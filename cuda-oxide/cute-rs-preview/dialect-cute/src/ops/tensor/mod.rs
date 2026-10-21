/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level operations for one-dimensional tensor views.
//!
//! These operations preserve the story written by the kernel:
//!
//! ```text
//! pointer + length
//!       │
//!       ▼
//! tensor_make → tensor_zipped_divide → tensor_slice
//!                                         │
//!                         ┌───────────────┼───────────────┐
//!                         ▼               ▼               ▼
//!                    is_full/base     load/store     tail store
//! ```
//!
//! They do not prescribe CUDA instructions or MLIR spelling. A backend first
//! chooses how to lower the preserved view.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::builtin::{
    op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
    types::{IntegerType, Signedness},
};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::{
    CuteAlignmentAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorLayoutAttr,
};
use crate::types::CuteTensorViewType;

/// Read tensor-view metadata from an SSA operand's type.
fn view_of(ctx: &Context, value: Value) -> Option<CuteTensorViewType> {
    let ty = value.get_type(ctx);
    ty.deref(ctx).downcast_ref::<CuteTensorViewType>().cloned()
}

/// Preserve tensor storage facts while replacing its static selection layout.
fn transformed_view_type(
    ctx: &mut Context,
    view: Value,
    layout: CuteTensorLayoutAttr,
) -> TypeHandle {
    let source =
        view_of(ctx, view).expect("cute tensor-view builder needs a CuteTensorViewType input");
    source.with_layout(ctx, layout).into()
}

/// Check that the SSA value has an unsigned 64-bit integer type.
fn is_u64(ctx: &Context, value: Value) -> bool {
    let ty = value.get_type(ctx);
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

/// Check for a one-bit integer predicate type.
fn is_i1(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 1)
}

/// Compare tensor storage and access facts independently of the selection layout.
fn same_view_facts(left: &CuteTensorViewType, right: &CuteTensorViewType) -> bool {
    left.logical == right.logical
        && left.storage == right.storage
        && left.format == right.format
        && left.role == right.role
        && left.space == right.space
        && left.access == right.access
        && left.alignment == right.alignment
}

/// Check the register carrier's pointer pointee and required write capability.
fn carrier_pointer(ctx: &Context, value: Value, pointee: TypeHandle, needs_write: bool) -> bool {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let Some(pointer) = ty_ref.downcast_ref::<MirPtrType>() else {
        return false;
    };
    pointer.pointee == pointee
        && (!needs_write || pointer.is_mutable)
        && [address_space::GENERIC, address_space::LOCAL].contains(&pointer.address_space)
}

/// Require a selected tile whose byte width and alignment support one vector transfer.
fn valid_full_tile_transfer(
    view: &CuteTensorViewType,
    ctx: &Context,
    alignment_bytes: u64,
) -> bool {
    let Some(tile_size) = view.selected_tile_size() else {
        return false;
    };
    let Some(storage_bytes) = view.storage_bytes(ctx) else {
        return false;
    };
    let Some(tile_bytes) = tile_size.checked_mul(storage_bytes) else {
        return false;
    };
    matches!(tile_bytes, 4 | 8 | 16)
        && alignment_bytes.is_power_of_two()
        && alignment_bytes >= tile_bytes
}

/// Start a contiguous tensor view from a data pointer and element count.
///
/// ```text
/// %data, %len ──► cute.tensor_make ──► contiguous tensor view
/// ```
///
/// The result is a ghost value: this operation does not load or allocate.
#[pliron_op(
    name = "cute.tensor_make",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteTensorMakeOp;

impl CuteTensorMakeOp {
    /// Build a global-memory tensor view. `logical` and `storage` are kept as
    /// separate arguments even though v0 requires them to match.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        data: Value,
        len: Value,
        logical: TypeHandle,
        storage: TypeHandle,
        access: CuteTensorAccessAttr,
        alignment_bytes: u64,
    ) -> Self {
        let view: TypeHandle = CuteTensorViewType::get(
            ctx,
            logical,
            storage,
            CuteTensorAddressSpaceAttr::Gmem,
            access,
            alignment_bytes,
            CuteTensorLayoutAttr::Contiguous1D,
        )
        .into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![view],
                vec![data, len],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorMakeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tensor_make needs 2 operands and 1 result");
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(view) = result_ty_ref.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(op.loc(), "cute.tensor_make result must be a tensor view");
        };
        if view.layout != CuteTensorLayoutAttr::Contiguous1D {
            return verify_err!(op.loc(), "cute.tensor_make result must use Contiguous1D");
        }
        if view.space != CuteTensorAddressSpaceAttr::Gmem {
            return verify_err!(op.loc(), "cute.tensor_make result must be global memory");
        }
        view.verify(ctx)?;

        let data_ty = op.get_operand(0).get_type(ctx);
        let data_ty_ref = data_ty.deref(ctx);
        let Some(data) = data_ty_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.tensor_make data must be a MIR pointer");
        };
        if data.pointee != view.storage {
            return verify_err!(
                op.loc(),
                "cute.tensor_make data pointee must match tensor storage"
            );
        }
        if ![address_space::GENERIC, address_space::GLOBAL].contains(&data.address_space) {
            return verify_err!(
                op.loc(),
                "cute.tensor_make data must point at global or generic memory"
            );
        }
        if view.access == CuteTensorAccessAttr::ReadWrite && !data.is_mutable {
            return verify_err!(
                op.loc(),
                "a writable cute.tensor_make view needs a mutable data pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.tensor_make length must be an unsigned 64-bit integer"
            );
        }
        Ok(())
    }
}

/// Split one contiguous row into fixed-width tiles.
///
/// ```text
/// [0 1 2 3 4 5 6 7] ── zipped_divide<4> ──► [0 1 2 3] [4 5 6 7]
/// ```
///
/// Only the layout changes. Element, storage, space, access, and alignment
/// must stay exactly the same.
#[pliron_op(
    name = "cute.tensor_zipped_divide",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ],
    attributes = (zipped_layout: CuteTensorLayoutAttr)
)]
pub struct CuteTensorZippedDivideOp;

impl CuteTensorZippedDivideOp {
    /// Build a detached subdivision into fixed-width element tiles without accessing memory.
    pub fn new(ctx: &mut Context, tensor: Value, tile_size: u64) -> Self {
        let layout = CuteTensorLayoutAttr::Zipped1D(tile_size);
        let result_ty = transformed_view_type(ctx, tensor, layout);
        let divide = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_ty],
                vec![tensor],
                vec![],
                0,
            ),
        };
        divide.set_attr_zipped_layout(ctx, layout);
        divide
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorZippedDivideOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide needs 1 operand and 1 result"
            );
        }
        let Some(input) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide input must be a tensor view"
            );
        };
        if input.layout != CuteTensorLayoutAttr::Contiguous1D {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide input must use Contiguous1D"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide result must be a tensor view"
            );
        };
        let Some(layout) = self.get_attr_zipped_layout(ctx).map(|layout| *layout) else {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide must carry its zipped layout"
            );
        };
        if !matches!(layout, CuteTensorLayoutAttr::Zipped1D(size) if size > 0) {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide layout must be Zipped1D with a positive tile size"
            );
        }
        if result.layout != layout {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide result layout must match its layout attribute"
            );
        }
        if !same_view_facts(&input, result) {
            return verify_err!(
                op.loc(),
                "cute.tensor_zipped_divide may change only the tensor layout"
            );
        }
        result.verify(ctx)
    }
}

/// Select tile number `tile_index` from a zipped view.
///
/// ```text
/// zipped tiles: [tile 0] [tile 1] [tile 2]
///                              ^ slice(..., 1)
/// ```
#[pliron_op(
    name = "cute.tensor_slice",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteTensorSliceOp;

impl CuteTensorSliceOp {
    /// Build a detached selection of one tile, retaining the original tensor storage facts.
    pub fn new(ctx: &mut Context, tensor: Value, tile_index: Value) -> Self {
        let input = view_of(ctx, tensor)
            .expect("cute.tensor_slice builder needs a CuteTensorViewType input");
        let tile_size = match input.layout {
            CuteTensorLayoutAttr::Zipped1D(size) => size,
            _ => panic!("cute.tensor_slice builder needs a Zipped1D input"),
        };
        let result_ty = input
            .with_layout(ctx, CuteTensorLayoutAttr::Tile1D(tile_size))
            .into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_ty],
                vec![tensor, tile_index],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorSliceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tensor_slice needs 2 operands and 1 result");
        }
        let Some(input) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(op.loc(), "cute.tensor_slice input must be a tensor view");
        };
        let CuteTensorLayoutAttr::Zipped1D(tile_size) = input.layout else {
            return verify_err!(op.loc(), "cute.tensor_slice input must use Zipped1D");
        };
        if tile_size == 0 {
            return verify_err!(op.loc(), "cute.tensor_slice tile size must be positive");
        }
        if !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.tensor_slice tile index must be an unsigned 64-bit integer"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(op.loc(), "cute.tensor_slice result must be a tensor view");
        };
        if result.layout != CuteTensorLayoutAttr::Tile1D(tile_size) {
            return verify_err!(
                op.loc(),
                "cute.tensor_slice result tile width must match its zipped input"
            );
        }
        if !same_view_facts(&input, result) {
            return verify_err!(
                op.loc(),
                "cute.tensor_slice may change only the tensor layout"
            );
        }
        result.verify(ctx)
    }
}

/// Ask whether every position in a selected tile is in bounds.
#[pliron_op(
    name = "cute.tensor_is_full",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ]
)]
pub struct CuteTensorIsFullOp;

impl CuteTensorIsFullOp {
    /// Build a detached predicate checking whether the selected tile fits the runtime extent.
    pub fn new(ctx: &mut Context, tensor: Value) -> Self {
        let result_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_ty],
                vec![tensor],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorIsFullOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tensor_is_full needs 1 operand and 1 result");
        }
        let Some(view) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(op.loc(), "cute.tensor_is_full input must be a tensor view");
        };
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(size) if size > 0) {
            return verify_err!(
                op.loc(),
                "cute.tensor_is_full input must be a selected Tile1D"
            );
        }
        if !is_i1(ctx, op.get_result(0).get_type(ctx)) {
            return verify_err!(op.loc(), "cute.tensor_is_full result must be i1");
        }
        Ok(())
    }
}

/// Return the selected tile's first absolute element index.
#[pliron_op(
    name = "cute.tensor_base",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ]
)]
pub struct CuteTensorBaseOp;

impl CuteTensorBaseOp {
    /// Build a detached query for the selected tile's absolute element offset.
    pub fn new(ctx: &mut Context, tensor: Value) -> Self {
        let result_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_ty],
                vec![tensor],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorBaseOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tensor_base needs 1 operand and 1 result");
        }
        let Some(view) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(op.loc(), "cute.tensor_base input must be a tensor view");
        };
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(size) if size > 0) {
            return verify_err!(op.loc(), "cute.tensor_base input must be a selected Tile1D");
        }
        if !is_u64(ctx, op.get_result(0)) {
            return verify_err!(
                op.loc(),
                "cute.tensor_base result must be an unsigned 64-bit integer"
            );
        }
        Ok(())
    }
}

/// Load one full tile into a thread-local register carrier.
///
/// The operation keeps the tile as an operand. A backend can lower it to an
/// aligned native vector copy or a tensor-to-register mapping.
///
/// The verifier checks the static tile width and the alignment promise. At
/// runtime, the caller must also prove both facts below before executing it:
///
/// ```text
/// tensor_is_full(tile) == true
/// tile address % promised alignment == 0
/// ```
#[pliron_op(
    name = "cute.tensor_load_into",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (load_alignment_bytes: CuteAlignmentAttr)
)]
pub struct CuteTensorLoadIntoOp;

impl CuteTensorLoadIntoOp {
    /// Build a detached full-tile transfer into registers with a selected-address alignment promise.
    pub fn new(
        ctx: &mut Context,
        tensor: Value,
        destination: Value,
        assumed_alignment_bytes: u64,
    ) -> Self {
        let load = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![tensor, destination],
                vec![],
                0,
            ),
        };
        load.set_attr_load_alignment_bytes(ctx, CuteAlignmentAttr(assumed_alignment_bytes));
        load
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorLoadIntoOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into needs 2 operands and 0 results"
            );
        }
        let Some(view) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into source must be a tensor view"
            );
        };
        if view.access != CuteTensorAccessAttr::ReadOnly {
            return verify_err!(op.loc(), "cute.tensor_load_into source must be read-only");
        }
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(size) if size > 0) {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into source must be a selected Tile1D"
            );
        }
        if !carrier_pointer(ctx, op.get_operand(1), view.storage, true) {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into destination must be a mutable local/generic pointer to storage"
            );
        }
        let Some(alignment) = self
            .get_attr_load_alignment_bytes(ctx)
            .map(|attribute| attribute.0)
        else {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into must carry an alignment promise"
            );
        };
        if !valid_full_tile_transfer(&view, ctx, alignment) {
            return verify_err!(
                op.loc(),
                "cute.tensor_load_into needs a supported full-tile width and sufficient power-of-two alignment"
            );
        }
        Ok(())
    }
}

/// Store one thread-local register carrier into a full writable tile.
///
/// The verifier checks the static tile width, writable access, and alignment
/// promise. At runtime the selected tile must be full, its address must meet
/// that promise, and no conflicting memory access may race this store.
#[pliron_op(
    name = "cute.tensor_store_from",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (store_alignment_bytes: CuteAlignmentAttr)
)]
pub struct CuteTensorStoreFromOp;

impl CuteTensorStoreFromOp {
    /// Operand order follows the data flow: register carrier, then tensor.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        tensor: Value,
        assumed_alignment_bytes: u64,
    ) -> Self {
        let store = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![source, tensor],
                vec![],
                0,
            ),
        };
        store.set_attr_store_alignment_bytes(ctx, CuteAlignmentAttr(assumed_alignment_bytes));
        store
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorStoreFromOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from needs 2 operands and 0 results"
            );
        }
        let Some(view) = view_of(ctx, op.get_operand(1)) else {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from destination must be a tensor view"
            );
        };
        if view.access != CuteTensorAccessAttr::ReadWrite {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from destination must be writable"
            );
        }
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(size) if size > 0) {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from destination must be a selected Tile1D"
            );
        }
        if !carrier_pointer(ctx, op.get_operand(0), view.storage, false) {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from source must be a local/generic pointer to storage"
            );
        }
        let Some(alignment) = self
            .get_attr_store_alignment_bytes(ctx)
            .map(|attribute| attribute.0)
        else {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from must carry an alignment promise"
            );
        };
        if !valid_full_tile_transfer(&view, ctx, alignment) {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_from needs a supported full-tile width and sufficient power-of-two alignment"
            );
        }
        Ok(())
    }
}

/// Store one scalar at an absolute index inside a selected writable tile.
///
/// This is the short-tail path. Keeping it explicit lets both backends see
/// that the scalar store still belongs to the same selected tensor tile.
/// The absolute index must be inside both the selected tile and the tensor's
/// valid element range. No conflicting memory access may race this store.
#[pliron_op(
    name = "cute.tensor_store_element_abs",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>]
)]
pub struct CuteTensorStoreElementAbsOp;

impl CuteTensorStoreElementAbsOp {
    /// Build a detached scalar-tail store retaining its selected tile and absolute element index.
    pub fn new(ctx: &mut Context, tensor: Value, absolute_index: Value, value: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![tensor, absolute_index, value],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTensorStoreElementAbsOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs needs 3 operands and 0 results"
            );
        }
        let Some(view) = view_of(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs destination must be a tensor view"
            );
        };
        if view.access != CuteTensorAccessAttr::ReadWrite {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs destination must be writable"
            );
        }
        if !matches!(view.layout, CuteTensorLayoutAttr::Tile1D(size) if size > 0) {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs destination must be a selected Tile1D"
            );
        }
        if !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs index must be an unsigned 64-bit integer"
            );
        }
        if op.get_operand(2).get_type(ctx) != view.logical {
            return verify_err!(
                op.loc(),
                "cute.tensor_store_element_abs value must match the tensor logical type"
            );
        }
        Ok(())
    }
}

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
