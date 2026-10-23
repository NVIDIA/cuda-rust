/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level block-scaled tensor operations used by GEMV.
//!
//! The dialect keeps the same small steps as the Rust kernel:
//!
//! ```text
//! packed E2M1 tensor ──┐
//!                      ├─ scaled_view ─ row ─ KTile<64> ─ load ─ fragment ─┐
//! UE8M0 scale tensor ──┘                                                   │
//!                                                                          ├─ dot(acc)
//! packed E2M1 tensor ──┐                                                   │
//!                      ├─ scaled_view ─ row ─ KTile<64> ─ load ─ fragment ─┘
//! UE8M0 scale tensor ──┘
//! ```
//!
//! Every arrow is its own operation. Nothing here means “run GEMV”. The
//! selected backend can lower the load and dot either to native pointer and
//! arithmetic operations or to the corresponding CuTe MLIR steps.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::builtin::{
    op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
    types::{FP32Type, IntegerType, Signedness},
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
    CuteAlignmentAttr, CuteScaledLayoutAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr,
    CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
};
use crate::types::{CuteFragmentType, CuteScaledViewType, CuteTensorViewType};

/// The first preserved GEMV slice loads 64 logical K values at a time.
pub const GEMV_K_TILE_WIDTH: u64 = 64;

/// Downcast an SSA operand's type to its packed tensor-view facts.
fn tensor_view_of(ctx: &Context, value: Value) -> Option<CuteTensorViewType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTensorViewType>()
        .cloned()
}

/// Downcast an SSA operand's type to its bound value/scale view.
fn scaled_view_of(ctx: &Context, value: Value) -> Option<CuteScaledViewType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteScaledViewType>()
        .cloned()
}

/// Read the source-view facts retained by a loaded block-scaled fragment.
fn fragment_of(ctx: &Context, value: Value) -> Option<CuteFragmentType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteFragmentType>()
        .cloned()
}

/// Check that the SSA value has an unsigned 64-bit integer type.
fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

/// Check for the FP32 scalar type required by accumulator carriers.
fn is_f32(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()
}

/// Compare the bound value/scale types and role, independently of the selected subview.
fn same_scaled_members(left: &CuteScaledViewType, right: &CuteScaledViewType) -> bool {
    left.values == right.values && left.scales == right.scales && left.role == right.role
}

/// Compare logical/storage types, format, and layout while allowing other view facts to differ.
fn same_tensor_facts_except_role(left: &CuteTensorViewType, right: &CuteTensorViewType) -> bool {
    left.logical == right.logical
        && left.storage == right.storage
        && left.format == right.format
        && left.layout == right.layout
}

/// Require compatible matrix/vector fragments for the supported block-scaled FP32 dot.
fn fragments_can_dot(ctx: &Context, matrix: &CuteFragmentType, vector: &CuteFragmentType) -> bool {
    let Some(matrix_source) = matrix.source_view(ctx) else {
        return false;
    };
    let Some(vector_source) = vector.source_view(ctx) else {
        return false;
    };
    if matrix_source.role != CuteTensorRoleAttr::Mkl
        || vector_source.role != CuteTensorRoleAttr::Nkl
        || matrix_source.layout != vector_source.layout
    {
        return false;
    }

    let matrix_values_ty = matrix_source.values.deref(ctx);
    let Some(matrix_values) = matrix_values_ty.downcast_ref::<CuteTensorViewType>() else {
        return false;
    };
    let vector_values_ty = vector_source.values.deref(ctx);
    let Some(vector_values) = vector_values_ty.downcast_ref::<CuteTensorViewType>() else {
        return false;
    };
    let matrix_scales_ty = matrix_source.scales.deref(ctx);
    let Some(matrix_scales) = matrix_scales_ty.downcast_ref::<CuteTensorViewType>() else {
        return false;
    };
    let vector_scales_ty = vector_source.scales.deref(ctx);
    let Some(vector_scales) = vector_scales_ty.downcast_ref::<CuteTensorViewType>() else {
        return false;
    };

    same_tensor_facts_except_role(matrix_values, vector_values)
        && same_tensor_facts_except_role(matrix_scales, vector_scales)
}

/// Recover rows and K extents only from a direct two-dimensional tensor constructor.
fn tensor_make_2d_shape(ctx: &Context, value: Value) -> Option<(Value, Value)> {
    let producer = value.defining_op()?;
    if Operation::get_opid(producer, ctx) != CuteTensorMake2DOp::get_opid_static() {
        return None;
    }
    let producer = producer.deref(ctx);
    Some((producer.get_operand(2), producer.get_operand(3)))
}

/// Describe one packed or scale tensor using its two logical dimensions.
///
/// `len` counts storage elements. `rows` and `k` count logical values. The
/// operation creates a view only; it does not read the pointer.
#[pliron_op(
    name = "cute.tensor_make_2d",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteTensorMake2DOp;

impl CuteTensorMake2DOp {
    /// Create a read-only global tensor with explicit format, role, and layout.
    ///
    /// `alignment_bytes` describes the storage pointer at view construction.
    /// Pass `1` when the safe source API has no stronger guarantee. An unsafe
    /// load records its selected-address promise on the load operation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        data: Value,
        len: Value,
        rows: Value,
        k: Value,
        logical: TypeHandle,
        storage: TypeHandle,
        format: CuteTensorFormatAttr,
        role: CuteTensorRoleAttr,
        alignment_bytes: u64,
        layout: CuteTensorLayoutAttr,
    ) -> Self {
        let result: TypeHandle = CuteTensorViewType::get_with_facts(
            ctx,
            logical,
            storage,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            alignment_bytes,
            format,
            role,
            layout,
        )
        .into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![data, len, rows, k],
                vec![],
                0,
            ),
        }
    }

    /// Create the packed E2M1 value view used by the GEMV matrix or vector.
    #[allow(clippy::too_many_arguments)]
    pub fn new_e2m1(
        ctx: &mut Context,
        data: Value,
        len: Value,
        rows: Value,
        k: Value,
        role: CuteTensorRoleAttr,
        alignment_bytes: u64,
    ) -> Self {
        let logical: TypeHandle = FP32Type::get(ctx).into();
        let storage: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        Self::new(
            ctx,
            data,
            len,
            rows,
            k,
            logical,
            storage,
            CuteTensorFormatAttr::E2M1,
            role,
            alignment_bytes,
            CuteTensorLayoutAttr::KMajor,
        )
    }

    /// Create the UE8M0 scale view paired with an E2M1 tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn new_ue8m0(
        ctx: &mut Context,
        data: Value,
        len: Value,
        rows: Value,
        k: Value,
        role: CuteTensorRoleAttr,
        alignment_bytes: u64,
        values_per_scale: u64,
    ) -> Self {
        let logical: TypeHandle = FP32Type::get(ctx).into();
        let storage: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        Self::new(
            ctx,
            data,
            len,
            rows,
            k,
            logical,
            storage,
            CuteTensorFormatAttr::UE8M0,
            role,
            alignment_bytes,
            CuteTensorLayoutAttr::BlockScaleKMajor(values_per_scale),
        )
    }

    /// Create the E4M3FN scale view paired with an E2M1 tensor (NVFP4).
    #[allow(clippy::too_many_arguments)]
    pub fn new_e4m3(
        ctx: &mut Context,
        data: Value,
        len: Value,
        rows: Value,
        k: Value,
        role: CuteTensorRoleAttr,
        alignment_bytes: u64,
        values_per_scale: u64,
    ) -> Self {
        let logical: TypeHandle = FP32Type::get(ctx).into();
        let storage: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        Self::new(
            ctx,
            data,
            len,
            rows,
            k,
            logical,
            storage,
            CuteTensorFormatAttr::E4M3,
            role,
            alignment_bytes,
            CuteTensorLayoutAttr::BlockScaleKMajor(values_per_scale),
        )
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the storage-element count from the source slice.
    #[must_use]
    pub fn storage_len(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the logical row count. Packed storage does not change it.
    #[must_use]
    pub fn logical_rows(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Return the logical K count. For E2M1, storage uses `K / 2` bytes.
    #[must_use]
    pub fn logical_k(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }
}

impl Verify for CuteTensorMake2DOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tensor_make_2d needs 4 operands and 1 result"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(view) = result_ty_ref.downcast_ref::<CuteTensorViewType>() else {
            return verify_err!(op.loc(), "cute.tensor_make_2d result must be a tensor view");
        };
        view.verify(ctx)?;
        if view.space != CuteTensorAddressSpaceAttr::Gmem
            || view.access != CuteTensorAccessAttr::ReadOnly
        {
            return verify_err!(
                op.loc(),
                "cute.tensor_make_2d must produce a read-only global tensor view"
            );
        }
        if !matches!(
            (view.format, view.layout),
            (CuteTensorFormatAttr::E2M1, CuteTensorLayoutAttr::KMajor)
                | (
                    CuteTensorFormatAttr::UE8M0 | CuteTensorFormatAttr::E4M3,
                    CuteTensorLayoutAttr::BlockScaleKMajor(_)
                )
        ) {
            return verify_err!(
                op.loc(),
                "cute.tensor_make_2d needs an E2M1 value or UE8M0/E4M3 scale layout"
            );
        }

        let data_ty = op.get_operand(0).get_type(ctx);
        let data_ty_ref = data_ty.deref(ctx);
        let Some(data) = data_ty_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.tensor_make_2d data must be a MIR pointer");
        };
        if data.pointee != view.storage {
            return verify_err!(
                op.loc(),
                "cute.tensor_make_2d data pointee must match tensor storage"
            );
        }
        if ![address_space::GENERIC, address_space::GLOBAL].contains(&data.address_space) {
            return verify_err!(
                op.loc(),
                "cute.tensor_make_2d data must point at global or generic memory"
            );
        }
        for (index, name) in [(1usize, "length"), (2, "rows"), (3, "K")] {
            if !is_u64(ctx, op.get_operand(index)) {
                return verify_err!(
                    op.loc(),
                    "cute.tensor_make_2d {name} must be an unsigned 64-bit integer"
                );
            }
        }
        Ok(())
    }
}

/// Bind a packed value tensor to the scale tensor that explains its range.
#[pliron_op(
    name = "cute.scaled_view_make",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteScaledViewMakeOp;

impl CuteScaledViewMakeOp {
    /// Bind value and scale tensor ghosts into one detached scaled-view operation.
    pub fn new(ctx: &mut Context, values: Value, scales: Value) -> Self {
        let values_view = tensor_view_of(ctx, values)
            .expect("cute.scaled_view_make builder needs a tensor-view values input");
        let result: TypeHandle = CuteScaledViewType::get(
            ctx,
            values.get_type(ctx),
            scales.get_type(ctx),
            values_view.role,
            CuteScaledLayoutAttr::Full,
        )
        .into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![values, scales],
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

impl Verify for CuteScaledViewMakeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make needs 2 operands and 1 result"
            );
        }
        if tensor_view_of(ctx, op.get_operand(0)).is_none()
            || tensor_view_of(ctx, op.get_operand(1)).is_none()
        {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make inputs must be tensor views"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteScaledViewType>() else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make result must be a scaled view"
            );
        };
        if result.values != op.get_operand(0).get_type(ctx)
            || result.scales != op.get_operand(1).get_type(ctx)
            || result.layout != CuteScaledLayoutAttr::Full
        {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make result must bind its two inputs at Full layout"
            );
        }
        let Some((value_rows, value_k)) = tensor_make_2d_shape(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make values must come directly from cute.tensor_make_2d"
            );
        };
        let Some((scale_rows, scale_k)) = tensor_make_2d_shape(ctx, op.get_operand(1)) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make scales must come directly from cute.tensor_make_2d"
            );
        };
        if value_rows != scale_rows || value_k != scale_k {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_make values and scales must share the same logical rows and K"
            );
        }
        result.verify(ctx)
    }
}

/// Select one logical row without reading values or scales.
#[pliron_op(
    name = "cute.scaled_view_row",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteScaledViewRowOp;

impl CuteScaledViewRowOp {
    /// Build a detached selection of one batch and row from a scaled tensor view.
    pub fn new(ctx: &mut Context, tensor: Value, batch: Value, row: Value) -> Self {
        let source = scaled_view_of(ctx, tensor)
            .expect("cute.scaled_view_row builder needs a scaled-view input");
        let result: TypeHandle = source.with_layout(ctx, CuteScaledLayoutAttr::Row).into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![tensor, batch, row],
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

impl Verify for CuteScaledViewRowOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_row needs 3 operands and 1 result"
            );
        }
        let Some(input) = scaled_view_of(ctx, op.get_operand(0)) else {
            return verify_err!(op.loc(), "cute.scaled_view_row input must be a scaled view");
        };
        if input.layout != CuteScaledLayoutAttr::Full {
            return verify_err!(op.loc(), "cute.scaled_view_row input must use Full layout");
        }
        if !is_u64(ctx, op.get_operand(1)) || !is_u64(ctx, op.get_operand(2)) {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_row batch and row must be unsigned 64-bit integers"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteScaledViewType>() else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_row result must be a scaled view"
            );
        };
        if !same_scaled_members(&input, result) || result.layout != CuteScaledLayoutAttr::Row {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_row may change only Full layout to Row"
            );
        }
        result.verify(ctx)
    }
}

/// Select 64 neighboring logical K values and their four scales.
#[pliron_op(
    name = "cute.scaled_view_k_tile",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteScaledViewKTileOp;

impl CuteScaledViewKTileOp {
    /// Build a detached K-tile selection with the source row's packed-format facts.
    pub fn new(ctx: &mut Context, row: Value, tile_index: Value) -> Self {
        let source = scaled_view_of(ctx, row)
            .expect("cute.scaled_view_k_tile builder needs a scaled row input");
        let result: TypeHandle = source
            .with_layout(ctx, CuteScaledLayoutAttr::KTile(GEMV_K_TILE_WIDTH))
            .into();
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![row, tile_index],
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

impl Verify for CuteScaledViewKTileOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile needs 2 operands and 1 result"
            );
        }
        let Some(input) = scaled_view_of(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile input must be a scaled view"
            );
        };
        if input.layout != CuteScaledLayoutAttr::Row {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile input must use Row layout"
            );
        }
        if !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile index must be an unsigned 64-bit integer"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteScaledViewType>() else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile result must be a scaled view"
            );
        };
        if !same_scaled_members(&input, result)
            || result.layout != CuteScaledLayoutAttr::KTile(GEMV_K_TILE_WIDTH)
        {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_k_tile may change only Row layout to KTile<64>"
            );
        }
        result.verify(ctx)
    }
}

/// Load one selected K tile into a register fragment.
///
/// The source view is safe to construct from byte slices and therefore makes
/// no strong alignment claim. This unsafe transfer carries the stronger
/// selected-address promises needed by its two physical reads:
///
/// ```text
/// 32 packed value bytes  -> address aligned to at least 16 bytes
///  4 scale bytes         -> address aligned to at least  4 bytes
/// ```
#[pliron_op(
    name = "cute.scaled_view_load",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ],
    attributes = (
        value_alignment_bytes: CuteAlignmentAttr,
        scale_alignment_bytes: CuteAlignmentAttr
    )
)]
pub struct CuteScaledViewLoadOp;

impl CuteScaledViewLoadOp {
    /// Build a detached block-scaled load with explicit selected-address alignment promises.
    pub fn new(
        ctx: &mut Context,
        tile: Value,
        value_alignment_bytes: u64,
        scale_alignment_bytes: u64,
    ) -> Self {
        let result: TypeHandle = CuteFragmentType::get(ctx, tile.get_type(ctx)).into();
        let load = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![tile],
                vec![],
                0,
            ),
        };
        load.set_attr_value_alignment_bytes(ctx, CuteAlignmentAttr(value_alignment_bytes));
        load.set_attr_scale_alignment_bytes(ctx, CuteAlignmentAttr(scale_alignment_bytes));
        load
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the promised alignment of the selected packed-value address.
    #[must_use]
    pub fn promised_value_alignment(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_value_alignment_bytes(ctx)
            .map(|alignment| alignment.0)
    }

    /// Return the promised alignment of the selected scale address.
    #[must_use]
    pub fn promised_scale_alignment(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_scale_alignment_bytes(ctx)
            .map(|alignment| alignment.0)
    }
}

impl Verify for CuteScaledViewLoadOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load needs 1 operand and 1 result"
            );
        }
        let Some(input) = scaled_view_of(ctx, op.get_operand(0)) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load input must be a scaled view"
            );
        };
        if input.layout != CuteScaledLayoutAttr::KTile(GEMV_K_TILE_WIDTH) {
            return verify_err!(op.loc(), "cute.scaled_view_load input must be KTile<64>");
        }
        let Some(value_alignment) = self.promised_value_alignment(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load must carry a packed-value alignment promise"
            );
        };
        let Some(scale_alignment) = self.promised_scale_alignment(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load must carry a scale alignment promise"
            );
        };
        if !value_alignment.is_power_of_two() || value_alignment < 16 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load needs packed-value alignment of at least 16 bytes"
            );
        }
        if !scale_alignment.is_power_of_two() || scale_alignment < 4 {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load needs scale alignment of at least 4 bytes"
            );
        }
        let result_ty = op.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let Some(result) = result_ty_ref.downcast_ref::<CuteFragmentType>() else {
            return verify_err!(op.loc(), "cute.scaled_view_load result must be a fragment");
        };
        if result.source != op.get_operand(0).get_type(ctx) {
            return verify_err!(
                op.loc(),
                "cute.scaled_view_load fragment must remember its input tile"
            );
        }
        result.verify(ctx)
    }
}

/// Add one M-row fragment × one N-row fragment dot product to `acc`.
///
/// This operation owns no loop or output store. It represents one K=64
/// reduction step and returns the next `f32` accumulator.
#[pliron_op(
    name = "cute.dot",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteDotOp;

impl CuteDotOp {
    /// Build a detached FP32 dot update from matching matrix/vector K fragments.
    pub fn new(ctx: &mut Context, matrix: Value, vector: Value, acc: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![acc.get_type(ctx)],
                vec![matrix, vector, acc],
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

impl Verify for CuteDotOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.dot needs 3 operands and 1 result");
        }
        let Some(matrix) = fragment_of(ctx, op.get_operand(0)) else {
            return verify_err!(op.loc(), "cute.dot left input must be a fragment");
        };
        let Some(vector) = fragment_of(ctx, op.get_operand(1)) else {
            return verify_err!(op.loc(), "cute.dot right input must be a fragment");
        };
        matrix.verify(ctx)?;
        vector.verify(ctx)?;
        if !fragments_can_dot(ctx, &matrix, &vector) {
            return verify_err!(op.loc(), "cute.dot needs compatible Mkl and Nkl fragments");
        }
        let acc_ty = op.get_operand(2).get_type(ctx);
        if !is_f32(ctx, acc_ty) {
            return verify_err!(op.loc(), "cute.dot accumulator must be f32");
        }
        if op.get_result(0).get_type(ctx) != acc_ty {
            return verify_err!(
                op.loc(),
                "cute.dot result must match its f32 accumulator type"
            );
        }
        Ok(())
    }
}

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
