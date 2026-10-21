/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level TMA transport used by the first GEMM slice.
//!
//! This layer keeps one simple story visible:
//!
//! ```text
//! descriptor ─► gmem carrier view ─┐
//!                                  ├─ tma_copy_2d(row, col, barrier)
//! shared base ─► smem carrier view ┘
//! ```
//!
//! TMA moves raw storage boxes. A packed-FP4 tile therefore travels as `u8`,
//! and a canonical scale tile travels as `u16`. These transport views are
//! deliberately `Plain` and `Generic`: E2M1, UE8M0, Mkl, and Nkl become
//! meaningful only when later shared-tensor and MMA operations interpret the
//! copied bytes.
//!
//! A backend continuation can lower each semantic copy without changing its
//! ordinary runtime ABI:
//!
//! ```text
//! tma_copy_2d(gmem, smem, row, col, barrier)
//!       │ resolve the two direct view producers
//!       ▼
//! copy_tma_2d(smem.base, barrier, gmem.descriptor, row, col)
//! ```

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::builtin::{
    op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    types::{FP32Type, IntegerType},
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
    CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr,
    CuteTensorRoleAttr,
};
use crate::layout::ComposedLayout;
use crate::types::{CuteTensorViewType, CuteTmaViewType};

/// Read a TMA transport view from an SSA operand's type.
fn tma_view_of(ctx: &Context, value: Value) -> Option<CuteTmaViewType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTmaViewType>()
        .cloned()
}

/// Return the supported scalar carrier width in bytes, excluding decoded packed formats.
fn transport_element_bytes(ctx: &Context, element: TypeHandle) -> Option<u64> {
    let element = element.deref(ctx);
    if element.downcast_ref::<MirFP16Type>().is_some() {
        Some(2)
    } else if element.downcast_ref::<FP32Type>().is_some() {
        Some(4)
    } else {
        element.downcast_ref::<IntegerType>().and_then(|integer| {
            match (integer.width(), integer.is_unsigned()) {
                (8, true) => Some(1),
                (16, true) => Some(2),
                _ => None,
            }
        })
    }
}

/// Check that the SSA value has an unsigned 64-bit integer type.
fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

/// Build the raw-carrier tensor/TMA type with the requested address space and access.
fn transport_view_type(
    ctx: &mut Context,
    element: TypeHandle,
    space: CuteTensorAddressSpaceAttr,
    access: CuteTensorAccessAttr,
    alignment_bytes: u64,
    smem_layout: ComposedLayout,
) -> TypeHandle {
    let tensor: TypeHandle = CuteTensorViewType::get_with_facts(
        ctx,
        element,
        element,
        space,
        access,
        alignment_bytes,
        CuteTensorFormatAttr::Plain,
        CuteTensorRoleAttr::Generic,
        CuteTensorLayoutAttr::Tma2D,
    )
    .into();
    CuteTmaViewType::get(ctx, tensor, smem_layout).into()
}

/// Wrap a TMA descriptor as a global raw-carrier tensor view.
///
/// The descriptor owns the real global pointer, dimensions, and strides. The
/// result type keeps the carrier element and the destination tile placement
/// visible without pretending the descriptor pointer is the data pointer.
/// The inner tensor's natural carrier alignment is therefore not a claim
/// about the hidden global data address; descriptor creation and the unsafe
/// copy contract own that separate requirement.
#[pliron_op(
    name = "cute.tma_gmem_view",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteTmaGmemViewOp;

impl CuteTmaGmemViewOp {
    /// Build a descriptor-backed transport view with explicit read or write access.
    fn new_with_access(
        ctx: &mut Context,
        descriptor: Value,
        element: TypeHandle,
        smem_layout: ComposedLayout,
        access: CuteTensorAccessAttr,
    ) -> Self {
        let alignment = transport_element_bytes(ctx, element)
            .expect("cute.tma_gmem_view builder needs a supported carrier element");
        let result = transport_view_type(
            ctx,
            element,
            CuteTensorAddressSpaceAttr::Gmem,
            access,
            alignment,
            smem_layout,
        );
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![descriptor],
                vec![],
                0,
            ),
        }
    }

    /// Wrap a descriptor used as a global source.
    pub fn new(
        ctx: &mut Context,
        descriptor: Value,
        element: TypeHandle,
        smem_layout: ComposedLayout,
    ) -> Self {
        Self::new_with_access(
            ctx,
            descriptor,
            element,
            smem_layout,
            CuteTensorAccessAttr::ReadOnly,
        )
    }

    /// Wrap a descriptor used as a global destination.
    ///
    /// The descriptor pointer is still immutable. `ReadWrite` describes the
    /// data reached through the descriptor, not the descriptor bytes.
    pub fn new_destination(
        ctx: &mut Context,
        descriptor: Value,
        element: TypeHandle,
        smem_layout: ComposedLayout,
    ) -> Self {
        Self::new_with_access(
            ctx,
            descriptor,
            element,
            smem_layout,
            CuteTensorAccessAttr::ReadWrite,
        )
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the descriptor handle hidden behind the semantic view.
    #[must_use]
    pub fn descriptor(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Verify for CuteTmaGmemViewOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tma_gmem_view needs 1 operand and 1 result");
        }
        let descriptor_ty = self.descriptor(ctx).get_type(ctx);
        let descriptor_ty = descriptor_ty.deref(ctx);
        let Some(descriptor) = descriptor_ty.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.tma_gmem_view descriptor must be a pointer");
        };
        if descriptor.is_mutable || descriptor.address_space != address_space::GENERIC {
            return verify_err!(
                op.loc(),
                "cute.tma_gmem_view descriptor must be an immutable generic pointer"
            );
        }
        let Some(view) = tma_view_of(ctx, op.get_result(0)) else {
            return verify_err!(op.loc(), "cute.tma_gmem_view result must be a TMA view");
        };
        view.verify(ctx)?;
        let tensor = view.tensor_view(ctx).expect("verified TMA tensor view");
        let element_bytes = tensor.storage_bytes(ctx).expect("verified carrier width");
        if tensor.space != CuteTensorAddressSpaceAttr::Gmem
            || !matches!(
                tensor.access,
                CuteTensorAccessAttr::ReadOnly | CuteTensorAccessAttr::ReadWrite
            )
            || tensor.alignment.0 != element_bytes
        {
            return verify_err!(
                op.loc(),
                "cute.tma_gmem_view must be a global carrier with natural alignment"
            );
        }
        Ok(())
    }
}

/// View one existing CTA-shared allocation as a TMA destination tile.
///
/// `capacity` counts carrier elements, not bytes or logical FP4 values.
/// `alignment_bytes` is the unsafe caller's promise about this selected stage
/// base; it may be stronger than the carrier's natural alignment.
#[pliron_op(
    name = "cute.tma_smem_view",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct CuteTmaSmemViewOp;

impl CuteTmaSmemViewOp {
    /// Build a detached transport view over shared storage and its element capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        base: Value,
        capacity: Value,
        element: TypeHandle,
        smem_layout: ComposedLayout,
        alignment_bytes: u64,
    ) -> Self {
        let result = transport_view_type(
            ctx,
            element,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadWrite,
            alignment_bytes,
            smem_layout,
        );
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![base, capacity],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the selected CTA-shared stage base.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the available carrier-element count from that base.
    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for CuteTmaSmemViewOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tma_smem_view needs 2 operands and 1 result");
        }
        let Some(view) = tma_view_of(ctx, op.get_result(0)) else {
            return verify_err!(op.loc(), "cute.tma_smem_view result must be a TMA view");
        };
        view.verify(ctx)?;
        let tensor = view.tensor_view(ctx).expect("verified TMA tensor view");
        if tensor.space != CuteTensorAddressSpaceAttr::Smem
            || tensor.access != CuteTensorAccessAttr::ReadWrite
        {
            return verify_err!(
                op.loc(),
                "cute.tma_smem_view must be a writable shared carrier"
            );
        }
        let element_bytes = i64::try_from(
            tensor
                .storage_bytes(ctx)
                .expect("verified TMA carrier has a byte width"),
        )
        .expect("supported TMA carrier width fits i64");
        let required_alignment =
            crate::layout::tma_phase_alignment_bytes(&view.smem_layout.0, element_bytes)
                .expect("verified TMA layout has a phase alignment");
        let promised_alignment = tensor.alignment.0;
        if promised_alignment < required_alignment as u64 {
            return verify_err!(
                op.loc(),
                "cute.tma_smem_view promises {promised_alignment}-byte alignment, but this layout needs {required_alignment} bytes"
            );
        }
        let base_ty = self.base(ctx).get_type(ctx);
        let base_ty = base_ty.deref(ctx);
        let Some(base) = base_ty.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.tma_smem_view base must be a MIR pointer");
        };
        if base.pointee != tensor.storage
            || !base.is_mutable
            || base.address_space != address_space::SHARED
        {
            return verify_err!(
                op.loc(),
                "cute.tma_smem_view base must be a mutable CTA-shared carrier pointer"
            );
        }
        if !is_u64(ctx, self.capacity(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_smem_view capacity must be an unsigned 64-bit integer"
            );
        }
        Ok(())
    }
}

/// Start one asynchronous two-dimensional copy from a descriptor-backed
/// global view into one selected shared stage.
///
/// The completion barrier stays explicit because the later pipeline slice
/// owns its initialization, expected byte count, wait, and release protocol.
#[pliron_op(
    name = "cute.tma_copy_2d",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>]
)]
pub struct CuteTmaCopy2dOp;

impl CuteTmaCopy2dOp {
    /// Build a detached TMA copy preserving the source/destination views, tile indices, and barrier.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        destination: Value,
        tile_row: Value,
        tile_column: Value,
        completion_barrier: Value,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![
                    source,
                    destination,
                    tile_row,
                    tile_column,
                    completion_barrier,
                ],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the source view operand of the asynchronous transfer.
    #[must_use]
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the destination view operand of the asynchronous transfer.
    #[must_use]
    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the runtime row index in units of complete TMA tiles.
    #[must_use]
    pub fn tile_row(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Return the runtime column index in units of complete TMA tiles.
    #[must_use]
    pub fn tile_column(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }

    /// Return the shared barrier pointer tracking completion of the transfer.
    #[must_use]
    pub fn completion_barrier(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(4)
    }
}

impl Verify for CuteTmaCopy2dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "cute.tma_copy_2d needs 5 operands and 0 results");
        }
        let Some(source) = tma_view_of(ctx, self.source(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_copy_2d source must be a TMA view");
        };
        let Some(destination) = tma_view_of(ctx, self.destination(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_copy_2d destination must be a TMA view");
        };
        source.verify(ctx)?;
        destination.verify(ctx)?;
        let source_tensor = source.tensor_view(ctx).expect("verified source tensor");
        let destination_tensor = destination
            .tensor_view(ctx)
            .expect("verified destination tensor");
        if source_tensor.space != CuteTensorAddressSpaceAttr::Gmem
            || source_tensor.access != CuteTensorAccessAttr::ReadOnly
            || destination_tensor.space != CuteTensorAddressSpaceAttr::Smem
            || destination_tensor.access != CuteTensorAccessAttr::ReadWrite
        {
            return verify_err!(
                op.loc(),
                "cute.tma_copy_2d direction must be read-only Gmem to writable Smem"
            );
        }
        if source_tensor.storage != destination_tensor.storage
            || source.smem_layout != destination.smem_layout
        {
            return verify_err!(
                op.loc(),
                "cute.tma_copy_2d source and destination need the same carrier and tile layout"
            );
        }
        if !is_u64(ctx, self.tile_row(ctx)) || !is_u64(ctx, self.tile_column(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_copy_2d tile row and column must be unsigned 64-bit integers"
            );
        }
        let barrier_ty = self.completion_barrier(ctx).get_type(ctx);
        let barrier_ty = barrier_ty.deref(ctx);
        let Some(barrier) = barrier_ty.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "cute.tma_copy_2d completion barrier must be a MIR pointer"
            );
        };
        let barrier_is_u64 = barrier
            .pointee
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned());
        if !barrier.is_mutable || barrier.address_space != address_space::SHARED || !barrier_is_u64
        {
            return verify_err!(
                op.loc(),
                "cute.tma_copy_2d completion barrier must be a mutable CTA-shared pointer to canonical unsigned u64 Barrier storage"
            );
        }
        Ok(())
    }
}

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
