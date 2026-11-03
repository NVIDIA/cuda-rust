/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Descriptor-backed raw transport views and their direct copy contracts.

use cute_layout::ComposedLayout;
use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use super::cooperative_elem_bytes;
use crate::attributes::CuteComposedLayoutAttr;

/// Frozen zero-result ABI for one hardware (TMA) tile copy into CTA-local
/// shared memory.
///
/// ```text
/// 0 smem base (T)    2 tensor map        4 tile column index
/// 1 mbarrier         3 tile row index
/// ```
///
/// The shared layout stays in a typed attribute so the compiler can prove
/// TMA encodability at build time; the descriptor itself carries the same
/// layout in its Rust TYPE, so pairing errors never reach this op.
/// Barrier lifecycle (init, arrive_expect_tx, wait) stays caller-owned.
/// Remote or multicast cluster-shared copies require a distinct operation;
/// accepting a generic destination here would erase the proof needed to
/// select PTX's CTA-local TMA form.
#[pliron_op(
    name = "cute.copy_tma_2d",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
    attributes = (
        tma_smem_layout: CuteComposedLayoutAttr,
        tma_elem: TypeAttr
    )
)]
pub struct CuteTmaLoad2dOp;

impl CuteTmaLoad2dOp {
    /// Build a detached CTA-local TMA load retaining its descriptor, barrier, and tile coordinates.
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 5],
        smem_layout: ComposedLayout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let load = Self { op };
        load.set_attr_tma_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        load.set_attr_tma_elem(ctx, TypeAttr::new(elem));
        load
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTmaLoad2dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d expects exactly 5 operands and 0 results"
            );
        }
        let Some(smem_layout) = self
            .get_attr_tma_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d must have a shared-memory layout"
            );
        };
        let Some(elem): Option<TypeHandle> =
            self.get_attr_tma_elem(ctx).map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.copy_tma_2d must have an element type");
        };
        let Some(elem_bytes) = cooperative_elem_bytes(ctx, elem) else {
            return verify_err!(op.loc(), "unsupported cute.copy_tma_2d element type");
        };
        if let Err(error) = cute_layout::validate_tma_encodable(&smem_layout, elem_bytes) {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d layout is not encodable: {error}"
            );
        }
        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d destination must be a MIR pointer"
            );
        };
        if pointer.pointee != elem
            || !pointer.is_mutable
            || pointer.address_space != address_space::SHARED
        {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d destination must be mutable CTA-shared elem memory"
            );
        }
        for index in [1usize, 2] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
                return verify_err!(
                    op.loc(),
                    "cute.copy_tma_2d operand {index} must be a pointer"
                );
            }
        }
        for (index, what) in [(3usize, "tile row"), (4usize, "tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{what} must be an unsigned 64-bit integer");
            }
        }
        Ok(())
    }
}

/// Frozen zero-result ABI for one hardware (TMA) tile copy from CTA-local
/// shared memory into global memory.
///
/// ```text
/// 0 smem source (T)   2 tile row index
/// 1 tensor map        3 tile column index
/// ```
///
/// The shared layout and element type remain typed attributes. The descriptor
/// carries the same witnesses at the Rust boundary, while this operation
/// preserves them for the selected backend continuation.
#[pliron_op(
    name = "cute.copy_tma_s2g_2d",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
    attributes = (
        tma_store_smem_layout: CuteComposedLayoutAttr,
        tma_store_elem: TypeAttr
    )
)]
pub struct CuteTmaStore2dOp;

impl CuteTmaStore2dOp {
    /// Build a detached CTA-local TMA store retaining its descriptor and tile coordinates.
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 4],
        smem_layout: ComposedLayout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let store = Self { op };
        store.set_attr_tma_store_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        store.set_attr_tma_store_elem(ctx, TypeAttr::new(elem));
        store
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTmaStore2dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d expects exactly 4 operands and 0 results"
            );
        }
        let Some(smem_layout) = self
            .get_attr_tma_store_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d must have a shared-memory layout"
            );
        };
        let Some(elem): Option<TypeHandle> = self
            .get_attr_tma_store_elem(ctx)
            .map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.copy_tma_s2g_2d must have an element type");
        };
        let Some(elem_bytes) = cooperative_elem_bytes(ctx, elem) else {
            return verify_err!(op.loc(), "unsupported cute.copy_tma_s2g_2d element type");
        };
        if let Err(error) = cute_layout::validate_tma_encodable(&smem_layout, elem_bytes) {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d layout is not encodable: {error}"
            );
        }
        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d source must be a MIR pointer"
            );
        };
        if pointer.pointee != elem || pointer.address_space != address_space::SHARED {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d source must be CTA-shared elem memory"
            );
        }
        let tensor_map_ty = op.get_operand(1).get_type(ctx);
        if tensor_map_ty
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .is_none()
        {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d tensor map must be a pointer"
            );
        }
        for (index, what) in [(2usize, "tile row"), (3usize, "tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{what} must be an unsigned 64-bit integer");
            }
        }
        Ok(())
    }
}
