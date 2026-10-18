/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative matrix loads with typed operand roles and shared placement.

use cute_layout::ComposedLayout;
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
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

use crate::attributes::{CuteComposedLayoutAttr, CuteMatrixRoleAttr};

/// Frozen zero-result ABI for one warp-cooperative `ldmatrix` fragment load.
///
/// ```text
/// 0 smem base (f16)   2 warp tile column   4 fragment slot (u32 array)
/// 1 warp tile row     3 lane id
/// ```
///
/// The shared layout (with any swizzle) and the operand role stay in typed
/// attributes. Backend lowering computes each lane's row address through the
/// composed byte map and selects the target load, storing the returned values
/// into the fragment slot. Operands 1..=2 are `u64`; operand 3 is `u32`.
/// As in the unsafe `cute-rs` matrix-load API, the lane must be the caller's
/// hardware lane (0..32) and the whole selected window must lie in the layout.
/// Lowering may use these bounds to simplify tile-relative address arithmetic;
/// this operation does not insert dynamic window or lane checks.
#[pliron_op(
    name = "cute.ldmatrix",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
    attributes = (
        matrix_role: CuteMatrixRoleAttr,
        ldmatrix_smem_layout: CuteComposedLayoutAttr,
        ldmatrix_elem: TypeAttr
    )
)]
pub struct CuteLdmatrixOp;

impl CuteLdmatrixOp {
    /// Build a detached warp matrix load with its typed shared-layout and operand-role contract.
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 5],
        role: CuteMatrixRoleAttr,
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
        load.set_attr_matrix_role(ctx, role);
        load.set_attr_ldmatrix_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        load.set_attr_ldmatrix_elem(ctx, TypeAttr::new(elem));
        load
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteLdmatrixOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let Some(role) = self.get_attr_matrix_role(ctx).map(|attr| *attr) else {
            return verify_err!(op.loc(), "cute.ldmatrix must have a matrix role");
        };
        let Some(smem_layout) = self
            .get_attr_ldmatrix_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(op.loc(), "cute.ldmatrix must have a shared-memory layout");
        };
        let Some(elem): Option<TypeHandle> = self
            .get_attr_ldmatrix_elem(ctx)
            .map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.ldmatrix must have an element type");
        };
        if elem.deref(ctx).downcast_ref::<MirFP16Type>().is_none() {
            return verify_err!(op.loc(), "cute.ldmatrix v0 supports f16 elements only");
        }
        let _ = role;
        let modes = smem_layout.inner().modes();
        if modes.len() != 2 {
            return verify_err!(op.loc(), "cute.ldmatrix shared layout needs two modes");
        }
        let (Some(rows), Some(columns)) = (modes[0].checked_size(), modes[1].checked_size()) else {
            return verify_err!(op.loc(), "cute.ldmatrix shared layout extents are invalid");
        };
        let byte_layout = match smem_layout.to_byte_offsets(2) {
            Ok(layout) => layout,
            Err(error) => {
                return verify_err!(
                    op.loc(),
                    "cute.ldmatrix shared layout has no byte form: {error}"
                );
            }
        };
        if let Err(error) = cute_layout::validate_ldmatrix_source(&byte_layout, rows, columns) {
            return verify_err!(op.loc(), "cute.ldmatrix source is not loadable: {error}");
        }

        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.ldmatrix source must be a MIR pointer");
        };
        if pointer.pointee != elem
            || ![address_space::GENERIC, address_space::SHARED].contains(&pointer.address_space)
        {
            return verify_err!(
                op.loc(),
                "cute.ldmatrix source must point at shared/generic f16"
            );
        }
        for (index, role_name) in [(1usize, "warp tile row"), (2usize, "warp tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{role_name} must be an unsigned 64-bit integer");
            }
        }
        let lane_ty = op.get_operand(3).get_type(ctx);
        if lane_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none_or(|integer| integer.width() != 32 || !integer.is_unsigned())
        {
            return verify_err!(op.loc(), "lane id must be an unsigned 32-bit integer");
        }
        let slot_ty = op.get_operand(4).get_type(ctx);
        let slot_ref = slot_ty.deref(ctx);
        let Some(slot) = slot_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "fragment slot must be a MIR pointer");
        };
        if !slot.is_mutable {
            return verify_err!(op.loc(), "fragment slot must be mutable");
        }
        Ok(())
    }
}
