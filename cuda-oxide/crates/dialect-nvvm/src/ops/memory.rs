/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory address-space conversion operations.
//!
//! CUDA source-level pointers are generic; hardware instruction descriptors
//! (WGMMA/tcgen05 SMEM descriptors, `ldmatrix`/`stmatrix` operands) consume
//! raw space-local offsets instead. These operations expose that conversion
//! as a first-class step, mirroring CUDA C++'s `__cvta_generic_to_shared_offset`.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

/// Convert a generic-address pointer into its raw `.shared` window offset.
///
/// The operand is a pointer in the generic (or already shared) address
/// space; the result is either the native `u32` shared address consumed by
/// instructions such as `ldmatrix`, or the `u64` offset carrier used to build
/// hardware SMEM descriptors. Lowered as `addrspacecast` to `addrspace(3)`
/// followed by `ptrtoint`, which `llc` selects as `cvta.to.shared`.
#[pliron_op(
    name = "nvvm.cvta_generic_to_shared_offset",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct CvtaGenericToSharedOffsetOp;

impl CvtaGenericToSharedOffsetOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtaGenericToSharedOffsetOp { op }
    }
}

fn is_shared_offset_integer(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            matches!(integer.width(), 32 | 64) && integer.signedness() == Signedness::Unsigned
        })
}

impl Verify for CvtaGenericToSharedOffsetOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.cvta_generic_to_shared_offset requires one operand and one result"
            );
        }
        let pointer_ty = op.get_operand(0).get_type(ctx);
        let pointer_ty_obj = pointer_ty.deref(ctx);
        let Some(pointer_ty) = pointer_ty_obj.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "nvvm.cvta_generic_to_shared_offset operand must be a MIR pointer"
            );
        };
        if !matches!(
            pointer_ty.address_space,
            address_space::GENERIC | address_space::SHARED
        ) {
            return verify_err!(
                op.loc(),
                "nvvm.cvta_generic_to_shared_offset operand must point to generic or shared memory"
            );
        }
        if !is_shared_offset_integer(ctx, op.get_result(0).get_type(ctx)) {
            return verify_err!(
                op.loc(),
                "nvvm.cvta_generic_to_shared_offset result must be u32 or u64"
            );
        }
        Ok(())
    }
}
