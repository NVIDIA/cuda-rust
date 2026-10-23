/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed per-thread cache-aware register-to-global copies.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::{
    builtin::types::{FP32Type, IntegerType},
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err, verify_err_noloc,
};
use pliron_derive::{pliron_attr, pliron_op};

/// Number of adjacent scalar values copied by one thread.
#[pliron_attr(name = "cute.global_store", format = "`<` $elements `>`")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CuteGlobalStoreAttr {
    /// Number of contiguous storage elements transferred by this operation.
    pub elements: u64,
}
impl Verify for CuteGlobalStoreAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.elements == 0 || self.elements > 16 {
            return verify_err_noloc!("global store needs 1..16 elements");
        }
        Ok(())
    }
}

/// Byte size of a scalar accepted by the cache-aware global-store boundary.
pub fn global_store_element_bytes(ctx: &Context, element: TypeHandle) -> Option<u64> {
    let ty = element.deref(ctx);
    if ty.downcast_ref::<MirFP16Type>().is_some() {
        Some(2)
    } else if ty.downcast_ref::<FP32Type>().is_some()
        || ty
            .downcast_ref::<IntegerType>()
            .is_some_and(|i| i.width() == 32 && i.is_unsigned())
    {
        Some(4)
    } else {
        None
    }
}

/// Weak global store with an L1 no-allocation cache hint.
///
/// Operands are a writable global pointer and a local/generic register-carrier
/// pointer with identical element types. The destination's full-width alignment,
/// liveness, bounds, and absence of conflicting accesses are unsafe caller
/// obligations. The source carrier must be 32-byte aligned; the public Rust carrier type
/// guarantees this independently of the unsafe destination contract.
#[pliron_op(name="cute.store_global_no_allocate",format,attributes=(global_store:CuteGlobalStoreAttr))]
pub struct CuteStoreGlobalNoAllocateOp;
impl CuteStoreGlobalNoAllocateOp {
    /// Build a detached cache-aware register-to-global store with a static element count.
    pub fn new(ctx: &mut Context, destination: Value, source: Value, elements: u64) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![destination, source],
                vec![],
                0,
            ),
        };
        op.set_attr_global_store(ctx, CuteGlobalStoreAttr { elements });
        op
    }
    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
    /// Return the static count of elements transferred by the store.
    pub fn elements(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_global_store(ctx).map(|a| a.elements)
    }
}
impl Verify for CuteStoreGlobalNoAllocateOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "global store needs two pointers and no results");
        }
        let pointers: Vec<_> = op
            .operands()
            .map(|v| {
                v.get_type(ctx)
                    .deref(ctx)
                    .downcast_ref::<MirPtrType>()
                    .cloned()
            })
            .collect();
        let (Some(dst), Some(src)) = (&pointers[0], &pointers[1]) else {
            return verify_err!(op.loc(), "global store operands must be typed pointers");
        };
        if !dst.is_mutable
            || dst.address_space != address_space::GLOBAL
            || !matches!(
                src.address_space,
                address_space::LOCAL | address_space::GENERIC
            )
            || dst.pointee != src.pointee
        {
            return verify_err!(
                op.loc(),
                "global store requires matching writable global destination and local/generic source"
            );
        }
        let Some(bytes) = global_store_element_bytes(ctx, dst.pointee) else {
            return verify_err!(
                op.loc(),
                "global store supports only f16, f32, or u32 storage"
            );
        };
        let Some(n) = self.elements(ctx) else {
            return verify_err!(op.loc(), "global store lost its static width");
        };
        if !matches!(n.checked_mul(bytes), Some(4 | 8 | 16 | 32)) {
            return verify_err!(op.loc(), "global store needs 4, 8, 16, or 32 bytes");
        }
        Ok(())
    }
}
