/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory address-space conversion intrinsics.
//!
//! `nvvm.cvta_generic_to_shared_offset` lowers without inline PTX: an
//! `addrspacecast` into `addrspace(3)` followed by `ptrtoint`, which `llc`
//! selects as `cvta.to.shared`. The `ptrtoint` here deliberately reads the
//! space-local shared offset; it is a hardware descriptor/instruction-address
//! boundary, not a Rust pointer-address observation (those genericize first,
//! see `convert/ops/cast.rs`).

use crate::convert::intrinsics::common::cast_to_shared_addrspace;
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert `nvvm.cvta_generic_to_shared_offset` to `addrspacecast` + `ptrtoint`.
pub(crate) fn convert_cvta_generic_to_shared_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("cvta_generic_to_shared_offset requires an operand");
    }
    let shared_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let Some(result_width) = result_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(IntegerType::width)
    else {
        return pliron::input_err_noloc!(
            "cvta_generic_to_shared_offset requires an integer result"
        );
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let ptr_to_int = llvm::PtrToIntOp::new(ctx, shared_ptr, result_ty.into());
    rewriter.insert_operation(ctx, ptr_to_int.get_operation());
    rewriter.replace_operation(ctx, op, ptr_to_int.get_operation());
    Ok(())
}
