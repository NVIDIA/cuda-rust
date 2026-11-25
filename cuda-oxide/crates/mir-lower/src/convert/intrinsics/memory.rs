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

use crate::convert::intrinsics::common::{
    cast_to_cluster_shared_addrspace, cast_to_shared_addrspace,
};
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
    let space = operands[0]
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|ty| ty.address_space());
    let shared_ptr = match space {
        Some(3 | 7) => operands[0], // AS7 retains the remote CTA rank.
        Some(0) => cast_to_shared_addrspace(ctx, rewriter, operands[0]),
        _ => {
            return pliron::input_err_noloc!(
                "shared offset requires generic, shared or cluster shared pointer"
            );
        }
    };

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

pub(crate) fn convert_cvta_generic_to_cluster_shared_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("cvta_generic_to_shared_offset requires an operand");
    }
    let space = operands[0]
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|ty| ty.address_space());
    if !matches!(space, Some(0 | 3 | 7)) {
        return pliron::input_err_noloc!(
            "cluster shared offset requires generic or shared pointer"
        );
    }
    let shared_ptr = cast_to_cluster_shared_addrspace(ctx, rewriter, operands[0]);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::{
        append_mir_return, build_kernel, find_all, kernel_blocks, make_ctx,
    };
    use dialect_mir::types::MirPtrType;
    use dialect_nvvm::ops::{CvtaGenericToClusterSharedOffsetOp, CvtaGenericToSharedOffsetOp};
    use pliron::r#type::TypeHandle;
    #[test]
    fn cvta_preserves_local_and_cluster_offsets() {
        for space in [0, 3, 7] {
            for width in [32, 64] {
                let mut ctx = make_ctx();
                let word: TypeHandle = IntegerType::get(&ctx, width, Signedness::Unsigned).into();
                let ptr: TypeHandle = MirPtrType::get(&mut ctx, word, true, space).into();
                let (module, block) = build_kernel(&mut ctx, vec![ptr], vec![word]);
                let arg = block.deref(&ctx).get_argument(0);
                let cvta = Operation::new(
                    &mut ctx,
                    CvtaGenericToSharedOffsetOp::get_concrete_op_info(),
                    vec![word],
                    vec![arg],
                    vec![],
                    0,
                );
                cvta.insert_at_back(block, &ctx);
                let value = cvta.deref(&ctx).get_result(0);
                append_mir_return(&mut ctx, block, vec![value]);
                crate::lower_mir_to_llvm(&mut ctx, module).unwrap();
                let blocks = kernel_blocks(&ctx, module);
                let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &blocks);
                let ints = find_all::<llvm::PtrToIntOp>(&ctx, &blocks);
                assert_eq!(casts.len(), usize::from(space == 0));
                assert_eq!(ints.len(), 1);
                let operand = ints[0].get_operation().deref(&ctx).get_operand(0);
                let ty = operand.get_type(&ctx);
                assert_eq!(
                    ty.deref(&ctx)
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .unwrap()
                        .address_space(),
                    if space == 0 { 3 } else { space }
                );
                if space == 0 {
                    assert_eq!(operand, casts[0].get_operation().deref(&ctx).get_result(0));
                }
            }
        }
    }
    #[test]
    fn cluster_cvta_targets_cluster_window_from_generic_and_named_pointers() {
        for space in [0, 3, 7] {
            for width in [32, 64] {
                let mut ctx = make_ctx();
                let word: TypeHandle = IntegerType::get(&ctx, width, Signedness::Unsigned).into();
                let ptr: TypeHandle = MirPtrType::get(&mut ctx, word, true, space).into();
                let (module, block) = build_kernel(&mut ctx, vec![ptr], vec![word]);
                let arg = block.deref(&ctx).get_argument(0);
                let cvta = Operation::new(
                    &mut ctx,
                    CvtaGenericToClusterSharedOffsetOp::get_concrete_op_info(),
                    vec![word],
                    vec![arg],
                    vec![],
                    0,
                );
                cvta.insert_at_back(block, &ctx);
                let value = cvta.deref(&ctx).get_result(0);
                append_mir_return(&mut ctx, block, vec![value]);
                crate::lower_mir_to_llvm(&mut ctx, module).unwrap();
                let blocks = kernel_blocks(&ctx, module);
                let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &blocks);
                let ints = find_all::<llvm::PtrToIntOp>(&ctx, &blocks);
                assert_eq!(casts.len(), usize::from(space != 7));
                assert_eq!(ints.len(), 1);
                let operand = ints[0].get_operation().deref(&ctx).get_operand(0);
                let ty = operand.get_type(&ctx);
                assert_eq!(
                    ty.deref(&ctx)
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .unwrap()
                        .address_space(),
                    7
                );
                if space != 7 {
                    assert_eq!(operand, casts[0].get_operation().deref(&ctx).get_result(0));
                }
            }
        }
    }
}
