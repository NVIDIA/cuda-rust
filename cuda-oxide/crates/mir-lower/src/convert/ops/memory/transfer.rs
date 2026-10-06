/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversions for `mir.memcpy` / `mir.memmove` and their shared lowering body.

use super::common::anyhow_to_pliron;
use crate::convert::types::{convert_type, llvm_type_size_align, mir_element_stride};
use crate::helpers;
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::IntBinArithOpWithOverflowFlag;
use llvm_export::ops as llvm;
use llvm_export::types::{FuncType, VoidType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;

/// Convert `mir.memcpy` to the matching `llvm.memcpy.p<dst>.p<src>.i<bits>`.
///
/// MIR's count is measured in pointee elements, while LLVM's memcpy intrinsic
/// expects bytes. The pre-conversion destination pointer type still carries the
/// MIR pointee, so use it to scale the count before emitting the call.
///
/// The intrinsic name is an overload: LLVM encodes each pointer's address
/// space and the length width into it, and its verifier rejects a call whose
/// argument types disagree with the name. Today every pointer reaching a
/// `copy_nonoverlapping` is a Rust raw pointer, which cuda-oxide normalizes to
/// the generic address space (an `addrspacecast` is inserted when the raw
/// pointer is formed), so the operands are always `p0` and `i64`. We still
/// derive the suffix from the real operand types rather than hardcoding
/// `p0.p0.i64`: it matches how every other overloaded intrinsic is named here
/// (`ctpop`, `fptosi.sat`, ...), and it keeps this lowering correct if raw
/// pointers ever start carrying a non-generic address space.
pub(crate) fn convert_memcpy(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mem_transfer(ctx, rewriter, op, operands_info, "memcpy")
}

/// Convert `mir.memmove` to the matching `llvm.memmove.p<dst>.p<src>.i<bits>`.
///
/// Identical to [`convert_memcpy`] except it emits the overlap-safe
/// `llvm.memmove` intrinsic. `mir.memmove` backs `core::intrinsics::copy`
/// (`ptr::copy`); `mir.memcpy` backs the non-overlapping variant.
pub(crate) fn convert_memmove(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mem_transfer(ctx, rewriter, op, operands_info, "memmove")
}

/// Shared lowering for `mir.memcpy` / `mir.memmove`. `intrinsic_base` selects
/// the LLVM intrinsic family ("memcpy" or "memmove"); both share the same
/// `(dst, src, len_bytes, isvolatile)` signature and element->byte count scaling.
///
/// The element type comes from the op's own `elem_type` attribute, stamped
/// at build time from dst's pointer type. Operand type history is not
/// usable here: a kind-only `mir.cast` lowers to a plain value forwarding,
/// history does not follow that edge, and a stale hit would scale the byte
/// count by the wrong element size.
fn convert_mem_transfer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_base: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (dst, src, count) = match operands.as_slice() {
        [dst, src, count] => (*dst, *src, *count),
        _ => {
            return pliron::input_err_noloc!(
                "{intrinsic_base} operation requires exactly 3 operands"
            );
        }
    };

    let pointee = {
        let op_ref = op.deref(ctx);
        op_ref
            .attributes
            .get::<pliron::builtin::attributes::TypeAttr>(
                &format!("{intrinsic_base}_elem_type").try_into().unwrap(),
            )
            .map(|attr| attr.get_type(ctx))
            .ok_or_else(|| {
                pliron::create_error!(
                    op_ref.loc(),
                    pliron::result::ErrorKind::VerificationFailed,
                    pliron::result::StringError(format!(
                        "mir.{intrinsic_base} missing its elem_type attribute; \
                         byte count has no fact to derive from"
                    ))
                )
            })?
    };
    // Byte-count policy: exact or error, never guessed. rustc's stride of the
    // stamped MIR elem type is the primary fact; the converted LLVM type's
    // natural layout is the fallback.
    let elem_size = match mir_element_stride(ctx, pointee) {
        Some(stride) => stride,
        None => {
            let elem_ty = convert_type(ctx, pointee).map_err(anyhow_to_pliron)?;
            match llvm_type_size_align(ctx, elem_ty) {
                Some((size, _)) => size,
                None => {
                    let type_display = pointee.deref(ctx).disp(ctx).to_string();
                    return Err(pliron::create_error!(
                        op.deref(ctx).loc(),
                        pliron::result::ErrorKind::VerificationFailed,
                        pliron::result::StringError(format!(
                            "mir.{intrinsic_base} element type `{type_display}` has no exact \
                             byte size; refusing to guess the copy length"
                        ))
                    ));
                }
            }
        }
    };

    let bytes = if elem_size == 1 {
        count
    } else {
        let count_ty = count.get_type(ctx);
        let bits = count_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(|ty| ty.width())
            .unwrap_or(64);
        let count_int_ty = IntegerType::get(ctx, bits, Signedness::Signless);
        let size_attr = IntegerAttr::new(
            count_int_ty,
            APInt::from_u64(
                elem_size,
                std::num::NonZeroUsize::new(bits as usize).unwrap(),
            ),
        );
        let size_const = llvm::ConstantOp::new(ctx, Box::new(size_attr));
        let size_val = size_const.get_operation().deref(ctx).get_result(0);
        rewriter.insert_operation(ctx, size_const.get_operation());

        let flags = IntegerOverflowFlagsAttr::default();
        let mul = llvm::MulOp::new_with_overflow_flag(ctx, count, size_val, flags);
        rewriter.insert_operation(ctx, mul.get_operation());
        mul.get_operation().deref(ctx).get_result(0)
    };

    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let false_attr = IntegerAttr::new(
        i1_ty,
        APInt::from_u64(0, std::num::NonZeroUsize::new(1).unwrap()),
    );
    let volatile = llvm::ConstantOp::new(ctx, Box::new(false_attr));
    rewriter.insert_operation(ctx, volatile.get_operation());
    let volatile_val = volatile.get_operation().deref(ctx).get_result(0);

    let void_ty = VoidType::get(ctx);
    let func_ty = FuncType::get(
        ctx,
        void_ty.into(),
        vec![
            dst.get_type(ctx),
            src.get_type(ctx),
            bytes.get_type(ctx),
            volatile_val.get_type(ctx),
        ],
        false,
    );
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            pliron::result::StringError(format!("{intrinsic_base} operation has no parent block"))
        )
    })?;
    // Derive the overload suffix from the real (already type-converted)
    // operands so the name can never disagree with the argument types.
    let dst_ty = dst.get_type(ctx);
    let dst_as = dst_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);
    let src_ty = src.get_type(ctx);
    let src_as = src_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);
    let len_bits = bytes
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .unwrap_or(64);
    let intrinsic_name = format!("llvm_{intrinsic_base}_p{dst_as}_p{src_as}_i{len_bits}");
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(anyhow_to_pliron)?;

    let callee: Identifier = intrinsic_name.as_str().try_into().map_err(|e| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            pliron::result::StringError(format!("Invalid memcpy intrinsic name: {e:?}"))
        )
    })?;
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(callee),
        func_ty,
        vec![dst, src, bytes, volatile_val],
    );
    crate::convert::preserve_location(ctx, op, call.get_operation());
    rewriter.insert_operation(ctx, call.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}
