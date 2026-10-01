/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::attributes::GepNoWrapFlags;
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

pub(super) fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::input_error_noloc!("{e}")
}

/// Copy an enum value into a fresh stack slot and return the pointer.
///
/// This is how we reach a payload field that has no struct slot of its
/// own (its bytes are shared with a different-typed field of another
/// variant): once the value sits in memory, a byte-precise pointer can
/// read or write any part of it, no struct field needed.
///
/// The slot is marked with the enum's real (rustc) alignment. The struct
/// type alone can look under-aligned: `{ i8, [7 x i8] }` says "align 1"
/// to LLVM, while Rust may require 8.
///
/// The alloca lands at the use site, same as
/// [`convert_extract_array_element`](super::array_extract::convert_extract_array_element);
/// the standard `opt -O2` run (SROA)
/// removes it again. Hoisting these into the function's entry block is a
/// known follow-up for the unoptimized (`CUDA_OXIDE_NO_OPT=1`) path.
pub(super) fn spill_enum_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    enum_val: Value,
    llvm_struct_ty: TypeHandle,
    abi_align: u64,
) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let one_apint = APInt::from_i64(1, NonZeroUsize::new(64).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, Box::new(one_attr));
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca_op = llvm::AllocaOp::new(ctx, llvm_struct_ty, one_val, 0);
    rewriter.insert_operation(ctx, alloca_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, alloca_op.get_operation(), abi_align as u32);
    }
    let slot_ptr = alloca_op.get_operation().deref(ctx).get_result(0);

    let store_op = llvm::StoreOp::new(ctx, enum_val, slot_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, store_op.get_operation(), abi_align as u32);
    }
    slot_ptr
}

/// Pointer to `base + offset` bytes (`getelementptr i8, ptr base, offset`).
///
/// Used whenever rustc's physical byte offset cannot be represented faithfully
/// by a typed LLVM aggregate GEP, as with overlapping enum payloads or packed
/// struct fields.
pub(super) fn byte_offset_gep(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    base: Value,
    offset: u64,
) -> Value {
    use llvm_export::ops::GepIndex;
    let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let offset = emit_integer_constant(ctx, rewriter, 64, u128::from(offset));
    let gep_op = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        base,
        vec![GepIndex::Value(offset)],
        i8_ty,
        GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep_op.get_operation());
    gep_op.get_operation().deref(ctx).get_result(0)
}

pub(super) fn emit_integer_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    width: u32,
    bits: u128,
) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let attr = pliron::builtin::attributes::IntegerAttr::new(
        ty,
        APInt::from_u128(bits, NonZeroUsize::new(width as usize).unwrap()),
    );
    let op = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, op.get_operation());
    op.get_operation().deref(ctx).get_result(0)
}
