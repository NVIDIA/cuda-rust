/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic NVVM intrinsic conversion for special registers.
//!
//! | Operation    | LLVM Intrinsic                    |
//! |--------------|-----------------------------------|
//! | `ReadTidX`   | `llvm_nvvm_read_ptx_sreg_tid_x`   |
//! | `ReadCtaidX` | `llvm_nvvm_read_ptx_sreg_ctaid_x` |
//! | `ReadNtidX`  | `llvm_nvvm_read_ptx_sreg_ntid_x`  |
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower a special-register read through exact inline PTX.
///
/// This is used when no LLVM intrinsic exists on every supported LLVM
/// version, when the modern PTX result is wider than LLVM's legacy intrinsic,
/// or when the register is a location sample that must be read again at every
/// source call. `kind` selects whether LLVM may common or remove the read.
pub(crate) fn convert_sreg_read_inline(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    result_width: u32,
    asm_template: &str,
    constraints: &str,
    kind: AsmKind,
) -> Result<()> {
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    let inline_asm = llvm_export::ops::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        vec![],
        asm_template,
        constraints,
        kind,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
