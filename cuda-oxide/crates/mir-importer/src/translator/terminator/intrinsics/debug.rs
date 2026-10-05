/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsics.
//!
//! Handles translation of debug/profiling primitives including:
//! - `clock()` - 32-bit GPU clock counter
//! - `clock64()` - 64-bit GPU clock counter
//! - `globaltimer()` - 64-bit GPU global timer
//! - `__gpu_assertfail()` - Device-side assertion diagnostics
//! - `__gpu_vprintf()` - Formatted output to host console

use super::super::helpers;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use dialect_mir::ops::MirConstructTupleOp;
use dialect_mir::types::MirTupleType;
use dialect_nvvm::ops::{AssertFailOp, VprintfOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emits `__gpu_assertfail()`: CUDA device-side assertion diagnostics.
///
/// # Generated Operation
///
/// `nvvm.assertfail` - Maps to CUDA
/// `__assertfail(message, file, line, function, char_size)`
///
/// # Arguments
///
/// * `args[0]` - Pointer to null-terminated assertion message
/// * `args[1]` - Pointer to null-terminated source file name
/// * `args[2]` - Source line number (`u32`)
/// * `args[3]` - Pointer to null-terminated function or module context
/// * `args[4]` - Character size (`usize`)
#[allow(clippy::too_many_arguments)]
pub fn emit_assertfail(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use crate::translator::rvalue;

    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "__gpu_assertfail expects 5 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut translated_args = Vec::with_capacity(5);
    let mut last_op = prev_op;
    for arg in args {
        let (value, next_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        translated_args.push(value);
        last_op = next_op;
    }

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let assertfail_op = AssertFailOp::build(
        ctx,
        translated_args[0],
        translated_args[1],
        translated_args[2],
        translated_args[3],
        translated_args[4],
    );
    assertfail_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        assertfail_op.insert_after(ctx, prev);
    } else {
        assertfail_op.insert_at_front(block_ptr, ctx);
    }

    // The Rust placeholder returns `()`. Materialize the unit value so the
    // normal call destination and success edge remain well-formed if the CUDA
    // system call is treated conservatively as returning void.
    let unit_ty = MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, assertfail_op);

    let unit_value = unit_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_value,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "__gpu_assertfail call without target block",
    )
}

/// Emits `__gpu_vprintf()`: Formatted output to host console.
///
/// # Generated Operation
///
/// `nvvm.vprintf` - Maps to CUDA `vprintf(format, args)`
///
/// # Arguments
///
/// * `args[0]` - Pointer to null-terminated format string (*const u8)
/// * `args[1]` - Pointer to packed argument buffer (*const u8)
///
/// # Returns
///
/// i32 - Number of arguments on success (negative on error)
#[allow(clippy::too_many_arguments)]
pub fn emit_vprintf(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use crate::translator::rvalue;

    // Validate we have exactly 2 arguments: format_ptr and args_ptr
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "__gpu_vprintf expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Translate the format pointer operand
    let (format_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Translate the args pointer operand
    let (args_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Create the vprintf operation
    let vprintf_op = VprintfOp::build(ctx, format_ptr, args_ptr);
    vprintf_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = last_op {
        vprintf_op.insert_after(ctx, prev);
    } else {
        vprintf_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result (i32) in the destination
    let result_value = vprintf_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        vprintf_op,
        value_map,
        block_map,
        loc,
        "__gpu_vprintf call without target block",
    )
}
