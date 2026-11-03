/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic block translation: MIR block → Pliron IR block contents.
//!
//! Translates the contents of a single MIR basic block (statements + terminator)
//! into Pliron IR operations. The block itself is created by [`super::body`];
//! this module just populates it.
//!
//! # Translation order
//!
//! 1. Statements are translated in order, emitting loads/stores against the
//!    per-local alloca slots recorded in [`ValueMap`].
//! 2. Terminator is translated last and emits zero-operand control-flow ops
//!    (`mir.goto`, `mir.cond_br`, `mir.switch`, `mir.return`). Cross-block
//!    data flow happens via the slots, not block arguments.
//!
//! Panic blocks are the one exception: a block that ends in a diverging call
//! into `core::panicking` lowers to a device trap alone, statements included
//! (see [`translate_block`]).

use super::statement;
use super::statement_debug;
use super::terminator;
use crate::error::TranslationResult;
use crate::pipeline::StatementDebugInfoBlock;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::operation::Operation;
use rustc_public::mir;

/// Translates a MIR basic block's contents into the corresponding Pliron IR block.
///
/// # Arguments
///
/// * `ctx` - Pliron IR context
/// * `body` - The full MIR body (needed for local declarations)
/// * `mir_block` - The MIR block to translate
/// * `_idx` - Block index (unused, kept for debugging)
/// * `block_ptr` - Target Pliron IR block (already created)
/// * `value_map` - MIR local → alloca slot mapping
/// * `block_map` - Block index → Pliron IR block mapping
/// * `rustc_mono_successors` - Exact successors selected by rustc's
///   monomorphization traversal for this block
/// * `legaliser` - Shared identifier legaliser for name uniqueness
/// * `entry_prev_op` - For the entry block only: the last op emitted by
///   `body::translate_body`'s alloca/store setup (see `emit_entry_allocas`),
///   so that statements are appended **after** that setup instead of being
///   inserted at the front. For every other block this must be `None`.
/// * `statement_debug_info` - rustc's debug-only assignments aligned with the
///   statement boundaries in this block. Present only for full variable debug.
#[allow(clippy::too_many_arguments)]
pub fn translate_block(
    ctx: &mut Context,
    body: &mir::Body,
    mir_block: &mir::BasicBlock,
    _idx: usize,
    block_ptr: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    rustc_mono_successors: &[usize],
    legaliser: &mut Legaliser,
    entry_prev_op: Option<Ptr<Operation>>,
    statement_debug_info: Option<&StatementDebugInfoBlock>,
) -> TranslationResult<()> {
    let mut prev_op: Option<Ptr<Operation>> = entry_prev_op;

    // A block that ends in a diverging call into `core::panicking` is lowered
    // to `nvvm.trap` + `mir.unreachable`; the call itself is dropped, because
    // no panic runtime exists on the device (see
    // `terminator::is_dropped_panic_call`). Its statements exist only to build
    // what that call would have consumed -- the message `&str`, the
    // `format_args!` pieces -- and the block has no successor, so nothing they
    // write can ever be read. Emitting the trap alone is what makes a panic
    // carrying a message compile at all: a materialized `&str` constant has no
    // device lowering, so translating those statements always fails.
    //
    // A store *through a pointer* in such a block is dropped along with the
    // message. That stays inside the model cuda-oxide already uses for panics:
    // the trap aborts the kernel, and the memory state of an aborted kernel is
    // unspecified, so the store has no defined observer either way.
    if terminator::is_dropped_panic_call(&mir_block.terminator) {
        terminator::emit_dropped_panic_trap(ctx, &mir_block.terminator, block_ptr, prev_op);
        return Ok(());
    }

    for (statement_index, stmt) in mir_block.statements.iter().enumerate() {
        if let Some(debug_info) = statement_debug_info {
            prev_op = statement_debug::translate_statement_debug_info(
                ctx,
                body,
                &debug_info.before_statements[statement_index],
                value_map,
                block_ptr,
                prev_op,
            )?;
        }
        let op_ptr =
            statement::translate_statement(ctx, body, stmt, value_map, block_ptr, prev_op)?;
        prev_op = op_ptr;
    }

    if let Some(debug_info) = statement_debug_info {
        prev_op = statement_debug::translate_statement_debug_info(
            ctx,
            body,
            &debug_info.before_terminator,
            value_map,
            block_ptr,
            prev_op,
        )?;
    }

    let _term_op_ptr = terminator::translate_terminator(
        ctx,
        body,
        &mir_block.terminator,
        value_map,
        block_ptr,
        prev_op,
        block_map,
        rustc_mono_successors,
        legaliser,
    )?;

    Ok(())
}
