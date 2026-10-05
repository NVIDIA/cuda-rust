/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conversions for the debug-value markers and debug-local metadata copying.

use llvm_export::ops as llvm;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

pub(super) fn copy_debug_local_variable(
    ctx: &mut Context,
    mir_op: Ptr<Operation>,
    llvm_op: Ptr<Operation>,
) {
    if let Some(info) = llvm_export::ops::debug_local_variable(ctx, mir_op) {
        llvm_export::ops::set_debug_local_variable(ctx, llvm_op, info);
    }
    let aliases = llvm_export::ops::debug_whole_variable_aliases(ctx, mir_op);
    if !aliases.is_empty() {
        llvm_export::ops::set_debug_whole_variable_aliases(ctx, llvm_op, &aliases);
    }
    let projected = llvm_export::ops::debug_projected_variables(ctx, mir_op);
    if !projected.is_empty() {
        llvm_export::ops::set_debug_projected_variables(ctx, llvm_op, &projected);
    }
    let fragments = llvm_export::ops::debug_fragment_variables(ctx, mir_op);
    if !fragments.is_empty() {
        llvm_export::ops::set_debug_fragment_variables(ctx, llvm_op, &fragments);
    }
    if let Some(scope) = llvm_export::ops::debug_local_source_scope(ctx, mir_op) {
        llvm_export::ops::set_debug_local_source_scope(ctx, llvm_op, scope);
    }
    if let Some((file, pos)) = llvm_export::ops::debug_local_declaration_location(ctx, mir_op) {
        llvm_export::ops::set_debug_local_declaration_location(
            ctx, llvm_op, file, pos.line, pos.column,
        );
    }
    if let Some(expression) = llvm_export::ops::debug_value_expression(ctx, mir_op) {
        llvm_export::ops::set_debug_value_expression(ctx, llvm_op, &expression);
    }
}

/// Convert `mir.dbg_value` to the LLVM-export debug marker.
///
/// The op is still debug-only after lowering. The textual LLVM exporter later
/// prints it as an `llvm.dbg.value` intrinsic call.
pub(crate) fn convert_dbg_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let value = op.deref(ctx).get_operand(0);
    let loc = op.deref(ctx).loc().clone();
    let llvm_dbg_value = llvm::DebugValueOp::new(ctx, value);
    llvm_dbg_value.get_operation().deref_mut(ctx).set_loc(loc);
    copy_debug_local_variable(ctx, op, llvm_dbg_value.get_operation());
    rewriter.insert_operation(ctx, llvm_dbg_value.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.dbg_value_list` to the LLVM-export multi-value debug marker.
///
/// The ordered operands become a `DIArgList` during textual export. The typed
/// location recipe is carried as generic metadata and copied unchanged here.
pub(crate) fn convert_dbg_value_list(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let values: Vec<_> = op.deref(ctx).operands().collect();
    if values.len() < 2 {
        return pliron::input_err_noloc!("mir.dbg_value_list requires at least two operands");
    }
    if llvm_export::ops::debug_value_expression(ctx, op).is_none() {
        return pliron::input_err_noloc!("mir.dbg_value_list is missing its debug expression");
    }

    let loc = op.deref(ctx).loc().clone();
    let llvm_dbg_value = llvm::DebugValueListOp::new(ctx, values);
    llvm_dbg_value.get_operation().deref_mut(ctx).set_loc(loc);
    copy_debug_local_variable(ctx, op, llvm_dbg_value.get_operation());
    rewriter.insert_operation(ctx, llvm_dbg_value.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}
