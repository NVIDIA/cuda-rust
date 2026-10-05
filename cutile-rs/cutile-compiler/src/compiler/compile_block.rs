/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_block.rs` — translates Rust `syn::Block`
//! AST nodes into tile-ir operations. Only type and IR-emission changes; the
//! control flow, dispatch logic, and variable binding are identical.

use super::_function::CUDATileFunctionCompiler;
use super::_value::{BlockTerminator, CompilerContext, LoopKind, Mutability, TileRustValue};
use super::shared_types::Kind;
use super::shared_utils::{STACK_GROW_SIZE, STACK_RED_ZONE};
use super::tile_rust_type::TileRustType;
use crate::error::{JITError, SpannedJITError};
use crate::generics::GenericVars;
use crate::syn_utils::*;
use crate::types::get_type_mutability;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{BlockId, Module};

use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{Expr, Item, Pat, Stmt};

impl<'m> CUDATileFunctionCompiler<'m> {
    fn carries_partition_axis_proof_origin(value: &TileRustValue) -> bool {
        let Some(ident) = get_type_ident(&value.ty.rust_ty) else {
            return false;
        };
        matches!(
            ident.to_string().as_str(),
            "Tensor"
                | "Partition"
                | "BoundedPartition"
                | "BoundedPartitionMut"
                | "PartitionMut"
                | "MappedPartitionMut"
        )
    }

    fn bind_pattern_value(
        &self,
        pat: &Pat,
        value: TileRustValue,
        inherited_mutability: bool,
        ctx: &mut CompilerContext,
    ) -> Result<(), JITError> {
        match pat {
            Pat::Ident(ident) => {
                let mut value = value;
                value.mutability = if inherited_mutability || ident.mutability.is_some() {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                if value.tensor_origin.is_none()
                    && Self::carries_partition_axis_proof_origin(&value)
                {
                    value.tensor_origin = Some(ident.ident.to_string());
                }
                ctx.vars.insert(ident.ident.to_string(), value.clone());
                if let Some((_at, subpat)) = &ident.subpat {
                    self.bind_pattern_value(subpat, value, inherited_mutability, ctx)?;
                }
                Ok(())
            }
            Pat::Type(pat_type) => self.bind_pattern_value(
                &pat_type.pat,
                value,
                inherited_mutability || get_type_mutability(&pat_type.ty),
                ctx,
            ),
            Pat::Paren(paren) => {
                self.bind_pattern_value(&paren.pat, value, inherited_mutability, ctx)
            }
            Pat::Reference(reference) => self.bind_pattern_value(
                &reference.pat,
                value,
                inherited_mutability || reference.mutability.is_some(),
                ctx,
            ),
            Pat::Tuple(tuple) => {
                let Some(elements) = value.values.clone() else {
                    return self.jit_error_result(
                        &tuple.span(),
                        "right-hand side of tuple destructuring must be a tuple expression",
                    );
                };
                if elements.len() != tuple.elems.len() {
                    return self.jit_error_result(
                        &tuple.span(),
                        &format!(
                            "tuple pattern has {} bindings but the expression produces {} values",
                            tuple.elems.len(),
                            elements.len()
                        ),
                    );
                }
                for (pat, value) in tuple.elems.iter().zip(elements) {
                    self.bind_pattern_value(pat, value, inherited_mutability, ctx)?;
                }
                Ok(())
            }
            Pat::Slice(slice) => {
                let Some(elements) = value.values.clone() else {
                    return self.jit_error_result(
                        &slice.span(),
                        "right-hand side of slice destructuring must be an array expression",
                    );
                };
                let pats = slice.elems.iter().collect::<Vec<_>>();
                let rest_pos = pats.iter().position(|pat| matches!(pat, Pat::Rest(_)));
                match rest_pos {
                    Some(rest_pos) => {
                        if pats.len().saturating_sub(1) > elements.len() {
                            return self.jit_error_result(
                                &slice.span(),
                                &format!(
                                    "slice pattern requires at least {} values but the expression produces {} values",
                                    pats.len() - 1,
                                    elements.len()
                                ),
                            );
                        }
                        for idx in 0..rest_pos {
                            self.bind_pattern_value(
                                pats[idx],
                                elements[idx].clone(),
                                inherited_mutability,
                                ctx,
                            )?;
                        }
                        let suffix_len = pats.len() - rest_pos - 1;
                        for suffix_idx in 0..suffix_len {
                            self.bind_pattern_value(
                                pats[rest_pos + 1 + suffix_idx],
                                elements[elements.len() - suffix_len + suffix_idx].clone(),
                                inherited_mutability,
                                ctx,
                            )?;
                        }
                    }
                    None => {
                        if pats.len() != elements.len() {
                            return self.jit_error_result(
                                &slice.span(),
                                &format!(
                                    "slice pattern has {} bindings but the expression produces {} values",
                                    pats.len(),
                                    elements.len()
                                ),
                            );
                        }
                        for (pat, value) in pats.into_iter().zip(elements) {
                            self.bind_pattern_value(pat, value, inherited_mutability, ctx)?;
                        }
                    }
                }
                Ok(())
            }
            Pat::Struct(pat_struct) => {
                if value.kind != Kind::Struct {
                    return self.jit_error_result(
                        &pat_struct.span(),
                        "right-hand side of struct destructuring must be a struct value",
                    );
                }
                let Some(fields) = value.fields.clone() else {
                    return self.jit_error_result(
                        &pat_struct.span(),
                        "struct value is missing its field data (internal)",
                    );
                };
                for field in &pat_struct.fields {
                    let syn::Member::Named(field_name) = &field.member else {
                        return self.jit_error_result(
                            &field.member.span(),
                            "tuple struct patterns are not supported in struct destructuring",
                        );
                    };
                    let Some(field_value) = fields.get(&field_name.to_string()).cloned() else {
                        return self.jit_error_result(
                            &field.member.span(),
                            &format!("{} is not a field", field_name),
                        );
                    };
                    self.bind_pattern_value(&field.pat, field_value, inherited_mutability, ctx)?;
                }
                Ok(())
            }
            Pat::TupleStruct(tuple_struct) => {
                let Some(elements) = value.values.clone() else {
                    return self.jit_error_result(
                        &tuple_struct.span(),
                        "right-hand side of tuple-struct destructuring must be a compound value",
                    );
                };
                if elements.len() != tuple_struct.elems.len() {
                    return self.jit_error_result(
                        &tuple_struct.span(),
                        &format!(
                            "tuple-struct pattern has {} bindings but the expression produces {} values",
                            tuple_struct.elems.len(),
                            elements.len()
                        ),
                    );
                }
                for (pat, value) in tuple_struct.elems.iter().zip(elements) {
                    self.bind_pattern_value(pat, value, inherited_mutability, ctx)?;
                }
                Ok(())
            }
            Pat::Or(or_pat) => {
                for case in &or_pat.cases {
                    self.bind_pattern_value(case, value.clone(), inherited_mutability, ctx)?;
                }
                Ok(())
            }
            Pat::Wild(_) | Pat::Rest(_) => Ok(()),
            _ => self.jit_error_result(
                &pat.span(),
                "this pattern form is not supported in let bindings",
            ),
        }
    }

    pub fn compile_block(
        &self,
        module: &mut Module,
        block_id: BlockId,
        block_expr: &syn::Block,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _block_debug_str = block_expr.to_token_stream().to_string();
            // Top-level returns yield the inlined function's value. Nested
            // returns need the separate 13.4 kernel-loop handling below.
            let is_fn_body = std::mem::replace(&mut ctx.fn_body, false);
            let mut terminator_encountered = None;
            let mut return_value: Option<TileRustValue> = None;
            let num_statements = &block_expr.stmts.len();
            for (i, statement) in block_expr.stmts.iter().enumerate() {
                let is_last = i == num_statements - 1;
                match statement {
                    Stmt::Local(local) => {
                        let mut bindings = Vec::new();
                        super::shared_utils::collect_pattern_bindings(&local.pat, &mut bindings)?;
                        let Some(init) = &local.init else {
                            return self.jit_error_result(
                                &local.span(),
                                "let bindings must have an initializer expression",
                            );
                        };
                        let annotated_ty = local_pattern_type(&local.pat);
                        let annotated_ct_ty = match annotated_ty {
                            Some(ty) => self.compile_type(ty, generic_args, &HashMap::new())?,
                            None => None,
                        };
                        let init_ty = self
                            .typeck_expr_tile_type(&init.expr, generic_args, &HashMap::new())?
                            .or(annotated_ct_ty);
                        let Some(value) = self.compile_expression(
                            module,
                            block_id,
                            &init.expr,
                            generic_args,
                            ctx,
                            init_ty,
                        )?
                        else {
                            return self.jit_error_result(
                                &init.expr.span(),
                                &format!(
                                    "failed to compile initializer: `{}`",
                                    init.expr.to_token_stream()
                                ),
                            );
                        };
                        // The initializer can install a token too. Check only
                        // after compiling it, before replacing the old binding.
                        for name in &bindings {
                            if ctx.explicit_token_updates.contains(name) {
                                return self.jit_error_result(
                                    &local.pat.span(),
                                    "cannot rebind a tensor after set_token/set_tensor_token; install the token on the final function-level binding before creating views",
                                );
                            }
                        }
                        self.bind_pattern_value(&local.pat, value, false, ctx)?;
                        for name in bindings {
                            if is_fn_body {
                                ctx.function_level_bindings.insert(name);
                            } else {
                                ctx.function_level_bindings.remove(&name);
                            }
                        }
                    }
                    Stmt::Item(item) => {
                        match item {
                            Item::Const(const_item) => {
                                let binding_name: Option<String> =
                                    Some(const_item.ident.to_string());
                                let ct_ty: Option<TileRustType> = self.compile_type(
                                    &const_item.ty,
                                    generic_args,
                                    &HashMap::new(),
                                )?;
                                let Some(binding_name) = binding_name else {
                                    return self.jit_error_result(
                                        &const_item.span(),
                                        "unable to determine name for const binding",
                                    );
                                };
                                match self.compile_expression(
                                    module,
                                    block_id,
                                    &const_item.expr,
                                    generic_args,
                                    ctx,
                                    ct_ty,
                                )? {
                                    Some(mut value) => {
                                        value.mutability = Mutability::Immutable;
                                        ctx.vars.insert(binding_name, value);
                                    }
                                    None => {
                                        return self.jit_error_result(
                                            &const_item.expr.span(),
                                            &format!(
                                                "failed to compile const initializer: `{}`",
                                                const_item.expr.to_token_stream()
                                            ),
                                        )
                                    }
                                }
                            }
                            _ => {
                                return self.jit_error_result(
                                    &item.span(),
                                    "only `const` item definitions are supported inside function bodies",
                                )
                            }
                        };
                    }
                    Stmt::Expr(expr, semicolon) => match expr {
                        Expr::Continue(_continue_expr) => {
                            let Some(loop_carry_var_names) = &ctx.carry_vars else {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "`continue` cannot be used outside of a loop",
                                );
                            };
                            terminator_encountered = Some(BlockTerminator::Continue);
                            let loop_carry_values = ctx.unpack_some_vars(loop_carry_var_names)?;
                            let (op_id, _) =
                                OpBuilder::new(Opcode::Continue, self.ir_location(&expr.span()))
                                    .operands(loop_carry_values.iter().copied())
                                    .build(module);
                            append_op(module, block_id, op_id);
                        }
                        Expr::Break(_break_expr) => {
                            let Some(loop_carry_var_names) = &ctx.carry_vars else {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "Executing break outside of loop is not supported.",
                                );
                            };
                            // `cuda_tile.for` has no early exit: a `cuda_tile.break`
                            // nested in it is rejected by the assembler ("can only
                            // be nested within 'cuda_tile.loop', 'cuda_tile.if'"),
                            // so the kernel would fail at `tileiras` time with an
                            // IR-level message. Reject it here, at the source.
                            if ctx.innermost_loop == Some(LoopKind::For) {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "`break` inside a `for` loop is not supported: Tile IR `for` loops \
                                     cannot exit early. Rewrite the loop as `while`/`loop` with an \
                                     explicit counter, or guard the remaining body with the exit \
                                     condition",
                                );
                            }
                            terminator_encountered = Some(BlockTerminator::Break);
                            let loop_carry_values = ctx.unpack_some_vars(loop_carry_var_names)?;
                            let (op_id, _) =
                                OpBuilder::new(Opcode::Break, self.ir_location(&expr.span()))
                                    .operands(loop_carry_values.iter().copied())
                                    .build(module);
                            append_op(module, block_id, op_id);
                        }
                        Expr::Assign(assign_expr) => {
                            let var_name: String = match &*assign_expr.left {
                                    Expr::Path(path_expr) => {
                                        get_ident_from_path_expr(path_expr).to_string()
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &assign_expr.left.span(),
                                            "only simple variable names are supported on the left side of an assignment",
                                        )
                                    }
                                };
                            let rhs_ty = self.typeck_expr_tile_type(
                                &assign_expr.right,
                                generic_args,
                                &HashMap::new(),
                            )?;
                            let mut ct_value: TileRustValue =
                                match self.compile_expression(
                                    module,
                                    block_id,
                                    &assign_expr.right,
                                    generic_args,
                                    ctx,
                                    rhs_ty,
                                )? {
                                    Some(value) => value,
                                    None => return self.jit_error_result(
                                        &assign_expr.right.span(),
                                        "failed to compile the right-hand side of this assignment",
                                    ),
                                };
                            ct_value.mutability = Mutability::Mutable;
                            ctx.vars.insert(var_name, ct_value);
                        }
                        Expr::Return(return_expr) => {
                            if !is_fn_body
                                && ctx.kernel_entry
                                && !ctx.inside_for
                                && ctx.innermost_loop == Some(LoopKind::Loop)
                            {
                                if let Some(caps) = &self.target_capabilities {
                                    caps.require_version(
                                        "cuda_tile.return inside loop",
                                        cutile_ir::requirements::Feature::LoopReturn.since(),
                                        &self.ir_location(&expr.span()),
                                    )
                                    .map_err(JITError::from)?;
                                }
                                if return_expr.expr.is_some() {
                                    return self.jit_error_result(
                                        &expr.span(),
                                        "returning a value from a kernel is not supported",
                                    );
                                }
                                let (op, _) =
                                    OpBuilder::new(Opcode::Return, self.ir_location(&expr.span()))
                                        .build(module);
                                append_op(module, block_id, op);
                                terminator_encountered = Some(BlockTerminator::Return);
                                break;
                            }
                            // A `return` below the function body has nothing to
                            // lower to: the enclosing block's terminator (a
                            // `yield`/`continue`) would be emitted instead and
                            // control would fall through to the statements after
                            // the `if`/loop — `if idx >= n { return; } p.load([idx])`
                            // performed the load (audit 2026-08). Reject it rather
                            // than miscompile.
                            if !is_fn_body {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "`return` is only supported at the top level of a function body; \
                                     inside `if`/`else`, a loop, or a nested block it cannot be \
                                     lowered (execution would fall through). Guard the remaining \
                                     statements with the condition instead, e.g. \
                                     `if idx < n { ... }`",
                                );
                            }
                            match &return_expr.expr {
                                Some(expr) => {
                                    return_value = self.compile_expression(
                                        module,
                                        block_id,
                                        expr,
                                        generic_args,
                                        ctx,
                                        return_type.clone(),
                                    )?;
                                }
                                None => return_value = None,
                            }
                            break;
                        }
                        _ => {
                            if is_last && semicolon.is_none() {
                                return_value = self.compile_expression(
                                    module,
                                    block_id,
                                    expr,
                                    generic_args,
                                    ctx,
                                    return_type.clone(),
                                )?;
                            } else {
                                self.compile_expression(
                                    module,
                                    block_id,
                                    expr,
                                    generic_args,
                                    ctx,
                                    None,
                                )?;
                            }
                        }
                    },
                    Stmt::Macro(macro_stmt) => {
                        self.compile_cuda_tile_macro(
                            module,
                            block_id,
                            &macro_stmt.mac,
                            generic_args,
                            ctx,
                            return_type.clone(),
                        )?;
                    }
                }
                // A constant-folded branch or bare block is emitted directly
                // into this block. Do not emit code after its kernel return.
                if module
                    .block(block_id)
                    .ops
                    .last()
                    .is_some_and(|id| module.op(*id).opcode == Opcode::Return)
                {
                    terminator_encountered = Some(BlockTerminator::Return);
                    break;
                }
            }
            if terminator_encountered.is_none() {
                let loop_carry_var_names = ctx.carry_vars.clone().unwrap_or(vec![]);
                match ctx.default_terminator {
                    Some(BlockTerminator::Yield) => {
                        let (cuda_tile_return_values, _) = {
                            if let Some(result) = &return_value {
                                // A tuple, array, or struct value has no single
                                // IR value to yield across the region boundary.
                                let Some(cuda_tile_value) = result.value else {
                                    return self.jit_error_result(
                                        &block_expr.span(),
                                        "this block produces a compound value (a tuple, array, or \
                                         struct) as the result of an `if`/`else`, which is not \
                                         supported; bind each component in its own `let` and \
                                         select them separately",
                                    );
                                };
                                (vec![cuda_tile_value], Some(result.ty.clone()))
                            } else {
                                (vec![], None)
                            }
                        };
                        let loop_carry_values = ctx.unpack_some_vars(&loop_carry_var_names)?;
                        let (op_id, _) =
                            OpBuilder::new(Opcode::Yield, self.ir_location(&block_expr.span()))
                                .operands(
                                    loop_carry_values
                                        .iter()
                                        .chain(cuda_tile_return_values.iter())
                                        .copied(),
                                )
                                .build(module);
                        append_op(module, block_id, op_id);
                    }
                    Some(BlockTerminator::Continue) => {
                        let loop_carry_values = ctx.unpack_some_vars(&loop_carry_var_names)?;
                        let (op_id, _) =
                            OpBuilder::new(Opcode::Continue, self.ir_location(&block_expr.span()))
                                .operands(loop_carry_values.iter().copied())
                                .build(module);
                        append_op(module, block_id, op_id);
                    }
                    Some(BlockTerminator::Return) => {
                        self.resolve_span(&block_expr.span())
                            .jit_assert(loop_carry_var_names.is_empty(), "unexpected state")?;
                        if return_value.is_some() {
                            return self.jit_error_result(
                                &block_expr.span(),
                                "returning a value from this function is not supported",
                            );
                        }
                        let (op_id, _) =
                            OpBuilder::new(Opcode::Return, self.ir_location(&block_expr.span()))
                                .build(module);
                        append_op(module, block_id, op_id);
                    }
                    Some(BlockTerminator::Break) => {
                        self.resolve_span(&block_expr.span())
                            .jit_error_result("unexpected default terminator type")?;
                    }
                    None => {}
                }
            }
            Ok(return_value)
        }) // stacker::maybe_grow
    }
}

fn local_pattern_type(pat: &Pat) -> Option<&syn::Type> {
    match pat {
        Pat::Type(pat_type) => Some(pat_type.ty.as_ref()),
        _ => None,
    }
}
