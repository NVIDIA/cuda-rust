/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Binary operation compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_binary_op.rs` — translates binary
//! arithmetic, comparison, and bitwise operations into tile-ir operations.
//! Only type and IR-emission changes; the dispatch logic and bounds
//! propagation are identical.

use quote::ToTokens;
use syn::spanned::Spanned;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue};
use super::shared_types::Kind;
use super::shared_utils::{get_tile_bop_from_rust_bop, TileBinaryOp};
use super::tile_rust_type::TileRustType;
use super::utils::{
    cmp_ordering_attr, cmp_pred_attr, flag_attr, rounding_mode_attr, rust_int_signedness,
    signedness_attr, NamedAttr,
};
use crate::error::JITError;
use crate::generics::GenericVars;
use crate::value_facts;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, BlockId, Module, ScalarType, TileElementType, TileType, Type};

use std::collections::HashMap;
use syn::ExprBinary;

/// Port of `get_cmp_predicate_attr` from `compiler/utils.rs`.
///
/// Returns a comparison-predicate named attribute for comparison binary ops,
/// or `None` for non-comparison ops.
fn get_cmp_predicate_attr_ir(expr: &TileBinaryOp) -> Result<Option<NamedAttr>, JITError> {
    match expr {
        TileBinaryOp::Eq => Ok(Some(cmp_pred_attr("equal"))),
        TileBinaryOp::Ne => Ok(Some(cmp_pred_attr("not_equal"))),
        TileBinaryOp::Lt => Ok(Some(cmp_pred_attr("less_than"))),
        TileBinaryOp::Le => Ok(Some(cmp_pred_attr("less_than_or_equal"))),
        TileBinaryOp::Gt => Ok(Some(cmp_pred_attr("greater_than"))),
        TileBinaryOp::Ge => Ok(Some(cmp_pred_attr("greater_than_or_equal"))),
        _ => Ok(None),
    }
}

/// Construct a tile-ir bool (i1) result type that mirrors the shape of `lhs_type`.
///
/// If `lhs_type` is a `Tile`, the result is a tile with the same shape but `I1`
/// element type. If it's a scalar, the result is `Scalar(I1)`.
fn make_bool_result_type(lhs_type: &Type) -> Type {
    match lhs_type {
        Type::Tile(tile_ty) => Type::Tile(TileType {
            shape: tile_ty.shape.clone(),
            element_type: TileElementType::Scalar(ScalarType::I1),
        }),
        _ => Type::Scalar(ScalarType::I1),
    }
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub fn compile_binary_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        bin_expr: &ExprBinary,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let tile_binary_op = get_tile_bop_from_rust_bop(&bin_expr.op)?;
        let is_comparison = matches!(
            tile_binary_op,
            TileBinaryOp::Eq
                | TileBinaryOp::Ne
                | TileBinaryOp::Lt
                | TileBinaryOp::Le
                | TileBinaryOp::Gt
                | TileBinaryOp::Ge
        );
        let is_logical = matches!(bin_expr.op, syn::BinOp::And(_) | syn::BinOp::Or(_));
        if is_logical {
            return self.compile_short_circuit_op(
                module,
                block_id,
                bin_expr,
                generic_vars,
                ctx,
                matches!(bin_expr.op, syn::BinOp::And(_)),
            );
        }
        let lhs_return_type = if is_comparison {
            None
        } else {
            return_type.clone()
        };
        let lhs = self.compile_expression(
            module,
            block_id,
            &bin_expr.left,
            generic_vars,
            ctx,
            lhs_return_type,
        )?;
        if lhs.is_none() {
            return self.jit_error_result(
                &bin_expr.left.span(),
                "failed to compile the left-hand side of this binary operation",
            );
        }
        let lhs = lhs.unwrap();
        let rhs_return_type = if is_comparison {
            Some(lhs.ty.clone())
        } else {
            return_type.clone().or_else(|| Some(lhs.ty.clone()))
        };
        let rhs = self.compile_expression(
            module,
            block_id,
            &bin_expr.right,
            generic_vars,
            ctx,
            rhs_return_type,
        )?;
        if rhs.is_none() {
            return self.jit_error_result(
                &bin_expr.right.span(),
                "failed to compile the right-hand side of this binary operation",
            );
        }
        let rhs = rhs.unwrap();
        Ok(Some(self.compile_binary_op_from_values(
            module,
            block_id,
            lhs,
            rhs,
            &tile_binary_op,
            generic_vars,
            ctx,
            return_type,
            &bin_expr.span(),
        )?))
    }

    /// `&&`/`||` operands must be scalar `bool`s: a primitive whose Rust type
    /// is `bool` and whose IR value is not a shaped tile (rustc already
    /// rejects tile operands; this guards the compiler's own callers).
    fn require_scalar_bool(
        &self,
        module: &Module,
        value: &TileRustValue,
        side: &str,
        op_name: &str,
        side_span: &proc_macro2::Span,
    ) -> Result<(), JITError> {
        let is_bool = matches!(&value.ty.rust_ty, syn::Type::Path(p) if p.path.is_ident("bool"));
        let is_scalar = value.value.is_some_and(
            |v| !matches!(module.value_type(v), Type::Tile(tile) if !tile.shape.is_empty()),
        );
        if value.kind != Kind::PrimitiveType || !is_bool || !is_scalar {
            return self.jit_error_result(
                side_span,
                &format!(
                    "the {side} operand of `{op_name}` must be a scalar `bool`, got `{}`; \
                     use `&`/`|` for element-wise tile logic",
                    value.ty.rust_ty.to_token_stream()
                ),
            );
        }
        Ok(())
    }

    /// Lowers `a && b` / `a || b` with Rust's short-circuit semantics.
    ///
    /// The right operand is compiled into the `then` (`&&`) or `else` (`||`)
    /// region of a `cuda_tile.if` on the left operand, so it executes only
    /// when the left operand does not already decide the result — exactly
    /// as in Rust. The former lowering evaluated both sides eagerly and
    /// combined them with `andi`/`ori`, which executed e.g. the `idx % n` in
    /// `n != 0 && idx % n == 0` for `n == 0` (audit 2026-08).
    ///
    /// A compile-time-known left operand folds the whole expression the way
    /// Rust's own evaluation would: the right side is compiled inline when it
    /// decides the result, and never compiled when it does not.
    #[allow(clippy::too_many_arguments)]
    fn compile_short_circuit_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        bin_expr: &ExprBinary,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        is_and: bool,
    ) -> Result<Option<TileRustValue>, JITError> {
        let op_name = if is_and { "&&" } else { "||" };
        let span = bin_expr.span();
        let bool_ty = syn::parse2::<syn::Type>("bool".parse().unwrap()).unwrap();
        let bool_tr_ty = self
            .compile_type(&bool_ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(&span, "failed to compile the `bool` type"))?;
        let Some(lhs) =
            self.compile_expression(module, block_id, &bin_expr.left, generic_vars, ctx, None)?
        else {
            return self.jit_error_result(
                &bin_expr.left.span(),
                &format!("failed to compile the left-hand side of `{op_name}`"),
            );
        };
        self.require_scalar_bool(module, &lhs, "left", op_name, &bin_expr.left.span())?;
        // Any assignment on the right would have to be threaded through the
        // conditional region as a carried value; nothing needs that today.
        let mutated = super::shared_utils::collect_mutated_variables_from_expr(&bin_expr.right)?;
        if let Some(name) = mutated.iter().next() {
            return self.jit_error_result(
                &bin_expr.right.span(),
                &format!(
                    "assignment to `{name}` inside the right-hand side of `{op_name}` is not \
                     supported; move it before the expression"
                ),
            );
        }
        let tile_op = if is_and {
            TileBinaryOp::BitAnd
        } else {
            TileBinaryOp::BitOr
        };
        // Compile-time-known left operand: fold exactly as Rust evaluates.
        if let Some(bounds) = lhs.bounds.filter(|b| b.is_exact()) {
            let lhs_true = bounds.start != 0;
            let rhs_decides = lhs_true == is_and;
            if !rhs_decides {
                // `false && b` / `true || b`: `b` is never evaluated.
                return Ok(Some(lhs));
            }
            let previous_region = ctx.token_update_in_region;
            ctx.token_update_in_region = true;
            let rhs =
                self.compile_expression(module, block_id, &bin_expr.right, generic_vars, ctx, None);
            ctx.token_update_in_region = previous_region;
            let Some(rhs) = rhs? else {
                return self.jit_error_result(
                    &bin_expr.right.span(),
                    &format!("failed to compile the right-hand side of `{op_name}`"),
                );
            };
            self.require_scalar_bool(module, &rhs, "right", op_name, &bin_expr.right.span())?;
            return Ok(Some(rhs));
        }
        let Some(cond) = lhs.value else {
            return self.jit_error_result(
                &bin_expr.left.span(),
                &format!("left-hand side of `{op_name}` did not produce a value"),
            );
        };
        // The region that evaluates `b`: `then` for `&&`, `else` for `||`.
        let (rhs_block_id, _) = cutile_ir::builder::build_block(module, &[]);
        let mut rhs_vars = ctx.clone();
        rhs_vars.token_update_in_region = true;
        rhs_vars.default_terminator = None;
        rhs_vars.carry_vars = None;
        let Some(rhs) = self.compile_expression(
            module,
            rhs_block_id,
            &bin_expr.right,
            generic_vars,
            &mut rhs_vars,
            None,
        )?
        else {
            return self.jit_error_result(
                &bin_expr.right.span(),
                &format!("failed to compile the right-hand side of `{op_name}`"),
            );
        };
        self.require_scalar_bool(module, &rhs, "right", op_name, &bin_expr.right.span())?;
        let rhs_value = rhs.value.expect("checked by require_scalar_bool");
        let result_ty = module.value_type(rhs_value).clone();
        let (rhs_yield, _) = OpBuilder::new(Opcode::Yield, self.ir_location(&span))
            .operand(rhs_value)
            .build(module);
        append_op(module, rhs_block_id, rhs_yield);
        let rhs_region = module.alloc_region(cutile_ir::ir::Region {
            blocks: vec![rhs_block_id],
        });
        // The region taken when `a` decides: yields the deciding constant
        // (`false` for `&&`, `true` for `||`).
        let (const_block_id, _) = cutile_ir::builder::build_block(module, &[]);
        let deciding = self.compile_bool_constant(module, const_block_id, generic_vars, !is_and)?;
        let deciding_value = deciding
            .value
            .ok_or_else(|| self.jit_error(&span, "failed to compile a `bool` constant"))?;
        let (const_yield, _) = OpBuilder::new(Opcode::Yield, self.ir_location(&span))
            .operand(deciding_value)
            .build(module);
        append_op(module, const_block_id, const_yield);
        let const_region = module.alloc_region(cutile_ir::ir::Region {
            blocks: vec![const_block_id],
        });
        let (then_region, else_region) = if is_and {
            (rhs_region, const_region)
        } else {
            (const_region, rhs_region)
        };
        let (if_op, results) = OpBuilder::new(Opcode::If, self.ir_location(&span))
            .operand(cond)
            .result(result_ty)
            .region(then_region)
            .region(else_region)
            .build(module);
        append_op(module, block_id, if_op);
        // Value facts: the interval lattice is the same truth table the eager
        // lowering had; the symbolic term does not apply to booleans.
        let facts =
            value_facts::transfer(&tile_op, &lhs, &rhs, value_facts::int_value_domain("bool"));
        let mut result = TileRustValue::new_value_kind_like(results[0], bool_tr_ty);
        result.bounds = facts.bounds;
        Ok(Some(result))
    }

    pub fn compile_binary_op_from_values(
        &self,
        module: &mut Module,
        block_id: BlockId,
        lhs: TileRustValue,
        rhs: TileRustValue,
        tile_rust_arithmetic_op: &TileBinaryOp,
        generic_vars: &GenericVars,
        _ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        if lhs.ty.rust_ty != rhs.ty.rust_ty {
            return self.jit_error_result(
                span,
                &format!(
                    "binary `{:?}` requires operands of the same type, but got `{}` and `{}`",
                    tile_rust_arithmetic_op,
                    lhs.ty.rust_ty.to_token_stream(),
                    rhs.ty.rust_ty.to_token_stream()
                ),
            );
        }
        let lhs_value = lhs.value;
        if lhs_value.is_none() {
            return self.jit_error_result(
                span,
                "left-hand side of binary operation did not produce a value",
            );
        }
        let lhs_value = lhs_value.unwrap();
        let rhs_value = rhs.value;
        if rhs_value.is_none() {
            return self.jit_error_result(
                span,
                "right-hand side of binary operation did not produce a value",
            );
        }
        let rhs_value = rhs_value.unwrap();
        let operand_type = lhs.ty.clone();
        let operand_rust_ty = &operand_type.rust_ty;
        let Some(operand_rust_element_type) =
            operand_type.get_instantiated_rust_element_type(self.modules.primitives())
        else {
            return self.jit_error_result(
                span,
                &format!(
                    "unable to determine element type for `{:?}` on `{}`",
                    tile_rust_arithmetic_op,
                    operand_type.rust_ty.to_token_stream()
                ),
            );
        };
        let Some(_operand_tile_ir_ty) = &operand_type.tile_ir_ty else {
            return self.jit_error_result(
                span,
                &format!(
                    "type `{}` cannot be used with binary `{:?}`",
                    operand_type.rust_ty.to_token_stream(),
                    tile_rust_arithmetic_op
                ),
            );
        };
        // For tile-ir, the result type for same-type operations comes from the
        // lhs value's type in the module (preserves tile shape).
        let operand_result_ty = module.value_type(lhs_value).clone();

        let Some(operand_cuda_tile_element_type) =
            operand_type.get_cuda_tile_element_type(self.modules.primitives())?
        else {
            return self.jit_error_result(
                span,
                &format!(
                    "unable to determine compiled element type for `{:?}`",
                    tile_rust_arithmetic_op
                ),
            );
        };
        let mut is_cmp = false;
        let signedness_str = rust_int_signedness(operand_rust_element_type.as_str());
        let sign_attr = signedness_attr("signedness", signedness_str);
        // Build the operation (allocates in module) but do NOT append to the
        // block yet. The old compiler defers `.build()` + `append_operation`
        // until after the exact-bounds early-return check, so we replicate that:
        // build now, check bounds, append only if we actually need the op.
        let (op_id, results) = match operand_cuda_tile_element_type.as_ref() {
            "i1" | "i4" | "i8" | "i32" | "i64" => {
                // TODO (hme): Add i4, i8, i16 support, as needed.
                if let Some(comparison_predicate) =
                    get_cmp_predicate_attr_ir(tile_rust_arithmetic_op)?
                {
                    is_cmp = true;
                    let bool_result_ty = make_bool_result_type(&operand_result_ty);
                    OpBuilder::new(Opcode::CmpI, self.ir_location(span))
                        .attr(comparison_predicate.0, comparison_predicate.1)
                        .attr(sign_attr.0, sign_attr.1)
                        .operand(lhs_value)
                        .operand(rhs_value)
                        .result(bool_result_ty)
                        .build(module)
                } else {
                    // If both operands have bounds, we can generate bounds on the output.
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => OpBuilder::new(Opcode::MinI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Max => OpBuilder::new(Opcode::MaxI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Add => OpBuilder::new(Opcode::AddI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Sub => OpBuilder::new(Opcode::SubI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Mul => OpBuilder::new(Opcode::MulI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Rem => OpBuilder::new(Opcode::RemI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Div => {
                            // Rust `/` truncates toward zero, and every analysis
                            // that models it (`bounds::Div`, the exact-range
                            // constant fold, `value_facts::FloorDiv`) uses Rust
                            // semantics, so the device op must round toward zero
                            // too. The former `negative_inf` (floor) lowering
                            // disagreed with the analysis for negative dividends
                            // and is rejected by the Tile IR verifier for
                            // unsigned operands. `RoundingMode::Zero == 1`.
                            // DivI uses "rounding" (not "rounding_mode") in bytecode
                            OpBuilder::new(Opcode::DivI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .attr(sign_attr.0, sign_attr.1)
                                .attr(
                                    "rounding",
                                    Attribute::i32(cutile_ir::ir::RoundingMode::Zero as i64),
                                )
                                .build(module)
                        }
                        TileBinaryOp::CeilDiv => {
                            // DivI uses "rounding" (not "rounding_mode") in bytecode
                            OpBuilder::new(Opcode::DivI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .attr(sign_attr.0, sign_attr.1)
                                .attr(
                                    "rounding",
                                    Attribute::i32(cutile_ir::ir::RoundingMode::PositiveInf as i64),
                                )
                                .build(module)
                        }
                        TileBinaryOp::BitAnd => {
                            OpBuilder::new(Opcode::AndI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
                        }
                        TileBinaryOp::BitOr => OpBuilder::new(Opcode::OrI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::BitXor => {
                            OpBuilder::new(Opcode::XOrI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
                        }
                        TileBinaryOp::Shl => OpBuilder::new(Opcode::ShLI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Shr => OpBuilder::new(Opcode::ShRI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        _ => {
                            return self.jit_error_result(
                                span,
                                &format!("Unimplemented binary op {tile_rust_arithmetic_op:#?}"),
                            );
                        }
                    }
                }
            }
            "bf16" | "f16" | "f32" | "f64" => {
                if let Some(comparison_predicate) =
                    get_cmp_predicate_attr_ir(tile_rust_arithmetic_op)?
                {
                    let comparison_ordering = cmp_ordering_attr("ordered");
                    is_cmp = true;
                    let bool_result_ty = make_bool_result_type(&operand_result_ty);
                    OpBuilder::new(Opcode::CmpF, self.ir_location(span))
                        .attr(comparison_predicate.0, comparison_predicate.1)
                        .attr(comparison_ordering.0, comparison_ordering.1)
                        .operand(lhs_value)
                        .operand(rhs_value)
                        .result(bool_result_ty)
                        .build(module)
                } else {
                    let default_rm_attr = rounding_mode_attr("nearest_even");
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => OpBuilder::new(Opcode::MinF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Max => OpBuilder::new(Opcode::MaxF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Add => OpBuilder::new(Opcode::AddF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Sub => OpBuilder::new(Opcode::SubF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Mul => OpBuilder::new(Opcode::MulF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Rem => OpBuilder::new(Opcode::RemF, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Div => OpBuilder::new(Opcode::DivF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::TrueDiv => {
                            let approx_rm_attr = rounding_mode_attr("approx");
                            let mut builder = OpBuilder::new(Opcode::DivF, self.ir_location(span))
                                .attr(approx_rm_attr.0, approx_rm_attr.1);
                            if operand_cuda_tile_element_type.as_str() == "f32" {
                                let ftz = flag_attr("flush_to_zero");
                                builder = builder.attr(ftz.0, ftz.1);
                            }
                            builder
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
                        }
                        _ => {
                            return self.jit_error_result(
                                span,
                                &format!("Unimplemented binary op {tile_rust_arithmetic_op:#?}"),
                            );
                        }
                    }
                }
            }
            _ => {
                return self.jit_error_result(
                    span,
                    &format!(
                        "Binary operation is not implemented for {}",
                        operand_rust_ty.to_token_stream()
                    ),
                );
            }
        };

        let return_type = match return_type {
            Some(rt) => rt,
            None => {
                // Try to infer from lhs/rhs.
                if is_cmp {
                    let bool_ty = syn::parse2::<syn::Type>("bool".parse().unwrap()).unwrap();
                    self.compile_type(&bool_ty, generic_vars, &HashMap::new())?
                        .unwrap()
                } else {
                    operand_type
                }
            }
        };

        // Value-fact transfer (interval + symbolic), gathered in `value_facts`.
        // Interval facts require primitive operands; the symbolic term still
        // propagates when only one side has a range (e.g. an affine index).
        let facts = if lhs.bounds.is_some() && rhs.bounds.is_some() {
            if !(lhs.kind == Kind::PrimitiveType && rhs.kind == Kind::PrimitiveType) {
                return self.jit_error_result(
                    span,
                    &format!(
                        "Expected PrimitiveType for binary op bounds, got lhs={:#?}, rhs={:#?}",
                        lhs.kind, rhs.kind
                    ),
                );
            }
            let result_domain = return_type
                .get_instantiated_rust_element_type(self.modules.primitives())
                .as_deref()
                .and_then(value_facts::int_value_domain);
            value_facts::transfer(tile_rust_arithmetic_op, &lhs, &rhs, result_domain)
        } else {
            value_facts::ScalarFacts {
                bounds: None,
                term: value_facts::propagate_term(tile_rust_arithmetic_op, &lhs, &rhs),
                floor_div: value_facts::propagate_floor_div(tile_rust_arithmetic_op, &lhs, &rhs),
            }
        };
        if let Some(bounds) = &facts.bounds {
            if bounds.is_exact() {
                // The lower/upper bounds are equivalent — emit a constant
                // instead. The op allocated above becomes dead (not appended
                // to any block).
                return self.compile_constant_from_exact_bounds(
                    module,
                    block_id,
                    *bounds,
                    return_type,
                );
            }
        }

        // Only now append the binary op to the block (mirrors the old
        // compiler which only calls `builder.append_operation` after the
        // bounds check).
        append_op(module, block_id, op_id);
        let value = results[0];
        let mut tr_value = TileRustValue::new_value_kind_like(value, return_type.clone());
        tr_value.bounds = facts.bounds;
        tr_value.term = facts.term;
        tr_value.floor_div = facts.floor_div;
        Ok(tr_value)
    }
}
