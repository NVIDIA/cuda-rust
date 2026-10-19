/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assumption compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_assume.rs` — translates assumption
//! function calls (e.g. `assume_div_by`, `assume_bounds`, `assume_same_elements_*`)
//! into tile-ir `Assume` operations. Only type and IR-emission changes; the
//! dispatch logic and validation are identical.

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue};
use super::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::GenericVars;
use crate::syn_utils::*;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, BlockId, Bounded, DivBy, Module, SameElements};

use syn::spanned::Spanned;
use syn::{Expr, ExprCall};

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Compiles an assumption function call into tile-ir.
    ///
    /// Assumption functions are compiler hints that provide optimization opportunities
    /// by asserting properties about values that the compiler can exploit. These
    /// functions use const-generic parameters to pass compile-time constants.
    ///
    /// ## Supported Assume Operations
    ///
    /// - `assume_div_by<DIVISOR>` - Value is divisible by DIVISOR
    /// - `assume_div_by_every_along<DIVISOR, EVERY, ALONG>` - Complex divisibility pattern
    /// - `assume_bounds_lower<LOWER>` - Value >= LOWER (inclusive)
    /// - `assume_bounds_upper<UPPER>` - Value <= UPPER (inclusive)
    /// - `assume_bounds<LOWER, UPPER>` - LOWER <= Value <= UPPER
    /// - `assume_same_elements_1d<GROUP0>` - Elements identical within groups (1D)
    /// - `assume_same_elements_2d<GROUP0, GROUP1>` - Elements identical within groups (2D)
    /// - `assume_same_elements_3d<GROUP0, GROUP1, GROUP2>` - Elements identical within groups (3D)
    /// - `assume_same_elements_4d<GROUP0, GROUP1, GROUP2, GROUP3>` - Elements identical within groups (4D)
    pub(crate) fn compile_assumption_call(
        &self,
        call_expr: &ExprCall,
        module: &mut Module,
        block_id: BlockId,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<TileRustValue, JITError> {
        let Expr::Path(path_expr) = &*call_expr.func else {
            return self.jit_error_result(
                &call_expr.func.span(),
                "expected a simple function path for assume invocation",
            );
        };
        let ident = get_ident_from_path_expr(path_expr);
        let compiler_op_function = ident.to_string();
        let mut args =
            self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
        let val = args.remove(0);
        let return_type = val.ty.clone();
        let Some(generic_args) = get_call_expression_generics(call_expr) else {
            return self.jit_error_result(
                &call_expr.span(),
                "`assume` requires generic arguments (e.g. `assume_bounds::<T, 0, 128>(...)`)",
            );
        };
        let predicate_args = get_generic_arg_ints::<i32>(&generic_args, Some(generic_vars));
        let Some(val_value) = val.value else {
            return self.jit_error_result(
                &call_expr.span(),
                "the first argument to `assume` must produce a value",
            );
        };
        self.compile_value_assumption(
            module,
            block_id,
            val_value,
            compiler_op_function.as_str(),
            &predicate_args,
            return_type,
            &call_expr.span(),
        )
    }

    /// Generates tile-ir assume operation with appropriate predicate attribute.
    pub(crate) fn compile_value_assumption(
        &self,
        module: &mut Module,
        block_id: BlockId,
        assume_val: cutile_ir::ir::Value,
        assume_op_rust_function: &str,
        predicate_args: &[i32],
        return_type: TileRustType,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        let predicate_attr = match assume_op_rust_function {
            "assume_div_by" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_div_by` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::DivBy(DivBy {
                    divisor: predicate_args[0] as u64,
                    every: None,
                    along: None,
                })
            }
            "assume_div_by_every_along" => {
                if predicate_args.len() != 3 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_div_by_every_along` requires 3 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::DivBy(DivBy {
                    divisor: predicate_args[0] as u64,
                    every: Some(predicate_args[1] as i64),
                    along: Some(predicate_args[2] as i64),
                })
            }
            "assume_bounds_lower" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds_lower` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::Bounded(Bounded {
                    lb: Some(predicate_args[0] as i64),
                    ub: None,
                })
            }
            "assume_bounds_upper" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds_upper` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::Bounded(Bounded {
                    lb: None,
                    ub: Some(predicate_args[0] as i64),
                })
            }
            "assume_bounds" => {
                if predicate_args.len() != 2 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds` requires 2 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::Bounded(Bounded {
                    lb: Some(predicate_args[0] as i64),
                    ub: Some(predicate_args[1] as i64),
                })
            }
            "assume_same_elements_1d" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_1d` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::SameElements(SameElements {
                    values: predicate_args.iter().map(|&v| v as i64).collect(),
                })
            }
            "assume_same_elements_2d" => {
                if predicate_args.len() != 2 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_2d` requires 2 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::SameElements(SameElements {
                    values: predicate_args.iter().map(|&v| v as i64).collect(),
                })
            }
            "assume_same_elements_3d" => {
                if predicate_args.len() != 3 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_3d` requires 3 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::SameElements(SameElements {
                    values: predicate_args.iter().map(|&v| v as i64).collect(),
                })
            }
            "assume_same_elements_4d" => {
                if predicate_args.len() != 4 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_4d` requires 4 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                Attribute::SameElements(SameElements {
                    values: predicate_args.iter().map(|&v| v as i64).collect(),
                })
            }
            _ => {
                return self.jit_error_result(
                    span,
                    &format!("unrecognized assume operation `{assume_op_rust_function}`"),
                );
            }
        };
        let result_ty = module.value_type(assume_val).clone();
        let (op_id, results) = OpBuilder::new(Opcode::Assume, self.ir_location(span))
            .operand(assume_val)
            .attr("predicate", predicate_attr)
            .result(result_ty)
            .build(module);
        append_op(module, block_id, op_id);
        let value = results[0];
        Ok(TileRustValue::new_value_kind_like(value, return_type))
    }
}
