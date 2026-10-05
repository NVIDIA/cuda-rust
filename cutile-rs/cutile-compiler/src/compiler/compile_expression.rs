/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expression compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_expression.rs` — translates Rust `syn::Expr`
//! AST nodes into tile-ir operations. Only type and IR-emission changes; the
//! control flow, dispatch logic, and variable binding are identical.

use super::_function::CUDATileFunctionCompiler;
use super::_value::{
    BlockTerminator, CompilerContext, DimOrigin, LoopFrame, LoopKind, PartitionAxisOrigin,
    TileRustValue,
};
use super::shared_types::Kind;
use super::shared_utils::{
    block_has_early_exit, collect_mutated_variables, collect_mutated_variables_from_block,
    collect_mutated_variables_from_expr, collect_mutated_variables_loop,
    collect_mutated_variables_while, dedup, update_outer_block_type_meta, TileBinaryOp,
    MAX_MAPPED_PARTITION_RANK, OWNED_MAP_DIM, STACK_GROW_SIZE, STACK_RED_ZONE,
};
use super::tile_rust_type::TileRustType;
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::generics::{
    get_cga_from_generic_argument, GenericVars, TypeInstance, TypeInstanceUserType,
};
use crate::passes::name_resolution::{DefKind, Res};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::builder::{append_op, build_block, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, Location, Module, Region, ScalarType, TileElementType, TileType,
    Type as TileIrType,
};

use quote::ToTokens;
use std::collections::{BTreeMap, HashMap};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{parse_quote, Expr, ExprForLoop, ExprMacro, Lit, Member, Pat, Token, UnOp};

/// A per-tensor ordering-token accumulator threaded through a loop. The loop
/// carries the tensor's token (a bare `token`, never the view); each store in
/// the body joins its output into it; the loop result is published to the tensor
/// at exit. Orders any later access to the tensor after all of the loop's writes.
pub(crate) struct LoopTokenAcc {
    /// Synthetic carry-var name holding the accumulator token.
    acc_var: String,
    /// The root tensor whose token the accumulator publishes.
    root: String,
    /// Views written in the body that root at `root`; rebound to the published
    /// token after the loop so a trailing store through them stays ordered.
    receivers: Vec<String>,
    /// True if this loop created the accumulator; false if it inherited one from
    /// an enclosing loop (nested loops writing the same tensor). Only the owner
    /// publishes the result to the tensor and drops the carry variable — a
    /// non-owner's result is left for the enclosing loop to carry further.
    owner: bool,
    /// True if some store to this tensor has a non-distinct index (a
    /// constant/repeated address, not varying with the loop variable). Those
    /// writes overlap, so the views are rebound to read the carried accumulator
    /// token — the stores serialize (each ordered after the previous). Distinct
    /// (branded) writes keep reading their invariant view token and fork.
    serialize: bool,
}

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Construct a ZST marker type placeholder from a path expression.
    ///
    /// Used for static_params like `ftz::Enabled`, `rounding::NearestEven`.
    /// These carry no tile-ir value — they're compile-time constants
    /// consumed by `resolve_static_params` during op compilation.
    fn make_zst_marker(path_expr: &syn::ExprPath) -> TileRustValue {
        let path_ty: syn::Type = syn::Type::Path(syn::TypePath {
            qself: None,
            path: path_expr.path.clone(),
        });
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: path_ty,
        });
        let ty = TileRustType::new_string(type_instance);
        TileRustValue::new_string(Expr::Path(path_expr.clone()), ty)
    }

    fn const_shape_macro_args(
        &self,
        mac_expr: &ExprMacro,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<Vec<String>, JITError> {
        let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
        let exprs = parser.parse2(mac_expr.mac.tokens.clone()).map_err(|err| {
            self.jit_error(
                &mac_expr.span(),
                &format!("failed to parse const-shape macro arguments: {err}"),
            )
        })?;
        let expr_count = exprs.len();
        let mut args = Vec::new();
        for expr in exprs {
            match &expr {
                Expr::Path(path) if path.path.segments.len() == 1 => {
                    let name = get_ident_from_path_expr(path).to_string();
                    if let Some(cga) = generic_vars.inst_array.get(&name) {
                        if expr_count != 1 {
                            return self.jit_error_result(
                                &expr.span(),
                                &format!(
                                    "`{name}` names a const generic array; use it alone or index it as `{name}[i]`"
                                ),
                            );
                        }
                        args.extend(cga.iter().map(|dim| dim.to_string()));
                        continue;
                    }
                    self.require_compile_time_shape_expr(&expr, generic_vars, ctx)?;
                    args.push(expr.to_token_stream().to_string());
                }
                Expr::Index(index) => {
                    if let Expr::Path(path) = index.expr.as_ref() {
                        let name = get_ident_from_path_expr(path).to_string();
                        if let Some(cga) = generic_vars.inst_array.get(&name) {
                            let i = parse_signed_literal_as_i32(&index.index);
                            let Some(dim) = cga.get(i as usize) else {
                                return self.jit_error_result(
                                    &index.index.span(),
                                    &format!(
                                        "index {i} out of bounds for const generic array `{name}` of length {}",
                                        cga.len()
                                    ),
                                );
                            };
                            args.push(dim.to_string());
                            continue;
                        }
                    }
                    return self.jit_error_result(
                        &expr.span(),
                        "only const generic array indexing like `S[0]` is supported in `const_shape!` and `const_array!`",
                    );
                }
                _ => {
                    self.require_compile_time_shape_expr(&expr, generic_vars, ctx)?;
                    args.push(expr.to_token_stream().to_string());
                }
            }
        }
        Ok(args)
    }

    fn require_compile_time_shape_expr(
        &self,
        expr: &Expr,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<(), JITError> {
        match expr {
            Expr::Lit(_) | Expr::Unary(_) => Ok(()),
            Expr::Path(path) if path.path.segments.len() == 1 => {
                let name = get_ident_from_path_expr(path).to_string();
                if generic_vars.get_i32(&name).is_some() {
                    return Ok(());
                }
                if ctx
                    .vars
                    .get(&name)
                    .and_then(|value| value.bounds)
                    .is_some_and(|bounds| bounds.is_exact())
                {
                    return Ok(());
                }
                let res = self
                    .modules
                    .name_resolver
                    .resolve_path(&path.path, &self.module_name);
                if let Res::Def(DefKind::Const, def_id) = res {
                    if self
                        .modules
                        .name_resolver
                        .get_const(&def_id)
                        .and_then(crate::type_aliases::const_item_scalar_expr)
                        .is_some()
                    {
                        return Ok(());
                    }
                }
                self.jit_error_result(
                    &expr.span(),
                    "all arguments to `const_shape!` must be compile-time constants",
                )
            }
            Expr::Paren(paren) => {
                self.require_compile_time_shape_expr(&paren.expr, generic_vars, ctx)
            }
            _ => self.jit_error_result(
                &expr.span(),
                "all arguments to `const_shape!` must be compile-time constants",
            ),
        }
    }

    fn make_option_type(rust_ty: syn::Type) -> TileRustType {
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: rust_ty,
        });
        TileRustType::new_enum(type_instance)
    }

    fn make_option_type_from_payload(payload_ty: &TileRustType) -> TileRustType {
        let payload_rust_ty = &payload_ty.rust_ty;
        let rust_ty: syn::Type = parse_quote!(Option<#payload_rust_ty>);
        Self::make_option_type(rust_ty)
    }

    fn path_looks_like_associated_const(
        &self,
        path_expr: &syn::ExprPath,
        generic_vars: &GenericVars,
    ) -> bool {
        if path_expr.qself.is_some() {
            return true;
        }
        if path_expr.path.segments.len() != 2 {
            return false;
        }
        let qualifier = path_expr.path.segments[0].ident.to_string();
        generic_vars.var_type(&qualifier).is_some()
            || self.modules.structs().contains_key(&qualifier)
            || self
                .modules
                .primitives()
                .keys()
                .any(|(_, self_name)| self_name == &qualifier)
    }

    fn expected_array_element_type(
        &self,
        expected: &TileRustType,
        generic_vars: &GenericVars,
    ) -> Result<Option<TileRustType>, JITError> {
        let elem_ty = match &expected.rust_ty {
            syn::Type::Array(array) => Some(&*array.elem),
            syn::Type::Slice(slice) => Some(&*slice.elem),
            _ => None,
        };
        let Some(elem_ty) = elem_ty else {
            return Ok(None);
        };
        self.compile_type(elem_ty, generic_vars, &HashMap::new())
    }

    fn compile_else_branch(
        &self,
        module: &mut Module,
        block_id: BlockId,
        else_expr: &Expr,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        match else_expr {
            Expr::Block(block_expr) => {
                self.compile_block(module, block_id, &block_expr.block, generic_vars, ctx, return_type)
            }
            Expr::If(_) => {
                let synthetic_block = syn::Block {
                    brace_token: Default::default(),
                    stmts: vec![syn::Stmt::Expr(else_expr.clone(), None)],
                };
                self.compile_block(module, block_id, &synthetic_block, generic_vars, ctx, return_type)
            }
            _ => self.jit_error_result(
                &else_expr.span(),
                "only block expressions (`{ ... }`) and chained `else if` expressions are supported in else branches",
            ),
        }
    }

    fn cga_type_arg(dims: &[i32]) -> String {
        let dims = dims
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{ [{dims}] }}")
    }

    pub(crate) fn mapped_partition_type_shapes(
        &self,
        value: &TileRustValue,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<(Vec<i32>, Vec<i32>), JITError> {
        let (type_ident, type_generic_args) = get_ident_generic_args(&value.ty.rust_ty);
        let Some(type_ident) = type_ident else {
            return self.jit_error_result(span, "expected a mapped partition type");
        };
        if !type_ident.to_string().starts_with("MappedPartitionMut") {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` for loops require a MappedPartitionMut receiver, got `{}`",
                    value.ty.rust_ty.to_token_stream()
                ),
            );
        }
        let Some(tile_shape_arg) = type_generic_args.args.iter().nth(1) else {
            return self.jit_error_result(
                span,
                "MappedPartitionMut is missing its tile-shape generic argument",
            );
        };
        let Some(map_shape_arg) = type_generic_args.args.iter().nth(2) else {
            return self.jit_error_result(
                span,
                "MappedPartitionMut is missing its map-shape generic argument",
            );
        };
        let Some(tile_shape) = get_cga_from_generic_argument(tile_shape_arg, generic_vars) else {
            return self.jit_error_result(
                span,
                "failed to resolve MappedPartitionMut tile-shape const generic array",
            );
        };
        let Some(map_shape) = get_cga_from_generic_argument(map_shape_arg, generic_vars) else {
            return self.jit_error_result(
                span,
                "failed to resolve MappedPartitionMut map-shape const generic array",
            );
        };
        if tile_shape.len() != map_shape.len() {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` requires matching tile and map ranks, got tile rank {} and map rank {}",
                    tile_shape.len(),
                    map_shape.len()
                ),
            );
        }
        if tile_shape.is_empty() || tile_shape.len() > MAX_MAPPED_PARTITION_RANK {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` supports rank-1 through rank-{MAX_MAPPED_PARTITION_RANK} MappedPartitionMut values, got rank {}",
                    tile_shape.len(),
                ),
            );
        }
        Ok((tile_shape, map_shape))
    }

    fn compile_i32_type(
        &self,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<TileRustType, JITError> {
        self.compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(span, "failed to compile i32 type"))
    }

    fn scalar_i32_ir_type() -> TileIrType {
        TileIrType::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::I32),
        })
    }

    fn compile_tile_block_tuple(
        &self,
        module: &mut Module,
        block_id: BlockId,
        opcode: Opcode,
        i32_ty: &TileRustType,
        span: &proc_macro2::Span,
    ) -> Vec<TileRustValue> {
        let is_tile_block_id = matches!(opcode, Opcode::GetTileBlockId);
        let is_num_tile_blocks = matches!(opcode, Opcode::GetNumTileBlocks);
        let scalar_i32_ty = Self::scalar_i32_ir_type();
        let mut op_builder = OpBuilder::new(opcode, self.ir_location(span));
        for _ in 0..3 {
            op_builder = op_builder.result(scalar_i32_ty.clone());
        }
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        results
            .into_iter()
            .enumerate()
            .map(|(axis, value)| {
                let bounds = self.const_grid.and_then(|const_grid| {
                    let axis_size = match axis {
                        0 => const_grid.0,
                        1 => const_grid.1,
                        2 => const_grid.2,
                        _ => unreachable!(),
                    } as i64;
                    if is_num_tile_blocks {
                        Some(Bounds::exact(axis_size))
                    } else if is_tile_block_id && axis_size > 0 {
                        Some(Bounds::new(0, axis_size - 1))
                    } else {
                        None
                    }
                });
                let mut result = TileRustValue::new_primitive(value, i32_ty.clone(), bounds);
                // Tag the special-register value with its canonical atom so a
                // partition-access obligation can discharge against the universal
                // `TileBlockId(k) < NumTileBlocks(k)` hardware axiom.
                if is_tile_block_id {
                    result.term = Some(cuda_async::predicate::Term::atom(
                        cuda_async::predicate::Atom::TileBlockId(axis),
                    ));
                } else if is_num_tile_blocks {
                    result.term = Some(cuda_async::predicate::Term::atom(
                        cuda_async::predicate::Atom::NumTileBlocks(axis),
                    ));
                }
                result
            })
            .collect()
    }

    fn compile_index_space_shape_values(
        &self,
        module: &mut Module,
        block_id: BlockId,
        partition_value: &TileRustValue,
        i32_ty: &TileRustType,
        span: &proc_macro2::Span,
    ) -> Result<Vec<TileRustValue>, JITError> {
        let view_value = partition_value.value.ok_or_else(|| {
            self.jit_error(span, "expected a direct value for mapped partition indices")
        })?;
        let view_ty = module.value_type(view_value).clone();
        let TileIrType::PartitionView(pv) = &view_ty else {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` expects a mapped partition view, got `{:?}`",
                    view_ty
                ),
            );
        };
        let rank = pv.tile_shape.len();
        if rank == 0 || rank > MAX_MAPPED_PARTITION_RANK {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` supports rank-1 through rank-{MAX_MAPPED_PARTITION_RANK} partitions, got rank {rank}"
                ),
            );
        }

        let scalar_i32_ty = Self::scalar_i32_ir_type();
        let mut op_builder =
            OpBuilder::new(Opcode::GetIndexSpaceShape, self.ir_location(span)).operand(view_value);
        for _ in 0..rank {
            op_builder = op_builder.result(scalar_i32_ty.clone());
        }
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);

        let mut values = Vec::with_capacity(rank);
        for axis in 0..rank {
            let mut value = TileRustValue::new_primitive(results[axis], i32_ty.clone(), None);
            let parent_axis = pv.dim_map.get(axis).copied().ok_or_else(|| {
                self.jit_error(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} is missing from partition dim_map {:?}",
                        pv.dim_map
                    ),
                )
            })?;
            if parent_axis < 0 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} maps to invalid parent axis {parent_axis}"
                    ),
                );
            }
            let parent_axis = parent_axis as usize;
            let Some(&parent_dim) = pv.tensor_view.shape.get(parent_axis) else {
                return self.jit_error_result(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} maps to parent axis {parent_axis}, but parent tensor rank is {}",
                        pv.tensor_view.shape.len()
                    ),
                );
            };
            let tile_dim = pv.tile_shape[axis] as i64;
            if tile_dim <= 0 {
                return self.jit_error_result(
                    span,
                    &format!("`iter_indices()` axis {axis} has invalid tile dimension {tile_dim}"),
                );
            }
            if parent_dim >= 0 {
                let num_tiles = (parent_dim + tile_dim - 1) / tile_dim;
                value.bounds = Some(Bounds::exact(num_tiles));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn simple_path_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path)
                if path.qself.is_none()
                    && path.path.leading_colon.is_none()
                    && path.path.segments.len() == 1 =>
            {
                Some(path.path.segments[0].ident.to_string())
            }
            Expr::Paren(paren) => Self::simple_path_name(&paren.expr),
            _ => None,
        }
    }

    fn is_dim_new_call(func: &Expr) -> bool {
        let Expr::Path(path) = func else {
            return false;
        };
        path.path.segments.len() == 2
            && path.path.segments[0].ident == "Dim"
            && path.path.segments[1].ident == "new"
    }

    fn compile_dim_new_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &syn::ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 1 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`Dim::new` expects 1 argument, got {}",
                    call_expr.args.len()
                ),
            );
        }
        let i32_type = self
            .compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(&call_expr.span(), "failed to compile i32 type"))?;
        let mut value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[0],
                generic_vars,
                ctx,
                Some(i32_type),
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.args[0].span(),
                    "failed to compile dimension size",
                )
            })?;
        let value_id = value.value.ok_or_else(|| {
            self.jit_error(
                &call_expr.args[0].span(),
                "dimension size must compile to a scalar value",
            )
        })?;
        value.dim_origin = Some(DimOrigin::Value(value_id));
        let dim_type = match return_type {
            Some(return_type) => return_type,
            None => self
                .compile_type(&parse_quote!(Dim), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(&call_expr.span(), "failed to compile Dim type"))?,
        };
        let dim_origin = value.dim_origin.clone();
        let mut fields = BTreeMap::new();
        fields.insert("size".to_string(), value);
        let mut dim = TileRustValue::new_struct(fields, dim_type);
        dim.dim_origin = dim_origin;
        Ok(Some(dim))
    }

    fn wrap_scalar_as_dim(
        &self,
        mut value: TileRustValue,
        generic_vars: &GenericVars,
        return_type: Option<TileRustType>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        let value_id = value
            .value
            .ok_or_else(|| self.jit_error(span, "dimension size must compile to a scalar value"))?;
        if value.dim_origin.is_none() {
            value.dim_origin = Some(DimOrigin::Value(value_id));
        }
        let dim_type = match return_type {
            Some(return_type) => return_type,
            None => self
                .compile_type(&parse_quote!(Dim), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(span, "failed to compile Dim type"))?,
        };
        let dim_origin = value.dim_origin.clone();
        let bounds = value.bounds;
        let mut fields = BTreeMap::new();
        fields.insert("size".to_string(), value);
        let mut dim = TileRustValue::new_struct(fields, dim_type);
        dim.dim_origin = dim_origin;
        dim.bounds = bounds;
        Ok(dim)
    }

    fn compile_into_dim_method(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &syn::ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.method != "into_dim" {
            return Ok(None);
        }
        if !method_call.args.is_empty() {
            return self.jit_error_result(
                &method_call.args.span(),
                "`IntoDim::into_dim` does not take arguments",
            );
        }
        let receiver = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile IntoDim receiver",
                )
            })?;
        if get_type_ident(&receiver.ty.rust_ty).is_some_and(|ident| ident == "Dim") {
            return Ok(Some(receiver));
        }
        Ok(Some(self.wrap_scalar_as_dim(
            receiver,
            generic_vars,
            return_type,
            &method_call.span(),
        )?))
    }

    fn compile_partition_with_bounds_method(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &syn::ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.method != "with_bounds" {
            return Ok(None);
        }
        if method_call.args.len() != 1 {
            return self.jit_error_result(
                &method_call.args.span(),
                "`Partition::with_bounds` expects exactly one tuple argument",
            );
        }
        let partition = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile Partition::with_bounds receiver",
                )
            })?;
        let bounds = self
            .compile_expression(
                module,
                block_id,
                &method_call.args[0],
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.args[0].span(),
                    "failed to compile Partition::with_bounds tuple",
                )
            })?;
        let Some(bound_values) = bounds.values else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "`Partition::with_bounds` expects a tuple of dimensions",
            );
        };
        self.apply_with_bounds(
            module,
            block_id,
            partition,
            bound_values,
            return_type,
            generic_vars,
            ctx,
            &method_call.args[0].span(),
        )
        .map(Some)
    }

    /// Bind a `Dim` per axis to `partition`, verifying each binding and
    /// retyping the result as the matching `Bounded*` partition.
    ///
    /// `with_bounds` is reachable two ways — as a method call, intercepted
    /// before inlining, and as the `partition_with_bounds` compiler op, which
    /// is `pub` and so callable directly. Both spellings must mean exactly the
    /// same thing, so both route here. They previously carried separate copies
    /// of this logic that had already drifted (differing diagnostics, and only
    /// one of them was ever exercised).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_with_bounds(
        &self,
        module: &mut Module,
        block_id: BlockId,
        mut partition: TileRustValue,
        bound_values: Vec<TileRustValue>,
        return_type: Option<TileRustType>,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        let (static_tile, _, _) = self.partition_static_geometry(&partition, span)?;
        if bound_values.len() != static_tile.len() {
            return self.jit_error_result(
                span,
                &format!(
                    "`Partition::with_bounds` expects rank-{} bounds for this partition, got rank {}",
                    static_tile.len(),
                    bound_values.len()
                ),
            );
        }
        let mut dim_origins = Vec::with_capacity(bound_values.len());
        // The partition's index-space shape is one op serving every axis. Built
        // on first use (a binding that folds at JIT never needs it) and shared
        // across the rest, so a rank-n `with_bounds` costs at most one query
        // rather than n.
        let mut index_space_shape: Option<Vec<cutile_ir::ir::Value>> = None;
        for (axis, value) in bound_values.into_iter().enumerate() {
            let Some(origin) = Self::value_dim_origin(&value) else {
                return self.jit_error_result(
                    span,
                    &format!(
                        "`Partition::with_bounds` bound {axis} must come from `num_tiles`, `Dim::new`, or `IntoDim::into_dim`"
                    ),
                );
            };
            // The binding is a claim; verify it (see the helper's doc).
            self.emit_with_bounds_binding_check(
                module,
                block_id,
                &partition,
                &value,
                axis,
                generic_vars,
                ctx,
                &mut index_space_shape,
                span,
            )?;
            dim_origins.push(origin);
        }
        let return_type = match return_type {
            Some(return_type) => return_type,
            None => {
                let mut bounded_ty = partition.ty.rust_ty.clone();
                let syn::Type::Path(path_ty) = &mut bounded_ty else {
                    return self.jit_error_result(
                        span,
                        "expected a partition type for `Partition::with_bounds`",
                    );
                };
                let Some(segment) = path_ty.path.segments.last_mut() else {
                    return self.jit_error_result(
                        span,
                        "expected a partition type path for `Partition::with_bounds`",
                    );
                };
                // Derive the bounded name from the receiver's own ident rather
                // than naming one target: `PartitionMut` must become
                // `BoundedPartitionMut`, not `BoundedPartition`.
                segment.ident =
                    syn::Ident::new(&format!("Bounded{}", segment.ident), segment.ident.span());
                let mut return_type = partition.ty.clone();
                return_type.rust_ty = bounded_ty;
                return_type
            }
        };
        partition.ty = return_type;
        partition.bounded_axes = Some(dim_origins);
        Ok(partition)
    }

    /// Reduce a `with_bounds` binding to a divisibility predicate, when the
    /// bound `Dim` is `floor(e / t)` over exactly the extent and tile extent
    /// this axis is partitioned by.
    ///
    /// The binding obliges `val(m) == tiles(P, axis) == ceil(e / t)`. With
    /// `val(m) == floor(e / t)`, the two agree precisely when `t` divides `e`,
    /// so the whole obligation collapses to `divisible_by(e, t)` — stated over
    /// the extent atom that already exists, with no new vocabulary.
    ///
    /// Returns `None` whenever any part of that shape is not established: a
    /// different divisor, an unlabelled extent, or a numerator the compiler
    /// cannot relate to this axis's extent. The caller then falls through to
    /// the device assert — fail closed (SC4). The extent atom's frame comes
    /// from [`Self::extent_atom`] (SC1).
    fn with_bounds_divisibility(
        &self,
        partition: &TileRustValue,
        bound: &TileRustValue,
        dim_map: &[i32],
        axis: usize,
        tile_dim: i64,
    ) -> Option<cuda_async::predicate::Predicate> {
        use crate::passes::obligation::{resolve, Obligation, Resolution};
        use cuda_async::predicate::{Predicate, Term};
        // The `Dim`'s scalar lives in a `size` field; look there too.
        let floor_div = bound.floor_div.as_ref().or_else(|| {
            bound
                .fields
                .as_ref()
                .and_then(|fields| fields.get("size"))
                .and_then(|size| size.floor_div.as_ref())
        })?;
        // The division must be by this axis's tile extent, or it says nothing
        // about this axis's tile count.
        if floor_div.divisor != tile_dim {
            return None;
        }
        // ...and the numerator must be this axis's extent.
        let param = *self.param_index.get(partition.tensor_origin.as_ref()?)?;
        let tensor_axis = *dim_map.get(axis).filter(|&&d| d >= 0)? as usize;
        let expected = Term::atom(self.extent_atom(param, tensor_axis));
        if floor_div.numerator == expected {
            return Predicate::divisible_by(expected, tile_dim);
        }
        // Cross-tensor binding (Theorem 2): the `Dim` divides a *different*
        // tensor's extent — the GEMM contraction pattern, `k` derived from `x`
        // but also bound to an axis of `y`. The reduction still applies when
        // the two extents are equal, and that equality is precisely what a
        // declared `preconditions` dim fact states — already verified by the
        // launcher before the kernel runs. So the binding discharges from the
        // *declared* equality plus the numerator's own divisibility check (the
        // very predicate the sibling binding emits): one check decides both.
        //
        // Entailed-only, deliberately. An *unproven* equality here also ranges
        // over launch-known operands, so `resolve` would happily hoist it — but
        // extent equality is strictly stronger than tile-count equality
        // (`ceil(100/64) == ceil(128/64)` with unequal extents), so hoisting it
        // unasked would reject launches today's device assert accepts. The
        // kernel must opt in by declaring the fact; otherwise fail closed.
        let equality = Obligation::new(
            Predicate::eq(&floor_div.numerator, &expected)?,
            "with_bounds shared-extent binding",
        );
        if matches!(resolve(&equality, &self.assumptions), Resolution::Jit) {
            return Predicate::divisible_by(floor_div.numerator.clone(), tile_dim);
        }
        None
    }

    /// Verify one `with_bounds` binding at run time: the declared `Dim` must
    /// equal this partition's tile count for `axis`.
    ///
    /// `with_bounds` binds a symbolic extent (a `Dim`) to `(partition, axis)`.
    /// Binding the *same* `Dim` to several partitions is how a kernel states
    /// that those axes share an extent — a GEMM binds one `Dim` to `x`'s axis 1
    /// and `y`'s axis 0 to name the contraction dimension. So a binding is a
    /// **claim**, and nothing else verifies it: unchecked, a wrong `Dim`
    /// discharges every branded access on that axis and admits an out-of-bounds
    /// access from safe code (both on stores and on loads).
    ///
    /// Emitted once per binding at the `with_bounds` site — loop-invariant in
    /// practice — never per access. The cost is therefore O(bindings), not
    /// O(accesses): the per-access checks stay discharged by the brand, so the
    /// register win on hot loops is unaffected.
    ///
    /// Static and launch-resolvable bindings are handled before this fallback.
    /// The remaining dynamic forms deliberately keep this device comparison:
    /// `with_bounds` is deprecated, and preserving its safety is more valuable
    /// than expanding the solver solely to optimize its residual cases.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_with_bounds_binding_check(
        &self,
        module: &mut Module,
        block_id: BlockId,
        partition: &TileRustValue,
        bound: &TileRustValue,
        axis: usize,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        index_space_shape: &mut Option<Vec<cutile_ir::ir::Value>>,
        span: &proc_macro2::Span,
    ) -> Result<(), JITError> {
        // The declared extent: the `Dim`'s runtime value.
        let declared = bound.value.or_else(|| {
            bound
                .fields
                .as_ref()
                .and_then(|fields| fields.get("size"))
                .and_then(|size| size.value)
        });
        let (Some(declared), Some(view_value)) = (declared, partition.value) else {
            return Ok(());
        };
        let view_ty = module.value_type(view_value).clone();
        let cutile_ir::ir::Type::PartitionView(pv) = &view_ty else {
            return Ok(());
        };
        let rank = pv.tile_shape.len();
        if axis >= rank {
            return Ok(());
        }
        // Discharge at JIT where possible — this is the zero-cost path.
        //
        // FRAME INVARIANT (SC2, and the reason this is sound): the declared
        // parameter shape may stand in for the view extent only for parameter
        // kinds where the two coincide. That holds for every kind that can reach
        // here, because `with_bounds` exists solely on `Partition` and
        // `PartitionMut`: an immutable `&Tensor` has view == root, and a
        // `&mut Tensor`'s declared shape *is* its per-CTA slab. It does NOT hold
        // for a `MappedPartitionMut`, whose declared shape is the *tile* shape —
        // if `with_bounds` is ever added to that type, this fold becomes wrong
        // and must first gate on the parameter kind. See
        // `.internal/tasks/in_progress/check_hoisting/BOUNDS_HOISTING_ANALYSIS.md` §SC1 amendment.
        //
        // The *declared* parameter shape is enforced at launch (the generated
        // launcher asserts `valid_shape == given_shape`), so the compiler may
        // trust it as this view's extent. That matters because the partition
        // view type carries `-1` for a `&mut Tensor` param by design, while the
        // validator still knows the declared shape. If both that extent and the
        // declared `Dim` are known, the binding is decided at compile time and
        // costs nothing at run time.
        let tile_dim = pv.tile_shape[axis] as i64;
        let declared_extent = self.declared_view_extent(partition, &pv.dim_map, axis);
        let declared_const = bound
            .bounds
            .filter(|b| b.is_exact())
            .map(|b| b.start)
            .or_else(|| {
                bound
                    .fields
                    .as_ref()
                    .and_then(|fields| fields.get("size"))
                    .and_then(|size| size.bounds)
                    .filter(|b| b.is_exact())
                    .map(|b| b.start)
            });
        if let (Some(extent), Some(dim_value)) = (declared_extent, declared_const) {
            if extent >= 0 && tile_dim > 0 {
                let expected = (extent as i64 + tile_dim - 1) / tile_dim;
                if dim_value == expected {
                    return Ok(());
                }
                return self.jit_error_result(
                    span,
                    &format!(
                        "`with_bounds` bound {axis} is {dim_value}, but this partition has {expected} tiles on axis {axis}"
                    ),
                );
            }
        }
        // Launch rung. Neither side folded, so the binding is about to cost a
        // device assert. But when the bound `Dim` is `floor(e / t)` for the very
        // extent and tile this axis is partitioned by, the obligation
        // `val(m) == ceil(e / t)` is exactly `t` divides `e` — a predicate the
        // existing vocabulary can state over the extent atom itself.
        //
        // Reducing to divisibility rather than naming the division as an atom is
        // what keeps this discharge-able: a declared `preconditions` fact
        // asserting the divisibility entails it, so `resolve` can settle it at
        // `Jit` and emit nothing. An opaque division atom would be evaluable but
        // unrelatable, forcing a host check even for a kernel that proved the
        // fact. (Same lever MLIR uses to simplify floordiv/ceildiv.)
        //
        // FRAME (SC1): the extent atom resolves against the host's *root* shape
        // array, and `label_param_extents` only labels immutable parameters, so
        // a slabbed `&mut Tensor` never reaches here — its `floor_div` carries
        // no term. Fail closed to the device assert below (SC4).
        if let Some(divisibility) =
            self.with_bounds_divisibility(partition, bound, &pv.dim_map, axis, tile_dim)
        {
            let cause = format!(
                "`with_bounds` bound {axis}: the partitioned extent must be divisible by the tile \
                 extent {tile_dim}, or the declared bound undercounts the tiles on axis {axis}"
            );
            if self.lower_obligation(divisibility, cause) {
                return Ok(());
            }
        }
        // Constant rung, launch tier: a constant bound `c` on a dynamic-extent
        // axis obliges `ceil(e / t) == c`, which is the pair of linear
        // inequalities `(c-1)·t < e ≤ c·t` — both over the extent atom, both
        // launch-known. The atom's frame comes from `extent_atom` (SC1): the
        // root extent for an immutable param, the slab extent for a `&mut`.
        if let (Some(c), Some(&param)) = (
            declared_const,
            partition
                .tensor_origin
                .as_ref()
                .and_then(|name| self.param_index.get(name)),
        ) {
            let tensor_axis = pv.dim_map.get(axis).copied().filter(|&d| d >= 0);
            if let (Some(tensor_axis), true, true) = (tensor_axis, c >= 1, tile_dim >= 1) {
                use cuda_async::predicate::{Predicate, Term};
                let e = Term::atom(self.extent_atom(param, tensor_axis as usize));
                // `(c-1)·t < e` and `e < c·t + 1`; bail to the device assert on
                // arithmetic overflow rather than weaken either side (the upper
                // bound must stay ≤-inclusive or the exact-fit extent — the
                // common case — would be rejected).
                let bounds_pair = (|| {
                    let lower = (c - 1).checked_mul(tile_dim)?;
                    let upper_plus_one = c.checked_mul(tile_dim)?.checked_add(1)?;
                    let above = Predicate::lt(&Term::constant(lower), &e)?;
                    let not_above = Predicate::lt(&e, &Term::constant(upper_plus_one))?;
                    Some((above, not_above))
                })();
                if let Some((above, not_above)) = bounds_pair {
                    let cause = format!(
                        "`with_bounds` bound {axis} declares {c} tiles of extent {tile_dim}, so \
                         axis {axis}'s extent must be in ({}, {}]",
                        (c - 1) * tile_dim,
                        c * tile_dim
                    );
                    if self.lower_obligation(above, cause.clone())
                        && self.lower_obligation(not_above, cause)
                    {
                        return Ok(());
                    }
                }
            }
        }
        // This partition's real tile count on `axis`, read off the one
        // index-space query shared by every axis of this binding.
        let shape_results = match index_space_shape {
            Some(cached) => cached,
            slot => {
                let i32_scalar_ty = cutile_ir::ir::Type::Tile(TileType {
                    shape: vec![],
                    element_type: TileElementType::Scalar(ScalarType::I32),
                });
                let mut op_builder =
                    OpBuilder::new(Opcode::GetIndexSpaceShape, self.ir_location(span))
                        .operand(view_value);
                for _ in 0..rank {
                    op_builder = op_builder.result(i32_scalar_ty.clone());
                }
                let (shape_op, results) = op_builder.build(module);
                append_op(module, block_id, shape_op);
                slot.insert(results)
            }
        };
        let actual = shape_results[axis];

        let i32_tr_type = self
            .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(span, "failed to synthesize `i32` type"))?;
        let eq = self.compile_binary_op_from_values(
            module,
            block_id,
            TileRustValue::new_primitive(declared, i32_tr_type.clone(), None),
            TileRustValue::new_primitive(actual, i32_tr_type, None),
            &TileBinaryOp::Eq,
            generic_vars,
            ctx,
            None,
            span,
        )?;
        // Statically decidable: discharge at JIT and emit nothing (the common
        // case when the view's extent and the declared `Dim` are both known), or
        // reject outright when the bound is provably wrong.
        if let Some(bounds) = eq.bounds {
            if bounds.is_exact() {
                if bounds.start != 0 {
                    return Ok(());
                }
                return self.jit_error_result(
                    span,
                    &format!(
                        "`with_bounds` bound {axis} does not equal this partition's tile count on axis {axis}"
                    ),
                );
            }
        }
        let Some(eq_result) = eq.value else {
            return Ok(());
        };
        self.deny_residual_check(
            &format!("the `with_bounds` binding check for axis {axis}"),
            "Declare the extent statically in the signature, or bind a `Dim` the \
             compiler can relate to this partition's tile count",
            span,
        )?;
        let (assert_op, _) = OpBuilder::new(Opcode::Assert, self.ir_location(span))
            .attr(
                "message",
                Attribute::String(format!(
                    "`with_bounds` bound {axis} does not equal this partition's tile count on axis {axis}"
                )),
            )
            .operand(eq_result)
            .build(module);
        append_op(module, block_id, assert_op);
        Ok(())
    }

    fn value_dim_origin(value: &TileRustValue) -> Option<DimOrigin> {
        value.dim_origin.clone().or_else(|| {
            value
                .fields
                .as_ref()
                .and_then(|fields| fields.get("size"))
                .and_then(|size| size.dim_origin.clone())
        })
    }

    /// The tensor-named axis provenance of a loop bound, looking through a
    /// `Dim` wrapper to the scalar it wraps (mirroring
    /// [`Self::value_dim_origin`], which does the same for the view-valued
    /// form).
    fn value_partition_axis_origin(value: &TileRustValue) -> Option<PartitionAxisOrigin> {
        value.partition_axis_origin.clone().or_else(|| {
            value
                .fields
                .as_ref()
                .and_then(|fields| fields.get("size"))
                .and_then(|size| size.partition_axis_origin.clone())
        })
    }

    fn dim_size_value(
        &self,
        dim_value: &TileRustValue,
        span: &proc_macro2::Span,
    ) -> Result<cutile_ir::ir::Value, JITError> {
        if let Some(value) = dim_value.value {
            return Ok(value);
        }
        let Some(fields) = dim_value.fields.as_ref() else {
            return self.jit_error_result(span, "dimension value is missing its scalar size");
        };
        let Some(size) = fields.get("size") else {
            return self.jit_error_result(span, "dimension value is missing its `size` field");
        };
        size.value
            .ok_or_else(|| self.jit_error(span, "dimension size must compile to a scalar value"))
    }

    /// Special lowering for `for idx in mapped_partition.iter_indices()`.
    ///
    /// This is intentionally separate from normal range-loop lowering. The
    /// iterator is a DSL proof boundary: the compiler lowers it to a persistent
    /// flat tile-id loop and mints `PartitionIndex` values branded with the
    /// mapped partition that produced them.
    fn try_compile_mapped_partition_indices_for_loop(
        &self,
        module: &mut Module,
        block_id: BlockId,
        for_expr: &ExprForLoop,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<bool, JITError> {
        let Expr::MethodCall(method_call) = &*for_expr.expr else {
            return Ok(false);
        };
        let method_name = method_call.method.to_string();
        let (has_ranges, is_shared) = match method_name.as_str() {
            "iter_indices" => (false, false),
            "iter_indices_with" => (false, true),
            "iter_indices_within" => (true, false),
            "iter_indices_within_with" => (true, true),
            _ => return Ok(false),
        };
        let expected_args = usize::from(has_ranges) + usize::from(is_shared);
        if method_call.args.len() != expected_args {
            return self.jit_error_result(
                &method_call.args.span(),
                &format!(
                    "MappedPartitionMut::{method_name} expects {expected_args} argument(s), got {}",
                    method_call.args.len()
                ),
            );
        }
        let ranges_arg = has_ranges.then(|| &method_call.args[0]);
        let other_arg_index = usize::from(has_ranges);
        let Pat::Ident(iterand_ident) = &*for_expr.pat else {
            return self.jit_error_result(
                &for_expr.pat.span(),
                "MappedPartitionMut::iter_indices loops must bind a simple index variable",
            );
        };

        let partition_value = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile mapped partition receiver",
                )
            })?;
        let partition_origin = partition_value.value.ok_or_else(|| {
            self.jit_error(
                &method_call.receiver.span(),
                "mapped partition receiver did not produce a direct value",
            )
        })?;
        let (tile_shape, map_shape) = self.mapped_partition_type_shapes(
            &partition_value,
            generic_vars,
            &method_call.receiver.span(),
        )?;
        let rank = tile_shape.len();
        // Owned axes (map dim OWNED = 0) are not traversed by the stream: each
        // stream item owns the axis's full extent (subtensor-per-CTA
        // exclusivity), and in-kernel loops traverse it with bounds-proven
        // indices. Only the streamed axes participate in the flat tile-id
        // schedule.
        if let Some(axis) = map_shape.iter().position(|&dim| dim < 0) {
            return self.jit_error_result(
                &method_call.receiver.span(),
                &format!(
                    "mapped partition map dimensions must be OWNED (0) or positive: axis {axis} is {}",
                    map_shape[axis]
                ),
            );
        }
        let streamed_axes: Vec<usize> = (0..rank)
            .filter(|&axis| map_shape[axis] != OWNED_MAP_DIM)
            .collect();
        if streamed_axes.is_empty() {
            return self.jit_error_result(
                &method_call.receiver.span(),
                "mapped partition requires at least one streamed (non-OWNED) map axis",
            );
        }
        let has_owned_axes = streamed_axes.len() != rank;
        if has_owned_axes && has_ranges {
            return self.jit_error_result(
                &method_call.args.span(),
                "sub-range iteration over a map with OWNED axes is not supported yet",
            );
        }
        let i32_ty = self.compile_i32_type(generic_vars, &for_expr.span())?;
        let index_space = self.compile_index_space_shape_values(
            module,
            block_id,
            &partition_value,
            &i32_ty,
            &method_call.receiver.span(),
        )?;

        // Shared maps: a second mapped partition whose stores accept the same
        // indices. The type already forces equal tile/map shapes; the logical
        // partition grids must also match so every minted index is in bounds
        // for both partitions. Static grids are compared here; dynamic grids
        // get a runtime assert per axis.
        let mut extra_origins: Vec<cutile_ir::ir::Value> = vec![];
        if is_shared {
            let other_arg = &method_call.args[other_arg_index];
            let other_expr: &Expr = match other_arg {
                Expr::Reference(reference) => &reference.expr,
                other => other,
            };
            let other_value = self
                .compile_expression(module, block_id, other_expr, generic_vars, ctx, None)?
                .ok_or_else(|| {
                    self.jit_error(
                        &other_arg.span(),
                        "failed to compile shared mapped partition argument",
                    )
                })?;
            let other_origin = other_value.value.ok_or_else(|| {
                self.jit_error(
                    &other_arg.span(),
                    "shared mapped partition argument did not produce a direct value",
                )
            })?;
            let (other_tile_shape, other_map_shape) =
                self.mapped_partition_type_shapes(&other_value, generic_vars, &other_arg.span())?;
            if other_tile_shape != tile_shape || other_map_shape != map_shape {
                return self.jit_error_result(
                    &other_arg.span(),
                    "iter_indices_with requires both mapped partitions to share tile and map shapes",
                );
            }
            let other_index_space = self.compile_index_space_shape_values(
                module,
                block_id,
                &other_value,
                &i32_ty,
                &other_arg.span(),
            )?;
            // Declared `dim(a, i) == dim(b, i)` preconditions discharge the
            // per-axis grid-equality asserts: equal tensor extents with
            // type-equal tile shapes (checked above) give equal tile grids.
            let receiver_name = Self::simple_path_name(&method_call.receiver)
                .or_else(|| partition_value.tensor_origin.clone());
            let other_name =
                Self::simple_path_name(other_expr).or_else(|| other_value.tensor_origin.clone());
            for axis in 0..rank {
                let lhs = &index_space[axis];
                let rhs = &other_index_space[axis];
                if let (Some(lhs_bounds), Some(rhs_bounds)) = (lhs.bounds, rhs.bounds) {
                    if lhs_bounds.is_exact() && rhs_bounds.is_exact() {
                        if lhs_bounds.start != rhs_bounds.start {
                            return self.jit_error_result(
                                &other_arg.span(),
                                &format!(
                                    "iter_indices_with requires equal logical partition grids: axis {axis} has {} vs {} tiles",
                                    lhs_bounds.start, rhs_bounds.start
                                ),
                            );
                        }
                        continue;
                    }
                }
                if let (Some(receiver_name), Some(other_name)) = (&receiver_name, &other_name) {
                    if self.resolve_dim_eq(receiver_name, axis, other_name, axis) {
                        self.check_stats
                            .discharged
                            .set(self.check_stats.discharged.get() + 1);
                        continue;
                    }
                }
                let eq_value = self.compile_binary_op_from_values(
                    module,
                    block_id,
                    lhs.clone(),
                    rhs.clone(),
                    &TileBinaryOp::Eq,
                    generic_vars,
                    ctx,
                    None,
                    &other_arg.span(),
                )?;
                let eq_result = eq_value.value.ok_or_else(|| {
                    self.jit_error(
                        &other_arg.span(),
                        "failed to compile shared mapped partition grid comparison",
                    )
                })?;
                let message = format!(
                    "shared mapped partitions require equal logical partition grids: axis {axis}"
                );
                self.deny_residual_check(
                    &format!("the shared mapped-partition grid check for axis {axis}"),
                    "Declare the two tensors' extents equal with a `preconditions` fact so \
                     the equality discharges at compile time or at launch",
                    &other_arg.span(),
                )?;
                let (assert_op_id, _) =
                    OpBuilder::new(Opcode::Assert, self.ir_location(&other_arg.span()))
                        .attr("message", Attribute::String(message))
                        .operand(eq_result)
                        .build(module);
                append_op(module, block_id, assert_op_id);
            }
            extra_origins.push(other_origin);
        }

        // Sub-range iteration: per-axis (start_tile, num_tiles) pairs narrow
        // the traversed grid. The schedule below runs over the effective
        // lengths and the axis starts are added to the minted coordinates.
        // Each range is validated against the grid — statically when bounds
        // prove it, otherwise with a runtime assert — so indices keep the
        // in-bounds store proof.
        let mut axis_starts: Vec<Option<TileRustValue>> = vec![None; rank];
        let mut axis_lens: Vec<TileRustValue> = index_space.clone();
        // Tensor-axis provenance means "this coordinate ranges over the
        // source tensor's whole axis" and can therefore justify a
        // cross-tensor `tiles(source) <= tiles(target)` obligation. A
        // validated sub-range still proves stores into its mapped partition,
        // but its coordinates range over only `[start, start + len)`: using
        // the full source axis there is conservative yet needlessly rejects
        // consumers such as a `[heads, max_seq, d]` cache updated from a
        // `[seq, heads, d]` source. Track the two proofs separately.
        let mut axis_covers_full_grid = vec![true; rank];
        if let Some(ranges_expr) = ranges_arg {
            let ranges_value = self
                .compile_expression(module, block_id, ranges_expr, generic_vars, ctx, None)?
                .ok_or_else(|| {
                    self.jit_error(
                        &ranges_expr.span(),
                        "failed to compile mapped partition sub-range argument",
                    )
                })?;
            let Some(range_tuples) = &ranges_value.values else {
                return self.jit_error_result(
                    &ranges_expr.span(),
                    "sub-range iteration expects an array of (start_tile, num_tiles) pairs",
                );
            };
            if range_tuples.len() != rank {
                return self.jit_error_result(
                    &ranges_expr.span(),
                    &format!(
                        "sub-range iteration expects {rank} (start_tile, num_tiles) pairs, got {}",
                        range_tuples.len()
                    ),
                );
            }
            for axis in 0..rank {
                let Some(parts) = &range_tuples[axis].values else {
                    return self.jit_error_result(
                        &ranges_expr.span(),
                        &format!("sub-range axis {axis} must be a (start_tile, num_tiles) pair"),
                    );
                };
                if parts.len() != 2 {
                    return self.jit_error_result(
                        &ranges_expr.span(),
                        &format!("sub-range axis {axis} must be a (start_tile, num_tiles) pair"),
                    );
                }
                let start = parts[0].clone();
                let len = parts[1].clone();
                let num_bid = index_space[axis].clone();

                let start_exact = start.bounds.filter(|bounds| bounds.is_exact());
                let len_exact = len.bounds.filter(|bounds| bounds.is_exact());
                let grid_exact = num_bid.bounds.filter(|bounds| bounds.is_exact());

                // Reject provably-bad ranges at compile time.
                if let Some(bounds) = start_exact {
                    if bounds.start < 0 {
                        return self.jit_error_result(
                            &ranges_expr.span(),
                            &format!("sub-range axis {axis} start {} is negative", bounds.start),
                        );
                    }
                }
                if let Some(bounds) = len_exact {
                    if bounds.start < -1 {
                        return self.jit_error_result(
                            &ranges_expr.span(),
                            &format!(
                                "sub-range axis {axis} length {} is negative (-1 means the rest of the axis)",
                                bounds.start
                            ),
                        );
                    }
                }

                // `num_tiles = -1`: the rest of the axis from `start_tile`.
                // The resulting length is in bounds by construction, so only
                // the start needs validation.
                let is_remainder = len_exact.is_some_and(|bounds| bounds.start == -1);
                axis_covers_full_grid[axis] = start_exact.is_some_and(|bounds| bounds.start == 0)
                    && (is_remainder
                        || matches!(
                            (len_exact, grid_exact),
                            (Some(len_bounds), Some(grid_bounds))
                                if len_bounds.start == grid_bounds.start
                        ));
                let len = if is_remainder {
                    self.compile_binary_op_from_values(
                        module,
                        block_id,
                        num_bid.clone(),
                        start.clone(),
                        &TileBinaryOp::Sub,
                        generic_vars,
                        ctx,
                        None,
                        &ranges_expr.span(),
                    )?
                } else {
                    len
                };

                let emit_range_assert = |compiler: &Self,
                                         module: &mut Module,
                                         ctx: &mut CompilerContext,
                                         lhs: TileRustValue,
                                         rhs: TileRustValue,
                                         op: TileBinaryOp,
                                         detail: &str|
                 -> Result<(), JITError> {
                    let cmp = compiler.compile_binary_op_from_values(
                        module,
                        block_id,
                        lhs,
                        rhs,
                        &op,
                        generic_vars,
                        ctx,
                        None,
                        &ranges_expr.span(),
                    )?;
                    let cmp_value = cmp.value.ok_or_else(|| {
                        compiler.jit_error(
                            &ranges_expr.span(),
                            "failed to compile sub-range bounds comparison",
                        )
                    })?;
                    let message =
                        format!("mapped partition sub-range out of bounds: axis {axis}: {detail}");
                    compiler.deny_residual_check(
                        &format!("the mapped-partition sub-range check for axis {axis} ({detail})"),
                        "Use compile-time-constant sub-range bounds so the check discharges \
                         at compile time",
                        &ranges_expr.span(),
                    )?;
                    let (assert_op_id, _) =
                        OpBuilder::new(Opcode::Assert, compiler.ir_location(&ranges_expr.span()))
                            .attr("message", Attribute::String(message))
                            .operand(cmp_value)
                            .build(module);
                    append_op(module, block_id, assert_op_id);
                    Ok(())
                };

                // start >= 0, unless proven above.
                if start_exact.is_none() {
                    let zero = self.compile_constant(module, block_id, generic_vars, 0)?;
                    emit_range_assert(
                        self,
                        module,
                        ctx,
                        start.clone(),
                        zero,
                        TileBinaryOp::Ge,
                        "start is negative",
                    )?;
                }
                if is_remainder {
                    // Remainder length: assert start <= grid so the length is
                    // non-negative, unless both are static.
                    let statically_ok = match (start_exact, grid_exact) {
                        (Some(start_bounds), Some(grid_bounds)) => {
                            if start_bounds.start > grid_bounds.start {
                                return self.jit_error_result(
                                    &ranges_expr.span(),
                                    &format!(
                                        "sub-range axis {axis} start {} exceeds the {}-tile grid",
                                        start_bounds.start, grid_bounds.start
                                    ),
                                );
                            }
                            true
                        }
                        // `(0, -1)` spells "the whole axis": in bounds for any
                        // grid, no assert needed.
                        (Some(start_bounds), None) => start_bounds.start == 0,
                        _ => false,
                    };
                    if !statically_ok {
                        emit_range_assert(
                            self,
                            module,
                            ctx,
                            start.clone(),
                            num_bid.clone(),
                            TileBinaryOp::Le,
                            "start exceeds the logical grid",
                        )?;
                    }
                } else {
                    // Explicit length: len >= 0 and start + len <= grid,
                    // unless both are static.
                    if len_exact.is_none() {
                        let zero = self.compile_constant(module, block_id, generic_vars, 0)?;
                        emit_range_assert(
                            self,
                            module,
                            ctx,
                            len.clone(),
                            zero,
                            TileBinaryOp::Ge,
                            "length is negative (the -1 rest-of-axis spelling must be a literal)",
                        )?;
                    }
                    let statically_ok = match (start_exact, len_exact, grid_exact) {
                        (Some(start_bounds), Some(len_bounds), Some(grid_bounds)) => {
                            if start_bounds.start + len_bounds.start > grid_bounds.start {
                                return self.jit_error_result(
                                    &ranges_expr.span(),
                                    &format!(
                                        "sub-range axis {axis} [{}, {}) exceeds the {}-tile grid",
                                        start_bounds.start,
                                        start_bounds.start + len_bounds.start,
                                        grid_bounds.start
                                    ),
                                );
                            }
                            true
                        }
                        _ => false,
                    };
                    if !statically_ok {
                        // Overflow-free form of `start + len <= grid`:
                        // `start <= grid - len`. Both operands of the
                        // subtraction are asserted nonnegative above, so the
                        // difference is representable in i32, whereas the sum
                        // wraps for `start = i32::MAX, len = 1` — and the
                        // wrapped value passed this assert and got the
                        // resulting coordinate branded as proven (issue #214).
                        let grid_minus_len = self.compile_binary_op_from_values(
                            module,
                            block_id,
                            num_bid.clone(),
                            len.clone(),
                            &TileBinaryOp::Sub,
                            generic_vars,
                            ctx,
                            None,
                            &ranges_expr.span(),
                        )?;
                        emit_range_assert(
                            self,
                            module,
                            ctx,
                            start.clone(),
                            grid_minus_len,
                            TileBinaryOp::Le,
                            "range end exceeds the logical grid",
                        )?;
                    }
                }

                // A statically-zero start needs no coordinate offset.
                if start_exact.is_none_or(|bounds| bounds.start != 0) {
                    axis_starts[axis] = Some(start);
                }
                axis_lens[axis] = len;
            }
        }

        // The persistent loop runs over the streamed tile count only; owned
        // axes contribute no work items.
        let mut total_tiles = axis_lens[streamed_axes[0]].clone();
        for &axis in streamed_axes.iter().skip(1) {
            total_tiles = self.compile_binary_op_from_values(
                module,
                block_id,
                total_tiles,
                axis_lens[axis].clone(),
                &TileBinaryOp::Mul,
                generic_vars,
                ctx,
                None,
                &for_expr.span(),
            )?;
        }
        let total_tiles_value = total_tiles.value.ok_or_else(|| {
            self.jit_error(
                &for_expr.span(),
                "failed to compute mapped partition tile count",
            )
        })?;

        let pid = self.compile_tile_block_tuple(
            module,
            block_id,
            Opcode::GetTileBlockId,
            &i32_ty,
            &for_expr.span(),
        );
        let grid = self.compile_tile_block_tuple(
            module,
            block_id,
            Opcode::GetNumTileBlocks,
            &i32_ty,
            &for_expr.span(),
        );
        let lower_bound = pid[0]
            .value
            .ok_or_else(|| self.jit_error(&for_expr.span(), "failed to compute tile-block id"))?;
        let step = grid[0].value.ok_or_else(|| {
            self.jit_error(&for_expr.span(), "failed to compute tile-block grid size")
        })?;

        let mut loop_carry_vars = collect_mutated_variables(for_expr)?
            .into_iter()
            .collect::<Vec<_>>();
        // Thread each written tensor's ordering token across the loop boundary.
        // The iterand (a branded `PartitionIndex`) makes every mapped store
        // distinct per iteration, so these fork.
        let persistent_loop_var = match &*for_expr.pat {
            syn::Pat::Ident(p) => Some(p.ident.to_string()),
            _ => None,
        };
        let token_accumulators = self.setup_loop_token_accumulators(
            &for_expr.body,
            persistent_loop_var.as_deref(),
            ctx,
            generic_vars,
        )?;
        for acc in &token_accumulators {
            loop_carry_vars.push(acc.acc_var.clone());
        }
        let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
        let loop_carry_arg_tys = loop_carry_args
            .iter()
            .map(|val| module.value_type(*val).clone())
            .collect::<Vec<_>>();

        let for_iterand_type = Self::scalar_i32_ir_type();
        let loop_block_arg_tys = [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
        let value_watermark = module.num_values() as u32;
        let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

        let mut for_variables = ctx.clone();
        let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
        for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
        self.bind_serialized_views(&token_accumulators, &mut for_variables);
        for_variables.carry_vars = Some(loop_carry_vars.clone());
        for_variables.default_terminator = Some(BlockTerminator::Continue);
        for_variables.innermost_loop = Some(LoopKind::For);
        for_variables.inside_for = true;
        // Persistent tile-id loop: the step is the physical grid size, so
        // induction-substitution hoisting stays off (unit_step = false), but
        // loop-invariant checks in the body can still move to the preheader.
        for_variables.loop_frames.push(LoopFrame {
            preheader_block: block_id,
            body_block: loop_block_id,
            value_watermark,
            induction_values: vec![loop_block_args[0]],
            lower: lower_bound,
            upper: total_tiles_value,
            unit_step: false,
            // Non-unit step: `[lower, upper - 1]` is not the induction range.
            induction_range: None,
            known_non_empty: false,
            has_early_exit: block_has_early_exit(&for_expr.body),
        });

        let tile_id = TileRustValue::new_primitive(loop_block_args[0], i32_ty.clone(), None);

        let tile_shape_arg = Self::cga_type_arg(&tile_shape);
        let index_ty: syn::Type = syn::parse_str(&format!("PartitionIndex<{tile_shape_arg}>"))
            .map_err(|err| {
                self.jit_error(
                    &for_expr.span(),
                    &format!("failed to build mapped partition index type: {err}"),
                )
            })?;
        let index_return_ty = self
            .compile_type(&index_ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| {
                self.jit_error(&for_expr.span(), "failed to compile PartitionIndex type")
            })?;
        // The schedule decomposes the flat tile id over the streamed axes
        // only; owned axes take a constant 0 coordinate (the base of the
        // owned subtensor), so the all-minted store path remains valid.
        let streamed_lens: Vec<TileRustValue> = streamed_axes
            .iter()
            .map(|&axis| axis_lens[axis].clone())
            .collect();
        let streamed_map: Vec<i32> = streamed_axes.iter().map(|&axis| map_shape[axis]).collect();
        let streamed_bids = self.emit_mapped_partition_schedule(
            module,
            loop_block_id,
            &tile_id,
            &streamed_lens,
            &streamed_map,
            generic_vars,
            &mut for_variables,
            &for_expr.span(),
        )?;
        let mut streamed_bids = streamed_bids.into_iter();
        let mut bids: Vec<TileRustValue> = Vec::with_capacity(rank);
        for axis in 0..rank {
            if map_shape[axis] != OWNED_MAP_DIM {
                bids.push(streamed_bids.next().ok_or_else(|| {
                    self.jit_error(
                        &for_expr.span(),
                        "mapped partition schedule produced too few streamed coordinates",
                    )
                })?);
            } else {
                let zero = self.compile_constant(module, loop_block_id, generic_vars, 0i32)?;
                bids.push(zero);
            }
        }
        // Sub-range iteration: offset the minted coordinates by the axis
        // starts. Bounds propagate through the add, so downstream proofs see
        // [start, start + len - 1].
        for (axis, start) in axis_starts.iter().enumerate() {
            let Some(start) = start else {
                continue;
            };
            bids[axis] = self.compile_binary_op_from_values(
                module,
                loop_block_id,
                start.clone(),
                bids[axis].clone(),
                &TileBinaryOp::Add,
                generic_vars,
                &mut for_variables,
                None,
                &for_expr.span(),
            )?;
        }
        let coords_ty = self.array_i32_type(rank, generic_vars, &for_expr.span())?;
        let coords = TileRustValue::new_compound(bids, coords_ty);
        let mut index_fields = BTreeMap::new();
        index_fields.insert("coords".to_string(), coords);
        let mut index_value = TileRustValue::new_struct(index_fields, index_return_ty);
        let mut origins = vec![partition_origin];
        origins.extend(extra_origins);
        index_value.partition_origins = Some(origins.clone());
        let index_tensor_origin = Self::simple_path_name(&method_call.receiver)
            .or_else(|| partition_value.tensor_origin.clone());
        if let Some(fields) = index_value.fields.as_mut() {
            if let Some(coords) = fields.get_mut("coords") {
                if let Some(values) = coords.values.as_mut() {
                    for (axis, value) in values.iter_mut().enumerate() {
                        // Per-axis branding: each projected component records
                        // the minting stream (all shared origins) and its
                        // axis, so composite store indices can prove
                        // streamed-axis provenance component-wise.
                        value.index_origin = Some(DimOrigin::PartitionAxis {
                            view: partition_origin,
                            axis,
                            tile_dim: tile_shape[axis],
                        });
                        value.partition_origins = Some(origins.clone());
                        // Whole-axis provenance is stronger than the private
                        // mapped-store brand above. A narrowed axis cannot
                        // inherit it: downstream cross-tensor loads must test
                        // the actual sub-range coordinate instead.
                        if axis_covers_full_grid[axis] {
                            if let Some(tensor_origin) = &index_tensor_origin {
                                value.partition_axis_origin = Some(PartitionAxisOrigin {
                                    tensor: tensor_origin.clone(),
                                    axis,
                                    tile_dim: tile_shape[axis],
                                });
                            }
                        }
                    }
                }
            }
        }
        for_variables
            .vars
            .insert(iterand_ident.ident.to_string(), index_value);

        self.compile_block(
            module,
            loop_block_id,
            &for_expr.body,
            generic_vars,
            &mut for_variables,
            return_type,
        )?;

        let region_id = module.alloc_region(Region {
            blocks: vec![loop_block_id],
        });
        let (for_op_id, result_values) =
            OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                .operands([lower_bound, total_tiles_value, step].iter().copied())
                .operands(loop_carry_args.iter().copied())
                .results(loop_carry_arg_tys.iter().cloned())
                .region(region_id)
                .build(module);
        append_op(module, block_id, for_op_id);

        if result_values.len() != loop_carry_args.len() {
            return self.jit_error_result(
                &for_expr.span(),
                &format!(
                    "mapped partition indices loop produces {} results but {} mutable variables are carried across iterations",
                    result_values.len(),
                    loop_carry_args.len()
                ),
            );
        }
        ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
        self.finish_loop_token_accumulators(&token_accumulators, ctx);
        Ok(true)
    }

    /// Special lowering for `for idx in dim`.
    ///
    /// A `Dim` is the source-level iterable proof object. Iterating it lowers
    /// to `for idx in 0..dim`, while the loop variable is tagged as an index
    /// produced by that dimension. Plain `i32` values, including `num_tiles`
    /// results, continue through normal range lowering.
    /// Set up per-tensor ordering-token accumulators for a loop body: for each
    /// tensor written in the body, a synthetic carry variable holding the
    /// tensor's token, seeded from its current value. Threaded as a loop iter-arg
    /// (a bare `token`, never the view — a `partition_view` is not a valid
    /// iter-arg), joined per store (see `accumulate_loop_token`), and published
    /// to the tensor at exit (see `finish_loop_token_accumulators`). This is the
    /// loop-scope case of resource token threading: a view roots at its tensor,
    /// and the tensor's token must survive the loop boundary so a later access
    /// (a second `partition_mut`, a store after the loop) is ordered after the
    /// loop's writes.
    fn setup_loop_token_accumulators(
        &self,
        body: &syn::Block,
        loop_var: Option<&str>,
        ctx: &mut CompilerContext,
        generic_vars: &GenericVars,
    ) -> Result<Vec<LoopTokenAcc>, JITError> {
        use std::collections::BTreeMap;
        let stores = super::shared_utils::collect_store_calls(body, loop_var);
        // Group the written views by the root tensor they store to, tracking
        // whether every store to that tensor is distinct per iteration.
        struct RootInfo {
            receivers: Vec<String>,
            all_distinct: bool,
        }
        let mut by_root: BTreeMap<String, RootInfo> = BTreeMap::new();
        for store in &stores {
            let Some(view) = ctx.vars.get(&store.receiver) else {
                continue;
            };
            let Some(root) = view.tensor_origin.clone() else {
                continue;
            };
            let has_token = ctx
                .vars
                .get(&root)
                .and_then(|t| t.type_meta.as_ref())
                .and_then(|m| m.fields.get("token"))
                .and_then(|f| f.value)
                .is_some();
            if has_token {
                let entry = by_root.entry(root).or_insert(RootInfo {
                    receivers: Vec::new(),
                    all_distinct: true,
                });
                if !entry.receivers.contains(&store.receiver) {
                    entry.receivers.push(store.receiver.clone());
                }
                entry.all_distinct &= store.index_distinct;
            }
        }
        if by_root.is_empty() {
            return Ok(vec![]);
        }
        let token_ty = self.token_type(generic_vars)?;
        let mut accumulators = Vec::new();
        for (root, info) in by_root {
            let acc_var = format!("__loop_token_acc__{root}");
            let serialize = !info.all_distinct;
            if ctx.vars.contains_key(&acc_var) {
                // An enclosing loop already threads this tensor's token; carry
                // the same accumulator through this loop so its writes compose
                // into it. The enclosing (owner) loop publishes the final result.
                accumulators.push(LoopTokenAcc {
                    acc_var,
                    root,
                    receivers: info.receivers,
                    owner: false,
                    serialize,
                });
            } else {
                let seed = ctx.vars[&root]
                    .type_meta
                    .as_ref()
                    .and_then(|m| m.fields.get("token"))
                    .and_then(|f| f.value)
                    .expect("root tensor token checked above");
                ctx.vars.insert(
                    acc_var.clone(),
                    TileRustValue::new_primitive(seed, token_ty.clone(), None),
                );
                accumulators.push(LoopTokenAcc {
                    acc_var,
                    root,
                    receivers: info.receivers,
                    owner: true,
                    serialize,
                });
            }
        }
        Ok(accumulators)
    }

    /// For accumulators whose writes overlap (serialize), rebind the receiver
    /// views' tokens to the carried accumulator so the stores read it and chain
    /// (each ordered after the previous). Called after the loop block args are
    /// bound. Distinct (fork) accumulators are left alone — their stores keep
    /// reading their invariant view token.
    fn bind_serialized_views(
        &self,
        accumulators: &[LoopTokenAcc],
        for_variables: &mut CompilerContext,
    ) {
        for acc in accumulators {
            if !acc.serialize {
                continue;
            }
            let Some(acc_token) = for_variables.vars.get(&acc.acc_var).and_then(|v| v.value) else {
                continue;
            };
            for receiver in &acc.receivers {
                super::shared_utils::set_view_token(receiver, acc_token, for_variables);
            }
        }
    }

    /// Join a store's output token into its tensor's loop accumulator, if the
    /// enclosing loop is threading one. Called at the method-inline boundary
    /// after the store advanced the view's token. The stores themselves keep
    /// reading their (loop-invariant) view token — they fork, disjoint writes
    /// stay parallel — while the accumulator collects `join(acc, output)` so the
    /// loop result dominates every write.
    pub(crate) fn accumulate_loop_token(
        &self,
        module: &mut Module,
        block_id: BlockId,
        ctx: &mut CompilerContext,
        view_var: &str,
        generic_vars: &GenericVars,
    ) -> Result<(), JITError> {
        let Some(view) = ctx.vars.get(view_var) else {
            return Ok(());
        };
        let Some(root) = view.tensor_origin.clone() else {
            return Ok(());
        };
        let acc_var = format!("__loop_token_acc__{root}");
        let Some(acc_token) = ctx.vars.get(&acc_var).and_then(|v| v.value) else {
            return Ok(());
        };
        let Some(view_token) = view
            .type_meta
            .as_ref()
            .and_then(|m| m.fields.get("token"))
            .and_then(|f| f.value)
        else {
            return Ok(());
        };
        let token_ty = self.token_type(generic_vars)?;
        let (op_id, results) = OpBuilder::new(Opcode::JoinTokens, Location::Unknown)
            .result(TileIrType::Token)
            .operand(acc_token)
            .operand(view_token)
            .build(module);
        append_op(module, block_id, op_id);
        ctx.vars.insert(
            acc_var,
            TileRustValue::new_primitive(results[0], token_ty, None),
        );
        Ok(())
    }

    /// Publish each loop accumulator's result (repacked from the loop op) to its
    /// tensor and to the views written in the body, then drop the synthetic
    /// carry variables. After this, a later `partition_mut` of the tensor or a
    /// store through one of those views is ordered after the loop's writes.
    fn finish_loop_token_accumulators(
        &self,
        accumulators: &[LoopTokenAcc],
        ctx: &mut CompilerContext,
    ) {
        for acc in accumulators {
            // A non-owner inherited the accumulator from an enclosing loop: leave
            // its result in place for that loop to carry and publish.
            if !acc.owner {
                continue;
            }
            if let Some(result_token) = ctx.vars.get(&acc.acc_var).and_then(|v| v.value) {
                super::shared_utils::set_view_token(&acc.root, result_token, ctx);
                for receiver in &acc.receivers {
                    super::shared_utils::set_view_token(receiver, result_token, ctx);
                }
            }
            ctx.vars.remove(&acc.acc_var);
        }
    }

    fn try_compile_dim_for_loop(
        &self,
        module: &mut Module,
        block_id: BlockId,
        for_expr: &ExprForLoop,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<bool, JITError> {
        let Some(dim_name) = Self::simple_path_name(&for_expr.expr) else {
            return Ok(false);
        };
        let Some(dim_value) = ctx.vars.get(&dim_name).cloned() else {
            return Ok(false);
        };
        if get_type_ident(&dim_value.ty.rust_ty).is_none_or(|ident| ident != "Dim") {
            return Ok(false);
        }
        let Some(dim_origin) = Self::value_dim_origin(&dim_value) else {
            return Ok(false);
        };
        let upper_bound = self.dim_size_value(&dim_value, &for_expr.expr.span())?;
        let maybe_iterand_ident = match &*for_expr.pat {
            Pat::Wild(_) => None,
            Pat::Ident(ident_pat) => Some(ident_pat),
            _ => {
                return self.jit_error_result(
                    &for_expr.pat.span(),
                    "dimension loops must bind a simple index variable or `_`",
                );
            }
        };

        let zero = self.compile_constant(module, block_id, generic_vars, 0i32)?;
        let one = self.compile_constant(module, block_id, generic_vars, 1i32)?;
        let lower_bound = zero.value.ok_or_else(|| {
            self.jit_error(
                &for_expr.span(),
                "failed to compile dimension loop lower bound",
            )
        })?;
        let step = one.value.ok_or_else(|| {
            self.jit_error(&for_expr.span(), "failed to compile dimension loop step")
        })?;

        let mut loop_carry_vars = collect_mutated_variables(for_expr)?
            .into_iter()
            .collect::<Vec<_>>();
        // Thread each written tensor's ordering token across the loop boundary.
        let loop_var_name = maybe_iterand_ident.map(|p| p.ident.to_string());
        let token_accumulators = self.setup_loop_token_accumulators(
            &for_expr.body,
            loop_var_name.as_deref(),
            ctx,
            generic_vars,
        )?;
        for acc in &token_accumulators {
            loop_carry_vars.push(acc.acc_var.clone());
        }
        let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
        let loop_carry_arg_tys = loop_carry_args
            .iter()
            .map(|val| module.value_type(*val).clone())
            .collect::<Vec<_>>();

        let for_iterand_type = module.value_type(upper_bound).clone();
        let loop_block_arg_tys = [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
        let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

        let mut for_variables = ctx.clone();
        let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
        for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
        self.bind_serialized_views(&token_accumulators, &mut for_variables);
        if let Some(iterand_ident) = maybe_iterand_ident {
            let iterand_name = iterand_ident.ident.to_string();
            let i32_type = self
                .compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(&for_expr.span(), "failed to compile i32 type"))?;
            let upper_bounds = dim_value.bounds.or_else(|| {
                dim_value
                    .fields
                    .as_ref()
                    .and_then(|fields| fields.get("size"))
                    .and_then(|size| size.bounds)
            });
            let mut iterand_val = if let Some(bounds) = upper_bounds {
                let upper = bounds.end - 1;
                if upper >= 0 {
                    let bounds = Bounds::new(0, upper);
                    let mut value = self.compile_value_assumption(
                        module,
                        loop_block_id,
                        loop_block_args[0],
                        "assume_bounds",
                        &[bounds.start as i32, bounds.end as i32],
                        i32_type.clone(),
                        &for_expr.span(),
                    )?;
                    value.bounds = Some(bounds);
                    value
                } else {
                    TileRustValue::new_value_kind_like(loop_block_args[0], i32_type.clone())
                }
            } else {
                TileRustValue::new_value_kind_like(loop_block_args[0], i32_type.clone())
            };
            iterand_val.index_origin = Some(dim_origin);
            iterand_val.partition_axis_origin = Self::value_partition_axis_origin(&dim_value);
            for_variables.vars.insert(iterand_name, iterand_val);
        }
        for_variables.carry_vars = Some(loop_carry_vars.clone());
        for_variables.default_terminator = Some(BlockTerminator::Continue);
        for_variables.innermost_loop = Some(LoopKind::For);
        for_variables.inside_for = true;

        self.compile_block(
            module,
            loop_block_id,
            &for_expr.body,
            generic_vars,
            &mut for_variables,
            return_type,
        )?;

        let region_id = module.alloc_region(Region {
            blocks: vec![loop_block_id],
        });
        let (for_op_id, result_values) =
            OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                .operands([lower_bound, upper_bound, step].iter().copied())
                .operands(loop_carry_args.iter().copied())
                .results(loop_carry_arg_tys.iter().cloned())
                .region(region_id)
                .build(module);
        append_op(module, block_id, for_op_id);

        if result_values.len() != loop_carry_args.len() {
            return self.jit_error_result(
                &for_expr.span(),
                &format!(
                    "dimension loop produces {} results but {} mutable variables are carried across iterations",
                    result_values.len(),
                    loop_carry_args.len()
                ),
            );
        }
        ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
        self.finish_loop_token_accumulators(&token_accumulators, ctx);
        Ok(true)
    }

    pub fn compile_expression(
        &self,
        module: &mut Module,
        block_id: BlockId,
        expr: &syn::Expr,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _expr_debug_str = expr.to_token_stream().to_string();
            match expr {
                Expr::ForLoop(for_expr) => {
                    if self.try_compile_mapped_partition_indices_for_loop(
                        module,
                        block_id,
                        for_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(None);
                    }
                    if self.try_compile_dim_for_loop(
                        module,
                        block_id,
                        for_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(None);
                    }

                    // A for loop: for pat in expr { ... }.
                    let maybe_iterand_ident = match &*for_expr.pat {
                        Pat::Wild(_) => {
                            // Iterand is not bounded.
                            None
                        }
                        Pat::Ident(ident_pat) => Some(ident_pat),
                        _ => return self.jit_error_result(
                            &for_expr.pat.span(),
                            "this loop pattern is not supported; use a simple variable name or `_`",
                        ),
                    };
                    // Extract range and optional step from the for-loop expression.
                    // Supports: `0..n` (step=1) and `(0..n).step_by(k)`.
                    let (range_expr, maybe_step_expr): (&syn::ExprRange, Option<&Expr>) =
                        match &*for_expr.expr {
                            Expr::Range(range) => (range, None),
                            Expr::MethodCall(mc) if mc.method == "step_by" => {
                                let receiver = match &*mc.receiver {
                                    Expr::Paren(p) => &*p.expr,
                                    other => other,
                                };
                                let Expr::Range(range) = receiver else {
                                    return self.jit_error_result(
                                        &mc.receiver.span(),
                                        "expected a range expression as the receiver of step_by (e.g. `(0..n).step_by(k)`)",
                                    );
                                };
                                if mc.args.len() != 1 {
                                    return self.jit_error_result(
                                        &mc.args.span(),
                                        "step_by expects exactly one argument",
                                    );
                                }
                                (range, Some(&mc.args[0]))
                            }
                            _ => {
                                return self.jit_error_result(
                                    &for_expr.expr.span(),
                                    "only range expressions (e.g. `0..n` or `(0..n).step_by(k)`) are supported in for loops",
                                );
                            }
                        };
                    // TODO (hme): Add meaningful errors and do more than just unwrap.
                    let Some(start_expr) = &range_expr.start else {
                        return self.jit_error_result(
                            &range_expr.span(),
                            "range expression is missing a start bound (e.g. `0..n`)",
                        );
                    };
                    let Some(end_expr) = &range_expr.end else {
                        return self.jit_error_result(
                            &range_expr.span(),
                            "range expression is missing an end bound (e.g. `0..n`)",
                        );
                    };
                    let start_return_type = self
                        .typeck_expr_tile_type(start_expr, generic_vars, &HashMap::new())?
                        .or(return_type.clone());
                    let Some(start_val) = self.compile_expression(
                        module,
                        block_id,
                        start_expr,
                        generic_vars,
                        ctx,
                        start_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &start_expr.span(),
                            "failed to compile range start expression",
                        );
                    };
                    let end_return_type = self
                        .typeck_expr_tile_type(end_expr, generic_vars, &HashMap::new())?
                        .or(return_type.clone());
                    let Some(end_val) = self.compile_expression(
                        module,
                        block_id,
                        end_expr,
                        generic_vars,
                        ctx,
                        end_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &end_expr.span(),
                            "failed to compile range end expression",
                        );
                    };
                    let iterand_lower_const = start_val.bounds;
                    let iterand_upper_const = end_val.bounds;
                    let lower_bound = start_val.value.unwrap();
                    let upper_bound = end_val.value.unwrap();
                    let step_value = if let Some(step_expr) = maybe_step_expr {
                        let Some(val) = self.compile_expression(
                            module,
                            block_id,
                            step_expr,
                            generic_vars,
                            ctx,
                            Some(start_val.ty.clone()),
                        )?
                        else {
                            return self.jit_error_result(
                                &step_expr.span(),
                                "failed to compile step_by expression",
                            );
                        };
                        val
                    } else {
                        self.compile_constant(module, block_id, generic_vars, 1)?
                    };
                    let step = step_value.value.ok_or_else(|| {
                        self.jit_error(
                            &for_expr.span(),
                            "internal: failed to produce step value for for-loop",
                        )
                    })?;
                    // The step when it is a compile-time constant (always for
                    // the unit-step form, which compiles the constant `1`).
                    let step_const: Option<i64> =
                        step_value.bounds.filter(|b| b.is_exact()).map(|b| b.start);
                    if let Some(step) = step_const {
                        if step <= 0 {
                            return self.jit_error_result(
                                &maybe_step_expr.map_or(for_expr.span(), |e| e.span()),
                                &format!("`step_by` requires a positive step, got {step}"),
                            );
                        }
                    }

                    // We skip verifying the op here and just require that each mutated mutable vars:
                    // 1. Is passed as an operand.
                    // 2. Is a block argument.
                    // 3. Is loop-carried.
                    // 4. Is returned.
                    let for_iterand_type = module.value_type(lower_bound).clone();
                    let loop_carry_vars = collect_mutated_variables(for_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    // Add iterand as first argument.
                    let loop_block_arg_tys =
                        [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
                    let value_watermark = module.num_values() as u32;
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

                    let mut for_variables = ctx.clone();
                    // Update loop carry variables within the for loop
                    // to the mutable variables accessed in this operation.
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
                    for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    if let Some(iterand_ident) = maybe_iterand_ident {
                        // maybe_iterand_ident is None if it is wild.
                        // If it's an ident, then add the iterand as a var.
                        let iterand_name = iterand_ident.ident.to_string();
                        let iterand_val = loop_block_args[0];
                        // This has the same type as start/end val.
                        let iterand_ty = start_val.ty.clone();
                        // If the loop bounds are const, then we can put a bound on the iterand.
                        // Subtract upper bound by 1, since it is the open end of the interval [start, end).
                        let mut iterand_val = match (iterand_lower_const, iterand_upper_const) {
                            (Some(iterand_lower_const), Some(iterand_upper_const)) => {
                                // `[lower, upper - 1]` over-approximates what a stepped
                                // loop attains: `(0..10).step_by(4)` visits {0, 4, 8},
                                // never 9. The interval fact must describe the ATTAINED
                                // set, because the static fold and the hoisted check test
                                // its extremes as values the loop reaches (audit 2026-08:
                                // "9 < 9" over nine tiles). With a constant step and an
                                // exact start the last attained value is
                                // `start + step * floor((end - 1 - start) / step)`; for a
                                // runtime step or an inexact start the residue class is
                                // unknown, so no interval fact is kept. The assumption
                                // below is emitted either way — it is a true
                                // over-approximation, and the device may use it.
                                let over_approx = Bounds::new(
                                    iterand_lower_const.start,
                                    iterand_upper_const.end - 1,
                                );
                                let attained = match step_const {
                                    Some(1) => Some(over_approx),
                                    Some(step) if iterand_lower_const.is_exact() => {
                                        let start = iterand_lower_const.start;
                                        let last = over_approx.end;
                                        let last_attained = if last >= start {
                                            start + step * ((last - start) / step)
                                        } else {
                                            start
                                        };
                                        Some(Bounds::new(start, last_attained))
                                    }
                                    _ => None,
                                };
                                let assumed = attained.unwrap_or(over_approx);
                                let mut iterand_val = self.compile_value_assumption(
                                    module,
                                    loop_block_id,
                                    iterand_val,
                                    "assume_bounds",
                                    &[assumed.start as i32, assumed.end as i32],
                                    iterand_ty,
                                    &for_expr.span(),
                                )?;
                                iterand_val.bounds = attained;
                                iterand_val
                            }
                            (Some(iterand_lower_const), None) => self.compile_value_assumption(
                                module,
                                loop_block_id,
                                iterand_val,
                                "assume_bounds_lower",
                                &[iterand_lower_const.start as i32],
                                iterand_ty,
                                &for_expr.span(),
                            )?,
                            (None, Some(iterand_upper_const)) => self.compile_value_assumption(
                                module,
                                loop_block_id,
                                iterand_val,
                                "assume_bounds_upper",
                                &[iterand_upper_const.end as i32 - 1],
                                iterand_ty,
                                &for_expr.span(),
                            )?,
                            (None, None) => TileRustValue::new_value_kind_like(
                                iterand_val,
                                start_val.ty.clone(),
                            ),
                        };
                        // A loop from exactly 0 up to a partition-axis count
                        // ranges over precisely that axis's tiles, so the
                        // iterand inherits the bound's axis provenance in both
                        // its forms. This is what makes `for i in 0..num_tiles(
                        // &p, a)` prove its own accesses safe, with no
                        // `with_bounds` annotation: an ordinary integer loop
                        // over a count the compiler minted carries the same
                        // evidence the annotation used to supply by hand.
                        if start_val
                            .bounds
                            .as_ref()
                            .is_some_and(|bounds| bounds.is_exact() && bounds.start == 0)
                        {
                            iterand_val.index_origin = Self::value_dim_origin(&end_val);
                            iterand_val.partition_axis_origin =
                                Self::value_partition_axis_origin(&end_val);
                        }
                        let mut induction_values = vec![loop_block_args[0]];
                        if let Some(alias) = iterand_val.value {
                            induction_values.push(alias);
                        }
                        // Static bounds prove the loop non-empty when the
                        // largest possible lower bound is below the smallest
                        // possible upper bound.
                        let known_non_empty = match (&start_val.bounds, &end_val.bounds) {
                            (Some(lower), Some(upper)) => lower.end < upper.start,
                            _ => false,
                        };
                        if let Some(alias) = iterand_val.value {
                            // Seed the induction variable's symbolic form: `1 *
                            // iv + 0`. Term arithmetic propagates it through the
                            // loop body (replaces the former AffineForm seed).
                            iterand_val.term = Some(cuda_async::predicate::Term::atom(
                                cuda_async::predicate::Atom::Iv(alias.index()),
                            ));
                        }
                        let unit_step = maybe_step_expr.is_none();
                        // When both bounds are compile-time constants and the
                        // step is unit, the induction variable's inclusive range
                        // is `[lower, upper - 1]` — used to derive an affine
                        // index's static range from its term.
                        let induction_range = if unit_step {
                            match (&start_val.bounds, &end_val.bounds) {
                                (Some(lo), Some(hi)) if lo.is_exact() && hi.is_exact() => {
                                    hi.start.checked_sub(1).map(|max| crate::bounds::Bounds {
                                        start: lo.start,
                                        end: max,
                                    })
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };
                        for_variables.loop_frames.push(LoopFrame {
                            preheader_block: block_id,
                            body_block: loop_block_id,
                            value_watermark,
                            induction_values,
                            lower: lower_bound,
                            upper: upper_bound,
                            unit_step,
                            induction_range,
                            known_non_empty,
                            has_early_exit: block_has_early_exit(&for_expr.body),
                        });
                        for_variables.vars.insert(iterand_name, iterand_val);
                    }
                    for_variables.carry_vars = Some(loop_carry_vars.clone());
                    for_variables.default_terminator = Some(BlockTerminator::Continue);
                    for_variables.innermost_loop = Some(LoopKind::For);
                    for_variables.inside_for = true;
                    // `return` inside the body is rejected by `compile_block`
                    // (it cannot be lowered); `break` too, since `cuda_tile.for`
                    // has no early exit.
                    self.compile_block(
                        module,
                        loop_block_id,
                        &for_expr.body,
                        generic_vars,
                        &mut for_variables,
                        return_type,
                    )?;

                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (for_op_id, result_values) =
                        OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                            .operands([lower_bound, upper_bound, step].iter().copied())
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, for_op_id);

                    // TODO (hme): This fails with "operand #0 does not dominate this use"
                    //  This may be a bug.
                    //  The compiled module in its entirety still passes verification.
                    // assert!(for_op.verify());
                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &for_expr.span(),
                            &format!(
                                "for loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::While(while_expr) => {
                    // While loop: while condition { body }
                    // Convert to cuda_tile.loop - simpler approach: body then check
                    let loop_carry_vars = collect_mutated_variables_while(while_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_carry_arg_tys);

                    let mut loop_variables = ctx.clone();
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args.clone();
                    loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    loop_variables.carry_vars = Some(loop_carry_vars.clone());
                    loop_variables.default_terminator = Some(BlockTerminator::Continue);
                    loop_variables.innermost_loop = Some(LoopKind::Loop);

                    // Evaluate condition
                    let Some(TileRustValue {
                        value: Some(condition_val),
                        ..
                    }) = self.compile_expression(
                        module,
                        loop_block_id,
                        &while_expr.cond,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &while_expr.cond.span(),
                            "failed to compile while-loop condition",
                        );
                    };

                    // Check condition first - if false, break immediately
                    // Then region: continue to body (just yield, body comes next)
                    let (then_block_id, _then_block_args) = build_block(module, &[]);
                    let (yield_op_id, _) =
                        OpBuilder::new(Opcode::Yield, self.ir_location(&while_expr.span()))
                            .build(module);
                    append_op(module, then_block_id, yield_op_id);
                    let then_region_id = module.alloc_region(Region {
                        blocks: vec![then_block_id],
                    });

                    // Else region: break out
                    let (else_block_id, _else_block_args) = build_block(module, &[]);
                    let break_values = loop_variables.unpack_some_vars(&loop_carry_vars)?;
                    let (break_op_id, _) =
                        OpBuilder::new(Opcode::Break, self.ir_location(&while_expr.span()))
                            .operands(break_values.iter().copied())
                            .build(module);
                    append_op(module, else_block_id, break_op_id);
                    let else_region_id = module.alloc_region(Region {
                        blocks: vec![else_block_id],
                    });

                    let (condition_check_id, _) =
                        OpBuilder::new(Opcode::If, self.ir_location(&while_expr.cond.span()))
                            .operand(condition_val)
                            .region(then_region_id)
                            .region(else_region_id)
                            .build(module);
                    append_op(module, loop_block_id, condition_check_id);

                    // Execute body
                    self.compile_block(
                        module,
                        loop_block_id,
                        &while_expr.body,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?;
                    // compile_block will inject continue at the end

                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (loop_op_id, result_values) =
                        OpBuilder::new(Opcode::Loop, self.ir_location(&while_expr.span()))
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, loop_op_id);

                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &while_expr.span(),
                            &format!(
                                "while loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::Loop(loop_expr) => {
                    // Infinite loop: loop { body }
                    // Same as while but without condition check
                    let loop_carry_vars = collect_mutated_variables_loop(loop_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_carry_arg_tys);

                    let mut loop_variables = ctx.clone();
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args.clone();
                    loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    loop_variables.carry_vars = Some(loop_carry_vars.clone());
                    loop_variables.default_terminator = Some(BlockTerminator::Continue);
                    loop_variables.innermost_loop = Some(LoopKind::Loop);

                    // Execute loop body (must contain break to exit)
                    // The body should handle its own terminator (break/continue)
                    self.compile_block(
                        module,
                        loop_block_id,
                        &loop_expr.body,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?;

                    // Note: compile_block will inject continue if not already present
                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (loop_op_id, result_values) =
                        OpBuilder::new(Opcode::Loop, self.ir_location(&loop_expr.span()))
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, loop_op_id);

                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &loop_expr.span(),
                            &format!(
                                "loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::If(if_expr) => {
                    // The condition is always bool -- don't propagate the if
                    // expression's return type into the condition.
                    let Some(conditional_val) = self.compile_expression(
                        module,
                        block_id,
                        &if_expr.cond,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &if_expr.cond.span(),
                            "failed to compile if-condition",
                        );
                    };
                    if let Some(bounds) = conditional_val.bounds {
                        if bounds.is_exact() {
                            // Emit the corresponding conditional, if it's defined.
                            let mut block_vars = ctx.clone();
                            block_vars.token_update_in_region = true;
                            // This is inlined, so no need to inject a terminator.
                            block_vars.default_terminator = None;
                            let (res, carry_vars) = match (bounds.start, &if_expr.else_branch) {
                                (1, _) => {
                                    let res = self.compile_block(
                                        module,
                                        block_id,
                                        &if_expr.then_branch,
                                        generic_vars,
                                        &mut block_vars,
                                        None,
                                    )?;
                                    let carry_vars =
                                        collect_mutated_variables_from_block(&if_expr.then_branch)?
                                            .into_iter()
                                            .collect::<Vec<_>>();
                                    (res, carry_vars)
                                }
                                (0, Some((_Else, else_expr))) => {
                                    let res = self.compile_else_branch(
                                        module,
                                        block_id,
                                        else_expr,
                                        generic_vars,
                                        &mut block_vars,
                                        None,
                                    )?;
                                    let carry_vars =
                                        collect_mutated_variables_from_expr(else_expr)?
                                            .into_iter()
                                            .collect::<Vec<_>>();
                                    (res, carry_vars)
                                }
                                _ => {
                                    // Do nothing since the conditional is false and there is no else branch.
                                    return Ok(None);
                                }
                            };
                            // The exact condition was inlined into this block,
                            // so there is no join and no competing definition.
                            // Publish the taken path's complete values —
                            // including facts established by its RHS — instead
                            // of reconstructing from the pre-branch templates
                            // and invalidating them (2026-08-12 review, P1).
                            ctx.replace_some_vars_from(&carry_vars, &block_vars)?;
                            return Ok(res);
                        }
                    }

                    // The if/then block must yield captured mutable variables.
                    let then_captured_vars =
                        collect_mutated_variables_from_block(&if_expr.then_branch)?
                            .into_iter()
                            .collect::<Vec<_>>();
                    let else_captured_vars = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            collect_mutated_variables_from_expr(else_expr)?
                                .into_iter()
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    };
                    let mut if_captured_var_names = if let Some(loop_carry_vars) = &ctx.carry_vars {
                        [
                            loop_carry_vars.clone(),
                            then_captured_vars.clone(),
                            else_captured_vars.clone(),
                        ]
                        .concat()
                    } else {
                        [then_captured_vars.clone(), else_captured_vars.clone()].concat()
                    };
                    dedup(&mut if_captured_var_names);

                    let Some(condition_val) = conditional_val.value else {
                        return self.jit_error_result(
                            &if_expr.cond.span(),
                            "failed to compile if-condition",
                        );
                    };
                    // Build then region.
                    let (then_region_id, then_return_type, branch_result_type) = {
                        let mut block_vars = ctx.clone();
                        block_vars.token_update_in_region = true;
                        block_vars.carry_vars = Some(if_captured_var_names.clone());
                        block_vars.default_terminator = Some(BlockTerminator::Yield);
                        let (then_block_id, _then_block_args) = build_block(module, &[]);
                        let result = self.compile_block(
                            module,
                            then_block_id,
                            &if_expr.then_branch,
                            generic_vars,
                            &mut block_vars,
                            return_type.clone(),
                        )?;
                        let (branch_result_type, return_type) = {
                            if let Some(result) = result {
                                let Some(cuda_tile_value) = result.value else {
                                    return self.jit_error_result(
                                        &if_expr.then_branch.span(),
                                        "an `if`/`else` cannot produce a compound value (a tuple, \
                                         array, or struct); bind each component in its own `let`",
                                    );
                                };
                                let result_ty = module.value_type(cuda_tile_value).clone();
                                (vec![result_ty], Some(result.ty.clone()))
                            } else {
                                (vec![], None)
                            }
                        };
                        let region_id = module.alloc_region(Region {
                            blocks: vec![then_block_id],
                        });
                        (region_id, return_type, branch_result_type)
                    };

                    // We don't need to check return type. Both Rust and Tile IR compiler perform this check.
                    let (else_region_id, _else_return_type) = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            let mut block_vars = ctx.clone();
                            block_vars.token_update_in_region = true;
                            block_vars.carry_vars = Some(if_captured_var_names.clone());
                            block_vars.default_terminator = Some(BlockTerminator::Yield);
                            let (else_block_id, _else_block_args) = build_block(module, &[]);
                            let result = self.compile_else_branch(
                                module,
                                else_block_id,
                                else_expr,
                                generic_vars,
                                &mut block_vars,
                                then_return_type.clone(),
                            )?;
                            let (_cuda_tile_return_values, return_type) = {
                                if let Some(result) = result {
                                    let Some(cuda_tile_value) = result.value else {
                                        return self.jit_error_result(
                                            &else_expr.span(),
                                            "an `if`/`else` cannot produce a compound value (a \
                                             tuple, array, or struct); bind each component in its \
                                             own `let`",
                                        );
                                    };
                                    (vec![cuda_tile_value], Some(result.ty.clone()))
                                } else {
                                    (vec![], None)
                                }
                            };
                            let region_id = module.alloc_region(Region {
                                blocks: vec![else_block_id],
                            });
                            (region_id, return_type)
                        } else {
                            if then_return_type.is_some() {
                                return self.jit_error_result(
                                    &if_expr.span(),
                                    "if-expression without an else branch cannot produce a return type",
                                );
                            }
                            let (else_block_id, _else_block_args) = build_block(module, &[]);
                            // If there is only a then branch, there is no return value. Yield only the captured mutable vars.
                            let captured_mutable_vars =
                                ctx.unpack_some_vars(&if_captured_var_names)?;
                            let (yield_op_id, _) =
                                OpBuilder::new(Opcode::Yield, self.ir_location(&if_expr.span()))
                                    .operands(captured_mutable_vars.iter().copied())
                                    .build(module);
                            append_op(module, else_block_id, yield_op_id);
                            let region_id = module.alloc_region(Region {
                                blocks: vec![else_block_id],
                            });
                            (region_id, None)
                        }
                    };

                    let if_result_types = {
                        let if_captured_var_args = ctx.unpack_some_vars(&if_captured_var_names)?;
                        let if_captured_var_arg_tys = if_captured_var_args
                            .iter()
                            .map(|val| module.value_type(*val).clone())
                            .collect::<Vec<_>>();
                        [if_captured_var_arg_tys, branch_result_type].concat()
                    };

                    let (if_op_id, mut result_values) =
                        OpBuilder::new(Opcode::If, self.ir_location(&if_expr.cond.span()))
                            .operand(condition_val)
                            .results(if_result_types.iter().cloned())
                            .region(then_region_id)
                            .region(else_region_id)
                            .build(module);
                    append_op(module, block_id, if_op_id);

                    if let Some(ty) = then_return_type {
                        if result_values.len() != if_captured_var_names.len() + 1 {
                            return self.jit_error_result(
                                &if_expr.span(),
                                &format!(
                                    "If expression result count ({}) does not match captured var count + 1 ({})",
                                    result_values.len(), if_captured_var_names.len() + 1
                                ),
                            );
                        }
                        let return_value = result_values.pop().unwrap();
                        ctx.repack_some_vars(&if_captured_var_names, &result_values, true)?;
                        let tr_value = TileRustValue::new_value_kind_like(return_value, ty);
                        Ok(Some(tr_value))
                    } else {
                        ctx.repack_some_vars(&if_captured_var_names, &result_values, true)?;
                        Ok(None)
                    }
                }
                Expr::Block(block_expr) => {
                    let mut inner_block_vars = ctx.clone();
                    inner_block_vars.default_terminator = None;
                    let outer_block_vars = ctx;
                    let carry_vars = collect_mutated_variables_from_block(&block_expr.block)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let result = self.compile_block(
                        module,
                        block_id,
                        &block_expr.block,
                        generic_vars,
                        &mut inner_block_vars,
                        return_type,
                    )?;
                    let result_values = inner_block_vars.unpack_some_vars(&carry_vars)?;
                    outer_block_vars.repack_some_vars(&carry_vars, &result_values, true)?;
                    // TODO (hme): Is this still needed if we're packing/unpacking above?
                    update_outer_block_type_meta(
                        &mut inner_block_vars,
                        outer_block_vars,
                        "token".to_string(),
                    );
                    Ok(result)
                }
                Expr::Unsafe(block_expr) => {
                    let mut inner_block_vars = ctx.clone();
                    inner_block_vars.default_terminator = None;
                    let outer_block_vars = ctx;
                    let carry_vars = collect_mutated_variables_from_block(&block_expr.block)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let result = self.compile_block(
                        module,
                        block_id,
                        &block_expr.block,
                        generic_vars,
                        &mut inner_block_vars,
                        return_type,
                    )?;
                    let result_values = inner_block_vars.unpack_some_vars(&carry_vars)?;
                    outer_block_vars.repack_some_vars(&carry_vars, &result_values, true)?;
                    // TODO (hme): Is this still needed if we're packing/unpacking above?
                    update_outer_block_type_meta(
                        &mut inner_block_vars,
                        outer_block_vars,
                        "token".to_string(),
                    );
                    Ok(result)
                }
                Expr::Struct(struct_expr) => {
                    let return_type = match return_type {
                        Some(return_type) => return_type,
                        None => {
                            return self.jit_error_result(
                                &struct_expr.span(),
                                "struct expressions require a known return type; try adding a type annotation",
                            )
                        }
                    };
                    let mut fields: BTreeMap<String, TileRustValue> = BTreeMap::new();
                    for field in struct_expr.fields.iter() {
                        let field_name: String = match &field.member {
                            Member::Named(named) => named.to_string(),
                            Member::Unnamed(_idx) => {
                                return self.jit_error_result(
                                    &struct_expr.span(),
                                    "unnamed (tuple) struct fields are not supported",
                                )
                            }
                        };
                        let struct_name = struct_expr.path.segments[0].ident.to_string();
                        let field_type = self
                            .modules
                            .get_struct_field_type(&struct_name, &field_name);
                        let tile_rust_ty = if let Some(field_type) = field_type {
                            // `Shape` and `Array` are compiler-known structs whose field
                            // expressions often need a concrete expected type during emission.
                            if ["Shape", "Array"].contains(&struct_name.as_str()) {
                                self.compile_type(&field_type, generic_vars, &HashMap::new())?
                            } else {
                                self.typeck_expr_tile_type(
                                    &field.expr,
                                    generic_vars,
                                    &HashMap::new(),
                                )?
                            }
                        } else {
                            self.typeck_expr_tile_type(&field.expr, generic_vars, &HashMap::new())?
                        };
                        let field_value: TileRustValue = match self.compile_expression(
                            module,
                            block_id,
                            &field.expr,
                            generic_vars,
                            ctx,
                            tile_rust_ty,
                        )? {
                            Some(field_value) => field_value,
                            None => {
                                return self.jit_error_result(
                                    &field.expr.span(),
                                    &format!("failed to compile value for field `{field_name}`"),
                                )
                            }
                        };
                        fields.insert(field_name, field_value);
                    }
                    Ok(Some(TileRustValue::new_struct(fields, return_type)))
                }
                Expr::Reference(ref_expr) => {
                    // TODO (hme): Check whether all expr types can be supported.
                    let return_type = match return_type {
                        Some(ty) => {
                            if let syn::Type::Reference(ref_type) = ty.rust_ty {
                                self.compile_type(&ref_type.elem, generic_vars, &HashMap::new())?
                            } else {
                                None
                            }
                        }
                        _ => return_type,
                    };
                    match &*ref_expr.expr {
                        Expr::Array(_array_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Path(_path_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Repeat(_repeat_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::MethodCall(_method_call_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        _ => self.jit_error_result(
                            &ref_expr.span(),
                            "this reference expression form is not supported",
                        ),
                    }
                }
                Expr::Tuple(tuple_expr) => {
                    let expected_elem_types = match return_type.as_ref().map(|ty| &ty.rust_ty) {
                        Some(syn::Type::Tuple(tuple_ty)) => {
                            Some(tuple_ty.elems.iter().cloned().collect::<Vec<_>>())
                        }
                        _ => None,
                    };
                    if let Some(expected_elem_types) = &expected_elem_types {
                        if expected_elem_types.len() != tuple_expr.elems.len() {
                            return self.jit_error_result(
                                &tuple_expr.span(),
                                &format!(
                                    "tuple expression has {} elements but expected tuple type has {} elements",
                                    tuple_expr.elems.len(),
                                    expected_elem_types.len()
                                ),
                            );
                        }
                    }
                    let mut rust_types: Vec<syn::Type> = vec![];
                    let mut values: Vec<TileRustValue> = vec![];
                    for (idx, elem) in tuple_expr.elems.iter().enumerate() {
                        let elem_return_type = expected_elem_types
                            .as_ref()
                            .and_then(|elem_types| elem_types.get(idx))
                            .and_then(|elem_ty| {
                                self.compile_type(elem_ty, generic_vars, &HashMap::new())
                                    .ok()
                                    .flatten()
                            });
                        match self.compile_expression(
                            module,
                            block_id,
                            elem,
                            generic_vars,
                            ctx,
                            elem_return_type,
                        )? {
                            Some(value) => {
                                rust_types.push(value.ty.rust_ty.clone());
                                values.push(value);
                            }
                            None => {
                                return self.jit_error_result(
                                    &elem.span(),
                                    "failed to compile tuple element",
                                )
                            }
                        };
                    }
                    let ty_string = rust_types
                        .iter()
                        .map(|rust_ty| rust_ty.to_token_stream().to_string())
                        .collect::<Vec<String>>()
                        .join(", ");
                    let ty: syn::Type =
                        match syn::parse2::<syn::Type>(format!("({ty_string})").parse().unwrap()) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &tuple_expr.span(),
                                    &format!(
                                        "failed to parse inferred tuple type `({ty_string})`: {e}"
                                    ),
                                )
                            }
                        };
                    let ct_ty = match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                        Some(ct_ty) => ct_ty,
                        None => {
                            return self.jit_error_result(
                                &tuple_expr.span(),
                                "unable to compile inferred tuple type",
                            )
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, ct_ty)))
                }
                Expr::Array(array_expr) => {
                    let mut values: Vec<TileRustValue> = vec![];
                    for elem in &array_expr.elems {
                        let elem_ty = match &return_type {
                            Some(return_type) => {
                                match &return_type.rust_ty {
                                    syn::Type::Array(array_type) => self.compile_type(
                                        &array_type.elem,
                                        generic_vars,
                                        &HashMap::new(),
                                    )?,
                                    syn::Type::Slice(slice) => {
                                        // TODO (hme): Confirm this is right.
                                        self.compile_type(
                                            &slice.elem,
                                            generic_vars,
                                            &HashMap::new(),
                                        )?
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &elem.span(),
                                            &format!(
                                                "unexpected element type `{}`",
                                                return_type.rust_ty.to_token_stream()
                                            ),
                                        )
                                    }
                                }
                            }
                            None => None,
                        };
                        match self.compile_expression(
                            module,
                            block_id,
                            elem,
                            generic_vars,
                            ctx,
                            elem_ty,
                        )? {
                            Some(value) => values.push(value),
                            None => {
                                return self.jit_error_result(
                                    &elem.span(),
                                    "failed to compile array element",
                                )
                            }
                        };
                    }
                    let return_type = if let Some(return_type) = return_type {
                        return_type
                    } else {
                        if values.is_empty() {
                            return self.jit_error_result(
                                &array_expr.span(),
                                "unable to infer type for empty array; add a type annotation",
                            );
                        }
                        let ty: &TileRustType = &values[0].ty;
                        let ty_string = ty.rust_ty.to_token_stream().to_string();
                        let ty: syn::Type = match syn::parse2::<syn::Type>(
                            format!("[{ty_string}]").parse().unwrap(),
                        ) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &array_expr.span(),
                                    &format!(
                                        "failed to parse inferred array type `[{ty_string}]`: {e}"
                                    ),
                                )
                            }
                        };
                        match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                            Some(ct_ty) => ct_ty,
                            None => {
                                return self.jit_error_result(
                                    &array_expr.span(),
                                    "unable to compile inferred array type",
                                )
                            }
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, return_type)))
                }
                Expr::Repeat(repeat_expr) => {
                    let len = {
                        let len_expr = &*repeat_expr.len;
                        if let Expr::Path(len_expr) = len_expr {
                            let var_name = len_expr.path.segments.last().unwrap().ident.to_string();
                            // Expecting a const generic primitive.
                            let Some(n) = generic_vars.get_i32(var_name.as_str()) else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    &format!("expected a const generic value for repeat length, but `{var_name}` is not a known const generic"),
                                );
                            };
                            n as usize
                        } else {
                            let Expr::Lit(lit_expr) = len_expr else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "repeat length must be a literal or const generic",
                                );
                            };
                            let Lit::Int(int_lit) = &lit_expr.lit else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "repeat length must be an integer literal",
                                );
                            };
                            let Ok(len) = int_lit.base10_parse::<usize>() else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "failed to parse repeat length as a valid integer",
                                );
                            };
                            len
                        }
                    };
                    let elem_return_type = match return_type.as_ref() {
                        Some(return_type) => {
                            self.expected_array_element_type(return_type, generic_vars)?
                        }
                        None => self.typeck_expr_tile_type(
                            &repeat_expr.expr,
                            generic_vars,
                            &HashMap::new(),
                        )?,
                    };
                    let Some(value) = self.compile_expression(
                        module,
                        block_id,
                        &repeat_expr.expr,
                        generic_vars,
                        ctx,
                        elem_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &repeat_expr.expr.span(),
                            "failed to compile repeat expression element",
                        );
                    };
                    let values: Vec<TileRustValue> = vec![value; len];
                    let return_type = if let Some(return_type) = return_type {
                        return_type
                    } else {
                        if values.is_empty() {
                            return self.jit_error_result(
                                &repeat_expr.span(),
                                "unable to infer type for zero-length repeat expression; add a type annotation",
                            );
                        }
                        let ty: &TileRustType = &values[0].ty;
                        let ty_string = ty.rust_ty.to_token_stream().to_string();
                        let ty: syn::Type = match syn::parse2::<syn::Type>(
                            format!("[{ty_string}]").parse().unwrap(),
                        ) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &repeat_expr.span(),
                                    &format!(
                                        "failed to parse inferred repeat type `[{ty_string}]`: {e}"
                                    ),
                                )
                            }
                        };
                        match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                            Some(ct_ty) => ct_ty,
                            None => {
                                return self.jit_error_result(
                                    &repeat_expr.span(),
                                    "unable to compile inferred repeat type",
                                )
                            }
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, return_type)))
                }
                Expr::Path(path_expr) => {
                    let var_name = path_expr.path.segments.last().unwrap().ident.to_string();

                    // Handle None specially — Rust Option::None, not a variable.
                    if path_expr.path.segments.len() == 1 && var_name == "None" {
                        if let Some(return_type) = return_type {
                            if return_type.kind == Kind::Enum {
                                return Ok(Some(TileRustValue::new_enum(
                                    "None",
                                    None,
                                    return_type,
                                )));
                            }
                        }
                        return Ok(None);
                    }

                    // 1. Local variable (single-segment paths, locals shadow module items).
                    if path_expr.path.segments.len() == 1 {
                        if let Some(value) = ctx.vars.get(&var_name) {
                            return Ok(Some(value.clone()));
                        }
                    }

                    // 2. Resolve via name resolver (module-level structs, functions, etc.).
                    let res = self
                        .modules
                        .name_resolver
                        .resolve_path(&path_expr.path, &self.module_name);
                    match res {
                        Res::Def(DefKind::Struct, _) => {
                            // Known DSL struct — return as ZST marker placeholder.
                            return Ok(Some(Self::make_zst_marker(path_expr)));
                        }
                        Res::Def(DefKind::Const, def_id) => {
                            let Some(const_item) = self.modules.name_resolver.get_const(&def_id)
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to resolve const `{var_name}`"),
                                );
                            };
                            let const_ty =
                                self.compile_type(&const_item.ty, generic_vars, &HashMap::new())?;
                            return self.compile_expression(
                                module,
                                block_id,
                                &const_item.expr,
                                generic_vars,
                                ctx,
                                const_ty,
                            );
                        }
                        Res::Def(DefKind::Static, def_id) => {
                            let Some(static_item) = self.modules.name_resolver.get_static(&def_id)
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to resolve static `{var_name}`"),
                                );
                            };
                            let Some(static_ty) =
                                self.compile_type(&static_item.ty, generic_vars, &HashMap::new())?
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to compile static `{var_name}` type"),
                                );
                            };
                            return Ok(Some(TileRustValue::new_struct(BTreeMap::new(), static_ty)));
                        }
                        _ => {}
                    }

                    // 3. Multi-segment path not in the resolver — treat as a ZST
                    //    marker type from a nested Rust module (ftz::Enabled,
                    //    rounding::NearestEven, nan::Disabled, etc.). These modules
                    //    are defined outside the #[cutile::module] block and aren't
                    //    in the DSL AST the resolver indexes. They're valid Rust
                    //    type paths consumed by resolve_static_params.
                    if path_expr.path.segments.len() > 1 {
                        if self.path_looks_like_associated_const(path_expr, generic_vars) {
                            return self.jit_error_result(
                                &path_expr.span(),
                                "associated const values are not supported in expression position; use a literal or pass supported element constants such as `T::ZERO` directly to a DSL operation that accepts them",
                            );
                        }
                        return Ok(Some(Self::make_zst_marker(path_expr)));
                    }

                    // 4. Single-segment, not a local, not in resolver — error.
                    let suggestion = self.modules.name_resolver.find_all_definitions(&var_name);
                    if suggestion.is_empty() {
                        self.jit_error_result(
                            &path_expr.span(),
                            &format!("undefined variable `{var_name}`"),
                        )
                    } else {
                        self.jit_error_result(
                            &path_expr.span(),
                            &format!(
                                "undefined variable `{var_name}` (did you mean the function defined in {}?)",
                                suggestion.join(", ")
                            ),
                        )
                    }
                }
                Expr::Call(call_expr) => {
                    let call_expr_func_str = call_expr.func.to_token_stream().to_string();
                    let _args_str = call_expr.args.to_token_stream().to_string();
                    match &*call_expr.func {
                        Expr::Path(path_expr) => {
                            if Self::is_dim_new_call(&call_expr.func) {
                                return self.compile_dim_new_call(
                                    module,
                                    block_id,
                                    call_expr,
                                    generic_vars,
                                    ctx,
                                    return_type,
                                );
                            }
                            let ident = get_ident_from_path_expr(path_expr);
                            // Handle Some(...) specially - it's a Rust Option constructor, not a function call
                            if ident == "Some" {
                                if call_expr.args.len() != 1 {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "`Some()` expects exactly one argument, got {}",
                                            call_expr.args.len()
                                        ),
                                    );
                                }

                                if let Some(return_type) = return_type {
                                    if return_type.kind == Kind::Enum {
                                        return Ok(Some(TileRustValue::new_enum(
                                            "Some",
                                            Some(call_expr.args[0].clone()),
                                            return_type,
                                        )));
                                    }
                                }

                                let Some(payload_value) = self.compile_expression(
                                    module,
                                    block_id,
                                    &call_expr.args[0],
                                    generic_vars,
                                    ctx,
                                    None,
                                )?
                                else {
                                    return self.jit_error_result(
                                        &call_expr.args[0].span(),
                                        "failed to compile `Some` payload",
                                    );
                                };
                                let option_type =
                                    Self::make_option_type_from_payload(&payload_value.ty);
                                return Ok(Some(TileRustValue::new_enum(
                                    "Some",
                                    Some(call_expr.args[0].clone()),
                                    option_type,
                                )));
                            }
                            if self
                                .modules
                                .get_cuda_tile_op_attrs(ident.to_string().as_str())
                                .is_some()
                            {
                                Ok(self.compile_cuda_tile_op_call(
                                    module,
                                    block_id,
                                    call_expr,
                                    generic_vars,
                                    ctx,
                                    return_type,
                                )?)
                            } else if let Some((module_name, fn_item)) = self
                                .modules
                                .get_function_by_name(ident.to_string().as_str())
                            {
                                if let Some(compiler_op_attrs) =
                                    get_meta_list("cuda_tile :: compiler_op", &fn_item.attrs)
                                {
                                    Ok(self.compile_compiler_op_call(
                                        module,
                                        block_id,
                                        call_expr,
                                        path_expr,
                                        fn_item,
                                        &compiler_op_attrs,
                                        generic_vars,
                                        ctx,
                                        return_type,
                                    )?)
                                } else {
                                    Ok(self.inline_function_call(
                                        module,
                                        block_id,
                                        module_name,
                                        fn_item,
                                        call_expr,
                                        generic_vars,
                                        ctx,
                                        return_type,
                                    )?)
                                }
                            } else {
                                self.jit_error_result(
                                    &call_expr.func.span(),
                                    &format!("call to `{}` is not supported", call_expr_func_str),
                                )
                            }
                        }
                        _ => self.jit_error_result(
                            &call_expr.func.span(),
                            &format!("Call to {} not supported.", call_expr_func_str),
                        ),
                    }
                }
                Expr::MethodCall(method_call_expr) => {
                    if let Some(value) = self.compile_into_dim_method(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    if let Some(value) = self.compile_partition_with_bounds_method(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    if let Some(value) = self.compile_global_method_call(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    Ok(self.inline_method_call(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type,
                    )?)
                }
                Expr::Field(field_expr) => {
                    let Some(base) = self.compile_expression(
                        module,
                        block_id,
                        &field_expr.base,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &field_expr.base.span(),
                            "failed to compile the receiver of this field access",
                        );
                    };
                    match &field_expr.member {
                        Member::Named(field_name) => {
                            if base.kind != Kind::Struct {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "expected a struct value for field access",
                                );
                            }
                            if base.fields.is_none() {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "struct is missing its field data (internal)",
                                );
                            }
                            let fields = &base.fields.clone().unwrap();
                            let Some(field_value) = fields.get(&field_name.to_string()) else {
                                return self.jit_error_result(
                                    &field_name.span(),
                                    &format!("{} is not a field.", field_name),
                                );
                            };
                            Ok(Some(field_value.clone()))
                        }
                        Member::Unnamed(idx) => {
                            if base.kind != Kind::Compound {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "expected a tuple or compound value for indexed field access",
                                );
                            }
                            if base.values.is_none() {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "compound value is missing its element list (internal)",
                                );
                            }
                            let values = base.values.as_ref().unwrap();
                            let index = idx.index as usize;
                            let value: Option<&TileRustValue> = values.get(index);
                            if value.is_none() {
                                return self.jit_error_result(
                                    &field_expr.span(),
                                    &format!(
                                        "Index {index} access failed with {} elements.",
                                        values.len()
                                    ),
                                );
                            }
                            Ok(Some(value.unwrap().clone()))
                        }
                    }
                }
                Expr::Unary(unary_expr) => {
                    let UnOp::Neg(_) = unary_expr.op else {
                        return self.jit_error_result(
                            &unary_expr.span(),
                            "Unary expression not supported",
                        );
                    };
                    match &*unary_expr.expr {
                        Expr::Lit(lit_expr) => {
                            let return_type = if return_type.is_none() {
                                match get_lit_type(lit_expr) {
                                    Some(ty) => {
                                        self.compile_type(&ty, generic_vars, &HashMap::new())?
                                    }
                                    None => None,
                                }
                            } else {
                                return_type
                            };
                            let Some(return_type) = return_type else {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "Failed to infer type for unary op expr.",
                                );
                            };
                            let (lit_string, bounds) = match &lit_expr.lit {
                                Lit::Float(float_lit) => {
                                    (format!("-{}", float_lit.base10_digits()), None)
                                }
                                Lit::Int(int_lit) => {
                                    let str = format!("-{}", int_lit.base10_digits());
                                    let val = -int_lit
                                        .base10_parse::<i32>()
                                        .unwrap_or_else(|_| panic!("Failed to parse literal {str}"))
                                        as i64;
                                    (str, Some(Bounds::exact(val)))
                                }
                                _ => {
                                    return self.jit_error_result(
                                        &lit_expr.span(),
                                        "Lit expression not implemented",
                                    )
                                }
                            };
                            let Some(cuda_tile_ty) = return_type
                                .get_cuda_tile_element_type(self.modules.primitives())?
                            else {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "unable to determine type for numeric literal; add a type annotation",
                                );
                            };

                            // Build Constant op with proper DenseElements encoding.
                            let (op_result, _tile_ir_ty) = build_constant_op(
                                module,
                                block_id,
                                &lit_string,
                                &cuda_tile_ty,
                                self.ir_location(&lit_expr.span()),
                            );

                            let rust_ty = return_type.rust_ty;
                            let ct_type =
                                self.compile_type(&rust_ty, generic_vars, &HashMap::new())?;
                            if ct_type.is_none() {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "failed to compile the type of this literal",
                                );
                            }
                            let ct_type = ct_type.unwrap();
                            if ct_type.kind != Kind::PrimitiveType {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    &format!(
                                        "expected a scalar type for this literal, got {:?}",
                                        ct_type.kind
                                    ),
                                );
                            }
                            Ok(Some(TileRustValue::new_primitive(
                                op_result, ct_type, bounds,
                            )))
                        }
                        _ => self.jit_error_result(
                            &unary_expr.span(),
                            "Non-const unary expressions not supported.",
                        ),
                    }
                }
                Expr::Cast(cast_expr) => {
                    let mut src_expr = self
                        .compile_expression(
                            module,
                            block_id,
                            &cast_expr.expr,
                            generic_vars,
                            ctx,
                            None,
                        )?
                        .unwrap();
                    let src_elem_ty: String = src_expr
                        .ty
                        .get_instantiated_rust_element_type(self.modules.primitives())
                        .unwrap();
                    let dst_elem_ty: String = get_rust_element_type_primitive(&cast_expr.ty);
                    match (src_elem_ty.as_str(), dst_elem_ty.as_str()) {
                        ("i32", "u32") => {}
                        ("i64", "u64") => {}
                        ("i32", "usize") => {}
                        ("usize", "i32") => {}
                        _ => {
                            return self.jit_error_result(
                                &cast_expr.span(),
                                &format!(
                                    "unsupported cast from `{src_elem_ty}` to `{dst_elem_ty}`"
                                ),
                            )
                        }
                    }
                    // The cast is a relabel — same bits, new value domain. An
                    // interval is a claim about the machine value under the
                    // NEW interpretation, so it survives only if it fits the
                    // destination's domain (e.g. a possibly-negative `i32`
                    // range says nothing about the value read as `u32`).
                    if let Some(b) = src_expr.bounds {
                        let keep = crate::value_facts::int_value_domain(&dst_elem_ty)
                            .is_some_and(|d| d.start <= b.start && b.end <= d.end);
                        if !keep {
                            src_expr.bounds = None;
                        }
                    }
                    Ok(Some(src_expr))
                }
                Expr::Lit(lit_expr) => {
                    let return_type = if return_type.is_none() {
                        let typeck_return_type =
                            self.typeck_expr_tile_type(expr, generic_vars, &HashMap::new())?;
                        if typeck_return_type.is_some() {
                            typeck_return_type
                        } else {
                            match get_lit_type(lit_expr) {
                                Some(ty) => {
                                    self.compile_type(&ty, generic_vars, &HashMap::new())?
                                }
                                None => None,
                            }
                        }
                    } else {
                        return_type
                    };
                    let Some(return_type) = return_type else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!(
                                "Failed to infer type for lit expr {}.",
                                lit_expr.to_token_stream()
                            ),
                        );
                    };
                    if let Lit::Str(_) = &lit_expr.lit {
                        return Ok(Some(TileRustValue::new_string(
                            Expr::Lit(lit_expr.clone()),
                            return_type,
                        )));
                    }
                    let (lit_string, bounds) = match &lit_expr.lit {
                        Lit::Float(float_lit) => (float_lit.base10_digits().to_string(), None),
                        Lit::Int(int_lit) => {
                            let str = int_lit.base10_digits().to_string();
                            let val = int_lit
                                .base10_parse::<i32>()
                                .unwrap_or_else(|_| panic!("Failed to parse literal {str}"))
                                as i64;
                            (str, Some(Bounds::exact(val)))
                        }
                        Lit::Bool(bool_lit) => (
                            format!("{}", bool_lit.value as i32),
                            Some(Bounds::exact(bool_lit.value as i64)),
                        ),
                        _ => {
                            return self.jit_error_result(
                                &lit_expr.span(),
                                "Lit expression not implemented",
                            )
                        }
                    };
                    let Some(cuda_tile_ty) =
                        return_type.get_cuda_tile_element_type(self.modules.primitives())?
                    else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "unable to determine type for numeric literal; add a type annotation",
                        );
                    };

                    // Build Constant op with proper DenseElements encoding.
                    let (op_result, _tile_ir_ty) = build_constant_op(
                        module,
                        block_id,
                        &lit_string,
                        &cuda_tile_ty,
                        self.ir_location(&lit_expr.span()),
                    );

                    let rust_ty = return_type.rust_ty;
                    let ct_type = self.compile_type(&rust_ty, generic_vars, &HashMap::new())?;
                    if ct_type.is_none() {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "failed to compile the type of this literal",
                        );
                    }
                    let ct_type = ct_type.unwrap();
                    if ct_type.kind != Kind::PrimitiveType {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!(
                                "expected a scalar type for this literal, got {:?}",
                                ct_type.kind
                            ),
                        );
                    }
                    Ok(Some(TileRustValue::new_primitive(
                        op_result, ct_type, bounds,
                    )))
                }
                Expr::Binary(bin_expr) => {
                    // These are type-checked by Rust, so just do whatever the expression is asking.
                    Ok(self.compile_binary_op(
                        module,
                        block_id,
                        bin_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?)
                }
                Expr::Paren(paren_expr) => Ok(self.compile_expression(
                    module,
                    block_id,
                    &paren_expr.expr,
                    generic_vars,
                    ctx,
                    return_type.clone(),
                )?),
                Expr::Macro(mac_expr) => {
                    let last_seg = mac_expr.mac.path.segments.last();
                    if last_seg.is_none() {
                        return self.jit_error_result(
                            &mac_expr.mac.path.span(),
                            "unrecognized macro invocation",
                        );
                    }
                    let last_seg = last_seg.unwrap();
                    let mac_name = last_seg.ident.to_string();
                    Ok(match mac_name.as_str() {
                        "const_shape" | "shape" | "const_array" => {
                            // TODO (hme): Remove special case for const_shape here
                            //  and on the proc-macro side (rank_instantiation.rs).
                            let args = self.const_shape_macro_args(mac_expr, generic_vars, ctx)?;
                            let cga_str = format!("{{[{}]}}", args.join(", "));
                            let ty_str = if mac_name == "const_shape" || mac_name == "shape" {
                                "Shape"
                            } else {
                                "Array"
                            };
                            let shape_expr = syn::parse2::<Expr>(
                                format!("{ty_str}::<{cga_str}>{{dims: &[]}}")
                                    .parse()
                                    .unwrap(),
                            )
                            .unwrap();
                            let return_type = if return_type.is_none() {
                                let shape_str = format!("{ty_str}<{cga_str}>");
                                let shape_ty =
                                    syn::parse2::<syn::Type>(shape_str.parse().unwrap()).unwrap();
                                self.compile_type(&shape_ty, generic_vars, &HashMap::new())?
                            } else {
                                return_type.clone()
                            };
                            self.compile_expression(
                                module,
                                block_id,
                                &shape_expr,
                                generic_vars,
                                ctx,
                                return_type,
                            )?
                        }
                        _ => self.compile_cuda_tile_macro(
                            module,
                            block_id,
                            &mac_expr.mac,
                            generic_vars,
                            ctx,
                            return_type.clone(),
                        )?,
                    })
                }
                Expr::Closure(closure_expr) => {
                    // Closures cannot be used as standalone expressions in CUDA Tile.
                    // They are only supported as arguments to specific operations (e.g., reduce, scan)
                    // that compile them into tile-ir regions.
                    self.jit_error_result(
                        &closure_expr.span(),
                        "closures are not supported as standalone values; \
                         they can only be used as arguments to operations like `reduce()` or `scan()`",
                    )
                }
                Expr::Index(index_expr) => {
                    let Some(expr_val) = self.compile_expression(
                        module,
                        block_id,
                        &index_expr.expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &index_expr.expr.span(),
                            "failed to compile the indexed expression",
                        );
                    };
                    // TODO (hme): Revisit this once we have proper type inference.
                    let i32_type: syn::Type = parse_quote! { i32 };
                    let i32_type = self.compile_type(&i32_type, generic_vars, &HashMap::new())?;
                    let Some(index_val) = self.compile_expression(
                        module,
                        block_id,
                        &index_expr.index,
                        generic_vars,
                        ctx,
                        i32_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &index_expr.index.span(),
                            "failed to compile index value",
                        );
                    };
                    let idx: i32 = {
                        let Some(index_bounds) = index_val.bounds else {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                "dynamic indices are not supported; the index must be a compile-time constant",
                            );
                        };
                        if !index_bounds.is_exact() {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                "index must be a compile-time constant with exact bounds",
                            );
                        }
                        index_bounds.start as i32
                    };
                    if idx < 0 {
                        return self.jit_error_result(
                            &index_expr.index.span(),
                            &format!("index must be non-negative, got {idx}"),
                        );
                    }
                    if expr_val.kind == Kind::Compound {
                        let Some(mut values) = expr_val.values else {
                            return self.jit_error_result(
                                &index_expr.expr.span(),
                                "internal: compound value is missing its element list during index access",
                            );
                        };
                        let index = idx as usize;
                        if index >= values.len() {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                &format!(
                                    "index {idx} out of bounds for compound value of length {}",
                                    values.len()
                                ),
                            );
                        }
                        return Ok(Some(values.remove(index)));
                    }
                    if let Some(fields) = expr_val.fields.as_ref() {
                        if let Some(dims) = fields.get("dims") {
                            let Some(mut values) = dims.values.clone() else {
                                return self.jit_error_result(
                                    &index_expr.expr.span(),
                                    "shape-like value has a `dims` field that is not indexable",
                                );
                            };
                            let index = idx as usize;
                            if index >= values.len() {
                                return self.jit_error_result(
                                    &index_expr.index.span(),
                                    &format!(
                                        "index {idx} out of bounds for shape of rank {}",
                                        values.len()
                                    ),
                                );
                            }
                            return Ok(Some(values.remove(index)));
                        }
                    }
                    self.jit_error_result(
                        &index_expr.expr.span(),
                        "indexing is only supported on tuple/compound values and shape-like descriptors",
                    )
                }
                _ => self.jit_error_result(&expr.span(), "this expression form is not supported"),
            }
        }) // stacker::maybe_grow
    }
}

/// Convert a CUDA Tile element type string (e.g. "f32", "i32") to a tile-ir scalar tile Type.
fn cuda_tile_element_type_to_tile_ir(cuda_tile_ty: &str) -> cutile_ir::ir::Type {
    use cutile_ir::ir::{ScalarType, TileElementType, TileType, Type};
    let scalar = super::_type::scalar_from_name(cuda_tile_ty).unwrap_or(ScalarType::I32);
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(scalar),
    })
}

/// Build a Constant op with a proper DenseElements value attribute.
/// `lit_string` is the numeric literal as text (e.g. "42", "-3.14", "0x3f800000").
/// `cuda_tile_ty` is the element type name (e.g. "f32", "i32").
fn build_constant_op(
    module: &mut cutile_ir::ir::Module,
    block_id: cutile_ir::ir::BlockId,
    lit_string: &str,
    cuda_tile_ty: &str,
    location: Location,
) -> (cutile_ir::ir::Value, cutile_ir::ir::Type) {
    use cutile_ir::ir::DenseElements;

    let result_ty = cuda_tile_element_type_to_tile_ir(cuda_tile_ty);
    let data = encode_literal_bytes(lit_string, cuda_tile_ty);

    let (op_id, results) = OpBuilder::new(Opcode::Constant, location)
        .result(result_ty.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: result_ty.clone(),
                shape: vec![],
                data,
            }),
        )
        .build(module);
    cutile_ir::builder::append_op(module, block_id, op_id);
    (results[0], result_ty)
}

/// Encode a literal value string into bytes for a DenseElements attribute.
pub fn encode_literal_bytes(lit_string: &str, cuda_tile_ty: &str) -> Vec<u8> {
    use cutile_ir::ir::ScalarType;
    let scalar = super::_type::scalar_from_name(cuda_tile_ty).unwrap_or(ScalarType::I32);
    match scalar {
        ScalarType::I1 => vec![if lit_string != "0" { 0xFF } else { 0x00 }],
        ScalarType::I4 => {
            let v: i8 = lit_string.parse().unwrap_or(0);
            vec![(v as u8) & 0x0F]
        }
        ScalarType::I8 => {
            let v: i8 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I16 => {
            let v: i16 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I32 => {
            let v: i32 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I64 => {
            let v: i64 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::F16 => {
            let v = parse_float_or_hex(lit_string);
            half::f16::from_f64(v).to_le_bytes().to_vec()
        }
        ScalarType::BF16 => {
            let v = parse_float_or_hex(lit_string);
            half::bf16::from_f64(v).to_le_bytes().to_vec()
        }
        ScalarType::F32 => {
            let v = parse_float_or_hex(lit_string);
            (v as f32).to_le_bytes().to_vec()
        }
        ScalarType::F64 | ScalarType::TF32 => {
            let v = parse_float_or_hex(lit_string);
            v.to_le_bytes().to_vec()
        }
        ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F8E5M3FNU => {
            let v: u8 = lit_string.parse().unwrap_or(0);
            vec![v]
        }
        ScalarType::F4E2M1FN => {
            let v: u8 = lit_string.parse().unwrap_or(0);
            vec![v & 0x0F]
        }
    }
}

/// Parse a float literal string, handling both decimal ("3.14") and hex ("0x40490fdb") forms.
fn parse_float_or_hex(s: &str) -> f64 {
    if s.starts_with("0x") || s.starts_with("-0x") {
        let negative = s.starts_with('-');
        let hex = if negative { &s[3..] } else { &s[2..] };
        let bits = u64::from_str_radix(hex, 16).unwrap_or(0);
        let v = match hex.len() {
            1..=4 => half::f16::from_bits(bits as u16).to_f64(),
            5..=8 => f32::from_bits(bits as u32) as f64,
            _ => f64::from_bits(bits),
        };
        if negative {
            -v
        } else {
            v
        }
    } else {
        s.parse::<f64>().unwrap_or(0.0)
    }
}
