/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering of calls to foreign tile functions (`#[cutile::foreign]`
//! declarations, see [`crate::foreign_decl`]) to Tile IR `preview$call`.
//!
//! A call `cholesky::<1, 32, f64>("potrf", "lower", a)` becomes
//!
//! ```text
//! %l, %info, %t = preview$call @cholesky<["potrf", "lower", 1, 32, f64]>(%a, %info0, %token)
//! ```
//!
//! The constant arguments are the strings the `&str` arguments resolve to
//! when the call is compiled, then the callee's generic arguments (see
//! [`crate::foreign_decl`] for the order). The operands are the other call
//! arguments followed by a zero-filled tile per
//! output-only result (`%info0`), and an ordering token
//! is threaded last, as cuTile Python does. `tileiras` resolves the callee in
//! the tile libraries the host configured (see [`crate::tile_library`]).

use std::collections::HashMap;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, Location, Module, PreviewConstArg, TileElementType, TileType,
    Type as TileIrType, Value,
};
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::{ExprCall, ItemFn};

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue};
use super::compile_expression::splat_dense_elements;
use super::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::foreign_decl::{ConstParamKind, ForeignFnDecl};
use crate::generics::{GenericArgInference, GenericVars};
use crate::passes::name_resolution::DefKind;
use crate::syn_utils::get_sig_types;

impl<'m> CUDATileFunctionCompiler<'m> {
    /// The callee's generic arguments: inferred from the operand types
    /// (`operand_tys`, one per operand argument), then taken from the
    /// turbofish (`_` stays inferred).
    fn foreign_call_generic_vars(
        &self,
        fn_item: &ItemFn,
        call: &ExprCall,
        caller_generic_vars: &GenericVars,
        operand_tys: &[syn::Type],
    ) -> Result<GenericVars, JITError> {
        let mut inference =
            GenericArgInference::new_function(ForeignFnDecl::operand_signature(&fn_item.sig));
        inference.map_args_to_params(&operand_tys.to_vec(), None, caller_generic_vars)?;
        inference.apply_provided_generics_fn_call(call, caller_generic_vars, &|ty| {
            self.string_tag_text(ty)
        });
        let unresolved: Vec<_> = inference
            .params
            .iter()
            .filter(|p| matches!(inference.param2arg.get(*p), Some(None)))
            .cloned()
            .collect();
        if !unresolved.is_empty() {
            return self.jit_error_result(
                &call.func.span(),
                &format!(
                    "cannot infer generic parameter(s) {} of foreign function `{}` from the \
                     arguments; pass them explicitly with `::<...>`",
                    unresolved
                        .iter()
                        .map(|p| format!("`{p}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    fn_item.sig.ident
                ),
            );
        }
        let mut generic_vars =
            inference.get_generic_vars_instance(caller_generic_vars, self.modules.primitives());
        generic_vars.ordered_param_vars = inference.params.clone();
        Ok(generic_vars)
    }

    /// The type of a foreign call, for type inference. `operand_tys` has one
    /// type per operand argument; `&str` arguments are left out.
    pub(crate) fn foreign_call_return_type(
        &self,
        module_name: &str,
        fn_item: &ItemFn,
        call: &ExprCall,
        caller_generic_vars: &GenericVars,
        operand_tys: &[syn::Type],
    ) -> Result<Option<TileRustType>, JITError> {
        let generic_vars =
            self.foreign_call_generic_vars(fn_item, call, caller_generic_vars, operand_tys)?;
        let (_, return_type) = get_sig_types(&fn_item.sig, None);
        let concrete = {
            let _module_scope = self.push_module_scope(module_name);
            self.concrete_type(&return_type, &generic_vars, call)?
        };
        self.compile_type(&concrete, caller_generic_vars, &HashMap::new())
    }

    /// `ty` with the callee's generic parameters replaced by their values
    /// (`Tile<T, {[M]}>` becomes `Tile<f32, {[64]}>`), so it means the same
    /// in the caller.
    fn concrete_type(
        &self,
        ty: &syn::Type,
        generic_vars: &GenericVars,
        call: &ExprCall,
    ) -> Result<syn::Type, JITError> {
        if let syn::Type::Tuple(tuple) = ty {
            let mut concrete = tuple.clone();
            for elem in concrete.elems.iter_mut() {
                *elem = self.concrete_type(elem, generic_vars, call)?;
            }
            return Ok(syn::Type::Tuple(concrete));
        }
        match self.compile_type(ty, generic_vars, &HashMap::new())? {
            Some(compiled) => Ok(compiled.type_instance.get_instantiated_type().clone()),
            None => self.jit_error_result(
                &call.span(),
                &format!("cannot compile type `{}`", ty.to_token_stream()),
            ),
        }
    }

    /// Compiles `ty` (with the callee's generic arguments) to a tile type.
    fn foreign_tile_type(
        &self,
        ty: &syn::Type,
        generic_vars: &GenericVars,
        call: &ExprCall,
        what: &str,
    ) -> Result<(TileRustType, TileIrType), JITError> {
        let compiled = self.compile_type(ty, generic_vars, &HashMap::new())?;
        match compiled {
            Some(rust_ty) => match &rust_ty.tile_ir_ty {
                Some(
                    ir_ty @ TileIrType::Tile(TileType {
                        element_type: TileElementType::Scalar(_),
                        ..
                    }),
                ) => {
                    let ir_ty = ir_ty.clone();
                    Ok((rust_ty, ir_ty))
                }
                _ => self.jit_error_result(
                    &call.span(),
                    &format!(
                        "{what} of type `{}` is not a tile or scalar; foreign tile functions \
                         take and return tiles and scalars only",
                        ty.to_token_stream()
                    ),
                ),
            },
            None => self.jit_error_result(
                &call.span(),
                &format!("cannot compile {what} type `{}`", ty.to_token_stream()),
            ),
        }
    }

    /// The string an `&str` argument of a foreign call stands for, resolved
    /// when the call is compiled: a literal, a local or parameter bound to
    /// one, a `const` item (`FILL`, `consts::FILL`), or an associated const
    /// of a string tag (`F::NAME` with `F` a generic parameter bound to
    /// `Lower`, `Lower::NAME`, or `<Lower as Trait>::NAME`). Runs in the
    /// caller's scope.
    pub(crate) fn foreign_string_arg(
        &self,
        expr: &syn::Expr,
        param: &str,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<String, JITError> {
        use syn::{Expr, ExprLit, Lit};
        let not_constant = || {
            self.jit_error(
                &expr.span(),
                &format!(
                    "`{param}` must be a string known when the kernel is compiled: a literal, \
                     a `const`, or an associated const such as `F::NAME`; got `{}`",
                    crate::syn_utils::user_expr_tokens(expr)
                ),
            )
        };
        let const_value = |item: &syn::ItemConst| match item.expr.as_ref() {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Ok(s.value()),
            _ => Err(not_constant()),
        };
        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Ok(s.value()),
            Expr::Paren(inner) => self.foreign_string_arg(&inner.expr, param, generic_vars, ctx),
            Expr::Group(inner) => self.foreign_string_arg(&inner.expr, param, generic_vars, ctx),
            Expr::Reference(r) if r.mutability.is_none() => {
                self.foreign_string_arg(&r.expr, param, generic_vars, ctx)
            }
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let name = path.path.segments[0].ident.to_string();
                if let Some(value) = ctx.vars.get(&name) {
                    return match &value.string_literal {
                        Some(Expr::Lit(ExprLit {
                            lit: Lit::Str(s), ..
                        })) => Ok(s.value()),
                        _ => Err(not_constant()),
                    };
                }
                let item = self
                    .modules
                    .name_resolver
                    .resolve_const(&name, &self.current_module())
                    .map_err(|message| self.jit_error(&expr.span(), &message))?
                    .ok_or_else(not_constant)?;
                const_value(item)
            }
            Expr::Path(path)
                if path.path.segments.len() >= 2
                    || path.qself.as_ref().is_some_and(|q| q.position == 0) =>
            {
                let segments = &path.path.segments;
                let const_name = segments.last().unwrap().ident.to_string();
                // A `const` item named by its path (`consts::FILL`).
                if path.qself.is_none() {
                    if let Ok(Some(def)) = self.modules.resolve_item(
                        &path.path,
                        &self.current_module(),
                        DefKind::Const,
                    ) {
                        if let Some(item) = self.modules.name_resolver.get_const(&def) {
                            return const_value(item);
                        }
                    }
                }
                // `Tag::NAME`; `<Tag>::NAME` for a type that is not a plain
                // path (`<Alias<Tag>>::NAME`); or `<Tag as Trait>::NAME`,
                // which reads only the impls of `Trait`.
                let (mut tag, of_trait): (syn::Path, Option<syn::Path>) = match &path.qself {
                    Some(qself) => match qself.ty.as_ref() {
                        syn::Type::Path(p)
                            if p.qself.is_none() && qself.position + 1 == segments.len() =>
                        {
                            let of_trait = (qself.position > 0).then(|| syn::Path {
                                leading_colon: path.path.leading_colon,
                                segments: segments.iter().take(qself.position).cloned().collect(),
                            });
                            (p.path.clone(), of_trait)
                        }
                        _ => return Err(not_constant()),
                    },
                    None => {
                        let mut tag = path.path.clone();
                        tag.segments.pop();
                        tag.segments.pop_punct();
                        (tag, None)
                    }
                };
                // A generic parameter stands for the type it is bound to,
                // which is spelled from the tag's defining module.
                if let Some(bound) = tag
                    .get_ident()
                    .and_then(|id| generic_vars.inst_types.get(&id.to_string()))
                {
                    tag = syn::parse_str(bound).map_err(|_| not_constant())?;
                }
                let def = self
                    .modules
                    .resolve_struct(&tag, &self.current_module())
                    .map_err(|message| self.jit_error(&expr.span(), &message))?
                    .ok_or_else(not_constant)?;
                let of_trait = match of_trait {
                    Some(trait_path) => Some(
                        self.modules
                            .resolve_item(&trait_path, &self.current_module(), DefKind::Trait)
                            .map_err(|message| self.jit_error(&expr.span(), &message))?
                            .ok_or_else(not_constant)?,
                    ),
                    None => None,
                };
                let mut values: Vec<String> = self
                    .modules
                    .associated_str_consts(&def, of_trait.as_ref())
                    .into_iter()
                    .filter(|(name, _)| *name == const_name)
                    .map(|(_, value)| value)
                    .collect();
                values.dedup();
                match values.as_slice() {
                    [value] => Ok(value.clone()),
                    [] => Err(self.jit_error(
                        &expr.span(),
                        &format!(
                            "`{}::{}` has no `&str` associated const `{const_name}` with a \
                             string literal value",
                            def.module, def.name
                        ),
                    )),
                    _ => Err(self.jit_error(
                        &expr.span(),
                        &format!(
                            "`{}::{}` has several associated consts `{const_name}` with \
                             different values ({}); name the trait, as in \
                             `<T as Trait>::{const_name}`",
                            def.module,
                            def.name,
                            values
                                .iter()
                                .map(|v| format!("\"{v}\""))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    )),
                }
            }
            _ => Err(not_constant()),
        }
    }

    /// The constant argument bound to `name`, a generic parameter of `kind`.
    fn foreign_const_arg(
        &self,
        name: &str,
        kind: ConstParamKind,
        generic_vars: &GenericVars,
        call: &ExprCall,
    ) -> Result<PreviewConstArg, JITError> {
        let missing = || {
            self.jit_error(
                &call.func.span(),
                &format!("no value for generic parameter `{name}` of this foreign call"),
            )
        };
        match kind {
            ConstParamKind::Int => generic_vars
                .inst_i32
                .get(name)
                .map(|v| PreviewConstArg::Int(i64::from(*v)))
                .ok_or_else(missing),
            ConstParamKind::Datatype => {
                let rust_name = generic_vars.inst_types.get(name).ok_or_else(missing)?;
                let scalar = self
                    .modules
                    .get_primitives_attrs("ElementType", rust_name)
                    .and_then(|attrs| attrs.parse_string("name"))
                    .and_then(|tile_name| super::_type::scalar_from_name(&tile_name))
                    .ok_or_else(|| {
                        self.jit_error(
                            &call.func.span(),
                            &format!(
                                "generic parameter `{name}` must be a tile element type, got \
                                 `{rust_name}`"
                            ),
                        )
                    })?;
                Ok(PreviewConstArg::Datatype(TileIrType::Scalar(scalar)))
            }
            ConstParamKind::Str(_) => unreachable!("string constants come from `&str` arguments"),
        }
    }

    /// Lowers a call to a foreign tile function.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_foreign_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        module_name: &str,
        fn_item: &ItemFn,
        call: &ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let decl = match ForeignFnDecl::from_item_fn(fn_item) {
            Ok(Some(decl)) => decl,
            Ok(None) => unreachable!("dispatched on the foreign attribute"),
            Err(e) => return self.jit_error_result(&call.span(), &e.to_string()),
        };
        let location = self.ir_location(&call.span());
        if let Some(capabilities) = &self.target_capabilities {
            cutile_ir::requirements::Feature::ForeignCall
                .check(capabilities, &location)
                .map_err(JITError::from)?;
        }
        if call.args.len() != fn_item.sig.inputs.len() {
            return self.jit_error_result(
                &call.span(),
                &format!(
                    "`{}` takes {} argument(s), got {}",
                    decl.ident,
                    fn_item.sig.inputs.len(),
                    call.args.len()
                ),
            );
        }
        // String constants resolve in the caller's scope, before entering
        // the declaration's module.
        let strings = decl
            .const_params
            .iter()
            .map(|p| match p.kind {
                ConstParamKind::Str(index) => self
                    .foreign_string_arg(&call.args[index], &p.name, generic_vars, ctx)
                    .map(Some),
                ConstParamKind::Int | ConstParamKind::Datatype => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let operand_exprs: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]> = decl
            .params
            .iter()
            .map(|p| call.args[p.arg_index].clone())
            .collect();
        let args = self.compile_call_args(module, block_id, &operand_exprs, generic_vars, ctx)?;
        let arg_rust_tys: Vec<_> = args.iter().map(|a| a.ty.rust_ty.clone()).collect();
        let callee_generic_vars =
            self.foreign_call_generic_vars(fn_item, call, generic_vars, &arg_rust_tys)?;
        // The declaration's types and consts resolve in its own module.
        let _module_scope = self.push_module_scope(module_name);

        let const_args = decl
            .const_params
            .iter()
            .zip(strings)
            .map(|(p, string)| {
                let arg = match string {
                    Some(value) => Ok(PreviewConstArg::Str(value)),
                    None => self.foreign_const_arg(&p.name, p.kind, &callee_generic_vars, call),
                };
                arg.map(Attribute::PreviewConstArg)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut operands = Vec::with_capacity(args.len() + decl.implicit_out_results().len() + 1);
        for ((param, arg), arg_expr) in decl.params.iter().zip(&args).zip(&operand_exprs) {
            let (_, param_ty) =
                self.foreign_tile_type(&param.ty, &callee_generic_vars, call, "parameter")?;
            let Some(value) = arg.value else {
                return self.jit_error_result(
                    &arg_expr.span(),
                    &format!(
                        "argument `{}` of a foreign call must be a tile or scalar value",
                        crate::syn_utils::user_expr_tokens(arg_expr)
                    ),
                );
            };
            let arg_ty = module.value_type(value);
            if arg_ty != &param_ty {
                return self.jit_error_result(
                    &arg_expr.span(),
                    &format!(
                        "argument `{}` of `{}` has type {}, but parameter `{}` expects {}",
                        crate::syn_utils::user_expr_tokens(arg_expr),
                        decl.ident,
                        cutile_ir::ir::format_type(arg_ty),
                        param.name,
                        cutile_ir::ir::format_type(&param_ty)
                    ),
                );
            }
            operands.push(value);
        }
        // Output-only results are trailing operands, passed zero-filled.
        for ty in decl.implicit_out_results() {
            let (_, out_ty) =
                self.foreign_tile_type(ty, &callee_generic_vars, call, "output-only result")?;
            operands.push(zero_tile(module, block_id, &out_ty, location.clone())?);
        }
        let (token, token_vars) = self.foreign_call_token(module, block_id, ctx, &location);
        operands.push(token);

        let concrete_results = decl
            .results
            .iter()
            .map(|ty| self.concrete_type(ty, &callee_generic_vars, call))
            .collect::<Result<Vec<_>, _>>()?;
        let results = concrete_results
            .iter()
            .map(|ty| self.foreign_tile_type(ty, generic_vars, call, "result"))
            .collect::<Result<Vec<_>, _>>()?;
        let (op_id, values) = OpBuilder::new(Opcode::PreviewCall, location)
            .operands(operands)
            .results(
                results
                    .iter()
                    .map(|(_, ir_ty)| ir_ty.clone())
                    .chain([TileIrType::Token]),
            )
            .attr("callee", Attribute::String(decl.symbol.clone()))
            .attr("const_args", Attribute::Array(const_args))
            .build(module);
        append_op(module, block_id, op_id);

        let result_token = *values.last().expect("the call returns a token");
        // Later accesses in this block are ordered after the call through
        // their tokens. Regions compile from a copy of the context, so these
        // updates stay inside one; the carried token takes the call's
        // ordering out of blocks, branches, loops and inlined calls (see
        // `publish_ffi_token`).
        for name in token_vars {
            super::shared_utils::set_view_token(&name, result_token, ctx);
        }
        for var in [FFI_TOKEN_VAR, FFI_ACCESS_VAR] {
            if let Some(token) = ctx.vars.get_mut(var) {
                token.value = Some(result_token);
            }
        }

        let mut tile_values: Vec<TileRustValue> = results
            .into_iter()
            .zip(&values)
            .map(|((rust_ty, _), &value)| TileRustValue::new_value_kind_like(value, rust_ty))
            .collect();
        match &fn_item.sig.output {
            syn::ReturnType::Type(_, ty) if matches!(ty.as_ref(), syn::Type::Tuple(t) if !t.elems.is_empty()) =>
            {
                let tuple: syn::Type = syn::parse_quote!((#(#concrete_results,)*));
                let Some(tuple_ty) = self.compile_type(&tuple, generic_vars, &HashMap::new())?
                else {
                    return self.jit_error_result(
                        &call.span(),
                        &format!(
                            "cannot compile the result type of foreign function `{}`",
                            decl.ident
                        ),
                    );
                };
                Ok(Some(TileRustValue::new_compound(tile_values, tuple_ty)))
            }
            _ => Ok(tile_values.pop()),
        }
    }

    /// The token a foreign call consumes: the join of the carried foreign-call
    /// token ([`FFI_TOKEN_VAR`]) and the carried memory-access token
    /// ([`FFI_ACCESS_VAR`]), which together cover every earlier foreign call
    /// and memory access. Also returns the names of the token-carrying views
    /// and tensors, which the call's result token is published to.
    fn foreign_call_token(
        &self,
        module: &mut Module,
        block_id: BlockId,
        ctx: &CompilerContext,
        location: &Location,
    ) -> (Value, Vec<String>) {
        let mut names: Vec<String> = ctx
            .vars
            .iter()
            .filter(|(_, v)| token_of(v).is_some())
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        let mut tokens: Vec<Value> = [FFI_TOKEN_VAR, FFI_ACCESS_VAR]
            .iter()
            .filter_map(|var| ctx.vars.get(*var).and_then(|v| v.value))
            .collect();
        tokens.dedup();
        let token = join_or_make_token(module, block_id, tokens, location);
        (token, names)
    }

    /// Creates the carried foreign-call token at the kernel's entry when a
    /// loaded module declares a foreign function. Kernels without foreign
    /// declarations keep their IR unchanged.
    pub(crate) fn seed_ffi_token(
        &self,
        module: &mut Module,
        block_id: BlockId,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        span: &proc_macro2::Span,
    ) -> Result<(), JITError> {
        if !self.modules.declares_foreign_functions() {
            return Ok(());
        }
        let token = join_or_make_token(module, block_id, vec![], &self.ir_location(span));
        let token_ty = self.token_type(generic_vars)?;
        for var in [FFI_TOKEN_VAR, FFI_ACCESS_VAR] {
            ctx.vars.insert(
                var.to_string(),
                TileRustValue::new_primitive(token, token_ty.clone(), None),
            );
        }
        Ok(())
    }

    /// Joins the tokens of the memory accesses emitted since `values_before`
    /// directly in `block_id` (loads, stores, atomics, fences) into the
    /// carried memory-access token, so later foreign calls are ordered after
    /// them, also from the next iteration of a loop and after a branch. Only
    /// foreign calls consume that token: accesses stay ordered among
    /// themselves as before. A no-op without a carried token.
    pub(crate) fn record_memory_accesses(
        &self,
        module: &mut Module,
        block_id: BlockId,
        ctx: &mut CompilerContext,
        values_before: usize,
    ) {
        let Some(access) = ctx.vars.get(FFI_ACCESS_VAR).and_then(|v| v.value) else {
            return;
        };
        // Most ops make no token: skip the scan of the whole block below.
        if !module
            .values_since(values_before)
            .any(|(_, data)| matches!(data.ty, TileIrType::Token))
        {
            return;
        }
        let in_block: std::collections::HashSet<cutile_ir::ir::OpId> =
            module.block(block_id).ops.iter().copied().collect();
        let tokens: Vec<Value> = module
            .values_since(values_before)
            .filter_map(|(value, data)| match data.producer {
                cutile_ir::ir::ValueProducer::OpResult { op, .. }
                    if matches!(data.ty, TileIrType::Token)
                        && in_block.contains(&op)
                        && !matches!(
                            module.op(op).opcode,
                            Opcode::MakeToken | Opcode::JoinTokens | Opcode::PreviewCall
                        ) =>
                {
                    Some(value)
                }
                _ => None,
            })
            .collect();
        if tokens.is_empty() {
            return;
        }
        let location = module
            .block(block_id)
            .ops
            .last()
            .map(|op| module.op_location(*op).clone())
            .unwrap_or(Location::Unknown);
        let joined = join_or_make_token(
            module,
            block_id,
            std::iter::once(access).chain(tokens).collect(),
            &location,
        );
        if let Some(var) = ctx.vars.get_mut(FFI_ACCESS_VAR) {
            var.value = Some(joined);
        }
    }

    /// Joins the carried foreign-call token into every view's and tensor's
    /// token, so later memory accesses are ordered after the foreign calls
    /// it covers. A no-op without a carried token.
    pub(crate) fn join_ffi_token_into_views(
        &self,
        module: &mut Module,
        block_id: BlockId,
        ctx: &mut CompilerContext,
        span: &proc_macro2::Span,
    ) {
        let Some(ffi) = ffi_token(ctx) else {
            return;
        };
        let location = self.ir_location(span);
        let mut joined: HashMap<Value, Value> = HashMap::new();
        let names: Vec<String> = ctx
            .vars
            .iter()
            .filter(|(_, v)| token_of(v).is_some_and(|t| t != ffi))
            .map(|(name, _)| name.clone())
            .collect();
        for name in names {
            let token = token_of(&ctx.vars[&name]).expect("filtered above");
            let new_token = *joined.entry(token).or_insert_with(|| {
                join_or_make_token(module, block_id, vec![token, ffi], &location)
            });
            super::shared_utils::set_view_token(&name, new_token, ctx);
        }
    }

    /// After a region (a `{}` or `unsafe {}` block, a branch, a loop, an
    /// inlined call): when foreign calls in it advanced the carried token past
    /// `before`, orders every later access after them.
    pub(crate) fn publish_ffi_token(
        &self,
        module: &mut Module,
        block_id: BlockId,
        ctx: &mut CompilerContext,
        before: Option<Value>,
        span: &proc_macro2::Span,
    ) {
        if ffi_token(ctx) != before {
            self.join_ffi_token_into_views(module, block_id, ctx, span);
        }
    }
}

/// The variable carrying the token of the latest foreign calls through
/// branches, loops and inlined calls, like a mutable variable (see
/// [`CUDATileFunctionCompiler::seed_ffi_token`]).
pub(crate) const FFI_TOKEN_VAR: &str = "__cutile_ffi_token__";

/// The variable carrying the join of every memory access's token since the
/// latest foreign call, through branches, loops and inlined calls (see
/// [`CUDATileFunctionCompiler::record_memory_accesses`]). Only foreign calls
/// consume it.
pub(crate) const FFI_ACCESS_VAR: &str = "__cutile_ffi_accesses__";

/// The carried foreign-call token, if the kernel has one.
pub(crate) fn ffi_token(ctx: &CompilerContext) -> Option<Value> {
    ctx.vars.get(FFI_TOKEN_VAR).and_then(|v| v.value)
}

/// Adds the carried foreign-call and memory-access tokens to a region's
/// carried variables.
pub(crate) fn carry_ffi_token(ctx: &CompilerContext, vars: &mut Vec<String>) {
    for var in [FFI_TOKEN_VAR, FFI_ACCESS_VAR] {
        if ctx.vars.contains_key(var) && !vars.iter().any(|v| v == var) {
            vars.push(var.to_string());
        }
    }
}

/// Copies the carried foreign-call and memory-access tokens from `from` to
/// `to`, into and out of an inlined call.
pub(crate) fn copy_ffi_tokens(from: &CompilerContext, to: &mut CompilerContext) {
    for var in [FFI_TOKEN_VAR, FFI_ACCESS_VAR] {
        if let Some(value) = from.vars.get(var) {
            to.vars.insert(var.to_string(), value.clone());
        }
    }
}

/// One token: the only one of `tokens`, their join, or a fresh token.
fn join_or_make_token(
    module: &mut Module,
    block_id: BlockId,
    tokens: Vec<Value>,
    location: &Location,
) -> Value {
    if let [token] = tokens.as_slice() {
        return *token;
    }
    let opcode = if tokens.is_empty() {
        Opcode::MakeToken
    } else {
        Opcode::JoinTokens
    };
    let (op_id, results) = OpBuilder::new(opcode, location.clone())
        .operands(tokens)
        .result(TileIrType::Token)
        .build(module);
    append_op(module, block_id, op_id);
    results[0]
}

fn token_of(value: &TileRustValue) -> Option<Value> {
    value.type_meta.as_ref()?.fields.get("token")?.value
}

/// A tile of `ty` filled with zeros (a splat constant).
fn zero_tile(
    module: &mut Module,
    block_id: BlockId,
    ty: &TileIrType,
    location: Location,
) -> Result<Value, JITError> {
    let TileIrType::Tile(TileType {
        element_type: TileElementType::Scalar(scalar),
        ..
    }) = ty
    else {
        unreachable!("foreign operand types are scalar-element tiles")
    };
    let value = splat_dense_elements(*scalar, vec![0; scalar.byte_width()], || Ok(ty.clone()))?;
    let (op_id, results) = OpBuilder::new(Opcode::Constant, location)
        .result(ty.clone())
        .attr("value", Attribute::DenseElements(value))
        .build(module);
    append_op(module, block_id, op_id);
    Ok(results[0])
}
