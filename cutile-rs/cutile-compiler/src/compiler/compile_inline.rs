/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_inline.rs` — handles inlining of
//! function calls and method calls. Only type and IR-emission changes; the
//! control flow, dispatch logic, and variable binding are identical.

use syn::spanned::Spanned;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, Mutability, TileRustValue};
use super::shared_utils::{STACK_GROW_SIZE, STACK_RED_ZONE};
use super::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::{GenericArgInference, GenericVars};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::ir::{BlockId, Module};

use quote::ToTokens;
use std::collections::HashMap;
use syn::{Expr, ExprCall, ExprMethodCall, ItemFn, Type};

/// Port of `crate::compiler::utils::update_type_meta` for compiler2 value types.
/// Copies mutable type metadata fields from inner to outer context using a variable name mapping.
fn update_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    outer2inner_vars: &HashMap<String, String>,
    field_name: String,
) {
    super::shared_utils::update_type_meta(
        inner_block_vars,
        outer_block_vars,
        outer2inner_vars,
        field_name,
    );
    let outer_keys: Vec<String> = outer_block_vars.var_keys();
    for outer_key in &outer_keys {
        let Some(outer_val) = outer_block_vars.vars.get(outer_key) else {
            continue;
        };
        if outer_val.mutability == Mutability::Mutable {
            if let Some(inner_key) = outer2inner_vars.get(outer_key) {
                if let Some(inner_val) = inner_block_vars.vars.get(inner_key) {
                    if inner_val.mutability == Mutability::Mutable {
                        // The callee advanced this resource's token; propagate it
                        // up the borrow link to the root tensor, so a later view
                        // of the same tensor is ordered after these writes. Here
                        // (the caller boundary) both the view and its root tensor
                        // are in scope, unlike inside the callee frame.
                        super::shared_utils::propagate_token_to_root(outer_key, outer_block_vars);
                    }
                }
            }
        }
    }
}

impl<'m> CUDATileFunctionCompiler<'m> {
    #[allow(clippy::ptr_arg)] // public signature stays as-is
    pub fn inline_function_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        module_name: &String,
        fn_item: &ItemFn,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let normalized_fn_item = crate::type_aliases::normalize_item_fn_param_type_aliases(
            fn_item,
            self.modules.type_aliases(),
        )
        .map_err(JITError::Generic)?;
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let fn_item = &normalized_fn_item;
            let _inline_function_call_debug_str = call_expr.to_token_stream().to_string();
            // println!("enter_function_call: {}", call_expr.to_token_stream().to_string());

            // Compile caller arguments.
            let call_arg_values =
                self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
            // Arguments above belong to the caller's frame; everything from
            // here on compiles the callee's body inline. User functions keep
            // their source spans, including across module/file boundaries.
            // Core operations remain attributed to the user's call site.
            let _call_site = if self.has_inline_source(module_name) {
                self.push_call_site(
                    &call_expr.span(),
                    &fn_item.sig.ident.to_string(),
                    &fn_item.sig.ident.span(),
                    module_name,
                )
            } else {
                self.push_opaque_call_site(&call_expr.span())
            };
            // Map function generic params to caller generic args.
            let mut generic_arg_inference = GenericArgInference::new_function(fn_item.sig.clone());
            let call_arg_rust_tys = call_arg_values
                .iter()
                .map(|arg| arg.ty.rust_ty.clone())
                .collect::<Vec<_>>();
            // println!("{call_arg_rust_tys:#?}");
            generic_arg_inference.map_args_to_params(&call_arg_rust_tys, None, generic_vars)?;
            // Bind new variables.
            // The variables must:
            // - Have the names of the parameters in the callee.
            // - Have the type of parameters in the callee. This is an inductive property.
            let param_names = get_sig_param_names(&fn_item.sig);
            let (input_params, _output_param) = get_sig_types(&fn_item.sig, None);
            let mut call_variables = CompilerContext::empty();
            // The callee body is compiled into the caller's current block, so
            // the caller's loop context governs check hoisting inside it.
            call_variables.loop_frames = ctx.loop_frames.clone();
            call_variables.token_update_in_region =
                ctx.token_update_in_region || ctx.inside_for || ctx.innermost_loop.is_some();
            call_variables.module_scope.push(module_name.clone());
            // The callee's body block is a function body: a top-level `return`
            // there yields the call's value.
            call_variables.fn_body = true;
            // The ABI wrapper's final call is the kernel body, not a helper.
            call_variables.kernel_entry = ctx.kernel_entry
                && module_name == &self.module_name
                && crate::kernel_naming::KernelNaming::canonical_public_name(
                    &fn_item.sig.ident.to_string(),
                ) == self._function_name;
            let mut outer2inner_map = HashMap::new();
            let sig_param_mutability = get_sig_param_mutability(&fn_item.sig);

            for i in 0..param_names.len() {
                let param_name = &param_names[i];
                let param_type = &input_params[i];
                let mut param_val = call_arg_values[i].clone();
                // TODO (hme): This may not be enough, depending on what level of inspection we require of compound / struct types.
                param_val.ty.rust_ty = param_type.clone();
                param_val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                call_variables.vars.insert(param_name.clone(), param_val);
                if let Some(call_arg_name) = get_ident_from_expr(&call_expr.args[i]) {
                    if ctx
                        .function_level_bindings
                        .contains(&call_arg_name.to_string())
                    {
                        call_variables
                            .function_level_bindings
                            .insert(param_name.clone());
                    }
                    outer2inner_map.insert(call_arg_name.to_string(), param_name.clone());
                };
            }
            // Remap generic parameters.
            let expr_generic_args = get_call_expression_generics(call_expr);
            let mut call_generic_vars = if GenericVars::is_empty(&fn_item.sig.generics) {
                // If there are no generics, we're done.
                GenericVars::empty(&fn_item.sig.generics)?
            } else if expr_generic_args.is_some() {
                // If the caller specifies generics args, use them.
                generic_vars.from_expr_generic_args(&fn_item.sig.generics, &expr_generic_args)?
            } else {
                // If nothing is specified, try to infer an instance of GenericVars.
                let mut generic_arg_inference =
                    GenericArgInference::new_function(fn_item.sig.clone());
                let call_arg_rust_tys = call_arg_values
                    .iter()
                    .map(|arg| arg.ty.rust_ty.clone())
                    .collect::<Vec<_>>();
                generic_arg_inference.map_args_to_params(&call_arg_rust_tys, None, generic_vars)?;
                // println!("inline_function_call {:#?}: generic_vars={generic_vars:#?} \nexpr_generic_args={expr_generic_args:#?} \ngeneric_arg_inference={generic_arg_inference:#?}", fn_item.sig.ident.to_string());
                generic_arg_inference
                    .get_generic_vars_instance(generic_vars, self.modules.primitives())
            };
            // The callee's body compiles in its own module's scope: its
            // `const`s, and the consts named in its types, are its module's.
            let _module_scope = self.push_module_scope(module_name);
            self.add_module_const_vars(&mut call_generic_vars);
            // Add function call const generics as variables.
            for (key, value) in call_generic_vars.ordered_inst_i32() {
                let tr_val = self.compile_constant(module, block_id, &call_generic_vars, value)?;
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            for (key, value) in call_generic_vars.ordered_inst_bool() {
                let tr_val =
                    self.compile_bool_constant(module, block_id, &call_generic_vars, value)?;
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            // Add function call CGAs arrays as variables.
            for (key, value) in call_generic_vars.ordered_inst_array() {
                let arr_expr = syn::parse2::<Expr>(format!("{value:?}").parse().unwrap()).unwrap();
                let arr_ty =
                    syn::parse2::<Type>(format!("[i32;{}]", value.len()).parse().unwrap()).unwrap();
                let ty = self.compile_type(&arr_ty, &call_generic_vars, &HashMap::new())?;
                let tr_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &arr_expr,
                        &call_generic_vars,
                        &mut call_variables,
                        ty,
                    )?
                    .expect("Failed to compile CGA as var.");
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            let initial_types = call_variables
                .vars
                .iter()
                .map(|(name, value)| (name.clone(), value.ty.clone()))
                .collect::<HashMap<_, _>>();
            let mut typed_fn_item = fn_item.clone();
            crate::passes::node_ids::assign_expr_ids(&mut typed_fn_item);
            let typeck_results = crate::passes::type_inference::infer_function(
                self,
                &typed_fn_item,
                &call_generic_vars,
                initial_types,
            )?;
            let lowered_fn_item = crate::passes::typed_dispatch_lowering::lower_function(
                &typed_fn_item,
                &typeck_results,
            );
            // println!("inline_function_call {:#?}: generic_args={generic_args:#?} \nexpr_generic_args={expr_generic_args:#?} \ncall_generic_args={call_generic_args:#?}", fn_item.sig.ident.to_string());
            // println!("inline_function_call {:#?}: \n variables={call_variables:#?}", fn_item.sig.ident.to_string());
            let previous_typeck_results = self.typeck_results.replace(Some(typeck_results));
            let result = self.compile_block(
                module,
                block_id,
                &lowered_fn_item.block,
                &call_generic_vars,
                &mut call_variables,
                return_type.clone(),
            );
            self.typeck_results.replace(previous_typeck_results);
            let result = result?;
            update_type_meta(
                &mut call_variables,
                ctx,
                &outer2inner_map,
                "token".to_string(),
            );
            // println!("exit_function_call: {}", call_expr.to_token_stream().to_string());
            if let Some(mut res) = result {
                if let Some(rt) = return_type {
                    // Use specified return type.
                    res.ty = rt;
                    return Ok(Some(res));
                };
                let type_params = res.ty.params;
                // println!("inline call res.ty.params: {:#?}", type_params);
                let Some(derived_ret_ty) = self.derive_type(
                    module,
                    block_id,
                    &Expr::Call(call_expr.clone()),
                    Some(type_params),
                    generic_vars,
                    ctx,
                )?
                else {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        &format!(
                            "Failed to determine typeck return type for inlined function call `{}`",
                            call_expr.to_token_stream()
                        ),
                    );
                };
                res.ty = derived_ret_ty;
                Ok(Some(res))
            } else {
                Ok(None)
            }
        }) // stacker::maybe_grow
    }

    pub fn inline_method_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _inline_method_call_debug_str = method_call_expr.to_token_stream().to_string();
            // println!("enter_method_call: {}", method_call_expr.to_token_stream().to_string());
            // Compile caller arguments.
            // Receiver is prepended to args, so value of receiver is present for method calls.
            // args have generics from outer scope for both receiver + method call args.
            let mut args = method_call_expr.args.clone();
            args.insert(0, *method_call_expr.receiver.clone());
            let call_arg_values =
                self.compile_call_args(module, block_id, &args, generic_vars, ctx)?;
            let receiver_rust_ty = &call_arg_values[0].ty.rust_ty;
            let call_arg_rust_tys = call_arg_values
                .iter()
                .map(|arg| arg.ty.rust_ty.clone())
                .collect::<Vec<_>>();
            let selected_method = self.typeck_method_selection(method_call_expr);
            let (module_name, impl_item, impl_method, selected_generic_vars) =
                if let Some(selection) = selected_method {
                    (
                        selection.module_name,
                        selection.impl_item,
                        selection.impl_method,
                        Some(selection.generic_vars),
                    )
                } else {
                    let impl_item_fn = self.modules.get_impl_item_fn(
                        receiver_rust_ty,
                        method_call_expr,
                        generic_vars,
                        &call_arg_rust_tys,
                    )?;
                    if impl_item_fn.is_none() {
                        return self.jit_error_result(
                            &method_call_expr.method.span(),
                            &format!(
                                "method `{}` not found for receiver type `{}`",
                                method_call_expr.method,
                                receiver_rust_ty.to_token_stream()
                            ),
                        );
                    }
                    let (module_name, impl_item, impl_method) = impl_item_fn.unwrap();
                    (module_name, impl_item, impl_method, None)
                };
            // User methods need the same source/inline-frame preservation
            // as free functions. Core methods use the user's call site.
            let _call_site = if self.has_inline_source(&module_name) {
                self.push_call_site(
                    &method_call_expr.span(),
                    &impl_method.sig.ident.to_string(),
                    &impl_method.sig.ident.span(),
                    &module_name,
                )
            } else {
                self.push_opaque_call_site(&method_call_expr.span())
            };
            // println!("Expr::MethodCall: {:#?}, generic_vars: {generic_vars:#?}", impl_item_fn.to_token_stream().to_string());

            // Remap function parameters.
            // Do this by constructing new values from the method's parameters.
            let self_ty = &*impl_item.self_ty;
            // Note that self_ty here is treated as a param type.
            // Bind new variables.
            // The variables must:
            // - Have the names of the parameters in the callee.
            // - Have the type of parameters in the callee. This is an inductive property.
            // get_sig_param_names includes value for self if the signature contains self.
            let param_names = get_sig_param_names(&impl_method.sig);
            let (input_params, _output_param) = get_sig_types(&impl_method.sig, Some(self_ty));
            let mut call_variables = CompilerContext::empty();
            // The callee body is compiled into the caller's current block, so
            // the caller's loop context governs check hoisting inside it.
            call_variables.loop_frames = ctx.loop_frames.clone();
            call_variables.token_update_in_region =
                ctx.token_update_in_region || ctx.inside_for || ctx.innermost_loop.is_some();
            call_variables.module_scope.push(module_name.clone());
            call_variables.fn_body = true;
            let mut outer2inner_map = HashMap::new();
            let sig_param_mutability = get_sig_param_mutability(&impl_method.sig);
            for i in 0..param_names.len() {
                let param_name = &param_names[i];
                let param_type = &input_params[i];
                let mut param_val = call_arg_values[i].clone();
                // TODO (hme): This may not be enough, depending on what level of inspection we require of compound / struct types.
                param_val.ty.rust_ty = param_type.clone();
                param_val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                call_variables.vars.insert(param_name.clone(), param_val);
                // Including self here.
                if let Some(call_arg_name) = get_ident_from_expr(&args[i]) {
                    if ctx
                        .function_level_bindings
                        .contains(&call_arg_name.to_string())
                    {
                        call_variables
                            .function_level_bindings
                            .insert(param_name.clone());
                    }
                    outer2inner_map.insert(call_arg_name.to_string(), param_name.clone());
                };
            }
            // Remap generic parameters.
            // This is different from a function call, because passing generics to a method
            // does not capture all generics available within the method.
            let mut call_generic_vars = if let Some(selected_generic_vars) = selected_generic_vars {
                selected_generic_vars
            } else {
                let generic_arg_inference =
                    GenericArgInference::new_method(&impl_item, &impl_method);
                if generic_arg_inference.param2arg.is_empty() {
                    // There are no generics in this method.
                    GenericVars::empty(&impl_method.sig.generics)?
                } else {
                    crate::passes::type_inference::infer_method_generics(
                        &impl_item,
                        &impl_method,
                        method_call_expr,
                        &call_arg_rust_tys,
                        self_ty,
                        generic_vars,
                        self.modules.primitives(),
                    )?
                }
            };
            // The callee's body compiles in its own module's scope: its
            // `const`s, and the consts named in its types, are its module's.
            let _module_scope = self.push_module_scope(module_name.as_str());
            self.add_module_const_vars(&mut call_generic_vars);

            // Add method call const generics as variables.
            for (key, value) in call_generic_vars.ordered_inst_i32() {
                let tr_val = self.compile_constant(module, block_id, generic_vars, value)?;
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            for (key, value) in call_generic_vars.ordered_inst_bool() {
                let tr_val = self.compile_bool_constant(module, block_id, generic_vars, value)?;
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            for (key, value) in call_generic_vars.ordered_inst_array() {
                let arr_expr = syn::parse2::<Expr>(format!("{value:?}").parse().unwrap()).unwrap();
                let arr_ty =
                    syn::parse2::<Type>(format!("[i32;{}]", value.len()).parse().unwrap()).unwrap();
                let ty = self.compile_type(&arr_ty, &call_generic_vars, &HashMap::new())?;
                let tr_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &arr_expr,
                        &call_generic_vars,
                        &mut call_variables,
                        ty,
                    )?
                    .expect("Failed to compile CGA as var.");
                call_variables.vars.insert(key.to_string(), tr_val);
            }
            // println!("inline_method_call {:#?}: generic_vars={generic_vars:#?} \nexpr_generic_args={expr_generic_args:#?} \ncall_generic_args={call_generic_args:#?}", impl_method.sig.ident.to_string());
            // Keep the body in its original source coordinate system.
            let mut compile_block = impl_method.block.clone();
            crate::passes::node_ids::assign_block_expr_ids(&mut compile_block);
            let initial_types = call_variables
                .vars
                .iter()
                .map(|(name, value)| (name.clone(), value.ty.clone()))
                .collect::<HashMap<_, _>>();
            let mut typed_method = impl_method.clone();
            typed_method.block = compile_block.clone();
            let typeck_results = crate::passes::type_inference::infer_method(
                self,
                &impl_item,
                &typed_method,
                self_ty,
                &call_generic_vars,
                initial_types,
            )?;
            let previous_typeck_results = self.typeck_results.replace(Some(typeck_results));
            let result = self.compile_block(
                module,
                block_id,
                &compile_block,
                &call_generic_vars,
                &mut call_variables,
                return_type.clone(),
            );
            self.typeck_results.replace(previous_typeck_results);
            let result = result?;
            update_type_meta(
                &mut call_variables,
                ctx,
                &outer2inner_map,
                "token".to_string(),
            );
            // If this method advanced a mutable view's token (a store) inside a
            // loop threading that tensor's token, join the output into the loop
            // accumulator. A no-op outside such a loop.
            for outer_key in outer2inner_map.keys() {
                let is_mutable = ctx
                    .vars
                    .get(outer_key)
                    .map(|v| v.mutability == Mutability::Mutable)
                    .unwrap_or(false);
                if is_mutable {
                    self.accumulate_loop_token(
                        module,
                        block_id,
                        ctx,
                        outer_key,
                        &call_generic_vars,
                    )?;
                }
            }
            // println!("exit_method_call: {}", method_call_expr.to_token_stream().to_string());
            if let Some(mut res) = result {
                if let Some(rt) = return_type {
                    // Use specified rust type.
                    // We don't want the entire provided type, because the computed TileRustValue
                    // contains cuda tile type information that can't be inferred.
                    res.ty.rust_ty = rt.rust_ty;
                    return Ok(Some(res));
                };
                // Reverse type inference for resulting rust type in res.
                let type_params = res.ty.params;
                let Some(derived_ret_ty) = self.derive_type(
                    module,
                    block_id,
                    &Expr::MethodCall(method_call_expr.clone()),
                    Some(type_params),
                    generic_vars,
                    ctx,
                )?
                else {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!(
                            "Failed to determine typeck return type for inlined method call `{}`",
                            method_call_expr.to_token_stream()
                        ),
                    );
                };
                res.ty = derived_ret_ty;
                Ok(Some(res))
            } else {
                Ok(None)
            }
        }) // stacker::maybe_grow
    }
}
