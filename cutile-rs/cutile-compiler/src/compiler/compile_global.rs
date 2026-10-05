/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-level `Global` lowering.
//!
//! Rust `static NAME: Global<A, { [] }>` items are ordinary Rust statics in the
//! expanded module, but the JIT treats them as Tile IR globals. The Rust value is
//! just an immutable descriptor; device-side mutability is expressed by the
//! load/store/atomic methods lowered here.

use super::_function::CUDATileFunctionCompiler;
use super::_type;
use super::_value::{CompilerContext, TileRustValue};
use super::shared_utils::{AtomicMode, ElementTypePrefix};
use super::tile_rust_type::TileRustType;
use crate::error::{JITError, SpannedJITError};
use crate::generics::GenericVars;
use crate::passes::name_resolution::{DefKind, Res};
use crate::syn_utils::get_type_ident;
use crate::types::TypeParam;
use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, DenseElements, Global as IrGlobal, Module, ScalarType, TileElementType,
    TileType, Type as TileIrType,
};
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{
    Expr, ExprCall, ExprLit, ExprMethodCall, GenericArgument, ItemStatic, Lit, PathArguments,
    StaticMutability, Type, UnOp,
};

#[derive(Clone)]
struct GlobalInfo {
    symbol: String,
    element_ty: Type,
    element_name: String,
    shape: Vec<i32>,
    constant: bool,
    visibility: cutile_ir::ir::SymbolVisibility,
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub(crate) fn emit_module_globals(&self, module: &mut Module) -> Result<(), JITError> {
        for (module_name, item) in self.modules.name_resolver.all_statics() {
            let Some(info) = self.global_info(module_name, item, &self.generic_vars)? else {
                return self.modules.resolve_span(module_name, &item.span()).jit_error_result(
                    "only `static NAME: Global<A, { [] }>` items are supported inside `#[cutile::module]`; use `const` for compile-time constants",
                );
            };
            if !info.shape.is_empty() {
                return self
                    .modules
                    .resolve_span(module_name, &item.span())
                    .jit_error_result(
                        "shaped `Global` statics are not supported yet; use `Global<A, { [] }>`",
                    );
            }

            let scalar = _type::scalar_from_name(&info.element_name).ok_or_else(|| {
                JITError::Generic(format!(
                    "unsupported Global element type `{}`",
                    info.element_name
                ))
            })?;
            if let Some(caps) = &self.target_capabilities {
                use cutile_ir::requirements::Feature;
                for (enabled, feature) in [
                    (info.constant, Feature::GlobalConstant),
                    (
                        info.visibility != cutile_ir::ir::SymbolVisibility::Public,
                        Feature::GlobalVisibility,
                    ),
                ] {
                    if enabled {
                        caps.require_version(
                            "global constant/visibility",
                            feature.since(),
                            &self.ir_location(&item.span()),
                        )?;
                    }
                }
            }
            let value_ty = TileIrType::Tile(TileType {
                shape: vec![1],
                element_type: TileElementType::Scalar(scalar),
            });
            let init_expr = self.global_static_initializer(item)?;
            let init_value = self.global_scalar_initializer_value(&init_expr, module_name)?;
            let data =
                super::compile_expression::encode_literal_bytes(&init_value, &info.element_name);

            module.globals.push(IrGlobal {
                sym_name: info.symbol,
                value: DenseElements {
                    element_type: value_ty,
                    shape: vec![1],
                    data,
                },
                alignment: scalar_alignment(scalar),
                constant: info.constant,
                symbol_visibility: info.visibility,
            });
        }
        Ok(())
    }

    pub(crate) fn compile_global_method_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let Some(info) = self.resolve_global_receiver(&method_call.receiver, generic_vars, ctx)?
        else {
            return Ok(None);
        };
        if !info.shape.is_empty() {
            return self.jit_error_result(
                &method_call.receiver.span(),
                "shaped `Global` method calls are not supported yet; use `Global<A, { [] }>`",
            );
        }

        if info.constant && method_call.method != "load" {
            return self.jit_error_result(
                &method_call.span(),
                "constant Global storage cannot be modified",
            );
        }
        match method_call.method.to_string().as_str() {
            "load" => self.compile_global_load(module, block_id, method_call, generic_vars, info, return_type),
            "store" => self.compile_global_store(module, block_id, method_call, generic_vars, ctx, info, return_type),
            "atomic_add" => self.compile_global_atomic_add(module, block_id, method_call, generic_vars, ctx, info, return_type),
            method => self.jit_error_result(
                &method_call.method.span(),
                &format!("unsupported `Global` method `{method}`; supported methods are `load`, `store`, and `atomic_add`"),
            ),
        }
    }

    fn compile_global_load(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &ExprMethodCall,
        generic_vars: &GenericVars,
        info: GlobalInfo,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.args.len() != 2 {
            return self.jit_error_result(
                &method_call.span(),
                "`Global::load` expects `(memory_ordering, memory_scope)`",
            );
        }
        let ptr = self.compile_global_ptr(module, block_id, method_call, &info)?;
        let tile_ty = self.global_tile_type(&info)?;
        let tile_ir_ty = _type::convert_type(&tile_ty).ok_or_else(|| {
            self.jit_error(&method_call.span(), "failed to compile Global load type")
        })?;
        let token_ty = self.token_type(generic_vars)?;

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&method_call.args[0], "memory_ordering")?;
        let memory_ordering_value = load_ordering_value(&memory_ordering).ok_or_else(|| {
            self.jit_error(
                &method_call.args[0].span(),
                "invalid Global load ordering; valid: Relaxed, Acquire",
            )
        })?;
        let memory_scope =
            super::shared_utils::extract_zst_type_name(&method_call.args[1], "memory_scope")?;
        let memory_scope_value = memory_scope_value(&memory_scope).ok_or_else(|| {
            self.jit_error(
                &method_call.args[1].span(),
                "invalid Global memory scope; valid: Device, System",
            )
        })?;

        let builder = OpBuilder::new(Opcode::LoadPtrTko, self.ir_location(&method_call.span()))
            .result(tile_ir_ty)
            .result(TileIrType::Token)
            .operand(ptr)
            .attr(
                "memory_ordering_semantics",
                Attribute::i32(memory_ordering_value),
            )
            .attr("memory_scope", Attribute::i32(memory_scope_value))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(0),
                    Attribute::i32(0),
                    Attribute::i32(0),
                ]),
            );
        let (op_id, results) = builder.build(module);
        append_op(module, block_id, op_id);

        let outer_ty = match return_type {
            Some(ty) => ty,
            None => {
                let tile_rust_ty = &tile_ty.rust_ty;
                let tuple_ty: Type = syn::parse_quote!((#tile_rust_ty, Token));
                self.compile_type(&tuple_ty, generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(
                            &method_call.span(),
                            "failed to compile Global load return type",
                        )
                    })?
            }
        };
        Ok(Some(TileRustValue::new_compound(
            vec![
                TileRustValue::new_structured_type(results[0], tile_ty, None),
                TileRustValue::new_primitive(results[1], token_ty, None),
            ],
            outer_ty,
        )))
    }

    fn compile_global_store(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        info: GlobalInfo,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.args.len() != 3 {
            return self.jit_error_result(
                &method_call.span(),
                "`Global::store` expects `(value, memory_ordering, memory_scope)`",
            );
        }
        let ptr = self.compile_global_ptr(module, block_id, method_call, &info)?;
        let tile_ty = self.global_tile_type(&info)?;
        let Some(value) = self.compile_expression(
            module,
            block_id,
            &method_call.args[0],
            generic_vars,
            ctx,
            Some(tile_ty.clone()),
        )?
        else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "failed to compile Global store value",
            );
        };
        let Some(value_ir) = value.value else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "Global store value did not produce an IR value",
            );
        };
        let token_ty = return_type.unwrap_or(self.token_type(generic_vars)?);

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&method_call.args[1], "memory_ordering")?;
        let memory_ordering_value = store_ordering_value(&memory_ordering).ok_or_else(|| {
            self.jit_error(
                &method_call.args[1].span(),
                "invalid Global store ordering; valid: Relaxed, Release",
            )
        })?;
        let memory_scope =
            super::shared_utils::extract_zst_type_name(&method_call.args[2], "memory_scope")?;
        let memory_scope_value = memory_scope_value(&memory_scope).ok_or_else(|| {
            self.jit_error(
                &method_call.args[2].span(),
                "invalid Global memory scope; valid: Device, System",
            )
        })?;

        let builder = OpBuilder::new(Opcode::StorePtrTko, self.ir_location(&method_call.span()))
            .result(TileIrType::Token)
            .operand(ptr)
            .operand(value_ir)
            .attr(
                "memory_ordering_semantics",
                Attribute::i32(memory_ordering_value),
            )
            .attr("memory_scope", Attribute::i32(memory_scope_value))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(1),
                    Attribute::i32(0),
                    Attribute::i32(0),
                ]),
            );
        let (op_id, results) = builder.build(module);
        append_op(module, block_id, op_id);
        Ok(Some(TileRustValue::new_primitive(
            results[0], token_ty, None,
        )))
    }

    fn compile_global_atomic_add(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        info: GlobalInfo,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.args.len() != 3 {
            return self.jit_error_result(
                &method_call.span(),
                "`Global::atomic_add` expects `(value, memory_ordering, memory_scope)`",
            );
        }
        let ptr = self.compile_global_ptr(module, block_id, method_call, &info)?;
        let tile_ty = self.global_tile_type(&info)?;
        let Some(value) = self.compile_expression(
            module,
            block_id,
            &method_call.args[0],
            generic_vars,
            ctx,
            Some(tile_ty.clone()),
        )?
        else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "failed to compile Global atomic value",
            );
        };
        let Some(value_ir) = value.value else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "Global atomic value did not produce an IR value",
            );
        };
        let token_ty = self.token_type(generic_vars)?;
        let tile_ir_ty = _type::convert_type(&tile_ty).ok_or_else(|| {
            self.jit_error(
                &method_call.span(),
                "failed to compile Global atomic result type",
            )
        })?;

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&method_call.args[1], "memory_ordering")?;
        let memory_ordering_value = atomic_ordering_value(&memory_ordering).ok_or_else(|| {
            self.jit_error(
                &method_call.args[1].span(),
                "invalid Global atomic ordering; valid: Relaxed, Acquire, Release, AcqRel",
            )
        })?;
        let memory_scope =
            super::shared_utils::extract_zst_type_name(&method_call.args[2], "memory_scope")?;
        let memory_scope_value = memory_scope_value(&memory_scope).ok_or_else(|| {
            self.jit_error(
                &method_call.args[2].span(),
                "invalid Global memory scope; valid: Device, System",
            )
        })?;

        let elem_prefix = ElementTypePrefix::new(&info.element_name)?;
        let mode = if elem_prefix == ElementTypePrefix::Float {
            AtomicMode::new("AddF", elem_prefix)?
        } else {
            AtomicMode::new("Add", elem_prefix)?
        } as i64;

        let (op_id, results) =
            OpBuilder::new(Opcode::AtomicRMW, self.ir_location(&method_call.span()))
                .result(tile_ir_ty)
                .result(TileIrType::Token)
                .operand(ptr)
                .operand(value_ir)
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                )
                .attr("memory_scope", Attribute::i32(memory_scope_value))
                .attr("mode", Attribute::i32(mode))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(0),
                    ]),
                )
                .build(module);
        append_op(module, block_id, op_id);

        let outer_ty = match return_type {
            Some(ty) => ty,
            None => {
                let tile_rust_ty = &tile_ty.rust_ty;
                let tuple_ty: Type = syn::parse_quote!((#tile_rust_ty, Token));
                self.compile_type(&tuple_ty, generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(
                            &method_call.span(),
                            "failed to compile Global atomic return type",
                        )
                    })?
            }
        };
        Ok(Some(TileRustValue::new_compound(
            vec![
                TileRustValue::new_structured_type(results[0], tile_ty, None),
                TileRustValue::new_primitive(results[1], token_ty, None),
            ],
            outer_ty,
        )))
    }

    fn compile_global_ptr(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &ExprMethodCall,
        info: &GlobalInfo,
    ) -> Result<cutile_ir::ir::Value, JITError> {
        let ptr_ty = TileRustType::from_scalar_ptr(&info.element_name, true).ok_or_else(|| {
            self.jit_error(
                &method_call.receiver.span(),
                &format!(
                    "failed to compile Global pointer type for `{}`",
                    info.element_name
                ),
            )
        })?;
        let ptr_ir_ty = _type::convert_type(&ptr_ty).ok_or_else(|| {
            self.jit_error(
                &method_call.receiver.span(),
                "failed to convert Global pointer type",
            )
        })?;
        let (op_id, results) = OpBuilder::new(
            Opcode::GetGlobal,
            self.ir_location(&method_call.receiver.span()),
        )
        .result(ptr_ir_ty)
        .attr("name", Attribute::String(info.symbol.clone()))
        .build(module);
        append_op(module, block_id, op_id);
        Ok(results[0])
    }

    fn resolve_global_receiver(
        &self,
        receiver: &Expr,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<Option<GlobalInfo>, JITError> {
        let Expr::Path(path) = receiver else {
            return Ok(None);
        };
        if path.path.segments.len() != 1 {
            return Ok(None);
        }
        let name = path.path.segments[0].ident.to_string();
        if ctx.vars.contains_key(&name) {
            return Ok(None);
        }
        let res = self
            .modules
            .name_resolver
            .resolve_path(&path.path, &self.module_name);
        let Res::Def(DefKind::Static, def_id) = res else {
            return Ok(None);
        };
        let Some(static_item) = self.modules.name_resolver.get_static(&def_id) else {
            return Ok(None);
        };
        self.global_info(&def_id.module, static_item, generic_vars)
    }

    fn global_info(
        &self,
        module_name: &str,
        item: &ItemStatic,
        generic_vars: &GenericVars,
    ) -> Result<Option<GlobalInfo>, JITError> {
        if !matches!(item.mutability, StaticMutability::None) {
            return self.modules.resolve_span(module_name, &item.span()).jit_error_result(
                "`Global` declarations should use immutable `static`; mutability is provided by `Global` methods",
            );
        }
        let normalized_ty = self
            .modules
            .normalize_type_aliases_in(&item.ty, Some(module_name))?;
        let Some(type_name) = get_type_ident(&normalized_ty) else {
            return Ok(None);
        };
        if type_name != "Global" {
            return Ok(None);
        }
        let atomic_ty = global_element_type(&normalized_ty).ok_or_else(|| {
            self.modules
                .resolve_span(module_name, &item.ty.span())
                .jit_error("`Global` requires an atomic type: `Global<AtomicI32, { [] }>`")
        })?;
        let element_ty = self.global_atomic_value_type(&atomic_ty, module_name)?;
        let element_compiled = self
            .compile_type(
                &element_ty,
                generic_vars,
                &HashMap::<String, TypeParam>::new(),
            )?
            .ok_or_else(|| {
                self.modules
                    .resolve_span(module_name, &element_ty.span())
                    .jit_error("failed to compile `Global` element type")
            })?;
        let element_name = element_compiled
            .get_cuda_tile_element_type(self.modules.primitives())?
            .ok_or_else(|| {
                self.modules
                    .resolve_span(module_name, &element_ty.span())
                    .jit_error("failed to determine `Global` element type")
            })?;
        let shape = global_shape(&normalized_ty).ok_or_else(|| {
            self.modules
                .resolve_span(module_name, &item.ty.span())
                .jit_error("`Global` requires a static shape: `Global<A, { [] }>`")
        })?;
        let options = crate::syn_utils::get_meta_list("cuda_tile :: global", &item.attrs);
        let constant = options
            .as_ref()
            .and_then(|o| o.parse_bool("constant"))
            .unwrap_or(false);
        let visibility = match options
            .as_ref()
            .and_then(|o| o.parse_string("visibility"))
            .as_deref()
        {
            None | Some("public") => cutile_ir::ir::SymbolVisibility::Public,
            Some("private") => cutile_ir::ir::SymbolVisibility::Private,
            Some(_) => {
                return self
                    .modules
                    .resolve_span(module_name, &item.span())
                    .jit_error_result("global visibility must be public or private")
            }
        };
        Ok(Some(GlobalInfo {
            symbol: global_symbol_name(module_name, &item.ident.to_string()),
            element_ty,
            element_name,
            shape,
            constant,
            visibility,
        }))
    }

    /// Resolve the sealed marker once, shared by inference and IR emission.
    pub(crate) fn global_atomic_value_type(
        &self,
        ty: &Type,
        module_name: &str,
    ) -> Result<Type, JITError> {
        let ty = self
            .modules
            .normalize_type_aliases_in(ty, Some(module_name))?;
        let payload = match &ty {
            Type::Path(path) => match self.modules.name_resolver.resolve_path(&path.path, module_name) {
                Res::Def(DefKind::Struct, id)
                    if Some(id.module.as_str()) == self.modules.name_resolver.core_module() =>
                {
                    atomic_payload_type(&id.name)
                }
                _ => None,
            },
            _ => None,
        }.ok_or_else(|| self.modules.resolve_span(module_name, &ty.span()).jit_error(
            "`Global` requires a supported sealed Atomic type (AtomicI32, AtomicU32, AtomicI64, AtomicU64, AtomicF32, or AtomicF64)",
        ))?;
        Ok(syn::parse_str(payload).expect("atomic payload is a scalar type"))
    }

    fn global_static_initializer(&self, item: &ItemStatic) -> Result<Expr, JITError> {
        let Expr::Call(call) = &*item.expr else {
            return self.jit_error_result(
                &item.expr.span(),
                "`Global` static initializer must be `Global::new(value)`",
            );
        };
        if !call_path_ends_with_new(call) || call.args.len() != 1 {
            return self.jit_error_result(
                &call.span(),
                "`Global` static initializer must be `Global::new(value)`",
            );
        }
        Ok(call.args[0].clone())
    }

    fn global_scalar_initializer_value(
        &self,
        expr: &Expr,
        module_name: &str,
    ) -> Result<String, JITError> {
        if let Some(value) = literal_scalar_value(expr) {
            return Ok(value);
        }
        if let Expr::Path(path) = expr {
            let res = self
                .modules
                .name_resolver
                .resolve_path(&path.path, module_name);
            if let Res::Def(DefKind::Const, def_id) = res {
                if let Some(const_item) = self.modules.name_resolver.get_const(&def_id) {
                    return self.global_scalar_initializer_value(&const_item.expr, &def_id.module);
                }
            }
        }
        self.modules.resolve_span(module_name, &expr.span()).jit_error_result(
            "`Global::new` currently requires a scalar literal or module-level scalar const initializer",
        )
    }

    fn global_tile_type(&self, info: &GlobalInfo) -> Result<TileRustType, JITError> {
        TileRustType::from_tile(&info.element_name, &info.shape).ok_or_else(|| {
            JITError::Generic(format!(
                "failed to compile Global tile type for `{}`",
                info.element_ty.to_token_stream()
            ))
        })
    }

    pub(crate) fn token_type(&self, generic_vars: &GenericVars) -> Result<TileRustType, JITError> {
        let token_ty: Type = syn::parse_quote!(Token);
        self.compile_type(&token_ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(&token_ty.span(), "failed to compile Token type"))
    }
}

fn global_element_type(ty: &Type) -> Option<Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    })
}

fn global_shape(ty: &Type) -> Option<Vec<i32>> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        GenericArgument::Const(expr) => const_shape_expr(expr),
        _ => None,
    })
}

fn const_shape_expr(expr: &Expr) -> Option<Vec<i32>> {
    match expr {
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [syn::Stmt::Expr(expr, _)] => const_shape_expr(expr),
            _ => None,
        },
        Expr::Array(array) => array.elems.iter().map(expr_i32).collect(),
        Expr::Repeat(repeat) => {
            let value = expr_i32(&repeat.expr)?;
            let len = expr_i32(&repeat.len)? as usize;
            Some(vec![value; len])
        }
        Expr::Paren(paren) => const_shape_expr(&paren.expr),
        _ => None,
    }
}

fn expr_i32(expr: &Expr) -> Option<i32> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) => value.base10_parse::<i32>().ok(),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            expr_i32(&unary.expr).map(|value| -value)
        }
        Expr::Paren(paren) => expr_i32(&paren.expr),
        _ => None,
    }
}

fn call_path_ends_with_new(call: &ExprCall) -> bool {
    let Expr::Path(path) = &*call.func else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "new")
}

fn literal_scalar_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Bool(value) => Some(if value.value() { "1" } else { "0" }.to_string()),
            Lit::Int(value) => Some(value.base10_digits().to_string()),
            Lit::Float(value) => Some(value.base10_digits().to_string()),
            _ => None,
        },
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => match &*unary.expr {
            Expr::Lit(ExprLit { lit, .. }) => match lit {
                Lit::Int(value) => Some(format!("-{}", value.base10_digits())),
                Lit::Float(value) => Some(format!("-{}", value.base10_digits())),
                _ => None,
            },
            _ => None,
        },
        Expr::Paren(paren) => literal_scalar_value(&paren.expr),
        _ => None,
    }
}

fn global_symbol_name(module_name: &str, name: &str) -> String {
    let mut symbol = String::new();
    for ch in module_name.chars().chain(['_']).chain(name.chars()) {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            symbol.push(ch);
        } else {
            symbol.push('_');
        }
    }
    symbol
}

fn scalar_alignment(scalar: ScalarType) -> u64 {
    match scalar {
        ScalarType::I1
        | ScalarType::I4
        | ScalarType::I8
        | ScalarType::F4E2M1FN
        | ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E5M3FNU
        | ScalarType::F8E8M0FNU => 1,
        ScalarType::I16 | ScalarType::F16 | ScalarType::BF16 => 2,
        ScalarType::I32 | ScalarType::F32 | ScalarType::TF32 => 4,
        ScalarType::I64 | ScalarType::F64 => 8,
    }
}

fn memory_scope_value(scope: &str) -> Option<i64> {
    match scope {
        "Device" => Some(1),
        "System" => Some(2),
        _ => None,
    }
}

fn load_ordering_value(ordering: &str) -> Option<i64> {
    match ordering {
        "Relaxed" => Some(1),
        "Acquire" => Some(2),
        _ => None,
    }
}

fn store_ordering_value(ordering: &str) -> Option<i64> {
    match ordering {
        "Relaxed" => Some(1),
        "Release" => Some(3),
        _ => None,
    }
}

fn atomic_ordering_value(ordering: &str) -> Option<i64> {
    match ordering {
        "Relaxed" => Some(1),
        "Acquire" => Some(2),
        "Release" => Some(3),
        "AcqRel" => Some(4),
        _ => None,
    }
}

// Must agree with the sealed Atomic implementations in cutile::core.
fn atomic_payload_type(name: &str) -> Option<&'static str> {
    match name {
        "AtomicI32" => Some("i32"),
        "AtomicU32" => Some("u32"),
        "AtomicI64" => Some("i64"),
        "AtomicU64" => Some("u64"),
        "AtomicF32" => Some("f32"),
        "AtomicF64" => Some("f64"),
        _ => None,
    }
}
