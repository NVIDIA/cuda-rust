/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler1-compatible type inference and DSL dispatch selection.
//!
//! This is intentionally narrower than a Rust type checker. It builds the
//! side-table shape needed by the compiler3 pass plan: expression types where
//! the DSL can infer them, selected method impls, and dispatch-wrapper calls
//! that should be erased before emission.

use crate::compiler::_function::CUDATileFunctionCompiler;
use crate::compiler::shared_utils;
use crate::compiler::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::{
    get_cga_from_generic_argument, GenericArgInference, GenericArgType, GenericVars, TypeInstance,
    TypeInstanceUserType,
};
use crate::passes::name_resolution::{DefKind, Res};
use crate::passes::node_ids::{self, NodeId};
use crate::syn_utils::*;
use crate::types::{get_lit_type, TypeParam};
use proc_macro2::TokenTree;
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use syn::{
    Expr, ExprCall, ExprLit, ExprMacro, ExprMethodCall, GenericArgument, Generics, ImplItemFn,
    ItemFn, ItemImpl, Lit, Pat, PathArguments, Stmt, TraitItem, Type, TypeParamBound,
    WherePredicate,
};

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedScalarType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F32,
    F64,
}

impl ResolvedScalarType {
    fn from_ident(ident: &str) -> Option<Self> {
        match ident {
            "bool" => Some(Self::Bool),
            "i8" => Some(Self::I8),
            "i16" => Some(Self::I16),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "i128" => Some(Self::I128),
            "isize" => Some(Self::Isize),
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            "u64" => Some(Self::U64),
            "u128" => Some(Self::U128),
            "usize" => Some(Self::Usize),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }

    fn to_syn_type(&self) -> Type {
        match self {
            Self::Bool => syn::parse_quote!(bool),
            Self::I8 => syn::parse_quote!(i8),
            Self::I16 => syn::parse_quote!(i16),
            Self::I32 => syn::parse_quote!(i32),
            Self::I64 => syn::parse_quote!(i64),
            Self::I128 => syn::parse_quote!(i128),
            Self::Isize => syn::parse_quote!(isize),
            Self::U8 => syn::parse_quote!(u8),
            Self::U16 => syn::parse_quote!(u16),
            Self::U32 => syn::parse_quote!(u32),
            Self::U64 => syn::parse_quote!(u64),
            Self::U128 => syn::parse_quote!(u128),
            Self::Usize => syn::parse_quote!(usize),
            Self::F32 => syn::parse_quote!(f32),
            Self::F64 => syn::parse_quote!(f64),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedType {
    Scalar(ResolvedScalarType),
    Adt {
        path: syn::Path,
    },
    Reference {
        mutable: bool,
        elem: Box<ResolvedType>,
    },
    Pointer {
        mutable: bool,
        elem: Box<ResolvedType>,
    },
    Tuple(Vec<ResolvedType>),
    Array {
        elem: Box<ResolvedType>,
        len: Expr,
    },
    Slice(Box<ResolvedType>),
    Unit,
    Never,
    Surface(Type),
}

impl ResolvedType {
    pub fn from_syn_type(ty: &Type) -> Self {
        match ty {
            Type::Path(path_ty) if path_ty.qself.is_none() => {
                if path_ty.path.segments.len() == 1 {
                    let segment = &path_ty.path.segments[0];
                    if matches!(segment.arguments, PathArguments::None) {
                        if let Some(scalar) =
                            ResolvedScalarType::from_ident(&segment.ident.to_string())
                        {
                            return Self::Scalar(scalar);
                        }
                    }
                }
                Self::Adt {
                    path: path_ty.path.clone(),
                }
            }
            Type::Reference(reference) => Self::Reference {
                mutable: reference.mutability.is_some(),
                elem: Box::new(Self::from_syn_type(&reference.elem)),
            },
            Type::Ptr(ptr) => Self::Pointer {
                mutable: ptr.mutability.is_some(),
                elem: Box::new(Self::from_syn_type(&ptr.elem)),
            },
            Type::Tuple(tuple) if tuple.elems.is_empty() => Self::Unit,
            Type::Tuple(tuple) => {
                Self::Tuple(tuple.elems.iter().map(Self::from_syn_type).collect())
            }
            Type::Array(array) => Self::Array {
                elem: Box::new(Self::from_syn_type(&array.elem)),
                len: array.len.clone(),
            },
            Type::Slice(slice) => Self::Slice(Box::new(Self::from_syn_type(&slice.elem))),
            Type::Never(_) => Self::Never,
            _ => Self::Surface(ty.clone()),
        }
    }

    pub fn to_syn_type(&self) -> Type {
        match self {
            Self::Scalar(scalar) => scalar.to_syn_type(),
            Self::Adt { path } => Type::Path(syn::TypePath {
                qself: None,
                path: path.clone(),
            }),
            Self::Reference { mutable, elem } => {
                let elem = elem.to_syn_type();
                if *mutable {
                    syn::parse_quote!(&mut #elem)
                } else {
                    syn::parse_quote!(&#elem)
                }
            }
            Self::Pointer { mutable, elem } => {
                let elem = elem.to_syn_type();
                if *mutable {
                    syn::parse_quote!(*mut #elem)
                } else {
                    syn::parse_quote!(*const #elem)
                }
            }
            Self::Tuple(elems) => {
                let elems = elems.iter().map(Self::to_syn_type).collect();
                Type::Tuple(syn::TypeTuple {
                    paren_token: syn::token::Paren::default(),
                    elems,
                })
            }
            Self::Array { elem, len } => {
                let elem = elem.to_syn_type();
                syn::parse_quote!([#elem; #len])
            }
            Self::Slice(elem) => {
                let elem = elem.to_syn_type();
                syn::parse_quote!([#elem])
            }
            Self::Unit => syn::parse_quote!(()),
            Self::Never => syn::parse_quote!(!),
            Self::Surface(ty) => ty.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MethodSelection {
    pub module_name: String,
    pub impl_item: ItemImpl,
    pub impl_method: ImplItemFn,
    pub generic_vars: GenericVars,
    pub return_type: Option<TileRustType>,
}

#[derive(Clone)]
struct ClosureSignature {
    inputs: Vec<Type>,
    output: Option<Type>,
}

#[derive(Clone, Default)]
pub struct TypeckResults {
    resolved_expr_types: HashMap<NodeId, ResolvedType>,
    expr_types: HashMap<NodeId, TileRustType>,
    method_selections: HashMap<NodeId, MethodSelection>,
    lowered_method_calls: HashMap<NodeId, ExprMethodCall>,
}

impl TypeckResults {
    pub fn insert_expr_type(&mut self, expr: &Expr, ty: TileRustType) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        self.insert_expr_type_by_id(id, ty);
    }

    fn insert_expr_type_by_id(&mut self, id: NodeId, ty: TileRustType) {
        self.resolved_expr_types
            .insert(id, ResolvedType::from_syn_type(&ty.rust_ty));
        self.expr_types.insert(id, ty);
    }

    pub fn insert_resolved_expr_type(&mut self, expr: &Expr, ty: ResolvedType) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        self.resolved_expr_types.insert(id, ty);
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TileRustType> {
        self.expr_types.get(&node_ids::expr_id(expr)?)
    }

    pub fn resolved_expr_type(&self, expr: &Expr) -> Option<&ResolvedType> {
        self.resolved_expr_types.get(&node_ids::expr_id(expr)?)
    }

    pub fn required_resolved_expr_type(&self, expr: &Expr) -> Result<&ResolvedType, JITError> {
        self.resolved_expr_type(expr)
            .ok_or_else(|| JITError::generic_err("missing typeck entry for expression"))
    }

    pub fn syn_expr_type(&self, expr: &Expr) -> Option<Type> {
        self.resolved_expr_type(expr).map(ResolvedType::to_syn_type)
    }

    pub fn required_syn_expr_type(&self, expr: &Expr) -> Result<Type, JITError> {
        self.syn_expr_type(expr)
            .ok_or_else(|| JITError::generic_err("missing typeck entry for expression"))
    }

    pub fn required_tile_expr_type(
        &self,
        compiler: &CUDATileFunctionCompiler<'_>,
        expr: &Expr,
        generic_vars: &GenericVars,
        type_params: &HashMap<String, TypeParam>,
    ) -> Result<TileRustType, JITError> {
        if let Some(ty) = self.expr_type(expr) {
            return Ok(ty.clone());
        }
        let syn_ty = self.required_syn_expr_type(expr)?;
        compiler
            .compile_type(&syn_ty, generic_vars, type_params)?
            .ok_or_else(|| JITError::generic_err("failed to compile typeck expression type"))
    }

    pub fn debug_dump(&self) -> String {
        let mut lines = Vec::new();
        for (id, ty) in &self.resolved_expr_types {
            let syn_ty = ty.to_syn_type();
            lines.push((
                id.0,
                0u8,
                format!("expr#{}: {}", id.0, syn_ty.to_token_stream()),
            ));
        }
        for (id, selection) in &self.method_selections {
            let return_ty = selection
                .return_type
                .as_ref()
                .map(|ty| ty.rust_ty.to_token_stream().to_string())
                .unwrap_or_else(|| "_".to_string());
            lines.push((
                id.0,
                1u8,
                format!(
                    "method#{}: {}::{} -> {}",
                    id.0, selection.module_name, selection.impl_method.sig.ident, return_ty
                ),
            ));
        }
        lines.sort_by(|lhs, rhs| (lhs.0, lhs.1, &lhs.2).cmp(&(rhs.0, rhs.1, &rhs.2)));
        lines
            .into_iter()
            .map(|(_, _, line)| line)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn insert_method_selection(
        &mut self,
        method_call: &ExprMethodCall,
        selection: MethodSelection,
    ) {
        self.insert_method_selection_for_expr(&Expr::MethodCall(method_call.clone()), selection);
    }

    pub fn insert_method_selection_for_expr(&mut self, expr: &Expr, selection: MethodSelection) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        self.method_selections.insert(id, selection);
    }

    pub fn method_selection(&self, method_call: &ExprMethodCall) -> Option<&MethodSelection> {
        self.method_selections
            .get(&node_ids::expr_id(&Expr::MethodCall(method_call.clone()))?)
    }

    pub fn insert_lowered_method_call(&mut self, call: &ExprCall, method_call: ExprMethodCall) {
        let Some(id) = node_ids::expr_id(&Expr::Call(call.clone())) else {
            return;
        };
        self.lowered_method_calls.insert(id, method_call);
    }

    pub fn lowered_method_call(&self, call: &ExprCall) -> Option<&ExprMethodCall> {
        self.lowered_method_calls
            .get(&node_ids::expr_id(&Expr::Call(call.clone()))?)
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct InferVarId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum InferVarKind {
    General,
    Int,
    Float,
}

#[derive(Clone, Debug)]
enum InferredTy {
    Known(TileRustType),
    Var(InferVarId),
    Unknown,
}

#[derive(Clone, Debug)]
enum BranchTerm {
    Value(InferredTy),
    Diverges,
    Unknown,
}

impl BranchTerm {
    fn from_term(term: InferredTy) -> Self {
        match term {
            InferredTy::Unknown => Self::Unknown,
            term => Self::Value(term),
        }
    }
}

#[derive(Clone, Debug)]
struct InferVar {
    kind: InferVarKind,
    parent: Option<InferVarId>,
    value: Option<TileRustType>,
    expr_ids: Vec<NodeId>,
    origins: Vec<Expr>,
    origin_propagated: bool,
}

#[derive(Clone, Debug, Default)]
struct InferenceState {
    vars: Vec<InferVar>,
    expr_terms: HashMap<NodeId, InferredTy>,
}

pub fn infer_function(
    compiler: &CUDATileFunctionCompiler<'_>,
    fn_item: &ItemFn,
    generic_vars: &GenericVars,
    initial_types: HashMap<String, TileRustType>,
) -> Result<TypeckResults, JITError> {
    let (_, return_type) = get_sig_types(&fn_item.sig, None);
    let expected_return = compiler
        .compile_type(&return_type, generic_vars, &HashMap::new())
        .ok()
        .flatten();
    let mut cx = TypeInferenceCx {
        compiler,
        generic_vars,
        generic_type_bounds: generic_type_bounds(&fn_item.sig.generics),
        syn_vars: initial_types
            .iter()
            .map(|(name, ty)| (name.clone(), ty.rust_ty.clone()))
            .collect(),
        mutable_vars: HashSet::new(),
        vars: HashMap::new(),
        local_terms: HashMap::new(),
        inference: InferenceState::default(),
        results: TypeckResults::default(),
    };
    for (name, ty) in initial_types {
        cx.bind_tile_var(name, ty);
    }
    cx.infer_block(&fn_item.block, expected_return)?;
    cx.finish_inference()?;
    Ok(cx.results)
}

pub fn infer_method(
    compiler: &CUDATileFunctionCompiler<'_>,
    impl_item: &ItemImpl,
    impl_method: &ImplItemFn,
    self_ty: &Type,
    generic_vars: &GenericVars,
    initial_types: HashMap<String, TileRustType>,
) -> Result<TypeckResults, JITError> {
    let (_, return_type) = get_sig_types(&impl_method.sig, Some(self_ty));
    let expected_return = compiler
        .compile_type(&return_type, generic_vars, &HashMap::new())
        .ok()
        .flatten();
    let mut bounds = generic_type_bounds(&impl_item.generics);
    merge_generic_type_bounds(&mut bounds, generic_type_bounds(&impl_method.sig.generics));
    let mut cx = TypeInferenceCx {
        compiler,
        generic_vars,
        generic_type_bounds: bounds,
        syn_vars: initial_types
            .iter()
            .map(|(name, ty)| (name.clone(), ty.rust_ty.clone()))
            .collect(),
        mutable_vars: HashSet::new(),
        vars: HashMap::new(),
        local_terms: HashMap::new(),
        inference: InferenceState::default(),
        results: TypeckResults::default(),
    };
    for (name, ty) in initial_types {
        cx.bind_tile_var(name, ty);
    }
    cx.infer_block(&impl_method.block, expected_return)?;
    cx.finish_inference()?;
    Ok(cx.results)
}

struct TypeInferenceCx<'a, 'm> {
    compiler: &'a CUDATileFunctionCompiler<'m>,
    generic_vars: &'a GenericVars,
    generic_type_bounds: HashMap<String, Vec<String>>,
    syn_vars: HashMap<String, Type>,
    mutable_vars: HashSet<String>,
    vars: HashMap<String, TileRustType>,
    local_terms: HashMap<String, InferredTy>,
    inference: InferenceState,
    results: TypeckResults,
}

enum ArrayLikeExpr {
    Array(syn::ExprArray),
    Repeat { len: usize },
}

impl ArrayLikeExpr {
    fn len(&self) -> usize {
        match self {
            Self::Array(array) => array.elems.len(),
            Self::Repeat { len } => *len,
        }
    }
}

impl<'a, 'm> TypeInferenceCx<'a, 'm> {
    fn infer_block(
        &mut self,
        block: &syn::Block,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        stacker::maybe_grow(
            shared_utils::STACK_RED_ZONE,
            shared_utils::STACK_GROW_SIZE,
            || self.infer_block_inner(block, expected),
        )
    }

    fn infer_block_inner(
        &mut self,
        block: &syn::Block,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let mut last_type = None;
        for (idx, stmt) in block.stmts.iter().enumerate() {
            let stmt_expected = if idx + 1 == block.stmts.len() || stmt_is_return(stmt) {
                expected.clone()
            } else {
                None
            };
            last_type = self.infer_stmt(stmt, stmt_expected)?;
        }
        Ok(last_type)
    }

    fn infer_stmt(
        &mut self,
        stmt: &Stmt,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        stacker::maybe_grow(
            shared_utils::STACK_RED_ZONE,
            shared_utils::STACK_GROW_SIZE,
            || self.infer_stmt_inner(stmt, expected),
        )
    }

    fn infer_stmt_inner(
        &mut self,
        stmt: &Stmt,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        match stmt {
            Stmt::Local(local) => {
                let annotated_syn_type = local_pattern_type(&local.pat).cloned();
                let annotated_type = match &annotated_syn_type {
                    Some(ty) => {
                        self.compiler
                            .compile_type(ty, self.generic_vars, &HashMap::new())?
                    }
                    _ => None,
                };

                let init_type = if let Some(init) = &local.init {
                    if annotated_type.is_some() {
                        self.infer_expr(&init.expr, annotated_type.clone())?
                    } else {
                        match self.infer_expr_term(&init.expr, None)? {
                            InferredTy::Known(ty) => Some(ty),
                            term @ InferredTy::Var(_) => {
                                if let Some(name) = local_binding_name(&local.pat) {
                                    self.bind_inferred_var(name, term);
                                }
                                None
                            }
                            InferredTy::Unknown => {
                                if let Some(name) = local_binding_name(&local.pat) {
                                    let term = self.new_infer_var(
                                        InferVarKind::General,
                                        Some(&init.expr),
                                        Some(*init.expr.clone()),
                                    );
                                    self.bind_inferred_var(name, term);
                                }
                                None
                            }
                        }
                    }
                } else {
                    None
                };

                let binding_type = annotated_type.or(init_type);
                if let Some(binding_type) = binding_type.clone() {
                    self.bind_pattern_with_tile_type(&local.pat, binding_type)?;
                } else if let Some(syn_type) = annotated_syn_type.or_else(|| {
                    local
                        .init
                        .as_ref()
                        .and_then(|init| self.results.syn_expr_type(&init.expr))
                }) {
                    self.bind_pattern_with_syn_type(&local.pat, &syn_type)?;
                } else if let Some(init) = &local.init {
                    self.bind_pattern_from_expr(&local.pat, &init.expr)?;
                }
                Ok(binding_type)
            }
            Stmt::Expr(expr, semicolon) => {
                let expr_expected = if semicolon.is_none() || matches!(expr, Expr::Return(_)) {
                    expected
                } else {
                    None
                };
                self.infer_expr(expr, expr_expected)
            }
            Stmt::Item(syn::Item::Const(item_const)) => {
                let expected = self.compiler.compile_type(
                    &item_const.ty,
                    self.generic_vars,
                    &HashMap::new(),
                )?;
                let init_ty = self.infer_expr(&item_const.expr, expected.clone())?;
                if let Some(ty) = expected.or(init_ty) {
                    self.bind_tile_var(item_const.ident.to_string(), ty);
                } else {
                    self.bind_syn_var(item_const.ident.to_string(), item_const.ty.as_ref().clone());
                }
                Ok(None)
            }
            Stmt::Item(_) | Stmt::Macro(_) => Ok(None),
        }
    }

    fn infer_expr(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        stacker::maybe_grow(
            shared_utils::STACK_RED_ZONE,
            shared_utils::STACK_GROW_SIZE,
            || self.infer_expr_inner(expr, expected),
        )
    }

    fn infer_expr_inner(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let inferred = match expr {
            Expr::Path(path) => {
                if path.path.segments.len() == 1 {
                    let name = get_ident_from_path_expr(path).to_string();
                    if name == "None" {
                        expected.filter(|ty| option_payload_syn_type(&ty.rust_ty).is_some())
                    } else if let Some(term) = self.local_terms.get(&name).cloned() {
                        let term = if let Some(expected) = expected.clone() {
                            self.unify_with_known(term, expected)?
                        } else {
                            term
                        };
                        self.record_expr_term(expr, term.clone());
                        self.term_known_type(&term)
                    } else if let Some(ty) = self.infer_global_const_type(path)? {
                        Some(ty)
                    } else if let Some(ty) = self.infer_global_static_type(path)? {
                        Some(ty)
                    } else {
                        self.vars.get(&name).cloned()
                    }
                } else if let Some(ty) = self.infer_associated_const_type(path)? {
                    Some(ty)
                } else {
                    self.infer_zst_marker_type(expr)
                }
            }
            Expr::Reference(reference) => {
                let inner = self.infer_expr(&reference.expr, None)?;
                inner.map(|mut ty| {
                    ty.rust_ty = Type::Reference(syn::TypeReference {
                        and_token: reference.and_token,
                        lifetime: None,
                        mutability: reference.mutability,
                        elem: Box::new(ty.rust_ty.clone()),
                    });
                    ty
                })
            }
            Expr::Paren(paren) => self.infer_expr(&paren.expr, expected.clone())?,
            Expr::Block(block) => {
                let block_ty = self.infer_block(&block.block, expected.clone())?;
                if block_ty.is_none() {
                    if let Some(syn_ty) = self.block_tail_syn_type(&block.block) {
                        self.results
                            .insert_resolved_expr_type(expr, ResolvedType::from_syn_type(&syn_ty));
                    }
                }
                block_ty
            }
            Expr::Tuple(tuple) => self.infer_tuple(tuple, expected.clone())?,
            Expr::Array(array) => {
                let elem_expected = match expected.as_ref() {
                    Some(ty) => self.array_element_type(ty)?,
                    None => None,
                };
                let mut elem_types = Vec::new();
                if let Some(elem_expected) = elem_expected {
                    for elem in &array.elems {
                        if let Some(elem_ty) = self.infer_expr(elem, Some(elem_expected.clone()))? {
                            elem_types.push(elem_ty.rust_ty);
                        }
                    }
                } else {
                    for elem in array
                        .elems
                        .iter()
                        .filter(|elem| !is_unsuffixed_numeric_literal(elem))
                    {
                        if let Some(elem_ty) = self.infer_expr(elem, None)? {
                            elem_types.push(elem_ty.rust_ty);
                        }
                    }
                    let inferred_elem_expected = elem_types.first().and_then(|ty| {
                        self.compiler
                            .compile_type(ty, self.generic_vars, &HashMap::new())
                            .ok()
                            .flatten()
                    });
                    for elem in array
                        .elems
                        .iter()
                        .filter(|elem| is_unsuffixed_numeric_literal(elem))
                    {
                        if let Some(elem_ty) =
                            self.infer_expr(elem, inferred_elem_expected.clone())?
                        {
                            elem_types.push(elem_ty.rust_ty);
                        }
                    }
                }
                if expected.is_some() {
                    expected
                } else if let Some(first_ty) = elem_types.first() {
                    if elem_types.iter().all(|ty| ty == first_ty) {
                        let len = array.elems.len();
                        let array_ty: Type = syn::parse_quote!([#first_ty; #len]);
                        self.compiler
                            .compile_type(&array_ty, self.generic_vars, &HashMap::new())?
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Expr::Repeat(repeat) => {
                let elem_expected = match expected.as_ref() {
                    Some(ty) => self.array_element_type(ty)?,
                    None => None,
                };
                let elem_ty = self.infer_expr(&repeat.expr, elem_expected)?;
                if expected.is_some() {
                    expected
                } else if let (Some(elem_ty), Some(len)) =
                    (elem_ty, repeat_length_value(&repeat.len, self.generic_vars))
                {
                    let elem_ty = elem_ty.rust_ty;
                    let array_ty: Type = syn::parse_quote!([#elem_ty; #len]);
                    self.compiler
                        .compile_type(&array_ty, self.generic_vars, &HashMap::new())?
                } else {
                    None
                }
            }
            Expr::Range(range) => self.infer_range(range)?,
            Expr::Field(field) => self.infer_field(field, expected.clone())?,
            Expr::Index(index) => self.infer_index(index, expected.clone())?,
            Expr::Call(call) => self.infer_call(call, expected.clone())?,
            Expr::MethodCall(method_call) => {
                self.infer_method_call(method_call, expected.clone())?
            }
            Expr::Struct(struct_expr) => self.infer_struct(struct_expr, expected.clone())?,
            Expr::Macro(macro_expr) => self.infer_macro(macro_expr)?,
            Expr::Cast(cast) => self.infer_cast_target(expr, &cast.expr, &cast.ty)?,
            Expr::Closure(closure) => {
                self.infer_closure(closure, &[], None)?;
                expected
            }
            Expr::Assign(assign) => {
                let lhs_name = match &*assign.left {
                    Expr::Path(path) if path.path.segments.len() == 1 => {
                        Some(get_ident_from_path_expr(path).to_string())
                    }
                    _ => None,
                };
                if let Some(name) = lhs_name {
                    self.infer_assignment(&name, &assign.right)?;
                }
                None
            }
            Expr::Binary(binary) => {
                if binary_result_is_bool(&binary.op) {
                    self.infer_bool_binary(binary)?
                } else if expected.is_none() && binary_result_matches_operands(&binary.op) {
                    match self.infer_expr_term(expr, None)? {
                        InferredTy::Known(ty) => Some(ty),
                        term @ InferredTy::Var(_) => {
                            self.record_expr_term(expr, term);
                            None
                        }
                        InferredTy::Unknown => None,
                    }
                } else if is_unsuffixed_numeric_literal(&binary.left) {
                    let rhs = self.infer_expr(&binary.right, expected.clone())?;
                    let lhs = self.infer_expr(&binary.left, rhs.clone().or(expected.clone()))?;
                    lhs.or(rhs).or(expected)
                } else {
                    let lhs = self.infer_expr(&binary.left, expected.clone())?;
                    let rhs = self.infer_expr(&binary.right, lhs.clone().or(expected.clone()))?;
                    if lhs.is_none() {
                        if let Some(rhs_ty) = rhs.clone() {
                            let _ = self.infer_expr(&binary.left, Some(rhs_ty.clone()))?;
                            Some(rhs_ty)
                        } else {
                            lhs.or(rhs).or(expected)
                        }
                    } else {
                        lhs.or(rhs).or(expected)
                    }
                }
            }
            Expr::Unary(unary) => {
                if expected.is_none() && matches!(unary.op, syn::UnOp::Neg(_)) {
                    match self.infer_expr_term(expr, None)? {
                        InferredTy::Known(ty) => Some(ty),
                        term @ InferredTy::Var(_) => {
                            self.record_expr_term(expr, term);
                            None
                        }
                        InferredTy::Unknown => None,
                    }
                } else {
                    self.infer_expr(&unary.expr, expected.clone())?
                }
            }
            Expr::Lit(lit) => {
                if expected.is_some() {
                    expected
                } else if let Some(ty) = get_lit_type(lit) {
                    self.compiler
                        .compile_type(&ty, self.generic_vars, &HashMap::new())?
                } else {
                    let term = self.literal_infer_var(expr, lit);
                    self.record_expr_term(expr, term);
                    None
                }
            }
            Expr::If(if_expr) => {
                let _ = self.infer_expr(&if_expr.cond, self.bool_type()?)?;
                if let Some(expected) = expected.clone() {
                    let then_ty = self.infer_block(&if_expr.then_branch, Some(expected.clone()))?;
                    let else_ty = if let Some((_, else_expr)) = &if_expr.else_branch {
                        self.infer_expr(else_expr, Some(expected.clone()))?
                    } else {
                        None
                    };
                    then_ty.or(else_ty).or(Some(expected))
                } else {
                    match self.infer_if_expr_term(if_expr)? {
                        InferredTy::Known(ty) => Some(ty),
                        term @ InferredTy::Var(_) => {
                            self.record_expr_term(expr, term);
                            None
                        }
                        InferredTy::Unknown => None,
                    }
                }
            }
            Expr::ForLoop(for_loop) => {
                let iter_ty = self.infer_for_iter_ty(&for_loop.expr)?;
                let (results, inference) = {
                    let mut nested = self.fork();
                    if let Some(iter_ty) = iter_ty {
                        nested.bind_pattern_with_tile_type(&for_loop.pat, iter_ty)?;
                    }
                    let _ = nested.infer_block(&for_loop.body, None)?;
                    (nested.results, nested.inference)
                };
                self.results = results;
                self.inference = inference;
                None
            }
            Expr::While(while_expr) => {
                let _ = self.infer_expr(&while_expr.cond, self.bool_type()?)?;
                self.infer_scoped_block(&while_expr.body)?;
                None
            }
            Expr::Loop(loop_expr) => {
                self.infer_scoped_block(&loop_expr.body)?;
                None
            }
            Expr::Return(return_expr) => {
                if let Some(return_expr) = &return_expr.expr {
                    let _ = self.infer_expr(return_expr, expected.clone())?;
                }
                expected
            }
            Expr::Unsafe(unsafe_expr) => {
                let block_ty = self.infer_block(&unsafe_expr.block, expected.clone())?;
                if block_ty.is_none() {
                    if let Some(syn_ty) = self.block_tail_syn_type(&unsafe_expr.block) {
                        self.results
                            .insert_resolved_expr_type(expr, ResolvedType::from_syn_type(&syn_ty));
                    }
                }
                block_ty
            }
            _ => expected,
        };

        if let Some(ty) = inferred.clone() {
            self.record_expr_term(expr, InferredTy::Known(ty.clone()));
            self.results.insert_expr_type(expr, ty);
        }
        Ok(inferred)
    }

    fn bind_tile_var(&mut self, name: String, ty: TileRustType) {
        self.syn_vars.insert(name.clone(), ty.rust_ty.clone());
        self.local_terms
            .insert(name.clone(), InferredTy::Known(ty.clone()));
        self.vars.insert(name, ty);
    }

    fn bind_syn_var(&mut self, name: String, ty: Type) {
        self.syn_vars.insert(name, ty);
    }

    fn bind_inferred_var(&mut self, name: String, term: InferredTy) {
        if let Some(ty) = self.term_known_type(&term) {
            self.bind_tile_var(name, ty);
        } else {
            self.local_terms.insert(name, term);
        }
    }

    fn infer_assignment(&mut self, lhs_name: &str, rhs: &Expr) -> Result<(), JITError> {
        if let Some(lhs_term) = self.local_terms.get(lhs_name).cloned() {
            let rhs_term = if let Some(lhs_ty) = self.term_known_type(&lhs_term) {
                self.infer_expr_term(rhs, Some(lhs_ty))?
            } else {
                self.infer_expr_term(rhs, None)?
            };
            let term = self.unify_terms(lhs_term, rhs_term)?;
            self.bind_inferred_var(lhs_name.to_string(), term);
            return Ok(());
        }

        let lhs_ty = self.vars.get(lhs_name).cloned();
        let rhs_ty = self.infer_expr(rhs, lhs_ty.clone())?;
        if let Some(ty) = rhs_ty.or(lhs_ty) {
            self.bind_tile_var(lhs_name.to_string(), ty);
        }
        Ok(())
    }

    fn infer_cast_target(
        &mut self,
        cast_expr: &Expr,
        source_expr: &Expr,
        target_ty: &Type,
    ) -> Result<Option<TileRustType>, JITError> {
        let _ = self.infer_expr(source_expr, None)?;
        self.results
            .insert_resolved_expr_type(cast_expr, ResolvedType::from_syn_type(target_ty));
        match self
            .compiler
            .compile_type(target_ty, self.generic_vars, &HashMap::new())
        {
            Ok(ty) => Ok(ty),
            Err(_) if is_surface_only_scalar_type(target_ty) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn infer_bool_binary(
        &mut self,
        binary: &syn::ExprBinary,
    ) -> Result<Option<TileRustType>, JITError> {
        let bool_ty = self.bool_type()?;
        if binary_operands_are_bool(&binary.op) {
            let _ = self.infer_expr(&binary.left, bool_ty.clone())?;
            let _ = self.infer_expr(&binary.right, bool_ty.clone())?;
            return Ok(bool_ty);
        }

        let lhs = self.infer_expr_term(&binary.left, None)?;
        let rhs = self.infer_expr_term(&binary.right, None)?;
        let _ = self.unify_terms(lhs, rhs)?;
        Ok(bool_ty)
    }

    fn bool_type(&self) -> Result<Option<TileRustType>, JITError> {
        let bool_ty: Type = syn::parse_quote!(bool);
        self.compiler
            .compile_type(&bool_ty, self.generic_vars, &HashMap::new())
    }

    fn infer_scoped_block(&mut self, block: &syn::Block) -> Result<(), JITError> {
        let (results, inference) = {
            let mut nested = self.fork();
            let _ = nested.infer_block(block, None)?;
            (nested.results, nested.inference)
        };
        self.results = results;
        self.inference = inference;
        Ok(())
    }

    fn infer_closure(
        &mut self,
        closure: &syn::ExprClosure,
        param_types: &[TileRustType],
        expected_return: Option<TileRustType>,
    ) -> Result<(), JITError> {
        let closure_info = parse_closure(closure);
        let (results, inference) = {
            let mut nested = self.fork();
            for (idx, param) in closure_info.params.iter().enumerate() {
                if let Some(param_ty) = param_types.get(idx) {
                    nested.bind_tile_var(param.name.clone(), param_ty.clone());
                } else if let Some(param_ty) = &param.ty {
                    if let Some(tile_ty) = nested.compiler.compile_type(
                        param_ty,
                        nested.generic_vars,
                        &HashMap::new(),
                    )? {
                        nested.bind_tile_var(param.name.clone(), tile_ty);
                    } else {
                        nested.bind_syn_var(param.name.clone(), param_ty.clone());
                    }
                } else {
                    let term = nested.new_infer_var(InferVarKind::General, None, None);
                    nested.bind_inferred_var(param.name.clone(), term);
                }
            }
            let _ = nested.infer_expr(&closure_info.body, expected_return)?;
            (nested.results, nested.inference)
        };
        self.results = results;
        self.inference = inference;
        Ok(())
    }

    fn bind_pattern_with_tile_type(&mut self, pat: &Pat, ty: TileRustType) -> Result<(), JITError> {
        self.bind_pattern_with_type(pat, &ty.rust_ty.clone(), Some(ty))
    }

    fn bind_pattern_with_syn_type(&mut self, pat: &Pat, ty: &Type) -> Result<(), JITError> {
        let tile_ty = self
            .compiler
            .compile_type(ty, self.generic_vars, &HashMap::new())?;
        self.bind_pattern_with_type(pat, ty, tile_ty)
    }

    fn bind_pattern_with_type(
        &mut self,
        pat: &Pat,
        ty: &Type,
        tile_ty: Option<TileRustType>,
    ) -> Result<(), JITError> {
        match pat {
            Pat::Ident(ident) => {
                let name = ident.ident.to_string();
                if ident.mutability.is_some() {
                    self.mutable_vars.insert(name.clone());
                }
                if let Some(tile_ty) = tile_ty {
                    self.bind_tile_var(name, tile_ty);
                } else {
                    self.bind_syn_var(name, ty.clone());
                }
                if let Some((_at, subpat)) = &ident.subpat {
                    self.bind_pattern_with_syn_type(subpat, ty)?;
                }
            }
            Pat::Type(pat_type) => {
                self.bind_pattern_with_syn_type(&pat_type.pat, &pat_type.ty)?;
            }
            Pat::Paren(paren) => {
                self.bind_pattern_with_type(&paren.pat, ty, tile_ty)?;
            }
            Pat::Reference(reference) => {
                if let Type::Reference(reference_ty) = ty {
                    self.bind_pattern_with_syn_type(&reference.pat, &reference_ty.elem)?;
                }
            }
            Pat::Tuple(tuple) => {
                if let Some(elem_types) = tuple_element_syn_types(ty) {
                    for (pat, elem_ty) in tuple.elems.iter().zip(elem_types.iter()) {
                        self.bind_pattern_with_syn_type(pat, elem_ty)?;
                    }
                }
            }
            Pat::Slice(slice) => {
                if let Some(elem_ty) = array_or_slice_element_syn_type(ty) {
                    for pat in slice
                        .elems
                        .iter()
                        .filter(|pat| !matches!(pat, Pat::Rest(_)))
                    {
                        self.bind_pattern_with_syn_type(pat, &elem_ty)?;
                    }
                }
            }
            Pat::Struct(pat_struct) => {
                for field in &pat_struct.fields {
                    if let Some(field_ty) = self.struct_field_syn_type(ty, &field.member) {
                        self.bind_pattern_with_syn_type(&field.pat, &field_ty)?;
                    }
                }
            }
            Pat::TupleStruct(tuple_struct) => {
                if let Some(elem_types) = self.tuple_struct_element_syn_types(ty) {
                    for (pat, elem_ty) in tuple_struct.elems.iter().zip(elem_types.iter()) {
                        self.bind_pattern_with_syn_type(pat, elem_ty)?;
                    }
                }
            }
            Pat::Or(or_pat) => {
                for case in &or_pat.cases {
                    self.bind_pattern_with_type(case, ty, tile_ty.clone())?;
                }
            }
            Pat::Wild(_) | Pat::Rest(_) => {}
            _ => {}
        }
        Ok(())
    }

    fn bind_pattern_from_expr(&mut self, pat: &Pat, expr: &Expr) -> Result<(), JITError> {
        match pat {
            Pat::Ident(_) => {
                let mut term = self.infer_expr_term(expr, None)?;
                if matches!(term, InferredTy::Unknown) {
                    term =
                        self.new_infer_var(InferVarKind::General, Some(expr), Some(expr.clone()));
                }
                self.bind_pattern_with_term(pat, term)
            }
            Pat::Type(pat_type) => {
                let expected =
                    self.compiler
                        .compile_type(&pat_type.ty, self.generic_vars, &HashMap::new())?;
                let _ = self.infer_expr(expr, expected)?;
                self.bind_pattern_with_syn_type(&pat_type.pat, &pat_type.ty)
            }
            Pat::Paren(paren) => self.bind_pattern_from_expr(&paren.pat, expr),
            Pat::Reference(reference) => {
                let Expr::Reference(expr_reference) = expr else {
                    if let Some(ty) = self.infer_expr(expr, None)? {
                        return self.bind_pattern_with_tile_type(pat, ty);
                    }
                    return Ok(());
                };
                self.bind_pattern_from_expr(&reference.pat, &expr_reference.expr)
            }
            Pat::Tuple(tuple_pat) => {
                if let Expr::Tuple(tuple_expr) = expr {
                    if tuple_pat.elems.len() == tuple_expr.elems.len() {
                        for (pat, expr) in tuple_pat.elems.iter().zip(tuple_expr.elems.iter()) {
                            self.bind_pattern_from_expr(pat, expr)?;
                        }
                        return Ok(());
                    }
                }
                if let Some(ty) = self.infer_expr(expr, None)? {
                    self.bind_pattern_with_tile_type(pat, ty)?;
                } else if let Some(syn_ty) = self.results.syn_expr_type(expr) {
                    self.bind_pattern_with_syn_type(pat, &syn_ty)?;
                }
                Ok(())
            }
            Pat::Slice(slice_pat) => {
                if let Expr::Array(array_expr) = expr {
                    self.bind_slice_pattern_from_array_expr(slice_pat, array_expr)?;
                    return Ok(());
                }
                if let Some(ty) = self.infer_expr(expr, None)? {
                    self.bind_pattern_with_tile_type(pat, ty)?;
                }
                Ok(())
            }
            Pat::Struct(_) | Pat::TupleStruct(_) => {
                if let Some(ty) = self.infer_expr(expr, None)? {
                    self.bind_pattern_with_tile_type(pat, ty)?;
                }
                Ok(())
            }
            Pat::Or(or_pat) => {
                for case in &or_pat.cases {
                    self.bind_pattern_from_expr(case, expr)?;
                }
                Ok(())
            }
            Pat::Wild(_) | Pat::Rest(_) => Ok(()),
            _ => {
                if let Some(ty) = self.infer_expr(expr, None)? {
                    self.bind_pattern_with_tile_type(pat, ty)?;
                }
                Ok(())
            }
        }
    }

    fn bind_pattern_with_term(&mut self, pat: &Pat, term: InferredTy) -> Result<(), JITError> {
        match pat {
            Pat::Ident(ident) => {
                if ident.mutability.is_some() {
                    self.mutable_vars.insert(ident.ident.to_string());
                }
                self.bind_inferred_var(ident.ident.to_string(), term.clone());
                if let Some((_at, subpat)) = &ident.subpat {
                    self.bind_pattern_with_term(subpat, term)?;
                }
            }
            Pat::Type(pat_type) => {
                let expected =
                    self.compiler
                        .compile_type(&pat_type.ty, self.generic_vars, &HashMap::new())?;
                if let Some(expected) = expected {
                    let _ = self.unify_with_known(term, expected)?;
                }
                self.bind_pattern_with_syn_type(&pat_type.pat, &pat_type.ty)?;
            }
            Pat::Paren(paren) => self.bind_pattern_with_term(&paren.pat, term)?,
            Pat::Or(or_pat) => {
                for case in &or_pat.cases {
                    self.bind_pattern_with_term(case, term.clone())?;
                }
            }
            Pat::Wild(_) | Pat::Rest(_) => {}
            _ => {
                if let Some(ty) = self.term_known_type(&term) {
                    self.bind_pattern_with_tile_type(pat, ty)?;
                }
            }
        }
        Ok(())
    }

    fn bind_slice_pattern_from_array_expr(
        &mut self,
        slice_pat: &syn::PatSlice,
        array_expr: &syn::ExprArray,
    ) -> Result<(), JITError> {
        let exprs = array_expr.elems.iter().collect::<Vec<_>>();
        let pats = slice_pat.elems.iter().collect::<Vec<_>>();
        let mut elem_terms = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            elem_terms.push(self.infer_expr_term(expr, None)?);
        }

        let mut unified_elem = InferredTy::Unknown;
        for term in &elem_terms {
            unified_elem = self.unify_terms(unified_elem, term.clone())?;
        }

        let rest_pos = pats.iter().position(|pat| matches!(pat, Pat::Rest(_)));
        let mut pairs = Vec::new();
        match rest_pos {
            Some(rest_pos) => {
                if pats.len().saturating_sub(1) > exprs.len() {
                    return Ok(());
                }
                for idx in 0..rest_pos {
                    pairs.push((pats[idx], idx));
                }
                let suffix_len = pats.len() - rest_pos - 1;
                for suffix_idx in 0..suffix_len {
                    pairs.push((
                        pats[rest_pos + 1 + suffix_idx],
                        exprs.len() - suffix_len + suffix_idx,
                    ));
                }
            }
            None => {
                if pats.len() != exprs.len() {
                    return Ok(());
                }
                for idx in 0..pats.len() {
                    pairs.push((pats[idx], idx));
                }
            }
        }

        for (pat, expr_idx) in pairs {
            let term = self.unify_terms(elem_terms[expr_idx].clone(), unified_elem.clone())?;
            self.bind_pattern_with_term(pat, term)?;
        }
        Ok(())
    }

    fn struct_field_syn_type(&self, ty: &Type, member: &syn::Member) -> Option<Type> {
        let struct_name = type_path_ident(ty)?;
        let item_struct = self.compiler.modules.structs().get(&struct_name)?;
        match (&item_struct.fields, member) {
            (syn::Fields::Named(fields), syn::Member::Named(field_name)) => fields
                .named
                .iter()
                .find(|field| {
                    field
                        .ident
                        .as_ref()
                        .is_some_and(|ident| ident == field_name)
                })
                .map(|field| field.ty.clone()),
            (syn::Fields::Unnamed(fields), syn::Member::Unnamed(index)) => fields
                .unnamed
                .iter()
                .nth(index.index as usize)
                .map(|field| field.ty.clone()),
            _ => None,
        }
    }

    fn tuple_struct_element_syn_types(&self, ty: &Type) -> Option<Vec<Type>> {
        let struct_name = type_path_ident(ty)?;
        let item_struct = self.compiler.modules.structs().get(&struct_name)?;
        let syn::Fields::Unnamed(fields) = &item_struct.fields else {
            return None;
        };
        Some(
            fields
                .unnamed
                .iter()
                .map(|field| field.ty.clone())
                .collect(),
        )
    }

    fn infer_expr_term(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<InferredTy, JITError> {
        stacker::maybe_grow(
            shared_utils::STACK_RED_ZONE,
            shared_utils::STACK_GROW_SIZE,
            || self.infer_expr_term_inner(expr, expected),
        )
    }

    fn infer_expr_term_inner(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<InferredTy, JITError> {
        if let Some(expected) = expected {
            if let Some(existing) = self.expr_term(expr) {
                let term = self.unify_with_known(existing, expected)?;
                self.record_expr_term(expr, term.clone());
                return Ok(term);
            }
            if let Some(ty) = self.infer_expr(expr, Some(expected.clone()))? {
                return Ok(InferredTy::Known(ty));
            }
            return Ok(InferredTy::Known(expected));
        }

        if let Some(existing) = self.expr_term(expr) {
            return Ok(existing);
        }

        match expr {
            Expr::Path(path) if path.path.segments.len() == 1 => {
                let name = get_ident_from_path_expr(path).to_string();
                if let Some(term) = self.local_terms.get(&name).cloned() {
                    self.record_expr_term(expr, term.clone());
                    Ok(term)
                } else if let Some(ty) = self.vars.get(&name).cloned() {
                    Ok(InferredTy::Known(ty))
                } else if let Some(ty) = self.infer_global_const_type(path)? {
                    Ok(InferredTy::Known(ty))
                } else {
                    Ok(InferredTy::Unknown)
                }
            }
            Expr::Path(path) => {
                if let Some(ty) = self.infer_associated_const_type(path)? {
                    Ok(InferredTy::Known(ty))
                } else {
                    Ok(InferredTy::Unknown)
                }
            }
            Expr::Lit(lit) => {
                if let Some(ty) = get_lit_type(lit) {
                    let Some(ty) =
                        self.compiler
                            .compile_type(&ty, self.generic_vars, &HashMap::new())?
                    else {
                        return Ok(InferredTy::Unknown);
                    };
                    Ok(InferredTy::Known(ty))
                } else {
                    Ok(self.literal_infer_var(expr, lit))
                }
            }
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
                let term = self.infer_expr_term(&unary.expr, None)?;
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Paren(paren) => {
                let term = self.infer_expr_term(&paren.expr, None)?;
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Block(block) => {
                let term = self.infer_block_term(&block.block)?;
                if matches!(term, InferredTy::Unknown) {
                    if let Some(ty) = self.infer_expr(expr, None)? {
                        return Ok(InferredTy::Known(ty));
                    }
                }
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Unsafe(unsafe_expr) => {
                let term = self.infer_block_term(&unsafe_expr.block)?;
                if matches!(term, InferredTy::Unknown) {
                    if let Some(ty) = self.infer_expr(expr, None)? {
                        return Ok(InferredTy::Known(ty));
                    }
                }
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Binary(binary) if binary_result_matches_operands(&binary.op) => {
                let lhs = self.infer_expr_term(&binary.left, None)?;
                let rhs = self.infer_expr_term(&binary.right, None)?;
                let term = self.unify_terms(lhs, rhs)?;
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Binary(binary) if binary_result_is_bool(&binary.op) => {
                let Some(bool_ty) = self.bool_type()? else {
                    return Ok(InferredTy::Unknown);
                };
                let _ = self.infer_bool_binary(binary)?;
                let term = InferredTy::Known(bool_ty);
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::Cast(cast) => {
                let Some(ty) = self.infer_cast_target(expr, &cast.expr, &cast.ty)? else {
                    return Ok(InferredTy::Unknown);
                };
                let term = InferredTy::Known(ty);
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            Expr::If(if_expr) => {
                let _ = self.infer_expr(&if_expr.cond, self.bool_type()?)?;
                let term = self.infer_if_expr_term(if_expr)?;
                self.record_expr_term(expr, term.clone());
                Ok(term)
            }
            _ => match self.infer_expr(expr, None)? {
                Some(ty) => Ok(InferredTy::Known(ty)),
                None => Ok(InferredTy::Unknown),
            },
        }
    }

    fn infer_if_expr_term(&mut self, if_expr: &syn::ExprIf) -> Result<InferredTy, JITError> {
        let then_term = self.infer_block_branch_term(&if_expr.then_branch)?;
        let else_term = if let Some((_, else_expr)) = &if_expr.else_branch {
            self.infer_expr_branch_term(else_expr)?
        } else {
            BranchTerm::Unknown
        };
        self.join_branch_terms(then_term, else_term)
    }

    fn infer_expr_branch_term(&mut self, expr: &Expr) -> Result<BranchTerm, JITError> {
        if expr_diverges(expr) {
            let _ = self.infer_expr(expr, None)?;
            return Ok(BranchTerm::Diverges);
        }
        self.infer_expr_term(expr, None).map(BranchTerm::from_term)
    }

    fn infer_block_branch_term(&mut self, block: &syn::Block) -> Result<BranchTerm, JITError> {
        if block_diverges(block) {
            let _ = self.infer_block(block, None)?;
            return Ok(BranchTerm::Diverges);
        }
        self.infer_block_term(block).map(BranchTerm::from_term)
    }

    fn join_branch_terms(
        &mut self,
        lhs: BranchTerm,
        rhs: BranchTerm,
    ) -> Result<InferredTy, JITError> {
        match (lhs, rhs) {
            (BranchTerm::Diverges, BranchTerm::Diverges) => Ok(InferredTy::Unknown),
            (BranchTerm::Diverges, BranchTerm::Value(term))
            | (BranchTerm::Value(term), BranchTerm::Diverges)
            | (BranchTerm::Unknown, BranchTerm::Value(term))
            | (BranchTerm::Value(term), BranchTerm::Unknown) => Ok(term),
            (BranchTerm::Value(lhs), BranchTerm::Value(rhs)) => self.unify_terms(lhs, rhs),
            (BranchTerm::Diverges, BranchTerm::Unknown)
            | (BranchTerm::Unknown, BranchTerm::Diverges)
            | (BranchTerm::Unknown, BranchTerm::Unknown) => Ok(InferredTy::Unknown),
        }
    }

    fn infer_block_term(&mut self, block: &syn::Block) -> Result<InferredTy, JITError> {
        let Some((last, prefix)) = block.stmts.split_last() else {
            return Ok(InferredTy::Unknown);
        };
        for stmt in prefix {
            let _ = self.infer_stmt(stmt, None)?;
        }
        match last {
            Stmt::Expr(expr, None) => self.infer_expr_term(expr, None),
            Stmt::Expr(expr, _) => {
                let _ = self.infer_expr(expr, None)?;
                Ok(InferredTy::Unknown)
            }
            Stmt::Local(_) | Stmt::Item(_) | Stmt::Macro(_) => {
                let _ = self.infer_stmt(last, None)?;
                Ok(InferredTy::Unknown)
            }
        }
    }

    fn literal_infer_var(&mut self, expr: &Expr, lit: &ExprLit) -> InferredTy {
        if let Some(existing) = self.expr_term(expr) {
            return existing;
        }
        let kind = match &lit.lit {
            Lit::Int(raw) if raw.suffix().is_empty() => InferVarKind::Int,
            Lit::Float(raw) if raw.suffix().is_empty() => InferVarKind::Float,
            _ => return InferredTy::Unknown,
        };
        self.new_infer_var(kind, Some(expr), Some(expr.clone()))
    }

    fn new_infer_var(
        &mut self,
        kind: InferVarKind,
        expr: Option<&Expr>,
        origin: Option<Expr>,
    ) -> InferredTy {
        let id = InferVarId(self.inference.vars.len());
        let expr_ids = expr
            .and_then(node_ids::expr_id)
            .map(|id| vec![id])
            .unwrap_or_default();
        self.inference.vars.push(InferVar {
            kind,
            parent: None,
            value: None,
            expr_ids,
            origins: origin.into_iter().collect(),
            origin_propagated: false,
        });
        let term = InferredTy::Var(id);
        if let Some(expr) = expr {
            self.record_expr_term(expr, term.clone());
        }
        term
    }

    fn expr_term(&self, expr: &Expr) -> Option<InferredTy> {
        self.inference
            .expr_terms
            .get(&node_ids::expr_id(expr)?)
            .cloned()
    }

    fn expr_or_local_term(&self, expr: &Expr) -> Option<InferredTy> {
        if let Some(term) = self.expr_term(expr) {
            return Some(term);
        }
        let Expr::Path(path) = expr else {
            return None;
        };
        if path.path.segments.len() != 1 {
            return None;
        }
        let name = get_ident_from_path_expr(path).to_string();
        self.local_terms.get(&name).cloned()
    }

    fn term_origin(&self, term: &InferredTy) -> Option<Expr> {
        let InferredTy::Var(id) = self.normalize_term(term.clone()) else {
            return None;
        };
        self.inference
            .vars
            .get(id.0)
            .and_then(|var| var.origins.first().cloned())
    }

    fn record_expr_term(&mut self, expr: &Expr, term: InferredTy) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        let term = self.normalize_term(term);
        self.inference.expr_terms.insert(id, term.clone());
        if let InferredTy::Var(var_id) = term {
            let Some(root_id) = self.find_var_id(var_id) else {
                return;
            };
            let Some(var) = self.inference.vars.get_mut(root_id.0) else {
                return;
            };
            if !var.expr_ids.contains(&id) {
                var.expr_ids.push(id);
            }
        }
    }

    fn term_known_type(&self, term: &InferredTy) -> Option<TileRustType> {
        match self.normalize_term(term.clone()) {
            InferredTy::Known(ty) => Some(ty.clone()),
            InferredTy::Var(id) => {
                let root_id = self.find_var_id(id)?;
                self.inference
                    .vars
                    .get(root_id.0)
                    .and_then(|var| var.value.clone())
            }
            InferredTy::Unknown => None,
        }
    }

    fn normalize_term(&self, term: InferredTy) -> InferredTy {
        match term {
            InferredTy::Var(id) => self
                .find_var_id(id)
                .map(InferredTy::Var)
                .unwrap_or(InferredTy::Unknown),
            _ => term,
        }
    }

    fn find_var_id(&self, mut id: InferVarId) -> Option<InferVarId> {
        let mut seen = 0usize;
        loop {
            let var = self.inference.vars.get(id.0)?;
            let Some(parent) = var.parent else {
                return Some(id);
            };
            id = parent;
            seen += 1;
            if seen > self.inference.vars.len() {
                return None;
            }
        }
    }

    fn unify_terms(&mut self, lhs: InferredTy, rhs: InferredTy) -> Result<InferredTy, JITError> {
        let lhs = self.normalize_term(lhs);
        let rhs = self.normalize_term(rhs);
        match (lhs, rhs) {
            (InferredTy::Unknown, term) | (term, InferredTy::Unknown) => Ok(term),
            (InferredTy::Known(lhs), InferredTy::Known(rhs)) => {
                if lhs.rust_ty == rhs.rust_ty {
                    Ok(InferredTy::Known(lhs))
                } else {
                    Ok(InferredTy::Unknown)
                }
            }
            (InferredTy::Known(known), term @ InferredTy::Var(_))
            | (term @ InferredTy::Var(_), InferredTy::Known(known)) => {
                self.unify_with_known(term, known)
            }
            (InferredTy::Var(lhs), InferredTy::Var(rhs)) => self.unify_vars(lhs, rhs),
        }
    }

    fn unify_vars(&mut self, lhs: InferVarId, rhs: InferVarId) -> Result<InferredTy, JITError> {
        let Some(lhs_root) = self.find_var_id(lhs) else {
            return Ok(InferredTy::Unknown);
        };
        let Some(rhs_root) = self.find_var_id(rhs) else {
            return Ok(InferredTy::Unknown);
        };
        if lhs_root == rhs_root {
            return Ok(InferredTy::Var(lhs_root));
        }

        let lhs_value = self.inference.vars[lhs_root.0].value.clone();
        let rhs_value = self.inference.vars[rhs_root.0].value.clone();
        let root_value = match (&lhs_value, &rhs_value) {
            (Some(lhs), Some(rhs)) if lhs.rust_ty == rhs.rust_ty => Some(lhs.clone()),
            (Some(lhs), None) if self.var_accepts_type(rhs_root, &lhs.rust_ty) => Some(lhs.clone()),
            (None, Some(rhs)) if self.var_accepts_type(lhs_root, &rhs.rust_ty) => Some(rhs.clone()),
            (None, None) => None,
            _ => return Ok(InferredTy::Unknown),
        };
        let Some(root_kind) = merge_infer_var_kinds(
            self.inference.vars[lhs_root.0].kind,
            self.inference.vars[rhs_root.0].kind,
        ) else {
            return Ok(InferredTy::Unknown);
        };

        let (root, child) = if lhs_value.is_some() && rhs_value.is_none() {
            (lhs_root, rhs_root)
        } else if rhs_value.is_some() && lhs_value.is_none() {
            (rhs_root, lhs_root)
        } else if lhs_root.0 <= rhs_root.0 {
            (lhs_root, rhs_root)
        } else {
            (rhs_root, lhs_root)
        };

        let child_var = self.inference.vars[child.0].clone();
        {
            let root_var = &mut self.inference.vars[root.0];
            root_var.kind = root_kind;
            if let Some(value) = root_value {
                root_var.value = Some(value);
            }
            for expr_id in child_var.expr_ids {
                if !root_var.expr_ids.contains(&expr_id) {
                    root_var.expr_ids.push(expr_id);
                }
            }
            for origin in child_var.origins {
                root_var.origins.push(origin);
            }
            root_var.origin_propagated = false;
        }
        self.inference.vars[child.0].parent = Some(root);
        Ok(InferredTy::Var(root))
    }

    fn var_accepts_type(&self, id: InferVarId, ty: &Type) -> bool {
        let Some(root) = self.find_var_id(id) else {
            return false;
        };
        literal_kind_accepts_type(self.inference.vars[root.0].kind, ty)
    }

    fn unify_with_known(
        &mut self,
        term: InferredTy,
        expected: TileRustType,
    ) -> Result<InferredTy, JITError> {
        match self.normalize_term(term) {
            InferredTy::Known(known) => Ok(InferredTy::Known(known)),
            InferredTy::Unknown => Ok(InferredTy::Known(expected)),
            InferredTy::Var(id) => {
                let Some(root_id) = self.find_var_id(id) else {
                    return Ok(InferredTy::Unknown);
                };
                let Some(var) = self.inference.vars.get_mut(root_id.0) else {
                    return Ok(InferredTy::Unknown);
                };
                if let Some(known) = &var.value {
                    return Ok(InferredTy::Known(known.clone()));
                }
                if literal_kind_accepts_type(var.kind, &expected.rust_ty) {
                    var.value = Some(expected.clone());
                    var.origin_propagated = false;
                    Ok(InferredTy::Known(expected))
                } else {
                    Ok(InferredTy::Var(id))
                }
            }
        }
    }

    fn finish_inference(&mut self) -> Result<(), JITError> {
        self.propagate_resolved_origins()?;
        self.fallback_unresolved_literals()?;
        self.propagate_resolved_origins()?;
        self.writeback_inferred_expr_types();
        Ok(())
    }

    fn propagate_resolved_origins(&mut self) -> Result<(), JITError> {
        loop {
            let mut work = Vec::new();
            for (idx, var) in self.inference.vars.iter().enumerate() {
                if var.parent.is_some() {
                    continue;
                }
                if var.origin_propagated {
                    continue;
                }
                let Some(value) = &var.value else {
                    continue;
                };
                if var.origins.is_empty() {
                    continue;
                }
                work.push((InferVarId(idx), var.origins.clone(), value.clone()));
            }
            if work.is_empty() {
                return Ok(());
            }
            for (id, origins, value) in work {
                if let Some(var) = self.inference.vars.get_mut(id.0) {
                    var.origin_propagated = true;
                }
                for origin in origins {
                    let _ = self.infer_expr(&origin, Some(value.clone()))?;
                }
            }
        }
    }

    fn fallback_unresolved_literals(&mut self) -> Result<(), JITError> {
        let fallbacks = self
            .inference
            .vars
            .iter()
            .map(|var| {
                if var.parent.is_some() || var.value.is_some() {
                    None
                } else {
                    match var.kind {
                        InferVarKind::Int => Some(syn::parse_quote!(i32)),
                        InferVarKind::Float => Some(syn::parse_quote!(f64)),
                        InferVarKind::General => None,
                    }
                }
            })
            .collect::<Vec<Option<Type>>>();
        for (idx, fallback) in fallbacks.into_iter().enumerate() {
            let Some(fallback) = fallback else {
                continue;
            };
            let Some(ty) =
                self.compiler
                    .compile_type(&fallback, self.generic_vars, &HashMap::new())?
            else {
                continue;
            };
            if let Some(var) = self.inference.vars.get_mut(idx) {
                var.value = Some(ty);
                var.origin_propagated = false;
            }
        }
        Ok(())
    }

    fn writeback_inferred_expr_types(&mut self) {
        let expr_terms = self.inference.expr_terms.clone();
        for (id, term) in expr_terms {
            if let Some(ty) = self.term_known_type(&term) {
                self.results.insert_expr_type_by_id(id, ty);
            }
        }
    }

    fn infer_expr_syn_type(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<Option<Type>, JITError> {
        if let Some(inferred) = self.infer_expr(expr, expected)? {
            return Ok(Some(inferred.rust_ty));
        }
        if let Some(syn_type) = self.results.syn_expr_type(expr) {
            return Ok(Some(syn_type));
        }
        match expr {
            Expr::Path(path) if path.path.segments.len() == 1 => {
                let name = get_ident_from_path_expr(path).to_string();
                Ok(self.syn_vars.get(&name).cloned())
            }
            Expr::Paren(paren) => self.infer_expr_syn_type(&paren.expr, None),
            Expr::Reference(reference) => {
                let Some(elem_ty) = self.infer_expr_syn_type(&reference.expr, None)? else {
                    return Ok(None);
                };
                Ok(Some(Type::Reference(syn::TypeReference {
                    and_token: reference.and_token,
                    lifetime: None,
                    mutability: reference.mutability,
                    elem: Box::new(elem_ty),
                })))
            }
            _ => Ok(None),
        }
    }

    fn array_element_type(
        &self,
        expected: &TileRustType,
    ) -> Result<Option<TileRustType>, JITError> {
        let elem_ty = match &expected.rust_ty {
            Type::Array(array) => Some(&*array.elem),
            Type::Slice(slice) => Some(&*slice.elem),
            _ => None,
        };
        let Some(elem_ty) = elem_ty else {
            return Ok(None);
        };
        self.compiler
            .compile_type(elem_ty, self.generic_vars, &HashMap::new())
    }

    fn infer_index(
        &mut self,
        index: &syn::ExprIndex,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let Some(base_ty) = self.infer_expr(&index.expr, None)? else {
            if let Some(expected_elem_ty) = expected.clone() {
                if let Some(array_ty) =
                    self.expected_array_type_from_index_base(&index.expr, &expected_elem_ty)?
                {
                    let _ = self.infer_expr(&index.expr, Some(array_ty))?;
                    return Ok(Some(expected_elem_ty));
                }
            }
            return Ok(None);
        };
        match &base_ty.rust_ty {
            Type::Array(array) => {
                self.compiler
                    .compile_type(&array.elem, self.generic_vars, &HashMap::new())
            }
            Type::Slice(slice) => {
                self.compiler
                    .compile_type(&slice.elem, self.generic_vars, &HashMap::new())
            }
            Type::Tuple(tuple) => {
                let Some(idx) = static_usize_index(&index.index) else {
                    return Ok(None);
                };
                let Some(elem_ty) = tuple.elems.iter().nth(idx) else {
                    return Ok(None);
                };
                self.compiler
                    .compile_type(elem_ty, self.generic_vars, &HashMap::new())
            }
            Type::Path(path)
                if path.qself.is_none()
                    && path.path.segments.last().is_some_and(|segment| {
                        let ident = segment.ident.to_string();
                        ident == "Shape" || ident.starts_with("Shape_")
                    }) =>
            {
                let i32_ty: Type = syn::parse_quote!(i32);
                self.compiler
                    .compile_type(&i32_ty, self.generic_vars, &HashMap::new())
            }
            _ => Ok(None),
        }
    }

    fn expected_array_type_from_index_base(
        &mut self,
        base: &Expr,
        expected_elem_ty: &TileRustType,
    ) -> Result<Option<TileRustType>, JITError> {
        let Some(array_like) = self.array_like_expr_or_origin(base) else {
            return Ok(None);
        };
        let elem_ty = &expected_elem_ty.rust_ty;
        let len = array_like.len();
        let array_syn_ty: Type = syn::parse_quote!([#elem_ty; #len]);
        let Some(array_ty) =
            self.compiler
                .compile_type(&array_syn_ty, self.generic_vars, &HashMap::new())?
        else {
            return Ok(None);
        };
        if let Some(term) = self.expr_or_local_term(base) {
            let term = self.unify_with_known(term, array_ty.clone())?;
            self.record_expr_term(base, term);
        }
        Ok(Some(array_ty))
    }

    fn array_like_expr_or_origin(&self, expr: &Expr) -> Option<ArrayLikeExpr> {
        match expr {
            Expr::Array(array) => Some(ArrayLikeExpr::Array(array.clone())),
            Expr::Repeat(repeat) => repeat_length_value(&repeat.len, self.generic_vars)
                .map(|len| ArrayLikeExpr::Repeat { len }),
            Expr::Paren(paren) => self.array_like_expr_or_origin(&paren.expr),
            _ => self
                .expr_or_local_term(expr)
                .and_then(|term| match self.term_origin(&term) {
                    Some(Expr::Array(array)) => Some(ArrayLikeExpr::Array(array)),
                    Some(Expr::Repeat(repeat)) => {
                        repeat_length_value(&repeat.len, self.generic_vars)
                            .map(|len| ArrayLikeExpr::Repeat { len })
                    }
                    _ => None,
                }),
        }
    }

    fn infer_struct(
        &mut self,
        struct_expr: &syn::ExprStruct,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let struct_ty = Type::Path(syn::TypePath {
            qself: None,
            path: struct_expr.path.clone(),
        });
        let struct_name = match struct_expr.path.segments.last() {
            Some(segment) => segment.ident.to_string(),
            None => return Ok(expected),
        };

        for field in &struct_expr.fields {
            let syn::Member::Named(field_name) = &field.member else {
                let _ = self.infer_expr(&field.expr, None)?;
                continue;
            };
            let field_expected = self
                .compiler
                .modules
                .get_struct_field_type(&struct_name, &field_name.to_string())
                .and_then(|field_ty| {
                    self.compiler
                        .compile_type(&field_ty, self.generic_vars, &HashMap::new())
                        .ok()
                        .flatten()
                });
            let _ = self.infer_expr(&field.expr, field_expected)?;
        }

        self.compiler
            .compile_type(&struct_ty, self.generic_vars, &HashMap::new())
            .map(|ty| ty.or(expected))
    }

    fn infer_for_iter_ty(&mut self, iter_expr: &Expr) -> Result<Option<TileRustType>, JITError> {
        if let Expr::MethodCall(method_call) = iter_expr {
            let is_mapped_indices_iter = matches!(
                (
                    method_call.method.to_string().as_str(),
                    method_call.args.len()
                ),
                ("iter_indices", 0)
                    | ("iter_indices_with", 1)
                    | ("iter_indices_within", 1)
                    | ("iter_indices_within_with", 2)
            );
            if is_mapped_indices_iter {
                if let Some(receiver_ty) = self.infer_expr_syn_type(&method_call.receiver, None)? {
                    if get_type_ident(&receiver_ty)
                        .is_some_and(|ident| ident.to_string().starts_with("MappedPartitionMut"))
                    {
                        let Some(type_generic_args) = maybe_generic_args(&receiver_ty) else {
                            return Ok(None);
                        };
                        let Some(tile_shape_arg) = type_generic_args.args.iter().nth(1) else {
                            return Ok(None);
                        };
                        let Some(tile_shape) =
                            get_cga_from_generic_argument(tile_shape_arg, self.generic_vars)
                        else {
                            return Ok(None);
                        };
                        let shape = tile_shape
                            .iter()
                            .map(|dim| dim.to_string())
                            .collect::<Vec<_>>()
                            .join(", ");
                        let index_ty: Type =
                            syn::parse_str(&format!("PartitionIndex<{{ [{shape}] }}>"))
                                .map_err(|err| JITError::Generic(err.to_string()))?;
                        return self.compiler.compile_type(
                            &index_ty,
                            self.generic_vars,
                            &HashMap::new(),
                        );
                    }
                }
            }
            if method_call.method == "step_by" {
                let receiver = match &*method_call.receiver {
                    Expr::Paren(paren) => &*paren.expr,
                    receiver => receiver,
                };
                if let Expr::Range(range) = receiver {
                    let _ = method_call
                        .args
                        .iter()
                        .map(|arg| self.infer_expr(arg, None))
                        .collect::<Result<Vec<_>, _>>()?;
                    return self.infer_range(range);
                }
            }
        }
        let iter_ty = self.infer_expr(iter_expr, None)?;
        if iter_ty
            .as_ref()
            .and_then(|ty| get_type_ident(&ty.rust_ty))
            .is_some_and(|ident| ident == "Dim")
        {
            return self.compiler.compile_type(
                &syn::parse_quote!(i32),
                self.generic_vars,
                &HashMap::new(),
            );
        }
        Ok(iter_ty)
    }

    fn block_tail_syn_type(&self, block: &syn::Block) -> Option<Type> {
        match block.stmts.last()? {
            Stmt::Expr(Expr::Return(return_expr), _) => return_expr
                .expr
                .as_ref()
                .and_then(|expr| self.results.syn_expr_type(expr)),
            Stmt::Expr(expr, None) => self.results.syn_expr_type(expr),
            _ => None,
        }
    }

    fn infer_range(&mut self, range: &syn::ExprRange) -> Result<Option<TileRustType>, JITError> {
        if let (Some(start), Some(end)) = (&range.start, &range.end) {
            if is_unsuffixed_numeric_literal(start) {
                let end_ty = self.infer_expr(end, None)?;
                let start_ty = self.infer_expr(start, end_ty.clone())?;
                return Ok(start_ty.or(end_ty));
            }
        }
        let start_ty = match &range.start {
            Some(start) => self.infer_expr(start, None)?,
            None => None,
        };
        let end_ty = match &range.end {
            Some(end) => self.infer_expr(end, start_ty.clone())?,
            None => None,
        };
        if start_ty.is_none() {
            if let (Some(start), Some(end_ty)) = (&range.start, end_ty.clone()) {
                let _ = self.infer_expr(start, Some(end_ty))?;
            }
        }
        Ok(start_ty.or(end_ty))
    }

    fn fork(&self) -> TypeInferenceCx<'a, 'm> {
        TypeInferenceCx {
            compiler: self.compiler,
            generic_vars: self.generic_vars,
            generic_type_bounds: self.generic_type_bounds.clone(),
            syn_vars: self.syn_vars.clone(),
            mutable_vars: self.mutable_vars.clone(),
            vars: self.vars.clone(),
            local_terms: self.local_terms.clone(),
            inference: self.inference.clone(),
            results: self.results.clone(),
        }
    }

    fn infer_zst_marker_type(&self, expr: &Expr) -> Option<TileRustType> {
        let Expr::Path(path_expr) = expr else {
            return None;
        };
        let path_ty = Type::Path(syn::TypePath {
            qself: None,
            path: path_expr.path.clone(),
        });
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: path_ty,
        });
        Some(TileRustType::new_string(type_instance))
    }

    fn infer_tuple(
        &mut self,
        tuple: &syn::ExprTuple,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        if let Some(expected) = expected {
            if let Some(elem_types) = tuple_element_syn_types(&expected.rust_ty) {
                if elem_types.len() == tuple.elems.len() {
                    for (elem, elem_ty) in tuple.elems.iter().zip(elem_types.iter()) {
                        let expected_elem = self.compiler.compile_type(
                            elem_ty,
                            self.generic_vars,
                            &HashMap::new(),
                        )?;
                        let _ = self.infer_expr(elem, expected_elem)?;
                    }
                    return Ok(Some(expected));
                }
            }
            return Ok(Some(expected));
        }

        let mut elem_types = Vec::new();
        for elem in &tuple.elems {
            if let Some(elem_ty) = self.infer_expr(elem, None)? {
                elem_types.push(elem_ty.rust_ty);
            }
        }
        if elem_types.len() != tuple.elems.len() {
            return Ok(None);
        }
        let tuple_ty = Type::Tuple(syn::TypeTuple {
            paren_token: syn::token::Paren::default(),
            elems: elem_types.into_iter().collect(),
        });
        self.compiler
            .compile_type(&tuple_ty, self.generic_vars, &HashMap::new())
    }

    fn infer_macro(&self, macro_expr: &ExprMacro) -> Result<Option<TileRustType>, JITError> {
        let Some(name) = macro_expr
            .mac
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
        else {
            return Ok(None);
        };
        let ty_name = match name.as_str() {
            "const_shape" | "shape" => "Shape",
            "const_array" => "Array",
            _ => return Ok(None),
        };
        let mut dims = Vec::new();
        for token in macro_expr.mac.tokens.clone() {
            match token {
                TokenTree::Literal(lit) => dims.push(lit.to_string()),
                TokenTree::Ident(ident) => {
                    let name = ident.to_string();
                    if let Some(array) = self.generic_vars.inst_array.get(&name) {
                        dims.extend(array.iter().map(i32::to_string));
                    } else {
                        dims.push(name);
                    }
                }
                TokenTree::Punct(punct) if punct.as_char() == ',' => {}
                _ => return Ok(None),
            }
        }
        let dims = dims.join(", ");
        let ty: Type = syn::parse_str(&format!("{ty_name}<{{ [{dims}] }}>"))
            .map_err(|err| JITError::Generic(err.to_string()))?;
        self.compiler
            .compile_type(&ty, self.generic_vars, &HashMap::new())
    }

    fn infer_field(
        &mut self,
        field: &syn::ExprField,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let Some(base_ty) = self.infer_expr(&field.base, None)? else {
            if let Some(expected) = expected {
                if self
                    .expected_tuple_type_from_field_base(&field.base, &field.member, &expected)?
                    .is_some()
                {
                    return Ok(Some(expected));
                }
            }
            return Ok(None);
        };
        match (&base_ty.rust_ty, &field.member) {
            (Type::Tuple(tuple), syn::Member::Unnamed(index)) => {
                let Some(elem_ty) = tuple.elems.iter().nth(index.index as usize) else {
                    return Ok(None);
                };
                self.compiler
                    .compile_type(elem_ty, self.generic_vars, &HashMap::new())
            }
            (Type::Path(path), syn::Member::Named(field_name)) if path.qself.is_none() => {
                let Some(struct_name) = path.path.segments.last().map(|segment| &segment.ident)
                else {
                    return Ok(None);
                };
                let Some(item_struct) = self
                    .compiler
                    .modules
                    .structs()
                    .get(&struct_name.to_string())
                else {
                    return Ok(None);
                };
                let syn::Fields::Named(fields) = &item_struct.fields else {
                    return Ok(None);
                };
                let Some(field) = fields.named.iter().find(|field| {
                    field
                        .ident
                        .as_ref()
                        .is_some_and(|ident| ident == field_name)
                }) else {
                    return Ok(None);
                };
                self.compiler
                    .compile_type(&field.ty, self.generic_vars, &HashMap::new())
            }
            _ => Ok(None),
        }
    }

    fn expected_tuple_type_from_field_base(
        &mut self,
        base: &Expr,
        member: &syn::Member,
        expected_elem_ty: &TileRustType,
    ) -> Result<Option<TileRustType>, JITError> {
        let syn::Member::Unnamed(index) = member else {
            return Ok(None);
        };
        let Some(tuple) = self.tuple_expr_or_origin(base) else {
            return Ok(None);
        };
        let target_idx = index.index as usize;
        if target_idx >= tuple.elems.len() {
            return Ok(None);
        }
        let mut elem_types = Vec::new();
        for (idx, elem) in tuple.elems.iter().enumerate() {
            if idx == target_idx {
                elem_types.push(expected_elem_ty.rust_ty.clone());
            } else if let Some(elem_ty) = self.infer_expr_syn_type(elem, None)? {
                elem_types.push(elem_ty);
            } else {
                return Ok(None);
            }
        }
        let tuple_ty = Type::Tuple(syn::TypeTuple {
            paren_token: syn::token::Paren::default(),
            elems: elem_types.into_iter().collect(),
        });
        let Some(tuple_ty) =
            self.compiler
                .compile_type(&tuple_ty, self.generic_vars, &HashMap::new())?
        else {
            return Ok(None);
        };
        let _ = self.infer_expr(base, Some(tuple_ty.clone()))?;
        Ok(Some(tuple_ty))
    }

    fn tuple_expr_or_origin(&self, expr: &Expr) -> Option<syn::ExprTuple> {
        match expr {
            Expr::Tuple(tuple) => Some(tuple.clone()),
            Expr::Paren(paren) => self.tuple_expr_or_origin(&paren.expr),
            _ => self
                .expr_or_local_term(expr)
                .and_then(|term| match self.term_origin(&term) {
                    Some(Expr::Tuple(tuple)) => Some(tuple),
                    _ => None,
                }),
        }
    }

    fn infer_call(
        &mut self,
        call: &ExprCall,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let Expr::Path(path) = &*call.func else {
            return Ok(expected);
        };
        if is_dim_new_call_path(path) && call.args.len() == 1 {
            let i32_ty = self.compiler.compile_type(
                &syn::parse_quote!(i32),
                self.generic_vars,
                &HashMap::new(),
            )?;
            if let Some(i32_ty) = i32_ty.clone() {
                let _ = self.infer_expr(&call.args[0], Some(i32_ty.clone()))?;
                return self.compiler.compile_type(
                    &syn::parse_quote!(Dim),
                    self.generic_vars,
                    &HashMap::new(),
                );
            }
        }
        let ident = get_ident_from_path_expr(path).to_string();
        // `coord((a, b, ...))` returns the arity-matched `Coord{N}`. The
        // declared return type is the associated `C::Coord`, which bottom-up
        // inference cannot project, so resolve it from the tuple arity here
        // (needed when the expected type is not supplied by a concrete
        // parameter, e.g. the generic `MappedPartitionMut::store` index).
        if ident == "coord" && call.args.len() == 1 {
            if let Some(tuple) = self.tuple_expr_or_origin(&call.args[0]) {
                let rank = tuple.elems.len();
                let _ = self.infer_call_arg_types(call.args.iter())?;
                if let Ok(coord_ty) = syn::parse_str::<Type>(&format!("Coord{rank}")) {
                    if let Some(ty) =
                        self.compiler
                            .compile_type(&coord_ty, self.generic_vars, &HashMap::new())?
                    {
                        return Ok(Some(ty));
                    }
                }
            }
        }
        if ident == "Some" && call.args.len() == 1 {
            if let Some(expected) = expected {
                let payload_expected =
                    option_payload_syn_type(&expected.rust_ty).and_then(|payload_ty| {
                        self.compiler
                            .compile_type(&payload_ty, self.generic_vars, &HashMap::new())
                            .ok()
                            .flatten()
                    });
                let _ = self.infer_expr(&call.args[0], payload_expected)?;
                return Ok(Some(expected));
            }
            let Some(payload_ty) = self.infer_expr(&call.args[0], None)? else {
                return Ok(None);
            };
            let payload_ty = payload_ty.rust_ty;
            let option_ty: Type = syn::parse_quote!(Option<#payload_ty>);
            return self
                .compiler
                .compile_type(&option_ty, self.generic_vars, &HashMap::new());
        }

        let Some((_, fn_item)) = self.compiler.modules.get_function_by_name(&ident) else {
            let _ = self.infer_call_arg_types(call.args.iter())?;
            return Ok(expected);
        };

        self.infer_signature_closure_args(fn_item, call, expected.as_ref())?;

        let arg_types = self.infer_function_call_arg_types(fn_item, call, expected.as_ref())?;
        if let Some(lowered_method_call) =
            crate::passes::typed_dispatch_lowering::lower_dispatch_wrapper_call(fn_item, call)
        {
            if let Some(selection) = self.infer_method_call_selection(&lowered_method_call)? {
                self.results
                    .insert_lowered_method_call(call, lowered_method_call.clone());
                self.results
                    .insert_method_selection_for_expr(&Expr::Call(call.clone()), selection.clone());
                return Ok(selection.return_type.or(expected));
            }
        }

        self.infer_function_return(fn_item, call, &arg_types, expected)
    }

    fn infer_signature_closure_args(
        &mut self,
        fn_item: &ItemFn,
        call: &ExprCall,
        expected_return: Option<&TileRustType>,
    ) -> Result<(), JITError> {
        let (param_types, return_type) = get_sig_types(&fn_item.sig, None);
        let closure_signatures = call
            .args
            .iter()
            .zip(param_types.iter())
            .map(|(arg, param_type)| match arg {
                Expr::Closure(_) => closure_signature_for_param(&fn_item.sig, param_type),
                _ => None,
            })
            .collect::<Vec<_>>();
        if closure_signatures.iter().all(Option::is_none) {
            return Ok(());
        }
        let closure_generics =
            closure_signature_generic_names(&fn_item.sig, closure_signatures.iter().flatten());

        let mut generic_arg_inf = GenericArgInference::new_function(fn_item.sig.clone());
        if let Some(expected_return) = expected_return {
            // Speculative seed from the expected type: a conflict just means
            // the hint doesn't apply, the strict check runs at inline time.
            let _ = generic_arg_inf.add_type_constraints(&return_type, &expected_return.rust_ty);
        }
        generic_arg_inf.apply_provided_generics_fn_call(call, self.generic_vars);

        for ((arg, param_type), signature) in call
            .args
            .iter()
            .zip(param_types.iter())
            .zip(closure_signatures.iter())
        {
            if let Expr::Closure(closure) = arg {
                if let Some(signature) = signature {
                    if let Some((param_types, return_type)) =
                        self.instantiate_closure_signature(signature, &generic_arg_inf)
                    {
                        self.infer_closure(closure, &param_types, return_type)?;
                    } else {
                        self.infer_closure(closure, &[], None)?;
                    }
                } else {
                    self.infer_closure(closure, &[], None)?;
                }
                continue;
            }

            let expected = substitute_resolved_generics(param_type, &generic_arg_inf)
                .as_ref()
                .and_then(|ty| self.compile_expected_param_type(ty))
                .or_else(|| {
                    concrete_scalar_param_type(param_type).and_then(|param_type| {
                        self.compiler
                            .compile_type(param_type, self.generic_vars, &HashMap::new())
                            .ok()
                            .flatten()
                    })
                });
            if let Some(arg_ty) = self.infer_expr_syn_type(arg, expected)? {
                if type_mentions_any_name(param_type, &closure_generics)
                    && can_add_type_constraint(param_type, &arg_ty)
                {
                    let _ = generic_arg_inf.add_type_constraints(param_type, &arg_ty);
                }
            }
        }
        Ok(())
    }

    fn instantiate_closure_signature(
        &self,
        signature: &ClosureSignature,
        inference: &GenericArgInference,
    ) -> Option<(Vec<TileRustType>, Option<TileRustType>)> {
        let mut param_types = Vec::with_capacity(signature.inputs.len());
        for param_type in &signature.inputs {
            let param_type = substitute_resolved_generics(param_type, inference)?;
            param_types.push(self.compile_closure_value_type(&param_type)?);
        }
        let return_type = signature
            .output
            .as_ref()
            .and_then(|return_type| substitute_resolved_generics(return_type, inference))
            .and_then(|return_type| self.compile_closure_value_type(&return_type));
        Some((param_types, return_type))
    }

    fn compile_closure_value_type(&self, ty: &Type) -> Option<TileRustType> {
        if let Some(name) = scalar_type_name(ty) {
            if name == "bool" || is_integer_scalar_name(&name) || is_float_scalar_name(&name) {
                if let Some(scalar_tile) = TileRustType::from_scalar_tile(&name) {
                    return Some(scalar_tile);
                }
            }
        }
        self.compile_expected_param_type(ty)
    }

    fn infer_method_call(
        &mut self,
        method_call: &ExprMethodCall,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        if let Some(global_return) = self.infer_global_method_call(method_call)? {
            return Ok(Some(global_return));
        }
        if let Some(selection) = self.infer_method_call_selection(method_call)? {
            self.results
                .insert_method_selection(method_call, selection.clone());
            Ok(selection.return_type.or(expected))
        } else {
            Ok(expected)
        }
    }

    fn infer_global_method_call(
        &mut self,
        method_call: &ExprMethodCall,
    ) -> Result<Option<TileRustType>, JITError> {
        let Some((element_ty, shape)) =
            self.global_receiver_element_and_shape(&method_call.receiver)?
        else {
            return Ok(None);
        };
        let tile_ty = global_tile_syn_type(&element_ty, &shape)?;
        let return_syn_ty = match method_call.method.to_string().as_str() {
            "load" | "atomic_add" => {
                for (idx, arg) in method_call.args.iter().enumerate() {
                    if method_call.method == "atomic_add" && idx == 0 {
                        let expected = self.compiler.compile_type(
                            &tile_ty,
                            self.generic_vars,
                            &HashMap::new(),
                        )?;
                        let _ = self.infer_expr(arg, expected)?;
                    } else {
                        let _ = self.infer_expr(arg, None)?;
                    }
                }
                syn::parse_quote!((#tile_ty, Token))
            }
            "store" => {
                for (idx, arg) in method_call.args.iter().enumerate() {
                    if idx == 0 {
                        let expected = self.compiler.compile_type(
                            &tile_ty,
                            self.generic_vars,
                            &HashMap::new(),
                        )?;
                        let _ = self.infer_expr(arg, expected)?;
                    } else {
                        let _ = self.infer_expr(arg, None)?;
                    }
                }
                syn::parse_quote!(Token)
            }
            _ => return Ok(None),
        };
        self.results.insert_resolved_expr_type(
            &Expr::MethodCall(method_call.clone()),
            ResolvedType::from_syn_type(&return_syn_ty),
        );
        self.compiler
            .compile_type(&return_syn_ty, self.generic_vars, &HashMap::new())
    }

    fn global_receiver_element_and_shape(
        &self,
        receiver: &Expr,
    ) -> Result<Option<(Type, Vec<i32>)>, JITError> {
        let Expr::Path(path) = receiver else {
            return Ok(None);
        };
        if path.path.segments.len() != 1 {
            return Ok(None);
        }
        let res = self
            .compiler
            .modules
            .name_resolver
            .resolve_path(&path.path, &self.compiler.module_name);
        let Res::Def(DefKind::Static, def_id) = res else {
            return Ok(None);
        };
        let Some(static_item) = self.compiler.modules.name_resolver.get_static(&def_id) else {
            return Ok(None);
        };
        let ty = self
            .compiler
            .modules
            .normalize_type_aliases_in(&static_item.ty, Some(&def_id.module))?;
        let Some(type_name) = get_type_ident(&ty) else {
            return Ok(None);
        };
        if type_name != "Global" {
            return Ok(None);
        }
        let Some(element_ty) = global_element_type(&ty) else {
            return Ok(None);
        };
        let element_ty = self
            .compiler
            .global_atomic_value_type(&element_ty, &def_id.module)?;
        let Some(shape) = global_shape(&ty) else {
            return Ok(None);
        };
        Ok(Some((element_ty, shape)))
    }

    fn infer_call_arg_types<'expr>(
        &mut self,
        args: impl Iterator<Item = &'expr Expr>,
    ) -> Result<Vec<TileRustType>, JITError> {
        let mut arg_types = Vec::new();
        for arg in args {
            let Some(arg_ty) = self.infer_expr(arg, None)? else {
                return Ok(Vec::new());
            };
            arg_types.push(arg_ty);
        }
        Ok(arg_types)
    }

    fn infer_function_call_arg_types(
        &mut self,
        fn_item: &ItemFn,
        call: &ExprCall,
        expected_return: Option<&TileRustType>,
    ) -> Result<Vec<TileRustType>, JITError> {
        let (param_types, _) = get_sig_types(&fn_item.sig, None);
        let expected_arg_types =
            self.expected_function_arg_types(fn_item, call, expected_return)?;
        let mut arg_types = Vec::new();
        for (idx, arg) in call.args.iter().enumerate() {
            let expected = expected_arg_types.get(idx).cloned().flatten().or_else(|| {
                param_types
                    .get(idx)
                    .and_then(concrete_scalar_param_type)
                    .and_then(|param_type| {
                        self.compiler
                            .compile_type(param_type, self.generic_vars, &HashMap::new())
                            .ok()
                            .flatten()
                    })
            });
            let Some(arg_ty) = self.infer_expr(arg, expected)? else {
                return Ok(Vec::new());
            };
            arg_types.push(arg_ty);
        }
        Ok(arg_types)
    }

    fn expected_function_arg_types(
        &self,
        fn_item: &ItemFn,
        call: &ExprCall,
        expected_return: Option<&TileRustType>,
    ) -> Result<Vec<Option<TileRustType>>, JITError> {
        let (param_types, return_type) = get_sig_types(&fn_item.sig, None);
        let mut generic_arg_inf = GenericArgInference::new_function(fn_item.sig.clone());
        if let Some(expected_return) = expected_return {
            // Speculative seed from the expected type: a conflict just means
            // the hint doesn't apply, the strict check runs at inline time.
            let _ = generic_arg_inf.add_type_constraints(&return_type, &expected_return.rust_ty);
        }
        generic_arg_inf.apply_provided_generics_fn_call(call, self.generic_vars);

        let mut expected_arg_types = Vec::new();
        for param_type in param_types {
            let expected = substitute_resolved_generics(&param_type, &generic_arg_inf)
                .as_ref()
                .and_then(|ty| self.compile_expected_param_type(ty))
                .or_else(|| self.compile_expected_param_type(&param_type));
            expected_arg_types.push(expected);
        }
        Ok(expected_arg_types)
    }

    fn compile_expected_param_type(&self, ty: &Type) -> Option<TileRustType> {
        if type_has_impl_trait(ty)
            || !type_is_fully_known(self.compiler, ty, self.generic_vars)
            || !type_is_resolvable(self.compiler, ty, self.generic_vars)
        {
            return None;
        }
        self.compiler
            .compile_type(ty, self.generic_vars, &HashMap::new())
            .ok()
            .flatten()
    }

    fn infer_global_const_type(
        &self,
        path: &syn::ExprPath,
    ) -> Result<Option<TileRustType>, JITError> {
        let res = self
            .compiler
            .modules
            .name_resolver
            .resolve_path(&path.path, &self.compiler.module_name);
        let Res::Def(DefKind::Const, def_id) = res else {
            return Ok(None);
        };
        let Some(const_item) = self.compiler.modules.name_resolver.get_const(&def_id) else {
            return Ok(None);
        };
        self.compiler
            .compile_type(&const_item.ty, self.generic_vars, &HashMap::new())
    }

    fn infer_global_static_type(
        &self,
        path: &syn::ExprPath,
    ) -> Result<Option<TileRustType>, JITError> {
        let res = self
            .compiler
            .modules
            .name_resolver
            .resolve_path(&path.path, &self.compiler.module_name);
        let Res::Def(DefKind::Static, def_id) = res else {
            return Ok(None);
        };
        let Some(static_item) = self.compiler.modules.name_resolver.get_static(&def_id) else {
            return Ok(None);
        };
        self.compiler
            .compile_type(&static_item.ty, self.generic_vars, &HashMap::new())
    }

    fn infer_associated_const_type(
        &self,
        path: &syn::ExprPath,
    ) -> Result<Option<TileRustType>, JITError> {
        let candidates = self.associated_const_syn_type_candidates(path);
        let Some(ty) = single_syn_type(candidates) else {
            return Ok(None);
        };
        self.compiler
            .compile_type(&ty, self.generic_vars, &HashMap::new())
    }

    fn associated_const_syn_type_candidates(&self, path: &syn::ExprPath) -> Vec<Type> {
        if let Some(qself) = &path.qself {
            let Some(const_name) = path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
            else {
                return Vec::new();
            };
            let Some(trait_name) = qself
                .position
                .checked_sub(1)
                .and_then(|idx| path.path.segments.iter().nth(idx))
                .map(|segment| segment.ident.to_string())
            else {
                return Vec::new();
            };
            return self
                .trait_associated_const_syn_type(&trait_name, &const_name, &qself.ty)
                .into_iter()
                .collect();
        }

        if path.path.segments.len() != 2 {
            return Vec::new();
        }
        let qualifier = &path.path.segments[0];
        let qualifier_name = qualifier.ident.to_string();
        let const_name = path.path.segments[1].ident.to_string();
        let self_ty = type_from_path_segment(qualifier);

        let mut candidates = Vec::new();
        if let Some(bounds) = self.generic_type_bounds.get(&qualifier_name) {
            for trait_name in bounds {
                if let Some(ty) =
                    self.trait_associated_const_syn_type(trait_name, &const_name, &self_ty)
                {
                    candidates.push(ty);
                }
            }
        }

        for (trait_name, self_name) in self.compiler.modules.trait_impls().keys() {
            if self_name == &qualifier_name {
                if let Some(ty) =
                    self.trait_associated_const_syn_type(trait_name, &const_name, &self_ty)
                {
                    candidates.push(ty);
                }
            }
        }
        for (trait_name, self_name) in self.compiler.modules.primitives().keys() {
            if self_name == &qualifier_name {
                if let Some(ty) =
                    self.trait_associated_const_syn_type(trait_name, &const_name, &self_ty)
                {
                    candidates.push(ty);
                }
            }
        }
        candidates
    }

    fn trait_associated_const_syn_type(
        &self,
        trait_name: &str,
        const_name: &str,
        self_ty: &Type,
    ) -> Option<Type> {
        let item_trait = self.compiler.modules.traits().get(trait_name)?;
        item_trait.items.iter().find_map(|item| {
            let TraitItem::Const(item_const) = item else {
                return None;
            };
            if item_const.ident != const_name {
                return None;
            }
            Some(substitute_self_type(&item_const.ty, self_ty))
        })
    }

    fn method_call_arg_rust_types_with_receiver(
        &mut self,
        method_call: &ExprMethodCall,
        receiver: Type,
    ) -> Result<Option<Vec<Type>>, JITError> {
        let expected_arg_types = self.unique_inherent_method_arg_types(&receiver, method_call);
        let mut arg_types = vec![receiver];
        for (idx, arg) in method_call.args.iter().enumerate() {
            let expected = expected_arg_types
                .as_ref()
                .and_then(|arg_types| arg_types.get(idx))
                .and_then(|param_type| self.compile_expected_param_type(param_type));
            let Some(arg_ty) = self.infer_expr_syn_type(arg, expected)? else {
                return Ok(None);
            };
            arg_types.push(arg_ty);
        }
        Ok(Some(arg_types))
    }

    fn infer_method_call_selection(
        &mut self,
        method_call: &ExprMethodCall,
    ) -> Result<Option<MethodSelection>, JITError> {
        let Some(receiver) = self.infer_expr_syn_type(&method_call.receiver, None)? else {
            let _ = self.infer_call_arg_types(method_call.args.iter())?;
            return Ok(None);
        };

        let allow_mut_borrow = self.receiver_allows_mut_borrow(&method_call.receiver, &receiver);
        for receiver in receiver_adjustment_candidates(&receiver, allow_mut_borrow) {
            let mut probe = self.fork();
            let Some(call_arg_types) =
                probe.method_call_arg_rust_types_with_receiver(method_call, receiver)?
            else {
                continue;
            };
            let Some(selection) = probe.select_method_call(method_call, &call_arg_types)? else {
                continue;
            };
            self.commit_typeck_probe(probe);
            return Ok(Some(selection));
        }

        let _ = self.infer_call_arg_types(method_call.args.iter())?;
        Ok(None)
    }

    fn receiver_allows_mut_borrow(&self, receiver_expr: &Expr, receiver_ty: &Type) -> bool {
        if type_is_mut_reference(receiver_ty) {
            return true;
        }
        let Expr::Path(path) = receiver_expr else {
            return false;
        };
        if path.path.segments.len() != 1 {
            return false;
        }
        let name = get_ident_from_path_expr(path).to_string();
        self.mutable_vars.contains(&name)
    }

    fn commit_typeck_probe(&mut self, probe: TypeInferenceCx<'_, '_>) {
        self.syn_vars = probe.syn_vars;
        self.mutable_vars = probe.mutable_vars;
        self.vars = probe.vars;
        self.local_terms = probe.local_terms;
        self.inference = probe.inference;
        self.results = probe.results;
    }

    fn unique_inherent_method_arg_types(
        &self,
        receiver_ty: &Type,
        method_call: &ExprMethodCall,
    ) -> Option<Vec<Type>> {
        let receiver_type = get_type_ident(receiver_ty)?.to_string();
        let method_name = method_call.method.to_string();
        let mut matches = Vec::new();

        // Same-named methods defined per rank (e.g. `BoundedPartition::load`
        // for rank 2 and rank 3) stay unique after filtering impls whose
        // const-generic array rank cannot match the receiver.
        let receiver_lookup_ty = crate::compiler::_module::instantiate_type_for_lookup(
            receiver_ty,
            self.generic_vars,
            self.compiler.modules.primitives(),
        );
        for (_module_name, item_impl) in self
            .compiler
            .modules
            .struct_impls()
            .get(&receiver_type)?
            .iter()
        {
            if !crate::compiler::_module::impl_self_type_rank_matches(
                item_impl,
                &receiver_lookup_ty,
                self.generic_vars,
            ) {
                continue;
            }
            for item in &item_impl.items {
                let syn::ImplItem::Fn(method) = item else {
                    continue;
                };
                if method.sig.ident == method_name {
                    matches.push((item_impl, method));
                }
            }
        }

        let [(item_impl, method)] = matches.as_slice() else {
            return None;
        };
        let self_ty = &*item_impl.self_ty;
        let (param_types, _return_type) = get_sig_types(&method.sig, Some(self_ty));
        let mut generic_arg_inf = GenericArgInference::new_method(item_impl, method);
        // Probing for a unique method match: constraint conflicts mean no match.
        let _ = generic_arg_inf.add_type_constraints(self_ty, receiver_ty);
        let receiver_rank = rank_from_type_shape_arg(receiver_ty, self.generic_vars);
        Some(
            param_types
                .into_iter()
                .skip(1)
                .map(|param_type| {
                    let resolved = substitute_resolved_generics(&param_type, &generic_arg_inf)
                        .unwrap_or(param_type);
                    receiver_rank
                        .map(|rank| replace_array_len_with_rank(resolved.clone(), rank))
                        .unwrap_or(resolved)
                })
                .collect(),
        )
    }

    fn select_method_call(
        &mut self,
        method_call: &ExprMethodCall,
        call_arg_rust_tys: &[Type],
    ) -> Result<Option<MethodSelection>, JITError> {
        if call_arg_rust_tys.is_empty() {
            return Ok(None);
        }
        let receiver_ty = &call_arg_rust_tys[0];
        let Some((module_name, impl_item, impl_method)) = self.compiler.modules.get_impl_item_fn(
            receiver_ty,
            method_call,
            self.generic_vars,
            call_arg_rust_tys,
        )?
        else {
            return Ok(None);
        };
        let self_ty = &*impl_item.self_ty;
        let call_generic_vars = infer_method_generics(
            &impl_item,
            &impl_method,
            method_call,
            call_arg_rust_tys,
            self_ty,
            self.generic_vars,
            self.compiler.modules.primitives(),
        )?;
        let return_type = self.infer_method_return_type(
            &impl_item,
            &impl_method,
            call_arg_rust_tys,
            self_ty,
            &call_generic_vars,
            method_call,
        )?;
        Ok(Some(MethodSelection {
            module_name,
            impl_item,
            impl_method,
            generic_vars: call_generic_vars,
            return_type,
        }))
    }

    fn infer_method_return_type(
        &mut self,
        impl_item: &ItemImpl,
        impl_method: &ImplItemFn,
        call_arg_rust_tys: &[Type],
        self_ty: &Type,
        _call_generic_vars: &GenericVars,
        method_call: &ExprMethodCall,
    ) -> Result<Option<TileRustType>, JITError> {
        let (arg_types, return_type) = get_sig_types(&impl_method.sig, Some(self_ty));
        if arg_types.iter().any(type_has_impl_trait) || type_has_impl_trait(&return_type) {
            return Ok(None);
        }
        let mut generic_arg_inf = GenericArgInference::new_method(impl_item, impl_method);
        generic_arg_inf.map_args_to_params(
            &call_arg_rust_tys.to_vec(),
            Some(self_ty),
            self.generic_vars,
        )?;
        generic_arg_inf.apply_provided_generics_method_call(method_call, self.generic_vars);
        if !generic_arg_inf.verify() {
            return Ok(None);
        }
        let call_output_type = generic_arg_inf.infer_type(&return_type, self.generic_vars);
        if !type_is_resolvable(self.compiler, &call_output_type, self.generic_vars) {
            self.results.insert_resolved_expr_type(
                &Expr::MethodCall(method_call.clone()),
                ResolvedType::from_syn_type(&call_output_type),
            );
            return Ok(None);
        }
        match self
            .compiler
            .compile_type(&call_output_type, self.generic_vars, &HashMap::new())
        {
            Ok(Some(ty)) => Ok(Some(ty)),
            Ok(None) => {
                self.results.insert_resolved_expr_type(
                    &Expr::MethodCall(method_call.clone()),
                    ResolvedType::from_syn_type(&call_output_type),
                );
                Ok(None)
            }
            Err(_) => {
                self.results.insert_resolved_expr_type(
                    &Expr::MethodCall(method_call.clone()),
                    ResolvedType::from_syn_type(&call_output_type),
                );
                Ok(None)
            }
        }
    }

    fn infer_function_return(
        &mut self,
        fn_item: &ItemFn,
        call: &ExprCall,
        arg_types: &[TileRustType],
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        if arg_types.is_empty() && !call.args.is_empty() {
            return Ok(expected);
        }
        let call_arg_rust_tys = arg_types
            .iter()
            .map(|arg| arg.rust_ty.clone())
            .collect::<Vec<_>>();
        let (fn_arg_types, return_type) = get_sig_types(&fn_item.sig, None);
        if fn_arg_types.iter().any(type_has_impl_trait) || type_has_impl_trait(&return_type) {
            return Ok(expected);
        }
        let mut generic_arg_inf = GenericArgInference::new_function(fn_item.sig.clone());
        generic_arg_inf.map_args_to_params(&call_arg_rust_tys, None, self.generic_vars)?;
        generic_arg_inf.apply_provided_generics_fn_call(call, self.generic_vars);
        if !generic_arg_inf.verify() {
            return Ok(expected);
        }
        let call_output_type = generic_arg_inf.infer_type(&return_type, self.generic_vars);
        if !type_is_resolvable(self.compiler, &call_output_type, self.generic_vars) {
            return Ok(expected);
        }
        let mut type_params = expected
            .as_ref()
            .map(type_params_by_name)
            .unwrap_or_default();
        let param_names = get_sig_param_names(&fn_item.sig);
        let mut arg_string_values: HashMap<String, String> = HashMap::new();
        let mut arg_zst_values: HashMap<String, String> = HashMap::new();

        for (idx, param_name) in param_names.iter().enumerate() {
            let Some(arg_expr) = call.args.get(idx) else {
                continue;
            };
            if let Some(value) = shared_utils::zst_type_name(arg_expr) {
                arg_zst_values.insert(param_name.clone(), value);
            }
            if let Expr::Lit(lit_expr) = arg_expr {
                if let syn::Lit::Str(s) = &lit_expr.lit {
                    arg_string_values.insert(param_name.clone(), s.value());
                }
            } else if param_name == "padding_value" {
                if let Some(value) = shared_utils::padding_zst_value(arg_expr) {
                    arg_string_values.insert(param_name.clone(), value);
                }
            }
        }

        if let Some(op_attrs) = self
            .compiler
            .modules
            .get_cuda_tile_op_attrs(fn_item.sig.ident.to_string().as_str())
        {
            if let Some(output_type_params) = op_attrs.parse_string_arr("output_type_params") {
                let arg_types_by_name = param_names
                    .iter()
                    .zip(arg_types.iter())
                    .map(|(name, ty)| (name.clone(), ty.clone()))
                    .collect::<HashMap<_, _>>();
                for type_param_name in output_type_params {
                    if should_skip_optional_output_type_param(&type_param_name, &arg_zst_values) {
                        continue;
                    }
                    if let Some(arg_type) = arg_types_by_name.get(&type_param_name) {
                        let cuda_tile_type_str = arg_type.get_cuda_tile_type_str();
                        let mut type_param = TypeParam::derive_param_from_type(
                            type_param_name.clone(),
                            arg_type.rust_ty.clone(),
                            cuda_tile_type_str,
                            Some(arg_type.type_instance.clone()),
                        );
                        if let TypeParam::Padding(ref mut padding) = type_param {
                            padding.padding_value =
                                arg_string_values.get(&type_param_name).cloned();
                        }
                        type_params.insert(type_param_name.clone(), type_param);
                    }
                }
            }
        }

        match self
            .compiler
            .compile_type(&call_output_type, self.generic_vars, &type_params)
        {
            Ok(Some(ty)) => Ok(Some(ty)),
            Ok(None) => {
                self.results.insert_resolved_expr_type(
                    &Expr::Call(call.clone()),
                    ResolvedType::from_syn_type(&call_output_type),
                );
                Ok(expected)
            }
            Err(_) => {
                self.results.insert_resolved_expr_type(
                    &Expr::Call(call.clone()),
                    ResolvedType::from_syn_type(&call_output_type),
                );
                Ok(expected)
            }
        }
    }
}

fn local_pattern_type(pat: &Pat) -> Option<&Type> {
    match pat {
        Pat::Type(pat_type) => Some(pat_type.ty.as_ref()),
        _ => None,
    }
}

fn generic_type_bounds(generics: &Generics) -> HashMap<String, Vec<String>> {
    let mut bounds: HashMap<String, Vec<String>> = HashMap::new();

    for param in &generics.params {
        let syn::GenericParam::Type(type_param) = param else {
            continue;
        };
        let type_name = type_param.ident.to_string();
        for bound in &type_param.bounds {
            if let Some(trait_name) = trait_bound_name(bound) {
                insert_unique_bound(&mut bounds, &type_name, trait_name);
            }
        }
    }

    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            let WherePredicate::Type(predicate_ty) = predicate else {
                continue;
            };
            let Some(type_name) = type_path_ident(&predicate_ty.bounded_ty) else {
                continue;
            };
            for bound in &predicate_ty.bounds {
                if let Some(trait_name) = trait_bound_name(bound) {
                    insert_unique_bound(&mut bounds, &type_name, trait_name);
                }
            }
        }
    }

    bounds
}

fn merge_generic_type_bounds(
    bounds: &mut HashMap<String, Vec<String>>,
    extra: HashMap<String, Vec<String>>,
) {
    for (type_name, trait_names) in extra {
        for trait_name in trait_names {
            insert_unique_bound(bounds, &type_name, trait_name);
        }
    }
}

fn trait_bound_name(bound: &TypeParamBound) -> Option<String> {
    match bound {
        TypeParamBound::Trait(trait_bound) => trait_bound
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        _ => None,
    }
}

fn insert_unique_bound(
    bounds: &mut HashMap<String, Vec<String>>,
    type_name: &str,
    trait_name: String,
) {
    let entry = bounds.entry(type_name.to_string()).or_default();
    if !entry.iter().any(|existing| existing == &trait_name) {
        entry.push(trait_name);
    }
}

fn type_from_path_segment(segment: &syn::PathSegment) -> Type {
    let segment = segment.clone();
    syn::parse_quote!(#segment)
}

fn single_syn_type(types: Vec<Type>) -> Option<Type> {
    match types.as_slice() {
        [ty] => Some(ty.clone()),
        _ => None,
    }
}

fn global_tile_syn_type(element_ty: &Type, shape: &[i32]) -> Result<Type, JITError> {
    let elem = element_ty.to_token_stream().to_string();
    let shape = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let ty = if shape.is_empty() {
        format!("Tile<{elem}, {{ [] }}>")
    } else {
        format!("Tile<{elem}, {{ [{shape}] }}>")
    };
    syn::parse_str::<Type>(&ty)
        .map_err(|err| JITError::Generic(format!("failed to parse Global tile type `{ty}`: {err}")))
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
        GenericArgument::Const(expr) => global_const_shape_expr(expr),
        _ => None,
    })
}

fn global_const_shape_expr(expr: &Expr) -> Option<Vec<i32>> {
    match expr {
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [Stmt::Expr(expr, _)] => global_const_shape_expr(expr),
            _ => None,
        },
        Expr::Array(array) => array.elems.iter().map(global_expr_i32).collect(),
        Expr::Repeat(repeat) => {
            let value = global_expr_i32(&repeat.expr)?;
            let len = global_expr_i32(&repeat.len)? as usize;
            Some(vec![value; len])
        }
        Expr::Paren(paren) => global_const_shape_expr(&paren.expr),
        _ => None,
    }
}

fn global_expr_i32(expr: &Expr) -> Option<i32> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) => value.base10_parse::<i32>().ok(),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            global_expr_i32(&unary.expr).map(|value| -value)
        }
        Expr::Paren(paren) => global_expr_i32(&paren.expr),
        _ => None,
    }
}

fn substitute_self_type(ty: &Type, self_ty: &Type) -> Type {
    match ty {
        Type::Path(path)
            if path.qself.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].ident == "Self" =>
        {
            self_ty.clone()
        }
        Type::Reference(reference) => {
            let mut reference = reference.clone();
            *reference.elem = substitute_self_type(&reference.elem, self_ty);
            Type::Reference(reference)
        }
        Type::Ptr(ptr) => {
            let mut ptr = ptr.clone();
            *ptr.elem = substitute_self_type(&ptr.elem, self_ty);
            Type::Ptr(ptr)
        }
        Type::Tuple(tuple) => {
            let mut tuple = tuple.clone();
            tuple.elems = tuple
                .elems
                .iter()
                .map(|elem| substitute_self_type(elem, self_ty))
                .collect();
            Type::Tuple(tuple)
        }
        Type::Array(array) => {
            let mut array = array.clone();
            *array.elem = substitute_self_type(&array.elem, self_ty);
            Type::Array(array)
        }
        Type::Slice(slice) => {
            let mut slice = slice.clone();
            *slice.elem = substitute_self_type(&slice.elem, self_ty);
            Type::Slice(slice)
        }
        Type::Paren(paren) => {
            let mut paren = paren.clone();
            *paren.elem = substitute_self_type(&paren.elem, self_ty);
            Type::Paren(paren)
        }
        Type::Group(group) => {
            let mut group = group.clone();
            *group.elem = substitute_self_type(&group.elem, self_ty);
            Type::Group(group)
        }
        _ => ty.clone(),
    }
}

fn receiver_adjustment_candidates(receiver_ty: &Type, allow_mut_borrow: bool) -> Vec<Type> {
    let mut candidates = Vec::new();
    let mut seen = HashSet::new();

    for base in autoderef_type_chain(receiver_ty) {
        push_unique_type(&mut candidates, &mut seen, base.clone());
        if !matches!(base, Type::Reference(_)) {
            let shared: Type = syn::parse_quote!(&#base);
            push_unique_type(&mut candidates, &mut seen, shared);
            if allow_mut_borrow {
                let mutable: Type = syn::parse_quote!(&mut #base);
                push_unique_type(&mut candidates, &mut seen, mutable);
            }
        }
    }

    candidates
}

fn autoderef_type_chain(ty: &Type) -> Vec<Type> {
    let mut chain = Vec::new();
    let mut current = ty.clone();
    loop {
        chain.push(current.clone());
        current = match current {
            Type::Reference(reference) => (*reference.elem).clone(),
            Type::Paren(paren) => (*paren.elem).clone(),
            Type::Group(group) => (*group.elem).clone(),
            _ => break,
        };
    }
    chain
}

fn push_unique_type(candidates: &mut Vec<Type>, seen: &mut HashSet<String>, ty: Type) {
    if seen.insert(ty.to_token_stream().to_string()) {
        candidates.push(ty);
    }
}

fn type_is_mut_reference(ty: &Type) -> bool {
    match ty {
        Type::Reference(reference) => reference.mutability.is_some(),
        Type::Paren(paren) => type_is_mut_reference(&paren.elem),
        Type::Group(group) => type_is_mut_reference(&group.elem),
        _ => false,
    }
}

fn local_binding_name(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(ident) => Some(ident.ident.to_string()),
        Pat::Type(pat_type) => local_binding_name(&pat_type.pat),
        _ => None,
    }
}

fn tuple_element_syn_types(ty: &Type) -> Option<Vec<Type>> {
    match ty {
        Type::Tuple(tuple) => Some(tuple.elems.iter().cloned().collect()),
        Type::Paren(paren) => tuple_element_syn_types(&paren.elem),
        _ => None,
    }
}

fn array_or_slice_element_syn_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Array(array) => Some((*array.elem).clone()),
        Type::Slice(slice) => Some((*slice.elem).clone()),
        Type::Reference(reference) => array_or_slice_element_syn_type(&reference.elem),
        Type::Paren(paren) => array_or_slice_element_syn_type(&paren.elem),
        _ => None,
    }
}

fn option_payload_syn_type(ty: &Type) -> Option<Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut payload_types = args.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty.clone()),
        _ => None,
    });
    let payload_ty = payload_types.next()?;
    if payload_types.next().is_some() {
        return None;
    }
    Some(payload_ty)
}

fn repeat_length_value(len: &Expr, generic_vars: &GenericVars) -> Option<usize> {
    match len {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(raw) => raw.base10_parse::<usize>().ok(),
            _ => None,
        },
        Expr::Path(path) if path.path.segments.len() == 1 => {
            let ident = path.path.segments[0].ident.to_string();
            generic_vars.get_i32(&ident).and_then(|value| {
                if value >= 0 {
                    Some(value as usize)
                } else {
                    None
                }
            })
        }
        Expr::Paren(paren) => repeat_length_value(&paren.expr, generic_vars),
        _ => None,
    }
}

fn type_path_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) if path.qself.is_none() => path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        Type::Paren(paren) => type_path_ident(&paren.elem),
        Type::Reference(reference) => type_path_ident(&reference.elem),
        _ => None,
    }
}

fn stmt_is_return(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(Expr::Return(_), _))
}

fn stmt_diverges(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_diverges(expr),
        Stmt::Local(_) | Stmt::Item(_) | Stmt::Macro(_) => false,
    }
}

fn block_diverges(block: &syn::Block) -> bool {
    block.stmts.iter().any(stmt_diverges)
}

fn expr_diverges(expr: &Expr) -> bool {
    match expr {
        Expr::Return(_) => true,
        Expr::Block(block) => block_diverges(&block.block),
        Expr::Unsafe(unsafe_expr) => block_diverges(&unsafe_expr.block),
        Expr::Paren(paren) => expr_diverges(&paren.expr),
        Expr::If(if_expr) => {
            let Some((_, else_expr)) = &if_expr.else_branch else {
                return false;
            };
            block_diverges(&if_expr.then_branch) && expr_diverges(else_expr)
        }
        _ => false,
    }
}

fn type_params_by_name(ty: &TileRustType) -> HashMap<String, TypeParam> {
    ty.params
        .iter()
        .filter_map(|param| param.name().map(|name| (name.to_string(), param.clone())))
        .collect()
}

fn concrete_scalar_param_type(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    ResolvedScalarType::from_ident(&segment.ident.to_string()).map(|_| ty)
}

fn is_unsuffixed_numeric_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(lit) => lit.suffix().is_empty(),
            Lit::Float(lit) => lit.suffix().is_empty(),
            _ => false,
        },
        Expr::Paren(paren) => is_unsuffixed_numeric_literal(&paren.expr),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            is_unsuffixed_numeric_literal(&unary.expr)
        }
        _ => false,
    }
}

fn literal_kind_accepts_type(kind: InferVarKind, ty: &Type) -> bool {
    match kind {
        InferVarKind::General => true,
        InferVarKind::Int => scalar_type_name(ty)
            .as_deref()
            .is_some_and(is_integer_scalar_name),
        InferVarKind::Float => scalar_type_name(ty)
            .as_deref()
            .is_some_and(is_float_scalar_name),
    }
}

fn merge_infer_var_kinds(lhs: InferVarKind, rhs: InferVarKind) -> Option<InferVarKind> {
    match (lhs, rhs) {
        (InferVarKind::General, kind) | (kind, InferVarKind::General) => Some(kind),
        (lhs, rhs) if lhs == rhs => Some(lhs),
        _ => None,
    }
}

fn binary_result_matches_operands(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::Add(_)
            | syn::BinOp::Sub(_)
            | syn::BinOp::Mul(_)
            | syn::BinOp::Div(_)
            | syn::BinOp::Rem(_)
            | syn::BinOp::BitAnd(_)
            | syn::BinOp::BitOr(_)
            | syn::BinOp::BitXor(_)
    )
}

fn binary_result_is_bool(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::Eq(_)
            | syn::BinOp::Ne(_)
            | syn::BinOp::Lt(_)
            | syn::BinOp::Le(_)
            | syn::BinOp::Gt(_)
            | syn::BinOp::Ge(_)
            | syn::BinOp::And(_)
            | syn::BinOp::Or(_)
    )
}

fn binary_operands_are_bool(op: &syn::BinOp) -> bool {
    matches!(op, syn::BinOp::And(_) | syn::BinOp::Or(_))
}

fn closure_signature_for_param(sig: &syn::Signature, param_ty: &Type) -> Option<ClosureSignature> {
    let param_name = bare_type_path_name(param_ty)?;
    for generic_param in &sig.generics.params {
        let syn::GenericParam::Type(type_param) = generic_param else {
            continue;
        };
        if type_param.ident == param_name {
            if let Some(signature) = closure_signature_from_bounds(&type_param.bounds) {
                return Some(signature);
            }
        }
    }
    let where_clause = sig.generics.where_clause.as_ref()?;
    for predicate in &where_clause.predicates {
        let syn::WherePredicate::Type(predicate) = predicate else {
            continue;
        };
        if bare_type_path_name(&predicate.bounded_ty).as_deref() == Some(param_name.as_str()) {
            if let Some(signature) = closure_signature_from_bounds(&predicate.bounds) {
                return Some(signature);
            }
        }
    }
    None
}

fn closure_signature_from_bounds(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
) -> Option<ClosureSignature> {
    for bound in bounds {
        let syn::TypeParamBound::Trait(trait_bound) = bound else {
            continue;
        };
        let segment = trait_bound.path.segments.last()?;
        if !matches!(
            segment.ident.to_string().as_str(),
            "Fn" | "FnMut" | "FnOnce"
        ) {
            continue;
        }
        let syn::PathArguments::Parenthesized(args) = &segment.arguments else {
            continue;
        };
        let output = match &args.output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(*ty.clone()),
        };
        return Some(ClosureSignature {
            inputs: args.inputs.iter().cloned().collect(),
            output,
        });
    }
    None
}

fn closure_signature_generic_names<'a>(
    sig: &syn::Signature,
    signatures: impl Iterator<Item = &'a ClosureSignature>,
) -> HashSet<String> {
    let function_generics = get_supported_generic_params(&sig.generics)
        .into_iter()
        .map(|(name, _)| name)
        .collect::<HashSet<_>>();
    let mut names = HashSet::new();
    for signature in signatures {
        for input in &signature.inputs {
            collect_type_names(input, &function_generics, &mut names);
        }
        if let Some(output) = &signature.output {
            collect_type_names(output, &function_generics, &mut names);
        }
    }
    names
}

fn collect_type_names(ty: &Type, allowed: &HashSet<String>, names: &mut HashSet<String>) {
    match ty {
        Type::Path(path) => {
            if path.qself.is_some() {
                return;
            }
            for segment in &path.path.segments {
                let name = segment.ident.to_string();
                if allowed.contains(&name) {
                    names.insert(name);
                }
                if let PathArguments::AngleBracketed(args) = &segment.arguments {
                    for arg in &args.args {
                        match arg {
                            GenericArgument::Type(ty) => collect_type_names(ty, allowed, names),
                            GenericArgument::Const(expr) => {
                                collect_const_expr_names(expr, allowed, names)
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Type::Reference(reference) => collect_type_names(&reference.elem, allowed, names),
        Type::Ptr(ptr) => collect_type_names(&ptr.elem, allowed, names),
        Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                collect_type_names(elem, allowed, names);
            }
        }
        Type::Array(array) => {
            collect_type_names(&array.elem, allowed, names);
            collect_const_expr_names(&array.len, allowed, names);
        }
        Type::Slice(slice) => collect_type_names(&slice.elem, allowed, names),
        _ => {}
    }
}

fn collect_const_expr_names(expr: &Expr, allowed: &HashSet<String>, names: &mut HashSet<String>) {
    match expr {
        Expr::Path(path) if path.path.segments.len() == 1 => {
            let name = path.path.segments[0].ident.to_string();
            if allowed.contains(&name) {
                names.insert(name);
            }
        }
        Expr::Block(block) => {
            for stmt in &block.block.stmts {
                if let Stmt::Expr(expr, _) = stmt {
                    collect_const_expr_names(expr, allowed, names);
                }
            }
        }
        Expr::Array(array) => {
            for elem in &array.elems {
                collect_const_expr_names(elem, allowed, names);
            }
        }
        Expr::Repeat(repeat) => {
            collect_const_expr_names(&repeat.expr, allowed, names);
            collect_const_expr_names(&repeat.len, allowed, names);
        }
        Expr::Unary(unary) => collect_const_expr_names(&unary.expr, allowed, names),
        Expr::Paren(paren) => collect_const_expr_names(&paren.expr, allowed, names),
        _ => {}
    }
}

fn type_mentions_any_name(ty: &Type, names: &HashSet<String>) -> bool {
    if names.is_empty() {
        return false;
    }
    let mut seen = HashSet::new();
    collect_type_names(ty, names, &mut seen);
    !seen.is_empty()
}

fn is_dim_new_call_path(path: &syn::ExprPath) -> bool {
    path.path.segments.len() == 2
        && path.path.segments[0].ident == "Dim"
        && path.path.segments[1].ident == "new"
}

fn bare_type_path_name(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    Some(segment.ident.to_string())
}

fn scalar_type_name(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &path.path.segments[0];
    if !matches!(segment.arguments, PathArguments::None) {
        return None;
    }
    Some(segment.ident.to_string())
}

fn is_integer_scalar_name(name: &str) -> bool {
    matches!(
        name,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
    )
}

fn is_float_scalar_name(name: &str) -> bool {
    matches!(
        name,
        "f16" | "bf16" | "f32" | "f64" | "tf32" | "f8e4m3fn" | "f8e5m2"
    )
}

fn is_builtin_scalar_name(name: &str) -> bool {
    name == "bool" || is_integer_scalar_name(name) || is_float_scalar_name(name)
}

fn is_surface_only_scalar_type(ty: &Type) -> bool {
    matches!(scalar_type_name(ty).as_deref(), Some("isize" | "usize"))
}

fn is_likely_unresolved_type_param(name: &str) -> bool {
    name.chars()
        .next()
        .is_some_and(|ch| ch == '_' || ch.is_ascii_uppercase())
}

fn replace_array_len_with_rank(mut ty: Type, rank: usize) -> Type {
    match &mut ty {
        Type::Array(array) => {
            if matches!(array.len, Expr::Path(_)) {
                let rank_lit = rank;
                array.len = syn::parse_quote!(#rank_lit);
            }
        }
        Type::Reference(reference) => {
            *reference.elem = replace_array_len_with_rank(*reference.elem.clone(), rank);
        }
        Type::Tuple(tuple) => {
            for elem in &mut tuple.elems {
                *elem = replace_array_len_with_rank(elem.clone(), rank);
            }
        }
        _ => {}
    }
    ty
}

fn rank_from_type_shape_arg(ty: &Type, generic_vars: &GenericVars) -> Option<usize> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    for arg in &args.args {
        match arg {
            GenericArgument::Const(expr) => {
                if let Some(rank) = rank_from_const_shape_expr(expr, generic_vars) {
                    return Some(rank);
                }
            }
            GenericArgument::Type(Type::Path(path)) if path.qself.is_none() => {
                let ident = path.path.segments.last()?.ident.to_string();
                if let Some(shape) = generic_vars.inst_array.get(&ident) {
                    return Some(shape.len());
                }
            }
            _ => {}
        }
    }
    None
}

fn rank_from_const_shape_expr(expr: &Expr, generic_vars: &GenericVars) -> Option<usize> {
    match expr {
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [Stmt::Expr(inner, _)] => rank_from_const_shape_expr(inner, generic_vars),
            _ => None,
        },
        Expr::Array(array) => Some(array.elems.len()),
        Expr::Repeat(repeat) => match &*repeat.len {
            Expr::Lit(lit) => match &lit.lit {
                Lit::Int(raw) => raw.base10_parse::<usize>().ok(),
                _ => None,
            },
            Expr::Path(path) => {
                let ident = get_ident_from_path_expr(path).to_string();
                generic_vars.get_i32(&ident).map(|value| value as usize)
            }
            _ => None,
        },
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.inst_array.get(&ident).map(Vec::len)
        }
        _ => None,
    }
}

fn substitute_resolved_generics(ty: &Type, inference: &GenericArgInference) -> Option<Type> {
    match ty {
        Type::Path(path_ty) if path_ty.qself.is_none() => {
            if let Some((kind, target)) = resolved_bare_type_path_arg(ty, inference)? {
                return match kind {
                    GenericArgType::Type => syn::parse_str::<Type>(target).ok(),
                    GenericArgType::GenericConstExpr => None,
                };
            }

            let mut result = path_ty.clone();
            for segment in &mut result.path.segments {
                let PathArguments::AngleBracketed(args) = &mut segment.arguments else {
                    continue;
                };
                for arg in &mut args.args {
                    *arg = substitute_generic_argument(arg, inference)?;
                }
            }
            Some(Type::Path(result))
        }
        Type::Reference(reference) => {
            let mut result = reference.clone();
            *result.elem = substitute_resolved_generics(&reference.elem, inference)?;
            Some(Type::Reference(result))
        }
        Type::Ptr(ptr) => {
            let mut result = ptr.clone();
            *result.elem = substitute_resolved_generics(&ptr.elem, inference)?;
            Some(Type::Ptr(result))
        }
        Type::Tuple(tuple) => {
            let mut result = tuple.clone();
            for elem in &mut result.elems {
                *elem = substitute_resolved_generics(elem, inference)?;
            }
            Some(Type::Tuple(result))
        }
        Type::Array(array) => {
            let mut result = array.clone();
            *result.elem = substitute_resolved_generics(&array.elem, inference)?;
            result.len = substitute_resolved_const_expr(&array.len, inference)?;
            Some(Type::Array(result))
        }
        Type::Slice(slice) => {
            let mut result = slice.clone();
            *result.elem = substitute_resolved_generics(&slice.elem, inference)?;
            Some(Type::Slice(result))
        }
        Type::ImplTrait(_) => None,
        _ => Some(ty.clone()),
    }
}

fn substitute_generic_argument(
    arg: &GenericArgument,
    inference: &GenericArgInference,
) -> Option<GenericArgument> {
    match arg {
        GenericArgument::Type(ty) => {
            if let Some((_kind, target)) = resolved_bare_type_path_arg(ty, inference)? {
                return syn::parse_str::<GenericArgument>(target).ok();
            }
            Some(GenericArgument::Type(substitute_resolved_generics(
                ty, inference,
            )?))
        }
        GenericArgument::Const(expr) => Some(GenericArgument::Const(
            substitute_resolved_const_expr(expr, inference)?,
        )),
        GenericArgument::Lifetime(lifetime) => Some(GenericArgument::Lifetime(lifetime.clone())),
        _ => None,
    }
}

fn substitute_resolved_const_expr(expr: &Expr, inference: &GenericArgInference) -> Option<Expr> {
    match expr {
        Expr::Path(path) if path.path.segments.len() == 1 => {
            let name = path.path.segments[0].ident.to_string();
            let Some(mapping) = inference.param2arg.get(&name) else {
                return Some(expr.clone());
            };
            let Some((_kind, target)) = mapping else {
                return None;
            };
            syn::parse_str::<Expr>(target).ok()
        }
        Expr::Block(block) => {
            let mut result = block.clone();
            let [Stmt::Expr(inner, semi)] = result.block.stmts.as_mut_slice() else {
                return Some(Expr::Block(result));
            };
            *inner = substitute_resolved_const_expr(inner, inference)?;
            *semi = None;
            Some(Expr::Block(result))
        }
        Expr::Array(array) => {
            let mut result = array.clone();
            for elem in &mut result.elems {
                *elem = substitute_resolved_const_expr(elem, inference)?;
            }
            Some(Expr::Array(result))
        }
        Expr::Repeat(repeat) => {
            let mut result = repeat.clone();
            *result.expr = substitute_resolved_const_expr(&repeat.expr, inference)?;
            *result.len = substitute_resolved_const_expr(&repeat.len, inference)?;
            Some(Expr::Repeat(result))
        }
        Expr::Unary(unary) => {
            let mut result = unary.clone();
            *result.expr = substitute_resolved_const_expr(&unary.expr, inference)?;
            Some(Expr::Unary(result))
        }
        Expr::Paren(paren) => {
            let mut result = paren.clone();
            *result.expr = substitute_resolved_const_expr(&paren.expr, inference)?;
            Some(Expr::Paren(result))
        }
        _ => Some(expr.clone()),
    }
}

fn resolved_bare_type_path_arg<'a>(
    ty: &Type,
    inference: &'a GenericArgInference,
) -> Option<Option<&'a (GenericArgType, String)>> {
    let Type::Path(path_ty) = ty else {
        return Some(None);
    };
    if path_ty.qself.is_some() || path_ty.path.segments.len() != 1 {
        return Some(None);
    }
    let segment = &path_ty.path.segments[0];
    if !matches!(segment.arguments, PathArguments::None) {
        return Some(None);
    }
    match inference.param2arg.get(&segment.ident.to_string()) {
        Some(Some(mapping)) => Some(Some(mapping)),
        Some(None) => None,
        None => Some(None),
    }
}

fn type_has_impl_trait(ty: &Type) -> bool {
    match ty {
        Type::ImplTrait(_) => true,
        Type::Reference(reference) => type_has_impl_trait(&reference.elem),
        Type::Tuple(tuple) => tuple.elems.iter().any(type_has_impl_trait),
        Type::Array(array) => type_has_impl_trait(&array.elem),
        Type::Ptr(ptr) => type_has_impl_trait(&ptr.elem),
        Type::Path(path) => path
            .path
            .segments
            .iter()
            .any(|segment| match &segment.arguments {
                PathArguments::AngleBracketed(args) => args.args.iter().any(|arg| match arg {
                    GenericArgument::Type(arg_ty) => type_has_impl_trait(arg_ty),
                    _ => false,
                }),
                _ => false,
            }),
        _ => false,
    }
}

fn can_add_type_constraint(param_ty: &Type, arg_ty: &Type) -> bool {
    if type_has_impl_trait(param_ty) {
        return false;
    }
    match (param_ty, arg_ty) {
        (Type::Path(param_path), Type::Path(arg_path)) => {
            if param_path.qself.is_some() || arg_path.qself.is_some() {
                return false;
            }
            let Some(param_segment) = param_path.path.segments.last() else {
                return false;
            };
            let Some(arg_segment) = arg_path.path.segments.last() else {
                return false;
            };
            match (&param_segment.arguments, &arg_segment.arguments) {
                (PathArguments::None, _) => true,
                (
                    PathArguments::AngleBracketed(param_args),
                    PathArguments::AngleBracketed(arg_args),
                ) => generic_arguments_are_compatible(param_args, arg_args),
                (PathArguments::AngleBracketed(_), _) => false,
                _ => true,
            }
        }
        (Type::Reference(param_ref), Type::Reference(arg_ref)) => {
            can_add_type_constraint(&param_ref.elem, &arg_ref.elem)
        }
        (Type::Ptr(param_ptr), Type::Ptr(arg_ptr)) => {
            can_add_type_constraint(&param_ptr.elem, &arg_ptr.elem)
        }
        (Type::Tuple(param_tuple), Type::Tuple(arg_tuple)) => {
            param_tuple.elems.len() == arg_tuple.elems.len()
                && param_tuple
                    .elems
                    .iter()
                    .zip(arg_tuple.elems.iter())
                    .all(|(param, arg)| can_add_type_constraint(param, arg))
        }
        (Type::Array(param_array), Type::Array(arg_array)) => {
            can_add_type_constraint(&param_array.elem, &arg_array.elem)
        }
        (Type::Slice(param_slice), Type::Slice(arg_slice)) => {
            can_add_type_constraint(&param_slice.elem, &arg_slice.elem)
        }
        (Type::ImplTrait(_), _) => false,
        _ => true,
    }
}

fn generic_arguments_are_compatible(
    param_args: &syn::AngleBracketedGenericArguments,
    arg_args: &syn::AngleBracketedGenericArguments,
) -> bool {
    if param_args.args.len() != arg_args.args.len() {
        return false;
    }
    param_args
        .args
        .iter()
        .zip(arg_args.args.iter())
        .all(|(param, arg)| match (param, arg) {
            (GenericArgument::Type(param_ty), GenericArgument::Type(arg_ty)) => {
                can_add_type_constraint(param_ty, arg_ty)
            }
            (GenericArgument::Const(_), GenericArgument::Const(_)) => true,
            (GenericArgument::Lifetime(_), GenericArgument::Lifetime(_)) => true,
            _ => false,
        })
}

fn type_is_resolvable(
    compiler: &CUDATileFunctionCompiler<'_>,
    ty: &Type,
    generic_vars: &GenericVars,
) -> bool {
    let normalized_ty = compiler
        .modules
        .normalize_type_aliases_in(ty, Some(&compiler.current_module()))
        .unwrap_or_else(|_| ty.clone());
    let ty = &normalized_ty;
    match ty {
        Type::Reference(reference) => type_is_resolvable(compiler, &reference.elem, generic_vars),
        Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .all(|elem| type_is_resolvable(compiler, elem, generic_vars)),
        Type::Array(array) => {
            type_is_resolvable(compiler, &array.elem, generic_vars)
                && const_expr_is_resolvable(&array.len, generic_vars)
        }
        Type::Ptr(ptr) => type_is_resolvable(compiler, &ptr.elem, generic_vars),
        Type::Path(path) => {
            if path.qself.is_some() {
                return false;
            }
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            match &segment.arguments {
                PathArguments::None => {
                    generic_vars.var_type(&ident).is_some()
                        || compiler.modules.structs().contains_key(&ident)
                        || compiler
                            .modules
                            .primitives()
                            .contains_key(&("ElementType".to_string(), ident.clone()))
                        || compiler
                            .modules
                            .primitives()
                            .contains_key(&("Scalar".to_string(), ident.clone()))
                        || is_builtin_scalar_name(&ident)
                }
                PathArguments::AngleBracketed(args) => args.args.iter().all(|arg| match arg {
                    GenericArgument::Type(arg_ty) => {
                        type_is_resolvable(compiler, arg_ty, generic_vars)
                    }
                    GenericArgument::Const(expr) => const_expr_is_resolvable(expr, generic_vars),
                    GenericArgument::Lifetime(_) => true,
                    _ => false,
                }),
                _ => false,
            }
        }
        _ => true,
    }
}

fn type_is_fully_known(
    compiler: &CUDATileFunctionCompiler<'_>,
    ty: &Type,
    generic_vars: &GenericVars,
) -> bool {
    let normalized_ty = compiler
        .modules
        .normalize_type_aliases_in(ty, Some(&compiler.current_module()))
        .unwrap_or_else(|_| ty.clone());
    let ty = &normalized_ty;
    match ty {
        Type::Reference(reference) => type_is_fully_known(compiler, &reference.elem, generic_vars),
        Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .all(|elem| type_is_fully_known(compiler, elem, generic_vars)),
        Type::Array(array) => {
            type_is_fully_known(compiler, &array.elem, generic_vars)
                && const_expr_is_fully_known(&array.len, generic_vars)
        }
        Type::Slice(slice) => type_is_fully_known(compiler, &slice.elem, generic_vars),
        Type::Ptr(ptr) => type_is_fully_known(compiler, &ptr.elem, generic_vars),
        Type::Path(path) => {
            if path.qself.is_some() {
                return false;
            }
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            if generic_vars.var_type(&ident).is_some() {
                return generic_vars.inst_types.contains_key(&ident)
                    || generic_vars.inst_i32.contains_key(&ident)
                    || generic_vars.inst_bool.contains_key(&ident)
                    || generic_vars.inst_array.contains_key(&ident)
                    || generic_vars.len2array.contains_key(&ident);
            }
            if is_likely_unresolved_type_param(&ident)
                && !generic_vars.inst_types.contains_key(&ident)
                && !compiler.modules.structs().contains_key(&ident)
            {
                return false;
            }
            let base_known = generic_vars.inst_types.contains_key(&ident)
                || compiler.modules.structs().contains_key(&ident)
                || compiler
                    .modules
                    .primitives()
                    .contains_key(&("ElementType".to_string(), ident.clone()))
                || compiler
                    .modules
                    .primitives()
                    .contains_key(&("Scalar".to_string(), ident.clone()))
                || is_builtin_scalar_name(&ident);
            if !base_known {
                return false;
            }
            match &segment.arguments {
                PathArguments::None => true,
                PathArguments::AngleBracketed(args) => args.args.iter().all(|arg| match arg {
                    GenericArgument::Type(arg_ty) => {
                        type_is_fully_known(compiler, arg_ty, generic_vars)
                    }
                    GenericArgument::Const(expr) => const_expr_is_fully_known(expr, generic_vars),
                    GenericArgument::Lifetime(_) => true,
                    _ => false,
                }),
                _ => false,
            }
        }
        Type::Never(_) => true,
        Type::Infer(_) | Type::ImplTrait(_) => false,
        _ => true,
    }
}

fn const_expr_is_resolvable(expr: &Expr, generic_vars: &GenericVars) -> bool {
    match expr {
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.var_type(&ident).is_some()
        }
        Expr::Index(index) => const_cga_index_is_resolvable(index, generic_vars),
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [Stmt::Expr(inner, _)] => const_expr_is_resolvable(inner, generic_vars),
            _ => false,
        },
        Expr::Array(array) => array
            .elems
            .iter()
            .all(|elem| const_array_elem_is_resolvable(elem, generic_vars)),
        Expr::Repeat(repeat) => {
            const_expr_is_resolvable(&repeat.expr, generic_vars)
                && const_expr_is_resolvable(&repeat.len, generic_vars)
        }
        Expr::Lit(_) => true,
        Expr::Unary(unary) => const_expr_is_resolvable(&unary.expr, generic_vars),
        Expr::Paren(paren) => const_expr_is_resolvable(&paren.expr, generic_vars),
        _ => false,
    }
}

fn const_expr_is_fully_known(expr: &Expr, generic_vars: &GenericVars) -> bool {
    match expr {
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.inst_i32.contains_key(&ident)
                || generic_vars.inst_bool.contains_key(&ident)
                || generic_vars.inst_array.contains_key(&ident)
                || generic_vars.len2array.contains_key(&ident)
        }
        Expr::Index(index) => const_cga_index_is_resolvable(index, generic_vars),
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [Stmt::Expr(inner, _)] => const_expr_is_fully_known(inner, generic_vars),
            _ => false,
        },
        Expr::Array(array) => array
            .elems
            .iter()
            .all(|elem| const_array_elem_is_fully_known(elem, generic_vars)),
        Expr::Repeat(repeat) => {
            const_expr_is_fully_known(&repeat.expr, generic_vars)
                && const_expr_is_fully_known(&repeat.len, generic_vars)
        }
        Expr::Lit(_) => true,
        Expr::Unary(unary) => const_expr_is_fully_known(&unary.expr, generic_vars),
        Expr::Paren(paren) => const_expr_is_fully_known(&paren.expr, generic_vars),
        _ => false,
    }
}

fn const_array_elem_is_resolvable(expr: &Expr, generic_vars: &GenericVars) -> bool {
    match expr {
        Expr::Lit(_) => true,
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.var_type(&ident).is_some()
        }
        Expr::Index(index) => const_cga_index_is_resolvable(index, generic_vars),
        Expr::Unary(unary) => const_array_elem_is_resolvable(&unary.expr, generic_vars),
        Expr::Paren(paren) => const_array_elem_is_resolvable(&paren.expr, generic_vars),
        _ => false,
    }
}

fn const_array_elem_is_fully_known(expr: &Expr, generic_vars: &GenericVars) -> bool {
    match expr {
        Expr::Lit(_) => true,
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.inst_i32.contains_key(&ident)
                || generic_vars.inst_bool.contains_key(&ident)
                || generic_vars.inst_array.contains_key(&ident)
                || generic_vars.len2array.contains_key(&ident)
        }
        Expr::Index(index) => const_cga_index_is_resolvable(index, generic_vars),
        Expr::Unary(unary) => const_array_elem_is_fully_known(&unary.expr, generic_vars),
        Expr::Paren(paren) => const_array_elem_is_fully_known(&paren.expr, generic_vars),
        _ => false,
    }
}

fn const_cga_index_is_resolvable(index: &syn::ExprIndex, generic_vars: &GenericVars) -> bool {
    let Expr::Path(path) = index.expr.as_ref() else {
        return false;
    };
    let ident = get_ident_from_path_expr(path).to_string();
    let Some(shape) = generic_vars.inst_array.get(&ident) else {
        return false;
    };
    let Some(i) = static_usize_index(&index.index) else {
        return false;
    };
    i < shape.len()
}

fn static_usize_index(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(raw) => raw.base10_parse::<usize>().ok(),
            _ => None,
        },
        Expr::Unary(unary) => {
            if matches!(unary.op, syn::UnOp::Neg(_)) {
                None
            } else {
                static_usize_index(&unary.expr)
            }
        }
        Expr::Paren(paren) => static_usize_index(&paren.expr),
        _ => None,
    }
}

pub fn infer_method_generics(
    impl_item: &ItemImpl,
    impl_method: &ImplItemFn,
    method_call: &ExprMethodCall,
    call_arg_rust_tys: &[Type],
    self_ty: &Type,
    caller_generic_vars: &GenericVars,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Result<GenericVars, JITError> {
    let generic_arg_inference = GenericArgInference::new_method(impl_item, impl_method);
    if generic_arg_inference.param2arg.is_empty() {
        return GenericVars::empty(&impl_method.sig.generics);
    }

    let mut generic_arg_inference = GenericArgInference::new_method(impl_item, impl_method);
    generic_arg_inference.map_args_to_params(
        &call_arg_rust_tys.to_vec(),
        Some(self_ty),
        caller_generic_vars,
    )?;
    let inferred = generic_arg_inference.get_generic_vars_instance(caller_generic_vars, primitives);

    if method_call.turbofish.is_some() {
        let passed = caller_generic_vars
            .from_expr_generic_args(&impl_method.sig.generics, &method_call.turbofish)?;
        inferred.merge(passed)
    } else {
        Ok(inferred)
    }
}

fn should_skip_optional_output_type_param(
    type_param_name: &str,
    arg_zst_values: &HashMap<String, String>,
) -> bool {
    matches!(
        (
            type_param_name,
            arg_zst_values.get(type_param_name).map(String::as_str)
        ),
        ("padding_value", Some("None")) | ("dim_map", Some("Identity"))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_type_round_trips_common_syn_types() {
        let cases: Vec<Type> = vec![
            syn::parse_quote!(i32),
            syn::parse_quote!(bool),
            syn::parse_quote!(&mut Tensor<f32, { [16, 32] }>),
            syn::parse_quote!((i32, f32)),
            syn::parse_quote!([i32; 4]),
            syn::parse_quote!(*mut f32),
            syn::parse_quote!(!),
        ];

        for ty in cases {
            let resolved = ResolvedType::from_syn_type(&ty);
            assert_eq!(resolved.to_syn_type(), ty);
        }
    }

    #[test]
    fn builtin_scalar_names_include_pointer_sized_integers() {
        assert!(is_builtin_scalar_name("usize"));
        assert!(is_builtin_scalar_name("isize"));
    }

    #[test]
    fn receiver_adjustments_include_autoref_and_autoderef_candidates() {
        let receiver: Type = syn::parse_quote!(&mut Tensor<f32, { [4] }>);
        let candidates = receiver_adjustment_candidates(&receiver, true)
            .into_iter()
            .map(|ty| ty.to_token_stream().to_string())
            .collect::<Vec<_>>();

        assert!(candidates.contains(&"& mut Tensor < f32 , { [4] } >".to_string()));
        assert!(candidates.contains(&"Tensor < f32 , { [4] } >".to_string()));
        assert!(candidates.contains(&"& Tensor < f32 , { [4] } >".to_string()));
    }

    #[test]
    fn receiver_adjustments_do_not_add_mut_borrow_for_immutable_receivers() {
        let receiver: Type = syn::parse_quote!(Tensor<f32, { [4] }>);
        let candidates = receiver_adjustment_candidates(&receiver, false)
            .into_iter()
            .map(|ty| ty.to_token_stream().to_string())
            .collect::<Vec<_>>();

        assert!(candidates.contains(&"Tensor < f32 , { [4] } >".to_string()));
        assert!(candidates.contains(&"& Tensor < f32 , { [4] } >".to_string()));
        assert!(!candidates.contains(&"& mut Tensor < f32 , { [4] } >".to_string()));
    }
}
