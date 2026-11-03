/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Utilities for inspecting and manipulating `syn` AST nodes—attribute parsing,
//! generic parameter extraction, closure analysis, and pretty-printing helpers.

use crate::generics::GenericVars;
use proc_macro2::Ident;
use quote::ToTokens;
use std::collections::HashSet;
use std::fmt::Display;
use std::str::FromStr;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{
    parse_quote, AngleBracketedGenericArguments, Attribute, ConstParam, Expr, ExprCall,
    ExprClosure, ExprMethodCall, ExprPath, FnArg, GenericArgument, GenericParam, Generics, Item,
    ItemFn, Lit, Meta, MetaList, Pat, PathArguments, ReturnType, Signature, Token, Type,
};

const CUTILE_INTERNAL_NODE_ID_ATTR: &str = "__cutile_node_id";

/// A parsed attribute meta list (e.g. `#[cuda_tile::ty(name = "f32")]`).
#[derive(Debug)]
pub struct SingleMetaList {
    name: Option<String>,
    meta_list: Option<MetaList>,
    variables: Vec<Meta>,
}

/// Render an expression for user-facing diagnostics.
///
/// Compiler-owned `syn` trees carry internal node-id attributes used by
/// typeck/lowering side tables. Those attributes are intentionally retained in
/// the AST but should not leak into source-facing error messages.
pub fn user_expr_tokens(expr: &Expr) -> String {
    let mut expr = expr.clone();
    strip_user_hidden_expr_attrs(&mut expr);
    expr.to_token_stream().to_string()
}

pub fn user_call_tokens(call_expr: &ExprCall) -> String {
    user_expr_tokens(&Expr::Call(call_expr.clone()))
}

pub fn user_method_call_tokens(method_call_expr: &ExprMethodCall) -> String {
    user_expr_tokens(&Expr::MethodCall(method_call_expr.clone()))
}

fn strip_user_hidden_expr_attrs(expr: &mut Expr) {
    let mut stripper = UserHiddenAttrStripper;
    stripper.visit_expr_mut(expr);
}

struct UserHiddenAttrStripper;

impl VisitMut for UserHiddenAttrStripper {
    fn visit_attribute_mut(&mut self, _attr: &mut Attribute) {
        // Do not recurse into attribute metadata.
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Some(attrs) = expr_attrs_mut(expr) {
            attrs.retain(|attr| !attr.path().is_ident(CUTILE_INTERNAL_NODE_ID_ATTR));
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}

fn expr_attrs_mut(expr: &mut Expr) -> Option<&mut Vec<Attribute>> {
    match expr {
        Expr::Array(expr) => Some(&mut expr.attrs),
        Expr::Assign(expr) => Some(&mut expr.attrs),
        Expr::Async(expr) => Some(&mut expr.attrs),
        Expr::Await(expr) => Some(&mut expr.attrs),
        Expr::Binary(expr) => Some(&mut expr.attrs),
        Expr::Block(expr) => Some(&mut expr.attrs),
        Expr::Break(expr) => Some(&mut expr.attrs),
        Expr::Call(expr) => Some(&mut expr.attrs),
        Expr::Cast(expr) => Some(&mut expr.attrs),
        Expr::Closure(expr) => Some(&mut expr.attrs),
        Expr::Const(expr) => Some(&mut expr.attrs),
        Expr::Continue(expr) => Some(&mut expr.attrs),
        Expr::Field(expr) => Some(&mut expr.attrs),
        Expr::ForLoop(expr) => Some(&mut expr.attrs),
        Expr::Group(expr) => Some(&mut expr.attrs),
        Expr::If(expr) => Some(&mut expr.attrs),
        Expr::Index(expr) => Some(&mut expr.attrs),
        Expr::Infer(expr) => Some(&mut expr.attrs),
        Expr::Let(expr) => Some(&mut expr.attrs),
        Expr::Lit(expr) => Some(&mut expr.attrs),
        Expr::Loop(expr) => Some(&mut expr.attrs),
        Expr::Macro(expr) => Some(&mut expr.attrs),
        Expr::Match(expr) => Some(&mut expr.attrs),
        Expr::MethodCall(expr) => Some(&mut expr.attrs),
        Expr::Paren(expr) => Some(&mut expr.attrs),
        Expr::Path(expr) => Some(&mut expr.attrs),
        Expr::Range(expr) => Some(&mut expr.attrs),
        Expr::RawAddr(expr) => Some(&mut expr.attrs),
        Expr::Reference(expr) => Some(&mut expr.attrs),
        Expr::Repeat(expr) => Some(&mut expr.attrs),
        Expr::Return(expr) => Some(&mut expr.attrs),
        Expr::Struct(expr) => Some(&mut expr.attrs),
        Expr::Try(expr) => Some(&mut expr.attrs),
        Expr::TryBlock(expr) => Some(&mut expr.attrs),
        Expr::Tuple(expr) => Some(&mut expr.attrs),
        Expr::Unary(expr) => Some(&mut expr.attrs),
        Expr::Unsafe(expr) => Some(&mut expr.attrs),
        Expr::While(expr) => Some(&mut expr.attrs),
        Expr::Yield(expr) => Some(&mut expr.attrs),
        Expr::Verbatim(_) => None,
        _ => None,
    }
}

impl SingleMetaList {
    /// Construct from a `syn::Attribute` that contains a meta list.
    pub fn from_attribute(attr: Attribute) -> Self {
        match attr.meta {
            Meta::List(meta_list) => {
                let tokens = meta_list.tokens.clone();
                let mut result = syn::parse2::<SingleMetaList>(tokens).unwrap();
                result.name = Some(meta_list.path.to_token_stream().to_string());
                result.meta_list = Some(meta_list);
                result
            }
            // The bare form (`#[cutile::entry]`): no arguments, every
            // knob at its default.
            Meta::Path(path) => Self {
                name: Some(path.to_token_stream().to_string()),
                meta_list: None,
                variables: Vec::new(),
            },
            other => panic!("Unexpected attribute list {other:#?}"),
        }
    }
    /// Returns the attribute path as a single string.
    pub fn name_as_str(&self) -> Option<String> {
        self.name.clone()
    }
    /// Returns the attribute path split by `::` separators.
    pub fn name_as_vec(&self) -> Option<Vec<&str>> {
        self.name
            .as_ref()
            .map(|s| s.as_str().split(" :: ").collect())
    }
    fn get_value(&self, name: &str) -> Option<&Expr> {
        for item in &self.variables {
            match item {
                Meta::NameValue(name_value) => {
                    let meta_ident = name_value.path.get_ident();
                    let meta_name = meta_ident.unwrap().to_string();
                    if name == meta_name {
                        return Some(&name_value.value);
                    }
                }
                _ => continue,
            }
        }
        None
    }
    /// Returns the raw expression for a named key-value entry.
    pub fn parse_custom_expr(&self, name: &str) -> Option<&Expr> {
        self.get_value(name)
    }
    /// Parses a named entry as an array of string literals.
    pub fn parse_string_arr(&self, name: &str) -> Option<Vec<String>> {
        let value = self.get_value(name);
        match value {
            Some(val) => {
                let Expr::Array(ref arr) = val else {
                    panic!("{name} is not an array: {val:#?}")
                };
                let mut res = vec![];
                for val in &arr.elems {
                    let Expr::Lit(ref lit) = val else {
                        panic!("{name} is not a literal: {val:#?}")
                    };
                    let Lit::Str(ref lit_str) = lit.lit else {
                        panic!("{name} is not a string: {lit:#?}")
                    };
                    res.push(lit_str.value().clone());
                }
                Some(res)
            }
            None => None,
        }
    }
    /// Parses a named entry as a single string literal.
    pub fn parse_string(&self, name: &str) -> Option<String> {
        let value = self.get_value(name);
        match value {
            Some(val) => {
                let Expr::Lit(ref lit) = val else {
                    panic!("{name} is not a literal: {val:#?}")
                };
                let Lit::Str(ref lit_str) = lit.lit else {
                    panic!("{name} is not a string: {lit:#?}")
                };
                Some(lit_str.value().clone())
            }
            None => None,
        }
    }
    /// Parses a named entry as a `u32` integer literal.
    pub fn parse_int(&self, name: &str) -> Option<u32> {
        let value = self.get_value(name);
        match value {
            Some(val) => {
                let Expr::Lit(ref lit) = val else {
                    panic!("{name} is not a literal: {val:#?}")
                };
                let Lit::Int(ref lit_int) = lit.lit else {
                    panic!("{name} is not a string: {lit:#?}")
                };
                Some(lit_int.base10_parse().unwrap())
            }
            None => None,
        }
    }
    /// Parses a named entry as a boolean literal. The bare form
    /// (`#[cutile::entry(unchecked_accesses)]`, a `Meta::Path`) means `true`,
    /// as the reference documents; it used to be silently ignored because
    /// only `name = value` entries were consulted.
    pub fn parse_bool(&self, name: &str) -> Option<bool> {
        if self
            .variables
            .iter()
            .any(|item| matches!(item, Meta::Path(path) if path.is_ident(name)))
        {
            return Some(true);
        }
        let value = self.get_value(name);
        match value {
            Some(val) => {
                let Expr::Lit(ref lit) = val else {
                    panic!("{name} is not a literal: {val:#?}")
                };
                let Lit::Bool(ref lit_bool) = lit.lit else {
                    panic!("{name} is not a string: {lit:#?}")
                };
                Some(lit_bool.value)
            }
            None => None,
        }
    }
}

impl Parse for SingleMetaList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let variables = Punctuated::<Meta, Token![,]>::parse_terminated(input)?
            .into_iter()
            .collect();
        Ok(Self {
            name: None,
            meta_list: None,
            variables,
        })
    }
}

impl From<SingleMetaList> for Vec<Attribute> {
    fn from(val: SingleMetaList) -> Self {
        let mut res = vec![];
        for meta in val.variables {
            let attr: Attribute = parse_quote! {
                #[noname(#meta)]
            };
            res.push(attr);
        }
        res
    }
}

/// Removes all attributes whose path matches one of the given names.
pub fn clear_attributes(attr_names: HashSet<&str>, attrs: &mut Vec<Attribute>) {
    // filter == keep
    *attrs = attrs
        .clone()
        .into_iter()
        .filter(|attr| match &attr.meta {
            Meta::Path(meta_path) => {
                let name = meta_path.to_token_stream().to_string();
                !attr_names.contains(name.as_str())
            }
            Meta::List(meta_list) => {
                let name = meta_list.path.to_token_stream().to_string();
                !attr_names.contains(name.as_str())
            }
            _ => true,
        })
        .collect::<Vec<Attribute>>();
}

/// Finds an attribute by path string, optionally matching only the last
/// segment. Matches both the argument-list form (`#[cutile::entry(...)]`)
/// and the bare-path form (`#[cutile::entry]`) — every argument has a
/// default, so the bare form is the documented spelling for a kernel that
/// needs none.
pub fn get_attribute(
    lookup_str: &str,
    outer_attrs: &Vec<Attribute>,
    last_seg_only: bool,
) -> Option<Attribute> {
    for attr in outer_attrs {
        let path = match &attr.meta {
            Meta::List(meta_list) => &meta_list.path,
            Meta::Path(path) => path,
            _ => continue,
        };
        let parsed_str = if last_seg_only {
            path.segments.last().unwrap().to_token_stream().to_string()
        } else {
            path.to_token_stream().to_string()
        };
        if parsed_str == lookup_str {
            return Some(attr.clone());
        }
    }
    None
}

/// Looks up an attribute by full path and parses it as a [`SingleMetaList`].
pub fn get_meta_list(attr_name: &str, outer_attrs: &Vec<Attribute>) -> Option<SingleMetaList> {
    get_attribute(attr_name, outer_attrs, false).map(SingleMetaList::from_attribute)
}

/// Like [`get_meta_list`] but matches only the last path segment.
pub fn get_meta_list_by_last_segment(
    last_seg: &str,
    outer_attrs: &Vec<Attribute>,
) -> Option<SingleMetaList> {
    get_attribute(last_seg, outer_attrs, true).map(SingleMetaList::from_attribute)
}

/// Finds the first `cuda_tile::*` attribute and parses it as a [`SingleMetaList`].
pub fn get_cuda_tile_meta_list(outer_attrs: &Vec<Attribute>) -> Option<SingleMetaList> {
    let mut found: Option<SingleMetaList> = None;
    for attr in outer_attrs {
        let Meta::List(meta_list) = &attr.meta else {
            continue;
        };
        let name = meta_list.path.to_token_stream().to_string();
        let name_parts = name.split(" :: ").collect::<Vec<&str>>();
        if name_parts[0] == "cuda_tile" {
            if found.is_some() {
                panic!("Found multiple cuda_tile attributes {outer_attrs:#?}")
            }
            found = Some(SingleMetaList::from_attribute(attr.clone()));
        }
    }
    found
}

#[derive(Debug, Clone)]
/// A const generic array parameter whose length is itself a generic variable.
pub struct VarCGAParameter {
    pub name: String,
    pub element_type: String,
    pub length_var: String,
}

impl VarCGAParameter {
    /// Instantiate with a concrete length to produce a [`CGAParameter`].
    pub fn instance(&self, length: u32) -> CGAParameter {
        CGAParameter {
            name: self.name.clone(),
            element_type: self.element_type.clone(),
            length,
        }
    }
}
impl VarCGAParameter {
    /// Extracts a variable-length CGA parameter from a `const` generic declaration.
    pub fn from_const_param(const_param: &ConstParam) -> VarCGAParameter {
        let name = const_param.ident.to_string();
        let Type::Array(ty_arr) = &const_param.ty else {
            panic!("Expected array type.")
        };
        let Type::Path(ref element_type) = *ty_arr.elem else {
            panic!("Expected type path.")
        };
        let element_type = element_type
            .path
            .get_ident()
            .unwrap()
            .to_string()
            .to_string();
        match &ty_arr.len {
            Expr::Path(length_expr) => {
                let length_var = length_expr
                    .path
                    .get_ident()
                    .unwrap()
                    .to_string()
                    .to_string();
                VarCGAParameter {
                    name,
                    element_type,
                    length_var,
                }
            }
            _ => {
                panic!("Unexpected path expression {:#?}.", ty_arr.len)
            }
        }
    }
    /// Returns `true` if the const param has a variable (non-literal) array length.
    pub fn is_var_cga(const_param: &ConstParam) -> bool {
        let Type::Array(ty_arr) = &const_param.ty else {
            panic!("Expected array type.")
        };
        if let Expr::Path(_length_expr) = &ty_arr.len {
            return true;
        };
        false
    }
    /// Returns `Some` if the const param is a variable-length CGA parameter.
    pub fn maybe_var_cga(const_param: &ConstParam) -> Option<VarCGAParameter> {
        if VarCGAParameter::is_var_cga(const_param) {
            Some(VarCGAParameter::from_const_param(const_param))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A const generic array parameter with a concrete length.
pub struct CGAParameter {
    pub name: String,
    pub element_type: String,
    pub length: u32,
}

#[derive(Debug, Clone)]
/// A scalar const generic parameter (e.g. `const N: i32`).
pub struct ConstParameter {
    pub name: String,
    pub ty: String,
}

impl CGAParameter {
    /// Extracts a fixed-length CGA parameter from a `const` generic declaration.
    pub fn from_const_param(const_param: &ConstParam) -> CGAParameter {
        let name = const_param.ident.to_string();
        let Type::Array(ty_arr) = &const_param.ty else {
            panic!("Expected array type.")
        };
        let Type::Path(ref element_type) = *ty_arr.elem else {
            panic!("Expected type path.")
        };
        let element_type = element_type
            .path
            .get_ident()
            .unwrap()
            .to_string()
            .to_string();
        match &ty_arr.len {
            Expr::Lit(expr_lit) => {
                let length: u32 = expr_lit
                    .to_token_stream()
                    .to_string()
                    .parse::<u32>()
                    .unwrap();
                CGAParameter {
                    name,
                    element_type,
                    length,
                }
            }
            _ => {
                panic!("Unexpected path expression {:#?}.", ty_arr.len)
            }
        }
    }
}

/// Separates generic params into const generic arrays and scalar const parameters.
pub fn parse_cgas(generics: &Generics) -> (Vec<CGAParameter>, Vec<ConstParameter>) {
    let mut cga_params: Vec<CGAParameter> = vec![];
    let mut const_params: Vec<ConstParameter> = vec![];
    for param in &generics.params {
        match param {
            GenericParam::Type(_type_param) => continue,
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(_ty_arr) => {
                    let arr_type_param = CGAParameter::from_const_param(const_param);
                    cga_params.push(arr_type_param);
                }
                Type::Path(type_path) => {
                    let name = const_param.ident.to_string();
                    let ty = type_path.to_token_stream().to_string();
                    const_params.push(ConstParameter { name, ty });
                }
                _ => continue,
            },
            _ => continue,
        }
    }
    (cga_params, const_params)
}

/// Returns the variable name for a function argument (including `self`).
pub fn get_fn_arg_var_name(arg: &FnArg) -> String {
    match arg {
        FnArg::Receiver(receiver) => receiver.self_token.to_token_stream().to_string(),
        FnArg::Typed(fn_param) => match fn_param.pat.as_ref() {
            syn::Pat::Ident(identifier) => identifier.ident.to_string(),
            _ => panic!("Unexpected argument pattern"),
        },
    }
}

/// Collects parameter names from a function signature.
pub fn get_sig_param_names(sig: &Signature) -> Vec<String> {
    let mut result = vec![];
    for arg in &sig.inputs {
        let name = get_fn_arg_var_name(arg);
        result.push(name);
    }
    result
}

/// Extracts angle-bracketed generic arguments from a call expression (e.g. `foo::<T, 3>(...)`).
pub fn get_call_expression_generics(
    call_expr: &ExprCall,
) -> Option<AngleBracketedGenericArguments> {
    match &*call_expr.func {
        Expr::Path(path_expr) => {
            let last_seg = path_expr.path.segments.last();
            match last_seg {
                Some(seg) => {
                    if let PathArguments::AngleBracketed(path_generic_args) = &seg.arguments {
                        // TODO (hme): Check if this is okay.
                        Some(path_generic_args.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Collects integer const generic arguments from angle brackets, resolving generic vars.
pub fn get_generic_arg_ints<T>(
    generic_args: &AngleBracketedGenericArguments,
    generic_vars: Option<&GenericVars>,
) -> Vec<T>
where
    T: FromStr + From<i32>,
    T::Err: Display,
{
    let mut result = vec![];
    for arg in &generic_args.args {
        match arg {
            GenericArgument::Const(expr) => {
                let Expr::Lit(lit) = expr else {
                    panic!("Unexpected expression.")
                };
                let Lit::Int(int_expr) = &lit.lit else {
                    panic!("Unexpected expression.")
                };
                let x = int_expr.base10_parse::<T>().expect("Failed to parse int.");
                result.push(x);
            }
            GenericArgument::Type(ty) => {
                if let Type::Path(path_ty) = ty {
                    if let Some(const_var) = path_ty.path.get_ident() {
                        if let Some(generic_vars) = generic_vars {
                            if let Some(const_val) =
                                generic_vars.inst_i32.get(&const_var.to_string())
                            {
                                result.push(T::from(*const_val));
                            }
                        }
                    }
                }
            }
            _ => continue,
        };
    }
    result
}

/// Returns `(input_types, return_type)` for a function signature.
pub fn get_sig_types(sig: &Signature, self_ty: Option<&Type>) -> (Vec<Type>, Type) {
    let mut input_tys: Vec<Type> = vec![];
    for input in sig.inputs.iter() {
        match input {
            FnArg::Typed(fn_param) => {
                let _name = {
                    match &*fn_param.pat {
                        Pat::Ident(ident) => ident.ident.to_string(),
                        _ => panic!("Unexpected function param pattern {:#?}.", fn_param.pat),
                    }
                };
                let ty = &*fn_param.ty;
                input_tys.push(ty.clone());
            }
            FnArg::Receiver(_fn_self) => {
                assert!(
                    self_ty.is_some(),
                    "bind_parameters for impls requires self_ty."
                );
                let self_ty = self_ty.unwrap().clone();
                input_tys.push(self_ty);
            }
        }
    }
    let ret_ty = get_sig_output_type(sig);
    (input_tys, ret_ty)
}

/// Returns the output type of a signature, defaulting to `()`.
pub fn get_sig_output_type(sig: &Signature) -> Type {
    match &sig.output {
        ReturnType::Type(_, return_type) => *return_type.clone(),
        ReturnType::Default => syn::parse2::<Type>("()".parse().unwrap()).unwrap(),
    }
}

/// Returns `true` if the function has a non-unit return type.
pub fn function_returns(fn_item: &ItemFn) -> bool {
    match &fn_item.sig.output {
        ReturnType::Type(_, return_type) => match &**return_type {
            Type::Tuple(type_tuple) => !type_tuple.elems.is_empty(),
            _ => true,
        },
        ReturnType::Default => false,
    }
}

/// Returns the last segment's ident from a `syn::Path`.
pub fn get_ident_from_path(path: &syn::Path) -> Ident {
    path.segments.last().unwrap().ident.clone()
}

/// Returns the last segment's ident from a path expression.
pub fn get_ident_from_path_expr(path_expr: &ExprPath) -> Ident {
    get_ident_from_path(&path_expr.path)
}

/// Tries to extract an ident from a path or reference expression.
pub fn get_ident_from_expr(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Path(path_expr) => Some(get_ident_from_path(&path_expr.path)),
        Expr::Reference(ref_expr) => get_ident_from_expr(&ref_expr.expr),
        _ => None,
    }
}

/// Returns the leaf ident of a type, following pointers and references.
pub fn get_type_ident(ty: &Type) -> Option<Ident> {
    match ty {
        Type::Path(type_path) => Some(type_path.path.segments.last().unwrap().ident.clone()),
        Type::Ptr(type_ptr) => get_type_ident(&type_ptr.elem),
        Type::Reference(type_ref) => get_type_ident(&type_ref.elem),
        _ => None,
    }
}

/// Returns a string representation of the type's leaf segment.
pub fn get_type_str(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(type_path) => Some(type_path.path.segments.last().unwrap().ident.to_string()),
        Type::Ptr(type_ptr) => Some(type_ptr.to_token_stream().to_string()),
        Type::Reference(type_ref) => Some(type_ref.to_token_stream().to_string()),
        _ => None,
    }
}

/// Returns `(ident, generic_args)` for a type of the form `T<...>`.
pub fn get_ident_generic_args(ty: &Type) -> (Option<Ident>, AngleBracketedGenericArguments) {
    match ty {
        Type::Path(type_path) => {
            let result_type = type_path.clone();
            let maybe_last_seg = result_type.path.segments.last().unwrap();
            let last_seg = maybe_last_seg.clone();
            match last_seg.arguments {
                // The type takes a const generic array as a type param:
                // f(..., shape: Shape<D>) -> ()
                PathArguments::AngleBracketed(type_params) => {
                    // This is a type of the form T<...>
                    (Some(last_seg.ident.clone()), type_params.clone())
                }
                _ => panic!(
                    "get_ident_generic_args: Unexpected generic arguments {:#?} for {ty:#?}",
                    last_seg.arguments
                ),
            }
        }
        Type::Reference(ref_type) => get_ident_generic_args(&ref_type.elem),
        _ => panic!("get_ident_generic_args: Unexpected type {:#?}", ty),
    }
}

/// Returns generic arguments if the type has them, or `None` otherwise.
pub fn maybe_generic_args(ty: &Type) -> Option<AngleBracketedGenericArguments> {
    match ty {
        Type::Path(type_path) => {
            let result_type = type_path.clone();
            let maybe_last_seg = result_type.path.segments.last().unwrap();
            let last_seg = maybe_last_seg.clone();
            match last_seg.arguments {
                // The type takes a const generic array as a type param:
                // f(..., shape: Shape<D>) -> ()
                PathArguments::AngleBracketed(type_params) => {
                    // This is a type of the form T<...>
                    Some(type_params.clone())
                }
                PathArguments::None => None,
                _ => panic!(
                    "get_ident_generic_args: Unexpected generic arguments {:#?} for {ty:#?}",
                    last_seg.arguments
                ),
            }
        }
        Type::Reference(ref_type) => maybe_generic_args(&ref_type.elem),
        Type::Ptr(_ptr_type) => None,
        Type::Array(_arr_type) => None,
        Type::Slice(_slice_type) => None,
        Type::Tuple(_) => {
            // Tuples don't have generic arguments at the top level.
            // "Top level" means on the tuple type itself - Rust tuple syntax is `(T1, T2, ...)`
            // not `Tuple<T1, T2, ...>`, so there are no angle brackets to extract.
            // The tuple's element types may themselves have generic arguments (e.g., `(i32, Vec<T>)`),
            // but those are handled separately when processing each element.
            None
        }
        _ => panic!("get_ident_generic_args: Unexpected type {:#?}", ty),
    }
}

/// Extracts type and const generic param names (skipping lifetimes).
pub fn get_supported_generic_params(generics: &Generics) -> Vec<(String, Option<Type>)> {
    let mut param_names: Vec<(String, Option<Type>)> = vec![];
    for param in &generics.params {
        match param {
            GenericParam::Type(type_param) => {
                let name = type_param.ident.to_string();
                param_names.push((name, None));
            }
            GenericParam::Const(const_param) => {
                let name = const_param.ident.to_string();
                let ty = const_param.ty.clone();
                param_names.push((name, Some(ty)));
            }
            GenericParam::Lifetime(_lifetime_param) => continue,
            #[allow(unreachable_patterns)]
            _ => panic!("Unexpected generic parameter {:#?}", param),
        }
    }
    param_names
}

/// Removes lifetime arguments from angle-bracketed generic arguments in place.
pub fn strip_generic_args_lifetimes(gen_args: &mut AngleBracketedGenericArguments) {
    let mut res = gen_args.args.clone();
    res.clear();
    for gen_arg in gen_args.args.iter() {
        if let GenericArgument::Lifetime(_gen_arg_lifetime) = gen_arg {
            continue;
        }
        res.push(gen_arg.clone());
    }
    gen_args.args = res;
}

/// Removes lifetime parameters from generics in place.
pub fn strip_generics_lifetimes(generics: &mut Generics) {
    let mut res = generics.params.clone();
    res.clear();
    for gen_param in generics.params.iter() {
        if let GenericParam::Lifetime(_) = gen_param {
            continue;
        }
        res.push(gen_param.clone());
    }
    generics.params = res;
}

/// Pretty-prints a single `syn::Item` via `prettyplease`.
pub fn item_string_pretty(item: &Item) -> String {
    let file = syn::File {
        attrs: vec![],
        items: vec![item.clone()],
        shebang: None,
    };
    file_item_string_pretty(&file)
}

/// Pretty-prints a `syn::File` via `prettyplease`.
pub fn file_item_string_pretty(file: &syn::File) -> String {
    prettyplease::unparse(file)
}

/// Represents a closure parameter with optional type annotation
#[derive(Debug, Clone)]
pub struct ClosureParam {
    pub name: String,
    pub ty: Option<Type>, // Some(ty) if |x: i32| ..., None if |x| ...
}

/// Parsed closure information
#[derive(Debug, Clone)]
pub struct ClosureInfo {
    pub params: Vec<ClosureParam>,
    pub body: Box<Expr>,
    pub is_async: bool,
    pub is_move: bool,
}

/// Extract closure parameters and body from ExprClosure
///
/// Example:
/// - `|x, y| x + y` -> params: [x, y], body: x + y
/// - `|acc: i32, x: i32| acc + x` -> params: [acc: i32, x: i32], body: acc + x
pub fn parse_closure(closure_expr: &ExprClosure) -> ClosureInfo {
    let mut params = Vec::new();

    for input in &closure_expr.inputs {
        match input {
            Pat::Ident(pat_ident) => {
                // Simple parameter: |x| ...
                params.push(ClosureParam {
                    name: pat_ident.ident.to_string(),
                    ty: None,
                });
            }
            Pat::Type(pat_type) => {
                // Typed parameter: |x: i32| ...
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    params.push(ClosureParam {
                        name: pat_ident.ident.to_string(),
                        ty: Some((*pat_type.ty).clone()),
                    });
                } else {
                    panic!("Unsupported closure parameter pattern: {:#?}", pat_type.pat);
                }
            }
            _ => panic!("Unsupported closure parameter pattern: {:#?}", input),
        }
    }

    ClosureInfo {
        params,
        body: closure_expr.body.clone(),
        is_async: closure_expr.asyncness.is_some(),
        is_move: closure_expr.capture.is_some(),
    }
}

/// Check if an expression contains a reference to any of the given variable names
/// Used to detect variable captures in closures
///
/// # Examples
///
/// **Example 1: Basic variable reference detection**
/// ```ignore
/// // Context: We have variables x, y, z in scope
/// // Code being analyzed: x + y * 2
///
/// let mut var_names = HashSet::new();
/// var_names.insert("x".to_string());
/// var_names.insert("y".to_string());
/// var_names.insert("z".to_string());  // z is in scope but not used
///
/// // Parse the expression "x + y * 2"
/// let captures = expr_references_vars(&expr, &var_names);
/// // Returns: ["x", "y"] - only variables actually referenced in the expression
/// ```
///
/// **Example 2: Method call with multiple references**
/// ```ignore
/// // Context: We have variables tile, view, index in scope
/// // Code being analyzed: tile.load(&view, index)
///
/// let mut var_names = HashSet::new();
/// var_names.insert("tile".to_string());
/// var_names.insert("view".to_string());
/// var_names.insert("index".to_string());
///
/// let captures = expr_references_vars(&expr, &var_names);
/// // Returns: ["tile", "view", "index"] - all three are referenced
/// ```
///
/// **Example 3: Closure capture detection**
/// ```ignore
/// // Context: Outer scope has variables: scale, offset
/// // Closure being analyzed: |acc, x| acc + x
/// // We want to find which outer variables the closure captures
///
/// let mut var_names = HashSet::new();
/// var_names.insert("scale".to_string());  // outer scope variable
/// var_names.insert("offset".to_string()); // outer scope variable
/// // Note: NOT including "acc" or "x" because they are the closure's parameters
///
/// // Parse the closure body expression "acc + x"
/// let captures = expr_references_vars(&closure_body_expr, &var_names);
/// // Returns: [] - the closure only uses its own parameters, not outer variables
/// ```
///
/// **Example 4: Closure that captures outer variables**
/// ```ignore
/// // Context: Outer scope has variables: scale, offset
/// // Closure being analyzed: |x| x * scale + offset
///
/// let mut var_names = HashSet::new();
/// var_names.insert("scale".to_string());
/// var_names.insert("offset".to_string());
///
/// // Parse the closure body expression "x * scale + offset"
/// let captures = expr_references_vars(&closure_body_expr, &var_names);
/// // Returns: ["scale", "offset"] - both outer variables are captured
/// ```
///
/// # Note
///
/// This is a recursive function that traverses the expression tree.
/// It handles:
/// - Binary expressions: `a + b`, `x * y`
/// - Unary expressions: `-x`, `!flag`
/// - Function calls: `minf(a, b)`
/// - Method calls: `tile.reshape(shape)`
/// - Field access: `point.x`
/// - Indexing: `array[i]`
/// - Blocks, if-expressions, and parenthesized expressions
///
/// # Future Use
///
/// TODO (np): This comprehensive expression traversal could be reused to identify
/// loop carry variables in for/while loops by checking which variables from outer
/// scope are referenced in the loop body.
fn expr_references_vars(expr: &Expr, var_names: &HashSet<String>) -> Vec<String> {
    let mut captured = Vec::new();

    match expr {
        Expr::Path(path_expr) => {
            if let Some(ident) = path_expr.path.get_ident() {
                let name = ident.to_string();
                if var_names.contains(&name) {
                    captured.push(name);
                }
            }
        }
        Expr::Binary(bin_expr) => {
            captured.extend(expr_references_vars(&bin_expr.left, var_names));
            captured.extend(expr_references_vars(&bin_expr.right, var_names));
        }
        Expr::Unary(unary_expr) => {
            captured.extend(expr_references_vars(&unary_expr.expr, var_names));
        }
        Expr::Call(call_expr) => {
            captured.extend(expr_references_vars(&call_expr.func, var_names));
            for arg in &call_expr.args {
                captured.extend(expr_references_vars(arg, var_names));
            }
        }
        Expr::MethodCall(method_call) => {
            captured.extend(expr_references_vars(&method_call.receiver, var_names));
            for arg in &method_call.args {
                captured.extend(expr_references_vars(arg, var_names));
            }
        }
        Expr::Field(field_expr) => {
            captured.extend(expr_references_vars(&field_expr.base, var_names));
        }
        Expr::Index(index_expr) => {
            captured.extend(expr_references_vars(&index_expr.expr, var_names));
            captured.extend(expr_references_vars(&index_expr.index, var_names));
        }
        Expr::Paren(paren_expr) => {
            captured.extend(expr_references_vars(&paren_expr.expr, var_names));
        }
        Expr::Block(block_expr) => {
            for stmt in &block_expr.block.stmts {
                if let syn::Stmt::Expr(expr, _) = stmt {
                    captured.extend(expr_references_vars(expr, var_names));
                }
            }
        }
        Expr::If(if_expr) => {
            captured.extend(expr_references_vars(&if_expr.cond, var_names));
            for stmt in &if_expr.then_branch.stmts {
                if let syn::Stmt::Expr(expr, _) = stmt {
                    captured.extend(expr_references_vars(expr, var_names));
                }
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                captured.extend(expr_references_vars(else_expr, var_names));
            }
        }
        // Add more expression types as needed
        _ => {}
    }

    captured
}

/// Get all variables captured by a closure from the outer scope
///
/// This function identifies which outer scope variables are referenced (captured) by a closure.
/// It performs **name-based** capture detection only - it does not determine capture mode
/// (by-value, by-reference, by-mutable-reference).
///
/// Example:
/// ```rust,ignore
/// let x = 5;
/// let y = 10;
/// let closure = |a| a + x;  // Captures 'x' but not 'y'
/// ```
///
/// # Capture Mode Detection
///
/// Rust closures can capture in three ways:
/// - **By value (move):** `move |x| { ... }` - explicit move keyword
/// - **By immutable reference:** `&T` - default for `Fn` closures
/// - **By mutable reference:** `&mut T` - used for `FnMut` closures
///
/// The example `|a, b| a + b` shown above "captures by move" in the sense that
/// if variables were captured, they would be moved. However, this function does
/// NOT distinguish between capture modes. Rust's type system determines the actual
/// capture mode based on how variables are used in the closure body:
/// - If a variable is only read → immutable borrow (`&T`)
/// - If a variable is mutated → mutable borrow (`&mut T`)
/// - If `move` keyword present → ownership transfer
///
/// # Loop Carry Variables
///
/// TODO (np): To use this for loops and determine mutability, we would need to:
/// 1. Track which captured variables are mutated inside the closure/loop body
/// 2. Analyze assignment expressions to identify mutations
/// 3. For now, we rely on Rust's type system to enforce valid capture semantics
pub fn get_closure_captures(
    closure_expr: &ExprClosure,
    outer_scope_vars: &[String],
) -> Vec<String> {
    let closure_info = parse_closure(closure_expr);

    // Build set of outer scope vars
    let outer_vars: HashSet<String> = outer_scope_vars.iter().cloned().collect();

    // Build set of closure parameter names (these shadow outer scope)
    let param_names: HashSet<String> = closure_info.params.iter().map(|p| p.name.clone()).collect();

    // Find references to outer scope vars that aren't shadowed by parameters
    let vars_available_to_capture: HashSet<String> =
        outer_vars.difference(&param_names).cloned().collect();

    let mut captured = expr_references_vars(&closure_info.body, &vars_available_to_capture);
    captured.sort();
    captured.dedup();
    captured
}

/// Check if an expression is a closure
pub fn is_closure(expr: &Expr) -> bool {
    matches!(expr, Expr::Closure(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_expr_tokens_strips_internal_node_id_attrs() {
        let expr: Expr = syn::parse_quote! {
            #[__cutile_node_id = 28]
            make_partition_view_mut(
                #[__cutile_node_id = 29] & #[__cutile_node_id = 30] z,
                #[__cutile_node_id = 31] const_shape![BM, BN],
                #[__cutile_node_id = 32] padding::None,
                #[__cutile_node_id = 33] z_token,
            )
        };

        assert!(expr
            .to_token_stream()
            .to_string()
            .contains("__cutile_node_id"));

        let rendered = user_expr_tokens(&expr);
        assert!(!rendered.contains("__cutile_node_id"));
        assert!(rendered.contains("make_partition_view_mut"));
        assert!(rendered.contains("const_shape"));
        assert!(expr
            .to_token_stream()
            .to_string()
            .contains("__cutile_node_id"));
    }
}
