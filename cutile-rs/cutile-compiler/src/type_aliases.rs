/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type-alias normalization for the DSL front-end.
//!
//! This is intentionally syntax-local: Pass 1 indexes `type` items, and
//! consumers that need the underlying DSL type ask this module to expand those
//! aliases before doing Tensor/scalar/pointer classification.

use quote::ToTokens;
use std::collections::HashMap;
use syn::visit_mut::{self, VisitMut};
use syn::{
    Expr, ExprPath, FnArg, GenericArgument, GenericParam, Item, ItemConst, ItemFn, ItemType, Lit,
    PatType, PathArguments, Type, TypePath,
};

const MAX_ALIAS_EXPANSION_DEPTH: usize = 64;

#[derive(Default)]
struct AliasSubstitution {
    types: HashMap<String, Type>,
    consts: HashMap<String, Expr>,
}

/// Collect type aliases from an inline module item list.
///
/// Nested inline modules are intentionally not merged into the parent scope.
/// Callers that process a submodule should call this again for that submodule's
/// own item list.
pub fn collect_type_aliases(items: &[Item]) -> HashMap<String, ItemType> {
    items
        .iter()
        .filter_map(|item| {
            let Item::Type(alias) = item else {
                return None;
            };
            Some((alias.ident.to_string(), alias.clone()))
        })
        .collect()
}

/// Expand all known type aliases in a type.
pub fn normalize_type_aliases(
    ty: &Type,
    aliases: &HashMap<String, ItemType>,
) -> Result<Type, String> {
    normalize_type_aliases_inner(ty, aliases, &mut Vec::new())
}

/// Replace paths to module-level scalar consts inside a type with their
/// literal expression. This keeps type-level shapes such as
/// `Tensor<f32, {[BLOCK]}>` consumable by the existing generic machinery.
pub fn normalize_const_paths_in_type(
    ty: &Type,
    consts: &HashMap<String, ItemConst>,
) -> Result<Type, String> {
    let mut out = ty.clone();
    ConstPathNormalizer { consts }.visit_type_mut(&mut out);
    Ok(out)
}

/// Return a supported scalar const expression suitable for substituting into
/// a type-level const expression.
pub fn const_item_scalar_expr(item: &ItemConst) -> Option<Expr> {
    let Type::Path(type_path) = item.ty.as_ref() else {
        return None;
    };
    let ty_name = type_path.path.segments.last()?.ident.to_string();
    match ty_name.as_str() {
        "bool" => match item.expr.as_ref() {
            Expr::Lit(lit) if matches!(lit.lit, Lit::Bool(_)) => Some(item.expr.as_ref().clone()),
            _ => None,
        },
        "i32" | "i64" | "u32" | "u64" => match item.expr.as_ref() {
            Expr::Lit(lit) if matches!(lit.lit, Lit::Int(_)) => Some(item.expr.as_ref().clone()),
            Expr::Unary(unary) => match unary.expr.as_ref() {
                Expr::Lit(lit) if matches!(lit.lit, Lit::Int(_)) => {
                    Some(item.expr.as_ref().clone())
                }
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

pub fn const_item_i32_value(item: &ItemConst) -> Option<i32> {
    let Type::Path(type_path) = item.ty.as_ref() else {
        return None;
    };
    if type_path.path.segments.last()?.ident != "i32" {
        return None;
    }
    match item.expr.as_ref() {
        Expr::Lit(lit) => match &lit.lit {
            Lit::Int(int_lit) => int_lit.base10_parse::<i32>().ok(),
            _ => None,
        },
        Expr::Unary(unary) => {
            if !matches!(unary.op, syn::UnOp::Neg(_)) {
                return None;
            }
            let Expr::Lit(lit) = unary.expr.as_ref() else {
                return None;
            };
            let Lit::Int(int_lit) = &lit.lit else {
                return None;
            };
            int_lit.base10_parse::<i32>().ok().map(|value| -value)
        }
        _ => None,
    }
}

pub fn const_item_bool_value(item: &ItemConst) -> Option<bool> {
    let Type::Path(type_path) = item.ty.as_ref() else {
        return None;
    };
    if type_path.path.segments.last()?.ident != "bool" {
        return None;
    }
    let Expr::Lit(lit) = item.expr.as_ref() else {
        return None;
    };
    let Lit::Bool(bool_lit) = &lit.lit else {
        return None;
    };
    Some(bool_lit.value)
}

/// Return a copy of `item` whose parameter types have aliases expanded.
pub fn normalize_item_fn_param_type_aliases(
    item: &ItemFn,
    aliases: &HashMap<String, ItemType>,
) -> Result<ItemFn, String> {
    let mut item = item.clone();
    for arg in &mut item.sig.inputs {
        let FnArg::Typed(PatType { ty, .. }) = arg else {
            continue;
        };
        **ty = normalize_type_aliases(ty, aliases)?;
    }
    Ok(item)
}

struct ConstPathNormalizer<'a> {
    consts: &'a HashMap<String, ItemConst>,
}

impl VisitMut for ConstPathNormalizer<'_> {
    fn visit_generic_argument_mut(&mut self, arg: &mut GenericArgument) {
        if let GenericArgument::Const(expr) = arg {
            self.visit_expr_mut(expr);
        } else {
            visit_mut::visit_generic_argument_mut(self, arg);
        }
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Path(path) = expr {
            if let Some(name) = single_segment_expr_ident(path) {
                if let Some(replacement) = self.consts.get(&name).and_then(const_item_scalar_expr) {
                    *expr = replacement;
                    return;
                }
            }
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}

fn normalize_type_aliases_inner(
    ty: &Type,
    aliases: &HashMap<String, ItemType>,
    stack: &mut Vec<String>,
) -> Result<Type, String> {
    if stack.len() > MAX_ALIAS_EXPANSION_DEPTH {
        return Err("type alias expansion exceeded recursion limit".to_string());
    }

    match ty {
        Type::Path(type_path) => normalize_path_type(type_path, aliases, stack),
        Type::Reference(type_ref) => {
            let mut out = type_ref.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Reference(out))
        }
        Type::Ptr(type_ptr) => {
            let mut out = type_ptr.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Ptr(out))
        }
        Type::Tuple(tuple) => {
            let mut out = tuple.clone();
            for elem in &mut out.elems {
                *elem = normalize_type_aliases_inner(elem, aliases, stack)?;
            }
            Ok(Type::Tuple(out))
        }
        Type::Array(array) => {
            let mut out = array.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Array(out))
        }
        Type::Slice(slice) => {
            let mut out = slice.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Slice(out))
        }
        Type::Paren(paren) => {
            let mut out = paren.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Paren(out))
        }
        Type::Group(group) => {
            let mut out = group.clone();
            *out.elem = normalize_type_aliases_inner(&out.elem, aliases, stack)?;
            Ok(Type::Group(out))
        }
        _ => Ok(ty.clone()),
    }
}

fn normalize_path_type(
    type_path: &TypePath,
    aliases: &HashMap<String, ItemType>,
    stack: &mut Vec<String>,
) -> Result<Type, String> {
    let Some(last_segment) = type_path.path.segments.last() else {
        return Ok(Type::Path(type_path.clone()));
    };
    let alias_name = last_segment.ident.to_string();

    if let Some(alias) = aliases.get(&alias_name) {
        if stack.contains(&alias_name) {
            return Err(format!(
                "recursive type alias expansion involving `{}`",
                alias_name
            ));
        }
        stack.push(alias_name.clone());
        let subst = build_alias_substitution(alias, &last_segment.arguments)?;
        let substituted = substitute_type(&alias.ty, &subst, aliases, stack)?;
        let expanded = normalize_type_aliases_inner(&substituted, aliases, stack)?;
        stack.pop();
        return Ok(expanded);
    }

    let mut out = type_path.clone();
    if let Some(last_segment) = out.path.segments.last_mut() {
        if let PathArguments::AngleBracketed(args) = &mut last_segment.arguments {
            for arg in &mut args.args {
                *arg = substitute_generic_arg(arg, &AliasSubstitution::default(), aliases, stack)?;
            }
        }
    }
    Ok(Type::Path(out))
}

fn build_alias_substitution(
    alias: &ItemType,
    args: &PathArguments,
) -> Result<AliasSubstitution, String> {
    let actual_args: Vec<GenericArgument> = match args {
        PathArguments::AngleBracketed(args) => args.args.iter().cloned().collect(),
        PathArguments::None => Vec::new(),
        PathArguments::Parenthesized(_) => {
            return Err(format!(
                "type alias `{}` uses unsupported parenthesized generic arguments",
                alias.ident
            ));
        }
    };
    let formals: Vec<&GenericParam> = alias
        .generics
        .params
        .iter()
        .filter(|param| !matches!(param, GenericParam::Lifetime(_)))
        .collect();

    if formals.len() != actual_args.len() {
        return Err(format!(
            "type alias `{}` expects {} generic argument(s), got {}",
            alias.ident,
            formals.len(),
            actual_args.len()
        ));
    }

    let mut subst = AliasSubstitution::default();
    for (formal, actual) in formals.into_iter().zip(actual_args) {
        match (formal, actual) {
            (GenericParam::Type(param), GenericArgument::Type(ty)) => {
                subst.types.insert(param.ident.to_string(), ty);
            }
            (GenericParam::Const(param), GenericArgument::Const(expr)) => {
                subst.consts.insert(param.ident.to_string(), expr);
            }
            (GenericParam::Const(param), GenericArgument::Type(Type::Path(path))) => {
                subst
                    .consts
                    .insert(param.ident.to_string(), expr_path_from_type_path(&path));
            }
            (GenericParam::Type(param), other) => {
                return Err(format!(
                    "type alias `{}` expected type argument for `{}`, got `{}`",
                    alias.ident,
                    param.ident,
                    other.to_token_stream()
                ));
            }
            (GenericParam::Const(param), other) => {
                return Err(format!(
                    "type alias `{}` expected const argument for `{}`, got `{}`",
                    alias.ident,
                    param.ident,
                    other.to_token_stream()
                ));
            }
            (GenericParam::Lifetime(_), _) => {}
        }
    }
    Ok(subst)
}

fn substitute_type(
    ty: &Type,
    subst: &AliasSubstitution,
    aliases: &HashMap<String, ItemType>,
    stack: &mut Vec<String>,
) -> Result<Type, String> {
    match ty {
        Type::Path(type_path) => {
            if let Some(name) = single_segment_ident(type_path) {
                if let Some(replacement) = subst.types.get(&name) {
                    return normalize_type_aliases_inner(replacement, aliases, stack);
                }
            }
            let mut out = type_path.clone();
            if let Some(last_segment) = out.path.segments.last_mut() {
                if let PathArguments::AngleBracketed(args) = &mut last_segment.arguments {
                    for arg in &mut args.args {
                        *arg = substitute_generic_arg(arg, subst, aliases, stack)?;
                    }
                }
            }
            normalize_type_aliases_inner(&Type::Path(out), aliases, stack)
        }
        Type::Reference(type_ref) => {
            let mut out = type_ref.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            Ok(Type::Reference(out))
        }
        Type::Ptr(type_ptr) => {
            let mut out = type_ptr.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            Ok(Type::Ptr(out))
        }
        Type::Tuple(tuple) => {
            let mut out = tuple.clone();
            for elem in &mut out.elems {
                *elem = substitute_type(elem, subst, aliases, stack)?;
            }
            Ok(Type::Tuple(out))
        }
        Type::Array(array) => {
            let mut out = array.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            substitute_expr_in_place(&mut out.len, subst);
            Ok(Type::Array(out))
        }
        Type::Slice(slice) => {
            let mut out = slice.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            Ok(Type::Slice(out))
        }
        Type::Paren(paren) => {
            let mut out = paren.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            Ok(Type::Paren(out))
        }
        Type::Group(group) => {
            let mut out = group.clone();
            *out.elem = substitute_type(&out.elem, subst, aliases, stack)?;
            Ok(Type::Group(out))
        }
        _ => Ok(ty.clone()),
    }
}

fn substitute_generic_arg(
    arg: &GenericArgument,
    subst: &AliasSubstitution,
    aliases: &HashMap<String, ItemType>,
    stack: &mut Vec<String>,
) -> Result<GenericArgument, String> {
    match arg {
        GenericArgument::Type(Type::Path(path)) => {
            if let Some(name) = single_segment_ident(path) {
                if let Some(expr) = subst.consts.get(&name) {
                    return Ok(generic_arg_from_const_expr(expr));
                }
            }
            Ok(GenericArgument::Type(substitute_type(
                &Type::Path(path.clone()),
                subst,
                aliases,
                stack,
            )?))
        }
        GenericArgument::Type(ty) => Ok(GenericArgument::Type(substitute_type(
            ty, subst, aliases, stack,
        )?)),
        GenericArgument::Const(expr) => {
            let mut expr = expr.clone();
            substitute_expr_in_place(&mut expr, subst);
            Ok(GenericArgument::Const(expr))
        }
        _ => Ok(arg.clone()),
    }
}

fn substitute_expr_in_place(expr: &mut Expr, subst: &AliasSubstitution) {
    struct Substituter<'a> {
        consts: &'a HashMap<String, Expr>,
    }

    impl VisitMut for Substituter<'_> {
        fn visit_expr_mut(&mut self, expr: &mut Expr) {
            if let Expr::Path(path) = expr {
                if let Some(name) = single_segment_expr_ident(path) {
                    if let Some(replacement) = self.consts.get(&name) {
                        *expr = replacement.clone();
                        return;
                    }
                }
            }
            visit_mut::visit_expr_mut(self, expr);
        }
    }

    Substituter {
        consts: &subst.consts,
    }
    .visit_expr_mut(expr);
}

fn single_segment_ident(type_path: &TypePath) -> Option<String> {
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    Some(type_path.path.segments.first()?.ident.to_string())
}

fn single_segment_expr_ident(expr_path: &ExprPath) -> Option<String> {
    if expr_path.qself.is_some() || expr_path.path.segments.len() != 1 {
        return None;
    }
    Some(expr_path.path.segments.first()?.ident.to_string())
}

fn expr_path_from_type_path(type_path: &TypePath) -> Expr {
    let qself = type_path.qself.clone();
    let path = type_path.path.clone();
    Expr::Path(ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    })
}

fn generic_arg_from_const_expr(expr: &Expr) -> GenericArgument {
    if let Expr::Path(path) = expr {
        return GenericArgument::Type(Type::Path(TypePath {
            qself: path.qself.clone(),
            path: path.path.clone(),
        }));
    }
    GenericArgument::Const(expr.clone())
}
