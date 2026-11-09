/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Declarations of foreign tile functions: functions a tile library
//! (`.tilelib`) implements and kernels call through Tile IR `preview$call`.
//!
//! A declaration is a body-less function inside a `#[cutile::module]`,
//! written like the library's Tile IR declaration:
//!
//! ```rust,ignore
//! /// # Safety
//! ///
//! /// The declaration matches `solver.tilelib`'s `cholesky`.
//! #[cutile::foreign]
//! pub unsafe fn cholesky<const Batch: i32, const M: i32, T: ElementType>(
//!     Function: &str,
//!     FillMode: &str,
//!     #[inout] A: Tile<T, { [Batch, M, M] }>,
//! ) -> (Tile<T, { [Batch, M, M] }>, Tile<i32, { [Batch] }>);
//! ```
//!
//! - The function is `unsafe fn` and calls are in `unsafe` blocks: the
//!   compiler cannot check the library, so the caller guarantees that the
//!   declaration matches it.
//! - The constant arguments of the call are the `&str` parameters (each a
//!   `string`, whose value the JIT resolves when it compiles the call),
//!   then the generic parameters (`const N: i32` an `index`,
//!   `T: ElementType` a `Datatype`), each group in order.
//!   `const_args = "Function, FillMode, Batch, M, T"` gives another order.
//! - Every other parameter is an operand the caller passes: read-only by
//!   default, read-write with `#[inout]`. Scalars are rank-0 tiles.
//! - The results start with the updated `#[inout]` operands. Every further
//!   result is an output-only operand the caller does not pass: the call
//!   appends a zero-filled operand of each, as a Tile C++
//!   `__experimental_foreign_decl__` declaration does.
//!
//! The macro parses the declaration to emit a rustc stub; the JIT parses
//! the same declaration to lower calls. Both use this module.

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::{
    Attribute, FnArg, ForeignItemFn, GenericParam, Item, ItemFn, ItemMod, LitStr, Pat, ReturnType,
    Signature, Type, TypeParamBound,
};

/// Last path segment of the declaration attribute, `#[cutile::foreign(..)]`.
const FOREIGN_ATTR: &str = "foreign";
/// Parameter attribute marking a read-write operand.
const INOUT_ATTR: &str = "inout";
/// Trait bound of a `Datatype` constant parameter.
const DATATYPE_BOUND: &str = "ElementType";

/// How the callee uses an operand the caller passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Read only.
    In,
    /// Read and overwritten; the updated value is returned.
    InOut,
}

/// Kind of a constant parameter, the declaration's `const_params` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstParamKind {
    /// `index`, from `const N: i32`.
    Int,
    /// `!cuda_tile<preview$Datatype>`, from `T: ElementType`.
    Datatype,
    /// `!cuda_tile<preview$string>`, from a `name: &str` parameter: the
    /// call argument at this index.
    Str(usize),
}

/// One constant parameter, in the order of the call's constant arguments.
#[derive(Debug, Clone)]
pub struct ConstParam {
    pub name: String,
    pub kind: ConstParamKind,
}

/// One operand the caller passes, in declaration order.
#[derive(Debug, Clone)]
pub struct ForeignParam {
    pub name: String,
    pub ty: Type,
    pub access: Access,
    /// Index of the call argument that supplies the operand.
    pub arg_index: usize,
}

/// A validated foreign function declaration.
#[derive(Debug, Clone)]
pub struct ForeignFnDecl {
    /// The Rust function name.
    pub ident: syn::Ident,
    /// The callee symbol in the tile library (`name = ".."`, default the
    /// Rust name).
    pub symbol: String,
    /// The constant parameters, in the order of the call's constant
    /// arguments.
    pub const_params: Vec<ConstParam>,
    /// The operands, in declaration order; `&str` parameters are constants,
    /// not operands.
    pub params: Vec<ForeignParam>,
    /// The declared results, in order.
    pub results: Vec<Type>,
}

impl ForeignFnDecl {
    /// Parses and validates a body-less declaration (`syn::Item::Verbatim`
    /// inside a module) into a function with a never-run body. `Ok(None)`
    /// when the tokens are not a function carrying the foreign attribute.
    pub fn parse_verbatim(tokens: &TokenStream) -> syn::Result<Option<ItemFn>> {
        let Ok(decl) = syn::parse2::<ForeignItemFn>(tokens.clone()) else {
            return Ok(None);
        };
        if find_attr(&decl.attrs, FOREIGN_ATTR).is_none() {
            return Ok(None);
        }
        let item = ItemFn {
            attrs: decl.attrs,
            vis: decl.vis,
            sig: decl.sig,
            block: Box::new(unreachable_body()),
        };
        Self::from_item_fn(&item)?;
        Ok(Some(item))
    }

    /// Parses a function carrying the foreign attribute; `Ok(None)` without it.
    pub fn from_item_fn(item: &ItemFn) -> syn::Result<Option<Self>> {
        let Some(attr) = find_attr(&item.attrs, FOREIGN_ATTR) else {
            return Ok(None);
        };
        let ForeignAttr { symbol, const_args } = parse_foreign_attr(attr)?;
        let sig = &item.sig;
        let ident = sig.ident.clone();
        let symbol = symbol.unwrap_or_else(|| ident.to_string());
        if sig.unsafety.is_none() {
            return Err(syn::Error::new(
                sig.fn_token.span,
                "a foreign function must be declared `unsafe fn`: the compiler cannot check \
                 the library's implementation, so callers must uphold its contract \
                 (see `Safety` in the Tile Libraries guide)",
            ));
        }
        let generic_params = sig
            .generics
            .params
            .iter()
            .map(const_param)
            .collect::<syn::Result<Vec<_>>>()?;
        let mut string_params = Vec::new();
        let mut params = Vec::new();
        for (arg_index, input) in sig.inputs.iter().enumerate() {
            match foreign_param(input, arg_index)? {
                Param::Operand(param) => params.push(param),
                Param::Str(name) => string_params.push(ConstParam {
                    name,
                    kind: ConstParamKind::Str(arg_index),
                }),
            }
        }
        let const_params = order_const_params(string_params, generic_params, const_args)?;
        let results = match &sig.output {
            ReturnType::Default => vec![],
            ReturnType::Type(_, ty) => match ty.as_ref() {
                Type::Tuple(tuple) => tuple.elems.iter().cloned().collect(),
                ty => vec![ty.clone()],
            },
        };
        for ty in &results {
            check_operand_type(ty, "result")?;
        }
        let decl = Self {
            ident,
            symbol,
            const_params,
            params,
            results,
        };
        decl.check_written_results(sig)?;
        Ok(Some(decl))
    }

    /// The `#[inout]` operands, in order. Their updated values are the first
    /// results.
    pub fn written_params(&self) -> impl Iterator<Item = &ForeignParam> {
        self.params.iter().filter(|p| p.access == Access::InOut)
    }

    /// The results that are implicit output-only operands: after the written
    /// operands. Each is passed zero-filled after the parameters.
    pub fn implicit_out_results(&self) -> &[Type] {
        &self.results[self.written_params().count()..]
    }

    /// `sig` without its `&str` parameters: the operands alone, which is
    /// what generic arguments are inferred from.
    pub fn operand_signature(sig: &Signature) -> Signature {
        let mut operands = sig.clone();
        operands.inputs = sig
            .inputs
            .iter()
            .filter(|input| !matches!(input, FnArg::Typed(p) if is_str_ref(&p.ty)))
            .cloned()
            .collect();
        operands
    }

    fn check_written_results(&self, sig: &Signature) -> syn::Result<()> {
        let written: Vec<_> = self.written_params().collect();
        let output_span = match &sig.output {
            ReturnType::Type(_, ty) => ty.span(),
            ReturnType::Default => sig.ident.span(),
        };
        if self.results.len() < written.len() {
            return Err(syn::Error::new(
                output_span,
                format!(
                    "a foreign function returns the updated value of each `#[inout]` \
                     parameter first; expected at least {} result(s)",
                    written.len()
                ),
            ));
        }
        for (i, (param, result)) in written.iter().zip(&self.results).enumerate() {
            if type_text(&param.ty) != type_text(result) {
                return Err(syn::Error::new(
                    result.span(),
                    format!(
                        "result {i} is the value of the `#[inout]` parameter `{}` and must have \
                         its type",
                        param.name
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// The signature callers see: the declaration without the foreign and
/// parameter attributes, with a body that is never run. The macro emits it
/// for rustc.
pub fn rustc_stub(item: &ItemFn) -> ItemFn {
    let mut stub = item.clone();
    stub.attrs.retain(|a| !attr_is(a, FOREIGN_ATTR));
    // Declarations keep the library's names (`Batch`, `A`).
    stub.attrs.push(syn::parse_quote!(
        #[allow(unused_variables, non_snake_case, non_upper_case_globals)]
    ));
    for input in stub.sig.inputs.iter_mut() {
        if let FnArg::Typed(pat) = input {
            pat.attrs.retain(|a| !a.path().is_ident(INOUT_ATTR));
        }
    }
    let message = format!(
        "foreign tile function `{}` can only be called from a cuTile kernel",
        item.sig.ident
    );
    stub.block = Box::new(syn::parse_quote!({ unreachable!(#message) }));
    stub
}

/// Replaces every foreign declaration (`syn::Item::Verbatim`) in `module`,
/// including inline submodules, by an equivalent `ItemFn` that keeps the
/// foreign attribute, so the JIT indexes and type-checks it like any
/// function. Invalid declarations are left in place; the macro has already
/// reported them.
pub fn normalize_module(module: &mut ItemMod) {
    if let Some((_, items)) = &mut module.content {
        normalize_items(items);
    }
}

fn normalize_items(items: &mut [Item]) {
    for item in items.iter_mut() {
        match item {
            Item::Verbatim(tokens) => {
                if let Ok(Some(item_fn)) = ForeignFnDecl::parse_verbatim(tokens) {
                    *item = Item::Fn(item_fn);
                }
            }
            Item::Mod(submodule) => normalize_module(submodule),
            _ => {}
        }
    }
}

/// Whether `item` is a foreign declaration (carries the foreign attribute).
pub fn is_foreign_fn(item: &ItemFn) -> bool {
    find_attr(&item.attrs, FOREIGN_ATTR).is_some()
}

/// Whether `attr`'s path ends in `name` (`#[cutile::name]` or `#[name]`).
fn attr_is(attr: &Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

fn find_attr<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a Attribute> {
    attrs.iter().find(|a| attr_is(a, name))
}

fn unreachable_body() -> syn::Block {
    syn::parse_quote!({ unreachable!() })
}

fn type_text(ty: &Type) -> String {
    ty.to_token_stream().to_string()
}

/// The arguments of `#[cutile::foreign(..)]`.
struct ForeignAttr {
    symbol: Option<String>,
    const_args: Option<LitStr>,
}

fn parse_foreign_attr(attr: &Attribute) -> syn::Result<ForeignAttr> {
    let mut symbol = None;
    let mut const_args = None;
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(ForeignAttr { symbol, const_args });
    }
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("name") {
            symbol = Some(meta.value()?.parse::<LitStr>()?.value());
            Ok(())
        } else if meta.path.is_ident("const_args") {
            const_args = Some(meta.value()?.parse::<LitStr>()?);
            Ok(())
        } else {
            Err(meta.error("expected `name = \"...\"` or `const_args = \"...\"`"))
        }
    })?;
    Ok(ForeignAttr { symbol, const_args })
}

fn const_param(param: &GenericParam) -> syn::Result<ConstParam> {
    match param {
        GenericParam::Const(c) => {
            let is_i32 =
                matches!(&c.ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident("i32"));
            if !is_i32 || c.default.is_some() {
                return Err(syn::Error::new(
                    c.span(),
                    "a constant parameter of a foreign function must be `const N: i32` \
                     (an `index` constant) without a default",
                ));
            }
            Ok(ConstParam {
                name: c.ident.to_string(),
                kind: ConstParamKind::Int,
            })
        }
        GenericParam::Type(t) => {
            let bounds: Vec<_> = t
                .bounds
                .iter()
                .filter_map(|bound| match bound {
                    TypeParamBound::Trait(tb) => tb.path.segments.last().map(|s| s.ident.clone()),
                    _ => None,
                })
                .collect();
            match bounds.as_slice() {
                [bound] if bound == DATATYPE_BOUND && t.default.is_none() => Ok(ConstParam {
                    name: t.ident.to_string(),
                    kind: ConstParamKind::Datatype,
                }),
                _ => Err(syn::Error::new(
                    t.span(),
                    format!(
                        "type parameter `{}` of a foreign function must be bounded by exactly \
                         `{DATATYPE_BOUND}` (a datatype constant), without a default",
                        t.ident
                    ),
                )),
            }
        }
        GenericParam::Lifetime(l) => Err(syn::Error::new(
            l.span(),
            "a foreign function cannot have lifetime parameters",
        )),
    }
}

/// A foreign function parameter: an operand or a string constant.
enum Param {
    Operand(ForeignParam),
    Str(String),
}

/// Whether `ty` is `&str` or `&'static str`.
fn is_str_ref(ty: &Type) -> bool {
    let Type::Reference(r) = ty else {
        return false;
    };
    r.mutability.is_none()
        && r.lifetime.as_ref().is_none_or(|l| l.ident == "static")
        && matches!(r.elem.as_ref(), Type::Path(p) if p.qself.is_none() && p.path.is_ident("str"))
}

/// The constant parameters in call order: `const_args` when given, else
/// the string parameters, then the generic parameters.
fn order_const_params(
    string_params: Vec<ConstParam>,
    generic_params: Vec<ConstParam>,
    const_args: Option<LitStr>,
) -> syn::Result<Vec<ConstParam>> {
    let mut all: Vec<ConstParam> = string_params.into_iter().chain(generic_params).collect();
    let Some(order) = const_args else {
        return Ok(all);
    };
    let names: Vec<String> = order
        .value()
        .split(',')
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect();
    let mut ordered = Vec::with_capacity(all.len());
    for name in &names {
        let Some(at) = all.iter().position(|p| &p.name == name) else {
            let known = ordered.iter().any(|p: &ConstParam| &p.name == name);
            return Err(syn::Error::new(
                order.span(),
                if known {
                    format!("`const_args` lists `{name}` twice")
                } else {
                    format!(
                        "`const_args` lists `{name}`, which is neither a generic parameter nor \
                         a `&str` parameter"
                    )
                },
            ));
        };
        ordered.push(all.remove(at));
    }
    if !all.is_empty() {
        return Err(syn::Error::new(
            order.span(),
            format!(
                "`const_args` must list every constant parameter; missing {}",
                all.iter()
                    .map(|p| format!("`{}`", p.name))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    Ok(ordered)
}

fn foreign_param(input: &FnArg, arg_index: usize) -> syn::Result<Param> {
    let FnArg::Typed(pat_type) = input else {
        return Err(syn::Error::new(
            input.span(),
            "a foreign function cannot take `self`",
        ));
    };
    let Pat::Ident(pat_ident) = pat_type.pat.as_ref() else {
        return Err(syn::Error::new(
            pat_type.pat.span(),
            "a foreign function parameter must be a plain name",
        ));
    };
    let mut access = Access::In;
    for attr in &pat_type.attrs {
        if !matches!(attr.meta, syn::Meta::Path(_)) || !attr.path().is_ident(INOUT_ATTR) {
            return Err(syn::Error::new(
                attr.span(),
                "the only attribute a foreign function parameter accepts is `#[inout]`",
            ));
        }
        access = Access::InOut;
    }
    let name = pat_ident.ident.to_string();
    if is_str_ref(&pat_type.ty) {
        if access == Access::InOut {
            return Err(syn::Error::new(
                pat_type.span(),
                format!("`{name}: &str` is a string constant and cannot be `#[inout]`"),
            ));
        }
        return Ok(Param::Str(name));
    }
    check_operand_type(&pat_type.ty, "parameter")?;
    Ok(Param::Operand(ForeignParam {
        name,
        ty: pat_type.ty.as_ref().clone(),
        access,
        arg_index,
    }))
}

/// Operands and results are tiles (`Tile<E, S>`) or scalars (an element type
/// name). Views, tensors, references and pointers are not supported.
fn check_operand_type(ty: &Type, what: &str) -> syn::Result<()> {
    let ok = match ty {
        Type::Path(p) if p.qself.is_none() => match p.path.segments.last() {
            Some(last) if last.ident == "Tile" => true,
            Some(last) => {
                p.path.segments.len() == 1 && matches!(last.arguments, syn::PathArguments::None)
            }
            None => false,
        },
        Type::Paren(inner) => return check_operand_type(&inner.elem, what),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(syn::Error::new(
            ty.span(),
            format!(
                "a foreign function {what} must be a tile (`Tile<E, S>`) or a scalar; \
                 views, tensors, references and pointers are not supported"
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn parse(tokens: TokenStream) -> syn::Result<Option<ForeignFnDecl>> {
        let item = ForeignFnDecl::parse_verbatim(&tokens)?;
        item.map_or(Ok(None), |item| ForeignFnDecl::from_item_fn(&item))
    }

    #[test]
    fn cholesky_declaration_maps_generics_operands_and_implicit_outs() {
        let decl = parse(quote! {
            #[cutile::foreign]
            pub unsafe fn cholesky<const BATCH: i32, const M: i32, T: ElementType>(
                function: &str,
                fill_mode: &'static str,
                #[inout] a: Tile<T, { [BATCH, M, M] }>,
            ) -> (Tile<T, { [BATCH, M, M] }>, Tile<i32, { [BATCH] }>);
        })
        .unwrap()
        .unwrap();
        assert_eq!(decl.symbol, "cholesky");
        let names: Vec<_> = decl.const_params.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["function", "fill_mode", "BATCH", "M", "T"]);
        assert_eq!(decl.const_params[1].kind, ConstParamKind::Str(1));
        assert_eq!(decl.params[0].arg_index, 2);
        let operands = ForeignFnDecl::operand_signature(
            &syn::parse2::<ItemFn>(quote! {
                unsafe fn f(a: &str, x: f32) {}
            })
            .unwrap()
            .sig,
        );
        assert_eq!(operands.inputs.len(), 1);
        let kinds: Vec<_> = decl.const_params.iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            [
                ConstParamKind::Str(0),
                ConstParamKind::Str(1),
                ConstParamKind::Int,
                ConstParamKind::Int,
                ConstParamKind::Datatype
            ]
        );
        assert_eq!(decl.params.len(), 1);
        assert_eq!(decl.params[0].access, Access::InOut);
        assert_eq!(decl.results.len(), 2);
        assert_eq!(decl.implicit_out_results().len(), 1);
    }

    #[test]
    fn const_args_reorders_the_constant_parameters() {
        let decl = parse(quote! {
            #[cutile::foreign(const_args = "N, mode, T")]
            unsafe fn f<const N: i32, T: ElementType>(mode: &str, x: Tile<T, { [N] }>) -> Tile<T, { [N] }>;
        })
        .unwrap()
        .unwrap();
        let names: Vec<_> = decl.const_params.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["N", "mode", "T"]);
    }

    #[test]
    fn written_operands_come_first_in_the_results() {
        // An `in` operand between written ones; results follow the written order.
        let decl = parse(quote! {
            #[cutile::foreign]
            unsafe fn f(#[inout] a: Tile<i32, { [4] }>, b: f32, #[inout] c: f32) -> (Tile<i32, { [4] }>, f32, i32);
        })
        .unwrap()
        .unwrap();
        let written: Vec<_> = decl.written_params().map(|p| p.name.as_str()).collect();
        assert_eq!(written, ["a", "c"]);
        assert_eq!(decl.implicit_out_results().len(), 1);
    }

    #[test]
    fn name_and_scalar_operands() {
        let decl = parse(quote! {
            #[cutile::foreign(name = "scale_tile")]
            unsafe fn scale<const N: i32, T: ElementType>(x: Tile<T, { [N] }>, alpha: T) -> Tile<T, { [N] }>;
        })
        .unwrap()
        .unwrap();
        assert_eq!(decl.symbol, "scale_tile");
        assert_eq!(decl.params.len(), 2);
        // No `#[inout]`: the result is an output-only operand.
        assert_eq!(decl.implicit_out_results().len(), 1);
    }

    #[test]
    fn a_declaration_without_operands_has_only_output_only_results() {
        let decl = parse(quote! {
            #[cutile::foreign]
            unsafe fn lane_ids(kind: &str) -> Tile<i32, { [32] }>;
        })
        .unwrap()
        .unwrap();
        assert!(decl.params.is_empty());
        assert_eq!(decl.const_params.len(), 1);
        assert_eq!(decl.implicit_out_results().len(), 1);
    }

    #[test]
    fn non_foreign_verbatim_items_are_ignored() {
        assert!(parse(quote! { fn plain(x: i32); }).unwrap().is_none());
        assert!(parse(quote! { struct S; }).unwrap().is_none());
    }

    #[test]
    fn invalid_declarations_are_rejected_with_reasons() {
        for (tokens, reason) in [
            (
                quote! { #[cutile::foreign] fn f(x: f32) -> f32; },
                "must be declared `unsafe fn`",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f<T>(x: T) -> T; },
                "bounded by exactly",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f<const N: usize>(x: f32); },
                "const N: i32",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(x: &Tile<f32, {[4]}>); },
                "tile (`Tile<E, S>`) or a scalar",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(#[restrict] x: f32); },
                "only attribute a foreign function parameter accepts",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(#[inout] x: f32) -> i32; },
                "`#[inout]` parameter `x` and must have its type",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(#[inout] x: f32); },
                "at least 1 result",
            ),
            (
                quote! { #[cutile::foreign(lib = "y")] unsafe fn f(x: f32); },
                "expected `name",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(#[inout] s: &str, x: f32) -> f32; },
                "cannot be `#[inout]`",
            ),
            (
                quote! { #[cutile::foreign] unsafe fn f(s: &mut str, x: f32); },
                "tile (`Tile<E, S>`) or a scalar",
            ),
            (
                quote! { #[cutile::foreign(const_args = "N, s")] unsafe fn f<const N: i32>(s: &str, t: &str, x: f32); },
                "missing `t`",
            ),
            (
                quote! { #[cutile::foreign(const_args = "s, s")] unsafe fn f(s: &str, x: f32); },
                "twice",
            ),
            (
                quote! { #[cutile::foreign(const_args = "q")] unsafe fn f(s: &str, x: f32); },
                "neither a generic parameter",
            ),
        ] {
            let message = parse(tokens).unwrap_err().to_string();
            assert!(message.contains(reason), "{message} (expected: {reason})");
        }
    }

    #[test]
    fn stub_and_normalized_items() {
        let tokens = quote! {
            #[cutile::foreign]
            pub unsafe fn f<const N: i32>(#[inout] x: Tile<f32, { [N] }>, y: f32) -> (Tile<f32, { [N] }>, f32);
        };
        let item = ForeignFnDecl::parse_verbatim(&tokens).unwrap().unwrap();
        let stub = rustc_stub(&item).to_token_stream().to_string();
        assert!(!stub.contains("cutile :: foreign"), "{stub}");
        assert!(!stub.contains("# [inout]"), "{stub}");
        assert!(stub.contains("non_upper_case_globals"), "{stub}");
        assert!(stub.contains("unreachable"), "{stub}");

        let mut module: ItemMod = syn::parse2(quote! {
            mod m {
                #tokens
                mod inner { #tokens }
            }
        })
        .unwrap();
        normalize_module(&mut module);
        let items = &module.content.as_ref().unwrap().1;
        assert!(matches!(&items[0], Item::Fn(f) if find_attr(&f.attrs, FOREIGN_ATTR).is_some()));
        let Item::Mod(inner) = &items[1] else {
            panic!("expected the submodule")
        };
        assert!(matches!(&inner.content.as_ref().unwrap().1[0], Item::Fn(_)));
    }
}
