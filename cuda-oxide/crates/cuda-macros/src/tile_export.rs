/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[tile_export]`: export a device function to cuTile through a tile
//! library.
//!
//! The attribute leaves the annotated function alone and adds two items next
//! to it: a `#[device] extern "C"` wrapper on the native SIMT ABI, and a
//! host-only `#[no_mangle]` metadata function whose body is the JSON
//! [`TileExportDesc`] of the export. The ABI rules live in `oxide-tile-export`;
//! this module maps Rust signatures onto them and emits the wrapper body.

use crate::common::{internal_ident, reserved_name_error, track_codegen_environment};
use oxide_tile_export::{
    CURRENT_VERSION, TileAccess, TileElem, TileExportDesc, TileExportError, TileParamDesc,
    TileSync, is_identifier,
};
use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use reserved_oxide_symbols::{RESERVED_ROOT, tile_export_meta_symbol};
use syn::{
    Expr, ExprArray, ExprLit, FnArg, Ident, ItemFn, Lit, LitInt, LitStr, Pat, ReturnType,
    Signature, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// The Rust element types `#[tile_export]` accepts, as the diagnostics list
/// them.
const ELEMENT_TYPES: &str = "f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64";

/// The attribute's parsed arguments. Every field is optional at parse time;
/// `expand_tile_export` reports the missing `tile_shape`.
#[derive(Default)]
pub(crate) struct TileExportArgs {
    pub(crate) tile_shape: Option<LitInt>,
    pub(crate) name: Option<LitStr>,
    pub(crate) symbol: Option<LitStr>,
    pub(crate) sync: Option<LitStr>,
    pub(crate) arch: Option<Vec<LitStr>>,
}

fn set_once<T>(slot: &mut Option<T>, key: &Ident, value: T) -> syn::Result<()> {
    if slot.is_some() {
        return Err(syn::Error::new(
            key.span(),
            format!("duplicate #[tile_export] argument `{key}`"),
        ));
    }
    *slot = Some(value);
    Ok(())
}

fn string_argument(key: &Ident, value: Expr) -> syn::Result<LitStr> {
    match value {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit), ..
        }) => Ok(lit),
        other => Err(syn::Error::new_spanned(
            other,
            format!("#[tile_export] `{key}` takes a string literal such as `{key} = \"...\"`"),
        )),
    }
}

impl Parse for TileExportArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "tile_shape" => {
                    let value: Expr = input.parse()?;
                    let lit = match value {
                        Expr::Lit(ExprLit {
                            lit: Lit::Int(lit), ..
                        }) => lit,
                        other => {
                            return Err(syn::Error::new_spanned(
                                other,
                                "#[tile_export] `tile_shape` must be an integer literal such as \
                                 `tile_shape = 128`; a static tile library cannot ship a \
                                 parametric implementation, so a constant or expression is \
                                 not accepted",
                            ));
                        }
                    };
                    set_once(&mut args.tile_shape, &key, lit)?;
                }
                "name" => {
                    let value = string_argument(&key, input.parse()?)?;
                    set_once(&mut args.name, &key, value)?;
                }
                "symbol" => {
                    let value = string_argument(&key, input.parse()?)?;
                    set_once(&mut args.symbol, &key, value)?;
                }
                "sync" => {
                    let value = string_argument(&key, input.parse()?)?;
                    set_once(&mut args.sync, &key, value)?;
                }
                "arch" => {
                    let array: ExprArray = input.parse().map_err(|err| {
                        syn::Error::new(
                            err.span(),
                            "#[tile_export] `arch` takes a list of string literals such as \
                             `arch = [\"sm_100\", \"sm_120\"]`",
                        )
                    })?;
                    let mut entries = Vec::with_capacity(array.elems.len());
                    for elem in array.elems {
                        match elem {
                            Expr::Lit(ExprLit {
                                lit: Lit::Str(lit), ..
                            }) => entries.push(lit),
                            other => {
                                return Err(syn::Error::new_spanned(
                                    other,
                                    "#[tile_export] `arch` entries must be string literals \
                                     such as \"sm_120\"",
                                ));
                            }
                        }
                    }
                    set_once(&mut args.arch, &key, entries)?;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[tile_export] argument `{other}`; accepted arguments \
                             are tile_shape, name, symbol, sync, arch"
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

pub(crate) fn tile_export_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    let args = parse_macro_input!(attr as TileExportArgs);
    let input = parse_macro_input!(item as ItemFn);
    match expand_tile_export(args, input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// How one Rust parameter travels across the native ABI.
enum ParamShape {
    /// `x: T`: `in`, by value.
    Scalar(TileElem),
    /// `x: &T`: `in`, behind `*const T`.
    Ref(TileElem),
    /// `x: &[T; F]`: `in` with footprint `F`, behind `*const T`.
    RefArray(TileElem, u32, LitInt),
    /// `x: &mut T`: `inout`, behind `*mut T`.
    RefMut(TileElem),
    /// `x: &mut [T; F]`: `inout` with footprint `F`, behind `*mut T`.
    RefMutArray(TileElem, u32, LitInt),
}

impl ParamShape {
    fn elem(&self) -> TileElem {
        match self {
            ParamShape::Scalar(elem)
            | ParamShape::Ref(elem)
            | ParamShape::RefArray(elem, _, _)
            | ParamShape::RefMut(elem)
            | ParamShape::RefMutArray(elem, _, _) => *elem,
        }
    }

    fn access(&self) -> TileAccess {
        match self {
            ParamShape::Scalar(_) | ParamShape::Ref(_) | ParamShape::RefArray(..) => TileAccess::In,
            ParamShape::RefMut(_) | ParamShape::RefMutArray(..) => TileAccess::InOut,
        }
    }

    fn footprint(&self) -> u32 {
        match self {
            ParamShape::RefArray(_, footprint, _) | ParamShape::RefMutArray(_, footprint, _) => {
                *footprint
            }
            _ => 1,
        }
    }

    fn by_value(&self) -> bool {
        matches!(self, ParamShape::Scalar(_))
    }
}

/// A type as the user would write it, for diagnostics: token printing puts
/// spaces around `&`, `;`, and `::` that no one types.
fn type_text(ty: &Type) -> String {
    ty.to_token_stream()
        .to_string()
        .replace("& mut ", "&mut ")
        .replace("& ", "&")
        .replace(" ;", ";")
        .replace(" ::", "::")
        .replace(":: ", "::")
        .replace(" <", "<")
        .replace("< ", "<")
        .replace(" >", ">")
        .replace(" ,", ",")
}

/// Peel `(T)` and macro-generated groups so a type written through another
/// macro classifies like one written by hand.
fn unwrap_type(mut ty: &Type) -> &Type {
    loop {
        match ty {
            Type::Paren(paren) => ty = &paren.elem,
            Type::Group(group) => ty = &group.elem,
            _ => return ty,
        }
    }
}

/// The element type a scalar type path names, if it is one of the eleven.
///
/// Accepts the bare name (`f32`) and the primitive module paths
/// (`core::primitive::f32`, `::std::primitive::f32`). Anything with generic
/// arguments or any other path is not an element type.
fn scalar_elem(ty: &Type) -> Option<TileElem> {
    let Type::Path(path) = unwrap_type(ty) else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let segments = &path.path.segments;
    if segments.iter().any(|segment| !segment.arguments.is_none()) {
        return None;
    }
    let last = segments.last()?;
    match segments.len() {
        1 => {}
        3 => {
            let crate_name = segments[0].ident.to_string();
            if !matches!(crate_name.as_str(), "core" | "std") || segments[1].ident != "primitive" {
                return None;
            }
        }
        _ => return None,
    }
    TileElem::from_rust_name(&last.ident.to_string())
}

fn classify_param_type(ty: &Type) -> syn::Result<ParamShape> {
    let unsupported = || {
        syn::Error::new_spanned(
            ty,
            format!(
                "#[tile_export] parameters must be `x: T`, `x: &T`, `x: &mut T`, \
                 `x: &[T; N]`, or `x: &mut [T; N]` with T one of {ELEMENT_TYPES} and N an \
                 integer literal; found `{}`. Slices, raw pointers, tuples, structs, \
                 generics, and other element types are not accepted",
                type_text(ty)
            ),
        )
    };
    if let Some(elem) = scalar_elem(ty) {
        return Ok(ParamShape::Scalar(elem));
    }
    let Type::Reference(reference) = unwrap_type(ty) else {
        return Err(unsupported());
    };
    let mutable = reference.mutability.is_some();
    let pointee = unwrap_type(&reference.elem);
    if let Some(elem) = scalar_elem(pointee) {
        return Ok(if mutable {
            ParamShape::RefMut(elem)
        } else {
            ParamShape::Ref(elem)
        });
    }
    let Type::Array(array) = pointee else {
        return Err(unsupported());
    };
    let Some(elem) = scalar_elem(&array.elem) else {
        return Err(unsupported());
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Int(len), ..
    }) = &array.len
    else {
        return Err(syn::Error::new_spanned(
            &array.len,
            "#[tile_export] array parameters need an integer literal length such as \
             `&[f32; 4]` or `&mut [f32; 4]`; the footprint must be known to the macro, so \
             a constant or expression is not accepted",
        ));
    };
    let footprint: u32 = len.base10_parse()?;
    if footprint == 0 {
        return Err(syn::Error::new_spanned(
            len,
            "#[tile_export] array parameters need at least one element",
        ));
    }
    Ok(if mutable {
        ParamShape::RefMutArray(elem, footprint, len.clone())
    } else {
        ParamShape::RefArray(elem, footprint, len.clone())
    })
}

fn classify_return_type(output: &ReturnType) -> syn::Result<Option<TileElem>> {
    let ty = match output {
        ReturnType::Default => return Ok(None),
        ReturnType::Type(_, ty) => ty,
    };
    if let Type::Tuple(tuple) = unwrap_type(ty)
        && tuple.elems.is_empty()
    {
        return Ok(None);
    }
    scalar_elem(ty).map(Some).ok_or_else(|| {
        syn::Error::new_spanned(
            ty,
            format!(
                "#[tile_export] functions must return a scalar (one of {ELEMENT_TYPES}) or \
                 nothing; found `{}`. References, pointers, arrays, tuples, and other \
                 element types are not accepted as return types",
                type_text(ty)
            ),
        )
    })
}

/// Reject a declaration or symbol name that cannot be a Tile IR symbol or
/// that lands in the reserved namespace.
fn check_symbol_literal(lit: &LitStr, what: &str) -> syn::Result<String> {
    let value = lit.value();
    if !is_identifier(&value) {
        return Err(syn::Error::new_spanned(
            lit,
            format!(
                "#[tile_export] {what} `{value}` must be an identifier (ASCII letters, digits \
                 and `_`, not starting with a digit)"
            ),
        ));
    }
    if value.starts_with(RESERVED_ROOT) {
        return Err(syn::Error::new_spanned(
            lit,
            format!(
                "#[tile_export] {what} `{value}` starts with the reserved cuda-oxide prefix \
                 `{RESERVED_ROOT}`; choose another name"
            ),
        ));
    }
    Ok(value)
}

/// A shared-rule violation as a diagnostic, pointed at the part of the
/// signature that breaks the rule: the parameter whose footprint differs, the
/// shape a warp export cannot split, or the function name for the rest.
fn validation_error(
    err: TileExportError,
    sig: &Signature,
    shape_lit: &LitInt,
    fn_ident: &Ident,
) -> syn::Error {
    let message = format!("#[tile_export]: {err}");
    match &err {
        TileExportError::MixedFootprints { other, .. } => {
            let offending = sig.inputs.iter().find_map(|arg| match arg {
                FnArg::Typed(pat_type) => match &*pat_type.pat {
                    Pat::Ident(pat_ident) if pat_ident.ident == other => Some(&pat_type.ty),
                    _ => None,
                },
                FnArg::Receiver(_) => None,
            });
            match offending {
                Some(ty) => syn::Error::new_spanned(ty, message),
                None => syn::Error::new(fn_ident.span(), message),
            }
        }
        TileExportError::WarpFootprintShape { .. } | TileExportError::WarpShape { .. } => {
            syn::Error::new_spanned(shape_lit, message)
        }
        _ => syn::Error::new(fn_ident.span(), message),
    }
}

/// Expand `#[tile_export(args)] fn ...` into the original function, the
/// `#[device] extern "C"` wrapper, and the metadata function.
pub(crate) fn expand_tile_export(args: TileExportArgs, input: ItemFn) -> syn::Result<TokenStream2> {
    let sig = &input.sig;
    let fn_ident = sig.ident.clone();
    if let Some(err) = reserved_name_error(&fn_ident) {
        return Err(err);
    }
    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "#[tile_export] functions cannot be generic: lifetime, type, and const \
             parameters are not accepted because a static tile library needs one concrete \
             symbol per export",
        ));
    }
    if let Some(asyncness) = &sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "#[tile_export] functions cannot be async; only plain `fn` items run on the device",
        ));
    }
    if let Some(variadic) = &sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "#[tile_export] functions cannot be variadic",
        ));
    }

    let Some(shape_lit) = args.tile_shape else {
        return Err(syn::Error::new(
            fn_ident.span(),
            "#[tile_export] requires `tile_shape = N`, the 1-D tile extent; a static tile library \
             cannot ship a parametric implementation, so every export names one extent",
        ));
    };
    let shape: i64 = shape_lit.base10_parse()?;
    if shape <= 0 {
        return Err(syn::Error::new_spanned(
            &shape_lit,
            format!("#[tile_export] `tile_shape` must be a positive integer, found {shape}"),
        ));
    }
    let sync = match &args.sync {
        None => TileSync::None,
        Some(lit) => match lit.value().as_str() {
            "none" => TileSync::None,
            "warp" => TileSync::Warp,
            other => {
                return Err(syn::Error::new_spanned(
                    lit,
                    format!(
                        "#[tile_export] `sync` must be \"none\" or \"warp\", found \"{other}\""
                    ),
                ));
            }
        },
    };
    // The footprint-1 form of the warp rule, reported before the parameters
    // are read so the message points at the shape. The footprint-aware form
    // comes from `validate` once the parameters are known.
    if sync == TileSync::Warp && shape % 32 != 0 {
        return Err(syn::Error::new_spanned(
            &shape_lit,
            format!(
                "#[tile_export] sync = \"warp\" needs a tile_shape that is a multiple of 32 so \
                 every invocation group fills whole warps, found {shape}"
            ),
        ));
    }
    let decl = match &args.name {
        Some(lit) => check_symbol_literal(lit, "name")?,
        None => fn_ident.to_string(),
    };
    let symbol = match &args.symbol {
        Some(lit) => check_symbol_literal(lit, "symbol")?,
        None => format!("{decl}_impl"),
    };
    let arch: Vec<String> = args
        .arch
        .iter()
        .flatten()
        .map(|lit| {
            let value = lit.value();
            if value.trim().is_empty() {
                Err(syn::Error::new_spanned(
                    lit,
                    "#[tile_export] `arch` entries must be non-empty names such as \"sm_120\"",
                ))
            } else {
                Ok(value)
            }
        })
        .collect::<syn::Result<_>>()?;

    // Parameters: plain identifiers with one of the four accepted types.
    let mut shapes = Vec::with_capacity(sig.inputs.len());
    let mut params = Vec::with_capacity(sig.inputs.len() + 1);
    let mut types = Vec::with_capacity(sig.inputs.len());
    for arg in &sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[tile_export] functions cannot take self; export a free function",
            ));
        };
        if let Some(attr) = pat_type.attrs.first() {
            return Err(syn::Error::new_spanned(
                attr,
                "#[tile_export] parameters cannot carry attributes",
            ));
        }
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "#[tile_export] parameters must be plain identifiers such as `x: f32`; \
                 patterns are not accepted because each parameter names one Tile IR argument",
            ));
        };
        if pat_ident.subpat.is_some() || pat_ident.by_ref.is_some() {
            return Err(syn::Error::new_spanned(
                pat_ident,
                "#[tile_export] parameters must be plain identifiers such as `x: f32`; \
                 `ref` bindings and subpatterns are not accepted",
            ));
        }
        let name = pat_ident.ident.to_string();
        if !is_identifier(&name) {
            return Err(syn::Error::new_spanned(
                &pat_ident.ident,
                format!(
                    "#[tile_export] parameter `{name}` must be an ASCII identifier without \
                     the `r#` prefix; it names a Tile IR argument"
                ),
            ));
        }
        let shape = classify_param_type(&pat_type.ty)?;
        params.push(TileParamDesc {
            name,
            elem: shape.elem(),
            access: shape.access(),
            footprint: shape.footprint(),
            by_value: shape.by_value(),
        });
        types.push((*pat_type.ty).clone());
        shapes.push(shape);
    }

    let return_elem = classify_return_type(&sig.output)?;
    if return_elem.is_none() && !params.iter().any(|param| param.access.is_output()) {
        return Err(syn::Error::new_spanned(
            sig,
            "#[tile_export] function has no output: return a scalar value or take at least \
             one `&mut T` or `&mut [T; N]` parameter, because a tile library declaration's \
             results are its `out` and `inout` arguments",
        ));
    }
    if let Some(elem) = return_elem {
        // The declaration argument that carries the return value. Keep the
        // conventional `<fn>_out` unless a parameter already took it.
        let mut out_name = format!("{fn_ident}_out");
        while params.iter().any(|param| param.name == out_name) {
            out_name.push('_');
        }
        params.push(TileParamDesc {
            name: out_name,
            elem,
            access: TileAccess::Out,
            footprint: 1,
            by_value: false,
        });
    }

    let desc = TileExportDesc {
        version: CURRENT_VERSION,
        decl,
        symbol,
        shape: vec![shape],
        params,
        sync,
        arch,
    };
    desc.validate()
        .map_err(|err| validation_error(err, sig, &shape_lit, &fn_ident))?;

    // The wrapper: pointer parameters are turned back into the references
    // the original function takes, the return value is stored through the
    // trailing out pointer.
    let mut abi_inputs = Vec::with_capacity(shapes.len() + 1);
    let mut call_args = Vec::with_capacity(shapes.len());
    for (index, (shape, ty)) in shapes.iter().zip(&types).enumerate() {
        let ident = internal_ident(&format!("__cuda_oxide_tile_arg_{index}"));
        match shape {
            ParamShape::Scalar(_) => {
                abi_inputs.push(quote! { #ident: #ty });
                call_args.push(quote! { #ident });
            }
            ParamShape::Ref(_) => {
                let Type::Reference(reference) = unwrap_type(ty) else {
                    unreachable!("classified as a reference");
                };
                let pointee = &reference.elem;
                abi_inputs.push(quote! { #ident: *const #pointee });
                call_args.push(quote! { unsafe { &*#ident } });
            }
            ParamShape::RefMut(_) => {
                let Type::Reference(reference) = unwrap_type(ty) else {
                    unreachable!("classified as a reference");
                };
                let pointee = &reference.elem;
                abi_inputs.push(quote! { #ident: *mut #pointee });
                call_args.push(quote! { unsafe { &mut *#ident } });
            }
            ParamShape::RefArray(_, _, len) => {
                let Type::Reference(reference) = unwrap_type(ty) else {
                    unreachable!("classified as a reference");
                };
                let Type::Array(array) = unwrap_type(&reference.elem) else {
                    unreachable!("classified as an array reference");
                };
                let elem_ty = &array.elem;
                abi_inputs.push(quote! { #ident: *const #elem_ty });
                call_args.push(quote! { unsafe { &*(#ident as *const [#elem_ty; #len]) } });
            }
            ParamShape::RefMutArray(_, _, len) => {
                let Type::Reference(reference) = unwrap_type(ty) else {
                    unreachable!("classified as a reference");
                };
                let Type::Array(array) = unwrap_type(&reference.elem) else {
                    unreachable!("classified as an array reference");
                };
                let elem_ty = &array.elem;
                abi_inputs.push(quote! { #ident: *mut #elem_ty });
                call_args.push(quote! { unsafe { &mut *(#ident as *mut [#elem_ty; #len]) } });
            }
        }
    }
    let call = quote! { #fn_ident(#(#call_args),*) };
    let body = match &sig.output {
        ReturnType::Type(_, ret_ty) if return_elem.is_some() => {
            let out_ident = internal_ident("__cuda_oxide_tile_out");
            let result_ident = internal_ident("__cuda_oxide_tile_result");
            abi_inputs.push(quote! { #out_ident: *mut #ret_ty });
            quote! {
                let #result_ident: #ret_ty = #call;
                unsafe { *#out_ident = #result_ident; }
            }
        }
        _ => quote! { #call; },
    };

    let wrapper_ident = Ident::new(&desc.symbol, fn_ident.span());
    let meta_ident = Ident::new(&tile_export_meta_symbol(&desc.symbol), fn_ident.span());
    let json = LitStr::new(&desc.to_json(), Span::call_site());
    let wrapper_doc = format!(
        "Tile library implementation of `{}`, on the native SIMT ABI.",
        desc.decl
    );

    Ok(quote! {
        #input

        #[doc = #wrapper_doc]
        #[doc(hidden)]
        #[allow(non_snake_case, clippy::missing_safety_doc, improper_ctypes_definitions)]
        #[::cuda_device::device]
        pub unsafe extern "C" fn #wrapper_ident(#(#abi_inputs),*) {
            #body
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        #[unsafe(no_mangle)]
        pub fn #meta_ident() -> &'static str {
            #json
        }
    })
}
