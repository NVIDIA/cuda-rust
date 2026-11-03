/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant-memory (`#[constant]` statics inside `#[cuda_module]`)
//! support: collection, symbol naming, and loader plumbing.

use crate::common::{attr_path_ends_with, has_attr_named};
use crate::cuda_module::{cuda_module_cfg_attrs, cuda_module_method_attrs};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::constant_symbol;
use syn::{Ident, Item, LitStr, Type, parse_quote};

/// A `#[constant]` static collected from a `#[cuda_module]` body.
pub(super) struct CudaModuleConstant {
    pub(super) ident: Ident,
    pub(super) ty: Box<Type>,
    pub(super) cfg_attrs: Vec<syn::Attribute>,
    pub(super) method_attrs: Vec<syn::Attribute>,
    pub(super) symbol: String,
}

pub(super) fn collect_cuda_module_constants(
    items: &[Item],
    module_ident: &Ident,
) -> syn::Result<Vec<CudaModuleConstant>> {
    let mut constants = Vec::new();
    for item in items {
        let Item::Static(item_static) = item else {
            continue;
        };
        if !has_attr_named(&item_static.attrs, "constant") {
            continue;
        }
        if extract_constant_inner_ty(&item_static.ty).is_none() {
            return Err(syn::Error::new_spanned(
                &item_static.ty,
                concat!(
                    "#[constant] static must have type `ConstantMemory<T>` ",
                    "(e.g. `static FOO: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;`). ",
                    "The wrapper prevents the compiler from constant-folding the initializer into kernel bodies.",
                ),
            ));
        }
        constants.push(CudaModuleConstant {
            ident: item_static.ident.clone(),
            ty: item_static.ty.clone(),
            cfg_attrs: cuda_module_cfg_attrs(&item_static.attrs)?,
            method_attrs: cuda_module_method_attrs(&item_static.attrs),
            symbol: cuda_module_constant_symbol(module_ident, &item_static.ident),
        });
    }
    Ok(constants)
}

fn cuda_module_constant_symbol(module_ident: &Ident, ident: &Ident) -> String {
    let start = ident.span().start();
    let base = format!(
        "{}_L{}C{}_{}",
        module_ident, start.line, start.column, ident
    );
    constant_symbol(&base)
}

pub(super) fn cuda_module_items_with_constant_symbols(
    items: &[Item],
    constants: &[CudaModuleConstant],
) -> Vec<TokenStream2> {
    let mut constants = constants.iter();
    items
        .iter()
        .map(|item| {
            let Item::Static(item_static) = item else {
                return quote! { #item };
            };
            if !has_attr_named(&item_static.attrs, "constant") {
                return quote! { #item };
            }
            let constant = constants
                .next()
                .expect("constant collection and rewrite order drifted");

            let mut item_static = item_static.clone();
            let symbol = LitStr::new(&constant.symbol, constant.ident.span());
            item_static.attrs = item_static
                .attrs
                .into_iter()
                .map(|attr| {
                    if attr_path_ends_with(&attr, "constant") {
                        let path = attr.path().clone();
                        parse_quote!(#[#path(export_name = #symbol)])
                    } else {
                        attr
                    }
                })
                .collect();
            quote! { #item_static }
        })
        .collect()
}

fn cuda_module_constant_field_ident(ident: &Ident) -> Ident {
    format_ident!("__{}", ident)
}

fn cuda_module_constant_resolver_ident(ident: &Ident) -> Ident {
    format_ident!("__resolve_{}", ident)
}

/// Extract `T` from a `ConstantMemory<T>` type path. Returns `None` for anything
/// that's not a path ending in `ConstantMemory<...>` with one generic argument.
pub(crate) fn extract_constant_inner_ty(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let last = type_path.path.segments.last()?;
    if last.ident != "ConstantMemory" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    args.args.iter().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

/// Like [`extract_constant_inner_ty`] but for sites that have already been
/// gated by `#[constant]`'s type-path validation, so extraction failure is
/// a compiler-internal invariant violation, not a user error.
fn constant_inner_ty(ty: &Type) -> &Type {
    extract_constant_inner_ty(ty).unwrap_or_else(|| {
        panic!(
            "#[cuda_module] collected a #[constant] static whose type is not ConstantMemory<T>; \
             this should have been rejected by the #[constant] attribute"
        )
    })
}

pub(super) fn generate_cuda_module_constant_field(constant: &CudaModuleConstant) -> TokenStream2 {
    let CudaModuleConstant {
        ident, cfg_attrs, ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    quote! {
        #(#cfg_attrs)*
        #field: ::std::sync::Arc<::std::sync::Mutex<::core::option::Option<::cuda_core::ConstantHandle>>>,
    }
}

pub(super) fn generate_cuda_module_constant_initializer(
    constant: &CudaModuleConstant,
) -> TokenStream2 {
    let CudaModuleConstant {
        ident, cfg_attrs, ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    quote! {
        #(#cfg_attrs)*
        #field: ::std::sync::Arc::new(::std::sync::Mutex::new(::core::option::Option::None)),
    }
}

pub(super) fn generate_cuda_module_constant_resolver_method(
    constant: &CudaModuleConstant,
) -> TokenStream2 {
    let CudaModuleConstant {
        ident,
        ty,
        cfg_attrs,
        symbol,
        ..
    } = constant;
    let field = cuda_module_constant_field_ident(ident);
    let resolver = cuda_module_constant_resolver_ident(ident);
    let symbol_lit = LitStr::new(symbol, ident.span());
    let inner_ty = constant_inner_ty(ty);
    let mismatch_msg = format!(
        "host/device size mismatch for #[constant] static `{ident}`: \
         device size {{}} bytes, host expected {{}} bytes (PTX symbol `{symbol}`)"
    );
    quote! {
        #(#cfg_attrs)*
        #[allow(non_snake_case)]
        fn #resolver(&self) -> ::core::result::Result<::cuda_core::ConstantHandle, ::cuda_core::DriverError> {
            let mut slot = self
                .#field
                .lock()
                .expect("cuda constant handle cache mutex poisoned");
            if let ::core::option::Option::Some(handle) = *slot {
                return ::core::result::Result::Ok(handle);
            }

            let (dptr, size) = self.__module.get_global(#symbol_lit)?;
            assert_eq!(
                size,
                ::core::mem::size_of::<#inner_ty>(),
                #mismatch_msg,
                size,
                ::core::mem::size_of::<#inner_ty>(),
            );
            // SAFETY: `dptr` was just resolved by `cuModuleGetGlobal` for a
            // module that the LoadedModule keeps alive, and the size matches
            // `size_of::<#inner_ty>()` (asserted above).
            let handle = unsafe { ::cuda_core::ConstantHandle::from_raw(dptr) };
            *slot = ::core::option::Option::Some(handle);
            ::core::result::Result::Ok(handle)
        }
    }
}

/// Generate stream-ordered `set_<name>` and one-shot `set_<name>_blocking`
/// methods on `LoadedModule`. The async setter stages owned host bytes so
/// temporaries remain valid until CUDA reaches the stream callback.
pub(super) fn generate_cuda_module_set_constant_method(
    constant: &CudaModuleConstant,
) -> TokenStream2 {
    let CudaModuleConstant {
        ident,
        ty,
        cfg_attrs,
        method_attrs,
        ..
    } = constant;
    let method_suffix = ident.to_string().to_ascii_lowercase();
    let method_name = format_ident!("set_{}", method_suffix);
    let blocking_name = format_ident!("set_{}_blocking", method_suffix);
    let resolver = cuda_module_constant_resolver_ident(ident);
    let inner_ty = constant_inner_ty(ty);

    quote! {
        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(non_snake_case)]
        pub fn #method_name(
            &self,
            stream: &::cuda_core::CudaStream,
            value: &#inner_ty,
        ) -> ::core::result::Result<(), ::cuda_core::DriverError> {
            let handle = self.#resolver()?;
            let num_bytes = ::core::mem::size_of::<#inner_ty>();
            let mut bytes = ::std::boxed::Box::<[u8]>::new_uninit_slice(num_bytes);
            unsafe {
                ::core::ptr::copy_nonoverlapping(
                    value as *const #inner_ty as *const u8,
                    bytes.as_mut_ptr() as *mut u8,
                    num_bytes,
                );
            }
            handle.write_async_staged(stream, bytes)
        }

        #(#cfg_attrs)*
        #(#method_attrs)*
        #[allow(non_snake_case)]
        pub fn #blocking_name(
            &self,
            value: &#inner_ty,
        ) -> ::core::result::Result<(), ::cuda_core::DriverError> {
            let handle = self.#resolver()?;
            // SAFETY: handle was size-checked against `#inner_ty` by the lazy
            // resolver; `value` is a valid host pointer for
            // `size_of::<#inner_ty>()`.
            unsafe {
                handle.write_blocking(
                    &self.__module,
                    value as *const #inner_ty as *const u8,
                    ::core::mem::size_of::<#inner_ty>(),
                )
            }
        }
    }
}
