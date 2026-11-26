/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device::{DeviceArgs, expand_device_function};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use reserved_oxide_symbols::{device_export_symbol, device_symbol};
use syn::{ItemFn, parse_quote};

fn expand(args: TokenStream2, func: ItemFn) -> syn::Result<String> {
    let args: DeviceArgs = syn::parse2(args)?;
    expand_device_function(args, func).map(|tokens| tokens.to_string().replace(' ', ""))
}

fn expand_err(args: TokenStream2, func: ItemFn) -> String {
    match expand(args, func) {
        Ok(expanded) => panic!("expected an error, got {expanded}"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn device_export_variants_use_export_prefix_and_keep_rust_wrapper() {
    let plain = expand(
        quote!(),
        parse_quote! {
            pub fn helper(x: f32) -> f32 { x + 1.0 }
        },
    )
    .unwrap();
    assert!(
        plain.contains(&device_symbol("helper")),
        "plain #[device] keeps helper prefix: {plain}"
    );

    let exported = expand(
        quote!(export),
        parse_quote! {
            pub fn __direct_callable__shade(x: f32) -> f32 { x * 2.0 }
        },
    )
    .unwrap();
    let expected_export = device_export_symbol("__direct_callable__shade");
    assert!(exported.contains(&expected_export), "{exported}");
    assert!(
        exported.contains("pubfn__direct_callable__shade("),
        "wrapper keeps the original Rust name: {exported}"
    );

    let renamed = expand(
        quote!(export_name = "__direct_callable__shade",),
        parse_quote! {
            pub fn shade(x: f32) -> f32 { x * 2.0 }
        },
    )
    .unwrap();
    assert!(renamed.contains(&expected_export), "{renamed}");
    assert!(
        renamed.contains("pubfnshade("),
        "in-crate Rust callers still call `shade`: {renamed}"
    );
}

#[test]
fn device_export_rejects_generics_and_invalid_names() {
    let generic_err = expand_err(
        quote!(export),
        parse_quote! {
            pub fn shade<T>(x: T) -> T { x }
        },
    );
    assert!(
        generic_err.contains("cannot have generic parameters"),
        "{generic_err}"
    );

    for bad in [
        quote!(export = "x"),
        quote!(unknown),
        quote!(export_name = ""),
        quote!(export_name = "9bad"),
        quote!(export_name = "bad$name"),
        quote!(export_name = "llvm_foo"),
        quote!(export_name = "cuda_oxide_foo"),
    ] {
        let err = expand_err(
            bad.clone(),
            parse_quote! {
                pub fn shade(x: f32) -> f32 { x }
            },
        );
        assert!(!err.is_empty(), "expected rejection for `{bad}`");
    }
}
