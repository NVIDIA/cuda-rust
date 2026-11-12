/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device::device_function_tokens;
use quote::{format_ident, quote};
use reserved_oxide_symbols::DEVICE_PREFIX;
use syn::{Attribute, ItemFn, parse_quote};

/// The `inline` attributes of the renamed function in the expansion of
/// `function`.
fn renamed_inline_attrs(function: ItemFn) -> Vec<String> {
    let name = format_ident!("{}{}", DEVICE_PREFIX, function.sig.ident);
    let expanded = device_function_tokens(function);
    let file: syn::File =
        syn::parse2(expanded.clone()).expect("generated tokens must parse as items");
    let renamed = file
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Fn(item_fn) if item_fn.sig.ident == name => Some(item_fn),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expansion has no fn `{name}`:\n{expanded}"));
    renamed
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("inline"))
        .map(|attr| quote!(#attr).to_string().replace(' ', ""))
        .collect()
}

#[test]
fn generic_device_function_keeps_the_inline_attribute_of_the_user() {
    let attrs: [Attribute; 3] = [
        parse_quote!(#[inline]),
        parse_quote!(#[inline(always)]),
        parse_quote!(#[inline(never)]),
    ];
    for attr in attrs {
        let expected = quote!(#attr).to_string().replace(' ', "");
        let function: ItemFn = parse_quote! {
            #attr
            fn scale<T: Copy>(value: T) -> T { value }
        };
        assert_eq!(renamed_inline_attrs(function), [expected]);
    }
}

#[test]
fn generic_device_function_is_inline_never_by_default() {
    let function: ItemFn = parse_quote! {
        fn scale<T: Copy>(value: T) -> T { value }
    };
    assert_eq!(renamed_inline_attrs(function), ["#[inline(never)]"]);
}

#[test]
fn other_attributes_do_not_replace_the_default() {
    let function: ItemFn = parse_quote! {
        #[cold]
        #[doc = "inline"]
        fn scale<T: Copy>(value: T) -> T { value }
    };
    assert_eq!(renamed_inline_attrs(function), ["#[inline(never)]"]);
}

#[test]
fn non_generic_device_function_gets_no_inline_attribute() {
    let function: ItemFn = parse_quote! {
        fn scale(value: f32) -> f32 { value }
    };
    assert!(renamed_inline_attrs(function).is_empty());
}
