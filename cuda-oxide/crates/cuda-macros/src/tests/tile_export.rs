/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::tile_export::{TileExportArgs, expand_tile_export};
use oxide_tile_export::{TileAccess, TileElem, TileExportDesc, TileSync};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use reserved_oxide_symbols::tile_export_meta_symbol;
use syn::{ItemFn, parse_quote};

fn expand(args: TokenStream2, func: ItemFn) -> syn::Result<String> {
    let args: TileExportArgs = syn::parse2(args)?;
    expand_tile_export(args, func).map(|tokens| tokens.to_string().replace(' ', ""))
}

fn expand_err(args: TokenStream2, func: ItemFn) -> String {
    match expand(args, func) {
        Ok(expanded) => panic!("expected an error, got {expanded}"),
        Err(err) => err.to_string(),
    }
}

/// Pull the embedded JSON back out of the expansion through the metadata
/// function's string literal.
fn embedded_desc(expanded: &str, symbol: &str) -> TileExportDesc {
    let meta = tile_export_meta_symbol(symbol);
    let start = expanded
        .find(&format!("fn{meta}()"))
        .unwrap_or_else(|| panic!("no metadata function `{meta}` in {expanded}"));
    let rest = &expanded[start..];
    let open = rest.find('"').expect("string literal");
    let json: String = syn::parse_str::<syn::LitStr>(
        &rest[open..rest[open..].find("\"}").map(|end| open + end + 1).unwrap()],
    )
    .expect("the metadata body is a string literal")
    .value();
    TileExportDesc::from_json(&json).expect("embedded JSON parses")
}

#[test]
fn scalar_in_and_returned_out_follow_the_table() {
    let func: ItemFn = parse_quote! {
        pub fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
    };
    let expanded = expand(quote! { tile_shape = 128 }, func).unwrap();

    // The original function survives unchanged.
    assert!(expanded.contains("pubfnrelu(x:f32)->f32"), "{expanded}");
    // The wrapper is a `#[device]` extern "C" function on the native ABI.
    assert!(
        expanded.contains("#[::cuda_device::device]pubunsafeextern\"C\"fnrelu_impl("),
        "{expanded}"
    );
    assert!(expanded.contains(":f32,"), "{expanded}");
    assert!(expanded.contains(":*mutf32)"), "{expanded}");
    assert!(expanded.contains("=relu("), "{expanded}");

    let desc = embedded_desc(&expanded, "relu_impl");
    desc.validate().unwrap();
    assert_eq!(desc.decl, "relu");
    assert_eq!(desc.symbol, "relu_impl");
    assert_eq!(desc.shape, vec![128]);
    assert_eq!(desc.sync, TileSync::None);
    assert!(desc.arch.is_empty());
    assert_eq!(desc.params.len(), 2);
    assert_eq!(desc.params[0].name, "x");
    assert_eq!(desc.params[0].access, TileAccess::In);
    assert!(desc.params[0].by_value);
    assert_eq!(desc.params[1].name, "relu_out");
    assert_eq!(desc.params[1].access, TileAccess::Out);
    assert_eq!(desc.params[1].elem, TileElem::F32);
    assert!(!desc.params[1].by_value);
}

#[test]
fn name_symbol_sync_and_arch_override_the_defaults() {
    let func: ItemFn = parse_quote! {
        pub fn scale(x: &mut f32, factor: f32) { *x *= factor; }
    };
    let expanded = expand(
        quote! { tile_shape = 128, name = "scale_in_place", symbol = "scale_impl_v2", sync = "warp", arch = ["sm_100", "sm_120"] },
        func,
    )
    .unwrap();
    assert!(
        expanded.contains("pubunsafeextern\"C\"fnscale_impl_v2("),
        "{expanded}"
    );
    assert!(expanded.contains(":*mutf32,"), "{expanded}");
    assert!(expanded.contains(":f32)"), "{expanded}");
    // `&mut T` is rebuilt from the pointer before the call.
    assert!(expanded.contains("&mut*"), "{expanded}");

    let desc = embedded_desc(&expanded, "scale_impl_v2");
    desc.validate().unwrap();
    assert_eq!(desc.decl, "scale_in_place");
    assert_eq!(desc.symbol, "scale_impl_v2");
    assert_eq!(desc.sync, TileSync::Warp);
    assert_eq!(desc.arch, vec!["sm_100".to_string(), "sm_120".to_string()]);
    assert_eq!(desc.params[0].access, TileAccess::InOut);
    assert!(!desc.params[0].by_value);
    assert_eq!(desc.params[1].access, TileAccess::In);
    assert!(desc.params[1].by_value);
}

#[test]
fn symbol_defaults_to_the_declaration_name() {
    let func: ItemFn = parse_quote! {
        pub fn scale(x: &mut f32) { *x *= 2.0; }
    };
    let expanded = expand(quote! { tile_shape = 64, name = "double" }, func).unwrap();
    let desc = embedded_desc(&expanded, "double_impl");
    assert_eq!(desc.decl, "double");
    assert_eq!(desc.symbol, "double_impl");
}

#[test]
fn array_parameters_carry_their_footprint() {
    let func: ItemFn = parse_quote! {
        pub fn window(acc: &mut [i32; 4], bias: &[i32; 4]) {
            let _ = (acc, bias);
        }
    };
    let expanded = expand(quote! { tile_shape = 64 }, func).unwrap();
    assert!(expanded.contains(":*muti32,"), "{expanded}");
    assert!(expanded.contains(":*consti32)"), "{expanded}");
    // Both arrays are rebuilt from their pointers before the call.
    assert!(expanded.contains("as*mut[i32;4]"), "{expanded}");
    assert!(expanded.contains("as*const[i32;4]"), "{expanded}");

    let desc = embedded_desc(&expanded, "window_impl");
    desc.validate().unwrap();
    let shapes: Vec<_> = desc
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.elem, p.access, p.footprint, p.by_value))
        .collect();
    assert_eq!(
        shapes,
        vec![
            ("acc", TileElem::I32, TileAccess::InOut, 4, false),
            ("bias", TileElem::I32, TileAccess::In, 4, false),
        ]
    );
}

#[test]
fn scalar_references_and_halves_carry_footprint_one() {
    let func: ItemFn = parse_quote! {
        pub fn mix(scale: &'_ u16, half: f16, out: &mut u16) {
            let _ = (scale, half, out);
        }
    };
    let expanded = expand(quote! { tile_shape = 64 }, func).unwrap();
    assert!(expanded.contains(":*constu16,"), "{expanded}");
    assert!(expanded.contains(":f16,"), "{expanded}");
    assert!(expanded.contains(":*mutu16)"), "{expanded}");

    let desc = embedded_desc(&expanded, "mix_impl");
    desc.validate().unwrap();
    let shapes: Vec<_> = desc
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.elem, p.access, p.footprint, p.by_value))
        .collect();
    assert_eq!(
        shapes,
        vec![
            ("scale", TileElem::U16, TileAccess::In, 1, false),
            ("half", TileElem::F16, TileAccess::In, 1, true),
            ("out", TileElem::U16, TileAccess::InOut, 1, false),
        ]
    );
}

#[test]
fn mixed_footprints_are_rejected_at_the_differing_parameter() {
    // The window needs 16 invocations of a 64-element tile, the scalar 64:
    // one implementation cannot have both.
    let func: ItemFn = parse_quote! {
        pub fn window(acc: &mut [i32; 4], bias: &i32) {
            let _ = (acc, bias);
        }
    };
    let err = expand_err(quote! { tile_shape = 64 }, func);
    assert!(
        err.contains("`acc` (footprint 4) and `bias` (footprint 1) differ"),
        "{err}"
    );
    // A by-value scalar is footprint 1 too, so it cannot sit beside a window.
    let func: ItemFn = parse_quote! {
        pub fn scale4(x: &mut [f32; 4], factor: f32) {
            let _ = (x, factor);
        }
    };
    let err = expand_err(quote! { tile_shape = 128 }, func);
    assert!(
        err.contains("`x` (footprint 4) and `factor` (footprint 1) differ"),
        "{err}"
    );
}

#[test]
fn warp_sync_with_a_footprint_needs_whole_warps_of_footprints() {
    let func: ItemFn = parse_quote! {
        pub fn scale4(x: &mut [f32; 4]) {
            let _ = x;
        }
    };
    // 64 is a multiple of 32, but a warp of 4-element footprints owns 128.
    let err = expand_err(quote! { tile_shape = 64, sync = "warp" }, func.clone());
    assert!(
        err.contains("footprint 4") && err.contains("multiple of 128") && err.contains("64"),
        "{err}"
    );
    expand(quote! { tile_shape = 128, sync = "warp" }, func.clone()).unwrap();
    // Without synchronization the footprint only has to divide the shape.
    expand(quote! { tile_shape = 64 }, func).unwrap();
}

#[test]
fn primitive_module_paths_are_element_types() {
    let func: ItemFn = parse_quote! {
        pub fn add(a: core::primitive::f64, b: ::std::primitive::f64) -> f64 { a + b }
    };
    let desc = embedded_desc(
        &expand(quote! { tile_shape = 32 }, func).unwrap(),
        "add_impl",
    );
    assert!(desc.params.iter().all(|p| p.elem == TileElem::F64));
}

#[test]
fn the_out_parameter_avoids_a_user_parameter_of_the_same_name() {
    let func: ItemFn = parse_quote! {
        pub fn f(f_out: f32) -> f32 { f_out }
    };
    let desc = embedded_desc(&expand(quote! { tile_shape = 32 }, func).unwrap(), "f_impl");
    assert_eq!(desc.params[1].name, "f_out_");
    desc.validate().unwrap();
}

#[test]
fn generics_self_and_async_are_rejected() {
    let generic: ItemFn = parse_quote! { pub fn relu<T>(x: T) -> T { x } };
    assert!(expand_err(quote! { tile_shape = 128 }, generic).contains("cannot be generic"));

    let lifetime: ItemFn = parse_quote! { pub fn relu<'a>(x: &'a f32) -> f32 { *x } };
    assert!(expand_err(quote! { tile_shape = 128 }, lifetime).contains("cannot be generic"));

    let method: ItemFn = parse_quote! { pub fn relu(self, x: f32) -> f32 { x } };
    assert!(expand_err(quote! { tile_shape = 128 }, method).contains("cannot take self"));

    let asynchronous: ItemFn = parse_quote! { pub async fn relu(x: f32) -> f32 { x } };
    assert!(expand_err(quote! { tile_shape = 128 }, asynchronous).contains("cannot be async"));
}

#[test]
fn shape_is_required_positive_and_literal() {
    let func: ItemFn = parse_quote! { pub fn relu(x: f32) -> f32 { x } };
    assert!(expand_err(quote! {}, func.clone()).contains("requires `tile_shape = N`"));
    assert!(expand_err(quote! { tile_shape = 0 }, func.clone()).contains("positive integer"));
    let err = syn::parse2::<TileExportArgs>(quote! { tile_shape = N })
        .err()
        .expect("a non-literal shape is a parse error")
        .to_string();
    assert!(err.contains("integer literal"), "{err}");
    let err = syn::parse2::<TileExportArgs>(quote! { tile_shape = 128, tile_shape = 64 })
        .err()
        .expect("duplicate keys are rejected")
        .to_string();
    assert!(err.contains("duplicate"), "{err}");
}

#[test]
fn unknown_and_malformed_arguments_are_rejected() {
    let err = syn::parse2::<TileExportArgs>(quote! { tile_shape = 128, foo = 1 })
        .err()
        .expect("unknown keys are rejected")
        .to_string();
    assert!(
        err.contains("unknown #[tile_export] argument `foo`")
            && err.contains("shape, name, symbol, sync, arch"),
        "{err}"
    );
    let err = syn::parse2::<TileExportArgs>(quote! { tile_shape = 128, arch = "sm_120" })
        .err()
        .expect("arch must be a list")
        .to_string();
    assert!(err.contains("list of string literals"), "{err}");
    let err = syn::parse2::<TileExportArgs>(quote! { tile_shape = 128, name = relu })
        .err()
        .expect("name must be a string literal")
        .to_string();
    assert!(err.contains("string literal"), "{err}");

    let func: ItemFn = parse_quote! { pub fn relu(x: f32) -> f32 { x } };
    assert!(
        expand_err(quote! { tile_shape = 128, sync = "cta" }, func.clone())
            .contains("must be \"none\" or \"warp\"")
    );
    assert!(
        expand_err(quote! { tile_shape = 128, symbol = "re-lu" }, func.clone())
            .contains("must be an identifier")
    );
    assert!(
        expand_err(
            quote! { tile_shape = 128, name = "cuda_oxide_relu" },
            func.clone()
        )
        .contains("reserved cuda-oxide prefix")
    );
    assert!(expand_err(quote! { tile_shape = 128, arch = [""] }, func).contains("non-empty"));
}

#[test]
fn unsupported_parameter_and_return_types_say_what_is_accepted() {
    for func in [
        parse_quote! { pub fn bad(x: &[f32]) -> f32 { x[0] } },
        parse_quote! { pub fn bad(x: *mut f32) -> f32 { 0.0 } },
        parse_quote! { pub fn bad(x: bool) -> f32 { 0.0 } },
        parse_quote! { pub fn bad(x: usize) -> f32 { 0.0 } },
        parse_quote! { pub fn bad(x: &mut [f32]) -> f32 { 0.0 } },
        parse_quote! { pub fn bad(x: &[bool; 4]) -> f32 { 0.0 } },
        parse_quote! { pub fn bad(x: (f32, f32)) -> f32 { x.0 } },
    ] {
        let err = expand_err(quote! { tile_shape = 128 }, func);
        assert!(
            err.contains(
                "parameters must be `x: T`, `x: &T`, `x: &mut T`, `x: &[T; N]`, or \
                 `x: &mut [T; N]`"
            ),
            "{err}"
        );
        assert!(
            err.contains("f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64"),
            "{err}"
        );
    }

    let pattern: ItemFn = parse_quote! { pub fn bad((a, b): (f32, f32)) -> f32 { a } };
    assert!(expand_err(quote! { tile_shape = 128 }, pattern).contains("plain identifiers"));

    let const_len: ItemFn = parse_quote! { pub fn bad(x: &mut [f32; N]) {} };
    assert!(expand_err(quote! { tile_shape = 128 }, const_len).contains("integer literal length"));

    for func in [
        parse_quote! { pub fn bad(x: f32) -> &'static f32 { &0.0 } },
        parse_quote! { pub fn bad(x: f32) -> (f32, f32) { (x, x) } },
        parse_quote! { pub fn bad(x: f32) -> bool { x > 0.0 } },
    ] {
        let err = expand_err(quote! { tile_shape = 128 }, func);
        assert!(err.contains("must return a scalar"), "{err}");
    }
}

#[test]
fn an_export_needs_an_output() {
    let sink: ItemFn = parse_quote! { pub fn sink(x: f32, y: &f32) { let _ = (x, y); } };
    let err = expand_err(quote! { tile_shape = 128 }, sink);
    assert!(err.contains("has no output"), "{err}");

    // `()` is fine once something is written in place.
    let in_place: ItemFn = parse_quote! { pub fn scale(x: &mut f32) -> () { *x *= 2.0; } };
    expand(quote! { tile_shape = 128 }, in_place).unwrap();
}

#[test]
fn warp_sync_needs_a_multiple_of_32() {
    let func: ItemFn = parse_quote! { pub fn s(x: f32) -> f32 { x } };
    let err = expand_err(quote! { tile_shape = 100, sync = "warp" }, func.clone());
    assert!(
        err.contains("multiple of 32") && err.contains("100"),
        "{err}"
    );
    expand(quote! { tile_shape = 96, sync = "warp" }, func).unwrap();
}

#[test]
fn reserved_function_names_are_rejected() {
    let func: ItemFn = parse_quote! { pub fn cuda_oxide_relu(x: f32) -> f32 { x } };
    assert!(expand_err(quote! { tile_shape = 128 }, func).contains("reserved cuda-oxide"));
}
