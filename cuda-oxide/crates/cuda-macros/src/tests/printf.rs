/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expansion tests for `gpu_printf!`.
//!
//! Every assertion runs on the token text with spaces removed, so byte string
//! literals appear without their inner spaces too (`b" = "` is `b"="`).

use crate::printf::{GpuPrintfInput, gpu_printf_impl};
use syn::parse_quote;

fn expanded(input: GpuPrintfInput) -> String {
    gpu_printf_impl(input).to_string().replace(' ', "")
}

/// Byte offset of `needle` in `haystack`, failing with the whole expansion.
fn position(haystack: &str, needle: &str) -> usize {
    haystack
        .find(needle)
        .unwrap_or_else(|| panic!("expected `{needle}` in {haystack}"))
}

fn assert_trait_route(expansion: &str) {
    assert!(
        expansion.contains("cuda_device::debug::GpuPrintfArg::promote(__val_"),
        "{expansion}"
    );
    assert!(
        expansion.contains("cuda_device::debug::__gpu_printf_format::<"),
        "{expansion}"
    );
    assert!(expansion.contains(">::FORMAT_CHAR"), "{expansion}");
    // The macro's own casts wrap the argument in parentheses; a cast the
    // user wrote (`value as u64`) renders without them. A runtime width is
    // cast to `i32` on every route, so it is not in this list.
    for cast in [")asi64", ")asu64", ")asf64", "%lld", "%llu", "%f"] {
        assert!(
            !expansion.contains(cast),
            "default placeholder must not pick a conversion in the macro: `{cast}` in {expansion}"
        );
    }
}

#[test]
fn default_placeholders_take_the_trait_route_whatever_the_argument_looks_like() {
    let inputs: [GpuPrintfInput; 8] = [
        parse_quote! { "{}", 42 },
        parse_quote! { "{}", 1.5 },
        parse_quote! { "{}", 18446744073709551615u64 },
        parse_quote! { "{}", 3.14159f32 },
        parse_quote! { "{}", true },
        parse_quote! { "{}", -x },
        parse_quote! { "{}", value as u64 },
        parse_quote! { "{}", data[tid] },
    ];
    for input in inputs {
        let expansion = expanded(input);
        assert_trait_route(&expansion);
        assert!(
            expansion.contains("__GpuPrintfSpec::Typed{prefix:b\"\","),
            "{expansion}"
        );
    }
}

#[test]
fn width_zero_pad_flags_and_precision_ride_in_the_typed_prefix() {
    let cases = [
        (parse_quote! { "{:8}", value }, "8"),
        (parse_quote! { "{:08}", value }, "08"),
        (parse_quote! { "{:.2}", value }, ".2"),
        (parse_quote! { "{:-8.3}", value }, "-8.3"),
        (parse_quote! { "{:+}", value }, "+"),
        (parse_quote! { "{: }", value }, " "),
        (parse_quote! { "{:#}", value }, "#"),
        (parse_quote! { "{:*}", width, value }, "*"),
    ];
    for (input, prefix) in cases {
        let expansion = expanded(input);
        assert_trait_route(&expansion);
        let expected = format!(
            "__GpuPrintfSpec::Typed{{prefix:b\"{}\",",
            prefix.replace(' ', "")
        );
        assert!(expansion.contains(&expected), "{expansion}");
    }
}

#[test]
fn explicit_type_characters_keep_their_casts_and_the_literal_format() {
    let cases = [
        (
            parse_quote! { "{:x}", 255u32 },
            "b\"%llx\\0\"",
            "(255u32)asu64",
        ),
        (
            parse_quote! { "{:08X}", bits },
            "b\"%08llX\\0\"",
            "(bits)asu64",
        ),
        (
            parse_quote! { "{:#o}", 64u32 },
            "b\"%#llo\\0\"",
            "(64u32)asu64",
        ),
        (parse_quote! { "{:.2f}", pi }, "b\"%.2f\\0\"", "(pi)asf64"),
        (
            parse_quote! { "{:e}", 1000.0f64 },
            "b\"%e\\0\"",
            "(1000.0f64)asf64",
        ),
        (
            parse_quote! { "{:d}", value },
            "b\"%lld\\0\"",
            "(value)asi64",
        ),
        (
            parse_quote! { "{:+d}", value },
            "b\"%+lld\\0\"",
            "(value)asi64",
        ),
        (parse_quote! { "{:c}", 'a' }, "b\"%c\\0\"", "('a')asi32"),
        (parse_quote! { "{:p}", ptr }, "b\"%llp\\0\"", "(ptr)asu64"),
    ];
    for (input, format, cast) in cases {
        let expansion = expanded(input);
        assert!(expansion.contains(format), "{expansion}");
        assert!(expansion.contains(cast), "{expansion}");
        assert!(
            !expansion.contains("__gpu_printf_format"),
            "all-fixed call sites take the literal route: {expansion}"
        );
        assert!(!expansion.contains("GpuPrintfArg::promote"), "{expansion}");
    }
}

#[test]
fn mixed_call_sites_hand_fixed_and_typed_specs_to_the_builder_in_order() {
    let expansion = expanded(parse_quote! { "{} = {:x} ({:.1})\n", count, bits, ratio });

    assert!(
        expansion.contains("&[b\"\",b\"=\",b\"(\",b\")\\n\"]"),
        "{expansion}"
    );
    let first = position(
        &expansion,
        "__GpuPrintfSpec::Typed{prefix:b\"\",format_char:<__T0as",
    );
    let second = position(&expansion, "__GpuPrintfSpec::Fixed(b\"%llx\")");
    let third = position(
        &expansion,
        "__GpuPrintfSpec::Typed{prefix:b\".1\",format_char:<__T1as",
    );
    assert!(first < second && second < third, "{expansion}");

    assert!(
        expansion.contains("__arg_0:cuda_device::debug::GpuPrintfArg::promote(__val_0)"),
        "{expansion}"
    );
    assert!(expansion.contains("__arg_1:__val_1"), "{expansion}");
    assert!(
        expansion.contains("__arg_2:cuda_device::debug::GpuPrintfArg::promote(__val_2)"),
        "{expansion}"
    );
    assert!(
        expansion.contains("__oxide_gpu_printf_format(&__val_0,&__val_2)"),
        "{expansion}"
    );
}

#[test]
fn the_format_buffer_is_sized_from_the_parts_and_specs() {
    // "a" + "%" + prefix + "ll" + type character + "b" + NUL
    let single = expanded(parse_quote! { "a{}b", value });
    assert!(single.contains("__gpu_printf_format::<7usize>"), "{single}");
    assert!(single.contains("->&'static[u8;7usize]"), "{single}");

    // "100%% " is six bytes once the percent sign is doubled.
    let percent = expanded(parse_quote! { "100% {}", value });
    assert!(percent.contains("b\"100%%\""), "{percent}");
    assert!(
        percent.contains("__gpu_printf_format::<11usize>"),
        "{percent}"
    );

    // "%llx" (4) + "%" + "08" + "ll" + type character (6) + NUL
    let mixed = expanded(parse_quote! { "{:x}{:08}", bits, value });
    assert!(mixed.contains("__gpu_printf_format::<11usize>"), "{mixed}");
}

#[test]
fn arguments_are_evaluated_in_source_order_on_both_routes() {
    let typed = expanded(parse_quote! { "{} [{:*}] {:x}\n", a(), w(), v(), h() });
    let lets = [
        "let__val_0=a();",
        "let__val_1=(w())asi32;",
        "let__val_2=v();",
        "let__val_3=(h())asu64;",
        "let__fmt=",
        "let__args=",
    ];
    let mut last = 0;
    for needle in lets {
        let at = position(&typed, needle);
        assert!(at > last, "`{needle}` out of order in {typed}");
        last = at;
    }
    assert!(typed.contains("__width_1:__val_1"), "{typed}");
    assert!(
        typed.contains("__arg_1:cuda_device::debug::GpuPrintfArg::promote(__val_2)"),
        "{typed}"
    );

    let fixed = expanded(parse_quote! { "[{:*d}] {:x}\n", w(), v(), h() });
    let lets = [
        "let__val_0=(w())asi32;",
        "let__val_1=(v())asi64;",
        "let__val_2=(h())asu64;",
        "let__args=",
    ];
    let mut last = 0;
    for needle in lets {
        let at = position(&fixed, needle);
        assert!(at > last, "`{needle}` out of order in {fixed}");
        last = at;
    }
    assert!(fixed.contains("b\"[%*lld]%llx\\n\\0\""), "{fixed}");
}

#[test]
fn literal_percent_signs_and_braces_are_escaped_on_both_routes() {
    let typed = expanded(parse_quote! { "{{{}}} 100%\n", value });
    assert!(typed.contains("&[b\"{\",b\"}100%%\\n\"]"), "{typed}");

    let fixed = expanded(parse_quote! { "{{{:d}}} 100%\n", value });
    assert!(fixed.contains("b\"{%lld}100%%\\n\\0\""), "{fixed}");
}

#[test]
fn a_call_without_placeholders_passes_a_null_argument_pointer() {
    let expansion = expanded(parse_quote! { "hello 100%\n" });
    assert!(expansion.contains("b\"hello100%%\\n\\0\""), "{expansion}");
    assert!(expansion.contains("core::ptr::null()"), "{expansion}");
    assert!(!expansion.contains("__GpuPrintfArgs"), "{expansion}");
}

#[test]
fn string_placeholders_keep_the_literal_for_concat() {
    let expansion = expanded(parse_quote! { "{:s}\n", "device" });
    assert!(
        expansion
            .contains("let__val_0={const__STR:&str=concat!(\"device\",\"\\0\");__STR.as_ptr()};"),
        "{expansion}"
    );
    assert!(expansion.contains("b\"%s\\n\\0\""), "{expansion}");
}

#[test]
fn runtime_precision_is_a_macro_error() {
    let expansion = expanded(parse_quote! { "{:.*}", 2, pi });
    assert!(expansion.contains("compile_error"), "{expansion}");
    assert!(
        expansion.contains("runtimeprecision`.*`isnotsupportedbyCUDAdeviceprintf"),
        "{expansion}"
    );

    let with_type = expanded(parse_quote! { "{:.*f}", 2, pi });
    assert!(with_type.contains("compile_error"), "{with_type}");
}

#[test]
fn argument_count_mismatches_are_macro_errors() {
    let missing = expanded(parse_quote! { "{} {}\n", 1 });
    assert!(missing.contains("compile_error"), "{missing}");
    assert!(
        missing.contains("requires2argumentsbut1weresupplied"),
        "{missing}"
    );

    // A runtime width takes an argument of its own.
    let width = expanded(parse_quote! { "{:*}\n", 1 });
    assert!(width.contains("compile_error"), "{width}");
    assert!(
        width.contains("requires2argumentsbut1weresupplied"),
        "{width}"
    );
}
