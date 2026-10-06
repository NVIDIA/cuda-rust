/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::printf::{GpuPrintfInput, gpu_printf_impl};
use syn::parse_quote;

fn expanded(input: GpuPrintfInput) -> String {
    gpu_printf_impl(input).to_string().replace(' ', "")
}

#[test]
fn precision_without_type_on_float_uses_float_conversion() {
    for tokens in [
        parse_quote! { "{:.2}", 3.14159f32 },
        parse_quote! { "{:.2}", 3.14159f64 },
        parse_quote! { "{:.2}", 3.14159 },
    ] {
        let expanded = expanded(tokens);
        assert!(
            expanded.contains("asf64"),
            "float precision must cast to f64, got {expanded}"
        );
        assert!(
            expanded.contains("%.2f"),
            "float precision must emit %.2f, got {expanded}"
        );
        assert!(
            !expanded.contains("lli") && !expanded.contains("lld") && !expanded.contains("asi64"),
            "float precision must not use the integer conversion, got {expanded}"
        );
    }

    // Explicit float and integer specifiers stay on their existing conversions.
    let explicit_float = expanded(parse_quote! { "{:.2f}", 3.14159f32 });
    assert!(explicit_float.contains("%.2f"), "{explicit_float}");
    assert!(explicit_float.contains("asf64"), "{explicit_float}");

    let width = expanded(parse_quote! { "{:08}", 42 });
    assert!(width.contains("%08lld"), "{width}");
    assert!(width.contains("asi64"), "{width}");

    let hex = expanded(parse_quote! { "{:x}", 255u32 });
    assert!(hex.contains("%llx"), "{hex}");
    assert!(hex.contains("asu64"), "{hex}");
}

#[test]
fn default_format_on_u64_uses_unsigned_conversion() {
    let unsigned = expanded(parse_quote! { "{}", 18446744073709551615u64 });
    assert!(
        unsigned.contains("asu64"),
        "u64 default must cast to u64, got {unsigned}"
    );
    assert!(
        unsigned.contains("%llu"),
        "u64 default must emit %llu, got {unsigned}"
    );
    assert!(
        !unsigned.contains("asi64") && !unsigned.contains("%lld"),
        "u64 default must not wrap through i64, got {unsigned}"
    );

    let cast = expanded(parse_quote! { "{}", value as u64 });
    assert!(
        cast.contains(")asu64"),
        "unsigned cast must be packed as u64, got {cast}"
    );
    assert!(cast.contains("%llu"), "{cast}");

    let float_default = expanded(parse_quote! { "{}", 1.5f32 });
    assert!(float_default.contains("asf64"), "{float_default}");
    assert!(float_default.contains("%f"), "{float_default}");
    assert!(!float_default.contains("%lld"), "{float_default}");

    // The macro cannot see a bare binding's type. Pack through GpuPrintfArg
    // so a u64 above i64::MAX is not printed as a negative %lld, and a float
    // is not truncated to an integer.
    let untyped = expanded(parse_quote! { "{}", value });
    assert!(untyped.contains("GpuPrintfArg::promote"), "{untyped}");
    assert!(untyped.contains("IS_FLOAT"), "{untyped}");
    assert!(untyped.contains("FORMAT_CHAR"), "{untyped}");
    assert!(!untyped.contains("asi64"), "{untyped}");

    // Signed defaults, width, and explicit conversions stay as they are.
    let signed = expanded(parse_quote! { "{}", 42 });
    assert!(signed.contains("asi64"), "{signed}");
    assert!(signed.contains("%lld"), "{signed}");

    let width = expanded(parse_quote! { "{:8}", 42 });
    assert!(width.contains("%8lld"), "{width}");
    assert!(width.contains("asi64"), "{width}");

    let zero_pad = expanded(parse_quote! { "{:08}", 42 });
    assert!(zero_pad.contains("%08lld"), "{zero_pad}");
    assert!(zero_pad.contains("asi64"), "{zero_pad}");

    let hex = expanded(parse_quote! { "{:x}", 255u64 });
    assert!(hex.contains("%llx"), "{hex}");
    assert!(hex.contains("asu64"), "{hex}");

    let scientific = expanded(parse_quote! { "{:e}", 1000.0f64 });
    assert!(scientific.contains("%e"), "{scientific}");
    assert!(scientific.contains("asf64"), "{scientific}");
    assert!(!scientific.contains("%llu"), "{scientific}");
}
