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
