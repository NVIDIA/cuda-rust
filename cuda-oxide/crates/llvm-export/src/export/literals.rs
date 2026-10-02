/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant and literal formatting for LLVM IR output.

pub(super) fn format_string_literal(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for byte in value.bytes() {
        match byte {
            // LLVM IR string constants use \XX byte escapes, where XX is a
            // two-digit hex value. See:
            // https://llvm.org/docs/LangRef.html#string-constants
            b'"' | b'\\' | 0x00..=0x1f | 0x7f..=0xff => {
                output.push_str(&format!("\\{byte:02X}"));
            }
            _ => output.push(byte as char),
        }
    }
    output.push('"');
    output
}

pub(super) fn format_half_literal(bits: u16) -> String {
    format!("0xH{bits:04X}")
}

/// Format a float value as an LLVM IR literal.
///
/// LLVM IR accepts decimal notation with a decimal point (e.g. `1.0`) or a
/// 16-hex-digit double bit pattern (e.g. `0x7FF0000000000000`). The hex form
/// is always interpreted as f64 and narrowed to the instruction's destination
/// type (`float`, `double`, or `half`). NaN and Inf must use the hex form
/// because the bare tokens `nan`/`inf` are not part of the LLVM IR grammar
/// and are rejected by both `llc` and libNVVM.
pub(super) fn format_float_literal(value: f64) -> String {
    if value.is_nan() {
        // Canonical quiet NaN. Sign bit is preserved for symmetry with the
        // Inf arm; payload bits are canonicalized to the qNaN marker only.
        if value.is_sign_negative() {
            "0xFFF8000000000000".to_string()
        } else {
            "0x7FF8000000000000".to_string()
        }
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "0x7FF0000000000000".to_string() // +inf
        } else {
            "0xFFF0000000000000".to_string() // -inf
        }
    } else {
        // Finite values: ensure a decimal point so LLVM does not mistake the
        // literal for an integer.
        let s = format!("{value}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

#[cfg(test)]
mod literal_tests {
    use super::{format_float_literal, format_string_literal};

    #[test]
    fn string_literals_escape_special_and_non_printable_bytes() {
        let output = format_string_literal("quote \" slash \\ newline\n tab\t");
        assert_eq!(output, "\"quote \\22 slash \\5C newline\\0A tab\\09\"");
    }

    #[test]
    fn string_literals_escape_utf8_as_bytes() {
        let output = format_string_literal("µ");
        assert_eq!(output, "\"\\C2\\B5\"");
    }

    #[test]
    fn nan_is_emitted_as_hex_qnan_not_bare_token() {
        // The bare token `nan` is not valid LLVM IR and is rejected by
        // libNVVM with "parse expected value token".
        assert_eq!(format_float_literal(f64::NAN), "0x7FF8000000000000");
        assert_eq!(
            format_float_literal(f64::from(f32::NAN)),
            "0x7FF8000000000000"
        );
    }

    #[test]
    fn negative_nan_preserves_sign() {
        assert_eq!(format_float_literal(-f64::NAN), "0xFFF8000000000000");
    }

    #[test]
    fn positive_infinity_is_emitted_as_hex() {
        assert_eq!(format_float_literal(f64::INFINITY), "0x7FF0000000000000");
    }

    #[test]
    fn negative_infinity_is_emitted_as_hex() {
        assert_eq!(
            format_float_literal(f64::NEG_INFINITY),
            "0xFFF0000000000000"
        );
    }

    #[test]
    fn finite_values_get_a_decimal_point() {
        let formatted = format_float_literal(42.0);
        assert!(
            formatted.contains('.') || formatted.contains('e') || formatted.contains('E'),
            "expected decimal point or exponent in `{formatted}`"
        );
        assert_eq!(format_float_literal(-0.0), "-0.0");
    }
}
