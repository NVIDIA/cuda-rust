/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU printf procedural macro implementation.
//!
//! Translates Rust-style format strings to C-style and generates
//! argument packing code for CUDA's vprintf.
//!
//! # Design
//!
//! The `gpu_printf!` macro:
//! 1. Parses Rust-style format string (`{}`, `{:x}`, `{:.2}`, etc.)
//! 2. Translates to C format string (`%d`, `%x`, `%.2f`, etc.)
//! 3. Generates a `#[repr(C)]` struct for argument packing
//! 4. Calls `__gpu_vprintf(format_ptr, args_ptr)` which the compiler recognizes
//!
//! # C Vararg Promotion Rules
//!
//! Arguments are promoted following C vararg rules:
//! - `i8`, `i16` → `i32`
//! - `u8`, `u16` → `u32`
//! - `f32` → `f64` (critical!)
//! - `bool` → `i32` (0 or 1)
//! - pointers/references → `u64` (64-bit)

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, LitStr, Token,
    parse::{Parse, ParseStream},
};

/// Maximum number of printf arguments (CUDA limitation)
const MAX_PRINTF_ARGS: usize = 32;

/// Input to gpu_printf! macro: format string followed by arguments
pub struct GpuPrintfInput {
    format_string: LitStr,
    args: Vec<Expr>,
}

impl Parse for GpuPrintfInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let format_string: LitStr = input.parse()?;
        let mut args = Vec::new();

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if !input.is_empty() {
                args.push(input.parse()?);
            }
        }

        Ok(GpuPrintfInput {
            format_string,
            args,
        })
    }
}

/// Parsed format placeholder from the format string
#[derive(Debug, Clone)]
struct Placeholder {
    /// Start position in format string (at the `{`)
    start: usize,
    /// End position in format string (after the `}`)
    end: usize,
    /// Parsed format specification
    spec: FormatSpec,
}

/// Format specification parsed from `{:...}`
#[derive(Debug, Clone, Default)]
struct FormatSpec {
    /// `-` flag: left-justify
    left_justify: bool,
    /// `+` flag: always show sign
    always_sign: bool,
    /// ` ` flag: space before positive numbers
    space_sign: bool,
    /// `#` flag: alternate form (0x prefix for hex, etc.)
    alternate: bool,
    /// `0` flag: zero-pad
    zero_pad: bool,
    /// Minimum width (None = no width specified)
    width: Option<Width>,
    /// Precision (None = no precision specified)
    precision: Option<Precision>,
    /// Format type character (x, X, o, e, E, f, g, G, a, A, p, c, s, d, etc.)
    format_type: Option<char>,
}

/// Width specification
#[derive(Debug, Clone)]
enum Width {
    /// Literal width value
    Literal(usize),
    /// Runtime width (`*`)
    Runtime,
}

/// Precision specification
#[derive(Debug, Clone)]
enum Precision {
    /// Literal precision value
    Literal(usize),
    /// Runtime precision (`.*`)
    Runtime,
}

/// Parse all placeholders from a format string
fn parse_format_string(format: &str) -> Result<Vec<Placeholder>, String> {
    let mut placeholders = Vec::new();
    let chars: Vec<char> = format.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                // Escaped `{{` - literal `{`
                i += 2;
                continue;
            }

            // Found a placeholder
            let start = i;
            i += 1; // Skip `{`

            // Parse until `}`
            let mut spec = FormatSpec::default();

            while i < chars.len() && chars[i] != '}' {
                if chars[i] == ':' {
                    i += 1;
                    // Parse format spec after `:`
                    i = parse_format_spec(&chars, i, &mut spec)?;
                } else {
                    i += 1;
                }
            }

            if i >= chars.len() {
                return Err("Unclosed format placeholder `{`".to_string());
            }

            i += 1; // Skip `}`

            placeholders.push(Placeholder {
                start,
                end: i,
                spec,
            });
        } else if chars[i] == '}' {
            if i + 1 < chars.len() && chars[i + 1] == '}' {
                // Escaped `}}` - literal `}`
                i += 2;
                continue;
            }
            return Err("Unmatched `}` in format string".to_string());
        } else {
            i += 1;
        }
    }

    Ok(placeholders)
}

/// Parse format specification after the `:` in `{:...}`
fn parse_format_spec(chars: &[char], mut i: usize, spec: &mut FormatSpec) -> Result<usize, String> {
    // Parse flags
    while i < chars.len() && chars[i] != '}' {
        match chars[i] {
            '-' => {
                spec.left_justify = true;
                i += 1;
            }
            '+' => {
                spec.always_sign = true;
                i += 1;
            }
            ' ' => {
                spec.space_sign = true;
                i += 1;
            }
            '#' => {
                spec.alternate = true;
                i += 1;
            }
            '0' if spec.width.is_none() => {
                // Leading 0 is zero-pad flag, not part of width
                spec.zero_pad = true;
                i += 1;
            }
            _ => break,
        }
    }

    // Parse width
    if i < chars.len() && chars[i] != '}' {
        if chars[i] == '*' {
            spec.width = Some(Width::Runtime);
            i += 1;
        } else if chars[i].is_ascii_digit() {
            let (width, new_i) = parse_number(chars, i);
            spec.width = Some(Width::Literal(width));
            i = new_i;
        }
    }

    // Parse precision
    if i < chars.len() && chars[i] == '.' {
        i += 1; // Skip `.`
        if i < chars.len() {
            if chars[i] == '*' {
                spec.precision = Some(Precision::Runtime);
                i += 1;
            } else if chars[i].is_ascii_digit() {
                let (prec, new_i) = parse_number(chars, i);
                spec.precision = Some(Precision::Literal(prec));
                i = new_i;
            } else {
                // `.` without number means precision 0
                spec.precision = Some(Precision::Literal(0));
            }
        }
    }

    // Parse type character
    if i < chars.len() && chars[i] != '}' {
        let c = chars[i];
        if matches!(
            c,
            'c' | 's'
                | 'd'
                | 'i'
                | 'o'
                | 'x'
                | 'X'
                | 'e'
                | 'E'
                | 'f'
                | 'g'
                | 'G'
                | 'a'
                | 'A'
                | 'p'
                | 'b'
        ) {
            spec.format_type = Some(c);
            i += 1;
        }
    }

    Ok(i)
}

/// Parse a decimal number from the character array
fn parse_number(chars: &[char], mut i: usize) -> (usize, usize) {
    let mut num = 0usize;
    while i < chars.len() && chars[i].is_ascii_digit() {
        num = num * 10 + (chars[i] as usize - '0' as usize);
        i += 1;
    }
    (num, i)
}

/// Count how many extra arguments are needed for runtime width/precision
fn count_runtime_args(spec: &FormatSpec) -> usize {
    let mut count = 0;
    if matches!(spec.width, Some(Width::Runtime)) {
        count += 1;
    }
    if matches!(spec.precision, Some(Precision::Runtime)) {
        count += 1;
    }
    count
}

/// Build the C format string from Rust format string
fn build_c_format_string(format: &str, placeholders: &[Placeholder]) -> String {
    let mut result = String::new();
    let chars: Vec<char> = format.chars().collect();
    let mut last_end = 0;

    for placeholder in placeholders {
        // Add literal text before this placeholder
        let mut i = last_end;
        while i < placeholder.start {
            if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
                result.push('{');
                i += 2;
            } else if chars[i] == '}' && i + 1 < chars.len() && chars[i + 1] == '}' {
                result.push('}');
                i += 2;
            } else if chars[i] == '%' {
                // Escape literal % as %%
                result.push_str("%%");
                i += 1;
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        // Add placeholder as %... (will be filled in by generate_c_specifier)
        // For now, add a marker that we'll replace
        result.push_str("__PLACEHOLDER__");

        last_end = placeholder.end;
    }

    // Add remaining literal text
    let mut i = last_end;
    while i < chars.len() {
        if chars[i] == '{' && i + 1 < chars.len() && chars[i + 1] == '{' {
            result.push('{');
            i += 2;
        } else if chars[i] == '}' && i + 1 < chars.len() && chars[i + 1] == '}' {
            result.push('}');
            i += 2;
        } else if chars[i] == '%' {
            result.push_str("%%");
            i += 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Generate C format specifier from FormatSpec.
fn generate_c_specifier(
    spec: &FormatSpec,
    is_64bit: bool,
    is_float: bool,
    is_string: bool,
    is_unsigned: bool,
) -> String {
    let mut result = String::from("%");

    // Add flags
    if spec.left_justify {
        result.push('-');
    }
    if spec.always_sign {
        result.push('+');
    }
    if spec.space_sign {
        result.push(' ');
    }
    if spec.alternate {
        result.push('#');
    }
    if spec.zero_pad {
        result.push('0');
    }

    // Add width
    match &spec.width {
        Some(Width::Literal(w)) => result.push_str(&w.to_string()),
        Some(Width::Runtime) => result.push('*'),
        None => {}
    }

    // Add precision
    match &spec.precision {
        Some(Precision::Literal(p)) => {
            result.push('.');
            result.push_str(&p.to_string());
        }
        Some(Precision::Runtime) => result.push_str(".*"),
        None => {}
    }

    // Add size modifier for 64-bit integers
    if is_64bit && !is_float && !is_string {
        result.push_str("ll");
    }

    // Add type character
    let type_char = match spec.format_type {
        Some('c') => 'c',
        Some('s') => 's',
        Some('d') | Some('i') => 'd',
        Some('o') => 'o',
        Some('x') => 'x',
        Some('X') => 'X',
        Some('e') => 'e',
        Some('E') => 'E',
        Some('f') => 'f',
        Some('g') => 'g',
        Some('G') => 'G',
        Some('a') => 'a',
        Some('A') => 'A',
        Some('p') => 'p',
        Some('b') => 'b', // Binary - we'll handle this specially
        Some(_) => {
            // Unknown format specifier - fall back to default
            if is_string {
                's'
            } else if is_float {
                'f'
            } else {
                'd'
            }
        }
        None => {
            // Default type based on argument type
            if is_string {
                's'
            } else if is_float {
                'f'
            } else if is_unsigned {
                'u'
            } else {
                'd'
            }
        }
    };
    result.push(type_char);

    result
}

/// How a `{}` argument with no type character should be packed.
enum DefaultArgKind {
    /// Float literal or `as f32` / `as f64`.
    Float,
    /// Unsigned literal suffix or `as u8` / `u16` / `u32` / `u64` / `usize`.
    Unsigned,
    /// Signed or unsuffixed integer literal, `bool`, or a signed cast.
    Signed,
    /// Binding, call, or other expression whose type the macro cannot see.
    Unknown,
}

fn peel_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Group(group) => peel_expr(&group.expr),
        Expr::Paren(paren) => peel_expr(&paren.expr),
        other => other,
    }
}

fn classify_type(ty: &syn::Type) -> DefaultArgKind {
    let syn::Type::Path(path) = ty else {
        return DefaultArgKind::Unknown;
    };
    if path.qself.is_some() {
        return DefaultArgKind::Unknown;
    }
    let Some(ident) = path.path.get_ident() else {
        return DefaultArgKind::Unknown;
    };
    match ident.to_string().as_str() {
        "f32" | "f64" => DefaultArgKind::Float,
        "u8" | "u16" | "u32" | "u64" | "usize" => DefaultArgKind::Unsigned,
        "i8" | "i16" | "i32" | "i64" | "isize" | "bool" => DefaultArgKind::Signed,
        _ => DefaultArgKind::Unknown,
    }
}

fn classify_default_arg(expr: &Expr) -> DefaultArgKind {
    match peel_expr(expr) {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Float(_) => DefaultArgKind::Float,
            syn::Lit::Int(int) => match int.suffix() {
                "u8" | "u16" | "u32" | "u64" | "usize" => DefaultArgKind::Unsigned,
                _ => DefaultArgKind::Signed,
            },
            syn::Lit::Bool(_) => DefaultArgKind::Signed,
            _ => DefaultArgKind::Unknown,
        },
        Expr::Cast(cast) => classify_type(&cast.ty),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            classify_default_arg(&unary.expr)
        }
        _ => DefaultArgKind::Unknown,
    }
}

fn append_ascii(bytes: &[u8]) -> TokenStream2 {
    quote! {
        #(
            __buf[__i] = #bytes;
            __i += 1;
        )*
    }
}

/// Specifier bytes for a default argument whose type is only known later.
///
/// `GpuPrintfArg` picks `%f`, `%u` / `%llu`, or `%d` / `%lld`. `ll` is omitted
/// for `%p` (pointers are already pointer-sized).
fn append_inferred_spec(ty: &syn::Ident, spec: &FormatSpec) -> TokenStream2 {
    let mut prefix = String::from("%");
    if spec.left_justify {
        prefix.push('-');
    }
    if spec.always_sign {
        prefix.push('+');
    }
    if spec.space_sign {
        prefix.push(' ');
    }
    if spec.alternate {
        prefix.push('#');
    }
    if spec.zero_pad {
        prefix.push('0');
    }
    match &spec.width {
        Some(Width::Literal(w)) => prefix.push_str(&w.to_string()),
        Some(Width::Runtime) => prefix.push('*'),
        None => {}
    }
    match &spec.precision {
        Some(Precision::Literal(p)) => {
            prefix.push('.');
            prefix.push_str(&p.to_string());
        }
        Some(Precision::Runtime) => prefix.push_str(".*"),
        None => {}
    }
    let prefix = append_ascii(prefix.as_bytes());
    quote! {
        #prefix
        if !#ty::IS_FLOAT && #ty::IS_64BIT && #ty::FORMAT_CHAR != 'p' {
            __buf[__i] = b'l';
            __i += 1;
            __buf[__i] = b'l';
            __i += 1;
        }
        __buf[__i] = if #ty::IS_FLOAT {
            b'f'
        } else {
            #ty::FORMAT_CHAR as u8
        };
        __i += 1;
    }
}

/// Fixed C specifier, or `None` when the argument type must be read from
/// [`cuda_device::debug::GpuPrintfArg`] at the call site.
enum SpecPiece {
    Fixed(String),
    Inferred(FormatSpec),
}

/// Main proc macro implementation (called from lib.rs)
pub fn gpu_printf_impl(input: GpuPrintfInput) -> TokenStream2 {
    let format_str = input.format_string.value();

    // Parse the format string
    let placeholders = match parse_format_string(&format_str) {
        Ok(p) => p,
        Err(e) => {
            return syn::Error::new(input.format_string.span(), e).to_compile_error();
        }
    };

    // Count total expected arguments (including runtime width/precision)
    let mut expected_args = 0;
    for p in &placeholders {
        expected_args += 1; // The value itself
        expected_args += count_runtime_args(&p.spec);
    }

    // Validate argument count
    if input.args.len() != expected_args {
        return syn::Error::new(
            input.format_string.span(),
            format!(
                "format string requires {} arguments but {} were supplied",
                expected_args,
                input.args.len()
            ),
        )
        .to_compile_error();
    }

    // Check max args limit
    if input.args.len() > MAX_PRINTF_ARGS {
        return syn::Error::new(
            input.format_string.span(),
            format!(
                "gpu_printf! supports at most {} arguments, {} were supplied",
                MAX_PRINTF_ARGS,
                input.args.len()
            ),
        )
        .to_compile_error();
    }

    // If no arguments, simple case
    if placeholders.is_empty() {
        let c_format = build_c_format_string(&format_str, &placeholders);
        // Create byte string literal for format string (avoids str type)
        let format_with_null = format!("{}\0", c_format);
        let format_lit =
            syn::LitByteStr::new(format_with_null.as_bytes(), proc_macro2::Span::call_site());

        return quote! {
            {
                // No arguments - pass null for args pointer
                // Use byte string literal directly (avoids str type which GPU doesn't support)
                cuda_device::debug::__gpu_vprintf(
                    #format_lit.as_ptr(),
                    core::ptr::null()
                )
            }
        };
    }

    // Generate the argument struct fields and initializers
    let mut field_defs: Vec<TokenStream2> = Vec::new();
    let mut field_inits: Vec<TokenStream2> = Vec::new();
    let mut c_specifiers: Vec<SpecPiece> = Vec::new();
    let mut arg_idx = 0;
    let mut inferred_vals: Vec<TokenStream2> = Vec::new();
    let mut inferred_val_idents: Vec<syn::Ident> = Vec::new();
    let mut inferred_tys: Vec<syn::Ident> = Vec::new();
    let mut promoted_tys: Vec<syn::Ident> = Vec::new();

    for (placeholder_idx, placeholder) in placeholders.iter().enumerate() {
        let spec = &placeholder.spec;

        // Handle runtime width
        if matches!(spec.width, Some(Width::Runtime)) {
            let field_name = format_ident!("__width_{}", placeholder_idx);
            let arg = &input.args[arg_idx];
            field_defs.push(quote! { #field_name: i32 });
            field_inits.push(quote! { #field_name: (#arg) as i32 });
            arg_idx += 1;
        }

        // Handle runtime precision
        if matches!(spec.precision, Some(Precision::Runtime)) {
            let field_name = format_ident!("__prec_{}", placeholder_idx);
            let arg = &input.args[arg_idx];
            field_defs.push(quote! { #field_name: i32 });
            field_inits.push(quote! { #field_name: (#arg) as i32 });
            arg_idx += 1;
        }

        // Handle the value argument
        let field_name = format_ident!("__arg_{}", placeholder_idx);
        let arg = &input.args[arg_idx];
        arg_idx += 1;

        // Generate field type and initialization based on format specifier
        // Use concrete types - C vararg promotion rules apply:
        // - f32 → f64 (always)
        // - i8, i16 → i32; u8, u16 → u32
        // - i32, u32, i64, u64 → unchanged
        // We use i64/u64 as the default to handle all integer sizes safely
        let (field_ty, field_init, is_64bit, is_float, is_string) = match spec.format_type {
            Some('c') => {
                // Character: promote to i32
                (
                    quote! { i32 },
                    quote! { (#arg) as i32 },
                    false,
                    false,
                    false,
                )
            }
            Some('s') => {
                // String: pass pointer
                (
                    quote! { *const u8 },
                    quote! { {
                        // Ensure string is null-terminated
                        const __STR: &str = concat!(#arg, "\0");
                        __STR.as_ptr()
                    }},
                    false,
                    false,
                    true,
                )
            }
            Some('p') => {
                // Pointer
                (quote! { u64 }, quote! { (#arg) as u64 }, true, false, false)
            }
            Some('e') | Some('E') | Some('f') | Some('g') | Some('G') | Some('a') | Some('A') => {
                // Float: always promote to f64
                (quote! { f64 }, quote! { (#arg) as f64 }, false, true, false)
            }
            Some('x') | Some('X') | Some('o') | Some('b') => {
                // Hex/octal/binary: use u64 to handle all unsigned integer sizes
                (quote! { u64 }, quote! { (#arg) as u64 }, true, false, false)
            }
            Some('d') | Some('i') => {
                // Signed integer: use i64 to handle all signed integer sizes
                (quote! { i64 }, quote! { (#arg) as i64 }, true, false, false)
            }
            // `{:.N}` has no type character. Precision on `%d` is a minimum
            // digit count, so packing a float as i64 prints "03" for 3.14.
            // The documented conversion is `%f`. Untyped arguments take the
            // same path: the macro cannot see their type, and guessing i64
            // would truncate the float.
            None if spec.precision.is_some() => {
                (quote! { f64 }, quote! { (#arg) as f64 }, false, true, false)
            }
            None => match classify_default_arg(arg) {
                // Visible unsigned values must not be narrowed through `as i64`:
                // a u64 above i64::MAX would print as a negative %lld.
                DefaultArgKind::Unsigned => {
                    (quote! { u64 }, quote! { (#arg) as u64 }, true, false, false)
                }
                DefaultArgKind::Float => {
                    (quote! { f64 }, quote! { (#arg) as f64 }, false, true, false)
                }
                DefaultArgKind::Signed => {
                    (quote! { i64 }, quote! { (#arg) as i64 }, true, false, false)
                }
                // Type is not in the tokens. `GpuPrintfArg` selects the
                // conversion so a float is not truncated and a u64 is not
                // wrapped. The sentinel tuple is replaced below.
                DefaultArgKind::Unknown => {
                    (quote! { i64 }, quote! { (#arg) as i64 }, true, false, false)
                }
            },
            _ => {
                // Fallback to i64
                (quote! { i64 }, quote! { (#arg) as i64 }, true, false, false)
            }
        };

        let inferred = spec.format_type.is_none()
            && spec.precision.is_none()
            && matches!(classify_default_arg(arg), DefaultArgKind::Unknown);
        if inferred {
            let infer_idx = inferred_tys.len();
            let ty_param = format_ident!("__T{}", infer_idx);
            let promoted = format_ident!("__P{}", infer_idx);
            let val = format_ident!("__val_{}", placeholder_idx);
            inferred_tys.push(ty_param);
            promoted_tys.push(promoted.clone());
            inferred_val_idents.push(val.clone());
            inferred_vals.push(quote! { let #val = #arg; });
            field_defs.push(quote! { #field_name: #promoted });
            field_inits
                .push(quote! { #field_name: cuda_device::debug::GpuPrintfArg::promote(#val) });
            c_specifiers.push(SpecPiece::Inferred(spec.clone()));
        } else {
            let is_unsigned = spec.format_type.is_none()
                && spec.precision.is_none()
                && matches!(classify_default_arg(arg), DefaultArgKind::Unsigned);
            field_defs.push(quote! { #field_name: #field_ty });
            field_inits.push(quote! { #field_name: #field_init });
            let c_spec = generate_c_specifier(spec, is_64bit, is_float, is_string, is_unsigned);
            c_specifiers.push(SpecPiece::Fixed(c_spec));
        }
    }

    // Generate unique struct name
    let struct_name = format_ident!("__GpuPrintfArgs");

    let expanded = if inferred_tys.is_empty() {
        // Build the final C format string
        let mut c_format = build_c_format_string(&format_str, &placeholders);
        for c_spec in &c_specifiers {
            let SpecPiece::Fixed(c_spec) = c_spec else {
                unreachable!("concrete printf args have fixed specifiers");
            };
            c_format = c_format.replacen("__PLACEHOLDER__", c_spec, 1);
        }

        // Create byte string literal for format string (avoids str type)
        let format_with_null = format!("{}\0", c_format);
        let format_lit =
            syn::LitByteStr::new(format_with_null.as_bytes(), proc_macro2::Span::call_site());

        quote! {
            {
                // Argument packing struct with C-compatible layout
                #[repr(C)]
                struct #struct_name {
                    #(#field_defs),*
                }

                // Pack arguments
                let __args = #struct_name {
                    #(#field_inits),*
                };

                // Call vprintf intrinsic
                // Use byte string literal directly (avoids str type which GPU doesn't support)
                cuda_device::debug::__gpu_vprintf(
                    #format_lit.as_ptr(),
                    &__args as *const #struct_name as *const u8
                )
            }
        }
    } else {
        let skeleton = build_c_format_string(&format_str, &placeholders);
        let parts: Vec<&str> = skeleton.split("__PLACEHOLDER__").collect();
        let mut writes = Vec::new();
        let mut infer_idx = 0usize;
        for (part_idx, part) in parts.iter().enumerate() {
            writes.push(append_ascii(part.as_bytes()));
            if part_idx >= c_specifiers.len() {
                continue;
            }
            match &c_specifiers[part_idx] {
                SpecPiece::Fixed(spec) => writes.push(append_ascii(spec.as_bytes())),
                SpecPiece::Inferred(spec) => {
                    writes.push(append_inferred_spec(&inferred_tys[infer_idx], spec));
                    infer_idx += 1;
                }
            }
        }
        // Flags + width + precision + `ll` + type character, per placeholder.
        let buf_len = format_str.len() + placeholders.len() * 48 + 8;
        let fmt_params = inferred_tys.clone();
        let fmt_param_refs = fmt_params.clone();
        quote! {
            {
                #[inline(always)]
                #[allow(unused_variables)]
                fn __oxide_gpu_printf_format<#(#fmt_params: cuda_device::debug::GpuPrintfArg),*>(
                    #( _: &#fmt_param_refs ),*
                ) -> &'static [u8] {
                    const { &{
                        let mut __buf = [0u8; #buf_len];
                        let mut __i = 0usize;
                        #(#writes)*
                        __buf[__i] = 0;
                        let _ = __i;
                        __buf
                    }}
                }

                #(#inferred_vals)*

                // Argument packing struct with C-compatible layout.
                // Inferred slots use the `GpuPrintfArg` promoted type.
                #[repr(C)]
                struct #struct_name<#(#promoted_tys),*> {
                    #(#field_defs),*
                }

                let __fmt = __oxide_gpu_printf_format(#(&#inferred_val_idents),*);
                let __args = #struct_name {
                    #(#field_inits),*
                };

                cuda_device::debug::__gpu_vprintf(
                    __fmt.as_ptr(),
                    &__args as *const _ as *const u8
                )
            }
        }
    };

    expanded
}
