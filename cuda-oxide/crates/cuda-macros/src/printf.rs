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
//! 1. Parses the Rust-style format string (`{}`, `{:x}`, `{:.2}`, and so on)
//! 2. Evaluates every argument, in order, into a local binding
//! 3. Packs the bindings into a `#[repr(C)]` struct
//! 4. Calls `__gpu_vprintf(format_ptr, args_ptr)`, which the compiler lowers
//!    to `vprintf`
//!
//! # Choosing the C conversion
//!
//! A placeholder with a type character (`{:x}`, `{:.2f}`, `{:d}`, `{:c}`, ...)
//! has a fixed conversion. The macro casts the argument to the matching C
//! vararg type (`as u64`, `as f64`, `as i64`, `as i32`) and, when every
//! placeholder is of this kind, writes the whole format string as a byte
//! string literal.
//!
//! A placeholder without a type character (`{}`, `{:8}`, `{:08}`, `{:.2}`)
//! takes the conversion from the argument's type through
//! `cuda_device::debug::GpuPrintfArg`: `%d` or `%lld` for signed integers,
//! `%u` or `%llu` for unsigned ones, `%f` for floats, `%d` for `bool` and `%p`
//! for raw pointers. The macro cannot see that type, so it emits a generic
//! helper with one `GpuPrintfArg` parameter per such placeholder. An inline
//! const inside the helper calls `cuda_device::debug::__gpu_printf_format`
//! with the literal text and the per-placeholder specs as data, and the
//! trait's constants pick the type character after monomorphization. The
//! helper returns `&'static [u8; N]`, the pointer-to-array constant the device
//! importer accepts. `N` is computed here from the same data and leaves room
//! for an `ll` modifier on every typed placeholder.
//!
//! # C vararg promotion rules
//!
//! Arguments are promoted following C vararg rules:
//! - `i8`, `i16` → `i32`
//! - `u8`, `u16` → `u32`
//! - `f32` → `f64`
//! - `bool` → `i32` (0 or 1)
//! - pointers → `u64`

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    Expr, LitByteStr, LitStr, Token,
    parse::{Parse, ParseStream},
};

/// Maximum number of printf arguments (CUDA limitation)
const MAX_PRINTF_ARGS: usize = 32;

/// Bytes a typed conversion needs beyond its prefix: `%`, `ll` and the type
/// character. The builder writes `ll` only for 64-bit integers; the rest of
/// the room stays zero after the NUL.
const TYPED_SPEC_OVERHEAD: usize = 4;

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
    /// Literal precision (None = no precision specified). Device printf
    /// accepts digits only here, so there is no runtime form.
    precision: Option<usize>,
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

    // Parse precision. CUDA device printf takes `*` for the width only; a
    // runtime precision prints its own specifier text instead of a value.
    if i < chars.len() && chars[i] == '.' {
        i += 1; // Skip `.`
        if i < chars.len() && chars[i] == '*' {
            return Err(
                "runtime precision `.*` is not supported by CUDA device printf; \
                 use a literal precision such as `{:.2}`"
                    .to_string(),
            );
        }
        if i < chars.len() && chars[i].is_ascii_digit() {
            let (prec, new_i) = parse_number(chars, i);
            spec.precision = Some(prec);
            i = new_i;
        } else {
            // `.` without number means precision 0
            spec.precision = Some(0);
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

/// Literal text around the placeholders, with `%` doubled and `{{` / `}}`
/// unescaped. Returns one more part than there are placeholders.
fn literal_parts(format: &str, placeholders: &[Placeholder]) -> Vec<String> {
    let chars: Vec<char> = format.chars().collect();
    let mut parts = Vec::with_capacity(placeholders.len() + 1);
    let mut last_end = 0;
    for placeholder in placeholders {
        parts.push(unescape_literal(&chars[last_end..placeholder.start]));
        last_end = placeholder.end;
    }
    parts.push(unescape_literal(&chars[last_end..]));
    parts
}

/// One run of literal text, converted to C printf text.
fn unescape_literal(chars: &[char]) -> String {
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '{' if i + 1 < chars.len() && chars[i + 1] == '{' => {
                result.push('{');
                i += 2;
            }
            '}' if i + 1 < chars.len() && chars[i + 1] == '}' => {
                result.push('}');
                i += 2;
            }
            '%' => {
                // Escape literal % as %%
                result.push_str("%%");
                i += 1;
            }
            c => {
                result.push(c);
                i += 1;
            }
        }
    }
    result
}

/// Flags, width and precision of a placeholder, as they appear between the
/// `%` and the type character.
fn spec_prefix(spec: &FormatSpec) -> String {
    let mut prefix = String::new();
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
    if let Some(p) = spec.precision {
        prefix.push('.');
        prefix.push_str(&p.to_string());
    }
    prefix
}

/// The complete C conversion of a placeholder with an explicit type character.
fn fixed_specifier(
    spec: &FormatSpec,
    type_char: char,
    is_64bit: bool,
    is_float: bool,
    is_string: bool,
) -> String {
    let mut result = String::from("%");
    result.push_str(&spec_prefix(spec));

    // Add size modifier for 64-bit integers
    if is_64bit && !is_float && !is_string {
        result.push_str("ll");
    }

    result.push(if type_char == 'i' { 'd' } else { type_char });
    result
}

/// A NUL-terminated byte string literal: the form of a static format string
/// the device accepts (a `str` would not be).
fn c_string_literal(text: &str) -> LitByteStr {
    LitByteStr::new(format!("{text}\0").as_bytes(), Span::call_site())
}

/// How one placeholder prints its argument.
enum Conversion {
    /// Explicit type character: the macro knows the whole C conversion.
    Fixed(String),
    /// No type character: the argument's `GpuPrintfArg` impl supplies the
    /// type character after monomorphization. `prefix` is the text between
    /// the `%` and that character, `ty` the helper's type parameter.
    Typed { prefix: String, ty: syn::Ident },
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

    // Count total expected arguments (including runtime width)
    let expected_args: usize = placeholders
        .iter()
        .map(|p| 1 + usize::from(matches!(p.spec.width, Some(Width::Runtime))))
        .sum();

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

    let parts = literal_parts(&format_str, &placeholders);

    // If no arguments, simple case
    if placeholders.is_empty() {
        let format_lit = c_string_literal(&parts[0]);
        return quote! {
            {
                // No arguments - pass null for args pointer
                cuda_device::debug::__gpu_vprintf(
                    #format_lit.as_ptr(),
                    core::ptr::null()
                )
            }
        };
    }

    // Every argument is evaluated into a local, in source order, before the
    // packing struct is built. The locals carry mixed-site spans so a later
    // argument expression cannot see an earlier binding.
    let mut hoisted: Vec<TokenStream2> = Vec::new();
    let mut field_defs: Vec<TokenStream2> = Vec::new();
    let mut field_inits: Vec<TokenStream2> = Vec::new();
    let mut conversions: Vec<Conversion> = Vec::new();
    let mut typed_params: Vec<syn::Ident> = Vec::new();
    let mut promoted_params: Vec<syn::Ident> = Vec::new();
    let mut typed_values: Vec<syn::Ident> = Vec::new();
    let mut arg_idx = 0;

    for (placeholder_idx, placeholder) in placeholders.iter().enumerate() {
        let spec = &placeholder.spec;

        // Handle runtime width
        if matches!(spec.width, Some(Width::Runtime)) {
            let value = format_ident!("__val_{}", arg_idx, span = Span::mixed_site());
            let field = format_ident!("__width_{}", placeholder_idx);
            let arg = &input.args[arg_idx];
            hoisted.push(quote! { let #value = (#arg) as i32; });
            field_defs.push(quote! { #field: i32 });
            field_inits.push(quote! { #field: #value });
            arg_idx += 1;
        }

        // Handle the value argument
        let value = format_ident!("__val_{}", arg_idx, span = Span::mixed_site());
        let field = format_ident!("__arg_{}", placeholder_idx);
        let arg = &input.args[arg_idx];
        arg_idx += 1;

        match spec.format_type {
            None => {
                let ty = format_ident!("__T{}", typed_params.len());
                let promoted = format_ident!("__P{}", typed_params.len());
                hoisted.push(quote! { let #value = #arg; });
                field_defs.push(quote! { #field: #promoted });
                field_inits.push(quote! {
                    #field: cuda_device::debug::GpuPrintfArg::promote(#value)
                });
                typed_params.push(ty.clone());
                promoted_params.push(promoted);
                typed_values.push(value);
                conversions.push(Conversion::Typed {
                    prefix: spec_prefix(spec),
                    ty,
                });
            }
            Some(type_char) => {
                // Explicit type character: cast to the C vararg type.
                // - f32 → f64 (always)
                // - integers → i64 or u64, so every width fits
                let (field_ty, init, is_64bit, is_float, is_string) = match type_char {
                    'c' => (
                        quote! { i32 },
                        quote! { (#arg) as i32 },
                        false,
                        false,
                        false,
                    ),
                    's' => (
                        quote! { *const u8 },
                        quote! { {
                            // Ensure string is null-terminated
                            const __STR: &str = concat!(#arg, "\0");
                            __STR.as_ptr()
                        } },
                        false,
                        false,
                        true,
                    ),
                    'p' => (quote! { u64 }, quote! { (#arg) as u64 }, true, false, false),
                    'e' | 'E' | 'f' | 'g' | 'G' | 'a' | 'A' => {
                        (quote! { f64 }, quote! { (#arg) as f64 }, false, true, false)
                    }
                    'x' | 'X' | 'o' | 'b' => {
                        (quote! { u64 }, quote! { (#arg) as u64 }, true, false, false)
                    }
                    // `d`, `i` and anything else the parser accepts
                    _ => (quote! { i64 }, quote! { (#arg) as i64 }, true, false, false),
                };
                hoisted.push(quote! { let #value = #init; });
                field_defs.push(quote! { #field: #field_ty });
                field_inits.push(quote! { #field: #value });
                conversions.push(Conversion::Fixed(fixed_specifier(
                    spec, type_char, is_64bit, is_float, is_string,
                )));
            }
        }
    }

    let args_ident = format_ident!("__args", span = Span::mixed_site());

    if typed_params.is_empty() {
        // Every conversion is known here: emit the format string as a literal.
        let mut c_format = String::new();
        for (part, conversion) in parts.iter().zip(&conversions) {
            c_format.push_str(part);
            match conversion {
                Conversion::Fixed(spec) => c_format.push_str(spec),
                Conversion::Typed { .. } => unreachable!("no typed placeholders on this route"),
            }
        }
        c_format.push_str(&parts[conversions.len()]);
        let format_lit = c_string_literal(&c_format);

        return quote! {
            {
                // Argument packing struct with C-compatible layout
                #[repr(C)]
                struct __GpuPrintfArgs {
                    #(#field_defs),*
                }

                #(#hoisted)*

                let #args_ident = __GpuPrintfArgs {
                    #(#field_inits),*
                };

                cuda_device::debug::__gpu_vprintf(
                    #format_lit.as_ptr(),
                    &#args_ident as *const __GpuPrintfArgs as *const u8
                )
            }
        };
    }

    // At least one conversion depends on an argument type. Hand the literal
    // text and the specs to the builder in cuda-device; its result is sized
    // from the same data.
    let part_lits: Vec<LitByteStr> = parts
        .iter()
        .map(|part| LitByteStr::new(part.as_bytes(), Span::call_site()))
        .collect();
    let mut buf_len = parts.iter().map(String::len).sum::<usize>() + 1;
    let mut spec_exprs: Vec<TokenStream2> = Vec::new();
    for conversion in &conversions {
        match conversion {
            Conversion::Fixed(spec) => {
                buf_len += spec.len();
                let lit = LitByteStr::new(spec.as_bytes(), Span::call_site());
                spec_exprs.push(quote! { cuda_device::debug::__GpuPrintfSpec::Fixed(#lit) });
            }
            Conversion::Typed { prefix, ty } => {
                buf_len += prefix.len() + TYPED_SPEC_OVERHEAD;
                let lit = LitByteStr::new(prefix.as_bytes(), Span::call_site());
                spec_exprs.push(quote! {
                    cuda_device::debug::__GpuPrintfSpec::Typed {
                        prefix: #lit,
                        format_char: <#ty as cuda_device::debug::GpuPrintfArg>::FORMAT_CHAR,
                        is_64bit: <#ty as cuda_device::debug::GpuPrintfArg>::IS_64BIT,
                        is_float: <#ty as cuda_device::debug::GpuPrintfArg>::IS_FLOAT,
                    }
                });
            }
        }
    }
    let fmt_ident = format_ident!("__fmt", span = Span::mixed_site());

    quote! {
        {
            // The format string of this call site, built once per
            // instantiation. The sized array is the constant form the
            // device importer accepts.
            #[inline(always)]
            fn __oxide_gpu_printf_format<#(#typed_params: cuda_device::debug::GpuPrintfArg),*>(
                #(_: &#typed_params),*
            ) -> &'static [u8; #buf_len] {
                const {
                    &cuda_device::debug::__gpu_printf_format::<#buf_len>(
                        &[#(#part_lits),*],
                        &[#(#spec_exprs),*],
                    )
                }
            }

            // Argument packing struct with C-compatible layout. Typed slots
            // hold the `GpuPrintfArg` promoted type.
            #[repr(C)]
            struct __GpuPrintfArgs<#(#promoted_params),*> {
                #(#field_defs),*
            }

            #(#hoisted)*

            let #fmt_ident = __oxide_gpu_printf_format(#(&#typed_values),*);
            let #args_ident = __GpuPrintfArgs {
                #(#field_inits),*
            };

            cuda_device::debug::__gpu_vprintf(
                #fmt_ident.as_ptr(),
                &#args_ident as *const _ as *const u8
            )
        }
    }
}
