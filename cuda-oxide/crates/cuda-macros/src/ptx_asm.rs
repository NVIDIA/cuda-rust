/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, LitStr, Token, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

const MAX_PTX_ASM_INPUTS: usize = 16;
// Blackwell tensor-memory loads expose up to 64 scalar destination registers
// in one indivisible PTX instruction. Keeping the public inline-PTX surface at
// 16 forces those results through an aggregate-return shim and defeats register
// promotion, so admit the hardware's complete output pack.
const MAX_PTX_ASM_OUTPUTS: usize = 64;
const REGISTER_ONLY_OPTION: &str = "register_only";
const MAY_DIVERGE_OPTION: &str = "may_diverge";
const REGISTER_ONLY_MAY_DIVERGE_OPTIONS: &str = "register_only,may_diverge";
const SUPPORTED_INPUT_CONSTRAINTS: &[&str] = &["h", "r", "l", "q", "f", "d", "n", "C"];
const SUPPORTED_OUTPUT_CONSTRAINTS: &[&str] = &["=h", "=r", "=l", "=q", "=f", "=d"];
const COMPILE_TIME_STRING_CONSTRAINT: &str = "C";
const SUPPORTED_INOUT_CONSTRAINTS: &[&str] = &["+h", "+r", "+l", "+q", "+f", "+d"];

pub struct PtxAsmInput {
    template: LitStr,
    operands: Vec<PtxAsmOperand>,
}

enum PtxAsmOperand {
    Out { constraint: LitStr, place: Expr },
    In { constraint: LitStr, expr: Expr },
    InOut { constraint: LitStr, place: Expr },
    Clobber { name: LitStr },
    Options { options: Vec<Ident> },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PtxAsmOutputKind {
    WriteOnly,
    ReadWrite,
}

struct PtxAsmOutput {
    constraint: LitStr,
    place: Expr,
    kind: PtxAsmOutputKind,
}

#[derive(Default)]
struct PtxAsmOptions {
    register_only: bool,
    may_diverge: bool,
}

impl Parse for PtxAsmInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let template: LitStr = input.parse()?;
        let mut operands = Vec::new();

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }

            if input.peek(Token![in]) {
                input.parse::<Token![in]>()?;
                let constraint = parse_parenthesized_string(input)?;
                let expr: Expr = input.parse()?;
                operands.push(PtxAsmOperand::In { constraint, expr });
                continue;
            }

            let ident: syn::Ident = input.parse()?;
            match ident.to_string().as_str() {
                "out" => {
                    let constraint = parse_parenthesized_string(input)?;
                    let place: Expr = input.parse()?;
                    operands.push(PtxAsmOperand::Out { constraint, place });
                }
                "inout" => {
                    let constraint = parse_parenthesized_string(input)?;
                    let place: Expr = input.parse()?;
                    operands.push(PtxAsmOperand::InOut { constraint, place });
                }
                "clobber" => {
                    let name = parse_parenthesized_string(input)?;
                    operands.push(PtxAsmOperand::Clobber { name });
                }
                "options" => {
                    let options = parse_parenthesized_options(input)?;
                    operands.push(PtxAsmOperand::Options { options });
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unsupported ptx_asm! operand `{other}`; expected `out`, `inout`, `in`, `clobber`, or `options`"
                        ),
                    ));
                }
            }
        }

        Ok(Self { template, operands })
    }
}

fn parse_parenthesized_string(input: ParseStream) -> syn::Result<LitStr> {
    let content;
    parenthesized!(content in input);
    let lit: LitStr = content.parse()?;
    if !content.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "expected a single string literal in parentheses",
        ));
    }
    Ok(lit)
}

fn parse_parenthesized_options(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    parenthesized!(content in input);
    let options: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
    if options.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "options(...) requires at least one option",
        ));
    }
    Ok(options.into_iter().collect())
}

pub fn ptx_asm_impl(input: PtxAsmInput) -> TokenStream2 {
    match build_ptx_asm(input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

fn build_ptx_asm(input: PtxAsmInput) -> syn::Result<TokenStream2> {
    let mut outputs: Vec<PtxAsmOutput> = Vec::new();
    let mut inputs: Vec<(LitStr, Expr)> = Vec::new();
    let mut clobbers: Vec<LitStr> = Vec::new();
    let mut options = PtxAsmOptions::default();
    let mut saw_input = false;

    for operand in input.operands {
        match operand {
            PtxAsmOperand::Out { constraint, place } => {
                if saw_input {
                    return Err(syn::Error::new(
                        constraint.span(),
                        "`out` operands must appear before `in` operands",
                    ));
                }
                validate_output_constraint(&constraint)?;
                outputs.push(PtxAsmOutput {
                    constraint,
                    place,
                    kind: PtxAsmOutputKind::WriteOnly,
                });
            }
            PtxAsmOperand::InOut { constraint, place } => {
                if saw_input {
                    return Err(syn::Error::new(
                        constraint.span(),
                        "`inout` operands must appear before `in` operands",
                    ));
                }
                validate_inout_constraint(&constraint)?;
                outputs.push(PtxAsmOutput {
                    constraint,
                    place,
                    kind: PtxAsmOutputKind::ReadWrite,
                });
            }
            PtxAsmOperand::In { constraint, expr } => {
                validate_input_constraint(&constraint)?;
                saw_input = true;
                inputs.push((constraint, expr));
            }
            PtxAsmOperand::Clobber { name } => clobbers.push(name),
            PtxAsmOperand::Options {
                options: option_idents,
            } => {
                for option in option_idents {
                    match option.to_string().as_str() {
                        REGISTER_ONLY_OPTION => {
                            if options.register_only {
                                return Err(syn::Error::new(
                                    option.span(),
                                    "`options(register_only)` was specified more than once",
                                ));
                            }
                            options.register_only = true;
                        }
                        MAY_DIVERGE_OPTION => {
                            if options.may_diverge {
                                return Err(syn::Error::new(
                                    option.span(),
                                    "`options(may_diverge)` was specified more than once",
                                ));
                            }
                            options.may_diverge = true;
                        }
                        other => {
                            return Err(syn::Error::new(
                                option.span(),
                                format!(
                                    "unsupported ptx_asm! option `{other}`; expected `register_only` or `may_diverge`"
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }

    if outputs.len() > MAX_PTX_ASM_OUTPUTS {
        return Err(syn::Error::new(
            input.template.span(),
            format!(
                "ptx_asm! supports at most {MAX_PTX_ASM_OUTPUTS} output operands across `out` and `inout`"
            ),
        ));
    }

    if options.register_only && outputs.is_empty() {
        return Err(syn::Error::new(
            input.template.span(),
            "`options(register_only)` requires an `out` or `inout` operand",
        ));
    }
    if options.register_only && !clobbers.is_empty() {
        return Err(syn::Error::new(
            clobbers[0].span(),
            "`options(register_only)` cannot be used with clobbers",
        ));
    }
    if options.may_diverge && !options.register_only {
        return Err(syn::Error::new(
            input.template.span(),
            "`options(may_diverge)` requires `register_only`",
        ));
    }

    if inputs.len() > MAX_PTX_ASM_INPUTS {
        return Err(syn::Error::new(
            input.template.span(),
            format!("ptx_asm! supports at most {MAX_PTX_ASM_INPUTS} input operands"),
        ));
    }

    // Hidden tied inputs initialize read-write outputs but are not visible
    // template operands, so they do not contribute to `%N` validation.
    let operand_count = outputs.len() + inputs.len();
    let converted_template = convert_cuda_template(&input.template, operand_count)?;
    let template_lit = syn::LitByteStr::new(converted_template.as_bytes(), input.template.span());

    let constraints = build_constraint_string(&outputs, &inputs, &clobbers)?;
    let constraints_lit = syn::LitByteStr::new(constraints.as_bytes(), input.template.span());
    let options_marker = if options.register_only && options.may_diverge {
        REGISTER_ONLY_MAY_DIVERGE_OPTIONS
    } else if options.register_only {
        REGISTER_ONLY_OPTION
    } else {
        ""
    };
    let options_lit = syn::LitByteStr::new(options_marker.as_bytes(), input.template.span());

    let input_exprs: Vec<TokenStream2> = inputs
        .iter()
        .map(|(constraint, expr)| {
            if constraint.value() == COMPILE_TIME_STRING_CONSTRAINT {
                // `in("C")` operands must be compile-time byte strings. The
                // typed helper turns any other operand type into a type error
                // at the call site, and the inline const keeps the operand a
                // MIR constant so the importer can splice its text.
                quote! { const { cuda_device::ptx::__ptx_asm_c(#expr) } }
            } else {
                quote! { #expr }
            }
        })
        .collect();

    // Evaluate each read-write place exactly once. A raw pointer keeps that
    // address available across explicit input evaluation without extending a
    // mutable reference borrow over the marker call.
    let mut inout_bindings = Vec::new();
    let mut inout_value_idents = Vec::new();
    let mut inout_ptr_idents: Vec<Option<Ident>> = Vec::with_capacity(outputs.len());

    for (output_index, output) in outputs.iter().enumerate() {
        match output.kind {
            PtxAsmOutputKind::WriteOnly => inout_ptr_idents.push(None),
            PtxAsmOutputKind::ReadWrite => {
                let ptr_ident = format_ident!("__ptx_inout_ptr_{output_index}");
                let value_ident = format_ident!("__ptx_inout_value_{output_index}");
                let place = &output.place;

                inout_bindings.push(quote! {
                    let #ptr_ident = &mut #place as *mut _;
                    let #value_ident = *#ptr_ident;
                });
                inout_value_idents.push(value_ident);
                inout_ptr_idents.push(Some(ptr_ident));
            }
        }
    }

    // LLVM operands follow all explicit inputs with the hidden values used by
    // numeric tied-input constraints.
    let marker_args: Vec<TokenStream2> = input_exprs
        .iter()
        .map(|expr| quote! { #expr })
        .chain(inout_value_idents.iter().map(|ident| quote! { #ident }))
        .collect();
    let arity = marker_args.len();

    if outputs.is_empty() {
        let fn_ident = format_ident!("__ptx_asm_void_{arity}");
        return Ok(quote! {{
            cuda_device::ptx::#fn_ident(
                #template_lit,
                #constraints_lit,
                #options_lit,
                #(#marker_args),*
            );
        }});
    }

    let fn_ident = format_ident!("__ptx_asm_out_{arity}");
    let tuple_vars: Vec<Ident> = (0..outputs.len())
        .map(|i| format_ident!("__ptx_out_{i}"))
        .collect();

    let assignments = outputs.iter().enumerate().zip(tuple_vars.iter()).map(
        |((output_index, output), result_ident)| match output.kind {
            PtxAsmOutputKind::WriteOnly => {
                let place = &output.place;
                quote! {
                    #place = #result_ident;
                }
            }
            PtxAsmOutputKind::ReadWrite => {
                let ptr_ident = inout_ptr_idents[output_index]
                    .as_ref()
                    .expect("read-write outputs always have a generated pointer");
                quote! {
                    *#ptr_ident = #result_ident;
                }
            }
        },
    );

    if outputs.len() == 1 {
        let result_ident = &tuple_vars[0];
        Ok(quote! {{
            #(#inout_bindings)*
            let #result_ident = cuda_device::ptx::#fn_ident(
                #template_lit,
                #constraints_lit,
                #options_lit,
                #(#marker_args),*
            );
            #(#assignments)*
        }})
    } else {
        Ok(quote! {{
            #(#inout_bindings)*
            let ( #(#tuple_vars),* ) = cuda_device::ptx::#fn_ident(
                #template_lit,
                #constraints_lit,
                #options_lit,
                #(#marker_args),*
            );
            #(#assignments)*
        }})
    }
}

fn build_constraint_string(
    outputs: &[PtxAsmOutput],
    inputs: &[(LitStr, Expr)],
    clobbers: &[LitStr],
) -> syn::Result<String> {
    let mut constraints = Vec::new();

    // LLVM represents a CUDA `+r` read-write operand as an `=r` output
    // followed by a numeric input constraint tied to that output.
    for output in outputs {
        match output.kind {
            PtxAsmOutputKind::WriteOnly => constraints.push(output.constraint.value()),
            PtxAsmOutputKind::ReadWrite => {
                let value = output.constraint.value();
                constraints.push(format!("={}", &value[1..]));
            }
        }
    }

    // Explicit inputs retain their user-visible operand order.
    constraints.extend(inputs.iter().map(|(constraint, _)| constraint.value()));

    // Hidden tied inputs follow explicit inputs so they do not change `%N`
    // numbering in the CUDA template.
    for (output_index, output) in outputs.iter().enumerate() {
        if output.kind == PtxAsmOutputKind::ReadWrite {
            constraints.push(output_index.to_string());
        }
    }

    for clobber in clobbers {
        constraints.push(normalize_clobber(clobber)?);
    }

    Ok(constraints.join(","))
}

fn convert_cuda_template(template: &LitStr, operand_count: usize) -> syn::Result<String> {
    let value = template.value();
    let mut converted = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' {
            converted.push_str("$$");
            continue;
        }

        if ch != '%' {
            converted.push(ch);
            continue;
        }

        match chars.peek().copied() {
            Some('%') => {
                chars.next();
                converted.push('%');
            }
            Some(next) if next.is_ascii_digit() => {
                let mut digits = String::new();
                converted.push('$');
                while let Some(digit) = chars.peek().copied() {
                    if digit.is_ascii_digit() {
                        chars.next();
                        digits.push(digit);
                        converted.push(digit);
                    } else {
                        break;
                    }
                }
                let index = digits.parse::<usize>().map_err(|_| {
                    syn::Error::new(
                        template.span(),
                        format!("ptx_asm! template placeholder `%{digits}` is too large"),
                    )
                })?;
                if index >= operand_count {
                    return Err(syn::Error::new(
                        template.span(),
                        format!(
                            "ptx_asm! template placeholder `%{digits}` has no matching operand"
                        ),
                    ));
                }
            }
            Some(other) => {
                let mut literal = String::new();
                for ch in chars.clone() {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.') {
                        literal.push(ch);
                    } else {
                        break;
                    }
                }
                if literal.is_empty() {
                    literal.push(other);
                }
                return Err(syn::Error::new(
                    template.span(),
                    format!(
                        "literal PTX register `%{literal}` must be escaped as `%%{literal}` in ptx_asm!"
                    ),
                ));
            }
            None => {
                return Err(syn::Error::new(
                    template.span(),
                    "trailing `%` in ptx_asm! template",
                ));
            }
        }
    }

    Ok(converted)
}

fn validate_single_constraint(constraint: &LitStr) -> syn::Result<String> {
    let value = constraint.value();
    if value.contains(',') {
        return Err(syn::Error::new(
            constraint.span(),
            "ptx_asm! operand constraints cannot contain `,`; use separate operands and clobber(...)",
        ));
    }
    Ok(value)
}

fn validate_inout_constraint(constraint: &LitStr) -> syn::Result<()> {
    let value = validate_single_constraint(constraint)?;
    if !value.starts_with('+') {
        return Err(syn::Error::new(
            constraint.span(),
            "`inout` constraints must use read-write syntax such as `\"+r\"`",
        ));
    }
    if !SUPPORTED_INOUT_CONSTRAINTS.contains(&value.as_str()) {
        return Err(syn::Error::new(
            constraint.span(),
            format!(
                "unsupported `inout` constraint `{value}`; expected one of {SUPPORTED_INOUT_CONSTRAINTS:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_output_constraint(constraint: &LitStr) -> syn::Result<()> {
    let value = validate_single_constraint(constraint)?;
    if !value.starts_with('=') {
        return Err(syn::Error::new(
            constraint.span(),
            "`out` constraints must use output syntax such as `\"=r\"`",
        ));
    }
    if !SUPPORTED_OUTPUT_CONSTRAINTS.contains(&value.as_str()) {
        return Err(syn::Error::new(
            constraint.span(),
            format!(
                "unsupported `out` constraint `{value}`; expected one of {SUPPORTED_OUTPUT_CONSTRAINTS:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_input_constraint(constraint: &LitStr) -> syn::Result<()> {
    let value = validate_single_constraint(constraint)?;
    if value.starts_with('=') || value.starts_with('+') {
        return Err(syn::Error::new(
            constraint.span(),
            "`in` constraints must use input syntax such as `\"r\"`",
        ));
    }
    if !SUPPORTED_INPUT_CONSTRAINTS.contains(&value.as_str()) {
        return Err(syn::Error::new(
            constraint.span(),
            format!(
                "unsupported `in` constraint `{value}`; expected one of {SUPPORTED_INPUT_CONSTRAINTS:?}"
            ),
        ));
    }
    Ok(())
}

fn normalize_clobber(clobber: &LitStr) -> syn::Result<String> {
    let value = clobber.value();
    if value == "memory" {
        Ok("~{memory}".to_string())
    } else if value.starts_with("~{") && value.ends_with('}') {
        Ok(value)
    } else {
        Err(syn::Error::new(
            clobber.span(),
            "only `clobber(\"memory\")` or raw LLVM clobbers like `clobber(\"~{memory}\")` are supported",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn converts_cuda_placeholders_and_escaped_registers() {
        let template: LitStr = parse_quote!("mov.u32 %0, %%laneid; add.u32 %10, %1, %2;");

        assert_eq!(
            convert_cuda_template(&template, 11).unwrap(),
            "mov.u32 $0, %laneid; add.u32 $10, $1, $2;"
        );
    }

    #[test]
    fn escapes_literal_dollars_for_llvm_inline_asm() {
        let template: LitStr = parse_quote!("$L__BB0: mov.u32 %0, %%laneid;");

        assert_eq!(
            convert_cuda_template(&template, 1).unwrap(),
            "$$L__BB0: mov.u32 $0, %laneid;"
        );
    }

    #[test]
    fn rejects_unescaped_literal_registers() {
        let template: LitStr = parse_quote!("mov.u32 %0, %laneid;");
        let err = convert_cuda_template(&template, 1).unwrap_err();

        assert!(
            err.to_string().contains("must be escaped as `%%laneid`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_placeholders_without_operands() {
        let template: LitStr = parse_quote!("add.u32 %2, %0, %1;");
        let err = convert_cuda_template(&template, 2).unwrap_err();

        assert!(
            err.to_string()
                .contains("placeholder `%2` has no matching operand"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn normalizes_memory_clobber() {
        let clobber: LitStr = parse_quote!("memory");
        assert_eq!(normalize_clobber(&clobber).unwrap(), "~{memory}");
    }

    #[test]
    fn validates_supported_cuda_constraints() {
        for constraint in SUPPORTED_INPUT_CONSTRAINTS {
            let input = LitStr::new(constraint, proc_macro2::Span::call_site());
            assert!(validate_input_constraint(&input).is_ok());
        }
        for constraint in SUPPORTED_OUTPUT_CONSTRAINTS {
            let output = LitStr::new(constraint, proc_macro2::Span::call_site());
            assert!(validate_output_constraint(&output).is_ok());
        }
        for constraint in SUPPORTED_INOUT_CONSTRAINTS {
            let inout = LitStr::new(constraint, proc_macro2::Span::call_site());
            assert!(validate_inout_constraint(&inout).is_ok());
        }
    }

    #[test]
    fn rejects_unsupported_cuda_constraints() {
        for constraint in ["", "x", "rf"] {
            let input = LitStr::new(constraint, proc_macro2::Span::call_site());
            let err = validate_input_constraint(&input).unwrap_err();
            assert!(
                err.to_string().contains("unsupported `in` constraint"),
                "unexpected error for `{constraint}`: {err}"
            );
        }

        for constraint in ["=", "=n", "=C", "=x", "=rf"] {
            let output = LitStr::new(constraint, proc_macro2::Span::call_site());
            let err = validate_output_constraint(&output).unwrap_err();
            assert!(
                err.to_string().contains("unsupported `out` constraint"),
                "unexpected error for `{constraint}`: {err}"
            );
        }

        for constraint in ["", "r", "=r"] {
            let inout = LitStr::new(constraint, proc_macro2::Span::call_site());
            let err = validate_inout_constraint(&inout).unwrap_err();
            assert!(
                err.to_string().contains("must use read-write syntax"),
                "unexpected error for `{constraint}`: {err}"
            );
        }

        for constraint in ["+n", "+C", "+x", "+rf"] {
            let inout = LitStr::new(constraint, proc_macro2::Span::call_site());
            let err = validate_inout_constraint(&inout).unwrap_err();
            assert!(
                err.to_string().contains("unsupported `inout` constraint"),
                "unexpected error for `{constraint}`: {err}"
            );
        }
    }

    #[test]
    fn builds_tied_constraints_without_exposing_hidden_operands() {
        let outputs = vec![
            PtxAsmOutput {
                constraint: parse_quote!("=r"),
                place: parse_quote!(write_only),
                kind: PtxAsmOutputKind::WriteOnly,
            },
            PtxAsmOutput {
                constraint: parse_quote!("+f"),
                place: parse_quote!(read_write),
                kind: PtxAsmOutputKind::ReadWrite,
            },
        ];
        let inputs = vec![(parse_quote!("r"), parse_quote!(input))];

        assert_eq!(
            build_constraint_string(&outputs, &inputs, &[]).unwrap(),
            "=r,=f,r,1"
        );
    }

    #[test]
    fn accepts_compile_time_string_input_constraint() {
        let input = LitStr::new("C", proc_macro2::Span::call_site());
        assert!(validate_input_constraint(&input).is_ok());

        for constraint in ["=C", "+C"] {
            let invalid = LitStr::new(constraint, proc_macro2::Span::call_site());
            assert!(validate_input_constraint(&invalid).is_err());
        }
    }

    #[test]
    fn accepts_maximum_output_pack() {
        let outputs = (0..64)
            .map(|index| format!("out(\"=r\") value_{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let input: PtxAsmInput = syn::parse_str(&format!("\"nop;\", {outputs}")).unwrap();

        build_ptx_asm(input)
            .expect("64 outputs must build: Blackwell tensor-memory loads need the full pack");
    }

    #[test]
    fn rejects_too_many_outputs() {
        let outputs = (0..65)
            .map(|index| format!("out(\"=r\") value_{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let input: PtxAsmInput = syn::parse_str(&format!("\"nop;\", {outputs}")).unwrap();

        let err = build_ptx_asm(input).unwrap_err();

        assert!(
            err.to_string().contains("at most 64 output operands"),
            "unexpected error: {err}"
        );
    }
}
