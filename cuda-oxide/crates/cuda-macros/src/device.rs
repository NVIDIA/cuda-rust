/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[device]` functions and extern blocks, plus `#[constant]` statics.

use crate::common::{
    impl_trait_parameter_error, reject_reserved_name, reserved_name_error,
    track_codegen_environment,
};
use crate::cuda_module::constants::extract_constant_inner_ty;
use crate::cuda_module::launchers::{codegen_generic_arguments, has_codegen_generics};
use crate::kernel::scope::{forwarding_inputs, inject_device_thread_index_scope};
use crate::launch_attrs::rewrite_loop_unroll_attrs;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use reserved_oxide_symbols::{
    DEVICE_EXPORT_PREFIX, DEVICE_EXTERN_PREFIX, DEVICE_PREFIX, constant_symbol,
};
use syn::{
    ForeignItem, Ident, ItemFn, ItemForeignMod, LitStr, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

pub(crate) struct ConstantArgs {
    pub(crate) export_name: Option<LitStr>,
}

impl Parse for ConstantArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { export_name: None });
        }

        let key: Ident = input.parse()?;
        if key != "export_name" {
            return Err(syn::Error::new_spanned(
                key,
                "expected `export_name = \"...\"`",
            ));
        }
        input.parse::<Token![=]>()?;
        let export_name: LitStr = input.parse()?;
        let value = export_name.value();
        let valid_ptx_ident = !value.is_empty()
            && value.chars().enumerate().all(|(idx, ch)| {
                ch.is_ascii_alphabetic()
                    || matches!(ch, '_' | '$')
                    || (idx > 0 && ch.is_ascii_digit())
            });
        if !valid_ptx_ident {
            return Err(syn::Error::new_spanned(
                &export_name,
                "`export_name` must be a non-empty PTX identifier (`[A-Za-z$_][A-Za-z$_0-9]*`)",
            ));
        }
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
            if !input.is_empty() {
                return Err(input.error("unexpected tokens after #[constant] export_name"));
            }
        }
        Ok(Self {
            export_name: Some(export_name),
        })
    }
}

pub(crate) fn constant_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    let args = parse_macro_input!(attr as ConstantArgs);
    let input = parse_macro_input!(item as syn::ItemStatic);

    if let Some(err) = reject_reserved_name(&input.ident) {
        return err;
    }

    if extract_constant_inner_ty(&input.ty).is_none() {
        return syn::Error::new_spanned(
            &input.ty,
            "#[constant] static must have type `ConstantMemory<T>` \
             (e.g. `static FOO: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;`). \
             The wrapper prevents the compiler from constant-folding the \
             initializer into kernel bodies.",
        )
        .to_compile_error()
        .into();
    }

    let export_name = args.export_name.unwrap_or_else(|| {
        LitStr::new(
            &constant_symbol(&input.ident.to_string()),
            input.ident.span(),
        )
    });

    quote! {
        #[unsafe(export_name = #export_name)]
        #input
    }
    .into()
}

/// Optional export directive on `#[device]`:
/// - `#[device]` -> helper collected transitively when kernels exist, or rooted
///   only in kernel-less standalone libraries.
/// - `#[device(export)]` -> always rooted as `.visible .func` under its Rust
///   declaration name, even alongside `#[kernel]` entry points.
/// - `#[device(export_name = "...")]` -> always rooted as `.visible .func`
///   under an explicit PTX symbol name while keeping the Rust function name for
///   in-crate callers.
pub(crate) enum DeviceExport {
    DeclarationName,
    CustomName(Ident),
}

pub(crate) struct DeviceArgs {
    pub(crate) export: Option<DeviceExport>,
}

impl Parse for DeviceArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { export: None });
        }

        let key: Ident = input.parse()?;
        let export = if key == "export" {
            if input.peek(Token![=]) {
                return Err(syn::Error::new_spanned(
                    key,
                    "expected `#[device(export)]` or `#[device(export_name = \"...\")]`",
                ));
            }
            DeviceExport::DeclarationName
        } else if key == "export_name" {
            input.parse::<Token![=]>()?;
            let lit: LitStr = input.parse()?;
            let value = lit.value();
            let valid = !value.is_empty()
                && value.chars().enumerate().all(|(idx, ch)| {
                    ch.is_ascii_alphabetic() || ch == '_' || (idx > 0 && ch.is_ascii_digit())
                });
            if !valid {
                return Err(syn::Error::new_spanned(
                    &lit,
                    "`export_name` must be a non-empty identifier (`[A-Za-z_][A-Za-z0-9_]*`)",
                ));
            }
            if value.starts_with("llvm_") {
                return Err(syn::Error::new_spanned(
                    &lit,
                    "`export_name` cannot start with `llvm_`, which cuda-oxide reserves for LLVM intrinsics",
                ));
            }
            let ident = Ident::new(&value, lit.span());
            if let Some(err) = reserved_name_error(&ident) {
                return Err(err);
            }
            DeviceExport::CustomName(ident)
        } else {
            return Err(syn::Error::new_spanned(
                key,
                "unsupported #[device] argument; expected `export` or `export_name = \"...\"`",
            ));
        };

        if !input.is_empty() {
            input.parse::<Token![,]>()?;
            if !input.is_empty() {
                return Err(input.error("unexpected tokens in #[device(...)]"));
            }
        }

        Ok(Self {
            export: Some(export),
        })
    }
}

pub(crate) fn device_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    let args = parse_macro_input!(attr as DeviceArgs);
    // Try parsing as a function definition first
    if let Ok(input) = syn::parse::<ItemFn>(item.clone()) {
        return match expand_device_function(args, input) {
            Ok(tokens) => tokens.into(),
            Err(err) => err.to_compile_error().into(),
        };
    }

    // Try parsing as an extern block
    if let Ok(input) = syn::parse::<ItemForeignMod>(item.clone()) {
        if args.export.is_some() {
            return syn::Error::new_spanned(
                &input.abi,
                "#[device] extern blocks cannot use `export` or `export_name`",
            )
            .to_compile_error()
            .into();
        }
        return generate_device_extern_block(input);
    }

    // Neither worked - emit error
    syn::Error::new_spanned(
        TokenStream2::from(item),
        "#[device] can only be applied to functions or extern blocks",
    )
    .to_compile_error()
    .into()
}

/// Generate a device function definition.
///
/// Renames the function into the reserved `cuda_oxide_device_<hash>_` (or
/// `cuda_oxide_device_export_<hash>_`) namespace for collector detection, and
/// generates a thin wrapper with the original name so user code can call
/// `my_func()` rather than the mangled internal symbol.
///
/// Handles both non-generic and generic device functions:
/// - **Non-generic**: `#[no_mangle]` on the prefixed function, `#[inline(always)]` wrapper.
/// - **Generic**: No `#[no_mangle]` (generics use mangled symbols), `#[inline(never)]` on
///   the prefixed function (so monomorphizations appear in CGUs for the collector),
///   `#[inline(always)]` wrapper with generics + turbofish forwarding.
///
/// This mirrors the pattern used by `#[kernel]` for generic kernels
/// (see `generate_generic_kernel_no_instantiation`).
pub(crate) fn expand_device_function(
    args: DeviceArgs,
    mut input: ItemFn,
) -> syn::Result<TokenStream2> {
    if let Some(err) = reserved_name_error(&input.sig.ident) {
        return Err(err);
    }
    if let Some(err) = impl_trait_parameter_error(&input, "device function") {
        return Err(err);
    }
    rewrite_loop_unroll_attrs(&mut input)?;
    inject_device_thread_index_scope(&mut input);

    let fn_name = input.sig.ident.clone();
    let vis = input.vis.clone();

    // Type and const parameters both create device monomorphizations.
    let has_generics = has_codegen_generics(&input.sig.generics);
    if args.export.is_some() && has_generics {
        return Err(syn::Error::new_spanned(
            &input.sig.generics,
            "#[device(export)] functions cannot have generic parameters; export a concrete wrapper instead",
        ));
    }

    let new_name = match &args.export {
        None => format_ident!("{}{}", DEVICE_PREFIX, fn_name),
        Some(DeviceExport::DeclarationName) => format_ident!("{}{}", DEVICE_EXPORT_PREFIX, fn_name),
        Some(DeviceExport::CustomName(export_ident)) => {
            format_ident!("{}{}", DEVICE_EXPORT_PREFIX, export_ident)
        }
    };

    let return_type = &input.sig.output;
    let generics = &input.sig.generics;
    let where_clause = &input.sig.generics.where_clause;
    let constness = &input.sig.constness;
    let unsafety = &input.sig.unsafety;
    let abi = &input.sig.abi;

    let (wrapper_inputs, params) = forwarding_inputs(&input.sig.inputs)?;

    // Rename the original function with the prefix
    input.sig.ident = new_name.clone();

    if has_generics {
        // Generic device function: mirrors the generic kernel pattern.
        //
        // - No #[no_mangle] - generic functions use mangled symbol names per
        //   monomorphization (e.g., `cuda_oxide_device_<hash>_add::<f32>` gets a
        //   unique mangled name). #[no_mangle] requires a single concrete symbol.
        //
        // - #[inline(never)] on the prefixed function - ensures each monomorphization
        //   appears as a distinct CGU item so the collector can find it. If it were
        //   inlined, the function would disappear from the CGU. It is a host-side
        //   boundary only: the backend does not turn it into device `noinline`
        //   (`device_codegen::is_generic_device_collector_boundary`).
        //
        // - The wrapper forwards type parameters via turbofish:
        //   `cuda_oxide_device_<hash>_add::<T>(a, b)`.

        let codegen_args = codegen_generic_arguments(generics);
        let turbofish = quote! { ::<#(#codegen_args),*> };
        let call = quote! { #new_name #turbofish (#(#params),*) };
        let call = if unsafety.is_some() {
            quote! { unsafe { #call } }
        } else {
            call
        };

        Ok(quote! {
            #[inline(never)]
            #input

            /// Wrapper for the generic device function with the original name.
            #[inline(always)]
            #vis #constness #unsafety #abi fn #fn_name #generics (#(#wrapper_inputs),*) #return_type #where_clause {
                #call
            }
        })
    } else {
        let call = quote! { #new_name(#(#params),*) };
        let call = if unsafety.is_some() {
            quote! { unsafe { #call } }
        } else {
            call
        };
        // Non-generic device function: simple case.
        Ok(quote! {
            #[allow(non_snake_case)]
            #[unsafe(no_mangle)]
            #input

            /// Wrapper for the device function with the original name.
            #[inline(always)]
            #vis #constness #unsafety #abi fn #fn_name #generics (#(#wrapper_inputs),*) #return_type #where_clause {
                #call
            }
        })
    }
}

/// Generate device extern block declarations (for FFI with external LTOIR).
///
/// For each function in the extern block:
/// 1. Rename it into the reserved `cuda_oxide_device_extern_<hash>_` namespace
///    (for collector detection)
/// 2. Generate a wrapper function with the original name (for user code)
///
/// User code calls `foo()` while the collector sees the hash-suffixed reserved
/// form. The `#[link_name]` attribute restores the original name in the binary
/// so external LTOIR resolves correctly.
fn generate_device_extern_block(mut input: ItemForeignMod) -> TokenStream {
    // Device extern declarations use CUDA's C calling convention. Reject
    // other Rust ABIs because their argument and return conventions may differ.
    if input
        .abi
        .name
        .as_ref()
        .is_some_and(|name| name.value() != "C")
    {
        return syn::Error::new_spanned(
            &input.abi,
            "#[device] extern blocks must use `extern \"C\"`",
        )
        .to_compile_error()
        .into();
    }

    let mut wrappers = Vec::new();

    // Process each item in the extern block
    for item in &mut input.items {
        if let ForeignItem::Fn(foreign_fn) = item {
            if let Some(err) = reject_reserved_name(&foreign_fn.sig.ident) {
                return err;
            }

            // `...` is purely syntactic, so it can be refused here rather than
            // in device codegen. That gives the user a span on the offending
            // tokens instead of a signature-less error at final-binary codegen,
            // and it also catches a variadic extern that is declared but never
            // called -- the collector only registers externs reached from
            // kernel MIR, so that one used to compile clean.
            //
            // The codegen rejection stays: the collector recognises device
            // externs by their reserved name prefix, so a hand-written extern
            // block never passes through this macro at all.
            if let Some(variadic) = &foreign_fn.sig.variadic {
                return syn::Error::new_spanned(variadic, "#[device] externs cannot be variadic")
                    .to_compile_error()
                    .into();
            }

            // Save original info for wrapper generation
            let original_name = foreign_fn.sig.ident.clone();
            let original_attrs = foreign_fn.attrs.clone();
            let original_sig = foreign_fn.sig.clone();

            let new_name = format_ident!("{}{}", DEVICE_EXTERN_PREFIX, original_name);
            foreign_fn.sig.ident = new_name.clone();

            // Store original name as link_name for the linker
            let original_name_str = original_name.to_string();
            foreign_fn.attrs.push(syn::parse_quote! {
                #[doc(hidden)]
            });
            foreign_fn.attrs.push(syn::parse_quote! {
                #[link_name = #original_name_str]
            });

            // Generate wrapper function with the original name. User code
            // calls `foo()`; the wrapper forwards to the reserved internal
            // symbol the macro just produced.
            let params: Vec<_> = original_sig
                .inputs
                .iter()
                .filter_map(|arg| {
                    if let syn::FnArg::Typed(pat_type) = arg
                        && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
                    {
                        return Some(pat_ident.ident.clone());
                    }
                    None
                })
                .collect();

            let return_type = &original_sig.output;
            let inputs = &original_sig.inputs;

            // Keep user's attributes (like #[convergent]) on the wrapper
            let wrapper = quote! {
                #(#original_attrs)*
                #[inline(always)]
                #[allow(non_snake_case)]
                pub unsafe fn #original_name(#inputs) #return_type {
                    #new_name(#(#params),*)
                }
            };
            wrappers.push(wrapper);
        }
    }

    let expanded = quote! {
        #input

        #(#wrappers)*
    };

    TokenStream::from(expanded)
}
