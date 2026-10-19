/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[kernel]` argument parsing, validation, and expansion dispatch.

pub(crate) mod codegen;
pub(crate) mod scope;

use crate::common::{
    attr_path_ends_with, grid_constant_pointee, impl_trait_parameter_error, reject_reserved_name,
    track_codegen_environment,
};
use crate::cuda_module::launchers::has_codegen_generics;
use crate::kernel::codegen::{
    generate_generic_kernel, generate_generic_kernel_no_instantiation, generate_simple_kernel,
};
use crate::launch_attrs::{add_launch_bounds_evaluatability_from_attrs, rewrite_loop_unroll_attrs};
use proc_macro::TokenStream;
use syn::{
    FnArg, GenericParam, Ident, ItemFn, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

/// Attribute arguments for `#[kernel(...)]`.
///
/// Legacy explicit-instantiation types, the optional launch-context binding,
/// and the bare `unchecked_indexing` flag may appear in any order:
///
/// ```ignore
/// #[kernel(launch_context = launch_context)]
/// #[kernel(f32, f64, launch_context = launch_context)]
/// #[kernel(unchecked_indexing)]
/// #[kernel(f32, unchecked_indexing)]
/// #[kernel(launch_context = launch_context, unchecked_indexing)]
/// ```
pub(crate) struct KernelArgs {
    /// Types to instantiate generic kernels for
    pub(crate) instantiate_types: Vec<Type>,
    /// User-selected name for the entry's typed launch context.
    pub(crate) launch_context: Option<Ident>,
    /// Elide slice/array bounds checks in this kernel's body (UB contract).
    pub(crate) unchecked_indexing: bool,
}

/// Returns true when the next attribute argument is exactly the bare flag
/// word `flag` (an identifier followed by `,` or end of input, never `=`).
///
/// A bare identifier also parses as a `Type`, so this peek must run before
/// the legacy instantiation-type fallback to keep flag words from being
/// swallowed as type arguments.
fn peek_bare_kernel_flag(input: ParseStream, flag: &str) -> bool {
    if !input.peek(Ident) || input.peek2(Token![=]) {
        return false;
    }
    let fork = input.fork();
    match fork.parse::<Ident>() {
        Ok(ident) => ident == flag && (fork.is_empty() || fork.peek(Token![,])),
        Err(_) => false,
    }
}

impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut instantiate_types = Vec::new();
        let mut launch_context = None;
        let mut unchecked_indexing = false;

        while !input.is_empty() {
            if input.peek(Ident) && input.peek2(Token![=]) {
                let name: Ident = input.parse()?;
                input.parse::<Token![=]>()?;
                if name != "launch_context" {
                    return Err(syn::Error::new(
                        name.span(),
                        format!(
                            "unknown #[kernel] named argument `{name}`; expected `launch_context = IDENT`"
                        ),
                    ));
                }
                let value: Ident = input.parse().map_err(|_| {
                    syn::Error::new(
                        input.span(),
                        "`launch_context` must be a single Rust identifier",
                    )
                })?;
                if launch_context.replace(value).is_some() {
                    return Err(syn::Error::new(
                        name.span(),
                        "duplicate `launch_context` argument in #[kernel]",
                    ));
                }
            } else if peek_bare_kernel_flag(input, "unchecked_indexing") {
                let flag: Ident = input.parse()?;
                if unchecked_indexing {
                    return Err(syn::Error::new(
                        flag.span(),
                        "duplicate `unchecked_indexing` argument in #[kernel]",
                    ));
                }
                unchecked_indexing = true;
            } else {
                instantiate_types.push(input.parse::<Type>()?);
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        Ok(KernelArgs {
            instantiate_types,
            launch_context,
            unchecked_indexing,
        })
    }
}

fn scope_parameter_collision(input: &ItemFn, scope: &Ident) -> Option<Ident> {
    struct Finder<'a> {
        scope: &'a Ident,
        found: Option<Ident>,
    }

    impl<'ast> syn::visit::Visit<'ast> for Finder<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            if self.found.is_none() && pattern.ident == *self.scope {
                self.found = Some(pattern.ident.clone());
            }
            syn::visit::visit_pat_ident(self, pattern);
        }
    }

    let mut finder = Finder { scope, found: None };
    for argument in &input.sig.inputs {
        if let FnArg::Typed(argument) = argument {
            syn::visit::Visit::visit_pat(&mut finder, &argument.pat);
        }
    }
    finder.found
}

/// Consume parameter-local `#[grid_constant]` declarations and plant one
/// source-index marker per parameter for the MIR importer.
///
/// An immutable reference is the source-level representation because LLVM's
/// grid-constant contract is a pointer parameter with `byval(T)`: device code
/// must keep the parameter-space address rather than receive a copied `T`.
pub(crate) fn inject_grid_constant_markers(input: &mut ItemFn) -> syn::Result<()> {
    let mut markers = Vec::new();
    for (index, argument) in input.sig.inputs.iter_mut().enumerate() {
        let FnArg::Typed(argument) = argument else {
            continue;
        };
        if grid_constant_pointee(argument)?.is_none() {
            continue;
        }
        argument
            .attrs
            .retain(|attribute| !attr_path_ends_with(attribute, "grid_constant"));
        markers.push(parse_quote! {
            // SAFETY: this marker is emitted only on the kernel entry whose
            // parameter declaration also determines host by-value marshalling.
            unsafe { ::cuda_device::thread::__grid_constant_config::<#index>(); }
        });
    }
    input.block.stmts.splice(0..0, markers);
    Ok(())
}

pub(crate) fn kernel_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    track_codegen_environment();
    let args = parse_macro_input!(attr as KernelArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let KernelArgs {
        instantiate_types,
        launch_context,
        unchecked_indexing,
    } = args;

    if let Some(err) = reject_reserved_name(&input.sig.ident) {
        return err;
    }
    if let Some(err) = impl_trait_parameter_error(&input, "kernel") {
        return err.to_compile_error().into();
    }
    if let Err(error) = inject_grid_constant_markers(&mut input) {
        return error.to_compile_error().into();
    }
    if let Some(launch_context) = &launch_context
        && let Some(parameter) = scope_parameter_collision(&input, launch_context)
    {
        return syn::Error::new_spanned(
            parameter,
            format!(
                "kernel launch-context binding `{launch_context}` conflicts with a function parameter; choose a distinct name in `#[kernel(launch_context = ...)]`"
            ),
        )
        .to_compile_error()
        .into();
    }

    // Consume any `#[unroll]` / `#[unroll(N)]` attributes written directly on
    // loops inside the kernel body. We strip the attribute from the loop
    // expression (so rustc never sees an expression attribute, keeping us off
    // nightly `stmt_expr_attributes`) and inject an
    // `__unroll_config::<FACTOR>()` marker into the loop body. If the visitor
    // hits a malformed attribute it records the error so we can surface it as a
    // compile error below.
    if let Err(err) = rewrite_loop_unroll_attrs(&mut input) {
        return err.to_compile_error().into();
    }
    if let Err(err) = add_launch_bounds_evaluatability_from_attrs(&mut input) {
        return err.to_compile_error().into();
    }

    // Insert the unchecked-indexing compiler marker at the top of the kernel
    // body. The MIR importer detects the call, elides bounds-check asserts in
    // the translated body, and strips the call before code generation. For
    // simple kernels the function itself becomes the entry, so the marker
    // stays put. Generic kernel generation strips the marker from the
    // re-emitted user-named implementation helper and confines it to the
    // generated entry wrapper (forwarded via
    // `top_level_kernel_configuration_markers`) plus a hidden unchecked twin
    // of the implementation that only the entry calls; see
    // `strip_unchecked_indexing_config_marker` and
    // `unchecked_indexing_impl_clone`.
    if unchecked_indexing {
        let marker_call: syn::Stmt = syn::parse_quote! {
            ::cuda_device::thread::__unchecked_indexing_config::<true>();
        };
        input.block.stmts.insert(0, marker_call);
    }

    // Only type and const parameters create distinct codegen instances.
    // Lifetimes are erased before monomorphization.
    let has_generics = has_codegen_generics(&input.sig.generics);

    if has_generics && !instantiate_types.is_empty() {
        let type_param_count = input
            .sig
            .generics
            .params
            .iter()
            .filter(|param| matches!(param, GenericParam::Type(_)))
            .count();
        let has_const_params = input
            .sig
            .generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Const(_)));
        let has_lifetime_params = input
            .sig
            .generics
            .params
            .iter()
            .any(|param| matches!(param, GenericParam::Lifetime(_)));
        if type_param_count != 1 || has_const_params || has_lifetime_params {
            return syn::Error::new_spanned(
                &input.sig.generics,
                "legacy #[kernel(Type, ...)] instantiation supports exactly one type parameter and no lifetime or const parameters; use #[kernel] and instantiate the kernel with a normal turbofish at the launch site",
            )
            .to_compile_error()
            .into();
        }
    }

    if has_generics && instantiate_types.is_empty() {
        // Generic kernel without explicit types - allow it!
        // Instantiation will happen from call sites (nvcc-style)
        return generate_generic_kernel_no_instantiation(input, launch_context);
    }

    if !has_generics && !instantiate_types.is_empty() {
        // Non-generic kernel with instantiation types - error
        return syn::Error::new_spanned(
            &input.sig.ident,
            "Instantiation types only apply to generic kernels",
        )
        .to_compile_error()
        .into();
    }

    if has_generics {
        // Generate wrapper kernels for each instantiation type
        generate_generic_kernel(input, instantiate_types, launch_context)
    } else {
        // Simple non-generic kernel
        generate_simple_kernel(input, launch_context)
    }
}
