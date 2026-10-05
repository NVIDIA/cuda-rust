/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hygiene-critical identifier construction, attribute helpers, and
//! reserved-name checks shared by every macro in this crate.

use proc_macro::TokenStream;
use reserved_oxide_symbols::{
    CODEGEN_FINGERPRINT_ENV, MATERIALIZE_CUBIN_ENV, MATERIALIZER_PROVENANCE_ENV, RESERVED_ROOT,
};
use syn::{FnArg, Ident, ItemFn, Meta, PatType, Type, visit::Visit};

/// Record cuda-oxide's exact device-codegen identity in the consuming crate's
/// dep-info. Cargo then rebuilds only crates that can own or instantiate device
/// code when output mode, architecture, policy, or tool provenance changes.
pub(crate) fn track_codegen_environment() {
    let _ = proc_macro::tracked::env_var(CODEGEN_FINGERPRINT_ENV);
    let _ = proc_macro::tracked::env_var(MATERIALIZE_CUBIN_ENV);
    let _ = proc_macro::tracked::env_var(MATERIALIZER_PROVENANCE_ENV);
}

/// Build a private identifier that cannot capture, or be captured by, a name
/// written in the user's kernel signature.
pub(crate) fn internal_ident(name: &str) -> Ident {
    let span = if proc_macro::is_available() {
        proc_macro::Span::def_site().into()
    } else {
        // Expansion helpers are also exercised as ordinary unit tests, where
        // the compiler's procedural-macro bridge is intentionally unavailable.
        proc_macro2::Span::call_site()
    };
    Ident::new(name, span)
}

pub(crate) fn cuda_module_async_lifetime() -> syn::Lifetime {
    let ident = internal_ident("__cuda_oxide_async");
    syn::Lifetime::new(&format!("'{}", ident), ident.span())
}

struct IdentFinder<'a> {
    name: &'a str,
    found: bool,
}

impl<'ast> Visit<'ast> for IdentFinder<'_> {
    fn visit_ident(&mut self, ident: &'ast Ident) {
        self.found |= ident == self.name;
    }
}

pub(crate) fn call_site_ident_avoiding_item(name: &str, item: &ItemFn) -> Ident {
    let mut candidate = name.to_owned();
    loop {
        let mut finder = IdentFinder {
            name: &candidate,
            found: false,
        };
        finder.visit_item_fn(item);
        if !finder.found {
            break;
        }
        candidate.push('_');
    }
    Ident::new(&candidate, proc_macro2::Span::call_site())
}

/// Reject function names that start with the reserved cuda-oxide prefix
/// (`cuda_oxide_`).
///
/// User code must not define functions in the cuda-oxide internal naming
/// namespace. Two failure modes this guards against:
///
/// 1. **Cosmetic.** `#[kernel] fn cuda_oxide_kernel_foo()` would expand to
///    a doubly-nested name like
///    `fn cuda_oxide_kernel_<hash>_cuda_oxide_kernel_foo()`, producing
///    confusing symbol names in MIR dumps and stack traces.
/// 2. **Forward-compatibility.** Future refactors may extend the namespace;
///    rejecting it at the source level keeps the contract clean.
///
/// Returns `Some(compile_error)` to be returned from the macro entry point,
/// or `None` if the name is safe.
pub(crate) fn reject_reserved_name(name: &Ident) -> Option<TokenStream> {
    let name_str = name.to_string();
    if name_str.starts_with(RESERVED_ROOT) {
        let msg = format!(
            "function name `{name_str}` starts with the reserved cuda-oxide \
             prefix `{RESERVED_ROOT}`; rename your function — this namespace \
             is reserved for cuda-oxide internal symbol mangling \
             (see cuda-oxide/crates/reserved-oxide-symbols)"
        );
        Some(syn::Error::new(name.span(), msg).to_compile_error().into())
    } else {
        None
    }
}

/// Reject argument-position `impl Trait`, which rustc represents as a hidden
/// type parameter that procedural macros cannot name at launch sites.
///
/// Without this check the host emits a non-generic lookup name while the
/// backend correctly emits a `_TID_...` specialization, causing a runtime
/// function-not-found error. A named generic preserves the same source-level
/// intent and gives both sides an explicit specialization identity.
pub(crate) fn impl_trait_parameter_error(input: &ItemFn, item_kind: &str) -> Option<syn::Error> {
    #[derive(Default)]
    struct Finder {
        first: Option<syn::TypeImplTrait>,
    }

    impl<'ast> syn::visit::Visit<'ast> for Finder {
        fn visit_type_impl_trait(&mut self, node: &'ast syn::TypeImplTrait) {
            if self.first.is_none() {
                self.first = Some(node.clone());
            }
        }
    }

    let mut finder = Finder::default();
    for arg in &input.sig.inputs {
        if let FnArg::Typed(pat_type) = arg {
            syn::visit::Visit::visit_type(&mut finder, &pat_type.ty);
        }
    }

    finder.first.map(|impl_trait| {
        syn::Error::new_spanned(
            impl_trait,
            format!(
                "{item_kind} parameters cannot use `impl Trait`; name the type parameter explicitly (for example, `fn named<T: Trait>(value: T)`) so host and device specialization identities agree"
            ),
        )
    })
}

pub(crate) fn has_attr_named(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr_path_ends_with(attr, name))
}

pub(crate) fn attr_path_ends_with(attr: &syn::Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .map(|segment| segment.ident == name)
        .unwrap_or(false)
}

/// Validate a parameter-local `#[grid_constant]` declaration and return the
/// immutable, sized pointee that defines both device and host ABI lowering.
pub(crate) fn grid_constant_pointee(pat_type: &PatType) -> syn::Result<Option<Type>> {
    let declarations: Vec<_> = pat_type
        .attrs
        .iter()
        .filter(|attribute| attr_path_ends_with(attribute, "grid_constant"))
        .collect();
    if declarations.is_empty() {
        return Ok(None);
    }
    if declarations.len() > 1 {
        return Err(syn::Error::new_spanned(
            declarations[1],
            "duplicate #[grid_constant] parameter attribute",
        ));
    }
    if !matches!(declarations[0].meta, Meta::Path(_)) {
        return Err(syn::Error::new_spanned(
            declarations[0],
            "#[grid_constant] takes no arguments",
        ));
    }
    let Type::Reference(reference) = pat_type.ty.as_ref() else {
        return Err(syn::Error::new_spanned(
            &pat_type.ty,
            "#[grid_constant] requires an immutable reference parameter such as &TmaDescriptor",
        ));
    };
    if reference.mutability.is_some() {
        return Err(syn::Error::new_spanned(
            reference,
            "#[grid_constant] parameters are read-only; use &T rather than &mut T",
        ));
    }
    if reference
        .lifetime
        .as_ref()
        .is_some_and(|lifetime| lifetime.ident != "_")
    {
        return Err(syn::Error::new_spanned(
            &reference.lifetime,
            "#[grid_constant] storage lives only for this kernel launch; use &T or &'_ T, not a named or 'static lifetime",
        ));
    }
    if matches!(reference.elem.as_ref(), Type::Slice(_)) {
        return Err(syn::Error::new_spanned(
            &reference.elem,
            "#[grid_constant] requires one sized pointee, not a slice",
        ));
    }
    Ok(Some(reference.elem.as_ref().clone()))
}
