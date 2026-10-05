/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-index call rewriting, kernel launch-context scopes, and
//! configuration-marker handling.

use crate::common::{attr_path_ends_with, call_site_ident_avoiding_item, internal_ident};
use crate::cuda_module::contract::LaunchContractArgs;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use reserved_oxide_symbols::KERNEL_SCOPE_LOCAL;
use syn::{
    Expr, ExprCall, ExprMethodCall, ExprPath, FnArg, GenericArgument, Ident, ItemFn, Path,
    PathArguments, Stmt, Type, parse_quote, parse_quote_spanned,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
};

/// Find the generic type parameter that has a Fn/FnMut/FnOnce bound (the closure type).
/// Returns the type parameter name if found.
pub(super) fn find_closure_generic(generics: &syn::Generics) -> Option<syn::Ident> {
    for param in &generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            for bound in &type_param.bounds {
                if is_fn_trait_bound(bound) {
                    return Some(type_param.ident.clone());
                }
            }
        }
    }

    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            let syn::WherePredicate::Type(predicate_type) = predicate else {
                continue;
            };
            if !predicate_type.bounds.iter().any(is_fn_trait_bound) {
                continue;
            }
            let Type::Path(type_path) = &predicate_type.bounded_ty else {
                continue;
            };
            if type_path.qself.is_none()
                && type_path.path.segments.len() == 1
                && let Some(segment) = type_path.path.segments.first()
            {
                return Some(segment.ident.clone());
            }
        }
    }

    None
}

fn is_fn_trait_bound(bound: &syn::TypeParamBound) -> bool {
    let syn::TypeParamBound::Trait(trait_bound) = bound else {
        return false;
    };
    trait_bound.path.segments.last().is_some_and(|segment| {
        matches!(
            segment.ident.to_string().as_str(),
            "Fn" | "FnMut" | "FnOnce"
        )
    })
}

/// Find which function parameter uses the closure type.
/// Returns the index and info of the closure parameter.
pub(super) fn find_closure_param<'a>(
    args_info: &'a [(&'a Ident, &'a Type)],
    closure_type_name: &syn::Ident,
) -> Option<(usize, &'a (&'a Ident, &'a Type))> {
    for (idx, (_name, ty)) in args_info.iter().enumerate() {
        // Check if the type is a simple path matching our closure generic
        if let Type::Path(type_path) = *ty
            && type_path.qself.is_none()
            && let Some(segment) = type_path.path.segments.first()
            && type_path.path.segments.len() == 1
            && segment.ident == *closure_type_name
        {
            return Some((idx, &args_info[idx]));
        }
    }
    None
}

/// Build wrapper parameters that can always be forwarded by value.
///
/// User functions may use any irrefutable parameter pattern, including `_` or
/// tuple destructuring. A generated wrapper cannot refer to those patterns
/// again, so every parameter gets a private synthetic identifier while keeping
/// its original type and attributes. The original implementation retains the
/// user's pattern.
pub(crate) fn forwarding_inputs(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> syn::Result<(Vec<FnArg>, Vec<Ident>)> {
    let mut wrapper_inputs = Vec::with_capacity(inputs.len());
    let mut forwarding_names = Vec::with_capacity(inputs.len());

    for (index, arg) in inputs.iter().enumerate() {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "CUDA kernels and device functions cannot take self parameters",
            ));
        };
        let name = internal_ident(&format!("__cuda_oxide_arg_{index}"));
        let mut wrapper_param = pat_type.clone();
        wrapper_param.pat = Box::new(parse_quote! { #name });
        wrapper_inputs.push(FnArg::Typed(wrapper_param));
        forwarding_names.push(name);
    }

    Ok((wrapper_inputs, forwarding_names))
}

/// True when `path`'s *last* segment is `name`.
///
/// We deliberately match on the tail only, so all of these resolve to the
/// same intrinsic:
///
/// ```ignore
/// index_1d()
/// thread::index_1d()
/// cuda_device::thread::index_1d()
/// ::cuda_device::thread::index_1d()
/// ```
///
/// And imports/aliases work too:
///
/// ```ignore
/// use cuda_device::thread::index_1d;          // bare ident → matches
/// use cuda_device::thread::index_1d as foo;   // aliased    → won't match (path tail is `foo`)
/// ```
///
/// The aliased form is intentionally not rewritten — if the user picked a
/// new name, they get the bare-stub-panic behaviour, not silent capture.
///
/// Caveat: if the user defines a *local* `fn index_1d` (or any other
/// reserved name) and calls it from inside `#[kernel]` / `#[device]`,
/// that call gets rewritten too. See the `Reserved names` section in
/// `ThreadIndex`'s doc-block — the convention is to pick a different
/// name (e.g. `compute_index_1d`) for any helper you want to keep.
fn is_thread_index_path(path: &Path, name: &str) -> bool {
    path.segments.last().is_some_and(|seg| seg.ident == name)
}

/// Build the rewritten path that points the user's call at the
/// `__internal::<name>` shim, preserving whatever prefix the user wrote.
///
/// The motivation is unused-import hygiene. If the user wrote
/// `use cuda_device::thread;` and called `thread::index_1d()`, replacing
/// the whole call with an absolute path makes rustc see the `thread`
/// import as unused. Instead, we splice `__internal` in front of the
/// last segment and keep everything before it intact:
///
/// ```text
/// thread::index_1d()                   →  thread::__internal::index_1d(&scope)
/// cuda_device::thread::index_1d()      →  cuda_device::thread::__internal::index_1d(&scope)
/// ::cuda_device::thread::index_1d()    →  ::cuda_device::thread::__internal::index_1d(&scope)
/// ```
///
/// Bare-ident calls are the only shape that can't carry a prefix, so for
/// those we fall back to the absolute path (the user wasn't naming
/// anything to import; see the bare-ident case in `is_thread_index_path`'s
/// doc-comment for why we still rewrite those):
///
/// ```text
/// index_1d()                           →  ::cuda_device::thread::__internal::index_1d(&scope)
/// ```
fn internal_thread_path(
    user_path: &Path,
    name: &str,
    arguments: syn::PathArguments,
    call_span: proc_macro2::Span,
) -> Path {
    let ident = Ident::new(name, call_span);
    let internal = Ident::new("__internal", call_span);

    if user_path.segments.len() == 1 {
        let mut absolute: Path =
            parse_quote_spanned! { call_span => ::cuda_device::thread::#internal::#ident };
        if let Some(last) = absolute.segments.last_mut() {
            last.arguments = arguments;
        }
        return absolute;
    }

    let leading_colon = user_path.leading_colon;
    let prefix_segments: Vec<&syn::PathSegment> = user_path
        .segments
        .iter()
        .take(user_path.segments.len() - 1)
        .collect();
    let mut rewritten: Path = parse_quote_spanned! {
        call_span => #leading_colon #(#prefix_segments)::* :: #internal :: #ident
    };
    if let Some(last) = rewritten.segments.last_mut() {
        last.arguments = arguments;
    }
    rewritten
}

/// One scoped intrinsic the rewriter knows about.
///
/// Adding a new `thread::*` function that needs the `'kernel` scope is a
/// one-line entry here, plus the matching public stub and `__internal::*`
/// impl in `cuda-device`.
struct ScopedIntrinsic {
    /// The unqualified function name (last segment of the path we match).
    name: &'static str,
    /// If true, copy the call-site's turbofish onto the rewritten path
    /// (e.g. `index_2d::<S>` → `__internal::index_2d::<S>`).
    preserve_turbofish: bool,
    /// If true, forward the original call arguments after the scope ref
    /// (e.g. `index_2d_runtime(s)` → `__internal::index_2d_runtime(&scope, s)`).
    forward_args: bool,
}

const SCOPED_INTRINSICS: &[ScopedIntrinsic] = &[
    ScopedIntrinsic {
        name: "index_1d",
        preserve_turbofish: false,
        forward_args: false,
    },
    ScopedIntrinsic {
        name: "index_2d",
        preserve_turbofish: true,
        forward_args: false,
    },
    ScopedIntrinsic {
        name: "index_2d_runtime",
        preserve_turbofish: false,
        forward_args: true,
    },
    ScopedIntrinsic {
        name: "warp_index",
        preserve_turbofish: false,
        forward_args: false,
    },
];

/// Method names whose zero-arg call sites get the kernel scope spliced in
/// as a leading `&scope` argument.
///
/// These are matched on the *method name only* (not the receiver type, which
/// the macro can't see anyway). The scope is only injected when the user
/// wrote a zero-arg call like `slice.get_mut_indexed()`; if they passed
/// arguments themselves, we leave the call alone and let typeck decide.
///
/// Same caveat as `SCOPED_INTRINSICS`: a local method on an unrelated type
/// with the same name and a zero-arg form will get the scope appended,
/// which will cause a typeck error ("expected 0 arguments, got 1"). Pick
/// a different name (e.g. `pop_indexed`) for any helper you want to keep.
const SCOPED_METHODS: &[&str] = &["get_mut_indexed"];

fn is_scoped_method(method: &Ident) -> bool {
    SCOPED_METHODS.iter().any(|name| method == name)
}

struct ThreadIndexCallRewriter {
    scope_ident: Ident,
    borrow_scope: bool,
    rewrote_index_call: bool,
}

fn ident_located_at(ident: &Ident, location: proc_macro2::Span) -> Ident {
    let mut relocated = ident.clone();
    relocated.set_span(ident.span().located_at(location));
    relocated
}

impl VisitMut for ThreadIndexCallRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        match expr {
            Expr::Call(call) => {
                let call_span = call.span();
                let ExprCall { func, args, .. } = call;
                let Expr::Path(ExprPath { path, .. }) = &mut **func else {
                    return;
                };
                let Some(intrinsic) = SCOPED_INTRINSICS
                    .iter()
                    .find(|i| is_thread_index_path(path, i.name))
                else {
                    return;
                };

                let last_args = path
                    .segments
                    .last()
                    .map(|seg| seg.arguments.clone())
                    .unwrap_or(syn::PathArguments::None);
                let path_args = if intrinsic.preserve_turbofish {
                    last_args
                } else {
                    syn::PathArguments::None
                };
                *path = internal_thread_path(path, intrinsic.name, path_args, call_span);
                // Keep the binding's name-resolution hygiene, but make this
                // generated reference part of the user's indexing expression
                // for diagnostics and line-table purposes.
                let scope_ident = ident_located_at(&self.scope_ident, call_span);
                let scope_arg: Expr = if self.borrow_scope {
                    parse_quote_spanned! { call_span => &#scope_ident }
                } else {
                    parse_quote_spanned! { call_span => #scope_ident }
                };

                if intrinsic.forward_args {
                    args.insert(0, scope_arg);
                } else {
                    args.clear();
                    args.push(scope_arg);
                }
                self.rewrote_index_call = true;
            }
            Expr::MethodCall(method_call) => {
                let call_span = method_call.span();
                let ExprMethodCall { method, args, .. } = method_call;
                if !is_scoped_method(method) || !args.is_empty() {
                    return;
                }
                let scope_ident = ident_located_at(&self.scope_ident, call_span);
                if self.borrow_scope {
                    args.push(parse_quote_spanned! { call_span => &#scope_ident });
                } else {
                    args.push(parse_quote_spanned! { call_span => #scope_ident });
                }
                self.rewrote_index_call = true;
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
pub(crate) struct RewrittenKernelScope {
    pub(crate) ident: Ident,
    pub(crate) domain: TokenStream2,
    pub(crate) coordinates: TokenStream2,
}

pub(super) fn rewrite_thread_index_calls(
    input: &mut ItemFn,
    borrow_scope: bool,
) -> Option<RewrittenKernelScope> {
    // Keep the ordinary call-site span so borrow-checker diagnostics continue
    // to point at the user's `#[kernel]` / `#[device]` item. Only rename the
    // binding when a generic parameter has deliberately taken the usual name.
    let scope_ident = call_site_ident_avoiding_item(KERNEL_SCOPE_LOCAL, input);
    if !rewrite_thread_index_calls_with_scope(input, &scope_ident, borrow_scope) {
        return None;
    }

    Some(kernel_scope_spec(input, scope_ident))
}

fn rewrite_thread_index_calls_with_scope(
    input: &mut ItemFn,
    scope_ident: &Ident,
    borrow_scope: bool,
) -> bool {
    let mut rewriter = ThreadIndexCallRewriter {
        scope_ident: scope_ident.clone(),
        borrow_scope,
        rewrote_index_call: false,
    };
    rewriter.visit_block_mut(&mut input.block);
    rewriter.rewrote_index_call
}

fn kernel_scope_spec(input: &ItemFn, ident: Ident) -> RewrittenKernelScope {
    let (domain, coordinates) = kernel_scope_marker_types(input);
    RewrittenKernelScope {
        ident,
        domain,
        coordinates,
    }
}

pub(crate) fn explicit_kernel_scope(input: &mut ItemFn, ident: Ident) -> RewrittenKernelScope {
    // Legacy helpers still receive the same capability, but only their
    // established names are rewritten. The new fast functions are ordinary
    // Rust calls that take this reference explicitly.
    rewrite_thread_index_calls_with_scope(input, &ident, false);
    kernel_scope_spec(input, ident)
}

pub(super) fn kernel_scope_binding(scope: &RewrittenKernelScope) -> Stmt {
    let RewrittenKernelScope {
        ident,
        domain,
        coordinates,
    } = scope;
    parse_quote! {
        let #ident = unsafe {
            ::cuda_device::thread::__internal::make_kernel_scope::<#domain, #coordinates>()
        };
    }
}

pub(crate) fn explicit_kernel_scope_bindings(scope: &RewrittenKernelScope) -> Vec<Stmt> {
    let RewrittenKernelScope {
        ident,
        domain,
        coordinates,
    } = scope;
    // Def-site-like hygiene keeps this generated storage binding distinct
    // from any source binding with the same text. Both its declaration and
    // reference are emitted by this one proc-macro expansion.
    let storage = Ident::new(
        &format!("{KERNEL_SCOPE_LOCAL}_storage"),
        proc_macro2::Span::mixed_site(),
    );
    vec![
        parse_quote! {
            let #storage = unsafe {
                ::cuda_device::thread::__internal::make_kernel_scope::<#domain, #coordinates>()
            };
        },
        parse_quote! {
            let #ident: ::cuda_device::thread::LaunchContextRef<'_, #domain, #coordinates> =
                &#storage;
        },
    ]
}

pub(super) fn append_kernel_scope_parameter(input: &mut ItemFn, scope: &RewrittenKernelScope) {
    let RewrittenKernelScope {
        ident,
        domain,
        coordinates,
    } = scope;
    input.sig.inputs.push(parse_quote! {
        #ident: ::cuda_device::thread::LaunchContextRef<'_, #domain, #coordinates>
    });
}

pub(crate) fn inject_thread_index_scope(input: &mut ItemFn) {
    if let Some(scope) = rewrite_thread_index_calls(input, true) {
        input.block.stmts.insert(0, kernel_scope_binding(&scope));
    }
}

pub(crate) fn inject_device_thread_index_scope(input: &mut ItemFn) {
    let Some(mut scope) = rewrite_thread_index_calls(input, true) else {
        return;
    };
    // A device helper has no host preparation boundary of its own. It may use
    // checked legacy witnesses, but it cannot mint a contract-backed fast
    // witness; callers must pass that witness or a checked view explicitly.
    scope.domain = quote! { ::cuda_device::thread::__internal::UnknownDomain };
    scope.coordinates = quote! { ::cuda_device::thread::__internal::NativeCoordinates };
    input.block.stmts.insert(0, kernel_scope_binding(&scope));
}

fn kernel_scope_marker_types(input: &ItemFn) -> (TokenStream2, TokenStream2) {
    let contract = input
        .attrs
        .iter()
        .find(|attr| attr_path_ends_with(attr, "launch_contract"))
        .and_then(|attr| attr.parse_args::<LaunchContractArgs>().ok())
        .map(|args| (args.domain, args.u32_coordinates))
        .or_else(|| {
            input
                .block
                .stmts
                .iter()
                .find_map(launch_contract_marker_values)
        });

    let (domain, u32_coordinates) = contract.unwrap_or((0, false));
    let domain = match domain {
        1 => quote! { ::cuda_device::thread::__internal::Domain1 },
        2 => quote! { ::cuda_device::thread::__internal::Domain2 },
        3 => quote! { ::cuda_device::thread::__internal::Domain3 },
        _ => quote! { ::cuda_device::thread::__internal::UnknownDomain },
    };
    let coordinates = if u32_coordinates {
        quote! { ::cuda_device::thread::__internal::U32Coordinates }
    } else {
        quote! { ::cuda_device::thread::__internal::NativeCoordinates }
    };
    (domain, coordinates)
}

fn launch_contract_marker_values(statement: &Stmt) -> Option<(u8, bool)> {
    let (call, unsafe_wrapped) = configuration_marker_call(statement)?;
    if !unsafe_wrapped {
        return None;
    }
    if !call.args.is_empty() {
        return None;
    }
    let Expr::Path(ExprPath {
        qself: None, path, ..
    }) = &*call.func
    else {
        return None;
    };
    let segments: Vec<_> = path.segments.iter().collect();
    if path.leading_colon.is_none()
        || segments.len() != 3
        || segments[0].ident != "cuda_device"
        || segments[1].ident != "thread"
        || segments[2].ident != "__launch_contract_config"
    {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &segments[2].arguments else {
        return None;
    };
    let mut arguments = arguments.args.iter();
    let GenericArgument::Const(Expr::Lit(domain)) = arguments.next()? else {
        return None;
    };
    let syn::Lit::Int(domain) = &domain.lit else {
        return None;
    };
    let GenericArgument::Const(Expr::Lit(coordinates)) = arguments.next()? else {
        return None;
    };
    let syn::Lit::Bool(coordinates) = &coordinates.lit else {
        return None;
    };
    if arguments.next().is_some() {
        return None;
    }
    Some((domain.base10_parse().ok()?, coordinates.value))
}

fn configuration_marker_call(statement: &Stmt) -> Option<(&ExprCall, bool)> {
    match statement {
        Stmt::Expr(Expr::Call(call), Some(_semicolon)) => Some((call, false)),
        Stmt::Expr(Expr::Unsafe(unsafe_block), _) if unsafe_block.block.stmts.len() == 1 => {
            let Stmt::Expr(Expr::Call(call), Some(_semicolon)) = &unsafe_block.block.stmts[0]
            else {
                return None;
            };
            Some((call, true))
        }
        _ => None,
    }
}

/// Return compiler configuration markers already materialized in a generic
/// kernel body by attributes that expanded before `#[kernel]`.
///
/// Only exact, zero-argument calls to cuda-oxide's internal marker paths are
/// forwarded. The launch-contract marker additionally retains its generated
/// `unsafe` boundary. Nested calls and unrelated top-level calls remain solely
/// in the helper body.
pub(crate) fn top_level_kernel_configuration_markers(input: &ItemFn) -> Vec<Stmt> {
    input
        .block
        .stmts
        .iter()
        .filter(|statement| is_kernel_configuration_marker(statement))
        .cloned()
        .collect()
}

pub(crate) fn is_kernel_configuration_marker(statement: &Stmt) -> bool {
    let Some((call, unsafe_wrapped)) = configuration_marker_call(statement) else {
        return false;
    };
    if !call.args.is_empty() {
        return false;
    }
    let Expr::Path(ExprPath {
        qself: None, path, ..
    }) = &*call.func
    else {
        return false;
    };
    let segments: Vec<_> = path.segments.iter().collect();
    if path.leading_colon.is_none()
        || segments.len() != 3
        || segments[0].ident != "cuda_device"
        || !matches!(segments[0].arguments, PathArguments::None)
        || !matches!(segments[1].arguments, PathArguments::None)
        || !matches!(segments[2].arguments, PathArguments::AngleBracketed(_))
    {
        return false;
    }

    let module = &segments[1].ident;
    let marker = &segments[2].ident;
    if unsafe_wrapped {
        module == "thread"
            && (marker == "__launch_contract_config" || marker == "__grid_constant_config")
    } else {
        (module == "thread"
            && (marker == "__launch_bounds_config"
                || marker == "__launch_contract_block_config"
                || marker == "__grid_constant_config"
                || marker == "__unchecked_indexing_config"))
            || (module == "cluster" && marker == "__cluster_config")
            || (module == "shared" && marker == "__dynamic_shared_alignment")
    }
}

fn is_named_kernel_configuration_marker(statement: &Stmt, expected: &str) -> bool {
    if !is_kernel_configuration_marker(statement) {
        return false;
    }
    let Some((call, _)) = configuration_marker_call(statement) else {
        return false;
    };
    let Expr::Path(ExprPath { path, .. }) = &*call.func else {
        return false;
    };
    path.segments
        .last()
        .is_some_and(|segment| segment.ident == expected)
}

/// True when `statement` is exactly the `__unchecked_indexing_config` marker
/// call that `#[kernel(unchecked_indexing)]` injects.
pub(crate) fn is_unchecked_indexing_config_marker(statement: &Stmt) -> bool {
    is_named_kernel_configuration_marker(statement, "__unchecked_indexing_config")
}

/// True when `statement` is one parameter-local `#[grid_constant]` marker.
pub(crate) fn is_grid_constant_config_marker(statement: &Stmt) -> bool {
    is_named_kernel_configuration_marker(statement, "__grid_constant_config")
}

/// Remove the `__unchecked_indexing_config` marker from a function body that
/// is about to be re-emitted as the user-named implementation helper of a
/// generic kernel.
///
/// The marker may only live in generated kernel ENTRY functions (the
/// `cuda_oxide_*kernel*`-prefixed symbols); the entry wrapper receives it via
/// `top_level_kernel_configuration_markers`. The helper is ordinary callable
/// Rust: if the marker stayed in its body, rustc's MIR inliner could splice
/// it into a different, non-opted kernel that calls the helper, and the MIR
/// importer would then elide that caller's bounds checks body-wide without
/// any opt-in. Stripping is fail-closed: if the helper is ever not
/// MIR-inlined into its own entry wrapper, the helper body simply keeps its
/// bounds checks.
pub(super) fn strip_unchecked_indexing_config_marker(input: &mut ItemFn) {
    input
        .block
        .stmts
        .retain(|statement| !is_unchecked_indexing_config_marker(statement));
}

/// Remove grid-constant ABI markers from a generic kernel's callable helper.
///
/// The generated entry wrapper receives the markers through
/// [`top_level_kernel_configuration_markers`]. Keeping them in the helper
/// would both duplicate the marker when rustc inlines that helper back into
/// its entry and leak the by-value entry ABI into unrelated kernels that call
/// the same helper.
pub(super) fn strip_grid_constant_config_markers(input: &mut ItemFn) {
    input
        .block
        .stmts
        .retain(|statement| !is_grid_constant_config_marker(statement));
}

/// Build the hidden unchecked twin of an opted-in generic kernel's
/// implementation, returning the identifier the generated entry wrapper must
/// call plus the emitted item.
///
/// `input` must already have the marker stripped (it is the exact body that
/// becomes the user-named helper); the clone re-inserts the marker as its
/// first statement.
///
/// Why a clone at all: the device pipeline translates the
/// `#[inline(always)]` implementation as its own device function (LLVM
/// inlines it into the entry later), so a marker on the entry wrapper alone
/// cannot elide the implementation body's bounds checks, and a marker left
/// in the user-named helper leaks elision into every other kernel that calls
/// the helper. The clone carries the marker instead. It is private,
/// `#[doc(hidden)]`, and named with a def-site (hygienic) identifier, so the
/// generated `cuda_oxide_*kernel*` entry wrapper of this same expansion is
/// its only possible caller: bounds-check elision stays confined to launches
/// of the opted kernel, and the user-named helper keeps every bounds check.
pub(super) fn unchecked_indexing_impl_clone(
    input: &ItemFn,
    helper_attrs: &[syn::Attribute],
) -> (Ident, TokenStream2) {
    // `Display` of a raw identifier keeps its `r#` prefix (e.g. `r#gen`),
    // which `Ident::new` rejects; strip it like `format_ident!` would.
    let implementation_name = input.sig.ident.to_string();
    let implementation_name = implementation_name
        .strip_prefix("r#")
        .unwrap_or(&implementation_name);
    let clone_name = internal_ident(&format!(
        "__cuda_oxide_unchecked_impl_{implementation_name}"
    ));
    let mut clone_fn = input.clone();
    clone_fn.sig.ident = clone_name.clone();
    clone_fn.vis = syn::Visibility::Inherited;
    // The twin's body is byte-identical to the user-named helper's, so it
    // carries the same routed attributes (cfg gates, lint levels such as
    // `#[allow]`/`#[expect]`, deprecation): a lint the author suppressed on
    // the helper must not re-fire on the twin. Entry directives and
    // `#[inline]` were already routed away by `route_generic_kernel_attrs`.
    clone_fn.attrs = helper_attrs.to_vec();
    let marker_call: Stmt = parse_quote! {
        ::cuda_device::thread::__unchecked_indexing_config::<true>();
    };
    clone_fn.block.stmts.insert(0, marker_call);
    let tokens = quote! {
        /// Hidden unchecked twin of the kernel implementation. Only the
        /// generated kernel entry calls it; its marker elides bounds checks
        /// solely for launches of the opted kernel.
        #[doc(hidden)]
        #[inline(always)]
        #[allow(non_snake_case)]
        #clone_fn
    };
    (clone_name, tokens)
}
