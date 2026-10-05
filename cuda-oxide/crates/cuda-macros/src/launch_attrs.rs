/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launch-configuration attributes: `#[launch_bounds]`,
//! `#[launch_contract]`, `#[cluster_launch]`, `#[cooperative_launch]`,
//! and `#[unroll]` loop-attribute rewriting.

use crate::common::attr_path_ends_with;
use crate::cuda_module::contract::{
    LaunchContractArgs, dynamic_shared_max, validate_requires_relations,
};
use crate::cuda_module::model::{
    CudaModuleParam, CudaModuleParamMarshal, cuda_module_param_from_typed, scalar_int_class,
};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, Pat, Path, Stmt, Token,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    visit::{self, Visit},
    visit_mut::{self, VisitMut},
};

pub(crate) fn launch_bounds_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: LaunchBoundsArgs = parse_macro_input!(attr as LaunchBoundsArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let max_threads = &args.max_threads.expr;
    let min_blocks = &args.min_blocks.expr;

    add_const_evaluatable_bound(&mut input.sig.generics, &args.max_threads);
    add_const_evaluatable_bound(&mut input.sig.generics, &args.min_blocks);

    // Inject the launch bounds config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        ::cuda_device::thread::__launch_bounds_config::<{ #max_threads }, { #min_blocks }>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

pub(crate) fn launch_contract_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as LaunchContractArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    // Validate `requires` here too, so a standalone contract (outside any
    // #[cuda_module]) rejects typos and bad grammar at the attribute site
    // instead of silently dropping the relations. Inside a #[cuda_module]
    // the module macro validates first against the source signature; a
    // module-level rejection replaces the whole module with the error, so
    // this attribute never expands there and the two validations cannot
    // stack duplicate diagnostics.
    if !args.requires.is_empty()
        && let Some(params) = standalone_requires_params(&input)
        && let Err(error) = validate_requires_relations(&args.requires, &params)
    {
        return error.to_compile_error().into();
    }

    inject_launch_contract_markers(&args, &mut input);

    quote! { #input }.into()
}

/// Best-effort parameter model for validating `requires` relations on the
/// function the `launch_contract` attribute macro receives.
///
/// Returns `None` when the function is a macro-generated generic entry
/// wrapper: its parameters carry synthetic `__cuda_oxide_arg_*` names, so
/// source-level relation identifiers cannot be resolved against it. Those
/// relations were already validated against the source signature, either by
/// the `#[cuda_module]` expansion or by `#[kernel]`'s generic routing (see
/// [`validate_routed_launch_contract_requires`]). Parameter shapes the
/// module marshaller rejects, reachable only outside `#[cuda_module]`, are
/// modelled as opaque scalars: referencing one in `requires` then fails with
/// the precise grammar error rather than a misleading "unknown identifier".
pub(crate) fn standalone_requires_params(item_fn: &ItemFn) -> Option<Vec<CudaModuleParam>> {
    requires_params_from_inputs(&item_fn.sig.inputs)
}

/// Shared core of [`standalone_requires_params`], usable with a saved copy
/// of a kernel's original inputs before scope parameters are appended or
/// wrapper signatures are synthesized.
fn requires_params_from_inputs(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> Option<Vec<CudaModuleParam>> {
    let mut params = Vec::new();
    for arg in inputs {
        let FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            continue;
        };
        if pat_ident.ident.to_string().starts_with("__cuda_oxide_arg_") {
            return None;
        }
        let ty = &pat_type.ty;
        let param = cuda_module_param_from_typed(pat_type).unwrap_or_else(|_| CudaModuleParam {
            name: pat_ident.ident.clone(),
            device_ty: (**ty).clone(),
            grid_constant_ty: None,
            sync_host_ty: quote! { #ty },
            async_host_ty: quote! { #ty },
            marshal: CudaModuleParamMarshal::Scalar,
            grid_constant: false,
            mutable_slice: false,
            disjoint_slice_ty: None,
            disjoint_slice_elem: None,
            uniform_ty: None,
            uniform_scalar: None,
            scalar_int: scalar_int_class(ty),
        });
        params.push(param);
    }
    Some(params)
}

/// Validate the `requires` relations of any `#[launch_contract]` attribute
/// that `#[kernel]` is about to route onto a generated generic entry
/// wrapper.
///
/// The wrapper's parameters carry synthetic `__cuda_oxide_arg_*` names, so
/// when the attribute macro finally expands there it cannot resolve
/// source-level identifiers and deliberately stands down (see
/// [`standalone_requires_params`]). Without this check, a typo'd relation on
/// a standalone generic kernel would compile clean while the identical
/// non-generic kernel errors. Malformed argument lists are skipped here: the
/// attribute macro reports those itself when it expands.
pub(crate) fn validate_routed_launch_contract_requires(
    attrs: &[syn::Attribute],
    source_inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> syn::Result<()> {
    for attr in attrs {
        if !attr_path_ends_with(attr, "launch_contract") {
            continue;
        }
        let Ok(args) = attr.parse_args::<LaunchContractArgs>() else {
            continue;
        };
        if args.requires.is_empty() {
            continue;
        }
        if let Some(params) = requires_params_from_inputs(source_inputs) {
            validate_requires_relations(&args.requires, &params)?;
        }
    }
    Ok(())
}

pub(crate) fn inject_launch_contract_markers(args: &LaunchContractArgs, input: &mut ItemFn) {
    let domain = args.domain;
    let u32_coordinates = args.u32_coordinates;
    let contract_marker: syn::Stmt = parse_quote! {
        unsafe {
            ::cuda_device::thread::__launch_contract_config::<#domain, #u32_coordinates>();
        }
    };
    input.block.stmts.insert(0, contract_marker);
    let mut next = 1;

    // Give the exact block shape to the device compiler as well, so ptxas
    // emits `.reqntid` and the driver rejects a mismatched block on every
    // axis. Without this the shape is known only to the host check.
    if let Some((x, y, z)) = args.exact_block {
        let block_marker: syn::Stmt = parse_quote! {
            ::cuda_device::thread::__launch_contract_block_config::<#x, #y, #z>();
        };
        input.block.stmts.insert(next, block_marker);
        next += 1;
    }

    if dynamic_shared_max(args.dynamic_shared) != 0 {
        let alignment = args.dynamic_shared_alignment as usize;
        let alignment_marker: syn::Stmt = parse_quote! {
            ::cuda_device::shared::__dynamic_shared_alignment::<#alignment>();
        };
        input.block.stmts.insert(next, alignment_marker);
    }
}

/// Arguments for `#[launch_bounds(...)]` attribute.
#[derive(Clone)]
pub(crate) struct LaunchBoundsArgs {
    pub(crate) max_threads: ConstU32Expr,
    pub(crate) min_blocks: ConstU32Expr,
}

impl Parse for LaunchBoundsArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(input)?;
        let values = args
            .into_iter()
            .map(ConstU32Expr::new)
            .collect::<syn::Result<Vec<_>>>()?;

        match values.len() {
            1 => {
                let max_threads = values.into_iter().next().unwrap();
                validate_literal_max_threads(&max_threads)?;
                Ok(LaunchBoundsArgs {
                    max_threads,
                    min_blocks: ConstU32Expr::literal(0),
                })
            }
            2 => {
                let mut values = values.into_iter();
                let max_threads = values.next().unwrap();
                let min_blocks = values.next().unwrap();
                validate_literal_max_threads(&max_threads)?;
                Ok(LaunchBoundsArgs {
                    max_threads,
                    min_blocks,
                })
            }
            _ => Err(syn::Error::new(
                input.span(),
                "launch_bounds expects 1 or 2 parameters: #[launch_bounds(max_threads)] or #[launch_bounds(max_threads, min_blocks)]",
            )),
        }
    }
}

/// A source expression whose expected type is `u32` at the generated marker.
///
/// Literal values are retained for early source-time diagnostics. Every other
/// expression stays as typed Rust syntax: rustc, not this procedural macro,
/// resolves associated constants and arithmetic for both device metadata and
/// each monomorphized host launch contract.
#[derive(Clone)]
pub(crate) struct ConstU32Expr {
    pub(crate) expr: Expr,
    pub(crate) literal_value: Option<u32>,
}

impl ConstU32Expr {
    pub(crate) fn new(expr: Expr) -> syn::Result<Self> {
        let literal_value = match &expr {
            Expr::Lit(expr_lit) => match &expr_lit.lit {
                syn::Lit::Int(value) => Some(value.base10_parse::<u32>()?),
                _ => None,
            },
            _ => None,
        };
        Ok(Self {
            expr,
            literal_value,
        })
    }

    pub(crate) fn literal(value: u32) -> Self {
        Self {
            expr: parse_quote! { #value },
            literal_value: Some(value),
        }
    }
}

fn validate_literal_max_threads(value: &ConstU32Expr) -> syn::Result<()> {
    if value.literal_value == Some(0) {
        Err(syn::Error::new_spanned(
            &value.expr,
            "launch_bounds maximum threads must be greater than zero",
        ))
    } else {
        Ok(())
    }
}

/// Returns whether an expression names one of the function's type or const
/// parameters and therefore needs a signature-level evaluatability witness.
///
/// Non-generic paths deliberately return false. In particular, a const item
/// declared inside the function body is in scope at the generated marker but
/// is not in scope in the function signature.
fn const_expr_depends_on_generics(expr: &Expr, generics: &syn::Generics) -> bool {
    let generic_names: Vec<&Ident> = generics
        .params
        .iter()
        .filter_map(|parameter| match parameter {
            GenericParam::Type(parameter) => Some(&parameter.ident),
            GenericParam::Const(parameter) => Some(&parameter.ident),
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    if generic_names.is_empty() {
        return false;
    }

    struct GenericReference<'a> {
        generic_names: &'a [&'a Ident],
        found: bool,
    }

    impl<'ast> Visit<'ast> for GenericReference<'_> {
        fn visit_path(&mut self, path: &'ast Path) {
            if path.segments.iter().any(|segment| {
                self.generic_names
                    .iter()
                    .any(|generic| segment.ident == **generic)
            }) {
                self.found = true;
                return;
            }
            visit::visit_path(self, path);
        }
    }

    let mut visitor = GenericReference {
        generic_names: &generic_names,
        found: false,
    };
    visitor.visit_expr(expr);
    visitor.found
}

/// Make a generic constant expression evaluatable at each monomorphization.
///
/// rustc requires this bound for expressions such as `P::MAX_THREADS`. The
/// marker itself supplies the expected `u32` type; the array length is only an
/// evaluatability witness and never reaches device code.
pub(crate) fn add_const_evaluatable_bound(generics: &mut syn::Generics, value: &ConstU32Expr) {
    if value.literal_value.is_some() || !const_expr_depends_on_generics(&value.expr, generics) {
        return;
    }
    let expr = &value.expr;
    let predicate: syn::WherePredicate = parse_quote! {
        [(); (#expr) as usize]:
    };
    generics.make_where_clause().predicates.push(predicate);
}

pub(crate) fn add_launch_bounds_evaluatability_from_attrs(input: &mut ItemFn) -> syn::Result<()> {
    let matching: Vec<_> = input
        .attrs
        .iter()
        .filter(|attr| attr_path_ends_with(attr, "launch_bounds"))
        .collect();
    if matching.len() > 1 {
        return Err(syn::Error::new_spanned(
            matching[1],
            "a kernel may have only one launch_bounds attribute",
        ));
    }
    let Some(attr) = matching.first() else {
        return Ok(());
    };
    let bounds = attr.parse_args::<LaunchBoundsArgs>()?;
    let max_threads = bounds.max_threads.clone();
    let min_blocks = bounds.min_blocks.clone();
    add_const_evaluatable_bound(&mut input.sig.generics, &max_threads);
    add_const_evaluatable_bound(&mut input.sig.generics, &min_blocks);
    Ok(())
}

/// Arguments for the `#[unroll]` / `#[unroll(N)]` attribute.
///
/// Bare `#[unroll]` parses to factor `0` (full unroll); `#[unroll(N)]` requires
/// `N >= 2`.
pub(crate) struct UnrollArgs {
    pub(crate) factor: ConstU32Expr,
}

impl Parse for UnrollArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Bare `#[unroll]` => full unroll (factor 0).
        if input.is_empty() {
            return Ok(UnrollArgs {
                factor: ConstU32Expr::literal(0),
            });
        }

        let args: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(input)?;
        let mut values = args
            .into_iter()
            .map(ConstU32Expr::new)
            .collect::<syn::Result<Vec<_>>>()?;

        match values.len() {
            1 => {
                let factor = values.pop().unwrap();
                match factor.literal_value {
                    Some(0 | 1) => Err(syn::Error::new_spanned(
                        &factor.expr,
                        "partial unroll factor must be at least 2; use #[unroll] for full unrolling",
                    )),
                    Some(value) if value > 1024 => Err(syn::Error::new_spanned(
                        &factor.expr,
                        "partial unroll factor cannot exceed 1024",
                    )),
                    _ => Ok(UnrollArgs { factor }),
                }
            }
            _ => Err(syn::Error::new(
                input.span(),
                "unroll expects no argument (full unroll) or one factor: #[unroll] or #[unroll(N)]",
            )),
        }
    }
}

/// `VisitMut` pass that consumes `#[unroll]` / `#[unroll(N)]` attributes written
/// directly on loops inside a `#[kernel]` (or `#[device]`) function body.
///
/// This mirrors how CubeCL's `#[cube]` macro handles per-loop `#[unroll]`: the
/// enclosing function macro owns the whole body AST, so it can strip the inner
/// attribute and lower it into a marker BEFORE rustc ever sees it. That keeps us
/// off the nightly `stmt_expr_attributes` feature (rustc otherwise rejects
/// attributes on expressions).
///
/// For every `for` / `while` / `loop` expression carrying an outer `#[unroll]`
/// attribute, the visitor:
///
/// 1. removes the `unroll` attribute from the loop expr's `attrs`, and
/// 2. inserts `cuda_device::thread::__unroll_config::<FACTOR>();` as the FIRST
///    statement of that loop's body block.
///
/// `FACTOR` follows [`UnrollArgs`]: bare `#[unroll]` => `0` (full unroll),
/// `#[unroll(N)]` => `N`. The importer reads the marker from the block it lands
/// in, so the request applies to that loop only.
///
/// The visitor recurses through nested blocks/loops/ifs via the default
/// `visit_mut` traversal, so an annotated loop anywhere in the function is
/// handled. Loops without an `#[unroll]` attribute are left untouched, so a
/// kernel with no per-loop annotations expands byte-identically to before.
#[derive(Default)]
pub(crate) struct LoopUnrollAttrVisitor {
    /// First parse error encountered (e.g. a malformed `#[unroll(...)]`). The
    /// caller surfaces this as a compile error.
    pub(crate) error: Option<syn::Error>,
    pub(crate) const_expressions: Vec<ConstU32Expr>,
}

impl LoopUnrollAttrVisitor {
    /// If `attrs` contains an outer `#[unroll]` / `#[unroll(N)]`, remove it and
    /// return the parsed factor. Returns `None` when no `unroll` attribute is
    /// present (leaving `attrs` untouched). Records a parse error and returns
    /// `None` if the attribute is malformed.
    fn take_unroll_factor(&mut self, attrs: &mut Vec<syn::Attribute>) -> Option<ConstU32Expr> {
        let idx = attrs
            .iter()
            .position(|attr| attr.path().is_ident("unroll"))?;
        let attr = attrs.remove(idx);

        // `#[unroll]` (bare) is `Meta::Path`; `#[unroll(N)]` is `Meta::List`.
        let factor = match &attr.meta {
            syn::Meta::Path(_) => ConstU32Expr::literal(0),
            syn::Meta::List(list) => match list.parse_args::<UnrollArgs>() {
                Ok(parsed) => parsed.factor,
                Err(err) => {
                    if self.error.is_none() {
                        self.error = Some(err);
                    }
                    return None;
                }
            },
            syn::Meta::NameValue(_) => {
                if self.error.is_none() {
                    self.error = Some(syn::Error::new_spanned(
                        &attr,
                        "unroll expects no argument (full unroll) or one factor: #[unroll] or #[unroll(N)]",
                    ));
                }
                return None;
            }
        };
        if factor.literal_value.is_none() {
            self.const_expressions.push(factor.clone());
        }
        Some(factor)
    }

    /// Build the `__unroll_config::<FACTOR>()` marker statement.
    fn marker_stmt(factor: &ConstU32Expr) -> Stmt {
        let expr = &factor.expr;
        parse_quote! {
            cuda_device::thread::__unroll_config::<{ #expr }>();
        }
    }
}

impl VisitMut for LoopUnrollAttrVisitor {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // Pull the unroll factor (if any) off this loop expr and inject the
        // marker into its body. We act only on loop expressions; everything
        // else falls through to the default recursion below.
        match expr {
            Expr::ForLoop(for_loop) => {
                if let Some(factor) = self.take_unroll_factor(&mut for_loop.attrs) {
                    for_loop.body.stmts.insert(0, Self::marker_stmt(&factor));
                }
            }
            Expr::While(while_loop) => {
                if let Some(factor) = self.take_unroll_factor(&mut while_loop.attrs) {
                    while_loop.body.stmts.insert(0, Self::marker_stmt(&factor));
                }
            }
            Expr::Loop(loop_expr) => {
                if let Some(factor) = self.take_unroll_factor(&mut loop_expr.attrs) {
                    loop_expr.body.stmts.insert(0, Self::marker_stmt(&factor));
                }
            }
            _ => {}
        }

        // Recurse into nested expressions/blocks so annotated loops nested
        // inside other loops, `if`s, or blocks are also handled.
        visit_mut::visit_expr_mut(self, expr);
    }
}

/// Consume per-loop unroll annotations in a kernel or device function and
/// replace them with the marker calls understood by the MIR importer.
pub(crate) fn rewrite_loop_unroll_attrs(input: &mut ItemFn) -> syn::Result<()> {
    let mut visitor = LoopUnrollAttrVisitor::default();
    visitor.visit_block_mut(&mut input.block);
    if let Some(err) = visitor.error {
        return Err(err);
    }
    for expression in &visitor.const_expressions {
        add_const_evaluatable_bound(&mut input.sig.generics, expression);
    }
    Ok(())
}

pub(crate) fn cluster_launch_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args: ClusterArgs = parse_macro_input!(attr as ClusterArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let x = args.x;
    let y = args.y;
    let z = args.z;

    // Inject the cluster config marker at the start of the function body
    let marker_call: syn::Stmt = syn::parse_quote! {
        ::cuda_device::cluster::__cluster_config::<#x, #y, #z>();
    };

    // Prepend the marker to the function body
    input.block.stmts.insert(0, marker_call);

    quote! {
        #input
    }
    .into()
}

/// Arguments for `#[cluster_launch(...)]` attribute.
pub(crate) struct ClusterArgs {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

impl Parse for ClusterArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        let values: Vec<u32> = args
            .iter()
            .map(|lit| lit.base10_parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;

        match values.len() {
            1 => Ok(ClusterArgs {
                x: values[0],
                y: 1,
                z: 1,
            }),
            2 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: 1,
            }),
            3 => Ok(ClusterArgs {
                x: values[0],
                y: values[1],
                z: values[2],
            }),
            _ => Err(syn::Error::new_spanned(
                args.first().unwrap(),
                "cluster expects 1, 2, or 3 dimensions: #[cluster(x)], #[cluster(x, y)], or #[cluster(x, y, z)]",
            )),
        }
    }
}

pub(crate) fn cooperative_launch_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "cooperative_launch takes no arguments: use a bare #[cooperative_launch]",
        )
        .to_compile_error()
        .into();
    }

    // Launch-time only: the marker is consumed by #[cuda_module]; the kernel
    // body and PTX are unchanged. Parse as a function so misuse on other
    // items is rejected with a clear error.
    let input = parse_macro_input!(item as ItemFn);
    quote! {
        #input
    }
    .into()
}
