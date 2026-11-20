/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parsing, parameter validation, and compiler-marker injection for the
//! `#[verify(...)]` formal-verification attribute.

use crate::common::attr_path_ends_with;
use crate::cuda_module::model::{CudaModuleParam, CudaModuleParamMarshal};
use crate::launch_attrs::{requires_params_from_inputs, standalone_requires_params};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Expr, FnArg, Ident, Item, ItemFn, Stmt, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
};

/// Extract `(lhs_param_index, rhs_param_index)` pairs for `requires` relations
/// of the form `lhs.len() >= rhs.len()` or `rhs.len() <= lhs.len()`.
pub(crate) fn extract_requires_len_ge_pairs(
    relations: &[Expr],
    params: &[CudaModuleParam],
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for relation in relations {
        let Expr::Binary(binary) = relation else {
            continue;
        };
        let (Some(left_idx), Some(right_idx)) = (
            slice_len_param_index(&binary.left, params),
            slice_len_param_index(&binary.right, params),
        ) else {
            continue;
        };
        match binary.op {
            syn::BinOp::Ge(_) => pairs.push((left_idx, right_idx)),
            syn::BinOp::Le(_) => pairs.push((right_idx, left_idx)),
            _ => {}
        }
    }
    pairs
}

fn slice_len_param_index(expr: &Expr, params: &[CudaModuleParam]) -> Option<usize> {
    let Expr::MethodCall(call) = expr else {
        return None;
    };
    if call.method != "len" || !call.args.is_empty() {
        return None;
    }
    let Expr::Path(path) = &*call.receiver else {
        return None;
    };
    let ident = path.path.get_ident()?;
    params.iter().position(|param| param.name == *ident)
}

/// Parsed clauses of `#[verify(...)]`.
#[derive(Clone, Default)]
pub(crate) struct VerifyArgs {
    /// Targets of `no_oob_idx(...)` (slice parameters or `SharedArray` statics).
    pub(crate) no_oob_idx: Vec<Ident>,
    /// Targets of `disjoint_writes(...)` (writable slice parameters).
    pub(crate) disjoint_writes: Vec<Ident>,
    /// Targets of `writes_all_elems(...)` (writable slice parameters).
    pub(crate) writes_all_elems: Vec<Ident>,
    /// Targets of `no_data_race(...)` / `no_data_races(...)` (`SharedArray` statics or writable slices).
    pub(crate) no_data_race: Vec<Ident>,
    /// Span of `uniform_barriers` when requested.
    pub(crate) uniform_barriers: Option<Ident>,
    /// Span of `tmem_lifecycle` when Blackwell `TMEM` lifecycle/bounds verification is requested.
    pub(crate) tmem_lifecycle: Option<Ident>,
}

impl Parse for VerifyArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "verify requires at least one obligation: `no_oob_idx(...)`, `disjoint_writes(...)`, `writes_all_elems(...)`, `no_data_race(...)`, `uniform_barriers`, or `tmem_lifecycle`",
            ));
        }
        let mut args = VerifyArgs::default();
        while !input.is_empty() {
            let clause: Ident = input.parse()?;
            if clause == "uniform_barriers" || clause == "tmem_lifecycle" {
                if input.peek(syn::token::Paren) {
                    let content;
                    syn::parenthesized!(content in input);
                    if !content.is_empty() {
                        return Err(syn::Error::new(
                            clause.span(),
                            format!("`{clause}` takes no arguments"),
                        ));
                    }
                }
                if clause == "uniform_barriers" {
                    args.uniform_barriers = Some(clause);
                } else {
                    args.tmem_lifecycle = Some(clause);
                }
                if !input.is_empty() {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }
            let content;
            syn::parenthesized!(content in input);
            let params: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
            if params.is_empty() {
                return Err(syn::Error::new(
                    clause.span(),
                    format!("`{clause}(...)` requires at least one buffer or shared-array name"),
                ));
            }
            match clause.to_string().as_str() {
                "no_oob_idx" => args.no_oob_idx.extend(params),
                "disjoint_writes" => args.disjoint_writes.extend(params),
                "writes_all_elems" => args.writes_all_elems.extend(params),
                "no_data_race" | "no_data_races" => args.no_data_race.extend(params),
                _ => {
                    return Err(syn::Error::new(
                        clause.span(),
                        format!(
                            "unknown #[verify] obligation `{clause}`; expected `no_oob_idx`, `disjoint_writes`, `writes_all_elems`, `no_data_race`, `uniform_barriers`, or `tmem_lifecycle`"
                        ),
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(args)
    }
}

/// Entry point for the `#[verify(...)]` attribute macro.
pub(crate) fn verify_entry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as VerifyArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    if let Some(params) = standalone_requires_params(&input)
        && let Err(error) = resolve_and_inject_verify_markers(&args, &params, &mut input)
    {
        return error.to_compile_error().into();
    }

    quote! { #input }.into()
}

/// Validate and build marker statements for any `#[verify(...)]` attributes
/// routed across generic kernel wrappers, and lift verified `SharedArray`
/// accesses so kernel authors do not need manual `unsafe` blocks.
pub(crate) fn validate_routed_verify_attrs(
    attrs: &[syn::Attribute],
    source_inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    body_stmts: &mut [Stmt],
) -> syn::Result<Vec<Stmt>> {
    let mut markers = Vec::new();
    let Some(params) = requires_params_from_inputs(source_inputs) else {
        return Ok(markers);
    };
    let shared_statics = collect_shared_array_statics(body_stmts);
    for attr in attrs {
        if !attr_path_ends_with(attr, "verify") {
            continue;
        }
        let args = attr.parse_args::<VerifyArgs>()?;
        markers.extend(build_verify_marker_stmts(&args, &params, &shared_statics)?);
        rewrite_verified_shared_accesses(&args, &shared_statics, body_stmts);
    }
    Ok(markers)
}

fn resolve_and_inject_verify_markers(
    args: &VerifyArgs,
    params: &[CudaModuleParam],
    input: &mut ItemFn,
) -> syn::Result<()> {
    let shared_statics = collect_shared_array_statics(&input.block.stmts);
    let stmts = build_verify_marker_stmts(args, params, &shared_statics)?;
    rewrite_verified_shared_accesses(args, &shared_statics, &mut input.block.stmts);
    for (idx, stmt) in stmts.into_iter().enumerate() {
        input.block.stmts.insert(idx, stmt);
    }
    Ok(())
}

/// AST visitor that wraps index reads/writes and verified async TMA / `mbarrier` /
/// `tcgen05` calls on Lean-verified `SharedArray`, `Barrier`, and `TMEM` resources
/// in `unsafe` blocks so the kernel body can be written in safe Rust.
struct VerifiedSharedArrayRewriter {
    /// In-body `SharedArray` statics listed in both `no_oob_idx` and `no_data_race`.
    verified_shared: Vec<Ident>,
    /// In-body `Barrier` (`mbarrier`) statics declared inside the kernel function.
    barrier_statics: Vec<Ident>,
    /// True when `tmem_lifecycle` is present on `#[verify(...)]`.
    tmem_lifecycle: bool,
    /// True while traversing inside an existing `unsafe { ... }` block.
    in_unsafe: bool,
}

/// Peel casts, raw-address-of, references, index projections, parens, and
/// `.as_ptr()` / `.as_mut_ptr()` to check whether `expr` directly targets a
/// static identifier in `allowed`.
fn expr_targets_static(expr: &Expr, allowed: &[Ident]) -> bool {
    match expr {
        Expr::Path(path_expr) => path_expr
            .path
            .get_ident()
            .is_some_and(|id| allowed.iter().any(|v| v == id)),
        Expr::Index(idx) => expr_targets_static(&idx.expr, allowed),
        Expr::Cast(cast) => expr_targets_static(&cast.expr, allowed),
        Expr::RawAddr(raw) => expr_targets_static(&raw.expr, allowed),
        Expr::Reference(refer) => expr_targets_static(&refer.expr, allowed),
        Expr::Paren(paren) => expr_targets_static(&paren.expr, allowed),
        Expr::Group(group) => expr_targets_static(&group.expr, allowed),
        Expr::MethodCall(mc)
            if (mc.args.is_empty()
                && (mc.method == "as_ptr"
                    || mc.method == "as_mut_ptr"
                    || mc.method == "cast"
                    || mc.method == "expose_provenance"))
                || (mc.args.len() == 1 && mc.method == "add") =>
        {
            expr_targets_static(&mc.receiver, allowed)
        }
        Expr::Binary(bin) if matches!(bin.op, syn::BinOp::BitAnd(_)) => {
            expr_targets_static(&bin.left, allowed) || expr_targets_static(&bin.right, allowed)
        }
        Expr::Call(call) => {
            if let Expr::Path(p) = &*call.func
                && let Some(last) = p.path.segments.last()
                && matches!(
                    last.ident.to_string().as_str(),
                    "map_shared_rank"
                        | "map_shared_rank_mut"
                        | "cvta_generic_to_shared_offset"
                        | "with_exposed_provenance_mut"
                        | "with_exposed_provenance"
                )
            {
                call.args
                    .first()
                    .is_some_and(|a| expr_targets_static(a, allowed))
            } else {
                false
            }
        }
        _ => false,
    }
}

impl VerifiedSharedArrayRewriter {
    /// Return true when `expr` indexes a `SharedArray` static in `self.verified_shared`.
    fn is_verified_shared_index(&self, expr: &Expr) -> bool {
        let Expr::Index(index_expr) = expr else {
            return false;
        };
        let Expr::Path(path_expr) = &*index_expr.expr else {
            return false;
        };
        path_expr
            .path
            .get_ident()
            .is_some_and(|id| self.verified_shared.iter().any(|v| v == id))
    }

    /// Return true when `expr` is a verified `mbarrier` op on `self.barrier_statics`,
    /// a `g2s` TMA bulk copy targeting `self.verified_shared` + `self.barrier_statics`,
    /// or (under `tmem_lifecycle`) a scoped `tcgen05` / `cvta_generic_to_shared_offset` call.
    fn is_verified_async_barrier_call(&self, expr: &Expr) -> bool {
        let Expr::Call(call_expr) = expr else {
            return false;
        };
        let Expr::Path(path_expr) = &*call_expr.func else {
            return false;
        };
        let Some(last) = path_expr.path.segments.last() else {
            return false;
        };
        let name = last.ident.to_string();
        match name.as_str() {
            "fence_proxy_async_shared_cta" => {
                call_expr.args.is_empty()
                    && (!self.barrier_statics.is_empty() || self.tmem_lifecycle)
            }
            "mbarrier_init"
            | "mbarrier_inval"
            | "mbarrier_arrive"
            | "mbarrier_arrive_expect_tx"
            | "mbarrier_arrive_expect_tx_cluster"
            | "mbarrier_arrive_cluster"
            | "mbarrier_arrive_cluster_relaxed"
            | "mbarrier_try_wait"
            | "mbarrier_try_wait_parity"
            | "mbarrier_try_wait_parity_cluster"
            | "mbarrier_try_wait_parity_cluster_relaxed"
            | "mbarrier_test_wait" => call_expr
                .args
                .first()
                .is_some_and(|arg| expr_targets_static(arg, &self.barrier_statics)),
            "map_shared_rank" | "map_shared_rank_mut" => call_expr.args.first().is_some_and(|arg| {
                expr_targets_static(arg, &self.verified_shared)
                    || expr_targets_static(arg, &self.barrier_statics)
            }),
            "clc_try_cancel" | "clc_try_cancel_multicast" => {
                call_expr
                    .args
                    .first()
                    .is_some_and(|dst| expr_targets_static(dst, &self.verified_shared))
                    && call_expr
                        .args
                        .get(1)
                        .is_some_and(|bar| expr_targets_static(bar, &self.barrier_statics))
            }
            "clc_query_is_canceled"
            | "clc_query_get_first_ctaid_x"
            | "clc_query_get_first_ctaid_y"
            | "clc_query_get_first_ctaid_z"
            | "cvt_bf16x2_f32" => true,
            _ if name.starts_with("cp_async_bulk_tensor_") && name.contains("_g2s") => {
                call_expr
                    .args
                    .first()
                    .is_some_and(|dst| expr_targets_static(dst, &self.verified_shared))
                    && call_expr
                        .args
                        .iter()
                        .any(|bar| expr_targets_static(bar, &self.barrier_statics))
            }
            "cvta_generic_to_shared_offset" => call_expr.args.first().is_some_and(|ptr| {
                (self.tmem_lifecycle && expr_targets_static(ptr, &self.verified_shared))
                    || expr_targets_static(ptr, &self.barrier_statics)
            }),
            "tcgen05_alloc" | "tcgen05_alloc_cg2" if self.tmem_lifecycle => call_expr
                .args
                .first()
                .is_some_and(|dst| expr_targets_static(dst, &self.verified_shared)),
            "tcgen05_commit"
            | "tcgen05_commit_cg2"
            | "tcgen05_commit_shared_cluster"
            | "tcgen05_commit_shared_cluster_cg2"
            | "tcgen05_commit_multicast"
            | "tcgen05_commit_multicast_cg2"
                if self.tmem_lifecycle =>
            {
                call_expr
                    .args
                    .first()
                    .is_some_and(|bar| expr_targets_static(bar, &self.barrier_statics))
            }
            "tcgen05_dealloc"
            | "tcgen05_dealloc_cg2"
            | "tcgen05_relinquish_alloc_permit"
            | "tcgen05_relinquish_alloc_permit_cg2"
            | "tcgen05_load_wait"
            | "tcgen05_store_wait"
            | "tcgen05_fence_before_thread_sync"
            | "tcgen05_fence_after_thread_sync"
                if self.tmem_lifecycle =>
            {
                true
            }
            _ if self.tmem_lifecycle
                && (name.starts_with("tcgen05_mma")
                    || name.starts_with("tcgen05_cp")
                    || name.starts_with("tcgen05_ld_")) =>
            {
                true
            }
            _ => false,
        }
    }
}

impl VisitMut for VerifiedSharedArrayRewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Unsafe(expr_unsafe) = expr {
            let prev = self.in_unsafe;
            self.in_unsafe = true;
            visit_mut::visit_expr_unsafe_mut(self, expr_unsafe);
            self.in_unsafe = prev;
            return;
        }

        if !self.in_unsafe {
            if let Expr::Assign(assign) = expr
                && self.is_verified_shared_index(&assign.left)
            {
                self.in_unsafe = true;
                visit_mut::visit_expr_mut(self, &mut assign.right);
                if let Expr::Index(idx) = &mut *assign.left {
                    visit_mut::visit_expr_mut(self, &mut idx.index);
                }
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if let Expr::RawAddr(raw) = expr
                && self.is_verified_shared_index(&raw.expr)
            {
                self.in_unsafe = true;
                if let Expr::Index(idx) = &mut *raw.expr {
                    visit_mut::visit_expr_mut(self, &mut idx.index);
                }
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if self.is_verified_shared_index(expr) {
                self.in_unsafe = true;
                if let Expr::Index(idx) = expr {
                    visit_mut::visit_expr_mut(self, &mut idx.index);
                }
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if let Expr::Unary(unary) = expr
                && matches!(unary.op, syn::UnOp::Deref(_))
                && expr_targets_static(&unary.expr, &self.verified_shared)
            {
                self.in_unsafe = true;
                visit_mut::visit_expr_mut(self, &mut unary.expr);
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if let Expr::MethodCall(mc) = expr
                && mc.method == "add"
                && mc.args.len() == 1
                && (expr_targets_static(&mc.receiver, &self.verified_shared)
                    || expr_targets_static(&mc.receiver, &self.barrier_statics))
            {
                self.in_unsafe = true;
                visit_mut::visit_expr_mut(self, &mut mc.receiver);
                for arg in &mut mc.args {
                    visit_mut::visit_expr_mut(self, arg);
                }
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
            if self.is_verified_async_barrier_call(expr) {
                self.in_unsafe = true;
                if let Expr::Call(call_expr) = expr {
                    for arg in &mut call_expr.args {
                        visit_mut::visit_expr_mut(self, arg);
                    }
                }
                self.in_unsafe = false;
                let inner = expr.clone();
                *expr = parse_quote! { unsafe { #inner } };
                return;
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }

    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        if let syn::Pat::Ident(pat_ident) = &local.pat
            && let Some(init) = &local.init
        {
            let id = pat_ident.ident.clone();
            if expr_targets_static(&init.expr, &self.verified_shared)
                && !self.verified_shared.contains(&id)
            {
                self.verified_shared.push(id.clone());
            }
            if expr_targets_static(&init.expr, &self.barrier_statics)
                && !self.barrier_statics.contains(&id)
            {
                self.barrier_statics.push(id);
            }
        }
        visit_mut::visit_local_mut(self, local);
    }
}

/// Wrap Lean-verified `SharedArray` index operations and scoped async `mbarrier` /
/// `g2s` TMA calls in `unsafe` blocks when `uniform_barriers` is also present.
fn rewrite_verified_shared_accesses(
    args: &VerifyArgs,
    shared_statics: &[Ident],
    stmts: &mut [Stmt],
) {
    if args.uniform_barriers.is_none() {
        return;
    }
    let barrier_array_statics = collect_barrier_array_statics(stmts);
    let verified_shared: Vec<Ident> = shared_statics
        .iter()
        .filter(|name| {
            let has_no_oob = args.no_oob_idx.iter().any(|n| n == *name);
            let has_no_race = args.no_data_race.iter().any(|n| n == *name);
            let is_bar_arr = barrier_array_statics.iter().any(|n| n == *name);
            has_no_oob && (has_no_race || is_bar_arr)
        })
        .cloned()
        .collect();
    if verified_shared.is_empty() {
        return;
    }
    let mut barrier_statics = collect_barrier_statics(stmts);
    for bar_arr in barrier_array_statics {
        if args.no_oob_idx.iter().any(|n| n == &bar_arr) && !barrier_statics.contains(&bar_arr) {
            barrier_statics.push(bar_arr);
        }
    }
    let mut rewriter = VerifiedSharedArrayRewriter {
        verified_shared,
        barrier_statics,
        tmem_lifecycle: args.tmem_lifecycle.is_some(),
        in_unsafe: false,
    };
    for stmt in stmts {
        rewriter.visit_stmt_mut(stmt);
    }
}

/// Collect in-body `static mut NAME: Barrier` declarations for verified `mbarrier` scoping.
fn collect_barrier_statics(stmts: &[Stmt]) -> Vec<Ident> {
    let mut names = Vec::new();
    for stmt in stmts {
        if let Stmt::Item(Item::Static(item_static)) = stmt
            && let Type::Path(type_path) = &*item_static.ty
            && type_path
                .path
                .segments
                .last()
                .is_some_and(|seg| seg.ident == "Barrier")
        {
            names.push(item_static.ident.clone());
        }
    }
    names.sort_by_key(|a| a.to_string());
    names
}

/// Collect in-body `static mut NAME: SharedArray<Barrier, ...>` declarations so
/// barrier arrays listed in `no_oob_idx(...)` receive scoped `mbarrier` lifting.
fn collect_barrier_array_statics(stmts: &[Stmt]) -> Vec<Ident> {
    let mut names = Vec::new();
    for stmt in stmts {
        if let Stmt::Item(Item::Static(item_static)) = stmt
            && is_barrier_shared_array_type(&item_static.ty)
        {
            names.push(item_static.ident.clone());
        }
    }
    names.sort_by_key(|a| a.to_string());
    names
}

/// Return true when `ty` is `SharedArray<Barrier, ...>`.
fn is_barrier_shared_array_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(seg) = type_path.path.segments.last() else {
        return false;
    };
    if seg.ident != "SharedArray" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(angle) = &seg.arguments else {
        return false;
    };
    angle.args.iter().find_map(|arg| {
        if let syn::GenericArgument::Type(Type::Path(inner)) = arg {
            Some(
                inner
                    .path
                    .segments
                    .last()
                    .is_some_and(|s| s.ident == "Barrier"),
            )
        } else {
            None
        }
    }) == Some(true)
}

/// Collect in-body `static mut NAME: SharedArray<T, N>` declarations sorted by
/// identifier name so `shared_id` (`0..N`) matches `lean-exporter`.
fn collect_shared_array_statics(stmts: &[Stmt]) -> Vec<Ident> {
    let mut names = Vec::new();
    for stmt in stmts {
        if let Stmt::Item(Item::Static(item_static)) = stmt
            && is_shared_array_type(&item_static.ty)
        {
            names.push(item_static.ident.clone());
        }
    }
    names.sort_by_key(|a| a.to_string());
    names
}

fn is_shared_array_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "SharedArray")
}

/// Ensure every `&mut [T]` parameter on a contracted/verified kernel is covered by
/// both `no_oob_idx(param)` and `disjoint_writes(param)` (or `no_data_race(param)`).
pub(crate) fn validate_mutable_slice_verify_coverage(
    verify_args: Option<&VerifyArgs>,
    params: &[CudaModuleParam],
) -> syn::Result<()> {
    for param in params.iter().filter(|p| p.mutable_slice) {
        let Some(args) = verify_args else {
            return Err(syn::Error::new(
                param.name.span(),
                "contracted kernels cannot take `&mut [T]`; use `DisjointSlice<T, IndexSpace>` so the launch domain and per-thread write ownership are explicit, or prove exclusivity with `#[verify(no_oob_idx(...), disjoint_writes(...))]`",
            ));
        };
        let has_no_oob = args.no_oob_idx.iter().any(|id| id == &param.name);
        let has_disjoint = args.disjoint_writes.iter().any(|id| id == &param.name)
            || args.no_data_race.iter().any(|id| id == &param.name);
        if !has_no_oob || !has_disjoint {
            return Err(syn::Error::new(
                param.name.span(),
                format!(
                    "mutable slice parameter `{0}` (`&mut [T]`) requires both `no_oob_idx({0})` and `disjoint_writes({0})` in `#[verify(...)]`; otherwise use `DisjointSlice<T>`",
                    param.name
                ),
            ));
        }
    }
    Ok(())
}

fn build_verify_marker_stmts(
    args: &VerifyArgs,
    params: &[CudaModuleParam],
    shared_statics: &[Ident],
) -> syn::Result<Vec<Stmt>> {
    validate_mutable_slice_verify_coverage(Some(args), params)?;
    let mut stmts = Vec::new();
    for ident in &args.no_oob_idx {
        if let Some(shared_idx) = shared_statics.iter().position(|name| name == ident) {
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_oob_shared_config::<#shared_idx>();
            });
        } else {
            let index = resolve_verify_slice_param(ident, params, false, "no_oob_idx")?;
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_oob_idx_config::<#index>();
            });
        }
    }
    for ident in &args.disjoint_writes {
        let index = resolve_verify_slice_param(ident, params, true, "disjoint_writes")?;
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_disjoint_writes_config::<#index>();
        });
    }
    for ident in &args.writes_all_elems {
        let index = resolve_verify_slice_param(ident, params, true, "writes_all_elems")?;
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_writes_all_elems_config::<#index>();
        });
    }
    for ident in &args.no_data_race {
        if let Some(shared_idx) = shared_statics.iter().position(|name| name == ident) {
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_no_data_race_config::<#shared_idx>();
            });
        } else {
            let index = resolve_verify_slice_param(ident, params, true, "no_data_race")?;
            stmts.push(parse_quote! {
                ::cuda_device::thread::__verify_disjoint_writes_config::<#index>();
            });
        }
    }
    if args.uniform_barriers.is_some() {
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_uniform_barriers_config();
        });
    }
    if args.tmem_lifecycle.is_some() {
        stmts.push(parse_quote! {
            ::cuda_device::thread::__verify_tmem_lifecycle_config();
        });
    }
    Ok(stmts)
}

fn resolve_verify_slice_param(
    ident: &Ident,
    params: &[CudaModuleParam],
    require_writable: bool,
    clause: &str,
) -> syn::Result<usize> {
    let Some((index, param)) = params
        .iter()
        .enumerate()
        .find(|(_, param)| param.name == *ident)
    else {
        return Err(syn::Error::new(
            ident.span(),
            format!("unknown kernel parameter `{ident}` in `{clause}(...)`"),
        ));
    };
    match param.marshal {
        CudaModuleParamMarshal::WritableDeviceBuffer { .. }
        | CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => Ok(index),
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } if !require_writable => Ok(index),
        CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. } => Err(syn::Error::new(
            ident.span(),
            format!(
                "`{clause}({ident})` requires a writable buffer (`DisjointSlice` or `&mut [T]`), but `{ident}` is read-only"
            ),
        )),
        CudaModuleParamMarshal::Scalar => Err(syn::Error::new(
            ident.span(),
            format!(
                "`{clause}({ident})` requires a slice or `DisjointSlice` parameter, found scalar `{ident}`"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_rejects_unknown_clause_readonly_writes_and_scalar_params() {
        let bad_clause = syn::parse_str::<VerifyArgs>("magic_clause(a)");
        assert!(
            bad_clause
                .err()
                .unwrap()
                .to_string()
                .contains("unknown #[verify] obligation `magic_clause`")
        );

        let func: ItemFn = parse_quote! {
            pub fn vecadd(a: &[f32], n: u32, mut c: DisjointSlice<f32>) {}
        };
        let params = standalone_requires_params(&func).expect("params");

        // `disjoint_writes(a)` on read-only slice `a: &[f32]` must be rejected.
        let readonly_write: VerifyArgs = syn::parse_str("disjoint_writes(a)").unwrap();
        let err = build_verify_marker_stmts(&readonly_write, &params, &[]).unwrap_err();
        assert!(err.to_string().contains("requires a writable buffer"));

        // `no_oob_idx(n)` on scalar `n: u32` must be rejected.
        let scalar_oob: VerifyArgs = syn::parse_str("no_oob_idx(n)").unwrap();
        let err = build_verify_marker_stmts(&scalar_oob, &params, &[]).unwrap_err();
        assert!(err.to_string().contains("found scalar `n`"));

        // `no_oob_idx(missing)` on unknown parameter must be rejected.
        let unknown_param: VerifyArgs = syn::parse_str("no_oob_idx(missing)").unwrap();
        let err = build_verify_marker_stmts(&unknown_param, &params, &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown kernel parameter `missing`")
        );

        // `uniform_barriers(x)` with arguments must be rejected.
        let bad_uniform = syn::parse_str::<VerifyArgs>("uniform_barriers(a)");
        assert!(
            bad_uniform
                .err()
                .unwrap()
                .to_string()
                .contains("`uniform_barriers` takes no arguments")
        );

        // `no_oob_idx(TILE_A)` and `no_data_race(TILE_A)` on in-body SharedArray statics succeed.
        let tiled_fn: ItemFn = parse_quote! {
            pub fn sgemm(a: &[f32], mut c: DisjointSlice<f32>) {
                static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
            }
        };
        let shared = collect_shared_array_statics(&tiled_fn.block.stmts);
        assert_eq!(shared.len(), 1);
        let tiled_args: VerifyArgs =
            syn::parse_str("no_oob_idx(TILE_A), no_data_race(TILE_A), uniform_barriers").unwrap();
        let markers = build_verify_marker_stmts(&tiled_args, &params, &shared).unwrap();
        assert_eq!(markers.len(), 3);

        // `rewrite_verified_shared_accesses` must wrap `g2s` TMA copies on verified `TILE_A`
        // + `BAR`, while leaving unverified `OTHER_TILE` and `s2g` TMA copies unwrapped.
        let mut async_fn: ItemFn = parse_quote! {
            pub fn async_k(a: &[f32], mut c: DisjointSlice<f32>) {
                static mut TILE_A: SharedArray<f32, 4096, 128> = SharedArray::UNINIT;
                static mut OTHER_TILE: SharedArray<f32, 64, 128> = SharedArray::UNINIT;
                static mut BAR: Barrier = Barrier::UNINIT;
                cp_async_bulk_tensor_2d_g2s_cta(&raw mut TILE_A as *mut u8, tmap, 0, 0, &raw mut BAR);
                cp_async_bulk_tensor_2d_g2s_cta(&raw mut OTHER_TILE as *mut u8, tmap, 0, 0, &raw mut BAR);
                cp_async_bulk_tensor_2d_s2g(&raw const TILE_A as *const u8, tmap, 0, 0);
            }
        };
        let shared_all = collect_shared_array_statics(&async_fn.block.stmts);
        rewrite_verified_shared_accesses(&tiled_args, &shared_all, &mut async_fn.block.stmts);
        let rendered = quote!(#async_fn).to_string();
        assert_eq!(
            rendered.matches("unsafe").count(),
            1,
            "only verified g2s TMA copy on TILE_A should be wrapped in unsafe: {rendered}"
        );

        // `tmem_lifecycle(a)` with arguments must be rejected.
        let bad_tmem = syn::parse_str::<VerifyArgs>("tmem_lifecycle(a)");
        assert!(
            bad_tmem
                .err()
                .unwrap()
                .to_string()
                .contains("`tmem_lifecycle` takes no arguments")
        );

        // Without `tmem_lifecycle`, `tcgen05_*` and `cvta_generic_to_shared_offset` must NOT
        // be wrapped in `unsafe`; with `tmem_lifecycle`, only verified statics/barriers are wrapped.
        let mut tmem_fn: ItemFn = parse_quote! {
            pub fn tmem_k(a: &[f32], mut c: DisjointSlice<f32>) {
                static mut TILE_A: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
                static mut OTHER_TILE: SharedArray<u8, 4096, 128> = SharedArray::UNINIT;
                static mut TMEM_ADDR: SharedArray<u32, 1, 4> = SharedArray::UNINIT;
                static mut BAR: Barrier = Barrier::UNINIT;
                static mut OTHER_BAR: u64 = 0;
                tcgen05_alloc(&raw mut TMEM_ADDR[0], 128);
                tcgen05_alloc(&raw mut OTHER_TILE as *mut u32, 128);
                let _a = cvta_generic_to_shared_offset(&raw const TILE_A as *const u8);
                let _b = cvta_generic_to_shared_offset(&raw const OTHER_TILE as *const u8);
                tcgen05_mma_f16(0, 0, 0, 0, false);
                tcgen05_commit_shared_cluster(&raw mut BAR as *mut u64);
                tcgen05_commit_shared_cluster(&raw mut OTHER_BAR);
                let _r = tcgen05_ld_32x32b_x1_raw(0);
                tcgen05_load_wait();
                tcgen05_dealloc(0, 128);
            }
        };
        let tmem_shared = collect_shared_array_statics(&tmem_fn.block.stmts);
        let mut tmem_without_clause = tmem_fn.clone();
        rewrite_verified_shared_accesses(
            &tiled_args,
            &tmem_shared,
            &mut tmem_without_clause.block.stmts,
        );
        assert_eq!(
            quote!(#tmem_without_clause)
                .to_string()
                .matches("unsafe")
                .count(),
            0,
            "tcgen05 calls must remain unwrapped when `tmem_lifecycle` is omitted"
        );
        let tmem_args: VerifyArgs = syn::parse_str(
            "no_oob_idx(TILE_A, TMEM_ADDR), no_data_race(TILE_A, TMEM_ADDR), uniform_barriers, tmem_lifecycle",
        )
        .unwrap();
        rewrite_verified_shared_accesses(&tmem_args, &tmem_shared, &mut tmem_fn.block.stmts);
        let rendered_tmem = quote!(#tmem_fn).to_string();
        assert_eq!(
            rendered_tmem.matches("unsafe").count(),
            7,
            "expected 7 scoped tcgen05/cvta calls wrapped in unsafe (leaving OTHER_TILE and OTHER_BAR unwrapped): {rendered_tmem}"
        );

        // Stage 8: `SharedArray<Barrier, 4, 8>` listed in `no_oob_idx(...)` lifts indexed
        // `&raw mut FULL_BAR[s]` / `mbarrier_*`, while leaving omitted `UNVERIFIED_BAR[s]` unwrapped!
        let mut ring_fn: ItemFn = parse_quote! {
            pub fn ring_k(a: &[f32], mut c: DisjointSlice<f32>) {
                static mut RING_BUF: SharedArray<f32, 256, 16> = SharedArray::UNINIT;
                static mut FULL_BAR: SharedArray<Barrier, 4, 8> = SharedArray::UNINIT;
                static mut UNVERIFIED_BAR: SharedArray<Barrier, 4, 8> = SharedArray::UNINIT;
                mbarrier_init(&raw mut FULL_BAR[0], 1);
                mbarrier_arrive(&raw const FULL_BAR[0]);
                let _ = mbarrier_try_wait_parity(&raw const FULL_BAR[0], 0);
                mbarrier_init(&raw mut UNVERIFIED_BAR[0], 1);
            }
        };
        let ring_shared = collect_shared_array_statics(&ring_fn.block.stmts);
        let ring_args: VerifyArgs = syn::parse_str(
            "no_oob_idx(RING_BUF, FULL_BAR), no_data_race(RING_BUF), uniform_barriers",
        )
        .unwrap();
        rewrite_verified_shared_accesses(&ring_args, &ring_shared, &mut ring_fn.block.stmts);
        let rendered_ring = quote!(#ring_fn).to_string();
        assert_eq!(
            rendered_ring.matches("unsafe").count(),
            3,
            "expected 3 FULL_BAR mbarrier calls wrapped in unsafe while leaving UNVERIFIED_BAR unwrapped: {rendered_ring}"
        );
    }
}
