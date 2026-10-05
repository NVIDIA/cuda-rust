/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shadow-dispatch synthesis for rank-polymorphic ops.
//!
//! For a rank-polymorphic op like:
//!
//! ```ignore
//! pub fn addf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
//!     lhs: Tile<E, S>, rhs: Tile<E, S>, rounding: R, ftz: F,
//! ) -> Tile<E, S>
//! ```
//!
//! this module synthesizes auxiliary scaffolding — a *shadow trait* (CGA-erased
//! and rank-polymorphic), one rank-instance impl per rank, and a free-fn wrapper —
//! that exists alongside the user's source for the sole purpose of letting
//! rustc's normal trait resolution dispatch each call site to the correct
//! rank-instance impl. None of these items are user-authored; they "shadow"
//! the user's variadic op, mirroring its semantics in a form rustc can resolve.
//! No macro-time call-site rewriting, no name-keyed registry consulted.
//!
//! ## Coverage
//!
//! The synthesizer supports three patterns of rank-polymorphic ops:
//!
//! - **Same-shape (case 3a).** All shape-bearing args and the return share a
//!   single CGA and the same type. Examples: `addf`, `subf`, `maxf`, `fma`,
//!   unary math. Emits `trait Op<...> { fn op(...) -> Self; }` with rank-instance
//!   impls on `Tile_N<E, S_0, ..., S_{N-1}>`.
//!
//! - **Different shape, bound (case 3b).** The return type differs from Self
//!   but every generic in the return is also referenced by an argument.
//!   Examples: `reshape`, `broadcast`, `constant`, comparisons (bool return).
//!   Emits `trait Op<Sh> { type Out; fn op(self, ...) -> Self::Out; }`.
//!
//! - **Different shape, free (case 3c).** The return type contains a generic
//!   param (type param or CGA) not present in any argument. Associated-type
//!   dispatch would violate coherence because multiple return shapes share the
//!   same Self + arg types, so `Out` is promoted to a shadow-trait generic and
//!   the caller ascribes the return type. Examples: `permute`, `reduce_sum`,
//!   `bitcast`, `exti`.
//!
//! Rank-linked multi-CGA ops (e.g. `broadcast` where `S` and `R` share length
//! `N`) emit one impl per shared rank. Independent CGAs (e.g. `reshape`'s
//! rank-N source and rank-M target) emit a Cartesian product.
//!
//! The trait+impl entry points ([`desugar_variadic_trait_decl`],
//! [`desugar_variadic_trait_impl`]) handle the same three cases for user-
//! authored variadic traits like `BroadcastScalar`. Here the user's trait
//! is the source of truth; the macro synthesizes a shadow rank-polymorphic
//! trait that the user's variadic trait desugars into.

use std::collections::BTreeMap;

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    spanned::Spanned, AngleBracketedGenericArguments, Expr, ExprPath, FnArg, GenericArgument,
    GenericParam, Generics, Ident, ImplItem, ImplItemFn, ItemFn, ItemImpl, ItemTrait, Pat,
    PathArguments, ReturnType, TraitItem, Type,
};

use crate::error::{syn_err, syn_error_at, Error};

/// Maximum rank the emitter produces rank-instance impls for. Must match the
/// range of `Tile_N` variants produced by `#[cuda_tile::variadic_struct]`.
pub const MAX_RANK: usize = 6;

/// Emit trait + impls + wrapper for a rank-polymorphic op.
///
/// `method_override` lets the caller set the trait method name to something
/// other than the fn ident. Used for ops whose user-facing method differs
/// from the free-fn name (e.g. `reshape_ptr` → method `reshape` so callers
/// write `ptr_tile.reshape(shape)`). When `None`, the fn ident is used.
///
/// `trait_name_override` lets the caller set the synthesized trait name to
/// something other than `PascalCase(fn_ident)`. Used to avoid collisions with
/// user-defined traits in the same module (e.g. `broadcast_scalar` → synth
/// `BroadcastScalar` would collide with the user's `BroadcastScalar` trait).
pub fn emit_shadow_dispatch(
    item: &ItemFn,
    method_override: Option<Ident>,
    trait_name_override: Option<Ident>,
) -> Result<TokenStream2, Error> {
    let spec = RankPolyOpSpec::parse(item, method_override, trait_name_override)?;

    let trait_decl = spec.emit_trait();
    let impls = spec.emit_impls();
    let wrapper = spec.emit_wrapper();

    Ok(quote! {
        #trait_decl
        #(#impls)*
        #wrapper
    })
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// One CGA generic: `const <cga_ident>: [i32; <length_ident>]`.
#[derive(Clone, Debug)]
struct CgaInfo {
    cga_ident: Ident,
    length_ident: Ident,
}

/// Parsed view of a rank-polymorphic op's signature.
struct RankPolyOpSpec {
    /// The free-fn's ident (used for wrapper name, trait name basis).
    fn_ident: Ident,
    /// The trait method ident (usually same as `fn_ident`, but may differ for
    /// ops whose user-facing method name is different from the free-fn name,
    /// e.g. `reshape_ptr` → method `reshape`).
    method_ident: Ident,
    trait_ident: Ident,

    /// All CGAs in the signature, in declaration order.
    cgas: Vec<CgaInfo>,
    /// CGAs grouped by shared length-ident (e.g., broadcast's S and R both
    /// use N → one group). Order preserves first appearance.
    length_groups: Vec<LengthGroup>,

    /// Non-shape-bearing-arg-referenced generics (trait-level).
    trait_level_params: Vec<GenericParam>,
    /// Generics that only appear inside shape-bearing arg types (impl-only).
    /// Excludes "dead" lifetimes that appear only in the return type — those
    /// would be unconstrained on the impl (E0207), so they're replaced with
    /// `'_` in the return type and dropped from impl generics.
    impl_only_params: Vec<GenericParam>,

    where_clause: Option<syn::WhereClause>,

    /// Arguments in original order.
    args: Vec<ParsedArg>,
    /// CGA used by return type, if return is shape-bearing.
    return_cga: Option<Ident>,
    /// Full return type (needed for return-type rewriting per impl). Dead
    /// lifetimes (return-only) are kept as-is; emit_impl rewrites them to a
    /// receiver-bound lifetime (or `'_` if no usable binder).
    return_type: Type,
    /// `true` if the original `fn` is `unsafe`. Propagated to the trait
    /// method, the impl method, and the wrapper free fn.
    is_unsafe: bool,
    /// Original lifetime params that appear only in the return.
    dead_lifetimes: Vec<String>,
    /// Synthesized trait-level lifetime idents, one per dead lifetime. These
    /// appear as trait generics so callers can ascribe an output lifetime
    /// (lifetime analog of the case-3c "Out as trait generic" pattern).
    dead_lt_idents: Vec<syn::Lifetime>,
    /// For each non-shape arg whose type references a CGA length ident (e.g.
    /// `idx: [i32; N]` in `load_tile`), the synthesized trait generic ident
    /// used in its place. Aligned to `args` by index — `None` for args that
    /// don't need promotion.
    rank_dep_arg_idents: Vec<Option<Ident>>,
}

#[derive(Clone, Debug)]
struct LengthGroup {
    length_ident: Ident,
    cgas: Vec<Ident>,
}

#[derive(Debug)]
struct ParsedArg {
    name: Ident,
    /// CGA this arg's type references (first match), if any.
    cga: Option<Ident>,
    ty: Type,
    /// Whether this arg's type is structurally identical to the first
    /// shape-bearing arg's type (so it should become `Self` in the method).
    is_self_like: bool,
}

impl RankPolyOpSpec {
    fn parse(
        item: &ItemFn,
        method_override: Option<Ident>,
        trait_name_override: Option<Ident>,
    ) -> Result<Self, Error> {
        let fn_ident = item.sig.ident.clone();
        let method_ident = method_override.unwrap_or_else(|| fn_ident.clone());
        let trait_ident = trait_name_override
            .unwrap_or_else(|| format_ident!("{}", pascal_case(&fn_ident.to_string())));

        let cgas = find_cgas(&item.sig.generics, fn_ident.span())?;
        if cgas.is_empty() {
            return syn_error_at(
                fn_ident.span(),
                "shadow_dispatch: no `const X: [i32; N]` generic found \
                 (is this op rank-polymorphic?)",
            );
        }
        let length_groups = group_cgas_by_length(&cgas);

        // Classify each argument.
        let mut args = Vec::new();
        let mut first_shape_bearing_ty: Option<Type> = None;
        for arg in &item.sig.inputs {
            match arg {
                FnArg::Typed(pat_type) => {
                    let name = match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                        _ => {
                            return syn_error_at(
                                pat_type.pat.span(),
                                "shadow_dispatch: unsupported argument pattern",
                            );
                        }
                    };
                    let ty = (*pat_type.ty).clone();
                    let cga = cgas
                        .iter()
                        .find(|c| type_references_ident(&ty, &c.cga_ident))
                        .map(|c| c.cga_ident.clone());
                    let is_self_like = if cga.is_some() {
                        match &first_shape_bearing_ty {
                            None => {
                                first_shape_bearing_ty = Some(ty.clone());
                                true
                            }
                            Some(prev) => types_equal(prev, &ty),
                        }
                    } else {
                        false
                    };
                    args.push(ParsedArg {
                        name,
                        cga,
                        ty,
                        is_self_like,
                    });
                }
                FnArg::Receiver(r) => {
                    return syn_error_at(
                        r.span(),
                        "shadow_dispatch: free fns only (no `self` receiver)",
                    );
                }
            }
        }

        if first_shape_bearing_ty.is_none() {
            return syn_error_at(
                fn_ident.span(),
                "shadow_dispatch: no shape-bearing argument found",
            );
        }

        // Return type info. Treat `()` (default / explicit unit) uniformly: no
        // CGA, no Out, no associated type — the trait method just returns ().
        let (raw_return_type, return_cga) = match &item.sig.output {
            ReturnType::Default => (syn::parse_quote! { () }, None),
            ReturnType::Type(_, ty) => {
                let cga = cgas
                    .iter()
                    .find(|c| type_references_ident(ty, &c.cga_ident))
                    .map(|c| c.cga_ident.clone());
                ((**ty).clone(), cga)
            }
        };

        // Identify "dead" lifetimes — those that appear in the return type but
        // not in any argument type. Leaving them in the impl's generic params
        // would be unconstrained (E0207). Replace them with `'_` in the return
        // type and exclude them from impl_only_params below.
        let all_arg_types: Vec<&Type> = args.iter().map(|a| &a.ty).collect();
        let dead_lifetimes: Vec<String> = item
            .sig
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let GenericParam::Lifetime(lt) = p {
                    let ident_s = lt.lifetime.ident.to_string();
                    let in_args = all_arg_types
                        .iter()
                        .any(|ty| type_references_ident(ty, &lt.lifetime.ident));
                    let in_return = type_references_ident(&raw_return_type, &lt.lifetime.ident);
                    if in_return && !in_args {
                        return Some(ident_s);
                    }
                }
                None
            })
            .collect();
        // Note: we keep raw_return_type as-is here. emit_impl will rewrite
        // dead lifetime occurrences either to the receiver's named lifetime
        // (when the first arg is a reference) or to `'_` as a fallback.
        let return_type = raw_return_type;

        // Classify non-CGA generics as trait-level (used in non-shape-bearing
        // arg types) vs impl-only (only inside shape-bearing arg types).
        let non_shape_bearing_types: Vec<&Type> = args
            .iter()
            .filter(|a| a.cga.is_none())
            .map(|a| &a.ty)
            .collect();

        let cga_idents: Vec<&Ident> = cgas.iter().map(|c| &c.cga_ident).collect();
        let length_idents: Vec<&Ident> = cgas.iter().map(|c| &c.length_ident).collect();

        let mut trait_level_params = Vec::new();
        let mut impl_only_params = Vec::new();
        for param in &item.sig.generics.params {
            // Skip dead lifetimes entirely — they've been elided from the
            // return type with `'_` and aren't constrained anywhere.
            if let GenericParam::Lifetime(lt) = param {
                if dead_lifetimes
                    .iter()
                    .any(|d| d == &lt.lifetime.ident.to_string())
                {
                    continue;
                }
            }
            let ident = match param {
                GenericParam::Type(t) => t.ident.clone(),
                GenericParam::Const(c) if cga_idents.iter().any(|i| **i == c.ident) => continue,
                GenericParam::Const(c) if length_idents.iter().any(|i| **i == c.ident) => continue,
                GenericParam::Const(c) => c.ident.clone(),
                GenericParam::Lifetime(lt) => lt.lifetime.ident.clone(),
            };
            let used_in_non_shape_arg = non_shape_bearing_types
                .iter()
                .any(|ty| type_references_ident(ty, &ident));
            if used_in_non_shape_arg {
                trait_level_params.push(param.clone());
            } else {
                impl_only_params.push(param.clone());
            }
        }

        let dead_lt_idents: Vec<syn::Lifetime> = (0..dead_lifetimes.len())
            .map(|i| syn::parse_str(&format!("'__td_lt{}", i)).unwrap())
            .collect();

        // Identify non-shape args whose type references a CGA length ident
        // (rank-dependent arg types like `[i32; N]`). These are promoted to
        // trait generics so the wrapper can hold them as opaque types.
        let mut rank_dep_arg_idents: Vec<Option<Ident>> = vec![None; args.len()];
        let mut next_ra = 0usize;
        for (i, arg) in args.iter().enumerate() {
            if arg.cga.is_some() {
                continue;
            }
            let mentions_length = cgas
                .iter()
                .any(|c| type_references_ident(&arg.ty, &c.length_ident));
            if mentions_length {
                rank_dep_arg_idents[i] = Some(format_ident!("__td_arg{}", next_ra));
                next_ra += 1;
            }
        }

        let is_unsafe = item.sig.unsafety.is_some();

        Ok(Self {
            fn_ident,
            method_ident,
            trait_ident,
            cgas,
            length_groups,
            trait_level_params,
            impl_only_params,
            where_clause: item.sig.generics.where_clause.clone(),
            args,
            return_cga,
            return_type,
            dead_lifetimes,
            dead_lt_idents,
            rank_dep_arg_idents,
            is_unsafe,
        })
    }

    /// Is this op case-3a (return type is same as Self-like first shape arg)?
    fn is_same_shape(&self) -> bool {
        match &self.return_cga {
            None => false,
            Some(ret_cga) => {
                // Return uses the primary (first shape-bearing arg's) CGA and the
                // return type structurally matches the first shape-bearing arg type.
                let first_shape_bearing = self.args.iter().find(|a| a.cga.is_some()).unwrap();
                let first_cga = first_shape_bearing.cga.as_ref().unwrap();
                ret_cga == first_cga && types_equal(&first_shape_bearing.ty, &self.return_type)
            }
        }
    }

    /// Returns true if the op returns `()` — handled like case-3a (Self), but
    /// with the unit type rather than Self. Skips Out entirely.
    fn is_void_return(&self) -> bool {
        matches!(&self.return_type, Type::Tuple(t) if t.elems.is_empty())
    }

    /// Case-3c: the return type contains generic params that aren't present
    /// in any argument type. Rustc can't derive an associated `Out` because
    /// multiple valid Outs would share the same Self + arg types (coherence
    /// error), so `Out` must become a trait generic that the caller ascribes
    /// at the return site. Examples: `permute`, `reduce_sum`, `exti`, `cat`.
    fn return_is_free(&self) -> bool {
        if self.is_same_shape() {
            return false;
        }
        if self.return_cga.is_none() {
            return false;
        }
        let arg_texts: Vec<String> = self
            .args
            .iter()
            .map(|a| {
                let ty = &a.ty;
                quote! { #ty }.to_string()
            })
            .collect();
        for param in &self.impl_only_params {
            let ident = match param {
                GenericParam::Type(t) => t.ident.clone(),
                GenericParam::Const(c) => c.ident.clone(),
                GenericParam::Lifetime(_) => continue,
            };
            let in_any_arg = arg_texts
                .iter()
                .any(|s| contains_whole_word(s, &ident.to_string()));
            if !in_any_arg && type_references_ident(&self.return_type, &ident) {
                return true;
            }
        }
        for ret_cga in &self.cgas {
            if type_references_ident(&self.return_type, &ret_cga.cga_ident)
                && !self
                    .args
                    .iter()
                    .any(|a| a.cga.as_ref() == Some(&ret_cga.cga_ident))
            {
                return true;
            }
        }
        false
    }

    fn emit_trait(&self) -> TokenStream2 {
        let trait_ident = &self.trait_ident;
        let method_ident = &self.method_ident;
        let trait_params = &self.trait_level_params;
        let where_clause = &self.where_clause;

        let extra_shape_generics = self.extra_shape_trait_generics();
        let extra_shape_generic_params: Vec<TokenStream2> = extra_shape_generics
            .iter()
            .map(|ident| quote! { #ident })
            .collect();

        let method_inputs = self.method_inputs_tokens(&extra_shape_generics);

        let (return_token, assoc_type, extra_out_trait_param): (
            TokenStream2,
            TokenStream2,
            Option<TokenStream2>,
        ) = if self.is_void_return() {
            (quote! { () }, quote! {}, None)
        } else if self.is_same_shape() {
            (quote! { Self }, quote! {}, None)
        } else if self.return_is_free() {
            // `Out` is a trait generic, not an associated type. Caller ascribes
            // the return type; rustc binds Out via return-type inference.
            (quote! { Out }, quote! {}, Some(quote! { Out }))
        } else {
            (quote! { Self::Out }, quote! { type Out; }, None)
        };

        let mut all_trait_params: Vec<TokenStream2> = Vec::new();
        // Dead-lifetime trait generics first (lifetimes lead in Rust generics).
        for lt in &self.dead_lt_idents {
            all_trait_params.push(quote! { #lt });
        }
        for p in &extra_shape_generic_params {
            all_trait_params.push(p.clone());
        }
        // Rank-dependent non-shape arg types as trait generics (e.g. `Idx0`
        // for `idx: [i32; N]`). Caller's array literal pins them.
        for id in self.rank_dep_arg_idents.iter().flatten() {
            all_trait_params.push(quote! { #id });
        }
        if let Some(ref out) = extra_out_trait_param {
            all_trait_params.push(out.clone());
        }
        for p in trait_params {
            all_trait_params.push(quote! { #p });
        }

        let unsafe_kw: TokenStream2 = if self.is_unsafe {
            quote! { unsafe }
        } else {
            quote! {}
        };

        quote! {
            #[allow(non_camel_case_types)]
            pub trait #trait_ident < #(#all_trait_params),* >
                #where_clause
            {
                #assoc_type
                #unsafe_kw fn #method_ident(#(#method_inputs),*) -> #return_token;
            }
        }
    }

    fn emit_impls(&self) -> Vec<TokenStream2> {
        let mut impls = Vec::new();
        // Iterate cartesian product of ranks across length groups.
        let rank_space = RankSpace::new(&self.length_groups, MAX_RANK);
        for combo in rank_space.iter() {
            impls.push(self.emit_impl(&combo));
        }
        impls
    }

    fn emit_impl(&self, combo: &RankCombo) -> TokenStream2 {
        let trait_ident = &self.trait_ident;
        let method_ident = &self.method_ident;
        let where_clause = &self.where_clause;
        let impl_only = &self.impl_only_params;
        let trait_level = &self.trait_level_params;

        // Scalar dim generics for every CGA across all groups, in declaration order.
        let dim_params: Vec<TokenStream2> = combo.dim_params_tokens().into_iter().collect();

        // Compute the Self type: rewrite the first shape-bearing arg's type
        // using the rank-instance dim names.
        let first_shape_bearing = self.args.iter().find(|a| a.cga.is_some()).unwrap();
        let self_type = rewrite_ty_for_rank(&first_shape_bearing.ty, combo, &self.cgas);

        // Extra shape generics bound to concrete types for this impl.
        let extra_shape_bindings: Vec<Type> = self
            .args
            .iter()
            .filter(|a| a.cga.is_some() && !a.is_self_like)
            .map(|a| rewrite_ty_for_rank(&a.ty, combo, &self.cgas))
            .collect();

        // Trait args = extra shape bindings + trait-level generics (passed through).
        let trait_level_args = self
            .trait_level_params
            .iter()
            .map(|p| match p {
                GenericParam::Type(t) => {
                    let i = &t.ident;
                    quote! { #i }
                }
                GenericParam::Const(c) => {
                    let i = &c.ident;
                    quote! { #i }
                }
                GenericParam::Lifetime(lt) => {
                    let l = &lt.lifetime;
                    quote! { #l }
                }
            })
            .collect::<Vec<_>>();

        // Return type concrete for this impl: rewrite self.return_type, then
        // substitute each dead lifetime (originally a return-only `'s`) with
        // its synthesized trait-level lifetime ident (`'__td_lt0`, etc).
        let mut return_concrete = rewrite_ty_for_rank(&self.return_type, combo, &self.cgas);
        for (orig, replacement) in self.dead_lifetimes.iter().zip(self.dead_lt_idents.iter()) {
            return_concrete =
                replace_lifetimes_with(&return_concrete, std::slice::from_ref(orig), replacement);
        }

        let mut trait_instantiation_args: Vec<TokenStream2> = Vec::new();
        // Dead-lifetime trait generics are passed through directly — each
        // impl just bumps them as impl-level generics.
        for lt in &self.dead_lt_idents {
            trait_instantiation_args.push(quote! { #lt });
        }
        for binding in &extra_shape_bindings {
            trait_instantiation_args.push(quote! { #binding });
        }
        // Rank-dep arg types: concrete-rewritten per-impl. Order matches the
        // trait declaration's ordering of rank_dep_arg_idents.
        for (i, slot) in self.rank_dep_arg_idents.iter().enumerate() {
            if slot.is_some() {
                let ty = rewrite_ty_for_rank(&self.args[i].ty, combo, &self.cgas);
                trait_instantiation_args.push(quote! { #ty });
            }
        }
        if !self.is_same_shape() && self.return_is_free() {
            trait_instantiation_args.push(quote! { #return_concrete });
        }
        for arg in &trait_level_args {
            trait_instantiation_args.push(arg.clone());
        }

        let method_inputs = self.impl_method_inputs_tokens(combo);
        let method_input_muted = self.method_arg_idents_muted();

        let (return_in_method, assoc_type_item): (TokenStream2, TokenStream2) =
            if self.is_void_return() {
                (quote! { () }, quote! {})
            } else if self.is_same_shape() {
                (quote! { Self }, quote! {})
            } else if self.return_is_free() {
                (quote! { #return_concrete }, quote! {})
            } else {
                (
                    quote! { Self::Out },
                    quote! { type Out = #return_concrete; },
                )
            };

        // Collect all impl-level generic params in one list so we can comma-join
        // them cleanly (avoiding stray leading commas when one bucket is empty).
        let mut all_impl_params: Vec<TokenStream2> = Vec::new();
        // Dead lifetimes appear at impl-level too (matching the trait's
        // lifetime generics).
        for lt in &self.dead_lt_idents {
            all_impl_params.push(quote! { #lt });
        }
        for p in impl_only {
            all_impl_params.push(quote! { #p });
        }
        for p in trait_level {
            all_impl_params.push(quote! { #p });
        }
        for p in &dim_params {
            all_impl_params.push(p.clone());
        }

        let unsafe_kw: TokenStream2 = if self.is_unsafe {
            quote! { unsafe }
        } else {
            quote! {}
        };

        quote! {
            impl < #(#all_impl_params),* > #trait_ident < #(#trait_instantiation_args),* >
                for #self_type
                #where_clause
            {
                #assoc_type_item
                #unsafe_kw fn #method_ident(#(#method_inputs),*) -> #return_in_method {
                    let _ = (#(#method_input_muted),*);
                    ::std::unreachable!()
                }
            }
        }
    }

    /// Method arg tokens for an **impl** block — extra-shape args use their
    /// concrete rewritten types, not trait generic names. `self` is always
    /// first regardless of where the first shape-bearing arg appears in the
    /// original source (constant's `(value, shape)` becomes `(self, value)`).
    /// Non-shape arg types are also rewritten so rank-dependent forms like
    /// `[i32; N]` resolve to a concrete length.
    fn impl_method_inputs_tokens(&self, combo: &RankCombo) -> Vec<TokenStream2> {
        let mut tokens = Vec::new();
        tokens.push(quote! { self });
        let mut seen_first_shape_bearing = false;
        for arg in &self.args {
            let name = &arg.name;
            if arg.cga.is_some() {
                if !seen_first_shape_bearing {
                    seen_first_shape_bearing = true;
                    continue;
                } else if arg.is_self_like {
                    tokens.push(quote! { #name: Self });
                } else {
                    let ty = rewrite_ty_for_rank(&arg.ty, combo, &self.cgas);
                    tokens.push(quote! { #name: #ty });
                }
            } else {
                let ty = rewrite_ty_for_rank(&arg.ty, combo, &self.cgas);
                tokens.push(quote! { #name: #ty });
            }
        }
        tokens
    }

    fn emit_wrapper(&self) -> TokenStream2 {
        let fn_ident = &self.fn_ident;
        let method_ident = &self.method_ident;
        let trait_ident = &self.trait_ident;
        let trait_params = &self.trait_level_params;
        let where_clause = &self.where_clause;

        let t_ident = Ident::new("__T", Span::call_site());
        let extra_shape_generics = self.extra_shape_trait_generics();
        let extra_shape_generic_idents: Vec<Ident> = extra_shape_generics.clone();

        // Detect receiver reference kind from the first shape-bearing arg's
        // type. The trait Self type IS the reference (e.g. `&Tensor_N<…>`),
        // but the wrapper must take a *concrete* `&T` / `&mut T` (with a
        // wrapper-level lifetime) so Rust's reborrow + `&mut → &` coercion
        // kick in at the call site. A by-value generic `tensor: T` would
        // move `&mut Tensor` callers and reject `&mut → &` coercion.
        let first_shape_arg = self.args.iter().find(|a| a.cga.is_some()).unwrap();
        let recv_ref_kind: Option<bool> = match &first_shape_arg.ty {
            Type::Reference(r) => Some(r.mutability.is_some()),
            _ => None,
        };
        let recv_lifetime: Option<syn::Lifetime> =
            recv_ref_kind.map(|_| syn::parse_quote! { '__td_recv });

        // The receiver type expression — used in wrapper arg, return (if 3a),
        // and trait bound. For non-ref receivers it's just `T`; for ref
        // receivers it's `&'a T` or `&'a mut T`.
        let receiver_ty_expr: TokenStream2 = match (&recv_ref_kind, &recv_lifetime) {
            (Some(true), Some(lt)) => quote! { & #lt mut #t_ident },
            (Some(false), Some(lt)) => quote! { & #lt #t_ident },
            _ => quote! { #t_ident },
        };

        let mut wrapper_args = Vec::new();
        let mut extra_shape_idx = 0;
        let mut seen_first_shape_bearing = false;
        for (i, arg) in self.args.iter().enumerate() {
            let name = &arg.name;
            if arg.cga.is_some() {
                if arg.is_self_like {
                    if !seen_first_shape_bearing {
                        seen_first_shape_bearing = true;
                        wrapper_args.push(quote! { #name: #receiver_ty_expr });
                    } else {
                        // Other self-like args (e.g. addf's `rhs`) share the
                        // receiver's type. For ref receivers this means a
                        // separate `&'a T` of the same kind.
                        wrapper_args.push(quote! { #name: #receiver_ty_expr });
                    }
                } else {
                    let sh = &extra_shape_generic_idents[extra_shape_idx];
                    extra_shape_idx += 1;
                    wrapper_args.push(quote! { #name: #sh });
                }
            } else if let Some(rd) = &self.rank_dep_arg_idents[i] {
                wrapper_args.push(quote! { #name: #rd });
            } else {
                // Apply literal-CGA rewrites (e.g. `PointerTile<P, {[]}>` →
                // `PointerTile_0<P>`). CGA references stay generic; the
                // wrapper is rank-polymorphic over them.
                let ty = rewrite_literal_cgas_only(&arg.ty);
                wrapper_args.push(quote! { #name: #ty });
            }
        }

        let trait_level_args = trait_params
            .iter()
            .map(|p| match p {
                GenericParam::Type(t) => {
                    let i = &t.ident;
                    quote! { #i }
                }
                GenericParam::Const(c) => {
                    let i = &c.ident;
                    quote! { #i }
                }
                GenericParam::Lifetime(lt) => {
                    let l = &lt.lifetime;
                    quote! { #l }
                }
            })
            .collect::<Vec<_>>();

        let out_ident = Ident::new("Out", Span::call_site());
        let use_free_out = !self.is_same_shape() && self.return_is_free();

        let mut trait_args: Vec<TokenStream2> = Vec::new();
        // Dead-lifetime trait generics first (lifetimes lead).
        for lt in &self.dead_lt_idents {
            trait_args.push(quote! { #lt });
        }
        for i in &extra_shape_generic_idents {
            trait_args.push(quote! { #i });
        }
        // Rank-dep arg generics, matching trait declaration ordering.
        for id in self.rank_dep_arg_idents.iter().flatten() {
            trait_args.push(quote! { #id });
        }
        if use_free_out {
            trait_args.push(quote! { #out_ident });
        }
        for arg in &trait_level_args {
            trait_args.push(arg.clone());
        }

        // Wrapper return: `()` for void, the receiver type for case-3a (Self),
        // `Out` for case-3c, `<receiver_ty as Trait>::Out` for case-3b.
        let wrapper_return = if self.is_void_return() {
            quote! { () }
        } else if self.is_same_shape() {
            receiver_ty_expr.clone()
        } else if use_free_out {
            quote! { #out_ident }
        } else {
            quote! { <#receiver_ty_expr as #trait_ident < #(#trait_args),* >>::Out }
        };

        let receiver_name = self
            .args
            .iter()
            .find(|a| a.cga.is_some())
            .map(|a| a.name.clone())
            .unwrap();

        let mut method_call_args: Vec<TokenStream2> = Vec::new();
        let mut seen_receiver = false;
        for arg in &self.args {
            if !seen_receiver && arg.cga.is_some() {
                seen_receiver = true;
                continue;
            }
            let n = &arg.name;
            method_call_args.push(quote! { #n });
        }

        let mut all_wrapper_generics: Vec<TokenStream2> = Vec::new();
        // Lifetime generics first (Rust order).
        if let Some(lt) = &recv_lifetime {
            all_wrapper_generics.push(quote! { #lt });
        }
        for lt in &self.dead_lt_idents {
            all_wrapper_generics.push(quote! { #lt });
        }
        all_wrapper_generics.push(quote! { #t_ident });
        for i in &extra_shape_generic_idents {
            all_wrapper_generics.push(quote! { #i });
        }
        for id in self.rank_dep_arg_idents.iter().flatten() {
            all_wrapper_generics.push(quote! { #id });
        }
        if use_free_out {
            all_wrapper_generics.push(quote! { #out_ident });
        }
        for p in trait_params.iter() {
            all_wrapper_generics.push(quote! { #p });
        }

        let wrapper_where = if let Some(wc) = where_clause {
            let preds = &wc.predicates;
            quote! { where #receiver_ty_expr: #trait_ident < #(#trait_args),* >, #preds }
        } else {
            quote! { where #receiver_ty_expr: #trait_ident < #(#trait_args),* > }
        };

        let unsafe_kw: TokenStream2 = if self.is_unsafe {
            quote! { unsafe }
        } else {
            quote! {}
        };
        // Inside an unsafe wrapper, calling the unsafe trait method requires
        // an unsafe block. For non-unsafe ops it's just a plain call.
        let body_call: TokenStream2 = if self.is_unsafe {
            quote! { unsafe { #receiver_name.#method_ident( #(#method_call_args),* ) } }
        } else {
            quote! { #receiver_name.#method_ident( #(#method_call_args),* ) }
        };

        quote! {
            pub #unsafe_kw fn #fn_ident < #(#all_wrapper_generics),* > ( #(#wrapper_args),* ) -> #wrapper_return
                #wrapper_where
            {
                #body_call
            }
        }
    }

    /// Token stream for the method's argument list — used in trait decl. `self`
    /// is always emitted first regardless of where the first shape-bearing arg
    /// appears in the source signature (ensures valid Rust syntax).
    fn method_inputs_tokens(&self, extra_shape_generics: &[Ident]) -> Vec<TokenStream2> {
        let mut tokens = Vec::new();
        tokens.push(quote! { self });
        let mut seen_first_shape_bearing = false;
        let mut extra_shape_idx = 0;
        for (i, arg) in self.args.iter().enumerate() {
            let name = &arg.name;
            if arg.cga.is_some() {
                if !seen_first_shape_bearing {
                    seen_first_shape_bearing = true;
                    continue;
                } else if arg.is_self_like {
                    tokens.push(quote! { #name: Self });
                } else {
                    let sh = &extra_shape_generics[extra_shape_idx];
                    extra_shape_idx += 1;
                    tokens.push(quote! { #name: #sh });
                }
            } else if let Some(rd) = &self.rank_dep_arg_idents[i] {
                tokens.push(quote! { #name: #rd });
            } else {
                // Apply literal-CGA rewrites (e.g. `PointerTile<P, {[]}>` →
                // `PointerTile_0<P>`); CGA references stay generic.
                let ty = rewrite_literal_cgas_only(&arg.ty);
                tokens.push(quote! { #name: #ty });
            }
        }
        tokens
    }

    fn method_arg_idents_muted(&self) -> Vec<TokenStream2> {
        let mut tokens = Vec::new();
        let mut seen_first_shape_bearing = false;
        for arg in &self.args {
            if arg.cga.is_some() && !seen_first_shape_bearing {
                seen_first_shape_bearing = true;
                continue;
            }
            let n = &arg.name;
            tokens.push(quote! { #n });
        }
        tokens
    }

    /// Trait generics allocated for "other shape-bearing args" that aren't
    /// Self-like. Named `Sh0`, `Sh1`, ....
    fn extra_shape_trait_generics(&self) -> Vec<Ident> {
        let mut count: usize = 0;
        let mut names = Vec::new();
        let mut seen_first_shape_bearing = false;
        for arg in &self.args {
            if arg.cga.is_some() {
                if !seen_first_shape_bearing {
                    seen_first_shape_bearing = true;
                } else if !arg.is_self_like {
                    names.push(format_ident!("Sh{}", count));
                    count += 1;
                }
            }
        }
        names
    }
}

// ---------------------------------------------------------------------------
// Rank-combination iteration
// ---------------------------------------------------------------------------

/// Space of rank combinations across length groups. For reshape (2
/// independent groups), iterates 0..=MAX_RANK × 0..=MAX_RANK. For broadcast
/// (1 group with 2 CGAs sharing N), iterates 0..=MAX_RANK once.
struct RankSpace {
    group_lengths: Vec<Ident>, // length idents in group order (e.g., [N, M])
    cga_to_group: BTreeMap<String, usize>, // cga_ident string → group index
    cgas: Vec<CgaInfo>,
    max_rank: usize,
}

impl RankSpace {
    fn new(groups: &[LengthGroup], max_rank: usize) -> Self {
        let group_lengths: Vec<Ident> = groups.iter().map(|g| g.length_ident.clone()).collect();
        let mut cga_to_group = BTreeMap::new();
        let mut cgas = Vec::new();
        for (idx, g) in groups.iter().enumerate() {
            for cga_ident in &g.cgas {
                cga_to_group.insert(cga_ident.to_string(), idx);
                cgas.push(CgaInfo {
                    cga_ident: cga_ident.clone(),
                    length_ident: g.length_ident.clone(),
                });
            }
        }
        Self {
            group_lengths,
            cga_to_group,
            cgas,
            max_rank,
        }
    }

    fn iter(&self) -> Vec<RankCombo> {
        let n_groups = self.group_lengths.len();
        if n_groups == 0 {
            return vec![];
        }
        // Cartesian product: for each group independently, iterate 0..=MAX.
        let mut result = Vec::new();
        let per_group_range: usize = self.max_rank + 1;
        let total = per_group_range.pow(n_groups as u32);
        for code in 0..total {
            let mut ranks_per_group = Vec::with_capacity(n_groups);
            let mut n = code;
            for _ in 0..n_groups {
                ranks_per_group.push(n % per_group_range);
                n /= per_group_range;
            }
            result.push(RankCombo {
                ranks_per_group,
                cga_to_group: self.cga_to_group.clone(),
                cgas: self.cgas.clone(),
            });
        }
        result
    }
}

/// One concrete (rank_of_group0, rank_of_group1, ...) combination.
struct RankCombo {
    ranks_per_group: Vec<usize>,
    cga_to_group: BTreeMap<String, usize>,
    cgas: Vec<CgaInfo>,
}

impl RankCombo {
    /// Rank for a given CGA ident.
    fn rank_of(&self, cga_ident: &Ident) -> usize {
        let group_idx = *self.cga_to_group.get(&cga_ident.to_string()).unwrap_or(&0);
        self.ranks_per_group[group_idx]
    }

    /// Rank for a given CGA-length ident (e.g., `N` → rank of any CGA using N).
    /// Used when rewriting array lengths like `[Tile<i32, {[]}>; N]`.
    fn rank_of_length(&self, length_ident: &Ident) -> Option<usize> {
        for cga in &self.cgas {
            if cga.length_ident == *length_ident {
                return Some(self.rank_of(&cga.cga_ident));
            }
        }
        None
    }

    /// Dim names for a given CGA. For CGA `S` at rank 2, returns `[S_0, S_1]`.
    fn dim_names_for(&self, cga_ident: &Ident) -> Vec<Ident> {
        let rank = self.rank_of(cga_ident);
        (0..rank)
            .map(|i| format_ident!("{}_{}", cga_ident, i))
            .collect()
    }

    /// All dim generic names across all CGAs (used when building dim_params).
    fn dim_names_per_cga(&self) -> BTreeMap<String, Vec<Ident>> {
        let mut out = BTreeMap::new();
        for cga in &self.cgas {
            out.insert(
                cga.cga_ident.to_string(),
                self.dim_names_for(&cga.cga_ident),
            );
        }
        out
    }

    /// Flattened list of `const X_i: i32` tokens for all dim generics in
    /// this combo, in CGA declaration order.
    fn dim_params_tokens(&self) -> Vec<TokenStream2> {
        let mut out = Vec::new();
        for cga in &self.cgas {
            for name in self.dim_names_for(&cga.cga_ident) {
                out.push(quote! { const #name: i32 });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_cgas(generics: &Generics, _span: Span) -> Result<Vec<CgaInfo>, Error> {
    let mut out = Vec::new();
    for param in &generics.params {
        if let GenericParam::Const(c) = param {
            if let Type::Array(arr) = &c.ty {
                let is_i32 = matches!(
                    &*arr.elem,
                    Type::Path(p) if p.path.is_ident("i32")
                );
                if !is_i32 {
                    continue;
                }
                let length_ident = match &arr.len {
                    Expr::Path(ExprPath { path, .. }) => path
                        .get_ident()
                        .cloned()
                        .ok_or_else(|| syn_err(c.ty.span(), "CGA length must be a simple ident"))?,
                    _ => continue,
                };
                out.push(CgaInfo {
                    cga_ident: c.ident.clone(),
                    length_ident,
                });
            }
        }
    }
    Ok(out)
}

fn group_cgas_by_length(cgas: &[CgaInfo]) -> Vec<LengthGroup> {
    let mut groups: Vec<LengthGroup> = Vec::new();
    for cga in cgas {
        if let Some(g) = groups
            .iter_mut()
            .find(|g| g.length_ident == cga.length_ident)
        {
            g.cgas.push(cga.cga_ident.clone());
        } else {
            groups.push(LengthGroup {
                length_ident: cga.length_ident.clone(),
                cgas: vec![cga.cga_ident.clone()],
            });
        }
    }
    groups
}

fn pascal_case(name: &str) -> String {
    let mut out = String::new();
    let mut upper_next = true;
    for ch in name.chars() {
        if ch == '_' {
            upper_next = true;
            continue;
        }
        if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

fn type_references_ident(ty: &Type, ident: &Ident) -> bool {
    let s = quote! { #ty }.to_string();
    contains_whole_word(&s, &ident.to_string())
}

fn contains_whole_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let nb = needle.as_bytes();
    let mut i = 0;
    while i + nb.len() <= bytes.len() {
        if &bytes[i..i + nb.len()] == nb {
            let prev_ok = i == 0 || !is_word_byte(bytes[i - 1]);
            let next_ok = i + nb.len() == bytes.len() || !is_word_byte(bytes[i + nb.len()]);
            if prev_ok && next_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn types_equal(a: &Type, b: &Type) -> bool {
    quote! { #a }.to_string() == quote! { #b }.to_string()
}

/// Literal-CGA-only type rewrite: walks a type and rewrites any
/// `Tile<E, {[…]}>`-style segment to its concrete `Tile_K<E, …>` form, leaving
/// CGA-name references (e.g. `Tile<E, S>`) untouched. Used by the wrapper
/// emitter where the wrapper is generic over CGAs and only literals can be
/// resolved at macro time.
fn rewrite_literal_cgas_only(ty: &Type) -> Type {
    match ty {
        Type::Path(tp) => {
            let mut new_tp = tp.clone();
            if let Some(last_seg) = new_tp.path.segments.last_mut() {
                if let PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    args, ..
                }) = &mut last_seg.arguments
                {
                    if let Some(rank) = args.iter().find_map(literal_cga_rank) {
                        let original_base = last_seg.ident.clone();
                        last_seg.ident = format_ident!("{}_{}", last_seg.ident, rank);
                        let already_has_lifetime = args
                            .iter()
                            .any(|a| matches!(a, GenericArgument::Lifetime(_)));
                        let needs_lifetime =
                            base_has_lifetime(&original_base) && !already_has_lifetime;
                        let mut new_args: syn::punctuated::Punctuated<
                            GenericArgument,
                            syn::Token![,],
                        > = syn::punctuated::Punctuated::new();
                        if needs_lifetime {
                            let lt: syn::Lifetime = syn::parse_quote! { '_ };
                            new_args.push(GenericArgument::Lifetime(lt));
                        }
                        for arg in args.iter() {
                            if let Some(dims) = literal_cga_dims(arg) {
                                for d in dims {
                                    let dim_expr: Expr = syn::parse_quote! { #d };
                                    new_args.push(GenericArgument::Const(dim_expr));
                                }
                            } else if let GenericArgument::Type(t) = arg {
                                new_args.push(GenericArgument::Type(rewrite_literal_cgas_only(t)));
                            } else {
                                new_args.push(arg.clone());
                            }
                        }
                        *args = new_args;
                        if let PathArguments::AngleBracketed(a) = &last_seg.arguments {
                            if a.args.is_empty() {
                                last_seg.arguments = PathArguments::None;
                            }
                        }
                    } else {
                        let mut new_args: syn::punctuated::Punctuated<
                            GenericArgument,
                            syn::Token![,],
                        > = syn::punctuated::Punctuated::new();
                        for arg in args.iter() {
                            if let GenericArgument::Type(t) = arg {
                                new_args.push(GenericArgument::Type(rewrite_literal_cgas_only(t)));
                            } else {
                                new_args.push(arg.clone());
                            }
                        }
                        *args = new_args;
                    }
                }
            }
            Type::Path(new_tp)
        }
        Type::Reference(r) => {
            let mut new_r = r.clone();
            *new_r.elem = rewrite_literal_cgas_only(&r.elem);
            Type::Reference(new_r)
        }
        Type::Array(a) => {
            let mut new_a = a.clone();
            *new_a.elem = rewrite_literal_cgas_only(&a.elem);
            Type::Array(new_a)
        }
        Type::Tuple(t) => {
            let mut new_t = t.clone();
            new_t.elems = t.elems.iter().map(rewrite_literal_cgas_only).collect();
            Type::Tuple(new_t)
        }
        other => other.clone(),
    }
}

/// If `arg` is a const generic of form `{[d0, d1, …]}` (a block containing an
/// array literal), return the array's element count (the literal CGA's rank).
/// Combo-independent — only handles array literals, not repeat exprs.
fn literal_cga_rank(arg: &GenericArgument) -> Option<usize> {
    if let GenericArgument::Const(Expr::Block(block)) = arg {
        if let Some(syn::Stmt::Expr(Expr::Array(arr), _)) = block.block.stmts.first() {
            return Some(arr.elems.len());
        }
    }
    None
}

/// Combo-aware literal CGA rank: handles both array literals (`{[d0, d1, …]}`)
/// and repeat exprs whose count is a CGA length ident (`{[expr; N]}`). Used in
/// per-impl type rewriting where `combo` resolves length idents to concrete ranks.
fn literal_cga_rank_with_combo(arg: &GenericArgument, combo: &RankCombo) -> Option<usize> {
    if let Some(rank) = literal_cga_rank(arg) {
        return Some(rank);
    }
    if let GenericArgument::Const(Expr::Block(block)) = arg {
        if let Some(syn::Stmt::Expr(Expr::Repeat(rep), _)) = block.block.stmts.first() {
            if let Expr::Path(p) = &*rep.len {
                if let Some(ident) = p.path.get_ident() {
                    return combo.rank_of_length(ident);
                }
            }
        }
    }
    None
}

/// If `arg` is a literal CGA `{[d0, d1, …]}`, return the dimension expressions
/// (`d0`, `d1`, …) so they can be reflowed as individual const generics.
fn literal_cga_dims(arg: &GenericArgument) -> Option<Vec<Expr>> {
    if let GenericArgument::Const(Expr::Block(block)) = arg {
        if let Some(syn::Stmt::Expr(Expr::Array(arr), _)) = block.block.stmts.first() {
            return Some(arr.elems.iter().cloned().collect());
        }
    }
    None
}

/// Combo-aware dimension expansion: for `{[expr; N]}` where N is a CGA length,
/// produce N copies of `expr`. Falls back to `literal_cga_dims` for plain arrays.
fn literal_cga_dims_with_combo(arg: &GenericArgument, combo: &RankCombo) -> Option<Vec<Expr>> {
    if let Some(dims) = literal_cga_dims(arg) {
        return Some(dims);
    }
    if let GenericArgument::Const(Expr::Block(block)) = arg {
        if let Some(syn::Stmt::Expr(Expr::Repeat(rep), _)) = block.block.stmts.first() {
            if let Expr::Path(p) = &*rep.len {
                if let Some(ident) = p.path.get_ident() {
                    if let Some(rank) = combo.rank_of_length(ident) {
                        return Some((0..rank).map(|_| (*rep.expr).clone()).collect());
                    }
                }
            }
        }
    }
    None
}

/// If `ty` is `&X` or `&mut X`, set its lifetime to `lt` (creating an explicit
/// binder where there was an elided one). Otherwise returns `ty` unchanged.
fn bind_outer_reference_lifetime(ty: &Type, lt: &syn::Lifetime) -> Type {
    if let Type::Reference(r) = ty {
        let mut new_r = r.clone();
        new_r.lifetime = Some(lt.clone());
        Type::Reference(new_r)
    } else {
        ty.clone()
    }
}

/// Walks a type and replaces every reference to a named lifetime in `dead`
/// with `replacement`. Used to bind return-only lifetimes to a synthesized
/// receiver-borrow lifetime so they're well-formed on the impl.
fn replace_lifetimes_with(ty: &Type, dead: &[String], replacement: &syn::Lifetime) -> Type {
    match ty {
        Type::Path(tp) => {
            let mut new_tp = tp.clone();
            for seg in new_tp.path.segments.iter_mut() {
                if let PathArguments::AngleBracketed(ab) = &mut seg.arguments {
                    let mut new_args: syn::punctuated::Punctuated<GenericArgument, syn::Token![,]> =
                        syn::punctuated::Punctuated::new();
                    for arg in ab.args.iter() {
                        match arg {
                            GenericArgument::Lifetime(lt)
                                if dead.iter().any(|d| d == &lt.ident.to_string()) =>
                            {
                                new_args.push(GenericArgument::Lifetime(replacement.clone()));
                            }
                            GenericArgument::Type(t) => new_args.push(GenericArgument::Type(
                                replace_lifetimes_with(t, dead, replacement),
                            )),
                            other => new_args.push(other.clone()),
                        }
                    }
                    ab.args = new_args;
                }
            }
            Type::Path(new_tp)
        }
        Type::Reference(r) => {
            let mut new_r = r.clone();
            if let Some(lt) = &r.lifetime {
                if dead.iter().any(|d| d == &lt.ident.to_string()) {
                    new_r.lifetime = Some(replacement.clone());
                }
            }
            *new_r.elem = replace_lifetimes_with(&r.elem, dead, replacement);
            Type::Reference(new_r)
        }
        Type::Array(a) => {
            let mut new_a = a.clone();
            *new_a.elem = replace_lifetimes_with(&a.elem, dead, replacement);
            Type::Array(new_a)
        }
        Type::Tuple(t) => {
            let mut new_t = t.clone();
            new_t.elems = t
                .elems
                .iter()
                .map(|e| replace_lifetimes_with(e, dead, replacement))
                .collect();
            Type::Tuple(new_t)
        }
        other => other.clone(),
    }
}

/// Base names of variadic-struct types that carry an `'a` lifetime parameter.
/// When `rewrite_ty_for_rank` rewrites `Shape<...>` → `Shape_N<...>`, we must
/// prepend `'_` to the rank-instance form because `Shape_N<'a, const D_0: i32, ...>`
/// has a lifetime that is NOT elidable in impl header positions (E0726).
const LIFETIME_BEARING_BASES: &[&str] = &[
    "Shape",
    "Array",
    "Partition",
    "PartitionMut",
    "BoundedPartition",
    "BoundedPartitionMut",
    "GatherScatterView",
    "StridedView",
];

fn base_has_lifetime(base: &Ident) -> bool {
    let s = base.to_string();
    LIFETIME_BEARING_BASES.iter().any(|b| *b == s)
}

/// Rewrite a type referencing CGAs to use rank-instance `Tile_N<E, S_0, ..., S_{N-1}>` form.
///
/// Handles nested paths and references. For each path segment, if it has
/// generic args that reference a CGA, we:
/// 1. Suffix the path segment ident with `_N` (where N is the CGA's rank).
/// 2. Replace the CGA generic arg with the scalar dim arg references.
/// 3. Prepend `'_` for known lifetime-bearing variadic types (Shape, Array, ...)
///    only when the original type doesn't already provide a lifetime arg.
///
/// Whether or not the outer segment is itself a CGA-bearing variadic type, we
/// **always recurse into its generic args** so nested CGAs (e.g.
/// `Option<Tile<bool, S>>`) get rewritten too.
fn rewrite_ty_for_rank(ty: &Type, combo: &RankCombo, cgas: &[CgaInfo]) -> Type {
    match ty {
        Type::Path(tp) => {
            let mut new_tp = tp.clone();
            if let Some(last_seg) = new_tp.path.segments.last_mut() {
                let original_base = last_seg.ident.clone();
                let mut cgas_for_this_seg: Vec<&CgaInfo> = Vec::new();
                let mut already_has_lifetime = false;
                if let PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    args, ..
                }) = &last_seg.arguments
                {
                    for arg in args.iter() {
                        if matches!(arg, GenericArgument::Lifetime(_)) {
                            already_has_lifetime = true;
                        }
                        if let Some(cga) = match_arg_to_cga(arg, cgas) {
                            cgas_for_this_seg.push(cga);
                        }
                    }
                }
                if !cgas_for_this_seg.is_empty() {
                    let rank_suffix = cgas_for_this_seg
                        .iter()
                        .map(|cga| combo.rank_of(&cga.cga_ident).to_string())
                        .collect::<Vec<_>>()
                        .join("_");
                    last_seg.ident = format_ident!("{}_{}", last_seg.ident, rank_suffix);
                    let needs_lifetime = base_has_lifetime(&original_base) && !already_has_lifetime;
                    if let PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                        args,
                        ..
                    }) = &mut last_seg.arguments
                    {
                        let mut new_args: syn::punctuated::Punctuated<
                            GenericArgument,
                            syn::Token![,],
                        > = syn::punctuated::Punctuated::new();
                        if needs_lifetime {
                            let lt: syn::Lifetime = syn::parse_quote! { '_ };
                            new_args.push(GenericArgument::Lifetime(lt));
                        }
                        for arg in args.iter() {
                            if let Some(cga_ref) = match_arg_to_cga(arg, cgas) {
                                let dim_names = combo.dim_names_for(&cga_ref.cga_ident);
                                for dim_name in dim_names {
                                    let dim_expr: Expr = syn::parse_quote! { #dim_name };
                                    new_args.push(GenericArgument::Const(dim_expr));
                                }
                            } else if let GenericArgument::Type(t) = arg {
                                let rewritten = rewrite_ty_for_rank(t, combo, cgas);
                                new_args.push(GenericArgument::Type(rewritten));
                            } else {
                                new_args.push(arg.clone());
                            }
                        }
                        *args = new_args;
                        if let PathArguments::AngleBracketed(a) = &last_seg.arguments {
                            if a.args.is_empty() {
                                last_seg.arguments = PathArguments::None;
                            }
                        }
                    } else if needs_lifetime {
                        // Rank-0 with no prior arg list: synthesize `<'_>`.
                        let args: AngleBracketedGenericArguments = syn::parse_quote! { <'_> };
                        last_seg.arguments = PathArguments::AngleBracketed(args);
                    }
                } else if let PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                    args,
                    ..
                }) = &mut last_seg.arguments
                {
                    // No CGA at this level — but the segment may carry a
                    // *literal* CGA like `Tile<E, {[128, 64]}>`. If so, suffix
                    // the segment by the literal rank and inline the array
                    // elements as individual const generics, e.g. `Tile_2<E, 128, 64>`.
                    let literal_ranks = args
                        .iter()
                        .filter_map(|a| literal_cga_rank_with_combo(a, combo))
                        .collect::<Vec<_>>();
                    if !literal_ranks.is_empty() {
                        let original_base = last_seg.ident.clone();
                        let rank_suffix = literal_ranks
                            .iter()
                            .map(|rank| rank.to_string())
                            .collect::<Vec<_>>()
                            .join("_");
                        last_seg.ident = format_ident!("{}_{}", last_seg.ident, rank_suffix);
                        let already_has_lifetime = args
                            .iter()
                            .any(|a| matches!(a, GenericArgument::Lifetime(_)));
                        let needs_lifetime =
                            base_has_lifetime(&original_base) && !already_has_lifetime;
                        let mut new_args: syn::punctuated::Punctuated<
                            GenericArgument,
                            syn::Token![,],
                        > = syn::punctuated::Punctuated::new();
                        if needs_lifetime {
                            let lt: syn::Lifetime = syn::parse_quote! { '_ };
                            new_args.push(GenericArgument::Lifetime(lt));
                        }
                        for arg in args.iter() {
                            if let Some(dims) = literal_cga_dims_with_combo(arg, combo) {
                                for d in dims {
                                    let dim_expr: Expr = syn::parse_quote! { #d };
                                    new_args.push(GenericArgument::Const(dim_expr));
                                }
                            } else if let GenericArgument::Type(t) = arg {
                                let rewritten = rewrite_ty_for_rank(t, combo, cgas);
                                new_args.push(GenericArgument::Type(rewritten));
                            } else {
                                new_args.push(arg.clone());
                            }
                        }
                        *args = new_args;
                        if let PathArguments::AngleBracketed(a) = &last_seg.arguments {
                            if a.args.is_empty() {
                                last_seg.arguments = PathArguments::None;
                            }
                        }
                    } else {
                        // No literal CGA — still recurse into nested type args
                        // so e.g. `Option<Tile<bool, S>>` gets its inner `Tile`
                        // rewritten to `Tile_N<bool, S_0>`.
                        let mut new_args: syn::punctuated::Punctuated<
                            GenericArgument,
                            syn::Token![,],
                        > = syn::punctuated::Punctuated::new();
                        for arg in args.iter() {
                            if let GenericArgument::Type(t) = arg {
                                let rewritten = rewrite_ty_for_rank(t, combo, cgas);
                                new_args.push(GenericArgument::Type(rewritten));
                            } else {
                                new_args.push(arg.clone());
                            }
                        }
                        *args = new_args;
                    }
                }
            }
            Type::Path(new_tp)
        }
        Type::Reference(r) => {
            let mut new_r = r.clone();
            *new_r.elem = rewrite_ty_for_rank(&r.elem, combo, cgas);
            Type::Reference(new_r)
        }
        Type::Array(a) => {
            let mut new_a = a.clone();
            *new_a.elem = rewrite_ty_for_rank(&a.elem, combo, cgas);
            if let Expr::Path(ExprPath { path, .. }) = &a.len {
                if let Some(ident) = path.get_ident() {
                    if let Some(rank) = combo.rank_of_length(ident) {
                        new_a.len = syn::parse_quote! { #rank };
                    }
                }
            }
            Type::Array(new_a)
        }
        Type::Tuple(t) => {
            let mut new_t = t.clone();
            new_t.elems = t
                .elems
                .iter()
                .map(|e| rewrite_ty_for_rank(e, combo, cgas))
                .collect();
            Type::Tuple(new_t)
        }
        other => other.clone(),
    }
}

/// If `arg` is a generic argument that resolves to exactly one of the CGA
/// idents, return that CGA.
fn match_arg_to_cga<'a>(arg: &GenericArgument, cgas: &'a [CgaInfo]) -> Option<&'a CgaInfo> {
    let ident = match arg {
        GenericArgument::Type(Type::Path(p)) => p.path.get_ident(),
        GenericArgument::Const(Expr::Path(ExprPath { path, .. })) => path.get_ident(),
        _ => None,
    }?;
    cgas.iter().find(|c| &c.cga_ident == ident)
}

// ---------------------------------------------------------------------------
// Variadic trait/impl desugaring
// ---------------------------------------------------------------------------
//
// For a `#[variadic_trait]` declaration like:
//
//     pub trait BroadcastScalar<E, const D: [i32; N]> where Self: ElementType {
//         fn broadcast(self, shape: Shape<D>) -> Tile<E, D>;
//     }
//
// `desugar_variadic_trait_decl` emits a single CGA-erased shadow trait:
//
//     pub trait BroadcastScalar<E, Sh0> where Self: ElementType {
//         type Out;
//         fn broadcast(self, shape: Sh0) -> Self::Out;
//     }
//
// And `desugar_variadic_trait_impl`, applied to the matching impl, emits one
// concrete impl per rank instance. At call sites rustc's normal trait resolution picks
// the right impl based on the receiver's element type and the shape arg's
// concrete type — no macro-time call-site rewriting needed.
//
// Two rank-polymorphic-trait shapes (mirroring the free-fn emitter):
//
// - **Case-3b (associated `Out`).** Every CGA referenced by the return is also
//   referenced by some method argument. Sh<i> uniquely determines the return,
//   so the trait carries an associated `type Out;` and the impls bind it.
//
// - **Case-3c (`Out` as trait generic).** At least one CGA appears only in the
//   return type (free). Associated-type dispatch can't cover this since the
//   same Self + arg types would correspond to multiple Outs (coherence error),
//   so `Out` is promoted to a trait generic and the caller ascribes the
//   return type at the call site.

/// One CGA's role in the rank-polymorphic-trait desugaring.
#[derive(Clone, Debug)]
enum CgaRole {
    /// Referenced by some method argument — gets a synthesized `Sh<i>` type
    /// generic in the shadow trait.
    ShapeBound { sh_ident: Ident },
    /// Only in return type(s) — collapses into the case-3c `Out` trait
    /// generic. The shadow trait has no `type Out;`; impls bind `Out` via the
    /// trait reference.
    Free,
}

/// Per-CGA classification + the shadow trait's overall return-type shape.
struct RankPolyShape {
    /// Aligned to the trait's CGA list — `roles[i]` describes what to
    /// substitute for `cgas[i]` in trait references.
    roles: Vec<CgaRole>,
    /// Trait generics for shape-bound CGAs, in declaration order.
    sh_idents: Vec<Ident>,
    /// `true` when at least one CGA is free (case-3c). Determines whether
    /// `Out` is a trait generic vs an associated type.
    has_free_cga: bool,
    /// `true` when at least one method's return type uses any CGA.
    any_return_uses_cga: bool,
}

/// Classify each CGA based on whether any method argument references it.
/// `methods_args_iter` yields (arg_type) over every typed argument across
/// every method; `methods_returns_iter` yields the optional return type for
/// each method. The caller passes both because we need to do this for both
/// `ItemTrait` (decl side) and `ItemImpl` (impl side) inputs.
fn classify_cgas<'a, AI, RI>(
    cgas: &[CgaInfo],
    methods_args_iter: AI,
    methods_returns_iter: RI,
) -> RankPolyShape
where
    AI: IntoIterator<Item = &'a Type>,
    RI: IntoIterator<Item = &'a Type>,
{
    let arg_types: Vec<&Type> = methods_args_iter.into_iter().collect();
    let return_types: Vec<&Type> = methods_returns_iter.into_iter().collect();

    let mut sh_count = 0usize;
    let mut sh_idents: Vec<Ident> = Vec::new();
    let mut roles: Vec<CgaRole> = Vec::with_capacity(cgas.len());
    for cga in cgas {
        let in_args = arg_types
            .iter()
            .any(|t| type_references_ident(t, &cga.cga_ident));
        if in_args {
            let sh = format_ident!("Sh{}", sh_count);
            sh_count += 1;
            sh_idents.push(sh.clone());
            roles.push(CgaRole::ShapeBound { sh_ident: sh });
        } else {
            roles.push(CgaRole::Free);
        }
    }

    let has_free_cga = roles.iter().any(|r| matches!(r, CgaRole::Free));
    let any_return_uses_cga = return_types.iter().any(|ret| {
        cgas.iter()
            .any(|c| type_references_ident(ret, &c.cga_ident))
    });

    RankPolyShape {
        roles,
        sh_idents,
        has_free_cga,
        any_return_uses_cga,
    }
}

/// Desugar a `#[cuda_tile::variadic_trait]`-annotated trait declaration into
/// a single CGA-erased shadow trait.
pub fn desugar_variadic_trait_decl(item: &ItemTrait) -> Result<TokenStream2, Error> {
    let cgas = find_cgas(&item.generics, item.ident.span())?;
    if cgas.is_empty() {
        return syn_error_at(
            item.ident.span(),
            "variadic_trait: no `const X: [i32; N]` generic found \
             (is this trait rank-polymorphic?)",
        );
    }

    // Walk every method to classify CGAs.
    let arg_types: Vec<&Type> = item
        .items
        .iter()
        .filter_map(|ti| match ti {
            TraitItem::Fn(tf) => Some(tf.sig.inputs.iter().filter_map(|a| match a {
                FnArg::Typed(pt) => Some(&*pt.ty),
                _ => None,
            })),
            _ => None,
        })
        .flatten()
        .collect();
    let return_types: Vec<&Type> = item
        .items
        .iter()
        .filter_map(|ti| match ti {
            TraitItem::Fn(tf) => match &tf.sig.output {
                ReturnType::Type(_, ret) => Some(&**ret),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let shape = classify_cgas(
        &cgas,
        arg_types.iter().copied(),
        return_types.iter().copied(),
    );

    let cga_idents: Vec<Ident> = cgas.iter().map(|c| c.cga_ident.clone()).collect();

    // Rewrite each method in the trait body.
    let mut new_items: Vec<TraitItem> = Vec::new();
    for ti in &item.items {
        match ti {
            TraitItem::Fn(tf) => {
                let mut new_tf = tf.clone();
                rewrite_trait_method_for_rank_poly(&mut new_tf.sig, &cgas, &shape);
                new_items.push(TraitItem::Fn(new_tf));
            }
            other => new_items.push(other.clone()),
        }
    }
    // Add `type Out;` only for case-3b. Case-3c puts Out in the trait generics.
    if shape.any_return_uses_cga && !shape.has_free_cga {
        new_items.insert(0, syn::parse_quote! { type Out; });
    }

    // Build new generics: drop CGA params; keep the rest in declaration order;
    // append synthesized Sh<i> generics; append `Out` if case-3c.
    let mut new_params: syn::punctuated::Punctuated<GenericParam, syn::Token![,]> =
        syn::punctuated::Punctuated::new();
    for param in &item.generics.params {
        let drop_it = matches!(
            param,
            GenericParam::Const(c) if cga_idents.contains(&c.ident)
        );
        if !drop_it {
            new_params.push(param.clone());
        }
    }
    for sh in &shape.sh_idents {
        new_params.push(syn::parse_quote! { #sh });
    }
    if shape.has_free_cga && shape.any_return_uses_cga {
        new_params.push(syn::parse_quote! { Out });
    }

    let trait_ident = &item.ident;
    let where_clause = &item.generics.where_clause;
    let supertraits_marker = if item.supertraits.is_empty() {
        quote! {}
    } else {
        let st = &item.supertraits;
        quote! { : #st }
    };
    let vis = &item.vis;
    let attrs = filter_cuda_tile_attrs(&item.attrs);

    Ok(quote! {
        #(#attrs)*
        #[allow(non_camel_case_types)]
        #vis trait #trait_ident < #new_params > #supertraits_marker
            #where_clause
        {
            #(#new_items)*
        }
    })
}

/// Desugar a `#[cuda_tile::variadic_trait_impl]`-annotated impl block into
/// rank-instance impls of the shadow trait. CGA args in the trait reference
/// are substituted with concrete rank-instance types — shape-bound CGAs to the
/// arg type that uses them (e.g. `Shape_R<'lt, …>`); free CGAs (case-3c) to
/// the rewritten return type. Case-3b impls bind `type Out = …`; case-3c
/// impls bind `Out` via the trait reference's last generic arg.
pub fn desugar_variadic_trait_impl(item: &ItemImpl) -> Result<TokenStream2, Error> {
    let cgas = find_cgas(&item.generics, item.span())?;
    if cgas.is_empty() {
        return syn_error_at(
            item.span(),
            "variadic_trait_impl: no `const X: [i32; N]` generic found",
        );
    }
    if item.trait_.is_none() {
        return syn_error_at(
            item.span(),
            "variadic_trait_impl: expected a trait impl (e.g. `impl T for X`)",
        );
    }
    let length_groups = group_cgas_by_length(&cgas);
    let rank_space = RankSpace::new(&length_groups, MAX_RANK);

    // Walk every method to:
    //  - find, per shape-bound CGA, the first arg type that uses it (used
    //    as the substitute in trait reference + method signature per rank instance);
    //  - capture the first CGA-using return type for `Out` binding (case-3b)
    //    or as the substitute for the free CGA's trait-arg slot (case-3c).
    let mut cga_shape_types: Vec<Option<Type>> = vec![None; cgas.len()];
    let mut return_type_for_out: Option<Type> = None;
    for ii in &item.items {
        if let ImplItem::Fn(impl_fn) = ii {
            for arg in &impl_fn.sig.inputs {
                if let FnArg::Typed(pt) = arg {
                    for (i, cga) in cgas.iter().enumerate() {
                        if cga_shape_types[i].is_none()
                            && type_references_ident(&pt.ty, &cga.cga_ident)
                        {
                            cga_shape_types[i] = Some((*pt.ty).clone());
                        }
                    }
                }
            }
            if let ReturnType::Type(_, ret) = &impl_fn.sig.output {
                let uses_cga = cgas
                    .iter()
                    .any(|c| type_references_ident(ret, &c.cga_ident));
                if uses_cga && return_type_for_out.is_none() {
                    return_type_for_out = Some((**ret).clone());
                }
            }
        }
    }

    // Classify CGAs to match the trait-decl side's shadow-trait shape.
    let arg_types: Vec<&Type> = item
        .items
        .iter()
        .filter_map(|ii| match ii {
            ImplItem::Fn(impl_fn) => Some(impl_fn.sig.inputs.iter().filter_map(|a| match a {
                FnArg::Typed(pt) => Some(&*pt.ty),
                _ => None,
            })),
            _ => None,
        })
        .flatten()
        .collect();
    let return_types: Vec<&Type> = item
        .items
        .iter()
        .filter_map(|ii| match ii {
            ImplItem::Fn(impl_fn) => match &impl_fn.sig.output {
                ReturnType::Type(_, ret) => Some(&**ret),
                _ => None,
            },
            _ => None,
        })
        .collect();
    let shape = classify_cgas(
        &cgas,
        arg_types.iter().copied(),
        return_types.iter().copied(),
    );

    // Sanity: any CGA that classify_cgas marked Free (only in return) must
    // have a return type captured to substitute for it. If not (e.g. a CGA
    // that's neither in args nor in any return), the impl can't substitute
    // the trait-arg slot — surface a clear error.
    if shape.has_free_cga && return_type_for_out.is_none() {
        return syn_error_at(
            item.span(),
            "variadic_trait_impl: free CGAs detected but no method has a return \
             type that uses them — cannot substitute case-3c trait args",
        );
    }

    let mut impls = Vec::new();
    for combo in rank_space.iter() {
        impls.push(emit_variadic_trait_impl_for_rank(
            item,
            &cgas,
            &shape,
            &cga_shape_types,
            &return_type_for_out,
            &combo,
        ));
    }
    Ok(quote! { #(#impls)* })
}

fn emit_variadic_trait_impl_for_rank(
    item: &ItemImpl,
    cgas: &[CgaInfo],
    shape: &RankPolyShape,
    cga_shape_types: &[Option<Type>],
    return_type_for_out: &Option<Type>,
    combo: &RankCombo,
) -> TokenStream2 {
    let recv_lt: syn::Lifetime = syn::parse_quote! { '__td_recv };
    let cga_idents: Vec<Ident> = cgas.iter().map(|c| c.cga_ident.clone()).collect();

    // Decide whether the impl needs the synthesized receiver lifetime. It's
    // load-bearing when the rank-instance trait/Self/return form pulls in a
    // lifetime-bearing variadic (Shape_R, Array_R, …) or a reference;
    // declaring it unconditionally on impls that don't would be unconstrained
    // (E0207). Heuristic: if any of the source types we substitute into the
    // impl carries a lifetime (directly or transitively), recv_lt is needed.
    let needs_recv_lt = cga_shape_types
        .iter()
        .any(|t| t.as_ref().is_some_and(type_uses_lifetime))
        || return_type_for_out.as_ref().is_some_and(type_uses_lifetime)
        || type_uses_lifetime(&item.self_ty);

    // Trait reference: substitute each CGA position with its rank-instance
    // concrete type (shape-bound → arg type; free → return type).
    let (_, trait_path_orig, _) = item.trait_.as_ref().unwrap();
    let trait_path = rewrite_trait_path_args_for_rank(
        trait_path_orig,
        cgas,
        shape,
        cga_shape_types,
        return_type_for_out,
        combo,
        &recv_lt,
    );

    // Self type: rewrite per rank instance (e.g. `for E` stays as `E`; `for Tile<E, D>`
    // becomes `for Tile_R<E, D_0, …>`). Bind any elided lifetimes that the
    // rewrite introduced — `'_` is not allowed in impl trait positions.
    let self_ty_rewritten = rewrite_ty_for_rank(&item.self_ty, combo, cgas);
    let self_ty_rewritten = bind_anon_lifetimes_to(&self_ty_rewritten, &recv_lt);

    // Impl generics: drop CGAs (and their `[i32; N]` length usage), keep
    // others, append rank-instance scalar dim consts. Prepend the receiver lifetime
    // only when at least one substituted form actually uses it.
    let dim_params = combo.dim_params_tokens();
    let mut all_impl_params: Vec<TokenStream2> = Vec::new();
    if needs_recv_lt {
        all_impl_params.push(quote! { #recv_lt });
    }
    for param in &item.generics.params {
        let skip = matches!(
            param,
            GenericParam::Const(c) if cga_idents.contains(&c.ident)
        );
        if !skip {
            all_impl_params.push(quote! { #param });
        }
    }
    for d in &dim_params {
        all_impl_params.push(d.clone());
    }

    // Out: case-3b emits `type Out = …;` in the impl body; case-3c binds
    // `Out` via the trait reference's last generic arg, so no body item.
    let out_binding = if shape.has_free_cga {
        quote! {}
    } else {
        match return_type_for_out {
            Some(ret) => {
                let ret_concrete = rewrite_ty_for_rank(ret, combo, cgas);
                let ret_concrete = bind_anon_lifetimes_to(&ret_concrete, &recv_lt);
                quote! { type Out = #ret_concrete; }
            }
            None => quote! {},
        }
    };

    // Rewrite each method: signature uses concrete rank-instance types, body becomes
    // unreachable!() (the JIT works from original source, not macro output).
    let mut method_tokens: Vec<TokenStream2> = Vec::new();
    for ii in &item.items {
        match ii {
            ImplItem::Fn(impl_fn) => {
                method_tokens.push(rewrite_impl_method_body_for_rank(
                    impl_fn, cgas, combo, &recv_lt,
                ));
            }
            other => method_tokens.push(quote! { #other }),
        }
    }

    let where_clause = &item.generics.where_clause;
    let attrs = filter_cuda_tile_attrs(&item.attrs);

    quote! {
        #(#attrs)*
        impl < #(#all_impl_params),* > #trait_path for #self_ty_rewritten
            #where_clause
        {
            #out_binding
            #(#method_tokens)*
        }
    }
}

/// Rewrite a trait method signature for the shadow trait: shape-bearing
/// arg types become their corresponding `Sh<i>`; CGA-using return types
/// become `Self::Out` (case-3b) or the trait-generic `Out` (case-3c).
fn rewrite_trait_method_for_rank_poly(
    sig: &mut syn::Signature,
    cgas: &[CgaInfo],
    shape: &RankPolyShape,
) {
    for arg in sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = arg {
            let cga_idx = cgas
                .iter()
                .enumerate()
                .find(|(_, c)| type_references_ident(&pt.ty, &c.cga_ident))
                .map(|(i, _)| i);
            if let Some(i) = cga_idx {
                if let CgaRole::ShapeBound { sh_ident } = &shape.roles[i] {
                    *pt.ty = syn::parse_quote! { #sh_ident };
                }
                // Free CGAs aren't in args by definition (classify_cgas's
                // post-condition), so reaching this branch with a Free role
                // shouldn't happen. Leaving the type alone is the safe
                // fallback if it ever does.
            }
        }
    }
    if let ReturnType::Type(_, ret) = &mut sig.output {
        let uses_cga = cgas
            .iter()
            .any(|c| type_references_ident(ret, &c.cga_ident));
        if uses_cga {
            let new_ret: Type = if shape.has_free_cga {
                syn::parse_quote! { Out }
            } else {
                syn::parse_quote! { Self::Out }
            };
            **ret = new_ret;
        }
    }
}

/// Rewrite a trait reference's generic args. Each CGA arg is substituted
/// with its rank-instance concrete type:
///   - shape-bound CGA → the corresponding `cga_shape_types[i]` rewritten
///     for `combo` (e.g. `BroadcastScalar<E, D>` → `BroadcastScalar<E,
///     Shape_R<'lt, D_0, …>>`);
///   - free CGA (case-3c) → the captured return type rewritten for `combo`
///     (the `Out` slot in the shadow trait).
fn rewrite_trait_path_args_for_rank(
    path: &syn::Path,
    cgas: &[CgaInfo],
    shape: &RankPolyShape,
    cga_shape_types: &[Option<Type>],
    return_type_for_out: &Option<Type>,
    combo: &RankCombo,
    recv_lt: &syn::Lifetime,
) -> syn::Path {
    let mut new_path = path.clone();
    if let Some(last_seg) = new_path.segments.last_mut() {
        if let PathArguments::AngleBracketed(ab) = &mut last_seg.arguments {
            let mut new_args: syn::punctuated::Punctuated<GenericArgument, syn::Token![,]> =
                syn::punctuated::Punctuated::new();
            for arg in ab.args.iter() {
                // syn parses bare-ident args (e.g. `BroadcastScalar<E, D>`) as
                // `GenericArgument::Type`, not `Const`, even when the binding
                // position is a const. Match both forms — the CGA's role is
                // determined by name, not by syn's parse classification.
                let cga_idx: Option<usize> = match arg {
                    GenericArgument::Const(Expr::Path(ExprPath { path, .. })) => path
                        .get_ident()
                        .and_then(|id| cgas.iter().position(|c| c.cga_ident == *id)),
                    GenericArgument::Type(Type::Path(p)) => p
                        .path
                        .get_ident()
                        .and_then(|id| cgas.iter().position(|c| c.cga_ident == *id)),
                    _ => None,
                };
                if let Some(idx) = cga_idx {
                    let substitute: Option<&Type> = match shape.roles[idx] {
                        CgaRole::ShapeBound { .. } => {
                            cga_shape_types.get(idx).and_then(|t| t.as_ref())
                        }
                        CgaRole::Free => return_type_for_out.as_ref(),
                    };
                    if let Some(src) = substitute {
                        let concrete = rewrite_ty_for_rank(src, combo, cgas);
                        let concrete = bind_anon_lifetimes_to(&concrete, recv_lt);
                        new_args.push(GenericArgument::Type(concrete));
                        continue;
                    }
                }
                new_args.push(arg.clone());
            }
            ab.args = new_args;
        }
    }
    new_path
}

/// Rewrite an impl method's signature for a rank-instance impl: arg and return
/// types become concrete rank-instance forms; body is replaced with `unreachable!()`.
/// The concrete return type works for both case-3b (matches the trait's
/// `Self::Out` after the impl's `type Out = …` binding) and case-3c
/// (matches the trait's `Out` after the trait reference's binding).
fn rewrite_impl_method_body_for_rank(
    impl_fn: &ImplItemFn,
    cgas: &[CgaInfo],
    combo: &RankCombo,
    recv_lt: &syn::Lifetime,
) -> TokenStream2 {
    let mut new_sig = impl_fn.sig.clone();
    for arg in new_sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = arg {
            let new_ty = rewrite_ty_for_rank(&pt.ty, combo, cgas);
            let new_ty = bind_anon_lifetimes_to(&new_ty, recv_lt);
            *pt.ty = new_ty;
        }
    }
    if let ReturnType::Type(_, ret) = &mut new_sig.output {
        let new_ret = rewrite_ty_for_rank(ret, combo, cgas);
        let new_ret = bind_anon_lifetimes_to(&new_ret, recv_lt);
        **ret = new_ret;
    }
    let muted_args: Vec<TokenStream2> = new_sig
        .inputs
        .iter()
        .filter_map(|a| match a {
            FnArg::Typed(pt) => match &*pt.pat {
                Pat::Ident(pi) => {
                    let n = &pi.ident;
                    Some(quote! { #n })
                }
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
        .collect();
    let attrs = filter_cuda_tile_attrs(&impl_fn.attrs);
    quote! {
        #(#attrs)*
        #new_sig {
            let _ = (#(#muted_args),*);
            ::std::unreachable!()
        }
    }
}

/// Walk a type and replace any anonymous lifetime (`'_`) with `lt`. Used after
/// `rewrite_ty_for_rank` to anchor lifetime-bearing variadic types
/// (Shape_R, Array_R, …) to the impl's lifetime parameter — required because
/// `'_` is not allowed in trait-impl positions.
fn bind_anon_lifetimes_to(ty: &Type, lt: &syn::Lifetime) -> Type {
    match ty {
        Type::Path(tp) => {
            let mut new_tp = tp.clone();
            for seg in new_tp.path.segments.iter_mut() {
                if let PathArguments::AngleBracketed(ab) = &mut seg.arguments {
                    let mut new_args: syn::punctuated::Punctuated<GenericArgument, syn::Token![,]> =
                        syn::punctuated::Punctuated::new();
                    for arg in ab.args.iter() {
                        match arg {
                            GenericArgument::Lifetime(l) if l.ident == "_" => {
                                new_args.push(GenericArgument::Lifetime(lt.clone()))
                            }
                            GenericArgument::Type(t) => {
                                new_args.push(GenericArgument::Type(bind_anon_lifetimes_to(t, lt)))
                            }
                            other => new_args.push(other.clone()),
                        }
                    }
                    ab.args = new_args;
                }
            }
            Type::Path(new_tp)
        }
        Type::Reference(r) => {
            let mut new_r = r.clone();
            if r.lifetime.is_none() {
                new_r.lifetime = Some(lt.clone());
            }
            *new_r.elem = bind_anon_lifetimes_to(&r.elem, lt);
            Type::Reference(new_r)
        }
        Type::Array(a) => {
            let mut new_a = a.clone();
            *new_a.elem = bind_anon_lifetimes_to(&a.elem, lt);
            Type::Array(new_a)
        }
        Type::Tuple(t) => {
            let mut new_t = t.clone();
            new_t.elems = t
                .elems
                .iter()
                .map(|e| bind_anon_lifetimes_to(e, lt))
                .collect();
            Type::Tuple(new_t)
        }
        other => other.clone(),
    }
}

/// True if `ty` syntactically pulls in a lifetime — either through a
/// lifetime-bearing variadic base (Shape, Array, Partition, PartitionMut), a
/// reference, or an explicit lifetime arg anywhere inside it. Used by the
/// impl emitter to decide whether the synthesized `'__td_recv` lifetime is
/// load-bearing.
fn type_uses_lifetime(ty: &Type) -> bool {
    match ty {
        Type::Path(tp) => {
            for seg in tp.path.segments.iter() {
                if base_has_lifetime(&seg.ident) {
                    return true;
                }
                if let PathArguments::AngleBracketed(ab) = &seg.arguments {
                    for arg in ab.args.iter() {
                        match arg {
                            GenericArgument::Lifetime(_) => return true,
                            GenericArgument::Type(t) if type_uses_lifetime(t) => return true,
                            _ => {}
                        }
                    }
                }
            }
            false
        }
        Type::Reference(_) => true,
        Type::Array(a) => type_uses_lifetime(&a.elem),
        Type::Tuple(t) => t.elems.iter().any(type_uses_lifetime),
        _ => false,
    }
}

/// Drop `#[cuda_tile::*]` attrs (they've served their purpose at this point).
fn filter_cuda_tile_attrs(attrs: &[syn::Attribute]) -> Vec<syn::Attribute> {
    attrs
        .iter()
        .filter(|a| {
            let path = a.path();
            path.segments.first().is_none_or(|s| s.ident != "cuda_tile")
        })
        .cloned()
        .collect()
}
