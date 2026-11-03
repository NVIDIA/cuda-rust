/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Launch-contract parsing and validation for `#[cuda_module]`,
//! including the `requires` relation pipeline.

use crate::common::{attr_path_ends_with, internal_ident};
use crate::cuda_module::model::{
    CudaModuleKernel, CudaModuleParam, CudaModuleParamMarshal, ScalarIntClass,
};
use crate::launch_attrs::{ClusterArgs, ConstU32Expr, LaunchBoundsArgs};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Expr, Ident, Token, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DynamicSharedContract {
    Exact(u32),
    Range { min_bytes: u32, max_bytes: u32 },
}

#[derive(Clone)]
pub(super) struct CudaModuleLaunchContract {
    pub(super) domain: u8,
    pub(super) u32_coordinates: bool,
    pub(super) exact_block: Option<(u32, u32, u32)>,
    pub(super) max_block_threads: Option<ConstU32Expr>,
    pub(super) dynamic_shared: DynamicSharedContract,
    pub(super) dynamic_shared_alignment: u32,
    pub(super) min_compute_capability: (u32, u32),
    /// Size requirements over the kernel's own parameters, validated
    /// against the parameter list at expansion time. Each relation becomes an
    /// overflow-safe host-side check in every checked launcher.
    pub(super) requires: Vec<Expr>,
}

/// Arguments accepted by `#[launch_contract(...)]`.
///
/// The attribute is intentionally declarative. The kernel author states the
/// launch domain and resource envelope; `#[cuda_module]` turns that statement
/// into a branded host configuration and validates it against the live
/// function/device once during preparation.
#[derive(Clone)]
pub(crate) struct LaunchContractArgs {
    pub(crate) domain: u8,
    pub(crate) u32_coordinates: bool,
    pub(crate) exact_block: Option<(u32, u32, u32)>,
    pub(crate) dynamic_shared: DynamicSharedContract,
    pub(crate) dynamic_shared_alignment: u32,
    pub(crate) min_compute_capability: (u32, u32),
    pub(crate) requires: Vec<Expr>,
}

impl Parse for LaunchContractArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut domain = None;
        let mut exact_block = None;
        let mut coordinates = None;
        let mut dynamic_shared = None;
        let mut dynamic_shared_range = None;
        let mut dynamic_shared_alignment = None;
        let mut min_compute_capability = None;
        let mut requires = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "domain" => {
                    reject_duplicate(&key, domain.is_some())?;
                    let value: syn::LitInt = input.parse()?;
                    domain = Some(value.base10_parse::<u8>()?);
                }
                "block" => {
                    reject_duplicate(&key, exact_block.is_some())?;
                    exact_block = Some(parse_u32_triplet(input, "block")?);
                }
                "coordinates" => {
                    reject_duplicate(&key, coordinates.is_some())?;
                    let width: Ident = input.parse()?;
                    if width != "u32" {
                        return Err(syn::Error::new(
                            width.span(),
                            "launch_contract coordinates currently supports only `u32`",
                        ));
                    }
                    coordinates = Some(true);
                }
                "dynamic_shared" => {
                    reject_duplicate(&key, dynamic_shared.is_some())?;
                    let value: syn::LitInt = input.parse()?;
                    dynamic_shared = Some(value.base10_parse::<u32>()?);
                }
                "dynamic_shared_range" => {
                    reject_duplicate(&key, dynamic_shared_range.is_some())?;
                    dynamic_shared_range = Some(parse_u32_pair(input, "dynamic_shared_range")?);
                }
                "dynamic_shared_alignment" => {
                    reject_duplicate(&key, dynamic_shared_alignment.is_some())?;
                    let value: syn::LitInt = input.parse()?;
                    dynamic_shared_alignment = Some(value.base10_parse::<u32>()?);
                }
                "min_compute_capability" => {
                    reject_duplicate(&key, min_compute_capability.is_some())?;
                    let (major, minor) = parse_u32_pair(input, "min_compute_capability")?;
                    min_compute_capability = Some((major, minor));
                }
                "requires" => {
                    reject_duplicate(&key, requires.is_some())?;
                    requires = Some(parse_requires_relations(input)?);
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "unknown launch_contract field; expected domain, coordinates, block, dynamic_shared, dynamic_shared_range, dynamic_shared_alignment, min_compute_capability, or requires",
                    ));
                }
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        let domain = domain.ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "launch_contract requires `domain = 1`, `2`, or `3`",
            )
        })?;
        if !(1..=3).contains(&domain) {
            return Err(syn::Error::new(
                input.span(),
                "launch_contract domain must be 1, 2, or 3",
            ));
        }
        if dynamic_shared.is_some() && dynamic_shared_range.is_some() {
            return Err(syn::Error::new(
                input.span(),
                "launch_contract accepts either `dynamic_shared` or `dynamic_shared_range`, not both",
            ));
        }
        let dynamic_shared = match (dynamic_shared, dynamic_shared_range) {
            (Some(bytes), None) => DynamicSharedContract::Exact(bytes),
            (None, Some((min_bytes, max_bytes))) if min_bytes <= max_bytes => {
                DynamicSharedContract::Range {
                    min_bytes,
                    max_bytes,
                }
            }
            (None, Some(_)) => {
                return Err(syn::Error::new(
                    input.span(),
                    "dynamic_shared_range minimum cannot exceed its maximum",
                ));
            }
            (None, None) => DynamicSharedContract::Exact(0),
            (Some(_), Some(_)) => unreachable!(),
        };
        let dynamic_shared_alignment = dynamic_shared_alignment.unwrap_or_else(|| {
            if dynamic_shared_max(dynamic_shared) == 0 {
                1
            } else {
                16
            }
        });
        if dynamic_shared_alignment == 0 || !dynamic_shared_alignment.is_power_of_two() {
            return Err(syn::Error::new(
                input.span(),
                "dynamic_shared_alignment must be a non-zero power of two",
            ));
        }
        if exact_block.is_none() {
            // A missing exact shape is valid only when #[launch_bounds]
            // supplies the compiled maximum. This is checked after all kernel
            // attributes have been collected so the two attributes remain one
            // source of truth rather than duplicate integers.
        }
        let min_compute_capability = min_compute_capability.unwrap_or((0, 0));
        Ok(Self {
            domain,
            u32_coordinates: coordinates.unwrap_or(false),
            exact_block,
            dynamic_shared,
            dynamic_shared_alignment,
            min_compute_capability,
            requires: requires.unwrap_or_default(),
        })
    }
}

/// Parses `requires = (relation, relation, ...)`: a parenthesized,
/// comma-separated, non-empty list of relation expressions. The grammar of
/// each relation is validated later, once the kernel's parameter list is
/// known (see [`validate_requires_relations`]).
fn parse_requires_relations(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    parenthesized!(content in input);
    let relations: Punctuated<Expr, Token![,]> = Punctuated::parse_terminated(&content)?;
    if relations.is_empty() {
        return Err(syn::Error::new(
            content.span(),
            "requires needs at least one relation, e.g. `requires = (a.len() >= n)`",
        ));
    }
    Ok(relations.into_iter().collect())
}

pub(crate) fn dynamic_shared_max(contract: DynamicSharedContract) -> u32 {
    match contract {
        DynamicSharedContract::Exact(bytes) => bytes,
        DynamicSharedContract::Range { max_bytes, .. } => max_bytes,
    }
}

fn reject_duplicate(key: &Ident, duplicate: bool) -> syn::Result<()> {
    if duplicate {
        Err(syn::Error::new(
            key.span(),
            format!("duplicate launch_contract field `{key}`"),
        ))
    } else {
        Ok(())
    }
}

fn parse_u32_triplet(input: ParseStream, field: &str) -> syn::Result<(u32, u32, u32)> {
    let content;
    parenthesized!(content in input);
    let values: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(&content)?;
    if values.len() != 3 {
        return Err(syn::Error::new(
            content.span(),
            format!("{field} must be a three-dimensional tuple `(x, y, z)`"),
        ));
    }
    let mut values = values.iter();
    Ok((
        values.next().unwrap().base10_parse()?,
        values.next().unwrap().base10_parse()?,
        values.next().unwrap().base10_parse()?,
    ))
}

fn parse_u32_pair(input: ParseStream, field: &str) -> syn::Result<(u32, u32)> {
    let content;
    parenthesized!(content in input);
    let values: Punctuated<syn::LitInt, Token![,]> = Punctuated::parse_terminated(&content)?;
    if values.len() != 2 {
        return Err(syn::Error::new(
            content.span(),
            format!("{field} must be a two-value tuple"),
        ));
    }
    let mut values = values.iter();
    Ok((
        values.next().unwrap().base10_parse()?,
        values.next().unwrap().base10_parse()?,
    ))
}

pub(super) fn cuda_module_cluster_dim(
    attrs: &[syn::Attribute],
) -> syn::Result<Option<(u32, u32, u32)>> {
    for attr in attrs {
        if attr_path_ends_with(attr, "cluster_launch") {
            let args = attr.parse_args::<ClusterArgs>()?;
            return Ok(Some((args.x, args.y, args.z)));
        }
    }
    Ok(None)
}

pub(super) fn cuda_module_cooperative(attrs: &[syn::Attribute]) -> syn::Result<bool> {
    for attr in attrs {
        if attr_path_ends_with(attr, "cooperative_launch") {
            if !matches!(attr.meta, syn::Meta::Path(_)) {
                return Err(syn::Error::new_spanned(
                    attr,
                    "cooperative_launch takes no arguments: use a bare #[cooperative_launch]",
                ));
            }
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn cuda_module_launch_contract(
    attrs: &[syn::Attribute],
    _fn_name: &Ident,
    params: &[CudaModuleParam],
    cluster_dim: Option<(u32, u32, u32)>,
) -> syn::Result<Option<CudaModuleLaunchContract>> {
    let contract_attrs: Vec<_> = attrs
        .iter()
        .filter(|attr| attr_path_ends_with(attr, "launch_contract"))
        .collect();
    if contract_attrs.len() > 1 {
        return Err(syn::Error::new_spanned(
            contract_attrs[1],
            "a kernel may have only one launch_contract",
        ));
    }
    let Some(attr) = contract_attrs.first() else {
        return Ok(None);
    };
    let args = attr.parse_args::<LaunchContractArgs>()?;

    if let Some(param) = params.iter().find(|param| param.mutable_slice) {
        return Err(syn::Error::new(
            param.name.span(),
            "contracted kernels cannot take `&mut [T]`; use `DisjointSlice<T, IndexSpace>` so the launch domain and per-thread write ownership are explicit",
        ));
    }
    if let Some(block) = args.exact_block {
        validate_dimensions_for_domain(block, args.domain, "block", attr.span())?;
    }
    if let Some(cluster) = cluster_dim {
        validate_dimensions_for_domain(cluster, args.domain, "cluster", attr.span())?;
    }

    let launch_bounds = cuda_module_launch_bounds(attrs)?;
    if args.exact_block.is_none() && launch_bounds.is_none() {
        return Err(syn::Error::new_spanned(
            attr,
            "launch_contract requires either an exact `block = (x, y, z)` or #[launch_bounds(max_threads)]",
        ));
    }
    let max_block_threads = launch_bounds.map(|bounds| bounds.max_threads);
    if let (Some(exact), Some(maximum)) = (
        args.exact_block,
        max_block_threads
            .as_ref()
            .and_then(|maximum| maximum.literal_value),
    ) {
        let exact_threads = u64::from(exact.0)
            .checked_mul(u64::from(exact.1))
            .and_then(|xy| xy.checked_mul(u64::from(exact.2)))
            .ok_or_else(|| {
                syn::Error::new_spanned(attr, "launch_contract block thread count overflows u64")
            })?;
        if exact_threads > u64::from(maximum) {
            return Err(syn::Error::new_spanned(
                attr,
                format!(
                    "launch_contract block {exact:?} has {exact_threads} threads, exceeding #[launch_bounds({maximum})]"
                ),
            ));
        }
    }

    let min_compute_capability = match cluster_dim {
        Some(_) if args.min_compute_capability < (9, 0) => (9, 0),
        _ => args.min_compute_capability,
    };

    validate_requires_relations(&args.requires, params)?;

    Ok(Some(CudaModuleLaunchContract {
        domain: args.domain,
        u32_coordinates: args.u32_coordinates,
        exact_block: args.exact_block,
        max_block_threads,
        dynamic_shared: args.dynamic_shared,
        dynamic_shared_alignment: args.dynamic_shared_alignment,
        min_compute_capability,
        requires: args.requires,
    }))
}

/// One-line reminder of the accepted `requires` grammar, appended to every
/// rejection so authors see what is allowed and which parameters exist.
fn requires_grammar_help(params: &[CudaModuleParam]) -> String {
    let mut names = Vec::new();
    for param in params {
        let name = &param.name;
        match param.marshal {
            // Only scalars that can actually appear in a relation are
            // offered; signed and non-integer scalars would be rejected.
            CudaModuleParamMarshal::Scalar => {
                if param.scalar_int == ScalarIntClass::Unsigned {
                    names.push(format!("`{name}`"));
                }
            }
            CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. }
            | CudaModuleParamMarshal::WritableDeviceBuffer { .. }
            | CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
                names.push(format!("`{name}.len()`"));
            }
        }
    }
    let available = if names.is_empty() {
        "(none of this kernel's parameters can appear in requires)".to_string()
    } else {
        names.join(", ")
    };
    format!(
        "each requires relation is one comparison (`>=`, `>`, `<=`, `<`, `==`, `!=`) between \
         expressions built from slice parameters as `<param>.len()`, unsigned integer scalar \
         parameters (u8/u16/u32/u64/usize), integer literals, parentheses, and `+`, `-`, `*`; \
         available operands: {available}"
    )
}

/// Validates every `requires` relation against the kernel's own parameter
/// list at expansion time, so a malformed relation is a compile error at the
/// attribute instead of a surprise at launch.
pub(crate) fn validate_requires_relations(
    relations: &[Expr],
    params: &[CudaModuleParam],
) -> syn::Result<()> {
    for relation in relations {
        let Expr::Binary(binary) = relation else {
            return Err(syn::Error::new_spanned(
                relation,
                format!(
                    "requires relation must be a comparison between two expressions; {}",
                    requires_grammar_help(params)
                ),
            ));
        };
        if !requires_comparison_op(&binary.op) {
            return Err(syn::Error::new_spanned(
                relation,
                format!(
                    "requires relation must compare with `>=`, `>`, `<=`, `<`, `==`, or `!=`; {}",
                    requires_grammar_help(params)
                ),
            ));
        }
        validate_requires_operand(&binary.left, params)?;
        validate_requires_operand(&binary.right, params)?;
    }
    Ok(())
}

fn requires_comparison_op(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::Ge(_)
            | syn::BinOp::Gt(_)
            | syn::BinOp::Le(_)
            | syn::BinOp::Lt(_)
            | syn::BinOp::Eq(_)
            | syn::BinOp::Ne(_)
    )
}

fn requires_arithmetic_op(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::Add(_) | syn::BinOp::Sub(_) | syn::BinOp::Mul(_)
    )
}

/// Validates one side of a `requires` comparison: an arithmetic expression
/// over `.len()` of slice-like parameters, unsigned integer scalar
/// parameters, and integer literals.
fn validate_requires_operand(expr: &Expr, params: &[CudaModuleParam]) -> syn::Result<()> {
    match expr {
        Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Int(int) => {
                int.base10_parse::<u64>().map_err(|_| {
                    syn::Error::new_spanned(literal, "integer literal in requires must fit in u64")
                })?;
                Ok(())
            }
            _ => Err(syn::Error::new_spanned(
                literal,
                format!(
                    "only integer literals are allowed in requires; {}",
                    requires_grammar_help(params)
                ),
            )),
        },
        Expr::Path(path) => {
            let Some(ident) = path.path.get_ident() else {
                return Err(syn::Error::new_spanned(
                    path,
                    format!(
                        "paths in requires must be bare kernel parameter names; {}",
                        requires_grammar_help(params)
                    ),
                ));
            };
            let Some(param) = params.iter().find(|param| param.name == *ident) else {
                return Err(syn::Error::new_spanned(
                    path,
                    format!(
                        "unknown identifier `{ident}` in requires: relations may only reference \
                         this kernel's parameters; {}",
                        requires_grammar_help(params)
                    ),
                ));
            };
            match param.marshal {
                CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. }
                | CudaModuleParamMarshal::WritableDeviceBuffer { .. }
                | CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => {
                    Err(syn::Error::new_spanned(
                        path,
                        format!(
                            "slice parameter `{ident}` may only appear in requires as \
                             `{ident}.len()`; {}",
                            requires_grammar_help(params)
                        ),
                    ))
                }
                CudaModuleParamMarshal::Scalar => match param.scalar_int {
                    ScalarIntClass::Unsigned => Ok(()),
                    ScalarIntClass::Signed => Err(syn::Error::new_spanned(
                        path,
                        format!(
                            "signed integer parameter `{ident}` is not supported in requires: \
                             relations are evaluated in u64 and signed widening semantics are \
                             not defined for v1; {}",
                            requires_grammar_help(params)
                        ),
                    )),
                    ScalarIntClass::Other => Err(syn::Error::new_spanned(
                        path,
                        format!(
                            "parameter `{ident}` is not an unsigned integer scalar, so it \
                             cannot appear in requires; {}",
                            requires_grammar_help(params)
                        ),
                    )),
                },
            }
        }
        Expr::MethodCall(call) => {
            if call.method != "len" {
                return Err(syn::Error::new_spanned(
                    &call.method,
                    format!(
                        "only the `.len()` method is allowed in requires; {}",
                        requires_grammar_help(params)
                    ),
                ));
            }
            if !call.args.is_empty() || call.turbofish.is_some() {
                return Err(syn::Error::new_spanned(
                    call,
                    "`.len()` in requires takes no arguments or turbofish",
                ));
            }
            let receiver_ident = match &*call.receiver {
                Expr::Path(path) => path.path.get_ident(),
                _ => None,
            };
            let Some(ident) = receiver_ident else {
                return Err(syn::Error::new_spanned(
                    &call.receiver,
                    format!(
                        "`.len()` in requires must be called directly on a slice parameter; {}",
                        requires_grammar_help(params)
                    ),
                ));
            };
            let Some(param) = params.iter().find(|param| param.name == *ident) else {
                return Err(syn::Error::new_spanned(
                    &call.receiver,
                    format!(
                        "unknown identifier `{ident}` in requires: relations may only reference \
                         this kernel's parameters; {}",
                        requires_grammar_help(params)
                    ),
                ));
            };
            match param.marshal {
                CudaModuleParamMarshal::ReadOnlyDeviceBuffer { .. }
                | CudaModuleParamMarshal::WritableDeviceBuffer { .. }
                | CudaModuleParamMarshal::RowWidthDeviceBuffer { .. } => Ok(()),
                CudaModuleParamMarshal::Scalar => Err(syn::Error::new_spanned(
                    call,
                    format!(
                        "`.len()` in requires is only available on slice or DisjointSlice \
                         parameters; `{ident}` is a scalar; {}",
                        requires_grammar_help(params)
                    ),
                )),
            }
        }
        Expr::Paren(paren) => validate_requires_operand(&paren.expr, params),
        Expr::Group(group) => validate_requires_operand(&group.expr, params),
        Expr::Binary(binary) if requires_arithmetic_op(&binary.op) => {
            validate_requires_operand(&binary.left, params)?;
            validate_requires_operand(&binary.right, params)
        }
        Expr::Binary(binary) if requires_comparison_op(&binary.op) => Err(syn::Error::new_spanned(
            binary,
            "nested comparisons are not supported in requires: each relation is exactly one \
                 comparison; split it into multiple comma-separated relations instead",
        )),
        Expr::Binary(binary) => {
            let op = &binary.op;
            Err(syn::Error::new_spanned(
                binary,
                format!(
                    "operator `{}` is not supported in requires; {}",
                    quote! { #op },
                    requires_grammar_help(params)
                ),
            ))
        }
        other => Err(syn::Error::new_spanned(
            other,
            format!(
                "unsupported expression in requires; {}",
                requires_grammar_help(params)
            ),
        )),
    }
}

/// How a generated `requires` check reads the length of a slice-like
/// parameter. Each checked launcher flavor sees slice arguments as a
/// different host type.
#[derive(Clone, Copy)]
pub(super) enum RequiresLenAccess {
    /// Sync prepared launcher: slice parameters are `&DeviceBuffer<T>` or
    /// `&mut DeviceBuffer<T>`, so `.len()` resolves to the inherent method.
    SyncBuffer,
    /// Async prepared launcher: slice parameters are `&impl KernelSliceArg`
    /// or `&mut impl KernelSliceArgMut`.
    AsyncRef,
    /// Owned async launcher: slice parameters are owned
    /// `R: KernelSliceArg(Mut)` resources.
    OwnedValue,
}

/// Renders a validated `requires` relation back to compact source text for
/// error messages, e.g. `a.len() >= m * k`.
fn render_requires_expr(expr: &Expr) -> String {
    match expr {
        Expr::Lit(literal) => {
            let lit = &literal.lit;
            quote!(#lit).to_string()
        }
        Expr::Path(path) => match path.path.get_ident() {
            Some(ident) => ident.to_string(),
            None => quote!(#path).to_string(),
        },
        Expr::MethodCall(call) => format!("{}.len()", render_requires_expr(&call.receiver)),
        Expr::Paren(paren) => format!("({})", render_requires_expr(&paren.expr)),
        Expr::Group(group) => render_requires_expr(&group.expr),
        Expr::Binary(binary) => {
            let op = &binary.op;
            format!(
                "{} {} {}",
                render_requires_expr(&binary.left),
                quote!(#op),
                render_requires_expr(&binary.right)
            )
        }
        other => quote!(#other).to_string(),
    }
}

/// Generates the host-side size checks for a contracted kernel, or
/// `None` when the contract declares no `requires` relations.
///
/// Every operand is widened to `u64` and every `+`/`-`/`*` goes through the
/// corresponding checked operation, so a relation whose arithmetic leaves the
/// `u64` range fails with `LaunchContractError::SizeRequirementOverflow` instead of wrapping. A
/// relation that evaluates to false fails with `LaunchContractError::SizeRequirementViolated`
/// carrying the relation's source text and both evaluated sides.
pub(super) fn generate_requires_checks(
    kernel: &CudaModuleKernel,
    access: RequiresLenAccess,
) -> Option<TokenStream2> {
    let contract = kernel.launch_contract.as_ref()?;
    if contract.requires.is_empty() {
        return None;
    }
    let kernel_name = kernel.fn_name.to_string();
    let lhs_binding = internal_ident("__cuda_oxide_requires_lhs");
    let rhs_binding = internal_ident("__cuda_oxide_requires_rhs");
    let checks = contract.requires.iter().map(|relation| {
        let Expr::Binary(binary) = relation else {
            unreachable!("requires relations are validated during contract construction");
        };
        let relation_text = render_requires_expr(relation);
        let op = &binary.op;
        let lhs = requires_operand_tokens(&binary.left, access, &kernel_name, &relation_text);
        let rhs = requires_operand_tokens(&binary.right, access, &kernel_name, &relation_text);
        quote! {
            {
                let #lhs_binding: u64 = #lhs;
                let #rhs_binding: u64 = #rhs;
                if !(#lhs_binding #op #rhs_binding) {
                    return ::core::result::Result::Err(
                        ::cuda_core::LaunchContractError::SizeRequirementViolated {
                            kernel: #kernel_name,
                            relation: #relation_text,
                            lhs: #lhs_binding,
                            rhs: #rhs_binding,
                        },
                    );
                }
            }
        }
    });
    Some(quote! { #(#checks)* })
}

/// Transliterates one validated side of a `requires` comparison into a `u64`
/// expression with checked arithmetic.
fn requires_operand_tokens(
    expr: &Expr,
    access: RequiresLenAccess,
    kernel_name: &str,
    relation_text: &str,
) -> TokenStream2 {
    match expr {
        Expr::Lit(literal) => {
            let syn::Lit::Int(int) = &literal.lit else {
                unreachable!("requires literals are validated during contract construction");
            };
            let value = int
                .base10_parse::<u64>()
                .expect("requires literals are validated during contract construction");
            quote! { #value }
        }
        Expr::Path(path) => {
            // Validated: a bare unsigned integer scalar parameter, so `as
            // u64` is a lossless widening.
            quote! { (#path as u64) }
        }
        Expr::MethodCall(call) => {
            let receiver = &call.receiver;
            match access {
                RequiresLenAccess::SyncBuffer => quote! { (#receiver.len() as u64) },
                RequiresLenAccess::AsyncRef => {
                    quote! { (::cuda_host::KernelSliceArg::len(#receiver) as u64) }
                }
                RequiresLenAccess::OwnedValue => {
                    quote! { (::cuda_host::KernelSliceArg::len(&#receiver) as u64) }
                }
            }
        }
        Expr::Paren(paren) => {
            requires_operand_tokens(&paren.expr, access, kernel_name, relation_text)
        }
        Expr::Group(group) => {
            requires_operand_tokens(&group.expr, access, kernel_name, relation_text)
        }
        Expr::Binary(binary) => {
            let lhs = requires_operand_tokens(&binary.left, access, kernel_name, relation_text);
            let rhs = requires_operand_tokens(&binary.right, access, kernel_name, relation_text);
            let checked = match binary.op {
                syn::BinOp::Add(_) => quote! { checked_add },
                syn::BinOp::Sub(_) => quote! { checked_sub },
                syn::BinOp::Mul(_) => quote! { checked_mul },
                _ => unreachable!("requires operators are validated during contract construction"),
            };
            quote! {
                #lhs.#checked(#rhs).ok_or(
                    ::cuda_core::LaunchContractError::SizeRequirementOverflow {
                        kernel: #kernel_name,
                        relation: #relation_text,
                    },
                )?
            }
        }
        _ => unreachable!("requires operands are validated during contract construction"),
    }
}

#[derive(Clone)]
struct ContractLaunchBounds {
    max_threads: ConstU32Expr,
}

fn cuda_module_launch_bounds(
    attrs: &[syn::Attribute],
) -> syn::Result<Option<ContractLaunchBounds>> {
    let matching: Vec<_> = attrs
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
        return Ok(None);
    };
    let args = attr.parse_args::<LaunchBoundsArgs>()?;
    Ok(Some(ContractLaunchBounds {
        max_threads: args.max_threads,
    }))
}

fn validate_dimensions_for_domain(
    dimensions: (u32, u32, u32),
    domain: u8,
    kind: &str,
    span: proc_macro2::Span,
) -> syn::Result<()> {
    if dimensions.0 == 0 || dimensions.1 == 0 || dimensions.2 == 0 {
        return Err(syn::Error::new(
            span,
            format!("launch_contract {kind} dimensions must be non-zero"),
        ));
    }
    let outside_domain = match domain {
        1 => dimensions.1 != 1 || dimensions.2 != 1,
        2 => dimensions.2 != 1,
        3 => false,
        _ => unreachable!(),
    };
    if outside_domain {
        return Err(syn::Error::new(
            span,
            format!(
                "launch_contract {kind} dimensions {dimensions:?} exceed the declared {domain}D domain"
            ),
        ));
    }
    Ok(())
}

impl CudaModuleLaunchContract {
    pub(super) fn cluster_tokens(
        &self,
        cluster_dim: Option<(u32, u32, u32)>,
    ) -> Option<TokenStream2> {
        cluster_dim.map(|(x, y, z)| quote! { .with_cluster((#x, #y, #z)) })
    }
}
