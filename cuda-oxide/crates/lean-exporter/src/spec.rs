/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reads the contract and verification attributes attached to `mir.func`.
//!
//! Before we look at any instructions inside the kernel body, we check what the
//! user wrote on the function signature and unpack those attributes into a
//! [`KernelVerificationSpec`]:
//!
//! ```text
//!   Attributes on `mir.func @vecadd`             `KernelVerificationSpec`
//!   ────────────────────────────────────────     ─────────────────────────────────
//!   reqntid_x = 256                          ──► block_size: 256
//!   contract_requires_len_ge = "0>=2,1>=2"   ──► requires_len_ge: {(0, 2), (1, 2)}
//!   verify_no_oob_idx = "0,1,2"              ──► no_oob_idx: [0, 1, 2]
//!   verify_disjoint_writes = "2"             ──► disjoint_writes: [2]
//!   verify_writes_all_elems = "2"            ──► writes_all_elems: [2]
//! ```

use std::collections::BTreeSet;

use dialect_mir::types::{MirDisjointSliceType, MirSliceType};
use pliron::{
    attribute::AttributeDict,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    context::{Context, Ptr},
    operation::Operation,
    r#type::Typed,
};

use crate::{ExportError, reject};

/// Parsed verification and launch-contract attributes on a `mir.func`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelVerificationSpec {
    /// Launch domain (`1` for 1D, `2` for 2D).
    pub domain: u32,
    /// Static block X dimension from `reqntid_x`.
    pub block_size: u32,
    /// Static block Y dimension from `reqntid_y` (`1` for 1D kernels).
    pub block_y: u32,
    /// Pairs `(lhs, rhs)` from `requires = (lhs.len() >= rhs.len())`.
    pub requires_len_ge: BTreeSet<(usize, usize)>,
    /// Parameter indices requested in `#[verify(no_oob_idx(...))]`.
    pub no_oob_idx: Vec<usize>,
    /// Parameter indices requested in `#[verify(disjoint_writes(...))]`.
    pub disjoint_writes: Vec<usize>,
    /// Parameter indices requested in `#[verify(writes_all_elems(...))]`.
    pub writes_all_elems: Vec<usize>,
    /// Static `SharedArray` indices requested in `#[verify(no_oob_idx(...))]`.
    pub no_oob_shared: Vec<usize>,
    /// Static `SharedArray` indices requested in `#[verify(no_data_race(...))]`.
    pub no_data_race: Vec<usize>,
    /// Whether `#[verify(uniform_barriers)]` was requested.
    pub uniform_barriers: bool,
    /// Whether `#[verify(tmem_lifecycle)]` was requested for Blackwell `tcgen05` / `TMEM`.
    pub tmem_lifecycle: bool,
}

/// Returns true if `function` carries any `verify_*` obligation attribute.
pub fn has_verification_attributes(ctx: &Context, function: Ptr<Operation>) -> bool {
    let attrs = &function.deref(ctx).attributes;
    attrs.0.keys().any(|key| {
        matches!(
            key.to_string().as_str(),
            "verify_no_oob_idx"
                | "verify_disjoint_writes"
                | "verify_writes_all_elems"
                | "verify_no_oob_shared"
                | "verify_no_data_race"
                | "verify_uniform_barriers"
                | "verify_tmem_lifecycle"
        )
    })
}

/// Look up an optional [`StringAttr`] named `name` on `attrs` and return its
/// owned `String` value when present.
pub(crate) fn get_string_attr(attrs: &AttributeDict, name: &str) -> Option<String> {
    attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == name)
        .and_then(|(_, v)| v.downcast_ref::<StringAttr>())
        .map(|s| String::from(s.clone()))
}

/// Look up an optional [`IntegerAttr`] named `name` on `attrs` and convert its
/// unsigned integer value to `u32`.
pub(crate) fn get_u32_attr(attrs: &AttributeDict, name: &str) -> Option<u32> {
    attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == name)
        .and_then(|(_, v)| v.downcast_ref::<IntegerAttr>())
        .and_then(|int| u32::try_from(int.value().to_u64()).ok())
}

/// Parse a comma-separated list of parameter or shared-array indices (such as
/// `"0,1,2"`) and validate that every index is strictly below `arity`.
fn parse_param_list(raw: &str, arity: usize, attr_name: &str) -> Result<Vec<usize>, ExportError> {
    let mut indices = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let idx = part.parse::<usize>().map_err(|_| {
            reject(
                attr_name,
                format!("invalid parameter index `{part}` in `{attr_name}`"),
            )
        })?;
        if idx >= arity {
            return Err(reject(
                attr_name,
                format!("parameter index {idx} out of range for {arity} parameters"),
            ));
        }
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    Ok(indices)
}

/// Parse a comma-separated list of `<lhs>>=<rhs>` parameter length relations from
/// `contract_requires_len_ge` and validate that every index is `< arity`.
fn parse_requires_len_ge(raw: &str, arity: usize) -> Result<BTreeSet<(usize, usize)>, ExportError> {
    let mut pairs = BTreeSet::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let Some((lhs, rhs)) = item.split_once(">=") else {
            return Err(reject(
                "contract_requires_len_ge",
                format!("malformed relation `{item}`; expected `<lhs>>=<rhs>`"),
            ));
        };
        let lhs = lhs.trim().parse::<usize>().map_err(|_| {
            reject(
                "contract_requires_len_ge",
                format!("invalid lhs parameter index in `{item}`"),
            )
        })?;
        let rhs = rhs.trim().parse::<usize>().map_err(|_| {
            reject(
                "contract_requires_len_ge",
                format!("invalid rhs parameter index in `{item}`"),
            )
        })?;
        if lhs >= arity || rhs >= arity {
            return Err(reject(
                "contract_requires_len_ge",
                format!("parameter index in `{item}` out of range for {arity} parameters"),
            ));
        }
        pairs.insert((lhs, rhs));
    }
    Ok(pairs)
}

/// Parse and validate the `#[launch_contract]` and `#[verify(...)]` attributes
/// and parameter types on `function`.
pub(crate) fn parse_kernel_verification_spec(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<(usize, KernelVerificationSpec), ExportError> {
    let func = function.deref(ctx);
    let attrs = &func.attributes;
    if get_string_attr(attrs, "gpu_kernel").as_deref() != Some("true") {
        return Err(reject(
            "kernel",
            "#[verify(...)] is only supported on #[kernel] entry functions",
        ));
    }

    let domain = get_u32_attr(attrs, "launch_contract_domain").ok_or_else(|| {
        reject(
            "kernel contract",
            "#[verify(...)] requires #[launch_contract(domain = ..., block = (...))]",
        )
    })?;
    if !matches!(domain, 1 | 2) {
        return Err(reject(
            "kernel contract",
            format!(
                "only 1D and 2D launch contracts (`domain = 1` or `2`) are supported, found domain = {domain}"
            ),
        ));
    }

    let block_x = get_u32_attr(attrs, "reqntid_x").ok_or_else(|| {
        reject(
            "kernel contract",
            "#[verify(...)] requires an exact block size in #[launch_contract(block = (x, y, 1))]",
        )
    })?;
    let block_y = get_u32_attr(attrs, "reqntid_y").unwrap_or(1);
    let block_z = get_u32_attr(attrs, "reqntid_z").unwrap_or(1);
    if domain == 1 && (block_x == 0 || block_y != 1 || block_z != 1) {
        return Err(reject(
            "kernel contract",
            format!("expected 1D block (x > 0, 1, 1), found ({block_x}, {block_y}, {block_z})"),
        ));
    }
    if domain == 2 && (block_x == 0 || block_y == 0 || block_z != 1) {
        return Err(reject(
            "kernel contract",
            format!("expected 2D block (x > 0, y > 0, 1), found ({block_x}, {block_y}, {block_z})"),
        ));
    }

    let type_attr = attrs
        .0
        .iter()
        .find(|(k, _)| k.to_string() == "mir_func_type")
        .and_then(|(_, v)| v.downcast_ref::<TypeAttr>())
        .ok_or_else(|| reject("kernel", "missing mir_func_type"))?;
    let fn_type_ref = type_attr.get_type(ctx).deref(ctx);
    let fn_type = fn_type_ref
        .downcast_ref::<FunctionType>()
        .ok_or_else(|| reject("kernel", "mir_func_type must be builtin.function"))?;

    let inputs = fn_type.arg_types();
    let arity = inputs.len();
    if arity == 0 {
        return Err(reject(
            "kernel",
            "verified kernel must have at least one parameter",
        ));
    }

    let requires_len_ge = match get_string_attr(attrs, "contract_requires_len_ge") {
        Some(raw) => parse_requires_len_ge(&raw, arity)?,
        None => BTreeSet::new(),
    };
    let no_oob_idx = match get_string_attr(attrs, "verify_no_oob_idx") {
        Some(raw) => parse_param_list(&raw, arity, "verify_no_oob_idx")?,
        None => Vec::new(),
    };
    let disjoint_writes = match get_string_attr(attrs, "verify_disjoint_writes") {
        Some(raw) => parse_param_list(&raw, arity, "verify_disjoint_writes")?,
        None => Vec::new(),
    };
    let writes_all_elems = match get_string_attr(attrs, "verify_writes_all_elems") {
        Some(raw) => parse_param_list(&raw, arity, "verify_writes_all_elems")?,
        None => Vec::new(),
    };
    let no_oob_shared = match get_string_attr(attrs, "verify_no_oob_shared") {
        Some(raw) => parse_param_list(&raw, 64, "verify_no_oob_shared")?,
        None => Vec::new(),
    };
    let no_data_race = match get_string_attr(attrs, "verify_no_data_race") {
        Some(raw) => parse_param_list(&raw, 64, "verify_no_data_race")?,
        None => Vec::new(),
    };
    let uniform_barriers =
        get_string_attr(attrs, "verify_uniform_barriers").as_deref() == Some("true");
    let tmem_lifecycle = get_string_attr(attrs, "verify_tmem_lifecycle").as_deref() == Some("true");

    let is_phased_shared =
        !no_oob_shared.is_empty() || !no_data_race.is_empty() || uniform_barriers || tmem_lifecycle;
    let mut writable_params = BTreeSet::new();
    for (idx, ty) in inputs.iter().enumerate() {
        let ty_ref = ty.deref(ctx);
        if let Some(slice) = ty_ref.downcast_ref::<MirSliceType>() {
            if slice.is_mutable() {
                writable_params.insert(idx);
            }
        } else if ty_ref.downcast_ref::<MirDisjointSliceType>().is_some() {
            writable_params.insert(idx);
        } else if domain == 1 && !is_phased_shared {
            return Err(reject(
                "kernel signature",
                format!("parameter {idx} must be a slice or DisjointSlice in the v0 kernel subset"),
            ));
        }
    }

    for &param in disjoint_writes.iter().chain(writes_all_elems.iter()) {
        if !writable_params.contains(&param) {
            return Err(reject(
                "kernel contract",
                format!("parameter {param} is read-only and cannot satisfy a write obligation"),
            ));
        }
    }

    Ok((
        arity,
        KernelVerificationSpec {
            domain,
            block_size: block_x,
            block_y,
            requires_len_ge,
            no_oob_idx,
            disjoint_writes,
            writes_all_elems,
            no_oob_shared,
            no_data_race,
            uniform_barriers,
            tmem_lifecycle,
        },
    ))
}
