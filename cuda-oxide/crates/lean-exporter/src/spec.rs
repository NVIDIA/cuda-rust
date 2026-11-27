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
//!   contract_requires = "s0>=s2,s1>=s2"      ──► requires: [s0 >= s2, s1 >= s2]
//!   verify_no_oob_idx = "0,1,2"              ──► no_oob_idx: [0, 1, 2]
//!   verify_disjoint_writes = "2"             ──► disjoint_writes: [2]
//!   verify_writes_all_elems = "2"            ──► writes_all_elems: [2]
//! ```

use std::{collections::BTreeSet, fmt};

use dialect_mir::types::{MirDisjointSliceType, MirPtrType, MirSliceType, MirStructType};
use pliron::{
    attribute::AttributeDict,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType},
    },
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

use crate::{ExportError, reject};

/// Comparison operator in a host-checked `#[launch_contract(requires = (...))]` relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReqCmp {
    /// `lhs >= rhs`.
    Ge,
    /// `lhs > rhs`.
    Gt,
    /// `lhs <= rhs`.
    Le,
    /// `lhs < rhs`.
    Lt,
    /// `lhs == rhs`.
    Eq,
    /// `lhs != rhs`.
    Ne,
}

impl ReqCmp {
    /// Return the Lean `CudaOxide.Mir.Program.ReqCmp` constructor literal for `self`.
    pub fn lean_ctor(self) -> &'static str {
        match self {
            Self::Ge => ".ge",
            Self::Gt => ".gt",
            Self::Le => ".le",
            Self::Lt => ".lt",
            Self::Eq => ".eq",
            Self::Ne => ".ne",
        }
    }

    /// Return the Rust source symbol (`>=`, `>`, `<=`, `<`, `==`, `!=`) for `self`.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Ge => ">=",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Lt => "<",
            Self::Eq => "==",
            Self::Ne => "!=",
        }
    }
}

/// Arithmetic expression node in a host-checked `#[launch_contract(requires = (...))]` relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReqExpr {
    /// Unsigned 64-bit integer constant literal.
    Const(u64),
    /// Unsigned integer scalar parameter (`param[idx]`).
    Scalar(usize),
    /// Slice or `DisjointSlice` element count (`param[idx].len()`).
    SliceLen(usize),
    /// Checked 64-bit unsigned addition (`lhs + rhs`).
    Add(Box<ReqExpr>, Box<ReqExpr>),
    /// Checked 64-bit unsigned subtraction (`lhs - rhs`).
    Sub(Box<ReqExpr>, Box<ReqExpr>),
    /// Checked 64-bit unsigned multiplication (`lhs * rhs`).
    Mul(Box<ReqExpr>, Box<ReqExpr>),
}

impl ReqExpr {
    /// Return true when `self` references `param[target].len()`.
    pub fn mentions_slice(&self, target: usize) -> bool {
        match self {
            Self::Const(_) | Self::Scalar(_) => false,
            Self::SliceLen(idx) => *idx == target,
            Self::Add(l, r) | Self::Sub(l, r) | Self::Mul(l, r) => {
                l.mentions_slice(target) || r.mentions_slice(target)
            }
        }
    }

    /// Operator precedence (`2` for atoms, `1` for `*`, `0` for `+` / `-`) for display formatting.
    fn prec(&self) -> u8 {
        match self {
            Self::Const(_) | Self::Scalar(_) | Self::SliceLen(_) => 2,
            Self::Mul(_, _) => 1,
            Self::Add(_, _) | Self::Sub(_, _) => 0,
        }
    }
}

impl fmt::Display for ReqExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => write!(f, "{v}"),
            Self::Scalar(p) => write!(f, "param[{p}]"),
            Self::SliceLen(p) => write!(f, "param[{p}].len()"),
            Self::Add(l, r) | Self::Sub(l, r) => {
                let op = if matches!(self, Self::Add(_, _)) {
                    '+'
                } else {
                    '-'
                };
                if r.prec() == 0 {
                    write!(f, "{l} {op} ({r})")
                } else {
                    write!(f, "{l} {op} {r}")
                }
            }
            Self::Mul(l, r) => {
                if l.prec() < 1 {
                    write!(f, "({l}) * ")?;
                } else {
                    write!(f, "{l} * ")?;
                }
                if r.prec() < 1 {
                    write!(f, "({r})")
                } else {
                    write!(f, "{r}")
                }
            }
        }
    }
}

/// One host-checked `lhs <cmp> rhs` precondition from `#[launch_contract(requires = (...))]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReqRel {
    /// Comparison operator relating `lhs` and `rhs`.
    pub cmp: ReqCmp,
    /// Left-hand arithmetic expression.
    pub lhs: ReqExpr,
    /// Right-hand arithmetic expression.
    pub rhs: ReqExpr,
}

impl ReqRel {
    /// Return true when either side of `self` references `param[target].len()`.
    pub fn mentions_slice(&self, target: usize) -> bool {
        self.lhs.mentions_slice(target) || self.rhs.mentions_slice(target)
    }
}

impl fmt::Display for ReqRel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.lhs, self.cmp.symbol(), self.rhs)
    }
}

/// Parsed verification and launch-contract attributes on a `mir.func`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelVerificationSpec {
    /// Launch domain (`1` for 1D, `2` for 2D).
    pub domain: u32,
    /// Static block X dimension from `reqntid_x`.
    pub block_size: u32,
    /// Static block Y dimension from `reqntid_y` (`1` for 1D kernels).
    pub block_y: u32,
    /// Static CTA cluster X dimension from `cluster_dim_x` (`1` when non-clustered).
    pub cluster_dim_x: u32,
    /// True when `u32_coordinates` was set on `#[launch_contract]` (`grid * block <= u32::MAX`).
    pub u32_coordinates: bool,
    /// Indices of `&mut [T]` (`MirSliceType` with `mutable: true`) parameters.
    pub mutable_slice_params: BTreeSet<usize>,
    /// Host-checked preconditions from `#[launch_contract(requires = (...))]`.
    pub requires: Vec<ReqRel>,
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

/// Return true when `ty` is a slice (`&[T]`, `&mut [T]`) or `DisjointSlice<T>` parameter type.
fn is_slice_param_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    ty_ref.downcast_ref::<MirSliceType>().is_some()
        || ty_ref.downcast_ref::<MirDisjointSliceType>().is_some()
}

/// Return true when `ty` is an unsigned integer scalar (`u8`/`u16`/`u32`/`u64`) or a single-field
/// wrapper struct around one (`Uniform<u32>`).
fn is_unsigned_scalar_param_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return int_ty.width() > 1 && int_ty.is_unsigned();
    }
    if let Some(st) = ty_ref.downcast_ref::<MirStructType>()
        && st.field_types.len() == 1
        && let Some(int_ty) = st.field_types[0].deref(ctx).downcast_ref::<IntegerType>()
    {
        return int_ty.width() > 1 && int_ty.is_unsigned();
    }
    false
}

/// Cursor-based recursive-descent parser for a single `contract_requires` expression.
struct ReqExprParser<'a> {
    /// Remaining ASCII bytes to parse.
    rest: &'a [u8],
    /// Parameter types of the enclosing `mir.func`.
    inputs: &'a [TypeHandle],
    /// Pliron IR context for parameter-type checks.
    ctx: &'a Context,
    /// Full relation text for error messages.
    rel_text: &'a str,
}

impl<'a> ReqExprParser<'a> {
    /// Skip leading ASCII whitespace.
    fn skip_ws(&mut self) {
        while let Some(b) = self.rest.first() {
            if b.is_ascii_whitespace() {
                self.rest = &self.rest[1..];
            } else {
                break;
            }
        }
    }

    /// Parse a sequence of decimal ASCII digits into `u64`.
    fn parse_u64_digits(&mut self) -> Result<u64, ExportError> {
        let len = self.rest.iter().take_while(|b| b.is_ascii_digit()).count();
        if len == 0 {
            return Err(reject(
                "contract_requires",
                format!("expected integer in `{}`", self.rel_text),
            ));
        }
        let s = std::str::from_utf8(&self.rest[..len]).unwrap();
        self.rest = &self.rest[len..];
        s.parse::<u64>().map_err(|_| {
            reject(
                "contract_requires",
                format!(
                    "integer literal `{s}` out of u64 range in `{}`",
                    self.rel_text
                ),
            )
        })
    }

    /// Parse a primary atom (`(expr)`, `s<idx>`, `p<idx>`, or `<u64>`).
    fn parse_atom(&mut self) -> Result<ReqExpr, ExportError> {
        self.skip_ws();
        let Some(&first) = self.rest.first() else {
            return Err(reject(
                "contract_requires",
                format!("unexpected end of expression in `{}`", self.rel_text),
            ));
        };
        match first {
            b'(' => {
                self.rest = &self.rest[1..];
                let inner = self.parse_add_sub()?;
                self.skip_ws();
                if self.rest.first() != Some(&b')') {
                    return Err(reject(
                        "contract_requires",
                        format!("unclosed `(` in `{}`", self.rel_text),
                    ));
                }
                self.rest = &self.rest[1..];
                Ok(inner)
            }
            b's' => {
                self.rest = &self.rest[1..];
                let idx = usize::try_from(self.parse_u64_digits()?).map_err(|_| {
                    reject(
                        "contract_requires",
                        format!("slice parameter index overflow in `{}`", self.rel_text),
                    )
                })?;
                let Some(&ty) = self.inputs.get(idx) else {
                    return Err(reject(
                        "contract_requires",
                        format!(
                            "parameter index {idx} in `{}` out of range for {} parameters",
                            self.rel_text,
                            self.inputs.len()
                        ),
                    ));
                };
                if !is_slice_param_type(self.ctx, ty) {
                    return Err(reject(
                        "contract_requires",
                        format!(
                            "parameter {idx} in `{}` is not a slice or DisjointSlice",
                            self.rel_text
                        ),
                    ));
                }
                Ok(ReqExpr::SliceLen(idx))
            }
            b'p' => {
                self.rest = &self.rest[1..];
                let idx = usize::try_from(self.parse_u64_digits()?).map_err(|_| {
                    reject(
                        "contract_requires",
                        format!("scalar parameter index overflow in `{}`", self.rel_text),
                    )
                })?;
                let Some(&ty) = self.inputs.get(idx) else {
                    return Err(reject(
                        "contract_requires",
                        format!(
                            "parameter index {idx} in `{}` out of range for {} parameters",
                            self.rel_text,
                            self.inputs.len()
                        ),
                    ));
                };
                if !is_unsigned_scalar_param_type(self.ctx, ty) {
                    return Err(reject(
                        "contract_requires",
                        format!(
                            "parameter {idx} in `{}` is not an unsigned integer scalar",
                            self.rel_text
                        ),
                    ));
                }
                Ok(ReqExpr::Scalar(idx))
            }
            b'0'..=b'9' => Ok(ReqExpr::Const(self.parse_u64_digits()?)),
            _ => Err(reject(
                "contract_requires",
                format!("unexpected token in `{}`", self.rel_text),
            )),
        }
    }

    /// Parse left-associative `*` chains.
    fn parse_mul(&mut self) -> Result<ReqExpr, ExportError> {
        let mut lhs = self.parse_atom()?;
        loop {
            self.skip_ws();
            if self.rest.first() == Some(&b'*') {
                self.rest = &self.rest[1..];
                let rhs = self.parse_atom()?;
                lhs = ReqExpr::Mul(Box::new(lhs), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    /// Parse left-associative `+` and `-` chains.
    fn parse_add_sub(&mut self) -> Result<ReqExpr, ExportError> {
        let mut lhs = self.parse_mul()?;
        loop {
            self.skip_ws();
            match self.rest.first() {
                Some(&b'+') => {
                    self.rest = &self.rest[1..];
                    let rhs = self.parse_mul()?;
                    lhs = ReqExpr::Add(Box::new(lhs), Box::new(rhs));
                }
                Some(&b'-') => {
                    self.rest = &self.rest[1..];
                    let rhs = self.parse_mul()?;
                    lhs = ReqExpr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }
}

/// Parse one `contract_requires` expression string and ensure all input bytes are consumed.
fn parse_req_expr(
    src: &str,
    rel_text: &str,
    inputs: &[TypeHandle],
    ctx: &Context,
) -> Result<ReqExpr, ExportError> {
    let mut parser = ReqExprParser {
        rest: src.as_bytes(),
        inputs,
        ctx,
        rel_text,
    };
    let expr = parser.parse_add_sub()?;
    parser.skip_ws();
    if !parser.rest.is_empty() {
        return Err(reject(
            "contract_requires",
            format!("trailing tokens in `{rel_text}`"),
        ));
    }
    Ok(expr)
}

/// Split one `contract_requires` item into `(lhs_str, ReqCmp, rhs_str)`.
fn split_req_cmp(item: &str) -> Result<(&str, ReqCmp, &str), ExportError> {
    let bytes = item.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if i + 1 < bytes.len() {
            let two = &bytes[i..i + 2];
            let cmp_opt = match two {
                b">=" => Some(ReqCmp::Ge),
                b"<=" => Some(ReqCmp::Le),
                b"==" => Some(ReqCmp::Eq),
                b"!=" => Some(ReqCmp::Ne),
                _ => None,
            };
            if let Some(cmp) = cmp_opt {
                return Ok((&item[..i], cmp, &item[i + 2..]));
            }
        }
        match bytes[i] {
            b'>' => return Ok((&item[..i], ReqCmp::Gt, &item[i + 1..])),
            b'<' => return Ok((&item[..i], ReqCmp::Lt, &item[i + 1..])),
            _ => i += 1,
        }
    }
    Err(reject(
        "contract_requires",
        format!("malformed relation `{item}`; expected comparison operator"),
    ))
}

/// Parse a comma-separated `contract_requires` attribute into validated [`ReqRel`] items.
fn parse_contract_requires(
    raw: &str,
    inputs: &[TypeHandle],
    ctx: &Context,
    out: &mut Vec<ReqRel>,
) -> Result<(), ExportError> {
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (lhs_str, cmp, rhs_str) = split_req_cmp(item)?;
        let lhs = parse_req_expr(lhs_str, item, inputs, ctx)?;
        let rhs = parse_req_expr(rhs_str, item, inputs, ctx)?;
        let rel = ReqRel { cmp, lhs, rhs };
        if !out.contains(&rel) {
            out.push(rel);
        }
    }
    Ok(())
}

/// Parse a legacy comma-separated `<lhs>>=<rhs>` parameter length attribute (`contract_requires_len_ge`).
fn parse_legacy_requires_len_ge(
    raw: &str,
    inputs: &[TypeHandle],
    ctx: &Context,
    out: &mut Vec<ReqRel>,
) -> Result<(), ExportError> {
    let arity = inputs.len();
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
        if !is_slice_param_type(ctx, inputs[lhs]) || !is_slice_param_type(ctx, inputs[rhs]) {
            return Err(reject(
                "contract_requires_len_ge",
                format!("parameter in `{item}` is not a slice or DisjointSlice"),
            ));
        }
        let rel = ReqRel {
            cmp: ReqCmp::Ge,
            lhs: ReqExpr::SliceLen(lhs),
            rhs: ReqExpr::SliceLen(rhs),
        };
        if !out.contains(&rel) {
            out.push(rel);
        }
    }
    Ok(())
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
    let cluster_dim_x = get_u32_attr(attrs, "cluster_dim_x").unwrap_or(1);
    let cluster_dim_y = get_u32_attr(attrs, "cluster_dim_y").unwrap_or(1);
    let cluster_dim_z = get_u32_attr(attrs, "cluster_dim_z").unwrap_or(1);
    if cluster_dim_x == 0 || cluster_dim_x > 16 || cluster_dim_y != 1 || cluster_dim_z != 1 {
        return Err(reject(
            "kernel contract",
            format!(
                "verified cluster launches require 1D X cluster (1..=16, 1, 1), found ({cluster_dim_x}, {cluster_dim_y}, {cluster_dim_z})"
            ),
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

    let mut requires = Vec::new();
    if let Some(raw) = get_string_attr(attrs, "contract_requires") {
        parse_contract_requires(&raw, &inputs, ctx, &mut requires)?;
    }
    if let Some(raw) = get_string_attr(attrs, "contract_requires_len_ge") {
        parse_legacy_requires_len_ge(&raw, &inputs, ctx, &mut requires)?;
    }
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
    if !writes_all_elems.is_empty() && domain != 1 {
        return Err(reject(
            "kernel contract",
            format!(
                "`writes_all_elems` requires a 1D launch contract (`domain = 1`), found domain = {domain}"
            ),
        ));
    }
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
    if !no_data_race.is_empty() && !uniform_barriers {
        // `MirNoSharedRace` orders accesses by per-thread `syncThreads` counts. Those counts
        // only line up across threads when `UniformBarriers` is also proved.
        return Err(reject(
            "kernel contract",
            "`no_data_race` on shared memory requires `uniform_barriers`",
        ));
    }

    let u32_coordinates =
        get_string_attr(attrs, "launch_contract_u32_coords").as_deref() == Some("true");
    let is_phased_shared =
        !no_oob_shared.is_empty() || !no_data_race.is_empty() || uniform_barriers || tmem_lifecycle;
    let mut writable_params = BTreeSet::new();
    let mut mutable_slice_params = BTreeSet::new();
    for (idx, ty) in inputs.iter().enumerate() {
        let ty_ref = ty.deref(ctx);
        if let Some(slice) = ty_ref.downcast_ref::<MirSliceType>() {
            if slice.is_mutable() {
                if !no_oob_idx.contains(&idx) || !disjoint_writes.contains(&idx) {
                    return Err(reject(
                        "kernel contract",
                        format!(
                            "mutable slice parameter {idx} (`&mut [T]`) requires both `no_oob_idx` and `disjoint_writes`"
                        ),
                    ));
                }
                writable_params.insert(idx);
                mutable_slice_params.insert(idx);
            }
        } else if ty_ref.downcast_ref::<MirDisjointSliceType>().is_some() {
            writable_params.insert(idx);
        } else if domain == 1 && !is_phased_shared && ty_ref.downcast_ref::<MirPtrType>().is_some()
        {
            return Err(reject(
                "kernel signature",
                format!("parameter {idx} cannot be a raw pointer in a verified 1D slice kernel"),
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
            cluster_dim_x,
            u32_coordinates,
            mutable_slice_params,
            requires,
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
