/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic proof facts for the cuTile DSL.
//!
//! This pass-side module owns the small predicate vocabulary accepted by
//! `#[cutile::entry(preconditions = ...)]` and exposes narrow proof queries to
//! IR emission. Codegen should consume these results, not parse or reason about
//! source-level predicates directly.

use crate::ast::SourceLocation;
use crate::compiler::shared_types::EntryAttrs;
use crate::error::{JITError, SpannedJITError};
use syn::{BinOp, Expr, ExprBinary, ExprCall};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MetadataExpr {
    Dim { tensor: String, axis: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MetadataFact {
    /// `dim(a, i) == dim(b, j)` — two axis extents are equal.
    DimEq {
        lhs: MetadataExpr,
        rhs: MetadataExpr,
    },
    /// `dim(t, k) % divisor == 0` — an axis extent is a multiple of a constant.
    /// The divisor must be a positive integer literal: it is embedded in the
    /// generated launcher's verification, which runs before any const-generic
    /// value exists.
    DimDivisible {
        tensor: String,
        axis: usize,
        divisor: i64,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProofResults {
    pub(crate) metadata_facts: Vec<MetadataFact>,
}

impl ProofResults {
    pub(crate) fn analyze_entry_attrs(entry_attrs: &EntryAttrs) -> Result<Self, JITError> {
        let Some(expr) = entry_attrs.get_entry_arg_expr("preconditions") else {
            return Ok(Self::default());
        };
        let mut metadata_facts = Vec::new();
        for expr in precondition_entries(expr)? {
            metadata_facts.push(parse_metadata_fact(expr)?);
        }
        Ok(Self { metadata_facts })
    }
}

fn precondition_entries(expr: &Expr) -> Result<Vec<&Expr>, JITError> {
    match expr {
        Expr::Tuple(tuple) => Ok(tuple.elems.iter().collect()),
        Expr::Paren(paren) => Ok(vec![paren.expr.as_ref()]),
        Expr::Binary(_) => Ok(vec![expr]),
        Expr::Call(call) => {
            if call_name(call).as_deref() == Some("same_partition_axis") {
                return SourceLocation::unknown().jit_error_result(
                    "`same_partition_axis` preconditions have been replaced by `dim(lhs, axis) == dim(rhs, axis)`",
                );
            }
            SourceLocation::unknown()
                .jit_error_result("`preconditions` entries must be metadata equalities")
        }
        _ => SourceLocation::unknown()
            .jit_error_result("`preconditions` must be an equality or tuple of equalities"),
    }
}

fn parse_metadata_fact(expr: &Expr) -> Result<MetadataFact, JITError> {
    let Expr::Binary(binary) = expr else {
        return SourceLocation::unknown()
            .jit_error_result("each `preconditions` entry must be a metadata equality");
    };
    parse_metadata_equality(binary)
}

fn parse_metadata_equality(binary: &ExprBinary) -> Result<MetadataFact, JITError> {
    if !matches!(binary.op, BinOp::Eq(_)) {
        return SourceLocation::unknown().jit_error_result("precondition predicates must use `==`");
    }
    // Divisibility spelling: `dim(t, k) % divisor == 0`.
    if let Expr::Binary(rem) = binary.left.as_ref() {
        if matches!(rem.op, BinOp::Rem(_)) {
            if !is_zero_literal(&binary.right) {
                return SourceLocation::unknown().jit_error_result(
                    "a `%` precondition must compare against literal `0`: `dim(t, k) % d == 0`",
                );
            }
            let MetadataExpr::Dim { tensor, axis } = parse_metadata_expr(&rem.left)?;
            let divisor = parse_divisor_literal(&rem.right)?;
            return Ok(MetadataFact::DimDivisible {
                tensor,
                axis,
                divisor,
            });
        }
    }
    Ok(MetadataFact::DimEq {
        lhs: parse_metadata_expr(&binary.left)?,
        rhs: parse_metadata_expr(&binary.right)?,
    })
}

fn is_zero_literal(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(i) => i.base10_parse::<i64>().map(|v| v == 0).unwrap_or(false),
            _ => false,
        },
        _ => false,
    }
}

fn parse_divisor_literal(expr: &Expr) -> Result<i64, JITError> {
    let Expr::Lit(lit) = expr else {
        return SourceLocation::unknown().jit_error_result(
            "a `%` precondition divisor must be a positive integer literal (a const generic \
             cannot be verified by the launcher, which runs before monomorphization)",
        );
    };
    let syn::Lit::Int(int_lit) = &lit.lit else {
        return SourceLocation::unknown()
            .jit_error_result("a `%` precondition divisor must be a positive integer literal");
    };
    let divisor = int_lit.base10_parse::<i64>().map_err(|err| {
        SourceLocation::unknown().jit_error(&format!("invalid `%` precondition divisor: {err}"))
    })?;
    if divisor < 1 {
        return SourceLocation::unknown().jit_error_result(&format!(
            "a `%` precondition divisor must be >= 1, got {divisor}"
        ));
    }
    Ok(divisor)
}

fn parse_metadata_expr(expr: &Expr) -> Result<MetadataExpr, JITError> {
    let Expr::Call(call) = expr else {
        return SourceLocation::unknown().jit_error_result(
            "precondition metadata expressions must be calls like `dim(tensor, axis)`",
        );
    };
    let Some(name) = call_name(call) else {
        return SourceLocation::unknown()
            .jit_error_result("precondition metadata expressions must use a function path");
    };
    match name.as_str() {
        "dim" => parse_dim_expr(call),
        other => SourceLocation::unknown().jit_error_result(&format!(
            "unsupported precondition metadata expression `{other}`; expected `dim`"
        )),
    }
}

fn parse_dim_expr(call: &ExprCall) -> Result<MetadataExpr, JITError> {
    if call.args.len() != 2 {
        return SourceLocation::unknown().jit_error_result(&format!(
            "`dim` expects 2 arguments, got {}",
            call.args.len()
        ));
    }
    Ok(MetadataExpr::Dim {
        tensor: parse_tensor_arg(&call.args[0])?,
        axis: parse_axis_arg(&call.args[1])?,
    })
}

fn call_name(call: &ExprCall) -> Option<String> {
    let Expr::Path(func_path) = call.func.as_ref() else {
        return None;
    };
    func_path
        .path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn parse_tensor_arg(expr: &Expr) -> Result<String, JITError> {
    let Expr::Path(path) = expr else {
        return SourceLocation::unknown()
            .jit_error_result("precondition tensor arguments must be parameter names");
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return SourceLocation::unknown()
            .jit_error_result("precondition tensor arguments must be simple parameter names");
    }
    Ok(path.path.segments[0].ident.to_string())
}

fn parse_axis_arg(expr: &Expr) -> Result<usize, JITError> {
    let axis = crate::types::parse_signed_literal_as_i32(expr);
    if axis < 0 {
        return SourceLocation::unknown().jit_error_result(&format!(
            "precondition axis must be non-negative, got {axis}"
        ));
    }
    Ok(axis as usize)
}

impl std::fmt::Display for MetadataExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataExpr::Dim { tensor, axis } => write!(f, "dim({tensor}, {axis})"),
        }
    }
}
