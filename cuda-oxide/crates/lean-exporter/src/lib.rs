/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![warn(missing_docs, clippy::missing_docs_in_private_items)]

//! Transcribes `PostMem2Reg` `dialect-mir` functions into Lean 4 (`MirFunc` +
//! `ControlFlowCfg`, or `Scalar.Expr` for standalone integer functions) so
//! Lean 4 proves kernel safety before PTX lowering.
//!
//! ```text
//!   mir.func (`PostMem2Reg` SSA)
//!     │
//!     ├─► 1. `spec.rs`               Read signature, `#[launch_contract]`, & `#[verify(...)]`
//!     ├─► 2. `translate/scalar.rs`   Single-block `BitVec` AST (`CudaOxide.Mir.Scalar`)
//!     ├─► 2. `translate/program.rs`  1-to-1 `MirFunc` + `ControlFlowCfg` (`CudaOxide.Mir.Program`)
//!     ├─► 3. `codegen.rs`            Render `.lean` definitions & `theorem` obligations
//!     └─► 4. `runner.rs`             Run `lake env lean`, reject `sorryAx`, erase proven `mir.assert`s
//! ```

pub mod codegen;
pub mod runner;
pub mod spec;
pub mod ssa;
pub mod translate;

pub use runner::verify_module_kernels;

use std::collections::HashSet;

use combine::{
    Parser, attempt,
    parser::char::{spaces, string},
    satisfy, skip_many,
};
use dialect_mir::ops::MirFuncOp;
use pliron::{
    attribute::AttributeDict,
    builtin::attributes::{GivenNamesAttr, IdentifierAttr},
    context::{Context, Ptr},
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::Operation,
    parsable::parse_from_str,
};

/// Unsupported and malformed input is rejected before any Lean is returned.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// Textual MIR failed to parse.
    #[error("cannot parse MIR: {0}")]
    Parse(String),
    /// An operation, attribute, or contract failed fail-closed validation.
    #[error("{scope}: {detail}")]
    Rejected {
        /// Diagnostic stage or operation scope.
        scope: String,
        /// Reason the input was rejected.
        detail: String,
    },
    /// Pliron structural or dominance verification failed.
    #[error("MIR structural verification failed: {0}")]
    Verification(String),
    /// The requested symbol was not present in the input MIR.
    #[error("function `{0}` was not found")]
    FunctionNotFound(String),
}

/// Construct a fail-closed [`ExportError::Rejected`] diagnostic for `scope`.
/// Used whenever an operation, attribute, or contract violates the verified subset.
pub(crate) fn reject(scope: &str, detail: impl Into<String>) -> ExportError {
    ExportError::Rejected {
        scope: scope.into(),
        detail: detail.into(),
    }
}

/// Validate that `attrs` contains only attributes listed in `permitted` (plus
/// well-formed `builtin_given_names` presentation metadata).
pub(crate) fn check_attributes(
    attrs: &AttributeDict,
    permitted: &[&str],
    scope: &str,
) -> Result<(), ExportError> {
    for (name, attr) in &attrs.0 {
        let name = name.to_string();
        if name == "builtin_given_names" {
            if attr.downcast_ref::<GivenNamesAttr>().is_none() {
                return Err(reject(scope, "malformed given-name metadata"));
            }
        } else if !permitted.contains(&name.as_str()) {
            return Err(reject(scope, format!("unsupported attribute `{name}`")));
        }
    }
    Ok(())
}

/// Look up and downcast a required attribute `name` of type `T` on `attrs`,
/// returning a fail-closed [`ExportError::Rejected`] error when missing or mistyped.
pub(crate) fn get_attribute<'a, T: pliron::attribute::Attribute>(
    attrs: &'a AttributeDict,
    name: &str,
    scope: &str,
) -> Result<&'a T, ExportError> {
    attrs
        .0
        .iter()
        .find(|(key, _)| key.to_string() == name)
        .and_then(|(_, value)| value.downcast_ref::<T>())
        .ok_or_else(|| reject(scope, format!("missing or malformed attribute `{name}`")))
}

/// Extract the `builtin_sym_name` string identifier from `op`.
/// Rejects operations that lack a valid symbol name attribute.
pub(crate) fn symbol_name(ctx: &Context, op: Ptr<Operation>) -> Result<String, ExportError> {
    let operation = op.deref(ctx);
    Ok(
        get_attribute::<IdentifierAttr>(&operation.attributes, "builtin_sym_name", "function")?
            .as_ref()
            .to_string(),
    )
}

/// Parse an entire MIR file and export a function by its exact symbol name.
///
/// Accepts a standalone `mir.func` or a `builtin.module` of direct functions.
/// The selected function alone is exported. Trailing input is never ignored.
pub fn export_named_source(source: &str, name: &str) -> Result<String, ExportError> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    let headers = skip_many(attempt(
        spaces()
            .with(string("//"))
            .skip(skip_many(satisfy(|c| c != '\n'))),
    ));
    let root = parse_from_str(
        headers
            .with(spaced(Operation::top_level_parser()))
            .skip(combine::eof()),
        &mut ctx,
        source,
    )
    .map_err(|error| ExportError::Parse(error.to_string()))?;

    if MirFuncOp::wrap(&ctx, root).is_some() {
        if symbol_name(&ctx, root)? != name {
            return Err(ExportError::FunctionNotFound(name.into()));
        }
        return export_function(&ctx, root);
    }
    if Operation::get_opid(root, &ctx).to_string() != "builtin.module" {
        return Err(reject("input", "expected mir.func or builtin.module"));
    }
    let module = root.deref(&ctx);
    if module.num_regions() != 1
        || module.get_num_results() != 0
        || module.get_num_operands() != 0
        || module.get_num_successors() != 0
    {
        return Err(reject("module", "malformed module structure"));
    }
    check_attributes(&module.attributes, &["builtin_sym_name"], "module")?;
    get_attribute::<IdentifierAttr>(&module.attributes, "builtin_sym_name", "module")?;
    let region = module.get_region(0).deref(&ctx);
    let blocks: Vec<_> = region.iter(&ctx).collect();
    if blocks.len() != 1 {
        return Err(reject("module", "expected exactly one module block"));
    }
    let block = blocks[0].deref(&ctx);
    if block.get_num_arguments() != 0 {
        return Err(reject("module", "module block cannot have arguments"));
    }
    check_attributes(&block.attributes, &[], "module block")?;
    let mut selected = None;
    let mut symbols = HashSet::new();
    for child in block.iter(&ctx) {
        if MirFuncOp::wrap(&ctx, child).is_none() {
            return Err(reject(
                "module",
                "only direct mir.func children are supported",
            ));
        }
        let symbol = symbol_name(&ctx, child)?;
        if !symbols.insert(symbol.clone()) {
            return Err(reject(
                "module",
                format!("duplicate function symbol `{symbol}`"),
            ));
        }
        if symbol == name {
            selected = Some(child);
        }
    }
    export_function(
        &ctx,
        selected.ok_or_else(|| ExportError::FunctionNotFound(name.into()))?,
    )
}

/// Translate a `mir.func` operation into Lean 4 source code using the unified
/// [`translate::translate_function`] and [`codegen::render_lean_module`] pipeline.
pub fn export_function(ctx: &Context, function: Ptr<Operation>) -> Result<String, ExportError> {
    if MirFuncOp::wrap(ctx, function).is_none() {
        return Err(reject("function", "expected mir.func"));
    }
    let translated = translate::translate_function(ctx, function)?;
    codegen::render_lean_module(&translated)
}
