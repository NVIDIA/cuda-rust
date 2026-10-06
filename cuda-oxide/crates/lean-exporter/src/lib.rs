/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A fail-closed exporter for single-block, unsigned, modular scalar MIR.
//!
//! This translates the selected function's actual SSA operations. It neither
//! proves the translation correct nor verifies a Rust kernel or generated PTX.

use std::collections::{HashMap, HashSet};

use combine::{
    Parser, attempt,
    parser::char::{spaces, string},
    satisfy, skip_many,
};
use dialect_mir::ops::MirFuncOp;
use pliron::{
    attribute::AttributeDict,
    builtin::{
        attributes::{GivenNamesAttr, IdentifierAttr, IntegerAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
        types::IntegerType,
    },
    context::{Context, Ptr},
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    parsable::parse_from_str,
    r#type::{TypeHandle, Typed},
    value::Value,
};

/// Unsupported and malformed input is rejected before any Lean is returned.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("cannot parse MIR: {0}")]
    Parse(String),
    #[error("{scope}: {detail}")]
    Rejected { scope: String, detail: String },
    #[error("MIR structural verification failed: {0}")]
    Verification(String),
    #[error("function `{0}` was not found")]
    FunctionNotFound(String),
}

fn reject(scope: &str, detail: impl Into<String>) -> ExportError {
    ExportError::Rejected {
        scope: scope.into(),
        detail: detail.into(),
    }
}

/// Only presentation metadata is ignored; all other attributes must be known.
fn check_attributes(
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

fn get_attribute<'a, T: pliron::attribute::Attribute>(
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

fn unsigned_width(ctx: &Context, ty: TypeHandle) -> Result<u32, ExportError> {
    let type_ref = ty.deref(ctx);
    let int = type_ref.downcast_ref::<IntegerType>().ok_or_else(|| {
        reject(
            "function signature",
            "only unsigned integer types are supported",
        )
    })?;
    if !int.is_unsigned() || !matches!(int.width(), 8 | 16 | 32 | 64) {
        return Err(reject(
            "function signature",
            "expected an unsigned integer of width 8, 16, 32, or 64",
        ));
    }
    Ok(int.width())
}

fn symbol_name(ctx: &Context, op: Ptr<Operation>) -> Result<String, ExportError> {
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
    let mut ctx = Context::new(); // Registers the builtin dialect.
    dialect_mir::register(&mut ctx);
    // Pliron's parser does not accept comments. Allow leading line comments for
    // licensing/provenance headers, retaining original source positions. All
    // actual IR, including the requirement for EOF, is parsed by Pliron.
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

/// Translate a single scalar MIR function into a closed Lean expression.
///
/// Supported functions have one block, one result and uniformly typed `ui8`,
/// `ui16`, `ui32`, or `ui64` arguments/results. Supported operations are integer
/// constants, wrapping add/sub/mul and return. Every operation is checked,
/// including dead operations. Names are regenerated, never copied into Lean.
pub fn export_function(ctx: &Context, function: Ptr<Operation>) -> Result<String, ExportError> {
    if MirFuncOp::wrap(ctx, function).is_none() {
        return Err(reject("function", "expected mir.func"));
    }
    let func = function.deref(ctx);
    if func.num_regions() != 1
        || func.get_num_operands() != 0
        || func.get_num_results() != 0
        || func.get_num_successors() != 0
    {
        return Err(reject(
            "function",
            "expected one region and no operands, results, or successors",
        ));
    }
    check_attributes(
        &func.attributes,
        &["builtin_sym_name", "mir_func_type"],
        "function",
    )?;
    get_attribute::<IdentifierAttr>(&func.attributes, "builtin_sym_name", "function")?;
    let type_attr = get_attribute::<TypeAttr>(&func.attributes, "mir_func_type", "function")?;
    let fn_type = type_attr.get_type(ctx).deref(ctx);
    let fn_type = fn_type.downcast_ref::<FunctionType>().ok_or_else(|| {
        reject(
            "function",
            "mir_func_type must contain a builtin.function type",
        )
    })?;
    let inputs = fn_type.arg_types();
    let outputs = fn_type.res_types();
    if outputs.len() != 1 {
        return Err(reject("function signature", "expected exactly one result"));
    }
    let value_type = outputs[0];
    let width = unsigned_width(ctx, value_type)?;
    if inputs.iter().any(|ty| *ty != value_type) {
        return Err(reject(
            "function signature",
            "all arguments must have the same type as the result",
        ));
    }
    let arity = inputs.len();
    let region = func.get_region(0).deref(ctx);
    let blocks: Vec<_> = region.iter(ctx).collect();
    if blocks.len() != 1 {
        return Err(reject(
            "function",
            "expected exactly one nonempty basic block",
        ));
    }
    let block = blocks[0].deref(ctx);
    check_attributes(&block.attributes, &[], "entry block")?;
    if block.get_num_arguments() != arity {
        return Err(reject(
            "entry block",
            "argument count does not match function signature",
        ));
    }

    let mut names: HashMap<Value, String> = HashMap::new();
    let mut lines = Vec::new();
    for (index, arg) in block.arguments().enumerate() {
        if arg.get_type(ctx) != value_type {
            return Err(reject(
                "entry block",
                "argument type does not match function signature",
            ));
        }
        let name = format!("arg{index}");
        lines.push(format!("  let {name} : CudaOxide.Model.Scalar.Expr {width} {arity} := .arg ⟨{index}, by decide⟩"));
        names.insert(arg, name);
    }

    let mut returned = None;
    let mut result_index = 0;
    for (index, op_ptr) in block.iter(ctx).enumerate() {
        let opid = Operation::get_opid(op_ptr, ctx).to_string();
        let scope = format!("operation {index} ({opid})");
        if returned.is_some() {
            return Err(reject(&scope, "operation after mir.return"));
        }
        let (expected_operands, expected_results, constant_key) = match opid.as_str() {
            "mir.add" | "mir.sub" | "mir.mul" => (2, 1, None),
            "mir.constant" => (0, 1, Some("value")),
            "builtin.constant" => (0, 1, Some("builtin_constant_value")),
            "mir.return" => (1, 0, None),
            _ => return Err(reject(&scope, "unsupported operation in scalar subset")),
        };
        let op = op_ptr.deref(ctx);
        if op.num_regions() != 0 || op.get_num_successors() != 0 {
            return Err(reject(&scope, "regions and successors are unsupported"));
        }
        if op.get_num_operands() != expected_operands || op.get_num_results() != expected_results {
            return Err(reject(&scope, "incorrect operand or result count"));
        }
        check_attributes(
            &op.attributes,
            &constant_key.into_iter().collect::<Vec<_>>(),
            &scope,
        )?;
        let mut operands = Vec::new();
        for operand in op.operands() {
            // Membership checks also reject forward, dangling and external SSA
            // references before consulting any referenced type or definition.
            let name = names.get(&operand).ok_or_else(|| {
                reject(
                    &scope,
                    "operand is not an earlier result or an entry argument",
                )
            })?;
            operands.push(name.clone());
        }
        for result in op.results() {
            if result.get_type(ctx) != value_type {
                return Err(reject(
                    &scope,
                    "result type does not match function signature",
                ));
            }
        }
        if opid == "mir.return" {
            returned = Some(operands[0].clone());
            continue;
        }
        let expression = if let Some(key) = constant_key {
            let literal = get_attribute::<IntegerAttr>(&op.attributes, key, &scope)?;
            if TypeHandle::from(literal.get_type()) != value_type
                || literal.value().bw() != width as usize
            {
                return Err(reject(
                    &scope,
                    "constant type/bit width does not match its result",
                ));
            }
            format!(
                ".literal (BitVec.ofNat {width} {})",
                literal.value().to_string_unsigned_decimal()
            )
        } else {
            let constructor = match opid.as_str() {
                "mir.add" => "add",
                "mir.sub" => "sub",
                "mir.mul" => "mul",
                _ => return Err(reject(&scope, "unsupported arithmetic operation")),
            };
            format!(".{constructor} {} {}", operands[0], operands[1])
        };
        let name = format!("v{result_index}");
        result_index += 1;
        lines.push(format!(
            "  let {name} : CudaOxide.Model.Scalar.Expr {width} {arity} := {expression}"
        ));
        names.insert(op.get_result(0), name);
    }
    let returned = returned.ok_or_else(|| reject("function", "missing mir.return terminator"))?;

    // Run Pliron's recursive structural/interface checks and dominance check
    // after the preflight above has ruled out dialect verifier panic cases.
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated from scalar MIR. This file is a model, not a proof certificate.\n\
         import CudaOxide.Model.Scalar\n\n\
         namespace CudaOxide.Generated\n\n\
         def program : CudaOxide.Model.Scalar.Expr {width} {arity} :=\n{}\n  {returned}\n\n\
         def evaluate (inputs : Fin {arity} → BitVec {width}) : BitVec {width} := program.eval inputs\n\n\
         end CudaOxide.Generated\n",
        lines.join("\n"),
    ))
}
