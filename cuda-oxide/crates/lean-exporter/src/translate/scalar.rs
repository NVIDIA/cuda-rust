/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-block pure scalar integer `mir.func` transcriber (`arithmetic.mir` ->
//! `CudaOxide.Mir.Scalar.Expr` `BitVec` AST).

use pliron::{
    builtin::{
        attributes::{IdentifierAttr, IntegerAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::{Operation, verify_operation},
    r#type::{TypeHandle, Typed},
};

use super::{TranslatedFunction, ir_utils::op_name};
use crate::{
    ExportError, check_attributes, get_attribute, reject,
    ssa::{SsaEnv, SsaKind},
};

/// Extract the unsigned bit width (`8`, `16`, `32`, or `64`) of a scalar integer `TypeHandle`.
/// Fails closed on non-integer, signed/signless, or unsupported integer bit widths.
fn unsigned_width(ctx: &Context, ty: TypeHandle) -> Result<u32, ExportError> {
    let ty = ty.deref(ctx);
    let Some(integer) = ty.downcast_ref::<IntegerType>() else {
        return Err(reject("type", "only unsigned integer types are supported"));
    };
    if !integer.is_unsigned() {
        return Err(reject(
            "type",
            "expected an unsigned integer of width 8, 16, 32, or 64",
        ));
    }
    match integer.width() {
        8 | 16 | 32 | 64 => Ok(integer.width()),
        _ => Err(reject(
            "type",
            "expected an unsigned integer of width 8, 16, 32, or 64",
        )),
    }
}

/// Transcribe a single-block pure scalar integer `mir.func` into [`TranslatedFunction::Scalar`].
/// Preserves exact SSA operand ordering and unsigned `BitVec` wraparound semantics.
pub(super) fn translate_scalar_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
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
    let blocks: Vec<_> = func.get_region(0).deref(ctx).iter(ctx).collect();
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

    let mut env = SsaEnv::new();
    let mut let_lines = Vec::new();
    for (index, arg) in block.arguments().enumerate() {
        if arg.get_type(ctx) != value_type {
            return Err(reject(
                "entry block",
                "argument type does not match function signature",
            ));
        }
        let name = env.bind_arg(arg, index, SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := .arg ⟨{index}, by decide⟩"
        ));
    }

    let mut returned_var = None;
    for (index, op_ptr) in block.iter(ctx).enumerate() {
        let opid = op_name(ctx, op_ptr);
        let scope = format!("operation {index} ({opid})");
        if returned_var.is_some() {
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
            let binding = env.get(&operand, &scope)?;
            if binding.kind != (SsaKind::ScalarWord { width }) {
                return Err(reject(&scope, "operand type mismatch in scalar expression"));
            }
            operands.push(binding.lean_name.clone());
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
            returned_var = Some(operands[0].clone());
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
        let name = env.bind_result(op.get_result(0), SsaKind::ScalarWord { width });
        let_lines.push(format!(
            "  let {name} : CudaOxide.Mir.Scalar.Expr {width} {arity} := {expression}"
        ));
    }
    let returned_var =
        returned_var.ok_or_else(|| reject("function", "missing mir.return terminator"))?;
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;

    Ok(TranslatedFunction::Scalar {
        width,
        arity,
        let_lines,
        returned_var,
    })
}
