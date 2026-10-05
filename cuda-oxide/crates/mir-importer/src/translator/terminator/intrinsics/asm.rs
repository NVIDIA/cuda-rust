/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline PTX marker-call translation.

use super::super::helpers::{self, emit_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops::MirConstructTupleOp;
use dialect_mir::types::{MirTupleType, pointer_carriers_in_type};
use dialect_nvvm::ops::InlinePtxOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

const OUT_PREFIX: &str = "cuda_device::ptx::__ptx_asm_out_";
const VOID_PREFIX: &str = "cuda_device::ptx::__ptx_asm_void_";
const REGISTER_ONLY_OPTION: &str = "register_only";
const REGISTER_ONLY_MAY_DIVERGE_OPTIONS: &str = "register_only,may_diverge";

const COMPILE_TIME_STRING_CONSTRAINT: &str = "C";

#[derive(Copy, Clone)]
struct InlinePtxOptions {
    sideeffect: bool,
    convergent: bool,
}

struct PreparedInlinePtx<'a> {
    template: String,
    constraints: String,
    runtime_args: Vec<&'a mir::Operand>,
}

enum InlinePtxInput {
    Runtime,
    CompileTime(String),
}

enum TemplateOperand<'a> {
    Runtime(usize),
    CompileTime(&'a str),
}

#[derive(Copy, Clone)]
pub enum InlinePtxCallKind {
    Output { inputs: usize },
    Void { inputs: usize },
}

impl InlinePtxCallKind {
    pub fn from_path(path: &str) -> Option<Self> {
        if let Some(rest) = path.strip_prefix(OUT_PREFIX) {
            return rest
                .parse::<usize>()
                .ok()
                .map(|inputs| InlinePtxCallKind::Output { inputs });
        }
        if let Some(rest) = path.strip_prefix(VOID_PREFIX) {
            return rest
                .parse::<usize>()
                .ok()
                .map(|inputs| InlinePtxCallKind::Void { inputs });
        }
        None
    }

    fn has_output(self) -> bool {
        matches!(self, InlinePtxCallKind::Output { .. })
    }

    fn inputs(self) -> usize {
        match self {
            InlinePtxCallKind::Output { inputs } | InlinePtxCallKind::Void { inputs } => inputs,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn emit_inline_ptx(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    kind: InlinePtxCallKind,
) -> TranslationResult<Ptr<Operation>> {
    let expected_args = 3 + kind.inputs();
    if args.len() != expected_args {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ptx_asm marker expected {expected_args} arguments, got {}",
                args.len()
            ))
        );
    }

    let template = literal_operand_string(&args[0], "ptx_asm template", loc.clone())?;
    let constraints = literal_operand_string(&args[1], "ptx_asm constraints", loc.clone())?;
    let options_marker = literal_operand_string(&args[2], "ptx_asm options", loc.clone())?;
    let options = parse_options(&options_marker, loc.clone())?;

    // Count outputs before removing compile-time string inputs. Output operands
    // retain their original indices; only runtime inputs after a `C` operand
    // need to be renumbered.
    let num_outputs = InlinePtxOp::count_output_constraints(&constraints);

    let prepared = prepare_inline_ptx(
        &template,
        &constraints,
        body,
        &args[3..],
        num_outputs,
        loc.clone(),
    )?;

    let mut input_values = Vec::with_capacity(prepared.runtime_args.len());
    let mut last_op = prev_op;

    for &arg in &prepared.runtime_args {
        let (value, arg_last_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        input_values.push(value);
        last_op = arg_last_op;
    }

    if !kind.has_output() {
        // Void call: no results.
        let inline_ptx = InlinePtxOp::build(
            ctx,
            vec![],
            input_values,
            &prepared.template,
            &prepared.constraints,
            options.sideeffect,
            options.convergent,
        );
        inline_ptx.deref_mut(ctx).set_loc(loc.clone());

        let inline_ptx = if let Some(prev) = last_op {
            inline_ptx.insert_after(ctx, prev);
            inline_ptx
        } else {
            inline_ptx.insert_at_front(block_ptr, ctx);
            inline_ptx
        };

        if let Some(target_idx) = target {
            return Ok(emit_goto(ctx, *target_idx, inline_ptx, block_map, loc));
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported("ptx_asm void call without target block".to_string())
            );
        }
    }

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    if num_outputs <= 1 {
        // Single-output (backward-compatible path): the destination type is the result type.
        let result_tys = vec![types::translate_destination_type(
            ctx,
            body,
            destination,
            &loc,
        )?];
        // Pointer authority follows rustc's destination type, never the PTX
        // template or register constraint chosen by the user.
        let has_pointer_result = result_tys
            .iter()
            .any(|result_ty| !pointer_carriers_in_type(ctx, *result_ty).is_empty());

        let inline_ptx = InlinePtxOp::build(
            ctx,
            result_tys,
            input_values,
            &prepared.template,
            &prepared.constraints,
            options.sideeffect,
            options.convergent,
        );
        if has_pointer_result {
            InlinePtxOp::new(inline_ptx)
                .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::InlineAsm);
        }
        inline_ptx.deref_mut(ctx).set_loc(loc.clone());

        let inline_ptx = if let Some(prev) = last_op {
            inline_ptx.insert_after(ctx, prev);
            inline_ptx
        } else {
            inline_ptx.insert_at_front(block_ptr, ctx);
            inline_ptx
        };

        let result_value = inline_ptx.deref(ctx).get_result(0);
        helpers::emit_prepared_result_and_goto(
            ctx,
            prepared_destination,
            result_value,
            target,
            block_ptr,
            inline_ptx,
            value_map,
            block_map,
            loc,
            "ptx_asm output call without target block",
        )
    } else {
        // Multi-output: the destination type is a tuple. Decompose it to get
        // per-output result types, then pack the results back into a tuple.
        let dest_ty = types::translate_destination_type(ctx, body, destination, &loc)?;
        let element_types = {
            let t = dest_ty.deref(ctx);
            match t.downcast_ref::<MirTupleType>() {
                Some(tup) if tup.get_types().len() == num_outputs => tup.get_types().to_vec(),
                _ => {
                    return input_err!(
                        loc.clone(),
                        TranslationErr::unsupported(format!(
                            "ptx_asm multi-output destination must be a {num_outputs}-element tuple"
                        ))
                    );
                }
            }
        };
        // Each result type is an element of rustc's exact destination tuple;
        // one pointer-carrying element marks the whole producer boundary.
        let has_pointer_result = element_types
            .iter()
            .any(|result_ty| !pointer_carriers_in_type(ctx, *result_ty).is_empty());

        let inline_ptx = InlinePtxOp::build(
            ctx,
            element_types,
            input_values,
            &prepared.template,
            &prepared.constraints,
            options.sideeffect,
            options.convergent,
        );
        if has_pointer_result {
            InlinePtxOp::new(inline_ptx)
                .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::InlineAsm);
        }
        inline_ptx.deref_mut(ctx).set_loc(loc.clone());

        let inline_ptx = if let Some(prev) = last_op {
            inline_ptx.insert_after(ctx, prev);
            inline_ptx
        } else {
            inline_ptx.insert_at_front(block_ptr, ctx);
            inline_ptx
        };

        // Collect results from the multi-output op.
        let result_values: Vec<_> = (0..num_outputs)
            .map(|i| inline_ptx.deref(ctx).get_result(i))
            .collect();

        // Pack results into a tuple to match the destination type.
        let tuple_op = Operation::new(
            ctx,
            MirConstructTupleOp::get_concrete_op_info(),
            vec![dest_ty],
            result_values,
            vec![],
            0,
        );
        tuple_op.deref_mut(ctx).set_loc(loc.clone());
        tuple_op.insert_after(ctx, inline_ptx);
        super::super::helpers::set_compiler_result_bundle_marker(ctx, tuple_op);
        let tuple_val = tuple_op.deref(ctx).get_result(0);

        helpers::emit_prepared_result_and_goto(
            ctx,
            prepared_destination,
            tuple_val,
            target,
            block_ptr,
            tuple_op,
            value_map,
            block_map,
            loc,
            "ptx_asm multi-output call without target block",
        )
    }
}

fn prepare_inline_ptx<'a>(
    template: &str,
    constraints: &str,
    body: &mir::Body,
    args: &'a [mir::Operand],
    num_outputs: usize,
    loc: Location,
) -> TranslationResult<PreparedInlinePtx<'a>> {
    let constraint_parts = split_constraints(constraints);
    let input_end = num_outputs + args.len();

    if constraint_parts.len() < input_end {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ptx_asm marker expected at least {input_end} operand constraints, got {} total constraints",
                constraint_parts.len()
            ))
        );
    }

    let outputs_are_prefix = constraint_parts[..num_outputs]
        .iter()
        .all(|constraint| constraint.starts_with('='))
        && constraint_parts[num_outputs..]
            .iter()
            .all(|constraint| !constraint.starts_with('='));

    if !outputs_are_prefix {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "ptx_asm marker constraints must list all output constraints before input constraints"
                    .to_string()
            )
        );
    }

    let input_constraints = &constraint_parts[num_outputs..input_end];

    if input_constraints
        .iter()
        .any(|constraint| constraint.starts_with("~{"))
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "ptx_asm marker has fewer input constraints than input arguments".to_string()
            )
        );
    }

    if constraint_parts[input_end..]
        .iter()
        .any(|constraint| !constraint.starts_with("~{"))
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "ptx_asm marker contains a non-clobber constraint after its input operands"
                    .to_string()
            )
        );
    }

    let mut inputs = Vec::with_capacity(args.len());
    let mut runtime_args = Vec::with_capacity(args.len());

    for (input_index, (constraint, arg)) in input_constraints.iter().copied().zip(args).enumerate()
    {
        if constraint == COMPILE_TIME_STRING_CONSTRAINT {
            let operand_index = num_outputs + input_index;

            // Only byte-string references may be spliced as compile-time
            // text. Anything else (e.g. a bare integer constant) would
            // silently splice its raw little-endian bytes into the template.
            let operand_ty = arg.ty(body.locals()).ok();
            if !operand_ty.as_ref().is_some_and(is_byte_array_ref) {
                let ty_desc = operand_ty
                    .map(|ty| format!("{:?}", ty.kind()))
                    .unwrap_or_else(|| "<unknown>".to_string());
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "ptx_asm `C` operand ${operand_index} must be a `&'static [u8; N]` \
                         byte-string constant, got {ty_desc}"
                    ))
                );
            }

            let kind_name = format!("ptx_asm `C` operand ${operand_index}");
            let value = literal_operand_string(arg, &kind_name, loc.clone())?;

            inputs.push(InlinePtxInput::CompileTime(
                trim_compile_time_string_terminator(value),
            ));
        } else {
            inputs.push(InlinePtxInput::Runtime);
            runtime_args.push(arg);
        }
    }

    let (template, constraints) =
        rewrite_inline_ptx(template, &constraint_parts, num_outputs, &inputs, loc)?;

    Ok(PreparedInlinePtx {
        template,
        constraints,
        runtime_args,
    })
}

/// Whether `ty` is a reference to a `u8` array (`&[u8; N]`), the only
/// operand type `in("C")` accepts.
fn is_byte_array_ref(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind, UintTy};

    let TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) = ty.kind() else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::Array(element, _)) = pointee.kind() else {
        return false;
    };
    matches!(element.kind(), TyKind::RigidTy(RigidTy::Uint(UintTy::U8)))
}

fn split_constraints(constraints: &str) -> Vec<&str> {
    if constraints.is_empty() {
        Vec::new()
    } else {
        constraints.split(',').collect()
    }
}

fn trim_compile_time_string_terminator(mut value: String) -> String {
    if value.as_bytes().last() == Some(&0) {
        value.pop();
    }

    value
}

fn rewrite_inline_ptx(
    template: &str,
    constraints: &[&str],
    num_outputs: usize,
    inputs: &[InlinePtxInput],
    loc: Location,
) -> TranslationResult<(String, String)> {
    let input_end = num_outputs + inputs.len();

    if constraints.len() < input_end {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ptx_asm rewrite expected at least {input_end} operand constraints, got {}",
                constraints.len()
            ))
        );
    }

    let mut template_operands = Vec::with_capacity(input_end);
    let mut rewritten_constraints = Vec::with_capacity(constraints.len());

    for (output_index, constraint) in constraints[..num_outputs].iter().copied().enumerate() {
        template_operands.push(TemplateOperand::Runtime(output_index));
        rewritten_constraints.push(constraint);
    }

    let mut next_runtime_index = num_outputs;

    for (input_index, (constraint, input)) in constraints[num_outputs..input_end]
        .iter()
        .copied()
        .zip(inputs)
        .enumerate()
    {
        let original_operand_index = num_outputs + input_index;

        match (constraint, input) {
            (COMPILE_TIME_STRING_CONSTRAINT, InlinePtxInput::CompileTime(value)) => {
                template_operands.push(TemplateOperand::CompileTime(value));
            }

            (COMPILE_TIME_STRING_CONSTRAINT, InlinePtxInput::Runtime) => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "ptx_asm `C` operand ${original_operand_index} was not resolved as compile-time text"
                    ))
                );
            }

            (_, InlinePtxInput::CompileTime(_)) => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "ptx_asm operand ${original_operand_index} was resolved as compile-time text but uses constraint `{constraint}`"
                    ))
                );
            }

            (_, InlinePtxInput::Runtime) => {
                template_operands.push(TemplateOperand::Runtime(next_runtime_index));
                rewritten_constraints.push(constraint);
                next_runtime_index += 1;
            }
        }
    }

    // Clobbers follow all output and input constraints and do not correspond
    // to numbered template operands.
    rewritten_constraints.extend_from_slice(&constraints[input_end..]);

    let rewritten_template = rewrite_template_operands(template, &template_operands, loc)?;

    Ok((rewritten_template, rewritten_constraints.join(",")))
}

fn rewrite_template_operands(
    template: &str,
    operands: &[TemplateOperand<'_>],
    loc: Location,
) -> TranslationResult<String> {
    let bytes = template.as_bytes();
    let mut rewritten = String::with_capacity(template.len());
    let mut cursor = 0;

    while cursor < bytes.len() {
        let Some(relative_dollar) = bytes[cursor..].iter().position(|byte| *byte == b'$') else {
            rewritten.push_str(&template[cursor..]);
            break;
        };

        let dollar = cursor + relative_dollar;
        rewritten.push_str(&template[cursor..dollar]);

        if dollar + 1 >= bytes.len() {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "ptx_asm LLVM template contains a trailing `$`".to_string()
                )
            );
        }

        match bytes[dollar + 1] {
            b'$' => {
                // A literal `$` was already escaped by the macro.
                rewritten.push_str("$$");
                cursor = dollar + 2;
            }

            b'0'..=b'9' => {
                let mut end = dollar + 2;

                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }

                let digits = &template[dollar + 1..end];
                let operand_index = match digits.parse::<usize>() {
                    Ok(index) => index,
                    Err(_) => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "ptx_asm template placeholder `${digits}` is too large"
                            ))
                        );
                    }
                };

                let Some(replacement) = operands.get(operand_index) else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "ptx_asm template placeholder `${digits}` has no matching operand"
                        ))
                    );
                };

                match replacement {
                    TemplateOperand::Runtime(new_index) => {
                        rewritten.push('$');
                        rewritten.push_str(&new_index.to_string());
                    }

                    TemplateOperand::CompileTime(value) => {
                        // Text introduced after macro expansion has not passed through
                        // convert_cuda_template(). Escape LLVM's `$` sigil here.
                        rewritten.push_str(&value.replace('$', "$$"));
                    }
                }

                cursor = end;
            }

            _ => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "ptx_asm LLVM template contains `$` not followed by `$` or an operand index"
                            .to_string()
                    )
                );
            }
        }
    }

    Ok(rewritten)
}

fn literal_operand_string(
    operand: &mir::Operand,
    kind_name: &str,
    loc: Location,
) -> TranslationResult<String> {
    let bytes = match operand {
        mir::Operand::Constant(constant) => {
            rvalue::constant_bytes(constant, kind_name, loc.clone())?
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{kind_name} must be a byte string literal, got MIR operand {other:?}"
                ))
            );
        }
    };

    String::from_utf8(bytes).map_err(|err| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "{kind_name} must be valid UTF-8: {err}"
        )))
    })
}

fn parse_options(marker: &str, loc: Location) -> TranslationResult<InlinePtxOptions> {
    match marker {
        "" => Ok(InlinePtxOptions {
            sideeffect: true,
            convergent: true,
        }),
        REGISTER_ONLY_OPTION => Ok(InlinePtxOptions {
            sideeffect: false,
            convergent: true,
        }),
        REGISTER_ONLY_MAY_DIVERGE_OPTIONS => Ok(InlinePtxOptions {
            sideeffect: false,
            convergent: false,
        }),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!("unsupported ptx_asm options marker `{other}`"))
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_for_test(
        template: &str,
        constraints: &str,
        num_outputs: usize,
        inputs: &[InlinePtxInput],
    ) -> (String, String) {
        let constraint_parts = split_constraints(constraints);

        rewrite_inline_ptx(
            template,
            &constraint_parts,
            num_outputs,
            inputs,
            Location::Unknown,
        )
        .unwrap()
    }

    #[test]
    fn register_only_keeps_inline_ptx_convergent() {
        let options = parse_options(REGISTER_ONLY_OPTION, Location::Unknown).unwrap();

        assert!(!options.sideeffect);
        assert!(options.convergent);
    }

    #[test]
    fn may_diverge_opt_in_drops_convergent() {
        let options = parse_options(REGISTER_ONLY_MAY_DIVERGE_OPTIONS, Location::Unknown).unwrap();

        assert!(!options.sideeffect);
        assert!(!options.convergent);
    }

    #[test]
    fn substitutes_compile_time_string_and_renumbers_operands() {
        let inputs = [
            InlinePtxInput::CompileTime(".wide".to_string()),
            InlinePtxInput::Runtime,
            InlinePtxInput::Runtime,
        ];

        let (template, constraints) =
            rewrite_for_test("mul$1.u32 $0, $2, $3;", "=l,C,r,r", 1, &inputs);

        assert_eq!(template, "mul.wide.u32 $0, $1, $2;");
        assert_eq!(constraints, "=l,r,r");
    }

    #[test]
    fn substitutes_repeated_compile_time_string_reference() {
        let inputs = [InlinePtxInput::CompileTime(".rn".to_string())];

        let (template, constraints) = rewrite_for_test("cvt$0$0.f32.s32;", "C", 0, &inputs);

        assert_eq!(template, "cvt.rn.rn.f32.s32;");
        assert_eq!(constraints, "");
    }

    #[test]
    fn strips_exactly_one_trailing_nul_from_compile_time_string() {
        assert_eq!(
            trim_compile_time_string_terminator(".wide\0".to_string()),
            ".wide"
        );
        assert_eq!(
            trim_compile_time_string_terminator(".wide".to_string()),
            ".wide"
        );
        assert_eq!(
            trim_compile_time_string_terminator(".wide\0\0".to_string()),
            ".wide\0"
        );
    }

    #[test]
    fn escapes_dollars_inside_compile_time_string() {
        let inputs = [InlinePtxInput::CompileTime("$mode".to_string())];

        let (template, constraints) = rewrite_for_test("instruction$0;", "C", 0, &inputs);

        assert_eq!(template, "instruction$$mode;");
        assert_eq!(constraints, "");
    }

    #[test]
    fn preserves_existing_escaped_dollars() {
        let inputs = [InlinePtxInput::Runtime];

        let (template, constraints) = rewrite_for_test("mov.u32 $0, $$L0;", "r", 0, &inputs);

        assert_eq!(template, "mov.u32 $0, $$L0;");
        assert_eq!(constraints, "r");
    }

    #[test]
    fn renumbers_multi_digit_operands_atomically() {
        let mut inputs = vec![InlinePtxInput::CompileTime(".wide".to_string())];
        inputs.extend((0..9).map(|_| InlinePtxInput::Runtime));

        let (template, constraints) =
            rewrite_for_test("use $1 and $10;", "=l,C,r,r,r,r,r,r,r,r,r", 1, &inputs);

        assert_eq!(template, "use .wide and $9;");
        assert_eq!(constraints, "=l,r,r,r,r,r,r,r,r,r");
    }

    #[test]
    fn rejects_out_of_range_template_operand() {
        let inputs = [InlinePtxInput::CompileTime(".wide".to_string())];
        let constraint_parts = split_constraints("C");

        let err = rewrite_inline_ptx(
            "instruction$1;",
            &constraint_parts,
            0,
            &inputs,
            Location::Unknown,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("placeholder `$1` has no matching operand"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn parses_output_marker_with_32_inputs() {
        let kind = InlinePtxCallKind::from_path("cuda_device::ptx::__ptx_asm_out_32")
            .expect("output marker should be recognized");

        assert!(matches!(kind, InlinePtxCallKind::Output { inputs: 32 }));
        assert_eq!(kind.inputs(), 32);
        assert!(kind.has_output());
    }
}
