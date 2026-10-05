/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic PTX assembly-syntax emission from structured operations.

use crate::ops::{
    PtxBranchTargetsOp, PtxCallableOp, PtxCfgBodyOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp,
    PtxModuleOp, PtxRawOp, PtxScopeOp, PtxTerminatorOp,
};
use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use std::collections::HashMap;
use std::fmt::{self, Write};

#[derive(Debug)]
pub enum EmitError {
    Verification(String),
    UnsupportedOperation(String),
    Format(fmt::Error),
}

impl fmt::Display for EmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verification(error) => write!(formatter, "invalid structured PTX: {error}"),
            Self::UnsupportedOperation(operation) => {
                write!(formatter, "cannot emit non-PTX operation {operation}")
            }
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EmitError {}

impl From<fmt::Error> for EmitError {
    fn from(error: fmt::Error) -> Self {
        Self::Format(error)
    }
}

/// Canonically emit one constructed or transformed PTX module.
///
/// This writer is intentionally distinct from lossless [`ptx_parse::EditScript`]
/// application. It may normalize formatting and should not be used for
/// minimal-perturbation source patching.
pub fn emit_canonical_module(ctx: &Context, module: &PtxModuleOp) -> Result<String, EmitError> {
    let mut output = String::new();
    write_canonical_module(ctx, module, &mut output)?;
    Ok(output)
}

/// Emit one structured PTX module into a caller-owned formatting sink.
pub fn write_canonical_module(
    ctx: &Context,
    module: &PtxModuleOp,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    module
        .get_operation()
        .deref(ctx)
        .verify(ctx)
        .map_err(|error| EmitError::Verification(error.to_string()))?;
    emit_block(ctx, module.body(ctx), 0, None, output)
}

fn emit_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    indent: usize,
    cfg: Option<&CfgEmitPlan>,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    for operation in block.deref(ctx).iter(ctx) {
        emit_operation(ctx, operation, indent, cfg, output)?;
    }
    Ok(())
}

fn emit_operation(
    ctx: &Context,
    operation: Ptr<Operation>,
    indent: usize,
    cfg: Option<&CfgEmitPlan>,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    if let Some(directive) = Operation::get_op::<PtxDirectiveOp>(operation, ctx) {
        write_indent(output, indent)?;
        for label in directive.labels(ctx) {
            output.write_str(&label)?;
            output.write_str(": ")?;
        }
        output.write_str(&directive.name(ctx))?;
        let arguments = directive.arguments(ctx);
        if !arguments.is_empty() {
            output.write_char(' ')?;
            output.write_str(&arguments)?;
        }
        output.write_char('\n')?;
        return Ok(());
    }
    if let Some(table) = Operation::get_op::<PtxBranchTargetsOp>(operation, ctx) {
        let cfg = cfg.expect("ptx.branch_targets only verifies inside native CFG");
        write_indent(output, indent)?;
        output.write_str(&table.name(ctx))?;
        output.write_str(": .branchtargets ")?;
        for (index, target) in cfg.table_targets[&table.name(ctx)].iter().enumerate() {
            if index != 0 {
                output.write_str(", ")?;
            }
            output.write_str(target)?;
        }
        output.write_str(";\n")?;
        return Ok(());
    }
    if let Some(callable) = Operation::get_op::<PtxCallableOp>(operation, ctx) {
        write_indent(output, indent)?;
        output.write_str(callable.header(ctx).trim())?;
        if callable.is_definition(ctx) {
            output.write_char('\n')?;
            write_indent(output, indent)?;
            output.write_str("{\n")?;
            if let Some(surface) = callable.surface_body(ctx) {
                emit_block(ctx, surface.body(ctx), indent + 1, None, output)?;
            } else if let Some(body) = callable.cfg_body(ctx) {
                let cfg = CfgEmitPlan::new(ctx, &body);
                for block in body.region(ctx).deref(ctx).iter(ctx) {
                    emit_block(ctx, block, indent + 1, Some(&cfg), output)?;
                }
            }
            write_indent(output, indent)?;
            output.write_str("}\n")?;
        } else {
            output.write_str(";\n")?;
        }
        return Ok(());
    }
    if let Some(label) = Operation::get_op::<PtxLabelOp>(operation, ctx) {
        write_indent(output, indent)?;
        output.write_str(&label.name(ctx))?;
        output.write_str(":\n")?;
        return Ok(());
    }
    if let Some(scope) = Operation::get_op::<PtxScopeOp>(operation, ctx) {
        let header = scope.header(ctx);
        if !header.is_empty() {
            write_indent(output, indent)?;
            output.write_str(header.trim())?;
            output.write_char('\n')?;
        }
        write_indent(output, indent)?;
        output.write_str("{\n")?;
        emit_block(ctx, scope.body(ctx), indent + 1, cfg, output)?;
        write_indent(output, indent)?;
        output.write_str("}\n")?;
        return Ok(());
    }
    if let Some(instruction) = Operation::get_op::<PtxInstructionOp>(operation, ctx) {
        write_indent(output, indent)?;
        if let Some(predicate) = instruction.predicate(ctx) {
            output.write_str(&predicate.guard_text())?;
            output.write_char(' ')?;
        }
        output.write_str(&instruction.head(ctx))?;
        let operands = instruction.operands(ctx);
        if !operands.is_empty() {
            output.write_char(' ')?;
            for (index, operand) in operands.iter().enumerate() {
                if index != 0 {
                    output.write_str(", ")?;
                }
                output.write_str(operand)?;
            }
        }
        output.write_str(";\n")?;
        return Ok(());
    }
    if let Some(terminator) = Operation::get_op::<PtxTerminatorOp>(operation, ctx) {
        if terminator.kind(ctx) == crate::attributes::TerminatorKindAttr::Fallthrough {
            return Ok(());
        }
        write_indent(output, indent)?;
        if let Some(predicate) = terminator.predicate(ctx) {
            output.write_str(&predicate.guard_text())?;
            output.write_char(' ')?;
        }
        output.write_str(&terminator.head(ctx))?;
        let mut operands = terminator.operands(ctx);
        let cfg = cfg.expect("ptx.terminator only verifies inside native CFG");
        let first_target = usize::from(terminator.has_fallthrough(ctx));
        match terminator.kind(ctx) {
            crate::attributes::TerminatorKindAttr::Branch => {
                let target = operation.deref(ctx).get_successor(first_target);
                operands.push(cfg.block_labels[&target].clone());
            }
            crate::attributes::TerminatorKindAttr::IndexedBranch => {
                operands.push(terminator.target_table(ctx));
            }
            _ => {}
        }
        if !operands.is_empty() {
            output.write_char(' ')?;
            for (index, operand) in operands.iter().enumerate() {
                if index != 0 {
                    output.write_str(", ")?;
                }
                output.write_str(operand)?;
            }
        }
        output.write_str(";\n")?;
        return Ok(());
    }
    if let Some(raw) = Operation::get_op::<PtxRawOp>(operation, ctx) {
        let text = raw.text(ctx);
        for line in text.trim().lines() {
            write_indent(output, indent)?;
            output.write_str(line.trim())?;
            output.write_char('\n')?;
        }
        return Ok(());
    }
    Err(EmitError::UnsupportedOperation(
        Operation::get_opid(operation, ctx).to_string(),
    ))
}

struct CfgEmitPlan {
    block_labels: HashMap<Ptr<BasicBlock>, String>,
    table_targets: HashMap<String, Vec<String>>,
}

impl CfgEmitPlan {
    fn new(ctx: &Context, body: &PtxCfgBodyOp) -> Self {
        let mut block_labels = HashMap::new();
        for block in body.region(ctx).deref(ctx).iter(ctx) {
            if let Some(label) = block
                .deref(ctx)
                .iter(ctx)
                .find_map(|operation| Operation::get_op::<PtxLabelOp>(operation, ctx))
            {
                block_labels.insert(block, label.name(ctx));
            }
        }
        let mut table_targets = HashMap::new();
        for block in body.region(ctx).deref(ctx).iter(ctx) {
            let Some(terminator) = block
                .deref(ctx)
                .iter(ctx)
                .last()
                .and_then(|operation| Operation::get_op::<PtxTerminatorOp>(operation, ctx))
            else {
                continue;
            };
            if terminator.kind(ctx) != crate::attributes::TerminatorKindAttr::IndexedBranch {
                continue;
            }
            let first_target = usize::from(terminator.has_fallthrough(ctx));
            let targets = terminator
                .get_operation()
                .deref(ctx)
                .successors()
                .skip(first_target)
                .map(|target| block_labels[&target].clone())
                .collect();
            table_targets.insert(terminator.target_table(ctx), targets);
        }
        Self {
            block_labels,
            table_targets,
        }
    }
}

fn write_indent(output: &mut impl Write, indent: usize) -> fmt::Result {
    for _ in 0..indent {
        output.write_str("    ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Projection;
    use crate::attributes::{CallableKindAttr, TerminatorKindAttr};
    use crate::ops::{PtxInstructionOp, PtxLabelOp, PtxTerminatorOp, PtxTerminatorSpec};

    #[test]
    fn projection_emits_canonical_nested_ptx() {
        let source = "\
.version 8.9
.target sm_120a
.address_size 64
.visible .entry kernel() {
    .reg .b32 %r<2>;
L0: @%p0 add.u32 %r0, %r0, 1;
    {
      mov.u32 %r0, 7;
    }
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let emitted = emit_canonical_module(&ctx, &projection.module()).unwrap();
        assert_eq!(
            emitted,
            "\
.version 8.9
.target sm_120a
.address_size 64
.visible .entry kernel()
{
    .reg .b32 %r<2>;
    L0:
    @%p0 add.u32 %r0, %r0, 1;
    {
        mov.u32 %r0, 7;
    }
    ret;
}
"
        );
        let reparsed = ptx_parse::Document::parse(&emitted).unwrap();
        assert!(reparsed.coverage().is_complete());
    }

    #[test]
    fn predicated_instructions_round_trip_through_the_typed_attribute() {
        let source = "\
.visible .entry kernel() {
    @%p1 add.u32 %r0, %r0, 1;
    @!%p1 bra L0;
L0:
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let emitted = emit_canonical_module(&ctx, &projection.module()).unwrap();
        assert_eq!(
            emitted,
            "\
.visible .entry kernel()
{
    @%p1 add.u32 %r0, %r0, 1;
    @!%p1 bra L0;
    L0:
    ret;
}
"
        );
        let reparsed = ptx_parse::Document::parse(&emitted).unwrap();
        assert!(reparsed.coverage().is_complete());
        let predicates: Vec<_> = reparsed
            .instructions()
            .iter()
            .map(|instruction| {
                instruction
                    .predicate()
                    .map(|predicate| (predicate.register().to_string(), predicate.is_negated()))
            })
            .collect();
        assert_eq!(
            predicates,
            vec![
                Some(("%p1".to_string(), false)),
                Some(("%p1".to_string(), true)),
                None
            ]
        );
    }

    #[test]
    fn emits_native_cfg_blocks_without_synthetic_fallthrough_text() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let module = PtxModuleOp::build(&mut ctx);
        PtxDirectiveOp::build(&mut ctx, ".version", "9.3")
            .get_operation()
            .insert_at_back(module.body(&ctx), &ctx);
        let callable = PtxCallableOp::build_cfg_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".visible .entry kernel()",
        );
        callable
            .get_operation()
            .insert_at_back(module.body(&ctx), &ctx);
        let body = callable.cfg_body(&ctx).unwrap();
        let entry = body.append_block(&mut ctx);
        let exit = body.append_block(&mut ctx);
        PtxLabelOp::build(&mut ctx, "L0")
            .get_operation()
            .insert_at_back(entry, &ctx);
        PtxInstructionOp::build(&mut ctx, None, "mov.u32", ["%r0", "1"])
            .get_operation()
            .insert_at_back(entry, &ctx);
        PtxTerminatorOp::fallthrough(&mut ctx, exit)
            .get_operation()
            .insert_at_back(entry, &ctx);
        PtxLabelOp::build(&mut ctx, "Done")
            .get_operation()
            .insert_at_back(exit, &ctx);
        PtxTerminatorOp::build(
            &mut ctx,
            PtxTerminatorSpec {
                kind: TerminatorKindAttr::Return,
                predicate: None,
                head: "ret",
                operands: Vec::new(),
                target_table: "",
                has_fallthrough: false,
            },
            std::iter::empty(),
        )
        .get_operation()
        .insert_at_back(exit, &ctx);

        assert_eq!(
            emit_canonical_module(&ctx, &module).unwrap(),
            "\
.version 9.3
.visible .entry kernel()
{
    L0:
    mov.u32 %r0, 1;
    Done:
    ret;
}
"
        );
    }

    #[test]
    fn derives_branch_text_from_successors_and_rejects_nonlocal_fallthrough() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let module = PtxModuleOp::build(&mut ctx);
        let callable = PtxCallableOp::build_cfg_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".entry kernel()",
        );
        callable
            .get_operation()
            .insert_at_back(module.body(&ctx), &ctx);
        let body = callable.cfg_body(&ctx).unwrap();
        let entry = body.append_block(&mut ctx);
        let skipped = body.append_block(&mut ctx);
        let target = body.append_block(&mut ctx);
        PtxTerminatorOp::build(
            &mut ctx,
            PtxTerminatorSpec {
                kind: TerminatorKindAttr::Branch,
                predicate: None,
                head: "bra",
                operands: Vec::new(),
                target_table: "",
                has_fallthrough: false,
            },
            [target],
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        for (block, label) in [(skipped, "Skipped"), (target, "Target")] {
            PtxLabelOp::build(&mut ctx, label)
                .get_operation()
                .insert_at_back(block, &ctx);
            PtxTerminatorOp::build(
                &mut ctx,
                PtxTerminatorSpec {
                    kind: TerminatorKindAttr::Return,
                    predicate: None,
                    head: "ret",
                    operands: Vec::new(),
                    target_table: "",
                    has_fallthrough: false,
                },
                std::iter::empty(),
            )
            .get_operation()
            .insert_at_back(block, &ctx);
        }
        let emitted = emit_canonical_module(&ctx, &module).unwrap();
        assert!(emitted.contains("bra Target;"));

        let invalid = PtxModuleOp::build(&mut ctx);
        let callable = PtxCallableOp::build_cfg_definition(
            &mut ctx,
            "invalid",
            CallableKindAttr::Entry,
            false,
            ".entry invalid()",
        );
        callable
            .get_operation()
            .insert_at_back(invalid.body(&ctx), &ctx);
        let body = callable.cfg_body(&ctx).unwrap();
        let first = body.append_block(&mut ctx);
        let next = body.append_block(&mut ctx);
        let wrong = body.append_block(&mut ctx);
        PtxTerminatorOp::fallthrough(&mut ctx, wrong)
            .get_operation()
            .insert_at_back(first, &ctx);
        for block in [next, wrong] {
            PtxTerminatorOp::build(
                &mut ctx,
                PtxTerminatorSpec {
                    kind: TerminatorKindAttr::Return,
                    predicate: None,
                    head: "ret",
                    operands: Vec::new(),
                    target_table: "",
                    has_fallthrough: false,
                },
                std::iter::empty(),
            )
            .get_operation()
            .insert_at_back(block, &ctx);
        }
        let error = emit_canonical_module(&ctx, &invalid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("fallthrough successor must be the next emitted block"));
    }
}
