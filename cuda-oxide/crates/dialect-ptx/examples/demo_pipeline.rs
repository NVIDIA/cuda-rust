/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tour of the structured PTX dialect: build a kernel with a
//! counting loop, show the typed IR (including the typed guard predicate on
//! the loop terminator), emit canonical PTX, prove the emitted text
//! re-parses losslessly with ptx-parse, then raise the same text into the
//! native Pliron CFG and show the loop block's ptx.terminator carrying the
//! typed predicate and real successor edges.

use dialect_ptx::attributes::{PredicateAttr, TerminatorKindAttr};
use dialect_ptx::ops::{PtxInstructionOp, PtxTerminatorOp};
use dialect_ptx::raising::NativeCfgPlan;
use dialect_ptx::{PtxBuilder, emit_canonical_module};
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ctx = Context::new();
    dialect_ptx::register(&mut ctx);

    let mut builder = PtxBuilder::new(&mut ctx);
    builder.version("8.9").target("sm_120a").address_size(64);
    let kernel = builder.visible_entry("demo_kernel", "()", |body| {
        body.directive(".reg", ".pred %p<2>;");
        body.directive(".reg", ".b32 %r<3>;");
        body.instruction("mov.u32", ["%r1", "0"]);
        body.label("$L_loop");
        body.instruction("add.u32", ["%r1", "%r1", "1"]);
        body.instruction("setp.lt.u32", ["%p1", "%r1", "16"]);
        body.predicated_instruction(PredicateAttr::new("%p1", false), "bra", ["$L_loop"]);
        body.instruction("ret", std::iter::empty::<&str>());
    });
    let module = builder.finish();

    println!("== typed IR ==");
    println!("{}", module.get_operation().disp(&ctx));

    let terminator = kernel
        .entry_block(&ctx)
        .expect("definition has a body")
        .deref(&ctx)
        .iter(&ctx)
        .filter(|operation| Operation::is_op::<PtxInstructionOp>(*operation, &ctx))
        .nth(3)
        .expect("loop back-edge instruction");
    println!("== loop terminator ==");
    println!("{}", terminator.disp(&ctx));

    let emitted = emit_canonical_module(&ctx, &module)?;
    println!("== canonical PTX ==");
    print!("{emitted}");

    let reparsed = ptx_parse::Document::parse(&emitted)?;
    if !reparsed.coverage().is_complete() {
        return Err(format!(
            "canonical PTX is structurally incomplete: {:?}",
            reparsed.coverage()
        )
        .into());
    }
    let back_edge = reparsed
        .instructions()
        .iter()
        .find(|instruction| instruction.head() == "bra")
        .expect("emitted kernel keeps its loop");
    let predicate = back_edge.predicate().expect("back-edge stays guarded");
    assert_eq!(predicate.register(), "%p1");
    assert!(!predicate.is_negated());
    println!("== round trip ==");
    println!(
        "back-edge guard survives: @{}{}",
        if predicate.is_negated() { "!" } else { "" },
        predicate.register()
    );

    // Raise the emitted text into the native Pliron CFG: labels become real
    // basic blocks and the guarded back-edge becomes a ptx.terminator whose
    // successors are CFG edges rather than label text.
    let mut cfg_ctx = Context::new();
    dialect_ptx::register(&mut cfg_ctx);
    let raised = NativeCfgPlan::analyze(&emitted)?.materialize(&mut cfg_ctx);
    println!("== native CFG ==");
    println!("{}", raised.module().get_operation().disp(&cfg_ctx));
    let loop_terminator = raised
        .blocks()
        .iter()
        .find_map(|block| {
            let terminator = block.block().deref(&cfg_ctx).iter(&cfg_ctx).last()?;
            let terminator = Operation::get_op::<PtxTerminatorOp>(terminator, &cfg_ctx)?;
            (terminator.kind(&cfg_ctx) == TerminatorKindAttr::Branch
                && terminator.predicate(&cfg_ctx).is_some())
            .then_some(terminator)
        })
        .expect("raised kernel keeps its guarded loop back-edge");
    println!("== raised loop-block terminator ==");
    println!("{}", loop_terminator.get_operation().disp(&cfg_ctx));
    let reemitted = emit_canonical_module(&cfg_ctx, &raised.module())?;
    assert_eq!(reemitted, emitted);
    println!("== native CFG round trip ==");
    println!("canonical emission from the raised CFG is byte-identical");
    Ok(())
}
