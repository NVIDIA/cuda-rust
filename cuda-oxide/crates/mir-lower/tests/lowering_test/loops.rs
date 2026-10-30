/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::{HashMap, HashSet};

use dialect_mir::ops as mir;
use llvm_export::export::{NvvmExportConfig, NvvmIrDialect};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::region::Region;
use pliron::value::Value;

use super::common::{append_return, build_test_kernel, make_test_ctx};
use super::wgmma_lowering::append_mir_unsigned_constant;

/// Append an op of `$op` with one `$result` to `$block` and return the result.
/// A macro because pliron's op-info type cannot be named outside pliron.
macro_rules! append_op {
    ($ctx:expr, $block:expr, $op:ty, $result:expr, $operands:expr) => {{
        let op = Operation::new(
            $ctx,
            <$op>::get_concrete_op_info(),
            vec![$result],
            $operands,
            vec![],
            0,
        );
        op.insert_at_back($block, $ctx);
        op.deref($ctx).get_result(0)
    }};
}

/// Append `while i < n { i += 1 }` after `from`, with a `mir.nounroll_hint`
/// first in the body when `nounroll` is set, and return the empty exit block.
fn append_counted_loop(
    ctx: &mut Context,
    region: Ptr<Region>,
    from: Ptr<BasicBlock>,
    n: Value,
    nounroll: bool,
) -> Ptr<BasicBlock> {
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let header = BasicBlock::new(ctx, None, vec![u32_ty.into()]);
    header.insert_at_back(region, ctx);
    let body = BasicBlock::new(ctx, None, vec![]);
    body.insert_at_back(region, ctx);
    let exit = BasicBlock::new(ctx, None, vec![]);
    exit.insert_at_back(region, ctx);

    let start = append_mir_unsigned_constant(ctx, from, u32_ty, 0);
    goto(ctx, from, header, vec![start]);

    let i = header.deref(ctx).get_argument(0);
    let lt = append_op!(ctx, header, mir::MirLtOp, i1_ty.into(), vec![i, n]);
    let done = append_op!(ctx, header, mir::MirNotOp, i1_ty.into(), vec![lt]);
    let (operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![done], vec![], vec![]]);
    let branch = Operation::new(
        ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![exit, body],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(branch, ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(ctx, segment_sizes);
    branch.insert_at_back(header, ctx);

    if nounroll {
        mir::MirNoUnrollHintOp::new(ctx)
            .get_operation()
            .insert_at_back(body, ctx);
    }
    let one = append_mir_unsigned_constant(ctx, body, u32_ty, 1);
    let next = append_op!(ctx, body, mir::MirAddOp, u32_ty.into(), vec![i, one]);
    goto(ctx, body, header, vec![next]);
    exit
}

fn goto(ctx: &mut Context, from: Ptr<BasicBlock>, to: Ptr<BasicBlock>, operands: Vec<Value>) {
    Operation::new(
        ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![to],
        0,
    )
    .insert_at_back(from, ctx);
}

/// Check the loop metadata in exported LLVM IR: `marked` latch branches, each
/// with its own `distinct` self-referencing node, one shared
/// `llvm.loop.unroll.disable` node, and no metadata ID defined twice.
fn assert_loop_metadata(ir: &str, marked: usize) {
    let mut definitions: HashMap<&str, &str> = HashMap::new();
    for line in ir.lines() {
        if let Some((id, value)) = line
            .strip_prefix('!')
            .and_then(|rest| rest.split_once(" = "))
            && id.chars().all(|c| c.is_ascii_digit())
        {
            assert!(
                definitions.insert(id, value).is_none(),
                "!{id} defined twice:\n{ir}"
            );
        }
    }

    let loop_ids: Vec<&str> = ir
        .lines()
        .filter(|line| line.trim_start().starts_with("br "))
        .filter_map(|line| line.split_once("!llvm.loop !"))
        .map(|(_, rest)| rest.split(|c: char| !c.is_ascii_digit()).next().unwrap())
        .collect();
    assert_eq!(loop_ids.len(), marked, "marked latch branches:\n{ir}");
    assert_eq!(
        loop_ids.iter().collect::<HashSet<_>>().len(),
        marked,
        "every marked loop needs its own node:\n{ir}"
    );

    let properties: Vec<&str> = definitions
        .iter()
        .filter(|(_, value)| **value == "!{!\"llvm.loop.unroll.disable\"}")
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(properties.len(), 1, "one shared property node:\n{ir}");
    for id in loop_ids {
        assert_eq!(
            definitions.get(id).copied(),
            Some(format!("distinct !{{!{id}, !{}}}", properties[0]).as_str()),
            "loop node !{id}:\n{ir}"
        );
    }
}

/// Lowering keeps the hinted loops' latch marks, and the exporter emits valid
/// loop metadata in both the PTX and the legacy NVVM dialects, alongside the
/// other metadata that shares the module's ID counter.
#[test]
fn nounroll_hints_become_loop_metadata_on_their_latches() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![u32_ty.into()]);
    let n = entry.deref(&ctx).get_argument(0);
    let region = entry
        .deref(&ctx)
        .get_parent_region()
        .expect("entry in a region");

    // hinted, unhinted, hinted
    let mut tail = entry;
    for nounroll in [true, false, true] {
        tail = append_counted_loop(&mut ctx, region, tail, n, nounroll);
    }
    append_return(&mut ctx, tail);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ptx_ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert_loop_metadata(&ptx_ir, 2);

    let nvvm_ir = llvm_export::export::export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .map_err(|error| anyhow::anyhow!(error))?;
    assert!(nvvm_ir.contains("!nvvmir.version"), "{nvvm_ir}");
    assert_loop_metadata(&nvvm_ir, 2);
    Ok(())
}
