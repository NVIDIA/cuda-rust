/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assertions must survive ordinary CFG cleanup and every copy of an unrolled
//! loop. A sole success edge does not make a potentially trapping operation an
//! unconditional branch.

mod common;

use common::{
    block, cond_br, counted_loop, func, goto, i1, iconst, mir_ctx, nested_counted_loop, ret,
    ret_values, u32t,
};
use dialect_mir::ops::{
    MirAssertOp, MirConstantOp, MirFuncOp, MirLtOp, MirReturnOp, MirUnrollHintOp,
};
use mir_transforms::{analyses::loop_info::LoopInfo, unroll::unroll_annotated_loops};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        ops::ConstantOp,
        types::FunctionType,
    },
    context::{Context, Ptr},
    graph::dominance::DomInfo,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::{Operation, verify_operation},
    opts::{constants::sccp::sccp, dce::dce, simplify_cfg::simplify_cfg},
    pass::AnalysisManager,
    region::Region,
    value::Value,
};

fn operations(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect()
}

fn assertions(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirAssertOp>(op, ctx).is_some())
        .collect()
}

fn constant(ctx: &Context, value: Value) -> Option<u64> {
    let op = value.defining_op()?;
    if let Some(constant) = Operation::get_op::<MirConstantOp>(op, ctx) {
        return constant.get_attr_value(ctx).map(|a| a.value().to_u64());
    }
    let attr = Operation::get_op::<ConstantOp>(op, ctx)?.get_value(ctx);
    (&*attr as &dyn Attribute)
        .downcast_ref::<IntegerAttr>()
        .map(|a| a.value().to_u64())
}

fn cleanup(ctx: &mut Context, module: Ptr<Operation>) {
    verify_operation(module, ctx).expect("valid input");
    simplify_cfg(module, ctx).unwrap();
    sccp(module, ctx).unwrap();
    simplify_cfg(module, ctx).unwrap();
    dce(module, ctx).unwrap();
    verify_operation(module, ctx).expect("valid optimized IR");
}

fn loop_count(ctx: &Context, region: Ptr<Region>) -> usize {
    let mut dom = DomInfo::default();
    LoopInfo::compute(ctx, region, dom.get_dom_tree(ctx, region))
        .loops()
        .len()
}

fn assert_before_terminator(ctx: &mut Context, block: Ptr<BasicBlock>, condition: Value) {
    let terminator = block.deref(ctx).get_terminator(ctx).unwrap();
    MirAssertOp::new(ctx, condition)
        .get_operation()
        .insert_before(ctx, terminator);
}

fn add_dynamic_condition(ctx: &mut Context, entry: Ptr<BasicBlock>) -> Value {
    let bool_type = i1(ctx);
    let argument = BasicBlock::push_argument(entry, ctx, bool_type.into());
    let function = entry.deref(ctx).get_parent_op(ctx).unwrap();
    let signature = FunctionType::get(ctx, vec![bool_type.into()], vec![]);
    Operation::get_op::<MirFuncOp>(function, ctx)
        .unwrap()
        .set_attr_mir_func_type(ctx, TypeAttr::new(signature.into()));
    entry.deref(ctx).get_argument(argument)
}

fn assert_less_than(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value, bound: Value) {
    let terminator = block.deref(ctx).get_terminator(ctx).unwrap();
    let bool_type = i1(ctx);
    let compare = Operation::new(
        ctx,
        MirLtOp::get_concrete_op_info(),
        vec![bool_type.into()],
        vec![value, bound],
        vec![],
        0,
    );
    compare.insert_before(ctx, terminator);
    let condition = compare.deref(ctx).get_result(0);
    assert_before_terminator(ctx, block, condition);
}

#[test]
fn cfg_cleanup_preserves_dynamic_true_and_false_assertions_and_forwarded_values() {
    let mut ctx = mir_ctx();
    let bool_type = i1(&mut ctx);
    let int_type = u32t(&mut ctx);
    let (module, region) = func(
        &mut ctx,
        vec![bool_type.into(), int_type.into()],
        vec![int_type.into()],
    );
    let entry = block(&mut ctx, region, vec![bool_type.into(), int_type.into()]);
    let second = block(&mut ctx, region, vec![int_type.into()]);
    let third = block(&mut ctx, region, vec![int_type.into()]);
    let condition = entry.deref(&ctx).get_argument(0);
    let payload = entry.deref(&ctx).get_argument(1);
    MirAssertOp::new(&mut ctx, condition)
        .get_operation()
        .insert_at_back(entry, &ctx);
    goto(&mut ctx, entry, second, vec![payload]);
    let always = iconst(&mut ctx, second, bool_type, 1);
    MirAssertOp::new(&mut ctx, always)
        .get_operation()
        .insert_at_back(second, &ctx);
    let forwarded = second.deref(&ctx).get_argument(0);
    goto(&mut ctx, second, third, vec![forwarded]);
    let never = iconst(&mut ctx, third, bool_type, 0);
    MirAssertOp::new(&mut ctx, never)
        .get_operation()
        .insert_at_back(third, &ctx);
    let forwarded = third.deref(&ctx).get_argument(0);
    ret_values(&mut ctx, third, vec![forwarded]);

    cleanup(&mut ctx, module);

    assert_eq!(
        region.deref(&ctx).iter(&ctx).count(),
        1,
        "gotos still merge"
    );
    let guards = assertions(&ctx, region);
    assert_eq!(
        guards.len(),
        3,
        "trapping effects survive every cleanup pass"
    );
    assert_eq!(guards[0].deref(&ctx).get_operand(0), condition);
    assert_eq!(
        constant(&ctx, guards[1].deref(&ctx).get_operand(0)),
        Some(1)
    );
    assert_eq!(
        constant(&ctx, guards[2].deref(&ctx).get_operand(0)),
        Some(0)
    );
    let terminator = entry.deref(&ctx).get_terminator(&ctx).unwrap();
    assert!(Operation::get_op::<MirReturnOp>(terminator, &ctx).is_some());
    assert_eq!(terminator.deref(&ctx).get_operand(0), payload);
}

#[test]
fn unreachable_assertion_is_removed_but_reachable_assertion_stays() {
    let mut ctx = mir_ctx();
    let bool_type = i1(&mut ctx);
    let (module, region) = func(&mut ctx, vec![bool_type.into()], vec![]);
    let entry = block(&mut ctx, region, vec![bool_type.into()]);
    let live = block(&mut ctx, region, vec![]);
    let dead = block(&mut ctx, region, vec![]);
    let dynamic = entry.deref(&ctx).get_argument(0);
    let always = iconst(&mut ctx, entry, bool_type, 1);
    cond_br(&mut ctx, entry, always, live, dead);
    MirAssertOp::new(&mut ctx, dynamic)
        .get_operation()
        .insert_at_back(live, &ctx);
    ret(&mut ctx, live);
    let never = iconst(&mut ctx, dead, bool_type, 0);
    MirAssertOp::new(&mut ctx, never)
        .get_operation()
        .insert_at_back(dead, &ctx);
    ret(&mut ctx, dead);

    cleanup(&mut ctx, module);
    let guards = assertions(&ctx, region);
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].deref(&ctx).get_operand(0), dynamic);
}

#[test]
fn full_partial_and_no_unroll_preserve_assertions_before_inside_and_after_loop() {
    for (factor, body_copies, loops) in [(None, 1, 1), (Some(0), 5, 0), (Some(2), 3, 2)] {
        let mut ctx = mir_ctx();
        let lp = counted_loop(&mut ctx, 5);
        let condition = add_dynamic_condition(&mut ctx, lp.preheader);
        for block in [lp.preheader, lp.latch, lp.exit] {
            assert_before_terminator(&mut ctx, block, condition);
        }
        if let Some(factor) = factor {
            MirUnrollHintOp::new(&mut ctx, factor)
                .get_operation()
                .insert_at_front(lp.latch, &ctx);
        }
        verify_operation(lp.module, &ctx).unwrap();
        unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default()).unwrap();
        verify_operation(lp.module, &ctx).unwrap();

        assert_eq!(loop_count(&ctx, lp.region), loops, "factor {factor:?}");
        let guards = assertions(&ctx, lp.region);
        assert_eq!(guards.len(), body_copies + 2, "factor {factor:?}");
        for guard in guards {
            assert_eq!(guard.deref(&ctx).get_operand(0), condition);
        }
    }
}

#[test]
fn full_unroll_keeps_assertions_on_each_accumulator_value_including_failure() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);
    // The header bound is 4. Accumulator values on entry to successive body
    // copies are 0, 0, 1, 3; asserting acc < 3 must fail in the fourth copy.
    let int_type = u32t(&mut ctx);
    let bound = iconst(&mut ctx, lp.preheader, int_type, 3);
    let bound_op = bound.defining_op().unwrap();
    bound_op.unlink(&ctx);
    let preheader_term = lp.preheader.deref(&ctx).get_terminator(&ctx).unwrap();
    bound_op.insert_before(&ctx, preheader_term);
    let accumulator = lp.header.deref(&ctx).get_argument(0);
    assert_less_than(&mut ctx, lp.latch, accumulator, bound);
    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);

    verify_operation(lp.module, &ctx).unwrap();
    unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default()).unwrap();
    verify_operation(lp.module, &ctx).unwrap();
    assert_eq!(loop_count(&ctx, lp.region), 0);
    let outcomes: Vec<_> = assertions(&ctx, lp.region)
        .into_iter()
        .map(|op| constant(&ctx, op.deref(&ctx).get_operand(0)))
        .collect();
    assert_eq!(outcomes, vec![Some(1), Some(1), Some(1), Some(0)]);
}

#[test]
fn nested_full_unroll_preserves_all_assertions() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);
    let condition = add_dynamic_condition(&mut ctx, lp.preheader);
    assert_before_terminator(&mut ctx, lp.inner_body, condition);
    for block in [lp.outer_body, lp.inner_body] {
        MirUnrollHintOp::new(&mut ctx, 0)
            .get_operation()
            .insert_at_front(block, &ctx);
    }
    verify_operation(lp.module, &ctx).unwrap();
    unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default()).unwrap();
    verify_operation(lp.module, &ctx).unwrap();
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(assertions(&ctx, lp.region).len(), 6);
}

#[test]
fn an_assertion_in_the_loop_header_prevents_bypassing_header_work() {
    for factor in [0, 2] {
        let mut ctx = mir_ctx();
        let lp = counted_loop(&mut ctx, 4);
        let condition = add_dynamic_condition(&mut ctx, lp.preheader);
        assert_before_terminator(&mut ctx, lp.header, condition);
        MirUnrollHintOp::new(&mut ctx, factor)
            .get_operation()
            .insert_at_front(lp.latch, &ctx);
        verify_operation(lp.module, &ctx).unwrap();
        unroll_annotated_loops(lp.module, &mut ctx, &mut AnalysisManager::default()).unwrap();
        verify_operation(lp.module, &ctx).unwrap();
        assert_eq!(loop_count(&ctx, lp.region), 1);
        let guards = assertions(&ctx, lp.region);
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].deref(&ctx).get_parent_block(), Some(lp.header));
    }
}
