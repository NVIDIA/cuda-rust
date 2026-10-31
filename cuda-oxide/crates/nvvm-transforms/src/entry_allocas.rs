/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Normalize fixed scratch slots after LLVM dialect legalization for NVVM.
//!
//! Non-entry allocas can introduce dynamic-stack restoration when helpers are
//! inlined. Moving eligible storage into entry lets NVVM use static slots.
//! Initialization and memory operations stay at their original control-flow sites.

use std::collections::HashSet;

use llvm_export::ops::{self, FuncOp};
use pliron::{
    builtin::attributes::IntegerAttr,
    context::{Context, Ptr},
    graph::traversals::region::sccs_in_topological_order,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};

/// Normalize every LLVM function in the module, leaving declarations alone.
/// The function-local analysis runs after all other NVVM compatibility rewrites
/// and updates the same IR that the pipeline subsequently verifies and exports.
pub(super) fn hoist_fixed_allocas(ctx: &mut Context, module: Ptr<Operation>) {
    let functions: Vec<_> = module
        .deref(ctx)
        .regions()
        .flat_map(|region| region.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .filter_map(|op| Operation::get_op::<FuncOp>(op, ctx))
        .collect();
    for function in functions {
        hoist_function_allocas(ctx, &function);
    }
}

/// A fixed slot outside a cycle executes at most once; its address stays unique.
/// Inside a cycle, reuse is allowed only when no iteration can retain or observe
/// the address. Dynamic counts and potentially escaping loop slots stay in place.
fn hoist_function_allocas(ctx: &mut Context, function: &FuncOp) {
    let Some(entry) = function.get_entry_block(ctx) else {
        return;
    };
    let region = function.get_operation().deref(ctx).get_region(0);
    let mut candidates = Vec::new();
    for block in region.deref(ctx).iter(ctx) {
        if block == entry {
            continue;
        }
        for allocation in block.deref(ctx).iter(ctx) {
            if Operation::get_op::<ops::AllocaOp>(allocation, ctx).is_none() {
                continue;
            }
            let Some(count) = allocation.deref(ctx).get_operand(0).defining_op() else {
                continue;
            };
            let Some(constant) = Operation::get_op::<ops::ConstantOp>(count, ctx) else {
                continue;
            };
            let attribute = constant.get_value(ctx);
            let Some(count) =
                (&*attribute as &dyn pliron::attribute::Attribute).downcast_ref::<IntegerAttr>()
            else {
                continue;
            };
            candidates.push((block, allocation, count.clone()));
        }
    }
    if candidates.is_empty() {
        return;
    }
    let cyclic_blocks: HashSet<_> = sccs_in_topological_order(ctx, &region)
        .into_iter()
        .filter(|component| component.is_cyclic)
        .flat_map(|component| component.nodes)
        .collect();
    let eligible: Vec<_> = candidates
        .into_iter()
        .filter(|(block, op, _)| {
            !cyclic_blocks.contains(block)
                || pointer_is_local_scratch(ctx, op.deref(ctx).get_result(0))
        })
        .collect();
    let anchor = entry.deref(ctx).iter(ctx).next();
    for (_, allocation, count) in eligible {
        // The count's old definition may be in a conditional block. Give the
        // moved allocation its own entry constant rather than violating SSA
        // dominance or moving a definition that other body operations still use.
        let constant = ops::ConstantOp::new(ctx, Box::new(count));
        if let Some(anchor) = anchor {
            constant.get_operation().insert_before(ctx, anchor);
        } else {
            constant.get_operation().insert_at_back(entry, ctx);
        }
        let count = constant.get_operation().deref(ctx).get_result(0);
        Operation::replace_operand(allocation, ctx, 0, count);
        // Move the original operation so alignment, source location, and debug
        // provenance survive. Stores and lifetime operations are not moved.
        allocation.unlink(ctx);
        if let Some(anchor) = anchor {
            allocation.insert_before(ctx, anchor);
        } else {
            allocation.insert_at_back(entry, ctx);
        }
    }
}

/// Follow pointer views, accepting only loads and stores through the address.
/// Calls, block arguments, comparisons/conversions, returns, and storing the
/// pointer as data can reveal allocation identity across iterations and stop us.
fn pointer_is_local_scratch(ctx: &Context, pointer: Value) -> bool {
    let mut pending = vec![pointer];
    let mut visited = HashSet::new();
    while let Some(pointer) = pending.pop() {
        if !visited.insert(pointer) {
            continue;
        }
        for usage in pointer.uses(ctx) {
            let operation = usage.user_op();
            let index = usage.find_index(ctx);
            let object = Operation::get_op_dyn(operation, ctx);
            let op = object.as_ref();
            if (op.is::<ops::LoadOp>() && index == 0) || (op.is::<ops::StoreOp>() && index == 1) {
                continue;
            }
            if index == 0
                && (op.is::<ops::GetElementPtrOp>()
                    || op.is::<ops::BitcastOp>()
                    || op.is::<ops::AddrSpaceCastOp>())
            {
                pending.push(operation.deref(ctx).get_result(0));
                continue;
            }
            return false;
        }
    }
    true
}
