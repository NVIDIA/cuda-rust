/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prove prepared collective calls, place storage, and lower their finalized ABI.

use dialect_mir::{
    attributes::MirCastKindAttr,
    library::{LibraryCallPlan, LibraryCallSite, LibraryRoleAttr},
    ops::{MirCastOp, MirLibraryOp, MirLibrarySourceOp, MirLibraryWriteCheckOp},
};
use pliron::linked_list::ContainsLinkedList;
use pliron::{
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op},
    irbuild::IRStatus,
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, PassResult},
    result::Result,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};

mod admission;
mod callback;
mod context_plans;
mod contexts;
mod planned;
mod stateful;
mod storage;
pub use admission::{admit_library_roots, capture_library_roles};
pub use callback::admit_library_callbacks;
pub use context_plans::{LibraryAssembly, assemble_library_plans};
pub use contexts::{materialize_library_contexts, resolve_library_call_target};
pub use stateful::analyze_libraries;
pub use storage::{RootedNativePlans, Validation, allocate_library_storage};
#[cfg(test)]
mod admission_tests;

/// Consume proof events before optional transforms. An unbound native call fails.
pub fn lower_library_operations(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
) -> Result<()> {
    lower(module, ctx, analyses, &BTreeMap::new())
}

/// Bind retained native calls to their authenticated provider results.
/// Validate every source site and ABI before rewriting; preserve source locations.
pub fn lower_planned_library_operations(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    plans: &BTreeMap<LibraryCallSite, LibraryCallPlan>,
) -> Result<()> {
    lower(module, ctx, analyses, plans)
}

fn lower(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    plans: &BTreeMap<LibraryCallSite, LibraryCallPlan>,
) -> Result<()> {
    let mut operations = (Vec::new(), Vec::new());
    walk_op(
        ctx,
        &mut operations,
        &WalkConfig::default(),
        module,
        |ctx, (retained, events), node| {
            if let IRNode::Operation(op) = node {
                if Operation::get_op::<MirLibraryOp>(op, ctx).is_some() {
                    retained.push(op);
                } else if Operation::get_op::<MirLibrarySourceOp>(op, ctx).is_some()
                    || Operation::get_op::<MirLibraryWriteCheckOp>(op, ctx).is_some()
                {
                    events.push(op);
                }
            }
        },
    );
    let (retained, events) = operations;
    for &op in &events {
        op.deref(ctx).verify(ctx)?;
    }
    let mut sites = BTreeSet::new();
    let mut boundaries = BTreeMap::new();
    let mut natives = BTreeMap::new();
    for &op in &retained {
        op.deref(ctx).verify(ctx)?;
        let library = MirLibraryOp::new(op);
        let site = (*library.get_attr_site(ctx).unwrap()).clone();
        let role = *library.get_attr_role(ctx).unwrap();
        if role == LibraryRoleAttr::NativeOperandBoundary {
            if boundaries.insert(site.clone(), op).is_some() {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "duplicate native operand boundary"
                );
            }
        } else if !sites.insert(site.clone()) {
            return pliron::verify_err!(
                op.deref(ctx).loc(),
                "duplicate retained library source site"
            );
        }
        if matches!(role, LibraryRoleAttr::Native(_)) {
            natives.insert(site.clone(), op);
        }
        if planned::operation(role).is_some() {
            let Some(plan) = plans.get(&site) else {
                return pliron::verify_err!(
                    op.deref(ctx).loc(),
                    "retained native/storage operation has no finalized plan"
                );
            };
            planned::validate(ctx, op, plan)?;
        }
    }
    for (site, &native) in &natives {
        let Some(&boundary) = boundaries.get(site) else {
            return pliron::verify_err!(
                native.deref(ctx).loc(),
                "native operation is missing its operand boundary"
            );
        };
        let same_callee = *MirLibraryOp::new(boundary)
            .get_attr_library_callee(ctx)
            .unwrap()
            == *MirLibraryOp::new(native)
                .get_attr_library_callee(ctx)
                .unwrap();
        let Some(block) = native.deref(ctx).get_parent_block() else {
            return pliron::verify_err!(native.deref(ctx).loc(), "native operation is detached");
        };
        if !same_callee
            || boundary.deref(ctx).get_parent_block() != Some(block)
            || !block
                .deref(ctx)
                .iter(ctx)
                .take_while(|&op| op != native)
                .any(|op| op == boundary)
        {
            return pliron::verify_err!(
                native.deref(ctx).loc(),
                "native operand boundary must precede its matching call in the same block"
            );
        }
    }
    for (site, &boundary) in &boundaries {
        if !natives.contains_key(site) {
            return pliron::verify_err!(
                boundary.deref(ctx).loc(),
                "native operand boundary is missing its call"
            );
        }
    }
    if retained.is_empty() && events.is_empty() {
        return Ok(());
    }
    let mut invalidated = PassResult::default();
    invalidated.ir_changed = IRStatus::Changed;
    analyses.retain_preserved(&invalidated);
    for op in retained {
        let site = (*MirLibraryOp::new(op).get_attr_site(ctx).unwrap()).clone();
        let role = *MirLibraryOp::new(op).get_attr_role(ctx).unwrap();
        let result = if let LibraryRoleAttr::WarpGroupIndex { width } = role {
            Some(planned::lower_warp_query(ctx, op, width))
        } else if let LibraryRoleAttr::WarpPrefixMember { width } = role {
            Some(planned::lower_warp_prefix_query(ctx, op, width))
        } else if let LibraryRoleAttr::SyncWarp { width } = role {
            planned::lower_warp_sync(ctx, op, width);
            None
        } else if role == LibraryRoleAttr::NativeOperandBoundary {
            planned::lower_boundary(ctx, op, &plans[&site]);
            None
        } else {
            planned::lower(ctx, op, &plans[&site])
        };
        if let Some(result) = result {
            let original = op.deref(ctx).get_result(0);
            original.replace_all_uses_with(ctx, &result);
        }
        Operation::erase(op, ctx);
    }
    for op in events {
        Operation::erase(op, ctx);
    }
    Ok(())
}

fn emit<O: Op>(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
    let loc = anchor.deref(ctx).loc().clone();
    op.deref_mut(ctx).set_loc(loc);
    op.insert_before(ctx, anchor);
    op
}
