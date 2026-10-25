/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prove prepared Block library calls and lower their finalized native ABI.

use dialect_mir::{
    attributes::MirCastKindAttr,
    library::{LibraryCallPlan, LibraryCallSite, LibraryRoleAttr},
    ops::{MirCastOp, MirLibraryOp, MirLibrarySourceOp, MirLibraryWriteCheckOp},
};
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
mod context_plans;
mod contexts;
mod planned;
mod stateful;
pub use admission::{admit_library_roots, capture_library_roles};
pub use context_plans::{LibraryAssembly, RootedNativePlans, assemble_library_plans};
pub use contexts::{materialize_library_contexts, resolve_library_call_target};
pub use stateful::{Validation, analyze_libraries};
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
    for &op in &retained {
        op.deref(ctx).verify(ctx)?;
        let site = (*MirLibraryOp::new(op).get_attr_site(ctx).unwrap()).clone();
        if !sites.insert(site.clone()) {
            return pliron::verify_err!(
                op.deref(ctx).loc(),
                "duplicate retained library source site"
            );
        }
        let Some(plan) = plans.get(&site) else {
            return pliron::verify_err!(
                op.deref(ctx).loc(),
                "retained native operation has no finalized plan"
            );
        };
        planned::validate(ctx, op, plan)?;
    }
    if retained.is_empty() && events.is_empty() {
        return Ok(());
    }
    let mut invalidated = PassResult::default();
    invalidated.ir_changed = IRStatus::Changed;
    analyses.retain_preserved(&invalidated);
    for op in retained {
        let site = (*MirLibraryOp::new(op).get_attr_site(ctx).unwrap()).clone();
        planned::lower(ctx, op, &plans[&site]);
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
