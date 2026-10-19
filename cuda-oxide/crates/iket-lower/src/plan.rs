/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Whole-IR instrumentation planning.

use crate::{
    event_name::{EncodedEventName, EventNameError, EventNameTable},
    method::{
        IketCompatibilityProfile, InstrumentMethod, InstrumentMethodPolicy, MethodSelectionError,
        select_instrument_method,
    },
};
use dialect_iket::ops::{IketMarkOp, IketRangePushOp, IketRangeStartOp};
use pliron::{
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
};
use std::collections::BTreeSet;
use thiserror::Error;

/// Immutable decisions shared by all IKET sites in one compiler root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IketLoweringPlan {
    pub instrument_method: InstrumentMethod,
    /// One entry per unique user event name, sorted for deterministic output.
    pub event_names: Vec<EncodedEventName>,
}

impl IketLoweringPlan {
    pub fn unique_user_event_count(&self) -> usize {
        self.event_names.len()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LoweringPlanError {
    #[error(transparent)]
    EventName(#[from] EventNameError),
    #[error(transparent)]
    MethodSelection(#[from] MethodSelectionError),
}

/// Analyze a complete operation tree before physical IKET lowering.
///
/// Repeated sites with the same name consume one user event ID. Range-pop is
/// excluded because IKET assigns it a reserved ID, while sentinel tokens do
/// not represent trace events at all.
pub fn plan_instrumentation(
    ctx: &Context,
    root: Ptr<Operation>,
    profile: IketCompatibilityProfile,
    policy: InstrumentMethodPolicy,
) -> Result<IketLoweringPlan, LoweringPlanError> {
    let mut unique_names = BTreeSet::new();
    collect_event_names(ctx, root, &mut unique_names);

    let instrument_method = select_instrument_method(profile, policy, unique_names.len())?;
    let mut event_name_table = EventNameTable::new(profile);
    let event_names = unique_names
        .into_iter()
        .map(|name| event_name_table.insert(&name))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(IketLoweringPlan {
        instrument_method,
        event_names,
    })
}

fn collect_event_names(ctx: &Context, operation: Ptr<Operation>, names: &mut BTreeSet<String>) {
    if let Some(name) = named_event(ctx, operation) {
        names.insert(name);
    }

    let regions = operation.deref(ctx).regions().collect::<Vec<_>>();
    for region in regions {
        let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();
        for block in blocks {
            let operations = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
            for nested in operations {
                collect_event_names(ctx, nested, names);
            }
        }
    }
}

fn named_event(ctx: &Context, operation: Ptr<Operation>) -> Option<String> {
    if let Some(op) = Operation::get_op::<IketMarkOp>(operation, ctx) {
        op.event_name(ctx)
    } else if let Some(op) = Operation::get_op::<IketRangeStartOp>(operation, ctx) {
        op.event_name(ctx)
    } else if let Some(op) = Operation::get_op::<IketRangePushOp>(operation, ctx) {
        op.event_name(ctx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::IKET_COMPATIBILITY_PROFILE;
    use dialect_iket::{attributes::IketPayloadKindAttr, ops::IketRangePopOp};
    use pliron::{basic_block::BasicBlock, builtin::ops::ModuleOp, op::Op};

    fn module_with_marks(ctx: &mut Context, names: &[String]) -> Ptr<Operation> {
        dialect_iket::register(ctx);
        let module = ModuleOp::new(ctx, "iket_test".try_into().unwrap());
        let region = module.get_operation().deref(ctx).get_region(0);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        for name in names {
            IketMarkOp::new(ctx, name, IketPayloadKindAttr::None, None)
                .get_operation()
                .insert_at_back(block, ctx);
        }
        module.get_operation()
    }

    #[test]
    fn plan_counts_unique_names_not_dynamic_sites() {
        let mut ctx = Context::new();
        let root = module_with_marks(
            &mut ctx,
            &["mma".to_string(), "mma".to_string(), "tma".to_string()],
        );
        let plan = plan_instrumentation(
            &ctx,
            root,
            IKET_COMPATIBILITY_PROFILE,
            InstrumentMethodPolicy::Auto,
        )
        .unwrap();
        assert_eq!(plan.unique_user_event_count(), 2);
        assert_eq!(plan.instrument_method, InstrumentMethod::NativeDump);
    }

    #[test]
    fn plan_switches_the_whole_module_after_thirty_names() {
        let mut ctx = Context::new();
        let names = (0..31)
            .map(|index| format!("event_{index}"))
            .collect::<Vec<_>>();
        let root = module_with_marks(&mut ctx, &names);
        let plan = plan_instrumentation(
            &ctx,
            root,
            IKET_COMPATIBILITY_PROFILE,
            InstrumentMethodPolicy::Auto,
        )
        .unwrap();
        assert_eq!(plan.instrument_method, InstrumentMethod::ExtendedNativeDump);
    }

    #[test]
    fn range_pop_does_not_consume_a_user_event_id() {
        let mut ctx = Context::new();
        let root = module_with_marks(&mut ctx, &[]);
        let region = root.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        IketRangePopOp::new(&mut ctx)
            .get_operation()
            .insert_at_back(block, &ctx);
        let plan = plan_instrumentation(
            &ctx,
            root,
            IKET_COMPATIBILITY_PROFILE,
            InstrumentMethodPolicy::Auto,
        )
        .unwrap();
        assert_eq!(plan.unique_user_event_count(), 0);
    }

    #[test]
    fn plan_encodes_long_names_with_the_cuda_cpp_placeholder() {
        let mut ctx = Context::new();
        let root = module_with_marks(&mut ctx, &["x".repeat(32)]);
        let plan = plan_instrumentation(
            &ctx,
            root,
            IKET_COMPATIBILITY_PROFILE,
            InstrumentMethodPolicy::Auto,
        )
        .unwrap();
        assert!(plan.event_names[0].uses_hash_placeholder);
    }
}
