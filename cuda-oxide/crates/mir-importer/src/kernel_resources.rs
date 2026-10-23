/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Finalized requirements for fixed views of the dynamic shared-memory base.
//! This is not an allocator: fixed views alias and combine by maximum extent.

use dialect_mir::ops::{MirCallOp, MirExternSharedOp, MirFuncOp};
use oxide_artifacts::KernelLaunchFacts;
use pliron::builtin::{attributes::IntegerAttr, op_interfaces::SymbolOpInterface};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const FIXED_SHARED_BYTES: &str = "cuda_oxide_fixed_shared_bytes";
pub(crate) const INFERRED_SHARED: &str = "cuda_oxide_inferred_shared";

fn integer_attr(ctx: &Context, op: Ptr<Operation>, name: &str) -> Option<u64> {
    let key: pliron::identifier::Identifier = name.try_into().unwrap();
    op.deref(ctx)
        .attributes
        .get::<IntegerAttr>(&key)
        .map(|value| value.value().to_u64())
}

/// Apply the exact geometry required by reachable native library operations.
/// Scratch extents are recorded on their actual shared-memory accesses and
/// collected together with ordinary Rust views below.
pub(crate) fn apply_library_block(
    ctx: &mut Context,
    entry: Ptr<Operation>,
    plan: &KernelLaunchFacts,
) -> Result<(), String> {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;

    let Some((x, y, z)) = plan.exact_block else {
        return Err("native library root is missing its exact block shape".into());
    };
    let required = u64::from(x) * u64::from(y) * u64::from(z);
    if let Some(maximum) = integer_attr(ctx, entry, "maxntid")
        && required > maximum
    {
        return Err(format!(
            "native library exact block ({x}, {y}, {z}) exceeds launch_bounds maximum {maximum}"
        ));
    }
    let axes = [("reqntid_x", x), ("reqntid_y", y), ("reqntid_z", z)];
    for (name, dimension) in axes {
        if let Some(existing) = integer_attr(ctx, entry, name)
            && existing != u64::from(dimension)
        {
            return Err(format!(
                "native library exact block ({x}, {y}, {z}) conflicts with kernel launch contract"
            ));
        }
    }
    let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    for (name, dimension) in axes {
        entry.deref_mut(ctx).attributes.set(
            name.try_into().unwrap(),
            IntegerAttr::new(
                ty,
                APInt::from_u64(
                    u64::from(dimension),
                    std::num::NonZeroUsize::new(64).unwrap(),
                ),
            ),
        );
    }
    Ok(())
}

#[derive(Default)]
struct FunctionResources {
    inferred: bool,
    fixed: bool,
    unconstrained: bool,
    bytes: u64,
    alignment: u64,
    callees: Vec<String>,
}

fn scan(ctx: &Context, op: Ptr<Operation>, resources: &mut FunctionResources) {
    if let Some(shared) = Operation::get_op::<MirExternSharedOp>(op, ctx) {
        match integer_attr(ctx, op, FIXED_SHARED_BYTES) {
            Some(bytes) => {
                resources.fixed = true;
                resources.bytes = resources.bytes.max(bytes);
                resources.alignment = resources.alignment.max(shared.get_alignment_value(ctx));
            }
            None => resources.unconstrained = true,
        }
    }
    if let Some(call) = Operation::get_op::<MirCallOp>(op, ctx)
        && let Some(callee) = call.get_attr_callee(ctx)
    {
        resources.callees.push(String::from((*callee).clone()));
    }
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                scan(ctx, child, resources);
            }
        }
    }
}

pub(crate) fn collect(
    ctx: &Context,
    module: Ptr<Operation>,
    kernels: impl IntoIterator<Item = String>,
) -> Result<BTreeMap<String, KernelLaunchFacts>, String> {
    let mut functions = BTreeMap::new();
    for region in module.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for op in block.deref(ctx).iter(ctx) {
                if let Some(function) = MirFuncOp::wrap(ctx, op) {
                    let mut resources = FunctionResources {
                        inferred: integer_attr(ctx, op, INFERRED_SHARED).is_some(),
                        alignment: integer_attr(ctx, op, "dynamic_shared_alignment").unwrap_or(1),
                        ..Default::default()
                    };
                    scan(ctx, op, &mut resources);
                    functions.insert(function.get_symbol_name(ctx).to_string(), (op, resources));
                }
            }
        }
    }
    let mut facts = BTreeMap::new();
    for kernel in kernels {
        let Some((entry, _)) = functions.get(&kernel) else {
            continue;
        };
        let mut visited = BTreeSet::new();
        let mut pending = vec![kernel.clone()];
        let mut combined = FunctionResources {
            alignment: 1,
            ..Default::default()
        };
        while let Some(name) = pending.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let Some((_, local)) = functions.get(&name) else {
                continue;
            };
            combined.inferred |= local.inferred;
            combined.fixed |= local.fixed;
            combined.unconstrained |= local.unconstrained;
            combined.bytes = combined.bytes.max(local.bytes);
            combined.alignment = combined.alignment.max(local.alignment);
            pending.extend(local.callees.iter().cloned());
        }
        if combined.inferred && combined.unconstrained {
            return Err(format!(
                "{kernel}: inferred dynamic shared memory cannot include unconstrained get/get_raw/offset accesses; describe the complete base region with DynamicSharedArray::get_fixed or declare an explicit launch contract"
            ));
        }
        if combined.unconstrained || !(combined.inferred || combined.fixed) {
            continue;
        }
        let narrow = |value| {
            u32::try_from(value)
                .map_err(|_| format!("{kernel}: compiled shared-memory layout exceeds u32"))
        };
        let exact_block = match (
            integer_attr(ctx, *entry, "reqntid_x"),
            integer_attr(ctx, *entry, "reqntid_y"),
            integer_attr(ctx, *entry, "reqntid_z"),
        ) {
            (Some(x), Some(y), Some(z)) => Some((narrow(x)?, narrow(y)?, narrow(z)?)),
            _ => None,
        };
        let layout = KernelLaunchFacts {
            exact_block,
            dynamic_shared_bytes: narrow(combined.bytes)?,
            dynamic_shared_alignment: narrow(combined.alignment)?,
        };
        facts.insert(kernel, layout);
    }
    Ok(facts)
}
