/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Finalized requirements for fixed views of the dynamic shared-memory base.
//! This is not an allocator: fixed views alias and combine by maximum extent.

use dialect_mir::{
    library::{LibraryCallPlan, LibraryCallSite, LibraryOperation},
    ops::{MirCallOp, MirExternSharedOp, MirFuncOp},
};
use oxide_artifacts::KernelLaunchFacts;
use pliron::builtin::{attributes::IntegerAttr, op_interfaces::SymbolOpInterface};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const PLANNED_ROOT: &str = "cuda_oxide_planned_storage_root";

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

#[derive(Clone, Default)]
struct FunctionResources {
    inferred: bool,
    fixed: bool,
    unconstrained: bool,
    bytes: u64,
    alignment: u64,
    declared_alignment: Option<u64>,
    callees: Vec<OrdinaryCall>,
}

#[derive(Clone)]
struct OrdinaryCall {
    site: Option<LibraryCallSite>,
    callee: String,
}

fn scan(ctx: &Context, op: Ptr<Operation>, resources: &mut FunctionResources, native_only: bool) {
    if let Some(shared) = Operation::get_op::<MirExternSharedOp>(op, ctx)
        && !(native_only && planned_root(ctx, op).is_some())
    {
        match integer_attr(ctx, op, FIXED_SHARED_BYTES) {
            Some(bytes) => {
                resources.fixed = true;
                resources.bytes = resources
                    .bytes
                    .max(shared.get_byte_offset_value(ctx).saturating_add(bytes));
                resources.alignment = resources.alignment.max(shared.get_alignment_value(ctx));
            }
            None => resources.unconstrained = true,
        }
    }
    if let Some(call) = Operation::get_op::<MirCallOp>(op, ctx)
        && let Some(callee) = call.get_attr_callee(ctx)
    {
        resources.callees.push(OrdinaryCall {
            site: call
                .get_attr_library_source_site(ctx)
                .map(|site| (*site).clone()),
            callee: String::from((*callee).clone()),
        });
    }
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                scan(ctx, child, resources, native_only);
            }
        }
    }
}

/// Ordinary requirements captured before preparation can remove views or calls.
/// Consumed by completion so compiler-planned offsets are relocated only once.
pub(crate) struct OrdinaryResources {
    kernels: BTreeMap<String, KernelResources>,
    alignments: BTreeMap<String, u64>,
    functions: BTreeMap<String, FunctionResources>,
}

struct KernelResources {
    resources: FunctionResources,
    exact_block: Option<(u32, u32, u32)>,
    maximum: Option<u64>,
}

fn narrow(kernel: &str, value: u64) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{kernel}: compiled shared-memory layout exceeds u32"))
}

/// Capture ordinary views, original helper reachability and launch constraints.
/// Unconstrained roots are retained even when they produce no launch fact.
pub(crate) fn capture_ordinary(
    ctx: &Context,
    module: Ptr<Operation>,
    kernels: impl IntoIterator<Item = String>,
) -> Result<OrdinaryResources, String> {
    capture(ctx, module, kernels, true)
}

/// Replay saved ordinary edges under the finalized helper contexts. Names here
/// are physical MIR symbols; source sites retain their original raw callers.
/// The target resolver is shared with actual context materialization.
pub(crate) fn bind_ordinary_contexts(
    ordinary: &mut OrdinaryResources,
    contexts: &BTreeMap<String, String>,
    mut resolve: impl FnMut(Option<&LibraryCallSite>, &str, &str) -> Result<String, String>,
) -> Result<(), String> {
    let replaced: BTreeSet<_> = contexts.values().collect();
    let mut targets: BTreeMap<_, _> = ordinary
        .functions
        .keys()
        .filter(|name| !replaced.contains(name))
        .map(|name| (name.clone(), name.clone()))
        .collect();
    targets.extend(contexts.clone());
    let mut functions = BTreeMap::new();
    for (emitted, source) in targets {
        let mut local = ordinary
            .functions
            .get(&source)
            .ok_or_else(|| format!("ordinary resource context has no source function {source}"))?
            .clone();
        for call in &mut local.callees {
            call.callee = resolve(call.site.as_ref(), &emitted, &call.callee)?;
        }
        functions.insert(emitted, local);
    }
    for kernel in ordinary.kernels.keys() {
        if !functions.contains_key(kernel) {
            return Err(format!("ordinary resource context removed kernel {kernel}"));
        }
    }
    ordinary.functions = functions;
    aggregate_ordinary(ordinary);
    Ok(())
}

/// Validate saved pre-preparation resources against the actual library plans.
/// Only Native and Reserve emit shared views; zero-byte views still count.
pub(crate) fn validate_ordinary_compatibility(
    ordinary: &OrdinaryResources,
    calls: &BTreeMap<LibraryCallSite, LibraryCallPlan>,
) -> Result<(), String> {
    let planned: BTreeSet<_> = calls
        .values()
        .filter(|plan| {
            matches!(
                plan.operation,
                LibraryOperation::Native | LibraryOperation::StorageReserve
            )
        })
        .filter_map(|plan| plan.planned_root.as_ref())
        .collect();
    for root in planned {
        let saved = ordinary
            .kernels
            .get(root)
            .ok_or("planned storage root has no captured ordinary resources")?;
        if saved.resources.unconstrained {
            return Err(format!(
                "{root}: planned storage cannot include an unconstrained native dynamic shared view"
            ));
        }
    }
    for (kernel, saved) in &ordinary.kernels {
        if saved.resources.inferred && saved.resources.unconstrained {
            return Err(format!(
                "{kernel}: inferred dynamic shared memory cannot include unconstrained get/get_raw/offset accesses; describe the complete base region with DynamicSharedArray::get_fixed or declare an explicit launch contract"
            ));
        }
    }
    Ok(())
}

fn capture(
    ctx: &Context,
    module: Ptr<Operation>,
    kernels: impl IntoIterator<Item = String>,
    ordinary_only: bool,
) -> Result<OrdinaryResources, String> {
    let mut functions = BTreeMap::new();
    let mut entries = BTreeMap::new();
    let mut operations = Vec::new();
    walk(ctx, module, &mut operations);
    for op in operations {
        if let Some(function) = MirFuncOp::wrap(ctx, op) {
            let declared_alignment = integer_attr(ctx, op, "dynamic_shared_alignment");
            let mut local = FunctionResources {
                inferred: integer_attr(ctx, op, INFERRED_SHARED).is_some(),
                alignment: declared_alignment.unwrap_or(1),
                declared_alignment,
                ..Default::default()
            };
            scan(ctx, op, &mut local, ordinary_only);
            let name = function.get_symbol_name(ctx).to_string();
            entries.insert(name.clone(), op);
            functions.insert(name, local);
        }
    }
    let mut captured = BTreeMap::new();
    for kernel in kernels {
        let Some(&entry) = entries.get(&kernel) else {
            continue;
        };
        let exact_block = match (
            integer_attr(ctx, entry, "reqntid_x"),
            integer_attr(ctx, entry, "reqntid_y"),
            integer_attr(ctx, entry, "reqntid_z"),
        ) {
            (Some(x), Some(y), Some(z)) => Some((
                narrow(&kernel, x)?,
                narrow(&kernel, y)?,
                narrow(&kernel, z)?,
            )),
            _ => None,
        };
        captured.insert(
            kernel,
            KernelResources {
                resources: FunctionResources::default(),
                exact_block,
                maximum: integer_attr(ctx, entry, "maxntid"),
            },
        );
    }
    let mut ordinary = OrdinaryResources {
        kernels: captured,
        alignments: BTreeMap::new(),
        functions,
    };
    aggregate_ordinary(&mut ordinary);
    Ok(ordinary)
}

/// The existing fixed-view maximum and alignment propagation over saved edges.
/// Replaying after context binding prevents unrelated roots' helper alignments
/// from being merged into every specialized copy.
fn aggregate_ordinary(ordinary: &mut OrdinaryResources) {
    ordinary.alignments.clear();
    for (name, local) in &ordinary.functions {
        let Some(alignment) = local.declared_alignment else {
            continue;
        };
        let mut pending = vec![name.clone()];
        let mut visited = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let Some(local) = ordinary.functions.get(&name) else {
                continue;
            };
            let saved = ordinary.alignments.entry(name).or_default();
            *saved = (*saved).max(alignment);
            pending.extend(local.callees.iter().map(|call| call.callee.clone()));
        }
    }
    for (kernel, saved) in &mut ordinary.kernels {
        let mut combined = FunctionResources {
            alignment: 1,
            ..Default::default()
        };
        let mut pending = vec![kernel.clone()];
        let mut visited = BTreeSet::new();
        while let Some(name) = pending.pop() {
            if !visited.insert(name.clone()) {
                continue;
            }
            let Some(local) = ordinary.functions.get(&name) else {
                continue;
            };
            merge_resources(&mut combined, local);
            combined.alignment = combined
                .alignment
                .max(ordinary.alignments.get(&name).copied().unwrap_or(1));
            pending.extend(local.callees.iter().map(|call| call.callee.clone()));
        }
        saved.resources = combined;
    }
}

fn planned_root(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    let key: pliron::identifier::Identifier = PLANNED_ROOT.try_into().unwrap();
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&key)
        .map(|value| String::from((*value).clone()))
}
fn walk(ctx: &Context, op: Ptr<Operation>, result: &mut Vec<Ptr<Operation>>) {
    result.push(op);
    for region in op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for child in block.deref(ctx).iter(ctx) {
                walk(ctx, child, result);
            }
        }
    }
}
fn merge_resources(saved: &mut FunctionResources, actual: &FunctionResources) {
    saved.inferred |= actual.inferred;
    saved.fixed |= actual.fixed;
    saved.unconstrained |= actual.unconstrained;
    saved.bytes = saved.bytes.max(actual.bytes);
    saved.alignment = saved.alignment.max(actual.alignment);
}

/// Relocate late planned views after the saved ordinary prefix and join facts.
/// Preparation is allowed to remove ordinary views/helper edges: their original
/// constraints still apply. The owned snapshot is not reusable for relocation.
pub(crate) fn complete_planned(
    ctx: &mut Context,
    module: Ptr<Operation>,
    mut ordinary: OrdinaryResources,
    roots: &BTreeMap<String, KernelLaunchFacts>,
) -> Result<BTreeMap<String, KernelLaunchFacts>, String> {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;

    // Raw and legacy calls also emit fixed shared views without a planned
    // tag. They contribute to the ordinary prefix, even for zero-byte scratch.
    let late = capture_ordinary(ctx, module, ordinary.kernels.keys().cloned())?;
    for (name, actual) in late.kernels {
        merge_resources(
            &mut ordinary.kernels.get_mut(&name).unwrap().resources,
            &actual.resources,
        );
    }
    for (name, alignment) in late.alignments {
        let saved = ordinary.alignments.entry(name).or_default();
        *saved = (*saved).max(alignment);
    }
    let mut operations = Vec::new();
    walk(ctx, module, &mut operations);
    let mut relocated = Vec::new();
    for &op in &operations {
        let Some(root) = planned_root(ctx, op) else {
            continue;
        };
        let plan = roots
            .get(&root)
            .ok_or("planned storage root has no finalized layout")?;
        let saved = ordinary
            .kernels
            .get(&root)
            .ok_or("planned storage root has no captured ordinary resources")?;
        if saved.resources.unconstrained {
            return Err(format!(
                "{root}: planned storage cannot include an unconstrained native dynamic shared view"
            ));
        }
        let alignment = u64::from(plan.dynamic_shared_alignment);
        if !alignment.is_power_of_two() {
            return Err("invalid planned storage alignment".into());
        }
        let base = saved
            .resources
            .bytes
            .checked_add(alignment - 1)
            .map(|n| n & !(alignment - 1))
            .ok_or("native prefix and storage plan overflow")?;
        let end = base
            .checked_add(u64::from(plan.dynamic_shared_bytes))
            .ok_or("native prefix and storage plan overflow")?;
        u32::try_from(end).map_err(|_| "native prefix and storage plan exceed u32")?;
        let shared = Operation::get_op::<MirExternSharedOp>(op, ctx)
            .ok_or("planned storage tag is not on a shared view")?;
        let offset = base
            .checked_add(shared.get_byte_offset_value(ctx))
            .ok_or("native prefix and storage plan overflow")?;
        let bytes = integer_attr(ctx, op, FIXED_SHARED_BYTES)
            .ok_or("planned storage view has no fixed extent")?;
        let extent = offset
            .checked_add(bytes)
            .ok_or("native prefix and storage plan overflow")?;
        narrow(&root, extent)?;
        relocated.push((op, root, offset, extent, shared.get_alignment_value(ctx)));
    }
    for (root, plan) in roots {
        let saved = ordinary
            .kernels
            .get_mut(root)
            .ok_or("native library root has no captured ordinary resources")?;
        let shape = plan
            .exact_block
            .ok_or("native library root is missing its exact block shape")?;
        if saved.exact_block.is_some_and(|old| old != shape) {
            let (x, y, z) = shape;
            return Err(format!(
                "native library exact block ({x}, {y}, {z}) conflicts with kernel launch contract"
            ));
        }
        let (x, y, z) = shape;
        if let Some(maximum) = saved.maximum
            && u64::from(x) * u64::from(y) * u64::from(z) > maximum
        {
            return Err(format!(
                "native library exact block ({x}, {y}, {z}) exceeds launch_bounds maximum {maximum}"
            ));
        }
        saved.exact_block = Some(shape);
    }
    for (op, root, offset, extent, alignment) in relocated {
        MirExternSharedOp::new(op).set_byte_offset_value(ctx, offset);
        let resources = &mut ordinary.kernels.get_mut(&root).unwrap().resources;
        resources.fixed = true;
        resources.bytes = resources.bytes.max(extent);
        resources.alignment = resources.alignment.max(alignment);
    }
    // Restore alignment propagation over the original helper graph before
    // propagating across the final graph, which also contains native calls.
    let layout_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    for op in operations {
        if let Some(function) = MirFuncOp::wrap(ctx, op) {
            let name = function.get_symbol_name(ctx).to_string();
            if let Some(&alignment) = ordinary.alignments.get(&name) {
                let alignment =
                    alignment.max(integer_attr(ctx, op, "dynamic_shared_alignment").unwrap_or(1));
                op.deref_mut(ctx).attributes.set(
                    "dynamic_shared_alignment".try_into().unwrap(),
                    IntegerAttr::new(
                        layout_ty,
                        APInt::from_u64(alignment, std::num::NonZeroUsize::new(64).unwrap()),
                    ),
                );
            }
            if let Some(plan) = roots.get(&name) {
                apply_library_block(ctx, op, plan)?;
            }
        }
    }
    mir_lower::propagate_kernel_dynamic_shared_alignments(ctx, module);
    let actual = capture(ctx, module, ordinary.kernels.keys().cloned(), false)?;
    for (name, actual) in actual.kernels {
        merge_resources(
            &mut ordinary.kernels.get_mut(&name).unwrap().resources,
            &actual.resources,
        );
    }
    let mut facts = BTreeMap::new();
    for (kernel, saved) in ordinary.kernels {
        let combined = saved.resources;
        if combined.inferred && combined.unconstrained {
            return Err(format!(
                "{kernel}: inferred dynamic shared memory cannot include unconstrained get/get_raw/offset accesses; describe the complete base region with DynamicSharedArray::get_fixed or declare an explicit launch contract"
            ));
        }
        if combined.unconstrained || !(combined.inferred || combined.fixed) {
            continue;
        }
        facts.insert(
            kernel.clone(),
            KernelLaunchFacts {
                exact_block: saved.exact_block,
                dynamic_shared_bytes: narrow(&kernel, combined.bytes)?,
                dynamic_shared_alignment: narrow(&kernel, combined.alignment)?,
            },
        );
    }
    Ok(facts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        ops::MirCallOp,
        types::{MirPtrType, MirTupleType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::SingleBlockRegionInterface,
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        irbuild::{
            listener::DummyListener,
            rewriter::{IRRewriter, Rewriter},
        },
        op::Op,
        utils::apint::APInt,
    };

    fn function(
        ctx: &mut Context,
        module: Ptr<Operation>,
        name: &str,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        let signature = FunctionType::get(ctx, vec![], vec![]);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, op, TypeAttr::new(signature.into()));
        function.set_symbol_name(ctx, name.try_into().unwrap());
        Operation::get_op::<ModuleOp>(module, ctx)
            .unwrap()
            .append_operation(ctx, op, 0);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(op.deref(ctx).get_region(0), ctx);
        (op, block)
    }

    fn attr(ctx: &mut Context, op: Ptr<Operation>, name: &str, value: u64) {
        let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        op.deref_mut(ctx).attributes.set(
            name.try_into().unwrap(),
            IntegerAttr::new(
                ty,
                APInt::from_u64(value, std::num::NonZeroUsize::new(64).unwrap()),
            ),
        );
    }

    fn call(ctx: &mut Context, block: Ptr<BasicBlock>, callee: &str) -> Ptr<Operation> {
        let unit = MirTupleType::get(ctx, vec![]).into();
        let op = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![unit],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(op).set_attr_callee(ctx, StringAttr::new(callee.to_owned()));
        op.insert_at_back(block, ctx);
        op
    }

    fn view(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        offset: u64,
        bytes: Option<u64>,
        planned: Option<&str>,
    ) -> Ptr<Operation> {
        let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let pointer = MirPtrType::get_shared(ctx, byte, true).into();
        let op = Operation::new(
            ctx,
            MirExternSharedOp::get_concrete_op_info(),
            vec![pointer],
            vec![],
            vec![],
            0,
        );
        let view = MirExternSharedOp::new(op);
        view.set_byte_offset_value(ctx, offset);
        view.set_alignment_value(ctx, 8);
        if let Some(bytes) = bytes {
            attr(ctx, op, FIXED_SHARED_BYTES, bytes);
        }
        if let Some(root) = planned {
            op.deref_mut(ctx).attributes.set(
                PLANNED_ROOT.try_into().unwrap(),
                StringAttr::new(root.to_owned()),
            );
        }
        op.insert_at_back(block, ctx);
        op
    }

    #[test]
    fn completion_preserves_deleted_helper_views_and_joins_late_native_scratch() {
        let ctx = &mut Context::new();
        dialect_mir::register(ctx);
        let module = ModuleOp::new(ctx, "resources".try_into().unwrap()).get_operation();
        let (root, body) = function(ctx, module, "kernel");
        let (other, other_body) = function(ctx, module, "other");
        let (_, helper) = function(ctx, module, "helper");
        let (_, raw) = function(ctx, module, "raw");
        attr(ctx, root, "dynamic_shared_alignment", 16);
        attr(ctx, other, "dynamic_shared_alignment", 128);
        let old_call = call(ctx, body, "helper");
        call(ctx, other_body, "helper");
        let old_view = view(ctx, helper, 5, Some(68), None);
        let saved =
            capture_ordinary(ctx, module, ["kernel".into(), "other".into(), "raw".into()]).unwrap();
        let mut rewriter = IRRewriter::<DummyListener>::default();
        rewriter.erase_operation(ctx, old_call);
        rewriter.erase_operation(ctx, old_view);
        view(ctx, body, 16, Some(32), None);
        view(ctx, raw, 0, Some(0), None);
        let planned = view(ctx, body, 4, Some(12), Some("kernel"));
        let roots = BTreeMap::from([(
            "kernel".into(),
            KernelLaunchFacts {
                exact_block: Some((64, 1, 1)),
                dynamic_shared_bytes: 16,
                dynamic_shared_alignment: 16,
            },
        )]);
        let facts = complete_planned(ctx, module, saved, &roots).unwrap();
        assert_eq!(
            MirExternSharedOp::new(planned).get_byte_offset_value(ctx),
            84
        );
        assert_eq!(facts["kernel"].dynamic_shared_bytes, 96);
        assert_eq!(facts["kernel"].dynamic_shared_alignment, 128);
        assert_eq!(facts["kernel"].exact_block, Some((64, 1, 1)));
        assert_eq!(facts["other"].dynamic_shared_bytes, 73);
        assert_eq!(facts["raw"].dynamic_shared_bytes, 0);
    }

    #[test]
    fn context_replay_preserves_distinct_alignments_after_edges_are_deleted() {
        let ctx = &mut Context::new();
        dialect_mir::register(ctx);
        let module = ModuleOp::new(ctx, "contexts".try_into().unwrap()).get_operation();
        let (a, a_body) = function(ctx, module, "root_a");
        let (b, b_body) = function(ctx, module, "root_b");
        let (_, helper) = function(ctx, module, "helper");
        attr(ctx, a, "dynamic_shared_alignment", 16);
        attr(ctx, b, "dynamic_shared_alignment", 128);
        let a_call = call(ctx, a_body, "helper");
        let b_call = call(ctx, b_body, "helper");
        for (op, caller) in [(a_call, "raw::a"), (b_call, "raw::b")] {
            MirCallOp::new(op).set_attr_library_source_site(
                ctx,
                LibraryCallSite {
                    caller: caller.into(),
                    block: 3,
                },
            );
        }
        let original_view = view(ctx, helper, 0, Some(13), None);
        let mut saved = capture_ordinary(ctx, module, ["root_a".into(), "root_b".into()]).unwrap();
        let mut rewriter = IRRewriter::<DummyListener>::default();
        for op in [a_call, b_call, original_view] {
            rewriter.erase_operation(ctx, op);
        }
        let contexts = BTreeMap::from([
            ("helper_a".into(), "helper".into()),
            ("helper_b".into(), "helper".into()),
        ]);
        bind_ordinary_contexts(&mut saved, &contexts, |site, caller, callee| {
            assert_eq!(callee, "helper");
            assert_eq!(site.unwrap().block, 3);
            Ok(if caller == "root_a" {
                "helper_a"
            } else {
                "helper_b"
            }
            .into())
        })
        .unwrap();
        let (clone_a, body_a) = function(ctx, module, "helper_a");
        let (clone_b, body_b) = function(ctx, module, "helper_b");
        let view_a = view(ctx, body_a, 0, Some(0), Some("root_a"));
        let view_b = view(ctx, body_b, 0, Some(0), Some("root_b"));
        let roots = ["root_a", "root_b"]
            .into_iter()
            .map(|root| {
                (
                    root.into(),
                    KernelLaunchFacts {
                        exact_block: Some((32, 1, 1)),
                        dynamic_shared_bytes: 0,
                        dynamic_shared_alignment: 8,
                    },
                )
            })
            .collect();
        let facts = complete_planned(ctx, module, saved, &roots).unwrap();
        assert_eq!(facts["root_a"].dynamic_shared_alignment, 16);
        assert_eq!(facts["root_b"].dynamic_shared_alignment, 128);
        for name in ["root_a", "root_b"] {
            assert_eq!(facts[name].dynamic_shared_bytes, 16);
        }
        assert_eq!(
            integer_attr(ctx, clone_a, "dynamic_shared_alignment"),
            Some(16)
        );
        assert_eq!(
            integer_attr(ctx, clone_b, "dynamic_shared_alignment"),
            Some(128)
        );
        for view in [view_a, view_b] {
            assert_eq!(MirExternSharedOp::new(view).get_byte_offset_value(ctx), 16);
        }
    }

    #[test]
    fn deleted_unconstrained_helper_still_rejects_planned_or_inferred_roots() {
        for planned in [false, true] {
            let ctx = &mut Context::new();
            dialect_mir::register(ctx);
            let module = ModuleOp::new(ctx, "unconstrained".try_into().unwrap()).get_operation();
            let (root, body) = function(ctx, module, "kernel");
            let (_, helper) = function(ctx, module, "helper");
            if !planned {
                attr(ctx, root, INFERRED_SHARED, 1);
            }
            let old_call = call(ctx, body, "helper");
            let old_view = view(ctx, helper, 0, None, None);
            let saved = capture_ordinary(ctx, module, ["kernel".into()]).unwrap();
            let mut rewriter = IRRewriter::<DummyListener>::default();
            rewriter.erase_operation(ctx, old_call);
            rewriter.erase_operation(ctx, old_view);
            let expected = if planned {
                "planned storage cannot include an unconstrained"
            } else {
                "inferred dynamic shared memory cannot include unconstrained"
            };
            let site = LibraryCallSite {
                caller: "kernel".into(),
                block: 0,
            };
            if planned {
                for operation in [LibraryOperation::Native, LibraryOperation::StorageReserve] {
                    let mut plan = LibraryCallPlan::internal(operation, "marker".into());
                    plan.planned_root = Some("kernel".into());
                    // Zero-byte planned views still conflict with the saved raw view.
                    let calls = BTreeMap::from([(site.clone(), plan)]);
                    let error = validate_ordinary_compatibility(&saved, &calls).unwrap_err();
                    assert!(error.contains(expected), "{error}");
                }
                for operation in [
                    LibraryOperation::StorageNew,
                    LibraryOperation::StorageReconfigure,
                    LibraryOperation::StorageRelease,
                    LibraryOperation::Redirect,
                ] {
                    let mut plan = LibraryCallPlan::internal(operation, "marker".into());
                    plan.planned_root = Some("kernel".into());
                    validate_ordinary_compatibility(
                        &saved,
                        &BTreeMap::from([(site.clone(), plan)]),
                    )
                    .unwrap();
                }
                let untagged = LibraryCallPlan::internal(LibraryOperation::Native, "raw".into());
                validate_ordinary_compatibility(&saved, &BTreeMap::from([(site, untagged)]))
                    .unwrap();
            } else {
                let error = validate_ordinary_compatibility(&saved, &BTreeMap::new()).unwrap_err();
                assert!(error.contains(expected), "{error}");
            }
            let mut roots = BTreeMap::new();
            if planned {
                view(ctx, body, 0, Some(16), Some("kernel"));
                roots.insert(
                    "kernel".into(),
                    KernelLaunchFacts {
                        exact_block: Some((32, 1, 1)),
                        dynamic_shared_bytes: 16,
                        dynamic_shared_alignment: 8,
                    },
                );
            }
            let error = complete_planned(ctx, module, saved, &roots).unwrap_err();
            assert!(
                error.contains(if planned {
                    "planned storage cannot include an unconstrained"
                } else {
                    "inferred dynamic shared memory cannot include unconstrained"
                }),
                "{error}"
            );
        }
    }
}
