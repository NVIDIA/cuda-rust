/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assemble the existing contextual names and finalized plans from MIR proofs.

use super::storage::{Plan, RootedNativePlans, Site, Validation};
use dialect_mir::{
    library::{
        LibraryCallKind, LibraryCallPlan, LibraryCallSite, LibraryCallTargetAttr,
        LibraryFunctionSourceAttr, LibraryOperation, LibraryRoleAttr,
    },
    ops::{MirCallOp, MirFuncOp},
};
use oxide_artifacts::KernelLaunchFacts;
use pliron::{
    context::{Context, Ptr},
    graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op},
    operation::Operation,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Owned final inputs consumed by the existing context materializer and lowerer.
#[derive(Default)]
pub struct LibraryAssembly {
    /// Emitted raw export to original raw export; legalization remains importer-owned.
    pub contexts: BTreeMap<String, String>,
    pub calls: BTreeMap<LibraryCallSite, LibraryCallPlan>,
    pub redirects: BTreeMap<String, BTreeMap<String, String>>,
    pub roots: BTreeMap<String, KernelLaunchFacts>,
}

fn source<'a>(
    sources: &'a BTreeMap<String, LibraryFunctionSourceAttr>,
    export: &str,
) -> Result<&'a LibraryFunctionSourceAttr, String> {
    sources
        .get(export)
        .ok_or_else(|| format!("library context has no authenticated source {export}"))
}

fn insert_context(
    contexts: &mut BTreeMap<String, String>,
    emitted: String,
    original: String,
) -> Result<(), String> {
    if let Some(previous) = contexts.insert(emitted, original.clone())
        && previous != original
    {
        return Err("finalized helper context has conflicting source identities".into());
    }
    Ok(())
}

fn insert_call(
    calls: &mut BTreeMap<LibraryCallSite, LibraryCallPlan>,
    site: LibraryCallSite,
    plan: LibraryCallPlan,
) -> Result<(), String> {
    if let Some(previous) = calls.get(&site) {
        if format!("{previous:?}") != format!("{plan:?}") {
            return Err("library call site has conflicting provider specializations".into());
        }
    } else {
        calls.insert(site, plan);
    }
    Ok(())
}

/// Preserve the old nonstorage naming and bottom-up storage context interning.
///
/// Provider requests retain original source reachability. Only planned storage
/// roots use trace liveness; their first trace and layout-aware plan are already
/// proved. This function performs no analysis, allocation or IR mutation.
#[allow(clippy::too_many_arguments)]
pub fn assemble_library_plans(
    module: Ptr<Operation>,
    ctx: &Context,
    rooted_functions: &BTreeMap<String, BTreeSet<String>>,
    source_targets: &BTreeMap<LibraryCallSite, BTreeSet<String>>,
    validation: &Validation,
    plans: &BTreeMap<String, Plan>,
    native_plans: &RootedNativePlans,
    provider_roots: BTreeMap<String, KernelLaunchFacts>,
) -> Result<LibraryAssembly, String> {
    let mut captured = (
        BTreeMap::<String, LibraryFunctionSourceAttr>::new(),
        BTreeMap::<LibraryCallSite, LibraryCallTargetAttr>::new(),
        false,
    );
    walk_op(
        ctx,
        &mut captured,
        &WalkConfig::default(),
        module,
        |ctx, (sources, selected, conflict), node| {
            let IRNode::Operation(op) = node else { return };
            if let Some(function) = Operation::get_op::<MirFuncOp>(op, ctx)
                && let Some(identity) = function.library_function_source(ctx)
                && let Some(previous) = sources.insert(identity.export.clone(), identity.clone())
                && previous != identity
            {
                *conflict = true;
            }
            if let Some(call) = Operation::get_op::<MirCallOp>(op, ctx)
                && let Some(site) = call.get_attr_library_source_site(ctx)
                && let Some(target) = call.get_attr_library_call_target(ctx)
                && let Some(previous) = selected.insert((*site).clone(), (*target).clone())
                && previous != *target
            {
                *conflict = true;
            }
        },
    );
    let (sources, selected, conflict) = captured;
    if conflict {
        return Err("library context has conflicting authenticated source identities".into());
    }
    let mut result = LibraryAssembly {
        roots: provider_roots,
        ..Default::default()
    };
    let roots: BTreeSet<_> = native_plans
        .keys()
        .chain(validation.shapes.keys())
        .collect();
    for root in roots {
        if plans.contains_key(root) {
            continue;
        }
        let mut context = BTreeMap::new();
        if validation.shapes.contains_key(root) {
            let root_source = source(&sources, root)?;
            for original in rooted_functions
                .get(root)
                .ok_or("checked root has no source reachability")?
            {
                let Some(identity) = sources
                    .get(original)
                    .filter(|source| source.checked_ancestor)
                else {
                    continue;
                };
                let name = if original == root {
                    root.clone()
                } else {
                    format!(
                        "{}{}_{}_{}",
                        reserved_oxide_symbols::LIBRARY_CONTEXT_PREFIX,
                        root_source.mangled.len(),
                        root_source.mangled,
                        identity.mangled,
                    )
                };
                insert_context(&mut result.contexts, name.clone(), original.clone())?;
                context.insert(original.clone(), name);
            }
            for (original, emitted) in &context {
                let redirects = result.redirects.entry(emitted.clone()).or_default();
                for (site, targets) in source_targets {
                    if site.caller != *original {
                        continue;
                    }
                    for target in targets {
                        if let Some(target_context) = context.get(target) {
                            redirects.insert(target.clone(), target_context.clone());
                        }
                    }
                }
            }
        }
        if let Some(natives) = native_plans.get(root) {
            for (site, plan) in natives {
                insert_call(
                    &mut result.calls,
                    LibraryCallSite {
                        caller: context.get(&site.caller).unwrap_or(&site.caller).clone(),
                        block: site.block,
                    },
                    plan.clone(),
                )?;
            }
        }
    }
    for (root, plan) in plans {
        let trace = validation
            .traces
            .get(root)
            .ok_or("storage root has no validated trace")?;
        let mut calls = BTreeMap::<Site, LibraryCallPlan>::new();
        for (site, event) in &trace.events {
            let mut call = if matches!(event.role, LibraryRoleAttr::Native(_)) {
                native_plans
                    .get(root)
                    .and_then(|natives| natives.get(&event.call))
                    .ok_or("missing storage provider plan")?
                    .clone()
            } else {
                LibraryCallPlan::internal(
                    match event.role {
                        LibraryRoleAttr::StorageNew(_) => LibraryOperation::StorageNew,
                        LibraryRoleAttr::StorageReconfigure { .. } => {
                            LibraryOperation::StorageReconfigure
                        }
                        LibraryRoleAttr::StorageReserve { .. } => LibraryOperation::StorageReserve,
                        LibraryRoleAttr::StorageRelease => LibraryOperation::StorageRelease,
                        _ => return Err("storage trace has an unsupported operation".into()),
                    },
                    event.marker.clone(),
                )
            };
            // Even a zero-byte native view binds fixed-prefix/resource checks.
            if matches!(event.role, LibraryRoleAttr::Native(_)) {
                call.planned_root = Some(root.clone());
            }
            if let Some(binding) = plan.sites.get(site) {
                call.scratch_offset = binding.offset;
                call.scratch_stride = binding.stride;
                call.scratch_bytes = binding.declared_bytes;
                call.scratch_alignment = binding.alignment;
                call.planned_root = Some(root.clone());
                call.barrier_before = binding.before;
                call.barrier_after = binding.after;
            }
            calls.insert(site.clone(), call);
        }
        let mut names = BTreeMap::<Site, String>::new();
        let mut intern = HashMap::<(String, Vec<(usize, String)>), String>::new();
        let mut contexts = Vec::new();
        let mut paths = Vec::new();
        for (path, original) in &trace.contexts {
            if source(&sources, original)?.checked_ancestor {
                paths.push((path, original));
            }
        }
        paths.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then(a.cmp(b)));
        for (path, original) in paths {
            let mut bindings: Vec<_> = calls
                .iter()
                .filter(|(site, _)| site.len() == path.len() + 1 && site.starts_with(path))
                .map(|(site, call)| (site[path.len()], format!("{call:?}")))
                .collect();
            let children: Vec<_> = names
                .iter()
                .filter(|(child, _)| child.len() == path.len() + 1 && child.starts_with(path))
                .map(|(child, name)| (child[path.len()], name.clone()))
                .collect();
            bindings.extend(
                children
                    .iter()
                    .map(|(bb, name)| (*bb, format!("child:{name}"))),
            );
            bindings.sort();
            let key = (original.clone(), bindings);
            let name = if path.is_empty() {
                root.clone()
            } else if let Some(name) = intern.get(&key) {
                name.clone()
            } else {
                format!(
                    "{}storage_{}_{}_{}",
                    reserved_oxide_symbols::LIBRARY_CONTEXT_PREFIX,
                    root.len(),
                    root,
                    contexts.len()
                )
            };
            let is_new = !intern.contains_key(&key);
            intern.insert(key, name.clone());
            names.insert(path.clone(), name.clone());
            if !is_new {
                continue;
            }
            contexts.push((original, name.clone()));
            insert_context(&mut result.contexts, name.clone(), original.clone())?;
            for (site, call) in &calls {
                if site.len() == path.len() + 1 && site.starts_with(path) {
                    insert_call(
                        &mut result.calls,
                        LibraryCallSite {
                            caller: name.clone(),
                            block: trace.events[site].call.block,
                        },
                        call.clone(),
                    )?;
                }
            }
            for (bb, target) in children {
                let edge = selected
                    .get(&LibraryCallSite {
                        caller: original.clone(),
                        block: bb,
                    })
                    .ok_or("storage child call edge is missing")?;
                if edge.kind != LibraryCallKind::Direct {
                    return Err("storage bindings require a direct helper call".into());
                }
                let mut call =
                    LibraryCallPlan::internal(LibraryOperation::Redirect, edge.marker.clone());
                call.symbol = target;
                insert_call(
                    &mut result.calls,
                    LibraryCallSite {
                        caller: name.clone(),
                        block: bb,
                    },
                    call,
                )?;
            }
        }
        let shape = validation
            .shapes
            .get(root)
            .ok_or("storage root has no admitted shape")?;
        result.roots.insert(
            root.clone(),
            KernelLaunchFacts {
                exact_block: Some((shape[0], shape[1], shape[2])),
                dynamic_shared_bytes: plan.bytes,
                dynamic_shared_alignment: plan.alignment,
            },
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::storage::{Binding, Event, Policy, Trace};
    use super::*;
    use dialect_mir::library::{LibraryNativeMetadata, LibraryScalar, NativeSpec};
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::FunctionType,
        },
        op::Op,
    };

    fn function(ctx: &mut Context, module: Ptr<Operation>, raw: &str) -> Ptr<BasicBlock> {
        let ty = FunctionType::get(ctx, vec![], vec![]);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
        function.set_symbol_name(ctx, format!("physical_{raw}").try_into().unwrap());
        function.set_library_function_source(
            ctx,
            LibraryFunctionSourceAttr {
                export: raw.into(),
                mangled: format!("mangled_{raw}"),
                diagnostic: raw.into(),
                checked_ancestor: true,
            },
        );
        Operation::get_op::<ModuleOp>(module, ctx)
            .unwrap()
            .append_operation(ctx, op, 0);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(op.deref(ctx).get_region(0), ctx);
        block
    }

    fn site(caller: &str, block: usize) -> LibraryCallSite {
        LibraryCallSite {
            caller: caller.into(),
            block,
        }
    }

    fn call(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        bb: usize,
        target: &str,
        kind: LibraryCallKind,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        let call = MirCallOp::new(op);
        call.set_attr_callee(ctx, StringAttr::new(format!("physical_{target}")));
        call.set_attr_library_source_site(ctx, site("root", bb));
        call.set_attr_library_call_target(
            ctx,
            LibraryCallTargetAttr {
                kind,
                export: target.into(),
                marker: format!("mangled_{target}"),
            },
        );
        op.insert_at_back(block, ctx);
        op
    }

    fn native_plan() -> LibraryCallPlan {
        let mut plan = LibraryCallPlan::internal(LibraryOperation::Native, "native_marker".into());
        plan.symbol = "native_symbol".into();
        plan.checked_block = true;
        plan.scratch_alignment = 1;
        plan
    }

    #[test]
    fn nonstorage_redirects_keep_all_source_targets_and_mangled_names() {
        let ctx = &mut Context::new();
        dialect_mir::register(ctx);
        let module = ModuleOp::new(ctx, "contexts".try_into().unwrap()).get_operation();
        let root_block = function(ctx, module, "root");
        function(ctx, module, "callable");
        function(ctx, module, "shim");
        call(ctx, root_block, 7, "callable", LibraryCallKind::Callable);
        let reached = BTreeMap::from([(
            "root".into(),
            BTreeSet::from(["root".into(), "callable".into(), "shim".into()]),
        )]);
        let targets = BTreeMap::from([(
            site("root", 7),
            BTreeSet::from(["callable".into(), "shim".into()]),
        )]);
        let validation = Validation {
            shapes: BTreeMap::from([("root".into(), [64, 1, 1])]),
            ..Default::default()
        };
        let natives = BTreeMap::from([(
            "root".into(),
            BTreeMap::from([(site("shim", 3), native_plan())]),
        )]);
        let result = assemble_library_plans(
            module,
            ctx,
            &reached,
            &targets,
            &validation,
            &BTreeMap::new(),
            &natives,
            BTreeMap::new(),
        )
        .unwrap();
        let expected = format!(
            "{}12_mangled_root_mangled_shim",
            reserved_oxide_symbols::LIBRARY_CONTEXT_PREFIX
        );
        assert_eq!(result.redirects["root"].len(), 2);
        assert_eq!(result.redirects["root"]["shim"], expected);
        assert_eq!(result.contexts[&expected], "shim");
        assert_eq!(result.calls[&site(&expected, 3)].symbol, "native_symbol");
    }

    #[test]
    fn storage_interning_uses_final_bindings_and_keeps_zero_layout_root() {
        let ctx = &mut Context::new();
        dialect_mir::register(ctx);
        let module = ModuleOp::new(ctx, "storage_contexts".try_into().unwrap()).get_operation();
        let root_block = function(ctx, module, "root");
        function(ctx, module, "helper");
        let first = call(ctx, root_block, 1, "helper", LibraryCallKind::Direct);
        call(ctx, root_block, 2, "helper", LibraryCallKind::Direct);
        call(ctx, root_block, 3, "helper", LibraryCallKind::Direct);
        let role = LibraryRoleAttr::Native(LibraryNativeMetadata::CheckedBlockStorage {
            spec: NativeSpec {
                scalar: LibraryScalar::Signed(32),
                items: 1,
            },
            storage: dialect_mir::library::LibraryStorageArg::Implicit,
            access: dialect_mir::library::LibraryAccess::Read,
            witness: dialect_mir::library::LibraryWitness::ReadOnlySlice,
        });
        let event = |caller, bb| Event {
            call: site(caller, bb),
            role,
            marker: "native_marker".into(),
            scope: super::super::storage::Domain::BLOCK,
            origin: None,
            policy: Policy::default(),
        };
        let trace = Trace {
            contexts: BTreeMap::from([
                (vec![], "root".into()),
                (vec![1], "helper".into()),
                (vec![2], "helper".into()),
                (vec![3], "helper".into()),
            ]),
            events: BTreeMap::from([
                (vec![1, 4], event("helper", 4)),
                (vec![2, 4], event("helper", 4)),
                (vec![3, 4], event("helper", 4)),
                (vec![9], event("root", 9)),
            ]),
        };
        let validation = Validation {
            shapes: BTreeMap::from([("root".into(), [64, 1, 1])]),
            traces: BTreeMap::from([("root".into(), trace)]),
        };
        let binding = |offset| Binding {
            offset,
            declared_bytes: 16,
            bytes: 16,
            alignment: 16,
            before: true,
            ..Default::default()
        };
        let plans = BTreeMap::from([(
            "root".into(),
            Plan {
                sites: BTreeMap::from([
                    (vec![1, 4], binding(0)),
                    (vec![2, 4], binding(0)),
                    (vec![3, 4], binding(16)),
                ]),
                bytes: 32,
                alignment: 16,
            },
        )]);
        let natives = BTreeMap::from([(
            "root".into(),
            BTreeMap::from([
                (site("helper", 4), native_plan()),
                (site("root", 9), native_plan()),
            ]),
        )]);
        let reached = BTreeMap::from([(
            "root".into(),
            BTreeSet::from(["root".into(), "helper".into()]),
        )]);
        let targets = [1, 2, 3]
            .into_iter()
            .map(|bb| (site("root", bb), BTreeSet::from(["helper".into()])))
            .collect();
        let result = assemble_library_plans(
            module,
            ctx,
            &reached,
            &targets,
            &validation,
            &plans,
            &natives,
            BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            result.calls[&site("root", 1)].symbol,
            result.calls[&site("root", 2)].symbol
        );
        assert_ne!(
            result.calls[&site("root", 1)].symbol,
            result.calls[&site("root", 3)].symbol
        );
        assert_eq!(result.contexts.len(), 3);
        assert_eq!(result.calls[&site("root", 9)].scratch_bytes, 0);
        assert_eq!(
            result.calls[&site("root", 9)].planned_root.as_deref(),
            Some("root")
        );
        assert_eq!(result.roots["root"].dynamic_shared_bytes, 32);
        assert_eq!(result.roots["root"].dynamic_shared_alignment, 16);
        let helper = &result.calls[&site("root", 3)].symbol;
        assert!(result.calls[&site(helper, 4)].barrier_before);
        assert_eq!(result.calls[&site(helper, 4)].scratch_offset, 16);
        let mut selected = (*MirCallOp::new(first)
            .get_attr_library_call_target(ctx)
            .unwrap())
        .clone();
        selected.kind = LibraryCallKind::Callable;
        MirCallOp::new(first).set_attr_library_call_target(ctx, selected);
        let error = assemble_library_plans(
            module,
            ctx,
            &reached,
            &targets,
            &validation,
            &plans,
            &natives,
            BTreeMap::new(),
        )
        .err()
        .unwrap();
        assert_eq!(error, "storage bindings require a direct helper call");
    }
}
