// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static full-block storage identities and checked placement. Provider recipe
//! identity is deliberately independent of every offset and synchronization site.

use super::{CallSite, LibraryCall, MarkerKind};
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::{Instance, Ty, TyCtxt, TyKind, TypingEnv};
use std::collections::BTreeMap;

/// A direct-call path followed by a MIR block. Loop iterations do not enter it.
pub(crate) type Site = Vec<usize>;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Policy {
    pub capacity: u32,
    pub alignment: u32,
    pub auto_sync: bool,
    pub exclusive: bool,
}

fn extent<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: rustc_middle::ty::GenericArgsRef<'tcx>,
    index: usize,
) -> Result<u32, String> {
    args.const_at(index)
        .try_to_target_usize(tcx)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "storage extent must be a concrete usize fitting u32".into())
}
fn boolean(args: rustc_middle::ty::GenericArgsRef<'_>, index: usize) -> Result<bool, String> {
    args.const_at(index)
        .try_to_leaf()
        .and_then(|n| n.try_to_bool().ok())
        .ok_or_else(|| "storage policy must be a concrete bool".into())
}
fn alignment(value: u32) -> Result<u32, String> {
    if value != 0 && !value.is_power_of_two() {
        return Err("storage alignment must be zero or a power of two".into());
    }
    Ok(value)
}

pub(crate) fn policy<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Policy, String> {
    let TyKind::Adt(def, args) = ty.kind() else {
        return Err("expected a genuine StorageDescriptor".into());
    };
    if Some(def.did()) != super::device_type(tcx, "library", "StorageDescriptor") {
        return Err("expected a genuine StorageDescriptor".into());
    }
    Ok(Policy {
        capacity: extent(tcx, args, 0)?,
        alignment: alignment(extent(tcx, args, 1)?)?,
        auto_sync: boolean(args, 2)?,
        exclusive: boolean(args, 3)?,
    })
}

pub(crate) fn storage_handle_type(tcx: TyCtxt<'_>, ty: Ty<'_>) -> bool {
    matches!(ty.kind(), TyKind::Adt(def, _) if ["StorageDescriptor","StorageReservation"].iter().any(|name|Some(def.did()) == super::device_type(tcx, "library", name)))
}

pub(crate) fn borrow_policy<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Policy, String> {
    let TyKind::Ref(_, inner, rustc_hir::Mutability::Mut) = ty.kind() else {
        return Err("storage must mutably borrow a genuine descriptor".into());
    };
    policy(tcx, *inner)
}

pub(crate) fn new_policy<'tcx>(
    tcx: TyCtxt<'tcx>,
    call: &LibraryCall<'tcx>,
) -> Result<Policy, String> {
    let args = call.instance.args;
    Ok(Policy {
        capacity: extent(tcx, args, 0)?,
        alignment: alignment(extent(tcx, args, 1)?)?,
        auto_sync: boolean(args, 2)?,
        exclusive: boolean(args, 3)?,
    })
}

pub(crate) fn reserve_layout<'tcx>(
    tcx: TyCtxt<'tcx>,
    call: &LibraryCall<'tcx>,
) -> Result<(u32, u32), String> {
    let args = call.instance.args;
    let element = args.type_at(0);
    super::scalar(element)?;
    // f16 is not in the frozen ten-type reservation set.
    if matches!(
        element.kind(),
        TyKind::Float(rustc_middle::ty::FloatTy::F16)
    ) {
        return Err("storage reservations require one of the ten fixed-width numeric types".into());
    }
    let count = extent(tcx, args, 2)?;
    if count == 0 {
        return Err("storage reservation extent must be positive".into());
    }
    let array = Ty::new_array(tcx, element, u64::from(count));
    let layout = tcx
        .layout_of(TypingEnv::fully_monomorphized().as_query_input(array))
        .map_err(|_| "cannot compute storage reservation layout")?;
    let bytes =
        u32::try_from(layout.size.bytes()).map_err(|_| "storage reservation exceeds u32")?;
    let align = u32::try_from(layout.align.abi.bytes())
        .map_err(|_| "storage reservation alignment exceeds u32")?;
    Ok((bytes, align.max(alignment(extent(tcx, args, 3)?)?)))
}

pub(crate) fn validate<'tcx>(tcx: TyCtxt<'tcx>, call: &LibraryCall<'tcx>) -> Result<(), String> {
    let args = call.instance.args;
    match call.kind {
        MarkerKind::StorageNew => {
            new_policy(tcx, call)?;
        }
        MarkerKind::StorageReconfigure => {
            policy(tcx, args.type_at(0))?;
            policy(tcx, args.type_at(1))?;
        }
        MarkerKind::StorageReserve => {
            policy(tcx, args.type_at(1))?;
            reserve_layout(tcx, call)?;
        }
        MarkerKind::StorageRelease => {
            super::scalar(args.type_at(0))?;
        }
        MarkerKind::CheckedBlockStorage => {
            let ty = args.type_at(3);
            if !matches!(ty.kind(), TyKind::Tuple(fields) if fields.is_empty()) {
                borrow_policy(tcx, ty)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct Event<'tcx> {
    pub call: CallSite<'tcx>,
    pub kind: MarkerKind,
    pub origin: Option<Site>,
    pub policy: Policy,
}
#[derive(Default)]
pub(crate) struct Trace<'tcx> {
    pub contexts: BTreeMap<Site, Instance<'tcx>>,
    pub events: BTreeMap<Site, Event<'tcx>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Binding {
    pub region: Site,
    pub offset: u32,
    /// Full descriptor allocation, including explicit capacity, from this view.
    pub declared_bytes: u32,
    pub bytes: u32,
    pub alignment: u32,
    pub implicit: bool,
    pub auto_sync: bool,
    pub before: bool,
    pub after: bool,
}
#[derive(Default)]
pub(crate) struct Plan {
    pub sites: BTreeMap<Site, Binding>,
    pub bytes: u32,
    pub alignment: u32,
}

pub(crate) fn align_up(value: u32, alignment: u32) -> Result<u32, String> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| "storage layout exceeds u32".into())
}

/// Place every static use once. `layouts` comes only from validated provider
/// responses or actual Rust array layouts; no CUB algorithm names are inspected.
pub(crate) fn allocate(
    trace: &Trace<'_>,
    layouts: &BTreeMap<Site, (u32, u32)>,
) -> Result<Plan, String> {
    type Uses = Vec<(Site, u32, u32)>;
    let mut descriptors: BTreeMap<Option<Site>, (Policy, Uses)> = BTreeMap::new();
    for (site, event) in &trace.events {
        let Some(&(bytes, align)) = layouts.get(site) else {
            continue;
        };
        if event.kind == MarkerKind::StorageReserve && event.policy.auto_sync {
            return Err("storage reservations cannot use auto_sync".into());
        }
        if bytes == 0 {
            continue;
        }
        let (policy, uses) = descriptors
            .entry(event.origin.clone())
            .or_insert_with(|| (event.policy.clone(), Vec::new()));
        if *policy != event.policy {
            return Err("one storage origin has incompatible policies after use".into());
        }
        uses.push((site.clone(), bytes, align));
    }
    let mut plan = Plan {
        alignment: 1,
        ..Default::default()
    };
    for (origin, (policy, uses)) in descriptors {
        let base_align = uses
            .iter()
            .map(|(_, _, a)| *a)
            .max()
            .unwrap_or(1)
            .max(policy.alignment);
        let mut end = 0;
        let mut placed = Vec::new();
        for (site, bytes, align) in uses {
            let offset = if policy.exclusive {
                align_up(end, align)?
            } else {
                0
            };
            end = end.max(
                offset
                    .checked_add(bytes)
                    .ok_or("storage layout exceeds u32")?,
            );
            placed.push((site, offset, bytes, align));
        }
        if policy.capacity != 0 && policy.capacity < end {
            return Err(format!(
                "storage capacity {} is smaller than complete plan {end}, including exclusive padding",
                policy.capacity
            ));
        }
        let allocation = end.max(policy.capacity);
        let base = align_up(plan.bytes, base_align)?;
        plan.bytes = base
            .checked_add(allocation)
            .ok_or("storage layout exceeds u32")?;
        plan.alignment = plan.alignment.max(base_align);
        for (site, offset, bytes, align) in placed {
            // Empty is the implicit pool. Explicit origins always contain a bb.
            let mut region = origin.clone().unwrap_or_default();
            if policy.exclusive {
                region.push(usize::MAX);
                region.extend(&site);
            }
            plan.sites.insert(
                site,
                Binding {
                    region,
                    offset: base + offset,
                    declared_bytes: allocation - offset,
                    bytes,
                    alignment: align.max(if offset == 0 { base_align } else { 1 }),
                    implicit: origin.is_none(),
                    auto_sync: policy.auto_sync,
                    before: false,
                    after: policy.auto_sync,
                },
            );
        }
    }
    Ok(plan)
}

/// Bind rooted static sites after the provider has supplied actual layouts.
/// Equal local/child binding vectors share one emitted helper body within a root.
pub(crate) fn finalize<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &crate::collector::CollectionResult<'tcx>,
    validation: crate::collective_validation::Validation<'tcx>,
    result: &mut super::FinalizedLibraries<'tcx>,
) -> Result<(), String> {
    use mir_importer::{LibraryCallPlan, LibraryOperation as Op};
    let mut plans = BTreeMap::new();
    let mut natives = BTreeMap::new();
    let mut traces = BTreeMap::new();
    for (root, trace) in validation.traces {
        if !trace.events.values().any(|event| event.kind.storage()) {
            continue;
        }
        let old_context = result.contexts.remove(&root).unwrap_or_default();
        let old_names: std::collections::HashSet<_> = old_context
            .iter()
            .map(|(_, name)| name.clone())
            .chain(std::iter::once(root.clone()))
            .collect();
        let mut native = std::collections::HashMap::new();
        result.calls.retain(|(site, caller, plan)| {
            if old_names.contains(caller) {
                native.insert((site.caller, site.block.index()), plan.clone());
                false
            } else {
                true
            }
        });
        for name in &old_names {
            result.redirects.remove(name);
        }
        let mut layouts = BTreeMap::new();
        for (site, event) in &trace.events {
            if event.kind == MarkerKind::CheckedBlockStorage {
                let plan = native
                    .get(&(event.call.caller, event.call.block.index()))
                    .ok_or("missing storage provider plan")?;
                layouts.insert(site.clone(), (plan.scratch_bytes, plan.scratch_alignment));
            } else if event.kind == MarkerKind::StorageReserve {
                let call = collection
                    .library_calls
                    .iter()
                    .find(|call| {
                        call.site.caller == event.call.caller && call.site.block == event.call.block
                    })
                    .unwrap();
                layouts.insert(site.clone(), reserve_layout(tcx, call)?);
            }
        }
        plans.insert(root.clone(), allocate(&trace, &layouts)?);
        natives.insert(root.clone(), native);
        traces.insert(root, trace);
    }
    if plans.is_empty() {
        return Ok(());
    }
    crate::collective_validation::validate(tcx, collection, Some(&mut plans))?;
    let affected = super::checked_helper_ancestors(collection);
    for (root, trace) in traces {
        let plan = plans.remove(&root).unwrap();
        let native = &natives[&root];
        let mut calls = BTreeMap::<Site, LibraryCallPlan>::new();
        for (site, event) in &trace.events {
            let mut call = if event.kind.native() {
                native[&(event.call.caller, event.call.block.index())].clone()
            } else {
                LibraryCallPlan::internal(
                    match event.kind {
                        MarkerKind::StorageNew => Op::StorageNew,
                        MarkerKind::StorageReconfigure => Op::StorageReconfigure,
                        MarkerKind::StorageReserve => Op::StorageReserve,
                        MarkerKind::StorageRelease => Op::StorageRelease,
                        _ => unreachable!(),
                    },
                    tcx.symbol_name(event.call.callee).name.to_string(),
                )
            };
            if let Some(binding) = plan.sites.get(site) {
                call.scratch_offset = binding.offset;
                call.scratch_bytes = binding.declared_bytes;
                call.scratch_alignment = binding.alignment;
                call.planned_root = Some(root.clone());
                call.barrier_before = binding.before;
                call.barrier_after = binding.after;
            }
            calls.insert(site.clone(), call);
        }
        let mut names = BTreeMap::<Site, String>::new();
        let mut intern: std::collections::HashMap<(Instance<'tcx>, Vec<(usize, String)>), String> =
            std::collections::HashMap::new();
        let mut contexts = Vec::new();
        // Longer paths first, so child identity is part of a parent's vector.
        let mut paths: Vec<_> = trace
            .contexts
            .iter()
            .filter(|(_, instance)| affected.contains(instance))
            .collect();
        paths.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then(a.cmp(b)));
        for (path, &instance) in paths {
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
            let key = (instance, bindings);
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
            contexts.push((instance, name.clone()));
            result.deferred.remove(&instance);
            for (site, call) in &calls {
                if site.len() == path.len() + 1 && site.starts_with(path) {
                    result
                        .calls
                        .push((trace.events[site].call, name.clone(), call.clone()));
                }
            }
            for (bb, target) in children {
                let edge = collection
                    .call_edges
                    .iter()
                    .find(|edge| edge.caller == instance && edge.block.index() == bb)
                    .ok_or("storage child call edge is missing")?;
                if edge.kind != super::EdgeKind::Direct {
                    return Err("storage bindings require a direct helper call".into());
                }
                let mut call = LibraryCallPlan::internal(
                    Op::Redirect,
                    tcx.symbol_name(edge.callee).name.to_string(),
                );
                call.symbol = target;
                result.calls.push((*edge, name.clone(), call));
            }
        }
        let shape = validation.shapes[&root];
        result.roots.insert(
            root.clone(),
            oxide_artifacts::KernelLaunchFacts {
                exact_block: Some((shape[0], shape[1], shape[2])),
                dynamic_shared_bytes: plan.bytes,
                dynamic_shared_alignment: plan.alignment,
            },
        );
        result.contexts.insert(root, contexts);
    }
    Ok(())
}
