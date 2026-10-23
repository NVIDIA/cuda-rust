/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned stateful traces and placement of validated native storage layouts.
use dialect_mir::library::{
    LibraryCallPlan, LibraryCallSite, LibraryNativeMetadata as Native, LibraryRoleAttr as Role,
};
use std::collections::BTreeMap;

/// The authenticated source policy, shared with the retained dialect contract.
pub type Policy = dialect_mir::library::LibraryStoragePolicy;
/// Original direct-call block path, followed by the source operation's block.
pub type Site = Vec<usize>;
/// Validated provider responses keyed by root and original source call site.
pub type RootedNativePlans = BTreeMap<String, BTreeMap<LibraryCallSite, LibraryCallPlan>>;
/// One validated static storage/native use in a concrete helper context.
#[derive(Clone, Debug)]
pub struct Event {
    pub call: LibraryCallSite,
    pub role: Role,
    pub marker: String,
    pub origin: Option<Site>,
    pub policy: Policy,
}
/// Ordered static contexts and uses from one layout-free proof.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub contexts: BTreeMap<Site, String>,
    pub events: BTreeMap<Site, Event>,
}
/// Owned output retained while the host provider runs.
#[derive(Clone, Debug, Default)]
pub struct Validation {
    pub shapes: BTreeMap<String, [u32; 3]>,
    pub traces: BTreeMap<String, Trace>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Binding {
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
pub struct Plan {
    pub sites: BTreeMap<Site, Binding>,
    pub bytes: u32,
    pub alignment: u32,
}

pub fn align_up(value: u32, alignment: u32) -> Result<u32, String> {
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(|| "storage layout exceeds u32".into())
}

/// Place every static use once. `layouts` comes only from validated provider
/// responses or actual Rust array layouts; no CUB algorithm names are inspected.
pub fn allocate(trace: &Trace, layouts: &BTreeMap<Site, (u32, u32)>) -> Result<Plan, String> {
    type Uses = Vec<(Site, u32, u32)>;
    let mut descriptors: BTreeMap<Option<Site>, (Policy, Uses)> = BTreeMap::new();
    for (site, event) in &trace.events {
        let Some(&(bytes, align)) = layouts.get(site) else {
            continue;
        };
        if matches!(event.role, Role::StorageReserve { .. }) && event.policy.auto_sync {
            return Err("storage reservations cannot use auto_sync".into());
        }
        if bytes == 0 {
            continue;
        }
        let (policy, uses) = descriptors
            .entry(event.origin.clone())
            .or_insert_with(|| (event.policy, Vec::new()));
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

/// Whether source reachability activates the planned storage subset.
pub fn planned_storage(role: Role) -> bool {
    matches!(
        role,
        Role::StorageNew(_)
            | Role::StorageReconfigure { .. }
            | Role::StorageReserve { .. }
            | Role::StorageRelease
            | Role::Native(Native::CheckedBlockStorage { .. })
    )
}

/// Join only validated provider layouts or authenticated reservation layouts.
pub fn allocate_library_storage(
    validation: &Validation,
    native_plans: &RootedNativePlans,
) -> Result<BTreeMap<String, Plan>, String> {
    let mut plans = BTreeMap::new();
    for (root, trace) in &validation.traces {
        if !trace
            .events
            .values()
            .any(|event| planned_storage(event.role))
        {
            continue;
        }
        let mut layouts = BTreeMap::new();
        for (site, event) in &trace.events {
            match event.role {
                Role::Native(Native::CheckedBlockStorage { .. }) => {
                    let plan = native_plans
                        .get(root)
                        .and_then(|calls| calls.get(&event.call))
                        .ok_or("missing storage provider plan")?;
                    layouts.insert(site.clone(), (plan.scratch_bytes, plan.scratch_alignment));
                }
                Role::StorageReserve { layout, .. } => {
                    layouts.insert(site.clone(), (layout.bytes, layout.alignment));
                }
                _ => {}
            }
        }
        plans.insert(root.clone(), allocate(trace, &layouts)?);
    }
    Ok(plans)
}
