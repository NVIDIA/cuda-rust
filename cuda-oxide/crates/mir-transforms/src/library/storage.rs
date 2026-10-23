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
/// Complete aligned partitions. Prefixes never enter this numeric ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scope {
    Warp(u32),
    Block,
}
impl Scope {
    pub(super) const LANE: Self = Self::Warp(1);
    pub(super) const HEIGHT: usize = 7;
}

/// A closed value/participation or scratch domain. Ord is only for map keys;
/// semantic implication and meet always use the explicit methods below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    Complete(Scope),
    Prefix(u32),
}
impl Domain {
    pub(super) const BLOCK: Self = Self::Complete(Scope::Block);
    pub(super) const LANE: Self = Self::Complete(Scope::LANE);

    pub(super) fn covers(self, needed: Self) -> bool {
        match (self, needed) {
            (Self::Complete(a), Self::Complete(b)) => a >= b,
            (Self::Complete(Scope::Block | Scope::Warp(32)), Self::Prefix(_)) => true,
            (Self::Prefix(a), Self::Prefix(b)) => a == b,
            (_, Self::LANE) => true,
            _ => false,
        }
    }
    pub(super) fn meet(self, other: Self) -> Self {
        if self.covers(other) {
            other
        } else if other.covers(self) {
            self
        } else {
            Self::LANE
        }
    }
    pub(super) fn complete_scope(self) -> Option<Scope> {
        match self {
            Self::Complete(scope) => Some(scope),
            Self::Prefix(_) => None,
        }
    }
    pub(super) fn replica_width(self) -> Option<u32> {
        match self {
            Self::Complete(Scope::Warp(w)) => Some(w),
            Self::Prefix(_) => Some(32),
            Self::Complete(Scope::Block) => None,
        }
    }
}

/// One validated static storage/native use in a concrete helper context.
#[derive(Clone, Debug)]
pub struct Event {
    pub call: LibraryCallSite,
    pub role: Role,
    pub marker: String,
    pub scope: Domain,
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
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Region {
    Implicit(Domain),
    Explicit(Site),
}
impl Default for Region {
    fn default() -> Self {
        Self::Implicit(Domain::BLOCK)
    }
}
impl Region {
    pub(super) fn completed_by(&self, scope: Domain) -> bool {
        match (scope, self) {
            (Domain::BLOCK, _) => true,
            (Domain::Complete(scope), Self::Implicit(Domain::Complete(producer))) => {
                scope >= *producer
            }
            (Domain::Complete(Scope::Warp(32)), Self::Implicit(Domain::Prefix(_))) => true,
            (Domain::Prefix(a), Self::Implicit(Domain::Prefix(b))) => a == *b,
            _ => false,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Binding {
    pub region: Region,
    /// Zero for Block views; otherwise the aligned per-Warp replica extent.
    pub stride: u32,
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
pub fn allocate(
    trace: &Trace,
    layouts: &BTreeMap<Site, (u32, u32)>,
    block_threads: u32,
) -> Result<Plan, String> {
    type Uses = Vec<(Site, u32, u32)>;
    let mut descriptors: BTreeMap<Region, (Policy, Uses)> = BTreeMap::new();
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
            .entry(match &event.origin {
                Some(origin) => Region::Explicit(origin.clone()),
                None => Region::Implicit(event.scope),
            })
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
        let width = match &origin {
            Region::Implicit(domain) => domain.replica_width(),
            Region::Explicit(_) => None,
        };
        let stride = if width.is_some() {
            align_up(end, base_align)?
        } else {
            0
        };
        let allocation = if let Some(width) = width {
            if !block_threads.is_multiple_of(width) {
                return Err("Warp pool width does not divide exact block".into());
            }
            stride
                .checked_mul(block_threads / width)
                .ok_or("replicated Warp storage exceeds u32")?
        } else {
            end.max(policy.capacity)
        };
        let base = align_up(plan.bytes, base_align)?;
        plan.bytes = base
            .checked_add(allocation)
            .ok_or("storage layout exceeds u32")?;
        plan.alignment = plan.alignment.max(base_align);
        for (site, offset, bytes, align) in placed {
            let mut region = origin.clone();
            if policy.exclusive
                && let Region::Explicit(id) = &mut region
            {
                id.push(usize::MAX);
                id.extend(&site);
            }
            plan.sites.insert(
                site,
                Binding {
                    region,
                    stride,
                    offset: base + offset,
                    declared_bytes: allocation - offset,
                    bytes,
                    alignment: align.max(if offset == 0 { base_align } else { 1 }),
                    implicit: matches!(origin, Region::Implicit(_)),
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
            | Role::Native(
                Native::CheckedBlockStorage { .. }
                    | Native::CheckedWarp { .. }
                    | Native::CheckedBlockPayload { .. }
                    | Native::CheckedBlockPayloadCallback { .. }
                    | Native::CheckedWarpPayload { .. }
            )
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
                Role::Native(
                    Native::CheckedBlockStorage { .. }
                    | Native::CheckedWarp { .. }
                    | Native::CheckedBlockPayload { .. }
                    | Native::CheckedBlockPayloadCallback { .. }
                    | Native::CheckedWarpPayload { .. },
                ) => {
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
        let shape = validation
            .shapes
            .get(root)
            .ok_or("storage root has no admitted MIR geometry")?;
        let threads = shape
            .iter()
            .copied()
            .try_fold(1_u32, u32::checked_mul)
            .filter(|n| *n != 0)
            .ok_or("storage plan requires a nonzero exact block extent fitting u32")?;
        plans.insert(root.clone(), allocate(trace, &layouts, threads)?);
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::library::{LibraryAccess, LibraryScalar, LibraryWitness, NativeSpec};

    #[test]
    fn replica_pools_are_aligned_disjoint_and_checked_for_overflow() {
        let spec = NativeSpec {
            scalar: LibraryScalar::Signed(32),
            items: 4,
        };
        let event = |width| Event {
            call: LibraryCallSite {
                caller: "root".into(),
                block: width as usize,
            },
            role: Role::Native(Native::CheckedWarp {
                spec,
                width,
                access: LibraryAccess::Read,
                witness: LibraryWitness::ReadOnlySlice,
            }),
            marker: "warp".into(),
            scope: Domain::Complete(Scope::Warp(width)),
            origin: None,
            policy: Policy::default(),
        };
        let mut trace = Trace::default();
        trace.events.insert(vec![8], event(8));
        trace.events.insert(vec![16], event(16));
        let layouts = BTreeMap::from([(vec![8], (148, 4)), (vec![16], (52, 16))]);
        let plan = allocate(&trace, &layouts, 128).unwrap();
        assert_eq!((plan.bytes, plan.alignment), (2880, 16));
        assert_eq!(
            (
                plan.sites[&vec![8]].stride,
                plan.sites[&vec![8]].declared_bytes
            ),
            (148, 2368)
        );
        assert_eq!(
            (plan.sites[&vec![16]].stride, plan.sites[&vec![16]].offset),
            (64, 2368)
        );
        assert_ne!(plan.sites[&vec![8]].region, plan.sites[&vec![16]].region);
        assert!(
            allocate(&trace, &layouts, 33)
                .err()
                .unwrap()
                .contains("does not divide")
        );
        let layouts = BTreeMap::from([(vec![8], (1 << 30, 1))]);
        assert!(
            allocate(&trace, &layouts, 64)
                .err()
                .unwrap()
                .contains("replicated Warp storage exceeds")
        );
        let layouts = BTreeMap::from([(vec![8], (u32::MAX, 4))]);
        assert!(
            allocate(&trace, &layouts, 64)
                .err()
                .unwrap()
                .contains("storage layout exceeds")
        );

        // Prefix widths name distinct pools, but each pool has one replica per
        // physical Warp rather than per WIDTH lanes.
        let mut prefix = Trace::default();
        for width in [17, 31] {
            let mut use_ = event(width);
            use_.role = Role::Native(Native::CheckedWarpPayload { spec, width });
            use_.scope = Domain::Prefix(width);
            prefix.events.insert(vec![width as usize], use_);
        }
        let layouts = BTreeMap::from([(vec![17], (17, 16)), (vec![31], (1, 16))]);
        let plan = allocate(&prefix, &layouts, 64).unwrap();
        assert_eq!((plan.bytes, plan.alignment), (96, 16));
        assert_eq!(
            (
                plan.sites[&vec![17]].stride,
                plan.sites[&vec![17]].declared_bytes
            ),
            (32, 64)
        );
        assert_eq!(
            (plan.sites[&vec![31]].stride, plan.sites[&vec![31]].offset),
            (16, 64)
        );
        assert_ne!(plan.sites[&vec![17]].region, plan.sites[&vec![31]].region);
        assert!(allocate(&prefix, &layouts, 63).is_err());
    }
}
