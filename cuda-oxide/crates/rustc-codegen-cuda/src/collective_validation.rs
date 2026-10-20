// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, per-call-context checks for the checked full-block library marker.
use crate::collector::{CollectionResult, device_mono_reachability};
use crate::library::storage::{self, Event, Plan, Policy, Site, Trace};
use crate::library::{EdgeKind, MarkerKind, reachable_instances};
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_public::{mir, rustc_internal};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fact {
    Unset,
    Uniform,
    Kernel(String),
    Resource(String),
    Descriptor(Site, Policy),
    Reservation(Site),
    Local(usize, Vec<usize>),
    Referent(Box<Fact>),
    Fields(Vec<Fact>),
    Varying,
}
impl Fact {
    fn uniform(&self) -> bool {
        matches!(
            self,
            Self::Uniform
                | Self::Kernel(_)
                | Self::Resource(_)
                | Self::Descriptor(..)
                | Self::Reservation(_)
        )
    }
    fn merge(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Unset, value) | (value, Self::Unset) => value.clone(),
            (Self::Uniform, Self::Kernel(_)) | (Self::Kernel(_), Self::Uniform) => Self::Uniform,
            (Self::Fields(a), Self::Fields(b)) if a.len() == b.len() => {
                Self::Fields(a.iter().zip(b).map(|(a, b)| a.merge(b)).collect())
            }
            _ => Self::Varying,
        }
    }
    fn taint(self) -> Self {
        match self {
            Self::Fields(v) => Self::Fields(v.into_iter().map(Self::taint).collect()),
            Self::Local(..) => self,
            _ => Self::Varying,
        }
    }
    fn has_resource(&self) -> bool {
        match self {
            Self::Resource(_) | Self::Descriptor(..) | Self::Reservation(_) => true,
            Self::Referent(value) => value.has_resource(),
            Self::Fields(values) => values.iter().any(Self::has_resource),
            _ => false,
        }
    }
    fn invalidate_borrow(&mut self) {
        match self {
            Self::Resource(_) | Self::Descriptor(..) | Self::Reservation(_) => {}
            Self::Referent(value) if value.has_resource() => value.invalidate_borrow(),
            Self::Fields(values) if values.iter().any(Self::has_resource) => {
                values.iter_mut().for_each(Self::invalidate_borrow);
            }
            _ => *self = Self::Varying,
        }
    }
    fn descriptor(&self) -> Option<(Site, Policy)> {
        match self {
            Self::Descriptor(origin, policy) => Some((origin.clone(), policy.clone())),
            Self::Referent(value) => value.descriptor(),
            _ => None,
        }
    }
    fn reservation(&self) -> Option<Site> {
        match self {
            Self::Reservation(site) => Some(site.clone()),
            Self::Referent(value) => value.reservation(),
            _ => None,
        }
    }
    fn resource_origin(&self) -> Option<&str> {
        match self {
            Self::Resource(origin) => Some(origin),
            Self::Referent(value) => value.resource_origin(),
            _ => None,
        }
    }
    fn invalidate_referents(&mut self) {
        match self {
            Self::Referent(_) => self.invalidate_borrow(),
            Self::Fields(fields) => fields.iter_mut().for_each(Self::invalidate_referents),
            _ => {}
        }
    }
    fn field(&self, index: usize) -> Self {
        match self {
            Self::Fields(v) => v.get(index).cloned().unwrap_or(Self::Varying),
            Self::Kernel(origin) | Self::Resource(origin) => {
                Self::Kernel(format!("{origin}.{index}"))
            }
            Self::Uniform | Self::Reservation(_) => Self::Uniform,
            _ => Self::Varying,
        }
    }
}
// Pending access bits: read=1, write=2. Joining paths retains every possible
// access; a proved full-block barrier clears them. This is not a lifetime token.
type Pending = BTreeMap<String, u8>;
fn merge_pending(into: &mut Pending, other: &Pending) -> bool {
    let mut changed = false;
    for (origin, bits) in other {
        let old = into.entry(origin.clone()).or_default();
        if *old | bits != *old {
            *old |= bits;
            changed = true;
        }
    }
    changed
}
#[derive(Clone, Default)]
struct Epochs {
    dirty: BTreeSet<Site>,
    live: BTreeMap<Site, Site>,
    used: BTreeMap<Site, BTreeSet<Policy>>,
}
impl Epochs {
    fn merge(&mut self, other: &Self) -> bool {
        let before = (self.dirty.len(), self.live.len());
        self.dirty.extend(other.dirty.iter().cloned());
        self.live
            .extend(other.live.iter().map(|(a, b)| (a.clone(), b.clone())));
        let mut changed = before != (self.dirty.len(), self.live.len());
        for (origin, policies) in &other.used {
            let current = self.used.entry(origin.clone()).or_default();
            let before = current.len();
            current.extend(policies.iter().cloned());
            changed |= before != current.len();
        }
        changed
    }
}
#[derive(Clone)]
struct State {
    locals: Vec<Fact>,
    pending: Pending,
    epochs: Epochs,
}
impl std::ops::Deref for State {
    type Target = Vec<Fact>;
    fn deref(&self) -> &Self::Target {
        &self.locals
    }
}
impl std::ops::DerefMut for State {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.locals
    }
}
fn local_read(state: &State, local: usize, fields: &[usize]) -> Fact {
    fields
        .iter()
        .fold(state[local].clone(), |fact, index| fact.field(*index))
}
fn read(place: &mir::Place, state: &State) -> Fact {
    let mut fact = state[place.local].clone();
    for projection in &place.projection {
        fact = match projection {
            mir::ProjectionElem::Field(index, _) => fact.field(*index),
            mir::ProjectionElem::Deref => match fact {
                Fact::Local(local, fields) => local_read(state, local, &fields),
                Fact::Referent(value) => *value,
                _ => Fact::Varying,
            },
            mir::ProjectionElem::Downcast(_) => fact,
            _ => Fact::Varying,
        };
    }
    fact
}
fn operand(value: &mir::Operand, state: &State) -> Fact {
    match value {
        mir::Operand::Copy(p) | mir::Operand::Move(p) => read(p, state),
        mir::Operand::Constant(_) | mir::Operand::RuntimeChecks(_) => Fact::Uniform,
    }
}
fn address(place: &mir::Place, state: &State) -> Fact {
    let mut local = place.local;
    let mut fields = Vec::new();
    for (index, projection) in place.projection.iter().enumerate() {
        match projection {
            mir::ProjectionElem::Field(index, _) => fields.push(*index),
            mir::ProjectionElem::Deref => match local_read(state, local, &fields) {
                Fact::Local(base, path) => {
                    local = base;
                    fields = path;
                }
                fact if index + 1 == place.projection.len() => return fact,
                _ => return Fact::Varying,
            },
            _ => return Fact::Varying,
        }
    }
    Fact::Local(local, fields)
}
fn rvalue(value: &mir::Rvalue, state: &State) -> Fact {
    match value {
        mir::Rvalue::Use(v, _) | mir::Rvalue::Cast(_, v, _) => operand(v, state),
        mir::Rvalue::CopyForDeref(p) => read(p, state),
        mir::Rvalue::Ref(_, _, p) | mir::Rvalue::AddressOf(_, p) => address(p, state),
        mir::Rvalue::Aggregate(_, values) => {
            Fact::Fields(values.iter().map(|v| operand(v, state)).collect())
        }
        mir::Rvalue::Repeat(v, _) => operand(v, state),
        mir::Rvalue::UnaryOp(_, v) => {
            if operand(v, state).uniform() {
                Fact::Uniform
            } else {
                Fact::Varying
            }
        }
        mir::Rvalue::BinaryOp(op, a, b) => {
            let a = operand(a, state);
            let b = operand(b, state);
            if matches!(op, mir::BinOp::Offset) && a.uniform() && b.uniform() {
                a
            } else if a.uniform() && b.uniform() {
                Fact::Uniform
            } else {
                Fact::Varying
            }
        }
        mir::Rvalue::CheckedBinaryOp(_, a, b) => {
            let f = if operand(a, state).uniform() && operand(b, state).uniform() {
                Fact::Uniform
            } else {
                Fact::Varying
            };
            Fact::Fields(vec![f.clone(), f])
        }
        _ => Fact::Varying,
    }
}
fn write_fields(target: &mut Fact, fields: &[usize], value: Fact) {
    if let Some((index, rest)) = fields.split_first() {
        if let Fact::Fields(values) = target
            && let Some(field) = values.get_mut(*index)
        {
            write_fields(field, rest, value);
            return;
        }
        *target = Fact::Varying;
    } else {
        *target = value;
    }
}
fn invalidate_borrows(state: &mut State, borrowed: &HashSet<usize>) {
    for (local, fact) in state.iter_mut().enumerate() {
        if borrowed.contains(&local) {
            fact.invalidate_borrow();
        } else {
            fact.invalidate_referents();
        }
    }
}
fn write(place: &mir::Place, value: Fact, state: &mut State, borrowed: &HashSet<usize>) {
    let mut local = place.local;
    let mut fields = Vec::new();
    for projection in &place.projection {
        match projection {
            mir::ProjectionElem::Field(index, _) => fields.push(*index),
            mir::ProjectionElem::Deref => {
                if let Fact::Local(base, path) = local_read(state, local, &fields) {
                    local = base;
                    fields = path;
                } else {
                    invalidate_borrows(state, borrowed);
                    return;
                }
            }
            _ => {
                if place
                    .projection
                    .iter()
                    .any(|projection| matches!(projection, mir::ProjectionElem::Deref))
                {
                    invalidate_borrows(state, borrowed);
                }
                state[local] = Fact::Varying;
                return;
            }
        }
    }
    write_fields(&mut state[local], &fields, value);
}
// Local indices belong to one MIR body. Resolve the entire reference graph
// before crossing a call boundary; a nested Local must never index the callee.
fn call_fact(fact: Fact, state: &State) -> Fact {
    fn resolve(fact: Fact, state: &State, visiting: &mut HashSet<usize>) -> Fact {
        match fact {
            Fact::Local(local, fields) => {
                if !visiting.insert(local) {
                    return Fact::Varying;
                }
                let value = resolve(local_read(state, local, &fields), state, visiting);
                visiting.remove(&local);
                Fact::Referent(Box::new(value))
            }
            Fact::Referent(value) => Fact::Referent(Box::new(resolve(*value, state, visiting))),
            Fact::Fields(values) => Fact::Fields(
                values
                    .into_iter()
                    .map(|value| resolve(value, state, visiting))
                    .collect(),
            ),
            other => other,
        }
    }
    resolve(fact, state, &mut HashSet::new())
}

// Worklist bottom and callee-owned addresses are not caller facts. This bounded
// analysis does not prove references newly created by a helper.
fn return_fact(fact: Fact) -> Fact {
    match fact {
        Fact::Unset | Fact::Local(..) => Fact::Varying,
        Fact::Fields(values) => Fact::Fields(values.into_iter().map(return_fact).collect()),
        Fact::Referent(value) => Fact::Referent(Box::new(return_fact(*value))),
        other => other,
    }
}
fn reachable(edges: &[Vec<usize>], start: usize, stop: Option<usize>) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    let mut pending = vec![start];
    while let Some(block) = pending.pop() {
        if Some(block) != stop && result.insert(block) {
            pending.extend(edges[block].iter().copied());
        }
    }
    result
}
fn joins(edges: &[Vec<usize>]) -> Vec<Option<usize>> {
    let exit = edges.len();
    let all: BTreeSet<_> = (0..=exit).collect();
    let mut post = vec![all; exit + 1];
    post[exit] = BTreeSet::from([exit]);
    loop {
        let mut changed = false;
        for block in (0..exit).rev() {
            let targets = if edges[block].is_empty() {
                vec![exit]
            } else {
                edges[block].clone()
            };
            let mut next = post[targets[0]].clone();
            for target in &targets[1..] {
                next = next.intersection(&post[*target]).copied().collect();
            }
            next.insert(block);
            if next != post[block] {
                post[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (0..exit)
        .map(|block| {
            post[block]
                .iter()
                .copied()
                .filter(|v| *v != block && *v != exit)
                .find(|candidate| {
                    post[block].iter().all(|other| {
                        *other == block || *other == *candidate || post[*candidate].contains(other)
                    })
                })
        })
        .collect()
}
struct Analyzer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    collection: &'a CollectionResult<'tcx>,
    root: &'a str,
    stack: Vec<Instance<'tcx>>,
    resource_root: bool,
    storage_root: bool,
    path: Site,
    trace: &'a mut Trace<'tcx>,
    plan: Option<&'a mut Plan>,
}
impl<'tcx> Analyzer<'_, 'tcx> {
    fn error(&self, instance: Instance<'tcx>, block: usize, message: &str) -> String {
        format!(
            "checked block collective in {}: {message}; {} MIR block {block}",
            self.root,
            self.tcx.def_path_str(instance.def_id())
        )
    }
    fn callee(&self, instance: Instance<'tcx>, block: usize) -> Option<(Instance<'tcx>, EdgeKind)> {
        self.collection
            .call_edges
            .iter()
            .find(|e| e.caller == instance && e.block.index() == block)
            .map(|e| (e.callee, e.kind))
    }
    fn intrinsic(&self, instance: Instance<'tcx>) -> Option<Fact> {
        let path = self.tcx.def_path_str(instance.def_id());
        let krate = self.tcx.crate_name(instance.def_id().krate);
        if !crate::generated_intrinsics::is_generated_intrinsic_placeholder(krate.as_str(), &path) {
            return None;
        }
        let uniform = matches!(
            path.rsplit("::").next().unwrap_or(""),
            "blockDim_x"
                | "blockDim_y"
                | "blockDim_z"
                | "blockIdx_x"
                | "blockIdx_y"
                | "blockIdx_z"
                | "gridDim_x"
                | "gridDim_y"
                | "gridDim_z"
                | "i0003"
                | "i0007"
                | "i0011"
                | "i0002"
                | "i0006"
                | "i0010"
                | "i0004"
                | "i0008"
                | "i0012"
        );
        Some(if uniform {
            Fact::Uniform
        } else {
            Fact::Varying
        })
    }
    fn analyze(
        &mut self,
        instance: Instance<'tcx>,
        arguments: Vec<Fact>,
        validate: bool,
        entry_varying: bool,
        pending: Pending,
        epochs: Epochs,
    ) -> Result<(Fact, Pending, Epochs), String> {
        if self.stack.contains(&instance) {
            return if validate {
                Err(self.error(instance, 0, "recursive helper participation is unsupported"))
            } else {
                Ok((Fact::Varying, pending, epochs))
            };
        }
        if matches!(instance.def, rustc_middle::ty::InstanceKind::Intrinsic(_)) {
            return Err(self.error(
                instance,
                0,
                "intrinsic is outside the supported checked block subset",
            ));
        }
        if validate {
            self.trace.contexts.insert(self.path.clone(), instance);
        }
        let stable: mir::mono::Instance = rustc_internal::stable(instance);
        let Some(body) = stable.body() else {
            return if validate {
                Err(self.error(
                    instance,
                    0,
                    "call without MIR cannot prove collective participation",
                ))
            } else {
                Ok((Fact::Varying, pending, epochs))
            };
        };
        let supplied = device_mono_reachability(self.tcx, instance);
        if body.blocks.len() != supplied.block_count {
            return Err("checked collective MIR bridge block count mismatch".into());
        }
        let mut edges = mir_importer::checked_collective_successors(
            &body,
            supplied.block_count,
            &supplied.successors,
        )?;
        // Exhaustive enum matches retain an impossible `otherwise` arm in
        // MIR. It cannot execute in a defined Rust program and must not make
        // the real arms appear to bypass their reconvergence point.
        for (block, data) in body.blocks.iter().enumerate() {
            if let mir::TerminatorKind::SwitchInt { targets, .. } = &data.terminator.kind {
                let otherwise = targets.otherwise();
                if body.blocks[otherwise].statements.is_empty()
                    && matches!(
                        body.blocks[otherwise].terminator.kind,
                        mir::TerminatorKind::Unreachable
                    )
                {
                    edges[block].retain(|target| *target != otherwise);
                }
            }
        }
        let live = reachable(&edges, 0, None);
        let joins = joins(&edges);
        let mut incoming: Vec<Option<State>> = vec![None; body.blocks.len()];
        let mut entry = State {
            locals: vec![Fact::Unset; body.locals().len()],
            pending,
            epochs,
        };
        for (local, fact) in entry.iter_mut().skip(1).zip(arguments) {
            *local = fact;
        }
        incoming[0] = Some(entry);
        let mut divergent = vec![entry_varying; body.blocks.len()];
        let mut varying_cycles = BTreeSet::new();
        let mut borrowed = HashSet::new();
        for &block in &live {
            for statement in &body.blocks[block].statements {
                if let mir::StatementKind::Assign(
                    _,
                    mir::Rvalue::Ref(_, _, p) | mir::Rvalue::AddressOf(_, p),
                ) = &statement.kind
                {
                    borrowed.insert(p.local);
                }
            }
        }
        self.stack.push(instance);
        let limit = body
            .blocks
            .len()
            .saturating_mul(
                body.locals().len()
                    + self.collection.library_calls.len() * 4
                    + self.plan.as_ref().map_or(0, |p| p.sites.len() * 4)
                    + 4,
            )
            .max(8);
        let mut settled = false;
        for _ in 0..limit {
            let mut changed = false;
            for &block in &live {
                let Some(mut state) = incoming[block].clone() else {
                    continue;
                };
                self.statements(&body, block, &mut state, divergent[block], &borrowed);
                if let mir::TerminatorKind::SwitchInt { discr, .. } =
                    &body.blocks[block].terminator.kind
                    && !operand(discr, &state).uniform()
                    && edges[block].len() > 1
                {
                    if edges[block]
                        .iter()
                        .any(|target| reachable(&edges, *target, joins[block]).contains(&block))
                    {
                        varying_cycles.insert(block);
                    }
                    for &target in &edges[block] {
                        let region = reachable(&edges, target, joins[block]);
                        for &member in &region {
                            if !divergent[member] {
                                divergent[member] = true;
                                changed = true;
                            }
                            if edges[member].iter().any(|next| {
                                region.contains(next)
                                    && reachable(&edges, *next, joins[block]).contains(&member)
                            }) {
                                varying_cycles.insert(block);
                            }
                        }
                    }
                }
                self.call(
                    &body,
                    instance,
                    block,
                    &mut state,
                    false,
                    divergent[block],
                    &borrowed,
                )?;
                for &target in &edges[block] {
                    if let Some(previous) = &mut incoming[target] {
                        changed |= merge_pending(&mut previous.pending, &state.pending);
                        changed |= previous.epochs.merge(&state.epochs);
                        for (old, new) in previous.iter_mut().zip(state.locals.iter()) {
                            let merged = old.merge(new);
                            if merged != *old {
                                *old = merged;
                                changed = true;
                            }
                        }
                    } else {
                        incoming[target] = Some(state.clone());
                        changed = true;
                    }
                }
            }
            if !changed {
                settled = true;
                break;
            }
        }
        if !settled {
            self.stack.pop();
            return Err(self.error(
                instance,
                0,
                "value analysis did not converge in its bounded subset",
            ));
        }
        if validate && let Some(&block) = varying_cycles.first() {
            self.stack.pop();
            return Err(self.error(
                instance,
                block,
                "lane-varying cycle cannot prove collective participation",
            ));
        }
        let mut returned = Fact::Unset;
        let mut returned_pending = Pending::new();
        let mut returned_epochs = Epochs::default();
        for &block in &live {
            let Some(mut state) = incoming[block].clone() else {
                continue;
            };
            if validate && (self.resource_root || self.storage_root) {
                self.check_resource_mutation(&body, instance, block, &state)?;
            }
            self.statements(&body, block, &mut state, divergent[block], &borrowed);
            if validate {
                if matches!(
                    body.blocks[block].terminator.kind,
                    mir::TerminatorKind::Unreachable
                ) {
                    return Err(self.error(
                        instance,
                        block,
                        "unreachable terminator is not a proved whole-kernel abort",
                    ));
                }
                if matches!(
                    body.blocks[block].terminator.kind,
                    mir::TerminatorKind::InlineAsm { .. }
                ) || (matches!(
                    body.blocks[block].terminator.kind,
                    mir::TerminatorKind::Drop { .. }
                ) && self.callee(instance, block).is_some())
                {
                    return Err(self.error(
                        instance,
                        block,
                        "inline assembly or nontrivial destructor participation is unsupported",
                    ));
                }
                self.call(
                    &body,
                    instance,
                    block,
                    &mut state,
                    true,
                    divergent[block],
                    &borrowed,
                )?;
            }
            if matches!(
                body.blocks[block].terminator.kind,
                mir::TerminatorKind::Return
            ) {
                returned = returned.merge(&return_fact(state[0].clone()));
                merge_pending(&mut returned_pending, &state.pending);
                returned_epochs.merge(&state.epochs);
            }
        }
        self.stack.pop();
        // Worklist bottom is not evidence about a call's result. In
        // particular, importer-rewritten host-panic placeholders have no
        // Rust Return block but can produce lane-varying device values.
        Ok((return_fact(returned), returned_pending, returned_epochs))
    }
    fn check_resource_mutation(
        &self,
        body: &mir::Body,
        instance: Instance<'tcx>,
        block: usize,
        state: &State,
    ) -> Result<(), String> {
        for statement in &body.blocks[block].statements {
            if matches!(
                statement.kind,
                mir::StatementKind::Intrinsic(mir::NonDivergingIntrinsic::CopyNonOverlapping(_))
            ) {
                return Err(self.error(
                    instance,
                    block,
                    "bulk memory mutation is unsupported in resource-checked roots",
                ));
            }
            let mir::StatementKind::Assign(place, _) = &statement.kind else {
                continue;
            };
            if !place
                .projection
                .iter()
                .any(|p| matches!(p, mir::ProjectionElem::Deref))
            {
                continue;
            }
            for length in 0..=place.projection.len() {
                let prefix = mir::Place {
                    local: place.local,
                    projection: place.projection[..length].to_vec(),
                };
                let mut carrier = false;
                if let Ok(ty) = prefix.ty(body.locals()) {
                    let mut ty = rustc_internal::internal(self.tcx, ty);
                    loop {
                        if crate::library::resource_parameter(self.tcx, ty)
                            || storage::storage_handle_type(self.tcx, ty)
                        {
                            carrier = true;
                            break;
                        }
                        match ty.kind() {
                            rustc_middle::ty::TyKind::Ref(_, inner, _)
                            | rustc_middle::ty::TyKind::RawPtr(inner, _) => ty = *inner,
                            _ => break,
                        }
                    }
                }
                if carrier || read(&prefix, state).has_resource() {
                    return Err(self.error(instance, block,
                        "mutation through a reference to a resource descriptor is unsupported; forward or move the carrier"));
                }
            }
        }
        Ok(())
    }
    fn statements(
        &self,
        body: &mir::Body,
        block: usize,
        state: &mut State,
        divergent: bool,
        borrowed: &HashSet<usize>,
    ) {
        for statement in &body.blocks[block].statements {
            if let mir::StatementKind::Assign(place, value) = &statement.kind {
                let value = rvalue(value, state);
                write(
                    place,
                    if divergent { value.taint() } else { value },
                    state,
                    borrowed,
                );
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn call(
        &mut self,
        body: &mir::Body,
        instance: Instance<'tcx>,
        block: usize,
        state: &mut State,
        validate: bool,
        divergent: bool,
        borrowed: &HashSet<usize>,
    ) -> Result<(), String> {
        let mir::TerminatorKind::Call {
            args,
            destination,
            target,
            ..
        } = &body.blocks[block].terminator.kind
        else {
            return Ok(());
        };
        if mir_importer::is_dropped_panic_call(&body.blocks[block].terminator) {
            return Ok(());
        }
        let Some((callee, kind)) = self.callee(instance, block) else {
            if validate {
                return Err(self.error(
                    instance,
                    block,
                    "unresolved call cannot prove collective participation",
                ));
            }
            invalidate_borrows(state, borrowed);
            write(destination, Fact::Varying, state, borrowed);
            return Ok(());
        };
        if let Some(call) = self
            .collection
            .library_calls
            .iter()
            .find(|c| c.site.caller == instance && c.site.block.index() == block)
        {
            let mut site = self.path.clone();
            site.push(block);
            if call.kind.storage() {
                if validate && divergent {
                    return Err(self.error(
                        instance,
                        block,
                        "storage operation is reached through lane-varying control flow",
                    ));
                }
                let mut origin = None;
                let mut policy = Policy::default();
                let result = match call.kind {
                    MarkerKind::StorageNew => {
                        policy = storage::new_policy(self.tcx, call)?;
                        Fact::Descriptor(site.clone(), policy.clone())
                    }
                    MarkerKind::StorageReconfigure => {
                        let descriptor = call_fact(operand(&args[0], state), state).descriptor();
                        if let Some((old, _)) = descriptor {
                            origin = Some(old.clone());
                            policy = storage::policy(self.tcx, call.instance.args.type_at(1))?;
                            if validate
                                && state.epochs.used.get(&old).is_some_and(|used| {
                                    used.iter().any(|previous| *previous != policy)
                                })
                            {
                                return Err(self.error(
                                    instance,
                                    block,
                                    "storage policy cannot change after use",
                                ));
                            }
                            Fact::Descriptor(old, policy.clone())
                        } else if validate {
                            return Err(self.error(
                                instance,
                                block,
                                "storage reconfiguration has ambiguous descriptor origin",
                            ));
                        } else {
                            Fact::Varying
                        }
                    }
                    MarkerKind::StorageReserve | MarkerKind::CheckedBlockStorage => {
                        let index = if call.kind == MarkerKind::StorageReserve {
                            0
                        } else {
                            2
                        };
                        let fact = call_fact(operand(&args[index], state), state);
                        if let Some((id, p)) = fact.descriptor() {
                            origin = Some(id);
                            policy = p;
                        } else if call.kind == MarkerKind::CheckedBlockStorage
                            && matches!(call.instance.args.type_at(3).kind(),rustc_middle::ty::TyKind::Tuple(fields) if fields.is_empty())
                        {
                        } else if validate {
                            return Err(self.error(
                                instance,
                                block,
                                "storage use has no unambiguous descriptor origin",
                            ));
                        }
                        if call.kind == MarkerKind::StorageReserve {
                            Fact::Reservation(site.clone())
                        } else {
                            Fact::Fields(vec![])
                        }
                    }
                    MarkerKind::StorageRelease => {
                        if let Some(reservation) =
                            call_fact(operand(&args[0], state), state).reservation()
                        {
                            if let Some(region) = state.epochs.live.remove(&reservation) {
                                state.epochs.dirty.insert(region);
                            } else if validate && self.plan.is_some() {
                                return Err(self.error(
                                    instance,
                                    block,
                                    "storage release has no live reservation",
                                ));
                            }
                        } else if validate {
                            return Err(self.error(
                                instance,
                                block,
                                "storage release has ambiguous reservation origin",
                            ));
                        }
                        Fact::Fields(vec![])
                    }
                    _ => unreachable!(),
                };
                if matches!(
                    call.kind,
                    MarkerKind::StorageReserve | MarkerKind::CheckedBlockStorage
                ) && let Some(origin) = &origin
                {
                    state
                        .epochs
                        .used
                        .entry(origin.clone())
                        .or_default()
                        .insert(policy.clone());
                }
                if validate {
                    self.trace.events.insert(
                        site.clone(),
                        Event {
                            call: call.site,
                            kind: call.kind,
                            origin,
                            policy,
                        },
                    );
                }
                if let Some(binding) = self.plan.as_mut().and_then(|p| p.sites.get_mut(&site)) {
                    if validate
                        && state
                            .epochs
                            .live
                            .values()
                            .any(|region| *region == binding.region)
                    {
                        return Err(format!(
                            "checked block collective in {}: storage use overlaps a live reservation at {site:?}",
                            self.root
                        ));
                    }
                    if state.epochs.dirty.contains(&binding.region) {
                        if binding.implicit {
                            binding.before = true;
                            state.pending.clear();
                            state.epochs.dirty.clear();
                        } else if validate {
                            return Err(format!(
                                "checked block collective in {}: explicit storage reuse requires a full-block barrier at {site:?}",
                                self.root
                            ));
                        }
                    }
                    state.epochs.dirty.insert(binding.region.clone());
                    if call.kind == MarkerKind::StorageReserve {
                        state
                            .epochs
                            .live
                            .insert(site.clone(), binding.region.clone());
                    }
                }
                if !call.kind.native() {
                    // Markers cannot mutate the descriptor; preserve its provenance.
                    write(destination, result, state, borrowed);
                    return Ok(());
                }
            }
            if validate && !call.kind.storage() {
                self.trace.events.insert(
                    site.clone(),
                    Event {
                        call: call.site,
                        kind: call.kind,
                        origin: None,
                        policy: Policy::default(),
                    },
                );
            }
            if validate {
                if !call.kind.checked()
                    || (self.resource_root && !call.kind.resource() && !call.kind.storage())
                {
                    return Err(self.error(
                        instance,
                        block,
                        if self.resource_root { "untracked and resource-checked library calls cannot be mixed in one checked root" } else { "raw and checked library calls cannot be mixed in one checked root" },
                    ));
                }
                if divergent {
                    return Err(self.error(
                        instance,
                        block,
                        "operation is reached through lane-varying control flow",
                    ));
                }
                let uniform = args
                    .get(if call.kind == MarkerKind::CheckedBlockStorage {
                        3
                    } else if call.kind.resource() {
                        2
                    } else {
                        1
                    })
                    .map(|arg| operand(arg, state))
                    .unwrap_or(Fact::Varying);
                let Fact::Fields(fields) = uniform else {
                    return Err(self.error(
                        instance,
                        block,
                        "uniform argument tuple has unsupported provenance",
                    ));
                };
                if fields.iter().any(|f| !f.uniform()) {
                    return Err(self.error(
                        instance,
                        block,
                        "memory carrier or scalar control is not proved block-uniform",
                    ));
                }
            }
            if call.kind.resource() {
                let witness = call_fact(operand(&args[1], state), state);
                if let Some(origin) = witness.resource_origin() {
                    let write = crate::library::resource_write(call);
                    let old = state.pending.get(origin).copied().unwrap_or(0);
                    if validate
                        && (!self.storage_root || self.plan.is_some())
                        && (old & 2 != 0 || (write && old != 0))
                    {
                        return Err(self.error(
                            instance,
                            block,
                            "conflicting collective resource access requires a full-block barrier",
                        ));
                    }
                    *state.pending.entry(origin.to_owned()).or_default() |=
                        if write { 2 } else { 1 };
                } else if validate {
                    return Err(self.error(
                        instance,
                        block,
                        "resource witness has no unambiguous kernel allocation origin",
                    ));
                }
            }
            if self
                .plan
                .as_ref()
                .and_then(|plan| plan.sites.get(&site))
                .is_some_and(|binding| binding.after)
            {
                state.pending.clear();
                state.epochs.dirty.clear();
            }
            // Native providers may mutate the payload, including a scalar
            // subsequently used as a control. Do not retain borrowed facts.
            invalidate_borrows(state, borrowed);
            write(destination, Fact::Fields(vec![]), state, borrowed);
            return Ok(());
        }
        let path = self.tcx.def_path_str(callee.def_id());
        let krate = self.tcx.crate_name(callee.def_id().krate);
        let is_sync = (krate.as_str() == "cuda_device" && path.ends_with("::sync_threads"))
            || (krate.as_str() == "cuda_intrinsics" && path.ends_with("::i0034"));
        if is_sync {
            state.pending.clear();
            state.epochs.dirty.clear();
            if validate && divergent {
                return Err(self.error(
                    instance,
                    block,
                    "block barrier is reached through lane-varying control flow",
                ));
            }
            write(destination, Fact::Fields(vec![]), state, borrowed);
            return Ok(());
        }
        if let Some(fact) = self.intrinsic(callee) {
            let name = path.rsplit("::").next().unwrap_or("");
            if validate
                && !fact.uniform()
                && !matches!(
                    name,
                    "threadIdx_x" | "threadIdx_y" | "threadIdx_z" | "i0001" | "i0005" | "i0009"
                )
            {
                return Err(self.error(
                    instance,
                    block,
                    "intrinsic is outside the supported checked block subset",
                ));
            }
            write(
                destination,
                if divergent { fact.taint() } else { fact },
                state,
                borrowed,
            );
            return Ok(());
        }
        if matches!(callee.def, rustc_middle::ty::InstanceKind::Intrinsic(_)) {
            let name = path.rsplit("::").next().unwrap_or("");
            if matches!(
                name,
                "integer_min"
                    | "integer_max"
                    | "unchecked_add"
                    | "unchecked_sub"
                    | "unchecked_mul"
                    | "arith_offset"
            ) {
                let fact = if args.iter().all(|arg| operand(arg, state).uniform()) {
                    if name == "arith_offset" {
                        operand(&args[0], state)
                    } else {
                        Fact::Uniform
                    }
                } else {
                    Fact::Varying
                };
                write(
                    destination,
                    if divergent { fact.taint() } else { fact },
                    state,
                    borrowed,
                );
                return Ok(());
            }
        }
        if target.is_none() && mir_importer::is_panic_entry_path(&path) {
            return Ok(());
        }
        if kind != EdgeKind::Direct && validate {
            return Err(self.error(
                instance,
                block,
                "indirect or destructor collective context is unsupported",
            ));
        }
        let facts = args
            .iter()
            .map(|arg| call_fact(operand(arg, state), state))
            .collect();
        self.path.push(block);
        let analyzed = self.analyze(
            callee,
            facts,
            validate,
            divergent,
            state.pending.clone(),
            state.epochs.clone(),
        );
        self.path.pop();
        let (mut result, pending, epochs) = analyzed?;
        state.pending = pending;
        state.epochs = epochs;
        if divergent {
            result = result.taint();
        }
        // Unknown pointers can alias a borrowed local even when their concrete
        // target was lost at a join. This subset has no interprocedural alias or
        // mutation summary: invalidate all borrowed facts after ordinary calls.
        invalidate_borrows(state, borrowed);
        write(destination, result, state, borrowed);
        Ok(())
    }
}
pub(crate) struct Validation<'tcx> {
    pub shapes: BTreeMap<String, [u32; 3]>,
    pub traces: BTreeMap<String, Trace<'tcx>>,
}
pub(crate) fn validate<'tcx>(
    tcx: TyCtxt<'tcx>,
    collection: &CollectionResult<'tcx>,
    mut plans: Option<&mut BTreeMap<String, Plan>>,
) -> Result<Validation<'tcx>, String> {
    rustc_internal::run(tcx, || {
        let mut shapes = BTreeMap::new();
        let mut traces = BTreeMap::new();
        for root in collection.functions.iter().filter(|f| f.is_kernel) {
            let reached = reachable_instances(collection, root.instance);
            if !collection.library_calls.iter().any(|c| c.kind.checked() && reached.contains(&c.site.caller)) { continue; }
            let stable: mir::mono::Instance = rustc_internal::stable(root.instance); let reach = device_mono_reachability(tcx, root.instance);
            let shape = mir_importer::kernel_exact_block(&stable, reach.block_count, &reach.successors)?
                .ok_or_else(|| format!("checked block collective root {} requires #[launch_contract(block = (...))] with an exact shape", root.export_name))?;
            let body = stable.body().ok_or("checked root has no MIR body")?;
            let resource_root = collection.library_calls.iter().any(|c| c.kind.resource() && reached.contains(&c.site.caller));
            let arguments = (1..=body.arg_locals().len()).map(|arg| {
                use rustc_middle::ty::TyKind;
                let ty = rustc_internal::internal(tcx, body.locals()[arg].ty);
                if resource_root && crate::library::resource_parameter(tcx, ty) {
                    Fact::Resource(format!("{}:{arg}", root.export_name))
                } else if matches!(ty.kind(), TyKind::Bool | TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_)) { Fact::Uniform }
                else { Fact::Kernel(format!("{}:{arg}", root.export_name)) }
            }).collect();
            let storage_root=collection.library_calls.iter().any(|c| c.kind.storage() && reached.contains(&c.site.caller));
            let trace=traces.entry(root.export_name.clone()).or_insert_with(Trace::default);
            let plan=plans.as_mut().and_then(|plans|plans.get_mut(&root.export_name));
            let mut analyzer = Analyzer { tcx, collection, root: &root.export_name, stack: Vec::new(), resource_root,storage_root,path:Vec::new(),trace,plan };
            analyzer.analyze(root.instance, arguments, true, false, Pending::new(),Epochs::default())?;
            shapes.insert(root.export_name.clone(), shape);
        }
        Ok(Validation{shapes,traces})
    }).map_err(|error| format!("cannot enter collective MIR analysis: {error:?}"))?
}
