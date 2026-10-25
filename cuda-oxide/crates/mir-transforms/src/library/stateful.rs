/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded per-context collective proof over prepared MIR values and local slots.
use super::storage::{
    Domain, Event, Plan, Policy, Region, Scope, Site, Trace, Validation, planned_storage,
};
use dialect_mir::{
    library::{
        self, LibraryAccess, LibraryCallKind, LibraryCallSite, LibraryCallbackEnvironment,
        LibraryNativeMetadata as Native, LibraryRoleAttr as Role, LibraryRootArgumentClass,
        LibrarySourceEffectKind as SourceKind, LibraryStorageArg, LibraryWriteKind,
    },
    ops::*,
    types::{MirPtrType, MirTupleType},
};
use pliron::linked_list::ContainsLinkedList;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::BranchOpInterface,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::op_cast,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    r#type::Typed,
    value::Value,
};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fact {
    Unset,
    // Exact 0/1 hardware predicate, independent of value uniformity.
    Member {
        width: u32,
        inverted: bool,
        boolean: bool,
    },
    Uniform(Domain),
    Kernel(String, Domain),
    Resource(String, Domain),
    Descriptor(Site, Policy),
    Reservation(Site),
    Local(usize, Vec<usize>),
    Referent(Box<Fact>),
    Fields(Vec<Fact>),
    // Fields are conditional on their presence in the active variant. The
    // tag's uniformity is independent of every payload field's uniformity.
    Enum {
        tag_uniform: Domain,
        fields: Vec<Fact>,
    },
    Varying,
}
impl Fact {
    fn precision(&self) -> Option<Domain> {
        match self {
            Self::Uniform(scope) | Self::Kernel(_, scope) | Self::Resource(_, scope) => {
                Some(*scope)
            }
            Self::Descriptor(..) | Self::Reservation(..) => Some(Domain::BLOCK),
            _ => None,
        }
    }
    fn uniform_at(&self, scope: Domain) -> bool {
        self.precision().is_some_and(|proved| proved.covers(scope))
    }
    fn merge(&self, other: &Self) -> Self {
        if self == other {
            return self.clone();
        }
        match (self, other) {
            (Self::Unset, value) | (value, Self::Unset) => value.clone(),
            (Self::Uniform(a), Self::Uniform(b))
            | (Self::Uniform(a), Self::Kernel(_, b))
            | (Self::Kernel(_, a), Self::Uniform(b)) => Self::Uniform((*a).meet(*b)),
            (Self::Kernel(a, x), Self::Kernel(b, y)) if a == b => {
                Self::Kernel(a.clone(), (*x).meet(*y))
            }
            (Self::Resource(a, x), Self::Resource(b, y)) if a == b => {
                Self::Resource(a.clone(), (*x).meet(*y))
            }
            (Self::Fields(a), Self::Fields(b)) if a.len() == b.len() => {
                Self::Fields(a.iter().zip(b).map(|(a, b)| a.merge(b)).collect())
            }
            (
                Self::Enum {
                    tag_uniform: a,
                    fields: af,
                },
                Self::Enum {
                    tag_uniform: b,
                    fields: bf,
                },
            ) => Self::Enum {
                tag_uniform: (*a).meet(*b),
                fields: (0..af.len().max(bf.len()))
                    .map(|i| self.variant_field(i).merge(&other.variant_field(i)))
                    .collect(),
            },
            (
                Self::Enum {
                    tag_uniform,
                    fields,
                },
                Self::Uniform(scope) | Self::Kernel(_, scope),
            )
            | (
                Self::Uniform(scope) | Self::Kernel(_, scope),
                Self::Enum {
                    tag_uniform,
                    fields,
                },
            ) => Self::Enum {
                tag_uniform: (*tag_uniform).meet(*scope),
                fields: fields
                    .iter()
                    .map(|f| f.merge(&Self::Uniform(*scope)))
                    .collect(),
            },
            _ => Self::Varying,
        }
    }
    fn taint(self, control: Domain) -> Self {
        match self {
            Self::Uniform(scope) => Self::Uniform(scope.meet(control)),
            Self::Kernel(origin, scope) => Self::Kernel(origin, scope.meet(control)),
            Self::Resource(origin, scope) => Self::Resource(origin, scope.meet(control)),
            Self::Fields(v) => Self::Fields(v.into_iter().map(|f| f.taint(control)).collect()),
            Self::Enum {
                tag_uniform,
                fields,
            } => Self::Enum {
                tag_uniform: tag_uniform.meet(control),
                fields: fields.into_iter().map(|f| f.taint(control)).collect(),
            },
            Self::Local(..) | Self::Member { .. } => self,
            Self::Referent(value) => Self::Referent(Box::new(value.taint(control))),
            _ if control == Domain::BLOCK => self,
            _ => Self::Varying,
        }
    }
    fn has_resource(&self) -> bool {
        match self {
            Self::Resource(..) | Self::Descriptor(..) | Self::Reservation(..) => true,
            Self::Referent(value) => value.has_resource(),
            Self::Fields(values) | Self::Enum { fields: values, .. } => {
                values.iter().any(Self::has_resource)
            }
            _ => false,
        }
    }
    fn has_local(&self) -> bool {
        match self {
            Self::Local(..) => true,
            Self::Referent(value) => value.has_local(),
            Self::Fields(values) | Self::Enum { fields: values, .. } => {
                values.iter().any(Self::has_local)
            }
            _ => false,
        }
    }
    fn invalidate_borrow(&mut self) {
        match self {
            Self::Resource(..) | Self::Descriptor(..) | Self::Reservation(..) => {}
            Self::Referent(value) if value.has_resource() => value.invalidate_borrow(),
            Self::Fields(values) if values.iter().any(Self::has_resource) => {
                values.iter_mut().for_each(Self::invalidate_borrow);
            }
            Self::Enum {
                tag_uniform,
                fields,
            } if fields.iter().any(Self::has_resource) => {
                *tag_uniform = Domain::LANE;
                fields.iter_mut().for_each(Self::invalidate_borrow);
            }
            _ => *self = Self::Varying,
        }
    }
    fn descriptor(&self) -> Option<(Site, Policy)> {
        match self {
            Self::Descriptor(origin, policy) => Some((origin.clone(), *policy)),
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
            Self::Resource(origin, _) => Some(origin),
            Self::Referent(value) => value.resource_origin(),
            _ => None,
        }
    }
    fn invalidate_referents(&mut self) {
        match self {
            Self::Referent(_) => self.invalidate_borrow(),
            Self::Fields(fields) | Self::Enum { fields, .. } => {
                fields.iter_mut().for_each(Self::invalidate_referents);
            }
            _ => {}
        }
    }
    fn field(&self, index: usize) -> Self {
        match self {
            Self::Fields(v) => v.get(index).cloned().unwrap_or(Self::Varying),
            Self::Kernel(origin, scope) | Self::Resource(origin, scope) => {
                Self::Kernel(format!("{origin}.{index}"), *scope)
            }
            Self::Reservation(_) => Self::Uniform(Domain::BLOCK),
            Self::Uniform(scope) => Self::Uniform(*scope),
            _ => Self::Varying,
        }
    }
    fn variant_field(&self, index: usize) -> Self {
        match self {
            // This conditional analysis fact is only valid at enum joins or
            // after a Downcast. It never reads an inactive variant's bytes.
            Self::Enum { fields, .. } => fields
                .get(index)
                .cloned()
                .unwrap_or(Self::Uniform(Domain::BLOCK)),
            Self::Uniform(_) | Self::Kernel(..) | Self::Resource(..) => self.field(index),
            _ => Self::Varying,
        }
    }
}
// The participant set is independent of value uniformity. In particular, a
// full Warp32 value does not prove that the suffix skipped a prefix call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Members {
    Full,
    Prefix(u32),
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Control {
    domain: Domain,
    members: Members,
}
impl Control {
    const FULL: Self = Self {
        domain: Domain::BLOCK,
        members: Members::Full,
    };
    const UNKNOWN: Self = Self {
        domain: Domain::LANE,
        members: Members::Unknown,
    };
    fn admits(self, domain: Domain) -> bool {
        self.domain.covers(domain)
            && match domain {
                Domain::Complete(_) => self.members == Members::Full,
                Domain::Prefix(w) => self.members == Members::Prefix(w),
            }
    }
    fn meet(self, other: Self) -> Self {
        let members = match (self.members, other.members) {
            (Members::Full, x) | (x, Members::Full) => x,
            (Members::Prefix(a), Members::Prefix(b)) if a == b => Members::Prefix(a),
            _ => Members::Unknown,
        };
        Self {
            domain: self.domain.meet(other.domain),
            members,
        }
    }
    fn restrict(self, restriction: Self, edge_dominates: bool) -> Self {
        // Prefix membership enables calls, unlike ordinary domain narrowing.
        // A reachable backedge or another predecessor cannot establish it.
        // An enclosing guard or caller may already prove this same prefix.
        if let Members::Prefix(width) = restriction.members
            && self.members != Members::Prefix(width)
            && !edge_dominates
        {
            self.meet(Self::UNKNOWN)
        } else {
            self.meet(restriction)
        }
    }
    fn narrow(self, precision: Domain) -> Self {
        Self {
            domain: self.domain.meet(precision),
            ..self
        }
    }
    fn member_branch(self, width: u32, member: bool) -> Self {
        if !member {
            return Self::UNKNOWN;
        }
        match self.members {
            Members::Full if self.domain.covers(Domain::Complete(Scope::Warp(32))) => Self {
                domain: Domain::Prefix(width),
                members: Members::Prefix(width),
            },
            Members::Prefix(w) if w == width => self,
            _ => Self::UNKNOWN,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Access {
    Read,
    Write,
}
// Retain each producer scope and access kind, together with the weakest
// completion proved on every incoming path. Only Block completion clears them.
type Pending = BTreeMap<(String, Scope, Access), Scope>;
fn merge_pending(into: &mut Pending, other: &Pending) -> bool {
    let mut changed = false;
    for (access, completion) in other {
        match into.entry(access.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(*completion);
                changed = true;
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let merged = (*entry.get()).min(*completion);
                changed |= merged != *entry.get();
                entry.insert(merged);
            }
        }
    }
    changed
}
#[derive(Clone, Default)]
struct Epochs {
    dirty: BTreeSet<Region>,
    live: BTreeMap<Site, Region>,
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
    values: FxHashMap<Value, Fact>,
    slots: Vec<Fact>,
    borrowed: BTreeSet<usize>,
    pending: Pending,
    epochs: Epochs,
}
impl State {
    fn fact(&self, value: Value) -> Fact {
        self.values.get(&value).cloned().unwrap_or(Fact::Unset)
    }
    fn merge(&mut self, other: &Self) -> bool {
        let mut changed = merge_pending(&mut self.pending, &other.pending);
        changed |= self.epochs.merge(&other.epochs);
        let n = self.borrowed.len();
        self.borrowed.extend(&other.borrowed);
        changed |= n != self.borrowed.len();
        for (key, value) in &other.values {
            let old = self.values.entry(*key).or_insert(Fact::Unset);
            let merged = old.merge(value);
            changed |= merged != *old;
            *old = merged;
        }
        for (old, value) in self.slots.iter_mut().zip(&other.slots) {
            let merged = old.merge(value);
            changed |= merged != *old;
            *old = merged;
        }
        changed
    }
    fn expose(&mut self, fact: &Fact) {
        match fact {
            Fact::Local(slot, _) => {
                self.borrowed.insert(*slot);
            }
            Fact::Fields(fields) | Fact::Enum { fields, .. } => {
                fields.iter().for_each(|f| self.expose(f))
            }
            Fact::Referent(f) => self.expose(f),
            _ => {}
        }
    }
}
fn local_read(state: &State, slot: usize, fields: &[usize]) -> Fact {
    fields
        .iter()
        .fold(state.slots[slot].clone(), |fact, field| fact.field(*field))
}
fn invalidate_borrows(state: &mut State) {
    for (slot, fact) in state.slots.iter_mut().enumerate() {
        if state.borrowed.contains(&slot) {
            fact.invalidate_borrow();
        } else {
            fact.invalidate_referents();
        }
    }
    for fact in state.values.values_mut() {
        fact.invalidate_referents();
    }
}
fn complete(state: &mut State, domain: Domain) {
    // Prefix synchronization cannot cover the sibling groups represented by
    // complete-partition global access records.
    if let Some(scope) = domain.complete_scope() {
        if scope == Scope::Block {
            state.pending.clear();
        } else {
            for ((_, producer, _), completion) in &mut state.pending {
                if *producer <= scope {
                    *completion = (*completion).max(scope);
                }
            }
        }
    }
    state
        .epochs
        .dirty
        .retain(|region| !region.completed_by(domain));
}
fn dereference(fact: Fact, state: &State) -> Fact {
    match fact {
        Fact::Local(slot, fields) => local_read(state, slot, &fields),
        Fact::Referent(f) => *f,
        _ => Fact::Varying,
    }
}
fn store(address: Fact, value: Fact, state: &mut State) {
    if let Fact::Local(slot, fields) = address {
        write_fields(&mut state.slots[slot], &fields, value);
    } else {
        invalidate_borrows(state);
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
            Fact::Enum {
                tag_uniform,
                fields,
            } => Fact::Enum {
                tag_uniform,
                fields: fields
                    .into_iter()
                    .map(|value| resolve(value, state, visiting))
                    .collect(),
            },
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
        Fact::Enum {
            tag_uniform,
            fields,
        } => Fact::Enum {
            tag_uniform,
            fields: fields.into_iter().map(return_fact).collect(),
        },
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
fn reachable_without_edge(edges: &[Vec<usize>], source: usize, target: usize) -> BTreeSet<usize> {
    let mut result = BTreeSet::new();
    let mut pending = vec![0];
    while let Some(block) = pending.pop() {
        if result.insert(block) {
            pending.extend(
                edges[block]
                    .iter()
                    .copied()
                    .filter(|next| block != source || *next != target),
            );
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

// Only immutable structure is cached. Context-dependent facts are recomputed
// for each call on this same prepared module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransferKind {
    Library,
    Source,
    WriteCheck,
    Call,
    Store,
    Bulk,
    SetDiscriminant,
    Unreachable,
    NoResult,
    Alloca,
    Load,
    Ref,
    FieldAddr,
    ExtractField,
    InsertField,
    ConstructEnum,
    GetDiscriminant,
    EnumPayload,
    Aggregate,
    Constant,
    Undef,
    Cast,
    PointerOffset,
    ArrayElementAddr,
    ExtractArrayElement,
    Eq,
    Ne,
    Not,
    CheckedArithmetic,
    Scalar,
    Unknown,
}

impl TransferKind {
    fn classify(ctx: &Context, op: Ptr<Operation>) -> Self {
        if Operation::get_op::<MirLibraryOp>(op, ctx).is_some() {
            Self::Library
        } else if Operation::get_op::<MirLibrarySourceOp>(op, ctx).is_some() {
            Self::Source
        } else if Operation::get_op::<MirLibraryWriteCheckOp>(op, ctx).is_some() {
            Self::WriteCheck
        } else if Operation::get_op::<MirCallOp>(op, ctx).is_some() {
            Self::Call
        } else if Operation::get_op::<MirStoreOp>(op, ctx).is_some() {
            Self::Store
        } else if Operation::get_op::<MirMemcpyOp>(op, ctx).is_some()
            || Operation::get_op::<MirMemmoveOp>(op, ctx).is_some()
        {
            Self::Bulk
        } else if Operation::get_op::<MirSetDiscriminantOp>(op, ctx).is_some() {
            Self::SetDiscriminant
        } else if Operation::get_op::<MirUnreachableOp>(op, ctx).is_some() {
            Self::Unreachable
        } else if op.deref(ctx).get_num_results() == 0 {
            Self::NoResult
        } else if Operation::get_op::<MirAllocaOp>(op, ctx).is_some() {
            Self::Alloca
        } else if Operation::get_op::<MirLoadOp>(op, ctx).is_some() {
            Self::Load
        } else if Operation::get_op::<MirRefOp>(op, ctx).is_some() {
            Self::Ref
        } else if Operation::get_op::<MirFieldAddrOp>(op, ctx).is_some() {
            Self::FieldAddr
        } else if Operation::get_op::<MirExtractFieldOp>(op, ctx).is_some() {
            Self::ExtractField
        } else if Operation::get_op::<MirInsertFieldOp>(op, ctx).is_some() {
            Self::InsertField
        } else if Operation::get_op::<MirConstructEnumOp>(op, ctx).is_some() {
            Self::ConstructEnum
        } else if Operation::get_op::<MirGetDiscriminantOp>(op, ctx).is_some() {
            Self::GetDiscriminant
        } else if Operation::get_op::<MirEnumPayloadOp>(op, ctx).is_some() {
            Self::EnumPayload
        } else {
            match Operation::get_opid(op, ctx).to_string().as_str() {
                "mir.construct_struct"
                | "mir.construct_tuple"
                | "mir.construct_array"
                | "mir.construct_slice"
                | "mir.construct_disjoint_slice" => Self::Aggregate,
                "mir.constant" | "mir.float_constant" => Self::Constant,
                "mir.undef" => Self::Undef,
                "mir.cast" => Self::Cast,
                "mir.ptr_offset" => Self::PointerOffset,
                "mir.array_element_addr" => Self::ArrayElementAddr,
                "mir.extract_array_element" => Self::ExtractArrayElement,
                "mir.eq" => Self::Eq,
                "mir.ne" => Self::Ne,
                "mir.not" => Self::Not,
                "mir.checked_add" | "mir.checked_sub" | "mir.checked_mul" => {
                    Self::CheckedArithmetic
                }
                "mir.add" | "mir.sub" | "mir.mul" | "mir.div" | "mir.rem" | "mir.neg"
                | "mir.shr" | "mir.shl" | "mir.bitand" | "mir.bitor" | "mir.bitxor" | "mir.lt"
                | "mir.le" | "mir.gt" | "mir.ge" | "mir.cmp" => Self::Scalar,
                _ => Self::Unknown,
            }
        }
    }
}

struct PreparedOperation {
    op: Ptr<Operation>,
    operands: Vec<Value>,
    results: Vec<Value>,
    kind: TransferKind,
}

impl PreparedOperation {
    fn new(ctx: &Context, op: Ptr<Operation>) -> Self {
        Self {
            op,
            operands: op.deref(ctx).operands().collect(),
            results: op.deref(ctx).results().collect(),
            kind: TransferKind::classify(ctx, op),
        }
    }
}

struct FunctionGraph {
    op: Ptr<Operation>,
    blocks: Vec<Ptr<BasicBlock>>,
    operations: Vec<Vec<PreparedOperation>>,
    edges: Vec<Vec<usize>>,
    successor_ordinals: Vec<Vec<usize>>,
    joins: Vec<Option<usize>>,
    bypasses: HashMap<(usize, usize), BTreeSet<usize>>,
    slots: HashMap<Ptr<Operation>, usize>,
    exposed_slots: HashMap<Ptr<Operation>, BTreeSet<usize>>,
    value_count: usize,
}
struct PreparedGraph {
    functions: BTreeMap<String, FunctionGraph>,
}

// Ordinary importer slots are not source borrows. Record each exposing use so
// it activates before joins or opaque operations lose Local, but not before
// earlier operations that cannot yet observe the address.
fn slot_exposures(ctx: &Context, address: Value) -> HashSet<Ptr<Operation>> {
    let mut pending = vec![address];
    let mut visited = HashSet::new();
    let mut exposures = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        for use_ in value.uses(ctx) {
            let op = use_.user_op();
            let index = use_.find_index(ctx);
            match TransferKind::classify(ctx, op) {
                TransferKind::Load | TransferKind::Store | TransferKind::SetDiscriminant
                    if index == 0 => {}
                TransferKind::WriteCheck => {}
                TransferKind::FieldAddr
                | TransferKind::PointerOffset
                | TransferKind::ArrayElementAddr
                | TransferKind::Cast
                    if index == 0 =>
                {
                    pending.extend(op.deref(ctx).results());
                }
                _ => {
                    exposures.insert(op);
                }
            }
        }
    }
    exposures
}

impl Analysis for PreparedGraph {
    fn name(&self) -> &str {
        "prepared-library-cfg"
    }
    fn compute(
        module: Ptr<Operation>,
        ctx: &Context,
        _: &mut AnalysisManager,
    ) -> pliron::result::Result<Self> {
        let mut functions = BTreeMap::new();
        for (name, op) in super::admission::functions(ctx, module) {
            if op.deref(ctx).num_regions() == 0 {
                continue;
            }
            let blocks: Vec<_> = op.deref(ctx).get_region(0).deref(ctx).iter(ctx).collect();
            if blocks.is_empty() {
                continue;
            }
            let indices: HashMap<_, _> = blocks.iter().enumerate().map(|(i, b)| (*b, i)).collect();
            let operations: Vec<Vec<_>> = blocks
                .iter()
                .map(|b| {
                    b.deref(ctx)
                        .iter(ctx)
                        .map(|op| PreparedOperation::new(ctx, op))
                        .collect()
                })
                .collect();
            let mut edges = Vec::new();
            let mut successor_ordinals = Vec::new();
            let mut slots = HashMap::new();
            let mut value_count = 0;
            for (block, ops) in blocks.iter().zip(&operations) {
                value_count += block.deref(ctx).get_num_arguments();
                for operation in ops {
                    if matches!(
                        operation.kind,
                        TransferKind::Library | TransferKind::Source | TransferKind::WriteCheck
                    ) {
                        operation.op.deref(ctx).verify(ctx)?;
                    }
                    value_count += operation.results.len();
                    if operation.kind == TransferKind::Alloca {
                        let n = slots.len();
                        slots.insert(operation.op, n);
                    }
                }
                let Some(term) = ops.last() else {
                    return pliron::verify_err!(
                        op.deref(ctx).loc(),
                        "library analysis requires terminated MIR blocks"
                    );
                };
                let term = term.op;
                let impossible = library::library_impossible_successor(ctx, term);
                let mut targets = Vec::new();
                let mut ordinals = Vec::new();
                for (ordinal, target) in term.deref(ctx).successors().enumerate() {
                    if impossible == Some(ordinal as u32) {
                        continue;
                    }
                    let Some(&index) = indices.get(&target) else {
                        return pliron::verify_err!(
                            term.deref(ctx).loc(),
                            "library CFG successor escapes its function"
                        );
                    };
                    targets.push(index);
                    ordinals.push(ordinal);
                }
                edges.push(targets);
                successor_ordinals.push(ordinals);
            }
            let mut exposed_slots: HashMap<_, BTreeSet<_>> = HashMap::new();
            for (&op, &slot) in &slots {
                for exposure in slot_exposures(ctx, op.deref(ctx).get_result(0)) {
                    exposed_slots.entry(exposure).or_default().insert(slot);
                }
            }
            let joins = joins(&edges);
            // Fixed prepared CFG: never derive membership from pruned successors.
            let bypasses = edges
                .iter()
                .enumerate()
                .flat_map(|(source, targets)| targets.iter().map(move |&target| (source, target)))
                .map(|edge| (edge, reachable_without_edge(&edges, edge.0, edge.1)))
                .collect();
            functions.insert(
                name,
                FunctionGraph {
                    op,
                    blocks,
                    operations,
                    edges,
                    successor_ordinals,
                    joins,
                    bypasses,
                    slots,
                    exposed_slots,
                    value_count,
                },
            );
        }
        Ok(Self { functions })
    }
}

fn branch_controls(
    fact: Fact,
    targets: &[(usize, bool)],
    control: Control,
) -> Vec<(usize, Control)> {
    let unique: BTreeSet<_> = targets.iter().map(|(target, _)| *target).collect();
    if let Fact::Member {
        width, inverted, ..
    } = fact
    {
        let known = control.members == Members::Prefix(width);
        unique
            .into_iter()
            .filter_map(|target| {
                let values: Vec<_> = targets
                    .iter()
                    .filter(|(t, v)| *t == target && (!known || *v != inverted))
                    .map(|(_, v)| *v)
                    .collect();
                match values.as_slice() {
                    [value] => Some((target, control.member_branch(width, *value != inverted))),
                    [_, _] => Some((target, control)),
                    _ => None,
                }
            })
            .collect()
    } else {
        let restricted = if unique.len() > 1 {
            control.narrow(fact.precision().unwrap_or(Domain::LANE))
        } else {
            control
        };
        unique
            .into_iter()
            .map(|target| (target, restricted))
            .collect()
    }
}
fn resource(role: Role) -> Option<LibraryAccess> {
    match role {
        Role::Native(
            Native::CheckedBlockResource { access, .. }
            | Native::CheckedBlockStorage { access, .. }
            | Native::CheckedWarp { access, .. },
        ) => Some(access),
        _ => None,
    }
}
fn checked(role: Role) -> bool {
    matches!(
        role,
        Role::Native(
            Native::CheckedBlock { .. }
                | Native::CheckedBlockResource { .. }
                | Native::CheckedBlockStorage { .. }
                | Native::CheckedWarp { .. }
                | Native::CheckedBlockPayload { .. }
                | Native::CheckedBlockPayloadCallback { .. }
                | Native::CheckedWarpPayload { .. }
        )
    )
}
fn controls_index(native: Native) -> Option<usize> {
    match native {
        Native::Raw { .. } | Native::Callback { .. } => None,
        Native::CheckedBlock { .. } | Native::CheckedWarpPayload { .. } => Some(0),
        Native::CheckedBlockResource { .. }
        | Native::CheckedWarp { .. }
        | Native::CheckedBlockPayload { .. }
        | Native::CheckedBlockPayloadCallback { .. } => Some(1),
        Native::CheckedBlockStorage { .. } => Some(2),
    }
}
fn scalar_precision(mut facts: impl Iterator<Item = Fact>) -> Fact {
    facts
        .try_fold(Domain::BLOCK, |domain, fact| {
            fact.precision().map(|p| domain.meet(p))
        })
        .map_or(Fact::Varying, Fact::Uniform)
}
fn unit(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirTupleType>()
        .is_some_and(|t| t.get_types().is_empty())
}

fn integer(ctx: &Context, value: Value) -> Option<u64> {
    let op = value.defining_op()?;
    Operation::get_op::<MirConstantOp>(op, ctx)?
        .get_attr_value(ctx)
        .map(|a| a.value().to_u64())
}
struct Analyzer<'a> {
    ctx: &'a Context,
    graph: &'a PreparedGraph,
    symbols: &'a BTreeMap<String, String>,
    callback_environments: &'a BTreeMap<LibraryCallSite, LibraryCallbackEnvironment>,
    root: &'a str,
    resource_root: bool,
    storage_root: bool,
    stack: Vec<String>,
    path: Site,
    trace: &'a mut Trace,
    plan: Option<&'a mut Plan>,
}
impl Analyzer<'_> {
    fn error(&self, function: &str, block: usize, reason: &str) -> String {
        let diagnostic = self
            .symbols
            .get(function)
            .and_then(|s| self.graph.functions.get(s))
            .and_then(|f| {
                MirFuncOp::wrap(self.ctx, f.op)
                    .unwrap()
                    .library_function_source(self.ctx)
            })
            .map_or_else(|| function.to_owned(), |s| s.diagnostic);
        format!(
            "checked block collective in {}: {reason}; {diagnostic} MIR block {block}",
            self.root
        )
    }
    fn source_block(&self, graph: &FunctionGraph, block: usize) -> usize {
        graph.operations[block]
            .last()
            .and_then(|term| library::library_source_site(self.ctx, term.op))
            .map_or(block, |site| site.block)
    }
    #[allow(clippy::too_many_arguments)]
    fn analyze(
        &mut self,
        function: &str,
        arguments: Vec<Fact>,
        validate: bool,
        entry_control: Control,
        pending: Pending,
        epochs: Epochs,
    ) -> Result<(Fact, Pending, Epochs), String> {
        if self.stack.iter().any(|f| f == function) {
            return if validate {
                Err(self.error(function, 0, "recursive helper participation is unsupported"))
            } else {
                Ok((Fact::Varying, pending, epochs))
            };
        }
        let Some(graph) = self
            .symbols
            .get(function)
            .and_then(|s| self.graph.functions.get(s))
        else {
            return if validate {
                Err(self.error(
                    function,
                    0,
                    "call without MIR cannot prove collective participation",
                ))
            } else {
                Ok((Fact::Varying, pending, epochs))
            };
        };
        if MirFuncOp::wrap(self.ctx, graph.op)
            .unwrap()
            .library_unchecked_indexing(self.ctx)
        {
            return Err(self.error(
                function,
                0,
                "checked block collectives do not support unchecked indexing; retain runtime bounds checks",
            ));
        }
        if validate {
            self.trace
                .contexts
                .insert(self.path.clone(), function.to_owned());
        }
        let mut entry = State {
            values: FxHashMap::default(),
            slots: vec![Fact::Unset; graph.slots.len()],
            borrowed: BTreeSet::new(),
            pending,
            epochs,
        };
        for (index, fact) in arguments.into_iter().enumerate() {
            entry
                .values
                .insert(graph.blocks[0].deref(self.ctx).get_argument(index), fact);
        }
        let mut incoming = vec![None; graph.blocks.len()];
        incoming[0] = Some(entry);
        let mut controls = vec![entry_control; graph.blocks.len()];
        let live = reachable(&graph.edges, 0, None);
        let limit = graph
            .blocks
            .len()
            .saturating_mul(
                graph.value_count
                    + graph.slots.len()
                    + self.plan.as_ref().map_or(0, |p| p.sites.len() * 4)
                    + 4,
            )
            .saturating_mul(Scope::HEIGHT * 3)
            .max(8);
        self.stack.push(function.to_owned());
        let mut settled = false;
        for _ in 0..limit {
            let mut changed = false;
            for &block in &live {
                let Some(mut state) = incoming[block].clone() else {
                    continue;
                };
                self.block(function, graph, block, &mut state, controls[block], false)?;
                let control = controls[block];
                let term = graph.operations[block].last().unwrap().op;
                let successors = if Operation::get_op::<MirCondBranchOp>(term, self.ctx).is_some() {
                    let targets: Vec<_> = graph.edges[block]
                        .iter()
                        .zip(&graph.successor_ordinals[block])
                        .map(|(&t, &i)| (t, i == 0))
                        .collect();
                    let branches = branch_controls(
                        state.fact(term.deref(self.ctx).get_operand(0)),
                        &targets,
                        control,
                    );
                    for &(target, restriction) in &branches {
                        for member in reachable(&graph.edges, target, graph.joins[block]) {
                            let dominates = !graph.bypasses[&(block, target)].contains(&member);
                            let restricted = controls[member].restrict(restriction, dominates);
                            changed |= restricted != controls[member];
                            controls[member] = restricted;
                        }
                    }
                    branches.into_iter().map(|(t, _)| t).collect::<Vec<_>>()
                } else {
                    graph.edges[block].clone()
                };
                for (edge, &target) in graph.edges[block].iter().enumerate() {
                    if !successors.contains(&target) {
                        continue;
                    }
                    let mut outgoing = Cow::Borrowed(&state);
                    let obj = Operation::get_op_dyn(term, self.ctx);
                    if let Some(branch) = op_cast::<dyn BranchOpInterface>(obj.as_ref()) {
                        for (arg, index) in branch
                            .successor_operand_range(
                                self.ctx,
                                graph.successor_ordinals[block][edge],
                            )
                            .enumerate()
                        {
                            let value = term.deref(self.ctx).get_operand(index);
                            let mut selection = controls[block].domain;
                            // Reconvergence does not erase a value selection. A
                            // conditional branch may pass different values to
                            // the same destination block argument on its edges.
                            if Operation::get_op::<MirCondBranchOp>(term, self.ctx).is_some()
                                && graph.edges[block].iter().enumerate().any(
                                    |(other, &destination)| {
                                        destination == target && {
                                            let other_index = branch
                                                .successor_operand_range(
                                                    self.ctx,
                                                    graph.successor_ordinals[block][other],
                                                )
                                                .nth(arg)
                                                .unwrap();
                                            term.deref(self.ctx).get_operand(other_index) != value
                                        }
                                    },
                                )
                            {
                                selection = selection.meet(
                                    state
                                        .fact(term.deref(self.ctx).get_operand(0))
                                        .precision()
                                        .unwrap_or(Domain::LANE),
                                );
                            }
                            let fact = state.fact(value).taint(selection);
                            outgoing.to_mut().values.insert(
                                graph.blocks[target].deref(self.ctx).get_argument(arg),
                                fact,
                            );
                        }
                    }
                    match &mut incoming[target] {
                        Some(previous) => changed |= previous.merge(&outgoing),
                        slot @ None => {
                            *slot = Some(outgoing.into_owned());
                            changed = true;
                        }
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
                function,
                0,
                "value analysis did not converge in its bounded subset",
            ));
        }
        let mut returned = Fact::Unset;
        let mut returned_pending = Pending::new();
        let mut returned_epochs = Epochs::default();
        for block in live {
            let term = graph.operations[block].last().unwrap().op;
            let is_return = Operation::get_op::<MirReturnOp>(term, self.ctx).is_some();
            // Non-return effects already flow through the settled incoming states.
            // Revisit all blocks only when validating their source obligations.
            if !validate && !is_return {
                continue;
            }
            let Some(mut state) = incoming[block].clone() else {
                continue;
            };
            self.block(
                function,
                graph,
                block,
                &mut state,
                controls[block],
                validate,
            )?;
            if is_return {
                let value = if term.deref(self.ctx).get_num_operands() == 0 {
                    Fact::Fields(vec![])
                } else {
                    state.fact(term.deref(self.ctx).get_operand(0))
                };
                returned = returned.merge(&return_fact(value.taint(controls[block].domain)));
                merge_pending(&mut returned_pending, &state.pending);
                returned_epochs.merge(&state.epochs);
            }
        }
        self.stack.pop();
        Ok((return_fact(returned), returned_pending, returned_epochs))
    }
    #[allow(clippy::too_many_arguments)]
    fn library(
        &mut self,
        function: &str,
        op: Ptr<Operation>,
        state: &mut State,
        control: Control,
        validate: bool,
    ) -> Result<Option<Fact>, String> {
        let library = MirLibraryOp::new(op);
        let role = *library.get_attr_role(self.ctx).unwrap();
        if let Role::WarpGroupIndex { width } = role {
            return Ok(Some(
                Fact::Uniform(Domain::Complete(Scope::Warp(width))).taint(control.domain),
            ));
        }
        let call = (*library.get_attr_site(self.ctx).unwrap()).clone();
        if let Role::WarpPrefixMember { width } = role {
            return Ok(Some(Fact::Member {
                width,
                inverted: false,
                boolean: false,
            }));
        }
        if let Role::SyncWarp { width } = role {
            let scope = Domain::Complete(Scope::Warp(width));
            if validate && !control.admits(scope) {
                return Err(self.error(
                    function,
                    call.block,
                    "WarpTile barrier is reached through control narrower than its width",
                ));
            }
            // Completion never releases a reservation or a wider storage epoch.
            complete(state, scope);
            return Ok(None);
        }
        let marker = String::from((*library.get_attr_library_callee(self.ctx).unwrap()).clone());
        let mut site = self.path.clone();
        site.push(call.block);
        let operands: Vec<_> = op.deref(self.ctx).operands().collect();
        let facts: Vec<_> = operands.iter().map(|v| state.fact(*v)).collect();
        let resolved: Vec<_> = facts.iter().cloned().map(|f| call_fact(f, state)).collect();
        let scope = if let Role::Native(
            Native::CheckedWarp { width, .. } | Native::CheckedWarpPayload { width, .. },
        ) = role
        {
            if matches!(role, Role::Native(Native::CheckedWarpPayload { .. }))
                && !width.is_power_of_two()
            {
                Domain::Prefix(width)
            } else {
                Domain::Complete(Scope::Warp(width))
            }
        } else {
            Domain::BLOCK
        };
        let mut origin = None;
        let mut policy = Policy::default();
        let mut result = None;
        let mut storage_use = false;
        match role {
            Role::NativeOperandBoundary
            | Role::WarpGroupIndex { .. }
            | Role::WarpPrefixMember { .. }
            | Role::SyncWarp { .. } => {
                return Ok(None);
            }
            Role::StorageNew(new) => {
                policy = new;
                result = Some(Fact::Descriptor(site.clone(), policy));
            }
            Role::StorageReconfigure { new, .. } => {
                if let Some((id, _)) = resolved[0].descriptor() {
                    if validate
                        && state
                            .epochs
                            .used
                            .get(&id)
                            .is_some_and(|used| used.iter().any(|old| *old != new))
                    {
                        return Err(self.error(
                            function,
                            call.block,
                            "storage policy cannot change after use",
                        ));
                    }
                    origin = Some(id.clone());
                    policy = new;
                    result = Some(Fact::Descriptor(id, policy));
                } else if validate {
                    return Err(self.error(
                        function,
                        call.block,
                        "storage reconfiguration has ambiguous descriptor origin",
                    ));
                } else {
                    result = Some(Fact::Varying);
                }
            }
            Role::StorageRelease => {
                if let Some(reservation) = resolved[0].reservation() {
                    if let Some(region) = state.epochs.live.remove(&reservation) {
                        state.epochs.dirty.insert(region);
                    } else if validate && self.plan.is_some() {
                        return Err(self.error(
                            function,
                            call.block,
                            "storage release has no live reservation",
                        ));
                    }
                } else if validate {
                    return Err(self.error(
                        function,
                        call.block,
                        "storage release has ambiguous reservation origin",
                    ));
                }
            }
            Role::StorageReserve { .. } => {
                storage_use = true;
                result = Some(Fact::Reservation(site.clone()));
            }
            Role::Native(
                Native::CheckedBlockStorage { .. }
                | Native::CheckedBlockPayload { .. }
                | Native::CheckedBlockPayloadCallback { .. },
            ) => {
                storage_use = true;
            }
            Role::Native(_) => {}
        }
        let storage_operation = matches!(
            role,
            Role::StorageNew(_)
                | Role::StorageReconfigure { .. }
                | Role::StorageReserve { .. }
                | Role::StorageRelease
        ) || storage_use;
        if storage_operation && validate && !control.admits(Domain::BLOCK) {
            return Err(self.error(
                function,
                call.block,
                "storage operation is reached through lane-varying control flow",
            ));
        }
        if storage_use {
            let (index, implicit) = match role {
                Role::Native(Native::CheckedBlockStorage { storage, .. }) => {
                    (1, matches!(storage, LibraryStorageArg::Implicit))
                }
                Role::Native(
                    Native::CheckedBlockPayload { storage, .. }
                    | Native::CheckedBlockPayloadCallback { storage, .. },
                ) => (0, matches!(storage, LibraryStorageArg::Implicit)),
                _ => (0, false),
            };
            if let Some((id, p)) = resolved[index].descriptor() {
                origin = Some(id);
                policy = p;
            } else if !implicit && validate {
                return Err(self.error(
                    function,
                    call.block,
                    "storage use has no unambiguous descriptor origin",
                ));
            }
            if let Some(id) = &origin {
                state
                    .epochs
                    .used
                    .entry(id.clone())
                    .or_default()
                    .insert(policy);
            }
        }
        if validate {
            self.trace.events.insert(
                site.clone(),
                Event {
                    call: call.clone(),
                    role,
                    marker,
                    scope,
                    origin,
                    policy,
                },
            );
        }
        if let Some(binding) = self.plan.as_ref().and_then(|p| p.sites.get(&site)).cloned() {
            if validate && state.epochs.live.values().any(|r| *r == binding.region) {
                return Err(format!(
                    "checked block collective in {}: storage use overlaps a live reservation at {site:?}",
                    self.root
                ));
            }
            if state.epochs.dirty.contains(&binding.region) {
                if binding.implicit {
                    self.plan
                        .as_mut()
                        .unwrap()
                        .sites
                        .get_mut(&site)
                        .unwrap()
                        .before = true;
                    complete(state, scope);
                } else if validate {
                    return Err(format!(
                        "checked block collective in {}: explicit storage reuse requires a full-block barrier at {site:?}",
                        self.root
                    ));
                }
            }
            state.epochs.dirty.insert(binding.region.clone());
            if matches!(role, Role::StorageReserve { .. }) {
                state.epochs.live.insert(site.clone(), binding.region);
            }
        }
        let Role::Native(native) = role else {
            return Ok(result);
        };
        let payload = matches!(
            native,
            Native::CheckedBlockPayload { .. }
                | Native::CheckedBlockPayloadCallback { .. }
                | Native::CheckedWarpPayload { .. }
        );
        if validate {
            if !checked(role)
                || ((self.resource_root || self.storage_root)
                    && resource(role).is_none()
                    && !storage_operation
                    && !payload)
            {
                return Err(self.error(function,call.block,if self.resource_root {"untracked and resource-checked library calls cannot be mixed in one checked root"}
                    else if self.storage_root {"untracked and planned-storage library calls cannot be mixed in one checked root"}
                    else {"raw and checked library calls cannot be mixed in one checked root"}));
            }
            if !control.admits(scope) {
                return Err(self.error(
                    function,
                    call.block,
                    "operation is reached through lane-varying control flow",
                ));
            }
            if matches!(native, Native::CheckedBlockPayloadCallback { .. }) {
                match self.callback_environments.get(&call) {
                    Some(
                        LibraryCallbackEnvironment::FunctionItem
                        | LibraryCallbackEnvironment::Closure {
                            capture_count: 0,
                            single_capture: None,
                        },
                    ) => {}
                    Some(LibraryCallbackEnvironment::Closure {
                        capture_count: 1,
                        single_capture: Some(library::LibraryScalar::Signed(32)),
                    }) => {
                        let environment_type = operands[3].get_type(self.ctx);
                        let environment_type = environment_type.deref(self.ctx);
                        let field =
                            environment_type
                                .downcast_ref::<MirPtrType>()
                                .and_then(|pointer| {
                                    pointer
                                        .pointee
                                        .deref(self.ctx)
                                        .downcast_ref::<dialect_mir::types::MirStructType>()
                                        .and_then(|structure| {
                                            (structure.field_count() == 1)
                                                .then(|| structure.get_field_type(0))
                                                .flatten()
                                        })
                                });
                        let i32_field = field.is_some_and(|field| {
                            field
                                .deref(self.ctx)
                                .downcast_ref::<IntegerType>()
                                .is_some_and(|ty| {
                                    ty.width() == 32 && ty.signedness() == Signedness::Signed
                                })
                        });
                        let contents = dereference(resolved[3].clone(), state);
                        if !i32_field
                            || !matches!(contents, Fact::Fields(ref fields)
                            if fields.len() == 1 && fields[0].uniform_at(Domain::BLOCK))
                        {
                            return Err(self.error(function, call.block,
                                "callback i32 capture is not proved block-uniform through its shared environment"));
                        }
                    }
                    _ => {
                        return Err(self.error(
                            function,
                            call.block,
                            "checked callback nominal environment is missing or unsupported",
                        ));
                    }
                }
            }
            if let Some(index) = controls_index(native)
                && !(payload && unit(self.ctx, operands[index]))
            {
                let Fact::Fields(fields) = &facts[index] else {
                    return Err(self.error(
                        function,
                        call.block,
                        "uniform argument tuple has unsupported provenance",
                    ));
                };
                if fields.iter().any(|f| !f.uniform_at(scope)) {
                    return Err(self.error(
                        function,
                        call.block,
                        if scope == Domain::BLOCK {
                            "memory carrier or scalar control is not proved block-uniform"
                        } else {
                            "memory carrier or scalar control is not proved Warp-uniform"
                        },
                    ));
                }
            }
        }
        if let Some(access) = resource(role) {
            let witness = call_fact(facts[0].clone(), state);
            if let Some(origin) = witness.resource_origin() {
                let scope = scope.complete_scope().ok_or_else(|| {
                    self.error(
                        function,
                        call.block,
                        "resource calls require a complete power-of-two Warp domain",
                    )
                })?;
                let write = access == LibraryAccess::Write;
                let kind = if write { Access::Write } else { Access::Read };
                let conflict =
                    state
                        .pending
                        .iter()
                        .any(|((old_origin, producer, old_kind), completion)| {
                            old_origin == origin
                                && (write || *old_kind == Access::Write)
                                && *completion < (*producer).max(scope)
                        });
                if validate && (!self.storage_root || self.plan.is_some()) && conflict {
                    return Err(self.error(
                        function,
                        call.block,
                        "conflicting collective resource access requires a barrier covering both accesses",
                    ));
                }
                state
                    .pending
                    .insert((origin.to_owned(), scope, kind), Scope::LANE);
            } else if validate {
                return Err(self.error(
                    function,
                    call.block,
                    "resource witness has no unambiguous kernel allocation origin",
                ));
            }
        }
        if self
            .plan
            .as_ref()
            .and_then(|p| p.sites.get(&site))
            .is_some_and(|b| b.after)
        {
            complete(state, scope);
        }
        facts.iter().for_each(|fact| state.expose(fact));
        invalidate_borrows(state);
        Ok(None)
    }
    #[allow(clippy::too_many_arguments)]
    fn block(
        &mut self,
        function: &str,
        graph: &FunctionGraph,
        block: usize,
        state: &mut State,
        control: Control,
        validate: bool,
    ) -> Result<(), String> {
        let mut aborted = false;
        let mut facts = Vec::new();
        for prepared in &graph.operations[block] {
            let op = prepared.op;
            if let Some(slots) = graph.exposed_slots.get(&op) {
                state.borrowed.extend(slots);
            }
            let operands = &prepared.operands;
            let kind = prepared.kind;
            if kind == TransferKind::Library {
                if let Some(fact) = self.library(function, op, state, control, validate)? {
                    for &result in &prepared.results {
                        state
                            .values
                            .insert(result, fact.clone().taint(control.domain));
                    }
                }
                continue;
            }
            facts.clear();
            facts.extend(operands.iter().map(|&v| state.fact(v)));
            if kind == TransferKind::Source {
                let source = MirLibrarySourceOp::new(op);
                let effect = source.get_attr_library_source_effect(self.ctx).unwrap();
                let fact = match effect.kind {
                    SourceKind::BlockSync => {
                        if validate && !control.admits(Domain::BLOCK) {
                            return Err(self.error(
                                function,
                                effect.site.block,
                                "block barrier is reached through lane-varying control flow",
                            ));
                        }
                        complete(state, Domain::BLOCK);
                        Fact::Fields(vec![])
                    }
                    SourceKind::BlockQuery => Fact::Uniform(Domain::BLOCK),
                    SourceKind::LaneQuery => Fact::Uniform(Domain::LANE),
                    SourceKind::UnitHint => Fact::Fields(vec![]),
                    SourceKind::ScalarIntrinsic | SourceKind::PointerOffset => {
                        facts.first().cloned().unwrap_or(Fact::Varying)
                    }
                    SourceKind::DynamicShared => Fact::Varying,
                    SourceKind::Abort => {
                        aborted = true;
                        Fact::Varying
                    }
                    SourceKind::UnsupportedIntrinsic
                    | SourceKind::UnresolvedCall
                    | SourceKind::NontrivialDrop
                    | SourceKind::InlineAssembly => {
                        if validate {
                            return Err(self.error(function,effect.site.block,match effect.kind {
                            SourceKind::UnsupportedIntrinsic=>"intrinsic is outside the supported checked block subset",
                            SourceKind::UnresolvedCall=>"unresolved call cannot prove collective participation",
                            _=>"inline assembly or nontrivial destructor participation is unsupported",
                        }));
                        }
                        invalidate_borrows(state);
                        Fact::Varying
                    }
                };
                if let Some(&value) = operands.first() {
                    state.values.insert(value, fact.taint(control.domain));
                }
                continue;
            }
            if kind == TransferKind::WriteCheck {
                let write = MirLibraryWriteCheckOp::new(op);
                if validate && (self.resource_root || self.storage_root) {
                    let source = write.get_attr_library_write_source(self.ctx).unwrap();
                    match source.kind {
                        LibraryWriteKind::Bulk => {
                            return Err(self.error(
                                function,
                                source.site.block,
                                "bulk memory mutation is unsupported in resource-checked roots",
                            ));
                        }
                        LibraryWriteKind::Indirect { nominal_protected }
                            if nominal_protected
                                || facts
                                    .iter()
                                    .any(|f| call_fact(f.clone(), state).has_resource()) =>
                        {
                            return Err(self.error(function,source.site.block,"mutation through a reference to a resource descriptor is unsupported; forward or move the carrier"));
                        }
                        _ => {}
                    }
                }
                continue;
            }
            if kind == TransferKind::Call {
                let call = MirCallOp::new(op);
                let result = if call.get_attr_library_expanded_source(self.ctx).is_some() {
                    scalar_precision(facts.iter().cloned())
                } else if let Some(site) = call.get_attr_library_source_site(self.ctx) {
                    let site = (*site).clone();
                    if let Some(target) = call.get_attr_library_call_target(self.ctx) {
                        if validate && target.kind != LibraryCallKind::Direct {
                            return Err(self.error(
                                function,
                                site.block,
                                "indirect or destructor collective context is unsupported",
                            ));
                        }
                        let arguments =
                            facts.iter().cloned().map(|f| call_fact(f, state)).collect();
                        facts.iter().for_each(|f| state.expose(f));
                        self.path.push(site.block);
                        let analyzed = self.analyze(
                            &target.export,
                            arguments,
                            validate,
                            control,
                            state.pending.clone(),
                            state.epochs.clone(),
                        );
                        self.path.pop();
                        let (result, pending, epochs) = analyzed?;
                        state.pending = pending;
                        state.epochs = epochs;
                        invalidate_borrows(state);
                        result
                    } else {
                        if validate {
                            return Err(self.error(
                                function,
                                site.block,
                                "unresolved call cannot prove collective participation",
                            ));
                        }
                        invalidate_borrows(state);
                        Fact::Varying
                    }
                } else {
                    Fact::Varying
                };
                for &value in &prepared.results {
                    state
                        .values
                        .insert(value, result.clone().taint(control.domain));
                }
                continue;
            }
            if kind == TransferKind::Store {
                let store_op = MirStoreOp::new(op);
                store(
                    state.fact(store_op.address_opd(self.ctx)),
                    state
                        .fact(store_op.value_opd(self.ctx))
                        .taint(control.domain),
                    state,
                );
                continue;
            }
            if kind == TransferKind::Bulk {
                // A transfer may span several fields or elements. Invalidate
                // escaped allocations instead of just the destination field.
                invalidate_borrows(state);
                continue;
            }
            if kind == TransferKind::SetDiscriminant {
                if let Some(address) = facts.first() {
                    store(address.clone(), Fact::Varying, state);
                }
                continue;
            }
            if validate && kind == TransferKind::Unreachable && !aborted {
                return Err(self.error(
                    function,
                    self.source_block(graph, block),
                    "unreachable terminator is not a proved whole-kernel abort",
                ));
            }
            if prepared.results.is_empty() {
                continue;
            }
            let result = prepared.results[0];
            let fact = if let Some(&slot) = graph.slots.get(&op) {
                Fact::Local(slot, vec![])
            } else if kind == TransferKind::Load {
                let load = MirLoadOp::new(op);
                dereference(state.fact(load.address_opd(self.ctx)), state)
            } else if kind == TransferKind::Ref {
                Fact::Referent(Box::new(facts[0].clone()))
            } else if kind == TransferKind::FieldAddr {
                let field = MirFieldAddrOp::new(op);
                let index = field.get_attr_field_index(self.ctx).unwrap().0 as usize;
                match &facts[0] {
                    Fact::Local(slot, path) => {
                        let mut path = path.clone();
                        path.push(index);
                        Fact::Local(*slot, path)
                    }
                    Fact::Referent(f) => Fact::Referent(Box::new(f.field(index))),
                    Fact::Resource(..) | Fact::Descriptor(..) | Fact::Reservation(..) => {
                        Fact::Referent(Box::new(facts[0].clone()))
                    }
                    fact => Fact::Referent(Box::new(fact.field(index))),
                }
            } else if kind == TransferKind::ExtractField {
                let field = MirExtractFieldOp::new(op);
                facts[0].field(field.get_attr_index(self.ctx).unwrap().0 as usize)
            } else if kind == TransferKind::InsertField {
                let field = MirInsertFieldOp::new(op);
                let mut base = facts[0].clone();
                let index = field.get_attr_insert_index(self.ctx).unwrap().0 as usize;
                if !matches!(base, Fact::Fields(_)) {
                    base = aggregate_bottom(self.ctx, result);
                }
                write_fields(&mut base, &[index], facts[1].clone());
                base
            } else if kind == TransferKind::ConstructEnum {
                Fact::Enum {
                    tag_uniform: Domain::BLOCK,
                    fields: facts.clone(),
                }
            } else if kind == TransferKind::GetDiscriminant {
                match &facts[0] {
                    Fact::Enum { tag_uniform, .. } => Fact::Uniform(*tag_uniform),
                    fact => scalar_precision(std::iter::once(fact.clone())),
                }
            } else if kind == TransferKind::EnumPayload {
                let payload = MirEnumPayloadOp::new(op);
                facts[0].variant_field(
                    payload.get_attr_payload_field_index(self.ctx).unwrap().0 as usize,
                )
            } else if kind == TransferKind::Aggregate {
                Fact::Fields(facts.clone())
            } else if kind == TransferKind::Constant {
                Fact::Uniform(Domain::BLOCK)
            } else if kind == TransferKind::Undef {
                aggregate_bottom(self.ctx, result)
            } else if kind == TransferKind::Cast {
                let fact = facts[0].clone();
                if let Fact::Member {
                    width, inverted, ..
                } = fact
                {
                    let target = result.get_type(self.ctx);
                    let target = target.deref(self.ctx);
                    if let Some(target) = target.downcast_ref::<IntegerType>()
                        && (target.width() == 1
                            || target.width() == 8 && target.signedness() == Signedness::Unsigned)
                    {
                        Fact::Member {
                            width,
                            inverted,
                            boolean: target.width() == 1,
                        }
                    } else {
                        Fact::Varying
                    }
                } else if fact.has_local() {
                    let source_ty = operands[0].get_type(self.ctx);
                    let target_ty = result.get_type(self.ctx);
                    let same_pointee = matches!(
                        (
                            source_ty.deref(self.ctx).downcast_ref::<MirPtrType>(),
                            target_ty.deref(self.ctx).downcast_ref::<MirPtrType>(),
                        ),
                        (Some(source), Some(target)) if source.pointee == target.pointee
                    );
                    if source_ty == target_ty || same_pointee {
                        fact
                    } else {
                        // A reinterpreted pointer no longer names the whole slot or field.
                        state.expose(&fact);
                        Fact::Varying
                    }
                } else {
                    fact
                }
            } else if matches!(
                kind,
                TransferKind::PointerOffset | TransferKind::ArrayElementAddr
            ) {
                state.expose(&facts[0]);
                if let Fact::Local(slot, path) = &facts[0] {
                    if kind == TransferKind::PointerOffset
                        && integer(self.ctx, operands[1]) == Some(0)
                    {
                        Fact::Local(*slot, path.clone())
                    } else {
                        Fact::Varying
                    }
                } else {
                    match (facts[0].precision(), facts[1].precision()) {
                        (Some(a), Some(b)) => facts[0].clone().taint(a.meet(b)),
                        _ => Fact::Varying,
                    }
                }
            } else if kind == TransferKind::ExtractArrayElement {
                integer(self.ctx, operands[1]).map_or(Fact::Varying, |i| facts[0].field(i as usize))
            } else if matches!(kind, TransferKind::Eq | TransferKind::Ne) {
                let predicate = facts.iter().enumerate().find_map(|(i, fact)| {
                    let Fact::Member {
                        width, inverted, ..
                    } = fact
                    else {
                        return None;
                    };
                    let constant = integer(self.ctx, operands[1 - i])?;
                    if constant > 1 {
                        return None;
                    }
                    Some(Fact::Member {
                        width: *width,
                        inverted: *inverted ^ (constant == 0) ^ (kind == TransferKind::Ne),
                        boolean: true,
                    })
                });
                predicate.unwrap_or_else(|| scalar_precision(facts.iter().cloned()))
            } else if kind == TransferKind::Not {
                match &facts[0] {
                    Fact::Member {
                        width,
                        inverted,
                        boolean: true,
                    } => Fact::Member {
                        width: *width,
                        inverted: !*inverted,
                        boolean: true,
                    },
                    _ => scalar_precision(facts.iter().cloned()),
                }
            } else if kind == TransferKind::CheckedArithmetic {
                let f = scalar_precision(facts.iter().cloned());
                Fact::Fields(vec![f.clone(), f])
            } else if kind == TransferKind::Scalar {
                scalar_precision(facts.iter().cloned())
            } else {
                Fact::Varying
            };
            state.values.insert(result, fact.taint(control.domain));
            for &value in prepared.results.iter().skip(1) {
                state.values.insert(value, Fact::Varying);
            }
        }
        Ok(())
    }
}
fn aggregate_bottom(ctx: &Context, value: Value) -> Fact {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    if let Some(tuple) = ty.downcast_ref::<MirTupleType>() {
        Fact::Fields(vec![Fact::Unset; tuple.get_types().len()])
    } else if let Some(structure) = ty.downcast_ref::<dialect_mir::types::MirStructType>() {
        Fact::Fields(vec![Fact::Unset; structure.field_count()])
    } else {
        Fact::Varying
    }
}

/// Run the one stateful collective analysis on the actual prepared module.
///
/// Original roles retain root eligibility across preparation. Only structural
/// CFG/SSA information is cached; contextual values, slots and pending resource
/// accesses are evaluated on that same prepared IR before provider execution.
#[allow(clippy::too_many_arguments)]
pub fn analyze_libraries(
    module: Ptr<Operation>,
    ctx: &Context,
    analyses: &mut AnalysisManager,
    rooted_functions: &BTreeMap<String, BTreeSet<String>>,
    symbols: &BTreeMap<String, String>,
    original_roles: &BTreeMap<LibraryCallSite, Role>,
    admitted_shapes: &BTreeMap<String, [u32; 3]>,
    callback_environments: &BTreeMap<LibraryCallSite, LibraryCallbackEnvironment>,
    mut plans: Option<&mut BTreeMap<String, Plan>>,
) -> pliron::result::Result<Validation> {
    let graph = analyses.get_analysis::<PreparedGraph>(module, ctx)?;
    let mut validation = Validation::default();
    for (root, reached) in rooted_functions {
        let roles: Vec<_> = original_roles
            .iter()
            .filter(|(site, _)| reached.contains(&site.caller))
            .map(|(_, role)| *role)
            .collect();
        if !roles.iter().any(|&r| checked(r) || planned_storage(r)) {
            continue;
        }
        let Some(&shape) = admitted_shapes.get(root) else {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "checked collective root {root} has no admitted MIR geometry"
            );
        };
        let resource_root = roles.iter().any(|&r| resource(r).is_some());
        let storage_root = roles.iter().any(|&r| planned_storage(r));
        let Some(function) = symbols.get(root).and_then(|s| graph.functions.get(s)) else {
            return pliron::verify_err!(
                module.deref(ctx).loc(),
                "checked root {root} has no prepared MIR body"
            );
        };
        let arguments = (0..function.blocks[0].deref(ctx).get_num_arguments())
            .map(|i| {
                match MirFuncOp::wrap(ctx, function.op)
                    .unwrap()
                    .library_root_argument(ctx, i)
                {
                    Some(LibraryRootArgumentClass::Scalar) => Fact::Uniform(Domain::BLOCK),
                    Some(LibraryRootArgumentClass::ResourceCandidate)
                        if resource_root || storage_root =>
                    {
                        Fact::Resource(format!("{root}:{}", i + 1), Domain::BLOCK)
                    }
                    Some(
                        LibraryRootArgumentClass::ResourceCandidate
                        | LibraryRootArgumentClass::Kernel,
                    ) => Fact::Kernel(format!("{root}:{}", i + 1), Domain::BLOCK),
                    Some(LibraryRootArgumentClass::StorageCandidate) | None => Fact::Varying,
                }
            })
            .collect();
        let mut trace = Trace::default();
        let mut analyzer = Analyzer {
            ctx,
            graph: &graph,
            symbols,
            callback_environments,
            root,
            resource_root,
            storage_root,
            stack: vec![],
            path: vec![],
            trace: &mut trace,
            plan: plans.as_mut().and_then(|plans| plans.get_mut(root)),
        };
        if let Err(error) = analyzer.analyze(
            root,
            arguments,
            true,
            Control::FULL,
            Pending::new(),
            Epochs::default(),
        ) {
            return pliron::verify_err!(function.op.deref(ctx).loc(), "{error}");
        }
        validation.shapes.insert(root.clone(), shape);
        validation.traces.insert(root.clone(), trace);
    }
    Ok(validation)
}

#[cfg(test)]
#[path = "stateful_tests.rs"]
mod tests;
