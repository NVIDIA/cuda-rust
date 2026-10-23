/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Follow semantic origins through ordinary MIR carriers.
//!
//! Facts are sets of possible origins, including an explicit unknown origin.
//! Family analyses first propagate known facts to a fixed point, allowing loop
//! backedges to inherit their entry origin. They then seed unrelated values,
//! poison missing forwarding inputs, and propagate again before requiring one
//! unambiguous origin. Seeding unknowns earlier would reject valid loop state.
//!
//! A merge of an entry state and its own advanced state retains one origin;
//! merging either with an arbitrary scalar adds `Unknown` and is rejected.
//! Closed local cells conservatively union all stores, irrespective of order.
//! This tracks identity, not dynamic values or general memory aliasing.

use std::collections::{HashMap, HashSet};

use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{
        MirAllocaOp, MirCastOp, MirConstructStructOp, MirConstructTupleOp, MirExtractFieldOp,
        MirFieldAddrOp, MirInsertFieldOp, MirLoadOp, MirStoreOp,
    },
    types::{MirArrayType, MirPtrType, MirStructType, MirTupleType, address_space},
};
use pliron::{
    builtin::{op_interfaces::BranchOpInterface, types::IntegerType},
    context::{Context, Ptr},
    op::{Op, op_cast},
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};

/// Identity of a whole SSA carrier or one nested aggregate field.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct CarrierKey {
    /// The whole SSA carrier before following any aggregate field path.
    pub(super) value: Value,
    /// Empty means the value itself.  Non-empty paths describe a field inside
    /// an ordinary aggregate before a matching `mir.extract_field` exposes it.
    pub(super) path: Vec<u32>,
}

/// Monotone sets of possible semantic origins, indexed by carrier and field path.
#[derive(Clone)]
pub(super) struct OrdinaryFacts<F> {
    /// Accumulated origins; multiple entries at one key represent a merge that consumers must resolve.
    pub(super) facts: HashMap<CarrierKey, HashSet<F>>,
}

impl<F> Default for OrdinaryFacts<F> {
    fn default() -> Self {
        Self {
            facts: HashMap::new(),
        }
    }
}

impl<F> OrdinaryFacts<F>
where
    F: Clone + Eq + std::hash::Hash,
{
    /// Add one origin to a field path and report whether the fixed-point state changed.
    pub(super) fn insert_path(&mut self, value: Value, path: Vec<u32>, fact: F) -> bool {
        self.facts
            .entry(CarrierKey { value, path })
            .or_default()
            .insert(fact)
    }

    /// Add an origin for the whole SSA carrier, with an empty field path.
    pub(super) fn insert(&mut self, value: Value, fact: F) -> bool {
        self.insert_path(value, Vec::new(), fact)
    }

    /// Borrow the possible origins at an exact carrier field path.
    pub(super) fn at_path(&self, value: Value, path: &[u32]) -> Option<&HashSet<F>> {
        self.facts.get(&CarrierKey {
            value,
            path: path.to_vec(),
        })
    }

    /// Borrow origins attached directly to a whole SSA carrier.
    pub(super) fn at(&self, value: Value) -> Option<&HashSet<F>> {
        self.at_path(value, &[])
    }

    /// Collect all nested field origins carried by one SSA value.
    pub(super) fn entries(&self, value: Value) -> Vec<(Vec<u32>, F)> {
        self.facts
            .iter()
            .filter(|(key, _)| key.value == value)
            .flat_map(|(key, facts)| facts.iter().cloned().map(|fact| (key.path.clone(), fact)))
            .collect()
    }

    /// Forward origins below a source field prefix into the corresponding destination prefix.
    pub(super) fn copy_entries(
        &mut self,
        source: Value,
        destination: Value,
        source_prefix: &[u32],
        destination_prefix: &[u32],
    ) -> bool {
        let mut changed = false;
        for (path, fact) in self.entries(source) {
            let Some(suffix) = path.strip_prefix(source_prefix) else {
                continue;
            };
            let mut destination_path = destination_prefix.to_vec();
            destination_path.extend_from_slice(suffix);
            changed |= self.insert_path(destination, destination_path, fact);
        }
        changed
    }
}

/// Read values passed to one successor through Pliron's branch interface.
pub(super) fn branch_successor_operands(
    ctx: &Context,
    terminator: Ptr<Operation>,
    successor_index: usize,
) -> Option<Vec<Value>> {
    let dynamic = Operation::get_op_dyn(terminator, ctx);
    let branch = op_cast::<dyn BranchOpInterface>(dynamic.as_ref())?;
    Some(branch.successor_operands(ctx, successor_index))
}

/// A provenance-preserving cast keeps the pointee and never strengthens
/// mutability. Local-cell aliases must also keep their address space; semantic
/// shared-memory carriers may use the compiler's exact generic/shared
/// round-trip, but no other address-space transition is accepted.
fn is_provenance_preserving_pointer_cast(
    ctx: &Context,
    operation: Ptr<Operation>,
    allow_shared_generic_transition: bool,
) -> bool {
    if Operation::get_opid(operation, ctx) != MirCastOp::get_opid_static()
        || operation.deref(ctx).get_num_operands() != 1
        || operation.deref(ctx).get_num_results() != 1
        || !MirCastOp::new(operation)
            .get_attr_cast_kind(ctx)
            .is_some_and(|kind| matches!(*kind, MirCastKindAttr::PtrToPtr))
    {
        return false;
    }
    let operation_ref = operation.deref(ctx);
    let source_type = operation_ref.get_operand(0).get_type(ctx);
    let result_type = operation_ref.get_result(0).get_type(ctx);
    let source_type_ref = source_type.deref(ctx);
    let result_type_ref = result_type.deref(ctx);
    let (Some(source), Some(result)) = (
        source_type_ref.downcast_ref::<MirPtrType>(),
        result_type_ref.downcast_ref::<MirPtrType>(),
    ) else {
        return false;
    };
    let address_space_is_preserved = source.address_space == result.address_space
        || (allow_shared_generic_transition
            && matches!(
                (source.address_space, result.address_space),
                (address_space::GENERIC, address_space::SHARED)
                    | (address_space::SHARED, address_space::GENERIC)
            ));
    source.pointee == result.pointee
        && address_space_is_preserved
        && (source.is_mutable || !result.is_mutable)
}

/// Accept pointer casts preserving pointee, address space, and non-strengthened mutability.
pub(super) fn is_local_cell_pointer_alias_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_provenance_preserving_pointer_cast(ctx, operation, false)
}

/// Also permit exact generic/shared pointer round-trips for semantic shared carriers.
pub(super) fn is_semantic_pointer_carrier_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_provenance_preserving_pointer_cast(ctx, operation, true)
}

/// Resolve a pointer to one compiler-created local cell and an exact constant
/// field path.  No pointer arithmetic, references, calls, or unknown casts are
/// accepted here.
pub(super) fn local_cell(ctx: &Context, mut pointer: Value) -> Option<(Value, Vec<u32>)> {
    let mut reversed_path = Vec::new();
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(pointer) {
            return None;
        }
        let definer = pointer.defining_op()?;
        let opid = Operation::get_opid(definer, ctx);
        let operation = definer.deref(ctx);
        if opid == MirAllocaOp::get_opid_static() {
            return (operation.get_num_results() == 1 && operation.get_result(0) == pointer).then(
                || {
                    reversed_path.reverse();
                    (pointer, reversed_path)
                },
            );
        }
        if opid == MirCastOp::get_opid_static()
            && operation.get_num_operands() == 1
            && operation.get_num_results() == 1
            && operation.get_result(0) == pointer
            && is_local_cell_pointer_alias_cast(ctx, definer)
        {
            pointer = operation.get_operand(0);
            continue;
        }
        if opid == MirFieldAddrOp::get_opid_static()
            && operation.get_num_operands() == 1
            && operation.get_num_results() == 1
            && operation.get_result(0) == pointer
        {
            let index = MirFieldAddrOp::new(definer).get_attr_field_index(ctx)?;
            reversed_path.push(index.0);
            pointer = operation.get_operand(0);
            continue;
        }
        return None;
    }
}

/// Every alias of an accepted cell must stay within the same small
/// cast/field/load/store vocabulary.  This lets the verifier conservatively
/// merge all stores to the cell without guessing about hidden aliases.
pub(super) fn local_cell_is_closed(ctx: &Context, root: Value) -> bool {
    let mut worklist = vec![root];
    let mut visited = HashSet::new();
    while let Some(pointer) = worklist.pop() {
        if !visited.insert(pointer) {
            continue;
        }
        for r#use in pointer.uses(ctx) {
            let user = r#use.user_op();
            let opid = Operation::get_opid(user, ctx);
            let operation = user.deref(ctx);
            if r#use.find_index(ctx) != 0 {
                return false;
            }
            let forwards_pointer = operation.get_num_results() == 1
                && ((opid == MirCastOp::get_opid_static()
                    && is_local_cell_pointer_alias_cast(ctx, user))
                    || (opid == MirFieldAddrOp::get_opid_static()
                        && MirFieldAddrOp::new(user)
                            .get_attr_field_index(ctx)
                            .is_some()));
            if forwards_pointer {
                worklist.push(operation.get_result(0));
            } else if opid == MirLoadOp::get_opid_static() {
                if MirLoadOp::new(user).is_volatile(ctx) {
                    return false;
                }
            } else if opid == MirStoreOp::get_opid_static() {
                if MirStoreOp::new(user).is_volatile(ctx) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    true
}

/// Propagate semantic facts through the exact ordinary SSA adapters that can
/// remain after common CuTe preparation.  This is deliberately a monotone
/// fixed-point transfer: loop block arguments become known once their entry
/// fact reaches them, then semantic advance operations can feed the same fact
/// around the backedge on a later iteration.
pub(super) fn propagate_ordinary_plumbing<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
) -> bool
where
    F: Clone + Eq + std::hash::Hash,
{
    let mut changed = false;
    let mut blocks = HashSet::new();
    for operation in operations {
        let operation_ref = operation.deref(ctx);
        if let Some(block) = operation_ref.get_parent_block() {
            blocks.insert(block);
        }
        let opid = Operation::get_opid(*operation, ctx);
        if opid == MirCastOp::get_opid_static()
            && is_semantic_pointer_carrier_cast(ctx, *operation)
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            changed |= facts.copy_entries(
                operation_ref.get_operand(0),
                operation_ref.get_result(0),
                &[],
                &[],
            );
        } else if (opid == MirConstructStructOp::get_opid_static()
            || opid == MirConstructTupleOp::get_opid_static())
            && operation_ref.get_num_results() == 1
        {
            let result = operation_ref.get_result(0);
            for (index, operand) in operation_ref.operands().enumerate() {
                changed |= facts.copy_entries(operand, result, &[], &[index as u32]);
            }
        } else if opid == MirInsertFieldOp::get_opid_static()
            && operation_ref.get_num_operands() == 2
            && operation_ref.get_num_results() == 1
        {
            let Some(index) = MirInsertFieldOp::new(*operation).get_attr_insert_index(ctx) else {
                continue;
            };
            let base = operation_ref.get_operand(0);
            let inserted = operation_ref.get_operand(1);
            let result = operation_ref.get_result(0);
            for (path, fact) in facts.entries(base) {
                if path.first().copied() != Some(index.0) {
                    changed |= facts.insert_path(result, path, fact);
                }
            }
            changed |= facts.copy_entries(inserted, result, &[], &[index.0]);
        } else if opid == MirExtractFieldOp::get_opid_static()
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            let Some(index) = MirExtractFieldOp::new(*operation).get_attr_index(ctx) else {
                continue;
            };
            changed |= facts.copy_entries(
                operation_ref.get_operand(0),
                operation_ref.get_result(0),
                &[index.0],
                &[],
            );
        } else if opid == MirStoreOp::get_opid_static() && operation_ref.get_num_operands() == 2 {
            let store = MirStoreOp::new(*operation);
            if !store.is_volatile(ctx)
                && let Some((root, path)) = local_cell(ctx, store.address_opd(ctx))
                && local_cell_is_closed(ctx, root)
            {
                changed |= facts.copy_entries(store.value_opd(ctx), root, &[], &path);
            }
        } else if opid == MirLoadOp::get_opid_static()
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            let load = MirLoadOp::new(*operation);
            if !load.is_volatile(ctx)
                && let Some((root, path)) = local_cell(ctx, load.address_opd(ctx))
                && local_cell_is_closed(ctx, root)
            {
                changed |= facts.copy_entries(root, operation_ref.get_result(0), &path, &[]);
            }
        }
    }

    for block in blocks {
        let block_ref = block.deref(ctx);
        for argument_index in 0..block_ref.get_num_arguments() {
            let argument = block_ref.get_argument(argument_index);
            for edge in block.uses(ctx) {
                let terminator = edge.user_op();
                let Some(operands) =
                    branch_successor_operands(ctx, terminator, edge.find_index(ctx))
                else {
                    continue;
                };
                if let Some(incoming) = operands.get(argument_index).copied() {
                    changed |= facts.copy_entries(incoming, argument, &[], &[]);
                }
            }
        }
    }
    changed
}

/// Enumerate integer fields of the requested bit width inside tuples, structs, and arrays.
fn integer_leaf_paths(
    ctx: &Context,
    ty: TypeHandle,
    width: u32,
    prefix: &mut Vec<u32>,
    output: &mut Vec<Vec<u32>>,
) {
    let ty_ref = ty.deref(ctx);
    if ty_ref
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == width)
    {
        output.push(prefix.clone());
    } else if let Some(structure) = ty_ref.downcast_ref::<MirStructType>() {
        for (index, field) in structure.field_types.iter().copied().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, field, width, prefix, output);
            prefix.pop();
        }
    } else if let Some(tuple) = ty_ref.downcast_ref::<MirTupleType>() {
        for (index, field) in tuple.get_types().iter().copied().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, field, width, prefix, output);
            prefix.pop();
        }
    } else if let Some(array) = ty_ref.downcast_ref::<MirArrayType>() {
        for index in 0..array.size() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, array.element_type(), width, prefix, output);
            prefix.pop();
        }
    }
}

/// Seed factless integer leaves only after known semantic facts stabilize.
/// Aggregates are visited by field path, so an unrelated scalar cannot hide
/// inside a carrier that also contains known fields.
pub(super) fn seed_unknown_integer_carriers<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
    width: u32,
    unknown: impl Fn(Value) -> F,
) where
    F: Clone + Eq + std::hash::Hash,
{
    let mut values = Vec::new();
    let mut blocks = HashSet::new();
    for operation in operations {
        let operation_ref = operation.deref(ctx);
        values.extend(operation_ref.results());
        if let Some(block) = operation_ref.get_parent_block()
            && blocks.insert(block)
        {
            let block_ref = block.deref(ctx);
            values.extend(
                (0..block_ref.get_num_arguments()).map(|index| block_ref.get_argument(index)),
            );
        }
    }

    for value in values {
        let mut paths = Vec::new();
        integer_leaf_paths(ctx, value.get_type(ctx), width, &mut Vec::new(), &mut paths);
        for path in paths {
            if facts.at_path(value, &path).is_none() {
                facts.insert_path(value, path, unknown(value));
            }
        }
    }
}

/// Mark missing inputs to a known block argument or closed local cell unknown.
/// Repeat this and ordinary propagation to a fixed point: one known incoming
/// edge must not conceal another edge or store with no semantic origin.
pub(super) fn poison_missing_forwarding_sources<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
    unknown: impl Fn(Value) -> F + Copy,
) -> bool
where
    F: Clone + Eq + std::hash::Hash,
{
    let mut changed = false;
    let mut blocks = HashSet::new();
    for operation in operations {
        if let Some(block) = operation.deref(ctx).get_parent_block() {
            blocks.insert(block);
        }
        if Operation::get_opid(*operation, ctx) != MirStoreOp::get_opid_static() {
            continue;
        }
        let store = MirStoreOp::new(*operation);
        let Some((root, cell_path)) = local_cell(ctx, store.address_opd(ctx)) else {
            continue;
        };
        if !local_cell_is_closed(ctx, root) {
            continue;
        }
        for (path, _) in facts.entries(root) {
            let Some(source_path) = path.strip_prefix(cell_path.as_slice()) else {
                continue;
            };
            if facts.at_path(store.value_opd(ctx), source_path).is_none() {
                changed |= facts.insert_path(
                    store.value_opd(ctx),
                    source_path.to_vec(),
                    unknown(store.value_opd(ctx)),
                );
            }
        }
    }

    for block in blocks {
        let block_ref = block.deref(ctx);
        for argument_index in 0..block_ref.get_num_arguments() {
            let argument = block_ref.get_argument(argument_index);
            let argument_entries = facts.entries(argument);
            if argument_entries.is_empty() {
                continue;
            }
            for edge in block.uses(ctx) {
                let terminator = edge.user_op();
                let Some(operands) =
                    branch_successor_operands(ctx, terminator, edge.find_index(ctx))
                else {
                    continue;
                };
                let Some(incoming) = operands.get(argument_index).copied() else {
                    continue;
                };
                for (path, _) in &argument_entries {
                    if facts.at_path(incoming, path).is_none() {
                        changed |= facts.insert_path(incoming, path.clone(), unknown(incoming));
                    }
                }
            }
        }
    }
    changed
}
