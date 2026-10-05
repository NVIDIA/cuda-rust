/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Forward compiler-owned multi-result operations across their Rust aggregate
//! return-ABI adapter.
//!
//! Some device operations produce many independent registers, while their Rust
//! API returns an array, tuple, or one-field SIMD wrapper.  The MIR importer has
//! to construct that aggregate and store it to the destination local.  Address
//! projections in the successor then prevent ordinary mem2reg from recovering
//! the original SSA results.  For wide operations this can become a large
//! insertvalue/store/GEP/load web and materially change register allocation.
//!
//! This pass is deliberately narrower than aggregate SROA.  It considers only
//! importer-marked `mir.compiler_result_bundle` constructors and rewrites only
//! after proving the complete boundary:
//!
//! * the construct tree contains every result of exactly one producer, in order;
//! * the outer bundle has one non-volatile store to one local alloca;
//! * the alloca has no other stores, copies, escapes, or unknown pointer users;
//! * every read is a non-volatile load through constant projections, or through
//!   one terminal array projection whose unsigned runtime index is proven to be
//!   `value % C` or guarded by an exact `assert(index < C)` with a small constant
//!   candidate set;
//! * the producer/store dominate every rewritten load.
//!
//! Any failed proof leaves the IR untouched.  Ordinary Rust aggregates are
//! never candidates because only the importer can attach the marker.

use dialect_mir::{
    attributes::{COMPILER_RESULT_BUNDLE_ATTR_KEY, CompilerResultBundleAttr},
    ops::{
        MAX_SCALARIZED_CANDIDATES, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirConstantOp,
        MirConstructArrayOp, MirConstructStructOp, MirConstructTupleOp, MirExtractArrayElementOp,
        MirFieldAddrOp, MirGotoOp, MirLoadOp, MirLtOp, MirRemOp, MirStoreOp,
    },
    types::MirArrayType,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    graph::{ControlFlowGraph, dominance::DomInfo},
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::{ContainsLinkedList, LinkedList},
    location::Located,
    op::Op,
    operation::Operation,
    pass::AnalysisManager,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use rustc_hash::FxHashSet;

#[derive(Clone, Copy)]
enum BoundedIndex {
    Direct(Value),
    Asserted { index: Value, bound: Value },
}

#[derive(Clone, Copy)]
enum LoadReplacement {
    Direct(Value),
    BoundedArrayElement {
        array: Value,
        index: BoundedIndex,
        result_type: TypeHandle,
    },
}

#[derive(Clone, Copy)]
struct LoadForwarding {
    load: Ptr<Operation>,
    replacement: LoadReplacement,
}
struct ForwardingPlan {
    store: Ptr<Operation>,
    bundle_nodes: Vec<Ptr<Operation>>,
    projection_nodes: Vec<Ptr<Operation>>,
    loads: Vec<LoadForwarding>,
    producer_block: Ptr<BasicBlock>,
    store_block: Ptr<BasicBlock>,
}

/// Remove proven compiler-created aggregate ABI boundaries before mem2reg.
pub fn forward_compiler_result_bundles(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    verbose: bool,
) -> Result<usize> {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let mut plans = Vec::new();
    for operation in operations {
        let Some(plan) = analyze_marked_bundle(ctx, operation) else {
            continue;
        };
        if domination_is_proven(module, ctx, analyses, &plan)? {
            plans.push(plan);
        }
    }

    let mut forwarded_loads = 0;
    for plan in plans {
        forwarded_loads += rewrite_plan(ctx, plan);
    }

    if forwarded_loads > 0 && verbose {
        eprintln!(
            "compiler-result forwarding: replaced {forwarded_loads} aggregate projection load(s)"
        );
    }
    Ok(forwarded_loads)
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, output: &mut Vec<Ptr<Operation>>) {
    output.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, output);
            }
        }
    }
}

fn analyze_marked_bundle(ctx: &Context, outer: Ptr<Operation>) -> Option<ForwardingPlan> {
    let key = COMPILER_RESULT_BUNDLE_ATTR_KEY.try_into().ok()?;
    let outer_ref = outer.deref(ctx);
    let marker = outer_ref.attributes.get::<CompilerResultBundleAttr>(&key)?;
    if !marker.0 || !is_construct_op(ctx, outer) {
        return None;
    }

    let outer_value = outer.deref(ctx).get_result(0);
    if outer_value.num_uses(ctx) != 1 {
        return None;
    }
    let outer_use = outer_value.uses(ctx).into_iter().next()?;
    let store_op = outer_use.user_op();
    let store = Operation::get_op::<MirStoreOp>(store_op, ctx)?;
    if outer_use.find_index(ctx) != 1 || store.is_volatile(ctx) {
        return None;
    }

    let root_pointer = store.address_opd(ctx);
    let alloca_op = root_pointer.defining_op()?;
    Operation::get_op::<MirAllocaOp>(alloca_op, ctx)?;

    let mut bundle_nodes = Vec::new();
    let mut leaves = Vec::new();
    flatten_construct_tree(ctx, outer_value, &mut bundle_nodes, &mut leaves)?;
    let first_leaf = *leaves.first()?;
    let producer = first_leaf.defining_op()?;
    if producer.deref(ctx).get_num_results() != leaves.len() || leaves.len() <= 1 {
        return None;
    }
    for (index, leaf) in leaves.iter().enumerate() {
        if *leaf != producer.deref(ctx).get_result(index)
            || leaf.defining_op() != Some(producer)
            || leaf.num_uses(ctx) != 1
        {
            return None;
        }
    }

    let producer_block = producer.deref(ctx).get_parent_block()?;
    let store_block = store_op.deref(ctx).get_parent_block()?;
    if producer_block != store_block || !operation_precedes(ctx, producer, store_op) {
        return None;
    }
    if bundle_nodes.iter().any(|node| {
        node.deref(ctx).get_parent_block() != Some(store_block)
            || !operation_precedes(ctx, *node, store_op)
    }) {
        return None;
    }

    let mut projection_nodes = Vec::new();
    let mut seen_projections = FxHashSet::default();
    let mut loads = Vec::new();
    for root_use in root_pointer.uses(ctx) {
        let user = root_use.user_op();
        let operand_index = root_use.find_index(ctx);
        if user == store_op && operand_index == 0 {
            continue;
        }
        if operand_index != 0
            || (Operation::get_op::<MirFieldAddrOp>(user, ctx).is_none()
                && Operation::get_op::<MirArrayElementAddrOp>(user, ctx).is_none())
        {
            return None;
        }
        analyze_projection(
            ctx,
            user,
            outer_value,
            &bundle_nodes,
            Vec::new(),
            &mut seen_projections,
            &mut projection_nodes,
            &mut loads,
        )?;
    }
    if loads.is_empty() {
        return None;
    }

    Some(ForwardingPlan {
        store: store_op,
        bundle_nodes,
        projection_nodes,
        loads,
        producer_block,
        store_block,
    })
}

fn flatten_construct_tree(
    ctx: &Context,
    value: Value,
    nodes: &mut Vec<Ptr<Operation>>,
    leaves: &mut Vec<Value>,
) -> Option<()> {
    let Some(operation) = value.defining_op() else {
        leaves.push(value);
        return Some(());
    };
    if !is_construct_op(ctx, operation) {
        leaves.push(value);
        return Some(());
    }
    if operation.deref(ctx).get_result(0) != value || value.num_uses(ctx) != 1 {
        return None;
    }
    nodes.push(operation);
    for index in 0..operation.deref(ctx).get_num_operands() {
        flatten_construct_tree(ctx, operation.deref(ctx).get_operand(index), nodes, leaves)?;
    }
    Some(())
}

fn is_construct_op(ctx: &Context, operation: Ptr<Operation>) -> bool {
    Operation::get_op::<MirConstructArrayOp>(operation, ctx).is_some()
        || Operation::get_op::<MirConstructTupleOp>(operation, ctx).is_some()
        || Operation::get_op::<MirConstructStructOp>(operation, ctx).is_some()
}

#[allow(clippy::too_many_arguments)]
fn analyze_projection(
    ctx: &Context,
    projection: Ptr<Operation>,
    bundle: Value,
    bundle_nodes: &[Ptr<Operation>],
    mut path: Vec<usize>,
    seen: &mut FxHashSet<Ptr<Operation>>,
    projection_nodes: &mut Vec<Ptr<Operation>>,
    loads: &mut Vec<LoadForwarding>,
) -> Option<()> {
    if !seen.insert(projection) {
        return None;
    }

    // Static projections continue to extend `path`. A non-constant array
    // projection is accepted only as the terminal projection into one of this
    // compiler-owned bundle's constructed arrays. Its candidate set must be
    // proven separately for each load, either by an unsigned `value % C` or by
    // an exact dominating `assert(index < C)`.
    let bounded_array = if let Some(field) = Operation::get_op::<MirFieldAddrOp>(projection, ctx) {
        path.push(field.get_attr_field_index(ctx)?.0 as usize);
        None
    } else if Operation::get_op::<MirArrayElementAddrOp>(projection, ctx).is_some() {
        let index = projection.deref(ctx).get_operand(1);
        if let Some(constant_index) = integer_constant_usize(ctx, index) {
            path.push(constant_index);
            None
        } else {
            let array = resolve_construct_path(ctx, bundle, &path)?;
            let array_definer = array.defining_op()?;
            if !bundle_nodes.contains(&array_definer)
                || Operation::get_op::<MirConstructArrayOp>(array_definer, ctx).is_none()
            {
                return None;
            }

            let array_type = array.get_type(ctx);
            let array_type_ref = array_type.deref(ctx);
            let array_info = array_type_ref.downcast_ref::<MirArrayType>()?;
            let element_type = array_info.element_type();

            Some((array, index, element_type, array_info.size()))
        }
    } else {
        return None;
    };

    projection_nodes.push(projection);
    let pointer = projection.deref(ctx).get_result(0);
    if pointer.num_uses(ctx) == 0 {
        return None;
    }
    for pointer_use in pointer.uses(ctx) {
        if pointer_use.find_index(ctx) != 0 {
            return None;
        }
        let user = pointer_use.user_op();
        if let Some(load) = Operation::get_op::<MirLoadOp>(user, ctx) {
            if load.is_volatile(ctx) {
                return None;
            }

            let result_type = user.deref(ctx).get_result(0).get_type(ctx);
            let replacement = if let Some((array, index, element_type, array_size)) = bounded_array
            {
                if result_type != element_type {
                    return None;
                }
                let projection_block = projection.deref(ctx).get_parent_block()?;
                let load_block = user.deref(ctx).get_parent_block()?;
                let index =
                    bounded_dynamic_index(ctx, index, projection_block, load_block, array_size)?;
                LoadReplacement::BoundedArrayElement {
                    array,
                    index,
                    result_type,
                }
            } else {
                let replacement = resolve_construct_path(ctx, bundle, &path)?;
                // A whole-subaggregate read (e.g. loading the array field of a
                // struct-wrapped bundle in one piece) resolves to one of the
                // bundle's own construct results. Forwarding it would leave live
                // uses of an operation this pass erases, so fail closed and keep
                // the memory path.
                if replacement
                    .defining_op()
                    .is_some_and(|definer| bundle_nodes.contains(&definer))
                {
                    return None;
                }
                if replacement.get_type(ctx) != result_type {
                    return None;
                }
                LoadReplacement::Direct(replacement)
            };

            loads.push(LoadForwarding {
                load: user,
                replacement,
            });
            continue;
        }

        // Once a runtime array element has been selected, no further pointer
        // projection is supported. Compiler result bundles are flat register
        // packs (possibly behind static struct/tuple wrappers); allowing a
        // dynamic projection into another aggregate would require a separate
        // proof about the selected subaggregate.
        if bounded_array.is_some() {
            return None;
        }
        if Operation::get_op::<MirFieldAddrOp>(user, ctx).is_none()
            && Operation::get_op::<MirArrayElementAddrOp>(user, ctx).is_none()
        {
            return None;
        }
        analyze_projection(
            ctx,
            user,
            bundle,
            bundle_nodes,
            path.clone(),
            seen,
            projection_nodes,
            loads,
        )?;
    }
    Some(())
}

fn integer_constant_usize(ctx: &Context, value: Value) -> Option<usize> {
    let operation = value.defining_op()?;
    let constant = Operation::get_op::<MirConstantOp>(operation, ctx)?;
    let attribute = constant.get_attr_value(ctx)?;
    let integer = attribute.value();
    if integer.bw() > usize::BITS as usize {
        return None;
    }
    usize::try_from(integer.to_u64()).ok()
}

fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
    let operation = value.defining_op()?;
    let constant = Operation::get_op::<MirConstantOp>(operation, ctx)?;
    let attribute = constant.get_attr_value(ctx)?;
    let integer = attribute.value();
    (integer.bw() <= 64).then(|| integer.to_u64())
}

fn validate_candidate_count(candidate_count: u64, array_size: u64) -> Option<()> {
    (candidate_count > 0
        && candidate_count <= array_size
        && candidate_count <= MAX_SCALARIZED_CANDIDATES)
        .then_some(())
}

fn bounded_dynamic_index(
    ctx: &Context,
    index: Value,
    projection_block: Ptr<BasicBlock>,
    load_block: Ptr<BasicBlock>,
    array_size: u64,
) -> Option<BoundedIndex> {
    let index_type = index.get_type(ctx);
    let index_type_ref = index_type.deref(ctx);
    let integer_type = index_type_ref.downcast_ref::<IntegerType>()?;
    if integer_type.signedness() != Signedness::Unsigned {
        return None;
    }

    if let Some(remainder) = index.defining_op()
        && Operation::get_op::<MirRemOp>(remainder, ctx).is_some()
    {
        let divisor = remainder.deref(ctx).get_operand(1);
        if divisor.get_type(ctx) != index_type {
            return None;
        }
        let candidate_count = integer_constant_u64(ctx, divisor)?;
        validate_candidate_count(candidate_count, array_size)?;
        return Some(BoundedIndex::Direct(index));
    }

    // Keep the assertion proof deliberately narrow. The dynamic projection
    // and its load must be in the same block, that block must have one
    // predecessor, and that predecessor must assert `index < constant`
    // immediately before its goto to the load. The comparison must use this
    // precise SSA index value. This is sufficient to establish domination without
    // introducing a general range analysis.
    if projection_block != load_block {
        return None;
    }
    let region = load_block.deref(ctx).get_parent_region()?;
    let predecessors = region.predecessors(ctx, &load_block);
    let [assert_block] = predecessors.as_slice() else {
        return None;
    };

    let terminator = assert_block.deref(ctx).get_terminator(ctx)?;
    Operation::get_op::<MirGotoOp>(terminator, ctx)?;
    if terminator.deref(ctx).get_num_successors() != 1
        || terminator.deref(ctx).get_successor(0) != load_block
    {
        return None;
    }

    let assertion = terminator.deref(ctx).get_prev()?;
    Operation::get_op::<MirAssertOp>(assertion, ctx)?;
    let condition = assertion.deref(ctx).get_operand(0);
    let comparison = condition.defining_op()?;
    Operation::get_op::<MirLtOp>(comparison, ctx)?;
    if comparison.deref(ctx).get_parent_block() != Some(*assert_block)
        || comparison.deref(ctx).get_operand(0) != index
    {
        return None;
    }

    let bound = comparison.deref(ctx).get_operand(1);
    if bound.get_type(ctx) != index_type {
        return None;
    }
    let candidate_count = integer_constant_u64(ctx, bound)?;
    validate_candidate_count(candidate_count, array_size)?;

    Some(BoundedIndex::Asserted { index, bound })
}

fn resolve_construct_path(ctx: &Context, mut value: Value, path: &[usize]) -> Option<Value> {
    for &index in path {
        let operation = value.defining_op()?;
        if !is_construct_op(ctx, operation) || operation.deref(ctx).get_result(0) != value {
            return None;
        }
        if index >= operation.deref(ctx).get_num_operands() {
            return None;
        }
        value = operation.deref(ctx).get_operand(index);
    }
    Some(value)
}

fn domination_is_proven(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    plan: &ForwardingPlan,
) -> Result<bool> {
    let Some(region) = plan.store_block.deref(ctx).get_parent_region() else {
        return Ok(false);
    };
    if plan.producer_block.deref(ctx).get_parent_region() != Some(region) {
        return Ok(false);
    }

    let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
    let dom = dom_info.get_dom_tree(ctx, region);
    for load in &plan.loads {
        let Some(load_block) = load.load.deref(ctx).get_parent_block() else {
            return Ok(false);
        };
        if load_block.deref(ctx).get_parent_region() != Some(region) {
            return Ok(false);
        }
        if load_block == plan.store_block {
            if !operation_precedes(ctx, plan.store, load.load) {
                return Ok(false);
            }
        } else if !dom.dominates(&plan.store_block, &load_block) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn operation_precedes(ctx: &Context, first: Ptr<Operation>, second: Ptr<Operation>) -> bool {
    let Some(block) = first.deref(ctx).get_parent_block() else {
        return false;
    };
    if second.deref(ctx).get_parent_block() != Some(block) {
        return false;
    }
    for operation in block.deref(ctx).iter(ctx) {
        if operation == first {
            return true;
        }
        if operation == second {
            return false;
        }
    }
    false
}

fn rewrite_plan(ctx: &mut Context, plan: ForwardingPlan) -> usize {
    let count = plan.loads.len();
    let mut rewriter = IRRewriter::<Recorder>::default();

    for forwarding in plan.loads {
        let replacement = match forwarding.replacement {
            LoadReplacement::Direct(replacement) => replacement,
            LoadReplacement::BoundedArrayElement {
                array,
                index,
                result_type,
            } => {
                let location = forwarding.load.deref(ctx).loc().clone();
                let bounded_index = match index {
                    BoundedIndex::Direct(index) => index,
                    BoundedIndex::Asserted { index, bound } => {
                        // The assertion proves `index < bound`, so this remainder
                        // is value-preserving. It carries the candidate count in
                        // the canonical form already consumed by mir-lower.
                        let remainder = Operation::new(
                            ctx,
                            MirRemOp::get_concrete_op_info(),
                            vec![index.get_type(ctx)],
                            vec![index, bound],
                            vec![],
                            0,
                        );
                        remainder.deref_mut(ctx).set_loc(location.clone());
                        remainder.insert_before(ctx, forwarding.load);
                        remainder.deref(ctx).get_result(0)
                    }
                };
                let extract = Operation::new(
                    ctx,
                    MirExtractArrayElementOp::get_concrete_op_info(),
                    vec![result_type],
                    vec![array, bounded_index],
                    vec![],
                    0,
                );
                extract.deref_mut(ctx).set_loc(location);
                extract.insert_before(ctx, forwarding.load);
                extract.deref(ctx).get_result(0)
            }
        };

        let old_result = forwarding.load.deref(ctx).get_result(0);
        old_result.replace_all_uses_with(ctx, &replacement);
        rewriter.erase_operation(ctx, forwarding.load);
    }
    for projection in plan.projection_nodes.into_iter().rev() {
        rewriter.erase_operation(ctx, projection);
    }
    rewriter.erase_operation(ctx, plan.store);

    // Static forwarding makes the complete construct tree dead. Bounded
    // dynamic extraction deliberately keeps the selected array constructor in
    // SSA, so erase only constructors whose result is actually unused after
    // the store/projection boundary has been removed. `bundle_nodes` is
    // preorder (outer before inner), which also releases inner constructors
    // when an otherwise-dead wrapper is erased.
    for constructor in plan.bundle_nodes {
        if constructor.deref(ctx).get_result(0).num_uses(ctx) == 0 {
            rewriter.erase_operation(ctx, constructor);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        attributes::{CompilerResultBundleAttr, FieldIndexAttr},
        ops::{MirAssertOp, MirCallOp, MirFuncOp, MirGotoOp, MirLtOp, MirReturnOp},
        types::{MirArrayType, MirPtrType, MirStructType},
    };
    use pliron::{
        builtin::{
            attributes::{IntegerAttr, StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        identifier::Identifier,
        op::Op,
        region::Region,
        r#type::TypeHandle,
        utils::apint::APInt,
    };
    use std::num::NonZeroUsize;

    struct Fixture {
        module: Ptr<Operation>,
        producer: Ptr<Operation>,
        return_op: Ptr<Operation>,
    }

    #[derive(Clone, Copy)]
    enum IndexShape {
        ConstantLast,
        Dynamic,
        BoundedRem { divisor: u64 },
        SignedRem { divisor: u64 },
        Asserted { bound: u64 },
        SignedAsserted { bound: u64 },
        MismatchedAsserted { bound: u64 },
        DynamicAssertedBound,
    }

    fn build_fixture(ctx: &mut Context, marked: bool, extra_store: bool, width: usize) -> Fixture {
        build_fixture_with_index(ctx, marked, extra_store, width, IndexShape::ConstantLast)
    }

    fn build_fixture_with_index(
        ctx: &mut Context,
        marked: bool,
        extra_store: bool,
        width: usize,
        index_shape: IndexShape,
    ) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, width as u64).into();
        let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, array_type, true).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![], vec![element_type]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, "kernel".try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![pointer_type],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        // A call is sufficient as a typed, independent multi-result producer
        // for this pass test. The forwarding proof does not depend on its
        // opcode.
        let producer = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type; width],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(producer).set_attr_callee(ctx, StringAttr::new("register_pack".to_string()));
        producer.insert_at_back(entry, ctx);

        let producer_results: Vec<_> = (0..width)
            .map(|index| producer.deref(ctx).get_result(index))
            .collect();
        let bundle = Operation::new(
            ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_type],
            producer_results,
            vec![],
            0,
        );
        if marked {
            bundle.deref_mut(ctx).attributes.set(
                Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY).unwrap(),
                CompilerResultBundleAttr(true),
            );
        }
        bundle.insert_at_back(entry, ctx);
        let bundle_value = bundle.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, bundle_value],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);
        if extra_store {
            let store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![slot, bundle_value],
                vec![],
                0,
            );
            store.insert_at_back(entry, ctx);
        }

        let signedness = match index_shape {
            IndexShape::SignedRem { .. } | IndexShape::SignedAsserted { .. } => Signedness::Signed,
            _ => Signedness::Unsigned,
        };
        let index_type = IntegerType::get(ctx, 64, signedness);
        let index_handle: TypeHandle = index_type.into();

        let index_value = match index_shape {
            IndexShape::ConstantLast => {
                let goto = Operation::new(
                    ctx,
                    MirGotoOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![body],
                    0,
                );
                goto.insert_at_back(entry, ctx);

                let index = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![index_handle],
                    vec![],
                    vec![],
                    0,
                );
                MirConstantOp::new(index).set_attr_value(
                    ctx,
                    IntegerAttr::new(
                        index_type,
                        APInt::from_u64((width - 1) as u64, NonZeroUsize::new(64).unwrap()),
                    ),
                );
                index.insert_at_back(body, ctx);
                index.deref(ctx).get_result(0)
            }
            IndexShape::Dynamic | IndexShape::BoundedRem { .. } | IndexShape::SignedRem { .. } => {
                let goto = Operation::new(
                    ctx,
                    MirGotoOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![body],
                    0,
                );
                goto.insert_at_back(entry, ctx);

                let raw_index = Operation::new(
                    ctx,
                    MirCallOp::get_concrete_op_info(),
                    vec![index_handle],
                    vec![],
                    vec![],
                    0,
                );
                MirCallOp::new(raw_index)
                    .set_attr_callee(ctx, StringAttr::new("runtime_index".to_string()));
                raw_index.insert_at_back(body, ctx);
                let raw_index_value = raw_index.deref(ctx).get_result(0);

                match index_shape {
                    IndexShape::Dynamic => raw_index_value,
                    IndexShape::BoundedRem { divisor } | IndexShape::SignedRem { divisor } => {
                        let constant = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![index_handle],
                            vec![],
                            vec![],
                            0,
                        );
                        MirConstantOp::new(constant).set_attr_value(
                            ctx,
                            IntegerAttr::new(
                                index_type,
                                APInt::from_u64(divisor, NonZeroUsize::new(64).unwrap()),
                            ),
                        );
                        constant.insert_at_back(body, ctx);
                        let divisor_value = constant.deref(ctx).get_result(0);

                        let remainder = Operation::new(
                            ctx,
                            MirRemOp::get_concrete_op_info(),
                            vec![index_handle],
                            vec![raw_index_value, divisor_value],
                            vec![],
                            0,
                        );
                        remainder.insert_at_back(body, ctx);
                        remainder.deref(ctx).get_result(0)
                    }
                    _ => unreachable!(),
                }
            }
            IndexShape::Asserted { .. }
            | IndexShape::SignedAsserted { .. }
            | IndexShape::MismatchedAsserted { .. }
            | IndexShape::DynamicAssertedBound => {
                let raw_index = Operation::new(
                    ctx,
                    MirCallOp::get_concrete_op_info(),
                    vec![index_handle],
                    vec![],
                    vec![],
                    0,
                );
                MirCallOp::new(raw_index)
                    .set_attr_callee(ctx, StringAttr::new("runtime_index".to_string()));
                raw_index.insert_at_back(entry, ctx);
                let raw_index_value = raw_index.deref(ctx).get_result(0);

                let compared_index = if matches!(index_shape, IndexShape::MismatchedAsserted { .. })
                {
                    let other_index = Operation::new(
                        ctx,
                        MirCallOp::get_concrete_op_info(),
                        vec![index_handle],
                        vec![],
                        vec![],
                        0,
                    );
                    MirCallOp::new(other_index)
                        .set_attr_callee(ctx, StringAttr::new("other_runtime_index".to_string()));
                    other_index.insert_at_back(entry, ctx);
                    other_index.deref(ctx).get_result(0)
                } else {
                    raw_index_value
                };

                let bound_value = match index_shape {
                    IndexShape::DynamicAssertedBound => {
                        let bound = Operation::new(
                            ctx,
                            MirCallOp::get_concrete_op_info(),
                            vec![index_handle],
                            vec![],
                            vec![],
                            0,
                        );
                        MirCallOp::new(bound)
                            .set_attr_callee(ctx, StringAttr::new("runtime_bound".to_string()));
                        bound.insert_at_back(entry, ctx);
                        bound.deref(ctx).get_result(0)
                    }
                    IndexShape::Asserted { bound }
                    | IndexShape::SignedAsserted { bound }
                    | IndexShape::MismatchedAsserted { bound } => {
                        let constant = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![index_handle],
                            vec![],
                            vec![],
                            0,
                        );
                        MirConstantOp::new(constant).set_attr_value(
                            ctx,
                            IntegerAttr::new(
                                index_type,
                                APInt::from_u64(bound, NonZeroUsize::new(64).unwrap()),
                            ),
                        );
                        constant.insert_at_back(entry, ctx);
                        constant.deref(ctx).get_result(0)
                    }
                    _ => unreachable!(),
                };

                let i1_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
                let comparison = Operation::new(
                    ctx,
                    MirLtOp::get_concrete_op_info(),
                    vec![i1_type],
                    vec![compared_index, bound_value],
                    vec![],
                    0,
                );
                comparison.insert_at_back(entry, ctx);
                let condition = comparison.deref(ctx).get_result(0);

                MirAssertOp::new(ctx, condition)
                    .get_operation()
                    .insert_at_back(entry, ctx);
                let goto = Operation::new(
                    ctx,
                    MirGotoOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![body],
                    0,
                );
                goto.insert_at_back(entry, ctx);

                raw_index_value
            }
        };

        finish_fixture(
            ctx,
            module.get_operation(),
            producer,
            body,
            slot,
            element_type,
            index_value,
        )
    }

    fn finish_fixture(
        ctx: &mut Context,
        module: Ptr<Operation>,
        producer: Ptr<Operation>,
        body: Ptr<BasicBlock>,
        slot: Value,
        element_type: TypeHandle,
        index_value: Value,
    ) -> Fixture {
        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![slot, index_value],
            vec![],
            0,
        );
        address.insert_at_back(body, ctx);
        let address_value = address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_type],
            vec![address_value],
            vec![],
            0,
        );
        load.insert_at_back(body, ctx);

        let load_value = load.deref(ctx).get_result(0);
        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![load_value],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module,
            producer,
            return_op,
        }
    }

    fn count<T: Op>(ctx: &Context, root: Ptr<Operation>) -> usize {
        let mut operations = Vec::new();
        collect_ops(ctx, root, &mut operations);
        operations
            .into_iter()
            .filter(|operation| Operation::get_op::<T>(*operation, ctx).is_some())
            .count()
    }

    /// Models the importer output for a struct-wrapped bundle such as
    /// `CuSimd([r0, r1])` that is read back in one piece (`CuSimd::to_array`):
    /// a `mir.field_addr` projection followed by a whole-array load.
    fn build_struct_wrapped_fixture(ctx: &mut Context) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, 2).into();
        let struct_type: TypeHandle = MirStructType::get(
            ctx,
            "CuSimd".to_string(),
            vec!["inner".to_string()],
            vec![array_type],
        )
        .into();
        let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, struct_type, true).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![], vec![array_type]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, "kernel".try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![pointer_type],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        let producer = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type, element_type],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(producer).set_attr_callee(ctx, StringAttr::new("register_pair".to_string()));
        producer.insert_at_back(entry, ctx);

        let producer_results = vec![
            producer.deref(ctx).get_result(0),
            producer.deref(ctx).get_result(1),
        ];
        let inner = Operation::new(
            ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_type],
            producer_results,
            vec![],
            0,
        );
        inner.insert_at_back(entry, ctx);
        let inner_value = inner.deref(ctx).get_result(0);

        let outer = Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![struct_type],
            vec![inner_value],
            vec![],
            0,
        );
        outer.deref_mut(ctx).attributes.set(
            Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY).unwrap(),
            CompilerResultBundleAttr(true),
        );
        outer.insert_at_back(entry, ctx);
        let outer_value = outer.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, outer_value],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);

        let goto = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![body],
            0,
        );
        goto.insert_at_back(entry, ctx);

        let array_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field_address = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![array_pointer],
            vec![slot],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field_address).set_attr_field_index(ctx, FieldIndexAttr(0));
        field_address.insert_at_back(body, ctx);
        let field_address_value = field_address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![array_type],
            vec![field_address_value],
            vec![],
            0,
        );
        load.insert_at_back(body, ctx);

        let load_value = load.deref(ctx).get_result(0);
        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![load_value],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module: module.get_operation(),
            producer,
            return_op,
        }
    }

    #[test]
    fn marked_exact_bundle_forwards_to_producer_results() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, false, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(
            fixture.return_op.deref(&ctx).get_operand(0),
            fixture.producer.deref(&ctx).get_result(1)
        );
    }

    /// A 64-result producer models the widest `ptx_asm!` output pack the
    /// macro now accepts (Blackwell tcgen05 tensor-memory loads).  The pass
    /// must forward the full-width bundle just like the two-wide case.
    #[test]
    fn sixty_four_result_pack_forwards_to_producer_results() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, false, 64);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(
            fixture.return_op.deref(&ctx).get_operand(0),
            fixture.producer.deref(&ctx).get_result(63)
        );
    }

    #[test]
    fn bounded_dynamic_index_forwards_to_ssa_extraction() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            32,
            IndexShape::BoundedRem { divisor: 4 },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(
            count::<MirConstructArrayOp>(&ctx, fixture.module),
            1,
            "the SSA array constructor must remain live for dynamic extraction"
        );

        let result = fixture.return_op.deref(&ctx).get_operand(0);
        let result_definer = result
            .defining_op()
            .expect("bounded result must be produced by an extraction op");
        assert!(Operation::get_op::<MirExtractArrayElementOp>(result_definer, &ctx).is_some());
    }

    #[test]
    fn asserted_dynamic_index_forwards_to_ssa_extraction() {
        let mut ctx = Context::new();
        let fixture =
            build_fixture_with_index(&mut ctx, true, false, 32, IndexShape::Asserted { bound: 4 });
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(
            count::<MirRemOp>(&ctx, fixture.module),
            1,
            "assert-proven indices must be canonicalized to bounded remainder form"
        );
        assert_eq!(
            count::<MirConstructArrayOp>(&ctx, fixture.module),
            1,
            "the SSA array constructor must remain live for dynamic extraction"
        );
    }

    #[test]
    fn maximum_asserted_candidate_count_is_forwarded() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            64,
            IndexShape::Asserted {
                bound: MAX_SCALARIZED_CANDIDATES,
            },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn oversized_asserted_candidate_count_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            64,
            IndexShape::Asserted {
                bound: MAX_SCALARIZED_CANDIDATES + 1,
            },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn asserted_candidate_count_larger_than_array_fails_closed() {
        let mut ctx = Context::new();
        let fixture =
            build_fixture_with_index(&mut ctx, true, false, 4, IndexShape::Asserted { bound: 5 });
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn signed_asserted_index_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            8,
            IndexShape::SignedAsserted { bound: 4 },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn assertion_about_different_index_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            8,
            IndexShape::MismatchedAsserted { bound: 4 },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn dynamic_assert_bound_fails_closed() {
        let mut ctx = Context::new();
        let fixture =
            build_fixture_with_index(&mut ctx, true, false, 8, IndexShape::DynamicAssertedBound);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn maximum_bounded_candidate_count_is_forwarded() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            64,
            IndexShape::BoundedRem {
                divisor: MAX_SCALARIZED_CANDIDATES,
            },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn oversized_bounded_candidate_count_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            64,
            IndexShape::BoundedRem {
                divisor: MAX_SCALARIZED_CANDIDATES + 1,
            },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn bounded_candidate_count_larger_than_array_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            4,
            IndexShape::BoundedRem { divisor: 5 },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn unbounded_dynamic_index_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(&mut ctx, true, false, 8, IndexShape::Dynamic);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn signed_remainder_index_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture_with_index(
            &mut ctx,
            true,
            false,
            8,
            IndexShape::SignedRem { divisor: 4 },
        );
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    /// Regression test: a whole-array field load out of a struct-wrapped
    /// bundle resolves to the bundle's own inner construct op.  Forwarding it
    /// would leave live uses of an erased operation (an ICE), so the pass must
    /// fail closed and leave the memory path intact.
    #[test]
    fn struct_wrapped_whole_array_load_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_struct_wrapped_fixture(&mut ctx);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructStructOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn ordinary_unmarked_aggregate_is_untouched() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, false, false, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn additional_bundle_store_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, true, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 2);
    }
}
