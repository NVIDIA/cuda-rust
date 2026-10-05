/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalarize aggregate-typed block arguments after lowering to the LLVM
//! dialect.
//!
//! # Why this pass exists
//!
//! pliron's `mem2reg` promotes a stack slot into a block argument of the
//! slot's complete type. For an enum local such as `Option<(f32, f32)>` (an
//! iterator's `next()` result), the control-flow join therefore carries one
//! block argument of the whole aggregate, which the textual exporter prints
//! as a PHI of a first-class aggregate:
//!
//! ```text
//! %v = phi { i32, { float, float } } [ %some, %bb1 ], [ %none, %bb2 ]
//! %d = extractvalue { i32, { float, float } } %v, 0
//! ```
//!
//! LLVM's -O2 pipeline cannot take such IR apart: SROA only splits allocas,
//! and InstCombine does not push `extractvalue` through a `phi`. A hot loop
//! that merges an iterator's `Option` result keeps a materialized
//! discriminant register plus one extra compare-and-branch per iteration,
//! which `rustc_codegen_llvm`-produced IR (enum locals in memory, split by
//! SROA into scalar phis) never exhibits.
//!
//! # What the pass does
//!
//! For every LLVM struct- or array-typed argument of a non-entry block with
//! two or more incoming edges (a single-incoming PHI folds on its own):
//!
//! 1. append one new block argument per scalar leaf of the aggregate,
//! 2. rebuild the aggregate at the block head with `llvm.undef` +
//!    `llvm.insertvalue`, and replace all uses of the old argument,
//! 3. in every predecessor, split the forwarded aggregate operand with
//!    `llvm.extractvalue` per leaf, and
//! 4. remove the old aggregate argument and operands.
//!
//! The exporter then prints scalar PHIs plus extractvalue-of-insertvalue
//! chains, which InstCombine and SimplifyCFG fold completely, the same shape
//! SROA produces for rustc's own codegen.

use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::region::Region;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_hash::FxHashMap;

/// Leaf budget per split argument. An aggregate that expands to more scalar
/// leaves than this (for example a large array) is left alone: the register
/// pressure of the split would outweigh the branch it saves, and such values
/// do not occur on the hot paths this pass exists for.
const MAX_LEAVES: usize = 16;

/// One block argument scheduled for splitting.
struct SplitPlan {
    /// Index of the aggregate argument in the block's original argument list.
    arg_idx: usize,
    /// Original aggregate type of the argument.
    aggregate_ty: TypeHandle,
    /// `(index path, leaf type)` for every scalar leaf, in field order.
    leaves: Vec<(Vec<u32>, TypeHandle)>,
    /// Block-argument indices of the appended leaf arguments.
    leaf_arg_indices: Vec<usize>,
}

/// Split every aggregate-typed non-entry block argument in `module_op` into
/// scalar leaves. Runs on the LLVM dialect module produced by the
/// `dialect-mir` conversion.
pub fn scalarize_aggregate_block_args(ctx: &mut Context, module_op: Ptr<Operation>) -> Result<()> {
    let functions: Vec<Ptr<Operation>> = {
        let module = module_op.deref(ctx);
        let region = module.get_region(0).deref(ctx);
        let mut ops = Vec::new();
        for block in region.iter(ctx) {
            ops.extend(block.deref(ctx).iter(ctx));
        }
        ops
    };
    for func in functions {
        let num_regions = func.deref(ctx).num_regions();
        for region_idx in 0..num_regions {
            let region = func.deref(ctx).get_region(region_idx);
            scalarize_region(ctx, region)?;
        }
    }
    Ok(())
}

fn scalarize_region(ctx: &mut Context, region: Ptr<Region>) -> Result<()> {
    let blocks: Vec<Ptr<BasicBlock>> = region.deref(ctx).iter(ctx).collect();
    if blocks.len() < 2 {
        return Ok(());
    }

    // Map each block to the branch edges that feed it: (terminator, succ_idx).
    let mut incoming_edges: FxHashMap<Ptr<BasicBlock>, Vec<(Ptr<Operation>, usize)>> =
        FxHashMap::default();
    for block in &blocks {
        let Some(term) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        let successors: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
        for (succ_idx, succ) in successors.into_iter().enumerate() {
            incoming_edges
                .entry(succ)
                .or_default()
                .push((term, succ_idx));
        }
    }

    // The entry block's arguments are the function parameters; skip it.
    for block in blocks.iter().skip(1) {
        let edges = incoming_edges.get(block).cloned().unwrap_or_default();
        scalarize_block(ctx, *block, &edges)?;
    }
    Ok(())
}

fn scalarize_block(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    edges: &[(Ptr<Operation>, usize)],
) -> Result<()> {
    let num_args = block.deref(ctx).get_num_arguments();
    if num_args == 0 {
        return Ok(());
    }

    // A block with fewer than two incoming edges exports as single-incoming
    // PHIs, which LLVM folds by itself even for aggregates; only a real merge
    // (two or more edges) produces the opaque aggregate PHI this pass exists
    // to split. Restricting to merges also avoids churning the argument list
    // of every kernel prologue successor.
    if edges.len() < 2 {
        return Ok(());
    }

    // Plan which arguments to split.
    let mut plans: Vec<SplitPlan> = Vec::new();
    for arg_idx in 0..num_args {
        let ty = block.deref(ctx).get_argument(arg_idx).get_type(ctx);
        if let Some(leaves) = aggregate_leaves(ctx, ty) {
            plans.push(SplitPlan {
                arg_idx,
                aggregate_ty: ty,
                leaves,
                leaf_arg_indices: Vec::new(),
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // An exotic terminator cannot be edited through BranchOpInterface. Leave
    // such blocks untouched.
    for (term, _) in edges {
        let term_obj = Operation::get_op_dyn(*term, ctx);
        if op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).is_none() {
            return Ok(());
        }
    }

    // 1. Append one new block argument per leaf, in plan order.
    for plan in &mut plans {
        for (_, leaf_ty) in &plan.leaves {
            let idx = BasicBlock::push_argument(block, ctx, *leaf_ty);
            plan.leaf_arg_indices.push(idx);
        }
    }

    // 2. Rebuild each aggregate at the block head and replace the old uses.
    let first_op = block
        .deref(ctx)
        .iter(ctx)
        .next()
        .expect("verified blocks end in a terminator");
    for plan in &plans {
        let undef_op = llvm::UndefOp::new(ctx, plan.aggregate_ty);
        undef_op.get_operation().insert_before(ctx, first_op);
        let mut rebuilt = undef_op.get_operation().deref(ctx).get_result(0);
        for ((path, _), leaf_arg_idx) in plan.leaves.iter().zip(&plan.leaf_arg_indices) {
            let leaf_value = block.deref(ctx).get_argument(*leaf_arg_idx);
            let insert_op = llvm::InsertValueOp::new(ctx, rebuilt, leaf_value, path.clone());
            insert_op.get_operation().insert_before(ctx, first_op);
            rebuilt = insert_op.get_operation().deref(ctx).get_result(0);
        }
        let old_arg = block.deref(ctx).get_argument(plan.arg_idx);
        old_arg.replace_all_uses_with(ctx, &rebuilt);
    }

    // 3. Split the forwarded aggregate operand on every incoming edge. The
    //    appends must mirror step 1's order so operand and argument indices
    //    line up.
    //
    //    Extract chains are cached per (terminator, forwarded aggregate
    //    value). One terminator can feed this block through several
    //    successor slots (the importer emits a cond_br with both
    //    destinations equal when a SwitchInt case target matches the
    //    otherwise target), and the textual exporter only accepts such
    //    duplicate same-destination conditional edges when the forwarded
    //    values are identical per position, deduplicating them into a
    //    single PHI incoming. Building a fresh chain per edge would hand
    //    the two segments distinct (merely equivalent) extract results
    //    and break that invariant, so every segment that forwards the
    //    same aggregate from the same terminator must receive the very
    //    same leaf values. The chain is inserted before its terminator,
    //    so caching never crosses a dominance boundary; distinct
    //    aggregate values keep distinct chains.
    let mut chain_cache: FxHashMap<(Ptr<Operation>, Value), Vec<Value>> = FxHashMap::default();
    for (term, succ_idx) in edges {
        for plan in &plans {
            let aggregate_operand = {
                let term_obj = Operation::get_op_dyn(*term, ctx);
                let branch =
                    op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).expect("checked above");
                branch.successor_operands(ctx, *succ_idx)[plan.arg_idx]
            };
            let leaf_values = match chain_cache.get(&(*term, aggregate_operand)) {
                Some(cached) => {
                    // A hit from another plan is fine: the forwarded value's
                    // type equals both block-arg types, so the leaf expansion
                    // is identical.
                    debug_assert_eq!(cached.len(), plan.leaves.len());
                    cached.clone()
                }
                None => {
                    let mut leaf_values = Vec::with_capacity(plan.leaves.len());
                    for (path, _) in &plan.leaves {
                        let extract_op =
                            llvm::ExtractValueOp::new(ctx, aggregate_operand, path.clone())?;
                        extract_op.get_operation().insert_before(ctx, *term);
                        leaf_values.push(extract_op.get_operation().deref(ctx).get_result(0));
                    }
                    chain_cache.insert((*term, aggregate_operand), leaf_values.clone());
                    leaf_values
                }
            };
            for leaf_value in leaf_values {
                let term_obj = Operation::get_op_dyn(*term, ctx);
                let branch =
                    op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).expect("checked above");
                branch.add_successor_operand(ctx, *succ_idx, leaf_value);
            }
        }
        // Drop the old aggregate operands, highest index first so the
        // remaining planned indices stay valid.
        let term_obj = Operation::get_op_dyn(*term, ctx);
        let branch = op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).expect("checked above");
        for plan in plans.iter().rev() {
            branch.remove_successor_operand(ctx, *succ_idx, plan.arg_idx);
        }
    }

    // 4. Drop the old aggregate arguments, highest index first.
    for plan in plans.iter().rev() {
        BasicBlock::remove_argument(block, ctx, plan.arg_idx);
    }

    Ok(())
}

/// Expand an LLVM struct or array type into its scalar leaves.
///
/// Returns `None` when `ty` is not an aggregate, is an opaque struct, or
/// expands to more than [`MAX_LEAVES`] leaves.
fn aggregate_leaves(ctx: &Context, ty: TypeHandle) -> Option<Vec<(Vec<u32>, TypeHandle)>> {
    if !is_splittable_aggregate(ctx, ty) {
        return None;
    }
    let mut leaves = Vec::new();
    let mut path = Vec::new();
    if collect_leaves(ctx, ty, &mut path, &mut leaves) {
        Some(leaves)
    } else {
        None
    }
}

fn is_splittable_aggregate(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return !struct_ty.is_opaque();
    }
    ty_ref.downcast_ref::<llvm_types::ArrayType>().is_some()
}

/// Depth-first leaf expansion. Returns `false` when the leaf budget is
/// exceeded or an opaque struct makes the layout unknowable.
fn collect_leaves(
    ctx: &Context,
    ty: TypeHandle,
    path: &mut Vec<u32>,
    leaves: &mut Vec<(Vec<u32>, TypeHandle)>,
) -> bool {
    enum Children {
        Fields(Vec<TypeHandle>),
        Elements(TypeHandle, u64),
        Leaf,
        Opaque,
    }
    let children = {
        let ty_ref = ty.deref(ctx);
        if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            if struct_ty.is_opaque() {
                Children::Opaque
            } else {
                Children::Fields(struct_ty.fields().collect())
            }
        } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            Children::Elements(array_ty.elem_type(), array_ty.size())
        } else {
            Children::Leaf
        }
    };
    match children {
        Children::Opaque => false,
        Children::Leaf => {
            if leaves.len() >= MAX_LEAVES {
                return false;
            }
            leaves.push((path.clone(), ty));
            true
        }
        Children::Fields(fields) => {
            for (idx, field_ty) in fields.into_iter().enumerate() {
                path.push(idx as u32);
                let ok = collect_leaves(ctx, field_ty, path, leaves);
                path.pop();
                if !ok {
                    return false;
                }
            }
            true
        }
        Children::Elements(elem_ty, size) => {
            if size > MAX_LEAVES as u64 {
                return false;
            }
            for idx in 0..size {
                path.push(idx as u32);
                let ok = collect_leaves(ctx, elem_ty, path, leaves);
                path.pop();
                if !ok {
                    return false;
                }
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::{append_block, build_kernel, make_ctx};
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::op::Op;

    /// entry(%a: i32, %b: i64):
    ///   %agg = insertvalue (insertvalue (undef {i32, {i64, i32}}), %a, 0), %b, 1, 0
    ///   br ^merge(%agg)
    /// ^other:
    ///   br ^merge(undef)
    /// ^merge(%agg: {i32, {i64, i32}}):   // two incoming edges
    ///   extractvalue %agg, 0
    ///   return
    #[test]
    fn splits_aggregate_block_argument_into_scalar_leaves() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let inner_ty: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (vec![i64_ty, i32_ty], llvm_types::StructLayout::Unpacked),
        )
        .into();
        let agg_ty: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (vec![i32_ty, inner_ty], llvm_types::StructLayout::Unpacked),
        )
        .into();

        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty, i64_ty], vec![]);
        let scalar_a = entry.deref(&ctx).get_argument(0);
        let scalar_b = entry.deref(&ctx).get_argument(1);

        let undef_op = llvm::UndefOp::new(&mut ctx, agg_ty);
        undef_op.get_operation().insert_at_back(entry, &ctx);
        let empty = undef_op.get_operation().deref(&ctx).get_result(0);
        let insert_a = llvm::InsertValueOp::new(&mut ctx, empty, scalar_a, vec![0]);
        insert_a.get_operation().insert_at_back(entry, &ctx);
        let with_a = insert_a.get_operation().deref(&ctx).get_result(0);
        let insert_b = llvm::InsertValueOp::new(&mut ctx, with_a, scalar_b, vec![1, 0]);
        insert_b.get_operation().insert_at_back(entry, &ctx);
        let aggregate = insert_b.get_operation().deref(&ctx).get_result(0);

        let other = append_block(&mut ctx, entry, vec![]);
        let merge = append_block(&mut ctx, entry, vec![agg_ty]);
        let br_op = llvm::BrOp::new(&mut ctx, merge, vec![aggregate]);
        br_op.get_operation().insert_at_back(entry, &ctx);

        let other_undef = llvm::UndefOp::new(&mut ctx, agg_ty);
        other_undef.get_operation().insert_at_back(other, &ctx);
        let other_agg = other_undef.get_operation().deref(&ctx).get_result(0);
        let other_br = llvm::BrOp::new(&mut ctx, merge, vec![other_agg]);
        other_br.get_operation().insert_at_back(other, &ctx);

        let merge_arg = merge.deref(&ctx).get_argument(0);
        let extract_op = llvm::ExtractValueOp::new(&mut ctx, merge_arg, vec![0]).unwrap();
        extract_op.get_operation().insert_at_back(merge, &ctx);
        let return_op = llvm::ReturnOp::new(&mut ctx, None);
        return_op.get_operation().insert_at_back(merge, &ctx);

        scalarize_aggregate_block_args(&mut ctx, module_ptr).unwrap();

        // The merge block now takes the three scalar leaves in field order.
        assert_eq!(merge.deref(&ctx).get_num_arguments(), 3);
        let leaf_tys: Vec<TypeHandle> = (0..3)
            .map(|idx| merge.deref(&ctx).get_argument(idx).get_type(&ctx))
            .collect();
        assert_eq!(leaf_tys, vec![i32_ty, i64_ty, i32_ty]);

        // The branch forwards three scalar leaves split from the aggregate.
        let term_obj = Operation::get_op_dyn(br_op.get_operation(), &ctx);
        let branch = op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).unwrap();
        let forwarded = branch.successor_operands(&ctx, 0);
        assert_eq!(forwarded.len(), 3);
        assert_eq!(forwarded[0].get_type(&ctx), i32_ty);
        assert_eq!(forwarded[1].get_type(&ctx), i64_ty);
        assert_eq!(forwarded[2].get_type(&ctx), i32_ty);

        // The old aggregate use is fed by a rebuild at the block head.
        let first_op = merge.deref(&ctx).iter(&ctx).next().unwrap();
        assert!(
            Operation::get_op::<llvm::UndefOp>(first_op, &ctx).is_some(),
            "merge block should start with the reassembly chain"
        );
        let rebuild_inserts = merge
            .deref(&ctx)
            .iter(&ctx)
            .filter(|op| Operation::get_op::<llvm::InsertValueOp>(*op, &ctx).is_some())
            .count();
        assert_eq!(rebuild_inserts, 3);
    }

    /// entry(%cond: i1, %a: i32, %b: i64):
    ///   %agg = insertvalue (insertvalue (undef {i32, i64}), %a, 0), %b, 1
    ///   cond_br %cond, ^merge(%agg), ^merge(%agg)   // BOTH edges, same value
    /// ^merge(%agg: {i32, i64}):
    ///   return
    ///
    /// The importer produces this shape when a MIR SwitchInt case target
    /// equals the otherwise target. The textual exporter dedupes duplicate
    /// same-destination conditional edges into one PHI incoming only when
    /// the forwarded values are identical, so the pass must materialize ONE
    /// extract chain and append the very same leaf values to both segments.
    #[test]
    fn duplicate_conditional_edges_share_one_extract_chain() {
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let agg_ty: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (vec![i32_ty, i64_ty], llvm_types::StructLayout::Unpacked),
        )
        .into();

        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty, i32_ty, i64_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);
        let scalar_a = entry.deref(&ctx).get_argument(1);
        let scalar_b = entry.deref(&ctx).get_argument(2);

        let undef_op = llvm::UndefOp::new(&mut ctx, agg_ty);
        undef_op.get_operation().insert_at_back(entry, &ctx);
        let empty = undef_op.get_operation().deref(&ctx).get_result(0);
        let insert_a = llvm::InsertValueOp::new(&mut ctx, empty, scalar_a, vec![0]);
        insert_a.get_operation().insert_at_back(entry, &ctx);
        let with_a = insert_a.get_operation().deref(&ctx).get_result(0);
        let insert_b = llvm::InsertValueOp::new(&mut ctx, with_a, scalar_b, vec![1]);
        insert_b.get_operation().insert_at_back(entry, &ctx);
        let aggregate = insert_b.get_operation().deref(&ctx).get_result(0);

        let merge = append_block(&mut ctx, entry, vec![agg_ty]);
        let cond_br = llvm::CondBrOp::new(
            &mut ctx,
            cond,
            merge,
            vec![aggregate],
            merge,
            vec![aggregate],
        );
        cond_br.get_operation().insert_at_back(entry, &ctx);

        let return_op = llvm::ReturnOp::new(&mut ctx, None);
        return_op.get_operation().insert_at_back(merge, &ctx);

        scalarize_aggregate_block_args(&mut ctx, module_ptr).unwrap();

        // (a) The merge block now takes the scalar leaves in field order.
        assert_eq!(merge.deref(&ctx).get_num_arguments(), 2);
        assert_eq!(merge.deref(&ctx).get_argument(0).get_type(&ctx), i32_ty);
        assert_eq!(merge.deref(&ctx).get_argument(1).get_type(&ctx), i64_ty);

        // (b) Both successor segments carry the IDENTICAL leaf values (same
        // Value identity per position), preserving the exporter's dedup
        // invariant for duplicate same-destination conditional edges.
        let term_obj = Operation::get_op_dyn(cond_br.get_operation(), &ctx);
        let branch = op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).unwrap();
        let true_opds = branch.successor_operands(&ctx, 0);
        let false_opds = branch.successor_operands(&ctx, 1);
        assert_eq!(true_opds.len(), 2);
        assert_eq!(
            true_opds, false_opds,
            "duplicate edges to the same block must forward identical values"
        );

        // (c) Exactly ONE extract chain was materialized in the predecessor:
        // one extractvalue per leaf, shared by both segments.
        let extract_count = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(*op, &ctx).is_some())
            .count();
        assert_eq!(extract_count, 2);
    }

    /// Same shape end to end through `lower_mir_to_llvm`: a `mir.cond_branch`
    /// whose true and false targets are both the merge block (the importer's
    /// lowering of a SwitchInt whose case target equals the otherwise
    /// target), forwarding one `mir.construct_tuple` result on both edges.
    /// After full lowering (which ends with this pass), the resulting
    /// `llvm.cond_br` must carry identical leaf values on both segments.
    #[test]
    fn duplicate_edges_survive_full_lowering() {
        use crate::convert::ops::test_util::{append_mir_return, find_first, kernel_blocks};
        use dialect_mir::ops as mir;
        use dialect_mir::types::MirTupleType;
        use pliron::builtin::op_interfaces::OperandSegmentInterface;

        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![i32_ty, i64_ty]).into();

        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty, i32_ty, i64_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);
        let scalar_a = entry.deref(&ctx).get_argument(1);
        let scalar_b = entry.deref(&ctx).get_argument(2);

        let tuple = Operation::new(
            &mut ctx,
            mir::MirConstructTupleOp::get_concrete_op_info(),
            vec![tuple_ty],
            vec![scalar_a, scalar_b],
            vec![],
            0,
        );
        tuple.insert_at_back(entry, &ctx);
        let tuple_value = tuple.deref(&ctx).get_result(0);

        let merge = append_block(&mut ctx, entry, vec![tuple_ty]);
        append_mir_return(&mut ctx, merge, vec![]);

        let (operands, segment_sizes) = mir::MirCondBranchOp::compute_segment_sizes(vec![
            vec![cond],
            vec![tuple_value],
            vec![tuple_value],
        ]);
        let cond_br = Operation::new(
            &mut ctx,
            mir::MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![merge, merge],
            0,
        );
        mir::MirCondBranchOp::new(cond_br).set_operand_segment_sizes(&ctx, segment_sizes);
        cond_br.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let llvm_br = find_first::<llvm::CondBrOp>(&ctx, &body).expect("expected llvm.cond_br");
        let true_opds = llvm_br.successor_operands(&ctx, 0);
        let false_opds = llvm_br.successor_operands(&ctx, 1);
        assert_eq!(true_opds.len(), 2, "tuple must be split into its leaves");
        assert_eq!(
            true_opds, false_opds,
            "duplicate edges to the same block must forward identical values"
        );
        let dest = llvm_br.get_operation().deref(&ctx).get_successor(0);
        assert_eq!(dest.deref(&ctx).get_num_arguments(), 2);
        assert_eq!(dest.deref(&ctx).get_argument(0).get_type(&ctx), i32_ty);
        assert_eq!(dest.deref(&ctx).get_argument(1).get_type(&ctx), i64_ty);
        let extract_count = body
            .iter()
            .flat_map(|b| b.deref(&ctx).iter(&ctx))
            .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(*op, &ctx).is_some())
            .count();
        assert_eq!(
            extract_count, 2,
            "exactly one extract chain, one op per leaf"
        );
    }

    /// A block argument whose aggregate expands past the leaf budget must be
    /// left untouched.
    #[test]
    fn leaves_oversized_array_arguments_alone() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let arr_ty: TypeHandle = llvm_types::ArrayType::get(&ctx, i32_ty, 32).into();

        let (module_ptr, entry) = build_kernel(&mut ctx, vec![], vec![]);

        let undef_op = llvm::UndefOp::new(&mut ctx, arr_ty);
        undef_op.get_operation().insert_at_back(entry, &ctx);
        let aggregate = undef_op.get_operation().deref(&ctx).get_result(0);

        let other = append_block(&mut ctx, entry, vec![]);
        let merge = append_block(&mut ctx, entry, vec![arr_ty]);
        let br_op = llvm::BrOp::new(&mut ctx, merge, vec![aggregate]);
        br_op.get_operation().insert_at_back(entry, &ctx);

        let other_undef = llvm::UndefOp::new(&mut ctx, arr_ty);
        other_undef.get_operation().insert_at_back(other, &ctx);
        let other_agg = other_undef.get_operation().deref(&ctx).get_result(0);
        let other_br = llvm::BrOp::new(&mut ctx, merge, vec![other_agg]);
        other_br.get_operation().insert_at_back(other, &ctx);

        let return_op = llvm::ReturnOp::new(&mut ctx, None);
        return_op.get_operation().insert_at_back(merge, &ctx);

        scalarize_aggregate_block_args(&mut ctx, module_ptr).unwrap();

        assert_eq!(merge.deref(&ctx).get_num_arguments(), 1);
        assert_eq!(merge.deref(&ctx).get_argument(0).get_type(&ctx), arr_ty);
        let term_obj = Operation::get_op_dyn(br_op.get_operation(), &ctx);
        let branch = op_cast::<dyn BranchOpInterface>(term_obj.as_ref()).unwrap();
        assert_eq!(branch.successor_operands(&ctx, 0).len(), 1);
    }
}
