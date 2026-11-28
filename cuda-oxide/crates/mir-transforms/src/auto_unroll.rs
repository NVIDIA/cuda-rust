/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Automatic full unrolling of loops that index a small local array with their
//! counter.
//!
//! A loop such as
//!
//! ```text
//!   let mut acc = [[0f32; 4]; 8];
//!   for n in 0..8 { acc[n] = mma(acc[n], a, b[n]); }
//! ```
//!
//! keeps `acc` in memory. mem2reg cannot promote an array indexed by a runtime
//! value, so `acc` stays a `mir.alloca` reached through
//! `mir.array_element_addr acc, n`. Unless LLVM's cost model happens to unroll
//! the loop (at `-O2` it often does not), the array becomes a `.local` stack
//! slot, and every access goes to memory (NVIDIA/cuda-rust#1815). Unrolling the
//! loop fully makes every index a constant, and the array becomes registers.
//!
//! [`plant_auto_unroll_hints`] finds such loops and plants a full-unroll
//! `mir.unroll_hint` in each, so the unroll pass handles them with the same
//! machinery as `#[unroll]`. A loop gets a hint when:
//!
//! - it contains no other loop and carries neither `#[unroll]` nor
//!   `#[nounroll]`;
//! - one of its blocks takes `mir.array_element_addr` of a function-local
//!   `mir.alloca` (directly or through further element and field address
//!   steps) with an index computed inside the loop, and the alloca holds at
//!   most [`AUTO_UNROLL_MAX_ARRAY_BYTES`] bytes of integers and floats.
//!
//! The unroll pass decides after normalizing the loop (`check`). It unrolls
//! only when:
//!
//! - the trip count is a compile-time constant of at most
//!   [`AUTO_UNROLL_MAX_COPIES`];
//! - such an access's index is computed from the counter and constants alone,
//!   so it folds to a constant in every copy;
//! - the copies stay within [`AUTO_UNROLL_MAX_OPS`] cloned operations.
//!
//! An automatic hint that fails any check is dropped without a warning, because
//! nobody asked for it. `#[nounroll]` opts a loop out, and setting
//! `CUDA_OXIDE_NO_AUTO_UNROLL` turns the rule off for a whole build.
//!
//! LLVM's AMDGPU backend has the same rule: `AMDGPUTTIImpl::getUnrollingPreferences`
//! raises the unroll threshold for a loop that indexes a small private
//! `alloca` with a loop-varying value. NVPTX has no equivalent.

use dialect_mir::ops::aggregate::{MirArrayElementAddrOp, MirExtractFieldOp, MirFieldAddrOp};
use dialect_mir::ops::arithmetic::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCheckedAddOp, MirCheckedMulOp,
    MirCheckedSubOp, MirDivOp, MirMulOp, MirRemOp, MirShlOp, MirShrOp, MirSubOp,
};
use dialect_mir::ops::cast::MirCastOp;
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::{MirNoUnrollHintOp, MirUnrollHintOp};
use dialect_mir::ops::memory::MirAllocaOp;
use dialect_mir::types::{MirArrayType, MirFP16Type};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::result::Result;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use rustc_hash::FxHashSet;

use crate::analyses::induction::LoopRecurrences;
use crate::analyses::loop_info::LoopInfo;
use crate::unroll::collect_functions;

/// Largest local array, in bytes, whose loop is unrolled automatically.
///
/// This is half of the 255 32-bit registers a thread can address, so the
/// promoted array leaves room for everything else the kernel keeps live. It is
/// large enough for typical per-thread `mma.sync` accumulator tiles (8 × 4
/// `f32` = 128 bytes, 16 × 8 = 512 bytes).
pub const AUTO_UNROLL_MAX_ARRAY_BYTES: u64 = 512;

pub const AUTO_UNROLL_MAX_COPIES: u64 = 64;

/// Most operations an automatic unroll may clone in total (copies × body).
///
/// This is far below the explicit `#[unroll]` limits: an annotation states that
/// the author wants the growth, an automatic unroll only buys registers.
pub const AUTO_UNROLL_MAX_OPS: u64 = 4_096;

pub const NO_AUTO_UNROLL_ENV: &str = "CUDA_OXIDE_NO_AUTO_UNROLL";

/// Plant a full-unroll hint in every loop that qualifies (see the module
/// documentation) and return the planted hint operations, so the unroll pass
/// can tell them from `#[unroll]` annotations.
pub fn plant_auto_unroll_hints(
    module: Ptr<Operation>,
    ctx: &mut Context,
) -> Result<FxHashSet<Ptr<Operation>>> {
    let mut planted = FxHashSet::default();
    if std::env::var_os(NO_AUTO_UNROLL_ENV).is_some() {
        return Ok(planted);
    }
    for func_op in collect_functions(module, ctx) {
        let region = func_op.deref(ctx).get_region(0);
        let info = {
            let mut analyses = AnalysisManager::default();
            let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
            let dom = dom_info.get_dom_tree(ctx, region);
            LoopInfo::compute(ctx, region, dom)
        };
        let mut sites = Vec::new();
        for lp in info.loops() {
            if !lp.children.is_empty() || has_unroll_annotation(ctx, &lp.blocks) {
                continue;
            }
            if let Some(site) = find_candidate_access(ctx, &lp.blocks, |index| {
                index
                    .get_parent_block(ctx)
                    .is_some_and(|block| lp.blocks.contains(&block))
            }) {
                sites.push(site);
            }
        }
        for site in sites {
            let hint = MirUnrollHintOp::new(ctx, 0).get_operation();
            hint.insert_before(ctx, site);
            planted.insert(hint);
        }
    }
    Ok(planted)
}

/// The unroll pass's final check for an automatic hint, after the loop has been
/// normalized and its counter analyzed. `Err` says why the loop stays rolled;
/// the pass reports it only in verbose builds.
pub(crate) fn check(
    ctx: &Context,
    info: &LoopInfo,
    loop_id: usize,
    rec: &LoopRecurrences,
) -> std::result::Result<(), String> {
    let lp = &info.loops()[loop_id];
    let trip = rec
        .trip_count
        .ok_or("the trip count is not a compile-time constant")?;
    if trip > AUTO_UNROLL_MAX_COPIES {
        return Err(format!(
            "{trip} iterations exceed the automatic limit of {AUTO_UNROLL_MAX_COPIES}"
        ));
    }
    let iv_index = rec.primary_iv.ok_or("no loop counter was found")?;
    let iv = lp.header.deref(ctx).get_argument(iv_index);
    if find_candidate_access(ctx, &lp.blocks, |index| {
        folds_per_copy(ctx, index, iv, &lp.blocks, 0)
    })
    .is_none()
    {
        return Err("no local-array index is computed from the counter and constants alone".into());
    }
    let mut ops_per_copy = 0u64;
    for &block in &lp.blocks {
        ops_per_copy += block.deref(ctx).iter(ctx).count() as u64;
    }
    let total = ops_per_copy.saturating_mul(trip);
    if total > AUTO_UNROLL_MAX_OPS {
        return Err(format!(
            "unrolling would clone {total} operations; the automatic limit is {AUTO_UNROLL_MAX_OPS}"
        ));
    }
    Ok(())
}

fn has_unroll_annotation(ctx: &Context, blocks: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    blocks.iter().any(|block| {
        block.deref(ctx).iter(ctx).any(|op| {
            Operation::get_op::<MirUnrollHintOp>(op, ctx).is_some()
                || Operation::get_op::<MirNoUnrollHintOp>(op, ctx).is_some()
        })
    })
}

/// The first `mir.array_element_addr` in `blocks` that indexes a small local
/// array with an index accepted by `index_ok`.
fn find_candidate_access(
    ctx: &Context,
    blocks: &FxHashSet<Ptr<BasicBlock>>,
    index_ok: impl Fn(Value) -> bool,
) -> Option<Ptr<Operation>> {
    // Visit blocks in region order so the result does not depend on hashing.
    let block = *blocks.iter().next()?;
    let region = block.deref(ctx).get_parent_region()?;
    for block in region.deref(ctx).iter(ctx) {
        if !blocks.contains(&block) {
            continue;
        }
        for op in block.deref(ctx).iter(ctx) {
            if Operation::get_op::<MirArrayElementAddrOp>(op, ctx).is_none() {
                continue;
            }
            let base = op.deref(ctx).get_operand(0);
            let index = op.deref(ctx).get_operand(1);
            if index_ok(index) && small_local_array(ctx, base) {
                return Some(op);
            }
        }
    }
    None
}

/// Whether `ptr` points into a function-local `mir.alloca` of at most
/// [`AUTO_UNROLL_MAX_ARRAY_BYTES`] bytes, through element and field address
/// steps only.
fn small_local_array(ctx: &Context, ptr: Value) -> bool {
    let mut ptr = ptr;
    for _ in 0..8 {
        let Some(def) = ptr.defining_op() else {
            return false;
        };
        if let Some(alloca) = Operation::get_op::<MirAllocaOp>(def, ctx) {
            return byte_size(ctx, alloca.pointee_type(ctx), 0)
                .is_some_and(|bytes| bytes <= AUTO_UNROLL_MAX_ARRAY_BYTES);
        }
        if Operation::get_op::<MirArrayElementAddrOp>(def, ctx).is_none()
            && Operation::get_op::<MirFieldAddrOp>(def, ctx).is_none()
        {
            return false;
        }
        ptr = def.deref(ctx).get_operand(0);
    }
    false
}

/// The size of an array of integers and floats, or of such a scalar; `None`
/// for any other type, which keeps the rule to plain register arrays.
fn byte_size(ctx: &Context, ty: TypeHandle, depth: u32) -> Option<u64> {
    if depth > 8 {
        return None;
    }
    let ty_ref = ty.deref(ctx);
    if let Some(array) = ty_ref.downcast_ref::<MirArrayType>() {
        let (element, len) = (array.element_type(), array.size());
        drop(ty_ref);
        return byte_size(ctx, element, depth + 1)?.checked_mul(len);
    }
    if let Some(int) = ty_ref.downcast_ref::<IntegerType>() {
        return Some(u64::from(int.width()).div_ceil(8));
    }
    if ty_ref.downcast_ref::<FP32Type>().is_some() {
        return Some(4);
    }
    if ty_ref.downcast_ref::<FP64Type>().is_some() {
        return Some(8);
    }
    if ty_ref.downcast_ref::<MirFP16Type>().is_some() {
        return Some(2);
    }
    None
}

/// Whether `v` is computed from the counter `iv` and constants alone, through
/// integer arithmetic, casts and field extraction (of checked arithmetic's
/// result pairs) inside the loop. In a full unroll the counter
/// is a literal in every copy, so such a value folds to a constant there.
fn folds_per_copy(
    ctx: &Context,
    v: Value,
    iv: Value,
    blocks: &FxHashSet<Ptr<BasicBlock>>,
    depth: u32,
) -> bool {
    if v == iv {
        return true;
    }
    if depth > 16 {
        return false;
    }
    let Some(def) = v.defining_op() else {
        return false;
    };
    if Operation::get_op::<MirConstantOp>(def, ctx).is_some() {
        return true;
    }
    let in_loop = def
        .deref(ctx)
        .get_parent_block()
        .is_some_and(|block| blocks.contains(&block));
    let foldable = Operation::get_op::<MirAddOp>(def, ctx).is_some()
        || Operation::get_op::<MirSubOp>(def, ctx).is_some()
        || Operation::get_op::<MirMulOp>(def, ctx).is_some()
        || Operation::get_op::<MirDivOp>(def, ctx).is_some()
        || Operation::get_op::<MirRemOp>(def, ctx).is_some()
        || Operation::get_op::<MirCheckedAddOp>(def, ctx).is_some()
        || Operation::get_op::<MirCheckedSubOp>(def, ctx).is_some()
        || Operation::get_op::<MirCheckedMulOp>(def, ctx).is_some()
        || Operation::get_op::<MirShlOp>(def, ctx).is_some()
        || Operation::get_op::<MirShrOp>(def, ctx).is_some()
        || Operation::get_op::<MirBitAndOp>(def, ctx).is_some()
        || Operation::get_op::<MirBitOrOp>(def, ctx).is_some()
        || Operation::get_op::<MirBitXorOp>(def, ctx).is_some()
        || Operation::get_op::<MirCastOp>(def, ctx).is_some()
        || Operation::get_op::<MirExtractFieldOp>(def, ctx).is_some();
    if !in_loop || !foldable {
        return false;
    }
    let operands: Vec<Value> = def.deref(ctx).operands().collect();
    operands
        .into_iter()
        .all(|operand| folds_per_copy(ctx, operand, iv, blocks, depth + 1))
}
