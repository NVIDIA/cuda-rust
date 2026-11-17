/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hoisted 2D global slice band-start provenance (`checked_raw_row_band_start` /
//! `checked_raw_col_band_start` in `sgemm_tiled_raw`) resolved over [`SsaEnv`] and [`BlockGraph`].

use std::collections::{HashMap, HashSet};

use pliron::{
    builtin::types::IntegerType, context::Context, linked_list::ContainsLinkedList, r#type::Typed,
    value::Value,
};

use super::ir_utils::{BlockGraph, bin_kinds, op_name};
use crate::ssa::{GlobalStripKind, IdxExpr, SsaEnv, SsaKind};

/// Provenance of a hoisted 2D global slice band start (`a_start` or `b_start`),
/// where valid incoming CFG paths checked the strip's last element `< slice.len()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BandStartProvenance {
    /// Sentinel `u64::MAX` (`18446744073709551615`).
    SentinelMax,
    /// Row strip start `start` satisfying `cols > 0` and `start + (cols - 1) < param.len()`.
    RowBand {
        /// Target slice parameter index.
        param: usize,
        /// Canonical SSA root of the row width `cols`.
        cols_root: Value,
    },
    /// Column strip start `start` satisfying `rows > 0` and `(rows - 1) * stride + start < param.len()`.
    ColBand {
        /// Target slice parameter index.
        param: usize,
        /// Canonical SSA root of the column height `rows`.
        rows_root: Value,
        /// Canonical SSA root of the row stride.
        stride_root: Value,
    },
}

/// Branch condition extracted on canonical integer SSA roots for 2D strip verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StripCond {
    /// Unsigned value is strictly positive (`val != 0`).
    NonZero(Value),
    /// Unsigned value equals zero (`val == 0`).
    IsZero(Value),
    /// Value equals sentinel `u64::MAX`.
    IsMax(Value),
    /// Value does not equal sentinel `u64::MAX`.
    NotMax(Value),
    /// Strict inequality `lhs < rhs`.
    Lt(Value, Value),
    /// Negated strict inequality `!(lhs < rhs)`.
    NotLt(Value, Value),
}

impl StripCond {
    /// Invert this condition across `mir.not` or the false arm of `mir.cond_br`.
    /// Preserves exact boolean equivalence on unsigned integer operands.
    fn negate(self) -> Self {
        match self {
            Self::NonZero(v) => Self::IsZero(v),
            Self::IsZero(v) => Self::NonZero(v),
            Self::IsMax(v) => Self::NotMax(v),
            Self::NotMax(v) => Self::IsMax(v),
            Self::Lt(a, b) => Self::NotLt(a, b),
            Self::NotLt(a, b) => Self::Lt(a, b),
        }
    }
}

/// Resolve 2D global slice strip guards (`RowStrip` / `ColStrip`) and propagate
/// `VerifiedGlobalStripIdx`, `IndexLtLen`, and `GlobalStripElementPtr` into `env`.
pub(super) fn populate_verified_2d_strips(ctx: &Context, graph: &BlockGraph, env: &mut SsaEnv) {
    for (val, (param, kind, offset)) in analyze_verified_2d_strips(ctx, graph, env) {
        env.bind_result(
            val,
            SsaKind::VerifiedGlobalStripIdx {
                param,
                kind,
                offset,
            },
        );
    }
    for _ in 0..graph.len() + 2 {
        let mut changed = false;
        for &b_ptr in &graph.blocks {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                let next = match (opid.as_str(), bin_kinds(env, &op)) {
                    ("mir.cast", (Some(k @ SsaKind::VerifiedGlobalStripIdx { .. }), _))
                        if op.get_num_operands() == 1 && op.get_num_results() == 1 =>
                    {
                        Some(k)
                    }
                    (
                        "mir.lt",
                        (
                            Some(SsaKind::VerifiedGlobalStripIdx {
                                param, offset: 0, ..
                            }),
                            Some(SsaKind::SliceLen(gp)),
                        ),
                    ) if op.get_num_results() == 1 && param == gp => {
                        Some(SsaKind::IndexLtLen { guard_param: gp })
                    }
                    (
                        "mir.ptr_offset",
                        (
                            Some(SsaKind::SlicePtr(param)),
                            Some(SsaKind::VerifiedGlobalStripIdx {
                                param: sp,
                                kind,
                                offset,
                            }),
                        ),
                    ) if op.get_num_results() == 1 && param == sp => {
                        Some(SsaKind::GlobalStripElementPtr {
                            param,
                            kind,
                            offset,
                        })
                    }
                    _ => None,
                };
                if let Some(k) = next
                    && env.get_kind(&op.get_result(0)).as_ref() != Some(&k)
                {
                    env.bind_result(op.get_result(0), k);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Analyze hoisted 2D band-start checks and dominating `col < a_eff` / `row < b_eff` guards.
/// Returns a map from verified index SSA values to `(param, GlobalStripKind, offset)`.
fn analyze_verified_2d_strips(
    ctx: &Context,
    graph: &BlockGraph,
    env: &SsaEnv,
) -> HashMap<Value, (usize, GlobalStripKind, u64)> {
    let mut cast_parent: HashMap<Value, Value> = HashMap::new();
    for &b_ptr in &graph.blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if op_name(ctx, op_ptr) == "mir.cast" {
                let op = op_ptr.deref(ctx);
                if op.get_num_operands() == 1 && op.get_num_results() == 1 {
                    let (src, dst) = (op.get_operand(0), op.get_result(0));
                    if let (Some(si), Some(di)) = (
                        src.get_type(ctx).deref(ctx).downcast_ref::<IntegerType>(),
                        dst.get_type(ctx).deref(ctx).downcast_ref::<IntegerType>(),
                    ) && si.is_unsigned()
                        && di.is_unsigned()
                        && si.width() <= di.width()
                    {
                        cast_parent.insert(dst, src);
                    }
                }
            }
        }
    }
    let root_of = |mut v: Value| -> Value {
        for _ in 0..16 {
            let Some(&p) = cast_parent.get(&v) else { break };
            v = p;
        }
        v
    };

    let mut const_vals: HashMap<Value, u128> = HashMap::new();
    let mut slice_lens: HashMap<Value, usize> = HashMap::new();
    let mut add_ops: HashMap<Value, (Value, Value)> = HashMap::new();
    let mut mul_ops: HashMap<Value, (Value, Value)> = HashMap::new();
    let mut sub_ops: Vec<(Value, Value, Value)> = Vec::new();
    let mut tx_derived: HashSet<Value> = HashSet::new();
    let mut ty_derived: HashSet<Value> = HashSet::new();

    for &b_ptr in &graph.blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            if op.get_num_results() == 1 {
                let res = op.get_result(0);
                let rr = root_of(res);
                match env.get_kind(&res) {
                    Some(SsaKind::ConstInt { value, .. }) => {
                        const_vals.insert(rr, value);
                    }
                    Some(SsaKind::SliceLen(param)) => {
                        slice_lens.insert(rr, param);
                    }
                    Some(SsaKind::Index(IdxExpr::Tx)) => {
                        tx_derived.insert(rr);
                    }
                    Some(SsaKind::Index(IdxExpr::Ty)) => {
                        ty_derived.insert(rr);
                    }
                    _ => {}
                }
                if op.get_num_operands() == 2 {
                    let (a, b) = (root_of(op.get_operand(0)), root_of(op.get_operand(1)));
                    match opid.as_str() {
                        "mir.add" => {
                            add_ops.insert(rr, (a, b));
                        }
                        "mir.mul" => {
                            mul_ops.insert(rr, (a, b));
                        }
                        "mir.sub" => sub_ops.push((rr, a, b)),
                        _ => {}
                    }
                }
            }
        }
    }
    let sub_one_of: HashMap<Value, Value> = sub_ops
        .into_iter()
        .filter_map(|(rr, a, b)| (const_vals.get(&b) == Some(&1)).then_some((rr, a)))
        .collect();

    for _ in 0..8 {
        let mut changed = false;
        for (&res, &(a, b)) in &add_ops {
            if (tx_derived.contains(&a) || tx_derived.contains(&b))
                && !ty_derived.contains(&a)
                && !ty_derived.contains(&b)
                && tx_derived.insert(res)
            {
                changed = true;
            }
            if (ty_derived.contains(&a) || ty_derived.contains(&b))
                && !tx_derived.contains(&a)
                && !tx_derived.contains(&b)
                && ty_derived.insert(res)
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut cond_map: HashMap<Value, StripCond> = HashMap::new();
    for _ in 0..4 {
        for &b_ptr in &graph.blocks {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                let opid = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 {
                    continue;
                }
                let res = op.get_result(0);
                match opid.as_str() {
                    "mir.eq" | "mir.ne" if op.get_num_operands() == 2 => {
                        let (a, b, is_eq) = (
                            root_of(op.get_operand(0)),
                            root_of(op.get_operand(1)),
                            opid == "mir.eq",
                        );
                        for (val_r, cst_r) in [(a, b), (b, a)] {
                            match const_vals.get(&cst_r).copied() {
                                Some(0) => {
                                    cond_map.insert(
                                        res,
                                        if is_eq {
                                            StripCond::IsZero(val_r)
                                        } else {
                                            StripCond::NonZero(val_r)
                                        },
                                    );
                                }
                                Some(c) if c == u128::from(u64::MAX) => {
                                    cond_map.insert(
                                        res,
                                        if is_eq {
                                            StripCond::IsMax(val_r)
                                        } else {
                                            StripCond::NotMax(val_r)
                                        },
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                    "mir.gt" if op.get_num_operands() == 2 => {
                        let (a, b) = (root_of(op.get_operand(0)), root_of(op.get_operand(1)));
                        if const_vals.get(&b) == Some(&0) {
                            cond_map.insert(res, StripCond::NonZero(a));
                        }
                    }
                    "mir.lt" if op.get_num_operands() == 2 => {
                        let (a, b) = (root_of(op.get_operand(0)), root_of(op.get_operand(1)));
                        let cond = if const_vals.get(&a) == Some(&0) {
                            StripCond::NonZero(b)
                        } else {
                            StripCond::Lt(a, b)
                        };
                        cond_map.insert(res, cond);
                    }
                    "mir.not" if op.get_num_operands() == 1 => {
                        if let Some(&inner) = cond_map.get(&op.get_operand(0)) {
                            cond_map.insert(res, inner.negate());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut branch_preds: Vec<(usize, usize, usize, StripCond)> = Vec::new();
    let mut goto_edges: Vec<(usize, usize, Value)> = Vec::new();
    for (src_blk, &b_ptr) in graph.blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            if opid == "mir.cond_br"
                && op.get_num_operands() >= 1
                && op.get_num_successors() == 2
                && let Some(&cond) = cond_map.get(&op.get_operand(0))
                && let (Some(t_idx), Some(f_idx)) = (
                    graph.idx_of(op.get_successor(0)),
                    graph.idx_of(op.get_successor(1)),
                )
            {
                branch_preds.push((src_blk, t_idx, f_idx, cond));
                branch_preds.push((src_blk, f_idx, t_idx, cond.negate()));
            } else if opid == "mir.goto"
                && op.get_num_operands() == 1
                && op.get_num_successors() == 1
                && let Some(dst_idx) = graph.idx_of(op.get_successor(0))
            {
                goto_edges.push((src_blk, dst_idx, op.get_operand(0)));
            }
        }
    }

    let dominates_pred = |b_idx: usize, pred: StripCond| -> bool {
        branch_preds.iter().any(|&(src_blk, in_dst, out_dst, p)| {
            p == pred
                && !graph.can_reach_without(0, b_idx, in_dst)
                && !graph.can_reach_without(out_dst, b_idx, src_blk)
        })
    };

    let mut band_vars: HashMap<Value, BandStartProvenance> = HashMap::new();
    for _ in 0..graph.len() + 2 {
        let mut changed = false;
        for (dst_blk, &b_ptr) in graph.blocks.iter().enumerate().skip(1) {
            let blk = b_ptr.deref(ctx);
            if blk.get_num_arguments() != 1 {
                continue;
            }
            let arg_root = root_of(blk.get_argument(0));
            let incoming: Vec<_> = goto_edges
                .iter()
                .filter(|&&(_, d, _)| d == dst_blk)
                .collect();
            if incoming.is_empty() {
                continue;
            }
            let mut merged: Option<BandStartProvenance> = None;
            let mut all_ok = true;
            for &&(src_blk, _, in_val) in &incoming {
                let in_r = root_of(in_val);
                let prov = if const_vals.get(&in_r) == Some(&u128::from(u64::MAX)) {
                    Some(BandStartProvenance::SentinelMax)
                } else if let Some(&existing) = band_vars.get(&in_r) {
                    Some(existing)
                } else {
                    branch_preds.iter().find_map(|&(_, _, _, pred)| {
                        let StripCond::Lt(last_r, len_r) = pred else {
                            return None;
                        };
                        let &param = slice_lens.get(&len_r)?;
                        if !dominates_pred(src_blk, pred) {
                            return None;
                        }
                        let &(x, y) = add_ops.get(&last_r)?;
                        for (start_cand, rem_cand) in [(x, y), (y, x)] {
                            if start_cand != in_r {
                                continue;
                            }
                            if let Some(&cols_root) = sub_one_of.get(&rem_cand)
                                && dominates_pred(src_blk, StripCond::NonZero(cols_root))
                            {
                                return Some(BandStartProvenance::RowBand { param, cols_root });
                            }
                            if let Some(&(m1, m2)) = mul_ops.get(&rem_cand) {
                                for (sub_cand, stride_root) in [(m1, m2), (m2, m1)] {
                                    if let Some(&rows_root) = sub_one_of.get(&sub_cand)
                                        && dominates_pred(src_blk, StripCond::NonZero(rows_root))
                                    {
                                        return Some(BandStartProvenance::ColBand {
                                            param,
                                            rows_root,
                                            stride_root,
                                        });
                                    }
                                }
                            }
                        }
                        None
                    })
                };
                match prov {
                    Some(BandStartProvenance::SentinelMax) => {}
                    Some(concrete) if merged.is_none() || merged == Some(concrete) => {
                        merged = Some(concrete);
                    }
                    _ => {
                        all_ok = false;
                        break;
                    }
                }
            }
            if all_ok
                && let Some(valid_prov) = merged
                && band_vars.insert(arg_root, valid_prov) != Some(valid_prov)
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut band_effs: HashMap<Value, Value> = HashMap::new();
    for (dst_blk, &b_ptr) in graph.blocks.iter().enumerate().skip(1) {
        let blk = b_ptr.deref(ctx);
        if blk.get_num_arguments() != 1 {
            continue;
        }
        let eff_root = root_of(blk.get_argument(0));
        let incoming: Vec<_> = goto_edges
            .iter()
            .filter(|&&(_, d, _)| d == dst_blk)
            .collect();
        if incoming.is_empty() {
            continue;
        }
        for (&band_root, &prov) in &band_vars {
            let extent_root = match prov {
                BandStartProvenance::RowBand { cols_root, .. } => cols_root,
                BandStartProvenance::ColBand { rows_root, .. } => rows_root,
                BandStartProvenance::SentinelMax => continue,
            };
            let mut saw_extent = false;
            let all_valid = incoming.iter().all(|&&(src_blk, _, op0)| {
                let r = root_of(op0);
                if const_vals.get(&r) == Some(&0) {
                    true
                } else if r == extent_root && dominates_pred(src_blk, StripCond::NotMax(band_root))
                {
                    saw_extent = true;
                    true
                } else {
                    false
                }
            });
            if all_valid && saw_extent {
                band_effs.insert(eff_root, band_root);
                break;
            }
        }
    }

    let step_is_guarded = |b_idx: usize, step_r: Value, band_r: Value, ext_r: Value| -> bool {
        band_effs.iter().any(|(&eff_r, &b_r)| {
            b_r == band_r && dominates_pred(b_idx, StripCond::Lt(step_r, eff_r))
        }) || (dominates_pred(b_idx, StripCond::NotMax(band_r))
            && dominates_pred(b_idx, StripCond::Lt(step_r, ext_r)))
    };

    let peel_const_add = |r: Value| -> Vec<(Value, u64)> {
        let mut out = vec![(r, 0u64)];
        if let Some(&(a, b)) = add_ops.get(&r) {
            for (base, cst) in [(a, b), (b, a)] {
                if let Some(&c) = const_vals.get(&cst)
                    && let Ok(off) = u64::try_from(c)
                {
                    out.push((base, off));
                }
            }
        }
        out
    };

    let mut verified_indices: HashMap<Value, (usize, GlobalStripKind, u64)> = HashMap::new();
    for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            for res in op_ptr.deref(ctx).results() {
                let idx_root = root_of(res);
                let matched =
                    peel_const_add(idx_root)
                        .into_iter()
                        .find_map(|(core_r, outer_off)| {
                            let &(lhs, rhs) = add_ops.get(&core_r)?;
                            for (band_cand, step_part) in [(lhs, rhs), (rhs, lhs)] {
                                match band_vars.get(&band_cand)? {
                                    &BandStartProvenance::RowBand { param, cols_root } => {
                                        for (step_r, inner_off) in peel_const_add(step_part) {
                                            if tx_derived.contains(&step_r)
                                                && step_is_guarded(
                                                    b_idx, step_r, band_cand, cols_root,
                                                )
                                            {
                                                return Some((
                                                    param,
                                                    GlobalStripKind::RowStrip,
                                                    outer_off.saturating_add(inner_off),
                                                ));
                                            }
                                        }
                                    }
                                    &BandStartProvenance::ColBand {
                                        param,
                                        rows_root,
                                        stride_root,
                                    } => {
                                        if let Some(&(m1, m2)) = mul_ops.get(&step_part) {
                                            for (row_cand, st_cand) in [(m1, m2), (m2, m1)] {
                                                if st_cand == stride_root
                                                    && ty_derived.contains(&row_cand)
                                                    && step_is_guarded(
                                                        b_idx, row_cand, band_cand, rows_root,
                                                    )
                                                {
                                                    return Some((
                                                        param,
                                                        GlobalStripKind::ColStrip,
                                                        outer_off,
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                    BandStartProvenance::SentinelMax => {}
                                }
                            }
                            None
                        });
                if let Some(info) = matched {
                    verified_indices.insert(res, info);
                }
            }
        }
    }
    verified_indices
}
