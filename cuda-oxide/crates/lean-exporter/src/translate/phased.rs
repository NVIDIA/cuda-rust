/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Top-level phased kernel driver (`translate_phased_kernel`) orchestrating
//! `BlockGraph` -> `ssa_prop` -> `strips_2d` -> `mir.assert` pairing -> `tcgen05`
//! into [`TranslatedFunction::TiledKernel2D`] and [`TranscribedControlFlowCfg`].

use std::collections::{BTreeSet, HashMap};

use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    value::Value,
};

use super::{
    AccessKind, GlobalStripAccess2D, PhasedGlobal1DAccess, TranslatedFunction,
    ir_utils::{
        BlockGraph, bin_kinds, int_attr_u64, is_mbarrier_shared_alloc, op_kind, op_name,
        push_shared_access, shared_alloc_elem_bytes,
    },
    ssa_prop::propagate_phased_ssa,
    strips_2d::populate_verified_2d_strips,
    tcgen05::{AsyncTranscribeCtx, transcribe_async_or_tcgen05_op},
};
use crate::{
    ExportError, reject,
    spec::{KernelVerificationSpec, get_string_attr},
    ssa::{
        CfgBranchCond, CfgOpItem, GuardExpr, SsaEnv, SsaKind, TileAccessPatternKind,
        TranscribedCfgBlock, TranscribedControlFlowCfg,
    },
};

/// Pair each in-block `mir.assert(mir.lt(idx, size))` check with the verified
/// `SharedElementPtr` produced by `mir.ptr_offset(shared_base, idx)`.
fn pair_shared_ptr_asserts(
    ctx: &Context,
    graph: &BlockGraph,
    spec: &KernelVerificationSpec,
    env: &SsaEnv,
) -> HashMap<Value, Ptr<Operation>> {
    let mut lt_const_bounds: HashMap<Value, (Value, u64)> = HashMap::new();
    for &b_ptr in &graph.blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            if op_name(ctx, op_ptr) == "mir.lt" {
                let op = op_ptr.deref(ctx);
                if op.get_num_operands() == 2
                    && op.get_num_results() == 1
                    && matches!(
                        op_kind(env, &op, 0),
                        Some(SsaKind::Index(_) | SsaKind::ConstInt { .. })
                    )
                    && let Some(SsaKind::ConstInt { value, .. }) = op_kind(env, &op, 1)
                    && let Ok(bound) = u64::try_from(value)
                {
                    lt_const_bounds.insert(op.get_result(0), (op.get_operand(0), bound));
                }
            }
        }
    }

    let mut ptr_asserts: HashMap<Value, Ptr<Operation>> = HashMap::new();
    for &b_ptr in &graph.blocks {
        let mut pending: Vec<(Ptr<Operation>, Value, u64)> = Vec::new();
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            match opid.as_str() {
                "mir.assert" if op.get_num_operands() == 1 => {
                    if let Some(&(idx_v, bnd)) = lt_const_bounds.get(&op.get_operand(0)) {
                        pending.push((op_ptr, idx_v, bnd));
                    }
                }
                "mir.array_element_addr" if op.get_num_operands() >= 2 => {
                    let idx_v = op.get_operand(1);
                    if let Some(pos) = pending.iter().rposition(|&(_, p, _)| p == idx_v) {
                        pending.remove(pos);
                    }
                }
                "mir.ptr_offset" if op.get_num_operands() >= 2 && op.get_num_results() == 1 => {
                    let idx_v = op.get_operand(1);
                    if let Some(SsaKind::SharedBasePtr {
                        shared_id, size, ..
                    }) = op_kind(env, &op, 0)
                    {
                        if let Some(pos) = pending
                            .iter()
                            .rposition(|&(_, p, b)| p == idx_v && b == size)
                        {
                            let (aop, _, _) = pending.remove(pos);
                            if spec.no_oob_shared.contains(&shared_id) {
                                ptr_asserts.insert(op.get_result(0), aop);
                            }
                        }
                    } else if let Some(pos) = pending.iter().rposition(|&(_, p, _)| p == idx_v) {
                        pending.remove(pos);
                    }
                }
                _ => {}
            }
        }
    }

    for _ in 0..graph.len() + 2 {
        let mut changed = false;
        for &b_ptr in &graph.blocks {
            for op_ptr in b_ptr.deref(ctx).iter(ctx) {
                if op_name(ctx, op_ptr) == "mir.cast" {
                    let op = op_ptr.deref(ctx);
                    if op.get_num_operands() == 1
                        && op.get_num_results() == 1
                        && let Some(&aop) = ptr_asserts.get(&op.get_operand(0))
                        && !ptr_asserts.contains_key(&op.get_result(0))
                    {
                        ptr_asserts.insert(op.get_result(0), aop);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    ptr_asserts
}

/// Transcribe any phased shared-memory, barrier, 2D tiled, `mbarrier`/TMA, or
/// Blackwell `tcgen05`/`TMEM` kernel into [`TranslatedFunction::TiledKernel2D`].
pub(super) fn translate_phased_kernel(
    ctx: &Context,
    function: Ptr<Operation>,
    arity: usize,
    spec: KernelVerificationSpec,
) -> Result<TranslatedFunction, ExportError> {
    let blocks: Vec<Ptr<BasicBlock>> = function
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .collect();
    if blocks.is_empty() {
        return Err(reject("kernel", "empty kernel region"));
    }
    let graph = BlockGraph::build(ctx, blocks);
    let num_blocks = graph.len();

    // 1. Discover static `SharedArray`, `Barrier` (`mbarrier`), and `tcgen05_ld_*` sites.
    let mut shared_map: HashMap<String, u64> = HashMap::new();
    let mut shared_elem_bytes_map: HashMap<String, u64> = HashMap::new();
    let mut bar_raw_list: Vec<String> = Vec::new();
    let mut shared_raw_list: Vec<(String, u64)> = Vec::new();
    let mut tmem_ld_id_map: HashMap<(usize, usize), usize> = HashMap::new();
    for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
        for (op_idx, op_ptr) in b_ptr.deref(ctx).iter(ctx).enumerate() {
            let opid = op_name(ctx, op_ptr);
            if opid.starts_with("nvvm.tcgen05_ld_") {
                let next_id = tmem_ld_id_map.len();
                tmem_ld_id_map.insert((b_idx, op_idx), next_id);
            } else if opid == "mir.shared_alloc" {
                let op = op_ptr.deref(ctx);
                let full = get_string_attr(&op.attributes, "source_name")
                    .unwrap_or_else(|| "SHARED".into());
                let short = full.rsplit("::").next().unwrap_or(&full).to_string();
                if is_mbarrier_shared_alloc(ctx, &op) {
                    if !bar_raw_list.contains(&short) {
                        bar_raw_list.push(short);
                    }
                } else {
                    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                    let elem_bytes = shared_alloc_elem_bytes(ctx, &op);
                    if !shared_map.contains_key(&short) {
                        shared_map.insert(short.clone(), size);
                        shared_elem_bytes_map.insert(short.clone(), elem_bytes);
                        shared_raw_list.push((short, size));
                    }
                }
            }
        }
    }
    shared_raw_list.sort_by(|a, b| a.0.cmp(&b.0));
    bar_raw_list.sort();

    // 2. Partition acyclic vs. cyclic `nvvm.barrier0` sites to label shared-access phase numbers
    //    (all barrier-uniformity and lifecycle proofs run in Lean 4 on `control_flow_cfg`).
    let mut acyclic_barriers = Vec::new();
    let mut cyclic_barriers = Vec::new();
    for &(b_idx, ord) in &graph.barrier_sites {
        if graph.is_cyclic(b_idx) {
            cyclic_barriers.push((b_idx, ord));
        } else {
            acyclic_barriers.push((b_idx, ord));
        }
    }
    let enclosing_loop_header: Option<usize> =
        cyclic_barriers.first().copied().and_then(|(sample, _)| {
            let cands: Vec<usize> = (0..num_blocks)
                .filter(|&h| {
                    graph.can_reach(h, sample)
                        && graph.can_reach(sample, h)
                        && !graph.can_reach_without(0, sample, h)
                })
                .collect();
            let hdr = cands.iter().copied().find(|&h| {
                cands
                    .iter()
                    .all(|&o| o == h || !graph.can_reach_without(0, o, h))
            })?;
            cyclic_barriers.sort_by(|&(ba, oa), &(bb, ob)| {
                if ba == bb {
                    return oa.cmp(&ob);
                }
                let a_before_b = graph.succs[hdr]
                    .iter()
                    .filter(|&&s| graph.can_reach(s, hdr))
                    .all(|&s| !graph.can_reach_avoiding_two(s, bb, ba, hdr));
                if a_before_b {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });
            Some(hdr)
        });
    let acyclic_dom_count = |target_blk: usize, seen_in_blk: usize| -> usize {
        let strict = acyclic_barriers
            .iter()
            .filter(|&&(ab, _)| {
                ab != target_blk
                    && graph.can_reach(ab, target_blk)
                    && !graph.can_reach_without(0, target_blk, ab)
            })
            .count();
        strict
            + if !graph.is_cyclic(target_blk) {
                seen_in_blk
            } else {
                0
            }
    };
    let block_phase = |b_idx: usize, seen_in_blk: usize| -> usize {
        if graph.barrier_sites.is_empty() {
            return 0;
        }
        if let Some(hdr) = enclosing_loop_header {
            let base = acyclic_dom_count(hdr, 0);
            let k = cyclic_barriers.len();
            if b_idx == hdr || (graph.can_reach(hdr, b_idx) && graph.can_reach(b_idx, hdr)) {
                if k <= 1 || b_idx == hdr {
                    return base;
                }
                let starts: Vec<usize> = graph.succs[hdr]
                    .iter()
                    .copied()
                    .filter(|&s| graph.can_reach(s, hdr))
                    .collect();
                let passed = cyclic_barriers
                    .iter()
                    .filter(|&&(bb, bo)| {
                        if bb == b_idx {
                            bo < seen_in_blk
                        } else {
                            !starts.is_empty()
                                && starts.iter().all(|&s| {
                                    graph.can_reach_without(s, b_idx, hdr)
                                        && !graph.can_reach_avoiding_two(s, b_idx, bb, hdr)
                                })
                        }
                    })
                    .count();
                return base + (passed % k);
            }
            if graph.can_reach(hdr, b_idx) {
                return acyclic_dom_count(b_idx, seen_in_blk) + k;
            }
        }
        acyclic_dom_count(b_idx, seen_in_blk)
    };

    // 3. Propagate `SsaEnv` + optional 2D global slice strips (`sgemm_tiled_raw`).
    let mut env = propagate_phased_ssa(
        ctx,
        &graph,
        &spec,
        &bar_raw_list,
        &shared_raw_list,
        &shared_elem_bytes_map,
        &tmem_ld_id_map,
    );
    if !spec.no_oob_idx.is_empty() {
        populate_verified_2d_strips(ctx, &graph, &mut env);
    }

    // 4. Transcribe block terminators into `cfg_blocks` once, and collect `mbarrier` / TMA operand facts.
    let mut cfg_blocks: Vec<TranscribedCfgBlock> = vec![TranscribedCfgBlock::default(); num_blocks];
    let mut branch_guards: Vec<(usize, usize, GuardExpr)> = Vec::new();
    let mut global_1d_guards: Vec<(usize, usize, usize, u64)> = Vec::new();
    let mut arrived_expect_tx_bars: BTreeSet<usize> = BTreeSet::new();
    let mut committed_tmem_bars: BTreeSet<usize> = BTreeSet::new();
    let mut arrived_tx_bytes_per_bar: HashMap<usize, u64> = HashMap::new();
    let mut async_bar_for_shared: HashMap<usize, usize> = HashMap::new();
    let mut mbarrier_wait_edges: Vec<(usize, usize, usize)> = Vec::new();
    for (w_blk, &b_ptr) in graph.blocks.iter().enumerate() {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            match opid.as_str() {
                "mir.return" => cfg_blocks[w_blk].is_return = true,
                "mir.goto" if op.get_num_successors() == 1 => {
                    if let Some(dst) = graph.idx_of(op.get_successor(0)) {
                        cfg_blocks[w_blk].succs.push((CfgBranchCond::Always, dst));
                    }
                }
                "mir.cond_br" if op.get_num_operands() >= 1 && op.get_num_successors() == 2 => {
                    if let (Some(t_idx), Some(f_idx)) = (
                        graph.idx_of(op.get_successor(0)),
                        graph.idx_of(op.get_successor(1)),
                    ) {
                        if graph.pure_unreachable.contains(&f_idx)
                            && !graph.pure_unreachable.contains(&t_idx)
                        {
                            cfg_blocks[w_blk].succs.push((CfgBranchCond::Always, t_idx));
                        } else if graph.pure_unreachable.contains(&t_idx)
                            && !graph.pure_unreachable.contains(&f_idx)
                        {
                            cfg_blocks[w_blk].succs.push((CfgBranchCond::Always, f_idx));
                        } else {
                            let is_cyc =
                                graph.can_reach(t_idx, w_blk) || graph.can_reach(f_idx, w_blk);
                            let (tc, fc) = match op_kind(&env, &op, 0) {
                                Some(SsaKind::Guard(g)) if g.is_loop_bound_test() => {
                                    (CfgBranchCond::Uniform, CfgBranchCond::Uniform)
                                }
                                Some(SsaKind::Guard(g)) => {
                                    branch_guards.push((w_blk, f_idx, g.clone()));
                                    branch_guards.push((w_blk, t_idx, g.negate()));
                                    (
                                        CfgBranchCond::Coord(g.clone()),
                                        CfgBranchCond::Coord(g.negate()),
                                    )
                                }
                                Some(SsaKind::MbarrierReady { bar_id }) => {
                                    mbarrier_wait_edges.push((w_blk, t_idx, bar_id));
                                    (
                                        CfgBranchCond::MbarReady(bar_id),
                                        CfgBranchCond::MbarSpin(bar_id),
                                    )
                                }
                                Some(SsaKind::MbarrierNotReady { bar_id }) => {
                                    mbarrier_wait_edges.push((w_blk, f_idx, bar_id));
                                    (
                                        CfgBranchCond::MbarSpin(bar_id),
                                        CfgBranchCond::MbarReady(bar_id),
                                    )
                                }
                                Some(SsaKind::UniformScalar) if is_cyc => {
                                    (CfgBranchCond::Uniform, CfgBranchCond::Uniform)
                                }
                                Some(SsaKind::Global1DLtLen {
                                    guard_param,
                                    group_size,
                                }) => {
                                    global_1d_guards.push((w_blk, f_idx, guard_param, group_size));
                                    branch_guards.push((w_blk, f_idx, GuardExpr::Unknown));
                                    branch_guards.push((w_blk, t_idx, GuardExpr::Unknown));
                                    (CfgBranchCond::Unknown, CfgBranchCond::Unknown)
                                }
                                Some(SsaKind::Global1DGeLen {
                                    guard_param,
                                    group_size,
                                }) => {
                                    global_1d_guards.push((w_blk, t_idx, guard_param, group_size));
                                    branch_guards.push((w_blk, f_idx, GuardExpr::Unknown));
                                    branch_guards.push((w_blk, t_idx, GuardExpr::Unknown));
                                    (CfgBranchCond::Unknown, CfgBranchCond::Unknown)
                                }
                                _ => {
                                    branch_guards.push((w_blk, f_idx, GuardExpr::Unknown));
                                    branch_guards.push((w_blk, t_idx, GuardExpr::Unknown));
                                    (CfgBranchCond::Unknown, CfgBranchCond::Unknown)
                                }
                            };
                            cfg_blocks[w_blk].succs.push((tc, t_idx));
                            cfg_blocks[w_blk].succs.push((fc, f_idx));
                        }
                    }
                }
                "nvvm.mbarrier_init_shared" => {
                    if !matches!(
                        bin_kinds(&env, &op),
                        (
                            Some(SsaKind::MbarrierPtr { .. }),
                            Some(SsaKind::ConstInt { .. })
                        )
                    ) {
                        return Err(reject(
                            &opid,
                            "mbarrier_init requires a static Barrier pointer and constant arrival count",
                        ));
                    }
                }
                "nvvm.mbarrier_arrive_shared" | "nvvm.mbarrier_arrive_cluster" => {
                    if !matches!(op_kind(&env, &op, 0), Some(SsaKind::MbarrierPtr { .. })) {
                        return Err(reject(
                            &opid,
                            "mbarrier_arrive requires a static Barrier pointer",
                        ));
                    }
                }
                "nvvm.mbarrier_arrive_expect_tx_shared"
                | "nvvm.mbarrier_arrive_expect_tx_cluster" => {
                    let (
                        Some(SsaKind::MbarrierPtr { bar_id }),
                        Some(SsaKind::ConstInt { value: raw_tx, .. }),
                    ) = bin_kinds(&env, &op)
                    else {
                        return Err(reject(
                            &opid,
                            "mbarrier_arrive_expect_tx requires a static Barrier pointer and constant byte count",
                        ));
                    };
                    let tx_bytes = u64::try_from(raw_tx).unwrap_or(u64::MAX);
                    if tx_bytes > 0 {
                        arrived_expect_tx_bars.insert(bar_id);
                        *arrived_tx_bytes_per_bar.entry(bar_id).or_insert(0) += tx_bytes;
                    }
                }
                _ if opid.starts_with("nvvm.tcgen05_commit") => {
                    if !spec.tmem_lifecycle {
                        return Err(reject(
                            &opid,
                            "tcgen05 operations in a #[verify] kernel require `tmem_lifecycle`",
                        ));
                    }
                    let Some(SsaKind::MbarrierPtr { bar_id }) = op_kind(&env, &op, 0) else {
                        return Err(reject(
                            &opid,
                            "tcgen05_commit requires a static Barrier pointer",
                        ));
                    };
                    committed_tmem_bars.insert(bar_id);
                }
                _ if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") => {
                    let (
                        Some(SsaKind::SharedBasePtr { shared_id, .. }),
                        Some(SsaKind::MbarrierPtr { bar_id }),
                    ) = bin_kinds(&env, &op)
                    else {
                        return Err(reject(
                            &opid,
                            "cp_async_bulk_tensor_g2s requires a static SharedArray destination and static Barrier",
                        ));
                    };
                    async_bar_for_shared.insert(shared_id, bar_id);
                }
                _ if opid.starts_with("nvvm.cp_async_bulk") => {
                    return Err(reject(
                        &opid,
                        format!("unsupported async bulk operation `{opid}` in verified kernel"),
                    ));
                }
                _ => {}
            }
        }
    }

    let block_guard_expr = |b_idx: usize| -> GuardExpr {
        branch_guards
            .iter()
            .fold(GuardExpr::True, |acc, (br_blk, out_dst, g)| {
                if *br_blk != b_idx
                    && graph.can_reach(0, b_idx)
                    && !graph.can_reach_without(0, b_idx, *br_blk)
                    && !graph.can_reach_without(*out_dst, b_idx, *br_blk)
                {
                    acc.and(g.clone())
                } else {
                    acc
                }
            })
    };
    let is_stage_ready = |b_idx: usize, bar_id: usize| -> bool {
        arrived_expect_tx_bars.contains(&bar_id)
            && mbarrier_wait_edges.iter().any(|&(w_blk, ready_dst, b_id)| {
                b_id == bar_id
                    && b_idx != w_blk
                    && !graph.can_reach_without(0, b_idx, ready_dst)
                    && graph.can_reach(ready_dst, b_idx)
            })
    };
    let mut uniform_mbar_ready_blocks: BTreeSet<usize> = BTreeSet::new();
    let mut effective_num_barriers = graph.barrier_sites.len();
    for &(_w_blk, ready_dst, bar_id) in &mbarrier_wait_edges {
        if (arrived_expect_tx_bars.contains(&bar_id) || committed_tmem_bars.contains(&bar_id))
            && graph
                .return_blocks
                .iter()
                .all(|&ret| !graph.can_reach_without(0, ret, ready_dst))
        {
            effective_num_barriers += 1;
            uniform_mbar_ready_blocks.insert(ready_dst);
        }
    }
    let effective_phase = |b_idx: usize, seen_sync: usize| -> usize {
        block_phase(b_idx, seen_sync)
            + uniform_mbar_ready_blocks
                .iter()
                .filter(|&&rd| graph.can_reach(rd, b_idx) && !graph.can_reach_without(0, b_idx, rd))
                .count()
    };

    let dom_global_1d_guard = |b_idx: usize, expected_group: u64| -> Option<usize> {
        global_1d_guards
            .iter()
            .find_map(|&(br_blk, out_dst, guard_param, gs)| {
                (br_blk != b_idx
                    && gs == expected_group
                    && graph.can_reach(0, b_idx)
                    && !graph.can_reach_without(0, b_idx, br_blk)
                    && !graph.can_reach_without(out_dst, b_idx, br_blk))
                .then_some(guard_param)
            })
    };

    // 5. Pair `mir.assert(mir.lt(idx, size))` with verified `SharedElementPtr`s.
    let mut ptr_asserts = pair_shared_ptr_asserts(ctx, &graph, &spec, &env);

    // 6. Populate `cfg_blocks[b_idx].ops` and access obligations in program order.
    let mut global_strip_accesses = Vec::new();
    let mut global_1d_accesses = Vec::new();
    let mut shared_accesses = Vec::new();
    let mut tmem_accesses = Vec::new();
    let mut proven_assert_ops = Vec::new();
    let mut tma_write_phase_for_shared: HashMap<usize, usize> = HashMap::new();

    for (b_idx, &b_ptr) in graph.blocks.iter().enumerate() {
        let dom_guard = block_guard_expr(b_idx);
        let leader_only = dom_guard.has_zero_leader_check();
        let active_threads = dom_guard.active_threads_bound(u64::from(spec.block_size));
        let residual_guard = dom_guard.custom_residual();
        let mut seen_barriers = 0usize;
        let cfg_blk = &mut cfg_blocks[b_idx];

        for (op_idx, op_ptr) in b_ptr.deref(ctx).iter(ctx).enumerate() {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);

            if opid != "mir.extract_field" {
                for operand in op.operands() {
                    if let Some(SsaKind::TmemLoadedReg { ld_id }) = env.get_kind(&operand) {
                        cfg_blk.ops.push(CfgOpItem::TmemLoadUse { ld_id });
                    }
                }
            }

            match opid.as_str() {
                "nvvm.barrier0" => {
                    cfg_blk.ops.push(CfgOpItem::SyncThreads);
                    seen_barriers += 1;
                    continue;
                }
                "nvvm.fence_proxy_async_shared_cta" => {
                    cfg_blk.ops.push(CfgOpItem::ProxyFenceAsyncShared);
                    continue;
                }
                "mir.return" | "mir.goto" | "mir.cond_br" => continue,
                _ => {}
            }

            let base_phase = effective_phase(b_idx, seen_barriers);
            let resolve_phase = |shared_id: usize, tma_phases: &HashMap<usize, usize>| -> usize {
                if let Some(&bar_id) = async_bar_for_shared.get(&shared_id)
                    && !is_stage_ready(b_idx, bar_id)
                    && let Some(&w_phase) = tma_phases.get(&shared_id)
                {
                    w_phase
                } else {
                    base_phase
                }
            };

            let mut acx = AsyncTranscribeCtx {
                ctx,
                spec: &spec,
                graph: &graph,
                env: &env,
                shared_raw_list: &shared_raw_list,
                tmem_ld_id_map: &tmem_ld_id_map,
                arrived_tx_bytes_per_bar: &arrived_tx_bytes_per_bar,
                ptr_asserts: &ptr_asserts,
                b_idx,
                op_idx,
                base_phase,
                leader_only,
                active_threads,
                residual_guard: &residual_guard,
                cfg_blk,
                shared_accesses: &mut shared_accesses,
                tmem_accesses: &mut tmem_accesses,
                proven_assert_ops: &mut proven_assert_ops,
                tma_write_phase_for_shared: &mut tma_write_phase_for_shared,
            };
            if transcribe_async_or_tcgen05_op(&opid, &op, &mut acx, resolve_phase)? {
                continue;
            }

            match opid.as_str() {
                "nvvm.cvta_generic_to_shared_offset" => {
                    if !matches!(
                        env.get_kind(&op.get_result(0)),
                        Some(SsaKind::SmemSharedOffset { .. })
                    ) {
                        return Err(reject(
                            &opid,
                            "`cvta_generic_to_shared_offset` must target a `SharedArray` listed in `no_oob_idx` and `no_data_race`",
                        ));
                    }
                }
                "mir.ptr_offset" => {
                    if matches!(op_kind(&env, &op, 0), Some(SsaKind::SharedBasePtr { .. }))
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(SsaKind::SharedElementPtr { .. })
                        )
                    {
                        return Err(reject(
                            "mir.ptr_offset",
                            "shared memory pointer offset must use a recognized 2D tile coordinate",
                        ));
                    }
                    if let Some(SsaKind::SlicePtr(param)) = op_kind(&env, &op, 0)
                        && (spec.no_oob_idx.contains(&param)
                            || spec.disjoint_writes.contains(&param)
                            || spec.mutable_slice_params.contains(&param))
                        && !matches!(
                            env.get_kind(&op.get_result(0)),
                            Some(
                                SsaKind::GlobalStripElementPtr { .. }
                                    | SsaKind::Global1DElementPtr { .. }
                            )
                        )
                    {
                        let reason = if spec.domain == 2 {
                            "2D global slice pointer offset must use a row-strip (`start + col`) or col-strip (`start + row * stride`) index"
                        } else {
                            "verified global slice pointer offset must use the 1D thread index (`(blockIdx_x as usize) * (blockDim_x as usize) + (threadIdx_x as usize)` or `thread::index_1d()`)"
                        };
                        return Err(reject("mir.ptr_offset", reason));
                    }
                }
                "mir.assert" => match op_kind(&env, &op, 0) {
                    Some(SsaKind::IndexLtLen {
                        guard_param: checked_param,
                    }) if spec.no_oob_idx.contains(&checked_param) => {
                        proven_assert_ops.push(op_ptr);
                    }
                    Some(SsaKind::Global1DLtLen {
                        guard_param: checked_param,
                        group_size,
                    }) if spec.no_oob_idx.contains(&checked_param)
                        && dom_global_1d_guard(b_idx, group_size).is_some() =>
                    {
                        proven_assert_ops.push(op_ptr);
                    }
                    _ => {}
                },
                "mir.load" | "mir.store" => {
                    let is_store = opid == "mir.store";
                    if is_store
                        && matches!(
                            op_kind(&env, &op, 1),
                            Some(
                                SsaKind::SharedBasePtr { .. }
                                    | SsaKind::SharedElementPtr { .. }
                                    | SsaKind::MbarrierPtr { .. }
                            )
                        )
                    {
                        return Err(reject(
                            "mir.store",
                            "verified shared-memory or mbarrier pointer cannot escape by being stored to memory",
                        ));
                    }
                    if is_store
                        && let Some(
                            SsaKind::BufferParam(p)
                            | SsaKind::SlicePtr(p)
                            | SsaKind::Global1DElementPtr { param: p, .. },
                        ) = op_kind(&env, &op, 1)
                        && (spec.mutable_slice_params.contains(&p)
                            || spec.disjoint_writes.contains(&p))
                    {
                        return Err(reject(
                            "mir.store",
                            "verified mutable slice pointer cannot escape by being stored to memory",
                        ));
                    }
                    match op_kind(&env, &op, 0) {
                        Some(SsaKind::SharedElementPtr {
                            shared_id,
                            size,
                            addr,
                        }) => {
                            if let Some(aop) = ptr_asserts.remove(&op.get_operand(0))
                                && !proven_assert_ops.contains(&aop)
                            {
                                proven_assert_ops.push(aop);
                            }
                            cfg_blk.ops.push(if is_store {
                                CfgOpItem::SharedWrite { shared_id }
                            } else {
                                CfgOpItem::SharedRead { shared_id }
                            });
                            let (pattern, loop_bound) = addr.classify_pattern(spec.block_size);
                            let guard = if matches!(
                                pattern,
                                TileAccessPatternKind::Expr(_)
                                    | TileAccessPatternKind::ColScan { .. }
                            ) && active_threads < u64::from(spec.block_size)
                            {
                                dom_guard.clone()
                            } else {
                                residual_guard.clone()
                            };
                            push_shared_access(
                                &mut shared_accesses,
                                &shared_raw_list,
                                if is_store {
                                    AccessKind::Write
                                } else {
                                    AccessKind::Read
                                },
                                shared_id,
                                size,
                                resolve_phase(shared_id, &tma_write_phase_for_shared),
                                pattern,
                                loop_bound,
                                leader_only,
                                active_threads,
                                guard,
                            );
                        }
                        Some(SsaKind::GlobalStripElementPtr {
                            param,
                            kind,
                            offset,
                        }) if !is_store => {
                            global_strip_accesses.push(GlobalStripAccess2D {
                                param,
                                kind,
                                offset,
                            });
                        }
                        Some(SsaKind::Global1DElementPtr {
                            param,
                            offset,
                            group_size,
                        }) if spec.no_oob_idx.contains(&param)
                            || spec.disjoint_writes.contains(&param)
                            || spec.mutable_slice_params.contains(&param) =>
                        {
                            let Some(guard_param) = dom_global_1d_guard(b_idx, group_size) else {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "unguarded global access on parameter {param}: must be dominated by a strict `< len` guard"
                                    ),
                                ));
                            };
                            if is_store
                                && group_size > 1
                                && !(group_size == u64::from(spec.block_size)
                                    && dom_guard.has_cta_leader_zero_check())
                            {
                                return Err(reject(
                                    &opid,
                                    "block-slot global write (`out[blockIdx_x]`) requires a dominating `tid == 0` guard",
                                ));
                            }
                            global_1d_accesses.push(PhasedGlobal1DAccess {
                                kind: if is_store {
                                    AccessKind::Write
                                } else {
                                    AccessKind::Read
                                },
                                target_param: param,
                                guard_param,
                                index_offset: offset,
                                group_size,
                            });
                        }
                        Some(SsaKind::SlicePtr(param))
                            if spec.no_oob_idx.contains(&param)
                                || spec.disjoint_writes.contains(&param)
                                || spec.mutable_slice_params.contains(&param) =>
                        {
                            return Err(reject(
                                &opid,
                                format!(
                                    "unindexed access on verified slice parameter {param} is not allowed"
                                ),
                            ));
                        }
                        _ => {}
                    }
                }
                "mir.cast"
                | "mir.construct_struct"
                | "mir.insert_field"
                | "mir.alloca"
                | "mir.field_addr"
                | "nvvm.mbarrier_inval_shared"
                | "nvvm.mbarrier_try_wait_parity_shared"
                | "nvvm.mbarrier_try_wait_shared"
                | "nvvm.mbarrier_test_wait_shared"
                | "nvvm.mbarrier_try_wait_parity_cluster"
                | "nvvm.mbarrier_try_wait_parity_cluster_relaxed" => {}
                _ => {
                    for operand in op.operands() {
                        match env.get_kind(&operand) {
                            Some(
                                SsaKind::SharedBasePtr { shared_id, .. }
                                | SsaKind::SharedElementPtr { shared_id, .. },
                            ) if spec.no_oob_shared.contains(&shared_id)
                                || spec.no_data_race.contains(&shared_id) =>
                            {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "verified shared-memory pointer cannot escape into unmodeled operation `{opid}`"
                                    ),
                                ));
                            }
                            Some(SsaKind::MbarrierPtr { .. }) => {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "verified mbarrier pointer cannot escape into unmodeled operation `{opid}`"
                                    ),
                                ));
                            }
                            Some(
                                SsaKind::BufferParam(p)
                                | SsaKind::SlicePtr(p)
                                | SsaKind::Global1DElementPtr { param: p, .. },
                            ) if (spec.mutable_slice_params.contains(&p)
                                || spec.disjoint_writes.contains(&p))
                                && opid != "mir.extract_field" =>
                            {
                                return Err(reject(
                                    &opid,
                                    format!(
                                        "verified mutable slice parameter {p} cannot escape into unmodeled operation `{opid}`"
                                    ),
                                ));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    shared_accesses.sort_by_key(|a| (a.shared_id, a.phase, a.kind));

    let mut control_flow_cfg = TranscribedControlFlowCfg {
        num_shared: shared_raw_list.len(),
        num_barriers: bar_raw_list.len(),
        blocks: cfg_blocks,
    };
    control_flow_cfg.compact(&BTreeSet::new());

    Ok(TranslatedFunction::TiledKernel2D {
        arity,
        spec,
        num_barriers: effective_num_barriers,
        global_strip_accesses,
        global_1d_accesses,
        shared_accesses,
        control_flow_cfg,
        tmem_accesses,
        proven_assert_ops,
    })
}
