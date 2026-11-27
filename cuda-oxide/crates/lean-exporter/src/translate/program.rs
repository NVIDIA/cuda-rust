/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Line-by-line `PostMem2Reg` `dialect-mir` -> `CudaOxide.Mir.Program.MirFunc` transcriber.
//!
//! Translates basic blocks, block arguments, integer/boolean SSA operations, pointer
//! offsets, loads, stores, assertions, `mbarrier`/TMA bulk copies, `cvta` SMEM
//! descriptors, Blackwell `tcgen05` / `TMEM` operations, and terminators 1-to-1
//! into Lean without any indexing-pattern recognizers in Rust.

use std::collections::{BTreeSet, HashMap, HashSet};

use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirStructType,
    MirTupleType,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::types::IntegerType,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};

use super::{
    TranslatedFunction,
    ir_utils::{
        BlockGraph, attribute_field_index, decode_int_constant, decode_tcgen05_idesc_shape,
        int_attr_u64, is_mbarrier_shared_alloc, mir_type_elem_bytes, op_name,
        parse_tcgen05_cp_shape, parse_tcgen05_ld_shape, ptr_pointee_elem_bytes,
        shared_alloc_elem_bytes,
    },
};
use crate::{
    ExportError, reject,
    spec::{KernelVerificationSpec, get_string_attr},
    ssa::{
        CfgBranchCond, CfgMbarWaitKind, CfgOpItem, GuardExpr, IdxExpr, TranscribedCfgBlock,
        TranscribedControlFlowCfg,
    },
    symbol_name,
};

/// Parameter kind in a transcribed [`MirProgramData`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirParamKind {
    /// Scalar unsigned integer parameter of `width` bits (`0` for float/opaque).
    Scalar(u32),
    /// Slice parameter `&[T]` (`false`) or `&mut [T]` (`true`).
    Slice(bool),
    /// Host-partitioned `DisjointSlice<T>` parameter.
    DisjointSlice,
}

/// Transcribed `Rhs` of a `Stmt.assign` in `CudaOxide.Mir.Program`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MirRhs {
    /// Alias / copy of `src` SSA variable.
    Copy(usize),
    /// Unsigned integer literal `(width, value)`.
    ConstInt(u32, u128),
    /// Boolean literal.
    ConstBool(bool),
    /// Hardware special register name (`"tidX"`, `"tidY"`, `"ctaidX"`, ...).
    Sreg(&'static str),
    /// 64-bit 1D global thread index (`thread::index_1d`).
    Index1D,
    /// Packed 2D `u32` coordinates `((row as u64) << 32) + (col as u64)` (`thread::coords_2d_u32`).
    Coord2DU32,
    /// Base pointer of slice parameter `param`.
    SlicePtr(usize),
    /// Element length of slice parameter `param`.
    SliceLen(usize),
    /// Base pointer of `SharedArray` `shared_id`.
    SharedBase(usize),
    /// Unsigned `width`-bit binary operator `(op_lean_name, width, lhs, rhs)`.
    BinOp(&'static str, u32, usize, usize),
    /// Unsigned `width`-bit overflow boolean `(op_lean_name, width, lhs, rhs)`.
    CheckedOvf(&'static str, u32, usize, usize),
    /// Unsigned integer comparison `(op_lean_name, lhs, rhs)`.
    CmpOp(&'static str, usize, usize),
    /// Boolean negation `!src`.
    NotBool(usize),
    /// Unsigned integer width cast `(dst_width, src)`.
    IntCast(u32, usize),
    /// Pointer element offset `base + idx`.
    PtrOffset(usize, usize),
    /// Memory load `*ptr`.
    Load(usize),
    /// `mbarrier_try_wait*(&BAR[slot], parity)` poll on `(bar_id, slot_var, is_acquire)`.
    MbarTryWait(usize, usize, bool),
    /// `.shared::cta` byte offset `cvta_generic_to_shared_offset(ptr_var)`.
    CvtaShared(usize),
    /// Load base `TMEM` handle of `cols` columns from shared-memory slot `ptr_var`.
    LoadTmemBase(usize, u64),
    /// Arbitrary `width`-bit unsigned integer produced by an opaque operation `(var_id, width)`.
    OpaqueUint(usize, u32),
    /// `width`-bit unsigned integer whose CTA-uniformity follows `src` (`(var_id, width, src)`).
    CtaUint1(usize, u32, usize),
    /// Zero-operand CTA-uniform constant (`mir.float_constant`).
    CtaConst,
    /// Single-operand pure combinational / CTA-query op (`clc_query_is_canceled`).
    CtaUnary(usize),
    /// Two-operand pure combinational non-index op (floating-point comparison).
    CtaBin(usize, usize),
    /// Non-index float or aggregate value.
    OpaqueVal,
}

/// Transcribed statement inside a [`MirBlockData`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MirStmt {
    /// SSA variable assignment `dst = rhs`.
    Assign(usize, MirRhs),
    /// Memory store `*ptr = val`.
    Store(usize, usize),
    /// Assertion `mir.assert(cond)` with deterministic `assert_id`.
    AssertCond(usize, usize),
    /// CTA-wide `thread::sync_threads()` (`nvvm.barrier0`).
    SyncThreads,
    /// Bounds-checked slot reference on a `SharedArray<Barrier, N>` pointer `ptr_var`.
    BarrierSlotCheck(usize),
    /// Bounds-checked peer CTA rank (`rank < clusterDimX`) on `cluster::map_shared_rank*`.
    ClusterRankCheck(usize),
    /// Plain arrival `mbarrier_arrive*(&BAR[slot])` on `(bar_id, slot_var, is_release)`.
    MbarArrive(usize, usize, bool),
    /// Byte-counted `mbarrier_arrive_expect_tx(&BAR[slot], count, tx_bytes)` on `(bar_id, slot_var, tx_bytes)`.
    MbarArriveExpectTx(usize, usize, u64),
    /// Asynchronous TMA copy `cp_async_bulk_tensor_*_g2s(ptr_var, bar_id, slot_var, span_elems)`.
    TmaCopyG2S(usize, usize, usize, u64),
    /// Warp-collective `tcgen05_alloc(ptr_var, cols)`.
    TmemAlloc(usize, u64),
    /// Asynchronous `tcgen05` shared-memory descriptor read `(desc_var, tile_bytes)`.
    SmemDescRead(usize, u64),
    /// Blackwell `TMEM` access `(tmem_var, height_lanes, width_cols, warp_restricted)`.
    TmemAccess(usize, u64, u64, bool),
}

/// Transcribed terminator ending a [`MirBlockData`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MirTerminator {
    /// `mir.return`.
    Ret,
    /// `mir.unreachable`.
    Unreachable,
    /// `mir.goto target(args)`.
    Goto(usize, Vec<usize>),
    /// `mir.cond_br cond ? then_blk(then_args) : else_blk(else_args)`.
    CondBr(usize, usize, Vec<usize>, usize, Vec<usize>),
}

/// One transcribed basic block in [`MirProgramData`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MirBlockData {
    /// Formal `(var_id, bit_width)` block arguments.
    pub args: Vec<(usize, u32)>,
    /// True when targeted by at least one cyclic backedge (`dst <= src` in RPO).
    pub is_loop_header: bool,
    /// Sequential statements inside this block.
    pub stmts: Vec<MirStmt>,
    /// Block-ending terminator.
    pub term: MirTerminator,
}

/// Complete line-by-line `MirFunc` transcription plus auxiliary CFG obligations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MirProgramData {
    /// Entry parameter kinds.
    pub params: Vec<MirParamKind>,
    /// Static `SharedArray` `(short_name, size, elem_bytes)` declarations.
    pub shared_decls: Vec<(String, u64, u64)>,
    /// `SharedArray` IDs proved race-free (`no_data_race`).
    pub race_free_shared: Vec<usize>,
    /// RPO-ordered basic blocks (`0` is the entry block).
    pub blocks: Vec<MirBlockData>,
    /// `(assert_id, op_ptr)` pairs for `mir.assert` checks guarding verified `no_oob_idx` accesses.
    pub candidate_asserts: Vec<(usize, Ptr<Operation>)>,
    /// Transcribed `ControlFlowCfg` for `uniform_barriers` and `tmem_lifecycle`.
    pub control_flow_cfg: TranscribedControlFlowCfg,
    /// True when at least one `cluster::map_shared_rank*` (`Stmt.clusterRankCheck`) was emitted.
    pub has_cluster_rank_checks: bool,
}

/// Return the unsigned integer bit width (`8`, `16`, `32`, `64`) of `ty`, `1` for `i1`, or `0` otherwise.
fn mir_uint_width(ctx: &Context, ty: TypeHandle) -> u32 {
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return 0;
    };
    if int_ty.width() == 1 {
        1
    } else if int_ty.is_unsigned() {
        int_ty.width()
    } else {
        0
    }
}

/// Minimal syntactic provenance tag tracked solely to expand `extract_field` on slice
/// parameters, detect escaping pointers, pair `mir.assert` checks, and wire hardware
/// `Barrier` / `TMEM` / SMEM descriptor handles into `ControlFlowCfg` and `MirFunc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalTag {
    /// Entry slice parameter `p` (verified when `true`, unverified when `false`).
    SliceParam(usize, bool),
    /// Verified slice base or element pointer on `p`.
    GlobalPtr(usize),
    /// Unverified global/stack/view pointer (`MatrixView32`, `Tile2dMut`, `mir.alloca`).
    UnverifiedPtr,
    /// `SharedArray` base or element pointer on `shared_id` of `size`.
    SharedPtr(usize, u64),
    /// Distributed Shared Memory (`map_shared_rank`) pointer into peer `shared_id` of `size`.
    DsmemSharedPtr(usize, u64),
    /// Static 64-bit hardware `Barrier` or `SharedArray<Barrier, N>` pointer.
    BarrierPtr {
        /// Barrier group index in `barrier_decls`.
        bar_id: usize,
        /// `Some((shared_id, size))` when backed by `SharedArray<Barrier, N>`, `None` for scalar `Barrier`.
        shared_opt: Option<(usize, u64)>,
    },
    /// 64-bit cluster-shared barrier byte address produced by `cvta` on `map_shared_rank(&BAR, rank)`.
    ClusterBarrierAddr {
        /// Barrier group index in `barrier_decls`.
        bar_id: usize,
        /// `Some((shared_id, size))` when backed by `SharedArray<Barrier, N>`, `None` for scalar `Barrier`.
        shared_opt: Option<(usize, u64)>,
    },
    /// `.shared::cta` byte offset or 64-bit `tcgen05` SMEM descriptor backed by `shared_id`.
    SmemDescSource(usize),
    /// Allocated Blackwell `TMEM` address handle backed by `alloc_cols` columns.
    TmemAddr(u64),
    /// Register returned by `nvvm.tcgen05_ld_*` awaiting `tcgen05_load_wait()`.
    TmemLoadedReg(usize),
    /// Single-field scalar wrapper parameter (`Uniform<u32>`, `ThreadIndex`) carrying `src_var`.
    ScalarWrapper(usize),
    /// `thread::index_1d()` wrapper struct carrying the 64-bit global index in field `0`.
    Index1DWrapper(usize),
    /// Thread-local stack slot (`mir.alloca`).
    StackSlot,
    /// Join of incompatible pointer origins across predecessors; fails closed on load/store/call.
    ConflictingPtr,
}

/// Join `incoming` into `dst`'s provenance tag, saturating to [`LocalTag::ConflictingPtr`]
/// whenever two control-flow predecessors disagree on a pointer's origin.
fn merge_local_tag(tags: &mut HashMap<Value, LocalTag>, dst: Value, incoming: LocalTag) {
    match tags.get(&dst).copied() {
        None => {
            tags.insert(dst, incoming);
        }
        Some(cur) if cur == incoming => {}
        Some(_) => {
            tags.insert(dst, LocalTag::ConflictingPtr);
        }
    }
}

/// Candidate `mir.assert` instruction awaiting pairing with the subsequent `mir.ptr_offset`
/// on the checked slice parameter or `SharedArray` so Lean can prove it redundant via
/// `MirAssertRedundant` and elide it before LLVM/PTX lowering.
#[derive(Clone, Copy, Debug)]
struct PendingAssert {
    /// Deterministic assertion ID emitted in `Stmt.assertCond`.
    assert_id: usize,
    /// Original Pliron `mir.assert` operation handle for post-proof elision.
    op_ptr: Ptr<Operation>,
    /// Checked index SSA value (for `SharedArray` `< size` assertions).
    idx_val: Value,
    /// Target slice parameter when the assertion checks `< slice.len()`.
    slice_param: Option<usize>,
    /// Target constant array bound when the assertion checks `< SHARED.size`.
    shared_bound: Option<u64>,
}

/// Compute Reverse Post-Order (RPO) over reachable non-`unreachable` blocks from `0`.
fn compute_rpo_blocks(graph: &BlockGraph) -> Vec<usize> {
    /// Depth-first traversal pushing non-unreachable block indices in post-order.
    fn dfs(u: usize, graph: &BlockGraph, vis: &mut HashSet<usize>, post: &mut Vec<usize>) {
        if !vis.insert(u) || graph.pure_unreachable.contains(&u) {
            return;
        }
        for &s in graph.succs[u].iter().rev() {
            dfs(s, graph, vis, post);
        }
        post.push(u);
    }
    let mut vis = HashSet::new();
    let mut post = Vec::new();
    dfs(0, graph, &mut vis, &mut post);
    post.reverse();
    post
}

/// Transcribe a `PostMem2Reg` `mir.func` line by line into [`TranslatedFunction::MirProgram`].
pub(super) fn translate_mir_program(
    ctx: &Context,
    function: Ptr<Operation>,
    arity: usize,
    spec: KernelVerificationSpec,
) -> Result<TranslatedFunction, ExportError> {
    let raw_blocks: Vec<Ptr<BasicBlock>> = function
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .collect();
    if raw_blocks.is_empty() {
        return Err(reject("kernel", "empty kernel region"));
    }
    let graph = BlockGraph::build(ctx, raw_blocks);
    let rpo_old = compute_rpo_blocks(&graph);
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (new_idx, &old_idx) in rpo_old.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    // 1. Discover static `SharedArray`, `Barrier` (`mbarrier`), and `tcgen05_ld` sites.
    let mut shared_decls: Vec<(String, u64, u64)> = Vec::new();
    let mut barrier_decls: Vec<String> = Vec::new();
    let mut tmem_ld_id_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut alloc_short_origin: HashMap<Value, String> = HashMap::new();
    for &old_idx in &rpo_old {
        for (op_idx, op_ptr) in graph.blocks[old_idx].deref(ctx).iter(ctx).enumerate() {
            let opid = op_name(ctx, op_ptr);
            if opid.starts_with("nvvm.tcgen05_ld_") {
                let next_id = tmem_ld_id_map.len();
                tmem_ld_id_map.insert((old_idx, op_idx), next_id);
            } else if opid == "mir.shared_alloc" {
                let op = op_ptr.deref(ctx);
                let full = get_string_attr(&op.attributes, "source_name")
                    .unwrap_or_else(|| "SHARED".into());
                let short = full.rsplit("::").next().unwrap_or(&full).to_string();
                if op.get_num_results() == 1 {
                    alloc_short_origin.insert(op.get_result(0), short.clone());
                }
                if is_mbarrier_shared_alloc(ctx, &op) {
                    if !barrier_decls.contains(&short) {
                        barrier_decls.push(short);
                    }
                } else {
                    let size = int_attr_u64(&op.attributes, "size").unwrap_or(0);
                    let eb = shared_alloc_elem_bytes(ctx, &op);
                    if !shared_decls.iter().any(|(n, _, _)| n == &short) {
                        shared_decls.push((short, size, eb));
                    }
                }
            }
        }
    }
    // Propagate `alloc_short_origin` across pointer offsets/casts/field_addr so `SharedArray<Barrier, N>`
    // statics passed to `nvvm.mbarrier_*`, `nvvm.cp_async_bulk_tensor_g2s*`, or `nvvm.tcgen05_commit*`
    // are registered in both `shared_decls` and `barrier_decls`.
    for _ in 0..4 {
        for &old_idx in &rpo_old {
            for op_ptr in graph.blocks[old_idx].deref(ctx).iter(ctx) {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if matches!(
                    on.as_str(),
                    "mir.ptr_offset"
                        | "mir.cast"
                        | "mir.field_addr"
                        | "nvvm.mapa_shared_cluster"
                        | "nvvm.cvta_generic_to_shared_offset"
                        | "mir.bitand"
                ) && op.get_num_operands() >= 1
                    && op.get_num_results() == 1
                    && let Some(orig) = alloc_short_origin.get(&op.get_operand(0)).cloned()
                {
                    alloc_short_origin.insert(op.get_result(0), orig);
                }
            }
        }
    }
    let mut barrier_shared_names: HashSet<String> = HashSet::new();
    for &old_idx in &rpo_old {
        for op_ptr in graph.blocks[old_idx].deref(ctx).iter(ctx) {
            let on = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            let bar_val_opt =
                if on.starts_with("nvvm.mbarrier_") || on.starts_with("nvvm.tcgen05_commit") {
                    op.operands().next()
                } else if on.starts_with("nvvm.cp_async_bulk_tensor_g2s")
                    || on.starts_with("nvvm.clc_try_cancel")
                {
                    op.operands().nth(1)
                } else {
                    None
                };
            if let Some(bv) = bar_val_opt
                && let Some(short) = alloc_short_origin.get(&bv)
                && shared_decls.iter().any(|(n, _, eb)| n == short && *eb == 8)
            {
                barrier_shared_names.insert(short.clone());
                if !barrier_decls.contains(short) {
                    barrier_decls.push(short.clone());
                }
            }
        }
    }
    shared_decls.sort_by(|a, b| a.0.cmp(&b.0));
    barrier_decls.sort();

    // 2. Assign sequential `var_id`s to every block argument and SSA result across `rpo_old`.
    let mut val_ids: HashMap<Value, usize> = HashMap::new();
    let mut next_var_id = 0usize;

    let mut params = Vec::with_capacity(arity);
    let mut slice_elem_bytes: HashMap<usize, u64> = HashMap::new();
    let mut tags: HashMap<Value, LocalTag> = HashMap::new();
    let is_verified_param = |p: usize| -> bool {
        spec.no_oob_idx.contains(&p)
            || spec.disjoint_writes.contains(&p)
            || spec.writes_all_elems.contains(&p)
            || spec.mutable_slice_params.contains(&p)
            || spec.requires_len_ge.iter().any(|&(l, r)| l == p || r == p)
    };
    // Parameters a `#[verify]` clause makes an access or coverage claim about.
    let is_claimed_param = |p: usize| -> bool {
        spec.no_oob_idx.contains(&p)
            || spec.disjoint_writes.contains(&p)
            || spec.writes_all_elems.contains(&p)
            || spec.mutable_slice_params.contains(&p)
    };
    let kernel_name = symbol_name(ctx, function).unwrap_or_else(|_| "<kernel>".to_string());
    for (idx, arg) in graph.blocks[0].deref(ctx).arguments().enumerate() {
        let arg_vid = *val_ids.entry(arg).or_insert_with(|| {
            let id = next_var_id;
            next_var_id += 1;
            id
        });
        let ty = arg.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        if let Some(sl) = ty_ref.downcast_ref::<MirSliceType>() {
            params.push(MirParamKind::Slice(sl.is_mutable()));
            slice_elem_bytes.insert(idx, mir_type_elem_bytes(ctx, sl.element_type()));
            tags.insert(arg, LocalTag::SliceParam(idx, is_verified_param(idx)));
        } else if let Some(ds) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
            params.push(MirParamKind::DisjointSlice);
            slice_elem_bytes.insert(idx, mir_type_elem_bytes(ctx, ds.element_type()));
            tags.insert(arg, LocalTag::SliceParam(idx, is_verified_param(idx)));
        } else if let Some(st) = ty_ref.downcast_ref::<MirStructType>()
            && st.field_types.len() == 1
        {
            let fw = mir_uint_width(ctx, st.field_types[0]);
            if fw > 1 {
                params.push(MirParamKind::Scalar(fw));
                tags.insert(arg, LocalTag::ScalarWrapper(arg_vid));
            } else {
                params.push(MirParamKind::Scalar(0));
                tags.insert(arg, LocalTag::UnverifiedPtr);
            }
        } else {
            let w = mir_uint_width(ctx, ty);
            if w <= 1
                && (ty_ref.downcast_ref::<MirPtrType>().is_some()
                    || ty_ref.downcast_ref::<MirStructType>().is_some())
            {
                tags.insert(arg, LocalTag::UnverifiedPtr);
            }
            params.push(MirParamKind::Scalar(if w <= 1 { 0 } else { w }));
        }
    }

    // Identify loop headers (`dst <= src` backedge in RPO).
    let mut loop_headers: HashSet<usize> = HashSet::new();
    for (new_src, &old_src) in rpo_old.iter().enumerate() {
        for &old_dst in &graph.succs[old_src] {
            if let Some(&new_dst) = old_to_new.get(&old_dst)
                && new_dst <= new_src
            {
                loop_headers.insert(new_dst);
            }
        }
    }

    // 3. Track CTA-uniform vs. thread-divergent SSA values for `ControlFlowCfg`.
    //    Reads from race-free `SharedArray`s (`spec.no_data_race`) at CTA-uniform indices
    //    return the same shared-memory word across all 32 lanes of an active warp.
    let mut race_free_shared_ptrs: HashSet<Value> = HashSet::new();
    for _ in 0..4 {
        for &old_idx in &rpo_old {
            for op_ptr in graph.blocks[old_idx].deref(ctx).iter(ctx) {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 {
                    continue;
                }
                let res = op.get_result(0);
                if on == "mir.shared_alloc" {
                    let full = get_string_attr(&op.attributes, "source_name")
                        .unwrap_or_else(|| "SHARED".into());
                    let short = full.rsplit("::").next().unwrap_or(&full);
                    if let Some(sid) = shared_decls.iter().position(|(n, _, _)| n == short)
                        && spec.no_data_race.contains(&sid)
                    {
                        race_free_shared_ptrs.insert(res);
                    }
                } else if matches!(on.as_str(), "mir.cast" | "mir.copy" | "mir.ptr_offset")
                    && op.get_num_operands() >= 1
                    && race_free_shared_ptrs.contains(&op.get_operand(0))
                {
                    race_free_shared_ptrs.insert(res);
                }
            }
        }
    }
    let mut divergent_vals: HashSet<Value> = HashSet::new();
    for _ in 0..graph.len() + 2 {
        let mut changed = false;
        for &old_idx in &rpo_old {
            let b_ptr = graph.blocks[old_idx];
            let blk = b_ptr.deref(ctx);
            if old_idx != 0 {
                for (arg_pos, arg) in blk.arguments().enumerate() {
                    if divergent_vals.contains(&arg) {
                        continue;
                    }
                    let mut any_div = false;
                    for &p_old in &rpo_old {
                        for pop in graph.blocks[p_old].deref(ctx).iter(ctx) {
                            let pn = op_name(ctx, pop);
                            let po = pop.deref(ctx);
                            if pn == "mir.goto"
                                && po.get_num_successors() == 1
                                && po.get_successor(0) == b_ptr
                                && arg_pos < po.get_num_operands()
                            {
                                any_div |= divergent_vals.contains(&po.get_operand(arg_pos));
                            } else if pn == "mir.cond_br" && po.get_num_successors() == 2 {
                                let tl = po.get_successor(0).deref(ctx).get_num_arguments();
                                if po.get_successor(0) == b_ptr
                                    && 1 + arg_pos < po.get_num_operands()
                                {
                                    any_div |= divergent_vals
                                        .contains(&po.get_operand(1 + arg_pos))
                                        || divergent_vals.contains(&po.get_operand(0));
                                }
                                if po.get_successor(1) == b_ptr
                                    && 1 + tl + arg_pos < po.get_num_operands()
                                {
                                    any_div |= divergent_vals
                                        .contains(&po.get_operand(1 + tl + arg_pos))
                                        || divergent_vals.contains(&po.get_operand(0));
                                }
                            }
                        }
                    }
                    if any_div && divergent_vals.insert(arg) {
                        changed = true;
                    }
                }
            }
            for op_ptr in blk.iter(ctx) {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 || divergent_vals.contains(&op.get_result(0)) {
                    continue;
                }
                let is_pure_uni_source = matches!(
                    on.as_str(),
                    "mir.constant"
                        | "builtin.constant"
                        | "mir.float_constant"
                        | "nvvm.read_ptx_sreg_ctaid_x"
                        | "nvvm.read_ptx_sreg_ctaid_y"
                        | "nvvm.read_ptx_sreg_ntid_x"
                        | "nvvm.read_ptx_sreg_ntid_y"
                        | "nvvm.read_ptx_sreg_cluster_ctaid_x"
                        | "nvvm.read_ptx_sreg_cluster_nctaid_x"
                        | "nvvm.clc_query_is_canceled"
                        | "nvvm.clc_query_get_first_ctaid_x"
                        | "nvvm.clc_query_get_first_ctaid_y"
                        | "nvvm.clc_query_get_first_ctaid_z"
                );
                let is_pure_comb = on.starts_with("mir.")
                    && !matches!(
                        on.as_str(),
                        "mir.load"
                            | "mir.call"
                            | "mir.alloca"
                            | "mir.shared_alloc"
                            | "mir.ptr_offset"
                    );
                let is_div = if is_pure_uni_source
                    || (on == "mir.shared_alloc"
                        && race_free_shared_ptrs.contains(&op.get_result(0)))
                {
                    false
                } else if is_pure_comb
                    || (on == "mir.ptr_offset" && race_free_shared_ptrs.contains(&op.get_result(0)))
                    || (on == "mir.load"
                        && op.get_num_operands() >= 1
                        && race_free_shared_ptrs.contains(&op.get_operand(0)))
                {
                    op.operands().any(|v| divergent_vals.contains(&v))
                } else {
                    true
                };
                if is_div && divergent_vals.insert(op.get_result(0)) {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // 3b. Track intra-warp lane divergence (`lane_divergent_vals`): `tx / 32` (`warp_id`) in a 1D
    //     block is constant across all 32 lanes of every warp, so warp-specialized branches and
    //     warp-uniform `SharedArray` loads stay warp-convergent (`CfgBranchCond::WarpUniform`)
    //     while any per-lane `tx`, `ty`, `lane_id`, or global load stays `lane_divergent_vals`.
    let mut pure_tx_vals: HashSet<Value> = HashSet::new();
    let mut early_const_u64: HashMap<Value, u64> = HashMap::new();
    for _ in 0..3 {
        for &old_idx in &rpo_old {
            for op_ptr in graph.blocks[old_idx].deref(ctx).iter(ctx) {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 {
                    continue;
                }
                let res = op.get_result(0);
                if matches!(on.as_str(), "mir.constant" | "builtin.constant")
                    && let Some((_, val)) = decode_int_constant(&op, &on)
                    && let Ok(v64) = u64::try_from(val)
                {
                    early_const_u64.insert(res, v64);
                } else if on == "nvvm.read_ptx_sreg_tid_x"
                    || (matches!(on.as_str(), "mir.cast" | "mir.copy")
                        && op.get_num_operands() == 1
                        && pure_tx_vals.contains(&op.get_operand(0)))
                {
                    pure_tx_vals.insert(res);
                }
            }
        }
    }
    let mut lane_divergent_vals: HashSet<Value> = HashSet::new();
    for _ in 0..graph.len() + 2 {
        let mut changed = false;
        for &old_idx in &rpo_old {
            let b_ptr = graph.blocks[old_idx];
            let blk = b_ptr.deref(ctx);
            if old_idx != 0 {
                for (arg_pos, arg) in blk.arguments().enumerate() {
                    if lane_divergent_vals.contains(&arg) {
                        continue;
                    }
                    let mut any_div = false;
                    for &p_old in &rpo_old {
                        for pop in graph.blocks[p_old].deref(ctx).iter(ctx) {
                            let pn = op_name(ctx, pop);
                            let po = pop.deref(ctx);
                            if pn == "mir.goto"
                                && po.get_num_successors() == 1
                                && po.get_successor(0) == b_ptr
                                && arg_pos < po.get_num_operands()
                            {
                                any_div |= lane_divergent_vals.contains(&po.get_operand(arg_pos));
                            } else if pn == "mir.cond_br" && po.get_num_successors() == 2 {
                                let tl = po.get_successor(0).deref(ctx).get_num_arguments();
                                if po.get_successor(0) == b_ptr
                                    && 1 + arg_pos < po.get_num_operands()
                                {
                                    any_div |= lane_divergent_vals
                                        .contains(&po.get_operand(1 + arg_pos))
                                        || lane_divergent_vals.contains(&po.get_operand(0));
                                }
                                if po.get_successor(1) == b_ptr
                                    && 1 + tl + arg_pos < po.get_num_operands()
                                {
                                    any_div |= lane_divergent_vals
                                        .contains(&po.get_operand(1 + tl + arg_pos))
                                        || lane_divergent_vals.contains(&po.get_operand(0));
                                }
                            }
                        }
                    }
                    if any_div && lane_divergent_vals.insert(arg) {
                        changed = true;
                    }
                }
            }
            for op_ptr in blk.iter(ctx) {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if op.get_num_results() != 1 || lane_divergent_vals.contains(&op.get_result(0)) {
                    continue;
                }
                let res = op.get_result(0);
                let is_pure_uni_source = matches!(
                    on.as_str(),
                    "mir.constant"
                        | "builtin.constant"
                        | "mir.float_constant"
                        | "nvvm.read_ptx_sreg_ctaid_x"
                        | "nvvm.read_ptx_sreg_ctaid_y"
                        | "nvvm.read_ptx_sreg_ntid_x"
                        | "nvvm.read_ptx_sreg_ntid_y"
                        | "nvvm.read_ptx_sreg_cluster_ctaid_x"
                        | "nvvm.read_ptx_sreg_cluster_nctaid_x"
                        | "nvvm.clc_query_is_canceled"
                        | "nvvm.clc_query_get_first_ctaid_x"
                        | "nvvm.clc_query_get_first_ctaid_y"
                        | "nvvm.clc_query_get_first_ctaid_z"
                );
                let is_warp_div_32 = spec.block_y == 1
                    && op.get_num_operands() == 2
                    && pure_tx_vals.contains(&op.get_operand(0))
                    && early_const_u64.get(&op.get_operand(1)).is_some_and(|&c| {
                        (on == "mir.div" && c >= 32 && c.is_power_of_two())
                            || (on == "mir.shr" && (5..=16).contains(&c))
                    });
                let is_pure_comb = on.starts_with("mir.")
                    && !matches!(
                        on.as_str(),
                        "mir.load"
                            | "mir.call"
                            | "mir.alloca"
                            | "mir.shared_alloc"
                            | "mir.ptr_offset"
                    );
                let is_div = if is_pure_uni_source
                    || is_warp_div_32
                    || (on == "mir.shared_alloc" && race_free_shared_ptrs.contains(&res))
                {
                    false
                } else if is_pure_comb
                    || (on == "mir.ptr_offset" && race_free_shared_ptrs.contains(&res))
                    || (on == "mir.load"
                        && op.get_num_operands() >= 1
                        && race_free_shared_ptrs.contains(&op.get_operand(0)))
                {
                    op.operands().any(|v| lane_divergent_vals.contains(&v))
                } else {
                    true
                };
                if is_div && lane_divergent_vals.insert(res) {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // 4. Pre-propagate integer constants, pointer/barrier/TMEM/descriptor tags, and `expect_tx` counts.
    let mut const_u128_vals: HashMap<Value, u128> = HashMap::new();
    let mut const_u64_vals: HashMap<Value, u64> = HashMap::new();
    let mut max_u64_vals: HashMap<Value, u64> = HashMap::new();
    let mut max_idesc_vals: HashMap<Value, u128> = HashMap::new();
    let mut enum_payload_tags: HashMap<Value, LocalTag> = HashMap::new();
    let mut tmem_slot_cols: HashMap<usize, u64> = HashMap::new();
    let mut struct_field_tags: HashMap<Value, Vec<Option<LocalTag>>> = HashMap::new();
    let mut alloca_field_tags: HashMap<Value, Vec<Option<LocalTag>>> = HashMap::new();
    let mut stack_field_ptrs: HashMap<Value, (Value, usize)> = HashMap::new();
    for _ in 0..8 {
        for &old_b in &rpo_old {
            let b_ptr = graph.blocks[old_b];
            for (op_idx, op_ptr) in b_ptr.deref(ctx).iter(ctx).enumerate() {
                let on = op_name(ctx, op_ptr);
                let op = op_ptr.deref(ctx);
                if matches!(on.as_str(), "mir.constant" | "builtin.constant")
                    && op.get_num_results() == 1
                    && let Some((_, val)) = decode_int_constant(&op, &on)
                {
                    let res = op.get_result(0);
                    const_u128_vals.insert(res, val);
                    if val >= 256 {
                        max_idesc_vals.insert(res, val);
                    }
                    if let Ok(v64) = u64::try_from(val) {
                        const_u64_vals.insert(res, v64);
                        max_u64_vals.insert(res, v64);
                    }
                } else if on == "mir.shl"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                    && let (Some(&vl), Some(&sh)) = (
                        const_u128_vals.get(&op.get_operand(0)),
                        const_u64_vals.get(&op.get_operand(1)),
                    )
                    && sh < 64
                {
                    let res = op.get_result(0);
                    let shifted = vl << sh;
                    const_u128_vals.insert(res, shifted);
                    if let Ok(v64) = u64::try_from(shifted) {
                        const_u64_vals.insert(res, v64);
                    }
                } else if on == "mir.shared_alloc" && op.get_num_results() == 1 {
                    let res = op.get_result(0);
                    let full = get_string_attr(&op.attributes, "source_name")
                        .unwrap_or_else(|| "SHARED".into());
                    let short = full.rsplit("::").next().unwrap_or(&full).to_string();
                    if is_mbarrier_shared_alloc(ctx, &op) {
                        if let Some(bid) = barrier_decls.iter().position(|n| n == &short) {
                            tags.insert(
                                res,
                                LocalTag::BarrierPtr {
                                    bar_id: bid,
                                    shared_opt: None,
                                },
                            );
                        }
                    } else if barrier_shared_names.contains(&short) {
                        if let (Some(sid), Some(bid)) = (
                            shared_decls.iter().position(|(n, _, _)| n == &short),
                            barrier_decls.iter().position(|n| n == &short),
                        ) {
                            let sz = shared_decls[sid].1;
                            tags.insert(
                                res,
                                LocalTag::BarrierPtr {
                                    bar_id: bid,
                                    shared_opt: Some((sid, sz)),
                                },
                            );
                        }
                    } else if let Some(sid) = shared_decls.iter().position(|(n, _, _)| n == &short)
                    {
                        let sz = shared_decls[sid].1;
                        tags.insert(res, LocalTag::SharedPtr(sid, sz));
                    }
                } else if on == "mir.alloca" && op.get_num_results() == 1 {
                    tags.insert(op.get_result(0), LocalTag::StackSlot);
                } else if on == "mir.field_addr"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    let base = op.get_operand(0);
                    let fi_opt = attribute_field_index(&op.attributes, "field_index")
                        .or_else(|| attribute_field_index(&op.attributes, "index"));
                    if let Some(&(root, outer_fi)) = stack_field_ptrs.get(&base) {
                        if fi_opt == Some(0) {
                            stack_field_ptrs.insert(op.get_result(0), (root, outer_fi));
                        }
                    } else if matches!(tags.get(&base), Some(LocalTag::StackSlot))
                        && let Some(fi) = fi_opt
                    {
                        stack_field_ptrs.insert(op.get_result(0), (base, fi as usize));
                    } else if let Some(t @ LocalTag::BarrierPtr { .. }) = tags.get(&base).copied() {
                        tags.insert(op.get_result(0), t);
                    }
                } else if on == "mir.store" && op.get_num_operands() >= 2 {
                    let (pv, vv) = (op.get_operand(0), op.get_operand(1));
                    if let Some(&(alloca, fi)) = stack_field_ptrs.get(&pv) {
                        let entry = alloca_field_tags.entry(alloca).or_default();
                        if entry.len() <= fi {
                            entry.resize(fi + 1, None);
                        }
                        entry[fi] = tags.get(&vv).copied();
                    } else if matches!(tags.get(&pv), Some(LocalTag::StackSlot)) {
                        if let Some(sft) = struct_field_tags.get(&vv).cloned() {
                            alloca_field_tags.insert(pv, sft);
                        } else if let Some(vt) = tags.get(&vv).copied() {
                            alloca_field_tags.insert(pv, vec![Some(vt)]);
                        }
                    }
                } else if matches!(
                    on.as_str(),
                    "mir.construct_struct" | "mir.construct_tuple" | "mir.construct_array"
                ) && op.get_num_results() == 1
                {
                    let res = op.get_result(0);
                    let fts: Vec<Option<LocalTag>> =
                        op.operands().map(|v| tags.get(&v).copied()).collect();
                    if let Some(Some(
                        t @ (LocalTag::SmemDescSource(_) | LocalTag::TmemLoadedReg(_)),
                    )) = fts.first().copied()
                    {
                        tags.insert(res, t);
                    }
                    struct_field_tags.insert(res, fts);
                } else if matches!(
                    on.as_str(),
                    "mir.extract_array_element" | "mir.array_element_addr"
                ) && op.get_num_operands() >= 1
                    && op.get_num_results() == 1
                {
                    if let Some(t @ (LocalTag::TmemLoadedReg(_) | LocalTag::StackSlot)) =
                        tags.get(&op.get_operand(0)).copied()
                    {
                        tags.insert(op.get_result(0), t);
                    }
                } else if on == "mir.insert_field"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                {
                    let (bs, fv, res) = (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    let mut fts = struct_field_tags.get(&bs).cloned().unwrap_or_default();
                    if let Some(fi) = attribute_field_index(&op.attributes, "insert_index")
                        .or_else(|| attribute_field_index(&op.attributes, "index"))
                        .or_else(|| attribute_field_index(&op.attributes, "field_index"))
                    {
                        let i = fi as usize;
                        if fts.len() <= i {
                            fts.resize(i + 1, None);
                        }
                        fts[i] = tags.get(&fv).copied();
                    }
                    if let Some(Some(t @ LocalTag::SmemDescSource(_))) = fts.first().copied() {
                        tags.insert(res, t);
                    }
                    struct_field_tags.insert(res, fts);
                } else if on == "mir.extract_field"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    let fi = attribute_field_index(&op.attributes, "index")
                        .or_else(|| attribute_field_index(&op.attributes, "field_index"));
                    if let Some(LocalTag::SliceParam(p, _)) = tags.get(&src).copied() {
                        if fi == Some(0) {
                            tags.insert(
                                res,
                                if is_claimed_param(p) {
                                    LocalTag::GlobalPtr(p)
                                } else {
                                    LocalTag::UnverifiedPtr
                                },
                            );
                        }
                    } else if let (Some(fts), Some(idx)) = (struct_field_tags.get(&src), fi)
                        && let Some(Some(t)) = fts.get(idx as usize).copied()
                    {
                        tags.insert(res, t);
                    } else if fi == Some(0)
                        && let Some(t @ (LocalTag::SmemDescSource(_) | LocalTag::TmemLoadedReg(_))) =
                            tags.get(&src).copied()
                    {
                        tags.insert(res, t);
                    }
                } else if on == "mir.ptr_offset"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                    && let Some(t) = tags.get(&op.get_operand(0)).copied()
                {
                    tags.insert(op.get_result(0), t);
                } else if on == "mir.cast"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    let cstr = format!("{:?}", op.attributes);
                    if cstr.contains("PointerExposeAddress") {
                        if let Some(t @ (LocalTag::BarrierPtr { .. } | LocalTag::UnverifiedPtr)) =
                            tags.get(&op.get_operand(0)).copied()
                        {
                            tags.insert(op.get_result(0), t);
                        }
                    } else if cstr.contains("PointerWithExposedProvenance") {
                        let src_op = op.get_operand(0);
                        if let Some(t @ LocalTag::BarrierPtr { .. }) = tags.get(&src_op).copied() {
                            tags.insert(op.get_result(0), t);
                        } else {
                            tags.insert(op.get_result(0), LocalTag::UnverifiedPtr);
                        }
                    } else if (cstr.contains("PtrToPtr")
                        || cstr.contains("StaticAddress")
                        || cstr.contains("PointerCoercionMutToConst"))
                        && let Some(t) = tags.get(&op.get_operand(0)).copied()
                    {
                        let src_is_slice = {
                            let ty = op.get_operand(0).get_type(ctx);
                            let r = ty.deref(ctx);
                            r.downcast_ref::<MirSliceType>().is_some()
                                || r.downcast_ref::<MirDisjointSliceType>().is_some()
                        };
                        let res_is_slice = {
                            let ty = op.get_result(0).get_type(ctx);
                            let r = ty.deref(ctx);
                            r.downcast_ref::<MirSliceType>().is_some()
                                || r.downcast_ref::<MirDisjointSliceType>().is_some()
                        };
                        if let LocalTag::SliceParam(p, _) = t
                            && src_is_slice
                            && !res_is_slice
                        {
                            tags.insert(
                                op.get_result(0),
                                if is_claimed_param(p) {
                                    LocalTag::GlobalPtr(p)
                                } else {
                                    LocalTag::UnverifiedPtr
                                },
                            );
                        } else {
                            tags.insert(op.get_result(0), t);
                        }
                        if let Some(&sf) = stack_field_ptrs.get(&op.get_operand(0)) {
                            stack_field_ptrs.insert(op.get_result(0), sf);
                        }
                    } else if cstr.contains("IntToInt") {
                        if let Some(&c) = const_u128_vals.get(&op.get_operand(0)) {
                            let dw = mir_uint_width(ctx, op.get_result(0).get_type(ctx));
                            if dw > 1 {
                                let masked = if dw >= 128 {
                                    c
                                } else {
                                    c & ((1u128 << dw) - 1)
                                };
                                const_u128_vals.insert(op.get_result(0), masked);
                                if let Ok(v64) = u64::try_from(masked) {
                                    const_u64_vals.insert(op.get_result(0), v64);
                                    max_u64_vals.insert(op.get_result(0), v64);
                                }
                            }
                        }
                        if let Some(
                            t @ (LocalTag::SmemDescSource(_)
                            | LocalTag::TmemAddr(_)
                            | LocalTag::TmemLoadedReg(_)),
                        ) = tags.get(&op.get_operand(0)).copied()
                        {
                            tags.insert(op.get_result(0), t);
                        }
                    }
                } else if on == "nvvm.mapa_shared_cluster"
                    && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                {
                    match tags.get(&op.get_operand(0)).copied() {
                        Some(LocalTag::SharedPtr(sid, sz) | LocalTag::DsmemSharedPtr(sid, sz)) => {
                            tags.insert(op.get_result(0), LocalTag::DsmemSharedPtr(sid, sz));
                        }
                        Some(btag @ LocalTag::BarrierPtr { .. }) => {
                            tags.insert(op.get_result(0), btag);
                        }
                        _ => {}
                    }
                } else if matches!(
                    on.as_str(),
                    "nvvm.cvta_generic_to_shared_offset"
                        | "nvvm.cvta_generic_to_cluster_shared_offset"
                ) && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                {
                    if let Some(LocalTag::SharedPtr(sid, _)) = tags.get(&op.get_operand(0)).copied()
                        && spec.no_oob_shared.contains(&sid)
                        && spec.no_data_race.contains(&sid)
                    {
                        tags.insert(op.get_result(0), LocalTag::SmemDescSource(sid));
                    } else if let Some(LocalTag::BarrierPtr { bar_id, shared_opt }) =
                        tags.get(&op.get_operand(0)).copied()
                    {
                        tags.insert(
                            op.get_result(0),
                            LocalTag::ClusterBarrierAddr { bar_id, shared_opt },
                        );
                    }
                } else if matches!(on.as_str(), "nvvm.tcgen05_alloc" | "nvvm.tcgen05_alloc_cg2")
                    && op.get_num_operands() >= 2
                {
                    if let (Some(LocalTag::SharedPtr(sid, _)), Some(&cols)) = (
                        tags.get(&op.get_operand(0)).copied(),
                        const_u64_vals.get(&op.get_operand(1)),
                    ) {
                        tmem_slot_cols.insert(sid, cols);
                    }
                } else if on.starts_with("nvvm.tcgen05_ld_") && op.get_num_results() >= 1 {
                    if let Some(&ld_id) = tmem_ld_id_map.get(&(old_b, op_idx)) {
                        for r_idx in 0..op.get_num_results() {
                            tags.insert(op.get_result(r_idx), LocalTag::TmemLoadedReg(ld_id));
                        }
                    }
                } else if on == "mir.load"
                    && op.get_num_operands() >= 1
                    && op.get_num_results() == 1
                {
                    let (pv, res) = (op.get_operand(0), op.get_result(0));
                    if let Some(LocalTag::SharedPtr(sid, _)) = tags.get(&pv).copied()
                        && let Some(&cols) = tmem_slot_cols.get(&sid)
                    {
                        tags.insert(res, LocalTag::TmemAddr(cols));
                    } else if let Some(&(alloca, fi)) = stack_field_ptrs.get(&pv)
                        && let Some(Some(ft)) = alloca_field_tags
                            .get(&alloca)
                            .and_then(|fs| fs.get(fi))
                            .copied()
                    {
                        tags.insert(res, ft);
                    } else if matches!(tags.get(&pv), Some(LocalTag::StackSlot))
                        && let Some(fts) = alloca_field_tags.get(&pv).cloned()
                    {
                        if let Some(Some(t)) = fts.first().copied() {
                            tags.insert(res, t);
                        }
                        struct_field_tags.insert(res, fts);
                    }
                } else if matches!(
                    on.as_str(),
                    "mir.add" | "mir.shr" | "mir.bitand" | "mir.bitor"
                ) && op.get_num_operands() == 2
                    && op.get_num_results() == 1
                {
                    let (t0, t1) = (
                        tags.get(&op.get_operand(0)).copied(),
                        tags.get(&op.get_operand(1)).copied(),
                    );
                    if let Some(LocalTag::SmemDescSource(sid)) = t0.or(t1) {
                        tags.insert(op.get_result(0), LocalTag::SmemDescSource(sid));
                    } else if on == "mir.add"
                        && let Some(LocalTag::TmemAddr(cols)) = t0.or(t1)
                    {
                        tags.insert(op.get_result(0), LocalTag::TmemAddr(cols));
                    } else if on == "mir.bitand"
                        && let Some(btag @ LocalTag::BarrierPtr { .. }) = t0.or(t1)
                        && (const_u64_vals
                            .get(&op.get_operand(0))
                            .is_some_and(|&m| m == 0xFEFF_FFF8)
                            || const_u64_vals
                                .get(&op.get_operand(1))
                                .is_some_and(|&m| m == 0xFEFF_FFF8))
                    {
                        tags.insert(op.get_result(0), btag);
                    }
                } else if on == "mir.call" && op.get_num_results() == 1 {
                    let res = op.get_result(0);
                    let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                    if callee.contains("Tcgen05SmemDescriptor") {
                        for ov in op.operands() {
                            if let Some(LocalTag::SmemDescSource(sid)) = tags.get(&ov).copied() {
                                tags.insert(res, LocalTag::SmemDescSource(sid));
                                struct_field_tags
                                    .insert(res, vec![Some(LocalTag::SmemDescSource(sid))]);
                                break;
                            }
                            if let Some(Some(ft @ LocalTag::SmemDescSource(_))) = alloca_field_tags
                                .get(&ov)
                                .and_then(|fs| fs.first())
                                .copied()
                            {
                                tags.insert(res, ft);
                                struct_field_tags.insert(res, vec![Some(ft)]);
                                break;
                            }
                        }
                    } else if callee.contains("from_bits")
                        && op.get_num_operands() == 1
                        && let Some(t @ LocalTag::TmemLoadedReg(_)) =
                            tags.get(&op.get_operand(0)).copied()
                    {
                        tags.insert(res, t);
                    }
                } else if on == "mir.construct_enum"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                    && attribute_field_index(&op.attributes, "construct_enum_variant_index")
                        == Some(1)
                    && let Some(&t) = tags.get(&op.get_operand(0))
                {
                    enum_payload_tags.insert(op.get_result(0), t);
                } else if on == "mir.enum_payload"
                    && op.get_num_operands() == 1
                    && op.get_num_results() == 1
                    && let Some(&t) = enum_payload_tags.get(&op.get_operand(0))
                {
                    tags.insert(op.get_result(0), t);
                } else if on == "mir.goto" && op.get_num_successors() == 1 {
                    let dst = op.get_successor(0);
                    for (i, v) in op.operands().enumerate() {
                        if i < dst.deref(ctx).get_num_arguments() {
                            let darg = dst.deref(ctx).get_argument(i);
                            if let Some(&pt) = enum_payload_tags.get(&v) {
                                enum_payload_tags.insert(darg, pt);
                            }
                            if let Some(&vt) = tags.get(&v) {
                                merge_local_tag(&mut tags, darg, vt);
                            }
                            if let Some(sft) = struct_field_tags.get(&v).cloned() {
                                struct_field_tags.insert(darg, sft);
                            }
                            if let Some(&mv) = max_u64_vals.get(&v) {
                                max_u64_vals
                                    .entry(darg)
                                    .and_modify(|cur| *cur = (*cur).max(mv))
                                    .or_insert(mv);
                            }
                            if let Some(&mi) = max_idesc_vals.get(&v) {
                                max_idesc_vals
                                    .entry(darg)
                                    .and_modify(|cur| {
                                        let s1 = decode_tcgen05_idesc_shape(*cur, false)
                                            .unwrap_or((0, 0));
                                        let s2 =
                                            decode_tcgen05_idesc_shape(mi, false).unwrap_or((0, 0));
                                        if (s2.0, s2.1) > (s1.0, s1.1) {
                                            *cur = mi;
                                        }
                                    })
                                    .or_insert(mi);
                            }
                        }
                    }
                } else if on == "mir.cond_br" && op.get_num_successors() == 2 {
                    let d0 = op.get_successor(0);
                    let d1 = op.get_successor(1);
                    let n0 = d0.deref(ctx).get_num_arguments();
                    for i in 0..n0 {
                        if 1 + i < op.get_num_operands() {
                            let sv = op.get_operand(1 + i);
                            let darg = d0.deref(ctx).get_argument(i);
                            if let Some(&pt) = enum_payload_tags.get(&sv) {
                                enum_payload_tags.insert(darg, pt);
                            }
                            if let Some(&vt) = tags.get(&sv) {
                                merge_local_tag(&mut tags, darg, vt);
                            }
                            if let Some(sft) = struct_field_tags.get(&sv).cloned() {
                                struct_field_tags.insert(darg, sft);
                            }
                            if let Some(&mv) = max_u64_vals.get(&sv) {
                                max_u64_vals
                                    .entry(darg)
                                    .and_modify(|cur| *cur = (*cur).max(mv))
                                    .or_insert(mv);
                            }
                            if let Some(&mi) = max_idesc_vals.get(&sv) {
                                max_idesc_vals
                                    .entry(darg)
                                    .and_modify(|cur| {
                                        let s1 = decode_tcgen05_idesc_shape(*cur, false)
                                            .unwrap_or((0, 0));
                                        let s2 =
                                            decode_tcgen05_idesc_shape(mi, false).unwrap_or((0, 0));
                                        if (s2.0, s2.1) > (s1.0, s1.1) {
                                            *cur = mi;
                                        }
                                    })
                                    .or_insert(mi);
                            }
                        }
                    }
                    for i in 0..d1.deref(ctx).get_num_arguments() {
                        if 1 + n0 + i < op.get_num_operands() {
                            let sv = op.get_operand(1 + n0 + i);
                            let darg = d1.deref(ctx).get_argument(i);
                            if let Some(&pt) = enum_payload_tags.get(&sv) {
                                enum_payload_tags.insert(darg, pt);
                            }
                            if let Some(&vt) = tags.get(&sv) {
                                merge_local_tag(&mut tags, darg, vt);
                            }
                            if let Some(sft) = struct_field_tags.get(&sv).cloned() {
                                struct_field_tags.insert(darg, sft);
                            }
                            if let Some(&mv) = max_u64_vals.get(&sv) {
                                max_u64_vals
                                    .entry(darg)
                                    .and_modify(|cur| *cur = (*cur).max(mv))
                                    .or_insert(mv);
                            }
                            if let Some(&mi) = max_idesc_vals.get(&sv) {
                                max_idesc_vals
                                    .entry(darg)
                                    .and_modify(|cur| {
                                        let s1 = decode_tcgen05_idesc_shape(*cur, false)
                                            .unwrap_or((0, 0));
                                        let s2 =
                                            decode_tcgen05_idesc_shape(mi, false).unwrap_or((0, 0));
                                        if (s2.0, s2.1) > (s1.0, s1.1) {
                                            *cur = mi;
                                        }
                                    })
                                    .or_insert(mi);
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Assign global `opaque_ids` and partition SSA values into persistent Tier 1 (`cross_vals`)
    //    vs. block-local Tier 2 temporaries (`val_ids`) after Step 4 has populated `tags`.
    //
    // Cross-block register lifetime invariant:
    // - Tier 1 (`0..local_base_vid`): globally unique persistent registers (entry/block args,
    //   `zero_slot_vid`, `enum_payload_ids`, and `cross_vals`), preserved across block boundaries.
    // - Tier 2 (`local_base_vid..max_local_vid`): per-block recycled temporaries, valid only
    //   inside the defining basic block.
    // - Tier 3 (`max_local_vid..next_var_id`): globally unique synthetic temporaries (`c4_vid`,
    //   `sum_vid`, `ovf_vid`) allocated during statement emission.
    // Every SSA value whose `val_ids[&v]` is persisted across blocks in a side table
    // (`struct_fields`, `alloca_vars`, `bar_slot_vars`, `enum_payload_tags` via `mir.construct_enum`,
    // or `LocalTag` wrapper IDs) must be in Tier 1 (`cross_vals`), and register-keyed side tables
    // (`smem_desc_ready`) must evict Tier-2 IDs at block boundaries.
    let mut enum_payload_ids: HashMap<Value, usize> = HashMap::new();
    let mut opaque_ids: HashMap<Value, usize> = HashMap::new();
    let mut next_opaque_id = 0usize;
    let is_enum_val = |v: Value| -> bool {
        v.get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirEnumType>()
            .is_some()
    };
    let is_aggregate_val = |v: Value| -> bool {
        let ty = v.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        ty_ref.downcast_ref::<MirStructType>().is_some()
            || ty_ref.downcast_ref::<MirTupleType>().is_some()
            || ty_ref.downcast_ref::<MirArrayType>().is_some()
    };

    let mut def_block: HashMap<Value, usize> = HashMap::new();
    let mut cross_vals: HashSet<Value> = HashSet::new();
    for &old_idx in &rpo_old {
        let blk = graph.blocks[old_idx].deref(ctx);
        for arg in blk.arguments() {
            def_block.insert(arg, old_idx);
            cross_vals.insert(arg);
            opaque_ids.entry(arg).or_insert_with(|| {
                let id = next_opaque_id;
                next_opaque_id += 1;
                id
            });
        }
        for op_ptr in blk.iter(ctx) {
            let op = op_ptr.deref(ctx);
            for r_idx in 0..op.get_num_results() {
                let res = op.get_result(r_idx);
                def_block.insert(res, old_idx);
                opaque_ids.entry(res).or_insert_with(|| {
                    let id = next_opaque_id;
                    next_opaque_id += 1;
                    id
                });
                if is_enum_val(res) || is_aggregate_val(res) {
                    cross_vals.insert(res);
                }
            }
        }
    }
    for &old_idx in &rpo_old {
        let blk = graph.blocks[old_idx].deref(ctx);
        for op_ptr in blk.iter(ctx) {
            let name = op_name(ctx, op_ptr);
            let is_term = matches!(name.as_str(), "mir.goto" | "mir.cond_br");
            let op = op_ptr.deref(ctx);
            let promotes_all_operands = is_term
                || matches!(
                    name.as_str(),
                    "mir.construct_struct"
                        | "mir.construct_tuple"
                        | "mir.construct_array"
                        | "mir.construct_enum"
                        | "mir.insert_field"
                );
            let promotes_rhs_operand = name == "mir.store"
                || (name == "mir.ptr_offset"
                    && op.get_num_operands() >= 2
                    && matches!(
                        tags.get(&op.get_operand(0)),
                        Some(LocalTag::BarrierPtr {
                            shared_opt: Some(_),
                            ..
                        })
                    ));
            for o_idx in 0..op.get_num_operands() {
                let ov = op.get_operand(o_idx);
                if promotes_all_operands
                    || (promotes_rhs_operand && o_idx == 1)
                    || def_block.get(&ov).copied() != Some(old_idx)
                {
                    cross_vals.insert(ov);
                }
            }
        }
    }

    let mut zero_slot_vid = 0usize;
    for &old_idx in &rpo_old {
        let blk = graph.blocks[old_idx].deref(ctx);
        for arg in blk.arguments() {
            val_ids.entry(arg).or_insert_with(|| {
                let id = next_var_id;
                next_var_id += 1;
                id
            });
            if old_idx != 0 && is_enum_val(arg) {
                let pid = next_var_id;
                next_var_id += 1;
                enum_payload_ids.insert(arg, pid);
            }
        }
        if old_idx == 0 {
            zero_slot_vid = next_var_id;
            next_var_id += 1;
        }
        for op_ptr in blk.iter(ctx) {
            let op = op_ptr.deref(ctx);
            for r_idx in 0..op.get_num_results() {
                let res = op.get_result(r_idx);
                if cross_vals.contains(&res) {
                    val_ids.entry(res).or_insert_with(|| {
                        let id = next_var_id;
                        next_var_id += 1;
                        id
                    });
                    if is_enum_val(res) {
                        let pid = next_var_id;
                        next_var_id += 1;
                        enum_payload_ids.insert(res, pid);
                    }
                }
            }
        }
    }
    let local_base_vid = next_var_id;
    let mut max_local_vid = local_base_vid;
    for &old_idx in &rpo_old {
        let blk = graph.blocks[old_idx].deref(ctx);
        let mut cur_local = local_base_vid;
        for op_ptr in blk.iter(ctx) {
            let op = op_ptr.deref(ctx);
            for r_idx in 0..op.get_num_results() {
                let res = op.get_result(r_idx);
                if let std::collections::hash_map::Entry::Vacant(entry) = val_ids.entry(res) {
                    entry.insert(cur_local);
                    cur_local += 1;
                }
            }
        }
        if cur_local > max_local_vid {
            max_local_vid = cur_local;
        }
    }
    next_var_id = max_local_vid;

    // Compute tight worst-case `mbarrier_arrive_expect_tx` byte counts per `Barrier` ID across reachable blocks,
    // and identify loop-header arguments initialized to `0` or `1` on forward entry edges.
    let mut arrived_tx_bytes_per_bar: HashMap<usize, u64> = HashMap::new();
    for &old_b in &rpo_old {
        for op_ptr in graph.blocks[old_b].deref(ctx).iter(ctx) {
            let on = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            if matches!(
                on.as_str(),
                "nvvm.mbarrier_arrive_expect_tx_shared" | "nvvm.mbarrier_arrive_expect_tx_cluster"
            ) {
                let (Some(LocalTag::BarrierPtr { bar_id, .. }), Some(&tx_bytes)) = (
                    op.operands().next().and_then(|v| tags.get(&v).copied()),
                    op.operands().nth(1).and_then(|v| max_u64_vals.get(&v)),
                ) else {
                    return Err(reject(
                        &on,
                        "mbarrier_arrive_expect_tx requires a static Barrier pointer and constant byte count",
                    ));
                };
                if tx_bytes > 0 {
                    arrived_tx_bytes_per_bar
                        .entry(bar_id)
                        .and_modify(|cur| *cur = (*cur).max(tx_bytes))
                        .or_insert(tx_bytes);
                }
            }
        }
    }

    let mut hdr_arg_entry_const: HashMap<Value, u64> = HashMap::new();
    for &hdr_new in &loop_headers {
        let hdr_old = rpo_old[hdr_new];
        let hdr_ptr = graph.blocks[hdr_old];
        for (arg_pos, arg) in hdr_ptr.deref(ctx).arguments().enumerate() {
            let mut init_c: Option<u64> = None;
            let mut consistent = true;
            for (src_new, &src_old) in rpo_old.iter().enumerate() {
                if src_new >= hdr_new {
                    continue;
                }
                for op_ptr in graph.blocks[src_old].deref(ctx).iter(ctx) {
                    let on = op_name(ctx, op_ptr);
                    let op = op_ptr.deref(ctx);
                    let passed_opt = if on == "mir.goto"
                        && op.get_num_successors() == 1
                        && op.get_successor(0) == hdr_ptr
                        && arg_pos < op.get_num_operands()
                    {
                        Some(op.get_operand(arg_pos))
                    } else if on == "mir.cond_br" && op.get_num_successors() == 2 {
                        let tl = op.get_successor(0).deref(ctx).get_num_arguments();
                        if op.get_successor(0) == hdr_ptr && 1 + arg_pos < op.get_num_operands() {
                            Some(op.get_operand(1 + arg_pos))
                        } else if op.get_successor(1) == hdr_ptr
                            && 1 + tl + arg_pos < op.get_num_operands()
                        {
                            Some(op.get_operand(1 + tl + arg_pos))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(pv) = passed_opt {
                        match (init_c, const_u64_vals.get(&pv).copied()) {
                            (None, Some(c)) => init_c = Some(c),
                            (Some(prev), Some(c)) if prev == c => {}
                            _ => consistent = false,
                        }
                    }
                }
            }
            if consistent && let Some(c) = init_c {
                hdr_arg_entry_const.insert(arg, c);
            }
        }
    }

    // Precompute TMA destination pointers and constant `mir.ptr_offset` sub-panel steps per `bar_id`.
    let mut tma_dests_per_bar: HashMap<usize, Vec<Value>> = HashMap::new();
    let mut ptr_const_offset: HashMap<Value, (Value, u64)> = HashMap::new();
    for &old_b in &rpo_old {
        for op_ptr in graph.blocks[old_b].deref(ctx).iter(ctx) {
            let on = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);
            if on == "mir.ptr_offset" && op.get_num_operands() == 2 && op.get_num_results() == 1 {
                if let Some(&off) = const_u64_vals.get(&op.get_operand(1)) {
                    ptr_const_offset.insert(op.get_result(0), (op.get_operand(0), off));
                }
            } else if matches!(on.as_str(), "mir.cast" | "mir.copy")
                && op.get_num_operands() == 1
                && op.get_num_results() == 1
                && let Some(&po) = ptr_const_offset.get(&op.get_operand(0))
            {
                ptr_const_offset.insert(op.get_result(0), po);
            } else if on.starts_with("nvvm.cp_async_bulk_tensor_g2s")
                && op.get_num_operands() >= 2
                && let Some(LocalTag::BarrierPtr { bar_id, .. }) =
                    tags.get(&op.get_operand(1)).copied()
            {
                tma_dests_per_bar
                    .entry(bar_id)
                    .or_default()
                    .push(op.get_operand(0));
            }
        }
    }

    let mut bar_slot_vars: HashMap<Value, usize> = HashMap::new();
    let mut pos_counted_lt: HashSet<Value> = HashSet::new();
    let mut pos_counted_ge: HashSet<Value> = HashSet::new();
    let mut slice_len_vals: HashMap<Value, usize> = HashMap::new();
    let mut idx1d_vals: HashSet<Value> = HashSet::new();
    let mut lt_slice_len: HashMap<Value, usize> = HashMap::new();
    let mut ge_slice_len: HashMap<Value, usize> = HashMap::new();
    let mut lt_const_bound: HashMap<Value, (Value, u64)> = HashMap::new();
    let mut coord_exprs: HashMap<Value, IdxExpr> = HashMap::new();
    let mut coord_guards: HashMap<Value, GuardExpr> = HashMap::new();
    let mut mbar_ready_bar: HashMap<Value, (usize, bool, CfgMbarWaitKind)> = HashMap::new();
    let mut mbar_not_ready_bar: HashMap<Value, (usize, bool, CfgMbarWaitKind)> = HashMap::new();
    let mut mir_blocks: Vec<MirBlockData> = Vec::with_capacity(rpo_old.len());
    let mut cfg_blocks: Vec<TranscribedCfgBlock> =
        vec![TranscribedCfgBlock::default(); rpo_old.len()];
    let mut accessed_params: BTreeSet<usize> = BTreeSet::new();
    let mut written_params: BTreeSet<usize> = BTreeSet::new();
    let mut accessed_shared: BTreeSet<usize> = BTreeSet::new();
    let mut has_cluster_rank_checks = false;
    let mut write_blocks_by_param: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut guard_entries_by_param: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut opt_entries_by_param: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut next_assert_id = 0usize;
    let mut ptr_assert_map: HashMap<Value, (usize, Ptr<Operation>)> = HashMap::new();
    let mut candidate_asserts: Vec<(usize, Ptr<Operation>)> = Vec::new();
    let mut struct_fields: HashMap<Value, Vec<(usize, Option<LocalTag>)>> = HashMap::new();
    let mut alloca_vars: HashMap<Value, Vec<(usize, Option<LocalTag>)>> = HashMap::new();
    let mut smem_desc_ready: HashSet<usize> = HashSet::new();
    let mut opt_disc_param: HashMap<Value, usize> = HashMap::new();
    let mut disc_vals: HashSet<Value> = HashSet::new();
    let mut opt_cond_true_param: HashMap<Value, usize> = HashMap::new();
    let mut opt_cond_false_param: HashMap<Value, usize> = HashMap::new();

    let expand_branch_args = |vals: &[Value],
                              val_ids: &HashMap<Value, usize>,
                              ep: &HashMap<Value, usize>|
     -> Vec<usize> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for v in vals {
            out.push(val_ids[v]);
            if let Some(&pid) = ep.get(v) {
                out.push(pid);
            }
        }
        out
    };

    let mut pending_asserts: Vec<PendingAssert> = Vec::new();

    // Capture `opaque_ids` directly so every fallback oracle ID is allocated from the
    // monotonic `opaque_ids` map (strictly increasing in RPO definition order) and can
    // never alias per-block `val_ids`.
    let fallback_rhs_for = |v: Value| -> MirRhs {
        let w = mir_uint_width(ctx, v.get_type(ctx));
        if w > 1 {
            MirRhs::OpaqueUint(opaque_ids[&v], w)
        } else {
            MirRhs::OpaqueVal
        }
    };

    for (new_b, &old_b) in rpo_old.iter().enumerate() {
        // Evict block-local Tier-2 register IDs (`local_base_vid..max_local_vid`) at block
        // boundaries so recycled temporary slots cannot falsely keep `smem_desc_ready` state.
        smem_desc_ready.retain(|&vid| vid < local_base_vid || vid >= max_local_vid);
        if graph.succs.iter().filter(|ss| ss.contains(&old_b)).count() != 1 {
            pending_asserts.clear();
        }
        let blk = graph.blocks[old_b].deref(ctx);
        let mut block_args: Vec<(usize, u32)> = Vec::new();
        for a in blk.arguments() {
            if let Some(&pid) = enum_payload_ids.get(&a) {
                block_args.push((val_ids[&a], 64));
                block_args.push((pid, 0));
            } else {
                let w = if old_b == 0
                    && let Some(MirParamKind::Scalar(pw)) = params.get(block_args.len()).copied()
                {
                    pw
                } else {
                    let raw = mir_uint_width(ctx, a.get_type(ctx));
                    if raw <= 1 { 0 } else { raw }
                };
                block_args.push((val_ids[&a], w));
            }
        }

        let mut stmts = Vec::new();
        if new_b == 0 {
            stmts.push(MirStmt::Assign(zero_slot_vid, MirRhs::ConstInt(64, 0)));
        }
        let mut term = MirTerminator::Unreachable;

        for (op_idx, op_ptr) in blk.iter(ctx).enumerate() {
            let opid = op_name(ctx, op_ptr);
            let op = op_ptr.deref(ctx);

            // Track consumer reads of `tcgen05_ld_*` results before `tcgen05_load_wait()`.
            let is_stack_bundle_store = opid == "mir.store"
                && op
                    .operands()
                    .next()
                    .is_some_and(|pv| matches!(tags.get(&pv), Some(LocalTag::StackSlot)));
            if !is_stack_bundle_store
                && !matches!(
                    opid.as_str(),
                    "mir.extract_field"
                        | "mir.construct_array"
                        | "mir.construct_struct"
                        | "mir.construct_tuple"
                )
            {
                for operand in op.operands() {
                    if let Some(LocalTag::TmemLoadedReg(ld_id)) = tags.get(&operand).copied() {
                        cfg_blocks[new_b].ops.push(CfgOpItem::TmemLoadUse { ld_id });
                    }
                }
            }

            if opid.starts_with("nvvm.tcgen05") && !spec.tmem_lifecycle {
                return Err(reject(
                    &opid,
                    "tcgen05 operations in a #[verify] kernel require `tmem_lifecycle`",
                ));
            }

            match opid.as_str() {
                "mir.storage_live" | "mir.storage_dead" | "mir.dbg_value" | "mir.dbg_declare"
                | "mir.nop" | "mir.unroll_hint" | "mir.nounroll_hint" | "nvvm.bar_warp_sync" => {}
                "mir.return" => {
                    term = MirTerminator::Ret;
                    cfg_blocks[new_b].is_return = true;
                }
                "mir.unreachable" => {
                    term = MirTerminator::Unreachable;
                }
                "mir.goto" if op.get_num_successors() == 1 => {
                    let Some(&dst) =
                        old_to_new.get(&graph.idx_of(op.get_successor(0)).unwrap_or(usize::MAX))
                    else {
                        term = MirTerminator::Unreachable;
                        continue;
                    };
                    let raw_ops: Vec<Value> = op.operands().collect();
                    let args = if dst == 0 {
                        raw_ops.iter().map(|v| val_ids[v]).collect()
                    } else {
                        expand_branch_args(&raw_ops, &val_ids, &enum_payload_ids)
                    };
                    term = MirTerminator::Goto(dst, args);
                    cfg_blocks[new_b].succs.push((CfgBranchCond::Always, dst));
                }
                "mir.cond_br" if op.get_num_operands() >= 1 && op.get_num_successors() == 2 => {
                    let cond_v = op.get_operand(0);
                    let t_old = graph.idx_of(op.get_successor(0)).unwrap_or(usize::MAX);
                    let f_old = graph.idx_of(op.get_successor(1)).unwrap_or(usize::MAX);
                    let t_len = op.get_successor(0).deref(ctx).get_num_arguments();
                    let t_raw: Vec<Value> = (0..t_len).map(|i| op.get_operand(1 + i)).collect();
                    let f_raw: Vec<Value> = ((1 + t_len)..op.get_num_operands())
                        .map(|i| op.get_operand(i))
                        .collect();
                    let t_args = expand_branch_args(&t_raw, &val_ids, &enum_payload_ids);
                    let f_args = expand_branch_args(&f_raw, &val_ids, &enum_payload_ids);
                    match (old_to_new.get(&t_old), old_to_new.get(&f_old)) {
                        (Some(&t_dst), Some(&f_dst)) => {
                            if let Some(&gp) = opt_cond_true_param.get(&cond_v) {
                                opt_entries_by_param.entry(gp).or_default().push(t_dst);
                            } else if let Some(&gp) = opt_cond_false_param.get(&cond_v) {
                                opt_entries_by_param.entry(gp).or_default().push(f_dst);
                            } else if let Some(&gp) = lt_slice_len.get(&cond_v) {
                                guard_entries_by_param.entry(gp).or_default().push(t_dst);
                            } else if let Some(&gp) = ge_slice_len.get(&cond_v) {
                                guard_entries_by_param.entry(gp).or_default().push(f_dst);
                            }
                            let is_cyc =
                                loop_headers.contains(&new_b) || t_dst <= new_b || f_dst <= new_b;
                            let br_cond = if let Some(&(bar_id, init_free, wk)) =
                                mbar_ready_bar.get(&cond_v)
                            {
                                (
                                    CfgBranchCond::MbarReady(bar_id, init_free, wk),
                                    CfgBranchCond::MbarSpin(bar_id),
                                )
                            } else if let Some(&(bar_id, init_free, wk)) =
                                mbar_not_ready_bar.get(&cond_v)
                            {
                                (
                                    CfgBranchCond::MbarSpin(bar_id),
                                    CfgBranchCond::MbarReady(bar_id, init_free, wk),
                                )
                            } else if loop_headers.contains(&new_b)
                                && pos_counted_lt.contains(&cond_v)
                                && !divergent_vals.contains(&cond_v)
                            {
                                (CfgBranchCond::Uniform, CfgBranchCond::PosLoopExit(t_dst))
                            } else if loop_headers.contains(&new_b)
                                && pos_counted_ge.contains(&cond_v)
                                && !divergent_vals.contains(&cond_v)
                            {
                                (CfgBranchCond::PosLoopExit(f_dst), CfgBranchCond::Uniform)
                            } else if is_cyc
                                && matches!(
                                    op_name_of_def(ctx, &graph, cond_v).as_deref(),
                                    Some("mir.lt" | "mir.ge" | "mir.not")
                                )
                                && !divergent_vals.contains(&cond_v)
                            {
                                (CfgBranchCond::Uniform, CfgBranchCond::Uniform)
                            } else if let Some(g) = coord_guards.get(&cond_v) {
                                (
                                    CfgBranchCond::Coord(g.clone()),
                                    CfgBranchCond::Coord(g.negate()),
                                )
                            } else if !divergent_vals.contains(&cond_v) && is_cyc {
                                (CfgBranchCond::Uniform, CfgBranchCond::Uniform)
                            } else if !lane_divergent_vals.contains(&cond_v) {
                                (CfgBranchCond::WarpUniform, CfgBranchCond::WarpUniform)
                            } else {
                                (CfgBranchCond::Unknown, CfgBranchCond::Unknown)
                            };
                            cfg_blocks[new_b].succs.push((br_cond.0, t_dst));
                            cfg_blocks[new_b].succs.push((br_cond.1, f_dst));
                            cfg_blocks[new_b].cond_var = Some(val_ids[&cond_v]);
                            term = MirTerminator::CondBr(
                                val_ids[&cond_v],
                                t_dst,
                                t_args,
                                f_dst,
                                f_args,
                            );
                        }
                        (Some(&t_dst), None) => {
                            cfg_blocks[new_b].succs.push((CfgBranchCond::Always, t_dst));
                            term = MirTerminator::Goto(t_dst, t_args);
                        }
                        (None, Some(&f_dst)) => {
                            cfg_blocks[new_b].succs.push((CfgBranchCond::Always, f_dst));
                            term = MirTerminator::Goto(f_dst, f_args);
                        }
                        (None, None) => term = MirTerminator::Unreachable,
                    }
                }
                "nvvm.barrier0" => {
                    stmts.push(MirStmt::SyncThreads);
                    cfg_blocks[new_b].ops.push(CfgOpItem::SyncThreads);
                }
                "nvvm.cluster_sync" | "nvvm.cluster_barrier" => {
                    stmts.push(MirStmt::SyncThreads);
                    cfg_blocks[new_b].ops.push(CfgOpItem::ClusterSync);
                }
                "nvvm.fence_proxy_async_shared_cta"
                | "nvvm.fence_mbarrier_init_release_cluster"
                | "nvvm.fence_acquire_shared_cluster_cluster"
                | "nvvm.fence_proxy_async_generic_acquire_shared_cluster_cluster"
                | "nvvm.fence_proxy_async_generic_release_shared_cta_cluster" => {
                    if opid == "nvvm.fence_proxy_async_shared_cta" {
                        cfg_blocks[new_b].ops.push(CfgOpItem::ProxyFenceAsyncShared);
                    }
                }
                "nvvm.mapa_shared_cluster"
                    if op.get_num_operands() == 2 && op.get_num_results() == 1 =>
                {
                    let (ptr_v, rank_v, res) =
                        (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    has_cluster_rank_checks = true;
                    stmts.push(MirStmt::ClusterRankCheck(val_ids[&rank_v]));
                    match tags.get(&ptr_v).copied() {
                        Some(LocalTag::SharedPtr(sid, sz) | LocalTag::DsmemSharedPtr(sid, sz)) => {
                            if !spec.no_oob_shared.contains(&sid) {
                                return Err(reject(
                                    &opid,
                                    "`map_shared_rank` target `SharedArray` must be listed in `no_oob_idx(...)`",
                                ));
                            }
                            tags.insert(res, LocalTag::DsmemSharedPtr(sid, sz));
                            if let Some(&pair) = ptr_assert_map.get(&ptr_v) {
                                ptr_assert_map.insert(res, pair);
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::Copy(val_ids[&ptr_v]),
                            ));
                        }
                        Some(btag @ LocalTag::BarrierPtr { .. }) => {
                            tags.insert(res, btag);
                            if let Some(&sv) = bar_slot_vars.get(&ptr_v) {
                                bar_slot_vars.insert(res, sv);
                            }
                            if let Some(&pair) = ptr_assert_map.get(&ptr_v) {
                                ptr_assert_map.insert(res, pair);
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::Copy(val_ids[&ptr_v]),
                            ));
                        }
                        _ => {
                            return Err(reject(
                                &opid,
                                "`map_shared_rank` requires a verified `SharedArray` or static `Barrier` pointer",
                            ));
                        }
                    }
                }
                "nvvm.mbarrier_init_shared" => {
                    let (
                        Some(bar_v),
                        Some(LocalTag::BarrierPtr { bar_id, shared_opt }),
                        Some(&count),
                    ) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1).and_then(|v| const_u64_vals.get(&v)),
                    )
                    else {
                        return Err(reject(
                            &opid,
                            "mbarrier_init requires a static Barrier pointer and constant arrival count",
                        ));
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    cfg_blocks[new_b]
                        .ops
                        .push(CfgOpItem::MbarInit { bar_id, count });
                }
                "nvvm.mbarrier_inval_shared" => {
                    let (Some(bar_v), Some(LocalTag::BarrierPtr { bar_id, shared_opt })) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                    ) else {
                        return Err(reject(
                            &opid,
                            "mbarrier_inval requires a static Barrier pointer",
                        ));
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    cfg_blocks[new_b].ops.push(CfgOpItem::MbarInval { bar_id });
                }
                "nvvm.mbarrier_arrive_shared"
                | "nvvm.mbarrier_arrive_cluster"
                | "nvvm.mbarrier_arrive_cluster_relaxed" => {
                    let (Some(bar_v), Some(bar_tag)) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                    ) else {
                        return Err(reject(
                            &opid,
                            "mbarrier_arrive requires a static Barrier pointer",
                        ));
                    };
                    let (bar_id, shared_opt, is_prechecked_addr) = match bar_tag {
                        LocalTag::BarrierPtr { bar_id, shared_opt } => (bar_id, shared_opt, false),
                        LocalTag::ClusterBarrierAddr { bar_id, shared_opt } => {
                            (bar_id, shared_opt, true)
                        }
                        _ => {
                            return Err(reject(
                                &opid,
                                "mbarrier_arrive requires a static Barrier pointer",
                            ));
                        }
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if !is_prechecked_addr {
                            if let Some(pair) = ptr_assert_map.remove(&bar_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                        }
                    }
                    let slot_vid = if shared_opt.is_some() {
                        bar_slot_vars.get(&bar_v).copied().ok_or_else(|| {
                            reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` pointer requires a resolved element slot index",
                            )
                        })?
                    } else {
                        zero_slot_vid
                    };
                    let is_cluster = opid != "nvvm.mbarrier_arrive_shared";
                    let is_relaxed = opid == "nvvm.mbarrier_arrive_cluster_relaxed";
                    cfg_blocks[new_b].ops.push(CfgOpItem::MbarArrive {
                        bar_id,
                        is_cluster,
                        is_relaxed,
                    });
                    stmts.push(MirStmt::MbarArrive(bar_id, slot_vid, !is_relaxed));
                    if op.get_num_results() == 1 {
                        stmts.push(MirStmt::Assign(
                            val_ids[&op.get_result(0)],
                            MirRhs::OpaqueVal,
                        ));
                    }
                }
                "nvvm.mbarrier_arrive_expect_tx_shared"
                | "nvvm.mbarrier_arrive_expect_tx_cluster" => {
                    let (
                        Some(bar_v),
                        Some(LocalTag::BarrierPtr { bar_id, shared_opt }),
                        Some(&tx_bytes),
                    ) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1).and_then(|v| max_u64_vals.get(&v)),
                    )
                    else {
                        return Err(reject(
                            &opid,
                            "mbarrier_arrive_expect_tx requires a static Barrier pointer and constant byte count",
                        ));
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    let slot_vid = if shared_opt.is_some() {
                        bar_slot_vars.get(&bar_v).copied().ok_or_else(|| {
                            reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` pointer requires a resolved element slot index",
                            )
                        })?
                    } else {
                        zero_slot_vid
                    };
                    let is_cluster = opid == "nvvm.mbarrier_arrive_expect_tx_cluster";
                    cfg_blocks[new_b].ops.push(CfgOpItem::MbarArriveExpectTx {
                        bar_id,
                        tx_bytes,
                        is_cluster,
                    });
                    stmts.push(MirStmt::MbarArriveExpectTx(bar_id, slot_vid, tx_bytes));
                    if op.get_num_results() == 1 {
                        stmts.push(MirStmt::Assign(
                            val_ids[&op.get_result(0)],
                            MirRhs::OpaqueVal,
                        ));
                    }
                }
                "nvvm.mbarrier_try_wait_parity_shared"
                | "nvvm.mbarrier_try_wait_shared"
                | "nvvm.mbarrier_test_wait_shared"
                | "nvvm.mbarrier_try_wait_parity_cluster"
                | "nvvm.mbarrier_try_wait_parity_cluster_relaxed"
                    if op.get_num_results() == 1 =>
                {
                    let (Some(bar_v), Some(LocalTag::BarrierPtr { bar_id, shared_opt })) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                    ) else {
                        return Err(reject(
                            &opid,
                            "mbarrier_try_wait requires a static Barrier pointer",
                        ));
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    let slot_vid = if shared_opt.is_some() {
                        bar_slot_vars.get(&bar_v).copied().ok_or_else(|| {
                            reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` pointer requires a resolved element slot index",
                            )
                        })?
                    } else {
                        zero_slot_vid
                    };
                    let init_free = op.operands().nth(1).is_some_and(|pv| {
                        const_u64_vals.get(&pv) == Some(&1)
                            || hdr_arg_entry_const.get(&pv) == Some(&1)
                    });
                    let wait_kind = match opid.as_str() {
                        "nvvm.mbarrier_try_wait_parity_cluster" => CfgMbarWaitKind::ClusterAcquire,
                        "nvvm.mbarrier_try_wait_parity_cluster_relaxed" => {
                            CfgMbarWaitKind::ClusterRelaxed
                        }
                        _ => CfgMbarWaitKind::CtaAcquire,
                    };
                    let is_acquire = wait_kind != CfgMbarWaitKind::ClusterRelaxed;
                    let res = op.get_result(0);
                    mbar_ready_bar.insert(res, (bar_id, init_free, wait_kind));
                    stmts.push(MirStmt::Assign(
                        val_ids[&res],
                        MirRhs::MbarTryWait(bar_id, slot_vid, is_acquire),
                    ));
                }
                _ if opid.starts_with("nvvm.cp_async_bulk") => {
                    if !opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") {
                        return Err(reject(
                            &opid,
                            format!("unsupported async bulk operation `{opid}`"),
                        ));
                    }
                    let (
                        Some(smem_v),
                        Some(LocalTag::SharedPtr(sid, sz)),
                        Some(bar_v),
                        Some(LocalTag::BarrierPtr { bar_id, shared_opt }),
                    ) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1),
                        op.operands().nth(1).and_then(|v| tags.get(&v).copied()),
                    )
                    else {
                        return Err(reject(
                            &opid,
                            "`cp_async_bulk_tensor_g2s` requires a verified `SharedArray` destination and static `Barrier`",
                        ));
                    };
                    if let Some((bar_sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&bar_sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(bar_sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    let elem_bytes = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                    if elem_bytes == 0 {
                        return Err(reject(
                            &opid,
                            "cannot determine SharedArray element byte size for async bulk copy",
                        ));
                    }
                    accessed_shared.insert(sid);
                    if let Some(pair) = ptr_assert_map.remove(&smem_v)
                        && !candidate_asserts.contains(&pair)
                    {
                        candidate_asserts.push(pair);
                    }
                    let slot_vid = if shared_opt.is_some() {
                        bar_slot_vars.get(&bar_v).copied().ok_or_else(|| {
                            reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` pointer requires a resolved element slot index",
                            )
                        })?
                    } else {
                        zero_slot_vid
                    };
                    let is_cg2 = opid.ends_with("_cg2");
                    let is_multicast = opid.contains("multicast");
                    let bar_dests = tma_dests_per_bar.get(&bar_id);
                    let total_sites = bar_dests.map_or(1, Vec::len);
                    let sub_panel_step = ptr_const_offset
                        .get(&smem_v)
                        .filter(|&&(parent_v, off)| {
                            off > 0 && bar_dests.is_some_and(|ds| ds.contains(&parent_v))
                        })
                        .and_then(|&(parent_v, _)| {
                            bar_dests?
                                .iter()
                                .filter_map(|dv| {
                                    ptr_const_offset
                                        .get(dv)
                                        .filter(|&&(pv, o)| pv == parent_v && o > 0)
                                        .map(|&(_, o)| o)
                                })
                                .min()
                        });
                    let span_elems = arrived_tx_bytes_per_bar
                        .get(&bar_id)
                        .map_or(sz, |&tx| {
                            let per_cta_tx = if is_cg2 { tx / 2 } else { tx };
                            let per_call_tx = if let Some(step_elems) = sub_panel_step {
                                step_elems * elem_bytes
                            } else if is_cg2 && total_sites >= 2 {
                                per_cta_tx / 2
                            } else {
                                per_cta_tx
                            };
                            per_call_tx.div_ceil(elem_bytes)
                        })
                        .max(1);
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmaCopyG2S {
                        shared_id: sid,
                        bar_id,
                        is_cg2,
                        is_multicast,
                    });
                    stmts.push(MirStmt::TmaCopyG2S(
                        val_ids[&smem_v],
                        bar_id,
                        slot_vid,
                        span_elems,
                    ));
                }
                "nvvm.clc_try_cancel" | "nvvm.clc_try_cancel_multicast" => {
                    let (
                        Some(resp_v),
                        Some(LocalTag::SharedPtr(sid, _)),
                        Some(bar_v),
                        Some(LocalTag::BarrierPtr { bar_id, shared_opt }),
                    ) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1),
                        op.operands().nth(1).and_then(|v| tags.get(&v).copied()),
                    )
                    else {
                        return Err(reject(
                            &opid,
                            "`clc_try_cancel` requires a verified `SharedArray` response buffer and static `Barrier`",
                        ));
                    };
                    if !spec.no_oob_shared.contains(&sid) || !spec.no_data_race.contains(&sid) {
                        return Err(reject(
                            &opid,
                            "`clc_try_cancel` response buffer must be listed in `no_oob_idx` and `no_data_race`",
                        ));
                    }
                    if let Some((bar_sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&bar_sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(bar_sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    let elem_bytes = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                    if elem_bytes == 0 {
                        return Err(reject(
                            &opid,
                            "cannot determine element byte size for `clc_try_cancel` response buffer",
                        ));
                    }
                    accessed_shared.insert(sid);
                    if let Some(pair) = ptr_assert_map.remove(&resp_v)
                        && !candidate_asserts.contains(&pair)
                    {
                        candidate_asserts.push(pair);
                    }
                    let slot_vid = if shared_opt.is_some() {
                        bar_slot_vars.get(&bar_v).copied().ok_or_else(|| {
                            reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` pointer requires a resolved element slot index",
                            )
                        })?
                    } else {
                        zero_slot_vid
                    };
                    let tx_bytes = arrived_tx_bytes_per_bar
                        .get(&bar_id)
                        .copied()
                        .unwrap_or(16)
                        .max(16);
                    let span_elems = tx_bytes.div_ceil(elem_bytes).max(1);
                    let is_multicast = opid == "nvvm.clc_try_cancel_multicast";
                    cfg_blocks[new_b].ops.push(CfgOpItem::ClcTryCancel {
                        shared_id: sid,
                        bar_id,
                        is_multicast,
                    });
                    stmts.push(MirStmt::TmaCopyG2S(
                        val_ids[&resp_v],
                        bar_id,
                        slot_vid,
                        span_elems,
                    ));
                }
                "nvvm.clc_query_is_canceled"
                | "nvvm.clc_query_get_first_ctaid_x"
                | "nvvm.clc_query_get_first_ctaid_y"
                | "nvvm.clc_query_get_first_ctaid_z"
                    if op.get_num_results() == 1 =>
                {
                    let res = op.get_result(0);
                    let w = mir_uint_width(ctx, res.get_type(ctx));
                    let rhs = if w > 1 {
                        match op.operands().next() {
                            Some(v) => MirRhs::CtaUint1(opaque_ids[&res], w, val_ids[&v]),
                            None => MirRhs::OpaqueUint(opaque_ids[&res], w),
                        }
                    } else {
                        match op.operands().next() {
                            Some(v) => MirRhs::CtaUnary(val_ids[&v]),
                            None => MirRhs::CtaConst,
                        }
                    };
                    stmts.push(MirStmt::Assign(val_ids[&res], rhs));
                }
                "nvvm.cvta_generic_to_shared_offset"
                | "nvvm.cvta_generic_to_cluster_shared_offset"
                    if op.get_num_operands() == 1 && op.get_num_results() == 1 =>
                {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    match tags.get(&src).copied() {
                        Some(LocalTag::SharedPtr(sid, _)) => {
                            if !spec.no_oob_shared.contains(&sid)
                                || !spec.no_data_race.contains(&sid)
                            {
                                return Err(reject(
                                    &opid,
                                    "`cvta_generic_to_shared_offset` must target a `SharedArray` listed in `no_oob_idx` and `no_data_race`",
                                ));
                            }
                            tags.insert(res, LocalTag::SmemDescSource(sid));
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::CvtaShared(val_ids[&src]),
                            ));
                        }
                        Some(LocalTag::BarrierPtr { bar_id, shared_opt }) => {
                            if let Some((bar_sid, _)) = shared_opt {
                                if !spec.no_oob_shared.contains(&bar_sid) {
                                    return Err(reject(
                                        &opid,
                                        "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                                    ));
                                }
                                accessed_shared.insert(bar_sid);
                                if let Some(pair) = ptr_assert_map.remove(&src)
                                    && !candidate_asserts.contains(&pair)
                                {
                                    candidate_asserts.push(pair);
                                }
                                stmts.push(MirStmt::BarrierSlotCheck(val_ids[&src]));
                            }
                            if let Some(&sv) = bar_slot_vars.get(&src) {
                                bar_slot_vars.insert(res, sv);
                            }
                            tags.insert(res, LocalTag::ClusterBarrierAddr { bar_id, shared_opt });
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                        }
                        _ => {
                            return Err(reject(
                                &opid,
                                "`cvta_generic_to_shared_offset` must target a verified `SharedArray` or static `Barrier`",
                            ));
                        }
                    }
                }
                "nvvm.tcgen05_alloc" | "nvvm.tcgen05_alloc_cg2" => {
                    let is_cg2 = opid.ends_with("_cg2");
                    let (Some(ptr_v), Some(LocalTag::SharedPtr(sid, _)), Some(&cols)) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1).and_then(|v| const_u64_vals.get(&v)),
                    ) else {
                        return Err(reject(
                            &opid,
                            "`tcgen05_alloc` requires a verified `SharedArray<u32, ...>` slot and constant column count",
                        ));
                    };
                    if !spec.no_oob_shared.contains(&sid) || !spec.no_data_race.contains(&sid) {
                        return Err(reject(
                            &opid,
                            "`tcgen05_alloc` shared-memory address slot must be listed in `no_oob_idx` and `no_data_race`",
                        ));
                    }
                    accessed_shared.insert(sid);
                    if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                        && !candidate_asserts.contains(&pair)
                    {
                        candidate_asserts.push(pair);
                    }
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemAlloc {
                        shared_id: sid,
                        cols,
                        is_cg2,
                    });
                    stmts.push(MirStmt::TmemAlloc(val_ids[&ptr_v], cols));
                }
                "nvvm.tcgen05_dealloc" | "nvvm.tcgen05_dealloc_cg2" => {
                    let is_cg2 = opid.ends_with("_cg2");
                    let (Some(addr_v), Some(LocalTag::TmemAddr(_)), Some(&cols)) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                        op.operands().nth(1).and_then(|v| const_u64_vals.get(&v)),
                    ) else {
                        return Err(reject(
                            &opid,
                            "`tcgen05_dealloc` requires the base `TMEM` address loaded from `TMEM_ADDR` and a constant column count",
                        ));
                    };
                    cfg_blocks[new_b]
                        .ops
                        .push(CfgOpItem::TmemDealloc { cols, is_cg2 });
                    stmts.push(MirStmt::TmemAccess(val_ids[&addr_v], 128, cols, false));
                }
                "nvvm.tcgen05_relinquish_alloc_permit"
                | "nvvm.tcgen05_relinquish_alloc_permit_cg2" => {
                    let is_cg2 = opid.ends_with("_cg2");
                    cfg_blocks[new_b]
                        .ops
                        .push(CfgOpItem::TmemRelinquishPermit { is_cg2 });
                }
                _ if opid.starts_with("nvvm.tcgen05_mma")
                    || opid.starts_with("nvvm.tcgen05_cp") =>
                {
                    let Some(tmem_v) = op.operands().next() else {
                        return Err(reject(&opid, "missing TMEM operand"));
                    };
                    if !matches!(tags.get(&tmem_v), Some(LocalTag::TmemAddr(_))) {
                        return Err(reject(
                            &opid,
                            format!(
                                "`{opid}` destination operand must derive from an allocated `TMEM` base address"
                            ),
                        ));
                    }
                    let is_cg2 = opid.ends_with("_cg2");
                    let (height_lanes, width_cols) = if opid.starts_with("nvvm.tcgen05_cp") {
                        parse_tcgen05_cp_shape(&opid).ok_or_else(|| {
                            reject(&opid, format!("unrecognized `tcgen05_cp` shape `{opid}`"))
                        })?
                    } else {
                        let idesc_val = op
                            .operands()
                            .find_map(|v| {
                                const_u128_vals
                                    .get(&v)
                                    .or_else(|| max_idesc_vals.get(&v))
                                    .copied()
                                    .filter(|&c| c >= 256)
                            })
                            .ok_or_else(|| {
                                reject(
                                    &opid,
                                    format!(
                                        "`{opid}` requires a compile-time constant `Tcgen05InstructionDescriptor`"
                                    ),
                                )
                            })?;
                        decode_tcgen05_idesc_shape(idesc_val, is_cg2).ok_or_else(|| {
                            reject(
                                &opid,
                                format!(
                                    "`{opid}` instruction descriptor encodes zero M or N dimension"
                                ),
                            )
                        })?
                    };
                    let mut desc_rank = 0usize;
                    let mut smem_ids = Vec::new();
                    for (arg_pos, operand) in op.operands().enumerate().skip(1) {
                        if !operand
                            .get_type(ctx)
                            .deref(ctx)
                            .downcast_ref::<IntegerType>()
                            .is_some_and(|i| i.width() == 64)
                        {
                            continue;
                        }
                        let Some(LocalTag::SmemDescSource(sid)) = tags.get(&operand).copied()
                        else {
                            return Err(reject(
                                &opid,
                                format!(
                                    "`{opid}` 64-bit SMEM descriptor operand {arg_pos} must derive from `cvta_generic_to_shared_offset` on a verified `SharedArray`"
                                ),
                            ));
                        };
                        let elem_bytes = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                        if elem_bytes == 0 {
                            return Err(reject(
                                &opid,
                                "cannot determine element byte width for `tcgen05` SMEM descriptor",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if !smem_ids.contains(&sid) {
                            smem_ids.push(sid);
                        }
                        let tile_bytes = if opid.starts_with("nvvm.tcgen05_cp") {
                            height_lanes * width_cols * 4
                        } else if desc_rank == 0 {
                            height_lanes * 32
                        } else {
                            (if is_cg2 { width_cols / 2 } else { width_cols }) * 32
                        };
                        desc_rank += 1;
                        stmts.push(MirStmt::SmemDescRead(val_ids[&operand], tile_bytes));
                    }
                    cfg_blocks[new_b]
                        .ops
                        .push(CfgOpItem::TmemAsyncIssue { smem_ids, is_cg2 });
                    stmts.push(MirStmt::TmemAccess(
                        val_ids[&tmem_v],
                        height_lanes,
                        width_cols,
                        false,
                    ));
                }
                _ if opid.starts_with("nvvm.tcgen05_ld_") && op.get_num_results() >= 1 => {
                    let (height_lanes, width_cols) =
                        parse_tcgen05_ld_shape(&opid).ok_or_else(|| {
                            reject(&opid, format!("unrecognized `tcgen05_ld` shape `{opid}`"))
                        })?;
                    let Some(tmem_v) = op.operands().next() else {
                        return Err(reject(&opid, "missing TMEM address operand"));
                    };
                    if !matches!(tags.get(&tmem_v), Some(LocalTag::TmemAddr(_))) {
                        return Err(reject(
                            &opid,
                            "`tcgen05_ld` operand must derive from an allocated `TMEM` base address",
                        ));
                    }
                    let ld_id = tmem_ld_id_map.get(&(old_b, op_idx)).copied().unwrap_or(0);
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemLoad { ld_id });
                    stmts.push(MirStmt::TmemAccess(
                        val_ids[&tmem_v],
                        height_lanes,
                        width_cols,
                        true,
                    ));
                    for r_idx in 0..op.get_num_results() {
                        let res = op.get_result(r_idx);
                        tags.insert(res, LocalTag::TmemLoadedReg(ld_id));
                        stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                    }
                }
                "nvvm.tcgen05_load_wait" => {
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemLoadWait);
                }
                _ if opid.starts_with("nvvm.tcgen05_commit") => {
                    let (Some(bar_v), Some(LocalTag::BarrierPtr { bar_id, shared_opt })) = (
                        op.operands().next(),
                        op.operands().next().and_then(|v| tags.get(&v).copied()),
                    ) else {
                        return Err(reject(
                            &opid,
                            "tcgen05_commit requires a static Barrier pointer",
                        ));
                    };
                    if let Some((sid, _)) = shared_opt {
                        if !spec.no_oob_shared.contains(&sid) {
                            return Err(reject(
                                &opid,
                                "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                            ));
                        }
                        accessed_shared.insert(sid);
                        if let Some(pair) = ptr_assert_map.remove(&bar_v)
                            && !candidate_asserts.contains(&pair)
                        {
                            candidate_asserts.push(pair);
                        }
                        stmts.push(MirStmt::BarrierSlotCheck(val_ids[&bar_v]));
                    }
                    let is_cg2 = opid.ends_with("_cg2");
                    let is_multicast = opid.contains("multicast");
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemCommit {
                        bar_id,
                        is_cg2,
                        is_multicast,
                    });
                }
                "nvvm.tcgen05_store_wait" => {
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemStoreWait);
                }
                "nvvm.tcgen05_fence_before_thread_sync" => {
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemFenceBeforeSync);
                }
                "nvvm.tcgen05_fence_after_thread_sync" => {
                    cfg_blocks[new_b].ops.push(CfgOpItem::TmemFenceAfterSync);
                }
                _ if opid.starts_with("nvvm.tcgen05") => {
                    return Err(reject(
                        &opid,
                        format!("unsupported `tcgen05` operation `{opid}` in verified kernel"),
                    ));
                }
                "mir.assert" if op.get_num_operands() == 1 => {
                    let aid = next_assert_id;
                    next_assert_id += 1;
                    let cv = op.get_operand(0);
                    stmts.push(MirStmt::AssertCond(aid, val_ids[&cv]));
                    let slice_gp = lt_slice_len.get(&cv).copied();
                    let shared_bnd = lt_const_bound.get(&cv).copied();
                    if let Some(gp) = slice_gp {
                        pending_asserts.push(PendingAssert {
                            assert_id: aid,
                            op_ptr,
                            idx_val: cv,
                            slice_param: Some(gp),
                            shared_bound: None,
                        });
                    } else if let Some((idx_v, bnd)) = shared_bnd {
                        pending_asserts.push(PendingAssert {
                            assert_id: aid,
                            op_ptr,
                            idx_val: idx_v,
                            slice_param: None,
                            shared_bound: Some(bnd),
                        });
                    }
                }
                "mir.store" if op.get_num_operands() >= 2 => {
                    let (ptr_v, val_v) = (op.get_operand(0), op.get_operand(1));
                    if matches!(
                        tags.get(&val_v),
                        Some(
                            LocalTag::SliceParam(_, true)
                                | LocalTag::GlobalPtr(_)
                                | LocalTag::SharedPtr(_, _)
                                | LocalTag::DsmemSharedPtr(_, _)
                                | LocalTag::BarrierPtr { .. }
                                | LocalTag::ClusterBarrierAddr { .. }
                        )
                    ) {
                        return Err(reject(
                            "mir.store",
                            "verified slice, shared-memory, or mbarrier pointer cannot escape by being stored to memory",
                        ));
                    }
                    if let Some(&(alloca, fi)) = stack_field_ptrs.get(&ptr_v) {
                        let entry = alloca_vars.entry(alloca).or_default();
                        if entry.len() <= fi {
                            entry.resize(fi + 1, (val_ids[&val_v], None));
                        }
                        entry[fi] = (val_ids[&val_v], tags.get(&val_v).copied());
                    } else if matches!(tags.get(&ptr_v), Some(LocalTag::StackSlot)) {
                        if let Some(flds) = struct_fields.get(&val_v).cloned() {
                            alloca_vars.insert(ptr_v, flds);
                        } else {
                            alloca_vars
                                .insert(ptr_v, vec![(val_ids[&val_v], tags.get(&val_v).copied())]);
                        }
                    }
                    match tags.get(&ptr_v).copied() {
                        Some(LocalTag::StackSlot | LocalTag::UnverifiedPtr) => {}
                        Some(LocalTag::GlobalPtr(p)) => {
                            if matches!(params.get(p), Some(MirParamKind::Slice(false))) {
                                return Err(reject(
                                    "mir.store",
                                    format!(
                                        "cannot store through read-only slice parameter {p} (`&[T]`) in verified kernel `{kernel_name}`"
                                    ),
                                ));
                            }
                            let expected_eb = slice_elem_bytes.get(&p).copied().unwrap_or(0);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.store",
                                    format!(
                                        "pointer cast changed slice parameter {p} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_params.insert(p);
                            written_params.insert(p);
                            let wb = write_blocks_by_param.entry(p).or_default();
                            if !wb.contains(&new_b) {
                                wb.push(new_b);
                            }
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            stmts.push(MirStmt::Store(val_ids[&ptr_v], val_ids[&val_v]));
                        }
                        Some(LocalTag::SharedPtr(sid, _)) => {
                            let expected_eb = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.store",
                                    format!(
                                        "pointer cast changed SharedArray {sid} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_shared.insert(sid);
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            cfg_blocks[new_b]
                                .ops
                                .push(CfgOpItem::SharedWrite { shared_id: sid });
                            stmts.push(MirStmt::Store(val_ids[&ptr_v], val_ids[&val_v]));
                        }
                        Some(LocalTag::DsmemSharedPtr(sid, _)) => {
                            let expected_eb = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.store",
                                    format!(
                                        "pointer cast changed DSMEM SharedArray {sid} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_shared.insert(sid);
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            cfg_blocks[new_b]
                                .ops
                                .push(CfgOpItem::DsmemAccess { shared_id: sid });
                            stmts.push(MirStmt::Store(val_ids[&ptr_v], val_ids[&val_v]));
                        }
                        _ => {
                            return Err(reject(
                                "mir.store",
                                format!(
                                    "unmodeled raw-pointer store in verified kernel `{kernel_name}`"
                                ),
                            ));
                        }
                    }
                }
                "mir.alloca" if op.get_num_results() == 1 => {
                    let res = op.get_result(0);
                    tags.insert(res, LocalTag::StackSlot);
                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                }
                "mir.field_addr" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                    let base = op.get_operand(0);
                    let res = op.get_result(0);
                    if let Some(&t) = tags.get(&base) {
                        tags.insert(res, t);
                    }
                    if matches!(tags.get(&base), Some(LocalTag::BarrierPtr { .. })) {
                        if let Some(&sv) = bar_slot_vars.get(&base) {
                            bar_slot_vars.insert(res, sv);
                        }
                        if let Some(&pair) = ptr_assert_map.get(&base) {
                            ptr_assert_map.insert(res, pair);
                        }
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(val_ids[&base])));
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                    }
                }
                "mir.construct_struct" | "mir.construct_tuple" | "mir.construct_array"
                    if op.get_num_results() == 1 =>
                {
                    let res = op.get_result(0);
                    for ov in op.operands() {
                        if matches!(
                            tags.get(&ov),
                            Some(
                                LocalTag::SliceParam(_, true)
                                    | LocalTag::GlobalPtr(_)
                                    | LocalTag::SharedPtr(_, _)
                                    | LocalTag::DsmemSharedPtr(_, _)
                                    | LocalTag::BarrierPtr { .. }
                                    | LocalTag::ClusterBarrierAddr { .. }
                                    | LocalTag::ConflictingPtr
                            )
                        ) {
                            return Err(reject(
                                &opid,
                                "verified slice, shared-memory, or mbarrier pointer cannot escape into aggregate",
                            ));
                        }
                    }
                    let flds: Vec<(usize, Option<LocalTag>)> = op
                        .operands()
                        .map(|v| (val_ids[&v], tags.get(&v).copied()))
                        .collect();
                    if flds.iter().any(|(_, t)| {
                        matches!(
                            t,
                            Some(LocalTag::UnverifiedPtr | LocalTag::SliceParam(_, false))
                        )
                    }) {
                        tags.insert(res, LocalTag::UnverifiedPtr);
                    } else if let Some(&(
                        _,
                        Some(t @ (LocalTag::SmemDescSource(_) | LocalTag::TmemLoadedReg(_))),
                    )) = flds.first()
                    {
                        tags.insert(res, t);
                    }
                    struct_fields.insert(res, flds);
                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                }
                "mir.extract_array_element" | "mir.array_element_addr"
                    if op.get_num_operands() >= 1 && op.get_num_results() == 1 =>
                {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    if let Some(
                        t @ (LocalTag::TmemLoadedReg(_)
                        | LocalTag::StackSlot
                        | LocalTag::UnverifiedPtr),
                    ) = tags.get(&src).copied()
                    {
                        tags.insert(res, t);
                    }
                    if op.get_num_operands() >= 2 {
                        let idx_v = op.get_operand(1);
                        if let Some(pos) =
                            pending_asserts.iter().rposition(|pa| pa.idx_val == idx_v)
                        {
                            pending_asserts.remove(pos);
                        }
                    }
                    stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                }
                "mir.insert_field" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
                    let (base_s, fv, res) =
                        (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    let mut flds = struct_fields.get(&base_s).cloned().unwrap_or_default();
                    if let Some(idx) = attribute_field_index(&op.attributes, "insert_index")
                        .or_else(|| attribute_field_index(&op.attributes, "index"))
                        .or_else(|| attribute_field_index(&op.attributes, "field_index"))
                    {
                        let i = idx as usize;
                        if flds.len() <= i {
                            flds.resize(i + 1, (val_ids[&res], None));
                        }
                        flds[i] = (val_ids[&fv], tags.get(&fv).copied());
                    }
                    struct_fields.insert(res, flds);
                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                }
                "mir.construct_enum" if op.get_num_results() == 1 => {
                    let res = op.get_result(0);
                    let pid = enum_payload_ids[&res];
                    match attribute_field_index(&op.attributes, "construct_enum_variant_index") {
                        Some(0) => {
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::ConstInt(64, 0)));
                            stmts.push(MirStmt::Assign(pid, MirRhs::OpaqueVal));
                        }
                        Some(1) if op.get_num_operands() == 1 => {
                            let pay_v = op.get_operand(0);
                            if let Some(&t) = tags.get(&pay_v) {
                                enum_payload_tags.insert(res, t);
                            }
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::ConstInt(64, 1)));
                            stmts.push(MirStmt::Assign(pid, MirRhs::Copy(val_ids[&pay_v])));
                        }
                        _ => {
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                            stmts.push(MirStmt::Assign(pid, MirRhs::OpaqueVal));
                        }
                    }
                }
                "mir.get_discriminant"
                    if op.get_num_operands() == 1 && op.get_num_results() == 1 =>
                {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    disc_vals.insert(res);
                    if let Some(LocalTag::GlobalPtr(p)) = enum_payload_tags.get(&src).copied() {
                        opt_disc_param.insert(res, p);
                    }
                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(val_ids[&src])));
                }
                "mir.enum_payload" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    if let Some(&t) = enum_payload_tags.get(&src) {
                        tags.insert(res, t);
                    } else {
                        tags.insert(res, LocalTag::UnverifiedPtr);
                    }
                    if let Some(&pid) = enum_payload_ids.get(&src) {
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(pid)));
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                    }
                }
                "mir.constant" | "builtin.constant" if op.get_num_results() == 1 => {
                    let res = op.get_result(0);
                    let w = mir_uint_width(ctx, res.get_type(ctx));
                    let raw_int_w = res
                        .get_type(ctx)
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .map_or(0, IntegerType::width);
                    if let Some((_, val)) = decode_int_constant(&op, &opid) {
                        if w == 1 {
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::ConstBool(val != 0)));
                        } else if w > 1 || (raw_int_w > 1 && val < 128) {
                            let eff_w = if w > 1 { w } else { 64 };
                            stmts
                                .push(MirStmt::Assign(val_ids[&res], MirRhs::ConstInt(eff_w, val)));
                        } else {
                            stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                        }
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                    }
                }
                "mir.float_constant" if op.get_num_results() == 1 => {
                    stmts.push(MirStmt::Assign(
                        val_ids[&op.get_result(0)],
                        MirRhs::CtaConst,
                    ));
                }
                "nvvm.read_ptx_sreg_tid_x"
                | "nvvm.read_ptx_sreg_tid_y"
                | "nvvm.read_ptx_sreg_ctaid_x"
                | "nvvm.read_ptx_sreg_ctaid_y"
                | "nvvm.read_ptx_sreg_ntid_x"
                | "nvvm.read_ptx_sreg_ntid_y"
                | "nvvm.read_ptx_sreg_laneid"
                | "nvvm.read_ptx_sreg_cluster_ctaid_x"
                    if op.get_num_results() == 1 =>
                {
                    let res = op.get_result(0);
                    let reg = match opid.as_str() {
                        "nvvm.read_ptx_sreg_tid_x" => {
                            coord_exprs.insert(res, IdxExpr::Tx);
                            "tidX"
                        }
                        "nvvm.read_ptx_sreg_tid_y" => {
                            coord_exprs.insert(res, IdxExpr::Ty);
                            "tidY"
                        }
                        "nvvm.read_ptx_sreg_ctaid_x" => "ctaidX",
                        "nvvm.read_ptx_sreg_ctaid_y" => "ctaidY",
                        "nvvm.read_ptx_sreg_ntid_x" => "ntidX",
                        "nvvm.read_ptx_sreg_ntid_y" => "ntidY",
                        "nvvm.read_ptx_sreg_cluster_ctaid_x" => {
                            coord_exprs.insert(res, IdxExpr::ClusterCtaidX);
                            "clusterCtaidX"
                        }
                        _ => {
                            let lane_ce = if spec.block_y == 1 {
                                IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)
                            } else {
                                IdxExpr::Mod(
                                    Box::new(IdxExpr::Add(
                                        Box::new(IdxExpr::Mul(
                                            Box::new(IdxExpr::Ty),
                                            Box::new(IdxExpr::Const(u64::from(spec.block_size))),
                                        )),
                                        Box::new(IdxExpr::Tx),
                                    )),
                                    32,
                                )
                            };
                            coord_exprs.insert(res, lane_ce);
                            "laneId"
                        }
                    };
                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Sreg(reg)));
                }
                "nvvm.read_ptx_sreg_cluster_nctaid_x" if op.get_num_results() == 1 => {
                    let res = op.get_result(0);
                    stmts.push(MirStmt::Assign(
                        val_ids[&res],
                        MirRhs::ConstInt(32, u128::from(spec.cluster_dim_x)),
                    ));
                }
                "mir.call" => {
                    for arg_v in op.operands() {
                        if let Some(tag) = tags.get(&arg_v) {
                            match tag {
                                LocalTag::SliceParam(p, true) | LocalTag::GlobalPtr(p)
                                    if is_claimed_param(*p) =>
                                {
                                    return Err(reject(
                                        "mir.call",
                                        format!(
                                            "verified slice parameter {p} cannot escape into `mir.call`"
                                        ),
                                    ));
                                }
                                LocalTag::SharedPtr(sid, _) | LocalTag::DsmemSharedPtr(sid, _)
                                    if spec.no_oob_shared.contains(sid)
                                        || spec.no_data_race.contains(sid) =>
                                {
                                    return Err(reject(
                                        "mir.call",
                                        "verified shared-memory pointer cannot escape into unmodeled operation `mir.call`",
                                    ));
                                }
                                LocalTag::BarrierPtr { .. }
                                | LocalTag::ClusterBarrierAddr { .. }
                                | LocalTag::ConflictingPtr => {
                                    return Err(reject(
                                        "mir.call",
                                        "verified mbarrier pointer cannot escape into unmodeled operation `mir.call`",
                                    ));
                                }
                                LocalTag::TmemAddr(_) if spec.tmem_lifecycle => {
                                    return Err(reject(
                                        "mir.call",
                                        "verified TMEM address cannot escape into unmodeled operation `mir.call`",
                                    ));
                                }
                                _ => {}
                            }
                        }
                    }
                    let callee = get_string_attr(&op.attributes, "callee").unwrap_or_default();
                    let is_cuda_dev = callee.contains("11cuda_device")
                        || callee.starts_with("cuda_device__")
                        || callee.starts_with("cuda_device::");
                    let is_pure_core = callee.contains("4core")
                        || callee.starts_with("core__")
                        || callee.starts_with("core::");
                    let is_pure_intrinsic =
                        dialect_mir::rust_intrinsics::is_pure_total_scalar_placeholder(&callee);
                    if op.get_num_results() != 1
                        || !(is_cuda_dev || is_pure_core || is_pure_intrinsic)
                    {
                        return Err(reject(
                            "mir.call",
                            format!(
                                "unmodeled function call `{callee}` in verified kernel; inline the helper (`#[inline(always)]`) or remove `#[verify]`"
                            ),
                        ));
                    }
                    {
                        let res = op.get_result(0);
                        let res_vid = val_ids[&res];
                        if is_cuda_dev && callee.contains("index_1d") {
                            tags.insert(res, LocalTag::Index1DWrapper(res_vid));
                            idx1d_vals.insert(res);
                            stmts.push(MirStmt::Assign(res_vid, MirRhs::Index1D));
                        } else if is_cuda_dev && callee.contains("coord_2d_u32") {
                            stmts.push(MirStmt::Assign(res_vid, MirRhs::Coord2DU32));
                            struct_fields.insert(res, vec![(res_vid, None)]);
                        } else if is_cuda_dev && callee.contains("lane_id") {
                            let lane_ce = if spec.block_y == 1 {
                                IdxExpr::Mod(Box::new(IdxExpr::Tx), 32)
                            } else {
                                IdxExpr::Mod(
                                    Box::new(IdxExpr::Add(
                                        Box::new(IdxExpr::Mul(
                                            Box::new(IdxExpr::Ty),
                                            Box::new(IdxExpr::Const(u64::from(spec.block_size))),
                                        )),
                                        Box::new(IdxExpr::Tx),
                                    )),
                                    32,
                                )
                            };
                            coord_exprs.insert(res, lane_ce);
                            stmts.push(MirStmt::Assign(res_vid, MirRhs::Sreg("laneId")));
                        } else if is_cuda_dev
                            && callee.contains("Tcgen05SmemDescriptor")
                            && let Some((src_vid, sid)) = op.operands().find_map(|ov| {
                                if let Some(LocalTag::SmemDescSource(s)) = tags.get(&ov).copied() {
                                    if let Some(flds) = struct_fields.get(&ov)
                                        && let Some(&(f_vid, Some(LocalTag::SmemDescSource(fs)))) =
                                            flds.first()
                                    {
                                        return Some((f_vid, fs));
                                    }
                                    return Some((val_ids[&ov], s));
                                }
                                if let Some(&(f_vid, Some(LocalTag::SmemDescSource(fs)))) =
                                    alloca_vars.get(&ov).and_then(|fs| fs.first())
                                {
                                    return Some((f_vid, fs));
                                }
                                None
                            })
                        {
                            if smem_desc_ready.contains(&src_vid) {
                                stmts.push(MirStmt::Assign(res_vid, MirRhs::Copy(src_vid)));
                            } else {
                                let c4_vid = next_var_id;
                                next_var_id += 1;
                                stmts.push(MirStmt::Assign(c4_vid, MirRhs::ConstInt(64, 4)));
                                stmts.push(MirStmt::Assign(
                                    res_vid,
                                    MirRhs::BinOp("shr", 64, src_vid, c4_vid),
                                ));
                            }
                            smem_desc_ready.insert(res_vid);
                            tags.insert(res, LocalTag::SmemDescSource(sid));
                            struct_fields
                                .insert(res, vec![(res_vid, Some(LocalTag::SmemDescSource(sid)))]);
                        } else if is_cuda_dev
                            && (callee.contains("ThreadIndex") || callee.contains("Uniform"))
                            && callee.contains("3get")
                            && op.get_num_operands() == 1
                            && let Some((src_vid, wrap_tag)) = op.operands().next().and_then(|ov| {
                                if let Some(
                                    &t @ (LocalTag::Index1DWrapper(sv)
                                    | LocalTag::ScalarWrapper(sv)),
                                ) = tags.get(&ov)
                                {
                                    return Some((sv, t));
                                }
                                alloca_vars.get(&ov).and_then(|fs| fs.first()).and_then(
                                    |&(sv, t_opt)| match t_opt {
                                        Some(
                                            t @ (LocalTag::Index1DWrapper(_)
                                            | LocalTag::ScalarWrapper(_)),
                                        ) => Some((sv, t)),
                                        _ => None,
                                    },
                                )
                            })
                        {
                            if matches!(wrap_tag, LocalTag::Index1DWrapper(_)) {
                                idx1d_vals.insert(res);
                            }
                            stmts.push(MirStmt::Assign(res_vid, MirRhs::Copy(src_vid)));
                        } else if let Some(&pid) = enum_payload_ids.get(&res) {
                            stmts.push(MirStmt::Assign(
                                res_vid,
                                MirRhs::OpaqueUint(opaque_ids[&res], 64),
                            ));
                            stmts.push(MirStmt::Assign(pid, MirRhs::OpaqueVal));
                        } else {
                            if !matches!(tags.get(&res), Some(LocalTag::TmemLoadedReg(_))) {
                                tags.insert(res, LocalTag::UnverifiedPtr);
                            }
                            stmts.push(MirStmt::Assign(res_vid, fallback_rhs_for(res)));
                        }
                    }
                }
                "mir.extract_field" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    let fidx = attribute_field_index(&op.attributes, "index")
                        .or_else(|| attribute_field_index(&op.attributes, "field_index"));
                    match (tags.get(&src).copied(), fidx) {
                        (Some(LocalTag::SliceParam(p, _)), Some(0)) if is_claimed_param(p) => {
                            tags.insert(res, LocalTag::GlobalPtr(p));
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::SlicePtr(p)));
                        }
                        (Some(LocalTag::SliceParam(_, _)), Some(0)) => {
                            tags.insert(res, LocalTag::UnverifiedPtr);
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                        }
                        (Some(LocalTag::SliceParam(p, _)), Some(1)) => {
                            slice_len_vals.insert(res, p);
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::SliceLen(p)));
                        }
                        (Some(LocalTag::ScalarWrapper(src_id)), Some(0)) => {
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(src_id)));
                        }
                        (Some(LocalTag::Index1DWrapper(src_id)), Some(0)) => {
                            idx1d_vals.insert(res);
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(src_id)));
                        }
                        _ => {
                            if let (Some(flds), Some(idx)) = (struct_fields.get(&src), fidx)
                                && let Some(&(f_vid, f_tag)) = flds.get(idx as usize)
                            {
                                if let Some(t) = f_tag {
                                    tags.insert(res, t);
                                }
                                if smem_desc_ready.contains(&f_vid) {
                                    smem_desc_ready.insert(val_ids[&res]);
                                }
                                stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(f_vid)));
                            } else {
                                if matches!(tags.get(&src), Some(LocalTag::UnverifiedPtr)) {
                                    tags.insert(res, LocalTag::UnverifiedPtr);
                                } else if let Some(t @ LocalTag::TmemLoadedReg(_)) =
                                    tags.get(&src).copied()
                                {
                                    tags.insert(res, t);
                                }
                                let rhs = if matches!(
                                    tags.get(&src),
                                    Some(LocalTag::UnverifiedPtr | LocalTag::TmemLoadedReg(_))
                                ) {
                                    fallback_rhs_for(res)
                                } else {
                                    let w = mir_uint_width(ctx, res.get_type(ctx));
                                    if w > 1 {
                                        MirRhs::CtaUint1(opaque_ids[&res], w, val_ids[&src])
                                    } else {
                                        MirRhs::CtaUnary(val_ids[&src])
                                    }
                                };
                                stmts.push(MirStmt::Assign(val_ids[&res], rhs));
                            }
                        }
                    }
                }
                "mir.shared_alloc" if op.get_num_results() == 1 => {
                    let res = op.get_result(0);
                    let full = get_string_attr(&op.attributes, "source_name")
                        .unwrap_or_else(|| "SHARED".into());
                    let short = full.rsplit("::").next().unwrap_or(&full).to_string();
                    if is_mbarrier_shared_alloc(ctx, &op) {
                        bar_slot_vars.insert(res, zero_slot_vid);
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                    } else if barrier_shared_names.contains(&short) {
                        let sid = shared_decls
                            .iter()
                            .position(|(n, _, _)| n == &short)
                            .unwrap_or(0);
                        let sz = shared_decls.get(sid).map_or(0, |(_, s, _)| *s);
                        let bid = barrier_decls.iter().position(|n| n == &short).unwrap_or(0);
                        tags.insert(
                            res,
                            LocalTag::BarrierPtr {
                                bar_id: bid,
                                shared_opt: Some((sid, sz)),
                            },
                        );
                        bar_slot_vars.insert(res, zero_slot_vid);
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::SharedBase(sid)));
                    } else {
                        let sid = shared_decls
                            .iter()
                            .position(|(n, _, _)| n == &short)
                            .unwrap_or(0);
                        let sz = shared_decls.get(sid).map_or(0, |(_, s, _)| *s);
                        tags.insert(res, LocalTag::SharedPtr(sid, sz));
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::SharedBase(sid)));
                    }
                }
                "mir.cast" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    let cast_str = format!("{:?}", op.attributes);
                    if cast_str.contains("PointerExposeAddress") {
                        match tags.get(&src).copied() {
                            Some(LocalTag::SliceParam(p, _)) if is_claimed_param(p) => {
                                return Err(reject(
                                    "mir.cast",
                                    format!(
                                        "pointer provenance exposure (`expose_provenance`) in kernel `{kernel_name}` is not modeled on verified slice parameter {p}"
                                    ),
                                ));
                            }
                            Some(
                                LocalTag::GlobalPtr(_)
                                | LocalTag::SharedPtr(_, _)
                                | LocalTag::DsmemSharedPtr(_, _)
                                | LocalTag::ClusterBarrierAddr { .. }
                                | LocalTag::ConflictingPtr,
                            ) => {
                                return Err(reject(
                                    "mir.cast",
                                    format!(
                                        "pointer provenance exposure (`expose_provenance`) in kernel `{kernel_name}` is not modeled on verified slice or shared-memory pointers"
                                    ),
                                ));
                            }
                            Some(t @ (LocalTag::BarrierPtr { .. } | LocalTag::UnverifiedPtr)) => {
                                tags.insert(res, t);
                                if let Some(&sv) = bar_slot_vars.get(&src) {
                                    bar_slot_vars.insert(res, sv);
                                }
                                if let Some(&pair) = ptr_assert_map.get(&src) {
                                    ptr_assert_map.insert(res, pair);
                                }
                                stmts.push(MirStmt::Assign(
                                    val_ids[&res],
                                    MirRhs::Copy(val_ids[&src]),
                                ));
                            }
                            _ => {
                                stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                            }
                        }
                    } else if cast_str.contains("PointerWithExposedProvenance") {
                        if let Some(t @ LocalTag::BarrierPtr { .. }) = tags.get(&src).copied() {
                            tags.insert(res, t);
                            if let Some(&sv) = bar_slot_vars.get(&src) {
                                bar_slot_vars.insert(res, sv);
                            }
                            if let Some(&pair) = ptr_assert_map.get(&src) {
                                ptr_assert_map.insert(res, pair);
                            }
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(val_ids[&src])));
                        } else {
                            tags.insert(res, LocalTag::UnverifiedPtr);
                            stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                        }
                    } else if cast_str.contains("PtrToPtr")
                        || cast_str.contains("StaticAddress")
                        || cast_str.contains("PointerCoercionMutToConst")
                    {
                        let src_is_slice = {
                            let ty = src.get_type(ctx);
                            let r = ty.deref(ctx);
                            r.downcast_ref::<MirSliceType>().is_some()
                                || r.downcast_ref::<MirDisjointSliceType>().is_some()
                        };
                        let res_is_slice = {
                            let ty = res.get_type(ctx);
                            let r = ty.deref(ctx);
                            r.downcast_ref::<MirSliceType>().is_some()
                                || r.downcast_ref::<MirDisjointSliceType>().is_some()
                        };
                        // `as_ptr` / `as_mut_ptr`: transcribe slice-to-thin-pointer casts 1-to-1 into
                        // `MirRhs::SlicePtr(p)` (`GlobalPtr(p)`) on claimed parameters so Lean 4 verifies
                        // safe raw pointer / `get_unchecked` offsets and rejects out-of-bounds offsets.
                        if let Some(LocalTag::SliceParam(p, _)) = tags.get(&src).copied()
                            && src_is_slice
                            && !res_is_slice
                        {
                            if is_claimed_param(p) {
                                tags.insert(res, LocalTag::GlobalPtr(p));
                                stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::SlicePtr(p)));
                            } else {
                                tags.insert(res, LocalTag::UnverifiedPtr);
                                stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                            }
                        } else {
                            if let Some(&t) = tags.get(&src) {
                                tags.insert(res, t);
                            }
                            if let Some(av) = alloca_vars.get(&src).cloned() {
                                alloca_vars.insert(res, av);
                            }
                            if let Some(&sv) = bar_slot_vars.get(&src) {
                                bar_slot_vars.insert(res, sv);
                            }
                            if let Some(&pair) = ptr_assert_map.get(&src) {
                                ptr_assert_map.insert(res, pair);
                            }
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(val_ids[&src])));
                        }
                    } else if cast_str.contains("IntToInt") {
                        let sw = mir_uint_width(ctx, src.get_type(ctx));
                        let dw = mir_uint_width(ctx, res.get_type(ctx));
                        let dw_raw = res
                            .get_type(ctx)
                            .deref(ctx)
                            .downcast_ref::<IntegerType>()
                            .map_or(0, IntegerType::width);
                        if (sw > 1 || disc_vals.contains(&src))
                            && (dw > 1 || (disc_vals.contains(&src) && dw_raw > 1))
                        {
                            let eff_dw = if dw > 1 { dw } else { dw_raw };
                            if disc_vals.contains(&src) {
                                disc_vals.insert(res);
                            }
                            if let Some(&gp) = opt_disc_param.get(&src) {
                                opt_disc_param.insert(res, gp);
                            }
                            if idx1d_vals.contains(&src) && eff_dw >= sw {
                                idx1d_vals.insert(res);
                            }
                            if smem_desc_ready.contains(&val_ids[&src]) && eff_dw >= 32 {
                                smem_desc_ready.insert(val_ids[&res]);
                            }
                            if let Some(ce) = coord_exprs.get(&src).cloned() {
                                if eff_dw >= sw
                                    || (eff_dw < 64
                                        && ce
                                            .exclusive_upper_bound(spec.block_size, spec.block_y)
                                            .is_some_and(|ub| ub <= (1u64 << eff_dw)))
                                {
                                    coord_exprs.insert(res, ce);
                                } else if eff_dw < 64 {
                                    coord_exprs
                                        .insert(res, IdxExpr::Mod(Box::new(ce), 1u64 << eff_dw));
                                }
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::IntCast(eff_dw, val_ids[&src]),
                            ));
                        } else {
                            let rhs = if dw > 1 {
                                MirRhs::CtaUint1(opaque_ids[&res], dw, val_ids[&src])
                            } else {
                                MirRhs::CtaUnary(val_ids[&src])
                            };
                            stmts.push(MirStmt::Assign(val_ids[&res], rhs));
                        }
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                    }
                }
                "mir.ptr_offset" if op.get_num_operands() == 2 && op.get_num_results() == 1 => {
                    let (base_v, idx_v, res) =
                        (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    match tags.get(&base_v).copied() {
                        Some(LocalTag::GlobalPtr(p)) => {
                            let expected_eb = slice_elem_bytes.get(&p).copied().unwrap_or(0);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, base_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.ptr_offset",
                                    format!(
                                        "pointer cast changed slice parameter {p} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            tags.insert(res, LocalTag::GlobalPtr(p));
                            if spec.no_oob_idx.contains(&p)
                                && let Some(pos) = pending_asserts
                                    .iter()
                                    .rposition(|pa| pa.slice_param == Some(p))
                            {
                                let pa = pending_asserts.remove(pos);
                                ptr_assert_map.insert(res, (pa.assert_id, pa.op_ptr));
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::PtrOffset(val_ids[&base_v], val_ids[&idx_v]),
                            ));
                        }
                        Some(
                            stag @ (LocalTag::SharedPtr(sid, sz)
                            | LocalTag::DsmemSharedPtr(sid, sz)),
                        ) => {
                            let expected_eb = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, base_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.ptr_offset",
                                    format!(
                                        "pointer cast changed SharedArray {sid} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            tags.insert(res, stag);
                            if let Some(pos) = pending_asserts
                                .iter()
                                .rposition(|pa| pa.idx_val == idx_v && pa.shared_bound == Some(sz))
                            {
                                let pa = pending_asserts.remove(pos);
                                if spec.no_oob_shared.contains(&sid) {
                                    ptr_assert_map.insert(res, (pa.assert_id, pa.op_ptr));
                                }
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::PtrOffset(val_ids[&base_v], val_ids[&idx_v]),
                            ));
                        }
                        Some(
                            btag @ LocalTag::BarrierPtr {
                                shared_opt: Some((sid, sz)),
                                ..
                            },
                        ) => {
                            if !spec.no_oob_shared.contains(&sid) {
                                return Err(reject(
                                    "mir.ptr_offset",
                                    "`SharedArray<Barrier, ...>` must be listed in `no_oob_idx(...)`",
                                ));
                            }
                            tags.insert(res, btag);
                            bar_slot_vars.insert(res, val_ids[&idx_v]);
                            if let Some(pos) = pending_asserts
                                .iter()
                                .rposition(|pa| pa.idx_val == idx_v && pa.shared_bound == Some(sz))
                            {
                                let pa = pending_asserts.remove(pos);
                                ptr_assert_map.insert(res, (pa.assert_id, pa.op_ptr));
                            }
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::PtrOffset(val_ids[&base_v], val_ids[&idx_v]),
                            ));
                        }
                        _ => {
                            if matches!(tags.get(&base_v), Some(LocalTag::UnverifiedPtr)) {
                                tags.insert(res, LocalTag::UnverifiedPtr);
                            }
                            if let Some(pos) =
                                pending_asserts.iter().rposition(|pa| pa.idx_val == idx_v)
                            {
                                pending_asserts.remove(pos);
                            }
                            stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                        }
                    }
                }
                "mir.load" if op.get_num_operands() >= 1 && op.get_num_results() == 1 => {
                    let (ptr_v, res) = (op.get_operand(0), op.get_result(0));
                    match tags.get(&ptr_v).copied() {
                        Some(LocalTag::GlobalPtr(p)) => {
                            let expected_eb = slice_elem_bytes.get(&p).copied().unwrap_or(0);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.load",
                                    format!(
                                        "pointer cast changed slice parameter {p} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_params.insert(p);
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            let lw = mir_uint_width(ctx, res.get_type(ctx));
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::Load(val_ids[&ptr_v]),
                            ));
                            if lw > 1 {
                                stmts.push(MirStmt::Assign(
                                    val_ids[&res],
                                    MirRhs::OpaqueUint(opaque_ids[&res], lw),
                                ));
                            }
                        }
                        Some(LocalTag::SharedPtr(sid, _)) => {
                            let expected_eb = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.load",
                                    format!(
                                        "pointer cast changed SharedArray {sid} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_shared.insert(sid);
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            cfg_blocks[new_b]
                                .ops
                                .push(CfgOpItem::SharedRead { shared_id: sid });
                            if let Some(&cols) = tmem_slot_cols.get(&sid) {
                                tags.insert(res, LocalTag::TmemAddr(cols));
                                stmts.push(MirStmt::Assign(
                                    val_ids[&res],
                                    MirRhs::LoadTmemBase(val_ids[&ptr_v], cols),
                                ));
                            } else {
                                let lw = mir_uint_width(ctx, res.get_type(ctx));
                                stmts.push(MirStmt::Assign(
                                    val_ids[&res],
                                    MirRhs::Load(val_ids[&ptr_v]),
                                ));
                                if lw > 1 {
                                    let rhs = if spec.no_data_race.contains(&sid) {
                                        MirRhs::CtaUint1(opaque_ids[&res], lw, val_ids[&res])
                                    } else {
                                        MirRhs::OpaqueUint(opaque_ids[&res], lw)
                                    };
                                    stmts.push(MirStmt::Assign(val_ids[&res], rhs));
                                }
                            }
                        }
                        Some(LocalTag::DsmemSharedPtr(sid, _)) => {
                            let expected_eb = shared_decls.get(sid).map_or(0, |&(_, _, eb)| eb);
                            let actual_eb = ptr_pointee_elem_bytes(ctx, ptr_v.get_type(ctx));
                            if expected_eb > 0 && actual_eb != expected_eb {
                                return Err(reject(
                                    "mir.load",
                                    format!(
                                        "pointer cast changed DSMEM SharedArray {sid} element size from {expected_eb}B to {actual_eb}B"
                                    ),
                                ));
                            }
                            accessed_shared.insert(sid);
                            if let Some(pair) = ptr_assert_map.remove(&ptr_v)
                                && !candidate_asserts.contains(&pair)
                            {
                                candidate_asserts.push(pair);
                            }
                            cfg_blocks[new_b]
                                .ops
                                .push(CfgOpItem::DsmemAccess { shared_id: sid });
                            let lw = mir_uint_width(ctx, res.get_type(ctx));
                            stmts.push(MirStmt::Assign(
                                val_ids[&res],
                                MirRhs::Load(val_ids[&ptr_v]),
                            ));
                            if lw > 1 {
                                stmts.push(MirStmt::Assign(
                                    val_ids[&res],
                                    MirRhs::OpaqueUint(opaque_ids[&res], lw),
                                ));
                            }
                        }
                        _ => {
                            if let (Some(&sf), Some(sv_pair)) = (
                                stack_field_ptrs.get(&ptr_v),
                                stack_field_ptrs
                                    .get(&ptr_v)
                                    .and_then(|&(a, fi)| alloca_vars.get(&a)?.get(fi).copied()),
                            ) {
                                let _ = sf;
                                let (sv_id, s_tag) = sv_pair;
                                if let Some(t) = s_tag {
                                    tags.insert(res, t);
                                }
                                if smem_desc_ready.contains(&sv_id) {
                                    smem_desc_ready.insert(val_ids[&res]);
                                }
                                stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::Copy(sv_id)));
                            } else if matches!(tags.get(&ptr_v), Some(LocalTag::StackSlot))
                                && let Some(flds) = alloca_vars.get(&ptr_v).cloned()
                            {
                                if let Some(&(sv_id, s_tag)) = flds.first() {
                                    if let Some(t) = s_tag {
                                        tags.insert(res, t);
                                    }
                                    if smem_desc_ready.contains(&sv_id) {
                                        smem_desc_ready.insert(val_ids[&res]);
                                    }
                                    if flds.len() == 1 {
                                        stmts.push(MirStmt::Assign(
                                            val_ids[&res],
                                            MirRhs::Copy(sv_id),
                                        ));
                                    } else {
                                        stmts.push(MirStmt::Assign(
                                            val_ids[&res],
                                            MirRhs::OpaqueVal,
                                        ));
                                    }
                                } else {
                                    stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                                }
                                struct_fields.insert(res, flds);
                            } else if matches!(
                                tags.get(&ptr_v),
                                Some(LocalTag::StackSlot | LocalTag::UnverifiedPtr)
                            ) {
                                tags.insert(res, LocalTag::UnverifiedPtr);
                                stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                            } else {
                                return Err(reject(
                                    "mir.load",
                                    format!(
                                        "unmodeled raw-pointer load in verified kernel `{kernel_name}`"
                                    ),
                                ));
                            }
                        }
                    }
                }
                "mir.checked_add" | "mir.checked_sub" | "mir.checked_mul"
                    if op.get_num_operands() == 2 && op.get_num_results() == 1 =>
                {
                    let (l, r, res) = (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    let lw = mir_uint_width(ctx, l.get_type(ctx));
                    let rw = mir_uint_width(ctx, r.get_type(ctx));
                    if lw > 1 && rw > 1 {
                        let lean_op = match opid.as_str() {
                            "mir.checked_add" => "add",
                            "mir.checked_sub" => "sub",
                            _ => "mul",
                        };
                        let sum_vid = next_var_id;
                        next_var_id += 1;
                        let ovf_vid = next_var_id;
                        next_var_id += 1;
                        stmts.push(MirStmt::Assign(
                            sum_vid,
                            MirRhs::BinOp(lean_op, lw, val_ids[&l], val_ids[&r]),
                        ));
                        stmts.push(MirStmt::Assign(
                            ovf_vid,
                            MirRhs::CheckedOvf(lean_op, lw, val_ids[&l], val_ids[&r]),
                        ));
                        struct_fields.insert(res, vec![(sum_vid, None), (ovf_vid, None)]);
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                    } else {
                        return Err(reject(&opid, format!("unsupported `{opid}` operand types")));
                    }
                }
                "mir.fma" | "mir.neg" if op.get_num_results() == 1 => {
                    stmts.push(MirStmt::Assign(
                        val_ids[&op.get_result(0)],
                        MirRhs::OpaqueVal,
                    ));
                }
                "mir.add" | "mir.sub" | "mir.mul" | "mir.div" | "mir.rem" | "mir.bitand"
                | "mir.bitor" | "mir.bitxor" | "mir.shl" | "mir.shr"
                    if op.get_num_operands() == 2 && op.get_num_results() == 1 =>
                {
                    let (l, r, res) = (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    let w = mir_uint_width(ctx, res.get_type(ctx));
                    let r_uint_ok = mir_uint_width(ctx, r.get_type(ctx)) > 1
                        || ((opid == "mir.shl" || opid == "mir.shr")
                            && const_u64_vals.contains_key(&r));
                    if opid == "mir.bitand"
                        && let Some(btag @ LocalTag::BarrierPtr { .. }) =
                            tags.get(&l).copied().or_else(|| tags.get(&r).copied())
                        && (const_u64_vals.get(&l) == Some(&0xFEFF_FFF8)
                            || const_u64_vals.get(&r) == Some(&0xFEFF_FFF8))
                    {
                        let bar_src = if matches!(tags.get(&l), Some(LocalTag::BarrierPtr { .. })) {
                            l
                        } else {
                            r
                        };
                        tags.insert(res, btag);
                        if let Some(&sv) = bar_slot_vars.get(&bar_src) {
                            bar_slot_vars.insert(res, sv);
                        }
                        stmts.push(MirStmt::Assign(
                            val_ids[&res],
                            MirRhs::Copy(val_ids[&bar_src]),
                        ));
                    } else if w > 1 && mir_uint_width(ctx, l.get_type(ctx)) > 1 && r_uint_ok {
                        let lean_op = match opid.as_str() {
                            "mir.add" => "add",
                            "mir.sub" => "sub",
                            "mir.mul" => "mul",
                            "mir.div" => "div",
                            "mir.rem" => "rem",
                            "mir.bitand" => "bitAnd",
                            "mir.bitor" => "bitOr",
                            "mir.bitxor" => "bitXor",
                            "mir.shl" => "shl",
                            _ => "shr",
                        };
                        if (opid == "mir.shr"
                            && matches!(tags.get(&l), Some(LocalTag::SmemDescSource(_)))
                            && const_u64_vals.get(&r) == Some(&4))
                            || (matches!(opid.as_str(), "mir.add" | "mir.bitand" | "mir.bitor")
                                && (smem_desc_ready.contains(&val_ids[&l])
                                    || smem_desc_ready.contains(&val_ids[&r])))
                        {
                            smem_desc_ready.insert(val_ids[&res]);
                        }
                        let lc_opt = coord_exprs
                            .get(&l)
                            .cloned()
                            .or_else(|| const_u64_vals.get(&l).copied().map(IdxExpr::Const));
                        let rc_opt = coord_exprs
                            .get(&r)
                            .cloned()
                            .or_else(|| const_u64_vals.get(&r).copied().map(IdxExpr::Const));
                        let has_coord =
                            coord_exprs.contains_key(&l) || coord_exprs.contains_key(&r);
                        let derived_coord = match opid.as_str() {
                            "mir.add" if has_coord => lc_opt
                                .zip(rc_opt)
                                .map(|(a, b)| IdxExpr::Add(Box::new(a), Box::new(b))),
                            "mir.mul" if has_coord => lc_opt
                                .zip(rc_opt)
                                .map(|(a, b)| IdxExpr::Mul(Box::new(a), Box::new(b))),
                            "mir.div" => coord_exprs
                                .get(&l)
                                .cloned()
                                .zip(const_u64_vals.get(&r).copied())
                                .filter(|&(_, d)| d > 0)
                                .map(|(a, d)| IdxExpr::Div(Box::new(a), d)),
                            "mir.rem" => coord_exprs
                                .get(&l)
                                .cloned()
                                .zip(const_u64_vals.get(&r).copied())
                                .filter(|&(_, d)| d > 0)
                                .map(|(a, d)| IdxExpr::Mod(Box::new(a), d)),
                            "mir.shr" => coord_exprs
                                .get(&l)
                                .cloned()
                                .zip(const_u64_vals.get(&r).copied())
                                .filter(|&(_, sh)| sh <= 32)
                                .map(|(a, sh)| IdxExpr::Div(Box::new(a), 1u64 << sh)),
                            "mir.shl" => coord_exprs
                                .get(&l)
                                .cloned()
                                .zip(const_u64_vals.get(&r).copied())
                                .filter(|&(_, sh)| sh <= 32)
                                .map(|(a, sh)| {
                                    IdxExpr::Mul(Box::new(a), Box::new(IdxExpr::Const(1u64 << sh)))
                                }),
                            "mir.bitand" => coord_exprs
                                .get(&l)
                                .cloned()
                                .zip(const_u64_vals.get(&r).copied())
                                .or_else(|| {
                                    coord_exprs
                                        .get(&r)
                                        .cloned()
                                        .zip(const_u64_vals.get(&l).copied())
                                })
                                .filter(|&(_, m)| m < u64::MAX && (m + 1).is_power_of_two())
                                .map(|(a, m)| IdxExpr::Mod(Box::new(a), m + 1)),
                            _ => None,
                        };
                        if let Some(ce) = derived_coord.filter(|expr| {
                            w >= 64
                                || expr
                                    .exclusive_upper_bound(spec.block_size, spec.block_y)
                                    .is_some_and(|ub| ub <= (1u64 << w))
                        }) {
                            coord_exprs.insert(res, ce);
                        }
                        stmts.push(MirStmt::Assign(
                            val_ids[&res],
                            MirRhs::BinOp(lean_op, w, val_ids[&l], val_ids[&r]),
                        ));
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], fallback_rhs_for(res)));
                    }
                }
                "mir.lt" | "mir.le" | "mir.gt" | "mir.ge" | "mir.eq" | "mir.ne"
                    if op.get_num_operands() == 2 && op.get_num_results() == 1 =>
                {
                    let (l, r, res) = (op.get_operand(0), op.get_operand(1), op.get_result(0));
                    let l_ok = mir_uint_width(ctx, l.get_type(ctx)) > 1
                        || disc_vals.contains(&l)
                        || const_u64_vals.contains_key(&l);
                    let r_ok = mir_uint_width(ctx, r.get_type(ctx)) > 1
                        || disc_vals.contains(&r)
                        || const_u64_vals.contains_key(&r);
                    if l_ok && r_ok {
                        let lean_cmp = match opid.as_str() {
                            "mir.lt" => "lt",
                            "mir.le" => "le",
                            "mir.gt" => "gt",
                            "mir.ge" => "ge",
                            "mir.eq" => "eq",
                            _ => "ne",
                        };
                        let disc_pair = opt_disc_param
                            .get(&l)
                            .and_then(|&gp| const_u64_vals.get(&r).map(|&c| (gp, c)))
                            .or_else(|| {
                                opt_disc_param
                                    .get(&r)
                                    .and_then(|&gp| const_u64_vals.get(&l).map(|&c| (gp, c)))
                            });
                        if opid == "mir.lt" {
                            if let Some(&gp) = slice_len_vals.get(&r) {
                                lt_slice_len.insert(res, gp);
                            }
                            if let Some(&bnd) = const_u64_vals.get(&r) {
                                lt_const_bound.insert(res, (l, bnd));
                                if let Some(&c_init) = hdr_arg_entry_const.get(&l)
                                    && c_init < bnd
                                {
                                    pos_counted_lt.insert(res);
                                }
                            }
                        } else if opid == "mir.gt"
                            && let Some(&gp) = slice_len_vals.get(&l)
                        {
                            lt_slice_len.insert(res, gp);
                        } else if opid == "mir.ge"
                            && let Some(&gp) = slice_len_vals.get(&r)
                        {
                            ge_slice_len.insert(res, gp);
                        } else if opid == "mir.le"
                            && let Some(&gp) = slice_len_vals.get(&l)
                        {
                            ge_slice_len.insert(res, gp);
                        } else if opid == "mir.eq"
                            && let Some((gp, c)) = disc_pair
                        {
                            if c == 1 {
                                opt_cond_true_param.insert(res, gp);
                            } else if c == 0 {
                                opt_cond_false_param.insert(res, gp);
                            }
                        } else if opid == "mir.ne"
                            && let Some((gp, c)) = disc_pair
                        {
                            if c == 0 {
                                opt_cond_true_param.insert(res, gp);
                            } else if c == 1 {
                                opt_cond_false_param.insert(res, gp);
                            }
                        }
                        let lc_opt = coord_exprs
                            .get(&l)
                            .cloned()
                            .or_else(|| const_u64_vals.get(&l).copied().map(IdxExpr::Const));
                        let rc_opt = coord_exprs
                            .get(&r)
                            .cloned()
                            .or_else(|| const_u64_vals.get(&r).copied().map(IdxExpr::Const));
                        if (coord_exprs.contains_key(&l) || coord_exprs.contains_key(&r))
                            && let (Some(cl), Some(cr)) = (lc_opt, rc_opt)
                        {
                            let g = match opid.as_str() {
                                "mir.lt" => GuardExpr::Lt(cl, cr),
                                "mir.ge" => GuardExpr::Ge(cl, cr),
                                "mir.gt" => GuardExpr::Lt(cr, cl),
                                "mir.le" => GuardExpr::Ge(cr, cl),
                                "mir.eq" => GuardExpr::Eq(cl, cr),
                                _ => GuardExpr::Ne(cl, cr),
                            };
                            coord_guards.insert(res, g);
                        }
                        stmts.push(MirStmt::Assign(
                            val_ids[&res],
                            MirRhs::CmpOp(lean_cmp, val_ids[&l], val_ids[&r]),
                        ));
                    } else {
                        stmts.push(MirStmt::Assign(
                            val_ids[&res],
                            MirRhs::CtaBin(val_ids[&l], val_ids[&r]),
                        ));
                    }
                }
                "mir.not" if op.get_num_operands() == 1 && op.get_num_results() == 1 => {
                    let (src, res) = (op.get_operand(0), op.get_result(0));
                    if mir_uint_width(ctx, src.get_type(ctx)) == 1 {
                        if let Some(&bar_pair) = mbar_ready_bar.get(&src) {
                            mbar_not_ready_bar.insert(res, bar_pair);
                        } else if let Some(&bar_pair) = mbar_not_ready_bar.get(&src) {
                            mbar_ready_bar.insert(res, bar_pair);
                        }
                        if pos_counted_lt.contains(&src) {
                            pos_counted_ge.insert(res);
                        } else if pos_counted_ge.contains(&src) {
                            pos_counted_lt.insert(res);
                        }
                        if let Some(g) = coord_guards.get(&src) {
                            coord_guards.insert(res, g.negate());
                        }
                        if let Some(&gp) = lt_slice_len.get(&src) {
                            ge_slice_len.insert(res, gp);
                        } else if let Some(&gp) = ge_slice_len.get(&src) {
                            lt_slice_len.insert(res, gp);
                        }
                        if let Some(&gp) = opt_cond_true_param.get(&src) {
                            opt_cond_false_param.insert(res, gp);
                        } else if let Some(&gp) = opt_cond_false_param.get(&src) {
                            opt_cond_true_param.insert(res, gp);
                        }
                        stmts.push(MirStmt::Assign(
                            val_ids[&res],
                            MirRhs::NotBool(val_ids[&src]),
                        ));
                    } else {
                        stmts.push(MirStmt::Assign(val_ids[&res], MirRhs::OpaqueVal));
                    }
                }
                _ if (opid.starts_with("nvvm.shfl_sync") || opid.starts_with("nvvm.cvt_"))
                    && op.get_num_results() == 1 =>
                {
                    stmts.push(MirStmt::Assign(
                        val_ids[&op.get_result(0)],
                        MirRhs::OpaqueVal,
                    ));
                }
                _ => {
                    return Err(reject(
                        &opid,
                        format!("unsupported operation `{opid}` in verified kernel"),
                    ));
                }
            }
        }

        mir_blocks.push(MirBlockData {
            args: block_args,
            is_loop_header: loop_headers.contains(&new_b),
            stmts,
            term,
        });
    }

    // Fail closed if an obligation names a slice parameter or SharedArray that was never accessed / written.
    for &p in &spec.no_oob_idx {
        if !accessed_params.contains(&p) {
            return Err(reject(
                "no_oob_idx",
                format!("parameter {p} is listed in `no_oob_idx` but never accessed in the kernel"),
            ));
        }
    }
    for &p in &spec.disjoint_writes {
        if !written_params.contains(&p) {
            return Err(reject(
                "disjoint_writes",
                format!(
                    "parameter {p} is listed in `disjoint_writes` but never written in the kernel"
                ),
            ));
        }
    }
    for &sid in &spec.no_oob_shared {
        if !accessed_shared.contains(&sid) {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "shared array {sid} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
    }
    for &sid in &spec.no_data_race {
        if !accessed_shared.contains(&sid) {
            return Err(reject(
                "no_data_race",
                format!(
                    "shared array {sid} is listed in `no_data_race` but never accessed in the kernel"
                ),
            ));
        }
    }

    for &p in &spec.writes_all_elems {
        let g_entries = opt_entries_by_param
            .remove(&p)
            .filter(|v| !v.is_empty())
            .or_else(|| guard_entries_by_param.remove(&p))
            .unwrap_or_default();
        let w_blocks = write_blocks_by_param.remove(&p).unwrap_or_default();
        if g_entries.is_empty() || w_blocks.is_empty() {
            return Err(reject(
                "writes_all_elems",
                format!(
                    "parameter {p} is listed in `writes_all_elems` without a guarded `< len` write path"
                ),
            ));
        }
    }

    let protected_cfg_blocks: BTreeSet<usize> = BTreeSet::new();
    let mut control_flow_cfg = TranscribedControlFlowCfg {
        num_shared: shared_decls.len(),
        num_barriers: barrier_decls.len(),
        blocks: cfg_blocks,
    };
    let _ = control_flow_cfg.compact(&protected_cfg_blocks);

    let proven_assert_ops: Vec<Ptr<Operation>> =
        candidate_asserts.iter().map(|&(_, op)| op).collect();

    let race_free_shared: Vec<usize> = spec.no_data_race.to_vec();
    check_no_tier2_cross_block_escape(&mir_blocks, local_base_vid, max_local_vid, &reject)?;

    Ok(TranslatedFunction::MirProgram {
        arity,
        spec: Box::new(spec),
        program: Box::new(MirProgramData {
            params,
            shared_decls,
            race_free_shared,
            blocks: mir_blocks,
            candidate_asserts,
            control_flow_cfg,
            has_cluster_rank_checks,
        }),
        proven_assert_ops,
    })
}

/// Collect the source register IDs read by `stmt` using an exhaustive match over
/// [`MirStmt`] and [`MirRhs`], pushing each register into `out`.
fn push_stmt_read_regs(stmt: &MirStmt, out: &mut Vec<usize>) {
    match stmt {
        MirStmt::Assign(_, rhs) => match rhs {
            MirRhs::Copy(src)
            | MirRhs::NotBool(src)
            | MirRhs::IntCast(_, src)
            | MirRhs::Load(src)
            | MirRhs::CvtaShared(src)
            | MirRhs::LoadTmemBase(src, _)
            | MirRhs::CtaUint1(_, _, src)
            | MirRhs::CtaUnary(src) => out.push(*src),
            MirRhs::BinOp(_, _, l, r)
            | MirRhs::CheckedOvf(_, _, l, r)
            | MirRhs::CmpOp(_, l, r)
            | MirRhs::PtrOffset(l, r)
            | MirRhs::CtaBin(l, r) => {
                out.push(*l);
                out.push(*r);
            }
            MirRhs::MbarTryWait(_, slot, _) => out.push(*slot),
            MirRhs::ConstInt(_, _)
            | MirRhs::ConstBool(_)
            | MirRhs::Sreg(_)
            | MirRhs::Index1D
            | MirRhs::Coord2DU32
            | MirRhs::SlicePtr(_)
            | MirRhs::SliceLen(_)
            | MirRhs::SharedBase(_)
            | MirRhs::OpaqueUint(_, _)
            | MirRhs::CtaConst
            | MirRhs::OpaqueVal => {}
        },
        MirStmt::Store(p, v) => {
            out.push(*p);
            out.push(*v);
        }
        MirStmt::AssertCond(_, c) => out.push(*c),
        MirStmt::BarrierSlotCheck(r)
        | MirStmt::ClusterRankCheck(r)
        | MirStmt::MbarArrive(_, r, _)
        | MirStmt::MbarArriveExpectTx(_, r, _)
        | MirStmt::TmemAlloc(r, _)
        | MirStmt::SmemDescRead(r, _)
        | MirStmt::TmemAccess(r, _, _, _) => out.push(*r),
        MirStmt::TmaCopyG2S(p, _, s, _) => {
            out.push(*p);
            out.push(*s);
        }
        MirStmt::SyncThreads => {}
    }
}

/// Fail closed in both debug and release builds if any basic block reads an upward-exposed
/// Tier-2 register ID (`local_base_vid..max_local_vid`) defined in another block.
fn check_no_tier2_cross_block_escape(
    blocks: &[MirBlockData],
    local_base_vid: usize,
    max_local_vid: usize,
    reject: &dyn Fn(&str, String) -> crate::ExportError,
) -> Result<(), crate::ExportError> {
    let mut read_regs = Vec::new();
    for (b_idx, blk) in blocks.iter().enumerate() {
        let mut defined_here: HashSet<usize> =
            HashSet::with_capacity(blk.args.len() + blk.stmts.len());
        for &(vid, _) in &blk.args {
            if vid >= local_base_vid {
                return Err(reject(
                    "register_allocation",
                    format!("block {b_idx} argument {vid} must be in Tier 1 (< {local_base_vid})"),
                ));
            }
            defined_here.insert(vid);
        }
        for stmt in &blk.stmts {
            read_regs.clear();
            push_stmt_read_regs(stmt, &mut read_regs);
            for &r in &read_regs {
                if !defined_here.contains(&r) && r >= local_base_vid && r < max_local_vid {
                    return Err(reject(
                        "register_allocation",
                        format!(
                            "block {b_idx} reads upward-exposed Tier-2 register {r} (in {local_base_vid}..{max_local_vid})"
                        ),
                    ));
                }
            }
            if let MirStmt::Assign(dst, _) = stmt {
                defined_here.insert(*dst);
            }
        }
        read_regs.clear();
        match &blk.term {
            MirTerminator::Ret | MirTerminator::Unreachable => {}
            MirTerminator::Goto(_, args) => read_regs.extend_from_slice(args),
            MirTerminator::CondBr(c, _, ta, _, fa) => {
                read_regs.push(*c);
                read_regs.extend_from_slice(ta);
                read_regs.extend_from_slice(fa);
            }
        }
        for &r in &read_regs {
            if !defined_here.contains(&r) && r >= local_base_vid && r < max_local_vid {
                return Err(reject(
                    "register_allocation",
                    format!(
                        "block {b_idx} terminator reads upward-exposed Tier-2 register {r} (in {local_base_vid}..{max_local_vid})"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Find the defining operation name of `val` across `graph.blocks`, if any.
fn op_name_of_def(ctx: &Context, graph: &BlockGraph, val: Value) -> Option<String> {
    for &b_ptr in &graph.blocks {
        for op_ptr in b_ptr.deref(ctx).iter(ctx) {
            let op = op_ptr.deref(ctx);
            for r_idx in 0..op.get_num_results() {
                if op.get_result(r_idx) == val {
                    return Some(op_name(ctx, op_ptr));
                }
            }
        }
    }
    None
}
