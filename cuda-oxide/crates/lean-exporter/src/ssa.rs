/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed SSA lookup table (`SsaEnv`) for the scalar `BitVec` exporter plus
//! control-flow graph types (`TranscribedControlFlowCfg`, `GuardExpr`, `IdxExpr`)
//! emitted alongside `MirFunc` into `CudaOxide.Mir.ControlFlow`.

use std::collections::{BTreeSet, HashMap, VecDeque};

use pliron::value::Value;

use crate::{ExportError, reject};

/// Compositional symbolic coordinate expression over intra-CTA thread coordinates
/// `(tx, ty)` and loop step `i`, mirrored 1-to-1 in `CudaOxide.Mir.ControlFlow.IdxExpr`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IdxExpr {
    /// Thread X coordinate (`threadIdx.x`).
    Tx,
    /// Thread Y coordinate (`threadIdx.y`).
    Ty,
    /// CTA cluster X rank (`cluster_ctaidX()`).
    ClusterCtaidX,
    /// Loop induction step `i`.
    LoopVar,
    /// Unsigned integer constant literal.
    Const(u64),
    /// Sum `lhs + rhs`.
    Add(Box<IdxExpr>, Box<IdxExpr>),
    /// Product `lhs * rhs`.
    Mul(Box<IdxExpr>, Box<IdxExpr>),
    /// Euclidean quotient `lhs / divisor` (from `/` or `>> sh`).
    Div(Box<IdxExpr>, u64),
    /// Euclidean remainder `lhs % divisor` (from `%` or `& (2^k - 1)`).
    Mod(Box<IdxExpr>, u64),
}

impl IdxExpr {
    /// Return a strict exclusive upper bound `U` (`self < U`) over all valid
    /// `(tx < block_x, ty < block_y)`, or `None` when unbounded or overflowing `u64`.
    pub fn exclusive_upper_bound(&self, block_x: u32, block_y: u32) -> Option<u64> {
        match self {
            Self::Tx => Some(u64::from(block_x)),
            Self::Ty => Some(u64::from(block_y)),
            Self::ClusterCtaidX => Some(16),
            Self::LoopVar => None,
            Self::Const(c) => c.checked_add(1),
            Self::Mod(_, m) if *m > 0 => Some(*m),
            Self::Mod(_, _) => None,
            Self::Div(inner, d) if *d > 0 => {
                let ub = inner.exclusive_upper_bound(block_x, block_y)?;
                if ub == 0 {
                    Some(0)
                } else {
                    ((ub - 1) / *d).checked_add(1)
                }
            }
            Self::Div(_, _) => None,
            Self::Add(a, b) => {
                let ua = a.exclusive_upper_bound(block_x, block_y)?;
                let ub = b.exclusive_upper_bound(block_x, block_y)?;
                if ua == 0 || ub == 0 {
                    Some(0)
                } else {
                    (ua - 1).checked_add(ub - 1)?.checked_add(1)
                }
            }
            Self::Mul(a, b) => {
                let ua = a.exclusive_upper_bound(block_x, block_y)?;
                let ub = b.exclusive_upper_bound(block_x, block_y)?;
                if ua == 0 || ub == 0 {
                    Some(0)
                } else if ua == 1 || ub == 1 {
                    Some(1)
                } else {
                    (ua - 1).checked_mul(ub - 1)?.checked_add(1)
                }
            }
        }
    }
}

/// Compositional boolean branch condition guarding a basic block in `ControlFlowCfg`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GuardExpr {
    /// Unguarded (`True`).
    True,
    /// Strict upper bound `lhs < rhs` (e.g. `tid < 64`).
    Lt(IdxExpr, IdxExpr),
    /// Lower bound `lhs >= rhs` (from `!(lhs < rhs)` or `mir.ge`).
    Ge(IdxExpr, IdxExpr),
    /// Equality `lhs == rhs` (e.g. `lane == 0` or `warp_id == 0`).
    Eq(IdxExpr, IdxExpr),
    /// Inequality `lhs != rhs` (from `!(lhs == rhs)` or `mir.ne`).
    Ne(IdxExpr, IdxExpr),
    /// Conjunction of two dominating branch guards (`p /\ q`).
    And(Box<GuardExpr>, Box<GuardExpr>),
    /// Unmodeled or data-dependent branch condition (`if flag`, `if data[tid] > 0.0`).
    /// Poisons warp-collective and block-uniformity checks until reconvergence.
    Unknown,
}

impl GuardExpr {
    /// Logical negation of a branch condition (`mir.not` or false branch of `mir.cond_br`).
    pub fn negate(&self) -> Self {
        match self {
            Self::True => Self::True,
            Self::Lt(a, b) => Self::Ge(a.clone(), b.clone()),
            Self::Ge(a, b) => Self::Lt(a.clone(), b.clone()),
            Self::Eq(a, b) => Self::Ne(a.clone(), b.clone()),
            Self::Ne(a, b) => Self::Eq(a.clone(), b.clone()),
            Self::And(_, _) | Self::Unknown => Self::Unknown,
        }
    }
}

/// Outgoing basic-block terminator branch condition transcribed into
/// `CudaOxide.Mir.ControlFlow.BranchCond`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgBranchCond {
    /// Unconditional `mir.goto`.
    Always,
    /// Thread-coordinate predicate (`tid == 0`, `warp_id == 0`, `tid < 64`, `lane == 0`).
    Coord(GuardExpr),
    /// `mbarrier_try_wait*` ready branch (`ready == true`, `init_free` for parity-1 empty rings).
    MbarReady(usize, bool),
    /// `mbarrier_try_wait*` spin branch (`ready == false`).
    MbarSpin(usize),
    /// Exit edge of a positive-trip counted loop (`0 < bound`) whose body starts at `body_blk`.
    PosLoopExit(usize),
    /// CTA-uniform loop bound condition (`tile < k`, `col < 128`).
    Uniform,
    /// Warp-convergent runtime condition (all 32 lanes of each active warp take the same branch).
    WarpUniform,
    /// Data-dependent or unmodeled runtime condition (`if flag`, `if data[tid] > 0.0`).
    Unknown,
}

/// Control-flow-sensitive GPU or shared-memory instruction transcribed in block
/// order into `CudaOxide.Mir.ControlFlow.CfgOp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgOpItem {
    /// Block-wide `thread::sync_threads()` (`nvvm.barrier0`).
    SyncThreads,
    /// Cluster-wide `cluster::cluster_sync()` (`nvvm.cluster_sync` / `nvvm.cluster_barrier`).
    ClusterSync,
    /// Generic-proxy shared-memory load on `shared_id`.
    SharedRead {
        /// Index of the `SharedArray` static (`0..num_shared`).
        shared_id: usize,
    },
    /// Generic-proxy shared-memory store on `shared_id`.
    SharedWrite {
        /// Index of the `SharedArray` static (`0..num_shared`).
        shared_id: usize,
    },
    /// Cross-CTA Distributed Shared Memory (`map_shared_rank`) access on `shared_id`.
    DsmemAccess {
        /// Index of the peer `SharedArray` static (`0..num_shared`).
        shared_id: usize,
    },
    /// `mbarrier_init(&BAR, count)` via generic proxy.
    MbarInit {
        /// Index of the `Barrier` static (`0..num_barriers`).
        bar_id: usize,
        /// Expected thread arrival count.
        count: u64,
    },
    /// `mbarrier_inval(&BAR)` resetting `bar_id` back to `.uninit`.
    MbarInval {
        /// Index of the `Barrier` static (`0..num_barriers`).
        bar_id: usize,
    },
    /// `fence_proxy_async_shared_cta()` flushing generic writes to the async proxy.
    ProxyFenceAsyncShared,
    /// Plain arrival `mbarrier_arrive(&BAR)`.
    MbarArrive {
        /// Index of the `Barrier` static (`0..num_barriers`).
        bar_id: usize,
    },
    /// Byte-counted arrival `mbarrier_arrive_expect_tx(&BAR, count, tx_bytes)`.
    MbarArriveExpectTx {
        /// Index of the `Barrier` static (`0..num_barriers`).
        bar_id: usize,
        /// Expected asynchronous byte transfer count.
        tx_bytes: u64,
    },
    /// Asynchronous TMA copy `cp_async_bulk_tensor_*_g2s(SMEM, BAR, ...)`.
    TmaCopyG2S {
        /// Destination `SharedArray` static index (`0..num_shared`).
        shared_id: usize,
        /// Completion `Barrier` static index (`0..num_barriers`).
        bar_id: usize,
        /// True for `.cta_group::2` (`_cg2`) multicast TMA.
        is_cg2: bool,
    },
    /// Asynchronous Blackwell CLC `clc_try_cancel*` writing `shared_id` on `bar_id`.
    ClcTryCancel {
        /// Destination `CLC_RESPONSE` `SharedArray` index (`0..num_shared`).
        shared_id: usize,
        /// Completion `CLC_BAR` `Barrier` index (`0..num_barriers`).
        bar_id: usize,
    },
    /// Warp-collective `tcgen05_alloc(&TMEM_ADDR[0], cols)`.
    TmemAlloc {
        /// `SharedArray` slot receiving the allocated `TMEM` base column address.
        shared_id: usize,
        /// Requested `TMEM` column count.
        cols: u64,
        /// True for `.cta_group::2` (`_cg2`) allocation.
        is_cg2: bool,
    },
    /// Warp-collective `tcgen05_dealloc(tmem_addr, cols)`.
    TmemDealloc {
        /// Freed `TMEM` column count.
        cols: u64,
        /// True for `.cta_group::2` (`_cg2`) deallocation.
        is_cg2: bool,
    },
    /// Warp-collective `tcgen05_relinquish_alloc_permit*`.
    TmemRelinquishPermit {
        /// True for `.cta_group::2` (`_cg2`) relinquish.
        is_cg2: bool,
    },
    /// Single-thread async `tcgen05_mma*` or `tcgen05_cp*` reading `smem_ids` into `TMEM`.
    TmemAsyncIssue {
        /// Backing `SharedArray` indices of all 64-bit SMEM descriptors read by the op.
        smem_ids: Vec<usize>,
        /// True for `.cta_group::2` (`_cg2`) MMA/cp.
        is_cg2: bool,
    },
    /// Single-thread `tcgen05_commit*(&BAR)` tying prior `TmemAsyncIssue` to `bar_id`.
    TmemCommit {
        /// Signaled `Barrier` static index (`0..num_barriers`).
        bar_id: usize,
        /// True for `.cta_group::2` (`_cg2`) commit.
        is_cg2: bool,
    },
    /// Warp-restricted async `tcgen05_ld_*` producing register token `ld_id`.
    TmemLoad {
        /// Deterministic load identifier (`0, 1, ...`).
        ld_id: usize,
    },
    /// `tcgen05_load_wait()` retiring in-flight `TmemLoad` operations.
    TmemLoadWait,
    /// Consumer read of the register produced by `TmemLoad { ld_id }`.
    TmemLoadUse {
        /// Identifier of the consumed `TmemLoad` result.
        ld_id: usize,
    },
    /// Warp-convergent `tcgen05_store_wait()` retiring async `tcgen05.st` operations.
    TmemStoreWait,
    /// `tcgen05_fence_before_thread_sync()` ordering `tcgen05` prior to thread synchronization.
    TmemFenceBeforeSync,
    /// `tcgen05_fence_after_thread_sync()` ordering `tcgen05` after thread synchronization.
    TmemFenceAfterSync,
}

/// One basic block transcribed into `CudaOxide.Mir.ControlFlow.CfgBlock`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TranscribedCfgBlock {
    /// Ordered control-flow operations inside this basic block.
    pub ops: Vec<CfgOpItem>,
    /// Outgoing `(condition, target_block_idx)` edges.
    pub succs: Vec<(CfgBranchCond, usize)>,
    /// True when this basic block ends with `mir.return`.
    pub is_return: bool,
}

/// Compact kernel control-flow graph transcribed 1-to-1 into
/// `CudaOxide.Mir.ControlFlow.ControlFlowCfg` for Lean 4 state-machine verification.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TranscribedControlFlowCfg {
    /// Number of static `SharedArray` allocations in the kernel.
    pub num_shared: usize,
    /// Number of static `Barrier` (`mbarrier`) allocations in the kernel.
    pub num_barriers: usize,
    /// Transcribed basic blocks (`0` is entry).
    pub blocks: Vec<TranscribedCfgBlock>,
}

impl TranscribedControlFlowCfg {
    /// Strip no-op pass-through `goto` trampolines (`ops: [], succs: [(Always, next)]`),
    /// collapse empty reconverging diamonds (`[(c0, d), (c1, d)]` with non-`mbarrier` edges),
    /// and prune unreachable blocks while preserving `protected` block indices.
    ///
    /// Returns a map from original block indices to compacted `0..K` block indices.
    pub fn compact(&mut self, protected: &BTreeSet<usize>) -> HashMap<usize, usize> {
        let n = self.blocks.len();
        if n == 0 {
            return HashMap::new();
        }
        let mut all_protected = protected.clone();
        for blk in &self.blocks {
            for (cond, _) in &blk.succs {
                if let CfgBranchCond::PosLoopExit(body_blk) = *cond {
                    all_protected.insert(body_blk);
                }
            }
        }
        let resolve_chain = |blocks: &[TranscribedCfgBlock], mut u: usize| -> usize {
            for _ in 0..n {
                if all_protected.contains(&u) || u >= n {
                    break;
                }
                let blk = &blocks[u];
                if blk.ops.is_empty()
                    && !blk.is_return
                    && let [(CfgBranchCond::Always, next)] = blk.succs.as_slice()
                    && *next != u
                {
                    u = *next;
                } else {
                    break;
                }
            }
            u
        };

        for _ in 0..n + 2 {
            let mut changed = false;
            let snap = self.blocks.clone();
            for blk in &mut self.blocks {
                for (_, dst) in &mut blk.succs {
                    let resolved = resolve_chain(&snap, *dst);
                    if resolved != *dst {
                        *dst = resolved;
                        changed = true;
                    }
                }
                if let [(c0, d0), (c1, d1)] = blk.succs.as_slice()
                    && d0 == d1
                    && !matches!(
                        c0,
                        CfgBranchCond::MbarReady(_, _)
                            | CfgBranchCond::MbarSpin(_)
                            | CfgBranchCond::PosLoopExit(_)
                    )
                    && !matches!(
                        c1,
                        CfgBranchCond::MbarReady(_, _)
                            | CfgBranchCond::MbarSpin(_)
                            | CfgBranchCond::PosLoopExit(_)
                    )
                {
                    let target = *d0;
                    blk.succs = vec![(CfgBranchCond::Always, target)];
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let entry = resolve_chain(&self.blocks, 0);
        let mut reachable_order = Vec::new();
        let mut old_to_new: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::from([entry]);
        while let Some(u) = queue.pop_front() {
            if u >= n || old_to_new.contains_key(&u) {
                continue;
            }
            let new_idx = reachable_order.len();
            old_to_new.insert(u, new_idx);
            reachable_order.push(u);
            for &(_, dst) in &self.blocks[u].succs {
                if !old_to_new.contains_key(&dst) {
                    queue.push_back(dst);
                }
            }
        }

        let mut full_remap = HashMap::new();
        for old_idx in 0..n {
            let target = resolve_chain(&self.blocks, old_idx);
            if let Some(&new_idx) = old_to_new.get(&target) {
                full_remap.insert(old_idx, new_idx);
            }
        }

        let mut new_blocks = Vec::with_capacity(reachable_order.len());
        for old_idx in reachable_order {
            let mut blk = self.blocks[old_idx].clone();
            blk.succs.retain(|(_, dst)| old_to_new.contains_key(dst));
            for (cond, dst) in &mut blk.succs {
                *dst = old_to_new[dst];
                if let CfgBranchCond::PosLoopExit(body_blk) = cond
                    && let Some(&new_body) = full_remap.get(body_blk)
                {
                    *body_blk = new_body;
                }
            }
            new_blocks.push(blk);
        }
        self.blocks = new_blocks;
        full_remap
    }
}

/// Semantic type of an SSA value in the single-block scalar `BitVec` subset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaKind {
    /// Fixed-width unsigned integer word (`ui8`, `ui16`, `ui32`, `ui64`).
    ScalarWord {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
    },
}

/// A Pliron SSA value paired with its generated Lean identifier and [`SsaKind`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedSsa {
    /// Deterministic Lean identifier (`"arg0"`, `"v0"`, `"v1"`, ...).
    pub lean_name: String,
    /// Semantic type of the SSA value.
    pub kind: SsaKind,
}

/// Typed SSA lookup table mapping Pliron [`Value`]s to [`TypedSsa`] bindings.
#[derive(Clone, Debug, Default)]
pub struct SsaEnv {
    /// Map from in-memory Pliron SSA [`Value`] to its [`TypedSsa`] binding.
    bindings: HashMap<Value, TypedSsa>,
    /// Monotonically increasing counter used to mint fresh `v0`, `v1`, ... names.
    next_var_id: usize,
}

impl SsaEnv {
    /// Create an empty SSA environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded SSA bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns true if no SSA bindings have been recorded.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Bind an entry block argument `arg{index}` with its [`SsaKind`].
    pub fn bind_arg(&mut self, value: Value, index: usize, kind: SsaKind) -> String {
        let lean_name = format!("arg{index}");
        self.bindings.insert(
            value,
            TypedSsa {
                lean_name: lean_name.clone(),
                kind,
            },
        );
        lean_name
    }

    /// Bind an operation result to a fresh `v{N}` Lean name and update its [`SsaKind`].
    pub fn bind_result(&mut self, value: Value, kind: SsaKind) -> String {
        if let Some(existing) = self.bindings.get_mut(&value) {
            existing.kind = kind;
            return existing.lean_name.clone();
        }
        let id = self.next_var_id;
        self.next_var_id += 1;
        let lean_name = format!("v{id}");
        self.bindings.insert(
            value,
            TypedSsa {
                lean_name: lean_name.clone(),
                kind,
            },
        );
        lean_name
    }

    /// Look up a previously defined SSA operand, rejecting forward, undefined,
    /// or external SSA references fail-closed.
    pub fn get(&self, value: &Value, scope: &str) -> Result<&TypedSsa, ExportError> {
        self.bindings.get(value).ok_or_else(|| {
            reject(
                scope,
                "operand is not an earlier result or an entry argument",
            )
        })
    }
}
