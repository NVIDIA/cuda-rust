/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed SSA lookup table (`SsaEnv`).
//!
//! Instead of mapping a Pliron SSA [`Value`] to an untyped `String`, every SSA
//! value is bound to a [`TypedSsa`] carrying **both**:
//! 1. `lean_name`: its generated Lean variable name (`"arg0"`, `"v0"`, `"v1"`, ...)
//! 2. `kind`: its MIR type and semantic role ([`SsaKind`])
//!
//! ```text
//!   Pliron `Value`      `TypedSsa { lean_name, kind }`
//!   ───────────────     ────────────────────────────────────────────────────
//!   `block` (arg 0) ──► `{ lean_name: "arg0", kind: ScalarWord { width: 64 } }`
//!   `a`     (arg 0) ──► `{ lean_name: "arg0", kind: BufferParam(0) }`
//!   `prod`  (op 0)  ──► `{ lean_name: "v0",   kind: ScalarWord { width: 64 } }`
//!   `idx`   (op 1)  ──► `{ lean_name: "v1",   kind: Index1D { offset: 0 } }`
//!   `c_len` (op 2)  ──► `{ lean_name: "v2",   kind: SliceLen(2) }`
//!   `cond`  (op 3)  ──► `{ lean_name: "v3",   kind: IndexLtLen { guard_param: 2 } }`
//! ```

use std::collections::{BTreeSet, HashMap, VecDeque};

use pliron::value::Value;

use crate::{ExportError, reject};

/// Result of checking whether an [`IdxExpr`] references zero, one, or conflicting
/// inner loop induction variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopScopeCheck {
    /// Expression does not reference any `LoopVar`.
    None,
    /// Expression references a single `LoopVar` with its active CFG scope.
    Single {
        /// Unique loop variable identifier.
        loop_id: usize,
        /// Header block index of the loop.
        header_block: usize,
        /// Active in-loop target block of `< bound`.
        body_block: usize,
        /// Loop-exit target block of `>= bound`.
        exit_block: usize,
    },
    /// Expression combines two distinct `LoopVar`s (`i` and `j`), which cannot be
    /// represented faithfully by a single `.loopVar` in Lean.
    Conflict,
}

impl LoopScopeCheck {
    /// Return true unless two distinct loop counters were combined.
    pub fn is_valid(self) -> bool {
        !matches!(self, Self::Conflict)
    }
}

/// Compositional symbolic index expression over thread coordinates `(tx, ty)` and loop step `i`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IdxExpr {
    /// Thread X coordinate (`threadIdx.x`).
    Tx,
    /// Thread Y coordinate (`threadIdx.y`).
    Ty,
    /// Inner scan loop induction variable `i` (`0 <= i < bound`).
    LoopVar {
        /// Unique loop variable identifier (`header_block * 16 + arg_pos`) so two distinct
        /// loop counters cannot silently collapse into the single `.loopVar` in Lean.
        loop_id: usize,
        /// Basic block index of the loop header that defines this counter.
        header_block: usize,
        /// In-loop target block of the `< bound` branch on `header_block`.
        body_block: usize,
        /// Loop-exit target block of the `>= bound` branch on `header_block`.
        exit_block: usize,
        /// Exclusive upper bound on `i` from `while i < bound`.
        bound: u64,
    },
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
    /// Return true if this expression depends on a thread coordinate (`Tx`, `Ty`)
    /// or a bounded loop induction variable (`LoopVar`).
    pub fn has_coord(&self) -> bool {
        match self {
            Self::Tx | Self::Ty | Self::LoopVar { .. } => true,
            Self::Const(_) => false,
            Self::Add(a, b) | Self::Mul(a, b) => a.has_coord() || b.has_coord(),
            Self::Div(a, _) | Self::Mod(a, _) => a.has_coord(),
        }
    }

    /// Return the exclusive upper bound of any `LoopVar` inside this expression
    /// (`1` if the expression does not reference an inner loop counter).
    pub fn loop_bound(&self) -> u64 {
        match self {
            Self::LoopVar { bound, .. } => *bound,
            Self::Tx | Self::Ty | Self::Const(_) => 1,
            Self::Add(a, b) | Self::Mul(a, b) => a.loop_bound().max(b.loop_bound()),
            Self::Div(a, _) | Self::Mod(a, _) => a.loop_bound(),
        }
    }

    /// Validate that at most one distinct `LoopVar` (`loop_id`) appears in this expression,
    /// returning its [`LoopScopeCheck`] scope.
    pub fn single_loop_scope(&self) -> LoopScopeCheck {
        match self {
            Self::LoopVar {
                loop_id,
                header_block,
                body_block,
                exit_block,
                ..
            } => LoopScopeCheck::Single {
                loop_id: *loop_id,
                header_block: *header_block,
                body_block: *body_block,
                exit_block: *exit_block,
            },
            Self::Tx | Self::Ty | Self::Const(_) => LoopScopeCheck::None,
            Self::Div(a, _) | Self::Mod(a, _) => a.single_loop_scope(),
            Self::Add(a, b) | Self::Mul(a, b) => {
                match (a.single_loop_scope(), b.single_loop_scope()) {
                    (LoopScopeCheck::Conflict, _) | (_, LoopScopeCheck::Conflict) => {
                        LoopScopeCheck::Conflict
                    }
                    (LoopScopeCheck::None, other) => other,
                    (some, LoopScopeCheck::None) => some,
                    (
                        sa @ LoopScopeCheck::Single { loop_id: id_a, .. },
                        LoopScopeCheck::Single { loop_id: id_b, .. },
                    ) if id_a == id_b => sa,
                    (LoopScopeCheck::Single { .. }, LoopScopeCheck::Single { .. }) => {
                        LoopScopeCheck::Conflict
                    }
                }
            }
        }
    }

    /// Return the largest power of two (`1`, `2`, `4`, `8`, ...) guaranteed to divide
    /// every runtime valuation of this expression.
    pub fn guaranteed_pow2_align(&self) -> u64 {
        match self {
            Self::Const(0) => 1u64 << 63,
            Self::Const(n) => 1u64 << n.trailing_zeros(),
            Self::Mul(a, b) => {
                let shift = (a.guaranteed_pow2_align().trailing_zeros()
                    + b.guaranteed_pow2_align().trailing_zeros())
                .min(63);
                1u64 << shift
            }
            Self::Add(a, b) => a.guaranteed_pow2_align().min(b.guaranteed_pow2_align()),
            Self::Tx | Self::Ty | Self::LoopVar { .. } | Self::Div(_, _) | Self::Mod(_, _) => 1,
        }
    }

    /// Return a strict exclusive upper bound `U` (`self < U`) over all valid
    /// `(tx < block_x, ty < block_y, i < loop_bound)`, or `None` on `u64` overflow.
    pub fn exclusive_upper_bound(&self, block_x: u32, block_y: u32) -> Option<u64> {
        match self {
            Self::Tx => Some(u64::from(block_x)),
            Self::Ty => Some(u64::from(block_y)),
            Self::LoopVar { bound, .. } => Some(*bound),
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

    /// Convert `lhs | rhs` into `Add(lhs, rhs)` if and only if the active bit ranges
    /// of `lhs` and `rhs` are provably disjoint (`(lhs & rhs) == 0`).
    pub fn disjoint_bitor(lhs: Self, rhs: Self, block_x: u32, block_y: u32) -> Option<Self> {
        if lhs == Self::Const(0) {
            return Some(rhs);
        }
        if rhs == Self::Const(0) {
            return Some(lhs);
        }
        let lhs_align = lhs.guaranteed_pow2_align();
        if lhs_align > 1
            && rhs
                .exclusive_upper_bound(block_x, block_y)
                .is_some_and(|ub| ub <= lhs_align)
        {
            return Some(Self::Add(Box::new(lhs), Box::new(rhs)));
        }
        let rhs_align = rhs.guaranteed_pow2_align();
        if rhs_align > 1
            && lhs
                .exclusive_upper_bound(block_x, block_y)
                .is_some_and(|ub| ub <= rhs_align)
        {
            return Some(Self::Add(Box::new(lhs), Box::new(rhs)));
        }
        None
    }

    /// Match `self` against `coord * stride` or `stride * coord`, returning `(coord, stride)`.
    fn as_scaled(expr: &Self) -> Option<(&Self, u64)> {
        match expr {
            Self::Mul(lhs, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
                (inner, Self::Const(s)) | (Self::Const(s), inner) if *s > 0 => Some((inner, *s)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Classify this [`IdxExpr`] into a named [`TileAccessPatternKind`] shortcut when
    /// it matches one of the standard patterns, or [`TileAccessPatternKind::Expr`] for
    /// any general index formula.
    pub fn classify_pattern(&self, block_size: u32) -> (TileAccessPatternKind, u64) {
        match self {
            Self::Tx => (
                TileAccessPatternKind::ThreadSlot {
                    stride: u64::from(block_size),
                },
                1,
            ),
            Self::LoopVar { bound, .. } => (TileAccessPatternKind::LinearScan, *bound),
            Self::Div(inner, group_size) if **inner == Self::Tx && *group_size > 0 => (
                TileAccessPatternKind::GroupSlot {
                    group_size: *group_size,
                },
                1,
            ),
            Self::Mod(inner, group_size) if **inner == Self::Tx && *group_size > 0 => (
                TileAccessPatternKind::GroupLaneScan {
                    group_size: *group_size,
                    pack: 1,
                },
                1,
            ),
            _ => {
                if let Some((inner, pack)) = Self::as_scaled(self) {
                    if *inner == Self::Tx {
                        return (TileAccessPatternKind::PackedSlot { pack }, 1);
                    }
                    if let Self::Mod(base, group_size) = inner
                        && **base == Self::Tx
                    {
                        return (
                            TileAccessPatternKind::GroupLaneScan {
                                group_size: *group_size,
                                pack,
                            },
                            1,
                        );
                    }
                }
                if let Self::Add(lhs, rhs) = self {
                    for (scaled_side, other_side) in
                        [(lhs.as_ref(), rhs.as_ref()), (rhs.as_ref(), lhs.as_ref())]
                    {
                        if let Some((row_coord, stride)) = Self::as_scaled(scaled_side) {
                            match (row_coord, other_side) {
                                (Self::Ty, Self::Tx) => {
                                    return (TileAccessPatternKind::ThreadSlot { stride }, 1);
                                }
                                (Self::Ty, Self::LoopVar { bound, .. }) => {
                                    return (TileAccessPatternKind::RowScan { stride }, *bound);
                                }
                                (Self::LoopVar { bound, .. }, Self::Tx) => {
                                    return (TileAccessPatternKind::ColScan { stride }, *bound);
                                }
                                (Self::Tx, Self::LoopVar { bound, .. }) => {
                                    return (
                                        TileAccessPatternKind::PackedSlot { pack: stride },
                                        *bound,
                                    );
                                }
                                (Self::Tx, Self::Const(c)) => {
                                    return (
                                        TileAccessPatternKind::PackedSlot { pack: stride },
                                        c.saturating_add(1),
                                    );
                                }
                                (Self::Mod(b, group_size), Self::LoopVar { bound, .. })
                                    if **b == Self::Tx =>
                                {
                                    return (
                                        TileAccessPatternKind::GroupLaneScan {
                                            group_size: *group_size,
                                            pack: stride,
                                        },
                                        *bound,
                                    );
                                }
                                (Self::Mod(b, group_size), Self::Const(c)) if **b == Self::Tx => {
                                    return (
                                        TileAccessPatternKind::GroupLaneScan {
                                            group_size: *group_size,
                                            pack: stride,
                                        },
                                        c.saturating_add(1),
                                    );
                                }
                                (Self::Div(b1, g1), Self::Mod(b2, g2))
                                    if **b1 == Self::Tx && **b2 == Self::Tx && g1 == g2 =>
                                {
                                    return (
                                        TileAccessPatternKind::BitSplitSlot {
                                            group_size: *g1,
                                            stride,
                                        },
                                        1,
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
                (TileAccessPatternKind::Expr(self.clone()), self.loop_bound())
            }
        }
    }
}

/// Compositional boolean branch condition guarding a basic block.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GuardExpr {
    /// Unguarded (`True`).
    True,
    /// Strict upper bound `lhs < rhs` (e.g. `tid < 64`).
    Lt(IdxExpr, IdxExpr),
    /// Lower bound `lhs >= rhs` (from `!(lhs < rhs)` or `mir.ge`).
    Ge(IdxExpr, IdxExpr),
    /// Equality `lhs == rhs` (e.g. `lane == 0` or `lane == 7`).
    Eq(IdxExpr, IdxExpr),
    /// Inequality `lhs != rhs` (from `!(lhs == rhs)` or `mir.ne`).
    Ne(IdxExpr, IdxExpr),
    /// Conjunction of two dominating branch guards (`p /\ q`).
    And(Box<GuardExpr>, Box<GuardExpr>),
    /// Unmodeled or data-dependent branch condition (`if flag`, `if data[tid] > 0.0`).
    /// Yields no narrowing assumption (`True`) on bounds obligations while poisoning
    /// warp-collective and block-uniformity checks.
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

    /// Conjoin two branch guards, simplifying trivial `GuardExpr::True` operands.
    pub fn and(self, other: Self) -> Self {
        match (&self, &other) {
            (Self::True, _) => other,
            (_, Self::True) => self,
            _ if self == other => self,
            _ => Self::And(Box::new(self), Box::new(other)),
        }
    }

    /// Return true if this guard includes a zero-lane leader check (`tx % G == 0` or `tx == 0`).
    pub fn has_zero_leader_check(&self) -> bool {
        match self {
            Self::Eq(lhs, rhs) => {
                matches!(
                    (lhs, rhs),
                    (IdxExpr::Mod(b, _), IdxExpr::Const(0))
                        | (IdxExpr::Const(0), IdxExpr::Mod(b, _))
                        if **b == IdxExpr::Tx
                ) || matches!(
                    (lhs, rhs),
                    (IdxExpr::Tx, IdxExpr::Const(0)) | (IdxExpr::Const(0), IdxExpr::Tx)
                )
            }
            Self::And(p, q) => p.has_zero_leader_check() || q.has_zero_leader_check(),
            _ => false,
        }
    }

    /// Return true if this guard includes a CTA-wide thread-0 check (`tx == 0`).
    pub fn has_cta_leader_zero_check(&self) -> bool {
        match self {
            Self::Eq(lhs, rhs) => matches!(
                (lhs, rhs),
                (IdxExpr::Tx, IdxExpr::Const(0)) | (IdxExpr::Const(0), IdxExpr::Tx)
            ),
            Self::And(p, q) => p.has_cta_leader_zero_check() || q.has_cta_leader_zero_check(),
            _ => false,
        }
    }

    /// Extract the tightest `tx < bound` upper bound implied by this guard (defaulting to `default_bound`).
    pub fn active_threads_bound(&self, default_bound: u64) -> u64 {
        match self {
            Self::Lt(IdxExpr::Tx, IdxExpr::Const(bound)) => default_bound.min(*bound),
            Self::And(p, q) => p
                .active_threads_bound(default_bound)
                .min(q.active_threads_bound(default_bound)),
            _ => default_bound,
        }
    }

    /// Extract any coordinate conjuncts that are not already represented by `active_threads`
    /// (`tx < N`), `leader_only` (`tx % G == 0` / `tx == 0`), or unmodeled `Unknown` guards.
    pub fn custom_residual(&self) -> Self {
        match self {
            Self::True | Self::Unknown => Self::True,
            Self::Lt(IdxExpr::Tx, IdxExpr::Const(_)) => Self::True,
            Self::Eq(_, _) if self.has_zero_leader_check() => Self::True,
            Self::And(p, q) => p.custom_residual().and(q.custom_residual()),
            other => other.clone(),
        }
    }

    /// Return true if this guard tests a bounded inner loop counter (`i < bound` or `i >= bound`).
    pub fn is_loop_bound_test(&self) -> bool {
        matches!(
            self,
            Self::Lt(a, _) | Self::Ge(a, _) if a.loop_bound() > 1
        )
    }
}

/// How threads in a block index into a shared-memory array (`SharedArray`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TileAccessPatternKind {
    /// One fixed slot per thread (`TILE[ty * stride + tx]`).
    ThreadSlot {
        /// Row stride of the shared tile.
        stride: u64,
    },
    /// Walk across row `ty`, one element per loop step `i` (`TILE[ty * stride + i]`).
    RowScan {
        /// Row stride of the shared tile.
        stride: u64,
    },
    /// Walk down column `tx`, one element per loop step `i` (`TILE[i * stride + tx]`).
    ColScan {
        /// Row stride of the shared tile.
        stride: u64,
    },
    /// One slot per warp or sub-warp team of `group_size` threads (`WARP_SUMS[tx / group_size]`).
    GroupSlot {
        /// Number of threads per group (`32` for a warp, `16` for a half-warp).
        group_size: u64,
    },
    /// Walk straight across a 1D array, one element per loop step `i` (`WARP_SUMS[i]`).
    LinearScan,
    /// A contiguous chunk of `pack` slots owned by each thread `tx` (`S_X[tx * pack + i]`).
    PackedSlot {
        /// Number of contiguous elements owned by each thread `tx`.
        pack: u64,
    },
    /// A chunk of `pack` slots indexed by a thread's lane `tx % group_size` inside its sub-warp (`S_XS[(tx % group_size) * pack + i]`).
    GroupLaneScan {
        /// Number of threads per sub-warp / group (`16`, `32`, ...).
        group_size: u64,
        /// Number of elements scanned per lane (`1`, `4`, `16`, ...).
        pack: u64,
    },
    /// A 1D thread ID `tx` split into a 2D `(row = tx / group_size, col = tx % group_size)` cell (`TILE[(tx / group_size) * stride + (tx % group_size)]`).
    BitSplitSlot {
        /// Sub-warp width (`2^k`, e.g. `16`).
        group_size: u64,
        /// Row stride in shared memory (`stride >= group_size`).
        stride: u64,
    },
    /// General compositional index expression `IdxExpr` built from SSA operations.
    Expr(IdxExpr),
}

/// Outgoing basic-block terminator branch condition transcribed into
/// `CudaOxide.Mir.ControlFlow.BranchCond`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgBranchCond {
    /// Unconditional `mir.goto`.
    Always,
    /// Thread-coordinate predicate (`tid == 0`, `warp_id == 0`, `tid < 64`, `lane == 0`).
    Coord(GuardExpr),
    /// `mbarrier_try_wait*` ready branch (`ready == true`).
    MbarReady(usize),
    /// `mbarrier_try_wait*` spin branch (`ready == false`).
    MbarSpin(usize),
    /// CTA-uniform loop bound condition (`tile < k`, `col < 128`).
    Uniform,
    /// Data-dependent or unmodeled runtime condition (`if flag`, `if data[tid] > 0.0`).
    Unknown,
}

/// Control-flow-sensitive GPU or shared-memory instruction transcribed in block
/// order into `CudaOxide.Mir.ControlFlow.CfgOp`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CfgOpItem {
    /// Block-wide `thread::sync_threads()` (`nvvm.barrier0`).
    SyncThreads,
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
    /// `mbarrier_init(&BAR, count)` via generic proxy.
    MbarInit {
        /// Index of the `Barrier` static (`0..num_barriers`).
        bar_id: usize,
        /// Expected thread arrival count.
        count: u64,
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
    /// Single-thread async `tcgen05_mma*` or `tcgen05_cp*` reading `smem_ids` into `TMEM`.
    TmemAsyncIssue {
        /// Backing `SharedArray` indices of all 64-bit SMEM descriptors read by the op.
        smem_ids: Vec<usize>,
    },
    /// Single-thread `tcgen05_commit*(&BAR)` tying prior `TmemAsyncIssue` to `bar_id`.
    TmemCommit {
        /// Signaled `Barrier` static index (`0..num_barriers`).
        bar_id: usize,
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
        let resolve_chain = |blocks: &[TranscribedCfgBlock], mut u: usize| -> usize {
            for _ in 0..n {
                if protected.contains(&u) || u >= n {
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
                    && !matches!(c0, CfgBranchCond::MbarReady(_) | CfgBranchCond::MbarSpin(_))
                    && !matches!(c1, CfgBranchCond::MbarReady(_) | CfgBranchCond::MbarSpin(_))
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
            for (_, dst) in &mut blk.succs {
                *dst = old_to_new[dst];
            }
            new_blocks.push(blk);
        }
        self.blocks = new_blocks;
        full_remap
    }
}

/// 2D global slice strip indexing shape in a tiled 2D kernel (`sgemm_tiled_raw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GlobalStripKind {
    /// Contiguous row strip `start + col` (derived from `tx` with unit stride).
    RowStrip,
    /// Strided column strip `start + row * stride` (derived from `ty * stride`).
    ColStrip,
}

/// Semantic type and provenance of an SSA value in `dialect-mir`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SsaKind {
    /// Fixed-width unsigned integer word (`ui8`, `ui16`, `ui32`, `ui64`).
    ScalarWord {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
    },
    /// Integer constant literal with known unsigned value.
    ConstInt {
        /// Bit width (`0` when signless or boolean).
        width: u32,
        /// Unsigned literal value.
        value: u128,
    },
    /// Entry block slice or `DisjointSlice` parameter at `param` (`0..arity`).
    BufferParam(usize),
    /// Stack alloca for the zero-sized `LaunchContext` token.
    LaunchContextAlloca,
    /// `ThreadIndex` struct returned by `__internal::index_1d`.
    ThreadIndexStruct,
    /// Hardware `threadIdx_x` special register (`nvvm.read_ptx_sreg_tid_x`).
    SregTidX {
        /// Integer bit width (`32` raw, `64` after widening cast to `usize`).
        bit_width: u32,
    },
    /// Hardware `blockIdx_x` special register (`nvvm.read_ptx_sreg_ctaid_x`).
    SregCtaidX {
        /// Integer bit width (`32` raw, `64` after widening cast to `usize`).
        bit_width: u32,
    },
    /// Hardware `blockDim_x` special register (`nvvm.read_ptx_sreg_ntid_x`).
    SregNtidX {
        /// Integer bit width (`32` raw, `64` after widening cast to `usize`).
        bit_width: u32,
    },
    /// Validated `blockIdx_x * blockDim_x` base product before adding `threadIdx_x`.
    BlockBase1D {
        /// Integer bit width (`64`, or `32` when `u32_coordinates` is active).
        bit_width: u32,
    },
    /// 1D global thread or group-slot index (`(blockIdx * blockDim + tx) / group_size + offset`).
    Global1DIdx {
        /// Constant element offset added to the index.
        offset: u64,
        /// Thread group size (`1` for per-thread `idx`, `block_size` for `blockIdx_x` leader).
        group_size: u64,
    },
    /// Pointer to `&buf[global_idx + offset]` on a global slice in a phased kernel.
    Global1DElementPtr {
        /// Target slice parameter index (`0..arity`).
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Thread group size (`1` or `block_size`).
        group_size: u64,
    },
    /// Boolean comparison `global_idx < param.len()` in a phased kernel.
    Global1DLtLen {
        /// Slice parameter whose `.len()` bounds the index.
        guard_param: usize,
        /// Thread group size (`1` or `block_size`) of the compared index.
        group_size: u64,
    },
    /// Negated comparison `!(global_idx < param.len())` in a phased kernel.
    Global1DGeLen {
        /// Slice parameter whose `.len()` bounds the index.
        guard_param: usize,
        /// Thread group size (`1` or `block_size`) of the compared index.
        group_size: u64,
    },
    /// 1D global thread index plus a constant element offset (`idx + offset`).
    Index1D {
        /// Constant offset added to `thread::index_1d()`.
        offset: u64,
    },
    /// Checked addition tuple `(Index1D + offset, overflow_bool)`.
    CheckedIndex1D {
        /// Constant offset added to `thread::index_1d()`.
        offset: u64,
    },
    /// Base pointer extracted from buffer parameter `param` (`field 0`).
    SlicePtr(usize),
    /// Element length extracted from buffer parameter `param` (`field 1`).
    SliceLen(usize),
    /// Pointer to `&buf[idx + offset]` on buffer parameter `param`.
    ElementPtr {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset added to the 1D thread index.
        offset: u64,
    },
    /// Boolean comparison `idx == usize::MAX` (sentinel check).
    IsSentinel,
    /// Boolean comparison `idx < param.len()`.
    IndexLtLen {
        /// Buffer parameter whose `.len()` bounds `idx`.
        guard_param: usize,
    },
    /// Boolean negation `!(idx < param.len())`.
    IndexGeLen {
        /// Buffer parameter whose `.len()` bounds `idx`.
        guard_param: usize,
    },
    /// `Option<&mut T>` carrying `None`.
    OptionNone,
    /// `Option<&mut T>` carrying `Some(&mut param[idx + offset])` constructed
    /// inside a block guarded by `idx < guard_param.len()`.
    OptionSome {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index.
        guard_param: usize,
    },
    /// Merged `Option<&mut T>` block argument at a join block after `get_mut`.
    OptionJoin {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index from the `Some` branch.
        guard_param: usize,
    },
    /// Discriminant extracted from `OptionJoin`.
    OptionDiscriminant {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index from the `Some` branch.
        guard_param: usize,
    },
    /// Boolean test `discriminant == 1` (`Option::is_some`).
    OptionIsSome {
        /// Target buffer parameter index.
        param: usize,
        /// Constant element offset.
        offset: u64,
        /// Dominating `< len` guard parameter index.
        guard_param: usize,
    },
    /// Boolean test `discriminant == 0` (`Option::is_none`).
    OptionIsNone,
    /// Compositional symbolic index expression (`tx`, `ty`, `i`, `+`, `*`, `/`, `%`).
    Index(IdxExpr),
    /// Compositional boolean branch or bounds guard (`<`, `>=`, `==`, `!=`).
    Guard(GuardExpr),
    /// Base pointer to a static `SharedArray<T, N>` allocation.
    SharedBasePtr {
        /// Index of the `SharedArray` static (`0`, `1`, ...).
        shared_id: usize,
        /// Number of elements in the `SharedArray`.
        size: u64,
        /// Byte size of one element `T` (`0` if unrecognized).
        elem_bytes: u64,
    },
    /// Pointer to a static `Barrier` (`mbarrier`) allocation in shared memory.
    MbarrierPtr {
        /// Deterministic index of the `Barrier` static (`0`, `1`, ...).
        bar_id: usize,
    },
    /// Boolean completion predicate returned by `mbarrier_try_wait` or `mbarrier_try_wait_parity`.
    MbarrierReady {
        /// Index of the awaited `Barrier` static (`0`, `1`, ...).
        bar_id: usize,
    },
    /// Negated `mbarrier` wait predicate (`!mbarrier_try_wait*`).
    MbarrierNotReady {
        /// Index of the awaited `Barrier` static (`0`, `1`, ...).
        bar_id: usize,
    },
    /// Element pointer `&TILE[addr]` into a `SharedArray`.
    SharedElementPtr {
        /// Index of the `SharedArray` static (`0`, `1`, ...).
        shared_id: usize,
        /// Number of elements in the `SharedArray`.
        size: u64,
        /// Compositional index expression used to offset into the `SharedArray`.
        addr: IdxExpr,
    },
    /// Verified 2D global strip index (`start + col + offset` or `start + row * stride + offset`)
    /// proven in MIR to be dominated by both the strip's last-element `< slice.len()` check and
    /// the per-element `< extent` check.
    VerifiedGlobalStripIdx {
        /// Target slice parameter index (`0..arity`).
        param: usize,
        /// Row-strip (`start + col`) vs. column-strip (`start + row * stride`) pattern.
        kind: GlobalStripKind,
        /// Constant element offset added to the strip index (`0` when exact).
        offset: u64,
    },
    /// Element pointer `&slice[strip_idx]` into a 2D global slice parameter.
    GlobalStripElementPtr {
        /// Target slice parameter index (`0..arity`).
        param: usize,
        /// Row-strip (`start + col`) vs. column-strip (`start + row * stride`) pattern.
        kind: GlobalStripKind,
        /// Constant element offset added to the strip index (`0` when exact).
        offset: u64,
    },
    /// `.shared::cta` byte offset returned by `nvvm.cvta_generic_to_shared_offset(&SMEM)`
    /// (plus any constant byte offset added before descriptor packing).
    SmemSharedOffset {
        /// Index of the verified `SharedArray` static (`0`, `1`, ...).
        shared_id: usize,
        /// Declared element count of the `SharedArray`.
        size: u64,
        /// Byte width of one element in the `SharedArray`.
        elem_bytes: u64,
        /// Constant byte offset from the start of the `SharedArray` (`0` at base).
        byte_offset: u64,
    },
    /// 64-bit `tcgen05` shared-memory matrix descriptor derived from [`Self::SmemSharedOffset`].
    SmemDesc {
        /// Index of the verified `SharedArray` static backing this descriptor.
        shared_id: usize,
        /// Declared element count of the `SharedArray`.
        size: u64,
        /// Byte width of one element in the `SharedArray`.
        elem_bytes: u64,
        /// Constant byte offset from the start of the `SharedArray`.
        byte_offset: u64,
    },
    /// 32-bit Blackwell Tensor Memory (`TMEM`) address (`base + (lane << 16) + col`)
    /// derived from an allocated `TMEM_ADDR` shared-memory slot.
    TmemAddr {
        /// Number of `TMEM` columns allocated by `tcgen05_alloc`.
        alloc_cols: u64,
        /// Compositional `TMEM` lane offset expression (`0..128`).
        lane_expr: IdxExpr,
        /// Compositional `TMEM` column offset expression (`0..alloc_cols`).
        col_expr: IdxExpr,
    },
    /// Register value returned by `nvvm.tcgen05_ld_*` that requires `tcgen05_load_wait()`
    /// before being read by downstream instructions.
    TmemLoadedReg {
        /// Deterministic load site identifier (`0, 1, ...`).
        ld_id: usize,
    },
    /// CTA-uniform scalar integer parameter or uniform loop counter (`m`, `n`, `k`, `tile`).
    UniformScalar,
    /// Field-wise SSA provenance of a `mir.construct_struct` value (such as
    /// `Tcgen05SmemDescriptor` or `Tcgen05SmemDescriptorBuilder`).
    StructFields(Vec<SsaKind>),
    /// Local stack allocation (`mir.alloca`) tracked flow-sensitively for aggregate builders.
    StackAlloca(Value),
    /// Pointer to field `field` of a local [`Self::StackAlloca`] (`mir.field_addr`).
    StackFieldPtr {
        /// Base `mir.alloca` SSA [`Value`].
        alloca: Value,
        /// Zero-based struct field index.
        field: usize,
    },
    /// Loaded element or arithmetic value that does not affect indexing.
    DataValue,
}

/// A Pliron SSA value paired with its generated Lean identifier and [`SsaKind`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedSsa {
    /// Deterministic Lean identifier (`"arg0"`, `"v0"`, `"v1"`, ...).
    pub lean_name: String,
    /// Semantic type and provenance of the SSA value.
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

    /// Bind an operation result to a fresh `v{N}` Lean name (or reuse its
    /// existing name during fixed-point refinement) and update its [`SsaKind`].
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

    /// Look up the [`SsaKind`] of `value` if it is already bound.
    pub fn get_kind(&self, value: &Value) -> Option<SsaKind> {
        self.bindings.get(value).map(|b| b.kind.clone())
    }
}
