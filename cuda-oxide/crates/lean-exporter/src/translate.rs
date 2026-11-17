/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thin, fail-closed `PostMem2Reg` `dialect-mir` transcriber powered by [`crate::ssa::SsaEnv`].
//!
//! Resolves SSA operand provenance (`IdxExpr`, `GuardExpr`, `SharedArray`,
//! `Barrier`, and `TMEM` addresses) and transcribes the basic-block control-flow
//! graph (`ControlFlowCfg`) 1-to-1 so Lean 4 owns all arithmetic, barrier,
//! proxy-coherence, `mbarrier`, and `tcgen05` / `TMEM` lifecycle proofs:
//!
//! ```text
//!                     translate_function(ctx, mir_func)
//!                                    │
//!             ┌──────────────────────┼──────────────────────┐
//!             ▼                      ▼                      ▼
//!       `scalar.rs`            `slice_1d.rs`          `phased.rs`
//!   • `ScalarWord` AST     • `KernelSummary1D`     • `ssa_prop.rs` -> `strips_2d.rs`
//!   • `BitVec` semantics   • `ControlFlowCfg`        -> `tcgen05.rs` -> `ControlFlowCfg`
//! ```

/// Shared `BlockGraph` adjacency queries and Pliron operand/cast/attribute helpers.
mod ir_utils;
/// Phased shared-memory, barrier, 2D tiled, TMA/`mbarrier`, and `TMEM` kernel driver.
mod phased;
/// Single-block pure scalar integer `mir.func` transcriber (`arithmetic.mir`).
mod scalar;
/// Barrier-free 1D global-slice kernel transcriber (`vecadd`, `warp_reduce_sum`).
mod slice_1d;
/// Fixed-point `SsaEnv` operand propagation (`IdxExpr`, `GuardExpr`, `SmemDesc`, `TmemAddr`).
mod ssa_prop;
/// Hoisted 2D global slice `RowBand` / `ColBand` provenance (`sgemm_tiled_raw`).
mod strips_2d;
/// Blackwell `tcgen05` / `TMEM`, `mbarrier`, and TMA instruction decoders and transcription.
mod tcgen05;

use pliron::{
    context::{Context, Ptr},
    operation::{Operation, verify_operation},
};

use crate::{
    ExportError,
    spec::{KernelVerificationSpec, has_verification_attributes, parse_kernel_verification_spec},
    ssa::{GlobalStripKind, GuardExpr, IdxExpr, TileAccessPatternKind, TranscribedControlFlowCfg},
};

/// Direction of an extracted buffer access.
/// Distinguishes read-only loads from mutating stores in Lean summaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    /// Read from a slice or shared-memory buffer (`mir.load`).
    Read,
    /// Write to a mutable slice, `DisjointSlice`, or shared-memory buffer (`mir.store`).
    Write,
}

/// One guarded 1D buffer access transcribed from `PostMem2Reg` `dialect-mir`.
/// Maps directly to `CudaOxide.Mir.Kernel.GuardedAccess1D` in Lean 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExtractedAccess {
    /// Read or write direction of this memory operation.
    pub kind: AccessKind,
    /// Accessed kernel parameter index (`0..arity`).
    pub target_param: usize,
    /// Kernel parameter index whose `.len()` bounds the dominating branch guard.
    pub guard_param: usize,
    /// Constant element offset added to the 1D thread index (`0` for `buf[idx]`).
    pub index_offset: u64,
}

/// One phased shared-memory tile or reduction access transcribed from `PostMem2Reg` MIR.
/// Maps directly to `CudaOxide.Mir.Kernel.PhasedTileAccess` in Lean 4.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhasedSharedAccess {
    /// Read or write direction of the shared-memory operation.
    pub kind: AccessKind,
    /// Static `SharedArray` index (`0` for `TILE_A`, `1` for `TILE_B`, ...).
    pub shared_id: usize,
    /// Short identifier of the `SharedArray` static (`"TILE_A"`, `"WARP_SUMS"`, ...).
    pub shared_name: String,
    /// Declared element capacity of the `SharedArray`.
    pub buffer_len: u64,
    /// Barrier phase index (`0` before first barrier, `1` after first, ...).
    pub phase: usize,
    /// Coordinate formula used to index the shared buffer.
    pub pattern: TileAccessPatternKind,
    /// Exclusive upper bound on loop index `i` (`1` outside a scan loop).
    pub loop_bound: u64,
    /// True when dominated by a single-leader guard (`lane == 0` or `tid == 0`).
    pub leader_only: bool,
    /// Dominating upper bound on `tx` (`block_size` when unguarded, or `N` under `if tid < N`).
    pub active_threads: u64,
    /// Residual coordinate branch guard (`GuardExpr::True` when unguarded).
    pub guard: GuardExpr,
}

/// One guarded 1D global slice access extracted inside a phased kernel (`&[T]`,
/// `&mut [T]`, or `DisjointSlice<T>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhasedGlobal1DAccess {
    /// Read or write direction.
    pub kind: AccessKind,
    /// Accessed kernel slice parameter index (`0..arity`).
    pub target_param: usize,
    /// Dominating `< guard_param.len()` slice parameter index.
    pub guard_param: usize,
    /// Constant element offset added to the 1D thread/group index.
    pub index_offset: u64,
    /// Group size (`1` for per-thread `idx`, `block_size` for `blockIdx_x` leader).
    pub group_size: u64,
}
/// transcribed from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalStripAccess2D {
    /// Accessed kernel slice parameter index (`0..arity`).
    pub param: usize,
    /// Row-strip (`start + col`) vs. column-strip (`start + row * stride`) shape.
    pub kind: GlobalStripKind,
    /// Constant element offset added to the strip index (`0` when exact).
    pub offset: u64,
}

/// One Blackwell `TMEM` operand or epilogue access (`tcgen05.mma`, `tcgen05.cp`, `tcgen05.ld`)
/// transcribed from `PostMem2Reg` `dialect-mir`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractedTmemAccess {
    /// Dynamically allocated `TMEM` column capacity backing this access.
    pub alloc_cols: u64,
    /// Starting `TMEM` lane expression (`0..128`).
    pub lane_expr: IdxExpr,
    /// Starting `TMEM` column expression (`0..alloc_cols`).
    pub col_expr: IdxExpr,
    /// Number of `TMEM` lanes touched (`16`, `32`, `64`, or `128`).
    pub height_lanes: u64,
    /// Number of 32-bit `TMEM` columns touched (`1..256`).
    pub width_cols: u64,
    /// Exclusive upper bound on loop counter `i` (`1` outside a loop).
    pub loop_bound: u64,
    /// True for `tcgen05.ld` / `tcgen05.st`, which must stay inside the warp's 32-lane slice.
    pub warp_restricted: bool,
    /// Dominating thread guard (`GuardExpr::True` when unguarded).
    pub guard: GuardExpr,
}

/// Transcribed output of [`translate_function`] before rendering to `.lean` source text.
/// Carries the extracted access summaries, `ControlFlowCfg`, and proven redundant asserts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedFunction {
    /// Pure single-block scalar function (`CudaOxide.Mir.Scalar.Expr`).
    Scalar {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
        /// Number of entry arguments.
        arity: usize,
        /// Ordered `let argN / vN := ...` lines.
        let_lines: Vec<String>,
        /// Final returned variable name (`"arg0"`, `"v1"`, ...).
        returned_var: String,
    },
    /// Verified barrier-free 1D slice GPU kernel (`KernelSummary1D` + `ControlFlowCfg`).
    Kernel {
        /// Number of kernel parameters.
        arity: usize,
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Guarded buffer reads and writes extracted from the kernel CFG.
        accesses: Vec<ExtractedAccess>,
        /// Transcribed basic-block control-flow graph for `WritesOnAllActivePaths`.
        control_flow_cfg: TranscribedControlFlowCfg,
        /// Per-parameter `(param, guard_entry_blocks, write_blocks)` triples for `writes_all_elems`.
        write_path_obligations: Vec<(usize, Vec<usize>, Vec<usize>)>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
    /// Verified phased shared-memory / barrier / async / `TMEM` GPU kernel.
    TiledKernel2D {
        /// Number of kernel parameters (`0..arity`).
        arity: usize,
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: KernelVerificationSpec,
        /// Number of effective barriers (`sync_threads()` + signaled `mbarrier` waits).
        num_barriers: usize,
        /// 2D global slice strip accesses extracted from the kernel CFG.
        global_strip_accesses: Vec<GlobalStripAccess2D>,
        /// 1D global slice accesses extracted from the kernel CFG (`&mut [T]` / `DisjointSlice<T>`).
        global_1d_accesses: Vec<PhasedGlobal1DAccess>,
        /// Phased shared-memory reads and writes extracted from the kernel CFG.
        shared_accesses: Vec<PhasedSharedAccess>,
        /// Transcribed basic-block control-flow graph verified in Lean 4 (`CudaOxide.Mir.ControlFlow`).
        control_flow_cfg: TranscribedControlFlowCfg,
        /// Extracted Blackwell `TMEM` accesses (`tcgen05.mma`, `tcgen05.cp`, `tcgen05.ld`).
        tmem_accesses: Vec<ExtractedTmemAccess>,
        /// `mir.assert` bounds-check operations proven redundant by `no_oob_idx`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
}

/// Transcribe any supported `mir.func` (scalar function, 1D slice kernel, or phased kernel)
/// into a [`TranslatedFunction`] ready for Lean 4 rendering.
pub fn translate_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    if !has_verification_attributes(ctx, function) {
        return scalar::translate_scalar_function(ctx, function);
    }
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;
    let (arity, spec) = parse_kernel_verification_spec(ctx, function)?;
    if spec.domain == 2
        || !spec.no_oob_shared.is_empty()
        || !spec.no_data_race.is_empty()
        || spec.uniform_barriers
        || spec.tmem_lifecycle
    {
        phased::translate_phased_kernel(ctx, function, arity, spec)
    } else {
        slice_1d::translate_slice_1d_kernel(ctx, function, arity, spec)
    }
}
