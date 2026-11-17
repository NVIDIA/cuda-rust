/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Blackwell `tcgen05` / Tensor Memory (`TMEM`), `mbarrier`, and Hopper/Blackwell
//! TMA bulk-copy instruction decoders and `CfgOpItem` transcription.

use std::collections::HashMap;

use pliron::{
    builtin::types::IntegerType,
    context::{Context, Ptr},
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use super::{
    AccessKind, ExtractedTmemAccess, PhasedSharedAccess,
    ir_utils::{BlockGraph, bin_kinds, op_kind, push_shared_access},
    ssa_prop::as_smem_desc_source,
};
use crate::{
    ExportError, reject,
    spec::KernelVerificationSpec,
    ssa::{
        CfgOpItem, GuardExpr, IdxExpr, SsaEnv, SsaKind, TileAccessPatternKind, TranscribedCfgBlock,
    },
};

/// Parse `(height_lanes, width_cols)` from an `nvvm.tcgen05_ld_*` operation name.
/// Covers `32x32b`, `16x64b`, `16x128b`, `16x256b`, and `16x32bx2` register shapes.
fn parse_tcgen05_ld_shape(opid: &str) -> Option<(u64, u64)> {
    let rest = opid.strip_prefix("nvvm.tcgen05_ld_")?;
    let rest = rest.strip_suffix("_raw").unwrap_or(rest);
    let rest = rest.strip_suffix("_pack16").unwrap_or(rest);
    let (shape_part, mult_part) = rest.split_once("_x")?;
    let mult = mult_part.parse::<u64>().ok()?;
    match shape_part {
        "32x32b" => Some((32, mult)),
        "16x64b" => Some((16, 2 * mult)),
        "16x128b" => Some((16, 4 * mult)),
        "16x256b" => Some((16, 8 * mult)),
        "16x32bx2" => Some((32, mult)),
        _ => None,
    }
}

/// Parse `(height_lanes, width_cols)` from an `nvvm.tcgen05_cp_*` operation name.
/// Covers 128-lane and warpx2/warpx4 SMEM-to-TMEM async matrix copy shapes.
fn parse_tcgen05_cp_shape(opid: &str) -> Option<(u64, u64)> {
    let rest = opid.strip_prefix("nvvm.tcgen05_cp_")?;
    let rest = rest.strip_suffix("_cg2").unwrap_or(rest);
    match rest {
        "128x256b" => Some((128, 8)),
        "128x128b" => Some((128, 4)),
        "4x256b" => Some((4, 8)),
        "32x128b_warpx2_01_13" | "32x128b_warpx2_02_13" => Some((64, 4)),
        "32x128b_warpx4" => Some((128, 4)),
        _ => None,
    }
}

/// Decode `(height_lanes, width_cols)` from a constant 32-bit `Tcgen05InstructionDescriptor`.
/// Halves the per-CTA `M` lane height when `.cta_group::2` (`is_cg2`) is active.
fn decode_tcgen05_idesc_shape(raw: u128, is_cg2: bool) -> Option<(u64, u64)> {
    let v = u32::try_from(raw).ok()?;
    let n_dim = u64::from(((v >> 17) & 0x3F) * 8);
    let m_full = u64::from(((v >> 24) & 0x1F) * 16);
    if n_dim == 0 || m_full == 0 {
        return None;
    }
    Some((if is_cg2 { m_full / 2 } else { m_full }, n_dim))
}

/// Shared per-block context passed into [`transcribe_async_or_tcgen05_op`].
/// Bundles read-only SSA/CFG metadata alongside mutable block and obligation outputs.
pub(super) struct AsyncTranscribeCtx<'a> {
    /// Pliron IR context.
    pub(super) ctx: &'a Context,
    /// Kernel verification contract (`#[verify(...)]`).
    pub(super) spec: &'a KernelVerificationSpec,
    /// Function basic-block graph.
    pub(super) graph: &'a BlockGraph,
    /// Converged SSA provenance map.
    pub(super) env: &'a SsaEnv,
    /// Sorted `(short_name, elem_count)` list of static `SharedArray`s.
    pub(super) shared_raw_list: &'a [(String, u64)],
    /// Deterministic `0..K` IDs assigned to each `(block_idx, op_idx)` `tcgen05_ld` site.
    pub(super) tmem_ld_id_map: &'a HashMap<(usize, usize), usize>,
    /// Accumulated `expect_tx` byte count per `Barrier` ID.
    pub(super) arrived_tx_bytes_per_bar: &'a HashMap<usize, u64>,
    /// Paired `mir.assert` operations keyed by `SharedArray` pointer SSA value.
    pub(super) ptr_asserts: &'a HashMap<Value, Ptr<Operation>>,
    /// Current basic-block index (`0..num_blocks`).
    pub(super) b_idx: usize,
    /// Current operation index inside `b_idx`.
    pub(super) op_idx: usize,
    /// Base barrier phase at this program point.
    pub(super) base_phase: usize,
    /// True when dominated by a single-leader thread guard (`tid == 0` / `lane == 0`).
    pub(super) leader_only: bool,
    /// Tightest `tx < N` active-thread bound at `b_idx`.
    pub(super) active_threads: u64,
    /// Residual coordinate guard at `b_idx`.
    pub(super) residual_guard: &'a GuardExpr,
    /// Mutable `CfgBlock` receiving transcribed `CfgOpItem`s.
    pub(super) cfg_blk: &'a mut TranscribedCfgBlock,
    /// Mutable phased shared-memory access list.
    pub(super) shared_accesses: &'a mut Vec<PhasedSharedAccess>,
    /// Mutable Blackwell `TMEM` access list.
    pub(super) tmem_accesses: &'a mut Vec<ExtractedTmemAccess>,
    /// Mutable list of `mir.assert` checks proven redundant.
    pub(super) proven_assert_ops: &'a mut Vec<Ptr<Operation>>,
    /// Mutable map recording the TMA write phase of each `SharedArray`.
    pub(super) tma_write_phase_for_shared: &'a mut HashMap<usize, usize>,
}

/// Transcribe an `mbarrier`, TMA `cp_async_bulk_tensor_g2s`, or Blackwell `tcgen05` op.
/// Returns `Ok(true)` if `opid` was handled here, or `Ok(false)` for generic MIR ops.
pub(super) fn transcribe_async_or_tcgen05_op(
    opid: &str,
    op: &Operation,
    acx: &mut AsyncTranscribeCtx<'_>,
    resolve_phase: impl Fn(usize, &HashMap<usize, usize>) -> usize,
) -> Result<bool, ExportError> {
    if opid.starts_with("nvvm.tcgen05") && !acx.spec.tmem_lifecycle {
        return Err(reject(
            opid,
            "tcgen05 operations in a #[verify] kernel require `tmem_lifecycle`",
        ));
    }
    if opid == "nvvm.mbarrier_init_shared" {
        if let (
            Some(SsaKind::MbarrierPtr { bar_id }),
            Some(SsaKind::ConstInt { value: raw_cnt, .. }),
        ) = bin_kinds(acx.env, op)
        {
            let count = u64::try_from(raw_cnt).unwrap_or(u64::MAX);
            acx.cfg_blk.ops.push(CfgOpItem::MbarInit { bar_id, count });
        }
        return Ok(true);
    }
    if matches!(
        opid,
        "nvvm.mbarrier_arrive_shared" | "nvvm.mbarrier_arrive_cluster"
    ) {
        if let Some(SsaKind::MbarrierPtr { bar_id }) = op_kind(acx.env, op, 0) {
            acx.cfg_blk.ops.push(CfgOpItem::MbarArrive { bar_id });
        }
        return Ok(true);
    }
    if matches!(
        opid,
        "nvvm.mbarrier_arrive_expect_tx_shared" | "nvvm.mbarrier_arrive_expect_tx_cluster"
    ) {
        if let (
            Some(SsaKind::MbarrierPtr { bar_id }),
            Some(SsaKind::ConstInt { value: raw_tx, .. }),
        ) = bin_kinds(acx.env, op)
        {
            let tx_bytes = u64::try_from(raw_tx).unwrap_or(u64::MAX);
            acx.cfg_blk
                .ops
                .push(CfgOpItem::MbarArriveExpectTx { bar_id, tx_bytes });
        }
        return Ok(true);
    }
    if opid.starts_with("nvvm.cp_async_bulk_tensor_g2s") {
        if let (
            Some(SsaKind::SharedBasePtr {
                shared_id,
                size,
                elem_bytes,
            }),
            Some(SsaKind::MbarrierPtr { bar_id }),
        ) = bin_kinds(acx.env, op)
        {
            if elem_bytes == 0 {
                return Err(reject(
                    opid,
                    "cannot determine SharedArray element byte size for async bulk copy",
                ));
            }
            acx.cfg_blk
                .ops
                .push(CfgOpItem::TmaCopyG2S { shared_id, bar_id });
            let copied_elems = acx
                .arrived_tx_bytes_per_bar
                .get(&bar_id)
                .map_or(size, |&tx| tx.div_ceil(elem_bytes));
            acx.tma_write_phase_for_shared
                .insert(shared_id, acx.base_phase);
            push_shared_access(
                acx.shared_accesses,
                acx.shared_raw_list,
                AccessKind::Write,
                shared_id,
                size,
                acx.base_phase,
                TileAccessPatternKind::LinearScan,
                copied_elems,
                acx.leader_only,
                acx.active_threads,
                GuardExpr::True,
            );
        }
        return Ok(true);
    }
    if matches!(opid, "nvvm.tcgen05_alloc" | "nvvm.tcgen05_alloc_cg2") {
        let is_cg2 = opid.ends_with("_cg2");
        let (
            Some(
                SsaKind::SharedBasePtr {
                    shared_id, size, ..
                }
                | SsaKind::SharedElementPtr {
                    shared_id,
                    size,
                    addr: IdxExpr::Const(0),
                },
            ),
            Some(SsaKind::ConstInt {
                value: raw_cols, ..
            }),
        ) = bin_kinds(acx.env, op)
        else {
            return Err(reject(
                opid,
                "`tcgen05_alloc` requires a verified `SharedArray<u32, ...>` slot and constant column count",
            ));
        };
        if !acx.spec.no_oob_shared.contains(&shared_id)
            || !acx.spec.no_data_race.contains(&shared_id)
        {
            return Err(reject(
                opid,
                "`tcgen05_alloc` shared-memory address slot must be listed in `no_oob_idx` and `no_data_race`",
            ));
        }
        if let Some(&aop) = acx.ptr_asserts.get(&op.get_operand(0))
            && !acx.proven_assert_ops.contains(&aop)
        {
            acx.proven_assert_ops.push(aop);
        }
        let cols = u64::try_from(raw_cols).unwrap_or(u64::MAX);
        acx.cfg_blk.ops.push(CfgOpItem::TmemAlloc {
            shared_id,
            cols,
            is_cg2,
        });
        push_shared_access(
            acx.shared_accesses,
            acx.shared_raw_list,
            AccessKind::Write,
            shared_id,
            size,
            acx.base_phase,
            TileAccessPatternKind::LinearScan,
            1,
            true,
            acx.active_threads,
            GuardExpr::True,
        );
        return Ok(true);
    }
    if matches!(opid, "nvvm.tcgen05_dealloc" | "nvvm.tcgen05_dealloc_cg2") {
        let is_cg2 = opid.ends_with("_cg2");
        let (
            Some(SsaKind::TmemAddr {
                lane_expr: IdxExpr::Const(0),
                col_expr: IdxExpr::Const(0),
                ..
            }),
            Some(SsaKind::ConstInt {
                value: raw_cols, ..
            }),
        ) = bin_kinds(acx.env, op)
        else {
            return Err(reject(
                opid,
                "`tcgen05_dealloc` requires the base `TMEM` address loaded from `TMEM_ADDR` and a constant column count",
            ));
        };
        let cols = u64::try_from(raw_cols).unwrap_or(u64::MAX);
        acx.cfg_blk
            .ops
            .push(CfgOpItem::TmemDealloc { cols, is_cg2 });
        return Ok(true);
    }
    if opid.starts_with("nvvm.tcgen05_mma") || opid.starts_with("nvvm.tcgen05_cp") {
        let Some(SsaKind::TmemAddr {
            alloc_cols,
            lane_expr,
            col_expr,
        }) = op_kind(acx.env, op, 0)
        else {
            return Err(reject(
                opid,
                format!(
                    "`{opid}` destination operand must derive from an allocated `TMEM` base address"
                ),
            ));
        };
        if !acx.graph.is_loop_expr_in_scope(&lane_expr, acx.b_idx)
            || !acx.graph.is_loop_expr_in_scope(&col_expr, acx.b_idx)
        {
            return Err(reject(
                opid,
                format!("`{opid}` TMEM address uses an out-of-scope or conflicting loop variable"),
            ));
        }
        let is_cg2 = opid.ends_with("_cg2");
        let (height_lanes, width_cols) = if opid.starts_with("nvvm.tcgen05_cp") {
            parse_tcgen05_cp_shape(opid)
                .ok_or_else(|| reject(opid, format!("unrecognized `tcgen05_cp` shape `{opid}`")))?
        } else {
            let idesc_val = op
                .operands()
                .find_map(|v| match acx.env.get_kind(&v) {
                    Some(SsaKind::ConstInt { value, .. }) if value >= 256 => Some(value),
                    _ => None,
                })
                .ok_or_else(|| {
                    reject(
                        opid,
                        format!(
                            "`{opid}` requires a compile-time constant `Tcgen05InstructionDescriptor`"
                        ),
                    )
                })?;
            decode_tcgen05_idesc_shape(idesc_val, is_cg2).ok_or_else(|| {
                reject(
                    opid,
                    format!("`{opid}` instruction descriptor encodes zero M or N dimension"),
                )
            })?
        };
        let mut desc_rank = 0usize;
        let mut smem_ids = Vec::new();
        for (arg_pos, operand) in op.operands().enumerate().skip(1) {
            if !operand
                .get_type(acx.ctx)
                .deref(acx.ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|i| i.width() == 64)
            {
                continue;
            }
            let Some((shared_id, size, elem_bytes, byte_offset)) = acx
                .env
                .get_kind(&operand)
                .as_ref()
                .and_then(as_smem_desc_source)
            else {
                return Err(reject(
                    opid,
                    format!(
                        "`{opid}` 64-bit SMEM descriptor operand {arg_pos} must derive from `cvta_generic_to_shared_offset` on a verified `SharedArray`"
                    ),
                ));
            };
            if elem_bytes == 0 {
                return Err(reject(
                    opid,
                    "cannot determine element byte width for `tcgen05` SMEM descriptor",
                ));
            }
            if !smem_ids.contains(&shared_id) {
                smem_ids.push(shared_id);
            }
            let tile_bytes = if opid.starts_with("nvvm.tcgen05_cp") {
                height_lanes * width_cols * 4
            } else if desc_rank == 0 {
                height_lanes * 32
            } else {
                (if is_cg2 { width_cols / 2 } else { width_cols }) * 32
            };
            desc_rank += 1;
            let req_elems = byte_offset.saturating_add(tile_bytes).div_ceil(elem_bytes);
            let phase = resolve_phase(shared_id, acx.tma_write_phase_for_shared);
            push_shared_access(
                acx.shared_accesses,
                acx.shared_raw_list,
                AccessKind::Read,
                shared_id,
                size,
                phase,
                TileAccessPatternKind::LinearScan,
                req_elems,
                acx.leader_only,
                acx.active_threads,
                GuardExpr::True,
            );
        }
        acx.cfg_blk.ops.push(CfgOpItem::TmemAsyncIssue { smem_ids });
        let loop_bound = lane_expr.loop_bound().max(col_expr.loop_bound());
        acx.tmem_accesses.push(ExtractedTmemAccess {
            alloc_cols,
            lane_expr,
            col_expr,
            height_lanes,
            width_cols,
            loop_bound,
            warp_restricted: false,
            guard: acx.residual_guard.clone(),
        });
        return Ok(true);
    }
    if opid.starts_with("nvvm.tcgen05_ld_") {
        let (height_lanes, width_cols) = parse_tcgen05_ld_shape(opid)
            .ok_or_else(|| reject(opid, format!("unrecognized `tcgen05_ld` shape `{opid}`")))?;
        let Some(SsaKind::TmemAddr {
            alloc_cols,
            lane_expr,
            col_expr,
        }) = op_kind(acx.env, op, 0)
        else {
            return Err(reject(
                opid,
                "`tcgen05_ld` operand must derive from an allocated `TMEM` base address",
            ));
        };
        if !acx.graph.is_loop_expr_in_scope(&lane_expr, acx.b_idx)
            || !acx.graph.is_loop_expr_in_scope(&col_expr, acx.b_idx)
        {
            return Err(reject(
                opid,
                "`tcgen05_ld` TMEM address uses an out-of-scope or conflicting loop variable",
            ));
        }
        let ld_id = acx
            .tmem_ld_id_map
            .get(&(acx.b_idx, acx.op_idx))
            .copied()
            .unwrap_or(0);
        acx.cfg_blk.ops.push(CfgOpItem::TmemLoad { ld_id });
        let loop_bound = lane_expr.loop_bound().max(col_expr.loop_bound());
        acx.tmem_accesses.push(ExtractedTmemAccess {
            alloc_cols,
            lane_expr,
            col_expr,
            height_lanes,
            width_cols,
            loop_bound,
            warp_restricted: true,
            guard: acx.residual_guard.clone(),
        });
        return Ok(true);
    }
    if opid == "nvvm.tcgen05_load_wait" {
        acx.cfg_blk.ops.push(CfgOpItem::TmemLoadWait);
        return Ok(true);
    }
    if opid.starts_with("nvvm.tcgen05_commit") {
        if let Some(SsaKind::MbarrierPtr { bar_id }) = op_kind(acx.env, op, 0) {
            acx.cfg_blk.ops.push(CfgOpItem::TmemCommit { bar_id });
        }
        return Ok(true);
    }
    if matches!(
        opid,
        "nvvm.tcgen05_store_wait"
            | "nvvm.tcgen05_fence_before_thread_sync"
            | "nvvm.tcgen05_fence_after_thread_sync"
    ) {
        return Ok(true);
    }
    if opid.starts_with("nvvm.tcgen05") {
        return Err(reject(
            opid,
            format!("unsupported `tcgen05` operation `{opid}` in verified kernel"),
        ));
    }
    Ok(false)
}
