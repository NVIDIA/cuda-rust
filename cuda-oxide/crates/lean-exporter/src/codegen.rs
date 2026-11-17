/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Renders a [`TranslatedFunction`] into a `.lean` source file.
//!
//! ```text
//!   TranslatedFunction
//!     ├── Scalar ──► emits `CudaOxide.Mir.Scalar.Expr` + `evaluate`
//!     └── Kernel ──► emits `CudaOxide.Mir.Kernel.KernelSummary1D` +
//!                    `NoOobIdx` / `DisjointWrites` / `WritesAllElems` theorems
//! ```

use crate::{
    ExportError, reject,
    spec::KernelVerificationSpec,
    ssa::{
        CfgBranchCond, CfgOpItem, GlobalStripKind, GuardExpr, IdxExpr, TileAccessPatternKind,
        TranscribedCfgBlock, TranscribedControlFlowCfg,
    },
    translate::{
        AccessKind, ExtractedAccess, ExtractedTmemAccess, GlobalStripAccess2D,
        PhasedGlobal1DAccess, PhasedSharedAccess, TranslatedFunction,
    },
};

/// Render any [`TranslatedFunction`] into a complete Lean 4 file.
pub fn render_lean_module(translated: &TranslatedFunction) -> Result<String, ExportError> {
    match translated {
        TranslatedFunction::Scalar {
            width,
            arity,
            let_lines,
            returned_var,
        } => Ok(format!(
            "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
             -- SPDX-License-Identifier: Apache-2.0\n\
             -- Generated from scalar MIR. This file is a model, not a proof certificate.\n\
             import CudaOxide.Mir.Scalar\n\n\
             namespace CudaOxide.Generated\n\n\
             def program : CudaOxide.Mir.Scalar.Expr {width} {arity} :=\n{}\n  {returned_var}\n\n\
             def evaluate (inputs : Fin {arity} → BitVec {width}) : BitVec {width} := program.eval inputs\n\n\
             end CudaOxide.Generated\n",
            let_lines.join("\n"),
        )),
        TranslatedFunction::Kernel {
            arity,
            spec,
            accesses,
            control_flow_cfg,
            write_path_obligations,
            ..
        } => render_verified_kernel_module(
            *arity,
            spec,
            accesses,
            control_flow_cfg,
            write_path_obligations,
        ),
        TranslatedFunction::TiledKernel2D {
            arity,
            spec,
            num_barriers,
            global_strip_accesses,
            global_1d_accesses,
            shared_accesses,
            control_flow_cfg,
            tmem_accesses,
            ..
        } => render_verified_tiled_2d_module(
            *arity,
            spec,
            *num_barriers,
            global_strip_accesses,
            global_1d_accesses,
            shared_accesses,
            control_flow_cfg,
            tmem_accesses,
        ),
    }
}

/// Return grouped, human-readable obligation descriptions proven for `spec`.
pub fn proved_obligation_descriptions(
    spec: &KernelVerificationSpec,
    global_strip_accesses: &[GlobalStripAccess2D],
) -> Vec<String> {
    let fmt_ids = |prefix: &str, ids: &[usize]| -> String {
        ids.iter()
            .map(|id| format!("{prefix} {id}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut items = Vec::new();
    if !spec.no_oob_idx.is_empty() {
        if spec.domain == 2 {
            for &p in &spec.no_oob_idx {
                let contract = match global_strip_accesses.iter().find(|a| a.param == p) {
                    Some(GlobalStripAccess2D {
                        kind: GlobalStripKind::ColStrip,
                        ..
                    }) => "CudaOxide.Contracts.NoOobColStrip",
                    _ => "CudaOxide.Contracts.NoOobRowStrip",
                };
                items.push(format!("no_oob_idx(param {p}) -> {contract}"));
            }
        } else {
            items.push(format!(
                "no_oob_idx({}) -> CudaOxide.Contracts.NoOobIdx",
                fmt_ids("param", &spec.no_oob_idx)
            ));
        }
    }
    if !spec.disjoint_writes.is_empty() {
        items.push(format!(
            "disjoint_writes({}) -> CudaOxide.Contracts.DisjointWrites",
            fmt_ids("param", &spec.disjoint_writes)
        ));
    }
    if !spec.writes_all_elems.is_empty() {
        items.push(format!(
            "writes_all_elems({}) -> CudaOxide.Contracts.WritesAllElems",
            fmt_ids("param", &spec.writes_all_elems)
        ));
    }
    if !spec.no_oob_shared.is_empty() {
        items.push(format!(
            "no_oob_idx({}) -> CudaOxide.Contracts.NoOobTileAccess",
            fmt_ids("shared", &spec.no_oob_shared)
        ));
    }
    if !spec.no_data_race.is_empty() {
        items.push(format!(
            "no_data_race({}) -> CudaOxide.Contracts.NoDataRace",
            fmt_ids("shared", &spec.no_data_race)
        ));
    }
    if spec.uniform_barriers {
        items.push(
            "uniform_barriers -> CudaOxide.Contracts.ControlFlow.UniformBarriers".to_string(),
        );
    }
    if spec.tmem_lifecycle {
        items.push(
            "tmem_lifecycle -> CudaOxide.Contracts.ControlFlow.TmemLifecycle + CudaOxide.Contracts.NoOobTmemAccess"
                .to_string(),
        );
    }
    items
}

/// Render a verified 1D kernel (`KernelSummary1D` + optional `ControlFlowCfg`) and its
/// `NoOobIdx`, `DisjointWrites`, `WritesAllElems`, and `WritesOnAllActivePaths`
/// theorem obligations into Lean 4.
fn render_verified_kernel_module(
    arity: usize,
    spec: &KernelVerificationSpec,
    accesses: &[ExtractedAccess],
    control_flow_cfg: &TranscribedControlFlowCfg,
    write_path_obligations: &[(usize, Vec<usize>, Vec<usize>)],
) -> Result<String, ExportError> {
    let block_size = spec.block_size;
    let mut access_entries = Vec::new();
    for access in accesses {
        let kind = match access.kind {
            AccessKind::Read => ".read",
            AccessKind::Write => ".write",
        };
        access_entries.push(format!(
            "    {{ kind := {kind}, targetParam := ⟨{}, by decide⟩, guardParam := ⟨{}, by decide⟩, indexOffset := {} }}",
            access.target_param, access.guard_param, access.index_offset
        ));
    }

    let mut theorems = Vec::new();
    let mut axiom_prints = Vec::new();

    for &param in &spec.no_oob_idx {
        let matching: Vec<_> = accesses
            .iter()
            .filter(|a| a.target_param == param)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "parameter {param} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, access) in matching.iter().enumerate() {
            let thm_name = if matching.len() == 1 {
                format!("verify_no_oob_idx_param{param}")
            } else {
                format!("verify_no_oob_idx_param{param}_{idx}")
            };
            let guard = access.guard_param;
            let offset = access.index_offset;
            if offset == 0 && param == guard {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat) :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_same_len"
                ));
            } else if offset == 0 && spec.requires_len_ge.contains(&(param, guard)) {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat)\n    \
                     (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩) :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_guard_le h_req"
                ));
            } else {
                let req_binder = if spec.requires_len_ge.contains(&(param, guard)) {
                    format!(
                        "\n    (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩)"
                    )
                } else {
                    String::new()
                };
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat){req_binder} :\n    \
                     CudaOxide.Contracts.NoOobIdx {block_size} {offset} (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     intro block lane active\n  \
                     omega"
                ));
            }
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &param in &spec.disjoint_writes {
        let param_accesses: Vec<_> = accesses
            .iter()
            .filter(|a| a.target_param == param)
            .collect();
        let writes: Vec<_> = param_accesses
            .iter()
            .copied()
            .filter(|a| a.kind == AccessKind::Write)
            .collect();
        if writes.is_empty() {
            return Err(reject(
                "disjoint_writes",
                format!(
                    "parameter {param} is listed in `disjoint_writes` but never written by the kernel"
                ),
            ));
        }
        for (idx, write) in writes.iter().enumerate() {
            let thm_name = if writes.len() == 1 {
                format!("verify_disjoint_writes_param{param}")
            } else {
                format!("verify_disjoint_writes_param{param}_{idx}")
            };
            let offset = write.index_offset;
            theorems.push(format!(
                "theorem {thm_name} :\n    \
                 CudaOxide.Contracts.DisjointWrites {block_size} {offset} := by\n  \
                 exact CudaOxide.Proofs.Launch.disjoint_writes_1d"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
        if param_accesses.len() > 1 {
            for (w_idx, write) in writes.iter().enumerate() {
                for (a_idx, other) in param_accesses.iter().enumerate() {
                    let pair_name =
                        format!("verify_disjoint_slice_pair_param{param}_{w_idx}_{a_idx}");
                    let off1 = write.index_offset;
                    let off2 = other.index_offset;
                    let body = if off1 == off2 {
                        "exact CudaOxide.Proofs.Launch.disjoint_slice_pair_same_offset"
                    } else {
                        "intro block₁ block₂ lane₁ lane₂ hlane₁ hlane₂ hsame\n  omega"
                    };
                    theorems.push(format!(
                        "theorem {pair_name} :\n    \
                         CudaOxide.Contracts.DisjointSliceAccessPair {block_size} {off1} {off2} := by\n  \
                         {body}"
                    ));
                    axiom_prints.push(format!("#print axioms {pair_name}"));
                }
            }
        }
    }

    for &param in &spec.writes_all_elems {
        let writes: Vec<_> = accesses
            .iter()
            .filter(|a| a.kind == AccessKind::Write && a.target_param == param)
            .collect();
        if writes.is_empty() {
            return Err(reject(
                "writes_all_elems",
                format!(
                    "parameter {param} is listed in `writes_all_elems` but never written by the kernel"
                ),
            ));
        }
        let write = writes[0];
        let thm_name = format!("verify_writes_all_elems_param{param}");
        if write.index_offset == 0 && write.guard_param == param {
            theorems.push(format!(
                "theorem {thm_name} (blocks : Nat) (lengths : Fin {arity} → Nat)\n    \
                 (hcoverage : lengths ⟨{param}, by decide⟩ ≤ blocks * {block_size}) :\n    \
                 CudaOxide.Contracts.WritesAllElems blocks {block_size} (lengths ⟨{param}, by decide⟩) := by\n  \
                 exact CudaOxide.Proofs.Launch.writes_all_elems_1d (by decide) hcoverage"
            ));
        } else {
            theorems.push(format!(
                "theorem {thm_name} (blocks : Nat) (lengths : Fin {arity} → Nat)\n    \
                 (hcoverage : lengths ⟨{param}, by decide⟩ ≤ blocks * {block_size}) :\n    \
                 CudaOxide.Contracts.WritesAllElems blocks {block_size} (lengths ⟨{param}, by decide⟩) := by\n  \
                 intro i hi\n  \
                 omega"
            ));
        }
        axiom_prints.push(format!("#print axioms {thm_name}"));
        if let Some((_, entries, write_blks)) =
            write_path_obligations.iter().find(|(p, _, _)| *p == param)
        {
            let path_thm = format!("verify_writes_all_elems_path_param{param}");
            let entries_str = entries
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let writes_str = write_blks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            theorems.push(format!(
                "theorem {path_thm} :\n    \
                 CudaOxide.Contracts.ControlFlow.WritesOnAllActivePaths controlFlowCfg [{entries_str}] [{writes_str}] := by\n  \
                 exact CudaOxide.Proofs.ControlFlow.writes_on_active_paths_sound (by decide)"
            ));
            axiom_prints.push(format!("#print axioms {path_thm}"));
        }
    }

    let cfg_section = if !write_path_obligations.is_empty() {
        format!(
            "{}\n\n",
            format_control_flow_cfg(u64::from(block_size), 1, control_flow_cfg)
        )
    } else {
        String::new()
    };

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated from PostMem2Reg dialect-mir.\n\
         import CudaOxide\n\n\
         set_option maxRecDepth 4096\n\n\
         namespace CudaOxide.Generated\n\n\
         def kernelSummary : CudaOxide.Mir.Kernel.KernelSummary1D {arity} :=\n  \
         {{ blockSize := {block_size}\n  \
         , accesses := [\n{}\n    ]\n  }}\n\n\
         {cfg_section}\
         {}\n\n\
         {}\n\n\
         end CudaOxide.Generated\n",
        access_entries.join(",\n"),
        theorems.join("\n\n"),
        axiom_prints.join("\n"),
    ))
}

/// Format a compositional [`IdxExpr`] into its Lean `CudaOxide.Mir.Kernel.IdxExpr` constructor tree.
fn format_idx_expr(expr: &IdxExpr) -> String {
    match expr {
        IdxExpr::Tx => ".tx".to_string(),
        IdxExpr::Ty => ".ty".to_string(),
        IdxExpr::LoopVar { .. } => ".loopVar".to_string(),
        IdxExpr::Const(n) => format!("(.const {n})"),
        IdxExpr::Add(a, b) => format!("(.add {} {})", format_idx_expr(a), format_idx_expr(b)),
        IdxExpr::Mul(a, b) => format!("(.mul {} {})", format_idx_expr(a), format_idx_expr(b)),
        IdxExpr::Div(a, d) => format!("(.div {} {d})", format_idx_expr(a)),
        IdxExpr::Mod(a, d) => format!("(.mod {} {d})", format_idx_expr(a)),
    }
}

/// Format a compositional [`GuardExpr`] into its Lean `CudaOxide.Mir.Kernel.GuardExpr` constructor tree.
fn format_guard_expr(guard: &GuardExpr) -> String {
    match guard {
        GuardExpr::True => ".tr".to_string(),
        GuardExpr::Lt(a, b) => format!("(.lt {} {})", format_idx_expr(a), format_idx_expr(b)),
        GuardExpr::Ge(a, b) => format!("(.ge {} {})", format_idx_expr(a), format_idx_expr(b)),
        GuardExpr::Eq(a, b) => format!("(.eq {} {})", format_idx_expr(a), format_idx_expr(b)),
        GuardExpr::Ne(a, b) => format!("(.ne {} {})", format_idx_expr(a), format_idx_expr(b)),
        GuardExpr::And(p, q) => format!("(.and {} {})", format_guard_expr(p), format_guard_expr(q)),
        GuardExpr::Unknown => ".unknown".to_string(),
    }
}

/// Format a [`CfgBranchCond`] into a Lean `CudaOxide.Mir.ControlFlow.BranchCond` expression.
fn format_cfg_branch_cond(cond: &CfgBranchCond) -> String {
    match cond {
        CfgBranchCond::Always => ".always".to_string(),
        CfgBranchCond::Coord(g) => format!("(.coord {})", format_guard_expr(g)),
        CfgBranchCond::MbarReady(b) => format!("(.mbarReady {b})"),
        CfgBranchCond::MbarSpin(b) => format!("(.mbarSpin {b})"),
        CfgBranchCond::Uniform => ".uniform".to_string(),
        CfgBranchCond::Unknown => ".unknown".to_string(),
    }
}

/// Format a [`CfgOpItem`] into a Lean `CudaOxide.Mir.ControlFlow.CfgOp` expression.
fn format_cfg_op(op: &CfgOpItem) -> String {
    match op {
        CfgOpItem::SyncThreads => ".syncThreads".to_string(),
        CfgOpItem::SharedRead { shared_id } => format!("(.sharedRead {shared_id})"),
        CfgOpItem::SharedWrite { shared_id } => format!("(.sharedWrite {shared_id})"),
        CfgOpItem::MbarInit { bar_id, count } => format!("(.mbarInit {bar_id} {count})"),
        CfgOpItem::ProxyFenceAsyncShared => ".proxyFenceAsyncShared".to_string(),
        CfgOpItem::MbarArrive { bar_id } => format!("(.mbarArrive {bar_id})"),
        CfgOpItem::MbarArriveExpectTx { bar_id, tx_bytes } => {
            format!("(.mbarArriveExpectTx {bar_id} {tx_bytes})")
        }
        CfgOpItem::TmaCopyG2S { shared_id, bar_id } => {
            format!("(.tmaCopyG2S {shared_id} {bar_id})")
        }
        CfgOpItem::TmemAlloc {
            shared_id,
            cols,
            is_cg2,
        } => format!("(.tmemAlloc {shared_id} {cols} {is_cg2})"),
        CfgOpItem::TmemDealloc { cols, is_cg2 } => {
            format!("(.tmemDealloc {cols} {is_cg2})")
        }
        CfgOpItem::TmemAsyncIssue { smem_ids } => {
            let list = smem_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("(.tmemAsyncIssue [{list}])")
        }
        CfgOpItem::TmemCommit { bar_id } => format!("(.tmemCommit {bar_id})"),
        CfgOpItem::TmemLoad { ld_id } => format!("(.tmemLoad {ld_id})"),
        CfgOpItem::TmemLoadWait => ".tmemLoadWait".to_string(),
        CfgOpItem::TmemLoadUse { ld_id } => format!("(.tmemLoadUse {ld_id})"),
    }
}

/// Format one [`TranscribedCfgBlock`] as a Lean `CudaOxide.Mir.ControlFlow.CfgBlock` literal.
fn format_cfg_block(blk: &TranscribedCfgBlock) -> String {
    let ops_str = blk
        .ops
        .iter()
        .map(format_cfg_op)
        .collect::<Vec<_>>()
        .join(", ");
    let succs_str = blk
        .succs
        .iter()
        .map(|(c, dst)| format!("({}, {dst})", format_cfg_branch_cond(c)))
        .collect::<Vec<_>>()
        .join(", ");
    let ret_suffix = if blk.is_return {
        ", isReturn := true"
    } else {
        ""
    };
    format!("{{ ops := [{ops_str}], succs := [{succs_str}]{ret_suffix} }}")
}

/// Format the full [`TranscribedControlFlowCfg`] as a Lean `def controlFlowCfg` declaration.
fn format_control_flow_cfg(bx: u64, by: u64, cfg: &TranscribedControlFlowCfg) -> String {
    let block_lines = cfg
        .blocks
        .iter()
        .map(|blk| format!("    {}", format_cfg_block(blk)))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "def controlFlowCfg : CudaOxide.Mir.ControlFlow.ControlFlowCfg :=\n  \
         {{ blockDimX := {bx}\n  \
         , blockDimY := {by}\n  \
         , numShared := {}\n  \
         , numBarriers := {}\n  \
         , blocks := [\n{block_lines}\n    ]\n  }}",
        cfg.num_shared, cfg.num_barriers
    )
}

/// Format a [`TileAccessPatternKind`] into its corresponding Lean
/// `CudaOxide.Mir.Kernel.TileAccessPattern` constructor expression.
fn format_tile_pattern(pat: &TileAccessPatternKind) -> String {
    match pat {
        TileAccessPatternKind::ThreadSlot { stride } => format!(".threadSlot {stride}"),
        TileAccessPatternKind::RowScan { stride } => format!(".rowScan {stride}"),
        TileAccessPatternKind::ColScan { stride } => format!(".colScan {stride}"),
        TileAccessPatternKind::GroupSlot { group_size } => format!(".groupSlot {group_size}"),
        TileAccessPatternKind::LinearScan => ".linearScan".to_string(),
        TileAccessPatternKind::PackedSlot { pack } => format!(".packedSlot {pack}"),
        TileAccessPatternKind::GroupLaneScan { group_size, pack } => {
            format!(".groupLaneScan {group_size} {pack}")
        }
        TileAccessPatternKind::BitSplitSlot { group_size, stride } => {
            format!(".bitSplitSlot {group_size} {stride}")
        }
        TileAccessPatternKind::Expr(expr) => format!(".expr {}", format_idx_expr(expr)),
    }
}

/// Format a [`PhasedSharedAccess`] into a Lean `CudaOxide.Mir.Kernel.PhasedTileAccess`
/// record literal for `kernelSummary` and `NoDataRace` theorem statements.
fn format_phased_access(a: &PhasedSharedAccess) -> String {
    let kind = match a.kind {
        AccessKind::Read => ".read",
        AccessKind::Write => ".write",
    };
    let pat = format_tile_pattern(&a.pattern);
    let leader_suffix = if a.leader_only {
        ", leaderOnly := true"
    } else {
        ""
    };
    let guard_suffix = if a.guard != GuardExpr::True {
        format!(", guard := {}", format_guard_expr(&a.guard))
    } else {
        String::new()
    };
    format!(
        "{{ kind := {kind}, space := .shared, bufferId := {}, phase := {}, pattern := {pat}, loopBound := {}{leader_suffix}{guard_suffix} }}",
        a.shared_id, a.phase, a.loop_bound
    )
}

/// Format an [`ExtractedTmemAccess`] into a Lean `CudaOxide.Mir.Kernel.TmemAccess` record literal.
fn format_tmem_access(a: &ExtractedTmemAccess) -> String {
    let guard_suffix = if a.guard != GuardExpr::True {
        format!(", guard := {}", format_guard_expr(&a.guard))
    } else {
        String::new()
    };
    format!(
        "{{ laneExpr := {}, colExpr := {}, heightLanes := {}, widthCols := {}, loopBound := {}, warpRestricted := {}{guard_suffix} }}",
        format_idx_expr(&a.lane_expr),
        format_idx_expr(&a.col_expr),
        a.height_lanes,
        a.width_cols,
        a.loop_bound,
        a.warp_restricted,
    )
}

/// Render a phased shared-memory or 2D tiled kernel (`TiledKernelSummary2D` + `ControlFlowCfg`)
/// and its `NoOobRowStrip`, `NoOobColStrip`, `NoOobTileAccess`, `NoDataRace`,
/// `UniformBarriers`, `TmemLifecycle`, and `NoOobTmemAccess` obligations into Lean 4.
#[allow(clippy::too_many_arguments)]
fn render_verified_tiled_2d_module(
    arity: usize,
    spec: &KernelVerificationSpec,
    num_barriers: usize,
    global_strip_accesses: &[GlobalStripAccess2D],
    global_1d_accesses: &[PhasedGlobal1DAccess],
    shared_accesses: &[PhasedSharedAccess],
    control_flow_cfg: &TranscribedControlFlowCfg,
    tmem_accesses: &[ExtractedTmemAccess],
) -> Result<String, ExportError> {
    let bx = u64::from(spec.block_size);
    let by = u64::from(spec.block_y);
    let access_entries: Vec<String> = shared_accesses
        .iter()
        .map(|a| format!("    {}", format_phased_access(a)))
        .collect();

    let mut theorems = Vec::new();
    let mut axiom_prints = Vec::new();

    for &param in &spec.no_oob_idx {
        let matching_strips: Vec<_> = global_strip_accesses
            .iter()
            .filter(|a| a.param == param)
            .collect();
        let matching_1d: Vec<_> = global_1d_accesses
            .iter()
            .filter(|a| a.target_param == param)
            .collect();
        if matching_strips.is_empty() && matching_1d.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "parameter {param} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, acc) in matching_strips.iter().enumerate() {
            let thm_name = if matching_strips.len() == 1 && matching_1d.is_empty() {
                format!("verify_no_oob_idx_param{param}")
            } else {
                format!("verify_no_oob_idx_param{param}_strip_{idx}")
            };
            let (contract, lemma) = match acc.kind {
                GlobalStripKind::RowStrip => (
                    "CudaOxide.Contracts.NoOobRowStrip",
                    "CudaOxide.Proofs.Bounds.no_oob_row_strip",
                ),
                GlobalStripKind::ColStrip => (
                    "CudaOxide.Contracts.NoOobColStrip",
                    "CudaOxide.Proofs.Bounds.no_oob_col_strip",
                ),
            };
            if acc.offset == 0 {
                theorems.push(format!(
                    "theorem {thm_name} (bufferLen : Nat) :\n    \
                     {contract} bufferLen := by\n  \
                     exact {lemma}"
                ));
            } else {
                let fallback_tactic = match acc.kind {
                    GlobalStripKind::RowStrip => {
                        "intro _start _cols _col _hpos _hlast _hcol\n  omega"
                    }
                    GlobalStripKind::ColStrip => {
                        "intro _start stride rows row _hpos _hlast _hrow\n  \
                         have hle : row ≤ rows - 1 := by omega\n  \
                         have _hmul := Nat.mul_le_mul_right stride hle\n  \
                         omega"
                    }
                };
                theorems.push(format!(
                    "theorem {thm_name} (bufferLen : Nat) :\n    \
                     {contract} bufferLen {} := by\n  \
                     {fallback_tactic}",
                    acc.offset
                ));
            }
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
        for (idx, acc) in matching_1d.iter().enumerate() {
            let thm_name = if matching_1d.len() == 1 && matching_strips.is_empty() {
                format!("verify_no_oob_idx_param{param}")
            } else {
                format!("verify_no_oob_idx_param{param}_{idx}")
            };
            let guard = acc.guard_param;
            let offset = acc.index_offset;
            let effective_bs = if acc.group_size > 1 { 1 } else { bx };
            if offset == 0 && param == guard {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat) :\n    \
                     CudaOxide.Contracts.NoOobIdx {effective_bs} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_same_len"
                ));
            } else if offset == 0 && spec.requires_len_ge.contains(&(param, guard)) {
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat)\n    \
                     (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩) :\n    \
                     CudaOxide.Contracts.NoOobIdx {effective_bs} 0 (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     exact CudaOxide.Proofs.Bounds.no_oob_idx_of_guard_le h_req"
                ));
            } else {
                let req_binder = if spec.requires_len_ge.contains(&(param, guard)) {
                    format!(
                        "\n    (h_req : lengths ⟨{guard}, by decide⟩ ≤ lengths ⟨{param}, by decide⟩)"
                    )
                } else {
                    String::new()
                };
                theorems.push(format!(
                    "theorem {thm_name} (lengths : Fin {arity} → Nat){req_binder} :\n    \
                     CudaOxide.Contracts.NoOobIdx {effective_bs} {offset} (lengths ⟨{guard}, by decide⟩) (lengths ⟨{param}, by decide⟩) := by\n  \
                     intro block lane active\n  \
                     omega"
                ));
            }
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &param in &spec.disjoint_writes {
        let param_accesses: Vec<_> = global_1d_accesses
            .iter()
            .filter(|a| a.target_param == param)
            .collect();
        let writes: Vec<_> = param_accesses
            .iter()
            .copied()
            .filter(|a| a.kind == AccessKind::Write)
            .collect();
        if writes.is_empty() {
            return Err(reject(
                "disjoint_writes",
                format!(
                    "parameter {param} is listed in `disjoint_writes` but never written by the kernel"
                ),
            ));
        }
        for (idx, write) in writes.iter().enumerate() {
            let thm_name = if writes.len() == 1 {
                format!("verify_disjoint_writes_param{param}")
            } else {
                format!("verify_disjoint_writes_param{param}_{idx}")
            };
            if write.group_size > 1 {
                theorems.push(format!(
                    "theorem {thm_name} :\n    \
                     CudaOxide.Contracts.DisjointGroupWrites {bx} {} := by\n  \
                     exact CudaOxide.Proofs.Launch.disjoint_group_writes_1d",
                    write.group_size
                ));
            } else {
                theorems.push(format!(
                    "theorem {thm_name} :\n    \
                     CudaOxide.Contracts.DisjointWrites {bx} {} := by\n  \
                     exact CudaOxide.Proofs.Launch.disjoint_writes_1d",
                    write.index_offset
                ));
            }
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
        if param_accesses.len() > 1 {
            for (w_idx, write) in writes.iter().enumerate() {
                for (a_idx, other) in param_accesses.iter().enumerate() {
                    let pair_name =
                        format!("verify_disjoint_slice_pair_param{param}_{w_idx}_{a_idx}");
                    let off1 = write.index_offset;
                    let off2 = other.index_offset;
                    let body = if write.group_size == other.group_size && off1 == off2 {
                        "exact CudaOxide.Proofs.Launch.disjoint_slice_pair_same_offset"
                    } else {
                        "intro block₁ block₂ lane₁ lane₂ hlane₁ hlane₂ hsame\n  omega"
                    };
                    let eff_bs = if write.group_size > 1 && write.group_size == other.group_size {
                        1
                    } else {
                        bx
                    };
                    theorems.push(format!(
                        "theorem {pair_name} :\n    \
                         CudaOxide.Contracts.DisjointSliceAccessPair {eff_bs} {off1} {off2} := by\n  \
                         {body}"
                    ));
                    axiom_prints.push(format!("#print axioms {pair_name}"));
                }
            }
        }
    }

    for &shared_id in &spec.no_oob_shared {
        let matching: Vec<_> = shared_accesses
            .iter()
            .filter(|a| a.shared_id == shared_id)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_oob_idx",
                format!(
                    "shared array {shared_id} is listed in `no_oob_idx` but never accessed in the kernel"
                ),
            ));
        }
        for (idx, acc) in matching.iter().enumerate() {
            let thm_name = format!("verify_no_oob_idx_shared{shared_id}_{idx}");
            let pat_str = format_tile_pattern(&acc.pattern);
            let buf_len = acc.buffer_len;
            let loop_bound = acc.loop_bound;
            let tx_bound = acc.active_threads;
            let fits_2d = by * bx <= buf_len;
            let proof_line = match &acc.pattern {
                TileAccessPatternKind::ThreadSlot { stride }
                    if *stride == tx_bound && by * tx_bound <= buf_len =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_slot (by decide)".to_string()
                }
                TileAccessPatternKind::ThreadSlot { .. } if by == 1 && tx_bound <= buf_len => {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_1d_slot (by decide)".to_string()
                }
                TileAccessPatternKind::RowScan { stride }
                    if *stride == bx && loop_bound == bx && fits_2d =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_row_scan (by decide)".to_string()
                }
                TileAccessPatternKind::ColScan { stride }
                    if *stride == bx && loop_bound == by && fits_2d =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_col_scan (by decide)".to_string()
                }
                TileAccessPatternKind::GroupSlot { group_size }
                    if *group_size > 0 && tx_bound.saturating_sub(1) / *group_size < buf_len =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_group_slot (by decide) (by decide)"
                        .to_string()
                }
                TileAccessPatternKind::LinearScan if loop_bound <= buf_len => {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_linear_scan (by decide)".to_string()
                }
                TileAccessPatternKind::PackedSlot { pack }
                    if loop_bound <= *pack && tx_bound * *pack <= buf_len =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_packed_slot (by decide) (by decide)"
                        .to_string()
                }
                TileAccessPatternKind::GroupLaneScan { group_size, pack }
                    if *group_size > 0
                        && loop_bound <= *pack
                        && *group_size * *pack <= buf_len =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_group_lane_scan (by decide) (by decide) (by decide)"
                        .to_string()
                }
                TileAccessPatternKind::BitSplitSlot { group_size, stride }
                    if *group_size > 0
                        && *group_size <= *stride
                        && (tx_bound.saturating_sub(1) / *group_size + 1) * *stride <= buf_len =>
                {
                    "exact CudaOxide.Proofs.Bounds.no_oob_tile_bit_split_slot (by decide) (by decide) (by decide)"
                        .to_string()
                }
                _ => {
                    if acc.guard != GuardExpr::True {
                        let guard_str = format_guard_expr(&acc.guard);
                        theorems.push(format!(
                            "theorem {thm_name} :\n    \
                             CudaOxide.Contracts.NoOobGuardedTileAccess {tx_bound} {by} {loop_bound} {guard_str} ({pat_str}) {buf_len} := by\n  \
                             intro tx ty i htx hty hi hguard\n  \
                             simp only [CudaOxide.Mir.Kernel.TileAccessPattern.address, CudaOxide.Mir.Kernel.IdxExpr.eval, CudaOxide.Mir.Kernel.GuardExpr.eval, CudaOxide.Mir.globalIndex] at hguard ⊢\n  \
                             omega"
                        ));
                        axiom_prints.push(format!("#print axioms {thm_name}"));
                        continue;
                    }
                    "intro tx ty i htx hty hi\n  simp only [CudaOxide.Mir.Kernel.TileAccessPattern.address, CudaOxide.Mir.Kernel.IdxExpr.eval, CudaOxide.Mir.globalIndex]\n  omega"
                        .to_string()
                }
            };
            theorems.push(format!(
                "theorem {thm_name} :\n    \
                 CudaOxide.Contracts.NoOobTileAccess {tx_bound} {by} {loop_bound} ({pat_str}) {buf_len} := by\n  \
                 {proof_line}"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    for &shared_id in &spec.no_data_race {
        let matching: Vec<_> = shared_accesses
            .iter()
            .filter(|a| a.shared_id == shared_id)
            .collect();
        if matching.is_empty() {
            return Err(reject(
                "no_data_race",
                format!(
                    "shared array {shared_id} is listed in `no_data_race` but never accessed in the kernel"
                ),
            ));
        }
        for (i, a1) in matching.iter().enumerate() {
            for (j, a2) in matching.iter().enumerate().skip(i) {
                if a1.kind == AccessKind::Read && a2.kind == AccessKind::Read {
                    continue;
                }
                let thm_name = format!("verify_no_data_race_shared{shared_id}_{i}_{j}");
                let a1_str = format_phased_access(a1);
                let a2_str = format_phased_access(a2);
                let has_custom_guard = a1.guard != GuardExpr::True || a2.guard != GuardExpr::True;
                let proof_line = if a1.phase != a2.phase {
                    "exact CudaOxide.Proofs.Launch.no_data_race_diff_phase (by decide)".to_string()
                } else if !has_custom_guard
                    && a1.pattern == (TileAccessPatternKind::ThreadSlot { stride: bx })
                    && a2.pattern == (TileAccessPatternKind::ThreadSlot { stride: bx })
                    && a1.loop_bound == a2.loop_bound
                    && !a1.leader_only
                    && !a2.leader_only
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_thread_slot".to_string()
                } else if !has_custom_guard
                    && by == 1
                    && a1.leader_only
                    && a2.leader_only
                    && a1.loop_bound == a2.loop_bound
                    && matches!(a1.pattern, TileAccessPatternKind::GroupSlot { .. })
                    && a1.pattern == a2.pattern
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_group_leader_slot".to_string()
                } else if !has_custom_guard
                    && by == 1
                    && a1.leader_only
                    && a2.leader_only
                    && a1.loop_bound == a2.loop_bound
                    && a1.pattern == TileAccessPatternKind::LinearScan
                    && a2.pattern == TileAccessPatternKind::LinearScan
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_single_leader".to_string()
                } else if !has_custom_guard
                    && by == 1
                    && !a1.leader_only
                    && !a2.leader_only
                    && let (
                        TileAccessPatternKind::PackedSlot { pack: p1 },
                        TileAccessPatternKind::PackedSlot { pack: p2 },
                    ) = (&a1.pattern, &a2.pattern)
                    && p1 == p2
                    && a1.loop_bound <= *p1
                    && a2.loop_bound <= *p2
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_packed_slot (by decide) (by decide)"
                        .to_string()
                } else if !has_custom_guard
                    && by == 1
                    && !a1.leader_only
                    && !a2.leader_only
                    && a1.loop_bound == a2.loop_bound
                    && let (
                        TileAccessPatternKind::BitSplitSlot {
                            group_size: g1,
                            stride: s1,
                        },
                        TileAccessPatternKind::BitSplitSlot {
                            group_size: g2,
                            stride: s2,
                        },
                    ) = (&a1.pattern, &a2.pattern)
                    && g1 == g2
                    && s1 == s2
                    && *g1 > 0
                    && *g1 <= *s1
                {
                    "exact CudaOxide.Proofs.Launch.no_data_race_bit_split_slot (by decide) (by decide)"
                        .to_string()
                } else {
                    "intro tx₁ ty₁ i₁ tx₂ ty₂ i₂ htx₁ hty₁ hi₁ htx₂ hty₂ hi₂ hact₁ hact₂ _ _ _ hsame\n  \
                     simp only [CudaOxide.Mir.Kernel.TileAccessPattern.address, CudaOxide.Mir.Kernel.PhasedTileAccess.active, CudaOxide.Mir.Kernel.IdxExpr.eval, CudaOxide.Mir.Kernel.GuardExpr.eval, CudaOxide.Mir.globalIndex] at hact₁ hact₂ hsame\n  \
                     omega"
                        .to_string()
                };
                theorems.push(format!(
                    "theorem {thm_name} :\n    \
                     CudaOxide.Contracts.NoDataRace {bx} {by}\n      {a1_str}\n      {a2_str} := by\n  \
                     {proof_line}"
                ));
                axiom_prints.push(format!("#print axioms {thm_name}"));
            }
        }
    }

    let need_uniform_barriers = spec.uniform_barriers
        || (!spec.no_data_race.is_empty() && num_barriers > 0)
        || (control_flow_cfg.num_barriers > 0 && !spec.tmem_lifecycle);
    let emit_cfg = need_uniform_barriers || spec.tmem_lifecycle;
    let cfg_def_section = if emit_cfg {
        format!("{}\n\n", format_control_flow_cfg(bx, by, control_flow_cfg))
    } else {
        String::new()
    };

    if need_uniform_barriers {
        let thm_name = "verify_uniform_barriers";
        theorems.push(format!(
            "theorem {thm_name} :\n    \
             CudaOxide.Contracts.ControlFlow.UniformBarriers controlFlowCfg := by\n  \
             exact CudaOxide.Proofs.ControlFlow.uniform_barriers_sound (by decide)"
        ));
        axiom_prints.push(format!("#print axioms {thm_name}"));
    }

    if spec.tmem_lifecycle {
        let lc_thm = "verify_tmem_lifecycle";
        theorems.push(format!(
            "theorem {lc_thm} :\n    \
             CudaOxide.Contracts.ControlFlow.TmemLifecycle controlFlowCfg := by\n  \
             exact CudaOxide.Proofs.ControlFlow.tmem_lifecycle_sound (by decide)"
        ));
        axiom_prints.push(format!("#print axioms {lc_thm}"));

        for (idx, acc) in tmem_accesses.iter().enumerate() {
            let thm_name = format!("verify_no_oob_tmem_{idx}");
            let acc_str = format_tmem_access(acc);
            let alloc_cols = acc.alloc_cols;
            let is_warp_lane_32 = matches!(
                &acc.lane_expr,
                IdxExpr::Mul(lhs, rhs)
                    if matches!(lhs.as_ref(), IdxExpr::Div(inner, 32) if **inner == IdxExpr::Tx)
                        && **rhs == IdxExpr::Const(32)
            );
            let proof_line = if !acc.warp_restricted
                && let (IdxExpr::Const(lane_off), IdxExpr::Const(col_off)) =
                    (&acc.lane_expr, &acc.col_expr)
                && lane_off + acc.height_lanes <= 128
                && col_off + acc.width_cols <= alloc_cols
                && alloc_cols <= 512
            {
                "exact CudaOxide.Proofs.Bounds.no_oob_tmem_mma (by decide) (by decide) (by decide)"
                    .to_string()
            } else if acc.warp_restricted
                && acc.guard == GuardExpr::True
                && bx <= 128
                && acc.height_lanes == 32
                && acc.loop_bound > 0
                && is_warp_lane_32
                && matches!(acc.col_expr, IdxExpr::LoopVar { .. })
                && (acc.loop_bound - 1) + acc.width_cols <= alloc_cols
                && alloc_cols <= 512
            {
                "exact CudaOxide.Proofs.Bounds.no_oob_tmem_warp_ld_unit_scan (by decide) (by decide) (by decide) (by decide)"
                    .to_string()
            } else {
                "intro tx ty i htx hty hi hguard\n  \
                 simp only [CudaOxide.Mir.Kernel.IdxExpr.eval, CudaOxide.Mir.Kernel.GuardExpr.eval] at *\n  \
                 refine ⟨by omega, by omega, by omega, ?_⟩\n  \
                 intro _hwarp\n  \
                 omega"
                    .to_string()
            };
            theorems.push(format!(
                "theorem {thm_name} :\n    \
                 CudaOxide.Contracts.NoOobTmemAccess {bx} {by} {alloc_cols}\n      {acc_str} := by\n  \
                 {proof_line}"
            ));
            axiom_prints.push(format!("#print axioms {thm_name}"));
        }
    }

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated from PostMem2Reg dialect-mir (2D Phased Shared-Memory Kernel).\n\
         import CudaOxide\n\n\
         set_option linter.unusedSimpArgs false\n\
         set_option maxRecDepth 4096\n\n\
         namespace CudaOxide.Generated\n\n\
         def kernelSummary : CudaOxide.Mir.Kernel.TiledKernelSummary2D :=\n  \
         {{ blockDimX := {bx}\n  \
         , blockDimY := {by}\n  \
         , numBarriers := {num_barriers}\n  \
         , sharedAccesses := [\n{}\n    ]\n  }}\n\n\
         {cfg_def_section}\
         {}\n\n\
         {}\n\n\
         end CudaOxide.Generated\n",
        access_entries.join(",\n"),
        theorems.join("\n\n"),
        axiom_prints.join("\n"),
    ))
}
