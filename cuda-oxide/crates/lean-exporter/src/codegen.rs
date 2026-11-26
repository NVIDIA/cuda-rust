/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Renders a [`TranslatedFunction`] into a `.lean` source file.
//!
//! ```text
//!   TranslatedFunction
//!     ├── Scalar     ──► emits `CudaOxide.Mir.Scalar.Expr` + `evaluate`
//!     └── MirProgram ──► emits `CudaOxide.Mir.Program.MirFunc` + `ControlFlowCfg`
//!                        + `MirNoOobGlobal`, `MirNoOobShared`, `MirDisjointWrites`,
//!                          `MirNoSharedRace`, `MirAssertRedundant`, `MirNoOobTmem`,
//!                          `UniformBarriers`, `TmemLifecycle`, `MirWritesAllElems`
//! ```

use crate::{
    ExportError,
    spec::KernelVerificationSpec,
    ssa::{
        CfgBranchCond, CfgOpItem, GuardExpr, IdxExpr, TranscribedCfgBlock,
        TranscribedControlFlowCfg,
    },
    translate::{
        TranslatedFunction,
        program::{MirParamKind, MirProgramData, MirRhs, MirStmt, MirTerminator},
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
        TranslatedFunction::MirProgram { spec, program, .. } => {
            render_mir_program_module(spec, program)
        }
    }
}

/// Return grouped, human-readable obligation descriptions proven for `spec`.
pub fn proved_obligation_descriptions(spec: &KernelVerificationSpec) -> Vec<String> {
    let fmt_ids = |prefix: &str, ids: &[usize]| -> String {
        ids.iter()
            .map(|id| format!("{prefix} {id}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut items = Vec::new();
    if !spec.no_oob_idx.is_empty() {
        items.push(format!(
            "no_oob_idx({}) -> CudaOxide.Contracts.Program.MirNoOobGlobal",
            fmt_ids("param", &spec.no_oob_idx)
        ));
    }
    if !spec.disjoint_writes.is_empty() {
        items.push(format!(
            "disjoint_writes({}) -> CudaOxide.Contracts.Program.MirDisjointWrites",
            fmt_ids("param", &spec.disjoint_writes)
        ));
    }
    if !spec.writes_all_elems.is_empty() {
        items.push(format!(
            "writes_all_elems({}) -> CudaOxide.Contracts.Program.MirWritesAllElems",
            fmt_ids("param", &spec.writes_all_elems)
        ));
    }
    if !spec.no_oob_shared.is_empty() {
        items.push(format!(
            "no_oob_idx({}) -> CudaOxide.Contracts.Program.MirNoOobShared",
            fmt_ids("shared", &spec.no_oob_shared)
        ));
    }
    if !spec.no_data_race.is_empty() {
        items.push(format!(
            "no_data_race({}) -> CudaOxide.Contracts.Program.MirNoSharedRace",
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
            "tmem_lifecycle -> CudaOxide.Contracts.ControlFlow.TmemLifecycle + CudaOxide.Contracts.Program.MirNoOobTmem"
                .to_string(),
        );
    }
    items
}

/// Format a compositional [`IdxExpr`] into its Lean `CudaOxide.Mir.ControlFlow.IdxExpr` constructor tree.
fn format_idx_expr(expr: &IdxExpr) -> String {
    match expr {
        IdxExpr::Tx => ".tx".to_string(),
        IdxExpr::Ty => ".ty".to_string(),
        IdxExpr::ClusterCtaidX => ".clusterCtaidX".to_string(),
        IdxExpr::LoopVar => ".loopVar".to_string(),
        IdxExpr::Const(n) => format!("(.const {n})"),
        IdxExpr::Add(a, b) => format!("(.add {} {})", format_idx_expr(a), format_idx_expr(b)),
        IdxExpr::Mul(a, b) => format!("(.mul {} {})", format_idx_expr(a), format_idx_expr(b)),
        IdxExpr::Div(a, d) => format!("(.div {} {d})", format_idx_expr(a)),
        IdxExpr::Mod(a, d) => format!("(.mod {} {d})", format_idx_expr(a)),
    }
}

/// Format a compositional [`GuardExpr`] into its Lean `CudaOxide.Mir.ControlFlow.GuardExpr` constructor tree.
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
        CfgBranchCond::MbarReady(b, init_free) => format!("(.mbarReady {b} {init_free})"),
        CfgBranchCond::MbarSpin(b) => format!("(.mbarSpin {b})"),
        CfgBranchCond::PosLoopExit(body) => format!("(.posLoopExit {body})"),
        CfgBranchCond::Uniform => ".uniform".to_string(),
        CfgBranchCond::WarpUniform => ".warpUniform".to_string(),
        CfgBranchCond::Unknown => ".unknown".to_string(),
    }
}

/// Format a [`CfgOpItem`] into a Lean `CudaOxide.Mir.ControlFlow.CfgOp` expression.
fn format_cfg_op(op: &CfgOpItem) -> String {
    match op {
        CfgOpItem::SyncThreads => ".syncThreads".to_string(),
        CfgOpItem::ClusterSync => ".clusterSync".to_string(),
        CfgOpItem::SharedRead { shared_id } => format!("(.sharedRead {shared_id})"),
        CfgOpItem::SharedWrite { shared_id } => format!("(.sharedWrite {shared_id})"),
        CfgOpItem::DsmemAccess { shared_id } => format!("(.dsmemAccess {shared_id})"),
        CfgOpItem::MbarInit { bar_id, count } => format!("(.mbarInit {bar_id} {count})"),
        CfgOpItem::MbarInval { bar_id } => format!("(.mbarInval {bar_id})"),
        CfgOpItem::ProxyFenceAsyncShared => ".proxyFenceAsyncShared".to_string(),
        CfgOpItem::MbarArrive { bar_id } => format!("(.mbarArrive {bar_id})"),
        CfgOpItem::MbarArriveExpectTx { bar_id, tx_bytes } => {
            format!("(.mbarArriveExpectTx {bar_id} {tx_bytes})")
        }
        CfgOpItem::TmaCopyG2S {
            shared_id,
            bar_id,
            is_cg2,
        } => format!("(.tmaCopyG2S {shared_id} {bar_id} {is_cg2})"),
        CfgOpItem::ClcTryCancel { shared_id, bar_id } => {
            format!("(.clcTryCancel {shared_id} {bar_id})")
        }
        CfgOpItem::TmemAlloc {
            shared_id,
            cols,
            is_cg2,
        } => format!("(.tmemAlloc {shared_id} {cols} {is_cg2})"),
        CfgOpItem::TmemDealloc { cols, is_cg2 } => {
            format!("(.tmemDealloc {cols} {is_cg2})")
        }
        CfgOpItem::TmemRelinquishPermit { is_cg2 } => {
            format!("(.tmemRelinquishPermit {is_cg2})")
        }
        CfgOpItem::TmemAsyncIssue { smem_ids, is_cg2 } => {
            let list = smem_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("(.tmemAsyncIssue [{list}] {is_cg2})")
        }
        CfgOpItem::TmemCommit { bar_id, is_cg2 } => format!("(.tmemCommit {bar_id} {is_cg2})"),
        CfgOpItem::TmemLoad { ld_id } => format!("(.tmemLoad {ld_id})"),
        CfgOpItem::TmemLoadWait => ".tmemLoadWait".to_string(),
        CfgOpItem::TmemLoadUse { ld_id } => format!("(.tmemLoadUse {ld_id})"),
        CfgOpItem::TmemStoreWait => ".tmemStoreWait".to_string(),
        CfgOpItem::TmemFenceBeforeSync => ".tmemFenceBeforeSync".to_string(),
        CfgOpItem::TmemFenceAfterSync => ".tmemFenceAfterSync".to_string(),
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
fn format_control_flow_cfg(bx: u64, by: u64, cdx: u64, cfg: &TranscribedControlFlowCfg) -> String {
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
         , clusterDimX := {cdx}\n  \
         , numShared := {}\n  \
         , numBarriers := {}\n  \
         , blocks := [\n{block_lines}\n    ]\n  }}",
        cfg.num_shared, cfg.num_barriers
    )
}

/// Format a [`MirRhs`] into its `CudaOxide.Mir.Program.Rhs` Lean constructor.
fn format_mir_rhs(rhs: &MirRhs) -> String {
    match rhs {
        MirRhs::Copy(src) => format!("(.copy {src})"),
        MirRhs::ConstInt(w, val) => format!("(.constInt {w} {val})"),
        MirRhs::ConstBool(b) => format!("(.constBool {b})"),
        MirRhs::Sreg(r) => format!("(.sreg .{r})"),
        MirRhs::Index1D => ".index1D".to_string(),
        MirRhs::Coord2DU32 => ".coord2DU32".to_string(),
        MirRhs::SlicePtr(p) => format!("(.slicePtr {p})"),
        MirRhs::SliceLen(p) => format!("(.sliceLen {p})"),
        MirRhs::SharedBase(s) => format!("(.sharedBase {s})"),
        MirRhs::BinOp(op, w, l, r) => format!("(.binOp .{op} {w} {l} {r})"),
        MirRhs::CheckedOvf(op, w, l, r) => format!("(.checkedOvf .{op} {w} {l} {r})"),
        MirRhs::CmpOp(op, l, r) => format!("(.cmpOp .{op} {l} {r})"),
        MirRhs::NotBool(s) => format!("(.notBool {s})"),
        MirRhs::IntCast(dw, s) => format!("(.intCast {dw} {s})"),
        MirRhs::PtrOffset(b, i) => format!("(.ptrOffset {b} {i})"),
        MirRhs::Load(p) => format!("(.load {p})"),
        MirRhs::MbarTryWait(b, s) => format!("(.mbarTryWait {b} {s})"),
        MirRhs::CvtaShared(p) => format!("(.cvtaShared {p})"),
        MirRhs::LoadTmemBase(p, cols) => format!("(.loadTmemBase {p} {cols})"),
        MirRhs::OpaqueUint(id, w) => format!("(.opaqueUint {id} {w})"),
        MirRhs::OpaqueVal => ".opaqueVal".to_string(),
    }
}

/// Render a line-by-line [`MirProgramData`] (`CudaOxide.Mir.Program.MirFunc`) and its
/// declarative safety obligations (`MirNoOobGlobal`, `MirNoOobShared`, `MirDisjointWrites`,
/// `MirNoSharedRace`, `MirAssertRedundant`, `MirNoOobTmem`, `UniformBarriers`, `TmemLifecycle`,
/// and `MirWritesAllElems`).
fn render_mir_program_module(
    spec: &KernelVerificationSpec,
    program: &MirProgramData,
) -> Result<String, ExportError> {
    let bx = u64::from(spec.block_size);
    let by = u64::from(spec.block_y);
    let cdx = u64::from(spec.cluster_dim_x);
    let param_strs: Vec<String> = program
        .params
        .iter()
        .map(|pk| match pk {
            MirParamKind::Scalar(w) => format!("(.scalar {w})"),
            MirParamKind::Slice(m) => format!("(.slice {m})"),
            MirParamKind::DisjointSlice => ".disjointSlice".to_string(),
        })
        .collect();
    let req_strs: Vec<String> = spec
        .requires_len_ge
        .iter()
        .map(|&(l, r)| format!("({l}, {r})"))
        .collect();
    let shared_strs: Vec<String> = program
        .shared_decls
        .iter()
        .map(|(_, sz, eb)| format!("{{ size := {sz}, elemBytes := {eb} }}"))
        .collect();

    let mut blk_strs = Vec::with_capacity(program.blocks.len());
    for blk in &program.blocks {
        let arg_strs: Vec<String> = blk
            .args
            .iter()
            .map(|&(vid, w)| format!("⟨{vid}, {w}⟩"))
            .collect();
        let stmt_strs: Vec<String> = blk
            .stmts
            .iter()
            .map(|s| match s {
                MirStmt::Assign(dst, rhs) => format!("(.assign {dst} {})", format_mir_rhs(rhs)),
                MirStmt::Store(p, v) => format!("(.store {p} {v})"),
                MirStmt::AssertCond(aid, c) => format!("(.assertCond {aid} {c})"),
                MirStmt::SyncThreads => ".syncThreads".to_string(),
                MirStmt::BarrierSlotCheck(p) => format!("(.barrierSlotCheck {p})"),
                MirStmt::ClusterRankCheck(r) => format!("(.clusterRankCheck {r})"),
                MirStmt::MbarArrive(b, s) => format!("(.mbarArrive {b} {s})"),
                MirStmt::MbarArriveExpectTx(b, s, tx) => {
                    format!("(.mbarArriveExpectTx {b} {s} {tx})")
                }
                MirStmt::TmaCopyG2S(p, b, s, sp) => {
                    format!("(.tmaCopyG2S {p} {b} {s} {sp})")
                }
                MirStmt::TmemAlloc(p, cols) => format!("(.tmemAlloc {p} {cols})"),
                MirStmt::SmemDescRead(d, tb) => format!("(.smemDescRead {d} {tb})"),
                MirStmt::TmemAccess(tv, hl, wc, wr) => {
                    format!("(.tmemAccess {tv} {hl} {wc} {wr})")
                }
            })
            .collect();
        let term_str = match &blk.term {
            MirTerminator::Ret => ".ret".to_string(),
            MirTerminator::Unreachable => ".unreachable".to_string(),
            MirTerminator::Goto(dst, args) => {
                let a = args
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("(.goto {dst} [{a}])")
            }
            MirTerminator::CondBr(c, tb, ta, fb, fa) => {
                let tas = ta
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let fas = fa
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("(.condBr {c} {tb} [{tas}] {fb} [{fas}])")
            }
        };
        blk_strs.push(format!(
            "    {{ args := [{}], isLoopHeader := {}, stmts := [{}], term := {term_str} }}",
            arg_strs.join(", "),
            blk.is_loop_header,
            stmt_strs.join(", "),
        ));
    }

    let need_uniform_barriers = spec.uniform_barriers
        || (program.control_flow_cfg.num_barriers > 0 && !spec.tmem_lifecycle);
    let need_cfg = need_uniform_barriers || spec.tmem_lifecycle;
    let cfg_section = if need_cfg {
        format!(
            "{}\n\ncfg_def_state_map controlFlowCfg\n\n",
            format_control_flow_cfg(bx, by, cdx, &program.control_flow_cfg)
        )
    } else {
        String::new()
    };

    let mut theorems = vec![
        "mir_def_sym_table mirFunc\n\n\
         theorem mirFunc_valid :\n    \
         CudaOxide.Proofs.Program.validSymFunc mirFunc symTable = true :=\n  \
         rfl\n\n\
         theorem mirEvents_eq :\n    \
         CudaOxide.Proofs.Program.collectSymMemEvents mirFunc symTable = symMemEvents :=\n  \
         rfl"
        .to_string(),
    ];
    if !program.candidate_asserts.is_empty() {
        theorems.push(
            "theorem mirAsserts_eq :\n    \
             CudaOxide.Proofs.Program.collectSymAssertEvents mirFunc symTable = symAssertEvents :=\n  \
             rfl"
                .to_string(),
        );
    }
    if spec.tmem_lifecycle {
        theorems.push(
            "theorem mirTmem_eq :\n    \
             CudaOxide.Proofs.Program.collectSymTmemEvents mirFunc symTable = symTmemEvents :=\n  \
             rfl"
            .to_string(),
        );
    }
    if program.has_cluster_rank_checks {
        theorems.push(
            "theorem mirRanks_eq :\n    \
             CudaOxide.Proofs.Program.collectSymRankEvents mirFunc symTable = symRankEvents :=\n  \
             rfl"
            .to_string(),
        );
    }
    let mut axiom_prints = vec!["#print axioms mirFunc_valid".to_string()];

    for &p in &spec.no_oob_idx {
        let thm = format!("verify_no_oob_idx_param{p}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirNoOobGlobal mirFunc {p} := by\n  \
             apply vcNoOobGlobal_of_table mirFunc_valid mirEvents_eq (by mir_bind_events)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }
    for &p in &spec.disjoint_writes {
        let thm = format!("verify_disjoint_writes_param{p}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirDisjointWrites mirFunc {p} := by\n  \
             apply vcDisjointWrites_of_table mirFunc_valid mirEvents_eq (by mir_bind_pairs)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }
    for &p in &spec.writes_all_elems {
        let thm = format!("verify_writes_all_elems_param{p}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirWritesAllElems mirFunc {p} := by\n  \
             apply vcWritesAllElems rfl (by mir_bind_writes)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }
    for &sid in &spec.no_oob_shared {
        let thm = format!("verify_no_oob_idx_shared{sid}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirNoOobShared mirFunc {sid} := by\n  \
             apply vcNoOobShared_of_table mirFunc_valid mirEvents_eq (by mir_bind_nat) (by mir_bind_events)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }
    if need_uniform_barriers {
        theorems.push(
            "theorem verify_uniform_barriers :\n    \
             CudaOxide.Contracts.ControlFlow.UniformBarriers controlFlowCfg := by\n  \
             exact CudaOxide.Proofs.ControlFlow.uniform_barriers_of_stateMap (cfg := controlFlowCfg) (inv := cfgStateMap) rfl"
                .to_string(),
        );
        axiom_prints.push("#print axioms verify_uniform_barriers".to_string());
    }
    for &sid in &spec.no_data_race {
        let thm = format!("verify_no_data_race_shared{sid}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirNoSharedRace mirFunc controlFlowCfg {sid} := by\n  \
             apply vcNoSharedRace_of_table verify_uniform_barriers mirFunc_valid mirEvents_eq (by mir_bind_shared_pairs)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }
    if spec.tmem_lifecycle {
        theorems.push(
            "theorem verify_no_oob_tmem :\n    \
             CudaOxide.Contracts.Program.MirNoOobTmem mirFunc := by\n  \
             apply vcNoOobTmem_of_table mirFunc_valid mirTmem_eq (by mir_bind_tmem)\n  \
             mir_vc mirFunc"
                .to_string(),
        );
        axiom_prints.push("#print axioms verify_no_oob_tmem".to_string());
    }
    if spec.tmem_lifecycle {
        theorems.push(
            "theorem verify_tmem_lifecycle :\n    \
             CudaOxide.Contracts.ControlFlow.TmemLifecycle controlFlowCfg := by\n  \
             exact CudaOxide.Proofs.ControlFlow.tmem_lifecycle_of_stateMap (cfg := controlFlowCfg) (inv := cfgStateMap) rfl"
                .to_string(),
        );
        axiom_prints.push("#print axioms verify_tmem_lifecycle".to_string());
    }
    if program.has_cluster_rank_checks {
        theorems.push(
            "theorem verify_no_oob_cluster_rank :\n    \
             CudaOxide.Contracts.Program.MirNoOobClusterRank mirFunc := by\n  \
             apply vcNoOobClusterRank_of_table mirFunc_valid mirRanks_eq (by mir_bind_ranks)\n  \
             mir_vc mirFunc"
                .to_string(),
        );
        axiom_prints.push("#print axioms verify_no_oob_cluster_rank".to_string());
    }
    for &(aid, _) in &program.candidate_asserts {
        let thm = format!("verify_assert_redundant_{aid}");
        theorems.push(format!(
            "theorem {thm} :\n    \
             CudaOxide.Contracts.Program.MirAssertRedundant mirFunc {aid} := by\n  \
             apply vcAssertRedundant_of_table mirFunc_valid mirAsserts_eq (by mir_bind_asserts)\n  \
             mir_vc mirFunc"
        ));
        axiom_prints.push(format!("#print axioms {thm}"));
    }

    Ok(format!(
        "-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\
         -- SPDX-License-Identifier: Apache-2.0\n\
         -- Generated line by line from PostMem2Reg dialect-mir (`CudaOxide.Mir.Program.MirFunc`).\n\
         import CudaOxide\n\n\
         set_option linter.unusedSimpArgs false\n\
         set_option maxRecDepth 65536\n\
         set_option maxHeartbeats 20000000\n\n\
         namespace CudaOxide.Generated\n\
         open CudaOxide.Proofs.Program\n\
         open CudaOxide.Proofs.ControlFlow\n\n\
         def mirFunc : CudaOxide.Mir.Program.MirFunc :=\n  \
         {{ domain := {}\n  \
         , blockDimX := {bx}\n  \
         , blockDimY := {by}\n  \
         , clusterDimX := {cdx}\n  \
         , u32Coords := {}\n  \
         , params := [{}]\n  \
         , requiresLenGe := [{}]\n  \
         , sharedDecls := [{}]\n  \
         , blocks := [\n{}\n    ]\n  }}\n\n\
         {cfg_section}\
         {}\n\n\
         {}\n\n\
         end CudaOxide.Generated\n",
        spec.domain,
        spec.u32_coordinates,
        param_strs.join(", "),
        req_strs.join(", "),
        shared_strs.join(", "),
        blk_strs.join(",\n"),
        theorems.join("\n\n"),
        axiom_prints.join("\n"),
    ))
}
