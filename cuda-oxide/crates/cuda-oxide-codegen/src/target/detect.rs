/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::features::{DetectedFeatures, ModuleRequirements, PtxIsaRequirement};
use crate::error::PipelineError;
use std::path::Path;

pub(super) fn contains_wgmma_features(contents: &str) -> bool {
    contents.contains("wgmma.fence")
        || contents.contains("wgmma.commit_group")
        || contents.contains("wgmma.wait_group")
        || contents.contains("wgmma.mma_async")
}

/// Checks for Thread Block Cluster instructions (sm_90+).
///
/// Cluster features require Hopper (sm_90) or newer:
/// - Cluster special registers (%cluster_ctaid, %cluster_nctaid)
/// - Cluster synchronization (cluster.sync)
/// - Distributed shared memory (mapa.shared::cluster)
pub(super) fn contains_cluster_features(contents: &str) -> bool {
    // Cluster special registers
    contents.contains("cluster_ctaid")
        || contents.contains("cluster_nctaid")
        || contents.contains("cluster_ctarank")
        || contents.contains("cluster_nctarank")
        || contents.contains("%clusterid")
        || contents.contains("%nclusterid")
        || contents.contains("%is_explicit_cluster")
        || contents.contains("!\"cluster_dim_x\"")
        || contents.contains("!\"cluster_dim_y\"")
        || contents.contains("!\"cluster_dim_z\"")
        // Cluster synchronization
        || contents.contains("cluster.sync")
        || contents.contains("barrier.cluster.")
        // Distributed shared memory
        || contents.contains("mapa.shared::cluster")
        || contents.contains(".shared::cluster")
        || contains_cluster_fence_features(contents)
        || contains_cluster_scoped_memory_features(contents)
}

fn contains_cluster_fence_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.sc.cluster")
            || statement.contains("fence.acq_rel.cluster")
            || statement.contains("fence.acquire.cluster")
            || statement.contains("fence.release.cluster")
    })
}

fn contains_cluster_scoped_memory_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        !statement.contains("multimem.")
            && statement.contains(".cluster.")
            && ["ld.", "st.", "atom.", "red."]
                .iter()
                .any(|mnemonic| statement.contains(mnemonic))
    })
}

/// Checks the one-way fence semantics added in PTX 8.6.
///
/// Unlike the older `.sc` / `.acq_rel` forms, `.acquire` and `.release`
/// require sm_90 for every scope, not just `.cluster`.
fn contains_fence_acquire_release_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.acquire.") || statement.contains("fence.release.")
    })
}

/// Checks the multimem instruction family introduced for sm_90.
///
/// Base forms need PTX 8.1. The pipeline currently has no 8.1 feature switch,
/// so PTX 8.6 is the nearest conservative version supported by LLVM.
fn contains_multimem_features(contents: &str) -> bool {
    contents.split(';').any(is_multimem_instruction)
}

fn is_multimem_instruction(statement: &str) -> bool {
    ["multimem.ld_reduce", "multimem.st", "multimem.red"]
        .iter()
        .any(|instruction| statement.contains(instruction))
}

/// Checks PTX 8.6 multimem formats that require a Blackwell family target.
fn contains_multimem_blackwell_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        is_multimem_instruction(statement)
            && [".e4m3", ".e5m2", ".acc::f16"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// Checks the PTX 8.6 floating-point extension to `redux.sync`.
fn contains_redux_f32_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("redux.sync") && statement.contains(".f32"))
}

/// Checks for forward-compatible instructions whose minimum target is sm_90.
///
/// Keep this category architecture-neutral: unlike WGMMA, these instructions
/// are not Hopper-specific and remain available on newer architectures.
pub(super) fn contains_sm90_features(contents: &str) -> bool {
    ["add.rn.bf16x2", "sub.rn.bf16x2", "mul.rn.bf16x2"]
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contains_packed_bf16_atomic_features(contents)
        || contains_stmatrix_features(contents)
        || contains_elect_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
}

/// Native two-lane f32 arithmetic requires PTX 8.6 and sm_100+.
///
/// Match the instruction family rather than cuda-oxide's currently emitted
/// spelling. External LLVM IR can contain any valid rounding/FTZ modifier
/// combination and must receive the same target floor.
pub(super) fn contains_f32x2_features(contents: &str) -> bool {
    contains_instruction_family_modifier(contents, &["add", "sub", "mul", "fma"], "f32x2")
}

/// Native packed bf16 atomic add was added in PTX 7.8 for sm_90.
pub(super) fn contains_packed_bf16_atomic_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "atom.global.add.noftz.bf16x2")
}

/// Packed f16 atomic add needs PTX 6.2. Its hardware floor predates
/// cuda-oxide's Volta floor, so only the independent PTX ISA requirement must
/// be raised.
pub(super) fn contains_packed_f16_atomic_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "atom.global.add.noftz.f16x2")
}

fn contains_elect_features(contents: &str) -> bool {
    contents.contains("elect.sync")
}

/// Checks for the register-only 8x8 matrix transpose (PTX 7.8, sm_75+).
pub(super) fn contains_movmatrix_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "movmatrix.sync.aligned.m8n8.trans.b16")
}

/// Checks the dense BF16 MMA form added by the typed device intrinsic.
///
/// MMA shapes and types have different architecture and PTX ISA floors, so
/// this intentionally matches the complete operation-specific mnemonic.
pub(super) fn contains_mma_m16n8k16_f32_bf16_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    )
}

/// Checks for the Ampere TF32 MMA operation (PTX 7.0, sm_80+).
pub(super) fn contains_mma_m16n8k8_f32_tf32_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    )
}

/// Checks the dense Ampere INT8 MMA forms (PTX 7.0, sm_80+).
pub(super) fn contains_dense_int8_mma_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the m8n8k16 INT8 MMA forms (PTX 6.5, sm_75+).
pub(super) fn contains_mma_m8n8k16_int8_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the m8n8k32 INT4 MMA forms (PTX 6.5, sm_75+).
pub(super) fn contains_mma_m8n8k32_int4_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the dense Ampere INT4 MMA forms (PTX 7.0, sm_80+).
pub(super) fn contains_dense_int4_mma_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

pub(super) const B1_XOR_MMA_MNEMONICS: &[&str] = &[
    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc",
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
];

pub(super) const B1_AND_MMA_MNEMONICS: &[&str] = &[
    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc",
    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc",
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
];

/// Checks the three dense binary XOR/POPC MMA forms (PTX 7.0).
pub(super) fn contains_b1_xor_mma_features(contents: &str) -> bool {
    B1_XOR_MMA_MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the three dense binary AND/POPC MMA forms (PTX 7.1, sm_80+).
pub(super) fn contains_b1_and_mma_features(contents: &str) -> bool {
    B1_AND_MMA_MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the only dense binary MMA form that can run below sm_80.
pub(super) fn contains_mma_m8n8k128_b1_xor_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, B1_XOR_MMA_MNEMONICS[0])
}

fn contains_sm80_b1_mma_features(contents: &str) -> bool {
    contains_b1_and_mma_features(contents)
        || B1_XOR_MMA_MNEMONICS[1..]
            .iter()
            .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks for the Ampere FP64 tensor-core MMA operation (PTX 7.0, sm_80+).
pub(super) fn contains_mma_m8n8k4_f64_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64")
}

/// Checks the dense F16 MMA form added by the typed device intrinsic.
///
/// MMA shapes and types have different architecture and PTX ISA floors, so
/// this intentionally matches the complete operation-specific mnemonic.
pub(super) fn contains_mma_m16n8k16_f32_f16_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    )
}

fn contains_instruction_mnemonic(contents: &str, mnemonic: &str) -> bool {
    contents.match_indices(mnemonic).any(|(index, _)| {
        let preceding = &contents[..index];
        let following = &contents[index + mnemonic.len()..];
        let escapes = ["\\09", "\\0A", "\\0B", "\\0C", "\\0D"];
        // Use PTX token delimiters rather than treating arbitrary punctuation
        // as a boundary. In particular, `$` and `%` participate in PTX
        // identifiers, and guarded opcodes have whitespace after `@{!}p`.
        let begins_at_instruction_boundary = preceding.is_empty()
            || preceding
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_whitespace() || matches!(ch, '"' | ';' | ':' | '{' | '}'))
            || escapes.iter().any(|escape| preceding.ends_with(escape))
            || preceding.ends_with("*/");
        let ends_at_instruction_boundary =
            following.chars().next().is_some_and(char::is_whitespace)
                || escapes.iter().any(|escape| following.starts_with(escape));
        begins_at_instruction_boundary && ends_at_instruction_boundary
    })
}

fn contains_instruction_family_modifier(
    contents: &str,
    operations: &[&str],
    required_modifier: &str,
) -> bool {
    const ESCAPED_WHITESPACE: [&str; 5] = ["\\09", "\\0A", "\\0B", "\\0C", "\\0D"];
    operations.iter().any(|operation| {
        contents.match_indices(operation).any(|(index, _)| {
            let preceding = &contents[..index];
            let following = &contents[index + operation.len()..];
            let begins_at_instruction_boundary = preceding.is_empty()
                || preceding.chars().next_back().is_some_and(|ch| {
                    ch.is_whitespace() || matches!(ch, '"' | ';' | ':' | '{' | '}')
                })
                || ESCAPED_WHITESPACE
                    .iter()
                    .any(|escape| preceding.ends_with(escape))
                || preceding.ends_with("*/");
            if !begins_at_instruction_boundary || !following.starts_with('.') {
                return false;
            }

            let plain_end = following
                .char_indices()
                .find_map(|(offset, ch)| {
                    (ch.is_whitespace() || matches!(ch, '"' | ';')).then_some(offset)
                })
                .unwrap_or(following.len());
            // An escape after the first ordinary delimiter cannot end this
            // token. Searching the whole suffix for every instruction makes
            // feature detection quadratic on large LLVM modules.
            let token_end = ESCAPED_WHITESPACE
                .iter()
                .filter_map(|escape| following[..plain_end].find(escape))
                .min()
                .unwrap_or(plain_end);
            following[..token_end]
                .split('.')
                .any(|modifier| modifier == required_modifier)
        })
    })
}

/// Checks the full PTX instruction families, including inline `ptx_asm!`
/// forms that cuda-oxide does not yet expose as typed wrappers.
///
/// Broad family matching is intentional. Missing a valid spelling can
/// silently select an architecture or PTX ISA that is too old; an invalid
/// spelling still reaches ptxas and fails there after conservative targeting.
fn contains_ldmatrix_features(contents: &str) -> bool {
    contents.contains("ldmatrix.sync.aligned.")
}

fn contains_stmatrix_features(contents: &str) -> bool {
    contents.contains("stmatrix.sync.aligned.")
}

/// PTX 8.6 matrix shapes/types have a Blackwell architecture-family floor.
fn contains_blackwell_matrix_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        let newer_ldmatrix = statement.contains("ldmatrix.sync.aligned.")
            && [".m16n16.", ".m8n16.", ".b8", ".src_fmt", ".dst_fmt"]
                .iter()
                .any(|token| statement.contains(token));
        let newer_stmatrix = statement.contains("stmatrix.sync.aligned.")
            && [".m16n8.", ".b8"]
                .iter()
                .any(|token| statement.contains(token));
        newer_ldmatrix || newer_stmatrix
    })
}

fn contains_ldmatrix_cta_state_space(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("ldmatrix.sync.aligned.") && statement.contains(".shared::cta.")
    })
}

/// Checks for features whose minimum target is sm_80.
///
/// This category includes packed bf16 operations introduced on Ampere and
/// non-bulk asynchronous copies. Match both the PTX spellings used in inline
/// assembly and the dotted LLVM NVVM intrinsic names for `cp.async`. Bulk and
/// tensor-copy forms have stronger requirements and are classified first.
pub(super) fn contains_sm80_features(contents: &str) -> bool {
    [
        "fma.rn.bf16x2",
        "fma.rn.relu.bf16x2",
        "min.bf16x2",
        "max.bf16x2",
        "neg.bf16x2",
        "abs.bf16x2",
    ]
    .iter()
    .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contains_mma_m8n8k4_f64_features(contents)
        || contents
            .split(';')
            .any(|statement| statement.contains("cvt.") && statement.contains(".bf16x2.f32"))
        || contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contents.contains("cp.async.ca.shared")
        || contents.contains("cp.async.cg.shared")
        || contents.contains("cp.async.commit_group")
        || contents.contains("cp.async.commit.group")
        || contents.contains("cp.async.wait_group")
        || contents.contains("cp.async.wait.group")
        || contents.contains("cp.async.wait_all")
        || contents.contains("cp.async.wait.all")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
        || contains_mma_m16n8k16_f32_f16_features(contents)
        || contains_mma_m16n8k8_f32_tf32_features(contents)
        || contains_dense_int8_mma_features(contents)
        || contains_dense_int4_mma_features(contents)
        || contains_sm80_b1_mma_features(contents)
}

/// Checks for TMA/mbarrier instructions (Hopper+ compatible with Blackwell).
///
/// These instructions work on BOTH Hopper and Blackwell:
/// - TMA: Tensor Memory Accelerator bulk copies
/// - mbarrier: Async hardware barriers with transaction tracking
///
/// The architecture floor is generic sm_90; automatic cross-compilation keeps
/// the existing sm_100 default for forward-compatible Blackwell PTX.
pub(super) fn contains_tma_features(contents: &str) -> bool {
    // TMA tensor copies and their commit/wait group controls.
    contains_cp_async_bulk_features(contents)
        || contains_mbarrier_sm90_features(contents)
        || contents.contains("fence.mbarrier_init")
        // Proxy fence for async operations
        || contents.contains("fence.proxy.async")
        || contents.contains(".sync_restrict")
}

fn contains_cp_async_bulk_features(contents: &str) -> bool {
    contents.contains("cp.async.bulk.")
}

fn contains_mbarrier_features(contents: &str) -> bool {
    contents.contains("mbarrier.") || contents.contains("llvm.nvvm.mbarrier")
}

fn contains_mbarrier_sm90_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        (statement.contains("mbarrier.") || statement.contains("llvm.nvvm.mbarrier"))
            && [
                "try_wait",
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
                ".relaxed",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

fn contains_mbarrier_ptx71_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("mbarrier.test_wait") && statement.contains(".parity"))
}

fn contains_mbarrier_ptx78_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && (statement.contains("try_wait") || statement.contains("shared::cta"))
    })
}

fn contains_mbarrier_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && [
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

/// Checks for Blackwell tcgen05 instructions (sm_100a+).
///
/// These instructions require a datacenter-Blackwell `a`/`f` target; consumer
/// sm_120 does not provide Tensor Memory:
/// - tcgen05: Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives)
///
/// Key differences from Hopper:
/// - tcgen05 MMA is single-thread (vs WGMMA's 128 threads)
/// - Uses Tensor Memory (TMEM) instead of registers
/// - Different synchronization model (mbarrier-based)
pub(super) fn contains_blackwell_features(contents: &str) -> bool {
    // Keep the instruction-family match broad enough for inline PTX and LLVM
    // intrinsic names, but do not treat debug filenames such as `tcgen05.rs`
    // as an instruction.
    [
        "tcgen05.alloc",
        "tcgen05.dealloc",
        "tcgen05.relinquish_alloc_permit",
        "tcgen05.fence",
        "tcgen05.commit",
        "tcgen05.mma",
        "tcgen05.cp",
        "tcgen05.shift",
        "tcgen05.ld",
        "tcgen05.st",
        "tcgen05.wait",
    ]
    .iter()
    .any(|instruction| contents.contains(instruction))
}

/// Checks for base TMA multicast in LLVM IR or inline PTX.
///
/// TMA multicast (`cp.async.bulk.tensor...multicast::cluster`) is an optional
/// qualifier that broadcasts a tile to all CTAs in a cluster. It is legal on
/// sm_90+, although NVIDIA advises an `a`/`f` target
/// for performance. In the LLVM intrinsic this is controlled by the trailing
/// `use_cta_mask` i1 argument being set to true.
pub(super) fn contains_tma_multicast(contents: &str) -> bool {
    contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i1 1, i1") || line.contains(", i1 true, i1"))
    }) || contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor") && statement.contains(".multicast::cluster")
    })
}

/// Checks Blackwell-only TMA forms with an explicit CTA-group qualifier.
pub(super) fn contains_tma_cta_group_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor")
            && (statement.contains(".cta_group::1") || statement.contains(".cta_group::2"))
    }) || contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i32 1)") || line.contains(", i32 2)"))
    })
}

/// Checks TMA copies whose destination is CTA-local shared memory.
///
/// `.shared::cta` already existed as a source state space for shared-to-global
/// copies, so the following `.global` source qualifier is part of the match.
/// The destination form was introduced in PTX 8.6 but is valid on sm_90.
pub(super) fn contains_tma_shared_cta_destination(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.") && statement.contains(".shared::cta.global")
    })
}

/// Checks PTX 8.6 TMA modifiers with a generic sm_100 architecture floor.
pub(super) fn contains_tma_sm100_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".cp_mask")
            || (contains_tma_gather_or_im2col(statement)
                && statement.contains(".shared::cta.global"))
    })
}

/// Checks PTX 8.6 TMA modes restricted to datacenter Blackwell targets.
pub(super) fn contains_tma_blackwell_accelerated_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".tile::scatter4")
            || statement.contains(".im2col::w::128")
            || (contains_tma_gather_or_im2col(statement)
                && !statement.contains(".shared::cta.global"))
    })
}

fn contains_tma_gather_or_im2col(statement: &str) -> bool {
    statement.contains(".tile::gather4")
        || (statement.contains(".im2col::w") && !statement.contains(".im2col::w::128"))
}

fn contains_tma_ptx86_features(contents: &str) -> bool {
    contains_tma_sm100_features(contents)
        || contains_tma_blackwell_accelerated_features(contents)
        || contents.contains(".sync_restrict")
        || contents
            .split(';')
            .any(|statement| statement.contains("mbarrier.") && statement.contains(".relaxed"))
}

fn contains_clc_features(contents: &str) -> bool {
    contents.contains("clusterlaunchcontrol.")
}

fn contains_clc_multicast_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("clusterlaunchcontrol.")
            && statement.contains(".multicast::cluster::all")
    })
}

fn contains_cluster_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("barrier.cluster.")
            && [".release", ".relaxed", ".acquire"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// Detect a direct LLVM intrinsic call, including its opaque-pointer overload.
///
/// Opaque-pointer LLVM IR spells the dynamic-stack intrinsics as
/// `llvm.stacksave.p0` and `llvm.stackrestore.p0`, while typed-pointer IR can
/// use the unsuffixed names. Keep the boundary strict so similarly named user
/// functions do not raise the module's target requirements. Only call
/// instructions count: declarations, comments, and quoted metadata or string
/// contents do not reach the backend.
fn contains_llvm_pointer_intrinsic_call(contents: &str, intrinsic: &str) -> bool {
    contents.lines().any(|line| {
        if !line.contains(intrinsic) {
            return false;
        }
        let code = llvm_code_without_comments_or_strings(line);
        code.match_indices(intrinsic).any(|(start, _)| {
            if !contains_llvm_keyword(&code[..start], "call") {
                return false;
            }

            let suffix = &code[start + intrinsic.len()..];
            if suffix.starts_with('(') {
                return true;
            }
            let Some(overload) = suffix.strip_prefix(".p") else {
                return false;
            };
            let digits = overload.bytes().take_while(u8::is_ascii_digit).count();
            digits != 0 && overload[digits..].starts_with('(')
        })
    })
}

fn llvm_code_without_comments_or_strings(line: &str) -> String {
    let mut code = String::with_capacity(line.len());
    let mut chars = line.chars();
    let mut in_string = false;

    while let Some(character) = chars.next() {
        if in_string {
            match character {
                '\\' => {
                    code.push(' ');
                    if chars.next().is_some() {
                        code.push(' ');
                    }
                }
                '"' => {
                    code.push(' ');
                    in_string = false;
                }
                _ => code.push(' '),
            }
        } else {
            match character {
                ';' => break,
                '"' => {
                    code.push(' ');
                    in_string = true;
                }
                _ => code.push(character),
            }
        }
    }

    code
}

fn contains_llvm_keyword(contents: &str, keyword: &str) -> bool {
    contents.match_indices(keyword).any(|(start, _)| {
        let before = contents[..start].bytes().next_back();
        let after = contents[start + keyword.len()..].bytes().next();
        before.is_none_or(|byte| !is_llvm_identifier_byte(byte))
            && after.is_none_or(|byte| !is_llvm_identifier_byte(byte))
    })
}

fn is_llvm_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'$' | b'.' | b'_')
}

fn contains_dynamic_stack_features(contents: &str) -> bool {
    contains_llvm_pointer_intrinsic_call(contents, "@llvm.stacksave")
        || contains_llvm_pointer_intrinsic_call(contents, "@llvm.stackrestore")
}

/// Detect every architecture requirement in exported LLVM text.
///
/// Both the ordinary PTX path and automatic libdevice mode use this exact
/// detector. The latter renders an in-memory preview before choosing the NVVM
/// pointer dialect.
pub fn detect_features_in_llvm_text(contents: &str) -> DetectedFeatures {
    let mut features = DetectedFeatures::empty();
    for (present, feature) in [
        (
            contains_blackwell_features(contents),
            DetectedFeatures::Blackwell,
        ),
        (
            contains_tma_cta_group_features(contents),
            DetectedFeatures::TmaCtaGroup,
        ),
        (
            contains_tma_blackwell_accelerated_features(contents),
            DetectedFeatures::BlackwellAccelerated,
        ),
        (
            contains_clc_multicast_features(contents),
            DetectedFeatures::BlackwellFamily,
        ),
        (
            contains_redux_f32_features(contents),
            DetectedFeatures::ReduxF32,
        ),
        (
            contains_multimem_blackwell_features(contents),
            DetectedFeatures::MultimemFp8,
        ),
        (
            contains_tma_multicast(contents),
            DetectedFeatures::TmaMulticast,
        ),
        (
            contains_blackwell_matrix_features(contents),
            DetectedFeatures::MatrixBlackwell,
        ),
        (contains_wgmma_features(contents), DetectedFeatures::Wgmma),
        (contains_tma_features(contents), DetectedFeatures::Tma),
        (
            contains_cluster_features(contents),
            DetectedFeatures::Cluster,
        ),
        (contains_sm90_features(contents), DetectedFeatures::Sm90),
        (contains_sm80_features(contents), DetectedFeatures::Sm80),
        (
            contains_mma_m8n8k16_int8_features(contents)
                || contains_mma_m8n8k32_int4_features(contents)
                || contains_mma_m8n8k128_b1_xor_features(contents),
            DetectedFeatures::Sm75,
        ),
        (
            contains_movmatrix_features(contents),
            DetectedFeatures::Movmatrix,
        ),
        (
            contains_ldmatrix_features(contents),
            DetectedFeatures::Ldmatrix,
        ),
        (
            contains_tma_sm100_features(contents)
                || contains_clc_features(contents)
                || contains_f32x2_features(contents),
            DetectedFeatures::Sm100,
        ),
        (
            contains_dynamic_stack_features(contents),
            DetectedFeatures::DynamicStack,
        ),
    ] {
        if present {
            features.insert(feature);
        }
    }
    if features == DetectedFeatures::empty() {
        features.insert(DetectedFeatures::Basic);
    }
    features
}

pub(super) fn detect_module_requirements_in_llvm_text(contents: &str) -> ModuleRequirements {
    let mut ptx_isa = PtxIsaRequirement::Default;
    if contains_packed_f16_atomic_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(62));
    }
    if contains_ldmatrix_features(contents)
        || contains_mma_m8n8k16_int8_features(contents)
        || contains_mma_m8n8k32_int4_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(65));
    }
    if contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
        || contains_mma_m16n8k16_f32_f16_features(contents)
        || contains_mma_m16n8k8_f32_tf32_features(contents)
        || contains_dense_int8_mma_features(contents)
        || contains_dense_int4_mma_features(contents)
        || contains_b1_xor_mma_features(contents)
        || contains_mma_m8n8k4_f64_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(70));
    }
    if contains_mbarrier_ptx71_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(71));
    }
    if contains_b1_and_mma_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(71));
    }
    if contains_dynamic_stack_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(73));
    }
    if contains_movmatrix_features(contents)
        || contains_stmatrix_features(contents)
        || contains_ldmatrix_cta_state_space(contents)
        || contains_cluster_features(contents)
        || contains_mbarrier_ptx78_features(contents)
        || contains_packed_bf16_atomic_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(78));
    }
    if contains_cp_async_bulk_features(contents)
        || contains_wgmma_features(contents)
        || contains_cluster_ptx80_features(contents)
        || contains_elect_features(contents)
        || contains_mbarrier_ptx80_features(contents)
        || contents.contains("fence.mbarrier_init")
        || contents.contains("fence.proxy.async")
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(80));
    }
    if contains_blackwell_matrix_features(contents)
        || contains_tma_cta_group_features(contents)
        || contains_tma_shared_cta_destination(contents)
        || contains_tma_ptx86_features(contents)
        || contains_clc_features(contents)
        || contains_blackwell_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
        || contains_redux_f32_features(contents)
        || contains_f32x2_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::new(86));
    }

    ModuleRequirements {
        features: detect_features_in_llvm_text(contents),
        ptx_isa,
    }
}

pub fn detect_module_requirements_in_llvm_file(
    ll_path: &Path,
) -> Result<ModuleRequirements, PipelineError> {
    let contents = std::fs::read_to_string(ll_path).map_err(|error| {
        PipelineError::PtxGeneration(format!(
            "failed to inspect generated LLVM IR {}: {error}",
            ll_path.display()
        ))
    })?;
    Ok(detect_module_requirements_in_llvm_text(&contents))
}
