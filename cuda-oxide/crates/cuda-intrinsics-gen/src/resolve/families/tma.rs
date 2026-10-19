/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, ImportedAddressSpace, ImportedIntrinsic, IntrinsicBackend,
    OverlayBackendLowering, OverlayIntrinsic, RuntimeValidation, Tma, TmaAdapter, TmaAdmission,
    TmaOperation, TmaReduction, TmaReductionAdmissionVariant, TmaReductionLoadMode,
    TmaReductionOperation,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use anyhow::{Context, Result, ensure};

use super::*;
use crate::resolve::abi_ledger::*;
use crate::resolve::guards::*;

pub(in crate::resolve) const TMA_BLACKWELL_TARGETS: &str = "sm_100a|sm_101a|sm_103a|sm_110a";
pub(in crate::resolve) const TENSOR_MAP_REPLACE_TARGETS: &str =
    "sm_100a|sm_100f|sm_103a|sm_103f|sm_110a|sm_110f|sm_120a|sm_120f|sm_121a|sm_121f|sm_90a";
pub(in crate::resolve) const TMA_OPERATIONS: [TmaOperation; 47] = [
    TmaOperation::G2sTile1d,
    TmaOperation::G2sTile2d,
    TmaOperation::G2sTile2dMulticast,
    TmaOperation::G2sTile2dMulticastCg2,
    TmaOperation::G2sTile3d,
    TmaOperation::G2sTile4d,
    TmaOperation::G2sTile5d,
    TmaOperation::S2gTile1d,
    TmaOperation::S2gTile2d,
    TmaOperation::S2gTile3d,
    TmaOperation::S2gTile4d,
    TmaOperation::S2gTile5d,
    TmaOperation::CommitGroup,
    TmaOperation::WaitGroup,
    TmaOperation::WaitGroupRead,
    TmaOperation::PrefetchTensorMap,
    TmaOperation::PrefetchTile1d,
    TmaOperation::PrefetchTile2d,
    TmaOperation::PrefetchTile3d,
    TmaOperation::PrefetchTile4d,
    TmaOperation::PrefetchTile5d,
    TmaOperation::PrefetchTileGather4TwoDimensional,
    TmaOperation::ReplaceBoxDim,
    TmaOperation::ReplaceElementStride,
    TmaOperation::ReplaceElementType,
    TmaOperation::ReplaceFillMode,
    TmaOperation::ReplaceGlobalAddress,
    TmaOperation::ReplaceGlobalDim,
    TmaOperation::ReplaceGlobalStride,
    TmaOperation::ReplaceInterleaveLayout,
    TmaOperation::ReplaceRank,
    TmaOperation::ReplaceSwizzleAtomicity,
    TmaOperation::ReplaceSwizzleMode,
    TmaOperation::FenceProxyTensorMapAcquireCluster,
    TmaOperation::FenceProxyTensorMapAcquireCta,
    TmaOperation::FenceProxyTensorMapAcquireGpu,
    TmaOperation::FenceProxyTensorMapAcquireSystem,
    TmaOperation::FenceProxyTensorMapReleaseCluster,
    TmaOperation::FenceProxyTensorMapReleaseCta,
    TmaOperation::FenceProxyTensorMapReleaseGpu,
    TmaOperation::FenceProxyTensorMapReleaseSystem,
    TmaOperation::PrefetchTile1dCacheHint,
    TmaOperation::PrefetchTile2dCacheHint,
    TmaOperation::PrefetchTile3dCacheHint,
    TmaOperation::PrefetchTile4dCacheHint,
    TmaOperation::PrefetchTile5dCacheHint,
    TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint,
];

pub(in crate::resolve) const TMA_REDUCTION_OPERATIONS: [TmaReductionOperation; 8] = [
    TmaReductionOperation::Add,
    TmaReductionOperation::And,
    TmaReductionOperation::Dec,
    TmaReductionOperation::Inc,
    TmaReductionOperation::Max,
    TmaReductionOperation::Min,
    TmaReductionOperation::Or,
    TmaReductionOperation::Xor,
];

pub(in crate::resolve) fn tma_reduction_matrix() -> Vec<TmaReduction> {
    let mut reductions = Vec::with_capacity(64);
    for operation in TMA_REDUCTION_OPERATIONS {
        for dimensions in 1..=5 {
            reductions.push(TmaReduction {
                operation,
                load_mode: TmaReductionLoadMode::Tile,
                dimensions,
            });
        }
        for dimensions in 3..=5 {
            reductions.push(TmaReduction {
                operation,
                load_mode: TmaReductionLoadMode::Im2col,
                dimensions,
            });
        }
    }
    reductions
}

#[cfg(test)]
pub(in crate::resolve) fn tma_reduction_admission_variants() -> Vec<TmaReductionAdmissionVariant> {
    tma_reduction_matrix()
        .into_iter()
        .enumerate()
        .map(|(index, reduction)| TmaReductionAdmissionVariant {
            // Preserve the repository's current ledger assignments in this test fixture.
            abi_id: format!("i{:04}", 923 + index),
            operation: reduction.operation,
            load_mode: reduction.load_mode,
            dimensions: reduction.dimensions,
        })
        .collect()
}

pub(in crate::resolve) fn tma_reduction_operation_name(
    operation: TmaReductionOperation,
) -> (&'static str, &'static str) {
    match operation {
        TmaReductionOperation::Add => ("add", "Add"),
        TmaReductionOperation::And => ("and", "And"),
        TmaReductionOperation::Dec => ("dec", "Dec"),
        TmaReductionOperation::Inc => ("inc", "Inc"),
        TmaReductionOperation::Max => ("max", "Max"),
        TmaReductionOperation::Min => ("min", "Min"),
        TmaReductionOperation::Or => ("or", "Or"),
        TmaReductionOperation::Xor => ("xor", "Xor"),
    }
}

pub(in crate::resolve) fn tma_reduction_load_mode_name(
    load_mode: TmaReductionLoadMode,
) -> (&'static str, &'static str, &'static str) {
    match load_mode {
        TmaReductionLoadMode::Tile => ("tile", "Tile", "tile"),
        TmaReductionLoadMode::Im2col => ("im2col", "Im2col", "im2col_no_offs"),
    }
}

pub(in crate::resolve) struct TmaReductionRecipe {
    id: String,
    operation_key: String,
    source_record: String,
    llvm_symbol: String,
    rust_arguments: Vec<&'static str>,
    dialect_op_type: String,
    dialect_op_name: String,
    dialect_operands: Vec<&'static str>,
    llvm_arguments: Vec<&'static str>,
    modifiers: Vec<String>,
    operands: Vec<OperandPattern>,
    summary: String,
}

pub(in crate::resolve) fn tma_reduction_recipe(
    reduction: TmaReduction,
) -> Result<TmaReductionRecipe> {
    ensure!(
        (1..=5).contains(&reduction.dimensions),
        "TMA reduction dimensionality must be in 1..=5"
    );
    ensure!(
        reduction.load_mode != TmaReductionLoadMode::Im2col || reduction.dimensions >= 3,
        "TMA reduction im2col mode requires at least three dimensions"
    );

    let (operation, operation_camel) = tma_reduction_operation_name(reduction.operation);
    let (load_mode, load_mode_camel, ptx_load_mode) =
        tma_reduction_load_mode_name(reduction.load_mode);
    let dimensions = reduction.dimensions as usize;
    let id = format!(
        "cp_async_bulk_tensor_reduce_{operation}_{load_mode}_{}d",
        reduction.dimensions
    );
    let mut rust_arguments = vec!["*const u8", "*const u8"];
    rust_arguments.extend(std::iter::repeat_n("i32", dimensions));
    let mut dialect_operands = vec!["ptr", "ptr"];
    dialect_operands.extend(std::iter::repeat_n("i32", dimensions));
    let mut llvm_arguments = vec!["shared_ptr", "ptr"];
    llvm_arguments.extend(std::iter::repeat_n("i32", dimensions));
    llvm_arguments.extend(["i64", "i1"]);

    Ok(TmaReductionRecipe {
        operation_key: format!(
            "memory.reduce.async.bulk.tensor.{operation}.{load_mode}.{}d",
            reduction.dimensions
        ),
        source_record: format!(
            "int_nvvm_cp_async_bulk_tensor_reduce_{operation}_{load_mode}_{}d",
            reduction.dimensions
        ),
        llvm_symbol: format!(
            "llvm.nvvm.cp.async.bulk.tensor.reduce.{operation}.{load_mode}.{}d",
            reduction.dimensions
        ),
        dialect_op_type: format!(
            "CpAsyncBulkTensorReduce{operation_camel}{load_mode_camel}{}dOp",
            reduction.dimensions
        ),
        dialect_op_name: format!(
            "nvvm.cp_async_bulk_tensor_reduce_{operation}_{load_mode}_{}d",
            reduction.dimensions
        ),
        rust_arguments,
        dialect_operands,
        llvm_arguments,
        modifiers: vec![
            "reduce".into(),
            "async".into(),
            "bulk".into(),
            "tensor".into(),
            format!("{}d", reduction.dimensions),
            "global".into(),
            "shared::cta".into(),
            operation.into(),
            ptx_load_mode.into(),
            "bulk_group".into(),
        ],
        operands: vec![OperandPattern::Address, OperandPattern::Address],
        summary: format!(
            "Starts a TMA tensor {operation} reduction from shared to global memory in {load_mode} mode."
        ),
        id,
    })
}

pub(in crate::resolve) fn expand_tma_reduction_variant(
    admission: &TmaAdmission,
    variant: &TmaReductionAdmissionVariant,
) -> Result<OverlayIntrinsic> {
    validate_abi_id(&variant.abi_id)?;
    let reduction = TmaReduction {
        operation: variant.operation,
        load_mode: variant.load_mode,
        dimensions: variant.dimensions,
    };
    let recipe = tma_reduction_recipe(reduction)?;

    Ok(OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: variant.abi_id.clone(),
        operation_key: recipe.operation_key.clone(),
        family: "tma".into(),
        source: None,
        source_record: Some(recipe.source_record.clone()),
        rust_module: "tma".into(),
        rust_name: recipe.id.clone(),
        rust_arguments: recipe
            .rust_arguments
            .iter()
            .map(|value| (*value).into())
            .collect(),
        rust_result: "()".into(),
        safe: false,
        must_use: false,
        safe_allowlist_reason: None,
        public_rust_path: format!("cuda_intrinsics::tma::{}", recipe.id),
        compatibility_rust_paths: vec![format!("cuda_device::tma::{}", recipe.id)],
        dialect_op_type: recipe.dialect_op_type.clone(),
        dialect_op_name: recipe.dialect_op_name.clone(),
        dialect_operands: recipe
            .dialect_operands
            .iter()
            .map(|value| (*value).into())
            .collect(),
        dialect_results: vec![],
        llvm_symbol: Some(recipe.llvm_symbol.clone()),
        resolved_llvm_symbol: None,
        llvm_arguments: recipe
            .llvm_arguments
            .iter()
            .map(|value| (*value).into())
            .collect(),
        llvm_results: vec![],
        pure: false,
        memory: "read_write".into(),
        convergent: true,
        execution_scope: "thread".into(),
        minimum_ptx: "8.0".into(),
        minimum_sm: Some("sm_90".into()),
        ptx_result: "()".into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section:
        "9.7.9.26.5.3 Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor"
            .into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor".into(),
        lowering: "generated_tma".into(),
        backend_lowerings: vec![
            OverlayBackendLowering {
                backend: IntrinsicBackend::LlvmNvptx,
                mechanism: BackendLoweringMechanism::TypedNvvm,
                evidence_profile: admission
                    .reduce_llvm_evidence_profile
                    .as_ref()
                    .expect("validated TMA reduction LLVM evidence profile")
                    .clone(),
                targets: None,
                minimum_ptx: Some("8.0".into()),
                minimum_sm: Some("sm_90".into()),
            },
            OverlayBackendLowering {
                backend: IntrinsicBackend::LibNvvm,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: admission
                    .reduce_libnvvm_evidence_profile
                    .as_ref()
                    .expect("validated TMA reduction libNVVM evidence profile")
                    .clone(),
                targets: None,
                minimum_ptx: Some("8.0".into()),
                minimum_sm: Some("sm_90".into()),
            },
        ],
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        integer_minmax: None,
        packed_conversion: None,
        scalar_conversion: None,
        scalar_arithmetic: None,
        scalar_math: None,
        extended_minmax: None,
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: None,
        prmt: None,
        cluster_barrier: None,
        wgmma_control: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        tma: Some(Tma {
            operation: TmaOperation::Reduce,
            reduction: Some(reduction),
            adapter: TmaAdapter::ReductionPointersCoordinatesInjectDefaults,
            runtime_validation: admission.runtime_validation,
        }),
        tcgen05: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "cp".into(),
            modifiers: recipe.modifiers,
            operands: recipe.operands,
        },
        summary: recipe.summary,
    })
}

pub(in crate::resolve) struct TmaRecipe {
    pub(in crate::resolve) operation: TmaOperation,
    pub(in crate::resolve) abi_id: &'static str,
    pub(in crate::resolve) id: &'static str,
    pub(in crate::resolve) operation_key: &'static str,
    pub(in crate::resolve) source_record: &'static str,
    pub(in crate::resolve) llvm_symbol: &'static str,
    pub(in crate::resolve) resolved_llvm_symbol: Option<String>,
    pub(in crate::resolve) selected_address_space: Option<ImportedAddressSpace>,
    pub(in crate::resolve) llvm_mechanism: BackendLoweringMechanism,
    pub(in crate::resolve) rust_arguments: Vec<&'static str>,
    pub(in crate::resolve) dialect_op_type: &'static str,
    pub(in crate::resolve) dialect_op_name: &'static str,
    pub(in crate::resolve) dialect_operands: Vec<&'static str>,
    pub(in crate::resolve) llvm_arguments: Vec<&'static str>,
    pub(in crate::resolve) adapter: TmaAdapter,
    pub(in crate::resolve) safe: bool,
    pub(in crate::resolve) safe_reason: Option<&'static str>,
    pub(in crate::resolve) convergent: bool,
    pub(in crate::resolve) minimum_ptx: &'static str,
    pub(in crate::resolve) minimum_sm: Option<&'static str>,
    pub(in crate::resolve) targets: &'static str,
    pub(in crate::resolve) memory: &'static str,
    pub(in crate::resolve) ptx_isa_section: &'static str,
    pub(in crate::resolve) ptx_isa_url: &'static str,
    pub(in crate::resolve) mnemonic: &'static str,
    pub(in crate::resolve) modifiers: Vec<String>,
    pub(in crate::resolve) operands: Vec<OperandPattern>,
    pub(in crate::resolve) summary: &'static str,
}

pub(in crate::resolve) fn tma_recipe(operation: TmaOperation) -> TmaRecipe {
    let (abi_id, id, operation_key, source_record, llvm_symbol, op_type, op_name) = match operation
    {
        TmaOperation::Reduce => unreachable!("TMA reductions use tma_reduction_recipe"),
        TmaOperation::G2sTile1d => (
            "i0328",
            "cp_async_bulk_tensor_1d_g2s",
            "memory.copy.async.bulk.tensor.g2s.tile.1d",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_1d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d",
            "CpAsyncBulkTensorG2sTile1dOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_1d",
        ),
        TmaOperation::G2sTile2d => (
            "i0329",
            "cp_async_bulk_tensor_2d_g2s",
            "memory.copy.async.bulk.tensor.g2s.tile.2d",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
            "CpAsyncBulkTensorG2sTile2dOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_2d",
        ),
        TmaOperation::G2sTile2dMulticast => (
            "i0330",
            "cp_async_bulk_tensor_2d_g2s_multicast",
            "memory.copy.async.bulk.tensor.g2s.tile.2d.multicast",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
            "CpAsyncBulkTensorG2sTile2dMulticastOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_2d_multicast",
        ),
        TmaOperation::G2sTile2dMulticastCg2 => (
            "i0331",
            "cp_async_bulk_tensor_2d_g2s_multicast_cg2",
            "memory.copy.async.bulk.tensor.g2s.tile.2d.multicast.cta_group_2",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
            "CpAsyncBulkTensorG2sTile2dMulticastCg2Op",
            "nvvm.cp_async_bulk_tensor_g2s_tile_2d_multicast_cg2",
        ),
        TmaOperation::G2sTile3d => (
            "i0332",
            "cp_async_bulk_tensor_3d_g2s",
            "memory.copy.async.bulk.tensor.g2s.tile.3d",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_3d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d",
            "CpAsyncBulkTensorG2sTile3dOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_3d",
        ),
        TmaOperation::G2sTile4d => (
            "i0333",
            "cp_async_bulk_tensor_4d_g2s",
            "memory.copy.async.bulk.tensor.g2s.tile.4d",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_4d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d",
            "CpAsyncBulkTensorG2sTile4dOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_4d",
        ),
        TmaOperation::G2sTile5d => (
            "i0334",
            "cp_async_bulk_tensor_5d_g2s",
            "memory.copy.async.bulk.tensor.g2s.tile.5d",
            "int_nvvm_cp_async_bulk_tensor_g2s_tile_5d",
            "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d",
            "CpAsyncBulkTensorG2sTile5dOp",
            "nvvm.cp_async_bulk_tensor_g2s_tile_5d",
        ),
        TmaOperation::S2gTile1d => (
            "i0335",
            "cp_async_bulk_tensor_1d_s2g",
            "memory.copy.async.bulk.tensor.s2g.tile.1d",
            "int_nvvm_cp_async_bulk_tensor_s2g_tile_1d",
            "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d",
            "CpAsyncBulkTensorS2gTile1dOp",
            "nvvm.cp_async_bulk_tensor_s2g_tile_1d",
        ),
        TmaOperation::S2gTile2d => (
            "i0336",
            "cp_async_bulk_tensor_2d_s2g",
            "memory.copy.async.bulk.tensor.s2g.tile.2d",
            "int_nvvm_cp_async_bulk_tensor_s2g_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d",
            "CpAsyncBulkTensorS2gTile2dOp",
            "nvvm.cp_async_bulk_tensor_s2g_tile_2d",
        ),
        TmaOperation::S2gTile3d => (
            "i0337",
            "cp_async_bulk_tensor_3d_s2g",
            "memory.copy.async.bulk.tensor.s2g.tile.3d",
            "int_nvvm_cp_async_bulk_tensor_s2g_tile_3d",
            "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d",
            "CpAsyncBulkTensorS2gTile3dOp",
            "nvvm.cp_async_bulk_tensor_s2g_tile_3d",
        ),
        TmaOperation::S2gTile4d => (
            "i0338",
            "cp_async_bulk_tensor_4d_s2g",
            "memory.copy.async.bulk.tensor.s2g.tile.4d",
            "int_nvvm_cp_async_bulk_tensor_s2g_tile_4d",
            "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d",
            "CpAsyncBulkTensorS2gTile4dOp",
            "nvvm.cp_async_bulk_tensor_s2g_tile_4d",
        ),
        TmaOperation::S2gTile5d => (
            "i0339",
            "cp_async_bulk_tensor_5d_s2g",
            "memory.copy.async.bulk.tensor.s2g.tile.5d",
            "int_nvvm_cp_async_bulk_tensor_s2g_tile_5d",
            "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d",
            "CpAsyncBulkTensorS2gTile5dOp",
            "nvvm.cp_async_bulk_tensor_s2g_tile_5d",
        ),
        TmaOperation::CommitGroup => (
            "i0340",
            "cp_async_bulk_commit_group",
            "memory.copy.async.bulk.group.commit",
            "int_nvvm_cp_async_bulk_commit_group",
            "llvm.nvvm.cp.async.bulk.commit.group",
            "CpAsyncBulkCommitGroupOp",
            "nvvm.cp_async_bulk_commit_group",
        ),
        TmaOperation::WaitGroup => (
            "i0341",
            "cp_async_bulk_wait_group",
            "memory.copy.async.bulk.group.wait_max_pending",
            "int_nvvm_cp_async_bulk_wait_group",
            "llvm.nvvm.cp.async.bulk.wait.group",
            "CpAsyncBulkWaitGroupOp",
            "nvvm.cp_async_bulk_wait_group",
        ),
        TmaOperation::WaitGroupRead => (
            "i0342",
            "cp_async_bulk_wait_group_read",
            "memory.copy.async.bulk.group.wait_read_max_pending",
            "int_nvvm_cp_async_bulk_wait_group_read",
            "llvm.nvvm.cp.async.bulk.wait.group.read",
            "CpAsyncBulkWaitGroupReadOp",
            "nvvm.cp_async_bulk_wait_group_read",
        ),
        TmaOperation::PrefetchTensorMap => (
            "i0887",
            "prefetch_tma_descriptor",
            "memory.prefetch.tensor_map",
            "int_nvvm_prefetch_tensormap",
            "llvm.nvvm.prefetch.tensormap",
            "PrefetchTensorMapOp",
            "nvvm.prefetch_tensormap",
        ),
        TmaOperation::PrefetchTile1d => (
            "i0888",
            "cp_async_bulk_prefetch_tensor_1d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.1d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_1d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d",
            "CpAsyncBulkPrefetchTensor1dL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_1d_l2",
        ),
        TmaOperation::PrefetchTile2d => (
            "i0889",
            "cp_async_bulk_prefetch_tensor_2d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.2d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d",
            "CpAsyncBulkPrefetchTensor2dL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_2d_l2",
        ),
        TmaOperation::PrefetchTile3d => (
            "i0890",
            "cp_async_bulk_prefetch_tensor_3d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.3d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_3d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d",
            "CpAsyncBulkPrefetchTensor3dL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_3d_l2",
        ),
        TmaOperation::PrefetchTile4d => (
            "i0891",
            "cp_async_bulk_prefetch_tensor_4d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.4d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_4d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d",
            "CpAsyncBulkPrefetchTensor4dL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_4d_l2",
        ),
        TmaOperation::PrefetchTile5d => (
            "i0892",
            "cp_async_bulk_prefetch_tensor_5d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.5d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_5d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d",
            "CpAsyncBulkPrefetchTensor5dL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_5d_l2",
        ),
        TmaOperation::PrefetchTileGather4TwoDimensional => (
            "i0893",
            "cp_async_bulk_prefetch_tensor_gather4_2d_l2",
            "memory.prefetch.async.bulk.tensor.global.tile.gather4.2d.l2",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_gather4_2d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.gather4.2d",
            "CpAsyncBulkPrefetchTensorGather4TwoDimensionalL2Op",
            "nvvm.cp_async_bulk_prefetch_tensor_gather4_2d_l2",
        ),
        TmaOperation::PrefetchTile1dCacheHint => (
            "i0917",
            "cp_async_bulk_prefetch_tensor_1d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.1d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_1d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d",
            "CpAsyncBulkPrefetchTensor1dL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_1d_l2_cache_hint",
        ),
        TmaOperation::PrefetchTile2dCacheHint => (
            "i0918",
            "cp_async_bulk_prefetch_tensor_2d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.2d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_2d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d",
            "CpAsyncBulkPrefetchTensor2dL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_2d_l2_cache_hint",
        ),
        TmaOperation::PrefetchTile3dCacheHint => (
            "i0919",
            "cp_async_bulk_prefetch_tensor_3d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.3d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_3d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d",
            "CpAsyncBulkPrefetchTensor3dL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_3d_l2_cache_hint",
        ),
        TmaOperation::PrefetchTile4dCacheHint => (
            "i0920",
            "cp_async_bulk_prefetch_tensor_4d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.4d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_4d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d",
            "CpAsyncBulkPrefetchTensor4dL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_4d_l2_cache_hint",
        ),
        TmaOperation::PrefetchTile5dCacheHint => (
            "i0921",
            "cp_async_bulk_prefetch_tensor_5d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.5d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_5d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d",
            "CpAsyncBulkPrefetchTensor5dL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_5d_l2_cache_hint",
        ),
        TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint => (
            "i0922",
            "cp_async_bulk_prefetch_tensor_gather4_2d_l2_cache_hint",
            "memory.prefetch.async.bulk.tensor.global.tile.gather4.2d.l2.cache_hint",
            "int_nvvm_cp_async_bulk_tensor_prefetch_tile_gather4_2d",
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.gather4.2d",
            "CpAsyncBulkPrefetchTensorGather4TwoDimensionalL2CacheHintOp",
            "nvvm.cp_async_bulk_prefetch_tensor_gather4_2d_l2_cache_hint",
        ),
        TmaOperation::ReplaceBoxDim => (
            "i0894",
            "tensormap_replace_box_dim",
            "memory.tensor_map.replace.box_dim",
            "int_nvvm_tensormap_replace_box_dim",
            "llvm.nvvm.tensormap.replace.box.dim",
            "ReplaceTensorMapBoxDimOp",
            "nvvm.tensormap_replace_box_dim",
        ),
        TmaOperation::ReplaceElementStride => (
            "i0895",
            "tensormap_replace_element_stride",
            "memory.tensor_map.replace.element_stride",
            "int_nvvm_tensormap_replace_element_stride",
            "llvm.nvvm.tensormap.replace.element.stride",
            "ReplaceTensorMapElementStrideOp",
            "nvvm.tensormap_replace_element_stride",
        ),
        TmaOperation::ReplaceElementType => (
            "i0896",
            "tensormap_replace_element_type",
            "memory.tensor_map.replace.element_type",
            "int_nvvm_tensormap_replace_elemtype",
            "llvm.nvvm.tensormap.replace.elemtype",
            "ReplaceTensorMapElementTypeOp",
            "nvvm.tensormap_replace_element_type",
        ),
        TmaOperation::ReplaceFillMode => (
            "i0897",
            "tensormap_replace_fill_mode",
            "memory.tensor_map.replace.fill_mode",
            "int_nvvm_tensormap_replace_fill_mode",
            "llvm.nvvm.tensormap.replace.fill.mode",
            "ReplaceTensorMapFillModeOp",
            "nvvm.tensormap_replace_fill_mode",
        ),
        TmaOperation::ReplaceGlobalAddress => (
            "i0898",
            "tensormap_replace_global_address",
            "memory.tensor_map.replace.global_address",
            "int_nvvm_tensormap_replace_global_address",
            "llvm.nvvm.tensormap.replace.global.address",
            "ReplaceTensorMapGlobalAddressOp",
            "nvvm.tensormap_replace_global_address",
        ),
        TmaOperation::ReplaceGlobalDim => (
            "i0899",
            "tensormap_replace_global_dim",
            "memory.tensor_map.replace.global_dim",
            "int_nvvm_tensormap_replace_global_dim",
            "llvm.nvvm.tensormap.replace.global.dim",
            "ReplaceTensorMapGlobalDimOp",
            "nvvm.tensormap_replace_global_dim",
        ),
        TmaOperation::ReplaceGlobalStride => (
            "i0900",
            "tensormap_replace_global_stride",
            "memory.tensor_map.replace.global_stride",
            "int_nvvm_tensormap_replace_global_stride",
            "llvm.nvvm.tensormap.replace.global.stride",
            "ReplaceTensorMapGlobalStrideOp",
            "nvvm.tensormap_replace_global_stride",
        ),
        TmaOperation::ReplaceInterleaveLayout => (
            "i0901",
            "tensormap_replace_interleave_layout",
            "memory.tensor_map.replace.interleave_layout",
            "int_nvvm_tensormap_replace_interleave_layout",
            "llvm.nvvm.tensormap.replace.interleave.layout",
            "ReplaceTensorMapInterleaveLayoutOp",
            "nvvm.tensormap_replace_interleave_layout",
        ),
        TmaOperation::ReplaceRank => (
            "i0902",
            "tensormap_replace_rank",
            "memory.tensor_map.replace.rank",
            "int_nvvm_tensormap_replace_rank",
            "llvm.nvvm.tensormap.replace.rank",
            "ReplaceTensorMapRankOp",
            "nvvm.tensormap_replace_rank",
        ),
        TmaOperation::ReplaceSwizzleAtomicity => (
            "i0903",
            "tensormap_replace_swizzle_atomicity",
            "memory.tensor_map.replace.swizzle_atomicity",
            "int_nvvm_tensormap_replace_swizzle_atomicity",
            "llvm.nvvm.tensormap.replace.swizzle.atomicity",
            "ReplaceTensorMapSwizzleAtomicityOp",
            "nvvm.tensormap_replace_swizzle_atomicity",
        ),
        TmaOperation::ReplaceSwizzleMode => (
            "i0904",
            "tensormap_replace_swizzle_mode",
            "memory.tensor_map.replace.swizzle_mode",
            "int_nvvm_tensormap_replace_swizzle_mode",
            "llvm.nvvm.tensormap.replace.swizzle.mode",
            "ReplaceTensorMapSwizzleModeOp",
            "nvvm.tensormap_replace_swizzle_mode",
        ),
        TmaOperation::FenceProxyTensorMapAcquireCluster => (
            "i0905",
            "fence_proxy_tensormap_generic_acquire_cluster",
            "memory.fence.proxy.tensor_map.generic.acquire.cluster",
            "int_nvvm_fence_proxy_tensormap_generic_acquire_cluster",
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster",
            "FenceProxyTensorMapGenericAcquireClusterOp",
            "nvvm.fence_proxy_tensormap_generic_acquire_cluster",
        ),
        TmaOperation::FenceProxyTensorMapAcquireCta => (
            "i0906",
            "fence_proxy_tensormap_generic_acquire_cta",
            "memory.fence.proxy.tensor_map.generic.acquire.cta",
            "int_nvvm_fence_proxy_tensormap_generic_acquire_cta",
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta",
            "FenceProxyTensorMapGenericAcquireCtaOp",
            "nvvm.fence_proxy_tensormap_generic_acquire_cta",
        ),
        TmaOperation::FenceProxyTensorMapAcquireGpu => (
            "i0907",
            "fence_proxy_tensormap_generic_acquire_gpu",
            "memory.fence.proxy.tensor_map.generic.acquire.gpu",
            "int_nvvm_fence_proxy_tensormap_generic_acquire_gpu",
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu",
            "FenceProxyTensorMapGenericAcquireGpuOp",
            "nvvm.fence_proxy_tensormap_generic_acquire_gpu",
        ),
        TmaOperation::FenceProxyTensorMapAcquireSystem => (
            "i0908",
            "fence_proxy_tensormap_generic_acquire_system",
            "memory.fence.proxy.tensor_map.generic.acquire.system",
            "int_nvvm_fence_proxy_tensormap_generic_acquire_sys",
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys",
            "FenceProxyTensorMapGenericAcquireSystemOp",
            "nvvm.fence_proxy_tensormap_generic_acquire_system",
        ),
        TmaOperation::FenceProxyTensorMapReleaseCluster => (
            "i0909",
            "fence_proxy_tensormap_generic_release_cluster",
            "memory.fence.proxy.tensor_map.generic.release.cluster",
            "int_nvvm_fence_proxy_tensormap_generic_release_cluster",
            "llvm.nvvm.fence.proxy.tensormap_generic.release.cluster",
            "FenceProxyTensorMapGenericReleaseClusterOp",
            "nvvm.fence_proxy_tensormap_generic_release_cluster",
        ),
        TmaOperation::FenceProxyTensorMapReleaseCta => (
            "i0910",
            "fence_proxy_tensormap_generic_release_cta",
            "memory.fence.proxy.tensor_map.generic.release.cta",
            "int_nvvm_fence_proxy_tensormap_generic_release_cta",
            "llvm.nvvm.fence.proxy.tensormap_generic.release.cta",
            "FenceProxyTensorMapGenericReleaseCtaOp",
            "nvvm.fence_proxy_tensormap_generic_release_cta",
        ),
        TmaOperation::FenceProxyTensorMapReleaseGpu => (
            "i0911",
            "fence_proxy_tensormap_generic_release_gpu",
            "memory.fence.proxy.tensor_map.generic.release.gpu",
            "int_nvvm_fence_proxy_tensormap_generic_release_gpu",
            "llvm.nvvm.fence.proxy.tensormap_generic.release.gpu",
            "FenceProxyTensorMapGenericReleaseGpuOp",
            "nvvm.fence_proxy_tensormap_generic_release_gpu",
        ),
        TmaOperation::FenceProxyTensorMapReleaseSystem => (
            "i0912",
            "fence_proxy_tensormap_generic_release_system",
            "memory.fence.proxy.tensor_map.generic.release.system",
            "int_nvvm_fence_proxy_tensormap_generic_release_sys",
            "llvm.nvvm.fence.proxy.tensormap_generic.release.sys",
            "FenceProxyTensorMapGenericReleaseSystemOp",
            "nvvm.fence_proxy_tensormap_generic_release_system",
        ),
    };

    let dimensions = operation.dimensions();
    let is_g2s = matches!(
        operation,
        TmaOperation::G2sTile1d
            | TmaOperation::G2sTile2d
            | TmaOperation::G2sTile2dMulticast
            | TmaOperation::G2sTile2dMulticastCg2
            | TmaOperation::G2sTile3d
            | TmaOperation::G2sTile4d
            | TmaOperation::G2sTile5d
    );
    let is_s2g = matches!(
        operation,
        TmaOperation::S2gTile1d
            | TmaOperation::S2gTile2d
            | TmaOperation::S2gTile3d
            | TmaOperation::S2gTile4d
            | TmaOperation::S2gTile5d
    );
    let multicast = matches!(
        operation,
        TmaOperation::G2sTile2dMulticast | TmaOperation::G2sTile2dMulticastCg2
    );
    let cg2 = operation == TmaOperation::G2sTile2dMulticastCg2;
    let prefetch_coordinates = operation.prefetch_coordinate_count();
    let is_release_fence = matches!(
        operation,
        TmaOperation::FenceProxyTensorMapReleaseCluster
            | TmaOperation::FenceProxyTensorMapReleaseCta
            | TmaOperation::FenceProxyTensorMapReleaseGpu
            | TmaOperation::FenceProxyTensorMapReleaseSystem
    );
    let is_acquire_fence = matches!(
        operation,
        TmaOperation::FenceProxyTensorMapAcquireCluster
            | TmaOperation::FenceProxyTensorMapAcquireCta
            | TmaOperation::FenceProxyTensorMapAcquireGpu
            | TmaOperation::FenceProxyTensorMapAcquireSystem
    );

    let mut rust_arguments = Vec::new();
    let mut dialect_operands = Vec::new();
    let mut llvm_arguments = Vec::new();
    let (adapter, safe, safe_reason, convergent, summary) = if is_g2s {
        rust_arguments.extend(["*mut u8", "*const u8"]);
        rust_arguments.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        rust_arguments.push("*mut u64");
        if multicast {
            rust_arguments.push("u16");
        }
        dialect_operands.extend(["ptr", "ptr", "ptr"]);
        dialect_operands.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        dialect_operands.extend(["i16", "i64"]);
        llvm_arguments.extend(["shared_cluster_ptr", "shared_ptr", "ptr"]);
        llvm_arguments.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        llvm_arguments.extend(["i16", "i64", "i1", "i1", "i32"]);
        (
            if multicast {
                TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults
            } else {
                TmaAdapter::G2sPointersCoordinatesBarrierInjectDefaults
            },
            false,
            None,
            true,
            if multicast {
                "Starts a multicast TMA tile copy from global to cluster shared memory."
            } else {
                "Starts a TMA tile copy from global to cluster shared memory."
            },
        )
    } else if is_s2g {
        rust_arguments.extend(["*const u8", "*const u8"]);
        rust_arguments.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        dialect_operands.extend(["ptr", "ptr"]);
        dialect_operands.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        llvm_arguments.extend(["shared_ptr", "ptr"]);
        llvm_arguments.extend(std::iter::repeat_n("i32", dimensions.unwrap()));
        llvm_arguments.extend(["i64", "i1"]);
        (
            TmaAdapter::S2gPointersCoordinatesInjectDefaults,
            false,
            None,
            true,
            "Starts a TMA tile copy from shared to global memory.",
        )
    } else if operation == TmaOperation::CommitGroup || is_release_fence {
        (
            TmaAdapter::NoOperands,
            true,
            Some(if operation == TmaOperation::CommitGroup {
                "committing this thread's pending bulk-copy group has no Rust memory-safety precondition."
            } else {
                "publishing prior tensor-map writes has no Rust memory-safety precondition."
            }),
            false,
            if operation == TmaOperation::CommitGroup {
                "Commits this thread's pending asynchronous bulk copies as one group."
            } else {
                "Publishes prior generic-proxy tensor-map writes to the selected tensor-map proxy scope."
            },
        )
    } else if operation == TmaOperation::PrefetchTensorMap {
        rust_arguments.push("*const u8");
        dialect_operands.push("ptr");
        llvm_arguments.push("anyptr");
        (
            TmaAdapter::DescriptorPointer,
            false,
            None,
            false,
            "Prefetches a live tensor-map descriptor into the tensor-map cache.",
        )
    } else if let Some(coordinate_count) = prefetch_coordinates {
        rust_arguments.push("*const u8");
        rust_arguments.extend(std::iter::repeat_n("i32", coordinate_count));
        dialect_operands.push("ptr");
        dialect_operands.extend(std::iter::repeat_n("i32", coordinate_count));
        if operation.uses_prefetch_cache_hint() {
            rust_arguments.push("u64");
            dialect_operands.push("i64");
        }
        llvm_arguments.push("ptr");
        llvm_arguments.extend(std::iter::repeat_n("i32", coordinate_count));
        llvm_arguments.extend(["i64", "i1"]);
        (
            if operation.uses_prefetch_cache_hint() {
                TmaAdapter::DescriptorCoordinatesCacheHintInjectFlag
            } else {
                TmaAdapter::DescriptorCoordinatesInjectDefaults
            },
            false,
            None,
            true,
            if operation.uses_prefetch_cache_hint() {
                "Prefetches one tensor tile through a tensor map into L2 using an explicit cache hint."
            } else {
                "Prefetches one tensor tile through a tensor map into L2."
            },
        )
    } else if operation == TmaOperation::ReplaceGlobalAddress {
        rust_arguments.extend(["*mut u8", "*const u8"]);
        dialect_operands.extend(["ptr", "ptr"]);
        llvm_arguments.extend(["anyptr", "i64"]);
        (
            TmaAdapter::DescriptorAndAddressPointers,
            false,
            None,
            false,
            "Replaces the global base address stored in a writable tensor-map descriptor.",
        )
    } else if matches!(
        operation,
        TmaOperation::ReplaceBoxDim
            | TmaOperation::ReplaceElementStride
            | TmaOperation::ReplaceGlobalDim
    ) {
        rust_arguments.extend(["*mut u8", "u32", "u32"]);
        dialect_operands.extend(["ptr", "i32", "i32"]);
        llvm_arguments.extend(["anyptr", "i32", "i32"]);
        (
            TmaAdapter::DescriptorOrdinalAndU32,
            false,
            None,
            false,
            "Replaces one indexed 32-bit field in a writable tensor-map descriptor.",
        )
    } else if operation == TmaOperation::ReplaceGlobalStride {
        rust_arguments.extend(["*mut u8", "u32", "u64"]);
        dialect_operands.extend(["ptr", "i32", "i64"]);
        llvm_arguments.extend(["anyptr", "i32", "i64"]);
        (
            TmaAdapter::DescriptorOrdinalAndU64,
            false,
            None,
            false,
            "Replaces one indexed 64-bit stride in a writable tensor-map descriptor.",
        )
    } else if matches!(
        operation,
        TmaOperation::ReplaceElementType
            | TmaOperation::ReplaceFillMode
            | TmaOperation::ReplaceInterleaveLayout
            | TmaOperation::ReplaceSwizzleAtomicity
            | TmaOperation::ReplaceSwizzleMode
    ) {
        rust_arguments.extend(["*mut u8", "u32"]);
        dialect_operands.extend(["ptr", "i32"]);
        llvm_arguments.extend(["anyptr", "i32"]);
        (
            TmaAdapter::DescriptorAndImmediateU32,
            false,
            None,
            false,
            "Replaces one immediate-valued field in a writable tensor-map descriptor.",
        )
    } else if operation == TmaOperation::ReplaceRank {
        rust_arguments.extend(["*mut u8", "u32"]);
        dialect_operands.extend(["ptr", "i32"]);
        llvm_arguments.extend(["anyptr", "i32"]);
        (
            TmaAdapter::DescriptorAndRuntimeU32,
            false,
            None,
            false,
            "Replaces the rank stored in a writable tensor-map descriptor.",
        )
    } else if is_acquire_fence {
        rust_arguments.push("*const u8");
        dialect_operands.push("ptr");
        llvm_arguments.extend(["ptr", "i32"]);
        (
            TmaAdapter::DescriptorPointerInjectBytes,
            false,
            None,
            false,
            "Acquires a published tensor-map descriptor into the selected tensor-map proxy scope.",
        )
    } else {
        rust_arguments.push("u32");
        dialect_operands.push("i32");
        llvm_arguments.push("i32");
        (
            TmaAdapter::CompileTimeConstantMaxPending,
            true,
            Some(
                "waiting on this thread's bulk-copy groups has no Rust memory-safety precondition.",
            ),
            false,
            if operation == TmaOperation::WaitGroupRead {
                "Waits for bulk-copy groups and completes their reads."
            } else {
                "Waits until at most the requested bulk-copy groups remain pending."
            },
        )
    };

    let is_replace = matches!(
        operation,
        TmaOperation::ReplaceBoxDim
            | TmaOperation::ReplaceElementStride
            | TmaOperation::ReplaceElementType
            | TmaOperation::ReplaceFillMode
            | TmaOperation::ReplaceGlobalAddress
            | TmaOperation::ReplaceGlobalDim
            | TmaOperation::ReplaceGlobalStride
            | TmaOperation::ReplaceInterleaveLayout
            | TmaOperation::ReplaceRank
            | TmaOperation::ReplaceSwizzleAtomicity
            | TmaOperation::ReplaceSwizzleMode
    );
    let is_fence = is_acquire_fence || is_release_fence;
    let descriptor_control = operation == TmaOperation::PrefetchTensorMap
        || prefetch_coordinates.is_some()
        || is_replace
        || is_fence;
    let blackwell_tma = cg2
        || matches!(
            operation,
            TmaOperation::PrefetchTileGather4TwoDimensional
                | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint
        );

    let (mnemonic, modifiers, operands) = if is_g2s {
        let mut modifiers = vec![
            "async".into(),
            "bulk".into(),
            "tensor".into(),
            format!("{}d", dimensions.unwrap()),
            "shared::cluster".into(),
            "global".into(),
            "tile".into(),
            "mbarrier::complete_tx::bytes".into(),
        ];
        if multicast {
            modifiers.push("multicast::cluster".into());
        }
        if cg2 {
            modifiers.push("cta_group::2".into());
        }
        let mut operands = vec![
            OperandPattern::Address,
            OperandPattern::Address,
            OperandPattern::Address,
        ];
        if multicast {
            operands.push(OperandPattern::Register);
        }
        ("cp", modifiers, operands)
    } else if is_s2g {
        (
            "cp",
            vec![
                "async".into(),
                "bulk".into(),
                "tensor".into(),
                format!("{}d", dimensions.unwrap()),
                "global".into(),
                "shared::cta".into(),
                "tile".into(),
                "bulk_group".into(),
            ],
            vec![OperandPattern::Address, OperandPattern::Address],
        )
    } else if operation == TmaOperation::CommitGroup {
        (
            "cp",
            vec!["async".into(), "bulk".into(), "commit_group".into()],
            vec![],
        )
    } else if matches!(
        operation,
        TmaOperation::WaitGroup | TmaOperation::WaitGroupRead
    ) {
        let mut modifiers = vec!["async".into(), "bulk".into(), "wait_group".into()];
        if operation == TmaOperation::WaitGroupRead {
            modifiers.push("read".into());
        }
        ("cp", modifiers, vec![OperandPattern::Immediate])
    } else if operation == TmaOperation::PrefetchTensorMap {
        (
            "prefetch",
            vec!["tensormap".into()],
            vec![OperandPattern::Address],
        )
    } else if prefetch_coordinates.is_some() {
        let (dimensionality, tile) = if matches!(
            operation,
            TmaOperation::PrefetchTileGather4TwoDimensional
                | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint
        ) {
            ("2d", "tile::gather4")
        } else {
            (
                match prefetch_coordinates.unwrap() {
                    1 => "1d",
                    2 => "2d",
                    3 => "3d",
                    4 => "4d",
                    5 => "5d",
                    _ => unreachable!("closed TMA prefetch dimensionality"),
                },
                "tile",
            )
        };
        let mut modifiers = vec![
            "async".into(),
            "bulk".into(),
            "prefetch".into(),
            "tensor".into(),
            dimensionality.into(),
            "L2".into(),
            "global".into(),
            tile.into(),
        ];
        let mut operands = vec![OperandPattern::Address];
        if operation.uses_prefetch_cache_hint() {
            modifiers.push("L2::cache_hint".into());
            operands.push(OperandPattern::Register);
        }
        ("cp", modifiers, operands)
    } else if is_replace {
        let (field, width, operands) = match operation {
            TmaOperation::ReplaceGlobalAddress => (
                "global_address",
                "b64",
                vec![OperandPattern::Address, OperandPattern::Register],
            ),
            TmaOperation::ReplaceRank => (
                "rank",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Register],
            ),
            TmaOperation::ReplaceBoxDim => (
                "box_dim",
                "b32",
                vec![
                    OperandPattern::Address,
                    OperandPattern::Immediate,
                    OperandPattern::Register,
                ],
            ),
            TmaOperation::ReplaceElementStride => (
                "element_stride",
                "b32",
                vec![
                    OperandPattern::Address,
                    OperandPattern::Immediate,
                    OperandPattern::Register,
                ],
            ),
            TmaOperation::ReplaceGlobalDim => (
                "global_dim",
                "b32",
                vec![
                    OperandPattern::Address,
                    OperandPattern::Immediate,
                    OperandPattern::Register,
                ],
            ),
            TmaOperation::ReplaceGlobalStride => (
                "global_stride",
                "b64",
                vec![
                    OperandPattern::Address,
                    OperandPattern::Immediate,
                    OperandPattern::Register,
                ],
            ),
            TmaOperation::ReplaceElementType => (
                "elemtype",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Immediate],
            ),
            TmaOperation::ReplaceFillMode => (
                "fill_mode",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Immediate],
            ),
            TmaOperation::ReplaceInterleaveLayout => (
                "interleave_layout",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Immediate],
            ),
            TmaOperation::ReplaceSwizzleAtomicity => (
                "swizzle_atomicity",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Immediate],
            ),
            TmaOperation::ReplaceSwizzleMode => (
                "swizzle_mode",
                "b32",
                vec![OperandPattern::Address, OperandPattern::Immediate],
            ),
            _ => unreachable!("TMA tensor-map replace operation was matched"),
        };
        (
            "tensormap",
            vec![
                "replace".into(),
                "tile".into(),
                field.into(),
                "global".into(),
                "b1024".into(),
                width.into(),
            ],
            operands,
        )
    } else if is_fence {
        let (semantics, scope) = match operation {
            TmaOperation::FenceProxyTensorMapAcquireCluster => ("acquire", "cluster"),
            TmaOperation::FenceProxyTensorMapAcquireCta => ("acquire", "cta"),
            TmaOperation::FenceProxyTensorMapAcquireGpu => ("acquire", "gpu"),
            TmaOperation::FenceProxyTensorMapAcquireSystem => ("acquire", "sys"),
            TmaOperation::FenceProxyTensorMapReleaseCluster => ("release", "cluster"),
            TmaOperation::FenceProxyTensorMapReleaseCta => ("release", "cta"),
            TmaOperation::FenceProxyTensorMapReleaseGpu => ("release", "gpu"),
            TmaOperation::FenceProxyTensorMapReleaseSystem => ("release", "sys"),
            _ => unreachable!("TMA tensor-map fence operation was matched"),
        };
        (
            "fence",
            vec![
                "proxy".into(),
                "tensormap::generic".into(),
                semantics.into(),
                scope.into(),
            ],
            if is_acquire_fence {
                vec![
                    OperandPattern::Address,
                    OperandPattern::Exact {
                        value: "128".into(),
                    },
                ]
            } else {
                vec![]
            },
        )
    } else {
        unreachable!("TMA operation category was matched")
    };

    let minimum_ptx = if blackwell_tma || operation == TmaOperation::ReplaceSwizzleAtomicity {
        "8.6"
    } else if is_replace || is_fence {
        "8.3"
    } else {
        "8.0"
    };
    let (minimum_sm, targets) = if blackwell_tma {
        (None, TMA_BLACKWELL_TARGETS)
    } else if operation == TmaOperation::ReplaceSwizzleAtomicity {
        (None, BLACKWELL_LDMATRIX_LLVM_TARGETS)
    } else if is_replace {
        (None, TENSOR_MAP_REPLACE_TARGETS)
    } else {
        (Some("sm_90"), "all")
    };
    let memory = if operation == TmaOperation::PrefetchTensorMap || prefetch_coordinates.is_some() {
        "read"
    } else if is_replace {
        "write"
    } else {
        "read_write"
    };
    let polymorphic_descriptor = operation == TmaOperation::PrefetchTensorMap;

    TmaRecipe {
        operation,
        abi_id,
        id,
        operation_key,
        source_record,
        llvm_symbol,
        resolved_llvm_symbol: polymorphic_descriptor.then(|| format!("{llvm_symbol}.p0")),
        selected_address_space: polymorphic_descriptor.then_some(ImportedAddressSpace::Generic),
        llvm_mechanism: if is_replace {
            BackendLoweringMechanism::InlinePtx
        } else {
            BackendLoweringMechanism::TypedNvvm
        },
        rust_arguments,
        dialect_op_type: op_type,
        dialect_op_name: op_name,
        dialect_operands,
        llvm_arguments,
        adapter,
        safe,
        safe_reason,
        convergent,
        minimum_ptx,
        minimum_sm,
        targets,
        memory,
        ptx_isa_section: if descriptor_control {
            "Tensor-map descriptor and asynchronous bulk tensor operations"
        } else {
            "9.7.9.26.5 Asynchronous bulk tensor copy"
        },
        ptx_isa_url: if descriptor_control {
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-map"
        } else {
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor"
        },
        mnemonic,
        modifiers,
        operands,
        summary,
    }
}

pub(in crate::resolve) fn expand_tma_admission(
    admission: &TmaAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "TMA runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact TMA admission requires both backend evidence profiles"
    );
    ensure!(
        admission
            .variants
            .iter()
            .map(|variant| variant.operation)
            .eq(TMA_OPERATIONS),
        "compact TMA admission must list all {} operations in canonical order",
        TMA_OPERATIONS.len()
    );

    let mut records = admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = tma_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "tma".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "tma".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: "()".into(),
                safe: recipe.safe,
                must_use: false,
                safe_allowlist_reason: recipe.safe_reason.map(Into::into),
                public_rust_path: format!("cuda_intrinsics::tma::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::tma::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: recipe.resolved_llvm_symbol.clone(),
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: vec![],
                pure: false,
                memory: recipe.memory.into(),
                convergent: recipe.convergent,
                execution_scope: "thread".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: "()".into(),
                targets: recipe.targets.into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: recipe.ptx_isa_url.into(),
                lowering: "generated_tma".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: recipe.llvm_mechanism,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        targets: None,
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                integer_minmax: None,
                packed_conversion: None,
                scalar_conversion: None,
                scalar_arithmetic: None,
                scalar_math: None,
                extended_minmax: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                wgmma_control: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                tma: Some(Tma {
                    operation: recipe.operation,
                    reduction: None,
                    adapter: recipe.adapter,
                    runtime_validation: admission.runtime_validation,
                }),
                tcgen05: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: recipe.selected_address_space,
                expected_ptx: InstructionPattern {
                    mnemonic: recipe.mnemonic.into(),
                    modifiers: recipe.modifiers,
                    operands: recipe.operands,
                },
                summary: recipe.summary.into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if !admission.reduce_variants.is_empty() {
        ensure!(
            admission.reduce_llvm_evidence_profile.is_some(),
            "compact TMA reduction admission requires reduce_llvm_evidence_profile"
        );
        ensure!(
            admission.reduce_libnvvm_evidence_profile.is_some(),
            "compact TMA reduction admission requires reduce_libnvvm_evidence_profile"
        );

        let expected_reductions = tma_reduction_matrix();
        ensure!(
            admission.reduce_variants.len() == expected_reductions.len()
                && admission
                    .reduce_variants
                    .iter()
                    .zip(expected_reductions.iter())
                    .all(|(variant, expected)| {
                        variant.operation == expected.operation
                            && variant.load_mode == expected.load_mode
                            && variant.dimensions == expected.dimensions
                    }),
            "compact TMA reduction admission must list all 64 operations in canonical order"
        );
        for variant in &admission.reduce_variants {
            records.push(expand_tma_reduction_variant(admission, variant)?);
        }
    }
    Ok(records)
}

pub(in crate::resolve) fn validate_tma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let tma = policy
        .tma
        .as_ref()
        .with_context(|| format!("{} has no closed TMA contract", policy.id))?;
    if tma.operation == TmaOperation::Reduce {
        return validate_tma_reduction_policy(policy, declaration, tma);
    }
    ensure!(
        tma.reduction.is_none(),
        "{} non-reduction TMA operation carries a reduction contract",
        policy.id
    );
    let recipe = tma_recipe(tma.operation);
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol == recipe.resolved_llvm_symbol
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} TMA identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tma"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == "()"
            && policy.safe == recipe.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.as_deref() == recipe.safe_reason
            && policy.public_rust_path == format!("cuda_intrinsics::tma::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::tma::{}", recipe.id)],
        "{} TMA Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results.is_empty()
            && declaration.arguments == recipe.llvm_arguments
            && declaration.results.is_empty()
            && policy.selected_address_space == recipe.selected_address_space
            && policy.lowering == "generated_tma",
        "{} TMA carrier or LLVM adapter changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent == recipe.convergent
            && policy.execution_scope == "thread"
            && tma.adapter == recipe.adapter
            && tma.runtime_validation == RuntimeValidation::Unexecuted,
        "{} TMA semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == recipe.targets
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url
            && policy.ptx_result == "()"
            && policy.expected_ptx.mnemonic == recipe.mnemonic
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == recipe.operands,
        "{} TMA target or PTX contract changed",
        policy.id
    );
    let valid_route = |backend, mechanism| {
        policy.backend_lowerings.iter().any(|route| {
            route.backend == backend
                && route.mechanism == mechanism
                && route.minimum_ptx.as_deref() == Some(recipe.minimum_ptx)
                && route.minimum_sm.as_deref() == recipe.minimum_sm
                && !route.evidence_profile.trim().is_empty()
        })
    };
    ensure!(
        policy.backend_lowerings.len() == 2
            && valid_route(IntrinsicBackend::LlvmNvptx, recipe.llvm_mechanism,)
            && valid_route(
                IntrinsicBackend::LibNvvm,
                BackendLoweringMechanism::InlinePtx,
            ),
        "{} TMA backend route changed",
        policy.id
    );
    let expected_properties = tma_imported_properties(tma.operation);
    ensure!(
        declaration.properties == expected_properties,
        "{} imported TMA declaration changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "TMA")?;
    Ok(())
}

pub(in crate::resolve) fn validate_tma_reduction_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    tma: &Tma,
) -> Result<()> {
    ensure!(
        tma.operation == TmaOperation::Reduce,
        "{} is not a TMA reduction",
        policy.id
    );
    let reduction = tma
        .reduction
        .with_context(|| format!("{} has no TMA reduction contract", policy.id))?;
    let recipe = tma_reduction_recipe(reduction)?;
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} TMA reduction identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "tma"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::tma::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::tma::{}", recipe.id)],
        "{} TMA reduction Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results.is_empty()
            && declaration.arguments == recipe.llvm_arguments
            && declaration.results.is_empty()
            && declaration.selections.is_empty()
            && policy.lowering == "generated_tma",
        "{} TMA reduction carrier or LLVM adapter changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "thread"
            && tma.adapter == TmaAdapter::ReductionPointersCoordinatesInjectDefaults
            && tma.runtime_validation == RuntimeValidation::Unexecuted,
        "{} TMA reduction semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.0"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.9.26.5.3 Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor"
            && policy.ptx_result == "()"
            && policy.expected_ptx.mnemonic == "cp"
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == recipe.operands,
        "{} TMA reduction target or PTX contract changed",
        policy.id
    );
    let valid_route = |backend, mechanism| {
        policy.backend_lowerings.iter().any(|route| {
            route.backend == backend
                && route.mechanism == mechanism
                && route.minimum_ptx.as_deref() == Some("8.0")
                && route.minimum_sm.as_deref() == Some("sm_90")
                && !route.evidence_profile.trim().is_empty()
        })
    };
    ensure!(
        policy.backend_lowerings.len() == 2
            && valid_route(
                IntrinsicBackend::LlvmNvptx,
                BackendLoweringMechanism::TypedNvvm,
            )
            && valid_route(
                IntrinsicBackend::LibNvvm,
                BackendLoweringMechanism::InlinePtx,
            ),
        "{} TMA reduction backend route changed",
        policy.id
    );
    ensure!(
        declaration.properties == tma_reduction_imported_properties(reduction.dimensions as usize),
        "{} imported TMA reduction declaration changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "TMA")?;
    Ok(())
}

pub(in crate::resolve) fn tma_imported_properties(operation: TmaOperation) -> Vec<String> {
    assert_ne!(operation, TmaOperation::Reduce);
    let dimensions = operation.dimensions();
    if matches!(
        operation,
        TmaOperation::G2sTile1d
            | TmaOperation::G2sTile2d
            | TmaOperation::G2sTile2dMulticast
            | TmaOperation::G2sTile2dMulticastCg2
            | TmaOperation::G2sTile3d
            | TmaOperation::G2sTile4d
            | TmaOperation::G2sTile5d
    ) {
        let dimensions = dimensions.unwrap();
        let mut properties = vec![
            format!("ImmArg<arg{}>", dimensions + 5),
            format!("ImmArg<arg{}>", dimensions + 6),
            format!("ImmArg<arg{}>", dimensions + 7),
            "IntrConvergent".into(),
            format!("Range<arg{},0,3>", dimensions + 7),
            "ReadOnly<arg2>".into(),
            "WriteOnly<arg0>".into(),
        ];
        properties.sort();
        return properties;
    }
    if matches!(
        operation,
        TmaOperation::S2gTile1d
            | TmaOperation::S2gTile2d
            | TmaOperation::S2gTile3d
            | TmaOperation::S2gTile4d
            | TmaOperation::S2gTile5d
    ) {
        return vec![
            format!("ImmArg<arg{}>", dimensions.unwrap() + 3),
            "IntrConvergent".into(),
            "ReadOnly<arg0>".into(),
            "ReadOnly<arg1>".into(),
        ];
    }
    if operation == TmaOperation::CommitGroup {
        return vec![];
    }
    if matches!(
        operation,
        TmaOperation::WaitGroup | TmaOperation::WaitGroupRead
    ) {
        return vec!["ImmArg<arg0>".into()];
    }
    if operation == TmaOperation::PrefetchTensorMap {
        return vec![
            "IntrArgMemOnly".into(),
            "NoCapture<arg0>".into(),
            "ReadOnly<arg0>".into(),
        ];
    }
    if let Some(coordinate_count) = operation.prefetch_coordinate_count() {
        return vec![
            format!("ImmArg<arg{}>", coordinate_count + 2),
            "IntrConvergent".into(),
            "ReadOnly<arg0>".into(),
        ];
    }
    if matches!(
        operation,
        TmaOperation::FenceProxyTensorMapAcquireCluster
            | TmaOperation::FenceProxyTensorMapAcquireCta
            | TmaOperation::FenceProxyTensorMapAcquireGpu
            | TmaOperation::FenceProxyTensorMapAcquireSystem
    ) {
        return vec![
            "ImmArg<arg1>".into(),
            "IntrArgMemOnly".into(),
            "IntrNoCallback".into(),
            "Range<arg1,128,129>".into(),
        ];
    }
    if matches!(
        operation,
        TmaOperation::FenceProxyTensorMapReleaseCluster
            | TmaOperation::FenceProxyTensorMapReleaseCta
            | TmaOperation::FenceProxyTensorMapReleaseGpu
            | TmaOperation::FenceProxyTensorMapReleaseSystem
    ) {
        return vec!["IntrNoCallback".into()];
    }

    let mut properties = vec![
        "IntrArgMemOnly".into(),
        "IntrWriteMem".into(),
        "NoCapture<arg0>".into(),
    ];
    match operation {
        TmaOperation::ReplaceBoxDim
        | TmaOperation::ReplaceElementStride
        | TmaOperation::ReplaceGlobalDim
        | TmaOperation::ReplaceGlobalStride => {
            // LLVM 23 added the Range<arg1,0,5> ordinal bound these
            // tensormap.replace declarations always implied.
            properties.extend(["ImmArg<arg1>".into(), "Range<arg1,0,5>".into()]);
        }
        TmaOperation::ReplaceElementType => {
            properties.extend([
                "ArgInfo<arg1>".into(),
                "ImmArg<arg1>".into(),
                "Range<arg1,0,16>".into(),
            ]);
        }
        TmaOperation::ReplaceFillMode => {
            properties.extend([
                "ArgInfo<arg1>".into(),
                "ImmArg<arg1>".into(),
                "Range<arg1,0,2>".into(),
            ]);
        }
        TmaOperation::ReplaceInterleaveLayout => {
            properties.extend([
                "ArgInfo<arg1>".into(),
                "ImmArg<arg1>".into(),
                "Range<arg1,0,3>".into(),
            ]);
        }
        TmaOperation::ReplaceSwizzleAtomicity => {
            properties.extend([
                "ArgInfo<arg1>".into(),
                "ImmArg<arg1>".into(),
                "Range<arg1,0,4>".into(),
            ]);
        }
        TmaOperation::ReplaceSwizzleMode => {
            properties.extend([
                "ArgInfo<arg1>".into(),
                "ImmArg<arg1>".into(),
                "Range<arg1,0,5>".into(),
            ]);
        }
        TmaOperation::ReplaceGlobalAddress | TmaOperation::ReplaceRank => {}
        _ => unreachable!("closed TMA operation property contract"),
    }
    properties.sort();
    properties
}

pub(in crate::resolve) fn tma_reduction_imported_properties(dimensions: usize) -> Vec<String> {
    vec![
        format!("ImmArg<arg{}>", dimensions + 3),
        "IntrConvergent".into(),
        "ReadOnly<arg0>".into(),
        "ReadOnly<arg1>".into(),
    ]
}
