/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{CatalogFile, CatalogIntrinsic};
use std::collections::BTreeSet;

mod misc;
mod mma;
mod packed;
mod scalar;
mod tcgen05;

pub(super) use misc::*;
pub(super) use mma::*;
pub(super) use packed::*;
pub(super) use scalar::*;
pub(super) use tcgen05::*;

pub(super) fn sregs(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "sreg")
}

pub(super) fn ldmatrix(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "ldmatrix")
}

pub(super) fn stmatrices(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "stmatrix")
}

pub(super) fn register_mmas(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "register_mma")
}

pub(super) fn sparse_mmas(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "sparse_mma")
}

pub(super) fn packed_atomics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_atomic")
}

pub(super) fn redux(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "redux")
}

pub(super) fn dot_products(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "dotprod")
}

pub(super) fn packed_alus(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_alu")
}

pub(super) fn packed_conversions(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_conversion")
}

pub(super) fn scalar_conversions(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "scalar_conversion")
}

pub(super) fn scalar_arithmetics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "scalar_arithmetic")
}

pub(super) fn extended_minmax(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "extended_minmax")
}

pub(super) fn scalar_maths(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "scalar_math")
}

pub(super) fn prmts(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "prmt")
}

pub(super) fn cluster_barriers(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cluster_barrier")
}

pub(super) fn debug_controls(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "debug_control")
}

pub(super) fn cluster_memory(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cluster_memory")
}

pub(super) fn clc_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "clc")
}

pub(super) fn tma_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "tma")
}

pub(super) fn execution_controls(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog.intrinsics.iter().filter(|record| {
        matches!(
            record.family.as_str(),
            "counted_barrier" | "grid_dependency" | "register_control"
        )
    })
}

pub(super) fn wgmma_controls(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "wgmma_control")
}

pub(super) fn tcgen05_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "tcgen05")
}

pub(super) fn tcgen05_mma_intrinsics(
    catalog: &CatalogFile,
) -> impl Iterator<Item = &CatalogIntrinsic> {
    tcgen05_intrinsics(catalog).filter(|record| {
        record
            .tcgen05
            .as_ref()
            .is_some_and(|tcgen05| tcgen05.mma.is_some())
    })
}

pub(super) fn tcgen05_non_mma_intrinsics(
    catalog: &CatalogFile,
) -> impl Iterator<Item = &CatalogIntrinsic> {
    tcgen05_intrinsics(catalog).filter(|record| {
        record
            .tcgen05
            .as_ref()
            .is_some_and(|tcgen05| tcgen05.mma.is_none())
    })
}

pub(super) fn cp_async_copies(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cp_async_copy")
}

pub(super) fn cp_async_controls(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cp_async_control")
}

pub(super) fn cp_async_mbarriers(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cp_async_mbarrier")
}

pub(super) fn mbarrier_basics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "mbarrier_basic")
}

pub(super) fn movmatrix(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "movmatrix")
}

pub(super) fn mbarrier_extended(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "mbarrier_extended")
}

pub(super) fn sync_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "sync")
}

pub(super) fn vote_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "vote")
}

pub(super) fn active_masks(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "active_mask")
}

pub(super) fn elect_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "elect")
}

pub(super) fn warp_matches(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_match")
}

pub(super) fn warp_barriers(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_barrier")
}

pub(super) fn warp_shuffles(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_shuffle")
}

pub(super) fn integer_minmaxes(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "integer_minmax")
}

/// Everything a sharded generated output may import from `dialect_nvvm::ops`:
/// each record's op type, the compatibility op types, and the attribute types
/// the renderers spell out inline. Shard emitters filter this list against the
/// shard body so each generated file imports exactly what it uses.
pub(super) fn dialect_nvvm_ops_import_candidates(catalog: &CatalogFile) -> Vec<String> {
    let mut candidates: BTreeSet<String> = [
        "BreakpointOp",
        "ClusterBarrierModeAttr",
        "ClusterBarrierOp",
        "ClusterSyncOp",
        "ExtendedMinMaxFormatAttr",
        "ExtendedMinMaxNanAttr",
        "ExtendedMinMaxOp",
        "ExtendedMinMaxOperationAttr",
        "ExtendedMinMaxSubnormalAttr",
        "ExtendedMinMaxXorSignAbsAttr",
        "LdmatrixElementAttr",
        "LdmatrixLayoutAttr",
        "LdmatrixMultiplicityAttr",
        "LdmatrixOp",
        "LdmatrixShapeAttr",
        "LdmatrixStateSpaceAttr",
        "NvvmAtomAddBf16x2Op",
        "NvvmAtomAddF16x2Op",
        "PackedAtomicAddOp",
        "PackedAtomicAtomicityAttr",
        "PackedAtomicFormatAttr",
        "PackedAtomicOrderingAttr",
        "PackedAtomicRoundingAttr",
        "PackedAtomicScopeAttr",
        "PackedAtomicStateSpaceAttr",
        "PackedAtomicSubnormalAttr",
        "PmEventOp",
        "PrmtModeAttr",
        "PrmtOp",
        "RegisterMmaAccumulatorAttr",
        "RegisterMmaElementAttr",
        "RegisterMmaKindAttr",
        "RegisterMmaLayoutAttr",
        "RegisterMmaOp",
        "RegisterMmaOperationAttr",
        "RegisterMmaOverflowAttr",
        "RegisterMmaShapeAttr",
        "ScalarArithmeticFormatAttr",
        "ScalarArithmeticOp",
        "ScalarArithmeticOperationAttr",
        "ScalarArithmeticRoundingAttr",
        "ScalarArithmeticSaturationAttr",
        "ScalarArithmeticSubnormalAttr",
        "ScalarConversionOp",
        "ScalarConversionRoundingAttr",
        "ScalarConversionSaturationAttr",
        "ScalarMathFormatAttr",
        "ScalarMathOp",
        "ScalarMathOperationAttr",
        "ScalarMathPrecisionAttr",
        "ScalarMathSubnormalAttr",
        "SparseMmaAccumulatorAttr",
        "SparseMmaElementAttr",
        "SparseMmaLayoutAttr",
        "SparseMmaMetadataAttr",
        "SparseMmaOp",
        "SparseMmaOverflowAttr",
        "SparseMmaSelectorAttr",
        "SparseMmaShapeAttr",
        "Tcgen05MmaBBufferAttr",
        "Tcgen05MmaBUsageAttr",
        "Tcgen05MmaCollectorAAttr",
        "Tcgen05MmaCtaGroupAttr",
        "Tcgen05MmaFormAttr",
        "Tcgen05MmaKindAttr",
        "Tcgen05MmaOp",
        "TrapOp",
        "WgmmaCommitGroupSyncAlignedOp",
        "WgmmaFenceSyncAlignedOp",
        "WgmmaWaitGroupSyncAlignedOp",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    for record in &catalog.intrinsics {
        candidates.insert(record.dialect.op_type.clone());
    }
    for record in ldmatrix(catalog) {
        if let Some((op_type, _)) = ldmatrix_compat_op(record) {
            candidates.insert(op_type.to_owned());
        }
    }
    for record in register_mmas(catalog) {
        if let Some(op_type) = register_mma_compat_op_type(record) {
            candidates.insert(op_type.to_owned());
        }
    }
    candidates.into_iter().collect()
}
