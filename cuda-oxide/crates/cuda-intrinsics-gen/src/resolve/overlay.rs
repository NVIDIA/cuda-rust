/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    OverlayFile, OverlayIntrinsic, OverlayShardFile, RegisterMmaAccumulator, Tcgen05Operation,
    TmaOperation,
};
use crate::util::sha256_bytes;
use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path};

use super::abi_ledger::*;
use super::families::*;
use super::guards::*;

pub(super) const OVERLAY_SCHEMA: u32 = 44;
pub(super) const MINIMUM_OVERLAY_SHARD_SCHEMA: u32 = 26;
pub(super) const OVERLAY_SHARD_SCHEMA: u32 = 64;
pub(super) const REGISTER_MMA_F8F6F4_SHARD_SCHEMA: u32 = 46;
pub(super) const REGISTER_MMA_F8F6F4_F16_SHARD_SCHEMA: u32 = 47;
pub(super) const REGISTER_MMA_MXF8F6F4_SHARD_SCHEMA: u32 = 60;
pub(super) const REGISTER_MMA_FP8_SHARD_SCHEMA: u32 = 48;
pub(super) const REGISTER_MMA_AMPERE_FLOAT_SHARD_SCHEMA: u32 = 49;
pub(super) const SPARSE_MMA_F8F6F4_SHARD_SCHEMA: u32 = 27;
pub(super) const SPARSE_MMA_F8F6F4_F16_SHARD_SCHEMA: u32 = 50;
pub(super) const SPARSE_MMA_AMPERE_FLOAT_SHARD_SCHEMA: u32 = 63;
pub(super) const SPARSE_MMA_FP8_F32_SHARD_SCHEMA: u32 = 64;
pub(super) const PRMT_SHARD_SCHEMA: u32 = 28;
pub(super) const PACKED_CONVERSION_FP8_SHARD_SCHEMA: u32 = 29;
pub(super) const PACKED_CONVERSION_FP8_F16X2_SHARD_SCHEMA: u32 = 59;
pub(super) const CLUSTER_SREG_SHARD_SCHEMA: u32 = 30;
pub(super) const CLUSTER_BARRIER_SHARD_SCHEMA: u32 = 31;
pub(super) const SPECIAL_REGISTER_SHARD_SCHEMA: u32 = 32;
pub(super) const DEBUG_CONTROL_SHARD_SCHEMA: u32 = 33;
pub(super) const THREADFENCE_SHARD_SCHEMA: u32 = 34;
pub(super) const STMATRIX_SHARD_SCHEMA: u32 = 35;
pub(super) const CLUSTER_MEMORY_SHARD_SCHEMA: u32 = 39;
pub(super) const CLC_SHARD_SCHEMA: u32 = 40;
pub(super) const TMA_SHARD_SCHEMA: u32 = 61;
pub(super) const TMA_REDUCTION_SHARD_SCHEMA: u32 = 62;
pub(super) const MBARRIER_EXTENDED_SHARD_SCHEMA: u32 = 40;
pub(super) const WGMMA_CONTROL_SHARD_SCHEMA: u32 = 38;
pub(super) const TCGEN05_SHARD_SCHEMA: u32 = 42;
pub(super) const SCALAR_CONVERSION_SHARD_SCHEMA: u32 = 43;
pub(super) const SCALAR_ARITHMETIC_SHARD_SCHEMA: u32 = 45;
pub(super) const EXTENDED_MINMAX_SHARD_SCHEMA: u32 = 60;
pub(super) const TCGEN05_CP_SHARD_SCHEMA: u32 = 52;
pub(super) const TCGEN05_LD_SHARD_SCHEMA: u32 = 53;
pub(super) const TCGEN05_ST_SHARD_SCHEMA: u32 = 54;
pub(super) const TCGEN05_OFFSET_LDST_SHARD_SCHEMA: u32 = 55;
pub(super) const TCGEN05_CONTROL_SHARD_SCHEMA: u32 = 56;
pub(super) const TCGEN05_MMA_SHARD_SCHEMA: u32 = 57;
pub(super) const SCALAR_MATH_SHARD_SCHEMA: u32 = 58;
pub(crate) const CATALOG_SCHEMA: u32 = 46;
pub(super) fn read_overlay(
    repo_root: &Path,
    manifest_path: &Path,
) -> Result<(OverlayFile, String)> {
    let manifest_bytes =
        fs::read(manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let mut overlay: OverlayFile = toml::from_slice(&manifest_bytes)
        .with_context(|| format!("parse {}", manifest_path.display()))?;

    ensure!(
        overlay.intrinsics.is_empty(),
        "overlay.toml must list family shards instead of inline intrinsic records"
    );
    ensure!(
        !overlay.shards.is_empty(),
        "overlay.toml must list at least one family shard"
    );

    let mut previous = None;
    let mut seen = BTreeSet::new();
    let mut hash_input = Vec::new();
    append_overlay_hash_input(&mut hash_input, "intrinsics/overlay.toml", &manifest_bytes);

    for shard_name in &overlay.shards {
        validate_overlay_shard_path(shard_name)?;
        ensure!(
            seen.insert(shard_name.as_str()),
            "overlay.toml lists duplicate shard {shard_name}"
        );
        if let Some(previous) = previous {
            ensure!(
                previous < shard_name.as_str(),
                "overlay.toml shards must be sorted"
            );
        }
        previous = Some(shard_name.as_str());

        let relative = Path::new("intrinsics").join(shard_name);
        let path = repo_root.join(&relative);
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let mut shard: OverlayShardFile =
            toml::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        validate_overlay_shard_schema(&shard, &path)?;
        let int4_mma_admission = shard.register_mma_int4.take();
        let int8_mma_admission = shard.register_mma_int8.take();
        let binary_mma_admission = shard.register_mma_b1.take();
        let f8f6f4_f32_mma_admission = shard.register_mma_f8f6f4_f32.take();
        let f8f6f4_f16_mma_admission = shard.register_mma_f8f6f4_f16.take();
        let mxf8f6f4_f32_mma_admission = shard.register_mma_mxf8f6f4_f32.take();
        let fp8_mma_admission = shard.register_mma_fp8.take();
        let ampere_float_mma_admission = shard.register_mma_ampere_float.take();
        let sparse_mma_admission = shard.sparse_mma_integer.take();
        let sparse_mma_f8f6f4_admission = shard.sparse_mma_f8f6f4_f32.take();
        let sparse_mma_f8f6f4_f16_admission = shard.sparse_mma_f8f6f4_f16.take();
        let sparse_mma_fp8_f32_admission = shard.sparse_mma_fp8_f32.take();
        let sparse_mma_ampere_float_admission = shard.sparse_mma_ordered_ampere_float.take();
        let prmt_admission = shard.prmt.take();
        let packed_conversion_fp8_admission = shard.packed_conversion_fp8.take();
        let packed_conversion_fp8_f16x2_admission = shard.packed_conversion_fp8_f16x2.take();
        let scalar_conversion_admission = shard.scalar_conversion.take();
        let scalar_arithmetic_admission = shard.scalar_arithmetic.take();
        let scalar_math_admission = shard.scalar_math.take();
        let extended_minmax_admission = shard.extended_minmax.take();
        let cluster_sreg_admission = shard.cluster_sreg.take();
        let cluster_barrier_admission = shard.cluster_barrier.take();
        let mbarrier_extended_admission = shard.mbarrier_extended.take();
        let special_register_admission = shard.special_registers.take();
        let debug_control_admission = shard.debug_control.take();
        let threadfence_admission = shard.threadfence.take();
        let cluster_memory_admission = shard.cluster_memory.take();
        let stmatrix_admission = shard.stmatrix.take();
        let clc_admission = shard.clc.take();
        let wgmma_control_admission = shard.wgmma_controls.take();
        let tma_admission = shard.tma.take();
        let tcgen05_admission = shard.tcgen05.take();
        let compact_mma_count = usize::from(int4_mma_admission.is_some())
            + usize::from(int8_mma_admission.is_some())
            + usize::from(binary_mma_admission.is_some())
            + usize::from(f8f6f4_f32_mma_admission.is_some())
            + usize::from(f8f6f4_f16_mma_admission.is_some())
            + usize::from(mxf8f6f4_f32_mma_admission.is_some())
            + usize::from(fp8_mma_admission.is_some())
            + usize::from(ampere_float_mma_admission.is_some())
            + usize::from(sparse_mma_admission.is_some())
            + usize::from(sparse_mma_f8f6f4_admission.is_some())
            + usize::from(sparse_mma_f8f6f4_f16_admission.is_some())
            + usize::from(sparse_mma_fp8_f32_admission.is_some())
            + usize::from(sparse_mma_ampere_float_admission.is_some());
        ensure!(
            compact_mma_count <= 1,
            "overlay shard {} contains more than one compact MMA admission",
            path.display()
        );
        let integer_mma_admission = int4_mma_admission
            .map(|admission| (RegisterMmaIntegerKind::Int4, admission))
            .or_else(|| {
                int8_mma_admission.map(|admission| (RegisterMmaIntegerKind::Int8, admission))
            });
        if let Some((kind, admission)) = integer_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact integer MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_integer_admission(kind, &admission)?;
        }
        if let Some(admission) = binary_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact binary MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_binary_admission(&admission)?;
        }
        if let Some(admission) = f8f6f4_f32_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact dense f8f6f4 MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics =
                expand_register_mma_f8f6f4_admission(&admission, RegisterMmaAccumulator::F32)?;
        }
        if let Some(admission) = f8f6f4_f16_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact dense f8f6f4 MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics =
                expand_register_mma_f8f6f4_admission(&admission, RegisterMmaAccumulator::F16)?;
        }
        if let Some(admission) = mxf8f6f4_f32_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact dense mxf8f6f4 MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_mxf8f6f4_admission(&admission)?;
        }
        if let Some(admission) = fp8_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact standard FP8 MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_fp8_admission(&admission)?;
        }
        if let Some(admission) = ampere_float_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact Ampere floating-point MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_ampere_float_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_integer_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_f8f6f4_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse f8f6f4 MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_f8f6f4_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_f8f6f4_f16_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse f8f6f4 F16 MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_f8f6f4_f16_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_fp8_f32_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse FP8 MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_fp8_f32_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_ampere_float_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact ordered Ampere floating sparse MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_ordered_ampere_float_admission(&admission)?;
        }
        if let Some(admission) = prmt_admission {
            ensure!(
                shard.family == "prmt" && shard.intrinsics.is_empty(),
                "compact prmt admission must be the only content of a prmt shard"
            );
            shard.intrinsics = expand_prmt_admission(&admission)?;
        }
        if let Some(admission) = packed_conversion_fp8_admission {
            ensure!(
                shard.family == "packed_conversion" && shard.intrinsics.is_empty(),
                "compact FP8 conversion admission must be the only content of a packed_conversion shard"
            );
            shard.intrinsics = expand_packed_conversion_fp8_admission(&admission)?;
        }
        if let Some(admission) = packed_conversion_fp8_f16x2_admission {
            ensure!(
                shard.family == "packed_conversion" && shard.intrinsics.is_empty(),
                "compact FP8 f16x2 conversion admission must be the only content of a packed_conversion shard"
            );
            shard.intrinsics = expand_packed_conversion_fp8_f16x2_admission(&admission)?;
        }
        if let Some(admission) = scalar_conversion_admission {
            ensure!(
                shard.family == "scalar_conversion" && shard.intrinsics.is_empty(),
                "compact scalar-conversion admission must be the only content of its shard"
            );
            shard.intrinsics = expand_scalar_conversion_admission(&admission)?;
        }
        if let Some(admission) = scalar_arithmetic_admission {
            ensure!(
                shard.family == "scalar_arithmetic" && shard.intrinsics.is_empty(),
                "compact scalar-arithmetic admission must be the only content of its shard"
            );
            shard.intrinsics = expand_scalar_arithmetic_admission(&admission)?;
        }
        if let Some(admission) = scalar_math_admission {
            ensure!(
                shard.family == "scalar_math" && shard.intrinsics.is_empty(),
                "compact scalar-math admission must be the only content of its shard"
            );
            shard.intrinsics = expand_scalar_math_admission(&admission)?;
        }
        if let Some(admission) = extended_minmax_admission {
            ensure!(
                shard.family == "extended_minmax" && shard.intrinsics.is_empty(),
                "compact extended-minmax admission must be the only content of its shard"
            );
            shard.intrinsics = expand_extended_minmax_admission(&admission)?;
        }
        if let Some(admission) = cluster_sreg_admission {
            ensure!(
                shard.family == "sreg" && shard.intrinsics.is_empty() && compact_mma_count == 0,
                "compact cluster-sreg admission must be the only content of an sreg shard"
            );
            shard.intrinsics = expand_cluster_sreg_admission(&admission)?;
        }
        if let Some(admission) = cluster_barrier_admission {
            ensure!(
                shard.family == "cluster_barrier" && shard.intrinsics.is_empty(),
                "compact cluster-barrier admission must be the only content of its shard"
            );
            shard.intrinsics = expand_cluster_barrier_admission(&admission)?;
        }
        if let Some(admission) = mbarrier_extended_admission {
            ensure!(
                shard.family == "mbarrier_extended" && shard.intrinsics.is_empty(),
                "compact extended-mbarrier admission must be the only content of its shard"
            );
            shard.intrinsics = expand_mbarrier_extended_admission(&admission)?;
        }
        if let Some(admission) = special_register_admission {
            ensure!(
                shard.family == "sreg" && shard.intrinsics.is_empty(),
                "compact special-register admission must be the only content of an sreg shard"
            );
            shard.intrinsics = expand_special_register_admission(&admission)?;
        }
        if let Some(admission) = debug_control_admission {
            ensure!(
                shard.family == "debug_control" && shard.intrinsics.is_empty(),
                "compact debug-control admission must be the only content of a debug_control shard"
            );
            shard.intrinsics = expand_debug_control_admission(&admission)?;
        }
        if let Some(admission) = threadfence_admission {
            ensure!(
                shard.family == "sync" && shard.intrinsics.is_empty(),
                "compact threadfence admission must be the only content of a sync shard"
            );
            shard.intrinsics = expand_threadfence_admission(&admission)?;
        }
        if let Some(admission) = cluster_memory_admission {
            ensure!(
                shard.family == "cluster_memory" && shard.intrinsics.is_empty(),
                "compact cluster-memory admission must be the only content of its shard"
            );
            shard.intrinsics = expand_cluster_memory_admission(&admission)?;
        }
        if let Some(admission) = stmatrix_admission {
            ensure!(
                shard.family == "stmatrix" && shard.intrinsics.is_empty(),
                "compact stmatrix admission must be the only content of its shard"
            );
            shard.intrinsics = expand_stmatrix_admission(&admission)?;
        }
        if let Some(admission) = clc_admission {
            ensure!(
                shard.family == "clc" && shard.intrinsics.is_empty(),
                "compact CLC admission must be the only content of a clc shard"
            );
            shard.intrinsics = expand_clc_admission(&admission)?;
        }
        if let Some(admission) = wgmma_control_admission {
            ensure!(
                shard.family == "wgmma_control" && shard.intrinsics.is_empty(),
                "compact WGMMA-control admission must be the only content of its shard"
            );
            shard.intrinsics = expand_wgmma_control_admission(&admission)?;
        }
        if let Some(admission) = tma_admission {
            ensure!(
                shard.family == "tma" && shard.intrinsics.is_empty(),
                "compact TMA admission must be the only content of a tma shard"
            );
            shard.intrinsics = expand_tma_admission(&admission)?;
        }
        if let Some(admission) = tcgen05_admission {
            ensure!(
                shard.family == "tcgen05" && shard.intrinsics.is_empty(),
                "compact tcgen05 admission must be the only content of a tcgen05 shard"
            );
            shard.intrinsics = expand_tcgen05_admission(&admission)?;
        }
        ensure!(
            !shard.intrinsics.is_empty(),
            "overlay shard {} contains no intrinsic records",
            path.display()
        );
        for record in &shard.intrinsics {
            ensure!(
                record.family == shard.family,
                "overlay shard {} declares family {}, but intrinsic {} uses family {}",
                path.display(),
                shard.family,
                record.id,
                record.family
            );
        }

        append_overlay_hash_input(
            &mut hash_input,
            relative
                .to_str()
                .context("overlay shard path is not valid UTF-8")?,
            &bytes,
        );
        overlay.intrinsics.extend(shard.intrinsics);
    }

    Ok((overlay, sha256_bytes(&hash_input)))
}

pub(super) fn validate_overlay_shard_schema(shard: &OverlayShardFile, path: &Path) -> Result<()> {
    validate_overlay_shard_schema_with_max(shard, path, OVERLAY_SHARD_SCHEMA)
}

pub(super) fn validate_overlay_shard_schema_with_max(
    shard: &OverlayShardFile,
    path: &Path,
    maximum_schema: u32,
) -> Result<()> {
    ensure!(
        (MINIMUM_OVERLAY_SHARD_SCHEMA..=maximum_schema).contains(&shard.schema),
        "unsupported overlay shard schema {} in {}",
        shard.schema,
        path.display()
    );
    ensure!(
        shard.sparse_mma_f8f6f4_f32.is_none() || shard.schema >= SPARSE_MMA_F8F6F4_SHARD_SCHEMA,
        "compact sparse f8f6f4 MMA admission requires overlay shard schema {}",
        SPARSE_MMA_F8F6F4_SHARD_SCHEMA
    );
    ensure!(
        shard.sparse_mma_f8f6f4_f16.is_none() || shard.schema >= SPARSE_MMA_F8F6F4_F16_SHARD_SCHEMA,
        "compact sparse f8f6f4 F16 MMA admission requires overlay shard schema {}",
        SPARSE_MMA_F8F6F4_F16_SHARD_SCHEMA
    );
    ensure!(
        shard.sparse_mma_ordered_ampere_float.is_none()
            || shard.schema >= SPARSE_MMA_AMPERE_FLOAT_SHARD_SCHEMA,
        "compact ordered Ampere floating sparse MMA admission requires overlay shard schema {}",
        SPARSE_MMA_AMPERE_FLOAT_SHARD_SCHEMA
    );
    ensure!(
        shard.sparse_mma_fp8_f32.is_none() || shard.schema >= SPARSE_MMA_FP8_F32_SHARD_SCHEMA,
        "compact sparse FP8 MMA admission requires overlay shard schema {}",
        SPARSE_MMA_FP8_F32_SHARD_SCHEMA
    );
    ensure!(
        shard.register_mma_f8f6f4_f32.is_none() || shard.schema >= REGISTER_MMA_F8F6F4_SHARD_SCHEMA,
        "compact dense f8f6f4 MMA admission requires overlay shard schema {}",
        REGISTER_MMA_F8F6F4_SHARD_SCHEMA
    );
    ensure!(
        shard.register_mma_f8f6f4_f16.is_none()
            || shard.schema >= REGISTER_MMA_F8F6F4_F16_SHARD_SCHEMA,
        "compact dense f8f6f4 F16 MMA admission requires overlay shard schema {}",
        REGISTER_MMA_F8F6F4_F16_SHARD_SCHEMA
    );
    ensure!(
        shard.register_mma_mxf8f6f4_f32.is_none()
            || shard.schema >= REGISTER_MMA_MXF8F6F4_SHARD_SCHEMA,
        "compact dense mxf8f6f4 MMA admission requires overlay shard schema {}",
        REGISTER_MMA_MXF8F6F4_SHARD_SCHEMA
    );
    ensure!(
        shard.register_mma_fp8.is_none() || shard.schema >= REGISTER_MMA_FP8_SHARD_SCHEMA,
        "compact standard FP8 MMA admission requires overlay shard schema {}",
        REGISTER_MMA_FP8_SHARD_SCHEMA
    );
    ensure!(
        shard.register_mma_ampere_float.is_none()
            || shard.schema >= REGISTER_MMA_AMPERE_FLOAT_SHARD_SCHEMA,
        "compact Ampere floating-point MMA admission requires overlay shard schema {}",
        REGISTER_MMA_AMPERE_FLOAT_SHARD_SCHEMA
    );
    ensure!(
        shard.prmt.is_none() || shard.schema >= PRMT_SHARD_SCHEMA,
        "compact prmt admission requires overlay shard schema {}",
        PRMT_SHARD_SCHEMA
    );
    ensure!(
        shard.packed_conversion_fp8.is_none() || shard.schema >= PACKED_CONVERSION_FP8_SHARD_SCHEMA,
        "compact FP8 conversion admission requires overlay shard schema {}",
        PACKED_CONVERSION_FP8_SHARD_SCHEMA
    );
    ensure!(
        shard.packed_conversion_fp8_f16x2.is_none()
            || shard.schema >= PACKED_CONVERSION_FP8_F16X2_SHARD_SCHEMA,
        "compact FP8 f16x2 conversion admission requires overlay shard schema {}",
        PACKED_CONVERSION_FP8_F16X2_SHARD_SCHEMA
    );
    ensure!(
        shard.scalar_conversion.is_none() || shard.schema >= SCALAR_CONVERSION_SHARD_SCHEMA,
        "compact scalar-conversion admission requires overlay shard schema {}",
        SCALAR_CONVERSION_SHARD_SCHEMA
    );
    ensure!(
        shard.scalar_arithmetic.is_none() || shard.schema >= SCALAR_ARITHMETIC_SHARD_SCHEMA,
        "compact scalar-arithmetic admission requires overlay shard schema {}",
        SCALAR_ARITHMETIC_SHARD_SCHEMA
    );
    ensure!(
        shard.scalar_math.is_none() || shard.schema >= SCALAR_MATH_SHARD_SCHEMA,
        "compact scalar-math admission requires overlay shard schema {}",
        SCALAR_MATH_SHARD_SCHEMA
    );
    ensure!(
        shard.extended_minmax.is_none() || shard.schema >= EXTENDED_MINMAX_SHARD_SCHEMA,
        "compact extended-minmax admission requires overlay shard schema {}",
        EXTENDED_MINMAX_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_sreg.is_none() || shard.schema >= CLUSTER_SREG_SHARD_SCHEMA,
        "compact cluster-sreg admission requires overlay shard schema {}",
        CLUSTER_SREG_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_barrier.is_none() || shard.schema >= CLUSTER_BARRIER_SHARD_SCHEMA,
        "compact cluster-barrier admission requires overlay shard schema {}",
        CLUSTER_BARRIER_SHARD_SCHEMA
    );
    ensure!(
        shard.special_registers.is_none() || shard.schema >= SPECIAL_REGISTER_SHARD_SCHEMA,
        "compact special-register admission requires overlay shard schema {}",
        SPECIAL_REGISTER_SHARD_SCHEMA
    );
    ensure!(
        shard.debug_control.is_none() || shard.schema >= DEBUG_CONTROL_SHARD_SCHEMA,
        "compact debug-control admission requires overlay shard schema {}",
        DEBUG_CONTROL_SHARD_SCHEMA
    );
    ensure!(
        shard.threadfence.is_none() || shard.schema >= THREADFENCE_SHARD_SCHEMA,
        "compact threadfence admission requires overlay shard schema {}",
        THREADFENCE_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_memory.is_none() || shard.schema >= CLUSTER_MEMORY_SHARD_SCHEMA,
        "compact cluster-memory admission requires overlay shard schema {}",
        CLUSTER_MEMORY_SHARD_SCHEMA
    );
    ensure!(
        shard.stmatrix.is_none() || shard.schema >= STMATRIX_SHARD_SCHEMA,
        "compact stmatrix admission requires overlay shard schema {}",
        STMATRIX_SHARD_SCHEMA
    );
    ensure!(
        shard.clc.is_none() || shard.schema >= CLC_SHARD_SCHEMA,
        "compact CLC admission requires overlay shard schema {}",
        CLC_SHARD_SCHEMA
    );
    ensure!(
        shard.tma.is_none() || shard.schema >= TMA_SHARD_SCHEMA,
        "compact TMA admission requires overlay shard schema {}",
        TMA_SHARD_SCHEMA
    );
    ensure!(
        shard
            .tma
            .as_ref()
            .is_none_or(|admission| admission.reduce_variants.is_empty())
            || shard.schema >= TMA_REDUCTION_SHARD_SCHEMA,
        "compact TMA reduction admission requires overlay shard schema {}",
        TMA_REDUCTION_SHARD_SCHEMA
    );
    ensure!(
        shard.mbarrier_extended.is_none() || shard.schema >= MBARRIER_EXTENDED_SHARD_SCHEMA,
        "compact extended-mbarrier admission requires overlay shard schema {}",
        MBARRIER_EXTENDED_SHARD_SCHEMA
    );
    ensure!(
        shard.wgmma_controls.is_none() || shard.schema >= WGMMA_CONTROL_SHARD_SCHEMA,
        "compact WGMMA-control admission requires overlay shard schema {}",
        WGMMA_CONTROL_SHARD_SCHEMA
    );
    ensure!(
        shard.tcgen05.is_none() || shard.schema >= TCGEN05_SHARD_SCHEMA,
        "compact tcgen05 admission requires overlay shard schema {}",
        TCGEN05_SHARD_SCHEMA
    );
    ensure!(
        shard
            .tcgen05
            .as_ref()
            .is_none_or(|admission| admission.cp_variants.is_empty())
            || shard.schema >= TCGEN05_CP_SHARD_SCHEMA,
        "compact tcgen05 copy admission requires overlay shard schema {}",
        TCGEN05_CP_SHARD_SCHEMA
    );
    ensure!(
        shard
            .tcgen05
            .as_ref()
            .is_none_or(|admission| admission.ld_variants.is_empty())
            || shard.schema >= TCGEN05_LD_SHARD_SCHEMA,
        "compact tcgen05 load admission requires overlay shard schema {}",
        TCGEN05_LD_SHARD_SCHEMA
    );
    ensure!(
        shard
            .tcgen05
            .as_ref()
            .is_none_or(|admission| admission.st_variants.is_empty())
            || shard.schema >= TCGEN05_ST_SHARD_SCHEMA,
        "compact tcgen05 store admission requires overlay shard schema {}",
        TCGEN05_ST_SHARD_SCHEMA
    );
    ensure!(
        shard.tcgen05.as_ref().is_none_or(|admission| {
            admission.ld_offset_variants.is_empty() && admission.st_offset_variants.is_empty()
        }) || shard.schema >= TCGEN05_OFFSET_LDST_SHARD_SCHEMA,
        "compact tcgen05 offset load/store admission requires overlay shard schema {}",
        TCGEN05_OFFSET_LDST_SHARD_SCHEMA
    );
    ensure!(
        shard.tcgen05.as_ref().is_none_or(|admission| {
            admission.control_llvm_evidence_profile.is_none()
                && admission.control_libnvvm_evidence_profile.is_none()
                && !admission.variants.iter().any(|variant| {
                    matches!(
                        variant.operation,
                        Tcgen05Operation::CommitMulticast
                            | Tcgen05Operation::ShiftDown
                            | Tcgen05Operation::ShiftDownCg2
                    )
                })
        }) || shard.schema >= TCGEN05_CONTROL_SHARD_SCHEMA,
        "compact tcgen05 control admission requires overlay shard schema {}",
        TCGEN05_CONTROL_SHARD_SCHEMA
    );
    if let Some(admission) = &shard.tcgen05
        && shard.schema >= TCGEN05_CONTROL_SHARD_SCHEMA
    {
        ensure!(
            admission
                .control_llvm_evidence_profile
                .as_deref()
                .is_some_and(|profile| !profile.trim().is_empty())
                && admission
                    .control_libnvvm_evidence_profile
                    .as_deref()
                    .is_some_and(|profile| !profile.trim().is_empty())
                && [
                    Tcgen05Operation::CommitMulticast,
                    Tcgen05Operation::ShiftDown,
                    Tcgen05Operation::ShiftDownCg2,
                ]
                .into_iter()
                .all(|operation| admission
                    .variants
                    .iter()
                    .any(|variant| variant.operation == operation)),
            "compact tcgen05 schema {} requires all three control variants and both backend evidence profiles",
            TCGEN05_CONTROL_SHARD_SCHEMA
        );
    }
    ensure!(
        shard.tcgen05.as_ref().is_none_or(|admission| {
            admission.mma_variants.is_empty()
                && admission.mma_llvm_evidence_profile.is_none()
                && admission.mma_libnvvm_evidence_profile.is_none()
                && admission.mma_llvm_target_contracts.is_empty()
                && admission.mma_libnvvm_target_contracts.is_empty()
        }) || shard.schema >= TCGEN05_MMA_SHARD_SCHEMA,
        "compact tcgen05 MMA admission requires overlay shard schema {}",
        TCGEN05_MMA_SHARD_SCHEMA
    );
    Ok(())
}

pub(super) fn validate_overlay_shard_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    ensure!(
        path.extension().and_then(|extension| extension.to_str()) == Some("toml"),
        "overlay shard path must name a TOML file: {}",
        path.display()
    );
    let components: Vec<_> = path.components().collect();
    ensure!(
        components.len() >= 2 && components[0] == Component::Normal("overlay".as_ref()),
        "overlay shard path must stay under intrinsics/overlay: {}",
        path.display()
    );
    ensure!(
        components
            .iter()
            .all(|component| matches!(component, Component::Normal(_))),
        "overlay shard path contains a non-normal component: {}",
        path.display()
    );
    Ok(())
}

pub(super) fn append_overlay_hash_input(output: &mut Vec<u8>, path: &str, contents: &[u8]) {
    output.extend_from_slice(&(path.len() as u64).to_le_bytes());
    output.extend_from_slice(path.as_bytes());
    output.extend_from_slice(&(contents.len() as u64).to_le_bytes());
    output.extend_from_slice(contents);
}

pub(super) fn shares_tma_2d_g2s_symbol(record: &OverlayIntrinsic, symbol: &str) -> bool {
    symbol == "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d"
        && record.tma.as_ref().is_some_and(|tma| {
            matches!(
                tma.operation,
                TmaOperation::G2sTile2d
                    | TmaOperation::G2sTile2dMulticast
                    | TmaOperation::G2sTile2dMulticastCg2
            )
        })
}

pub(super) fn shares_tma_prefetch_tile_symbol(record: &OverlayIntrinsic, symbol: &str) -> bool {
    let Some(tma) = record.tma.as_ref() else {
        return false;
    };
    let expected_symbol = match tma.operation {
        TmaOperation::PrefetchTile1d | TmaOperation::PrefetchTile1dCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d"
        }
        TmaOperation::PrefetchTile2d | TmaOperation::PrefetchTile2dCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d"
        }
        TmaOperation::PrefetchTile3d | TmaOperation::PrefetchTile3dCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d"
        }
        TmaOperation::PrefetchTile4d | TmaOperation::PrefetchTile4dCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d"
        }
        TmaOperation::PrefetchTile5d | TmaOperation::PrefetchTile5dCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d"
        }
        TmaOperation::PrefetchTileGather4TwoDimensional
        | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint => {
            "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.gather4.2d"
        }
        _ => return false,
    };
    symbol == expected_symbol
}

pub(super) fn shares_tcgen05_mma_symbol(record: &OverlayIntrinsic, symbol: &str) -> bool {
    record.tcgen05.as_ref().is_some_and(|tcgen05| {
        (symbol == "llvm.nvvm.tcgen05.mma.ws.tensor"
            && matches!(
                tcgen05.operation,
                Tcgen05Operation::MmaWsF16
                    | Tcgen05Operation::MmaWsBf16
                    | Tcgen05Operation::MmaWsTf32
            ))
            || tcgen05
                .mma
                .as_ref()
                .is_some_and(|mma| symbol == tcgen05_mma_llvm_symbol(mma.form))
    })
}

pub(super) fn shares_tcgen05_ld_symbol(record: &OverlayIntrinsic, symbol: &str) -> bool {
    let Some(tcgen05) = record.tcgen05.as_ref() else {
        return false;
    };
    if let Some(ld) = tcgen05.ld {
        return tcgen05.operation == Tcgen05Operation::Ld
            && record.source_record.as_deref() == Some(tcgen05_ld_source_record(ld).as_str())
            && symbol == tcgen05_ld_llvm_symbol(ld);
    }
    matches!(
        (tcgen05.operation, record.source_record.as_deref(), symbol),
        (
            Tcgen05Operation::Ld16x256bPure,
            Some("int_nvvm_tcgen05_ld_16x256b_x1"),
            "llvm.nvvm.tcgen05.ld.16x256b.x1"
        ) | (
            Tcgen05Operation::Ld16x256bX8Pure,
            Some("int_nvvm_tcgen05_ld_16x256b_x8"),
            "llvm.nvvm.tcgen05.ld.16x256b.x8"
        )
    )
}

pub(super) fn shares_tcgen05_st_symbol(record: &OverlayIntrinsic, symbol: &str) -> bool {
    record.tcgen05.as_ref().is_some_and(|tcgen05| {
        tcgen05.st.is_some_and(|st| {
            tcgen05.operation == Tcgen05Operation::St
                && record.source_record.as_deref() == Some(tcgen05_st_source_record(st).as_str())
                && symbol == tcgen05_st_llvm_symbol(st)
        })
    })
}

pub(super) fn validate_unique_overlay(
    records: &[OverlayIntrinsic],
    intrinsic_abi: u32,
) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut abi_ids = BTreeSet::new();
    let mut operation_keys = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut op_variants = BTreeSet::new();
    let mut op_type_names = BTreeMap::new();
    let mut symbol_bases = BTreeMap::new();
    let mut symbols = BTreeSet::new();
    let mut rust_items = BTreeSet::new();
    for record in records {
        insert_unique(&mut ids, &record.id, "catalog ID")?;
        validate_abi_id(&record.abi_id)?;
        insert_unique(&mut abi_ids, &record.abi_id, "intrinsic ABI ID")?;
        validate_operation_key(&record.operation_key)?;
        insert_unique(
            &mut operation_keys,
            &record.operation_key,
            "intrinsic operation key",
        )?;
        if let Some(previous_name) = op_type_names.insert(
            record.dialect_op_type.as_str(),
            record.dialect_op_name.as_str(),
        ) {
            ensure!(
                previous_name == record.dialect_op_name,
                "dialect op type {} maps to both {} and {}",
                record.dialect_op_type,
                previous_name,
                record.dialect_op_name
            );
        }
        let op_variant = format!(
            "{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
            record.dialect_op_name,
            record.ldmatrix_variant,
            record.packed_atomic,
            record.redux,
            record.vote,
            record.active_mask,
            record.warp_match,
            record.warp_barrier,
            record.warp_shuffle,
            record.dot_product,
            record.cp_async_copy,
            record.cp_async_control,
            record.cp_async_mbarrier,
            record.mbarrier_basic,
            record.movmatrix,
            record.mbarrier_extended,
            record.register_mma,
            record.sparse_mma,
            record.prmt,
            record.cluster_barrier,
            record.cluster_memory,
            record.scalar_conversion,
            record.scalar_arithmetic,
            record.scalar_math,
            record.extended_minmax,
            record.tcgen05,
        );
        insert_unique(&mut op_variants, &op_variant, "dialect op variant")?;
        if let Some(symbol) = &record.llvm_symbol {
            let is_resolved = record.resolved_llvm_symbol.is_some();
            let shares_reviewed_symbol = shares_tma_2d_g2s_symbol(record, symbol)
                || shares_tma_prefetch_tile_symbol(record, symbol)
                || shares_tcgen05_mma_symbol(record, symbol)
                || shares_tcgen05_ld_symbol(record, symbol)
                || shares_tcgen05_st_symbol(record, symbol);
            if let Some((previous_was_resolved, previous_shared_symbol)) =
                symbol_bases.insert(symbol, (is_resolved, shares_reviewed_symbol))
            {
                ensure!(
                    (previous_was_resolved && is_resolved)
                        || (previous_shared_symbol && shares_reviewed_symbol),
                    "duplicate LLVM symbol {symbol} is reused without a resolved symbol"
                );
            }
            if !shares_reviewed_symbol {
                insert_unique(
                    &mut symbols,
                    record.resolved_llvm_symbol.as_ref().unwrap_or(symbol),
                    "resolved LLVM symbol",
                )?;
            }
        }
        insert_unique(
            &mut rust_items,
            &format!("{}::{}", record.rust_module, record.rust_name),
            "raw Rust item",
        )?;
        insert_unique(
            &mut paths,
            &canonical_rust_path(intrinsic_abi, &record.abi_id),
            "canonical Rust path",
        )?;
        insert_unique(&mut paths, &record.public_rust_path, "public Rust path")?;
        for path in &record.compatibility_rust_paths {
            insert_unique(&mut paths, path, "compatibility Rust path")?;
        }
    }
    Ok(())
}
