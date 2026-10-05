/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{CatalogFile, ExtendedMinMaxFormat, IntrinsicBackend, PackedAluFormat};
#[cfg(test)]
use crate::render::collector_targets::render_targets;
use crate::render::collector_targets::{render_collector, render_targets_files};
use crate::render::common::backend_label;
use crate::render::compat::{
    render_compat_clc, render_compat_cluster_barrier, render_compat_cluster_memory,
    render_compat_cluster_sreg, render_compat_counted_barrier, render_compat_cp_async_copy,
    render_compat_debug_control, render_compat_dotprod, render_compat_fence,
    render_compat_float_output, render_compat_grid_dependency, render_compat_integer_minmax,
    render_compat_ldmatrix, render_compat_mbarrier_basic, render_compat_mbarrier_extended,
    render_compat_movmatrix, render_compat_packed_alu, render_compat_packed_atomic,
    render_compat_packed_conversion, render_compat_prmt, render_compat_register_control,
    render_compat_register_mma, render_compat_scalar_minmax, render_compat_sparse_mma,
    render_compat_special_register_module, render_compat_sreg, render_compat_stmatrix,
    render_compat_tcgen05, render_compat_tma, render_compat_wgmma_control,
};
use crate::render::dialect::{
    render_dialect_active_mask, render_dialect_clc, render_dialect_cluster_barrier,
    render_dialect_cluster_memory, render_dialect_cp_async_copy, render_dialect_debug_control,
    render_dialect_dotprod, render_dialect_elect, render_dialect_execution_control,
    render_dialect_extended_minmax, render_dialect_integer_minmax, render_dialect_ldmatrix,
    render_dialect_mbarrier_basic, render_dialect_mbarrier_extended, render_dialect_mod,
    render_dialect_movmatrix, render_dialect_packed_alu, render_dialect_packed_atomic,
    render_dialect_packed_conversion, render_dialect_prmt, render_dialect_redux,
    render_dialect_register_mma, render_dialect_scalar_arithmetic,
    render_dialect_scalar_conversion, render_dialect_scalar_math, render_dialect_sparse_mma,
    render_dialect_sreg, render_dialect_stmatrix, render_dialect_sync, render_dialect_tcgen05,
    render_dialect_tma, render_dialect_vote, render_dialect_warp_barrier,
    render_dialect_warp_match, render_dialect_warp_shuffle, render_dialect_wgmma_control,
};
use crate::render::families::{
    clc_intrinsics, cluster_memory, debug_controls, elect_intrinsics, execution_controls,
    extended_minmax, extended_minmax_contract, integer_minmaxes, mbarrier_extended, movmatrix,
    scalar_arithmetics, scalar_conversions, scalar_maths, stmatrices, sync_intrinsics,
    tcgen05_intrinsics, threadfence_ptx_level, tma_intrinsics, wgmma_controls,
};
#[cfg(test)]
use crate::render::importer::render_importer;
use crate::render::importer::render_importer_files;
#[cfg(test)]
use crate::render::lowering::render_lowering;
use crate::render::lowering::render_lowering_files;
use crate::render::probes::{render_elect_probe, render_special_register_probe};
use crate::render::raw_abi::{render_raw_abi, render_raw_mod};
use crate::render::reference::render_reference;
use crate::render::validate::validate_renderable;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::PathBuf;

mod collector_targets;
mod common;
mod compat;
mod dialect;
mod families;
mod importer;
mod lowering;
mod probes;
mod raw_abi;
mod reference;
#[cfg(test)]
mod tests;
mod validate;

pub(crate) use probes::render_probe;

pub fn all_outputs(
    catalog: &CatalogFile,
    catalog_json: String,
    catalog_sha256: &str,
) -> Result<BTreeMap<PathBuf, String>> {
    validate_renderable(catalog)?;
    let mut outputs = BTreeMap::new();
    outputs.insert("intrinsics/catalog.json".into(), catalog_json);
    outputs.insert(
        "crates/cuda-intrinsics/src/generated/mod.rs".into(),
        render_raw_mod(catalog, catalog_sha256),
    );
    outputs.insert(
        format!(
            "crates/cuda-intrinsics/src/generated/abi_v{}.rs",
            catalog.intrinsic_abi
        )
        .into(),
        render_raw_abi(catalog, catalog_sha256)?,
    );
    outputs.insert(
        "crates/cuda-device/src/generated/register_mma.rs".into(),
        render_compat_register_mma(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/ldmatrix.rs".into(),
        render_compat_ldmatrix(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/sparse_mma.rs".into(),
        render_compat_sparse_mma(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/sreg.rs".into(),
        render_compat_sreg(catalog, catalog_sha256),
    );
    for module in ["debug", "grid", "shared", "warp"] {
        outputs.insert(
            format!("crates/cuda-device/src/generated/{module}_sreg.rs").into(),
            render_compat_special_register_module(catalog, catalog_sha256, module),
        );
    }
    outputs.insert(
        "crates/cuda-device/src/generated/cluster_sreg.rs".into(),
        render_compat_cluster_sreg(catalog, catalog_sha256),
    );
    if sync_intrinsics(catalog).any(|record| threadfence_ptx_level(record).is_some()) {
        outputs.insert(
            "crates/cuda-device/src/generated/fence.rs".into(),
            render_compat_fence(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/cuda-device/src/generated/dotprod.rs".into(),
        render_compat_dotprod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/prmt.rs".into(),
        render_compat_prmt(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/cluster_barrier.rs".into(),
        render_compat_cluster_barrier(catalog, catalog_sha256),
    );
    if cluster_memory(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/cluster_memory.rs".into(),
            render_compat_cluster_memory(catalog, catalog_sha256),
        );
    }
    if stmatrices(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/stmatrix.rs".into(),
            render_compat_stmatrix(catalog, catalog_sha256),
        );
    }
    if movmatrix(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/movmatrix.rs".into(),
            render_compat_movmatrix(catalog, catalog_sha256),
        );
    }
    if wgmma_controls(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/wgmma_control.rs".into(),
            render_compat_wgmma_control(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/wgmma_control.rs".into(),
            render_dialect_wgmma_control(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/cuda-device/src/generated/atomic.rs".into(),
        render_compat_packed_atomic(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/async_copy.rs".into(),
        render_compat_cp_async_copy(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/mbarrier_basic.rs".into(),
        render_compat_mbarrier_basic(catalog, catalog_sha256),
    );
    if mbarrier_extended(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/mbarrier_extended.rs".into(),
            render_compat_mbarrier_extended(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/cuda-device/src/generated/bf16x2.rs".into(),
        render_compat_packed_alu(catalog, catalog_sha256, PackedAluFormat::Bf16x2),
    );
    for (module, format) in [
        ("f16", ExtendedMinMaxFormat::F16),
        ("bf16", ExtendedMinMaxFormat::Bf16),
    ] {
        if extended_minmax(catalog).any(|record| extended_minmax_contract(record).format == format)
        {
            outputs.insert(
                format!("crates/cuda-device/src/generated/{module}.rs").into(),
                render_compat_scalar_minmax(catalog, catalog_sha256, module),
            );
        }
    }
    outputs.insert(
        "crates/cuda-device/src/generated/f16x2.rs".into(),
        render_compat_packed_alu(catalog, catalog_sha256, PackedAluFormat::F16x2),
    );
    for module in ["i16x2", "int"] {
        if integer_minmaxes(catalog).any(|record| record.rust.module == module) {
            outputs.insert(
                format!("crates/cuda-device/src/generated/{module}.rs").into(),
                render_compat_integer_minmax(catalog, catalog_sha256, module),
            );
        }
    }
    if integer_minmaxes(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/integer_minmax.rs".into(),
            render_dialect_integer_minmax(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/cuda-device/src/generated/f32x2.rs".into(),
        render_compat_packed_alu(catalog, catalog_sha256, PackedAluFormat::F32x2),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/convert.rs".into(),
        render_compat_packed_conversion(
            catalog,
            catalog_sha256,
            "cuda_device::convert::",
            "convert",
            ("lo", "hi"),
        ),
    );
    if let Some((path, source)) = render_compat_float_output(catalog, catalog_sha256) {
        outputs.insert(path, source);
    }
    if extended_minmax(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/extended_minmax.rs".into(),
            render_dialect_extended_minmax(catalog, catalog_sha256),
        );
    }
    if debug_controls(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/debug_control.rs".into(),
            render_compat_debug_control(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/debug_control.rs".into(),
            render_dialect_debug_control(catalog, catalog_sha256),
        );
    }
    if clc_intrinsics(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/clc.rs".into(),
            render_compat_clc(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/clc.rs".into(),
            render_dialect_clc(catalog, catalog_sha256),
        );
    }
    if tma_intrinsics(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/tma.rs".into(),
            render_compat_tma(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/tma.rs".into(),
            render_dialect_tma(catalog, catalog_sha256),
        );
    }
    if execution_controls(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/counted_barrier.rs".into(),
            render_compat_counted_barrier(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/cuda-device/src/generated/grid_dependency.rs".into(),
            render_compat_grid_dependency(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/cuda-device/src/generated/register_control.rs".into(),
            render_compat_register_control(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/execution_control.rs".into(),
            render_dialect_execution_control(catalog, catalog_sha256),
        );
    }
    if tcgen05_intrinsics(catalog).next().is_some() {
        outputs.insert(
            "crates/cuda-device/src/generated/tcgen05.rs".into(),
            render_compat_tcgen05(catalog, catalog_sha256),
        );
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/tcgen05.rs".into(),
            render_dialect_tcgen05(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/mod.rs".into(),
        render_dialect_mod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sreg.rs".into(),
        render_dialect_sreg(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/dotprod.rs".into(),
        render_dialect_dotprod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/ldmatrix.rs".into(),
        render_dialect_ldmatrix(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/register_mma.rs".into(),
        render_dialect_register_mma(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sparse_mma.rs".into(),
        render_dialect_sparse_mma(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_atomic.rs".into(),
        render_dialect_packed_atomic(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_alu.rs".into(),
        render_dialect_packed_alu(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_conversion.rs".into(),
        render_dialect_packed_conversion(catalog, catalog_sha256),
    );
    if scalar_conversions(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/scalar_conversion.rs".into(),
            render_dialect_scalar_conversion(catalog, catalog_sha256),
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/scalar_arithmetic.rs".into(),
            render_dialect_scalar_arithmetic(catalog, catalog_sha256),
        );
    }
    if scalar_maths(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/scalar_math.rs".into(),
            render_dialect_scalar_math(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/prmt.rs".into(),
        render_dialect_prmt(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/cluster_barrier.rs".into(),
        render_dialect_cluster_barrier(catalog, catalog_sha256),
    );
    if cluster_memory(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/cluster_memory.rs".into(),
            render_dialect_cluster_memory(catalog, catalog_sha256),
        );
    }
    if stmatrices(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/stmatrix.rs".into(),
            render_dialect_stmatrix(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/redux.rs".into(),
        render_dialect_redux(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sync.rs".into(),
        render_dialect_sync(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/vote.rs".into(),
        render_dialect_vote(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/active_mask.rs".into(),
        render_dialect_active_mask(catalog, catalog_sha256),
    );
    if elect_intrinsics(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/elect.rs".into(),
            render_dialect_elect(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/cp_async.rs".into(),
        render_dialect_cp_async_copy(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/mbarrier_basic.rs".into(),
        render_dialect_mbarrier_basic(catalog, catalog_sha256),
    );
    if movmatrix(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/movmatrix.rs".into(),
            render_dialect_movmatrix(catalog, catalog_sha256),
        );
    }
    if mbarrier_extended(catalog).next().is_some() {
        outputs.insert(
            "crates/dialect-nvvm/src/ops/generated/mbarrier_extended.rs".into(),
            render_dialect_mbarrier_extended(catalog, catalog_sha256),
        );
    }
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_match.rs".into(),
        render_dialect_warp_match(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_barrier.rs".into(),
        render_dialect_warp_barrier(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_shuffle.rs".into(),
        render_dialect_warp_shuffle(catalog, catalog_sha256),
    );
    for (path, contents) in render_importer_files(catalog, catalog_sha256) {
        outputs.insert(path, contents);
    }
    for (path, contents) in render_lowering_files(catalog, catalog_sha256) {
        outputs.insert(path, contents);
    }
    outputs.insert(
        "crates/rustc-codegen-cuda/src/generated_intrinsics.rs".into(),
        render_collector(catalog, catalog_sha256),
    );
    for (path, contents) in render_targets_files(catalog, catalog_sha256) {
        outputs.insert(path, contents);
    }
    for record in &catalog.intrinsics {
        if record.special_register.is_some() || record.family == "elect" {
            for backend in [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm] {
                outputs.insert(
                    format!(
                        "intrinsics/probes/{}.{}.ll",
                        record.id,
                        backend_label(backend)
                    )
                    .into(),
                    if record.family == "elect" {
                        render_elect_probe(catalog, record, catalog_sha256, backend)
                    } else {
                        render_special_register_probe(catalog, record, catalog_sha256, backend)
                    },
                );
            }
        } else {
            outputs.insert(
                format!("intrinsics/probes/{}.ll", record.id).into(),
                render_probe(catalog, record, catalog_sha256),
            );
        }
    }
    outputs.insert(
        "intrinsics/generated-reference.md".into(),
        render_reference(catalog, catalog_sha256),
    );
    Ok(outputs)
}
