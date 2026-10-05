/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile-IR optimization hint attribute builders for compiler2.
//!
//! Re-exports the shared `OptimizationHints` / `SMHints` types from `crate::hints`,
//! and provides tile-ir-specific attribute construction functions.

use cutile_ir::ir::{Attribute, OptimizationHints as TileIrOptHints, ScalarType, Type};
use std::collections::HashMap;

// Re-export the shared hint types.
pub use crate::hints::{OptimizationHints, SMHints};

// ---------------------------------------------------------------------------
// Tile-IR attribute builders
// ---------------------------------------------------------------------------

/// Builds a tile-ir `OptimizationHints` attribute for the entry function.
pub fn build_entry_optimization_hints(hints: &OptimizationHints) -> Option<Attribute> {
    let mut entries = Vec::new();
    for (arch, sm_hints) in &hints.tile_as_hints {
        let mut arch_hints = Vec::new();
        if let Some(num_cta) = sm_hints.num_cta_in_cga {
            arch_hints.push((
                "num_cta_in_cga".to_string(),
                Attribute::Integer(num_cta as i64, Type::Scalar(ScalarType::I32)),
            ));
        }
        if let Some(occ) = sm_hints.occupancy {
            arch_hints.push((
                "occupancy".to_string(),
                Attribute::Integer(occ as i64, Type::Scalar(ScalarType::I32)),
            ));
        }
        if let Some(num_worker_warps) = sm_hints.num_worker_warps_per_cta {
            arch_hints.push((
                "num_worker_warps_per_cta".to_string(),
                Attribute::Integer(num_worker_warps as i64, Type::Scalar(ScalarType::I32)),
            ));
        }
        if !arch_hints.is_empty() {
            entries.push((arch.clone(), arch_hints));
        }
    }
    if entries.is_empty() {
        None
    } else {
        Some(Attribute::OptimizationHints(TileIrOptHints { entries }))
    }
}

/// Builds a tile-ir `OptimizationHints` attribute for load/store operations.
///
/// Only emits per-op call-site hint_params (latency, allow_tma, etc.).
/// Entry-level arch hints are NOT merged here — they are applied at the entry
/// function level via `build_entry_optimization_hints`.
pub fn build_load_store_hints(
    hints: &OptimizationHints,
    hint_params: HashMap<String, i32>,
) -> Option<Attribute> {
    if hint_params.is_empty() {
        return None;
    }
    let target_arch = hints
        .target_gpu_name
        .as_ref()
        .expect("Target gpu not yet specified. Did you compile?");
    let arch_hints: Vec<(String, Attribute)> = hint_params
        .iter()
        .map(|(k, v)| {
            if k == "allow_tma" {
                // allow_tma is a boolean: 0 = false, nonzero = true.
                (k.clone(), Attribute::Bool(*v != 0))
            } else {
                (
                    k.clone(),
                    Attribute::Integer(*v as i64, Type::Scalar(ScalarType::I32)),
                )
            }
        })
        .collect();
    Some(Attribute::OptimizationHints(TileIrOptHints {
        entries: vec![(target_arch.clone(), arch_hints)],
    }))
}
