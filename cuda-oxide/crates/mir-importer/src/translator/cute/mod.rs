/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Translate `cute-rs` calls into CuTe operations.
//!
//! CuTe-specific call handling lives here. The shared translator calls this
//! module from `translate_call`, as it does for CUDA intrinsics.
//!
//! Calls are matched by their definition path, found by walking the `DefId`
//! parent chain. This keeps matching stable when a function is re-exported or
//! a trait method is printed with a different name:
//!
//! ```text
//! public name:      cute_rs::load_tile
//! definition path:  cute_rs::tile::load_tile -> CUTE_FNS entry
//! ```
//!
//! A recognized call becomes a CuTe operation; its Rust body is not translated.
//! The importer checks its specialized generic arguments against [`SubstKind`]
//! before reading static configuration. Ordinary Rust helpers in explicitly
//! allowed modules use normal body translation and are inlined by LLVM.
//! Any other unknown `cute_rs` call is an error: translating a compiler stub's
//! body could turn layout operations into scalar code and lose their meaning.

pub(crate) mod block_scaled;
pub(crate) mod block_scaled_emit;
pub(crate) mod emit;
pub(crate) mod epilogue;
pub(crate) mod epilogue_emit;
pub(crate) mod layout;
pub(crate) mod pipeline;
pub(crate) mod pipeline_emit;
pub(crate) mod scheduler;
pub(crate) mod scheduler_emit;
pub(crate) mod sm100_emit;
pub(crate) mod smem_mma;
pub(crate) mod smem_mma_emit;
pub(crate) mod static_config;
pub(crate) mod tensor;
pub(crate) mod tensor_emit;

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::input_error_noloc;
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::FnDef;

/// Recognized cute-rs entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CuteFn {
    Sm100(sm100_emit::Sm100Fn),
    MakeTensorRead,
    MakeTensorWrite,
    ZippedDivideRead,
    ZippedDivideWrite,
    SliceRead,
    SliceWrite,
    TensorTileIsFull,
    TensorTileBase,
    TensorLoadTile,
    TensorStoreTile,
    TensorStoreElementAbs,
    BlockScaledMake,
    BlockScaledThreadRow,
    BlockScaledKTile,
    BlockScaledLoadK64,
    BlockScaledDotK64,
    SchedulerNew1d,
    SchedulerHasWork,
    SchedulerCurrentTile,
    WorkTileCoordinates,
    SchedulerAdvance,
    TmaLoadPipelineMake,
    TmaLoadPipelineFromRawPartsRejected,
    TmaLoadPipelineInit,
    PipelineStateNew,
    PipelineStateSlot,
    PipelineStateAdvance,
    PipelineProducerAcquire,
    PipelineProducerExpectTx,
    PipelineConsumerWait,
    PipelineConsumerRelease,
    PipelineProducerTail,
    SharedTensorOverlay,
    TiledMmaSlice,
    MmaFragmentFill,
    MmaLoadScales,
    MmaFragmentSliceK,
    MmaLoadA,
    MmaPartitionB,
    TiledGemm,
    EpilogueOverlay,
    EpilogueWarpSlice,
    EpilogueStoreFragment,
    EpilogueSyncReusable,
    EpilogueSyncReadyForTma,
    EpilogueHalf,
    TmaStoreAcquire,
    TmaStoreCommit,
    TmaStoreTail,
    LoadTile,
    StoreTile,
    AssumeDiv,
    CopyG2S,
    LoadMatrixA,
    LoadMatrixB,
    CopyTma2d,
    CopyTmaS2g2d,
}

/// Expected kind of one generic argument in a recognized function's substs.
#[allow(dead_code)] // `Lifetime` makes future early-bound lifetimes explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubstKind {
    Type,
    Const,
    Lifetime,
}

/// A recognized function's definition path and expected generic arguments.
/// Re-export paths do not appear here.
pub(crate) struct CuteFnSpec {
    pub canonical: &'static str,
    pub func: CuteFn,
    pub schema: &'static [SubstKind],
}

/// One entry per recognized function, including its generic argument schema.
pub(crate) const CUTE_FNS: &[CuteFnSpec] = &[
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tma_store",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::TmaStore),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_commit",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::StoreCommit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_acquire",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::StoreAcquire),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_tail",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::StoreTail),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tmem_alloc",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::TmemAlloc),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tmem_dealloc",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::TmemDealloc),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tiled_mma",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::TiledMma),
        schema: &[
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Const,
            SubstKind::Const,
            SubstKind::Const,
        ],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tmem_epilogue",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::TmemEpilogue),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_cluster_tma_load",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::ClusterTmaLoad),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_init",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::PipelineInit),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_acquire",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::PipelineAcquire),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_expect",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::PipelineExpect),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_wait",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::PipelineWait),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_release",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::PipelineRelease),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_init",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::AccumulatorInit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_acquire",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::AccumulatorAcquire),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_commit",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::AccumulatorCommit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_wait",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::AccumulatorWait),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_release",
        func: CuteFn::Sm100(sm100_emit::Sm100Fn::AccumulatorRelease),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::make_tensor_read",
        func: CuteFn::MakeTensorRead,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::make_tensor_write",
        func: CuteFn::MakeTensorWrite,
        schema: &[SubstKind::Lifetime, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::zipped_divide_read",
        func: CuteFn::ZippedDivideRead,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::zipped_divide_write",
        func: CuteFn::ZippedDivideWrite,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::slice_read",
        func: CuteFn::SliceRead,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::slice_write",
        func: CuteFn::SliceWrite,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::tensor_tile_is_full",
        func: CuteFn::TensorTileIsFull,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::tensor_tile_base",
        func: CuteFn::TensorTileBase,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::tensor_load_tile",
        func: CuteFn::TensorLoadTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::tensor_store_tile",
        func: CuteFn::TensorStoreTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tensor::tensor_store_element_abs",
        func: CuteFn::TensorStoreElementAbs,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::block_scaled_make",
        func: CuteFn::BlockScaledMake,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::block_scaled_thread_row",
        func: CuteFn::BlockScaledThreadRow,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::block_scaled_k_tile",
        func: CuteFn::BlockScaledKTile,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::block_scaled_load_k64",
        func: CuteFn::BlockScaledLoadK64,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::block_scaled_dot_k64",
        func: CuteFn::BlockScaledDotK64,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::scheduler::__compiler::scheduler_new_1d",
        func: CuteFn::SchedulerNew1d,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::scheduler::__compiler::scheduler_has_work",
        func: CuteFn::SchedulerHasWork,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::scheduler::__compiler::scheduler_current_tile",
        func: CuteFn::SchedulerCurrentTile,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::scheduler::__compiler::work_tile_coordinates",
        func: CuteFn::WorkTileCoordinates,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::scheduler::__compiler::scheduler_advance",
        func: CuteFn::SchedulerAdvance,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::tma_load_pipeline_from_raw_base",
        func: CuteFn::TmaLoadPipelineMake,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::tma_load_pipeline_init",
        func: CuteFn::TmaLoadPipelineInit,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_state_new_producer",
        func: CuteFn::PipelineStateNew,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_state_new_consumer",
        func: CuteFn::PipelineStateNew,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_state_slot",
        func: CuteFn::PipelineStateSlot,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_state_advance",
        func: CuteFn::PipelineStateAdvance,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_producer_acquire",
        func: CuteFn::PipelineProducerAcquire,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_producer_expect_tx",
        func: CuteFn::PipelineProducerExpectTx,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_consumer_wait",
        func: CuteFn::PipelineConsumerWait,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_consumer_release",
        func: CuteFn::PipelineConsumerRelease,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::pipeline_producer_tail",
        func: CuteFn::PipelineProducerTail,
        schema: &[SubstKind::Const, SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tiled_copy::__compiler::shared_tensor_overlay",
        func: CuteFn::SharedTensorOverlay,
        schema: &[
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
        ],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::tiled_mma_slice",
        func: CuteFn::TiledMmaSlice,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::fragment_fill",
        func: CuteFn::MmaFragmentFill,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::mma_load_scales",
        func: CuteFn::MmaLoadScales,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::fragment_slice_k",
        func: CuteFn::MmaFragmentSliceK,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::mma_load_a",
        func: CuteFn::MmaLoadA,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::mma_partition_b",
        func: CuteFn::MmaPartitionB,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::tiled_gemm",
        func: CuteFn::TiledGemm,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::__compiler::epilogue_smem_overlay",
        func: CuteFn::EpilogueOverlay,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::__compiler::epilogue_warp_slice",
        func: CuteFn::EpilogueWarpSlice,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled_mma::__compiler::epilogue_store_fragment",
        func: CuteFn::EpilogueStoreFragment,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::__compiler::epilogue_sync_reusable",
        func: CuteFn::EpilogueSyncReusable,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::__compiler::epilogue_sync_ready_for_tma",
        func: CuteFn::EpilogueSyncReadyForTma,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::__compiler::epilogue_half",
        func: CuteFn::EpilogueHalf,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::tma_store_producer_acquire",
        func: CuteFn::TmaStoreAcquire,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::tma_store_producer_commit",
        func: CuteFn::TmaStoreCommit,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::pipeline::__compiler::tma_store_producer_tail",
        func: CuteFn::TmaStoreTail,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tile::load_tile",
        func: CuteFn::LoadTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tile::store_tile",
        func: CuteFn::StoreTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::tile::assume_div",
        func: CuteFn::AssumeDiv,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_a",
        func: CuteFn::LoadMatrixA,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_b",
        func: CuteFn::LoadMatrixB,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::tma::copy_tma_2d",
        func: CuteFn::CopyTma2d,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::tma::copy_tma_s2g_2d",
        func: CuteFn::CopyTmaS2g2d,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::cooperative::copy_g2s",
        func: CuteFn::CopyG2S,
        schema: &[
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            // Rust's argument-position `impl LeadingDimMarker` is one
            // inferred type. It keeps the user-facing turbofish at six
            // entries while making the pitch promise visible here.
            SubstKind::Type,
        ],
    },
];

enum CuteIdentity {
    /// Callee is not from the cute_rs crate; normal translation proceeds.
    NotCuteCrate,
    /// A recognized entry point.
    Known(&'static CuteFnSpec),
    /// Ordinary zero-cost Rust whose body is intentionally translated.
    SourceAbstraction,
    /// In the cute_rs crate but not in the table: hard error (fall-through
    /// would body-translate and scalarize the layout).
    Unknown(String),
}

/// Definition modules containing ordinary Rust source abstractions rather
/// than importer-owned recognition stubs.
///
/// This is deliberately a closed list. Adding a module means every function
/// defined directly or in an impl below it must have a real, device-safe body.
/// Panic-only recognition stubs are forbidden in these modules: they would be
/// body-translated instead of dispatched through [`CUTE_FNS`]. New modules
/// therefore require an explicit audit and an update to the exact-list test.
const CUTE_SOURCE_ABSTRACTION_MODULES: &[&str] = &[
    "block_scaled",
    "block_scaled_mma",
    "epilogue",
    "numeric",
    "pipeline",
    "scheduler",
    "sm100",
    "tensor",
    "tiled_copy",
];

fn is_source_abstraction_path(path: &str, crate_name: &str) -> bool {
    let Some(rest) = path
        .strip_prefix(crate_name)
        .and_then(|rest| rest.strip_prefix("::"))
    else {
        return false;
    };
    let first_module = rest.split("::").next().unwrap_or_default();
    CUTE_SOURCE_ABSTRACTION_MODULES.contains(&first_module)
}

/// Rebuild the canonical definition path by walking DefId parents,
/// mirroring `intrinsics::generated::classify_raw_intrinsic`.
fn canonical_path(fn_def: &FnDef, crate_name: &str) -> String {
    let mut segments = Vec::new();
    let mut current = Some(fn_def.def_id());
    while let Some(def_id) = current {
        let printed = def_id.name();
        let segment = printed.as_str().rsplit("::").next().unwrap_or_default();
        segments.push(segment.to_owned());
        current = def_id.parent();
    }
    canonical_path_from_leaf_segments(crate_name, segments)
}

/// Join a DefId parent walk without erasing a same-named nested module.
pub(super) fn canonical_path_from_leaf_segments(
    crate_name: &str,
    mut segments: Vec<String>,
) -> String {
    segments.reverse();
    // Only the first segment is the crate root. A nested module may legally
    // have the same spelling and must remain part of the identity:
    //
    // cute_rs::cute_rs::tile::load_tile != cute_rs::tile::load_tile
    if segments
        .first()
        .is_some_and(|segment| segment == crate_name)
    {
        segments.remove(0);
    }
    format!("{crate_name}::{}", segments.join("::"))
}

#[cfg(test)]
mod canonical_path_tests {
    use super::{
        block_scaled_method_name_from_path, canonical_path_from_leaf_segments,
        is_source_abstraction_path, pipeline_method_name_from_path,
        scheduler_method_name_from_path, smem_mma_method_name_from_path,
        tensor_method_name_from_path,
    };

    #[test]
    fn removes_only_the_crate_root_segment() {
        let path = canonical_path_from_leaf_segments(
            "cute_rs",
            ["load_tile", "tile", "cute_rs", "cute_rs"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        assert_eq!(path, "cute_rs::cute_rs::tile::load_tile");
    }

    #[test]
    fn source_abstraction_modules_are_exact_and_closed() {
        assert_eq!(
            super::CUTE_SOURCE_ABSTRACTION_MODULES,
            &[
                "block_scaled",
                "block_scaled_mma",
                "epilogue",
                "numeric",
                "pipeline",
                "scheduler",
                "sm100",
                "tensor",
                "tiled_copy",
            ]
        );
        assert!(is_source_abstraction_path(
            "cute_rs::block_scaled::BlockScaledTensor<...>::thread_row",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute-rs::numeric::PackedE2M1x2::to_f32x2",
            "cute-rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::tensor::Tensor<...>::layout",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::tiled_copy::TiledCopy<...>::copy",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::block_scaled_mma::Mxfp4TiledMma<...>::gemm",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::epilogue::Sm120EpilogueWarp128x128::store_atom",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::pipeline::TmaLoadPipeline<...>::consumer_wait",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::scheduler::StaticPersistentTileScheduler<...>::current",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::tile::load_tile",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::block_scaled_extra::load",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "other::block_scaled::load",
            "cute_rs"
        ));
    }

    #[test]
    fn tensor_method_path_accepts_rustc_impl_owner_spellings() {
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::Contiguous1D>::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::{impl#6}::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute-rs::tensor::Tile1D<TILE>>::store_linear"),
            Some("store_linear")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::from_slice"),
            None,
            "a free function is not an inherent method"
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::other::Tile1D<TILE>>::base"),
            None
        );
    }

    #[test]
    fn block_scaled_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            block_scaled_method_name_from_path("cute_rs::block_scaled::KMajor<Mode>>::from_slices"),
            Some("from_slices")
        );
        assert_eq!(
            block_scaled_method_name_from_path("cute_rs::block_scaled::{impl#21}::dot_accumulate"),
            Some("dot_accumulate")
        );
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::__compiler::block_scaled_dot_k64"
            ),
            Some("block_scaled_dot_k64"),
            "the exact helper table takes precedence over method classification"
        );
        assert_eq!(
            block_scaled_method_name_from_path("cute_rs::other::{impl#1}::load"),
            None
        );
    }

    #[test]
    fn scheduler_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            scheduler_method_name_from_path(
                "cute_rs::scheduler::StaticPersistentTileScheduler<...>>::has_work"
            ),
            Some("has_work")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::{impl#4}::coordinates"),
            Some("coordinates")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::__compiler::scheduler_has_work"),
            Some("scheduler_has_work"),
            "the exact helper table takes precedence over method classification"
        );
    }

    #[test]
    fn pipeline_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::TmaLoadPipeline<...>>::producer_expect_tx"
            ),
            Some("producer_expect_tx")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::{impl#4}::advance"),
            Some("advance")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::__compiler::pipeline_consumer_wait"),
            Some("pipeline_consumer_wait"),
            "the exact helper table takes precedence over method classification"
        );
    }

    #[test]
    fn shared_mma_method_path_keeps_module_and_method_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::Mxfp4TiledMma<...>>::load_a_128",
                "block_scaled_mma"
            ),
            Some("load_a_128")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute-rs::tiled_copy::{impl#7}::from_raw_parts",
                "tiled_copy"
            ),
            Some("from_raw_parts")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::other::Mxfp4TiledMma<...>>::load_a_128",
                "block_scaled_mma"
            ),
            None
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::load_a_128",
                "block_scaled_mma"
            ),
            None,
            "a free function is not an inherent method"
        );
    }

    #[test]
    fn epilogue_method_paths_keep_each_owner_module_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::Sm120Epilogue128x128::sync_ready_for_tma",
                "epilogue"
            ),
            Some("sync_ready_for_tma")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::{impl#9}::store_tile",
                "block_scaled_mma"
            ),
            Some("store_tile")
        );
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::TmaStorePipeline<1>::producer_commit"
            ),
            Some("producer_commit")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::Sm120EpilogueWarp128x128::store_tile",
                "block_scaled_mma"
            ),
            None,
            "store_tile is defined in block_scaled_mma, not epilogue"
        );
    }
}

fn classify_cute_call(fn_def: &FnDef) -> CuteIdentity {
    let crate_name = fn_def.krate().name.to_string();
    if !matches!(crate_name.as_str(), "cute_rs" | "cute-rs") {
        return CuteIdentity::NotCuteCrate;
    }
    let path = canonical_path(fn_def, &crate_name);
    match CUTE_FNS.iter().find(|spec| spec.canonical == path) {
        Some(spec) => CuteIdentity::Known(spec),
        None if is_source_abstraction_path(&path, &crate_name) => CuteIdentity::SourceAbstraction,
        None => CuteIdentity::Unknown(path),
    }
}

/// Recognize only the ergonomic inherent methods that expose the v0 tensor
/// flow. The DefId's direct parent is an implementation block, whose printed
/// number is intentionally ignored; the receiver/result ADT supplies the
/// stable identity instead.
fn tensor_method_name_from_path(path: &str) -> Option<&str> {
    let mut segments = path.split("::");
    let crate_name = segments.next()?;
    let module = segments.next()?;
    // rustc_public currently prints an impl parent using the tail of its self
    // type (for example `Tile1D<TILE>>`), while older revisions exposed an
    // `{impl#N}` segment. That spelling is not identity. The exact receiver
    // and result ADTs checked by `classify_tensor_method_call` are.
    let impl_owner = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some()
        || !matches!(crate_name, "cute_rs" | "cute-rs")
        || module != "tensor"
        || impl_owner.is_empty()
    {
        return None;
    }
    Some(method)
}

/// Return an inherent method name from the exact block-scaled module.
///
/// rustc's spelling for the implementation owner is not stable. Receiver and
/// result ADTs below provide the identity instead.
fn block_scaled_method_name_from_path(path: &str) -> Option<&str> {
    let mut segments = path.split("::");
    let crate_name = segments.next()?;
    let module = segments.next()?;
    let impl_owner = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some()
        || !matches!(crate_name, "cute_rs" | "cute-rs")
        || module != "block_scaled"
        || impl_owner.is_empty()
    {
        return None;
    }
    Some(method)
}

/// Return an inherent method name from the exact shared-copy or MMA module.
fn smem_mma_method_name_from_path<'a>(path: &'a str, expected_module: &str) -> Option<&'a str> {
    let mut segments = path.split("::");
    let crate_name = segments.next()?;
    let module = segments.next()?;
    let impl_owner = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some()
        || !matches!(crate_name, "cute_rs" | "cute-rs")
        || module != expected_module
        || impl_owner.is_empty()
    {
        return None;
    }
    Some(method)
}

fn classify_smem_mma_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let destination_ty = destination.ty(body.locals()).ok();
    let operand_ty = |index: usize| args.get(index)?.ty(body.locals()).ok();

    if smem_mma_method_name_from_path(path, "tiled_copy") == Some("from_raw_parts")
        && args.len() == 2
        && destination_ty
            .as_ref()
            .is_some_and(|ty| smem_mma::decode_shared_tensor(ty).ok().flatten().is_some())
    {
        return Some(CuteFn::SharedTensorOverlay);
    }

    let method = smem_mma_method_name_from_path(path, "block_scaled_mma")?;
    match method {
        "get_slice"
            if args.len() == 1
                && destination_ty
                    .as_ref()
                    .is_some_and(|ty| smem_mma::decode_tiled_mma(ty).ok().flatten().is_some()) =>
        {
            Some(CuteFn::TiledMmaSlice)
        }
        "zero"
            if args.is_empty()
                && destination_ty
                    .as_ref()
                    .is_some_and(|ty| smem_mma::is_accumulator(ty).unwrap_or(false)) =>
        {
            Some(CuteFn::MmaFragmentFill)
        }
        "load_scale_atom_128"
            if args.len() == 5
                && operand_ty(0).is_some_and(|ty| {
                    smem_mma::decode_tiled_mma_receiver(&ty)
                        .ok()
                        .flatten()
                        .is_some()
                })
                && destination_ty
                    .as_ref()
                    .is_some_and(|ty| smem_mma::is_scale_stage(ty).unwrap_or(false)) =>
        {
            Some(CuteFn::MmaLoadScales)
        }
        "pairs_at_unchecked"
            if args.len() == 2
                && operand_ty(0)
                    .is_some_and(|ty| smem_mma::is_scale_stage(&ty).unwrap_or(false))
                && destination_ty
                    .as_ref()
                    .is_some_and(|ty| smem_mma::is_scale_k64(ty).unwrap_or(false)) =>
        {
            Some(CuteFn::MmaFragmentSliceK)
        }
        "load_a_128"
            if args.len() == 4
                && operand_ty(0).is_some_and(|ty| {
                    smem_mma::decode_tiled_mma_receiver(&ty)
                        .ok()
                        .flatten()
                        .is_some()
                }) =>
        {
            Some(CuteFn::MmaLoadA)
        }
        "get_b_tile_k64"
            if args.len() == 4
                && destination_ty
                    .as_ref()
                    .is_some_and(|ty| smem_mma::decode_b_tile(ty).ok().flatten().is_some()) =>
        {
            Some(CuteFn::MmaPartitionB)
        }
        "accumulate_k64"
            if args.len() == 5
                && operand_ty(0)
                    .is_some_and(|ty| smem_mma::is_accumulator(&ty).unwrap_or(false)) =>
        {
            Some(CuteFn::TiledGemm)
        }
        _ => None,
    }
}

fn classify_epilogue_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let operand_ty = |index: usize| args.get(index)?.ty(body.locals()).ok();
    let destination_ty = destination.ty(body.locals()).ok();

    if let Some(method) = smem_mma_method_name_from_path(path, "epilogue") {
        let tile_receiver = operand_ty(0)
            .as_ref()
            .is_some_and(|ty| epilogue::is_epilogue_tile_receiver(ty).unwrap_or(false));
        return match method {
            "from_raw"
                if args.len() == 1
                    && destination_ty
                        .as_ref()
                        .is_some_and(|ty| epilogue::is_epilogue_tile(ty).unwrap_or(false)) =>
            {
                Some(CuteFn::EpilogueOverlay)
            }
            "get_slice"
                if args.len() == 3
                    && tile_receiver
                    && destination_ty.as_ref().is_some_and(|ty| {
                        epilogue::is_epilogue_warp_slice(ty).unwrap_or(false)
                    }) =>
            {
                Some(CuteFn::EpilogueWarpSlice)
            }
            "sync_reusable" if args.len() == 1 && tile_receiver => {
                Some(CuteFn::EpilogueSyncReusable)
            }
            "sync_ready_for_tma" if args.len() == 1 && tile_receiver => {
                Some(CuteFn::EpilogueSyncReadyForTma)
            }
            "tma_half" if args.len() == 1 && tile_receiver => Some(CuteFn::EpilogueHalf),
            _ => None,
        };
    }

    if smem_mma_method_name_from_path(path, "block_scaled_mma") == Some("store_tile")
        && args.len() == 2
        && operand_ty(0)
            .as_ref()
            .is_some_and(|ty| epilogue::is_epilogue_warp_slice_receiver(ty).unwrap_or(false))
        && operand_ty(1)
            .as_ref()
            .is_some_and(|ty| epilogue::is_accumulator(ty).unwrap_or(false))
    {
        return Some(CuteFn::EpilogueStoreFragment);
    }

    let method = pipeline_method_name_from_path(path)?;
    let store_pipeline = operand_ty(0)
        .as_ref()
        .and_then(|ty| epilogue::decode_store_pipeline(ty).ok().flatten());
    match method {
        "producer_acquire" if args.len() == 1 && store_pipeline == Some(1) => {
            Some(CuteFn::TmaStoreAcquire)
        }
        "producer_commit" if args.len() == 1 && store_pipeline == Some(1) => {
            Some(CuteFn::TmaStoreCommit)
        }
        "producer_tail" if args.len() == 1 && store_pipeline == Some(1) => {
            Some(CuteFn::TmaStoreTail)
        }
        _ => None,
    }
}

/// Return an inherent method name from the exact scheduler module.
///
/// The impl-block spelling changes between rustc revisions. Exact scheduler
/// and work-tile receiver types below provide the stable identity.
fn scheduler_method_name_from_path(path: &str) -> Option<&str> {
    let mut segments = path.split("::");
    let crate_name = segments.next()?;
    let module = segments.next()?;
    let impl_owner = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some()
        || !matches!(crate_name, "cute_rs" | "cute-rs")
        || module != "scheduler"
        || impl_owner.is_empty()
    {
        return None;
    }
    Some(method)
}

/// Return an inherent method name from the exact pipeline module.
///
/// The impl-block spelling is not stable. Exact pipeline/state Rust types in
/// `classify_pipeline_method_call` provide the method identity.
fn pipeline_method_name_from_path(path: &str) -> Option<&str> {
    let mut segments = path.split("::");
    let crate_name = segments.next()?;
    let module = segments.next()?;
    let impl_owner = segments.next()?;
    let method = segments.next()?;
    if segments.next().is_some()
        || !matches!(crate_name, "cute_rs" | "cute-rs")
        || module != "pipeline"
        || impl_owner.is_empty()
    {
        return None;
    }
    Some(method)
}

fn classify_pipeline_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let method = pipeline_method_name_from_path(path)?;
    let destination_ty = destination.ty(body.locals()).ok();
    let destination_pipeline = destination_ty
        .as_ref()
        .and_then(|ty| pipeline::decode_tma_load_pipeline(ty).ok().flatten());
    let destination_state = destination_ty
        .as_ref()
        .and_then(|ty| pipeline::decode_pipeline_state(ty).ok().flatten());
    let receiver_pipeline = args.first().and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| pipeline::decode_tma_load_pipeline(&ty).ok().flatten())
    });
    let receiver_state = args.first().and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| pipeline::decode_pipeline_state_receiver(&ty).ok().flatten())
    });
    let lifecycle_state = args.get(1).and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| pipeline::decode_pipeline_state_receiver(&ty).ok().flatten())
    });

    match method {
        "from_raw_base" if args.len() == 1 && destination_pipeline.is_some() => {
            Some(CuteFn::TmaLoadPipelineMake)
        }
        "from_raw_parts" if args.len() == 2 && destination_pipeline.is_some() => {
            Some(CuteFn::TmaLoadPipelineFromRawPartsRejected)
        }
        "init" if args.len() == 2 && receiver_pipeline.is_some() => {
            Some(CuteFn::TmaLoadPipelineInit)
        }
        "new" if args.is_empty() && destination_state.is_some() => Some(CuteFn::PipelineStateNew),
        "slot" if args.len() == 1 && receiver_state.is_some() => Some(CuteFn::PipelineStateSlot),
        "advance" if args.len() == 1 && receiver_state.is_some() => {
            Some(CuteFn::PipelineStateAdvance)
        }
        "producer_acquire"
            if args.len() == 2
                && receiver_pipeline.is_some()
                && lifecycle_state.is_some_and(|state| {
                    state.role == dialect_cute::attributes::CutePipelineRoleAttr::Producer
                }) =>
        {
            Some(CuteFn::PipelineProducerAcquire)
        }
        "producer_expect_tx"
            if args.len() == 2
                && receiver_pipeline.is_some()
                && lifecycle_state.is_some_and(|state| {
                    state.role == dialect_cute::attributes::CutePipelineRoleAttr::Producer
                }) =>
        {
            Some(CuteFn::PipelineProducerExpectTx)
        }
        "consumer_wait"
            if args.len() == 2
                && receiver_pipeline.is_some()
                && lifecycle_state.is_some_and(|state| {
                    state.role == dialect_cute::attributes::CutePipelineRoleAttr::Consumer
                }) =>
        {
            Some(CuteFn::PipelineConsumerWait)
        }
        "consumer_release"
            if args.len() == 2
                && receiver_pipeline.is_some()
                && lifecycle_state.is_some_and(|state| {
                    state.role == dialect_cute::attributes::CutePipelineRoleAttr::Consumer
                }) =>
        {
            Some(CuteFn::PipelineConsumerRelease)
        }
        "producer_tail"
            if args.len() == 2
                && receiver_pipeline.is_some()
                && lifecycle_state.is_some_and(|state| {
                    state.role == dialect_cute::attributes::CutePipelineRoleAttr::Producer
                }) =>
        {
            Some(CuteFn::PipelineProducerTail)
        }
        _ => None,
    }
}

fn classify_scheduler_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let method = scheduler_method_name_from_path(path)?;
    let destination_ty = destination.ty(body.locals()).ok();
    let destination_scheduler = destination_ty
        .as_ref()
        .and_then(|ty| scheduler::decode_scheduler(ty).ok().flatten());
    let destination_tile = destination_ty
        .as_ref()
        .and_then(|ty| scheduler::decode_work_tile(ty).ok().flatten());
    let receiver_scheduler = args.first().and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| scheduler::decode_scheduler_receiver(&ty).ok().flatten())
    });
    let receiver_tile = args.first().and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| scheduler::decode_work_tile_receiver(&ty).ok().flatten())
    });

    match method {
        "new_1d" if args.is_empty() && destination_scheduler.is_some() => {
            Some(CuteFn::SchedulerNew1d)
        }
        "has_work" if args.len() == 1 && receiver_scheduler.is_some() => {
            Some(CuteFn::SchedulerHasWork)
        }
        "current_tile"
            if args.len() == 1
                && matches!((receiver_scheduler, destination_tile), (Some(source), Some(result)) if source == result) =>
        {
            Some(CuteFn::SchedulerCurrentTile)
        }
        "coordinates" if args.len() == 1 && receiver_tile.is_some() => {
            Some(CuteFn::WorkTileCoordinates)
        }
        "advance" if args.len() == 1 && receiver_scheduler.is_some() => {
            Some(CuteFn::SchedulerAdvance)
        }
        _ => None,
    }
}

fn classify_block_scaled_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    use block_scaled::BlockScaledStage;
    use dialect_cute::attributes::CuteTensorRoleAttr;

    let method = block_scaled_method_name_from_path(path)?;
    let destination = destination
        .ty(body.locals())
        .ok()
        .and_then(|ty| block_scaled::decode_block_scaled(&ty).ok().flatten());
    let receiver = args.first().and_then(|operand| {
        operand.ty(body.locals()).ok().and_then(|ty| {
            block_scaled::decode_block_scaled_receiver(&ty)
                .ok()
                .flatten()
        })
    });

    match method {
        "from_slices"
            if args.len() == 4
                && destination.is_some_and(|view| view.stage == BlockScaledStage::Full) =>
        {
            Some(CuteFn::BlockScaledMake)
        }
        "thread_row"
            if args.len() == 3
                && matches!((receiver, destination), (Some(source), Some(result))
                    if source.stage == BlockScaledStage::Full
                        && result.stage == BlockScaledStage::Row
                        && source.role == result.role) =>
        {
            Some(CuteFn::BlockScaledThreadRow)
        }
        "k_tile"
            if args.len() == 2
                && matches!((receiver, destination), (Some(source), Some(result))
                    if source.stage == BlockScaledStage::Row
                        && result.stage == BlockScaledStage::KTile64
                        && source.role == result.role) =>
        {
            Some(CuteFn::BlockScaledKTile)
        }
        "load"
            if args.len() == 1
                && matches!((receiver, destination), (Some(source), Some(result))
                    if source.stage == BlockScaledStage::KTile64
                        && result.stage == BlockScaledStage::Fragment64
                        && source.role == result.role) =>
        {
            Some(CuteFn::BlockScaledLoadK64)
        }
        "dot_accumulate"
            if args.len() == 3
                && receiver.is_some_and(|lhs| {
                    lhs.stage == BlockScaledStage::Fragment64 && lhs.role == CuteTensorRoleAttr::Mkl
                })
                && args.get(1).is_some_and(|operand| {
                    operand.ty(body.locals()).ok().is_some_and(|ty| {
                        block_scaled::decode_block_scaled_receiver(&ty)
                            .ok()
                            .flatten()
                            .is_some_and(|rhs| {
                                rhs.stage == BlockScaledStage::Fragment64
                                    && rhs.role == CuteTensorRoleAttr::Nkl
                            })
                    })
                }) =>
        {
            Some(CuteFn::BlockScaledDotK64)
        }
        _ => None,
    }
}

fn classify_tensor_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let method = tensor_method_name_from_path(path)?;

    let destination_ty = destination.ty(body.locals()).ok();
    let destination_view = destination_ty
        .as_ref()
        .and_then(|ty| tensor::decode_tensor_view(ty).ok().flatten());
    let receiver_view = args.first().and_then(|operand| {
        operand
            .ty(body.locals())
            .ok()
            .and_then(|ty| tensor::decode_tensor_view_receiver(&ty).ok().flatten())
    });

    use dialect_cute::attributes::{CuteTensorAccessAttr, CuteTensorLayoutAttr};
    match method {
        "from_slice"
            if destination_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadOnly
                    && view.layout == CuteTensorLayoutAttr::Contiguous1D
            }) =>
        {
            Some(CuteFn::MakeTensorRead)
        }
        "from_disjoint_slice"
            if destination_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadWrite
                    && view.layout == CuteTensorLayoutAttr::Contiguous1D
            }) =>
        {
            Some(CuteFn::MakeTensorWrite)
        }
        "zipped_divide" => match (receiver_view, destination_view) {
            (Some(source), Some(result))
                if source.layout == CuteTensorLayoutAttr::Contiguous1D
                    && matches!(result.layout, CuteTensorLayoutAttr::Zipped1D(_))
                    && source.access == result.access =>
            {
                Some(if source.access == CuteTensorAccessAttr::ReadOnly {
                    CuteFn::ZippedDivideRead
                } else {
                    CuteFn::ZippedDivideWrite
                })
            }
            _ => None,
        },
        "slice" => match (receiver_view, destination_view) {
            (Some(source), Some(result))
                if matches!(source.layout, CuteTensorLayoutAttr::Zipped1D(_))
                    && matches!(result.layout, CuteTensorLayoutAttr::Tile1D(_))
                    && source.access == result.access =>
            {
                Some(if source.access == CuteTensorAccessAttr::ReadOnly {
                    CuteFn::SliceRead
                } else {
                    CuteFn::SliceWrite
                })
            }
            _ => None,
        },
        "is_full"
            if receiver_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadOnly
                    && matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_))
            }) =>
        {
            Some(CuteFn::TensorTileIsFull)
        }
        "base"
            if receiver_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadOnly
                    && matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_))
            }) =>
        {
            Some(CuteFn::TensorTileBase)
        }
        "load"
            if receiver_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadOnly
                    && matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_))
            }) =>
        {
            Some(CuteFn::TensorLoadTile)
        }
        "store"
            if receiver_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadWrite
                    && matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_))
            }) =>
        {
            Some(CuteFn::TensorStoreTile)
        }
        "store_linear"
            if receiver_view.is_some_and(|view| {
                view.access == CuteTensorAccessAttr::ReadWrite
                    && matches!(view.layout, CuteTensorLayoutAttr::Tile1D(_))
            }) =>
        {
            Some(CuteFn::TensorStoreElementAbs)
        }
        _ => None,
    }
}

/// Validate the callee's generics against the spec's schema BEFORE any
/// positional extraction, so a cute-rs signature refactor fails at the
/// boundary with a message naming the drift instead of misreading deep in
/// the importer.
fn validate_schema(
    spec: &CuteFnSpec,
    substs: &rustc_public::ty::GenericArgs,
    loc: &Location,
) -> TranslationResult<()> {
    let found: Vec<&'static str> = substs
        .0
        .iter()
        .map(|arg| match arg {
            rustc_public::ty::GenericArgKind::Type(_) => "Type",
            rustc_public::ty::GenericArgKind::Const(_) => "Const",
            rustc_public::ty::GenericArgKind::Lifetime(_) => "Lifetime",
        })
        .collect();
    let expected: Vec<&'static str> = spec
        .schema
        .iter()
        .map(|k| match k {
            SubstKind::Type => "Type",
            SubstKind::Const => "Const",
            SubstKind::Lifetime => "Lifetime",
        })
        .collect();
    if found != expected {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{}: generics schema mismatch: the importer expects [{}] but the call \
                 carries [{}]; cute-rs and the importer are out of sync (was the \
                 function's signature refactored without updating CUTE_FNS?)",
                spec.canonical,
                expected.join(", "),
                found.join(", ")
            ))
        );
    }
    Ok(())
}

/// Dispatch hook called from `translate_call`.
///
/// Returns `None` when the callee is not a cute-rs function or is an explicit
/// source abstraction (the caller falls through to normal translation).
/// Returns `Some(result)` when this module owns the call, in which case the
/// shared path must not touch it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_translate_cute_call(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: &Location,
) -> Option<TranslationResult<Ptr<Operation>>> {
    // Structural callee extraction: only direct calls to a constant FnDef
    // can be cute calls (indirect dispatch of cute_rs items is rejected by
    // `requires_direct_dispatch` in the function-item path).
    let mir::Operand::Constant(const_op) = func else {
        return None;
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, substs)) =
        const_op.const_.ty().kind()
    else {
        return None;
    };

    let (cute_fn, spec) = match classify_cute_call(&fn_def) {
        CuteIdentity::NotCuteCrate => return None,
        CuteIdentity::SourceAbstraction => {
            let path = canonical_path(&fn_def, &fn_def.krate().name.to_string());
            let method = classify_tensor_method_call(&path, body, args, destination)
                .or_else(|| classify_block_scaled_method_call(&path, body, args, destination))
                .or_else(|| classify_smem_mma_method_call(&path, body, args, destination))
                .or_else(|| classify_epilogue_method_call(&path, body, args, destination))
                .or_else(|| classify_scheduler_method_call(&path, body, args, destination))
                .or_else(|| classify_pipeline_method_call(&path, body, args, destination))?;
            (method, None)
        }
        CuteIdentity::Known(spec) => (spec.func, Some(spec)),
        CuteIdentity::Unknown(path) => {
            return Some(Err(input_error_noloc!(TranslationErr::unsupported(
                format!(
                    "`{path}` is in the recognized cute-rs namespace but has no importer \
                     handler and is not in an approved source-abstraction module. Add an \
                     importer-owned primitive to CUTE_FNS, or deliberately classify its \
                     device-safe Rust module as a source abstraction."
                )
            ))));
        }
    };

    if let Some(spec) = spec
        && let Err(e) = validate_schema(spec, &substs, loc)
    {
        return Some(Err(e));
    }

    Some(match cute_fn {
        CuteFn::Sm100(kind) => sm100_emit::emit(
            ctx, body, func, args, kind, target, block_ptr, prev_op, value_map, block_map, loc.clone(),
        ),
        CuteFn::MakeTensorRead => tensor_emit::emit_make(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            dialect_cute::attributes::CuteTensorAccessAttr::ReadOnly,
        ),
        CuteFn::MakeTensorWrite => tensor_emit::emit_make(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            dialect_cute::attributes::CuteTensorAccessAttr::ReadWrite,
        ),
        CuteFn::ZippedDivideRead | CuteFn::ZippedDivideWrite => tensor_emit::emit_zipped_divide(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SliceRead | CuteFn::SliceWrite => tensor_emit::emit_slice(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TensorTileIsFull | CuteFn::TensorTileBase => tensor_emit::emit_query(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            matches!(cute_fn, CuteFn::TensorTileIsFull),
        ),
        CuteFn::TensorLoadTile => tensor_emit::emit_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TensorStoreTile => tensor_emit::emit_store(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TensorStoreElementAbs => tensor_emit::emit_store_element_abs(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::BlockScaledMake => block_scaled_emit::emit_make(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::BlockScaledThreadRow => block_scaled_emit::emit_row(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::BlockScaledKTile => block_scaled_emit::emit_k_tile(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::BlockScaledLoadK64 => block_scaled_emit::emit_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::BlockScaledDotK64 => block_scaled_emit::emit_dot(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SchedulerNew1d => scheduler_emit::emit_new_1d(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SchedulerHasWork => scheduler_emit::emit_has_work(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SchedulerCurrentTile => scheduler_emit::emit_current_tile(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::WorkTileCoordinates => scheduler_emit::emit_coordinates(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SchedulerAdvance => scheduler_emit::emit_advance(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaLoadPipelineMake => pipeline_emit::emit_make(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaLoadPipelineFromRawPartsRejected => input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "TmaLoadPipeline::from_raw_parts is not supported by the v0 semantic pipeline; use one contiguous [full][empty] ring with from_raw_base"
                    .to_owned()
            )
        ),
        CuteFn::TmaLoadPipelineInit => pipeline_emit::emit_init(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineStateNew => pipeline_emit::emit_state_new(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineStateSlot => pipeline_emit::emit_state_slot(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineStateAdvance => pipeline_emit::emit_state_advance(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineProducerAcquire => pipeline_emit::emit_producer_acquire(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineProducerExpectTx => pipeline_emit::emit_producer_expect_tx(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineConsumerWait => pipeline_emit::emit_consumer_wait(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineConsumerRelease => pipeline_emit::emit_consumer_release(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::PipelineProducerTail => pipeline_emit::emit_producer_tail(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::SharedTensorOverlay => smem_mma_emit::emit_overlay(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::TiledMmaSlice => smem_mma_emit::emit_mma_slice(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaFragmentFill => smem_mma_emit::emit_fragment_fill(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaLoadScales => smem_mma_emit::emit_load_scales(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaFragmentSliceK => smem_mma_emit::emit_slice_k(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaLoadA => smem_mma_emit::emit_load_a(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaPartitionB => smem_mma_emit::emit_partition_b(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::TiledGemm => smem_mma_emit::emit_tiled_gemm(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueOverlay => epilogue_emit::emit_overlay(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueWarpSlice => epilogue_emit::emit_warp_slice(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueStoreFragment => epilogue_emit::emit_store_fragment(
            ctx,
            body,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueSyncReusable => epilogue_emit::emit_sync(
            ctx,
            body,
            args,
            dialect_cute::attributes::CuteEpilogueSyncPhaseAttr::Reusable,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueSyncReadyForTma => epilogue_emit::emit_sync(
            ctx,
            body,
            args,
            dialect_cute::attributes::CuteEpilogueSyncPhaseAttr::ReadyForTma,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueHalf => epilogue_emit::emit_half(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaStoreAcquire => epilogue_emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue_emit::StorePipelineEffect::Acquire,
            target,
            block_ptr,
            prev_op,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaStoreCommit => epilogue_emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue_emit::StorePipelineEffect::Commit,
            target,
            block_ptr,
            prev_op,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaStoreTail => epilogue_emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue_emit::StorePipelineEffect::Tail,
            target,
            block_ptr,
            prev_op,
            block_map,
            loc.clone(),
        ),
        CuteFn::LoadTile => emit::emit_load_tile(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::StoreTile => emit::emit_store_tile(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::AssumeDiv => emit::emit_assume_div(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::LoadMatrixA => emit::emit_load_matrix(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            dialect_cute::attributes::CuteMatrixRoleAttr::A,
        ),
        CuteFn::LoadMatrixB => emit::emit_load_matrix(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            dialect_cute::attributes::CuteMatrixRoleAttr::B,
        ),
        CuteFn::CopyTma2d => emit::emit_copy_tma(
            ctx,
            body,
            func,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::CopyTmaS2g2d => emit::emit_copy_tma_s2g(
            ctx,
            body,
            func,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
        CuteFn::CopyG2S => emit::emit_copy_g2s(
            ctx,
            body,
            func,
            args,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_well_formed() {
        for (i, spec) in CUTE_FNS.iter().enumerate() {
            assert!(
                spec.canonical.starts_with("cute_rs::"),
                "canonical paths are definition paths in the cute_rs crate"
            );
            for other in &CUTE_FNS[i + 1..] {
                assert_ne!(spec.canonical, other.canonical, "duplicate canonical path");
            }
        }
    }
}
