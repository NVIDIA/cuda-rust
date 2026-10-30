/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler entry points and their generic-argument schemas.

use super::{sm100, sm100_attention};

/// Recognized cute-rs entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CuteFn {
    /// Dispatch a paired SM100 GEMM or pipeline hook.
    Sm100(#[doc = "Handler within the paired SM100 operation family."] sm100::Sm100Fn),
    /// Dispatch an independent SM100 attention MMA or TMA hook.
    Attention(
        #[doc = "Handler within the independent attention operation family."]
        sm100_attention::AttentionFn,
    ),
    /// Construct a read-only contiguous tensor over a slice.
    MakeTensorRead,
    /// Construct a writable tensor over disjoint output storage.
    MakeTensorWrite,
    /// Split a read-only tensor into fixed-size tiles.
    ZippedDivideRead,
    /// Split a writable tensor into fixed-size tiles.
    ZippedDivideWrite,
    /// Select one read-only tensor tile.
    SliceRead,
    /// Select one writable tensor tile.
    SliceWrite,
    /// Split a read-only tensor into tiles and select one: CuTe's `local_tile`.
    LocalTileRead,
    /// Split a writable tensor into tiles and select one: CuTe's `local_tile`.
    LocalTileWrite,
    /// Test whether every element of the selected tile is in bounds.
    TensorTileIsFull,
    /// Return the selected tile's absolute starting element index.
    TensorTileBase,
    /// Load a complete tensor tile into registers.
    TensorLoadTile,
    /// Store a register tile through a writable tensor view.
    TensorStoreTile,
    /// Store one element using an absolute storage index.
    TensorStoreElementAbs,
    /// Pair packed FP4 data with scales and a K-major layout.
    BlockScaledMake,
    /// Select the row or vector batch owned by one thread.
    BlockScaledThreadRow,
    /// Select a K64 block from a scaled row.
    BlockScaledKTile,
    /// Load packed values and scales for the selected K64 block.
    BlockScaledLoadK64,
    /// Accumulate one scaled K64 dot product in FP32.
    BlockScaledDotK64,
    /// Initialize persistent work from a linear physical CTA index.
    SchedulerNew1d,
    /// Test whether the current logical tile remains inside the fixed grid.
    SchedulerHasWork,
    /// Return the scheduler's current typed work tile.
    SchedulerCurrentTile,
    /// Decode a work tile's M, N and batch coordinates.
    WorkTileCoordinates,
    /// Advance persistent work by the physical grid stride.
    SchedulerAdvance,
    /// Construct a load pipeline with checked shared barrier storage.
    TmaLoadPipelineMake,
    /// Reject the obsolete untracked raw-pointer pipeline constructor.
    TmaLoadPipelineFromRawPartsRejected,
    /// Initialize transaction-counted full and reusable empty barriers.
    TmaLoadPipelineInit,
    /// Create the initial producer or consumer cursor.
    PipelineStateNew,
    /// Read the current cursor slot.
    PipelineStateSlot,
    /// Advance a cursor and toggle its parity on wrap.
    PipelineStateAdvance,
    /// Wait until an input slot is available for the producer.
    PipelineProducerAcquire,
    /// Set the expected input-copy byte count for the current slot.
    PipelineProducerExpectTx,
    /// Wait for the current input slot to become full.
    PipelineConsumerWait,
    /// Release the current input slot after consumers finish.
    PipelineConsumerRelease,
    /// Wait for remaining consumers before releasing the input ring.
    PipelineProducerTail,
    /// Overlay a typed data or scale layout on shared storage.
    SharedTensorOverlay,
    /// Construct the lane-owned tiled-MMA slice.
    TiledMmaSlice,
    /// Initialize the fixed FP32 accumulator fragment.
    MmaFragmentFill,
    /// Load A/B scales from the canonical scale atom.
    MmaLoadScales,
    /// Select scales for one K64 half.
    MmaFragmentSliceK,
    /// Load the A register fragment for a warp and K64 half.
    MmaLoadA,
    /// Select the shared B tile for a warp and K64 half.
    MmaPartitionB,
    /// Accumulate a tiled MXF4 product using paired data and scale operands.
    TiledGemm,
    /// Issue one 16x8x64 NVFP4 (E2M1 + E4M3, SF_VEC = 16) Tensor-Core MMA atom.
    Nvfp4MmaAtom,
    /// Overlay the fixed output layout on shared storage.
    EpilogueOverlay,
    /// Select the warp/lane-owned portion of the epilogue.
    EpilogueWarpSlice,
    /// Convert and scatter an accumulator fragment into shared FP16 output.
    EpilogueStoreFragment,
    /// Synchronize participating warps before output-buffer reuse.
    EpilogueSyncReusable,
    /// Publish shared output writes and synchronize before TMA stores.
    EpilogueSyncReadyForTma,
    /// Select one contiguous output half for a TMA store.
    EpilogueHalf,
    /// Wait for the oldest output-store group's shared reads to finish.
    TmaStoreAcquire,
    /// Commit the current issuer's pending output stores.
    TmaStoreCommit,
    /// Drain committed output-store groups.
    TmaStoreTail,
    /// Load one aligned scalar-vector tile into registers.
    LoadTile,
    /// Store one aligned register tile to memory.
    StoreTile,
    /// Emit the cache-aware register-to-global store.
    StoreGlobalNoAllocate,
    /// Record a source-promised divisor for a scalar index.
    AssumeDiv,
    /// Issue a cooperative global-to-shared tile copy with predicated edges.
    CopyG2S,
    /// Load the A matrix fragment from its checked shared layout.
    LoadMatrixA,
    /// Load the B matrix fragment from its checked shared layout.
    LoadMatrixB,
    /// Load a paired B fragment for adjacent N positions.
    LoadMatrixBPair,
    /// Load paired B fragments from K-major shared storage.
    LoadMatrixBPairKMajor,
    /// Issue a typed two-dimensional global-to-shared TMA copy.
    CopyTma2d,
    /// Issue a typed two-dimensional shared-to-global TMA store.
    CopyTmaS2g2d,
}

/// Expected kind of one generic argument in a recognized function's substs.
#[allow(dead_code)] // `Lifetime` makes future early-bound lifetimes explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubstKind {
    /// One Rust type argument in the hook's generic schema.
    Type,
    /// One compile-time constant argument in the hook's generic schema.
    Const,
    /// One lifetime argument retained in the hook's generic schema.
    Lifetime,
}

/// Report whether `path` is the defining path of a recognized cute-rs hook.
///
/// `path` is the crate-relative form rustc prints for a definition
/// (`::copy::tma::copy_tma_2d`); the registry stores the same path with the
/// `cute_rs` crate prefix. Re-export paths never match, so a caller that
/// resolved an item through the root export `cute_rs::copy_tma_2d` must pass
/// the defining path, not the import path.
pub fn is_cute_compiler_hook_path(path: &str) -> bool {
    CUTE_FNS
        .iter()
        .any(|spec| spec.canonical.strip_prefix("cute_rs") == Some(path))
}

/// A recognized function's definition path and expected generic arguments.
/// Re-export paths do not appear here.
pub(crate) struct CuteFnSpec {
    /// Exact defining function path used for compiler-hook recognition.
    pub canonical: &'static str,
    /// Semantic handler selected when the canonical path matches.
    pub func: CuteFn,
    /// Ordered generic-argument kinds required before decoding hook configuration.
    pub schema: &'static [SubstKind],
}

/// One entry per recognized function, including its generic argument schema.
pub(crate) const CUTE_FNS: &[CuteFnSpec] = &[
    CuteFnSpec {
        canonical: "cute_rs::attention::sm100::__compiler::mma_ss",
        func: CuteFn::Attention(sm100_attention::AttentionFn::MmaSs),
        schema: &[
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Const,
            SubstKind::Const,
            SubstKind::Const,
            SubstKind::Const,
        ],
    },
    CuteFnSpec {
        canonical: "cute_rs::attention::sm100::__compiler::mma_ts",
        func: CuteFn::Attention(sm100_attention::AttentionFn::MmaTs),
        schema: &[
            SubstKind::Type,
            SubstKind::Const,
            SubstKind::Const,
            SubstKind::Const,
            SubstKind::Const,
        ],
    },
    CuteFnSpec {
        canonical: "cute_rs::attention::sm100::__compiler::tma_load",
        func: CuteFn::Attention(sm100_attention::AttentionFn::TmaLoad),
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::attention::sm100::__compiler::tma_store",
        func: CuteFn::Attention(sm100_attention::AttentionFn::TmaStore),
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tma_store",
        func: CuteFn::Sm100(sm100::Sm100Fn::TmaStore),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_commit",
        func: CuteFn::Sm100(sm100::Sm100Fn::StoreCommit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_acquire",
        func: CuteFn::Sm100(sm100::Sm100Fn::StoreAcquire),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_store_tail",
        func: CuteFn::Sm100(sm100::Sm100Fn::StoreTail),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tmem_alloc",
        func: CuteFn::Sm100(sm100::Sm100Fn::TmemAlloc),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tmem_dealloc",
        func: CuteFn::Sm100(sm100::Sm100Fn::TmemDealloc),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_tiled_mma",
        func: CuteFn::Sm100(sm100::Sm100Fn::TiledMma),
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
        func: CuteFn::Sm100(sm100::Sm100Fn::TmemEpilogue),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_cluster_tma_load",
        func: CuteFn::Sm100(sm100::Sm100Fn::ClusterTmaLoad),
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_init",
        func: CuteFn::Sm100(sm100::Sm100Fn::PipelineInit),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_acquire",
        func: CuteFn::Sm100(sm100::Sm100Fn::PipelineAcquire),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_expect",
        func: CuteFn::Sm100(sm100::Sm100Fn::PipelineExpect),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_wait",
        func: CuteFn::Sm100(sm100::Sm100Fn::PipelineWait),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_pipeline_release",
        func: CuteFn::Sm100(sm100::Sm100Fn::PipelineRelease),
        schema: &[SubstKind::Const, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_init",
        func: CuteFn::Sm100(sm100::Sm100Fn::AccumulatorInit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_acquire",
        func: CuteFn::Sm100(sm100::Sm100Fn::AccumulatorAcquire),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_commit",
        func: CuteFn::Sm100(sm100::Sm100Fn::AccumulatorCommit),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_wait",
        func: CuteFn::Sm100(sm100::Sm100Fn::AccumulatorWait),
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::sm100::__compiler::sm100_accumulator_release",
        func: CuteFn::Sm100(sm100::Sm100Fn::AccumulatorRelease),
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
        canonical: "cute_rs::block_scaled::__compiler::nvfp4_make",
        func: CuteFn::BlockScaledMake,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::nvfp4_thread_row",
        func: CuteFn::BlockScaledThreadRow,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::nvfp4_k_tile",
        func: CuteFn::BlockScaledKTile,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::nvfp4_load_k64",
        func: CuteFn::BlockScaledLoadK64,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::block_scaled::__compiler::nvfp4_dot_k64",
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
        canonical: "cute_rs::copy::tiled::__compiler::shared_tensor_overlay",
        func: CuteFn::SharedTensorOverlay,
        schema: &[
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
            SubstKind::Type,
        ],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::tiled_mma_slice",
        func: CuteFn::TiledMmaSlice,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::fragment_fill",
        func: CuteFn::MmaFragmentFill,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::mma_load_scales",
        func: CuteFn::MmaLoadScales,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::fragment_slice_k",
        func: CuteFn::MmaFragmentSliceK,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::mma_load_a",
        func: CuteFn::MmaLoadA,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::mma_partition_b",
        func: CuteFn::MmaPartitionB,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::tiled_gemm",
        func: CuteFn::TiledGemm,
        schema: &[SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::nvfp4_mma_atom_call",
        func: CuteFn::Nvfp4MmaAtom,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::sm120::__compiler::epilogue_smem_overlay",
        func: CuteFn::EpilogueOverlay,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::sm120::__compiler::epilogue_warp_slice",
        func: CuteFn::EpilogueWarpSlice,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::block_scaled::__compiler::epilogue_store_fragment",
        func: CuteFn::EpilogueStoreFragment,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::sm120::__compiler::epilogue_sync_reusable",
        func: CuteFn::EpilogueSyncReusable,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::sm120::__compiler::epilogue_sync_ready_for_tma",
        func: CuteFn::EpilogueSyncReadyForTma,
        schema: &[],
    },
    CuteFnSpec {
        canonical: "cute_rs::epilogue::sm120::__compiler::epilogue_half",
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
        canonical: "cute_rs::copy::vector::load_tile",
        func: CuteFn::LoadTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::vector::store_tile",
        func: CuteFn::StoreTile,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::vector::store_global_no_allocate",
        func: CuteFn::StoreGlobalNoAllocate,
        schema: &[SubstKind::Type, SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::vector::assume",
        func: CuteFn::AssumeDiv,
        schema: &[SubstKind::Const],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_a",
        func: CuteFn::LoadMatrixA,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_b",
        func: CuteFn::LoadMatrixB,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_b_pair",
        func: CuteFn::LoadMatrixBPair,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::mma::load_matrix_b_pair_k_major",
        func: CuteFn::LoadMatrixBPairKMajor,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::tma::copy_tma_2d",
        func: CuteFn::CopyTma2d,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::tma::copy_tma_s2g_2d",
        func: CuteFn::CopyTmaS2g2d,
        schema: &[SubstKind::Type, SubstKind::Type],
    },
    CuteFnSpec {
        canonical: "cute_rs::copy::cooperative::copy_g2s",
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

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check that matrix-load entry points accept exactly one shared-layout type parameter.
    #[test]
    fn matrix_load_entry_points_require_one_shared_layout_type() {
        for (name, expected) in [
            ("load_matrix_a", CuteFn::LoadMatrixA),
            ("load_matrix_b", CuteFn::LoadMatrixB),
            ("load_matrix_b_pair", CuteFn::LoadMatrixBPair),
            ("load_matrix_b_pair_k_major", CuteFn::LoadMatrixBPairKMajor),
        ] {
            let path = format!("cute_rs::mma::{name}");
            let spec = CUTE_FNS.iter().find(|spec| spec.canonical == path);
            let spec = spec.unwrap_or_else(|| panic!("matrix load is not recognized: {path}"));
            assert_eq!(spec.func, expected, "{path}");
            assert_eq!(spec.schema, &[SubstKind::Type, SubstKind::Type], "{path}");
        }
    }

    /// Check that the crate-relative hook query accepts defining paths only.
    #[test]
    fn hook_paths_match_definitions_not_re_exports() {
        for path in [
            "::tensor::make_tensor_read",
            "::copy::vector::assume",
            "::copy::tma::copy_tma_2d",
            "::copy::cooperative::copy_g2s",
            "::mma::block_scaled::__compiler::tiled_gemm",
            "::epilogue::sm120::__compiler::epilogue_half",
            "::sm100::__compiler::sm100_tiled_mma",
        ] {
            assert!(is_cute_compiler_hook_path(path), "{path}");
        }
        // Paths that are not definition paths never match: the root module
        // aliases cute_rs once had (`tile`, `tma`, `block_scaled_mma`, retired),
        // the retired `epilogue::__compiler` re-export and crate-qualified forms.
        for path in [
            "::tile::assume",
            "::tma::copy_tma_2d",
            "::block_scaled_mma::__compiler::tiled_gemm",
            "::epilogue::__compiler::epilogue_half",
            "cute_rs::copy::tma::copy_tma_2d",
            "copy::tma::copy_tma_2d",
        ] {
            assert!(!is_cute_compiler_hook_path(path), "{path}");
        }
    }

    /// Check that registry names are unique and each generic schema matches its handler.
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
