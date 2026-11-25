/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared GEMM foundation registration, including registrations used by SM100.
//! This single owner preserves registration order across capability modules.

use crate::cute::{
    common::{
        ExternSharedTranslation, IdentityFirstOperandTranslation, IdentityTranslation,
        InsertFieldTranslation, StaticContractTypeTranslation, StaticMultiIdentityTranslation,
    },
    copy::tma::{TmaLoadTranslation, TmaStoreTranslation, TmaViewTypeTranslation},
    epilogue::sm120::{
        EpilogueHalfTranslation, EpilogueStoreTranslation, EpilogueSyncTranslation,
        NoOperandRenameTranslation, TmaStoreWaitTranslation,
    },
    mma::sm120::{
        FragmentFillTranslation, FragmentSliceKTranslation, MmaLoadATranslation,
        MmaLoadScalesTranslation, Nvfp4MmaAtomTranslation, TiledGemmTranslation,
    },
    pipeline::sm120::{
        LoadPipelineTypeTranslation, PipelineExpectTxTranslation, PipelineInitTranslation,
        PipelineReleaseTranslation, PipelineStateAdvanceTranslation, PipelineStateNewTranslation,
        PipelineStateSlotTranslation, PipelineTailTranslation, PipelineWaitTranslation,
    },
    scheduler::sm120::{
        SchedulerAdvanceTranslation, SchedulerHasWorkTranslation, SchedulerNewTranslation,
        WorkTileCoordinatesTranslation, WorkTileTypeTranslation,
    },
};
use crate::{DropAttribute, TranslationError, TranslationRegistry};
use dialect_cute::{
    attributes::{
        CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
        CuteMmaAtomAttr, CuteMmaCarrierKindAttr, CutePipelineStateAttr, CuteTileGridAttr,
        CuteTiledMmaPlanAttr, CuteTmaStorePipelineAttr,
    },
    epilogue_ops::{
        CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp,
        CuteEpilogueSyncOp, CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp,
        CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
    },
    gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp},
    pipeline_ops::{
        CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
        CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
        CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
        CuteTmaLoadPipelineMakeOp,
    },
    scheduler_ops::{
        CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp,
        CuteSchedulerNew1dOp, CuteWorkTileCoordinatesOp,
    },
    smem_mma_ops::{
        CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
        CuteMmaPartitionBOp, CuteNvfp4MmaAtomOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp,
        CuteTiledMmaSliceOp,
    },
    types::{
        CuteEpilogueTileType, CuteSmemTensorType, CuteTmaLoadPipelineType, CuteTmaViewType,
        CuteWorkTileType,
    },
};
use dialect_mir::ops::{MirExternSharedOp, MirInsertFieldOp};

/// Register the runtime foundation shared by TMA, the mainloop, and epilogue.
pub(crate) fn register_cute_gemm_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_type::<CuteWorkTileType>(WorkTileTypeTranslation)?;
    registry.register_type::<CuteTmaLoadPipelineType>(LoadPipelineTypeTranslation)?;
    registry.register_type::<CuteTmaViewType>(TmaViewTypeTranslation)?;
    // These types occur only inside source TypeAttr values. Their recipes read
    // the verified source objects directly, so the translated attribute value
    // is deliberately an empty sentinel and is removed before emission.
    registry.register_type::<CuteSmemTensorType>(StaticContractTypeTranslation)?;
    registry.register_type::<CuteEpilogueTileType>(StaticContractTypeTranslation)?;

    registry.register_attribute::<CuteTileGridAttr>(DropAttribute)?;
    registry.register_attribute::<CutePipelineStateAttr>(DropAttribute)?;
    registry.register_attribute::<CuteMmaAtomAttr>(DropAttribute)?;
    registry.register_attribute::<CuteTiledMmaPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteMmaCarrierKindAttr>(DropAttribute)?;
    registry.register_attribute::<CuteTmaStorePipelineAttr>(DropAttribute)?;
    registry.register_attribute::<CuteCountedCtaBarrierAttr>(DropAttribute)?;
    registry.register_attribute::<CuteEpilogueSyncPhaseAttr>(DropAttribute)?;
    registry.register_attribute::<CuteEpilogueHalfAttr>(DropAttribute)?;

    registry.register_operation::<MirInsertFieldOp>(InsertFieldTranslation)?;
    registry.register_operation::<MirExternSharedOp>(ExternSharedTranslation)?;

    registry.register_operation::<CuteSchedulerNew1dOp>(SchedulerNewTranslation)?;
    registry.register_operation::<CuteSchedulerHasWorkOp>(SchedulerHasWorkTranslation)?;
    registry.register_operation::<CuteSchedulerCurrentOp>(IdentityTranslation(
        "cute.scheduler_current",
    ))?;
    registry.register_operation::<CuteWorkTileCoordinatesOp>(WorkTileCoordinatesTranslation)?;
    registry.register_operation::<CuteSchedulerAdvanceOp>(SchedulerAdvanceTranslation)?;

    registry.register_operation::<CuteTmaLoadPipelineMakeOp>(IdentityFirstOperandTranslation(
        "cute.tma_load_pipeline_make",
    ))?;
    registry.register_operation::<CuteTmaLoadPipelineInitOp>(PipelineInitTranslation)?;
    registry.register_operation::<CutePipelineStateNewOp>(PipelineStateNewTranslation)?;
    registry.register_operation::<CutePipelineStateSlotOp>(PipelineStateSlotTranslation)?;
    registry.register_operation::<CutePipelineStateAdvanceOp>(PipelineStateAdvanceTranslation)?;
    registry.register_operation::<CutePipelineProducerAcquireOp>(PipelineWaitTranslation {
        empty_ring: true,
        name: "cute.pipeline_producer_acquire",
    })?;
    registry.register_operation::<CutePipelineConsumerWaitOp>(PipelineWaitTranslation {
        empty_ring: false,
        name: "cute.pipeline_consumer_wait",
    })?;
    registry.register_operation::<CutePipelineProducerExpectTxOp>(PipelineExpectTxTranslation)?;
    registry.register_operation::<CutePipelineConsumerReleaseOp>(PipelineReleaseTranslation)?;
    registry.register_operation::<CutePipelineProducerTailOp>(PipelineTailTranslation)?;

    registry.register_operation::<CuteTmaGmemViewOp>(IdentityFirstOperandTranslation(
        "cute.tma_gmem_view",
    ))?;
    registry.register_operation::<CuteTmaSmemViewOp>(IdentityFirstOperandTranslation(
        "cute.tma_smem_view",
    ))?;
    registry.register_operation::<CuteTmaCopy2dOp>(TmaLoadTranslation)?;
    registry.register_operation::<CuteTmaStore2dSemanticOp>(TmaStoreTranslation)?;

    registry.register_operation::<CuteTmaStoreAcquireOp>(TmaStoreWaitTranslation {
        name: "cute.tma_store_acquire",
    })?;
    registry.register_operation::<CuteTmaStoreCommitOp>(NoOperandRenameTranslation(
        "nvvm.cp.async.bulk.commit.group",
    ))?;
    registry.register_operation::<CuteTmaStoreTailOp>(TmaStoreWaitTranslation {
        name: "cute.tma_store_tail",
    })?;
    // Pointer/coordinate-only epilogue operations are included here. The
    // accumulator-to-stmatrix leaf is registered by the MMA rung below.
    registry.register_operation::<CuteEpilogueSmemOverlayOp>(StaticMultiIdentityTranslation {
        name: "cute.epilogue_smem_overlay",
        attributes: &["epilogue_overlay_tile"],
    })?;
    registry.register_operation::<CuteEpilogueWarpSliceOp>(StaticMultiIdentityTranslation {
        name: "cute.epilogue_warp_slice",
        attributes: &["epilogue_warp_slice_tile"],
    })?;
    registry.register_operation::<CuteEpilogueHalfOp>(EpilogueHalfTranslation)?;
    registry.register_operation::<CuteEpilogueSyncOp>(EpilogueSyncTranslation)?;

    register_mma_and_epilogue_mappings(registry)
}

/// Register SM120 MMA carrier operations and accumulator-to-shared epilogue stores.
fn register_mma_and_epilogue_mappings(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_operation::<CuteSmemTensorOverlayOp>(StaticMultiIdentityTranslation {
        name: "cute.smem_tensor_overlay",
        attributes: &["smem_overlay_view"],
    })?;
    registry
        .register_operation::<CuteTiledMmaSliceOp>(IdentityTranslation("cute.tiled_mma_slice"))?;
    registry.register_operation::<CuteFragmentFillOp>(FragmentFillTranslation)?;
    registry.register_operation::<CuteMmaLoadScalesOp>(MmaLoadScalesTranslation)?;
    registry.register_operation::<CuteFragmentSliceKOp>(FragmentSliceKTranslation)?;
    registry.register_operation::<CuteMmaLoadAOp>(MmaLoadATranslation)?;
    registry.register_operation::<CuteMmaPartitionBOp>(StaticMultiIdentityTranslation {
        name: "cute.mma_partition_b",
        attributes: &["mma_partition_b_view"],
    })?;
    registry.register_operation::<CuteTiledGemmOp>(TiledGemmTranslation)?;
    registry.register_operation::<CuteNvfp4MmaAtomOp>(Nvfp4MmaAtomTranslation)?;
    registry.register_operation::<CuteEpilogueStoreFragmentOp>(EpilogueStoreTranslation)?;
    Ok(())
}
