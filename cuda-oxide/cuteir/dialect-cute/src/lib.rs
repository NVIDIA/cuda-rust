/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe dialect definition.
//!
//! Represents CuTe layouts and operations in Pliron, alongside MIR control
//! flow and scalar operations.
//!
//! Layouts are stored as typed attributes backed by `cute-layout` (never as
//! strings). Layout operations compute their result types when the IR is
//! built. The verifier checks how tensors, copies, pipelines, and MMA
//! operations fit together before a backend lowers them.
//!
//! The implemented surface is deliberately narrow: elementwise tensor tiles,
//! one block-scaled GEMV fragment flow, raw-carrier TMA views for the four GEMM
//! stage copies, shared-tensor/tiled-MMA compute, and SM100 two-CTA tensor
//! memory with collector-aware MMA, cluster TMA, and asynchronous pipelines.
//! The selected backend lowers these operations to GPU code.

pub mod attributes;
pub mod epilogue_ops;
pub mod gemm_tma_ops;
pub mod gemv_ops;
pub mod ops;
pub mod pipeline_ops;
pub mod scheduler_ops;
pub mod sm100_ops;
pub mod smem_mma_ops;
pub mod tensor_ops;
pub mod types;
pub mod verify;

/// Layout math shared by the importer, verifier, and backends.
pub use cute_layout as layout;

use pliron::attribute::Attribute;
use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};
use pliron::op::Op;
use pliron::r#type::Type;

pub const CUTE_DIALECT_NAME: &str = "cute";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(CUTE_DIALECT_NAME).expect("valid dialect name"),
    );

    sm100_ops::register(ctx);

    // The #[pliron_op]/#[pliron_attr] macros auto-register via Context::default;
    // registering them again is safe and matches the other dialects.
    ops::CuteCopyOp::register(ctx);
    ops::CuteAssumeDivOp::register(ctx);
    ops::CuteCopyG2SOp::register(ctx);
    ops::CuteLdmatrixOp::register(ctx);
    ops::CuteTmaLoad2dOp::register(ctx);
    ops::CuteTmaStore2dOp::register(ctx);
    tensor_ops::CuteTensorMakeOp::register(ctx);
    tensor_ops::CuteTensorZippedDivideOp::register(ctx);
    tensor_ops::CuteTensorSliceOp::register(ctx);
    tensor_ops::CuteTensorIsFullOp::register(ctx);
    tensor_ops::CuteTensorBaseOp::register(ctx);
    tensor_ops::CuteTensorLoadIntoOp::register(ctx);
    tensor_ops::CuteTensorStoreFromOp::register(ctx);
    tensor_ops::CuteTensorStoreElementAbsOp::register(ctx);
    gemv_ops::CuteTensorMake2DOp::register(ctx);
    gemv_ops::CuteScaledViewMakeOp::register(ctx);
    gemv_ops::CuteScaledViewRowOp::register(ctx);
    gemv_ops::CuteScaledViewKTileOp::register(ctx);
    gemv_ops::CuteScaledViewLoadOp::register(ctx);
    gemv_ops::CuteDotOp::register(ctx);
    gemm_tma_ops::CuteTmaGmemViewOp::register(ctx);
    gemm_tma_ops::CuteTmaSmemViewOp::register(ctx);
    gemm_tma_ops::CuteTmaCopy2dOp::register(ctx);
    epilogue_ops::CuteEpilogueSmemOverlayOp::register(ctx);
    epilogue_ops::CuteEpilogueWarpSliceOp::register(ctx);
    epilogue_ops::CuteEpilogueStoreFragmentOp::register(ctx);
    epilogue_ops::CuteEpilogueSyncOp::register(ctx);
    epilogue_ops::CuteEpilogueHalfOp::register(ctx);
    epilogue_ops::CuteTmaStoreAcquireOp::register(ctx);
    epilogue_ops::CuteTmaStoreCommitOp::register(ctx);
    epilogue_ops::CuteTmaStoreTailOp::register(ctx);
    epilogue_ops::CuteTmaStore2dSemanticOp::register(ctx);
    scheduler_ops::CuteSchedulerNew1dOp::register(ctx);
    scheduler_ops::CuteSchedulerHasWorkOp::register(ctx);
    scheduler_ops::CuteSchedulerCurrentOp::register(ctx);
    scheduler_ops::CuteWorkTileCoordinatesOp::register(ctx);
    scheduler_ops::CuteSchedulerAdvanceOp::register(ctx);
    pipeline_ops::CuteTmaLoadPipelineMakeOp::register(ctx);
    pipeline_ops::CuteTmaLoadPipelineInitOp::register(ctx);
    pipeline_ops::CutePipelineStateNewOp::register(ctx);
    pipeline_ops::CutePipelineStateSlotOp::register(ctx);
    pipeline_ops::CutePipelineStateAdvanceOp::register(ctx);
    pipeline_ops::CutePipelineProducerAcquireOp::register(ctx);
    pipeline_ops::CutePipelineProducerExpectTxOp::register(ctx);
    pipeline_ops::CutePipelineConsumerWaitOp::register(ctx);
    pipeline_ops::CutePipelineConsumerReleaseOp::register(ctx);
    pipeline_ops::CutePipelineProducerTailOp::register(ctx);
    smem_mma_ops::CuteSmemTensorOverlayOp::register(ctx);
    smem_mma_ops::CuteTiledMmaSliceOp::register(ctx);
    smem_mma_ops::CuteFragmentFillOp::register(ctx);
    smem_mma_ops::CuteMmaLoadScalesOp::register(ctx);
    smem_mma_ops::CuteFragmentSliceKOp::register(ctx);
    smem_mma_ops::CuteMmaLoadAOp::register(ctx);
    smem_mma_ops::CuteMmaPartitionBOp::register(ctx);
    smem_mma_ops::CuteTiledGemmOp::register(ctx);
    attributes::CuteLayoutAttr::register(ctx);
    attributes::CuteDivisibilityAttr::register(ctx);
    attributes::CuteCopyAtomAttr::register(ctx);
    attributes::CuteComposedLayoutAttr::register(ctx);
    attributes::CuteTensorAddressSpaceAttr::register(ctx);
    attributes::CuteTensorAccessAttr::register(ctx);
    attributes::CuteTensorFormatAttr::register(ctx);
    attributes::CuteTensorRoleAttr::register(ctx);
    attributes::CuteTensorLayoutAttr::register(ctx);
    attributes::CuteScaledLayoutAttr::register(ctx);
    attributes::CuteAlignmentAttr::register(ctx);
    attributes::CuteMatrixRoleAttr::register(ctx);
    attributes::CuteTileGridAttr::register(ctx);
    attributes::CutePipelineRoleAttr::register(ctx);
    attributes::CutePipelineStateAttr::register(ctx);
    attributes::CuteMmaAccumulatorAttr::register(ctx);
    attributes::CuteMmaCarrierKindAttr::register(ctx);
    attributes::CuteMmaAtomAttr::register(ctx);
    attributes::CuteTiledMmaPlanAttr::register(ctx);
    attributes::CuteTmaStorePipelineAttr::register(ctx);
    attributes::CuteCountedCtaBarrierAttr::register(ctx);
    attributes::CuteEpilogueSyncPhaseAttr::register(ctx);
    attributes::CuteEpilogueHalfAttr::register(ctx);
    attributes::CuteEpiloguePlanAttr::register(ctx);
    types::CuteTensorViewType::register(ctx);
    types::CuteTmaViewType::register(ctx);
    types::CuteScaledViewType::register(ctx);
    types::CuteFragmentType::register(ctx);
    types::CuteWorkTileType::register(ctx);
    types::CuteTmaLoadPipelineType::register(ctx);
    types::CuteSmemTensorType::register(ctx);
    types::CuteEpilogueTileType::register(ctx);
}
