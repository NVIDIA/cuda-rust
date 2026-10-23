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
//!
//! Operation families live under [`ops`]. The root `*_ops` modules are
//! compatibility aliases to the same modules; [`attributes`], [`types`],
//! and [`verify`] retain their separate roles.

pub mod attributes;
pub mod ops;
pub mod types;
pub mod verify;

/// Compatibility alias for [`ops::copy`].
pub use ops::copy as copy_ops;
/// Compatibility alias for [`ops::epilogue`].
pub use ops::epilogue as epilogue_ops;
/// Compatibility alias for [`ops::gemm_tma`].
pub use ops::gemm_tma as gemm_tma_ops;
/// Compatibility alias for [`ops::gemv`].
pub use ops::gemv as gemv_ops;
/// Compatibility alias for [`ops::mma`].
pub use ops::mma as smem_mma_ops;
/// Compatibility alias for [`ops::pipeline`].
pub use ops::pipeline as pipeline_ops;
/// Compatibility alias for [`ops::scheduler`].
pub use ops::scheduler as scheduler_ops;
/// Compatibility alias for [`ops::sm100`].
pub use ops::sm100 as sm100_ops;
/// Compatibility alias for [`ops::sm100_attention`].
pub use ops::sm100_attention as sm100_attention_ops;
/// Compatibility alias for [`ops::tensor`].
pub use ops::tensor as tensor_ops;

/// Layout math shared by the importer, verifier, and backends.
pub use cute_layout as layout;

use pliron::attribute::Attribute;
use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};
use pliron::op::Op;
use pliron::r#type::Type;

/// Namespace used for all CuTe operation and attribute identifiers.
pub const CUTE_DIALECT_NAME: &str = "cute";

/// Register the CuTe dialect and its supported operation, attribute, and type families.
pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(CUTE_DIALECT_NAME).expect("valid dialect name"),
    );

    ops::copy::CuteStoreGlobalNoAllocateOp::register(ctx);
    ops::copy::CuteGlobalStoreAttr::register(ctx);
    ops::sm100::register(ctx);
    ops::sm100_attention::register(ctx);

    // The #[pliron_op]/#[pliron_attr] macros auto-register via Context::default;
    // registering them again is safe and matches the other dialects.
    ops::CuteCopyOp::register(ctx);
    ops::CuteAssumeDivOp::register(ctx);
    ops::CuteCopyG2SOp::register(ctx);
    ops::CuteLdmatrixOp::register(ctx);
    ops::CuteTmaLoad2dOp::register(ctx);
    ops::CuteTmaStore2dOp::register(ctx);
    ops::tensor::CuteTensorMakeOp::register(ctx);
    ops::tensor::CuteTensorZippedDivideOp::register(ctx);
    ops::tensor::CuteTensorSliceOp::register(ctx);
    ops::tensor::CuteTensorIsFullOp::register(ctx);
    ops::tensor::CuteTensorBaseOp::register(ctx);
    ops::tensor::CuteTensorLoadIntoOp::register(ctx);
    ops::tensor::CuteTensorStoreFromOp::register(ctx);
    ops::tensor::CuteTensorStoreElementAbsOp::register(ctx);
    ops::gemv::CuteTensorMake2DOp::register(ctx);
    ops::gemv::CuteScaledViewMakeOp::register(ctx);
    ops::gemv::CuteScaledViewRowOp::register(ctx);
    ops::gemv::CuteScaledViewKTileOp::register(ctx);
    ops::gemv::CuteScaledViewLoadOp::register(ctx);
    ops::gemv::CuteDotOp::register(ctx);
    ops::gemm_tma::CuteTmaGmemViewOp::register(ctx);
    ops::gemm_tma::CuteTmaSmemViewOp::register(ctx);
    ops::gemm_tma::CuteTmaCopy2dOp::register(ctx);
    ops::epilogue::CuteEpilogueSmemOverlayOp::register(ctx);
    ops::epilogue::CuteEpilogueWarpSliceOp::register(ctx);
    ops::epilogue::CuteEpilogueStoreFragmentOp::register(ctx);
    ops::epilogue::CuteEpilogueSyncOp::register(ctx);
    ops::epilogue::CuteEpilogueHalfOp::register(ctx);
    ops::epilogue::CuteTmaStoreAcquireOp::register(ctx);
    ops::epilogue::CuteTmaStoreCommitOp::register(ctx);
    ops::epilogue::CuteTmaStoreTailOp::register(ctx);
    ops::epilogue::CuteTmaStore2dSemanticOp::register(ctx);
    ops::scheduler::CuteSchedulerNew1dOp::register(ctx);
    ops::scheduler::CuteSchedulerHasWorkOp::register(ctx);
    ops::scheduler::CuteSchedulerCurrentOp::register(ctx);
    ops::scheduler::CuteWorkTileCoordinatesOp::register(ctx);
    ops::scheduler::CuteSchedulerAdvanceOp::register(ctx);
    ops::pipeline::CuteTmaLoadPipelineMakeOp::register(ctx);
    ops::pipeline::CuteTmaLoadPipelineInitOp::register(ctx);
    ops::pipeline::CutePipelineStateNewOp::register(ctx);
    ops::pipeline::CutePipelineStateSlotOp::register(ctx);
    ops::pipeline::CutePipelineStateAdvanceOp::register(ctx);
    ops::pipeline::CutePipelineProducerAcquireOp::register(ctx);
    ops::pipeline::CutePipelineProducerExpectTxOp::register(ctx);
    ops::pipeline::CutePipelineConsumerWaitOp::register(ctx);
    ops::pipeline::CutePipelineConsumerReleaseOp::register(ctx);
    ops::pipeline::CutePipelineProducerTailOp::register(ctx);
    ops::mma::CuteSmemTensorOverlayOp::register(ctx);
    ops::mma::CuteTiledMmaSliceOp::register(ctx);
    ops::mma::CuteFragmentFillOp::register(ctx);
    ops::mma::CuteMmaLoadScalesOp::register(ctx);
    ops::mma::CuteFragmentSliceKOp::register(ctx);
    ops::mma::CuteMmaLoadAOp::register(ctx);
    ops::mma::CuteMmaPartitionBOp::register(ctx);
    ops::mma::CuteTiledGemmOp::register(ctx);
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
