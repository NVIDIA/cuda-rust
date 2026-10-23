/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check CuTe values and operation order before either backend lowers them.
//!
//! This module follows where values come from and how they are used. It checks
//! pipeline ordering and restrictions on compiler-only types without changing
//! the input module.
//!
//! Verification runs local operation checks, the ghost escape audit, then
//! family checks within each MIR function (or the root for standalone IR).
//! The supported graph shapes are intentionally specific:
//!
//! - Each TMA load pipeline has one initialization and one complete producer
//!   and consumer state. Each lifecycle event has one static operation site;
//!   loop iterations reuse those sites.
//! - The SM120 epilogue has one overlay, warp slice, and fragment store, followed
//!   by two output halves in one committed store group per function. Its writer
//!   and issuer sequences may cross unconditional blocks with no side entry.
//!
//! These are limits of the current pattern checks, not proofs of arbitrary
//! concurrent programs. Dynamic bounds, participation, and barrier phases
//! remain source obligations. `cfg.rs` distinguishes dominance, reachability,
//! and straight-line order; `provenance.rs` follows ordinary carrier values.

/// Function scopes, reachability, dominance, and source-order protocol checks.
mod cfg;
/// Shared output placement and asynchronous store synchronization contracts.
mod epilogue;
/// Packed-value and scale-view contracts for the block-scaled GEMV flow.
mod gemv;
/// Restrictions preventing compiler-only CuTe descriptions from entering runtime storage.
mod ghosts;
/// Shared tensor placement, register carriers, and tiled matrix-instruction contracts.
mod mma;
/// Barrier-ring configuration and producer/consumer cursor contracts.
mod pipeline;
mod provenance;
/// Persistent output-tile grids and their ordinary cursor provenance.
mod scheduler;
/// Tensor storage, access, format, and selection metadata.
mod tensor;
/// Descriptor-backed raw transport views and their direct copy contracts.
mod tma;

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;

use cfg::operations_by_scope;
use epilogue::verify_epilogue_story;
use gemv::verify_gemv_story;
use ghosts::audit_cute_ghosts;
pub use ghosts::type_contains_cute_ghost;
use mma::verify_smem_mma_story;
use pipeline::verify_pipeline_story;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use scheduler::verify_scheduler_story;
use tensor::verify_tensor_story;
use tma::verify_tma_story;

use crate::{
    ops::epilogue::{
        CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp,
        CuteEpilogueSyncOp, CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp,
        CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
    },
    ops::gemm_tma::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp},
    ops::gemv::{
        CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
        CuteScaledViewRowOp, CuteTensorMake2DOp,
    },
    ops::mma::{
        CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
        CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
    },
    ops::pipeline::{
        CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
        CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
        CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
        CuteTmaLoadPipelineMakeOp,
    },
    ops::scheduler::{
        CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp,
        CuteSchedulerNew1dOp, CuteWorkTileCoordinatesOp,
    },
    ops::tensor::{
        CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
        CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
        CuteTensorZippedDivideOp,
    },
    ops::{
        CuteAssumeDivOp, CuteCopyG2SOp, CuteCopyOp, CuteLdmatrixOp, CuteTmaLoad2dOp,
        CuteTmaStore2dOp,
    },
};

/// Failure reported by the shared semantic verifier.
///
/// Native expansion and MLIR translation can both wrap this type without
/// depending on one another's error or lowering representations.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// The operation graph violates a supported semantic contract.
    #[error("invalid CuTe semantic module: {0}")]
    Invalid(#[doc = "Explanation of the violated semantic contract."] String),
    /// The graph contains a CuTe operation outside the verifier's supported set.
    #[error("unsupported CuTe semantic operation: {0}")]
    Unsupported(#[doc = "Unsupported operation or feature description."] String),
}

/// Result type shared by backend-neutral verification and backend adapters.
pub type VerifyResult<T = ()> = std::result::Result<T, VerifyError>;

/// Internal shorthand for the backend-neutral semantic verification result.
type Result<T = ()> = VerifyResult<T>;

/// Wrap a semantic contract failure in the shared invalid-module error variant.
fn invalid(message: impl Into<String>) -> VerifyError {
    VerifyError::Invalid(message.into())
}

/// Visit the root and all nested operations in region/block order without changing the IR.
fn collect_ops(ctx: &Context, root: Ptr<Operation>, output: &mut Vec<Ptr<Operation>>) {
    output.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, output);
            }
        }
    }
}

/// Dispatch each known CuTe operation's local verifier and reject unknown CuTe operation IDs.
fn verify_local_op(ctx: &Context, operation: Ptr<Operation>) -> Result {
    let opid = Operation::get_opid(operation, ctx);
    /// Wrap the operation as its recognized concrete type and run local verification.
    macro_rules! verify_as {
        ($ty:ty) => {
            <$ty>::wrap(operation).verify(ctx)
        };
    }

    let result = if opid == crate::ops::copy::CuteStoreGlobalNoAllocateOp::get_opid_static() {
        Some(verify_as!(crate::ops::copy::CuteStoreGlobalNoAllocateOp))
    } else if let Some(result) = crate::ops::sm100_attention::verify_local(ctx, operation) {
        Some(result)
    } else if let Some(result) = crate::ops::sm100::verify_local(ctx, operation) {
        Some(result)
    } else if opid == CuteTensorMakeOp::get_opid_static() {
        Some(verify_as!(CuteTensorMakeOp))
    } else if opid == CuteTensorZippedDivideOp::get_opid_static() {
        Some(verify_as!(CuteTensorZippedDivideOp))
    } else if opid == CuteTensorSliceOp::get_opid_static() {
        Some(verify_as!(CuteTensorSliceOp))
    } else if opid == CuteTensorIsFullOp::get_opid_static() {
        Some(verify_as!(CuteTensorIsFullOp))
    } else if opid == CuteTensorBaseOp::get_opid_static() {
        Some(verify_as!(CuteTensorBaseOp))
    } else if opid == CuteTensorLoadIntoOp::get_opid_static() {
        Some(verify_as!(CuteTensorLoadIntoOp))
    } else if opid == CuteTensorStoreFromOp::get_opid_static() {
        Some(verify_as!(CuteTensorStoreFromOp))
    } else if opid == CuteTensorStoreElementAbsOp::get_opid_static() {
        Some(verify_as!(CuteTensorStoreElementAbsOp))
    } else if opid == CuteTensorMake2DOp::get_opid_static() {
        Some(verify_as!(CuteTensorMake2DOp))
    } else if opid == CuteScaledViewMakeOp::get_opid_static() {
        Some(verify_as!(CuteScaledViewMakeOp))
    } else if opid == CuteScaledViewRowOp::get_opid_static() {
        Some(verify_as!(CuteScaledViewRowOp))
    } else if opid == CuteScaledViewKTileOp::get_opid_static() {
        Some(verify_as!(CuteScaledViewKTileOp))
    } else if opid == CuteScaledViewLoadOp::get_opid_static() {
        Some(verify_as!(CuteScaledViewLoadOp))
    } else if opid == CuteDotOp::get_opid_static() {
        Some(verify_as!(CuteDotOp))
    } else if opid == CuteSchedulerNew1dOp::get_opid_static() {
        Some(verify_as!(CuteSchedulerNew1dOp))
    } else if opid == CuteSchedulerHasWorkOp::get_opid_static() {
        Some(verify_as!(CuteSchedulerHasWorkOp))
    } else if opid == CuteSchedulerCurrentOp::get_opid_static() {
        Some(verify_as!(CuteSchedulerCurrentOp))
    } else if opid == CuteWorkTileCoordinatesOp::get_opid_static() {
        Some(verify_as!(CuteWorkTileCoordinatesOp))
    } else if opid == CuteSchedulerAdvanceOp::get_opid_static() {
        Some(verify_as!(CuteSchedulerAdvanceOp))
    } else if opid == CuteTmaLoadPipelineMakeOp::get_opid_static() {
        Some(verify_as!(CuteTmaLoadPipelineMakeOp))
    } else if opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
        Some(verify_as!(CuteTmaLoadPipelineInitOp))
    } else if opid == CutePipelineStateNewOp::get_opid_static() {
        Some(verify_as!(CutePipelineStateNewOp))
    } else if opid == CutePipelineStateSlotOp::get_opid_static() {
        Some(verify_as!(CutePipelineStateSlotOp))
    } else if opid == CutePipelineStateAdvanceOp::get_opid_static() {
        Some(verify_as!(CutePipelineStateAdvanceOp))
    } else if opid == CutePipelineProducerAcquireOp::get_opid_static() {
        Some(verify_as!(CutePipelineProducerAcquireOp))
    } else if opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        Some(verify_as!(CutePipelineProducerExpectTxOp))
    } else if opid == CutePipelineConsumerWaitOp::get_opid_static() {
        Some(verify_as!(CutePipelineConsumerWaitOp))
    } else if opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        Some(verify_as!(CutePipelineConsumerReleaseOp))
    } else if opid == CutePipelineProducerTailOp::get_opid_static() {
        Some(verify_as!(CutePipelineProducerTailOp))
    } else if opid == CuteTmaGmemViewOp::get_opid_static() {
        Some(verify_as!(CuteTmaGmemViewOp))
    } else if opid == CuteTmaSmemViewOp::get_opid_static() {
        Some(verify_as!(CuteTmaSmemViewOp))
    } else if opid == CuteTmaCopy2dOp::get_opid_static() {
        Some(verify_as!(CuteTmaCopy2dOp))
    } else if opid == CuteSmemTensorOverlayOp::get_opid_static() {
        Some(verify_as!(CuteSmemTensorOverlayOp))
    } else if opid == CuteTiledMmaSliceOp::get_opid_static() {
        Some(verify_as!(CuteTiledMmaSliceOp))
    } else if opid == CuteFragmentFillOp::get_opid_static() {
        Some(verify_as!(CuteFragmentFillOp))
    } else if opid == CuteMmaLoadScalesOp::get_opid_static() {
        Some(verify_as!(CuteMmaLoadScalesOp))
    } else if opid == CuteFragmentSliceKOp::get_opid_static() {
        Some(verify_as!(CuteFragmentSliceKOp))
    } else if opid == CuteMmaLoadAOp::get_opid_static() {
        Some(verify_as!(CuteMmaLoadAOp))
    } else if opid == CuteMmaPartitionBOp::get_opid_static() {
        Some(verify_as!(CuteMmaPartitionBOp))
    } else if opid == CuteTiledGemmOp::get_opid_static() {
        Some(verify_as!(CuteTiledGemmOp))
    } else if opid == CuteEpilogueSmemOverlayOp::get_opid_static() {
        Some(verify_as!(CuteEpilogueSmemOverlayOp))
    } else if opid == CuteEpilogueWarpSliceOp::get_opid_static() {
        Some(verify_as!(CuteEpilogueWarpSliceOp))
    } else if opid == CuteEpilogueStoreFragmentOp::get_opid_static() {
        Some(verify_as!(CuteEpilogueStoreFragmentOp))
    } else if opid == CuteEpilogueSyncOp::get_opid_static() {
        Some(verify_as!(CuteEpilogueSyncOp))
    } else if opid == CuteEpilogueHalfOp::get_opid_static() {
        Some(verify_as!(CuteEpilogueHalfOp))
    } else if opid == CuteTmaStoreAcquireOp::get_opid_static() {
        Some(verify_as!(CuteTmaStoreAcquireOp))
    } else if opid == CuteTmaStoreCommitOp::get_opid_static() {
        Some(verify_as!(CuteTmaStoreCommitOp))
    } else if opid == CuteTmaStoreTailOp::get_opid_static() {
        Some(verify_as!(CuteTmaStoreTailOp))
    } else if opid == CuteTmaStore2dSemanticOp::get_opid_static() {
        Some(verify_as!(CuteTmaStore2dSemanticOp))
    } else if opid == CuteCopyOp::get_opid_static() {
        Some(verify_as!(CuteCopyOp))
    } else if opid == CuteCopyG2SOp::get_opid_static() {
        Some(verify_as!(CuteCopyG2SOp))
    } else if opid == CuteAssumeDivOp::get_opid_static() {
        Some(verify_as!(CuteAssumeDivOp))
    } else if opid == CuteLdmatrixOp::get_opid_static() {
        Some(verify_as!(CuteLdmatrixOp))
    } else if opid == CuteTmaLoad2dOp::get_opid_static() {
        Some(verify_as!(CuteTmaLoad2dOp))
    } else if opid == CuteTmaStore2dOp::get_opid_static() {
        Some(verify_as!(CuteTmaStore2dOp))
    } else {
        None
    };

    if let Some(result) = result {
        result.map_err(|error| invalid(format!("`{opid}` failed local verification: {error}")))?;
    } else if opid.dialect.to_string() == crate::CUTE_DIALECT_NAME {
        return Err(VerifyError::Unsupported(format!(
            "unhandled operation `{opid}`"
        )));
    }
    Ok(())
}

/// Validate the complete high-level CuTe module without cloning or mutating
/// the IR, allocating replacement operations, or invoking either backend.
///
/// Unsafe runtime promises such as bounds, thread participation, and TMA
/// descriptor validity remain caller contracts. This verifier checks the
/// static semantic graph and rejects any compiler-only value that escapes it.
pub fn verify_cute_semantics(ctx: &Context, module: Ptr<Operation>) -> Result {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);

    for operation in &all_ops {
        verify_local_op(ctx, *operation)?;
    }
    audit_cute_ghosts(ctx, &all_ops)?;
    for scope in operations_by_scope(ctx, module, &all_ops) {
        verify_scheduler_story(ctx, &scope)?;
        verify_pipeline_story(ctx, &scope)?;
        verify_tma_story(ctx, &scope)?;
        verify_smem_mma_story(ctx, &scope)?;
        verify_gemv_story(ctx, &scope)?;
        verify_tensor_story(ctx, &scope)?;
        verify_epilogue_story(ctx, &scope)?;
        crate::ops::sm100::verify_story(ctx, &scope).map_err(invalid)?;
        crate::ops::sm100_attention::verify_story(ctx, &scope).map_err(invalid)?;
    }
    Ok(())
}
