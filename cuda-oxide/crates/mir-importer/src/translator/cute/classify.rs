/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Recognize ergonomic methods using exact defining modules and receiver types.

use super::registry::CuteFn;
use super::{block_scaled, epilogue, pipeline, scheduler, smem_mma, tensor};
use rustc_public::mir;

/// Match an inherent method in one exact defining module.
///
/// The printed impl owner varies across rustc revisions; callers separately
/// validate the receiver/result ADT. Neither extra module nesting nor a
/// public reexport path is accepted as the defining identity.
fn method_in_module<'a>(path: &'a str, module: &str) -> Option<&'a str> {
    let (crate_name, rest) = path.split_once("::")?;
    if !matches!(crate_name, "cute_rs" | "cute-rs") {
        return None;
    }
    let rest = rest.strip_prefix(module)?.strip_prefix("::")?;
    let (impl_owner, method) = rest.split_once("::")?;
    if impl_owner.is_empty() || method.contains("::") {
        return None;
    }
    Some(method)
}

/// Extract a method only from the exact tensor implementation module.
pub(super) fn tensor_method_name_from_path(path: &str) -> Option<&str> {
    method_in_module(path, "tensor")
}

/// Extract a method from the supported block-scaled tensor or GEMV implementation.
pub(super) fn block_scaled_method_name_from_path(path: &str) -> Option<&str> {
    method_in_module(path, "block_scaled::tensor")
        .or_else(|| method_in_module(path, "block_scaled::gemv"))
}

/// Extract a method from the supported tiled-MMA or fragment implementation.
pub(super) fn smem_mma_method_name_from_path<'a>(
    path: &'a str,
    expected_module: &str,
) -> Option<&'a str> {
    method_in_module(path, expected_module)
}

/// Recognize shared-MMA methods by defining module and receiver/result ADT.
pub(super) fn classify_smem_mma_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let destination_ty = destination.ty(body.locals()).ok();
    let operand_ty = |index: usize| args.get(index)?.ty(body.locals()).ok();

    if smem_mma_method_name_from_path(path, "copy::tiled") == Some("from_raw_parts")
        && args.len() == 2
        && destination_ty
            .as_ref()
            .is_some_and(|ty| smem_mma::decode_shared_tensor(ty).ok().flatten().is_some())
    {
        return Some(CuteFn::SharedTensorOverlay);
    }

    let method = smem_mma_method_name_from_path(path, "mma::block_scaled::tiled")
        .filter(|method| {
            matches!(
                *method,
                "get_slice" | "load_scale_atom_128" | "load_a_128" | "get_b_tile_k64"
            )
        })
        .or_else(|| {
            smem_mma_method_name_from_path(path, "mma::block_scaled::fragments")
                .filter(|method| matches!(*method, "zero" | "pairs_at_unchecked" | "gemm"))
        })?;
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
        "gemm"
            if args.len() == 5
                && operand_ty(0)
                    .is_some_and(|ty| smem_mma::is_accumulator(&ty).unwrap_or(false)) =>
        {
            Some(CuteFn::TiledGemm)
        }
        _ => None,
    }
}

/// Recognize epilogue and store-pipeline methods with their exact source types.
pub(super) fn classify_epilogue_method_call(
    path: &str,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
) -> Option<CuteFn> {
    let operand_ty = |index: usize| args.get(index)?.ty(body.locals()).ok();
    let destination_ty = destination.ty(body.locals()).ok();

    if let Some(method) = smem_mma_method_name_from_path(path, "epilogue::sm120") {
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

    if smem_mma_method_name_from_path(path, "mma::block_scaled::epilogue") == Some("store_tile")
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
pub(super) fn scheduler_method_name_from_path(path: &str) -> Option<&str> {
    method_in_module(path, "scheduler")
}

/// Return an inherent method name from the exact pipeline module.
///
/// The impl-block spelling is not stable. Exact pipeline/state Rust types in
/// `classify_pipeline_method_call` provide the method identity.
pub(super) fn pipeline_method_name_from_path(path: &str) -> Option<&str> {
    method_in_module(path, "pipeline")
}

/// Recognize load-pipeline and cursor methods without treating other helpers as primitives.
pub(super) fn classify_pipeline_method_call(
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
        "index" if args.len() == 1 && receiver_state.is_some() => Some(CuteFn::PipelineStateSlot),
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

/// Recognize persistent-scheduler methods and validate receiver/result ownership.
pub(super) fn classify_scheduler_method_call(
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
        "get_current_work"
            if args.len() == 1
                && matches!((receiver_scheduler, destination_tile), (Some(source), Some(result)) if source == result) =>
        {
            Some(CuteFn::SchedulerCurrentTile)
        }
        "tile_idx" if args.len() == 1 && receiver_tile.is_some() => {
            Some(CuteFn::WorkTileCoordinates)
        }
        "advance_to_next_work" if args.len() == 1 && receiver_scheduler.is_some() => {
            Some(CuteFn::SchedulerAdvance)
        }
        _ => None,
    }
}

/// Recognize scaled-tensor methods while checking their selection stage and role.
pub(super) fn classify_block_scaled_method_call(
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

/// Recognize tensor-view methods from exact types and access mode.
pub(super) fn classify_tensor_method_call(
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
        // CuTe's `local_tile` for a one-dimensional tiler: one call whose
        // emitter produces the same zipped_divide and slice pair as the chain.
        "local_tile" => match (receiver_view, destination_view) {
            (Some(source), Some(result))
                if source.layout == CuteTensorLayoutAttr::Contiguous1D
                    && matches!(result.layout, CuteTensorLayoutAttr::Tile1D(_))
                    && source.access == result.access =>
            {
                Some(if source.access == CuteTensorAccessAttr::ReadOnly {
                    CuteFn::LocalTileRead
                } else {
                    CuteFn::LocalTileWrite
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
        // The Tile1D guard keeps `TensorMut<Contiguous1D>::store_at`, ordinary
        // Rust with the same method name, on the body-translation path.
        "store_at"
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

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::method_in_module;

    /// Check that recognition uses exact defining modules rather than public reexport paths.
    #[test]
    fn nested_defining_modules_are_exact_not_reexport_paths() {
        for module in [
            "block_scaled::tensor",
            "block_scaled::gemv",
            "mma::block_scaled::tiled",
            "mma::block_scaled::fragments",
            "mma::block_scaled::epilogue",
        ] {
            assert_eq!(
                method_in_module(&format!("cute_rs::{module}::{{impl#7}}::method"), module),
                Some("method")
            );
            for path in [
                format!("other::{module}::{{impl#7}}::method"),
                format!("cute_rs::{module}_extra::{{impl#7}}::method"),
                format!("cute_rs::{module}::nested::{{impl#7}}::method"),
                format!("cute_rs::{module}::method"),
                format!(
                    "cute_rs::{}::{{impl#7}}::method",
                    module.split("::").next().unwrap()
                ),
            ] {
                assert_eq!(method_in_module(&path, module), None, "{path}");
            }
        }
    }
}
