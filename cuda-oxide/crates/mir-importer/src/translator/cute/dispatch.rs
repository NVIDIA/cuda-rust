/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Validate recognized calls and dispatch to operation-family emitters.

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

use super::classify::{
    classify_block_scaled_method_call, classify_epilogue_method_call,
    classify_pipeline_method_call, classify_scheduler_method_call, classify_smem_mma_method_call,
    classify_tensor_method_call,
};
use super::identity::{CuteIdentity, canonical_path, classify_cute_call};
use super::registry::{CuteFn, CuteFnSpec, SubstKind};
use super::{
    block_scaled, emit, epilogue, pipeline, scheduler, sm100, sm100_attention, smem_mma, tensor,
};

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
/// Returns `None` for other crates or an approved source abstraction that has
/// no recognized ergonomic method (the caller uses normal body translation).
/// Returns `Some(result)` when this module owns the call, in which case the
/// shared path must not touch it. Registered primitives validate their generic
/// schema; ergonomic methods instead use their family-specific ADT decoders.
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
    // can require CuTe recognition. Compiler hooks remain direct-call-only;
    // audited source-only modules may also be imported as Fn-trait callbacks.
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
        CuteFn::Attention(kind) => sm100_attention::emit(ctx, body, func, args, kind, target, block_ptr, prev_op, value_map, block_map, loc.clone()),
        CuteFn::Sm100(kind) => sm100::emit(
            ctx, body, func, args, kind, target, block_ptr, prev_op, value_map, block_map, loc.clone(),
        ),
        CuteFn::MakeTensorRead => tensor::emit::emit_make(
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
        CuteFn::MakeTensorWrite => tensor::emit::emit_make(
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
        CuteFn::ZippedDivideRead | CuteFn::ZippedDivideWrite => tensor::emit::emit_zipped_divide(
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
        CuteFn::SliceRead | CuteFn::SliceWrite => tensor::emit::emit_slice(
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
        CuteFn::LocalTileRead | CuteFn::LocalTileWrite => tensor::emit::emit_local_tile(
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
        CuteFn::TensorTileIsFull | CuteFn::TensorTileBase => tensor::emit::emit_query(
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
        CuteFn::TensorLoadTile => tensor::emit::emit_load(
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
        CuteFn::TensorStoreTile => tensor::emit::emit_store(
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
        CuteFn::TensorStoreElementAbs => tensor::emit::emit_store_element_abs(
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
        CuteFn::BlockScaledMake => block_scaled::emit::emit_make(
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
        CuteFn::BlockScaledThreadRow => block_scaled::emit::emit_row(
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
        CuteFn::BlockScaledKTile => block_scaled::emit::emit_k_tile(
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
        CuteFn::BlockScaledLoadK64 => block_scaled::emit::emit_load(
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
        CuteFn::BlockScaledDotK64 => block_scaled::emit::emit_dot(
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
        CuteFn::SchedulerNew1d => scheduler::emit::emit_new_1d(
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
        CuteFn::SchedulerHasWork => scheduler::emit::emit_has_work(
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
        CuteFn::SchedulerCurrentTile => scheduler::emit::emit_current_tile(
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
        CuteFn::WorkTileCoordinates => scheduler::emit::emit_coordinates(
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
        CuteFn::SchedulerAdvance => scheduler::emit::emit_advance(
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
        CuteFn::TmaLoadPipelineMake => pipeline::emit::emit_make(
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
                "PipelineTmaAsync::from_raw_parts is not supported by the v0 semantic pipeline; use one contiguous [full][empty] ring with from_raw_base"
                    .to_owned()
            )
        ),
        CuteFn::TmaLoadPipelineInit => pipeline::emit::emit_init(
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
        CuteFn::PipelineStateNew => pipeline::emit::emit_state_new(
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
        CuteFn::PipelineStateSlot => pipeline::emit::emit_state_slot(
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
        CuteFn::PipelineStateAdvance => pipeline::emit::emit_state_advance(
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
        CuteFn::PipelineProducerAcquire => pipeline::emit::emit_producer_acquire(
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
        CuteFn::PipelineProducerExpectTx => pipeline::emit::emit_producer_expect_tx(
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
        CuteFn::PipelineConsumerWait => pipeline::emit::emit_consumer_wait(
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
        CuteFn::PipelineConsumerRelease => pipeline::emit::emit_consumer_release(
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
        CuteFn::PipelineProducerTail => pipeline::emit::emit_producer_tail(
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
        CuteFn::SharedTensorOverlay => smem_mma::emit::emit_overlay(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::TiledMmaSlice => smem_mma::emit::emit_mma_slice(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaFragmentFill => smem_mma::emit::emit_fragment_fill(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaLoadScales => smem_mma::emit::emit_load_scales(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaFragmentSliceK => smem_mma::emit::emit_slice_k(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaLoadA => smem_mma::emit::emit_load_a(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::MmaPartitionB => smem_mma::emit::emit_partition_b(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::TiledGemm => smem_mma::emit::emit_tiled_gemm(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::Nvfp4MmaAtom => smem_mma::emit::emit_nvfp4_mma_atom(
            ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map,
            loc.clone(),
        ),
        CuteFn::EpilogueOverlay => epilogue::emit::emit_overlay(
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
        CuteFn::EpilogueWarpSlice => epilogue::emit::emit_warp_slice(
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
        CuteFn::EpilogueStoreFragment => epilogue::emit::emit_store_fragment(
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
        CuteFn::EpilogueSyncReusable => epilogue::emit::emit_sync(
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
        CuteFn::EpilogueSyncReadyForTma => epilogue::emit::emit_sync(
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
        CuteFn::EpilogueHalf => epilogue::emit::emit_half(
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
        CuteFn::TmaStoreAcquire => epilogue::emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue::emit::StorePipelineEffect::Acquire,
            target,
            block_ptr,
            prev_op,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaStoreCommit => epilogue::emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue::emit::StorePipelineEffect::Commit,
            target,
            block_ptr,
            prev_op,
            block_map,
            loc.clone(),
        ),
        CuteFn::TmaStoreTail => epilogue::emit::emit_store_pipeline_effect(
            ctx,
            func,
            args,
            epilogue::emit::StorePipelineEffect::Tail,
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
        CuteFn::StoreGlobalNoAllocate => emit::emit_store_global_no_allocate(
            ctx,body,func,args,target,block_ptr,prev_op,value_map,block_map,loc.clone(),
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
        CuteFn::LoadMatrixBPair => emit::emit_load_matrix(
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
            dialect_cute::attributes::CuteMatrixRoleAttr::BPair,
        ),
        CuteFn::LoadMatrixBPairKMajor => emit::emit_load_matrix(
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
            dialect_cute::attributes::CuteMatrixRoleAttr::BPairKMajor,
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
