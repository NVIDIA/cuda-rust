/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Restrictions preventing compiler-only CuTe descriptions from entering runtime storage.

use std::collections::HashSet;

use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirStructType,
    MirTupleType, MirUnionType,
};
use pliron::{
    attribute::attr_cast,
    builtin::{attr_interfaces::TypedAttrInterface, type_interfaces::FunctionTypeInterface},
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::{Op, OpId},
    operation::Operation,
    r#type::{TypeHandle, Typed, type_cast},
};

use super::{Result, invalid};
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
    types::{
        CuteEpilogueTileType, CuteFragmentType, CuteScaledViewType, CuteSmemTensorType,
        CuteTensorViewType, CuteTmaLoadPipelineType, CuteTmaViewType, CuteWorkTileType,
    },
};

/// List the tensor operations allowed to consume or produce direct tensor ghosts.
pub(super) fn tensor_semantic_ids() -> [OpId; 8] {
    [
        CuteTensorMakeOp::get_opid_static(),
        CuteTensorZippedDivideOp::get_opid_static(),
        CuteTensorSliceOp::get_opid_static(),
        CuteTensorIsFullOp::get_opid_static(),
        CuteTensorBaseOp::get_opid_static(),
        CuteTensorLoadIntoOp::get_opid_static(),
        CuteTensorStoreFromOp::get_opid_static(),
        CuteTensorStoreElementAbsOp::get_opid_static(),
    ]
}

/// List the packed-value view and fragment operations covered by the GEMV audit.
pub(super) fn gemv_semantic_ids() -> [OpId; 6] {
    [
        CuteTensorMake2DOp::get_opid_static(),
        CuteScaledViewMakeOp::get_opid_static(),
        CuteScaledViewRowOp::get_opid_static(),
        CuteScaledViewKTileOp::get_opid_static(),
        CuteScaledViewLoadOp::get_opid_static(),
        CuteDotOp::get_opid_static(),
    ]
}

/// List scheduler operations participating in work-tile provenance checks.
pub(super) fn scheduler_semantic_ids() -> [OpId; 5] {
    [
        CuteSchedulerNew1dOp::get_opid_static(),
        CuteSchedulerHasWorkOp::get_opid_static(),
        CuteSchedulerCurrentOp::get_opid_static(),
        CuteWorkTileCoordinatesOp::get_opid_static(),
        CuteSchedulerAdvanceOp::get_opid_static(),
    ]
}

/// List load-pipeline and cursor operations allowed by the lifecycle audit.
pub(super) fn pipeline_semantic_ids() -> [OpId; 10] {
    [
        CuteTmaLoadPipelineMakeOp::get_opid_static(),
        CuteTmaLoadPipelineInitOp::get_opid_static(),
        CutePipelineStateNewOp::get_opid_static(),
        CutePipelineStateSlotOp::get_opid_static(),
        CutePipelineStateAdvanceOp::get_opid_static(),
        CutePipelineProducerAcquireOp::get_opid_static(),
        CutePipelineProducerExpectTxOp::get_opid_static(),
        CutePipelineConsumerWaitOp::get_opid_static(),
        CutePipelineConsumerReleaseOp::get_opid_static(),
        CutePipelineProducerTailOp::get_opid_static(),
    ]
}

/// List direct carrier-view constructors and their semantic TMA copy consumer.
fn tma_semantic_ids() -> [OpId; 3] {
    [
        CuteTmaGmemViewOp::get_opid_static(),
        CuteTmaSmemViewOp::get_opid_static(),
        CuteTmaCopy2dOp::get_opid_static(),
    ]
}

/// List operations whose ordinary carriers carry shared-memory MMA meaning.
pub(super) fn smem_mma_semantic_ids() -> [OpId; 8] {
    [
        CuteSmemTensorOverlayOp::get_opid_static(),
        CuteTiledMmaSliceOp::get_opid_static(),
        CuteFragmentFillOp::get_opid_static(),
        CuteMmaLoadScalesOp::get_opid_static(),
        CuteFragmentSliceKOp::get_opid_static(),
        CuteMmaLoadAOp::get_opid_static(),
        CuteMmaPartitionBOp::get_opid_static(),
        CuteTiledGemmOp::get_opid_static(),
    ]
}

/// List output-tile, synchronization, and asynchronous store protocol operations.
pub(super) fn epilogue_semantic_ids() -> [OpId; 9] {
    [
        CuteEpilogueSmemOverlayOp::get_opid_static(),
        CuteEpilogueWarpSliceOp::get_opid_static(),
        CuteEpilogueStoreFragmentOp::get_opid_static(),
        CuteEpilogueSyncOp::get_opid_static(),
        CuteEpilogueHalfOp::get_opid_static(),
        CuteTmaStoreAcquireOp::get_opid_static(),
        CuteTmaStoreCommitOp::get_opid_static(),
        CuteTmaStoreTailOp::get_opid_static(),
        CuteTmaStore2dSemanticOp::get_opid_static(),
    ]
}

/// Collect the closed set of operations allowed to carry semantic ghosts or metadata.
fn all_high_level_semantic_ids() -> Vec<OpId> {
    let mut ids = tensor_semantic_ids().to_vec();
    ids.push(crate::ops::copy::CuteStoreGlobalNoAllocateOp::get_opid_static());
    ids.extend(gemv_semantic_ids());
    ids.extend(scheduler_semantic_ids());
    ids.extend(pipeline_semantic_ids());
    ids.extend(tma_semantic_ids());
    ids.extend(smem_mma_semantic_ids());
    ids.extend(epilogue_semantic_ids());
    ids.extend(crate::ops::sm100::semantic_ids());
    ids.extend(crate::ops::sm100_attention::semantic_ids());
    ids
}

/// Recognize ghost SSA types, excluding types permitted only as static metadata.
fn is_direct_cute_ghost_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<CuteTensorViewType>().is_some()
        || ty.downcast_ref::<CuteTmaViewType>().is_some()
        || ty.downcast_ref::<CuteScaledViewType>().is_some()
        || ty.downcast_ref::<CuteFragmentType>().is_some()
        || ty.downcast_ref::<CuteWorkTileType>().is_some()
        || ty.downcast_ref::<CuteTmaLoadPipelineType>().is_some()
}

/// `true` when `ty` is, or structurally contains, a CuTe compiler-only
/// (ghost) type. Ghost values must remain direct semantic SSA; preparation
/// passes use this to decide which slots must be dissolved before this
/// module's semantic verification runs.
pub fn type_contains_cute_ghost(ctx: &Context, ty: TypeHandle) -> bool {
    /// Recursively inspect nested type structure for ghosts, using a visited set to break recursive cycles.
    fn walk(ctx: &Context, ty: TypeHandle, seen: &mut HashSet<TypeHandle>) -> bool {
        if !seen.insert(ty) {
            return false;
        }
        let ty_ref = ty.deref(ctx);
        if ty_ref.downcast_ref::<CuteTensorViewType>().is_some()
            || ty_ref.downcast_ref::<CuteTmaViewType>().is_some()
            || ty_ref.downcast_ref::<CuteScaledViewType>().is_some()
            || ty_ref.downcast_ref::<CuteFragmentType>().is_some()
            || ty_ref.downcast_ref::<CuteWorkTileType>().is_some()
            || ty_ref.downcast_ref::<CuteTmaLoadPipelineType>().is_some()
            || ty_ref.downcast_ref::<CuteSmemTensorType>().is_some()
            || ty_ref.downcast_ref::<CuteEpilogueTileType>().is_some()
        {
            return true;
        }
        if let Some(function) = type_cast::<dyn FunctionTypeInterface>(&*ty_ref) {
            return function
                .arg_types()
                .into_iter()
                .chain(function.res_types())
                .any(|child| walk(ctx, child, seen));
        }
        if let Some(pointer) = ty_ref.downcast_ref::<MirPtrType>() {
            return walk(ctx, pointer.pointee, seen);
        }
        if let Some(tuple) = ty_ref.downcast_ref::<MirTupleType>() {
            return tuple.types.iter().any(|child| walk(ctx, *child, seen));
        }
        if let Some(structure) = ty_ref.downcast_ref::<MirStructType>() {
            return structure
                .field_types
                .iter()
                .any(|child| walk(ctx, *child, seen));
        }
        if let Some(union) = ty_ref.downcast_ref::<MirUnionType>() {
            return union
                .field_types
                .iter()
                .any(|child| walk(ctx, *child, seen));
        }
        if let Some(array) = ty_ref.downcast_ref::<MirArrayType>() {
            return walk(ctx, array.element_ty, seen);
        }
        if let Some(slice) = ty_ref.downcast_ref::<MirSliceType>() {
            return walk(ctx, slice.element_ty, seen);
        }
        if let Some(slice) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
            return walk(ctx, slice.element_ty, seen)
                || slice.space_tys.iter().any(|child| walk(ctx, *child, seen));
        }
        if let Some(enumeration) = ty_ref.downcast_ref::<MirEnumType>() {
            return walk(ctx, enumeration.discriminant_ty, seen)
                || enumeration
                    .all_field_types
                    .iter()
                    .any(|child| walk(ctx, *child, seen));
        }
        false
    }

    walk(ctx, ty, &mut HashSet::new())
}

/// Reject ghost values nested in runtime carriers, signatures, block arguments, or unrelated operations.
pub(super) fn audit_cute_ghosts(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let semantic_ids = all_high_level_semantic_ids();
    let escape = |where_: String| {
        invalid(format!(
            "CuTe compiler-only values must remain direct semantic SSA or typed semantic metadata; a ghost CuTe value escaped in {where_}"
        ))
    };

    for op_ptr in all_ops {
        let opid = Operation::get_opid(*op_ptr, ctx);
        let operation = op_ptr.deref(ctx);
        for value in operation.operands().chain(operation.results()) {
            let ty = value.get_type(ctx);
            if !type_contains_cute_ghost(ctx, ty) {
                continue;
            }
            if !semantic_ids.contains(&opid) || !is_direct_cute_ghost_type(ctx, ty) {
                return Err(escape(format!("an operand or result of `{opid}`")));
            }
        }
        for (name, attribute) in &operation.attributes.0 {
            let Some(typed) = attr_cast::<dyn TypedAttrInterface>(&**attribute) else {
                continue;
            };
            let ty = typed.get_type(ctx);
            if !type_contains_cute_ghost(ctx, ty) {
                continue;
            }
            let direct_metadata = ty.deref(ctx).downcast_ref::<CuteSmemTensorType>().is_some()
                || ty
                    .deref(ctx)
                    .downcast_ref::<CuteEpilogueTileType>()
                    .is_some();
            if !semantic_ids.contains(&opid) || !direct_metadata {
                return Err(escape(format!("attribute `{name}` on `{opid}`")));
            }
        }
        for region in operation.regions() {
            for block in region.deref(ctx).iter(ctx) {
                for argument in block.deref(ctx).arguments() {
                    if type_contains_cute_ghost(ctx, argument.get_type(ctx)) {
                        return Err(escape(format!("a block argument below `{opid}`")));
                    }
                }
                for (name, attribute) in &block.deref(ctx).attributes.0 {
                    if let Some(typed) = attr_cast::<dyn TypedAttrInterface>(&**attribute)
                        && type_contains_cute_ghost(ctx, typed.get_type(ctx))
                    {
                        return Err(escape(format!("block attribute `{name}` below `{opid}`")));
                    }
                }
            }
        }
    }
    Ok(())
}
