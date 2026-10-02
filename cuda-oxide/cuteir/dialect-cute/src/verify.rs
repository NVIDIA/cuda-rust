/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Check CuTe values and operation order before either backend lowers them.
//!
//! This module follows where values come from and how they are used. It checks
//! pipeline ordering and restrictions on compiler-only types without changing
//! the input module.

use std::collections::{HashMap, HashSet};

use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::control_flow::MirCondBranchOp;
use dialect_mir::ops::function::MirFuncOp;
use dialect_mir::ops::{
    MirAddOp, MirAllocaOp, MirCallOp, MirCastOp, MirConstantOp, MirConstructStructOp,
    MirConstructTupleOp, MirExtractFieldOp, MirFieldAddrOp, MirGotoOp, MirInsertFieldOp, MirLoadOp,
    MirNotOp, MirRefOp, MirStorageDeadOp, MirStorageLiveOp, MirStoreOp, MirUndefOp,
};
use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirStructType,
    MirTupleType, MirUnionType, address_space,
};
use pliron::attribute::attr_cast;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attr_interfaces::TypedAttrInterface;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::IntegerType;
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::{Op, OpId, op_cast};
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed, type_cast};
use pliron::value::Value;

use crate::attributes::{
    CuteEpilogueSyncPhaseAttr, CutePipelineRoleAttr, CutePipelineStateAttr, CuteTensorRoleAttr,
    CuteTileGridAttr, CuteTiledMmaPlanAttr,
};
use crate::epilogue_ops::{
    CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp,
    CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp, CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp,
    CuteTmaStoreTailOp,
};
use crate::gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp};
use crate::gemv_ops::{
    CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
    CuteScaledViewRowOp, CuteTensorMake2DOp,
};
use crate::ops::{
    CuteAssumeDivOp, CuteCopyG2SOp, CuteCopyOp, CuteLdmatrixOp, CuteTmaLoad2dOp, CuteTmaStore2dOp,
};
use crate::pipeline_ops::{
    CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
    CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
    CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
    CuteTmaLoadPipelineMakeOp,
};
use crate::scheduler_ops::{
    CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp, CuteSchedulerNew1dOp,
    CuteWorkTileCoordinatesOp,
};
use crate::smem_mma_ops::{
    CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
    CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
};
use crate::tensor_ops::{
    CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
    CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
    CuteTensorZippedDivideOp,
};
use crate::types::{
    CuteEpilogueTileType, CuteFragmentType, CuteScaledViewType, CuteSmemTensorType,
    CuteTensorViewType, CuteTmaLoadPipelineType, CuteTmaViewType, CuteWorkTileType,
};

/// Failure reported by the shared semantic verifier.
///
/// Native expansion and MLIR translation can both wrap this type without
/// depending on one another's error or lowering representations.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("invalid CuTe semantic module: {0}")]
    Invalid(String),
    #[error("unsupported CuTe semantic operation: {0}")]
    Unsupported(String),
}

/// Result type shared by backend-neutral verification and backend adapters.
pub type VerifyResult<T = ()> = std::result::Result<T, VerifyError>;

type Result<T = ()> = VerifyResult<T>;

fn invalid(message: impl Into<String>) -> VerifyError {
    VerifyError::Invalid(message.into())
}

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

fn tensor_semantic_ids() -> [OpId; 8] {
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

fn gemv_semantic_ids() -> [OpId; 6] {
    [
        CuteTensorMake2DOp::get_opid_static(),
        CuteScaledViewMakeOp::get_opid_static(),
        CuteScaledViewRowOp::get_opid_static(),
        CuteScaledViewKTileOp::get_opid_static(),
        CuteScaledViewLoadOp::get_opid_static(),
        CuteDotOp::get_opid_static(),
    ]
}

fn scheduler_semantic_ids() -> [OpId; 5] {
    [
        CuteSchedulerNew1dOp::get_opid_static(),
        CuteSchedulerHasWorkOp::get_opid_static(),
        CuteSchedulerCurrentOp::get_opid_static(),
        CuteWorkTileCoordinatesOp::get_opid_static(),
        CuteSchedulerAdvanceOp::get_opid_static(),
    ]
}

fn pipeline_semantic_ids() -> [OpId; 10] {
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

fn tma_semantic_ids() -> [OpId; 3] {
    [
        CuteTmaGmemViewOp::get_opid_static(),
        CuteTmaSmemViewOp::get_opid_static(),
        CuteTmaCopy2dOp::get_opid_static(),
    ]
}

fn smem_mma_semantic_ids() -> [OpId; 8] {
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

fn epilogue_semantic_ids() -> [OpId; 9] {
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

fn all_high_level_semantic_ids() -> Vec<OpId> {
    let mut ids = tensor_semantic_ids().to_vec();
    ids.extend(gemv_semantic_ids());
    ids.extend(scheduler_semantic_ids());
    ids.extend(pipeline_semantic_ids());
    ids.extend(tma_semantic_ids());
    ids.extend(smem_mma_semantic_ids());
    ids.extend(epilogue_semantic_ids());
    ids.extend(crate::sm100_ops::semantic_ids());
    ids
}

fn verify_local_op(ctx: &Context, operation: Ptr<Operation>) -> Result {
    let opid = Operation::get_opid(operation, ctx);
    macro_rules! verify_as {
        ($ty:ty) => {
            <$ty>::wrap(operation).verify(ctx)
        };
    }

    let result = if let Some(result) = crate::sm100_ops::verify_local(ctx, operation) {
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

fn audit_cute_ghosts(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
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

#[derive(Clone, Copy)]
struct TensorViewState {
    tile_size: Option<u64>,
    tile_index: Option<Value>,
}

fn resolve_tensor_view(ctx: &Context, value: Value, depth: usize) -> Result<TensorViewState> {
    if depth > 16 {
        return Err(invalid(
            "CuTe tensor-view producer chain is unexpectedly deep",
        ));
    }
    let Some(defining_op) = value.defining_op() else {
        return Err(invalid(
            "CuTe tensor view reached a block argument; a direct make/divide/slice chain is required",
        ));
    };
    let opid = Operation::get_opid(defining_op, ctx);
    if opid == CuteTensorMakeOp::get_opid_static() {
        return Ok(TensorViewState {
            tile_size: None,
            tile_index: None,
        });
    }
    if opid == CuteTensorZippedDivideOp::get_opid_static() {
        let mut state = resolve_tensor_view(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        if state.tile_size.is_some() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.tensor_zipped_divide must occur exactly once before tensor_slice",
            ));
        }
        let result_ty = value.get_type(ctx);
        let result_ty = result_ty.deref(ctx);
        let view = result_ty
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| invalid("cute.tensor_zipped_divide result is not a tensor view"))?;
        state.tile_size = view.layout.tile_size();
        return Ok(state);
    }
    if opid == CuteTensorSliceOp::get_opid_static() {
        let mut state = resolve_tensor_view(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        if state.tile_size.is_none() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.tensor_slice must follow exactly one zipped divide and may select only once",
            ));
        }
        state.tile_index = Some(defining_op.deref(ctx).get_operand(1));
        return Ok(state);
    }
    Err(invalid(format!(
        "tensor view is produced by unsupported operation `{opid}`"
    )))
}

fn verify_tensor_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = tensor_semantic_ids();
    let consumer_ids = &ids[3..];
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        let op = operation.deref(ctx);
        if opid == ids[0] || opid == ids[1] || opid == ids[2] {
            let result = op.get_result(0);
            let uses = result.uses(ctx);
            if uses.is_empty() {
                return Err(invalid(format!(
                    "tensor view from `{opid}` has no semantic consumer"
                )));
            }
            for r#use in uses {
                let user = r#use.user_op();
                let user_id = Operation::get_opid(user, ctx);
                let allowed = if opid == ids[0] {
                    user_id == ids[1] && r#use.find_index(ctx) == 0
                } else if opid == ids[1] {
                    user_id == ids[2] && r#use.find_index(ctx) == 0
                } else {
                    consumer_ids.contains(&user_id)
                };
                if !allowed {
                    return Err(invalid(format!(
                        "tensor view from `{opid}` has unsupported consumer `{user_id}`"
                    )));
                }
            }
        } else {
            let tensor = if opid == CuteTensorStoreFromOp::get_opid_static() {
                op.get_operand(1)
            } else {
                op.get_operand(0)
            };
            let state = resolve_tensor_view(ctx, tensor, 0)?;
            if state.tile_size.is_none() || state.tile_index.is_none() {
                return Err(invalid(format!(
                    "`{opid}` needs a complete make -> zipped_divide -> slice tensor story"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct GemvTensorState {
    rows: Value,
    k: Value,
    role: CuteTensorRoleAttr,
}

#[derive(Clone, Copy)]
struct GemvSelectionState {
    values: GemvTensorState,
    scales: GemvTensorState,
    batch: Option<Value>,
    row: Option<Value>,
    tile_index: Option<Value>,
}

fn resolve_gemv_tensor(ctx: &Context, value: Value) -> Result<GemvTensorState> {
    let Some(producer) = value.defining_op() else {
        return Err(invalid(
            "GEMV tensor reached a block argument; direct semantic SSA is required",
        ));
    };
    let opid = Operation::get_opid(producer, ctx);
    if opid != CuteTensorMake2DOp::get_opid_static() {
        return Err(invalid(format!(
            "GEMV tensor is produced by `{opid}` instead of `cute.tensor_make_2d`"
        )));
    }
    let op = producer.deref(ctx);
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let view = ty
        .downcast_ref::<CuteTensorViewType>()
        .ok_or_else(|| invalid("cute.tensor_make_2d result is not a tensor view"))?;
    Ok(GemvTensorState {
        rows: op.get_operand(2),
        k: op.get_operand(3),
        role: view.role,
    })
}

fn resolve_gemv_selection(ctx: &Context, value: Value, depth: usize) -> Result<GemvSelectionState> {
    if depth > 16 {
        return Err(invalid(
            "GEMV scaled-view producer chain is unexpectedly deep",
        ));
    }
    let Some(producer) = value.defining_op() else {
        return Err(invalid(
            "GEMV scaled view reached a block argument; direct semantic SSA is required",
        ));
    };
    let opid = Operation::get_opid(producer, ctx);
    let op = producer.deref(ctx);
    if opid == CuteScaledViewMakeOp::get_opid_static() {
        let values = resolve_gemv_tensor(ctx, op.get_operand(0))?;
        let scales = resolve_gemv_tensor(ctx, op.get_operand(1))?;
        if values.rows != scales.rows || values.k != scales.k {
            return Err(invalid(
                "cute.scaled_view_make values and scales must use the same runtime rows and K operands",
            ));
        }
        return Ok(GemvSelectionState {
            values,
            scales,
            batch: None,
            row: None,
            tile_index: None,
        });
    }
    if opid == CuteScaledViewRowOp::get_opid_static() {
        let mut state = resolve_gemv_selection(ctx, op.get_operand(0), depth + 1)?;
        if state.batch.is_some() || state.row.is_some() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.scaled_view_row must select one row before selecting a K tile",
            ));
        }
        state.batch = Some(op.get_operand(1));
        state.row = Some(op.get_operand(2));
        return Ok(state);
    }
    if opid == CuteScaledViewKTileOp::get_opid_static() {
        let mut state = resolve_gemv_selection(ctx, op.get_operand(0), depth + 1)?;
        if state.batch.is_none() || state.row.is_none() || state.tile_index.is_some() {
            return Err(invalid(
                "cute.scaled_view_k_tile must follow exactly one row selection",
            ));
        }
        state.tile_index = Some(op.get_operand(1));
        return Ok(state);
    }
    Err(invalid(format!(
        "scaled GEMV view is produced by unsupported operation `{opid}`"
    )))
}

fn resolve_gemv_fragment(ctx: &Context, value: Value) -> Result {
    let Some(load) = value.defining_op() else {
        return Err(invalid(
            "GEMV fragment reached a block argument; a direct load -> dot edge is required",
        ));
    };
    let opid = Operation::get_opid(load, ctx);
    if opid != CuteScaledViewLoadOp::get_opid_static() {
        return Err(invalid(format!(
            "GEMV fragment is produced by `{opid}` instead of `cute.scaled_view_load`"
        )));
    }
    let typed = CuteScaledViewLoadOp::wrap(load);
    let value_alignment = typed
        .promised_value_alignment(ctx)
        .ok_or_else(|| invalid("cute.scaled_view_load is missing its value alignment promise"))?;
    let scale_alignment = typed
        .promised_scale_alignment(ctx)
        .ok_or_else(|| invalid("cute.scaled_view_load is missing its scale alignment promise"))?;
    if value_alignment < 16 || scale_alignment < 4 {
        return Err(invalid(
            "GEMV K=64 load needs value alignment >= 16 and scale alignment >= 4",
        ));
    }
    let state = resolve_gemv_selection(ctx, load.deref(ctx).get_operand(0), 0)?;
    if state.batch.is_none() || state.row.is_none() || state.tile_index.is_none() {
        return Err(invalid("GEMV load needs a selected row and K tile"));
    }
    if state.values.role != state.scales.role {
        return Err(invalid("GEMV values and scales must keep the same role"));
    }
    Ok(())
}

fn require_one_user(ctx: &Context, producer: Ptr<Operation>, expected: &OpId) -> Result {
    let result = producer.deref(ctx).get_result(0);
    let uses = result.uses(ctx);
    if uses.len() != 1
        || Operation::get_opid(uses[0].user_op(), ctx) != *expected
        || uses[0].find_index(ctx) >= uses[0].user_op().deref(ctx).get_num_operands()
    {
        return Err(invalid(format!(
            "GEMV semantic value from `{}` must have exactly one `{expected}` consumer",
            Operation::get_opid(producer, ctx)
        )));
    }
    Ok(())
}

fn verify_gemv_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = gemv_semantic_ids();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        let expected = if opid == ids[0] {
            Some(&ids[1])
        } else if opid == ids[1] {
            Some(&ids[2])
        } else if opid == ids[2] {
            Some(&ids[3])
        } else if opid == ids[3] {
            Some(&ids[4])
        } else if opid == ids[4] {
            Some(&ids[5])
        } else {
            None
        };
        if let Some(expected) = expected {
            require_one_user(ctx, *operation, expected)?;
        } else if opid == ids[5] {
            let op = operation.deref(ctx);
            resolve_gemv_fragment(ctx, op.get_operand(0))?;
            resolve_gemv_fragment(ctx, op.get_operand(1))?;
        }
    }
    Ok(())
}

/// The semantic verifier treats every MIR function as an independent story
/// namespace.  Tests and tools may also place operations directly below a
/// module, in which case the verifier root is their namespace.
fn semantic_scope(
    ctx: &Context,
    root: Ptr<Operation>,
    operation: Ptr<Operation>,
) -> Ptr<Operation> {
    let mut current = operation;
    while let Some(parent) = current.deref(ctx).get_parent_op(ctx) {
        if Operation::get_opid(parent, ctx) == MirFuncOp::get_opid_static() {
            return parent;
        }
        if parent == root {
            return root;
        }
        current = parent;
    }
    root
}

fn operations_by_scope(
    ctx: &Context,
    root: Ptr<Operation>,
    all_ops: &[Ptr<Operation>],
) -> Vec<Vec<Ptr<Operation>>> {
    let mut scopes: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    for operation in all_ops {
        let scope = semantic_scope(ctx, root, *operation);
        scopes.entry(scope).or_default().push(*operation);
    }
    scopes.into_values().collect()
}

fn block_reaches(ctx: &Context, start: Ptr<BasicBlock>, destination: Ptr<BasicBlock>) -> bool {
    let Some(region) = start.deref(ctx).get_parent_region() else {
        return false;
    };
    if destination.deref(ctx).get_parent_region() != Some(region) {
        return false;
    }
    let mut worklist = vec![start];
    let mut visited = HashSet::new();
    while let Some(block) = worklist.pop() {
        if !visited.insert(block) {
            continue;
        }
        if block == destination {
            return true;
        }
        let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };
        worklist.extend(terminator.deref(ctx).successors());
    }
    false
}

fn operation_position(
    ctx: &Context,
    operation: Ptr<Operation>,
) -> Option<(Ptr<BasicBlock>, usize)> {
    let block = operation.deref(ctx).get_parent_block()?;
    let index = block
        .deref(ctx)
        .iter(ctx)
        .position(|candidate| candidate == operation)?;
    Some((block, index))
}

/// Existential CFG reachability with source order inside one block.  This is
/// used only to connect role-specific semantic milestones; each role's local
/// sequence is checked separately by `require_linear_protocol_order` where the
/// importer promises a straight-line call chain.
fn operation_reaches(ctx: &Context, source: Ptr<Operation>, destination: Ptr<Operation>) -> bool {
    let Some((source_block, source_index)) = operation_position(ctx, source) else {
        return false;
    };
    let Some((destination_block, destination_index)) = operation_position(ctx, destination) else {
        return false;
    };
    if source_block == destination_block {
        return source_index < destination_index;
    }
    let Some(terminator) = source_block.deref(ctx).get_terminator(ctx) else {
        return false;
    };
    terminator
        .deref(ctx)
        .successors()
        .any(|successor| block_reaches(ctx, successor, destination_block))
}

fn operation_dominates(ctx: &Context, source: Ptr<Operation>, destination: Ptr<Operation>) -> bool {
    let Some((source_block, source_index)) = operation_position(ctx, source) else {
        return false;
    };
    let Some((destination_block, destination_index)) = operation_position(ctx, destination) else {
        return false;
    };
    if source_block == destination_block {
        return source_index < destination_index;
    }
    let Some(region) = source_block.deref(ctx).get_parent_region() else {
        return false;
    };
    if destination_block.deref(ctx).get_parent_region() != Some(region) {
        return false;
    }
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    let Some(entry) = blocks.first().copied() else {
        return false;
    };
    let all: HashSet<_> = blocks.iter().copied().collect();
    let mut dominators: HashMap<Ptr<BasicBlock>, HashSet<Ptr<BasicBlock>>> = blocks
        .iter()
        .copied()
        .map(|block| {
            if block == entry {
                (block, HashSet::from([entry]))
            } else {
                (block, all.clone())
            }
        })
        .collect();

    loop {
        let mut changed = false;
        for block in blocks.iter().copied().filter(|block| *block != entry) {
            let predecessors: Vec<_> = block
                .uses(ctx)
                .into_iter()
                .filter_map(|edge| edge.user_op().deref(ctx).get_parent_block())
                .collect();
            let mut next = if let Some(first) = predecessors.first() {
                dominators.get(first).cloned().unwrap_or_default()
            } else {
                HashSet::new()
            };
            for predecessor in predecessors.iter().skip(1) {
                let predecessor_dominators =
                    dominators.get(predecessor).cloned().unwrap_or_default();
                next.retain(|candidate| predecessor_dominators.contains(candidate));
            }
            next.insert(block);
            if dominators.get(&block) != Some(&next) {
                dominators.insert(block, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dominators
        .get(&destination_block)
        .is_some_and(|set| set.contains(&source_block))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CarrierKey {
    value: Value,
    /// Empty means the value itself.  Non-empty paths describe a field inside
    /// an ordinary aggregate before a matching `mir.extract_field` exposes it.
    path: Vec<u32>,
}

#[derive(Clone)]
struct OrdinaryFacts<F> {
    facts: HashMap<CarrierKey, HashSet<F>>,
}

impl<F> Default for OrdinaryFacts<F> {
    fn default() -> Self {
        Self {
            facts: HashMap::new(),
        }
    }
}

impl<F> OrdinaryFacts<F>
where
    F: Clone + Eq + std::hash::Hash,
{
    fn insert_path(&mut self, value: Value, path: Vec<u32>, fact: F) -> bool {
        self.facts
            .entry(CarrierKey { value, path })
            .or_default()
            .insert(fact)
    }

    fn insert(&mut self, value: Value, fact: F) -> bool {
        self.insert_path(value, Vec::new(), fact)
    }

    fn at_path(&self, value: Value, path: &[u32]) -> Option<&HashSet<F>> {
        self.facts.get(&CarrierKey {
            value,
            path: path.to_vec(),
        })
    }

    fn at(&self, value: Value) -> Option<&HashSet<F>> {
        self.at_path(value, &[])
    }

    fn entries(&self, value: Value) -> Vec<(Vec<u32>, F)> {
        self.facts
            .iter()
            .filter(|(key, _)| key.value == value)
            .flat_map(|(key, facts)| facts.iter().cloned().map(|fact| (key.path.clone(), fact)))
            .collect()
    }

    fn copy_entries(
        &mut self,
        source: Value,
        destination: Value,
        source_prefix: &[u32],
        destination_prefix: &[u32],
    ) -> bool {
        let mut changed = false;
        for (path, fact) in self.entries(source) {
            let Some(suffix) = path.strip_prefix(source_prefix) else {
                continue;
            };
            let mut destination_path = destination_prefix.to_vec();
            destination_path.extend_from_slice(suffix);
            changed |= self.insert_path(destination, destination_path, fact);
        }
        changed
    }
}

fn branch_successor_operands(
    ctx: &Context,
    terminator: Ptr<Operation>,
    successor_index: usize,
) -> Option<Vec<Value>> {
    let dynamic = Operation::get_op_dyn(terminator, ctx);
    let branch = op_cast::<dyn BranchOpInterface>(dynamic.as_ref())?;
    Some(branch.successor_operands(ctx, successor_index))
}

/// A provenance-preserving cast keeps the pointee and never strengthens
/// mutability. Local-cell aliases must also keep their address space; semantic
/// shared-memory carriers may use the compiler's exact generic/shared
/// round-trip, but no other address-space transition is accepted.
fn is_provenance_preserving_pointer_cast(
    ctx: &Context,
    operation: Ptr<Operation>,
    allow_shared_generic_transition: bool,
) -> bool {
    if Operation::get_opid(operation, ctx) != MirCastOp::get_opid_static()
        || operation.deref(ctx).get_num_operands() != 1
        || operation.deref(ctx).get_num_results() != 1
        || !MirCastOp::new(operation)
            .get_attr_cast_kind(ctx)
            .is_some_and(|kind| matches!(*kind, MirCastKindAttr::PtrToPtr))
    {
        return false;
    }
    let operation_ref = operation.deref(ctx);
    let source_type = operation_ref.get_operand(0).get_type(ctx);
    let result_type = operation_ref.get_result(0).get_type(ctx);
    let source_type_ref = source_type.deref(ctx);
    let result_type_ref = result_type.deref(ctx);
    let (Some(source), Some(result)) = (
        source_type_ref.downcast_ref::<MirPtrType>(),
        result_type_ref.downcast_ref::<MirPtrType>(),
    ) else {
        return false;
    };
    let address_space_is_preserved = source.address_space == result.address_space
        || (allow_shared_generic_transition
            && matches!(
                (source.address_space, result.address_space),
                (address_space::GENERIC, address_space::SHARED)
                    | (address_space::SHARED, address_space::GENERIC)
            ));
    source.pointee == result.pointee
        && address_space_is_preserved
        && (source.is_mutable || !result.is_mutable)
}

fn is_local_cell_pointer_alias_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_provenance_preserving_pointer_cast(ctx, operation, false)
}

fn is_semantic_pointer_carrier_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_provenance_preserving_pointer_cast(ctx, operation, true)
}

/// Resolve a pointer to one compiler-created local cell and an exact constant
/// field path.  No pointer arithmetic, references, calls, or unknown casts are
/// accepted here.
fn local_cell(ctx: &Context, mut pointer: Value) -> Option<(Value, Vec<u32>)> {
    let mut reversed_path = Vec::new();
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(pointer) {
            return None;
        }
        let definer = pointer.defining_op()?;
        let opid = Operation::get_opid(definer, ctx);
        let operation = definer.deref(ctx);
        if opid == MirAllocaOp::get_opid_static() {
            return (operation.get_num_results() == 1 && operation.get_result(0) == pointer).then(
                || {
                    reversed_path.reverse();
                    (pointer, reversed_path)
                },
            );
        }
        if opid == MirCastOp::get_opid_static()
            && operation.get_num_operands() == 1
            && operation.get_num_results() == 1
            && operation.get_result(0) == pointer
            && is_local_cell_pointer_alias_cast(ctx, definer)
        {
            pointer = operation.get_operand(0);
            continue;
        }
        if opid == MirFieldAddrOp::get_opid_static()
            && operation.get_num_operands() == 1
            && operation.get_num_results() == 1
            && operation.get_result(0) == pointer
        {
            let index = MirFieldAddrOp::new(definer).get_attr_field_index(ctx)?;
            reversed_path.push(index.0);
            pointer = operation.get_operand(0);
            continue;
        }
        return None;
    }
}

/// Every alias of an accepted cell must stay within the same small
/// cast/field/load/store vocabulary.  This lets the verifier conservatively
/// merge all stores to the cell without guessing about hidden aliases.
fn local_cell_is_closed(ctx: &Context, root: Value) -> bool {
    let mut worklist = vec![root];
    let mut visited = HashSet::new();
    while let Some(pointer) = worklist.pop() {
        if !visited.insert(pointer) {
            continue;
        }
        for r#use in pointer.uses(ctx) {
            let user = r#use.user_op();
            let opid = Operation::get_opid(user, ctx);
            let operation = user.deref(ctx);
            if r#use.find_index(ctx) != 0 {
                return false;
            }
            let forwards_pointer = operation.get_num_results() == 1
                && ((opid == MirCastOp::get_opid_static()
                    && is_local_cell_pointer_alias_cast(ctx, user))
                    || (opid == MirFieldAddrOp::get_opid_static()
                        && MirFieldAddrOp::new(user)
                            .get_attr_field_index(ctx)
                            .is_some()));
            if forwards_pointer {
                worklist.push(operation.get_result(0));
            } else if opid == MirLoadOp::get_opid_static() {
                if MirLoadOp::new(user).is_volatile(ctx) {
                    return false;
                }
            } else if opid == MirStoreOp::get_opid_static() {
                if MirStoreOp::new(user).is_volatile(ctx) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    true
}

/// Propagate semantic facts through the exact ordinary SSA adapters that can
/// remain after common CuTe preparation.  This is deliberately a monotone
/// fixed-point transfer: loop block arguments become known once their entry
/// fact reaches them, then semantic advance operations can feed the same fact
/// around the backedge on a later iteration.
fn propagate_ordinary_plumbing<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
) -> bool
where
    F: Clone + Eq + std::hash::Hash,
{
    let mut changed = false;
    let mut blocks = HashSet::new();
    for operation in operations {
        let operation_ref = operation.deref(ctx);
        if let Some(block) = operation_ref.get_parent_block() {
            blocks.insert(block);
        }
        let opid = Operation::get_opid(*operation, ctx);
        if opid == MirCastOp::get_opid_static()
            && is_semantic_pointer_carrier_cast(ctx, *operation)
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            changed |= facts.copy_entries(
                operation_ref.get_operand(0),
                operation_ref.get_result(0),
                &[],
                &[],
            );
        } else if (opid == MirConstructStructOp::get_opid_static()
            || opid == MirConstructTupleOp::get_opid_static())
            && operation_ref.get_num_results() == 1
        {
            let result = operation_ref.get_result(0);
            for (index, operand) in operation_ref.operands().enumerate() {
                changed |= facts.copy_entries(operand, result, &[], &[index as u32]);
            }
        } else if opid == MirInsertFieldOp::get_opid_static()
            && operation_ref.get_num_operands() == 2
            && operation_ref.get_num_results() == 1
        {
            let Some(index) = MirInsertFieldOp::new(*operation).get_attr_insert_index(ctx) else {
                continue;
            };
            let base = operation_ref.get_operand(0);
            let inserted = operation_ref.get_operand(1);
            let result = operation_ref.get_result(0);
            for (path, fact) in facts.entries(base) {
                if path.first().copied() != Some(index.0) {
                    changed |= facts.insert_path(result, path, fact);
                }
            }
            changed |= facts.copy_entries(inserted, result, &[], &[index.0]);
        } else if opid == MirExtractFieldOp::get_opid_static()
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            let Some(index) = MirExtractFieldOp::new(*operation).get_attr_index(ctx) else {
                continue;
            };
            changed |= facts.copy_entries(
                operation_ref.get_operand(0),
                operation_ref.get_result(0),
                &[index.0],
                &[],
            );
        } else if opid == MirStoreOp::get_opid_static() && operation_ref.get_num_operands() == 2 {
            let store = MirStoreOp::new(*operation);
            if !store.is_volatile(ctx)
                && let Some((root, path)) = local_cell(ctx, store.address_opd(ctx))
                && local_cell_is_closed(ctx, root)
            {
                changed |= facts.copy_entries(store.value_opd(ctx), root, &[], &path);
            }
        } else if opid == MirLoadOp::get_opid_static()
            && operation_ref.get_num_operands() == 1
            && operation_ref.get_num_results() == 1
        {
            let load = MirLoadOp::new(*operation);
            if !load.is_volatile(ctx)
                && let Some((root, path)) = local_cell(ctx, load.address_opd(ctx))
                && local_cell_is_closed(ctx, root)
            {
                changed |= facts.copy_entries(root, operation_ref.get_result(0), &path, &[]);
            }
        }
    }

    for block in blocks {
        let block_ref = block.deref(ctx);
        for argument_index in 0..block_ref.get_num_arguments() {
            let argument = block_ref.get_argument(argument_index);
            for edge in block.uses(ctx) {
                let terminator = edge.user_op();
                let Some(operands) =
                    branch_successor_operands(ctx, terminator, edge.find_index(ctx))
                else {
                    continue;
                };
                if let Some(incoming) = operands.get(argument_index).copied() {
                    changed |= facts.copy_entries(incoming, argument, &[], &[]);
                }
            }
        }
    }
    changed
}

fn integer_leaf_paths(
    ctx: &Context,
    ty: TypeHandle,
    width: u32,
    prefix: &mut Vec<u32>,
    output: &mut Vec<Vec<u32>>,
) {
    let ty_ref = ty.deref(ctx);
    if ty_ref
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == width)
    {
        output.push(prefix.clone());
    } else if let Some(structure) = ty_ref.downcast_ref::<MirStructType>() {
        for (index, field) in structure.field_types.iter().copied().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, field, width, prefix, output);
            prefix.pop();
        }
    } else if let Some(tuple) = ty_ref.downcast_ref::<MirTupleType>() {
        for (index, field) in tuple.get_types().iter().copied().enumerate() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, field, width, prefix, output);
            prefix.pop();
        }
    } else if let Some(array) = ty_ref.downcast_ref::<MirArrayType>() {
        for index in 0..array.size() {
            let Ok(index) = u32::try_from(index) else {
                continue;
            };
            prefix.push(index);
            integer_leaf_paths(ctx, array.element_type(), width, prefix, output);
            prefix.pop();
        }
    }
}

fn seed_unknown_integer_carriers<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
    width: u32,
    unknown: impl Fn(Value) -> F,
) where
    F: Clone + Eq + std::hash::Hash,
{
    let mut values = Vec::new();
    let mut blocks = HashSet::new();
    for operation in operations {
        let operation_ref = operation.deref(ctx);
        values.extend(operation_ref.results());
        if let Some(block) = operation_ref.get_parent_block()
            && blocks.insert(block)
        {
            let block_ref = block.deref(ctx);
            values.extend(
                (0..block_ref.get_num_arguments()).map(|index| block_ref.get_argument(index)),
            );
        }
    }

    for value in values {
        let mut paths = Vec::new();
        integer_leaf_paths(ctx, value.get_type(ctx), width, &mut Vec::new(), &mut paths);
        for path in paths {
            if facts.at_path(value, &path).is_none() {
                facts.insert_path(value, path, unknown(value));
            }
        }
    }
}

fn poison_missing_forwarding_sources<F>(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &mut OrdinaryFacts<F>,
    unknown: impl Fn(Value) -> F + Copy,
) -> bool
where
    F: Clone + Eq + std::hash::Hash,
{
    let mut changed = false;
    let mut blocks = HashSet::new();
    for operation in operations {
        if let Some(block) = operation.deref(ctx).get_parent_block() {
            blocks.insert(block);
        }
        if Operation::get_opid(*operation, ctx) != MirStoreOp::get_opid_static() {
            continue;
        }
        let store = MirStoreOp::new(*operation);
        let Some((root, cell_path)) = local_cell(ctx, store.address_opd(ctx)) else {
            continue;
        };
        if !local_cell_is_closed(ctx, root) {
            continue;
        }
        for (path, _) in facts.entries(root) {
            let Some(source_path) = path.strip_prefix(cell_path.as_slice()) else {
                continue;
            };
            if facts.at_path(store.value_opd(ctx), source_path).is_none() {
                changed |= facts.insert_path(
                    store.value_opd(ctx),
                    source_path.to_vec(),
                    unknown(store.value_opd(ctx)),
                );
            }
        }
    }

    for block in blocks {
        let block_ref = block.deref(ctx);
        for argument_index in 0..block_ref.get_num_arguments() {
            let argument = block_ref.get_argument(argument_index);
            let argument_entries = facts.entries(argument);
            if argument_entries.is_empty() {
                continue;
            }
            for edge in block.uses(ctx) {
                let terminator = edge.user_op();
                let Some(operands) =
                    branch_successor_operands(ctx, terminator, edge.find_index(ctx))
                else {
                    continue;
                };
                let Some(incoming) = operands.get(argument_index).copied() else {
                    continue;
                };
                for (path, _) in &argument_entries {
                    if facts.at_path(incoming, path).is_none() {
                        changed |= facts.insert_path(incoming, path.clone(), unknown(incoming));
                    }
                }
            }
        }
    }
    changed
}

fn scheduler_grid(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Result<CuteTileGridAttr> {
    let grid = if *opid == CuteSchedulerNew1dOp::get_opid_static() {
        CuteSchedulerNew1dOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerHasWorkOp::get_opid_static() {
        CuteSchedulerHasWorkOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerCurrentOp::get_opid_static() {
        CuteSchedulerCurrentOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteSchedulerAdvanceOp::get_opid_static() {
        CuteSchedulerAdvanceOp::wrap(operation).tile_grid(ctx)
    } else if *opid == CuteWorkTileCoordinatesOp::get_opid_static() {
        CuteWorkTileCoordinatesOp::wrap(operation)
            .work_tile(ctx)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<CuteWorkTileType>()
            .map(|tile| tile.grid)
    } else {
        None
    };
    grid.ok_or_else(|| invalid(format!("`{opid}` is missing its scheduler tile grid")))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SchedulerFact {
    Current(Ptr<Operation>),
    Stride(Ptr<Operation>),
    Unknown(Value),
}

fn scheduler_root(
    facts: &OrdinaryFacts<SchedulerFact>,
    value: Value,
    current: bool,
    what: &str,
) -> Result<Ptr<Operation>> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no scheduler provenance after ordinary SSA forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges values from different scheduler stories"
        )));
    }
    match found.iter().next().copied() {
        Some(SchedulerFact::Current(root)) if current => Ok(root),
        Some(SchedulerFact::Stride(root)) if !current => Ok(root),
        _ => Err(invalid(format!(
            "{what} is not derived from the required scheduler_new_1d result"
        ))),
    }
}

fn build_scheduler_facts(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> OrdinaryFacts<SchedulerFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        if Operation::get_opid(*operation, ctx) == CuteSchedulerNew1dOp::get_opid_static() {
            let scheduler = CuteSchedulerNew1dOp::wrap(*operation);
            facts.insert(scheduler.current(ctx), SchedulerFact::Current(*operation));
            facts.insert(scheduler.stride(ctx), SchedulerFact::Stride(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CuteSchedulerAdvanceOp::get_opid_static() {
                continue;
            }
            let advance = CuteSchedulerAdvanceOp::wrap(*operation);
            let Some(current_facts) = facts.at(advance.current(ctx)).cloned() else {
                continue;
            };
            let Some(stride_facts) = facts.at(advance.stride(ctx)).cloned() else {
                continue;
            };
            for current in &current_facts {
                let SchedulerFact::Current(root) = current else {
                    continue;
                };
                if stride_facts.contains(&SchedulerFact::Stride(*root)) {
                    changed |= facts.insert(advance.next(ctx), SchedulerFact::Current(*root));
                }
            }
        }
        if !changed {
            break;
        }
    }

    // A factless scalar is an ordinary, unrelated value.  Seeding it only
    // after the semantic fixed point lets a valid loop recurrence establish
    // itself first, while still poisoning a merge with an arbitrary edge.
    seed_unknown_integer_carriers(ctx, operations, &mut facts, 64, SchedulerFact::Unknown);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, SchedulerFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

fn scheduler_guard(
    ctx: &Context,
    has_work: Value,
) -> Option<(Ptr<Operation>, Ptr<BasicBlock>, Ptr<BasicBlock>)> {
    let mut value = has_work;
    let mut negated = false;
    loop {
        let uses = value.uses(ctx);
        if uses.len() != 1 || uses[0].find_index(ctx) != 0 {
            return None;
        }
        let user = uses[0].user_op();
        let opid = Operation::get_opid(user, ctx);
        if opid == MirNotOp::get_opid_static() {
            let operation = user.deref(ctx);
            if operation.get_num_operands() != 1 || operation.get_num_results() != 1 {
                return None;
            }
            value = operation.get_result(0);
            negated = !negated;
            continue;
        }
        if opid != MirCondBranchOp::get_opid_static() {
            return None;
        }
        let operation = user.deref(ctx);
        if operation.get_num_successors() != 2 || operation.get_operand(0) != value {
            return None;
        }
        let work_index = usize::from(negated);
        return Some((
            user,
            operation.get_successor(work_index),
            operation.get_successor(1 - work_index),
        ));
    }
}

fn verify_scheduler_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = scheduler_semantic_ids();
    let facts = build_scheduler_facts(ctx, all_ops);
    let mut roots = HashSet::new();
    let mut has_work_roots = HashSet::new();
    let mut current_roots = HashSet::new();
    let mut advance_roots = HashSet::new();
    let mut coordinate_roots = HashSet::new();
    let mut has_work_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut current_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut advance_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut coordinates_by_root: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut currents = Vec::new();
    let mut coordinates = Vec::new();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "scheduler operation `{opid}` is not attached to a block"
            )));
        }
        let grid = scheduler_grid(ctx, *operation, &opid)?;
        if opid == CuteSchedulerNew1dOp::get_opid_static() {
            roots.insert(*operation);
        } else if opid == CuteSchedulerHasWorkOp::get_opid_static() {
            let semantic = CuteSchedulerHasWorkOp::wrap(*operation);
            let root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_has_work current",
            )?;
            if CuteSchedulerNew1dOp::wrap(root).tile_grid(ctx) != Some(grid) {
                return Err(invalid(
                    "cute.scheduler_has_work grid differs from its scheduler_new_1d root",
                ));
            }
            if !semantic.has_work(ctx).is_used(ctx) {
                return Err(invalid(
                    "cute.scheduler_has_work result is unused; it must guard the scheduler body",
                ));
            }
            has_work_roots.insert(root);
            has_work_by_root.entry(root).or_default().push(*operation);
        } else if opid == CuteSchedulerCurrentOp::get_opid_static() {
            let semantic = CuteSchedulerCurrentOp::wrap(*operation);
            let root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_current current",
            )?;
            if CuteSchedulerNew1dOp::wrap(root).tile_grid(ctx) != Some(grid) {
                return Err(invalid(
                    "cute.scheduler_current grid differs from its scheduler_new_1d root",
                ));
            }
            current_roots.insert(root);
            current_by_root.entry(root).or_default().push(*operation);
            currents.push(*operation);
        } else if opid == CuteWorkTileCoordinatesOp::get_opid_static() {
            coordinates.push(*operation);
        } else if opid == CuteSchedulerAdvanceOp::get_opid_static() {
            let semantic = CuteSchedulerAdvanceOp::wrap(*operation);
            let current_root = scheduler_root(
                &facts,
                semantic.current(ctx),
                true,
                "cute.scheduler_advance current",
            )?;
            let stride_root = scheduler_root(
                &facts,
                semantic.stride(ctx),
                false,
                "cute.scheduler_advance stride",
            )?;
            if current_root != stride_root
                || CuteSchedulerNew1dOp::wrap(current_root).tile_grid(ctx) != Some(grid)
            {
                return Err(invalid(
                    "cute.scheduler_advance must use current, stride, and grid from one scheduler_new_1d",
                ));
            }
            advance_roots.insert(current_root);
            advance_by_root
                .entry(current_root)
                .or_default()
                .push(*operation);
        }
    }

    for current in &currents {
        let tile = CuteSchedulerCurrentOp::wrap(*current).work_tile(ctx);
        let uses = tile.uses(ctx);
        if uses.is_empty()
            || uses.iter().any(|r#use| {
                Operation::get_opid(r#use.user_op(), ctx)
                    != CuteWorkTileCoordinatesOp::get_opid_static()
                    || r#use.find_index(ctx) != 0
            })
        {
            return Err(invalid(
                "cute.scheduler_current work tile must be used only by at least one cute.work_tile_coordinates",
            ));
        }
    }
    for coordinates in coordinates {
        let tile = CuteWorkTileCoordinatesOp::wrap(coordinates).work_tile(ctx);
        let Some(producer) = tile.defining_op() else {
            return Err(invalid(
                "cute.work_tile_coordinates reached a block argument; a direct scheduler_current result is required",
            ));
        };
        if Operation::get_opid(producer, ctx) != CuteSchedulerCurrentOp::get_opid_static()
            || !currents.contains(&producer)
        {
            return Err(invalid(
                "cute.work_tile_coordinates must use a scheduler_current attached to this module",
            ));
        }
        let root = scheduler_root(
            &facts,
            CuteSchedulerCurrentOp::wrap(producer).current(ctx),
            true,
            "cute.work_tile_coordinates scheduler current",
        )?;
        coordinate_roots.insert(root);
        coordinates_by_root
            .entry(root)
            .or_default()
            .push(coordinates);
    }

    for root in roots {
        if !has_work_roots.contains(&root)
            || !current_roots.contains(&root)
            || !coordinate_roots.contains(&root)
            || !advance_roots.contains(&root)
        {
            return Err(invalid(
                "each cute.scheduler_new_1d needs a complete has_work -> current -> coordinates -> advance story",
            ));
        }
        let has_work = &has_work_by_root[&root];
        let current = &current_by_root[&root];
        let coordinates = &coordinates_by_root[&root];
        let advances = &advance_by_root[&root];
        if has_work.len() != 1
            || current.len() != 1
            || coordinates.len() != 1
            || advances.len() != 1
        {
            return Err(invalid(
                "each scheduler root needs exactly one static has_work/current/coordinates/advance protocol",
            ));
        }
        let Some((guard, work_successor, exit_successor)) =
            scheduler_guard(ctx, CuteSchedulerHasWorkOp::wrap(has_work[0]).has_work(ctx))
        else {
            return Err(invalid(
                "cute.scheduler_has_work must exclusively guard one conditional branch (optionally through mir.not)",
            ));
        };
        let current_block = current[0]
            .deref(ctx)
            .get_parent_block()
            .expect("attached scheduler current");
        if !operation_reaches(ctx, root, has_work[0])
            || !block_reaches(ctx, work_successor, current_block)
            || block_reaches(ctx, exit_successor, current_block)
            || !operation_reaches(ctx, current[0], coordinates[0])
            || !operation_reaches(ctx, coordinates[0], advances[0])
            || !operation_reaches(ctx, advances[0], guard)
        {
            return Err(invalid(
                "scheduler CFG must be new -> guarded work current -> coordinates -> advance -> next guard, with the exit edge outside the work body",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LoadPipelineFacts {
    make: Ptr<Operation>,
    stages: u64,
    transaction_bytes: u32,
}

fn resolve_load_pipeline(ctx: &Context, value: Value) -> Result<LoadPipelineFacts> {
    let Some(make) = value.defining_op() else {
        return Err(invalid(
            "TMA load-pipeline handle reached a block argument; a direct tma_load_pipeline_make result is required",
        ));
    };
    let producer_id = Operation::get_opid(make, ctx);
    if producer_id != CuteTmaLoadPipelineMakeOp::get_opid_static() {
        return Err(invalid(format!(
            "TMA load-pipeline handle is produced by `{producer_id}` instead of `cute.tma_load_pipeline_make`"
        )));
    }
    let make_op = CuteTmaLoadPipelineMakeOp::wrap(make);
    if make_op.pipeline(ctx) != value {
        return Err(invalid(
            "TMA load-pipeline operand is not the make operation's pipeline result",
        ));
    }
    let pipeline = make_op
        .pipeline_type(ctx)
        .ok_or_else(|| invalid("TMA load-pipeline make has no pipeline type"))?;
    Ok(LoadPipelineFacts {
        make,
        stages: pipeline.stages,
        transaction_bytes: pipeline.transaction_bytes,
    })
}

fn pipeline_state_for_op(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Option<CutePipelineStateAttr> {
    if *opid == CutePipelineStateNewOp::get_opid_static() {
        CutePipelineStateNewOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineStateSlotOp::get_opid_static() {
        CutePipelineStateSlotOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineStateAdvanceOp::get_opid_static() {
        CutePipelineStateAdvanceOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerAcquireOp::get_opid_static() {
        CutePipelineProducerAcquireOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        CutePipelineProducerExpectTxOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineConsumerWaitOp::get_opid_static() {
        CutePipelineConsumerWaitOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        CutePipelineConsumerReleaseOp::wrap(operation).state(ctx)
    } else if *opid == CutePipelineProducerTailOp::get_opid_static() {
        CutePipelineProducerTailOp::wrap(operation).state(ctx)
    } else {
        None
    }
}

fn pipeline_operand(ctx: &Context, operation: Ptr<Operation>, opid: &OpId) -> Option<Value> {
    if *opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
        Some(CuteTmaLoadPipelineInitOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerAcquireOp::get_opid_static() {
        Some(CutePipelineProducerAcquireOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        Some(CutePipelineProducerExpectTxOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineConsumerWaitOp::get_opid_static() {
        Some(CutePipelineConsumerWaitOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        Some(CutePipelineConsumerReleaseOp::wrap(operation).pipeline(ctx))
    } else if *opid == CutePipelineProducerTailOp::get_opid_static() {
        Some(CutePipelineProducerTailOp::wrap(operation).pipeline(ctx))
    } else {
        None
    }
}

/// Sum the static byte sizes of every direct TMA copy completed by one
/// `expect_tx` result. The result may not escape through any other operand.
fn expected_tma_transaction_bytes(ctx: &Context, operation: Ptr<Operation>) -> Result<u64> {
    let barrier = CutePipelineProducerExpectTxOp::wrap(operation).completion_barrier(ctx);
    let uses = barrier.uses(ctx);
    if uses.is_empty() {
        return Err(invalid(
            "cute.pipeline_producer_expect_tx needs at least one direct cute.tma_copy_2d user",
        ));
    }
    let mut copies = HashSet::new();
    let mut bytes = 0u64;
    for r#use in uses {
        let user = r#use.user_op();
        if Operation::get_opid(user, ctx) != CuteTmaCopy2dOp::get_opid_static()
            || r#use.find_index(ctx) != 4
            || CuteTmaCopy2dOp::wrap(user).completion_barrier(ctx) != barrier
        {
            return Err(invalid(
                "cute.pipeline_producer_expect_tx result may only be the completion-barrier operand of cute.tma_copy_2d",
            ));
        }
        if !copies.insert(user) {
            continue;
        }
        let copy = CuteTmaCopy2dOp::wrap(user);
        let source_ty = copy.source(ctx).get_type(ctx);
        let source_ty = source_ty.deref(ctx);
        let view = source_ty
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| invalid("TMA copy attached to expect_tx has no TMA view type"))?;
        let tile_bytes = view.tile_bytes(ctx).ok_or_else(|| {
            invalid("TMA copy attached to expect_tx has no static tile byte count")
        })?;
        bytes = bytes
            .checked_add(tile_bytes)
            .ok_or_else(|| invalid("TMA transaction byte total overflows u64"))?;
    }
    Ok(bytes)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineFact {
    Slot(Ptr<Operation>),
    Phase(Ptr<Operation>),
    Unknown(Value),
}

fn pipeline_state_root(
    facts: &OrdinaryFacts<PipelineFact>,
    value: Value,
    slot: bool,
    what: &str,
) -> Result<Ptr<Operation>> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no pipeline-state provenance after ordinary SSA forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges values from different pipeline-state stories"
        )));
    }
    match found.iter().next().copied() {
        Some(PipelineFact::Slot(root)) if slot => Ok(root),
        Some(PipelineFact::Phase(root)) if !slot => Ok(root),
        _ => Err(invalid(format!(
            "{what} is not derived from the required cute.pipeline_state_new result"
        ))),
    }
}

fn build_pipeline_facts(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> OrdinaryFacts<PipelineFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        if Operation::get_opid(*operation, ctx) == CutePipelineStateNewOp::get_opid_static() {
            let state = CutePipelineStateNewOp::wrap(*operation);
            facts.insert(state.slot(ctx), PipelineFact::Slot(*operation));
            facts.insert(state.phase(ctx), PipelineFact::Phase(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CutePipelineStateAdvanceOp::get_opid_static()
            {
                continue;
            }
            let advance = CutePipelineStateAdvanceOp::wrap(*operation);
            let Some(slot_facts) = facts.at(advance.slot(ctx)).cloned() else {
                continue;
            };
            let Some(phase_facts) = facts.at(advance.phase(ctx)).cloned() else {
                continue;
            };
            for slot in &slot_facts {
                let PipelineFact::Slot(root) = slot else {
                    continue;
                };
                if phase_facts.contains(&PipelineFact::Phase(*root))
                    && CutePipelineStateNewOp::wrap(*root).state(ctx) == advance.state(ctx)
                {
                    changed |= facts.insert(advance.next_slot(ctx), PipelineFact::Slot(*root));
                    changed |= facts.insert(advance.next_phase(ctx), PipelineFact::Phase(*root));
                }
            }
        }
        if !changed {
            break;
        }
    }

    seed_unknown_integer_carriers(ctx, operations, &mut facts, 32, PipelineFact::Unknown);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, PipelineFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

#[derive(Default)]
struct PipelineStateEvents {
    slots: Vec<Ptr<Operation>>,
    advances: Vec<Ptr<Operation>>,
    acquires: Vec<Ptr<Operation>>,
    expects: Vec<Ptr<Operation>>,
    waits: Vec<Ptr<Operation>>,
    releases: Vec<Ptr<Operation>>,
    tails: Vec<Ptr<Operation>>,
}

fn require_producer_pipeline_order(
    ctx: &Context,
    root: Ptr<Operation>,
    acquire: Ptr<Operation>,
    expect: Ptr<Operation>,
    advance: Ptr<Operation>,
    tail: Ptr<Operation>,
) -> Result {
    if operation_dominates(ctx, root, acquire)
        && operation_dominates(ctx, acquire, expect)
        && operation_dominates(ctx, expect, advance)
        && operation_reaches(ctx, advance, tail)
    {
        Ok(())
    } else {
        Err(invalid(
            "producer pipeline CFG must order state_new -> acquire -> expect_tx -> advance, with tail reachable only after the producer loop",
        ))
    }
}

fn verify_pipeline_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = pipeline_semantic_ids();
    let facts = build_pipeline_facts(ctx, all_ops);
    let makes: Vec<_> = all_ops
        .iter()
        .copied()
        .filter(|operation| {
            Operation::get_opid(*operation, ctx) == CuteTmaLoadPipelineMakeOp::get_opid_static()
        })
        .collect();
    let mut init_by_make: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
    let mut events: HashMap<Ptr<Operation>, PipelineStateEvents> = HashMap::new();
    let mut binding: HashMap<Ptr<Operation>, Ptr<Operation>> = HashMap::new();

    for make in &makes {
        if make.deref(ctx).get_parent_block().is_none() {
            return Err(invalid("TMA load-pipeline make is not attached to a block"));
        }
        let handle = CuteTmaLoadPipelineMakeOp::wrap(*make).pipeline(ctx);
        let uses = handle.uses(ctx);
        if uses.is_empty() {
            return Err(invalid(
                "cute.tma_load_pipeline_make has no lifecycle consumer",
            ));
        }
        for r#use in uses {
            let user = r#use.user_op();
            let user_id = Operation::get_opid(user, ctx);
            if !ids.contains(&user_id)
                || r#use.find_index(ctx) != 0
                || pipeline_operand(ctx, user, &user_id) != Some(handle)
            {
                return Err(invalid(format!(
                    "TMA load-pipeline handle has an invalid use by `{user_id}`"
                )));
            }
        }
    }

    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "pipeline operation `{opid}` is not attached to a block"
            )));
        }
        if opid == CutePipelineStateNewOp::get_opid_static() {
            events.entry(*operation).or_default();
            continue;
        }

        let state_root = if opid == CutePipelineStateSlotOp::get_opid_static() {
            let semantic = CutePipelineStateSlotOp::wrap(*operation);
            let root = pipeline_state_root(
                &facts,
                semantic.slot(ctx),
                true,
                "cute.pipeline_state_slot slot",
            )?;
            if semantic.state(ctx) != CutePipelineStateNewOp::wrap(root).state(ctx) {
                return Err(invalid(
                    "cute.pipeline_state_slot facts differ from its pipeline_state_new root",
                ));
            }
            events.entry(root).or_default().slots.push(*operation);
            Some(root)
        } else if opid == CutePipelineStateAdvanceOp::get_opid_static() {
            let semantic = CutePipelineStateAdvanceOp::wrap(*operation);
            let slot_root = pipeline_state_root(
                &facts,
                semantic.slot(ctx),
                true,
                "cute.pipeline_state_advance slot",
            )?;
            let phase_root = pipeline_state_root(
                &facts,
                semantic.phase(ctx),
                false,
                "cute.pipeline_state_advance phase",
            )?;
            if slot_root != phase_root
                || semantic.state(ctx) != CutePipelineStateNewOp::wrap(slot_root).state(ctx)
            {
                return Err(invalid(
                    "cute.pipeline_state_advance must consume slot and phase from one matching pipeline_state_new",
                ));
            }
            events
                .entry(slot_root)
                .or_default()
                .advances
                .push(*operation);
            Some(slot_root)
        } else if opid == CuteTmaLoadPipelineMakeOp::get_opid_static() {
            None
        } else if opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
            let pipeline = resolve_load_pipeline(
                ctx,
                CuteTmaLoadPipelineInitOp::wrap(*operation).pipeline(ctx),
            )?;
            if !makes.contains(&pipeline.make) {
                return Err(invalid(
                    "pipeline init refers to a make outside its semantic scope",
                ));
            }
            init_by_make
                .entry(pipeline.make)
                .or_default()
                .push(*operation);
            None
        } else {
            let state = pipeline_state_for_op(ctx, *operation, &opid)
                .ok_or_else(|| invalid(format!("pipeline operation `{opid}` has no state")))?;
            let operation_ref = operation.deref(ctx);
            let slot_root = pipeline_state_root(
                &facts,
                operation_ref.get_operand(1),
                true,
                &format!("`{opid}` slot"),
            )?;
            let needs_phase = opid == CutePipelineProducerAcquireOp::get_opid_static()
                || opid == CutePipelineConsumerWaitOp::get_opid_static()
                || opid == CutePipelineProducerTailOp::get_opid_static();
            if needs_phase {
                let phase_root = pipeline_state_root(
                    &facts,
                    operation_ref.get_operand(2),
                    false,
                    &format!("`{opid}` phase"),
                )?;
                if phase_root != slot_root {
                    return Err(invalid(format!(
                        "`{opid}` mixes slot and phase from different pipeline states"
                    )));
                }
            }
            if CutePipelineStateNewOp::wrap(slot_root).state(ctx) != Some(state) {
                return Err(invalid(format!(
                    "`{opid}` state facts differ from its pipeline_state_new root"
                )));
            }

            let handle = pipeline_operand(ctx, *operation, &opid)
                .ok_or_else(|| invalid(format!("`{opid}` has no load-pipeline handle")))?;
            let pipeline = resolve_load_pipeline(ctx, handle)?;
            if !makes.contains(&pipeline.make) {
                return Err(invalid(
                    "pipeline lifecycle operation refers to a make outside its semantic scope",
                ));
            }
            if state.stages != pipeline.stages {
                return Err(invalid(format!(
                    "pipeline state on `{opid}` has {} stages but its handle has {}",
                    state.stages, pipeline.stages
                )));
            }
            if let Some(previous) = binding.insert(slot_root, pipeline.make)
                && previous != pipeline.make
            {
                return Err(invalid(
                    "one ordinary pipeline state is used with multiple load-pipeline handles",
                ));
            }

            let root_events = events.entry(slot_root).or_default();
            if opid == CutePipelineProducerAcquireOp::get_opid_static() {
                root_events.acquires.push(*operation);
            } else if opid == CutePipelineProducerExpectTxOp::get_opid_static() {
                root_events.expects.push(*operation);
                let copied = expected_tma_transaction_bytes(ctx, *operation)?;
                if copied != u64::from(pipeline.transaction_bytes) {
                    return Err(invalid(format!(
                        "cute.pipeline_producer_expect_tx promises {} bytes, but its direct TMA copies move {copied} bytes",
                        pipeline.transaction_bytes
                    )));
                }
            } else if opid == CutePipelineConsumerWaitOp::get_opid_static() {
                root_events.waits.push(*operation);
            } else if opid == CutePipelineConsumerReleaseOp::get_opid_static() {
                root_events.releases.push(*operation);
            } else if opid == CutePipelineProducerTailOp::get_opid_static() {
                root_events.tails.push(*operation);
            }
            Some(slot_root)
        };

        if let Some(root) = state_root {
            let root_state = CutePipelineStateNewOp::wrap(root)
                .state(ctx)
                .expect("locally verified pipeline state root");
            if let Some(state) = pipeline_state_for_op(ctx, *operation, &opid)
                && state != root_state
            {
                return Err(invalid(format!(
                    "`{opid}` does not carry its pipeline-state root's role and stage count"
                )));
            }
        }
    }

    let mut roles_by_make: HashMap<Ptr<Operation>, (usize, usize)> = HashMap::new();
    for (root, root_events) in &events {
        let state = CutePipelineStateNewOp::wrap(*root)
            .state(ctx)
            .expect("locally verified pipeline state root");
        let Some(make) = binding.get(root).copied() else {
            return Err(invalid(
                "cute.pipeline_state_new is not connected to any load-pipeline lifecycle",
            ));
        };
        if root_events.slots.is_empty() || root_events.advances.is_empty() {
            return Err(invalid(
                "each pipeline state needs state_slot and state_advance operations",
            ));
        }
        let roles = roles_by_make.entry(make).or_default();
        match state.role {
            CutePipelineRoleAttr::Producer => {
                roles.0 += 1;
                if root_events.acquires.len() != 1
                    || root_events.expects.len() != 1
                    || root_events.advances.len() != 1
                    || root_events.slots.len() != 1
                    || root_events.tails.len() != 1
                    || !root_events.waits.is_empty()
                    || !root_events.releases.is_empty()
                {
                    return Err(invalid(
                        "producer pipeline state needs acquire, expect_tx, advance, and exactly one tail, with no consumer events",
                    ));
                }
                let acquire = root_events.acquires[0];
                let expect = root_events.expects[0];
                let advance = root_events.advances[0];
                let tail = root_events.tails[0];
                require_producer_pipeline_order(ctx, *root, acquire, expect, advance, tail)?;
                let barrier = CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(ctx);
                for r#use in barrier.uses(ctx) {
                    let copy = r#use.user_op();
                    if !operation_dominates(ctx, expect, copy)
                        || !operation_dominates(ctx, copy, advance)
                    {
                        return Err(invalid(
                            "every TMA copy must be CFG-ordered after expect_tx and before producer state advance",
                        ));
                    }
                }
            }
            CutePipelineRoleAttr::Consumer => {
                roles.1 += 1;
                if root_events.waits.len() != 1
                    || root_events.releases.len() != 1
                    || root_events.advances.len() != 1
                    || root_events.slots.len() != 1
                    || !root_events.acquires.is_empty()
                    || !root_events.expects.is_empty()
                    || !root_events.tails.is_empty()
                {
                    return Err(invalid(
                        "consumer pipeline state needs wait, release, and advance, with no producer events",
                    ));
                }
                let wait = root_events.waits[0];
                let release = root_events.releases[0];
                let advance = root_events.advances[0];
                if !operation_dominates(ctx, *root, wait)
                    || !operation_dominates(ctx, wait, release)
                    || !operation_dominates(ctx, release, advance)
                    || !operation_reaches(ctx, advance, wait)
                {
                    return Err(invalid(
                        "consumer pipeline CFG must order state_new -> wait -> release -> advance -> next wait",
                    ));
                }
            }
        }
    }
    for make in makes {
        if init_by_make
            .get(&make)
            .is_none_or(|initializers| initializers.len() != 1)
        {
            return Err(invalid(
                "each tma_load_pipeline_make needs exactly one initialization",
            ));
        }
        if roles_by_make.get(&make).copied() != Some((1, 1)) {
            return Err(invalid(
                "each tma_load_pipeline_make needs exactly one complete producer state and one complete consumer state",
            ));
        }
        let initializer = init_by_make[&make][0];
        for (root, bound_make) in &binding {
            if *bound_make == make && !operation_dominates(ctx, initializer, *root) {
                return Err(invalid(
                    "load-pipeline initialization must dominate both producer and consumer state creation",
                ));
            }
        }
    }
    Ok(())
}

fn direct_tma_view_producer(
    ctx: &Context,
    value: Value,
    expected: &OpId,
    what: &str,
    scope: &[Ptr<Operation>],
) -> Result<Ptr<Operation>> {
    let Some(producer) = value.defining_op() else {
        return Err(invalid(format!(
            "semantic TMA {what} reached a block argument; a direct view producer is required"
        )));
    };
    let actual = Operation::get_opid(producer, ctx);
    if actual != *expected {
        return Err(invalid(format!(
            "semantic TMA {what} is produced by `{actual}` instead of `{expected}`"
        )));
    }
    if !scope.contains(&producer) {
        return Err(invalid(format!(
            "semantic TMA {what} is produced outside its function/story scope"
        )));
    }
    Ok(producer)
}

fn verify_tma_view_uses(ctx: &Context, producer: Ptr<Operation>, gmem: bool) -> Result {
    let value = producer.deref(ctx).get_result(0);
    let uses = value.uses(ctx);
    if uses.is_empty() {
        return Err(invalid(format!(
            "semantic TMA view from `{}` has no copy or store consumer",
            Operation::get_opid(producer, ctx)
        )));
    }
    for r#use in uses {
        let user = r#use.user_op();
        let user_id = Operation::get_opid(user, ctx);
        let index = r#use.find_index(ctx);
        let allowed = if gmem {
            (user_id == CuteTmaCopy2dOp::get_opid_static() && index == 0)
                || (user_id == CuteTmaStore2dSemanticOp::get_opid_static() && index == 1)
        } else {
            (user_id == CuteTmaCopy2dOp::get_opid_static() && index == 1)
                || (user_id == CuteTmaStore2dSemanticOp::get_opid_static() && index == 0)
        };
        if !allowed {
            return Err(invalid(format!(
                "semantic TMA view from `{}` has unsupported use by `{user_id}` operand {index}",
                Operation::get_opid(producer, ctx)
            )));
        }
    }
    Ok(())
}

fn verify_tma_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if opid == CuteTmaGmemViewOp::get_opid_static() {
            verify_tma_view_uses(ctx, *operation, true)?;
        } else if opid == CuteTmaSmemViewOp::get_opid_static() {
            verify_tma_view_uses(ctx, *operation, false)?;
        } else if opid == CuteTmaCopy2dOp::get_opid_static() {
            let copy = CuteTmaCopy2dOp::wrap(*operation);
            direct_tma_view_producer(
                ctx,
                copy.source(ctx),
                &CuteTmaGmemViewOp::get_opid_static(),
                "load source",
                all_ops,
            )?;
            direct_tma_view_producer(
                ctx,
                copy.destination(ctx),
                &CuteTmaSmemViewOp::get_opid_static(),
                "load destination",
                all_ops,
            )?;
            let barrier = copy.completion_barrier(ctx);
            let Some(expect) = barrier.defining_op() else {
                return Err(invalid(
                    "cute.tma_copy_2d completion barrier must come directly from pipeline_producer_expect_tx",
                ));
            };
            if Operation::get_opid(expect, ctx) != CutePipelineProducerExpectTxOp::get_opid_static()
                || !all_ops.contains(&expect)
                || CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(ctx) != barrier
            {
                return Err(invalid(
                    "cute.tma_copy_2d completion barrier is not the result of an expect_tx in this semantic scope",
                ));
            }
        } else if opid == CuteTmaStore2dSemanticOp::get_opid_static() {
            let store = CuteTmaStore2dSemanticOp::wrap(*operation);
            direct_tma_view_producer(
                ctx,
                store.source(ctx),
                &CuteTmaSmemViewOp::get_opid_static(),
                "store source",
                all_ops,
            )?;
            direct_tma_view_producer(
                ctx,
                store.destination(ctx),
                &CuteTmaGmemViewOp::get_opid_static(),
                "store destination",
                all_ops,
            )?;
        }
    }
    Ok(())
}

fn mma_plan_for_op(
    ctx: &Context,
    operation: Ptr<Operation>,
    opid: &OpId,
) -> Option<CuteTiledMmaPlanAttr> {
    if *opid == CuteTiledMmaSliceOp::get_opid_static() {
        CuteTiledMmaSliceOp::wrap(operation).plan(ctx)
    } else if *opid == CuteFragmentFillOp::get_opid_static() {
        CuteFragmentFillOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaLoadScalesOp::get_opid_static() {
        CuteMmaLoadScalesOp::wrap(operation).plan(ctx)
    } else if *opid == CuteFragmentSliceKOp::get_opid_static() {
        CuteFragmentSliceKOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaLoadAOp::get_opid_static() {
        CuteMmaLoadAOp::wrap(operation).plan(ctx)
    } else if *opid == CuteMmaPartitionBOp::get_opid_static() {
        CuteMmaPartitionBOp::wrap(operation).plan(ctx)
    } else if *opid == CuteTiledGemmOp::get_opid_static() {
        CuteTiledGemmOp::wrap(operation).plan(ctx)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MmaFact {
    Lane(Ptr<Operation>),
    SmemBase(Ptr<Operation>),
    SmemCapacity(Ptr<Operation>),
    StageScales(Ptr<Operation>),
    SlicedScales(Ptr<Operation>),
    AFragment(Ptr<Operation>),
    BBase(Ptr<Operation>),
    BCapacity(Ptr<Operation>),
    BWarpN(Ptr<Operation>),
    BKHalf(Ptr<Operation>),
    Accumulator(Ptr<Operation>),
    Scalar(Value),
    Unknown(Value),
}

fn one_mma_fact(facts: &OrdinaryFacts<MmaFact>, value: Value, what: &str) -> Result<MmaFact> {
    let Some(found) = facts.at(value) else {
        return Err(invalid(format!(
            "{what} has no shared-MMA provenance after ordinary carrier forwarding"
        )));
    };
    if found.len() != 1 {
        return Err(invalid(format!(
            "{what} merges carriers from different shared-MMA stories"
        )));
    }
    Ok(*found.iter().next().expect("one fact"))
}

fn require_mma_lane(
    facts: &OrdinaryFacts<MmaFact>,
    value: Value,
    what: &str,
) -> Result<Ptr<Operation>> {
    match one_mma_fact(facts, value, what)? {
        MmaFact::Lane(root) => Ok(root),
        _ => Err(invalid(format!(
            "{what} must come from cute.tiled_mma_slice"
        ))),
    }
}

fn require_smem_overlay_pair(
    facts: &OrdinaryFacts<MmaFact>,
    base: Value,
    capacity: Value,
    what: &str,
) -> Result<Ptr<Operation>> {
    let base_root = match one_mma_fact(facts, base, &format!("{what} base"))? {
        MmaFact::SmemBase(root) => root,
        _ => {
            return Err(invalid(format!(
                "{what} base must come from cute.smem_tensor_overlay"
            )));
        }
    };
    let capacity_root = match one_mma_fact(facts, capacity, &format!("{what} capacity"))? {
        MmaFact::SmemCapacity(root) => root,
        _ => {
            return Err(invalid(format!(
                "{what} capacity must come from cute.smem_tensor_overlay"
            )));
        }
    };
    if base_root != capacity_root {
        return Err(invalid(format!(
            "{what} mixes base and capacity from different shared overlays"
        )));
    }
    Ok(base_root)
}

fn require_same_mma_scalar(
    facts: &OrdinaryFacts<MmaFact>,
    left: Value,
    right: Value,
    what: &str,
) -> Result {
    if left == right {
        return Ok(());
    }
    match (
        one_mma_fact(facts, left, what)?,
        one_mma_fact(facts, right, what)?,
    ) {
        (MmaFact::Scalar(left), MmaFact::Scalar(right)) if left == right => Ok(()),
        _ => Err(invalid(format!(
            "{what} must be one identical ordinary SSA carrier"
        ))),
    }
}

fn build_mma_facts(ctx: &Context, operations: &[Ptr<Operation>]) -> OrdinaryFacts<MmaFact> {
    let mut facts = OrdinaryFacts::default();
    for operation in operations {
        let opid = Operation::get_opid(*operation, ctx);
        let operation_ref = operation.deref(ctx);
        if opid == CuteSmemTensorOverlayOp::get_opid_static() {
            let overlay = CuteSmemTensorOverlayOp::wrap(*operation);
            facts.insert(overlay.base(ctx), MmaFact::SmemBase(*operation));
            facts.insert(overlay.capacity(ctx), MmaFact::SmemCapacity(*operation));
        } else if opid == CuteTiledMmaSliceOp::get_opid_static() {
            facts.insert(
                CuteTiledMmaSliceOp::wrap(*operation).sliced_lane(ctx),
                MmaFact::Lane(*operation),
            );
        } else if opid == CuteFragmentFillOp::get_opid_static() {
            facts.insert(
                CuteFragmentFillOp::wrap(*operation).fragment(ctx),
                MmaFact::Accumulator(*operation),
            );
        } else if opid == CuteMmaLoadScalesOp::get_opid_static() {
            facts.insert(
                CuteMmaLoadScalesOp::wrap(*operation).scales(ctx),
                MmaFact::StageScales(*operation),
            );
        } else if opid == CuteFragmentSliceKOp::get_opid_static() {
            facts.insert(
                CuteFragmentSliceKOp::wrap(*operation).scales(ctx),
                MmaFact::SlicedScales(*operation),
            );
        } else if opid == CuteMmaLoadAOp::get_opid_static() {
            facts.insert(
                CuteMmaLoadAOp::wrap(*operation).fragment(ctx),
                MmaFact::AFragment(*operation),
            );
        } else if opid == CuteMmaPartitionBOp::get_opid_static() {
            facts.insert(operation_ref.get_result(0), MmaFact::BBase(*operation));
            facts.insert(operation_ref.get_result(1), MmaFact::BCapacity(*operation));
            facts.insert(operation_ref.get_result(2), MmaFact::BWarpN(*operation));
            facts.insert(operation_ref.get_result(3), MmaFact::BKHalf(*operation));
        }
    }

    loop {
        let mut changed = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        for operation in operations {
            if Operation::get_opid(*operation, ctx) != CuteTiledGemmOp::get_opid_static() {
                continue;
            }
            let operation_ref = operation.deref(ctx);
            let Some(accumulator_facts) = facts.at(operation_ref.get_operand(7)).cloned() else {
                continue;
            };
            for accumulator in accumulator_facts {
                if let MmaFact::Accumulator(root) = accumulator {
                    changed |= facts.insert(
                        CuteTiledGemmOp::wrap(*operation).accumulator(ctx),
                        MmaFact::Accumulator(root),
                    );
                }
            }
        }
        if !changed {
            break;
        }
    }

    seed_unknown_integer_carriers(ctx, operations, &mut facts, 32, MmaFact::Scalar);
    seed_unknown_integer_carriers(ctx, operations, &mut facts, 64, MmaFact::Scalar);
    loop {
        let poisoned =
            poison_missing_forwarding_sources(ctx, operations, &mut facts, MmaFact::Unknown);
        let propagated = propagate_ordinary_plumbing(ctx, operations, &mut facts);
        if !poisoned && !propagated {
            break;
        }
    }
    facts
}

fn mma_fact_at_path(
    facts: &OrdinaryFacts<MmaFact>,
    value: Value,
    path: &[u32],
    fact: MmaFact,
) -> bool {
    facts
        .at_path(value, path)
        .is_some_and(|found| found.contains(&fact))
}

fn mma_semantic_consumer(
    ctx: &Context,
    user: Ptr<Operation>,
    operand_index: usize,
    fact: MmaFact,
) -> bool {
    let opid = Operation::get_opid(user, ctx);
    match fact {
        MmaFact::Lane(_) => {
            operand_index == 0
                && (opid == CuteMmaLoadScalesOp::get_opid_static()
                    || opid == CuteMmaLoadAOp::get_opid_static()
                    || opid == CuteTiledGemmOp::get_opid_static())
        }
        MmaFact::SmemBase(_) => {
            (opid == CuteMmaLoadScalesOp::get_opid_static() && matches!(operand_index, 1 | 3))
                || (opid == CuteMmaLoadAOp::get_opid_static() && operand_index == 1)
                || (opid == CuteMmaPartitionBOp::get_opid_static() && operand_index == 0)
        }
        MmaFact::SmemCapacity(_) => {
            (opid == CuteMmaLoadScalesOp::get_opid_static() && matches!(operand_index, 2 | 4))
                || (opid == CuteMmaLoadAOp::get_opid_static() && operand_index == 2)
                || (opid == CuteMmaPartitionBOp::get_opid_static() && operand_index == 1)
        }
        MmaFact::StageScales(_) => {
            opid == CuteFragmentSliceKOp::get_opid_static() && operand_index == 0
        }
        MmaFact::SlicedScales(_) => {
            opid == CuteTiledGemmOp::get_opid_static() && operand_index == 6
        }
        MmaFact::AFragment(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 1,
        MmaFact::BBase(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 2,
        MmaFact::BCapacity(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 3,
        MmaFact::BWarpN(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 4,
        MmaFact::BKHalf(_) => opid == CuteTiledGemmOp::get_opid_static() && operand_index == 5,
        MmaFact::Accumulator(_) => {
            (opid == CuteTiledGemmOp::get_opid_static() && operand_index == 7)
                || (opid == CuteEpilogueStoreFragmentOp::get_opid_static() && operand_index == 3)
        }
        MmaFact::Scalar(_) | MmaFact::Unknown(_) => true,
    }
}

fn mma_branch_forwards_fact(
    ctx: &Context,
    facts: &OrdinaryFacts<MmaFact>,
    user: Ptr<Operation>,
    value: Value,
    path: &[u32],
    fact: MmaFact,
) -> bool {
    let operation = user.deref(ctx);
    (0..operation.get_num_successors()).any(|successor_index| {
        let successor = operation.get_successor(successor_index);
        branch_successor_operands(ctx, user, successor_index).is_some_and(|operands| {
            operands
                .iter()
                .copied()
                .enumerate()
                .any(|(index, incoming)| {
                    incoming == value
                        && index < successor.deref(ctx).get_num_arguments()
                        && mma_fact_at_path(
                            facts,
                            successor.deref(ctx).get_argument(index),
                            path,
                            fact,
                        )
                })
        })
    })
}

fn mma_ordinary_use_forwards_fact(
    ctx: &Context,
    facts: &OrdinaryFacts<MmaFact>,
    key: &CarrierKey,
    fact: MmaFact,
    user: Ptr<Operation>,
    operand_index: usize,
) -> bool {
    let opid = Operation::get_opid(user, ctx);
    let operation = user.deref(ctx);
    if opid == MirCastOp::get_opid_static()
        && operand_index == 0
        && is_semantic_pointer_carrier_cast(ctx, user)
        && operation.get_num_results() == 1
    {
        return mma_fact_at_path(facts, operation.get_result(0), &key.path, fact);
    }
    if (opid == MirConstructStructOp::get_opid_static()
        || opid == MirConstructTupleOp::get_opid_static())
        && operation.get_num_results() == 1
    {
        let mut destination_path = vec![operand_index as u32];
        destination_path.extend_from_slice(&key.path);
        return mma_fact_at_path(facts, operation.get_result(0), &destination_path, fact);
    }
    if opid == MirInsertFieldOp::get_opid_static()
        && operation.get_num_operands() == 2
        && operation.get_num_results() == 1
    {
        let Some(index) = MirInsertFieldOp::new(user).get_attr_insert_index(ctx) else {
            return false;
        };
        if operand_index == 0 {
            if key.path.first().copied() == Some(index.0) {
                return true;
            }
            return mma_fact_at_path(facts, operation.get_result(0), &key.path, fact);
        }
        if operand_index == 1 {
            let mut destination_path = vec![index.0];
            destination_path.extend_from_slice(&key.path);
            return mma_fact_at_path(facts, operation.get_result(0), &destination_path, fact);
        }
        return false;
    }
    if opid == MirExtractFieldOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
    {
        let Some(index) = MirExtractFieldOp::new(user).get_attr_index(ctx) else {
            return false;
        };
        if key.path.first().copied() != Some(index.0) {
            return true;
        }
        return mma_fact_at_path(facts, operation.get_result(0), &key.path[1..], fact);
    }
    if opid == MirStoreOp::get_opid_static() && operation.get_num_operands() == 2 {
        let store = MirStoreOp::new(user);
        if operand_index == 0 {
            return local_cell(ctx, store.address_opd(ctx))
                .is_some_and(|(root, _)| local_cell_is_closed(ctx, root));
        }
        if operand_index == 1
            && let Some((root, mut destination_path)) = local_cell(ctx, store.address_opd(ctx))
            && local_cell_is_closed(ctx, root)
        {
            destination_path.extend_from_slice(&key.path);
            return mma_fact_at_path(facts, root, &destination_path, fact);
        }
        return false;
    }
    if opid == MirLoadOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
    {
        let load = MirLoadOp::new(user);
        let Some((root, cell_path)) = local_cell(ctx, load.address_opd(ctx)) else {
            return false;
        };
        if !local_cell_is_closed(ctx, root) {
            return false;
        }
        let Some(result_path) = key.path.strip_prefix(cell_path.as_slice()) else {
            return true;
        };
        return mma_fact_at_path(facts, operation.get_result(0), result_path, fact);
    }
    if opid == MirFieldAddrOp::get_opid_static()
        && operand_index == 0
        && operation.get_num_results() == 1
        && MirFieldAddrOp::new(user)
            .get_attr_field_index(ctx)
            .is_some()
    {
        return local_cell(ctx, operation.get_result(0))
            .is_some_and(|(root, _)| local_cell_is_closed(ctx, root));
    }
    if opid == MirCondBranchOp::get_opid_static() && operand_index == 0 {
        return false;
    }
    mma_branch_forwards_fact(ctx, facts, user, key.value, &key.path, fact)
}

fn verify_mma_use_closure(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    facts: &OrdinaryFacts<MmaFact>,
) -> Result {
    for (key, found) in &facts.facts {
        for fact in found.iter().copied() {
            if matches!(fact, MmaFact::Scalar(_) | MmaFact::Unknown(_)) {
                continue;
            }
            for r#use in key.value.uses(ctx) {
                let user = r#use.user_op();
                let operand_index = r#use.find_index(ctx);
                if operations.contains(&user)
                    && (mma_semantic_consumer(ctx, user, operand_index, fact)
                        || mma_ordinary_use_forwards_fact(
                            ctx,
                            facts,
                            key,
                            fact,
                            user,
                            operand_index,
                        ))
                {
                    continue;
                }
                return Err(invalid(format!(
                    "shared-MMA carrier has an unsupported outgoing use by `{}` operand {operand_index}",
                    Operation::get_opid(user, ctx)
                )));
            }
        }
    }
    Ok(())
}

fn require_same_mma_plan(
    ctx: &Context,
    left: Ptr<Operation>,
    right: Ptr<Operation>,
    what: &str,
) -> Result {
    let left_id = Operation::get_opid(left, ctx);
    let right_id = Operation::get_opid(right, ctx);
    if mma_plan_for_op(ctx, left, &left_id) != mma_plan_for_op(ctx, right, &right_id) {
        return Err(invalid(format!(
            "{what} connects shared-MMA operations with different tiled-MMA plans"
        )));
    }
    Ok(())
}

fn verify_smem_mma_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = smem_mma_semantic_ids();
    let facts = build_mma_facts(ctx, all_ops);
    let mut overlays = HashSet::new();
    let mut used_overlays = HashSet::new();
    let mut slices = HashSet::new();
    let mut used_slices = HashSet::new();
    let mut fills = HashSet::new();
    let mut used_fills = HashSet::new();
    let mut scale_loads = HashSet::new();
    let mut used_scale_loads = HashSet::new();
    let mut scale_slices = HashSet::new();
    let mut used_scale_slices = HashSet::new();
    let mut a_loads = HashSet::new();
    let mut used_a_loads = HashSet::new();
    let mut b_partitions = HashSet::new();
    let mut used_b_partitions = HashSet::new();
    let mut gemms = 0usize;

    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if operation.deref(ctx).get_parent_block().is_none() {
            return Err(invalid(format!(
                "shared/MMA operation `{opid}` is not attached to a block"
            )));
        }
        let operation_ref = operation.deref(ctx);
        if opid == CuteSmemTensorOverlayOp::get_opid_static() {
            overlays.insert(*operation);
        } else if opid == CuteTiledMmaSliceOp::get_opid_static() {
            slices.insert(*operation);
        } else if opid == CuteFragmentFillOp::get_opid_static() {
            fills.insert(*operation);
        } else if opid == CuteMmaLoadScalesOp::get_opid_static() {
            scale_loads.insert(*operation);
            let semantic = CuteMmaLoadScalesOp::wrap(*operation);
            let lane = require_mma_lane(
                &facts,
                operation_ref.get_operand(0),
                "cute.mma_load_scales lane",
            )?;
            require_same_mma_plan(ctx, lane, *operation, "mma scale load")?;
            used_slices.insert(lane);
            let scale_a = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(1),
                operation_ref.get_operand(2),
                "cute.mma_load_scales A",
            )?;
            let scale_b = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(3),
                operation_ref.get_operand(4),
                "cute.mma_load_scales B",
            )?;
            if semantic.scale_a_view(ctx) != CuteSmemTensorOverlayOp::wrap(scale_a).view(ctx)
                || semantic.scale_b_view(ctx) != CuteSmemTensorOverlayOp::wrap(scale_b).view(ctx)
            {
                return Err(invalid(
                    "cute.mma_load_scales typed views do not match its exact shared overlays",
                ));
            }
            used_overlays.extend([scale_a, scale_b]);
        } else if opid == CuteFragmentSliceKOp::get_opid_static() {
            scale_slices.insert(*operation);
            let load = match one_mma_fact(
                &facts,
                operation_ref.get_operand(0),
                "cute.fragment_slice_k stage scales",
            )? {
                MmaFact::StageScales(load) => load,
                _ => {
                    return Err(invalid(
                        "cute.fragment_slice_k must consume cute.mma_load_scales",
                    ));
                }
            };
            require_same_mma_plan(ctx, load, *operation, "scale K slice")?;
            used_scale_loads.insert(load);
        } else if opid == CuteMmaLoadAOp::get_opid_static() {
            a_loads.insert(*operation);
            let semantic = CuteMmaLoadAOp::wrap(*operation);
            let lane =
                require_mma_lane(&facts, operation_ref.get_operand(0), "cute.mma_load_a lane")?;
            require_same_mma_plan(ctx, lane, *operation, "MMA A load")?;
            used_slices.insert(lane);
            let overlay = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(1),
                operation_ref.get_operand(2),
                "cute.mma_load_a",
            )?;
            if semantic.view(ctx) != CuteSmemTensorOverlayOp::wrap(overlay).view(ctx) {
                return Err(invalid(
                    "cute.mma_load_a typed view does not match its exact shared overlay",
                ));
            }
            used_overlays.insert(overlay);
        } else if opid == CuteMmaPartitionBOp::get_opid_static() {
            b_partitions.insert(*operation);
            let semantic = CuteMmaPartitionBOp::wrap(*operation);
            let overlay = require_smem_overlay_pair(
                &facts,
                operation_ref.get_operand(0),
                operation_ref.get_operand(1),
                "cute.mma_partition_b",
            )?;
            if semantic.view(ctx) != CuteSmemTensorOverlayOp::wrap(overlay).view(ctx) {
                return Err(invalid(
                    "cute.mma_partition_b typed view does not match its exact shared overlay",
                ));
            }
            used_overlays.insert(overlay);
        } else if opid == CuteTiledGemmOp::get_opid_static() {
            gemms += 1;
            let semantic = CuteTiledGemmOp::wrap(*operation);
            let lane =
                require_mma_lane(&facts, operation_ref.get_operand(0), "cute.tiled_gemm lane")?;
            let a = match one_mma_fact(
                &facts,
                operation_ref.get_operand(1),
                "cute.tiled_gemm A fragment",
            )? {
                MmaFact::AFragment(load) => load,
                _ => return Err(invalid("cute.tiled_gemm A must come from cute.mma_load_a")),
            };
            let b = match (
                one_mma_fact(&facts, operation_ref.get_operand(2), "tiled_gemm B base")?,
                one_mma_fact(
                    &facts,
                    operation_ref.get_operand(3),
                    "tiled_gemm B capacity",
                )?,
                one_mma_fact(&facts, operation_ref.get_operand(4), "tiled_gemm B warp-N")?,
                one_mma_fact(&facts, operation_ref.get_operand(5), "tiled_gemm B K-half")?,
            ) {
                (
                    MmaFact::BBase(base),
                    MmaFact::BCapacity(capacity),
                    MmaFact::BWarpN(warp_n),
                    MmaFact::BKHalf(k_half),
                ) if base == capacity && base == warp_n && base == k_half => base,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm B carriers must be the four matching results of one mma_partition_b",
                    ));
                }
            };
            let scales = match one_mma_fact(
                &facts,
                operation_ref.get_operand(6),
                "cute.tiled_gemm scales",
            )? {
                MmaFact::SlicedScales(slice) => slice,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm scales must come from cute.fragment_slice_k",
                    ));
                }
            };
            let fill = match one_mma_fact(
                &facts,
                operation_ref.get_operand(7),
                "cute.tiled_gemm accumulator",
            )? {
                MmaFact::Accumulator(fill) => fill,
                _ => {
                    return Err(invalid(
                        "cute.tiled_gemm accumulator must be the closed recurrence seeded by fragment_fill",
                    ));
                }
            };
            for producer in [lane, a, b, scales, fill] {
                require_same_mma_plan(ctx, producer, *operation, "tiled GEMM spine")?;
            }
            let scale_load = match one_mma_fact(
                &facts,
                scales.deref(ctx).get_operand(0),
                "tiled GEMM scale-load origin",
            )? {
                MmaFact::StageScales(load) => load,
                _ => unreachable!("fragment_slice_k was verified above"),
            };
            let scale_load_ref = scale_load.deref(ctx);
            let a_ref = a.deref(ctx);
            let b_ref = b.deref(ctx);
            let scale_lane = require_mma_lane(
                &facts,
                scale_load_ref.get_operand(0),
                "mma_load_scales lane",
            )?;
            let a_lane = require_mma_lane(&facts, a_ref.get_operand(0), "mma_load_a lane")?;
            if scale_lane != lane || a_lane != lane {
                return Err(invalid(
                    "tiled GEMM, scale load, and A load must use one tiled-MMA lane slice",
                ));
            }
            require_same_mma_scalar(
                &facts,
                scale_load_ref.get_operand(5),
                a_ref.get_operand(3),
                "shared-MMA warp-M",
            )?;
            require_same_mma_scalar(
                &facts,
                scale_load_ref.get_operand(6),
                b_ref.get_operand(2),
                "shared-MMA warp-N",
            )?;
            require_same_mma_scalar(
                &facts,
                scales.deref(ctx).get_operand(1),
                a_ref.get_operand(4),
                "shared-MMA K-half",
            )?;
            require_same_mma_scalar(
                &facts,
                a_ref.get_operand(4),
                b_ref.get_operand(3),
                "shared-MMA K-half",
            )?;
            if semantic.b_view(ctx) != CuteMmaPartitionBOp::wrap(b).view(ctx) {
                return Err(invalid(
                    "cute.tiled_gemm B view does not match its mma_partition_b producer",
                ));
            }
            used_slices.insert(lane);
            used_a_loads.insert(a);
            used_b_partitions.insert(b);
            used_scale_slices.insert(scales);
            used_fills.insert(fill);
        }
    }

    let found = overlays.len()
        + slices.len()
        + fills.len()
        + scale_loads.len()
        + scale_slices.len()
        + a_loads.len()
        + b_partitions.len()
        + gemms;
    if found == 0 {
        return Ok(());
    }
    verify_mma_use_closure(ctx, all_ops, &facts)?;
    if gemms == 0
        || overlays != used_overlays
        || slices != used_slices
        || fills != used_fills
        || scale_loads != used_scale_loads
        || scale_slices != used_scale_slices
        || a_loads != used_a_loads
        || b_partitions != used_b_partitions
    {
        return Err(invalid(
            "shared-MMA story is incomplete or has an unconsumed overlay/slice/load/partition/fill producer",
        ));
    }
    Ok(())
}

/// Follow one straight-line source sequence across the block splits emitted
/// for Rust call terminators. Every crossed edge must be an unconditional
/// goto into a block with no side entry.
fn require_linear_protocol_order(
    ctx: &Context,
    label: &str,
    ordered: &[Ptr<Operation>],
) -> Result<Vec<Ptr<Operation>>> {
    let Some(first) = ordered.first() else {
        return Err(invalid(format!("{label} has no operations to order")));
    };
    let first_block = first
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| invalid(format!("{label} contains an operation outside a block")))?;
    let region = first_block
        .deref(ctx)
        .get_parent_region()
        .ok_or_else(|| invalid(format!("{label} starts in a block outside a region")))?;
    if ordered.iter().any(|operation| {
        operation
            .deref(ctx)
            .get_parent_block()
            .and_then(|block| block.deref(ctx).get_parent_region())
            != Some(region)
    }) {
        return Err(invalid(format!(
            "{label} operations must stay in one region"
        )));
    }

    let mut current = first_block;
    let mut visited = HashSet::new();
    let mut next_expected = 0usize;
    let mut range = Vec::new();
    loop {
        if !visited.insert(current) {
            return Err(invalid(format!(
                "{label} loops before reaching its final operation"
            )));
        }
        let operations: Vec<_> = current.deref(ctx).iter(ctx).collect();
        let start = if current == first_block {
            operations
                .iter()
                .position(|operation| operation == first)
                .ok_or_else(|| {
                    invalid(format!(
                        "{label} contains an operation missing from its parent block"
                    ))
                })?
        } else {
            0
        };
        for operation in operations.into_iter().skip(start) {
            range.push(operation);
            if ordered.get(next_expected) == Some(&operation) {
                next_expected += 1;
                if next_expected == ordered.len() {
                    return Ok(range);
                }
            } else if ordered.contains(&operation) {
                return Err(invalid(format!(
                    "{label} is not in the required source order"
                )));
            }
        }

        let terminator = current
            .deref(ctx)
            .get_terminator(ctx)
            .ok_or_else(|| invalid(format!("{label} ends a block before its final operation")))?;
        if Operation::get_opid(terminator, ctx) != MirGotoOp::get_opid_static()
            || terminator.deref(ctx).get_num_successors() != 1
        {
            return Err(invalid(format!(
                "{label} crosses an extra exit instead of one unconditional successor"
            )));
        }
        let successor = terminator.deref(ctx).get_successor(0);
        if successor.deref(ctx).get_parent_region() != Some(region) {
            return Err(invalid(format!("{label} leaves its source region")));
        }
        if successor.preds(ctx).len() != 1 {
            return Err(invalid(format!(
                "{label} crosses a block with a side entry"
            )));
        }
        current = successor;
    }
}

fn is_epilogue_pointer_identity_cast(ctx: &Context, operation: Ptr<Operation>) -> bool {
    is_semantic_pointer_carrier_cast(ctx, operation)
}

fn is_epilogue_writer_plumbing(ctx: &Context, operation: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(operation, ctx);
    opid == MirGotoOp::get_opid_static()
        || opid == MirStorageLiveOp::get_opid_static()
        || opid == MirStorageDeadOp::get_opid_static()
        || opid == MirExtractFieldOp::get_opid_static()
        || is_epilogue_pointer_identity_cast(ctx, operation)
        || is_closed_local_cell_load(ctx, operation)
}

/// A non-volatile `mir.load` from a closed compiler-created local cell only
/// re-materializes carrier state (e.g. the epilogue handle struct spilled to
/// an alloca slot); it cannot touch tile memory, so it is admissible plumbing
/// between the writer-protocol anchor operations.
fn is_closed_local_cell_load(ctx: &Context, operation: Ptr<Operation>) -> bool {
    if Operation::get_opid(operation, ctx) != MirLoadOp::get_opid_static() {
        return false;
    }
    let load = MirLoadOp::new(operation);
    {
        let operation_ref = operation.deref(ctx);
        if operation_ref.get_num_operands() != 1 || operation_ref.get_num_results() != 1 {
            return false;
        }
    }
    if load.is_volatile(ctx) {
        return false;
    }
    let Some((root, _path)) = local_cell(ctx, load.address_opd(ctx)) else {
        return false;
    };
    local_cell_is_closed(ctx, root)
}

/// Accept only the dead immutable reference Rust leaves around a no-inline
/// call on the fieldless, zero-sized TMA store-pipeline token.
fn is_dead_epilogue_store_pipeline_ref(ctx: &Context, operation: Ptr<Operation>) -> bool {
    if Operation::get_opid(operation, ctx) != MirRefOp::get_opid_static() {
        return false;
    }
    let reference = MirRefOp::new(operation);
    if reference.verify(ctx).is_err()
        || reference
            .get_attr_mutable(ctx)
            .is_none_or(|mutable| mutable.0)
    {
        return false;
    }
    let operation_ref = operation.deref(ctx);
    let referent = operation_ref.get_operand(0);
    let result = operation_ref.get_result(0);
    if result.is_used(ctx) {
        return false;
    }

    let referent_ty = referent.get_type(ctx);
    let referent_ty_ref = referent_ty.deref(ctx);
    let Some(store_pipeline) = referent_ty_ref.downcast_ref::<MirStructType>() else {
        return false;
    };
    if store_pipeline.name != "TmaStorePipeline"
        || !store_pipeline.field_names.is_empty()
        || !store_pipeline.field_types.is_empty()
        || !store_pipeline.mem_to_decl.is_empty()
        || !store_pipeline.field_offsets.is_empty()
        || store_pipeline.total_size != 0
        || store_pipeline.abi_align != 1
    {
        return false;
    }
    let result_ty = result.get_type(ctx);
    let result_ty_ref = result_ty.deref(ctx);
    let Some(pointer) = result_ty_ref.downcast_ref::<MirPtrType>() else {
        return false;
    };
    if pointer.pointee != referent_ty
        || pointer.is_mutable
        || pointer.address_space != address_space::GENERIC
    {
        return false;
    }

    let Some(construct) = referent.defining_op() else {
        return false;
    };
    if Operation::get_opid(construct, ctx) != MirConstructStructOp::get_opid_static() {
        return false;
    }
    let construct_ref = construct.deref(ctx);
    if construct_ref.get_num_operands() != 0
        || construct_ref.get_num_results() != 1
        || construct_ref.get_result(0) != referent
    {
        return false;
    }
    let uses = referent.uses(ctx);
    uses.len() == 1 && uses[0].user_op() == operation && uses[0].find_index(ctx) == 0
}

fn is_epilogue_issuer_plumbing(ctx: &Context, operation: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(operation, ctx);
    is_epilogue_writer_plumbing(ctx, operation)
        || opid == MirUndefOp::get_opid_static()
        || opid == MirInsertFieldOp::get_opid_static()
        || opid == MirConstructStructOp::get_opid_static()
        || opid == MirConstructTupleOp::get_opid_static()
        || opid == MirConstantOp::get_opid_static()
        || opid == MirAddOp::get_opid_static()
        || is_dead_epilogue_store_pipeline_ref(ctx, operation)
}

fn resolve_inserted_epilogue_field(
    ctx: &Context,
    mut aggregate: Value,
    wanted_index: u32,
) -> Result<Option<Value>> {
    for _ in 0..16 {
        let Some(definer) = aggregate.defining_op() else {
            return Ok(None);
        };
        if Operation::get_opid(definer, ctx) != MirInsertFieldOp::get_opid_static() {
            return Ok(None);
        }
        let insert = MirInsertFieldOp::new(definer);
        let Some(index) = insert.get_attr_insert_index(ctx) else {
            return Err(invalid(
                "epilogue carrier has an insert_field without a constant index",
            ));
        };
        let operation = definer.deref(ctx);
        if index.0 == wanted_index {
            return Ok(Some(operation.get_operand(1)));
        }
        aggregate = operation.get_operand(0);
    }
    Err(invalid(
        "epilogue carrier insert/extract chain is unexpectedly deep",
    ))
}

/// Peel only lossless SSA carrier plumbing. Memory, arithmetic, control flow,
/// and arbitrary aliases deliberately stop provenance resolution.
fn resolve_epilogue_ssa_carrier(ctx: &Context, mut value: Value) -> Result<Value> {
    for _ in 0..16 {
        let Some(definer) = value.defining_op() else {
            return Ok(value);
        };
        let opid = Operation::get_opid(definer, ctx);
        if opid == MirCastOp::get_opid_static() {
            if is_semantic_pointer_carrier_cast(ctx, definer) {
                value = definer.deref(ctx).get_operand(0);
                continue;
            }
            return Ok(value);
        }
        if opid == MirExtractFieldOp::get_opid_static() {
            let extract = MirExtractFieldOp::new(definer);
            let Some(index) = extract.get_attr_index(ctx) else {
                return Err(invalid(
                    "epilogue carrier has an extract_field without a constant index",
                ));
            };
            let aggregate = definer.deref(ctx).get_operand(0);
            if let Some(aggregate_definer) = aggregate.defining_op() {
                let aggregate_id = Operation::get_opid(aggregate_definer, ctx);
                if aggregate_id == MirConstructStructOp::get_opid_static()
                    || aggregate_id == MirConstructTupleOp::get_opid_static()
                {
                    let aggregate_operation = aggregate_definer.deref(ctx);
                    if usize::try_from(index.0)
                        .ok()
                        .is_some_and(|index| index < aggregate_operation.get_num_operands())
                    {
                        value = aggregate_operation.get_operand(index.0 as usize);
                        continue;
                    }
                }
            }
            let Some(inserted) = resolve_inserted_epilogue_field(ctx, aggregate, index.0)? else {
                return Ok(value);
            };
            value = inserted;
            continue;
        }
        return Ok(value);
    }
    Err(invalid(
        "epilogue carrier cast/insert/extract chain is unexpectedly deep",
    ))
}

fn require_same_epilogue_carrier(
    ctx: &Context,
    actual: Value,
    expected: Value,
    message: &str,
) -> Result {
    if resolve_epilogue_ssa_carrier(ctx, actual)? != resolve_epilogue_ssa_carrier(ctx, expected)? {
        return Err(invalid(message));
    }
    Ok(())
}

fn epilogue_u64_constant(ctx: &Context, value: Value) -> Option<u64> {
    let integer = value.get_type(ctx);
    let integer = integer.deref(ctx);
    if !integer
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
    {
        return None;
    }
    let constant = value.defining_op()?;
    if Operation::get_opid(constant, ctx) != MirConstantOp::get_opid_static() {
        return None;
    }
    let attribute = MirConstantOp::new(constant).get_attr_value(ctx)?;
    (attribute.value().bw() <= 64).then(|| attribute.value().to_u64())
}

fn require_next_epilogue_column(ctx: &Context, left: Value, right: Value) -> Result {
    let left = resolve_epilogue_ssa_carrier(ctx, left)?;
    let right = resolve_epilogue_ssa_carrier(ctx, right)?;
    let Some(add) = right.defining_op() else {
        return Err(invalid(
            "epilogue half 1 must target the tile column immediately after half 0",
        ));
    };
    if Operation::get_opid(add, ctx) != MirAddOp::get_opid_static() {
        return Err(invalid("epilogue half 1 must target `half_0_column + 1`"));
    }
    let add = add.deref(ctx);
    let lhs = resolve_epilogue_ssa_carrier(ctx, add.get_operand(0))?;
    let rhs = resolve_epilogue_ssa_carrier(ctx, add.get_operand(1))?;
    if !((lhs == left && epilogue_u64_constant(ctx, rhs) == Some(1))
        || (rhs == left && epilogue_u64_constant(ctx, lhs) == Some(1)))
    {
        return Err(invalid("epilogue half 1 must target `half_0_column + 1`"));
    }
    Ok(())
}

fn require_epilogue_cfg_links(
    ctx: &Context,
    acquire: Ptr<Operation>,
    reusable_sync: Ptr<Operation>,
    ready_sync: Ptr<Operation>,
    half_zero: Ptr<Operation>,
    commit: Ptr<Operation>,
    tail: Ptr<Operation>,
) -> Result {
    if operation_reaches(ctx, acquire, reusable_sync)
        && operation_reaches(ctx, ready_sync, half_zero)
        && operation_reaches(ctx, commit, tail)
    {
        Ok(())
    } else {
        Err(invalid(
            "epilogue CFG must connect acquire -> writer hand-off -> TMA issuer -> commit -> tail inside one role-anchored story",
        ))
    }
}

#[derive(Clone, Copy)]
struct EpilogueStoreFacts {
    operation: Ptr<Operation>,
    source_view: Ptr<Operation>,
    destination_view: Ptr<Operation>,
}

fn epilogue_store_facts(
    ctx: &Context,
    operation: Ptr<Operation>,
    scope: &[Ptr<Operation>],
) -> Result<EpilogueStoreFacts> {
    let store = CuteTmaStore2dSemanticOp::wrap(operation);
    let source_view = direct_tma_view_producer(
        ctx,
        store.source(ctx),
        &CuteTmaSmemViewOp::get_opid_static(),
        "epilogue store source",
        scope,
    )?;
    let destination_view = direct_tma_view_producer(
        ctx,
        store.destination(ctx),
        &CuteTmaGmemViewOp::get_opid_static(),
        "epilogue store destination",
        scope,
    )?;
    Ok(EpilogueStoreFacts {
        operation,
        source_view,
        destination_view,
    })
}

fn verify_epilogue_story(ctx: &Context, all_ops: &[Ptr<Operation>]) -> Result {
    let ids = epilogue_semantic_ids();
    let mut overlays = Vec::new();
    let mut slices = Vec::new();
    let mut fragments = Vec::new();
    let mut syncs = Vec::new();
    let mut halves = Vec::new();
    let mut acquires = Vec::new();
    let mut commits = Vec::new();
    let mut tails = Vec::new();
    let mut stores = Vec::new();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if opid == ids[0] {
            overlays.push(*operation);
        } else if opid == ids[1] {
            slices.push(*operation);
        } else if opid == ids[2] {
            fragments.push(*operation);
        } else if opid == ids[3] {
            syncs.push(*operation);
        } else if opid == ids[4] {
            halves.push(*operation);
        } else if opid == ids[5] {
            acquires.push(*operation);
        } else if opid == ids[6] {
            commits.push(*operation);
        } else if opid == ids[7] {
            tails.push(*operation);
        } else if opid == ids[8] {
            stores.push(*operation);
        }
    }

    let found = overlays.len()
        + slices.len()
        + fragments.len()
        + syncs.len()
        + halves.len()
        + acquires.len()
        + commits.len()
        + tails.len()
        + stores.len();
    if found == 0 {
        return Ok(());
    }
    if overlays.len() != 1
        || slices.len() != 1
        || fragments.len() != 1
        || syncs.len() != 2
        || halves.len() != 2
        || acquires.len() != 1
        || commits.len() != 1
        || tails.len() != 1
        || stores.len() != 2
    {
        return Err(invalid(format!(
            "epilogue story needs exactly 1 overlay, 1 warp slice, 1 fragment store, 2 syncs, 2 halves, 1 acquire, 2 TMA stores, 1 commit, and 1 tail; found {}, {}, {}, {}, {}, {}, {}, {}, {}",
            overlays.len(),
            slices.len(),
            fragments.len(),
            syncs.len(),
            halves.len(),
            acquires.len(),
            stores.len(),
            commits.len(),
            tails.len()
        )));
    }

    let overlay = overlays[0];
    let warp_slice = slices[0];
    let store_fragment = fragments[0];
    let acquire = acquires[0];
    let commit = commits[0];
    let tail = tails[0];
    let overlay_view = CuteEpilogueSmemOverlayOp::wrap(overlay);
    let warp_view = CuteEpilogueWarpSliceOp::wrap(warp_slice);
    let fragment = CuteEpilogueStoreFragmentOp::wrap(store_fragment);
    let overlay_tile = overlay_view
        .tile(ctx)
        .expect("locally verified epilogue overlay has a tile");
    if warp_view.tile(ctx).as_ref() != Some(&overlay_tile)
        || fragment.tile(ctx).as_ref() != Some(&overlay_tile)
    {
        return Err(invalid(
            "epilogue overlay, warp slice, and fragment store must carry the same tile plan",
        ));
    }
    let mma_facts = build_mma_facts(ctx, all_ops);
    let accumulator_seed = match one_mma_fact(
        &mma_facts,
        fragment.accumulator(ctx),
        "epilogue fragment accumulator",
    )? {
        MmaFact::Accumulator(seed) => seed,
        _ => {
            return Err(invalid(
                "epilogue fragment accumulator must be the closed tiled-GEMM recurrence",
            ));
        }
    };
    if CuteFragmentFillOp::wrap(accumulator_seed).plan(ctx)
        != Some(overlay_tile.plan.tiled_mma.clone())
    {
        return Err(invalid(
            "epilogue accumulator plan differs from its shared-MMA story",
        ));
    }
    require_same_epilogue_carrier(
        ctx,
        warp_view.input_base(ctx),
        overlay_view.base(ctx),
        "epilogue warp slice must use the base produced by the epilogue overlay",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.base(ctx),
        warp_view.base(ctx),
        "epilogue fragment store must use the base produced by the warp slice",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.warp_id(ctx),
        warp_view.warp_id(ctx),
        "epilogue fragment store must use the warp ID produced by the warp slice",
    )?;
    require_same_epilogue_carrier(
        ctx,
        fragment.lane(ctx),
        warp_view.lane(ctx),
        "epilogue fragment store must use the lane produced by the warp slice",
    )?;

    let reusable_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::Reusable)
        })
        .ok_or_else(|| invalid("epilogue story is missing the Reusable sync"))?;
    let ready_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::ReadyForTma)
        })
        .ok_or_else(|| invalid("epilogue story is missing the ReadyForTma sync"))?;
    if reusable_sync == ready_sync {
        return Err(invalid(
            "epilogue story needs one distinct sync for each phase",
        ));
    }
    for sync in [reusable_sync, ready_sync] {
        let sync = CuteEpilogueSyncOp::wrap(sync);
        if sync.tile(ctx).as_ref() != Some(&overlay_tile) {
            return Err(invalid(
                "both epilogue syncs must carry the overlay tile plan",
            ));
        }
        require_same_epilogue_carrier(
            ctx,
            sync.base(ctx),
            overlay_view.base(ctx),
            "both epilogue syncs must use the base produced by the epilogue overlay",
        )?;
    }

    let writer_range = require_linear_protocol_order(
        ctx,
        "epilogue writer protocol",
        &[reusable_sync, store_fragment, ready_sync],
    )?;
    for operation in &writer_range {
        let expected = [reusable_sync, store_fragment, ready_sync].contains(operation);
        if !expected && !is_epilogue_writer_plumbing(ctx, *operation) {
            return Err(invalid(format!(
                "epilogue writer protocol has interposed non-carrier operation `{}`",
                Operation::get_opid(*operation, ctx)
            )));
        }
    }
    let writer_events: Vec<_> = writer_range
        .iter()
        .copied()
        .filter(|operation| {
            let opid = Operation::get_opid(*operation, ctx);
            opid.dialect.to_string() == crate::CUTE_DIALECT_NAME
                || opid == MirCallOp::get_opid_static()
        })
        .collect();
    if writer_events != [reusable_sync, store_fragment, ready_sync] {
        return Err(invalid(
            "epilogue writer protocol must be exactly Reusable -> fragment store -> ReadyForTma, whose semantic boundary includes publication to the TMA proxy",
        ));
    }

    let half_zero = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|half| half.0 == 0)
        })
        .ok_or_else(|| invalid("epilogue story is missing half 0"))?;
    let half_one = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|half| half.0 == 1)
        })
        .ok_or_else(|| invalid("epilogue story is missing half 1"))?;
    if half_zero == half_one {
        return Err(invalid("epilogue story needs two distinct shared halves"));
    }
    for half in [half_zero, half_one] {
        let half = CuteEpilogueHalfOp::wrap(half);
        if half.tile(ctx).as_ref() != Some(&overlay_tile) {
            return Err(invalid(
                "both epilogue halves must carry the overlay tile plan",
            ));
        }
        require_same_epilogue_carrier(
            ctx,
            half.full_base(ctx),
            overlay_view.base(ctx),
            "both epilogue halves must use the base produced by the epilogue overlay",
        )?;
    }

    let store_zero = epilogue_store_facts(ctx, stores[0], all_ops)?;
    let store_one = epilogue_store_facts(ctx, stores[1], all_ops)?;
    for (half, store) in [(half_zero, store_zero), (half_one, store_one)] {
        let half = CuteEpilogueHalfOp::wrap(half);
        let source = CuteTmaSmemViewOp::wrap(store.source_view);
        require_same_epilogue_carrier(
            ctx,
            source.base(ctx),
            half.half_base(ctx),
            "each semantic TMA store source base must come from the matching epilogue half through only the compiler's SSA carrier adapter",
        )?;
        require_same_epilogue_carrier(
            ctx,
            source.capacity(ctx),
            half.capacity(ctx),
            "each semantic TMA store source capacity must come from the matching epilogue half through only the compiler's SSA carrier adapter",
        )?;
    }
    if store_zero.operation.deref(ctx).get_operand(1).get_type(ctx)
        != store_one.operation.deref(ctx).get_operand(1).get_type(ctx)
    {
        return Err(invalid(
            "both epilogue stores must target the same typed output tile",
        ));
    }
    let destination_zero = CuteTmaGmemViewOp::wrap(store_zero.destination_view);
    let destination_one = CuteTmaGmemViewOp::wrap(store_one.destination_view);
    require_same_epilogue_carrier(
        ctx,
        destination_one.descriptor(ctx),
        destination_zero.descriptor(ctx),
        "both epilogue halves must target the same TMA destination descriptor",
    )?;
    let semantic_zero = CuteTmaStore2dSemanticOp::wrap(store_zero.operation);
    let semantic_one = CuteTmaStore2dSemanticOp::wrap(store_one.operation);
    require_same_epilogue_carrier(
        ctx,
        semantic_one.tile_row(ctx),
        semantic_zero.tile_row(ctx),
        "both epilogue halves must target the same output tile row",
    )?;
    require_next_epilogue_column(
        ctx,
        semantic_zero.tile_column(ctx),
        semantic_one.tile_column(ctx),
    )?;

    let issuer_range = require_linear_protocol_order(
        ctx,
        "epilogue TMA issuer protocol",
        &[
            half_zero,
            store_zero.operation,
            half_one,
            store_one.operation,
            commit,
        ],
    )?;
    for operation in issuer_range {
        let opid = Operation::get_opid(operation, ctx);
        let allowed_tma_view = opid == CuteTmaGmemViewOp::get_opid_static()
            || opid == CuteTmaSmemViewOp::get_opid_static();
        let expected = [
            half_zero,
            store_zero.operation,
            half_one,
            store_one.operation,
            commit,
        ]
        .contains(&operation);
        if !allowed_tma_view && !expected && !is_epilogue_issuer_plumbing(ctx, operation) {
            return Err(invalid(format!(
                "epilogue TMA issuer protocol has interposed operation `{opid}`"
            )));
        }
    }

    let acquire_pipeline = CuteTmaStoreAcquireOp::wrap(acquire).pipeline(ctx);
    let commit_pipeline = CuteTmaStoreCommitOp::wrap(commit).pipeline(ctx);
    let tail_pipeline = CuteTmaStoreTailOp::wrap(tail).pipeline(ctx);
    if acquire_pipeline != commit_pipeline || acquire_pipeline != tail_pipeline {
        return Err(invalid(
            "epilogue acquire, commit, and tail must carry the same TMA store-pipeline facts",
        ));
    }

    require_epilogue_cfg_links(
        ctx,
        acquire,
        reusable_sync,
        ready_sync,
        half_zero,
        commit,
        tail,
    )
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
        crate::sm100_ops::verify_story(ctx, &scope).map_err(invalid)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::{
        CutePipelineStateAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr,
        CuteTensorLayoutAttr,
    };
    use crate::layout::{ComposedLayout, OffsetUnit};
    use dialect_mir::ops::{MirReturnOp, MirUndefOp};
    use dialect_mir::types::{MirPtrType, address_space};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::op_interfaces::{OperandSegmentInterface, SymbolOpInterface};
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FP32Type, FunctionType, IntegerType, Signedness};

    fn module_top(ctx: &mut Context) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        dialect_mir::register(ctx);
        crate::register(ctx);
        let module = ModuleOp::new(ctx, "cute_verify_test".try_into().unwrap());
        let module = module.get_operation();
        let region = module.deref(ctx).get_region(0);
        let block = region.deref(ctx).iter(ctx).next().unwrap();
        (module, block)
    }

    fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
        let operation = MirUndefOp::new(ctx, ty).get_operation();
        operation.insert_at_back(block, ctx);
        operation.deref(ctx).get_result(0)
    }

    fn ghost_tensor_type(ctx: &mut Context) -> TypeHandle {
        let element: TypeHandle = FP32Type::get(ctx).into();
        CuteTensorViewType::get(
            ctx,
            element,
            element,
            CuteTensorAddressSpaceAttr::Gmem,
            CuteTensorAccessAttr::ReadOnly,
            4,
            CuteTensorLayoutAttr::Contiguous1D,
        )
        .into()
    }

    fn local_cell(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
        let pointer: TypeHandle = MirPtrType::get_generic(ctx, ty, true).into();
        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![pointer],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(block, ctx);
        alloca.deref(ctx).get_result(0)
    }

    fn store(ctx: &mut Context, block: Ptr<BasicBlock>, pointer: Value, value: Value) {
        Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![pointer, value],
            vec![],
            0,
        )
        .insert_at_back(block, ctx);
    }

    fn load(ctx: &mut Context, block: Ptr<BasicBlock>, pointer: Value, ty: TypeHandle) -> Value {
        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![ty],
            vec![pointer],
            vec![],
            0,
        );
        load.insert_at_back(block, ctx);
        load.deref(ctx).get_result(0)
    }

    fn pointer_cast(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        pointer: Value,
        result_type: TypeHandle,
    ) -> Ptr<Operation> {
        let cast = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_type],
            vec![pointer],
            vec![],
            0,
        );
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        cast.insert_at_back(block, ctx);
        cast
    }

    fn goto(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        target: Ptr<BasicBlock>,
        operands: Vec<Value>,
    ) {
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![target],
            0,
        )
        .insert_at_back(block, ctx);
    }

    fn cond_branch(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        condition: Value,
        true_target: Ptr<BasicBlock>,
        false_target: Ptr<BasicBlock>,
    ) {
        let (operands, sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], Vec::new(), Vec::new()]);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_target, false_target],
            0,
        );
        MirCondBranchOp::new(branch).set_operand_segment_sizes(ctx, sizes);
        branch.insert_at_back(block, ctx);
    }

    fn append_complete_scheduler_function(
        ctx: &mut Context,
        module_block: Ptr<BasicBlock>,
        name: &str,
        grid: CuteTileGridAttr,
        misroute_work_edge: bool,
    ) {
        let function_type = FunctionType::get(ctx, vec![], vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, name.try_into().unwrap());
        function.insert_at_back(module_block, ctx);
        let region = function.deref(ctx).get_region(0);
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let entry = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_back(region, ctx);
        let header = BasicBlock::new(ctx, None, vec![u64_ty, u64_ty]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(region, ctx);

        let start = CuteSchedulerNew1dOp::new(ctx, grid).get_operation();
        start.insert_at_back(entry, ctx);
        let start = CuteSchedulerNew1dOp::wrap(start);
        goto(
            ctx,
            entry,
            header,
            vec![start.current(ctx), start.stride(ctx)],
        );

        let current = header.deref(ctx).get_argument(0);
        let stride = header.deref(ctx).get_argument(1);
        let has_work = CuteSchedulerHasWorkOp::new(ctx, current, grid).get_operation();
        has_work.insert_at_back(header, ctx);
        let not = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![
                CuteSchedulerHasWorkOp::wrap(has_work)
                    .has_work(ctx)
                    .get_type(ctx),
            ],
            vec![CuteSchedulerHasWorkOp::wrap(has_work).has_work(ctx)],
            vec![],
            0,
        );
        not.insert_at_back(header, ctx);
        let finished = not.deref(ctx).get_result(0);
        let (true_target, false_target) = if misroute_work_edge {
            (body, exit)
        } else {
            (exit, body)
        };
        cond_branch(ctx, header, finished, true_target, false_target);

        let selected = CuteSchedulerCurrentOp::new(ctx, current, grid).get_operation();
        selected.insert_at_back(body, ctx);
        CuteWorkTileCoordinatesOp::new(ctx, CuteSchedulerCurrentOp::wrap(selected).work_tile(ctx))
            .get_operation()
            .insert_at_back(body, ctx);
        let advance = CuteSchedulerAdvanceOp::new(ctx, current, stride, grid).get_operation();
        advance.insert_at_back(body, ctx);
        goto(
            ctx,
            body,
            header,
            vec![CuteSchedulerAdvanceOp::wrap(advance).next(ctx), stride],
        );

        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(exit, ctx);
    }

    #[test]
    fn rejects_a_ghost_nested_in_an_ordinary_pointer_without_any_cute_op() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let ghost = ghost_tensor_type(&mut ctx);
        let inner: TypeHandle = MirPtrType::get_generic(&mut ctx, ghost, false).into();
        let outer: TypeHandle = MirPtrType::get_generic(&mut ctx, inner, false).into();
        undef(&mut ctx, block, outer);

        let error = verify_cute_semantics(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("ghost CuTe value"), "{error}");
        assert!(error.contains("operand or result"), "{error}");
    }

    #[test]
    fn accepts_independent_scheduler_stories_in_two_functions_without_mutation() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        append_complete_scheduler_function(
            &mut ctx,
            block,
            "first",
            CuteTileGridAttr::new(4, 4, 1),
            false,
        );
        append_complete_scheduler_function(
            &mut ctx,
            block,
            "second",
            CuteTileGridAttr::new(5, 4, 1),
            false,
        );
        let mut before = Vec::new();
        collect_ops(&ctx, module, &mut before);

        verify_cute_semantics(&ctx, module).unwrap();

        let mut after = Vec::new();
        collect_ops(&ctx, module, &mut after);
        assert_eq!(after, before);
    }

    #[test]
    fn rejects_a_scheduler_body_on_the_exit_edge_even_when_preorder_looks_ordered() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        append_complete_scheduler_function(
            &mut ctx,
            block,
            "misrouted",
            CuteTileGridAttr::new(4, 4, 1),
            true,
        );

        let error = verify_cute_semantics(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("scheduler CFG"), "{error}");
    }

    #[test]
    fn accepts_a_closed_local_cell_carrying_scheduler_provenance() {
        let mut ctx = Context::new();
        module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let grid = CuteTileGridAttr::new(4, 4, 1);
        let start = CuteSchedulerNew1dOp::new(&mut ctx, grid).get_operation();
        start.insert_at_back(block, &ctx);
        let pointer = local_cell(&mut ctx, block, u64_ty);
        let current = CuteSchedulerNew1dOp::wrap(start).current(&ctx);
        store(&mut ctx, block, pointer, current);
        let carried = load(&mut ctx, block, pointer, u64_ty);
        let operations: Vec<_> = block.deref(&ctx).iter(&ctx).collect();

        let facts = build_scheduler_facts(&ctx, &operations);
        assert_eq!(
            scheduler_root(&facts, carried, true, "test carrier").unwrap(),
            start
        );
    }

    #[test]
    fn factless_block_argument_store_poisons_a_valid_scheduler_cell() {
        let mut ctx = Context::new();
        module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![u64_ty]);
        let arbitrary = block.deref(&ctx).get_argument(0);
        let grid = CuteTileGridAttr::new(4, 4, 1);
        let start = CuteSchedulerNew1dOp::new(&mut ctx, grid).get_operation();
        start.insert_at_back(block, &ctx);
        let pointer = local_cell(&mut ctx, block, u64_ty);
        let current = CuteSchedulerNew1dOp::wrap(start).current(&ctx);
        store(&mut ctx, block, pointer, current);
        store(&mut ctx, block, pointer, arbitrary);
        let poisoned = load(&mut ctx, block, pointer, u64_ty);
        let operations: Vec<_> = block.deref(&ctx).iter(&ctx).collect();

        let facts = build_scheduler_facts(&ctx, &operations);
        let error = scheduler_root(&facts, poisoned, true, "test carrier")
            .unwrap_err()
            .to_string();
        assert!(error.contains("merges values"), "{error}");
    }

    #[test]
    fn non_identity_pointer_casts_cannot_alias_a_local_semantic_cell() {
        let mut ctx = Context::new();
        module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let pointer = local_cell(&mut ctx, block, u64_ty);
        let shared_pointer_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let different_pointee_ty: TypeHandle =
            MirPtrType::get_generic(&mut ctx, u32_ty, true).into();
        for result_type in [shared_pointer_ty, different_pointee_ty] {
            let cast = pointer_cast(&mut ctx, block, pointer, result_type);
            let alias = cast.deref(&ctx).get_result(0);
            assert!(!is_local_cell_pointer_alias_cast(&ctx, cast));
            assert!(super::local_cell(&ctx, alias).is_none());
        }
        let read_only_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, false).into();
        let read_only = pointer_cast(&mut ctx, block, pointer, read_only_type);
        let read_only_alias = read_only.deref(&ctx).get_result(0);
        assert!(is_local_cell_pointer_alias_cast(&ctx, read_only));
        assert!(super::local_cell(&ctx, read_only_alias).is_some());

        let mutable_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, true).into();
        let escalation = pointer_cast(&mut ctx, block, read_only_alias, mutable_type);
        let escalated_alias = escalation.deref(&ctx).get_result(0);
        assert!(!is_local_cell_pointer_alias_cast(&ctx, escalation));
        assert!(super::local_cell(&ctx, escalated_alias).is_none());
        assert!(!local_cell_is_closed(&ctx, pointer));
    }

    #[test]
    fn semantic_pointer_carriers_allow_only_shared_generic_address_space_roundtrips() {
        let mut ctx = Context::new();
        module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let shared_type: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let generic_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, true).into();
        let shared = undef(&mut ctx, block, shared_type);
        let to_generic = pointer_cast(&mut ctx, block, shared, generic_type);
        assert!(is_semantic_pointer_carrier_cast(&ctx, to_generic));
        let generic = to_generic.deref(&ctx).get_result(0);
        let to_shared = pointer_cast(&mut ctx, block, generic, shared_type);
        assert!(is_semantic_pointer_carrier_cast(&ctx, to_shared));

        let global_type: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::GLOBAL).into();
        let global = undef(&mut ctx, block, global_type);
        let global_to_generic = pointer_cast(&mut ctx, block, global, generic_type);
        assert!(!is_semantic_pointer_carrier_cast(&ctx, global_to_generic));

        let generic_u32_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, true).into();
        let wrong_pointee = pointer_cast(&mut ctx, block, shared, generic_u32_type);
        assert!(!is_semantic_pointer_carrier_cast(&ctx, wrong_pointee));

        let read_only_shared_type: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, false, address_space::SHARED).into();
        let read_only_shared = undef(&mut ctx, block, read_only_shared_type);
        let mutability_escalation = pointer_cast(&mut ctx, block, read_only_shared, generic_type);
        assert!(!is_semantic_pointer_carrier_cast(
            &ctx,
            mutability_escalation
        ));
    }

    #[test]
    fn rejects_pipeline_lifecycle_state_from_the_wrong_role_root() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let barrier_base = undef(&mut ctx, block, barrier_ty);
        let make =
            CuteTmaLoadPipelineMakeOp::new(&mut ctx, barrier_base, 3, 8, 8, 8).get_operation();
        make.insert_at_back(block, &ctx);
        let pipeline = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(&ctx);
        let producer = CutePipelineStateNewOp::new(&mut ctx, CutePipelineStateAttr::producer(3))
            .get_operation();
        producer.insert_at_back(block, &ctx);
        let producer = CutePipelineStateNewOp::wrap(producer);
        let producer_slot = producer.slot(&ctx);
        let producer_phase = producer.phase(&ctx);
        CutePipelineConsumerWaitOp::new(
            &mut ctx,
            pipeline,
            producer_slot,
            producer_phase,
            CutePipelineStateAttr::consumer(3),
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        let error = verify_cute_semantics(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("state facts differ"), "{error}");
    }

    #[test]
    fn rejects_producer_tail_that_precedes_its_lifecycle() {
        let mut ctx = Context::new();
        module_top(&mut ctx);
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let root = undef(&mut ctx, block, u64_ty).defining_op().unwrap();
        let tail = undef(&mut ctx, block, u64_ty).defining_op().unwrap();
        let acquire = undef(&mut ctx, block, u64_ty).defining_op().unwrap();
        let expect = undef(&mut ctx, block, u64_ty).defining_op().unwrap();
        let advance = undef(&mut ctx, block, u64_ty).defining_op().unwrap();

        let error = require_producer_pipeline_order(&ctx, root, acquire, expect, advance, tail)
            .unwrap_err()
            .to_string();
        assert!(error.contains("producer pipeline CFG"), "{error}");
    }

    #[test]
    fn rejects_tma_copy_with_an_arbitrary_typed_completion_barrier() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let arbitrary_barrier = undef(&mut ctx, block, barrier_ty);
        let layout =
            ComposedLayout::from_layout("(1,16):(16,1)".parse().unwrap(), OffsetUnit::Elements);
        let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let descriptor = undef(&mut ctx, block, descriptor_ty);
        let shared = undef(&mut ctx, block, shared_ty);
        let capacity = undef(&mut ctx, block, u64_ty);
        let row = undef(&mut ctx, block, u64_ty);
        let column = undef(&mut ctx, block, u64_ty);
        let source =
            CuteTmaGmemViewOp::new(&mut ctx, descriptor, u8_ty, layout.clone()).get_operation();
        source.insert_at_back(block, &ctx);
        let destination =
            CuteTmaSmemViewOp::new(&mut ctx, shared, capacity, u8_ty, layout, 16).get_operation();
        destination.insert_at_back(block, &ctx);
        let source_view = source.deref(&ctx).get_result(0);
        let destination_view = destination.deref(&ctx).get_result(0);
        CuteTmaCopy2dOp::new(
            &mut ctx,
            source_view,
            destination_view,
            row,
            column,
            arbitrary_barrier,
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        let error = verify_cute_semantics(&ctx, module).unwrap_err().to_string();
        assert!(
            error.contains("completion barrier is not the result of an expect_tx"),
            "{error}"
        );
    }

    #[test]
    fn rejects_mma_base_and_capacity_from_unrelated_same_typed_overlays() {
        let mut ctx = Context::new();
        let (_module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let first_base = undef(&mut ctx, block, shared_ty);
        let first_capacity = undef(&mut ctx, block, u64_ty);
        let second_base = undef(&mut ctx, block, shared_ty);
        let second_capacity = undef(&mut ctx, block, u64_ty);
        let first = CuteSmemTensorOverlayOp::new(&mut ctx, first_base, first_capacity, u64_ty)
            .get_operation();
        first.insert_at_back(block, &ctx);
        let second = CuteSmemTensorOverlayOp::new(&mut ctx, second_base, second_capacity, u64_ty)
            .get_operation();
        second.insert_at_back(block, &ctx);
        let operations: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        let facts = build_mma_facts(&ctx, &operations);

        let error = require_smem_overlay_pair(
            &facts,
            CuteSmemTensorOverlayOp::wrap(first).base(&ctx),
            CuteSmemTensorOverlayOp::wrap(second).capacity(&ctx),
            "test MMA carrier",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("different shared overlays"), "{error}");
    }

    #[test]
    fn rejects_mma_overlay_with_a_valid_partition_edge_and_an_unrelated_load() {
        let mut ctx = Context::new();
        let (_module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let base = undef(&mut ctx, block, shared_ty);
        let capacity = undef(&mut ctx, block, u64_ty);
        let overlay =
            CuteSmemTensorOverlayOp::new(&mut ctx, base, capacity, u64_ty).get_operation();
        overlay.insert_at_back(block, &ctx);
        let overlay = CuteSmemTensorOverlayOp::wrap(overlay);
        let overlay_base = overlay.base(&ctx);
        let overlay_capacity = overlay.capacity(&ctx);
        let warp_n = undef(&mut ctx, block, u64_ty);
        let k_half = undef(&mut ctx, block, u64_ty);
        let placement =
            ComposedLayout::from_layout("(128,32):(32,1)".parse().unwrap(), OffsetUnit::Elements);
        CuteMmaPartitionBOp::new(
            &mut ctx,
            overlay_base,
            overlay_capacity,
            warp_n,
            k_half,
            u64_ty,
            CuteTiledMmaPlanAttr::mxf4_128x128x128(placement),
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        Operation::new(
            &mut ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![u8_ty],
            vec![overlay_base],
            vec![],
            0,
        )
        .insert_at_back(block, &ctx);
        let operations: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        let facts = build_mma_facts(&ctx, &operations);

        let error = verify_mma_use_closure(&ctx, &operations, &facts)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsupported outgoing use"), "{error}");
        assert!(error.contains("mir.load"), "{error}");
    }

    #[test]
    fn rejects_epilogue_writer_and_issuer_in_mutually_exclusive_preorder_blocks() {
        let mut ctx = Context::new();
        let (_module, module_block) = module_top(&mut ctx);
        let function_type = FunctionType::get(&ctx, vec![], vec![]);
        let function = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, function, TypeAttr::new(function_type.into()))
            .set_symbol_name(&mut ctx, "epilogue_cfg".try_into().unwrap());
        function.insert_at_back(module_block, &ctx);
        let region = function.deref(&ctx).get_region(0);
        let entry = BasicBlock::new(&mut ctx, None, vec![]);
        entry.insert_at_back(region, &ctx);
        let writer = BasicBlock::new(&mut ctx, None, vec![]);
        writer.insert_at_back(region, &ctx);
        let issuer = BasicBlock::new(&mut ctx, None, vec![]);
        issuer.insert_at_back(region, &ctx);
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let condition = undef(&mut ctx, entry, i1_ty);
        cond_branch(&mut ctx, entry, condition, writer, issuer);
        let acquire = undef(&mut ctx, writer, u64_ty).defining_op().unwrap();
        let reusable = undef(&mut ctx, writer, u64_ty).defining_op().unwrap();
        let ready = undef(&mut ctx, writer, u64_ty).defining_op().unwrap();
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(writer, &ctx);
        let half_zero = undef(&mut ctx, issuer, u64_ty).defining_op().unwrap();
        let commit = undef(&mut ctx, issuer, u64_ty).defining_op().unwrap();
        let tail = undef(&mut ctx, issuer, u64_ty).defining_op().unwrap();
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(issuer, &ctx);

        let error =
            require_epilogue_cfg_links(&ctx, acquire, reusable, ready, half_zero, commit, tail)
                .unwrap_err()
                .to_string();
        assert!(error.contains("epilogue CFG"), "{error}");
    }

    #[test]
    fn rejects_expect_tx_when_direct_tma_bytes_do_not_match() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let barrier_base = undef(&mut ctx, block, barrier_ty);
        let make =
            CuteTmaLoadPipelineMakeOp::new(&mut ctx, barrier_base, 3, 8, 8, 8).get_operation();
        make.insert_at_back(block, &ctx);
        let pipeline = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(&ctx);
        let state = CutePipelineStateAttr::producer(3);
        let state_root = CutePipelineStateNewOp::new(&mut ctx, state).get_operation();
        state_root.insert_at_back(block, &ctx);
        let slot = CutePipelineStateNewOp::wrap(state_root).slot(&ctx);
        let expect =
            CutePipelineProducerExpectTxOp::new(&mut ctx, pipeline, slot, state).get_operation();
        expect.insert_at_back(block, &ctx);

        let layout =
            ComposedLayout::from_layout("(1,16):(16,1)".parse().unwrap(), OffsetUnit::Elements);
        let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let descriptor = undef(&mut ctx, block, descriptor_ty);
        let shared = undef(&mut ctx, block, shared_ty);
        let capacity = undef(&mut ctx, block, u64_ty);
        let row = undef(&mut ctx, block, u64_ty);
        let column = undef(&mut ctx, block, u64_ty);
        let source =
            CuteTmaGmemViewOp::new(&mut ctx, descriptor, u8_ty, layout.clone()).get_operation();
        source.insert_at_back(block, &ctx);
        let destination =
            CuteTmaSmemViewOp::new(&mut ctx, shared, capacity, u8_ty, layout, 16).get_operation();
        destination.insert_at_back(block, &ctx);
        let source_view = source.deref(&ctx).get_result(0);
        let destination_view = destination.deref(&ctx).get_result(0);
        let completion = CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(&ctx);
        CuteTmaCopy2dOp::new(
            &mut ctx,
            source_view,
            destination_view,
            row,
            column,
            completion,
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        let error = verify_cute_semantics(&ctx, module).unwrap_err().to_string();
        assert!(error.contains("promises 8 bytes"), "{error}");
        assert!(error.contains("move 16 bytes"), "{error}");
    }
}
