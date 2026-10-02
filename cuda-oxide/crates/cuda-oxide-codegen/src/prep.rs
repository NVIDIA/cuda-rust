/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::mir_pass_registry::{MirPassStage, SelectedMirPasses};
use crate::verify::verify_operation;
use dialect_cute::epilogue_ops::{
    CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp,
    CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp, CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp,
    CuteTmaStoreTailOp,
};
use dialect_cute::gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp};
use dialect_cute::gemv_ops::{
    CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
    CuteScaledViewRowOp, CuteTensorMake2DOp,
};
use dialect_cute::pipeline_ops::{
    CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
    CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
    CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
    CuteTmaLoadPipelineMakeOp,
};
use dialect_cute::scheduler_ops::{
    CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp, CuteSchedulerNew1dOp,
    CuteWorkTileCoordinatesOp,
};
use dialect_cute::sm100_ops::*;
use dialect_cute::smem_mma_ops::{
    CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
    CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
};
use dialect_cute::tensor_ops::{
    CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
    CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
    CuteTensorZippedDivideOp,
};
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirAllocaOp, MirCastOp};
use dialect_mir::types::MirPtrType;
use pliron::context::{Context, Ptr};
use pliron::irbuild::{
    listener::Recorder,
    rewriter::{IRRewriter, Rewriter},
};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::Typed;

/// Controls the reusable dialect-mir preparation stage.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirPreparation<'a> {
    /// Promote stack slots to SSA and run annotation-driven loop unrolling.
    pub promote_and_unroll: bool,
    /// Print preparation-pass progress notes to stderr. Threaded from the
    /// pipeline's `BackendOptions`; the scalarization passes read this flag
    /// instead of the environment (loop unrolling still checks
    /// `CUDA_OXIDE_VERBOSE` on its own).
    pub verbose: bool,
    /// Optional pass pipeline; `None` or empty preserves the defaults.
    pub mir_pass_pipeline: Option<&'a str>,
}

/// Verify and prepare a dialect-mir module before LLVM lowering.
///
/// The one shared post-translation orchestrator calls this helper for both the
/// rustc and standalone frontends.
#[doc(hidden)]
pub fn prepare_mir_module(
    ctx: &mut Context,
    module: Ptr<Operation>,
    preparation: MirPreparation<'_>,
) -> Result<(), PipelineError> {
    verify_operation(ctx, module, "module")?;
    // Capture this before any optional preparation pass mutates the module.
    // The extra promotion pass belongs only to the high-level tensor slice;
    // ordinary modules keep their existing preparation pipeline.
    let has_high_level_cute_operations = has_high_level_cute_operations(ctx, module);
    let has_pass_pipeline = preparation
        .mir_pass_pipeline
        .is_some_and(|pipeline| !pipeline.trim().is_empty());
    if !preparation.promote_and_unroll {
        if has_pass_pipeline {
            return Err(PipelineError::InvalidMirPassPipeline(
                "optional MIR passes are unavailable with full variable debug info".to_string(),
            ));
        }
        if has_high_level_cute_operations {
            return Err(PipelineError::Lowering(
                "high-level CuTe operations require MIR variable promotion; full variable debug information is not supported for this kernel"
                    .to_string(),
            ));
        }
        return Ok(());
    }

    // Validate every requested pass before any transformation runs. This keeps
    // an invalid later-stage name from leaving a module partially transformed.
    let selected_passes = select_optional_mir_passes(preparation.mir_pass_pipeline)?;

    let mut analyses = pliron::pass::AnalysisManager::default();
    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PrePreparation,
        &mut analyses,
    )?;

    // Compiler-owned multi-result device operations are temporarily adapted
    // to Rust aggregate return values by the MIR importer. Prove and remove
    // that ABI-only boundary before mem2reg so the independent register values
    // remain SSA all the way into LLVM/PTX lowering.
    mir_transforms::forward_compiler_result_bundles::forward_compiler_result_bundles(
        module,
        ctx,
        &mut analyses,
        preparation.verbose,
    )
    .map_err(|error| PipelineError::Verification {
        name: "compiler-result forwarding".to_string(),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;
    verify_operation(ctx, module, "module post-compiler-result-forwarding")?;

    // A by-value aggregate argument initially lives in a MIR alloca. Read-only
    // field/index projections make that alloca non-promotable even though the
    // original entry-block argument is already an SSA value. Canonicalize the
    // validated pointer chains back to value extraction before mem2reg.
    mir_transforms::scalarize_borrowed_aggregate_reads::canonicalize_read_only_aggregate_arguments(
        module,
        ctx,
        preparation.verbose,
    );
    verify_operation(
        ctx,
        module,
        "module post-borrowed-aggregate-read-canonicalization",
    )?;
    pliron::opts::mem2reg::mem2reg(module, ctx, &mut analyses).map_err(|error| {
        PipelineError::Verification {
            name: "mem2reg".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        }
    })?;
    verify_operation(ctx, module, "module post-mem2reg")?;

    // Source-level dynamic indices commonly pass through temporary MIR slots.
    // After mem2reg, bounded forms such as `index % C` are direct SSA values,
    // while result-bundle allocas blocked by dynamic array projections remain.
    // Re-run the same fail-closed forwarding proof here so those bundles can
    // be recovered without teaching the pass to trace arbitrary stack slots.
    mir_transforms::forward_compiler_result_bundles::forward_compiler_result_bundles(
        module,
        ctx,
        &mut analyses,
        preparation.verbose,
    )
    .map_err(|error| PipelineError::Verification {
        name: "post-mem2reg compiler-result forwarding".to_string(),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;
    verify_operation(
        ctx,
        module,
        "module post-mem2reg-compiler-result-forwarding",
    )?;

    // Formation passes that need promoted SSA values but must still see the
    // original loop CFG run here. In particular, a reduction formation pass
    // cannot safely infer a source loop once generic unrolling has cloned it.
    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PostMem2Reg,
        &mut analyses,
    )?;

    // An immutable aggregate pointer argument in an always-inline helper can
    // still retain dynamic field/array pointer chains after mem2reg. Recover
    // bounded read-only accesses in typed MIR before LLVM lowering.
    mir_transforms::scalarize_borrowed_aggregate_reads::
        canonicalize_bounded_borrowed_pointer_arguments(module, ctx, preparation.verbose);
    verify_operation(
        ctx,
        module,
        "module post-borrowed-pointer-read-canonicalization",
    )?;

    if has_high_level_cute_operations {
        // Borrowing a local CuTe value (`&mut tensor`) can leave a `mir.cast`
        // on its stack-slot pointer. The cast changes only the pointer kind,
        // but prevents mem2reg from removing the slot. CuTe ghost values must
        // stay in SSA, so remove these casts when every use is a load or store
        // address. The next mem2reg pass can then remove the CuTe slot.
        fold_ghost_slot_kind_only_casts(ctx, module);
        verify_operation(ctx, module, "module post-ghost-slot-kind-cast-folding")?;

        // The first mem2reg pass can remove a temporary pointer slot that was
        // the only reason another tensor slot looked as though its address
        // escaped. That newly exposed inner slot is promotable only after the
        // pointer cleanup above, so give this tensor pipeline one more pass.
        pliron::opts::mem2reg::mem2reg(module, ctx, &mut analyses).map_err(|error| {
            PipelineError::Verification {
                name: "second mem2reg".to_string(),
                message: error.disp(ctx).to_string(),
                operation: None,
            }
        })?;
        verify_operation(ctx, module, "module post-second-mem2reg")?;
    }

    mir_transforms::unroll::unroll_annotated_loops(module, ctx, &mut analyses).map_err(
        |error| PipelineError::Verification {
            name: "loop-unroll".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        },
    )?;
    verify_operation(ctx, module, "module post-unroll")?;

    run_optional_mir_passes(
        ctx,
        module,
        &selected_passes,
        MirPassStage::PostPreparation,
        &mut analyses,
    )
}

fn has_high_level_cute_operations(ctx: &Context, root: Ptr<Operation>) -> bool {
    let mut pending = vec![root];
    while let Some(operation) = pending.pop() {
        if Operation::is_op::<CuteTensorMakeOp>(operation, ctx)
            || Operation::is_op::<CuteTensorZippedDivideOp>(operation, ctx)
            || Operation::is_op::<CuteTensorSliceOp>(operation, ctx)
            || Operation::is_op::<CuteTensorIsFullOp>(operation, ctx)
            || Operation::is_op::<CuteTensorBaseOp>(operation, ctx)
            || Operation::is_op::<CuteTensorLoadIntoOp>(operation, ctx)
            || Operation::is_op::<CuteTensorStoreFromOp>(operation, ctx)
            || Operation::is_op::<CuteTensorStoreElementAbsOp>(operation, ctx)
            || Operation::is_op::<CuteTensorMake2DOp>(operation, ctx)
            || Operation::is_op::<CuteScaledViewMakeOp>(operation, ctx)
            || Operation::is_op::<CuteScaledViewRowOp>(operation, ctx)
            || Operation::is_op::<CuteScaledViewKTileOp>(operation, ctx)
            || Operation::is_op::<CuteScaledViewLoadOp>(operation, ctx)
            || Operation::is_op::<CuteDotOp>(operation, ctx)
            || Operation::is_op::<CuteTmaGmemViewOp>(operation, ctx)
            || Operation::is_op::<CuteTmaSmemViewOp>(operation, ctx)
            || Operation::is_op::<CuteTmaCopy2dOp>(operation, ctx)
            || Operation::is_op::<CuteSchedulerNew1dOp>(operation, ctx)
            || Operation::is_op::<CuteSchedulerHasWorkOp>(operation, ctx)
            || Operation::is_op::<CuteSchedulerCurrentOp>(operation, ctx)
            || Operation::is_op::<CuteWorkTileCoordinatesOp>(operation, ctx)
            || Operation::is_op::<CuteSchedulerAdvanceOp>(operation, ctx)
            || Operation::is_op::<CuteTmaLoadPipelineMakeOp>(operation, ctx)
            || Operation::is_op::<CuteTmaLoadPipelineInitOp>(operation, ctx)
            || Operation::is_op::<CutePipelineStateNewOp>(operation, ctx)
            || Operation::is_op::<CutePipelineStateSlotOp>(operation, ctx)
            || Operation::is_op::<CutePipelineStateAdvanceOp>(operation, ctx)
            || Operation::is_op::<CutePipelineProducerAcquireOp>(operation, ctx)
            || Operation::is_op::<CutePipelineProducerExpectTxOp>(operation, ctx)
            || Operation::is_op::<CutePipelineConsumerWaitOp>(operation, ctx)
            || Operation::is_op::<CutePipelineConsumerReleaseOp>(operation, ctx)
            || Operation::is_op::<CutePipelineProducerTailOp>(operation, ctx)
            || Operation::is_op::<CuteSmemTensorOverlayOp>(operation, ctx)
            || Operation::is_op::<CuteTiledMmaSliceOp>(operation, ctx)
            || Operation::is_op::<CuteFragmentFillOp>(operation, ctx)
            || Operation::is_op::<CuteMmaLoadScalesOp>(operation, ctx)
            || Operation::is_op::<CuteFragmentSliceKOp>(operation, ctx)
            || Operation::is_op::<CuteMmaLoadAOp>(operation, ctx)
            || Operation::is_op::<CuteMmaPartitionBOp>(operation, ctx)
            || Operation::is_op::<CuteTiledGemmOp>(operation, ctx)
            || Operation::is_op::<CuteEpilogueSmemOverlayOp>(operation, ctx)
            || Operation::is_op::<CuteEpilogueWarpSliceOp>(operation, ctx)
            || Operation::is_op::<CuteEpilogueStoreFragmentOp>(operation, ctx)
            || Operation::is_op::<CuteEpilogueSyncOp>(operation, ctx)
            || Operation::is_op::<CuteEpilogueHalfOp>(operation, ctx)
            || Operation::is_op::<CuteTmaStoreAcquireOp>(operation, ctx)
            || Operation::is_op::<CuteTmaStoreCommitOp>(operation, ctx)
            || Operation::is_op::<CuteTmaStoreTailOp>(operation, ctx)
            || Operation::is_op::<CuteTmaStore2dSemanticOp>(operation, ctx)
            || Operation::is_op::<CuteSm100TmemAllocOp>(operation, ctx)
            || Operation::is_op::<CuteSm100TmaStoreOp>(operation, ctx)
            || Operation::is_op::<CuteSm100StoreCommitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100StoreAcquireOp>(operation, ctx)
            || Operation::is_op::<CuteSm100StoreTailOp>(operation, ctx)
            || Operation::is_op::<CuteSm100TmemDeallocOp>(operation, ctx)
            || Operation::is_op::<CuteSm100TiledMmaOp>(operation, ctx)
            || Operation::is_op::<CuteSm100TmemEpilogueOp>(operation, ctx)
            || Operation::is_op::<CuteSm100ClusterTmaLoadOp>(operation, ctx)
            || Operation::is_op::<CuteSm100PipelineInitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100PipelineAcquireOp>(operation, ctx)
            || Operation::is_op::<CuteSm100PipelineExpectOp>(operation, ctx)
            || Operation::is_op::<CuteSm100PipelineWaitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100PipelineReleaseOp>(operation, ctx)
            || Operation::is_op::<CuteSm100AccumulatorInitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100AccumulatorAcquireOp>(operation, ctx)
            || Operation::is_op::<CuteSm100AccumulatorCommitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100AccumulatorWaitOp>(operation, ctx)
            || Operation::is_op::<CuteSm100AccumulatorReleaseOp>(operation, ctx)
        {
            return true;
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    false
}

/// Fold pointer-kind-only `mir.cast`s on alloca slot pointers whose pointee
/// contains a CuTe ghost type, replacing the cast result with the slot
/// pointer. Only the pointer kind (and its implied authority metadata) is
/// erased; the pointee and address space must match exactly. This runs before
/// mem2reg so a `&mut tensor` reborrow does not keep the ghost slot alive.
fn fold_ghost_slot_kind_only_casts(ctx: &mut Context, root: Ptr<Operation>) {
    let mut foldable = Vec::new();
    let mut pending = vec![root];
    while let Some(operation) = pending.pop() {
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                for child in block.deref(ctx).iter(ctx) {
                    pending.push(child);
                    if let Some(cast) = Operation::get_op::<MirCastOp>(child, ctx)
                        && is_ghost_slot_kind_only_cast(ctx, &cast)
                    {
                        foldable.push(child);
                    }
                }
            }
        }
    }
    let mut rewriter = IRRewriter::<Recorder>::default();
    for cast in foldable {
        let (base, result) = {
            let operation = cast.deref(ctx);
            (operation.get_operand(0), operation.get_result(0))
        };
        result.replace_all_uses_with(ctx, &base);
        rewriter.erase_operation(ctx, cast);
    }
}

fn is_ghost_slot_kind_only_cast(ctx: &Context, cast: &MirCastOp) -> bool {
    if cast
        .get_attr_cast_kind(ctx)
        .is_none_or(|kind| *kind != MirCastKindAttr::PtrToPtr)
    {
        return false;
    }
    let operation = cast.get_operation().deref(ctx);
    if operation.get_num_operands() != 1 || operation.get_num_results() != 1 {
        return false;
    }
    let base = operation.get_operand(0);
    let Some(defining_op) = base.defining_op() else {
        return false;
    };
    if Operation::get_op::<MirAllocaOp>(defining_op, ctx).is_none() {
        return false;
    }
    let base_ty = base.get_type(ctx);
    let result_ty = operation.get_result(0).get_type(ctx);
    let base_ty_ref = base_ty.deref(ctx);
    let result_ty_ref = result_ty.deref(ctx);
    let (Some(base_ptr), Some(result_ptr)) = (
        base_ty_ref.downcast_ref::<MirPtrType>(),
        result_ty_ref.downcast_ref::<MirPtrType>(),
    ) else {
        return false;
    };
    if base_ptr.pointee != result_ptr.pointee
        || base_ptr.address_space != result_ptr.address_space
        || !dialect_cute::verify::type_contains_cute_ghost(ctx, base_ptr.pointee)
    {
        return false;
    }
    // Fail closed: folding retypes the value at each use, which is only
    // meaning-preserving where the pointer is consumed as a memory address
    // (operand 0 of a load or store). A use that carries the pointer as data
    // (stored value, call argument, branch operand, ...) must keep the cast.
    operation.get_result(0).uses(ctx).iter().all(|r#use| {
        let user = r#use.user_op();
        r#use.find_index(ctx) == 0
            && (Operation::get_op::<dialect_mir::ops::MirLoadOp>(user, ctx).is_some()
                || Operation::get_op::<dialect_mir::ops::MirStoreOp>(user, ctx).is_some())
    })
}

fn select_optional_mir_passes(spec: Option<&str>) -> Result<SelectedMirPasses, PipelineError> {
    crate::mir_pass_registry::registry()
        .select(spec.unwrap_or_default())
        .map_err(|error| PipelineError::InvalidMirPassPipeline(error.to_string()))
}

fn run_optional_mir_passes(
    ctx: &mut Context,
    module: Ptr<Operation>,
    selected: &SelectedMirPasses,
    stage: MirPassStage,
    analyses: &mut pliron::pass::AnalysisManager,
) -> Result<(), PipelineError> {
    // Nothing selected for this stage: skip the pass-manager run and the extra
    // module verification so a default build pays nothing for the hooks.
    if !selected.has_stage(stage) {
        return Ok(());
    }

    let mut passes = crate::mir_pass_registry::registry().build_stage_pipeline(selected, stage);

    <pliron::pass::Passes as pliron::pass::PassManager>::run_pass(
        &mut passes,
        module,
        ctx,
        analyses,
    )
    .map_err(|error| PipelineError::Verification {
        name: format!("optional MIR passes ({stage:?})"),
        message: error.disp(ctx).to_string(),
        operation: None,
    })?;

    verify_operation(
        ctx,
        module,
        &format!("module post-optional-mir-passes ({stage:?})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_cute::{
        attributes::{
            CutePipelineStateAttr, CuteTensorAccessAttr, CuteTensorRoleAttr, CuteTileGridAttr,
            CuteTiledMmaPlanAttr, CuteTmaStorePipelineAttr,
        },
        epilogue_ops::CuteTmaStoreAcquireOp,
        gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp},
        gemv_ops::CuteTensorMake2DOp,
        layout::{ComposedLayout, Layout, OffsetUnit, Swizzle},
        pipeline_ops::CutePipelineStateNewOp,
        scheduler_ops::CuteSchedulerNew1dOp,
        smem_mma_ops::CuteTiledMmaSliceOp,
        tensor_ops::CuteTensorMakeOp,
    };
    use dialect_mir::ops::{
        MirAllocaOp, MirFuncOp, MirLoadOp, MirReturnOp, MirStoreOp, MirUndefOp,
    };
    use dialect_mir::types::{MirPtrType, address_space};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface};
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FP32Type, FunctionType, IntegerType, Signedness};
    use pliron::op::Op;
    use pliron::r#type::TypeHandle;

    fn assert_high_level_cute_op_is_detected<O: Op>() {
        let mut ctx = Context::new();
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        Operation::new(
            &mut ctx,
            O::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }

    #[test]
    fn debug_mode_rejects_requested_mir_passes() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = prepare_mir_module(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: false,
                verbose: false,
                mir_pass_pipeline: Some("future-pass"),
            },
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }

    #[test]
    fn invalid_staged_pipeline_is_rejected_before_preparation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let error = prepare_mir_module(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: true,
                verbose: false,
                mir_pass_pipeline: Some("missing-pass"),
            },
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::InvalidMirPassPipeline(_)));
    }

    #[test]
    fn second_mem2reg_promotes_slot_exposed_by_pointer_slot_promotion() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let element: TypeHandle = FP32Type::get(&ctx).into();
        let inner_pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, element, true).into();
        let outer_pointer: TypeHandle =
            MirPtrType::get_generic(&mut ctx, inner_pointer, true).into();
        let global_pointer: TypeHandle =
            MirPtrType::get(&mut ctx, element, false, address_space::GLOBAL).into();
        let length_type: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(&ctx, vec![element], vec![element]);
        let function = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(&mut ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(&mut ctx, "kernel".try_into().unwrap());
        module.append_operation(&mut ctx, function, 0);

        let region = function.deref(&ctx).get_region(0);
        let entry = BasicBlock::new(&mut ctx, None, vec![element]);
        entry.insert_at_back(region, &ctx);
        let input = entry.deref(&ctx).get_argument(0);

        // This harmless view makes the fixture take the tensor-only second
        // promotion path. The nested slots below reproduce the writable
        // `store_linear(&mut self)` shape that exposed the ordering bug.
        let data = MirUndefOp::new(&mut ctx, global_pointer).get_operation();
        data.insert_at_back(entry, &ctx);
        let data = data.deref(&ctx).get_result(0);
        let length = MirUndefOp::new(&mut ctx, length_type).get_operation();
        length.insert_at_back(entry, &ctx);
        let length = length.deref(&ctx).get_result(0);
        CuteTensorMakeOp::new(
            &mut ctx,
            data,
            length,
            element,
            element,
            CuteTensorAccessAttr::ReadOnly,
            4,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);

        let inner = Operation::new(
            &mut ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![inner_pointer],
            vec![],
            vec![],
            0,
        );
        inner.insert_at_back(entry, &ctx);
        let inner_slot = inner.deref(&ctx).get_result(0);

        let outer = Operation::new(
            &mut ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![outer_pointer],
            vec![],
            vec![],
            0,
        );
        outer.insert_at_back(entry, &ctx);
        let outer_slot = outer.deref(&ctx).get_result(0);

        for (address, value) in [(outer_slot, inner_slot), (inner_slot, input)] {
            Operation::new(
                &mut ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![address, value],
                vec![],
                0,
            )
            .insert_at_back(entry, &ctx);
        }

        let load_pointer = Operation::new(
            &mut ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![inner_pointer],
            vec![outer_slot],
            vec![],
            0,
        );
        load_pointer.insert_at_back(entry, &ctx);
        let loaded_pointer = load_pointer.deref(&ctx).get_result(0);
        let load_value = Operation::new(
            &mut ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element],
            vec![loaded_pointer],
            vec![],
            0,
        );
        load_value.insert_at_back(entry, &ctx);
        let loaded_value = load_value.deref(&ctx).get_result(0);
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![loaded_value],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);

        prepare_mir_module(
            &mut ctx,
            module.get_operation(),
            MirPreparation {
                promote_and_unroll: true,
                verbose: false,
                mir_pass_pipeline: None,
            },
        )
        .unwrap();

        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .all(|operation| !Operation::is_op::<MirAllocaOp>(operation, &ctx)),
            "both the outer pointer slot and the inner value slot must be promoted"
        );
    }

    #[test]
    fn full_variable_debug_rejects_high_level_tensor_operations_early() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        let element: TypeHandle = FP32Type::get(&ctx).into();
        let pointer: TypeHandle =
            MirPtrType::get(&mut ctx, element, false, address_space::GLOBAL).into();
        let length: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();

        let pointer_value = MirUndefOp::new(&mut ctx, pointer).get_operation();
        pointer_value.insert_at_back(block, &ctx);
        let pointer_value = pointer_value.deref(&ctx).get_result(0);
        let length_value = MirUndefOp::new(&mut ctx, length).get_operation();
        length_value.insert_at_back(block, &ctx);
        let length_value = length_value.deref(&ctx).get_result(0);
        CuteTensorMakeOp::new(
            &mut ctx,
            pointer_value,
            length_value,
            element,
            element,
            CuteTensorAccessAttr::ReadOnly,
            4,
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        let error = prepare_mir_module(
            &mut ctx,
            module_operation,
            MirPreparation {
                promote_and_unroll: false,
                verbose: false,
                mir_pass_pipeline: None,
            },
        )
        .unwrap_err();
        let PipelineError::Lowering(message) = error else {
            panic!("expected a full-debug lowering rejection");
        };
        assert!(message.contains("CuTe operations require MIR variable promotion"));
    }

    #[test]
    fn gemv_view_operations_enable_the_tensor_preparation_gate() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        let storage: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointer: TypeHandle =
            MirPtrType::get(&mut ctx, storage, false, address_space::GLOBAL).into();
        let usize_type: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();

        let pointer_value = MirUndefOp::new(&mut ctx, pointer).get_operation();
        pointer_value.insert_at_back(block, &ctx);
        let pointer_value = pointer_value.deref(&ctx).get_result(0);
        let size_value = MirUndefOp::new(&mut ctx, usize_type).get_operation();
        size_value.insert_at_back(block, &ctx);
        let size_value = size_value.deref(&ctx).get_result(0);
        CuteTensorMake2DOp::new_e2m1(
            &mut ctx,
            pointer_value,
            size_value,
            size_value,
            size_value,
            CuteTensorRoleAttr::Mkl,
            16,
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }

    #[test]
    fn scheduler_operations_enable_the_cute_preparation_gate() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        CuteSchedulerNew1dOp::new(&mut ctx, CuteTileGridAttr::new(16, 16, 1))
            .get_operation()
            .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }

    #[test]
    fn pipeline_operations_enable_the_cute_preparation_gate() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        CutePipelineStateNewOp::new(&mut ctx, CutePipelineStateAttr::producer(3))
            .get_operation()
            .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }

    #[test]
    fn sm100_operations_enable_the_cute_preparation_gate() {
        assert_high_level_cute_op_is_detected::<CuteSm100TmaStoreOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100StoreCommitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100StoreAcquireOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100StoreTailOp>();

        assert_high_level_cute_op_is_detected::<CuteSm100TmemAllocOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100TmemDeallocOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100TiledMmaOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100TmemEpilogueOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100ClusterTmaLoadOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100PipelineInitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100PipelineAcquireOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100PipelineExpectOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100PipelineWaitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100PipelineReleaseOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100AccumulatorInitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100AccumulatorAcquireOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100AccumulatorCommitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100AccumulatorWaitOp>();
        assert_high_level_cute_op_is_detected::<CuteSm100AccumulatorReleaseOp>();
    }

    #[test]
    fn tma_operations_enable_the_cute_preparation_gate() {
        assert_high_level_cute_op_is_detected::<CuteTmaGmemViewOp>();
        assert_high_level_cute_op_is_detected::<CuteTmaSmemViewOp>();
        assert_high_level_cute_op_is_detected::<CuteTmaCopy2dOp>();
    }

    #[test]
    fn shared_mma_operations_enable_the_cute_preparation_gate() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        let lane_type: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let lane = MirUndefOp::new(&mut ctx, lane_type).get_operation();
        lane.insert_at_back(block, &ctx);
        let lane = lane.deref(&ctx).get_result(0);

        let inner: Layout = "(128,32):(32,1)".parse().unwrap();
        let placement =
            ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements).unwrap();
        CuteTiledMmaSliceOp::new(
            &mut ctx,
            lane,
            CuteTiledMmaPlanAttr::mxf4_128x128x128(placement),
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }

    #[test]
    fn epilogue_operations_enable_the_cute_preparation_gate() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let module_operation = module.get_operation();
        let region = module_operation.deref(&ctx).get_region(0);
        let block = region.deref(&ctx).iter(&ctx).next().unwrap();
        CuteTmaStoreAcquireOp::new(&mut ctx, CuteTmaStorePipelineAttr::new(1))
            .get_operation()
            .insert_at_back(block, &ctx);

        assert!(has_high_level_cute_operations(&ctx, module_operation));
    }
}
