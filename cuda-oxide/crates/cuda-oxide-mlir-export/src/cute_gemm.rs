/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM120 block-scaled GEMM mappings for the official CUTLASS 4.7 compiler.
//!
//! The source dialect keeps scheduler, pipeline, TMA, MMA, and epilogue
//! operations visible. This pack removes compile-time handles and maps their
//! runtime work to operations accepted by the public PRE_COMPILED pipeline
//! in `libCutlassCompiler.so`.

use dialect_cute::{
    attributes::{
        CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
        CuteMmaCarrierKindAttr, CutePipelineRoleAttr, CutePipelineStateAttr,
        CuteTensorAddressSpaceAttr, CuteTileGridAttr, CuteTiledMmaPlanAttr,
        CuteTmaStorePipelineAttr,
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
        CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
    },
    types::{
        CuteEpilogueTileType, CuteSmemTensorType, CuteTmaLoadPipelineType, CuteTmaViewType,
        CuteWorkTileType,
    },
};
use dialect_mir::{
    ops::{MirConstructTupleOp, MirExternSharedOp, MirInsertFieldOp},
    types::{MirArrayType, MirStructType, MirTupleType},
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirBlock, MlirLocation, MlirOperation, MlirRegion, MlirResult,
    MlirType, MlirValueUse, OperationInput, OperationTranslation, TranslationError,
    TranslationRegistry, TranslationSession, TypeTranslation,
};

const PIPELINE_WAIT_TICKS: i128 = 10_000_000;
const LDMATRIX_BYTE_SWIZZLE_MASK: u64 = 0x180;
const LDMATRIX_BYTE_SWIZZLE_SHIFT: u64 = 3;
const COUNTED_CTA_SYNC_OP: &str = "nvvm.barrier.cta.sync";

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
    registry.register_attribute::<CuteTiledMmaPlanAttr>(DropAttribute)?;
    registry.register_attribute::<CuteMmaCarrierKindAttr>(DropAttribute)?;
    registry.register_attribute::<CuteTmaStorePipelineAttr>(DropAttribute)?;
    registry.register_attribute::<CuteCountedCtaBarrierAttr>(DropAttribute)?;
    registry.register_attribute::<CuteEpilogueSyncPhaseAttr>(DropAttribute)?;
    registry.register_attribute::<CuteEpilogueHalfAttr>(DropAttribute)?;

    registry.register_operation::<MirConstructTupleOp>(ConstructAggregateTranslation)?;
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

struct WorkTileTypeTranslation;

impl TypeTranslation for WorkTileTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        source
            .deref(ctx)
            .downcast_ref::<CuteWorkTileType>()
            .ok_or_else(|| "expected cute.work_tile".to_owned())?
            .verify(ctx)
            .map_err(stringify)?;
        Ok(MlirType::Integer(64))
    }
}

struct LoadPipelineTypeTranslation;

impl TypeTranslation for LoadPipelineTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        source
            .deref(ctx)
            .downcast_ref::<CuteTmaLoadPipelineType>()
            .ok_or_else(|| "expected cute.tma_load_pipeline".to_owned())?
            .verify(ctx)
            .map_err(stringify)?;
        shared_pointer_type()
    }
}

struct TmaViewTypeTranslation;

impl TypeTranslation for TmaViewTypeTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        let source_ref = source.deref(ctx);
        let view = source_ref
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| "expected cute.tma_view".to_owned())?;
        view.verify(ctx).map_err(stringify)?;
        let tensor = view
            .tensor_view(ctx)
            .ok_or_else(|| "cute.tma_view lost its tensor facts".to_owned())?;
        match tensor.space {
            CuteTensorAddressSpaceAttr::Gmem => generic_pointer_type(),
            CuteTensorAddressSpaceAttr::Smem => shared_pointer_type(),
        }
    }
}

struct StaticContractTypeTranslation;

impl TypeTranslation for StaticContractTypeTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: TypeHandle,
        _registry: &TranslationRegistry,
    ) -> Result<MlirType, String> {
        Ok(MlirType::Tuple(vec![]))
    }
}

struct ConstructAggregateTranslation;

impl OperationTranslation for ConstructAggregateTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        MirConstructTupleOp::new(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("mir.construct_tuple", &input, 1, input.operands.len())?;
        build_aggregate(
            session,
            input.results[0].clone(),
            input.operands,
            &input.location,
        )
    }
}

struct InsertFieldTranslation;

impl OperationTranslation for InsertFieldTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = MirInsertFieldOp::new(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("mir.insert_field", &input, 1, 2)?;
        let index = semantic
            .get_attr_insert_index(ctx)
            .ok_or_else(|| "mir.insert_field has no insert_index".to_owned())?
            .0 as i64;
        let mut target = operation(
            "llvm.insertvalue",
            input.results,
            input.operands,
            &input.location,
        )?;
        target
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![index]));
        Ok(vec![target])
    }
}

struct ExternSharedTranslation;

impl OperationTranslation for ExternSharedTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = MirExternSharedOp::new(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("extern_byte_offset");
        input.attributes.remove("extern_alignment");
        expect_shape("mir.extern_shared", &input, 1, 0)?;
        let offset = semantic.get_byte_offset_value(ctx);
        let alignment = semantic.get_alignment_value(ctx);
        let cute_pointer = MlirType::dialect(format!("!cute.ptr<i8, smem, align<{alignment}>>"))?;
        let (raw_result, raw) = fresh(session, cute_pointer);
        let raw_op = operation(
            "cute_nvgpu.arch.get_dyn_smem",
            vec![raw_result],
            vec![],
            &input.location,
        )?;
        let (base_result, base) = if offset == 0 {
            (
                input.results[0].clone(),
                MlirValueUse {
                    id: input.results[0].id,
                    ty: input.results[0].ty.clone(),
                },
            )
        } else {
            fresh(session, shared_pointer_type()?)
        };
        let bridge = operation(
            "builtin.unrealized_conversion_cast",
            vec![base_result],
            vec![raw],
            &input.location,
        )?;
        if offset == 0 {
            return Ok(vec![raw_op, bridge]);
        }

        let index = constant_value(
            session,
            MlirType::Integer(64),
            offset as i128,
            &input.location,
        )?;
        let mut gep = operation(
            "llvm.getelementptr",
            input.results,
            vec![base, index.1],
            &input.location,
        )?;
        gep_properties(&mut gep, MlirType::Integer(8), vec![i32::MIN]);
        Ok(vec![raw_op, bridge, index.0, gep])
    }
}

struct SchedulerNewTranslation;

impl OperationTranslation for SchedulerNewTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteSchedulerNew1dOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.scheduler_new_1d", &input, 2, 0)?;
        let mut operations = vec![];
        for (name, result) in [
            ("nvvm.read.ptx.sreg.ctaid.x", input.results[0].clone()),
            ("nvvm.read.ptx.sreg.nctaid.x", input.results[1].clone()),
        ] {
            let (narrow_result, narrow) = fresh(session, MlirType::Integer(32));
            operations.push(operation(
                name,
                vec![narrow_result],
                vec![],
                &input.location,
            )?);
            operations.push(operation(
                "arith.extui",
                vec![result],
                vec![narrow],
                &input.location,
            )?);
        }
        Ok(operations)
    }
}

struct SchedulerHasWorkTranslation;

impl OperationTranslation for SchedulerHasWorkTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteSchedulerHasWorkOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("cute.scheduler_has_work", &input, 1, 1)?;
        let total = semantic
            .tile_grid(ctx)
            .and_then(CuteTileGridAttr::total_tiles)
            .ok_or_else(|| "scheduler tile count overflowed".to_owned())?;
        let (constant, total_use) = constant_value(
            session,
            MlirType::Integer(64),
            u64_bits(total),
            &input.location,
        )?;
        let mut compare = operation(
            "arith.cmpi",
            input.results,
            vec![input.operands[0].clone(), total_use],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 6, // unsigned less-than
                ty: MlirType::Integer(64),
            },
        );
        Ok(vec![constant, compare])
    }
}

struct WorkTileCoordinatesTranslation;

impl OperationTranslation for WorkTileCoordinatesTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteWorkTileCoordinatesOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.work_tile_coordinates", &input, 4, 1)?;
        let tile_ty = source.deref(ctx).get_operand(0).get_type(ctx);
        let tile_ref = tile_ty.deref(ctx);
        let grid = tile_ref
            .downcast_ref::<CuteWorkTileType>()
            .ok_or_else(|| "work_tile_coordinates operand lost its grid".to_owned())?
            .grid;
        let linear = input.operands[0].clone();
        let mut operations = vec![];
        let zero = emit_i64_constant(session, &mut operations, 0, &input.location)?;
        operations.push(binary(
            "arith.addi",
            input.results[0].clone(),
            linear.clone(),
            zero,
            &input.location,
        )?);

        let m_width = emit_i64_constant(session, &mut operations, grid.m_tiles, &input.location)?;
        let quotient_m = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            linear.clone(),
            m_width.clone(),
            &input.location,
        )?;
        let consumed_m = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            quotient_m.clone(),
            m_width,
            &input.location,
        )?;
        operations.push(binary(
            "arith.subi",
            input.results[1].clone(),
            linear,
            consumed_m,
            &input.location,
        )?);

        let n_width = emit_i64_constant(session, &mut operations, grid.n_tiles, &input.location)?;
        let batch = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            quotient_m.clone(),
            n_width.clone(),
            &input.location,
        )?;
        let consumed_n = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            batch.clone(),
            n_width,
            &input.location,
        )?;
        operations.push(binary(
            "arith.subi",
            input.results[2].clone(),
            quotient_m,
            consumed_n,
            &input.location,
        )?);
        let zero = emit_i64_constant(session, &mut operations, 0, &input.location)?;
        operations.push(binary(
            "arith.addi",
            input.results[3].clone(),
            batch,
            zero,
            &input.location,
        )?);
        Ok(operations)
    }
}

struct SchedulerAdvanceTranslation;

impl OperationTranslation for SchedulerAdvanceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteSchedulerAdvanceOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.scheduler_advance", &input, 1, 2)?;
        let mut operations = vec![];
        let wrapped = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            input.operands[0].clone(),
            input.operands[1].clone(),
            &input.location,
        )?;
        let (overflow_result, overflow) = fresh(session, MlirType::Integer(1));
        let mut compare = operation(
            "arith.cmpi",
            vec![overflow_result],
            vec![wrapped.clone(), input.operands[0].clone()],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 6, // unsigned less-than
                ty: MlirType::Integer(64),
            },
        );
        operations.push(compare);
        let maximum = emit_i64_constant(session, &mut operations, u64::MAX, &input.location)?;
        operations.push(operation(
            "arith.select",
            input.results,
            vec![overflow, maximum, wrapped],
            &input.location,
        )?);
        Ok(operations)
    }
}

struct IdentityTranslation(&'static str);

impl OperationTranslation for IdentityTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        expect_shape(self.0, &input, 1, 1)?;
        Ok(vec![operation(
            "builtin.unrealized_conversion_cast",
            input.results,
            input.operands,
            &input.location,
        )?])
    }
}

struct IdentityFirstOperandTranslation(&'static str);

impl OperationTranslation for IdentityFirstOperandTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if input.results.len() != 1 || input.operands.is_empty() {
            return Err(format!(
                "{} needs one result and at least one operand",
                self.0
            ));
        }
        if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty()
        {
            return Err(format!("{} retained source-only structure", self.0));
        }
        Ok(vec![operation(
            "builtin.unrealized_conversion_cast",
            input.results,
            vec![input.operands[0].clone()],
            &input.location,
        )?])
    }
}

// The remaining definitions in this file are kept below the scalar foundation
// so each official-parser rung can be tested independently.

struct PipelineInitTranslation;

impl OperationTranslation for PipelineInitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTmaLoadPipelineInitOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.tma_load_pipeline_init", &input, 0, 2)?;
        let pipeline = load_pipeline_from_operand(ctx, source, 0)?;
        let mut operations = vec![];

        let (tid_result, tid) = fresh(session, MlirType::Integer(32));
        operations.push(operation(
            "nvvm.read.ptx.sreg.tid.x",
            vec![tid_result],
            vec![],
            &input.location,
        )?);
        let (initializer_result, initializer) = fresh(session, MlirType::Integer(1));
        let mut compare = operation(
            "arith.cmpi",
            vec![initializer_result],
            vec![tid, input.operands[1].clone()],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 0, // equal
                ty: MlirType::Integer(64),
            },
        );
        operations.push(compare);

        let mut init_operations = vec![];
        for stage in 0..pipeline.stages {
            for (slot, count) in [
                (stage, 1i128),
                (pipeline.stages + stage, i128::from(pipeline.consumer_warps)),
            ] {
                let barrier = barrier_pointer_constant(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    slot,
                    &input.location,
                )?;
                let count = emit_i32_constant(session, &mut operations, count, &input.location)?;
                init_operations.push(mbarrier_init(barrier, count, &input.location)?);
            }
        }
        init_operations.push(operation("scf.yield", vec![], vec![], &input.location)?);
        let mut conditional = operation("scf.if", vec![], vec![initializer], &input.location)?;
        conditional.regions = vec![
            MlirRegion {
                blocks: vec![MlirBlock {
                    id: session.fresh_block(),
                    arguments: vec![],
                    operations: init_operations,
                }],
            },
            MlirRegion { blocks: vec![] },
        ];
        operations.push(conditional);
        operations.push(operation(
            "nvvm.fence.mbarrier.init",
            vec![],
            vec![],
            &input.location,
        )?);
        operations.push(operation("nvvm.barrier", vec![], vec![], &input.location)?);
        Ok(operations)
    }
}

struct PipelineStateNewTranslation;

impl OperationTranslation for PipelineStateNewTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CutePipelineStateNewOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("cute.pipeline_state_new", &input, 2, 0)?;
        let state = semantic
            .state(ctx)
            .ok_or_else(|| "pipeline_state_new has no state".to_owned())?;
        Ok(vec![
            integer_constant(input.results[0].clone(), 0, &input.location)?,
            integer_constant(
                input.results[1].clone(),
                i128::from(state.role == CutePipelineRoleAttr::Producer),
                &input.location,
            )?,
        ])
    }
}

struct PipelineStateSlotTranslation;

impl OperationTranslation for PipelineStateSlotTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CutePipelineStateSlotOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.pipeline_state_slot", &input, 1, 1)?;
        Ok(vec![operation(
            "arith.extui",
            input.results,
            input.operands,
            &input.location,
        )?])
    }
}

struct PipelineStateAdvanceTranslation;

impl OperationTranslation for PipelineStateAdvanceTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CutePipelineStateAdvanceOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        expect_shape("cute.pipeline_state_advance", &input, 2, 2)?;
        let stages = semantic
            .state(ctx)
            .ok_or_else(|| "pipeline_state_advance has no state".to_owned())?
            .stages;
        emit_pipeline_state_advance(
            session,
            input.operands[0].clone(),
            input.operands[1].clone(),
            stages,
            Some([input.results[0].clone(), input.results[1].clone()]),
            &input.location,
        )
        .map(|(operations, _)| operations)
    }
}

struct PipelineWaitTranslation {
    empty_ring: bool,
    name: &'static str,
}

impl OperationTranslation for PipelineWaitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        match self.name {
            "cute.pipeline_producer_acquire" => CutePipelineProducerAcquireOp::wrap(source)
                .verify(ctx)
                .map_err(stringify)?,
            "cute.pipeline_consumer_wait" => CutePipelineConsumerWaitOp::wrap(source)
                .verify(ctx)
                .map_err(stringify)?,
            _ => return Err(format!("unknown pipeline wait mapping {}", self.name)),
        }
        expect_shape(self.name, &input, 0, 3)?;
        let pipeline = load_pipeline_from_operand(ctx, source, 0)?;
        let mut operations = vec![];
        let barrier = barrier_pointer_dynamic(
            session,
            &mut operations,
            input.operands[0].clone(),
            input.operands[1].clone(),
            self.empty_ring.then_some(pipeline.stages),
            &input.location,
        )?;
        let ticks = emit_i32_constant(
            session,
            &mut operations,
            PIPELINE_WAIT_TICKS,
            &input.location,
        )?;
        // This is the LLVM/NVVM high-level potentially-blocking operation.
        // Its official definition lowers to a retry loop and guarantees phase
        // completion; it is not one raw bounded PTX probe.
        operations.push(operation(
            "nvvm.mbarrier.try_wait.parity",
            vec![],
            vec![barrier, input.operands[2].clone(), ticks],
            &input.location,
        )?);
        Ok(operations)
    }
}

struct PipelineExpectTxTranslation;

impl OperationTranslation for PipelineExpectTxTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CutePipelineProducerExpectTxOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.pipeline_producer_expect_tx", &input, 1, 2)?;
        let pipeline = load_pipeline_from_operand(ctx, source, 0)?;
        let mut operations = vec![];
        let barrier = barrier_pointer_dynamic(
            session,
            &mut operations,
            input.operands[0].clone(),
            input.operands[1].clone(),
            None,
            &input.location,
        )?;
        let bytes = emit_i32_constant(
            session,
            &mut operations,
            i128::from(pipeline.transaction_bytes),
            &input.location,
        )?;
        operations.push(operation(
            "nvvm.mbarrier.arrive.expect_tx",
            vec![],
            vec![barrier.clone(), bytes],
            &input.location,
        )?);
        operations.push(operation(
            "builtin.unrealized_conversion_cast",
            input.results,
            vec![barrier],
            &input.location,
        )?);
        Ok(operations)
    }
}

struct PipelineReleaseTranslation;

impl OperationTranslation for PipelineReleaseTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CutePipelineConsumerReleaseOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.pipeline_consumer_release", &input, 0, 2)?;
        let pipeline = load_pipeline_from_operand(ctx, source, 0)?;
        let mut operations = vec![];
        let barrier = barrier_pointer_dynamic(
            session,
            &mut operations,
            input.operands[0].clone(),
            input.operands[1].clone(),
            Some(pipeline.stages),
            &input.location,
        )?;

        let (lane_result, lane) = fresh(session, MlirType::Integer(32));
        operations.push(operation(
            "nvvm.read.ptx.sreg.laneid",
            vec![lane_result],
            vec![],
            &input.location,
        )?);
        let zero = emit_i32_constant(session, &mut operations, 0, &input.location)?;
        let (elected_result, elected) = fresh(session, MlirType::Integer(1));
        let mut compare = operation(
            "arith.cmpi",
            vec![elected_result],
            vec![lane, zero],
            &input.location,
        )?;
        compare.properties.insert(
            "predicate".into(),
            MlirAttribute::Integer {
                value: 0,
                ty: MlirType::Integer(64),
            },
        );
        operations.push(compare);

        let mut arrive = operation(
            "nvvm.mbarrier.arrive",
            vec![],
            vec![barrier],
            &input.location,
        )?;
        let yield_ = operation("scf.yield", vec![], vec![], &input.location)?;
        arrive.location = input.location.clone();
        let mut conditional = operation("scf.if", vec![], vec![elected], &input.location)?;
        conditional.regions = vec![
            MlirRegion {
                blocks: vec![MlirBlock {
                    id: session.fresh_block(),
                    arguments: vec![],
                    operations: vec![arrive, yield_],
                }],
            },
            MlirRegion { blocks: vec![] },
        ];
        operations.push(conditional);
        Ok(operations)
    }
}

struct PipelineTailTranslation;

impl OperationTranslation for PipelineTailTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CutePipelineProducerTailOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.pipeline_producer_tail", &input, 0, 3)?;
        let pipeline = load_pipeline_from_operand(ctx, source, 0)?;
        let mut slot = input.operands[1].clone();
        let mut phase = input.operands[2].clone();
        let mut operations = vec![];
        for stage in 0..pipeline.stages {
            let barrier = barrier_pointer_dynamic(
                session,
                &mut operations,
                input.operands[0].clone(),
                slot.clone(),
                Some(pipeline.stages),
                &input.location,
            )?;
            let ticks = emit_i32_constant(
                session,
                &mut operations,
                PIPELINE_WAIT_TICKS,
                &input.location,
            )?;
            operations.push(operation(
                "nvvm.mbarrier.try_wait.parity",
                vec![],
                vec![barrier, phase.clone(), ticks],
                &input.location,
            )?);
            if stage + 1 != pipeline.stages {
                let (advance, next) = emit_pipeline_state_advance(
                    session,
                    slot,
                    phase,
                    pipeline.stages,
                    None,
                    &input.location,
                )?;
                operations.extend(advance);
                [slot, phase] = next;
            }
        }
        Ok(operations)
    }
}

struct TmaLoadTranslation;

impl OperationTranslation for TmaLoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTmaCopy2dOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.tma_copy_2d", &input, 0, 5)?;
        let (rows, cols) = tma_tile_extents(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        let mut operations = vec![];
        let column = tma_coordinate(
            session,
            &mut operations,
            input.operands[3].clone(),
            cols,
            &input.location,
        )?;
        let row = tma_coordinate(
            session,
            &mut operations,
            input.operands[2].clone(),
            rows,
            &input.location,
        )?;
        let mut copy = operation(
            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
            vec![],
            vec![
                input.operands[1].clone(),
                input.operands[0].clone(),
                column,
                row,
                input.operands[4].clone(),
            ],
            &input.location,
        )?;
        copy.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 0, 0, 0]),
        );
        copy.attributes
            .insert("isCTAOnly".into(), MlirAttribute::Bool(true));
        operations.push(copy);
        Ok(operations)
    }
}

struct TmaStoreTranslation;

impl OperationTranslation for TmaStoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTmaStore2dSemanticOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.tma_store_2d", &input, 0, 4)?;
        let (rows, cols) = tma_tile_extents(ctx, source.deref(ctx).get_operand(0).get_type(ctx))?;
        let mut operations = vec![];
        let column = tma_coordinate(
            session,
            &mut operations,
            input.operands[3].clone(),
            cols,
            &input.location,
        )?;
        let row = tma_coordinate(
            session,
            &mut operations,
            input.operands[2].clone(),
            rows,
            &input.location,
        )?;
        let mut copy = operation(
            "nvvm.cp.async.bulk.tensor.global.shared.cta",
            vec![],
            vec![
                input.operands[1].clone(),
                input.operands[0].clone(),
                column,
                row,
            ],
            &input.location,
        )?;
        copy.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 0, 0]),
        );
        operations.push(copy);
        Ok(operations)
    }
}

struct TmaStoreWaitTranslation {
    name: &'static str,
}

impl OperationTranslation for TmaStoreWaitTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let group = match self.name {
            "cute.tma_store_acquire" => {
                let semantic = CuteTmaStoreAcquireOp::wrap(source);
                semantic.verify(ctx).map_err(stringify)?;
                semantic
                    .pipeline(ctx)
                    .and_then(CuteTmaStorePipelineAttr::max_pending)
                    .ok_or_else(|| "tma_store_acquire has no valid pipeline".to_owned())?
            }
            "cute.tma_store_tail" => {
                CuteTmaStoreTailOp::wrap(source)
                    .verify(ctx)
                    .map_err(stringify)?;
                0
            }
            _ => return Err(format!("unknown TMA store wait mapping {}", self.name)),
        };
        expect_shape(self.name, &input, 0, 0)?;
        let mut wait = operation(
            "nvvm.cp.async.bulk.wait_group",
            vec![],
            vec![],
            &input.location,
        )?;
        wait.properties.insert(
            "group".into(),
            MlirAttribute::Integer {
                value: i128::from(group),
                ty: MlirType::Integer(32),
            },
        );
        wait.attributes.insert("read".into(), MlirAttribute::Unit);
        Ok(vec![wait])
    }
}

struct NoOperandRenameTranslation(&'static str);

impl OperationTranslation for NoOperandRenameTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if self.0 == "nvvm.cp.async.bulk.commit.group" {
            CuteTmaStoreCommitOp::wrap(source)
                .verify(ctx)
                .map_err(stringify)?;
        }
        expect_shape("no-operand rename", &input, 0, 0)?;
        Ok(vec![operation(self.0, vec![], vec![], &input.location)?])
    }
}

struct EpilogueHalfTranslation;

impl OperationTranslation for EpilogueHalfTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteEpilogueHalfOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("epilogue_half_tile");
        expect_shape("cute.epilogue_half", &input, 2, 1)?;
        let tile = semantic
            .tile(ctx)
            .ok_or_else(|| "epilogue_half has no tile".to_owned())?;
        let half = semantic
            .half(ctx)
            .ok_or_else(|| "epilogue_half has no index".to_owned())?;
        let half_elements = tile
            .half_elements()
            .ok_or_else(|| "epilogue half size overflowed".to_owned())?;
        let mut operations = vec![];
        if half.0 == 0 {
            operations.push(operation(
                "builtin.unrealized_conversion_cast",
                vec![input.results[0].clone()],
                vec![input.operands[0].clone()],
                &input.location,
            )?);
        } else {
            let index =
                emit_i64_constant(session, &mut operations, half_elements, &input.location)?;
            let mut gep = operation(
                "llvm.getelementptr",
                vec![input.results[0].clone()],
                vec![input.operands[0].clone(), index],
                &input.location,
            )?;
            gep_properties(&mut gep, epilogue_gep_element_type(), vec![i32::MIN]);
            operations.push(gep);
        }
        operations.push(integer_constant(
            input.results[1].clone(),
            i128::from(half_elements),
            &input.location,
        )?);
        Ok(operations)
    }
}

struct EpilogueSyncTranslation;

impl OperationTranslation for EpilogueSyncTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let semantic = CuteEpilogueSyncOp::wrap(source);
        semantic.verify(ctx).map_err(stringify)?;
        input.attributes.remove("epilogue_sync_tile");
        expect_shape("cute.epilogue_sync", &input, 0, 1)?;
        let phase = semantic
            .phase(ctx)
            .ok_or_else(|| "epilogue_sync has no phase".to_owned())?;
        let barrier = semantic
            .barrier(ctx)
            .ok_or_else(|| "epilogue_sync has no barrier".to_owned())?;
        let threads = barrier
            .participant_threads()
            .ok_or_else(|| "epilogue barrier thread count overflowed".to_owned())?;
        let mut operations = vec![];
        let id = emit_i32_constant(
            session,
            &mut operations,
            i128::from(barrier.barrier_id),
            &input.location,
        )?;
        let threads = emit_i32_constant(
            session,
            &mut operations,
            i128::from(threads),
            &input.location,
        )?;
        operations.extend(epilogue_sync_operations(
            phase,
            id,
            threads,
            &input.location,
        )?);
        Ok(operations)
    }
}

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
    registry.register_operation::<CuteEpilogueStoreFragmentOp>(EpilogueStoreTranslation)?;
    Ok(())
}

struct StaticMultiIdentityTranslation {
    name: &'static str,
    attributes: &'static [&'static str],
}

impl OperationTranslation for StaticMultiIdentityTranslation {
    fn translate(
        &self,
        _ctx: &Context,
        _source: Ptr<Operation>,
        mut input: OperationInput,
        _session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        for attribute in self.attributes {
            input.attributes.remove(*attribute);
        }
        if input.results.len() != input.operands.len() {
            return Err(format!(
                "{} expected matching result and operand counts",
                self.name
            ));
        }
        expect_shape(self.name, &input, input.results.len(), input.operands.len())?;
        input
            .results
            .into_iter()
            .zip(input.operands)
            .map(|(result, operand)| {
                operation(
                    "builtin.unrealized_conversion_cast",
                    vec![result],
                    vec![operand],
                    &input.location,
                )
            })
            .collect()
    }
}

#[derive(Clone)]
struct CarrierLayout {
    target: MlirType,
    kind: CarrierKind,
}

#[derive(Clone)]
enum CarrierKind {
    Leaf,
    Aggregate(Vec<(i64, CarrierLayout)>),
}

impl CarrierLayout {
    fn from_source(
        ctx: &Context,
        source: TypeHandle,
        session: &TranslationSession<'_>,
        depth: usize,
    ) -> Result<Self, String> {
        if depth > 64 {
            return Err("MMA carrier aggregate is unexpectedly deep".into());
        }
        let target = session.translate_type(source).map_err(stringify)?;
        let source_ref = source.deref(ctx);
        if let Some(array) = source_ref.downcast_ref::<MirArrayType>() {
            let count = usize::try_from(array.size())
                .map_err(|_| "MMA carrier array is too large".to_owned())?;
            let child = Self::from_source(ctx, array.element_type(), session, depth + 1)?;
            return Ok(Self {
                target,
                kind: CarrierKind::Aggregate(
                    (0..count)
                        .map(|index| (index as i64, child.clone()))
                        .collect(),
                ),
            });
        }

        let fields = if let Some(tuple) = source_ref.downcast_ref::<MirTupleType>() {
            Some(tuple.get_types().to_vec())
        } else {
            source_ref
                .downcast_ref::<MirStructType>()
                .map(|structure| structure.field_types.clone())
        };
        let Some(fields) = fields else {
            return Ok(Self {
                target,
                kind: CarrierKind::Leaf,
            });
        };
        drop(source_ref);

        let plan = crate::mir_memory::aggregate_plan(ctx, source, |ty| {
            session.translate_type(ty).map_err(stringify)
        })?;
        if plan.ty != target {
            return Err("MMA carrier aggregate disagrees with MIR type translation".into());
        }
        let mut children = Vec::new();
        for (declaration, field) in fields.into_iter().enumerate() {
            let Some(slot) = plan.decl_to_slot[declaration] else {
                continue;
            };
            children.push((
                slot as i64,
                Self::from_source(ctx, field, session, depth + 1)?,
            ));
        }
        Ok(Self {
            target,
            kind: CarrierKind::Aggregate(children),
        })
    }

    fn leaf_count(&self) -> usize {
        match &self.kind {
            CarrierKind::Leaf => 1,
            CarrierKind::Aggregate(children) => {
                children.iter().map(|(_, child)| child.leaf_count()).sum()
            }
        }
    }
}

fn extract_carrier(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    layout: &CarrierLayout,
    leaves: &mut Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<(), String> {
    match &layout.kind {
        CarrierKind::Leaf => {
            leaves.push(value);
            Ok(())
        }
        CarrierKind::Aggregate(children) => {
            for (slot, child) in children {
                let (result, extracted) = fresh(session, child.target.clone());
                let mut extract = operation(
                    "llvm.extractvalue",
                    vec![result],
                    vec![value.clone()],
                    location,
                )?;
                extract
                    .properties
                    .insert("position".into(), MlirAttribute::DenseI64Array(vec![*slot]));
                operations.push(extract);
                extract_carrier(session, operations, extracted, child, leaves, location)?;
            }
            Ok(())
        }
    }
}

fn rebuild_carrier(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    layout: &CarrierLayout,
    leaves: &[MlirValueUse],
    final_result: MlirResult,
    location: &MlirLocation,
) -> Result<(), String> {
    let mut cursor = 0;
    build_carrier_node(
        session,
        operations,
        layout,
        leaves,
        &mut cursor,
        Some(final_result),
        location,
    )?;
    if cursor != leaves.len() {
        return Err(format!(
            "MMA carrier consumed {cursor} leaves but received {}",
            leaves.len()
        ));
    }
    Ok(())
}

fn build_carrier_node(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    layout: &CarrierLayout,
    leaves: &[MlirValueUse],
    cursor: &mut usize,
    final_result: Option<MlirResult>,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    if matches!(layout.kind, CarrierKind::Leaf) {
        let leaf = leaves
            .get(*cursor)
            .cloned()
            .ok_or_else(|| "MMA carrier did not receive enough leaves".to_owned())?;
        *cursor += 1;
        if let Some(result) = final_result {
            let use_ = MlirValueUse {
                id: result.id,
                ty: result.ty.clone(),
            };
            operations.push(operation(
                "builtin.unrealized_conversion_cast",
                vec![result],
                vec![leaf],
                location,
            )?);
            return Ok(use_);
        }
        return Ok(leaf);
    }

    let CarrierKind::Aggregate(children) = &layout.kind else {
        unreachable!()
    };
    let mut built_children = Vec::with_capacity(children.len());
    for (slot, child) in children {
        built_children.push((
            *slot,
            build_carrier_node(session, operations, child, leaves, cursor, None, location)?,
        ));
    }

    let poison_result = if built_children.is_empty() {
        final_result
            .clone()
            .unwrap_or_else(|| fresh(session, layout.target.clone()).0)
    } else {
        fresh(session, layout.target.clone()).0
    };
    let mut aggregate = MlirValueUse {
        id: poison_result.id,
        ty: poison_result.ty.clone(),
    };
    operations.push(operation(
        "llvm.mlir.poison",
        vec![poison_result],
        vec![],
        location,
    )?);
    let child_count = built_children.len();
    for (index, (slot, child)) in built_children.into_iter().enumerate() {
        let result = if index + 1 == child_count {
            final_result
                .clone()
                .unwrap_or_else(|| fresh(session, layout.target.clone()).0)
        } else {
            fresh(session, layout.target.clone()).0
        };
        let next = MlirValueUse {
            id: result.id,
            ty: result.ty.clone(),
        };
        let mut insert = operation(
            "llvm.insertvalue",
            vec![result],
            vec![aggregate, child],
            location,
        )?;
        insert
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![slot]));
        operations.push(insert);
        aggregate = next;
    }
    Ok(aggregate)
}

struct FragmentFillTranslation;

impl OperationTranslation for FragmentFillTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteFragmentFillOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.fragment_fill", &input, 1, 1)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 64 {
            return Err("cute.fragment_fill needs exactly 64 f32 leaves".into());
        }
        let leaves = vec![input.operands[0].clone(); 64];
        let mut operations = vec![];
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &leaves,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

struct MmaLoadScalesTranslation;

impl OperationTranslation for MmaLoadScalesTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteMmaLoadScalesOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("mma_load_scales_a_view");
        input.attributes.remove("mma_load_scales_b_view");
        expect_shape("cute.mma_load_scales", &input, 1, 7)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 10 {
            return Err("cute.mma_load_scales needs exactly ten u32 leaves".into());
        }

        let mut operations = vec![];
        let lane = cast_int(
            session,
            &mut operations,
            input.operands[0].clone(),
            MlirType::Integer(64),
            "arith.extui",
            &input.location,
        )?;
        let four = emit_i64_constant(session, &mut operations, 4, &input.location)?;
        let q = emit_binary(
            session,
            &mut operations,
            "arith.divui",
            lane.clone(),
            four,
            &input.location,
        )?;
        let three = emit_i64_constant(session, &mut operations, 3, &input.location)?;
        let r = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane,
            three,
            &input.location,
        )?;
        let one = emit_i64_constant(session, &mut operations, 1, &input.location)?;
        let parity = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            r,
            one,
            &input.location,
        )?;
        let eight = emit_i64_constant(session, &mut operations, 8, &input.location)?;
        let parity_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            parity,
            eight,
            &input.location,
        )?;
        let a_provider = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            q.clone(),
            parity_band,
            &input.location,
        )?;
        let sixteen = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let a_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[5].clone(),
            sixteen.clone(),
            &input.location,
        )?;
        let a_row0 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            a_band,
            a_provider,
            &input.location,
        )?;
        let sixty_four = emit_i64_constant(session, &mut operations, 64, &input.location)?;
        let a_row1 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            a_row0.clone(),
            sixty_four,
            &input.location,
        )?;
        let b_band = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[6].clone(),
            sixteen,
            &input.location,
        )?;
        let b_row0 = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            b_band,
            q,
            &input.location,
        )?;
        let mut rows = vec![a_row0, a_row1, b_row0.clone()];
        for delta in [8_u64, 32, 40, 64, 72, 96, 104] {
            let delta = emit_i64_constant(session, &mut operations, delta, &input.location)?;
            rows.push(emit_binary(
                session,
                &mut operations,
                "arith.addi",
                b_row0.clone(),
                delta,
                &input.location,
            )?);
        }
        let mut words = Vec::with_capacity(10);
        for (index, row) in rows.into_iter().enumerate() {
            words.push(load_scale_word(
                session,
                &mut operations,
                if index < 2 {
                    input.operands[1].clone()
                } else {
                    input.operands[3].clone()
                },
                row,
                &input.location,
            )?);
        }
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &words,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

struct FragmentSliceKTranslation;

impl OperationTranslation for FragmentSliceKTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteFragmentSliceKOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        expect_shape("cute.fragment_slice_k", &input, 1, 2)?;
        let source_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_operand(0).get_type(ctx),
            session,
            0,
        )?;
        let result_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        let mut operations = vec![];
        let mut words = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[0].clone(),
            &source_layout,
            &mut words,
            &input.location,
        )?;
        if words.len() != 10 || result_layout.leaf_count() != 10 {
            return Err("cute.fragment_slice_k needs ten packed scale words".into());
        }
        let sixteen = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let shift64 = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            input.operands[1].clone(),
            sixteen,
            &input.location,
        )?;
        let shift = cast_int(
            session,
            &mut operations,
            shift64,
            MlirType::Integer(32),
            "arith.trunci",
            &input.location,
        )?;
        let mask = emit_i32_constant(session, &mut operations, 0xffff, &input.location)?;
        let mut selected = Vec::with_capacity(10);
        for word in words {
            let shifted = emit_binary(
                session,
                &mut operations,
                "arith.shrui",
                word,
                shift.clone(),
                &input.location,
            )?;
            selected.push(emit_binary(
                session,
                &mut operations,
                "arith.andi",
                shifted,
                mask.clone(),
                &input.location,
            )?);
        }
        rebuild_carrier(
            session,
            &mut operations,
            &result_layout,
            &selected,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

struct MmaLoadATranslation;

impl OperationTranslation for MmaLoadATranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteMmaLoadAOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("mma_load_a_view");
        expect_shape("cute.mma_load_a", &input, 1, 5)?;
        let layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_result(0).get_type(ctx),
            session,
            0,
        )?;
        if layout.leaf_count() != 8 {
            return Err("cute.mma_load_a needs exactly eight u32 leaves".into());
        }
        let mut operations = vec![];
        let first = emit_ldmatrix_x4(
            session,
            &mut operations,
            input.operands[1].clone(),
            input.operands[3].clone(),
            input.operands[4].clone(),
            input.operands[0].clone(),
            &input.location,
        )?;
        let four = emit_i64_constant(session, &mut operations, 4, &input.location)?;
        let second_row = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            input.operands[3].clone(),
            four,
            &input.location,
        )?;
        let second = emit_ldmatrix_x4(
            session,
            &mut operations,
            input.operands[1].clone(),
            second_row,
            input.operands[4].clone(),
            input.operands[0].clone(),
            &input.location,
        )?;
        let mut registers = first.to_vec();
        registers.extend(second);
        rebuild_carrier(
            session,
            &mut operations,
            &layout,
            &registers,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

fn emit_ldmatrix_x4(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    warp_tile_row: MlirValueUse,
    warp_tile_column: MlirValueUse,
    lane: MlirValueUse,
    location: &MlirLocation,
) -> Result<[MlirValueUse; 4], String> {
    let lane = cast_int(
        session,
        operations,
        lane,
        MlirType::Integer(64),
        "arith.extui",
        location,
    )?;
    let eight = emit_i64_constant(session, operations, 8, location)?;
    let submatrix = emit_binary(
        session,
        operations,
        "arith.divui",
        lane.clone(),
        eight.clone(),
        location,
    )?;
    let row_in = emit_binary(
        session,
        operations,
        "arith.remui",
        lane,
        eight.clone(),
        location,
    )?;
    let two = emit_i64_constant(session, operations, 2, location)?;
    let parity = emit_binary(
        session,
        operations,
        "arith.remui",
        submatrix.clone(),
        two.clone(),
        location,
    )?;
    let row_offset = emit_binary(
        session,
        operations,
        "arith.muli",
        parity,
        eight.clone(),
        location,
    )?;
    let sixteen = emit_i64_constant(session, operations, 16, location)?;
    let row = emit_binary(
        session,
        operations,
        "arith.muli",
        warp_tile_row,
        sixteen.clone(),
        location,
    )?;
    let row = emit_binary(session, operations, "arith.addi", row, row_offset, location)?;
    let row = emit_binary(session, operations, "arith.addi", row, row_in, location)?;

    let column = emit_binary(
        session,
        operations,
        "arith.muli",
        warp_tile_column,
        sixteen,
        location,
    )?;
    let sub_half = emit_binary(
        session,
        operations,
        "arith.divui",
        submatrix,
        two.clone(),
        location,
    )?;
    let column_offset = emit_binary(session, operations, "arith.muli", sub_half, eight, location)?;
    let column = emit_binary(
        session,
        operations,
        "arith.addi",
        column,
        column_offset,
        location,
    )?;

    // The verified shared layout is f16 (128,32):(32,1), converted to byte
    // strides (64,2), followed by S<2,3,3>.
    let sixty_four = emit_i64_constant(session, operations, 64, location)?;
    let row_bytes = emit_binary(session, operations, "arith.muli", row, sixty_four, location)?;
    let column_bytes = emit_binary(session, operations, "arith.muli", column, two, location)?;
    let plain = emit_binary(
        session,
        operations,
        "arith.addi",
        row_bytes,
        column_bytes,
        location,
    )?;
    // The source layout's element-unit S<2,3,3> becomes S<2,4,3>
    // after converting f16 elements to byte offsets.  Mask bits 8..7 and
    // shift them onto bits 5..4, preserving the 16-byte ldmatrix segment.
    let mask = emit_i64_constant(session, operations, LDMATRIX_BYTE_SWIZZLE_MASK, location)?;
    let source_bits = emit_binary(
        session,
        operations,
        "arith.andi",
        plain.clone(),
        mask,
        location,
    )?;
    let three = emit_i64_constant(session, operations, LDMATRIX_BYTE_SWIZZLE_SHIFT, location)?;
    let moved = emit_binary(
        session,
        operations,
        "arith.shrui",
        source_bits,
        three,
        location,
    )?;
    let byte_offset = emit_binary(session, operations, "arith.xori", plain, moved, location)?;
    let pointer = gep_value(
        session,
        operations,
        base,
        byte_offset,
        MlirType::Integer(8),
        location,
    )?;

    let result_type = MlirType::dialect("!llvm.struct<(i32, i32, i32, i32)>")?;
    let (matrix_result, matrix) = fresh(session, result_type);
    let mut load = operation(
        "nvvm.ldmatrix",
        vec![matrix_result],
        vec![pointer],
        location,
    )?;
    load.properties.insert(
        "layout".into(),
        MlirAttribute::dialect("#nvvm.mma_layout<row>")?,
    );
    load.properties.insert(
        "num".into(),
        MlirAttribute::Integer {
            value: 4,
            ty: MlirType::Integer(32),
        },
    );
    operations.push(load);
    let mut registers = Vec::with_capacity(4);
    for index in 0..4 {
        let (result, use_) = fresh(session, MlirType::Integer(32));
        let mut extract = operation(
            "llvm.extractvalue",
            vec![result],
            vec![matrix.clone()],
            location,
        )?;
        extract
            .properties
            .insert("position".into(), MlirAttribute::DenseI64Array(vec![index]));
        operations.push(extract);
        registers.push(use_);
    }
    registers
        .try_into()
        .map_err(|_| "ldmatrix did not produce four registers".to_owned())
}

struct TiledGemmTranslation;

impl OperationTranslation for TiledGemmTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteTiledGemmOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("tiled_gemm_b_view");
        expect_shape("cute.tiled_gemm", &input, 1, 8)?;
        let operation_ref = source.deref(ctx);
        let a_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(1).get_type(ctx),
            session,
            0,
        )?;
        let scales_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(6).get_type(ctx),
            session,
            0,
        )?;
        let accumulator_layout = CarrierLayout::from_source(
            ctx,
            operation_ref.get_operand(7).get_type(ctx),
            session,
            0,
        )?;
        let result_layout =
            CarrierLayout::from_source(ctx, operation_ref.get_result(0).get_type(ctx), session, 0)?;
        drop(operation_ref);

        let mut operations = vec![];
        let mut a = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[1].clone(),
            &a_layout,
            &mut a,
            &input.location,
        )?;
        let mut scales = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[6].clone(),
            &scales_layout,
            &mut scales,
            &input.location,
        )?;
        let mut accumulator = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[7].clone(),
            &accumulator_layout,
            &mut accumulator,
            &input.location,
        )?;
        if a.len() != 8
            || scales.len() != 10
            || accumulator.len() != 64
            || result_layout.leaf_count() != 64
        {
            return Err("cute.tiled_gemm carrier shape changed after verification".into());
        }

        for pair in 0..4_usize {
            let pair_offset =
                emit_i64_constant(session, &mut operations, (pair * 2) as u64, &input.location)?;
            let b_row = emit_binary(
                session,
                &mut operations,
                "arith.addi",
                input.operands[4].clone(),
                pair_offset,
                &input.location,
            )?;
            let loaded = emit_ldmatrix_x4(
                session,
                &mut operations,
                input.operands[2].clone(),
                b_row,
                input.operands[5].clone(),
                input.operands[0].clone(),
                &input.location,
            )?;
            let b_fragments = [
                [loaded[0].clone(), loaded[2].clone()],
                [loaded[1].clone(), loaded[3].clone()],
            ];
            for m in 0..2_usize {
                for (n_in_pair, b_fragment) in b_fragments.iter().enumerate() {
                    let n = pair * 2 + n_in_pair;
                    let cell = m * 8 + n;
                    let updated = emit_mxf4_mma(
                        session,
                        &mut operations,
                        &a[m * 4..m * 4 + 4],
                        b_fragment,
                        &accumulator[cell * 4..cell * 4 + 4],
                        scales[m].clone(),
                        scales[2 + n].clone(),
                        &input.location,
                    )?;
                    accumulator[cell * 4..cell * 4 + 4].clone_from_slice(&updated);
                }
            }
        }
        rebuild_carrier(
            session,
            &mut operations,
            &result_layout,
            &accumulator,
            input.results[0].clone(),
            &input.location,
        )?;
        Ok(operations)
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_mxf4_mma(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    a: &[MlirValueUse],
    b: &[MlirValueUse],
    accumulator: &[MlirValueUse],
    scale_a: MlirValueUse,
    scale_b: MlirValueUse,
    location: &MlirLocation,
) -> Result<[MlirValueUse; 4], String> {
    if a.len() != 4 || b.len() != 2 || accumulator.len() != 4 {
        return Err("SM120 MXF4 MMA received the wrong register count".into());
    }
    let scale_a = cast_int(
        session,
        operations,
        scale_a,
        MlirType::Integer(16),
        "arith.trunci",
        location,
    )?;
    let scale_b = cast_int(
        session,
        operations,
        scale_b,
        MlirType::Integer(16),
        "arith.trunci",
        location,
    )?;
    let mut operands = Vec::with_capacity(12);
    operands.extend_from_slice(a);
    operands.extend_from_slice(b);
    operands.extend_from_slice(accumulator);
    operands.extend([scale_a, scale_b]);

    let result_type = MlirType::Float(pliron_mlir_export::MlirFloatType::F32);
    let mut results = Vec::with_capacity(4);
    let mut uses = Vec::with_capacity(4);
    for _ in 0..4 {
        let (result, use_) = fresh(session, result_type.clone());
        results.push(result);
        uses.push(use_);
    }
    let mut operation_ = operation(
        "cute_nvgpu.arch.mma.SM120.block_scaled",
        results,
        operands,
        location,
    )?;
    operation_.properties.insert(
        "a_type".into(),
        MlirAttribute::Type(MlirType::Float(pliron_mlir_export::MlirFloatType::Other(
            "f4E2M1FN".into(),
        ))),
    );
    operation_.properties.insert(
        "b_type".into(),
        MlirAttribute::Type(MlirType::Float(pliron_mlir_export::MlirFloatType::Other(
            "f4E2M1FN".into(),
        ))),
    );
    operation_.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![4, 2, 4, 1, 1, 0, 0]),
    );
    operation_.properties.insert(
        "sf_type".into(),
        MlirAttribute::Type(MlirType::Float(pliron_mlir_export::MlirFloatType::Other(
            "f8E8M0FNU".into(),
        ))),
    );
    operation_.properties.insert(
        "shape_MNK".into(),
        MlirAttribute::dialect("#cute.shape<\"(16,8,64)\">")?,
    );
    operation_.properties.insert(
        "thread_id_a".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(16),
        },
    );
    operation_.properties.insert(
        "thread_id_b".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(16),
        },
    );
    operation_.properties.insert(
        "vec_size".into(),
        MlirAttribute::Integer {
            value: 32,
            ty: MlirType::Integer(32),
        },
    );
    operations.push(operation_);
    uses.try_into()
        .map_err(|_| "SM120 MXF4 MMA did not produce four accumulators".to_owned())
}

struct EpilogueStoreTranslation;

impl OperationTranslation for EpilogueStoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CuteEpilogueStoreFragmentOp::wrap(source)
            .verify(ctx)
            .map_err(stringify)?;
        input.attributes.remove("epilogue_store_fragment_tile");
        expect_shape("cute.epilogue_store_fragment", &input, 0, 4)?;
        let accumulator_layout = CarrierLayout::from_source(
            ctx,
            source.deref(ctx).get_operand(3).get_type(ctx),
            session,
            0,
        )?;
        let mut operations = vec![];
        let mut accumulator = vec![];
        extract_carrier(
            session,
            &mut operations,
            input.operands[3].clone(),
            &accumulator_layout,
            &mut accumulator,
            &input.location,
        )?;
        if accumulator.len() != 64 {
            return Err("cute.epilogue_store_fragment needs 64 f32 leaves".into());
        }

        let three = emit_i64_constant(session, &mut operations, 3, &input.location)?;
        let warp_m = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            input.operands[1].clone(),
            three,
            &input.location,
        )?;
        let two = emit_i64_constant(session, &mut operations, 2, &input.location)?;
        let warp_n = emit_binary(
            session,
            &mut operations,
            "arith.shrui",
            input.operands[1].clone(),
            two,
            &input.location,
        )?;
        let lane = cast_int(
            session,
            &mut operations,
            input.operands[2].clone(),
            MlirType::Integer(64),
            "arith.extui",
            &input.location,
        )?;
        let fifteen = emit_i64_constant(session, &mut operations, 15, &input.location)?;
        let lane15 = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane.clone(),
            fifteen,
            &input.location,
        )?;
        let warp_row_stride = emit_i64_constant(session, &mut operations, 1024, &input.location)?;
        let warp_rows = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            warp_m,
            warp_row_stride,
            &input.location,
        )?;
        let lane_row_stride = emit_i64_constant(session, &mut operations, 64, &input.location)?;
        let lane_rows = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            lane15,
            lane_row_stride,
            &input.location,
        )?;
        let row_zero = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            warp_rows,
            lane_rows,
            &input.location,
        )?;
        let band_stride = emit_i64_constant(session, &mut operations, 4096, &input.location)?;
        let row_one = emit_binary(
            session,
            &mut operations,
            "arith.addi",
            row_zero.clone(),
            band_stride,
            &input.location,
        )?;
        let columns_per_warp = emit_i64_constant(session, &mut operations, 16, &input.location)?;
        let column_base = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            warp_n,
            columns_per_warp,
            &input.location,
        )?;
        let seven = emit_i64_constant(session, &mut operations, 7, &input.location)?;
        let lane7 = emit_binary(
            session,
            &mut operations,
            "arith.andi",
            lane,
            seven,
            &input.location,
        )?;
        let eight = emit_i64_constant(session, &mut operations, 8, &input.location)?;
        let swizzle = emit_binary(
            session,
            &mut operations,
            "arith.muli",
            lane7,
            eight,
            &input.location,
        )?;
        let half_stride = emit_i64_constant(session, &mut operations, 8192, &input.location)?;

        const N_OFFSETS: [u64; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
        for m_band in 0..2_usize {
            for (n_slot, n_offset) in N_OFFSETS.into_iter().enumerate() {
                let local =
                    emit_i64_constant(session, &mut operations, n_offset % 64, &input.location)?;
                let column = emit_binary(
                    session,
                    &mut operations,
                    "arith.addi",
                    column_base.clone(),
                    local,
                    &input.location,
                )?;
                let logical = emit_binary(
                    session,
                    &mut operations,
                    "arith.addi",
                    if m_band == 0 {
                        row_zero.clone()
                    } else {
                        row_one.clone()
                    },
                    column,
                    &input.location,
                )?;
                let mut physical = emit_binary(
                    session,
                    &mut operations,
                    "arith.xori",
                    logical,
                    swizzle.clone(),
                    &input.location,
                )?;
                if n_offset >= 64 {
                    physical = emit_binary(
                        session,
                        &mut operations,
                        "arith.addi",
                        physical,
                        half_stride.clone(),
                        &input.location,
                    )?;
                }
                let pointer = gep_value(
                    session,
                    &mut operations,
                    input.operands[0].clone(),
                    physical,
                    // `physical` is an f16 element offset.  The source path
                    // performs `base.add(physical)` before casting to u8 for
                    // stmatrix, preserving each 16-byte row address.
                    epilogue_gep_element_type(),
                    &input.location,
                )?;
                let cell = (m_band * 8 + n_slot) * 4;
                let top = emit_f16x2_pack(
                    session,
                    &mut operations,
                    accumulator[cell].clone(),
                    accumulator[cell + 1].clone(),
                    &input.location,
                )?;
                let bottom = emit_f16x2_pack(
                    session,
                    &mut operations,
                    accumulator[cell + 2].clone(),
                    accumulator[cell + 3].clone(),
                    &input.location,
                )?;
                let mut store = operation(
                    "nvvm.stmatrix",
                    vec![],
                    vec![pointer, top, bottom],
                    &input.location,
                )?;
                store.properties.insert(
                    "layout".into(),
                    MlirAttribute::dialect("#nvvm.mma_layout<row>")?,
                );
                operations.push(store);
            }
        }
        Ok(operations)
    }
}

fn emit_f16x2_pack(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    low: MlirValueUse,
    high: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let vector_type = MlirType::Vector {
        shape: vec![2],
        element: Box::new(MlirType::Float(pliron_mlir_export::MlirFloatType::F16)),
    };
    let (call_result, converted) = fresh(session, vector_type);
    let mut call = operation(
        "llvm.call_intrinsic",
        vec![call_result],
        // `ff2f16x2` follows PTX source order: its first f32 becomes the
        // high half.  CuTe's `cvt_f16x2_f32(lo, hi)` contract is low-first,
        // so preserve the shared low-first semantic contract by reversing the
        // intrinsic operands.
        f16x2_intrinsic_operands(low, high).into(),
        location,
    )?;
    call.properties.insert(
        "intrin".into(),
        MlirAttribute::String("llvm.nvvm.ff2f16x2.rn".into()),
    );
    call.properties.insert(
        "op_bundle_sizes".into(),
        MlirAttribute::DenseI32Array(vec![]),
    );
    call.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![2, 0]),
    );
    operations.push(call);
    let (bits_result, bits) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "llvm.bitcast",
        vec![bits_result],
        vec![converted],
        location,
    )?);
    Ok(bits)
}

fn f16x2_intrinsic_operands(low: MlirValueUse, high: MlirValueUse) -> [MlirValueUse; 2] {
    [high, low]
}

fn cast_int(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    target: MlirType,
    name: &str,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, target);
    operations.push(operation(name, vec![result], vec![value], location)?);
    Ok(use_)
}

fn load_scale_word(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    row: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let mask31 = emit_i64_constant(session, operations, 31, location)?;
    let low = emit_binary(
        session,
        operations,
        "arith.andi",
        row.clone(),
        mask31,
        location,
    )?;
    let four = emit_i64_constant(session, operations, 4, location)?;
    let low = emit_binary(session, operations, "arith.muli", low, four, location)?;
    let five = emit_i64_constant(session, operations, 5, location)?;
    let quadrant = emit_binary(session, operations, "arith.shrui", row, five, location)?;
    let three = emit_i64_constant(session, operations, 3, location)?;
    let quadrant = emit_binary(session, operations, "arith.andi", quadrant, three, location)?;
    let offset = emit_binary(session, operations, "arith.addi", low, quadrant, location)?;
    let pointer = gep_value(
        session,
        operations,
        base,
        offset,
        MlirType::Integer(32),
        location,
    )?;
    let (result, use_) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "llvm.load",
        vec![result],
        vec![pointer],
        location,
    )?);
    Ok(use_)
}

fn load_pipeline_from_operand(
    ctx: &Context,
    source: Ptr<Operation>,
    index: usize,
) -> Result<CuteTmaLoadPipelineType, String> {
    let ty = source.deref(ctx).get_operand(index).get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<CuteTmaLoadPipelineType>()
        .cloned()
        .ok_or_else(|| "expected a cute.tma_load_pipeline operand".to_owned())
}

fn tma_tile_extents(ctx: &Context, ty: TypeHandle) -> Result<(u64, u64), String> {
    let ty_ref = ty.deref(ctx);
    let view = ty_ref
        .downcast_ref::<CuteTmaViewType>()
        .ok_or_else(|| "expected a cute.tma_view operand".to_owned())?;
    let modes = view.smem_layout.0.inner().modes();
    if modes.len() != 2 {
        return Err(format!("TMA mapping needs two modes, got {}", modes.len()));
    }
    let rows = modes[0]
        .checked_size()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "TMA row extent is invalid".to_owned())?;
    let cols = modes[1]
        .checked_size()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "TMA column extent is invalid".to_owned())?;
    Ok((rows, cols))
}

fn tma_coordinate(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    tile: MlirValueUse,
    extent: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let extent = emit_i64_constant(session, operations, extent, location)?;
    let elements = emit_binary(session, operations, "arith.muli", tile, extent, location)?;
    let (result, use_) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "arith.trunci",
        vec![result],
        vec![elements],
        location,
    )?);
    Ok(use_)
}

fn barrier_pointer_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    slot: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let slot = emit_i64_constant(session, operations, slot, location)?;
    gep_value(
        session,
        operations,
        base,
        slot,
        MlirType::Integer(64),
        location,
    )
}

fn barrier_pointer_dynamic(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    slot: MlirValueUse,
    add_stages: Option<u64>,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (wide_result, mut index) = fresh(session, MlirType::Integer(64));
    operations.push(operation(
        "arith.extui",
        vec![wide_result],
        vec![slot],
        location,
    )?);
    if let Some(stages) = add_stages {
        let stages = emit_i64_constant(session, operations, stages, location)?;
        index = emit_binary(session, operations, "arith.addi", index, stages, location)?;
    }
    gep_value(
        session,
        operations,
        base,
        index,
        MlirType::Integer(64),
        location,
    )
}

fn emit_pipeline_state_advance(
    session: &mut TranslationSession<'_>,
    slot: MlirValueUse,
    phase: MlirValueUse,
    stages: u64,
    final_results: Option<[MlirResult; 2]>,
    location: &MlirLocation,
) -> Result<(Vec<MlirOperation>, [MlirValueUse; 2]), String> {
    let mut operations = vec![];
    let one = emit_i32_constant(session, &mut operations, 1, location)?;
    let incremented = emit_binary(session, &mut operations, "arith.addi", slot, one, location)?;
    let width = emit_i32_constant(session, &mut operations, i128::from(stages), location)?;
    let (wrap_result, wraps) = fresh(session, MlirType::Integer(1));
    let mut compare = operation(
        "arith.cmpi",
        vec![wrap_result],
        vec![incremented.clone(), width],
        location,
    )?;
    compare.properties.insert(
        "predicate".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(64),
        },
    );
    operations.push(compare);
    let zero = emit_i32_constant(session, &mut operations, 0, location)?;

    let (slot_result, next_slot) = if let Some(results) = &final_results {
        (
            results[0].clone(),
            MlirValueUse {
                id: results[0].id,
                ty: results[0].ty.clone(),
            },
        )
    } else {
        fresh(session, MlirType::Integer(32))
    };
    operations.push(operation(
        "arith.select",
        vec![slot_result],
        vec![wraps.clone(), zero, incremented],
        location,
    )?);

    let (wrap_u32_result, wrap_u32) = fresh(session, MlirType::Integer(32));
    operations.push(operation(
        "arith.extui",
        vec![wrap_u32_result],
        vec![wraps],
        location,
    )?);
    let (phase_result, next_phase) = if let Some(results) = final_results {
        (
            results[1].clone(),
            MlirValueUse {
                id: results[1].id,
                ty: results[1].ty.clone(),
            },
        )
    } else {
        fresh(session, MlirType::Integer(32))
    };
    operations.push(operation(
        "arith.xori",
        vec![phase_result],
        vec![phase, wrap_u32],
        location,
    )?);
    Ok((operations, [next_slot, next_phase]))
}

fn gep_value(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    base: MlirValueUse,
    index: MlirValueUse,
    element: MlirType,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, base.ty.clone());
    let mut gep = operation(
        "llvm.getelementptr",
        vec![result],
        vec![base, index],
        location,
    )?;
    gep_properties(&mut gep, element, vec![i32::MIN]);
    operations.push(gep);
    Ok(use_)
}

fn gep_properties(operation: &mut MlirOperation, element: MlirType, indices: Vec<i32>) {
    operation
        .properties
        .insert("elem_type".into(), MlirAttribute::Type(element));
    operation.properties.insert(
        "noWrapFlags".into(),
        MlirAttribute::Integer {
            value: 0,
            ty: MlirType::Integer(32),
        },
    );
    operation.properties.insert(
        "rawConstantIndices".into(),
        MlirAttribute::DenseI32Array(indices),
    );
}

fn build_aggregate(
    session: &mut TranslationSession<'_>,
    final_result: MlirResult,
    fields: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    if fields.is_empty() {
        return Ok(vec![operation(
            "llvm.mlir.poison",
            vec![final_result],
            vec![],
            location,
        )?]);
    }
    let (poison_result, mut aggregate) = fresh(session, final_result.ty.clone());
    let mut operations = vec![operation(
        "llvm.mlir.poison",
        vec![poison_result],
        vec![],
        location,
    )?];
    let field_count = fields.len();
    for (position, field) in fields.into_iter().enumerate() {
        let result = if position + 1 == field_count {
            final_result.clone()
        } else {
            fresh(session, final_result.ty.clone()).0
        };
        let next = MlirValueUse {
            id: result.id,
            ty: result.ty.clone(),
        };
        let mut insert = operation(
            "llvm.insertvalue",
            vec![result],
            vec![aggregate, field],
            location,
        )?;
        insert.properties.insert(
            "position".into(),
            MlirAttribute::DenseI64Array(vec![position as i64]),
        );
        operations.push(insert);
        aggregate = next;
    }
    Ok(operations)
}

fn emit_i64_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: u64,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (operation_, use_) =
        constant_value(session, MlirType::Integer(64), u64_bits(value), location)?;
    operations.push(operation_);
    Ok(use_)
}

fn emit_i32_constant(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (operation_, use_) = constant_value(session, MlirType::Integer(32), value, location)?;
    operations.push(operation_);
    Ok(use_)
}

fn constant_value(
    session: &mut TranslationSession<'_>,
    ty: MlirType,
    value: i128,
    location: &MlirLocation,
) -> Result<(MlirOperation, MlirValueUse), String> {
    let (result, use_) = fresh(session, ty);
    Ok((integer_constant(result, value, location)?, use_))
}

fn emit_binary(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    name: &str,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let (result, use_) = fresh(session, left.ty.clone());
    operations.push(binary(name, result, left, right, location)?);
    Ok(use_)
}

fn binary(
    name: &str,
    result: MlirResult,
    left: MlirValueUse,
    right: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(name, vec![result], vec![left, right], location)
}

fn integer_constant(
    result: MlirResult,
    value: i128,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = operation("arith.constant", vec![result.clone()], vec![], location)?;
    operation.properties.insert(
        "value".into(),
        MlirAttribute::Integer {
            value,
            ty: result.ty,
        },
    );
    Ok(operation)
}

fn expect_shape(
    name: &str,
    input: &OperationInput,
    results: usize,
    operands: usize,
) -> Result<(), String> {
    if input.results.len() != results || input.operands.len() != operands {
        return Err(format!(
            "{name} expected {results} results and {operands} operands, got {} and {}",
            input.results.len(),
            input.operands.len()
        ));
    }
    if !input.successors.is_empty() || !input.regions.is_empty() || !input.attributes.is_empty() {
        return Err(format!("{name} retained source-only target structure"));
    }
    Ok(())
}

fn operation(
    name: &str,
    results: Vec<MlirResult>,
    operands: Vec<MlirValueUse>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.results = results;
    operation.operands = operands;
    operation.location = location.clone();
    Ok(operation)
}

fn mbarrier_init(
    barrier: MlirValueUse,
    arrival_count: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(
        "nvvm.mbarrier.init",
        vec![],
        vec![barrier, arrival_count],
        location,
    )
}

fn counted_cta_sync(
    barrier_id: MlirValueUse,
    participant_threads: MlirValueUse,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    operation(
        COUNTED_CTA_SYNC_OP,
        vec![],
        vec![barrier_id, participant_threads],
        location,
    )
}

fn async_shared_proxy_fence(location: &MlirLocation) -> Result<MlirOperation, String> {
    let mut fence = operation("nvvm.fence.proxy", vec![], vec![], location)?;
    fence.attributes.insert(
        "kind".into(),
        MlirAttribute::dialect("#nvvm.proxy_kind<async.shared>")?,
    );
    fence.attributes.insert(
        "space".into(),
        MlirAttribute::dialect("#nvvm.shared_space<cta>")?,
    );
    Ok(fence)
}

fn epilogue_sync_operations(
    phase: CuteEpilogueSyncPhaseAttr,
    barrier_id: MlirValueUse,
    participant_threads: MlirValueUse,
    location: &MlirLocation,
) -> Result<Vec<MlirOperation>, String> {
    let mut operations = Vec::with_capacity(match phase {
        CuteEpilogueSyncPhaseAttr::Reusable => 1,
        CuteEpilogueSyncPhaseAttr::ReadyForTma => 2,
    });
    if phase == CuteEpilogueSyncPhaseAttr::ReadyForTma {
        operations.push(async_shared_proxy_fence(location)?);
    }
    operations.push(counted_cta_sync(barrier_id, participant_threads, location)?);
    Ok(operations)
}

fn epilogue_gep_element_type() -> MlirType {
    MlirType::Float(pliron_mlir_export::MlirFloatType::F16)
}

#[cfg(test)]
const fn ldmatrix_swizzled_byte_offset(plain: u64) -> u64 {
    plain ^ ((plain & LDMATRIX_BYTE_SWIZZLE_MASK) >> LDMATRIX_BYTE_SWIZZLE_SHIFT)
}

fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}

fn generic_pointer_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.ptr")
}

fn shared_pointer_type() -> Result<MlirType, String> {
    MlirType::dialect("!llvm.ptr<3>")
}

fn u64_bits(value: u64) -> i128 {
    i64::from_ne_bytes(value.to_ne_bytes()) as i128
}

fn stringify(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron_mlir_export::{MlirFloatType, MlirValueId};

    fn value(id: u64, ty: MlirType) -> MlirValueUse {
        MlirValueUse {
            id: MlirValueId(id),
            ty,
        }
    }

    #[test]
    fn byte_domain_ldmatrix_swizzle_preserves_sixteen_byte_segments() {
        for warp_row in 0..8_u64 {
            for warp_column in 0..2_u64 {
                for lane in 0..32_u64 {
                    let submatrix = lane / 8;
                    let row = warp_row * 16 + (submatrix % 2) * 8 + lane % 8;
                    let column = warp_column * 16 + (submatrix / 2) * 8;
                    let plain = row * 64 + column * 2;
                    assert_eq!(
                        ldmatrix_swizzled_byte_offset(plain) % 16,
                        0,
                        "warp_row={warp_row} warp_column={warp_column} lane={lane}"
                    );
                }
            }
        }

        // The old element-domain mask toggled bit 3 for lane 1 after the
        // layout was converted to bytes, reproducing the observed 0x6448
        // misaligned shared address.
        let lane_one_plain = 64_u64;
        let old = lane_one_plain ^ ((lane_one_plain & 0xc0) >> 3);
        assert_eq!(old, 72);
        assert_eq!(old % 16, 8);
        assert_eq!(ldmatrix_swizzled_byte_offset(lane_one_plain), 64);
    }

    #[test]
    fn packed_f16_intrinsic_reverses_low_first_source_operands() {
        let low = value(7, MlirType::Float(MlirFloatType::F32));
        let high = value(8, MlirType::Float(MlirFloatType::F32));
        let [first, second] = f16x2_intrinsic_operands(low.clone(), high.clone());
        assert_eq!(first, high);
        assert_eq!(second, low);
    }

    #[test]
    fn epilogue_ready_publishes_then_syncs_while_reusable_only_syncs() {
        let barrier_id = value(1, MlirType::Integer(32));
        let participants = value(2, MlirType::Integer(32));
        let ready = epilogue_sync_operations(
            CuteEpilogueSyncPhaseAttr::ReadyForTma,
            barrier_id.clone(),
            participants.clone(),
            &MlirLocation::Unknown,
        )
        .unwrap();
        assert_eq!(
            ready
                .iter()
                .map(|operation| operation.name.as_str())
                .collect::<Vec<_>>(),
            ["nvvm.fence.proxy", "nvvm.barrier.cta.sync"]
        );
        assert_eq!(
            ready[0].attributes.get("kind"),
            Some(&MlirAttribute::dialect("#nvvm.proxy_kind<async.shared>").unwrap())
        );
        assert_eq!(
            ready[0].attributes.get("space"),
            Some(&MlirAttribute::dialect("#nvvm.shared_space<cta>").unwrap())
        );
        assert_eq!(
            ready[1].operands,
            vec![barrier_id.clone(), participants.clone()]
        );

        let reusable = epilogue_sync_operations(
            CuteEpilogueSyncPhaseAttr::Reusable,
            barrier_id,
            participants,
            &MlirLocation::Unknown,
        )
        .unwrap();
        assert_eq!(reusable.len(), 1);
        assert_eq!(reusable[0].name, "nvvm.barrier.cta.sync");
        assert_eq!(
            epilogue_gep_element_type(),
            MlirType::Float(MlirFloatType::F16)
        );
    }

    #[test]
    fn pipeline_initialization_is_unpredicated_inside_the_source_branch() {
        let barrier = value(3, MlirType::dialect("!llvm.ptr<3>").unwrap());
        let count = value(4, MlirType::Integer(32));
        let init = mbarrier_init(barrier.clone(), count.clone(), &MlirLocation::Unknown).unwrap();
        assert_eq!(init.name, "nvvm.mbarrier.init");
        assert_eq!(init.operands, vec![barrier, count]);
    }
}
