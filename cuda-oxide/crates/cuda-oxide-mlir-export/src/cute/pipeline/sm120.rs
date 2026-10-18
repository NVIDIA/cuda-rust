/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Load-pipeline initialization, ring state, waits, and barrier operations.
//!
//! Shared storage holds `stages` full barriers, then `stages` empty barriers.
//! Consumers wait on full slots; producers acquire empty slots. Both track
//! `(slot, phase)`: consumers start at `(0, 0)`, producers at `(0, 1)` so the
//! initially empty slots are available. Advancing wraps the slot and flips the
//! phase only on wrap. For two stages, the state calculation is:
//!
//! ```
//! let advance = |(slot, phase): (u32, u32)| {
//!     let next = (slot + 1) % 2;
//!     (next, phase ^ u32::from(next == 0))
//! };
//! let producer = advance((0, 1));
//! assert_eq!(producer, (1, 1));
//! assert_eq!(advance(producer), (0, 0));
//! ```

use crate::cute::common::{
    emit_binary, emit_i32_constant, emit_i64_constant, expect_shape, fresh, gep_value,
    integer_constant, operation, shared_pointer_type, stringify,
};
use crate::{
    MlirAttribute, MlirBlock, MlirLocation, MlirOperation, MlirRegion, MlirResult, MlirType,
    MlirValueUse, OperationInput, OperationTranslation, TranslationRegistry, TranslationSession,
    TypeTranslation,
};
use dialect_cute::{
    attributes::CutePipelineRoleAttr,
    pipeline_ops::{
        CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
        CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
        CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
    },
    types::CuteTmaLoadPipelineType,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Bound for each hardware wait probe; the high-level NVVM wait retries to completion.
const PIPELINE_WAIT_TICKS: i128 = 10_000_000;
/// Erase a verified load-pipeline handle to its CTA-shared barrier pointer.
pub(in crate::cute) struct LoadPipelineTypeTranslation;

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

/// Initialize full/empty barrier rings on the chosen thread, then publish and synchronize.
pub(in crate::cute) struct PipelineInitTranslation;

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

/// Create slot zero with producer or consumer initial parity from the source role.
pub(in crate::cute) struct PipelineStateNewTranslation;

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

/// Forward the runtime slot while erasing the source pipeline-state contract.
pub(in crate::cute) struct PipelineStateSlotTranslation;

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

/// Advance the slot and toggle its phase only when the ring wraps.
pub(in crate::cute) struct PipelineStateAdvanceTranslation;

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

/// Wait for completion in the selected full or empty barrier ring.
pub(in crate::cute) struct PipelineWaitTranslation {
    /// Select the second barrier ring for producer acquisition.
    pub(in crate::cute) empty_ring: bool,
    /// Source wait operation used to select and validate its semantic contract.
    pub(in crate::cute) name: &'static str,
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

/// Announce the verified transaction-byte count to the full-stage barrier.
pub(in crate::cute) struct PipelineExpectTxTranslation;

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

/// Have the selected consumer lane arrive on the corresponding empty-stage barrier.
pub(in crate::cute) struct PipelineReleaseTranslation;

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

/// Drain every remaining stage using its matching slot and parity.
pub(in crate::cute) struct PipelineTailTranslation;

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

/// Recover the verified pipeline type from a source operand before type erasure.
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

/// Form a shared barrier pointer from a compile-time slot counted in i64 elements.
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

/// Form a shared barrier pointer and optionally offset into the empty ring.
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

/// Emit slot increment, wrap-to-zero, and parity toggle, optionally reusing caller result IDs.
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

/// Build an unpredicated barrier initialization with the given arrival count.
pub(in crate::cute) fn mbarrier_init(
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
