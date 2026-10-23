/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use dialect_mir::ops::MirUndefOp;

/// Stage-ring size used by the pipeline regression fixture.
const STAGES: u64 = 3;
/// Number of consuming warps in the pipeline regression fixture.
const CONSUMER_WARPS: u32 = 8;
/// Expected bytes per completed stage in the pipeline regression fixture.
const TRANSACTION_BYTES: u32 = 17_408;

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Construct a mutable shared-pointer type with the fixture's chosen pointee.
fn shared_pointer(ctx: &mut Context, pointee: TypeHandle, mutable: bool) -> Value {
    let pointer: TypeHandle = MirPtrType::get(ctx, pointee, mutable, address_space::SHARED).into();
    undef(ctx, pointer)
}

/// Context and scalar/shared-pointer carriers for the load-pipeline verifier tests.
struct Setup {
    /// IR context owning all handles used by this fixture.
    ctx: Context,
    /// Shared storage pointer used by this fixture.
    base: Value,
    /// Unsigned 32-bit cursor/phase fixture value.
    u32_value: Value,
    /// Unsigned 64-bit extent/index fixture value.
    u64_value: Value,
}

/// Create the barrier-ring pointer and correctly typed u32/u64 cursor carriers.
fn setup() -> Setup {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    let barrier = u64_type(&ctx);
    let base = shared_pointer(&mut ctx, barrier, true);
    let u32_ty = u32_type(&ctx);
    let u64_ty = u64_type(&ctx);
    let u32_value = undef(&mut ctx, u32_ty);
    let u64_value = undef(&mut ctx, u64_ty);
    Setup {
        ctx,
        base,
        u32_value,
        u64_value,
    }
}

/// Creation, initialization, cursor updates, and producer/consumer effects retain their separate operation boundaries.
#[test]
fn load_pipeline_stays_a_composable_ring_protocol() {
    let Setup {
        mut ctx,
        base,
        u32_value,
        ..
    } = setup();
    let make = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        base,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    let pipeline = make.pipeline(&ctx);

    let init = CuteTmaLoadPipelineInitOp::new(&mut ctx, pipeline, u32_value);

    let producer_state = CutePipelineStateAttr::producer(STAGES);
    let producer = CutePipelineStateNewOp::new(&mut ctx, producer_state);
    let producer_slot = producer.slot(&ctx);
    let producer_phase = producer.phase(&ctx);
    let producer_index = CutePipelineStateSlotOp::new(&mut ctx, producer_slot, producer_state);
    let acquire = CutePipelineProducerAcquireOp::new(
        &mut ctx,
        pipeline,
        producer_slot,
        producer_phase,
        producer_state,
    );
    let expect =
        CutePipelineProducerExpectTxOp::new(&mut ctx, pipeline, producer_slot, producer_state);
    let producer_advance =
        CutePipelineStateAdvanceOp::new(&mut ctx, producer_slot, producer_phase, producer_state);
    let producer_next_slot = producer_advance.next_slot(&ctx);
    let producer_next_phase = producer_advance.next_phase(&ctx);
    let tail = CutePipelineProducerTailOp::new(
        &mut ctx,
        pipeline,
        producer_next_slot,
        producer_next_phase,
        producer_state,
    );

    let consumer_state = CutePipelineStateAttr::consumer(STAGES);
    let consumer = CutePipelineStateNewOp::new(&mut ctx, consumer_state);
    let consumer_slot = consumer.slot(&ctx);
    let consumer_phase = consumer.phase(&ctx);
    let consumer_index = CutePipelineStateSlotOp::new(&mut ctx, consumer_slot, consumer_state);
    let wait = CutePipelineConsumerWaitOp::new(
        &mut ctx,
        pipeline,
        consumer_slot,
        consumer_phase,
        consumer_state,
    );
    let release =
        CutePipelineConsumerReleaseOp::new(&mut ctx, pipeline, consumer_slot, consumer_state);
    let consumer_advance =
        CutePipelineStateAdvanceOp::new(&mut ctx, consumer_slot, consumer_phase, consumer_state);

    assert!(make.verify(&ctx).is_ok());
    assert!(init.verify(&ctx).is_ok());
    assert!(producer.verify(&ctx).is_ok());
    assert!(producer_index.verify(&ctx).is_ok());
    assert!(acquire.verify(&ctx).is_ok());
    assert!(expect.verify(&ctx).is_ok());
    assert!(producer_advance.verify(&ctx).is_ok());
    assert!(tail.verify(&ctx).is_ok());
    assert!(consumer.verify(&ctx).is_ok());
    assert!(consumer_index.verify(&ctx).is_ok());
    assert!(wait.verify(&ctx).is_ok());
    assert!(release.verify(&ctx).is_ok());
    assert!(consumer_advance.verify(&ctx).is_ok());
    assert_eq!(make.pipeline_type(&ctx).unwrap().storage_bytes(), Some(48));
    assert_eq!(
        expect.completion_barrier(&ctx).get_type(&ctx),
        base.get_type(&ctx)
    );
}

/// Zero/overflowing stages and unsupported transaction or participation counts fail configuration checks.
#[test]
fn pipeline_config_rejects_impossible_rings() {
    let ctx = Context::new();
    assert!(
        CuteTmaLoadPipelineType {
            stages: 0,
            consumer_warps: 8,
            transaction_bytes: 1,
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        CuteTmaLoadPipelineType {
            stages: 3,
            consumer_warps: 33,
            transaction_bytes: 1,
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        CuteTmaLoadPipelineType {
            stages: 3,
            consumer_warps: 8,
            transaction_bytes: 0,
        }
        .verify(&ctx)
        .is_err()
    );
    let overflow = CuteTmaLoadPipelineType {
        stages: u64::MAX,
        consumer_warps: 8,
        transaction_bytes: 1,
    };
    assert_eq!(overflow.storage_bytes(), None);
    assert!(overflow.verify(&ctx).is_err());
    assert!(CutePipelineStateAttr::producer(0).verify(&ctx).is_err());
}

/// Pipeline creation requires matching shared barrier storage, capacity, and alignment.
#[test]
fn make_rejects_wrong_storage_and_weak_promises() {
    let Setup {
        mut ctx,
        base,
        u32_value,
        ..
    } = setup();
    let weak = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        base,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        4,
    );
    assert!(
        weak.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("at least 8")
    );

    let wrong_pointee_ty = u32_value.get_type(&ctx);
    let wrong_pointee = shared_pointer(&mut ctx, wrong_pointee_ty, true);
    let wrong = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        wrong_pointee,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    assert!(
        wrong
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("unsigned u64 Barrier")
    );

    let barrier = u64_type(&ctx);
    let immutable = shared_pointer(&mut ctx, barrier, false);
    let immutable = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        immutable,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    assert!(immutable.verify(&ctx).is_err());

    let barrier = u64_type(&ctx);
    let generic_pointer_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, barrier, true).into();
    let generic_pointer = undef(&mut ctx, generic_pointer_ty);
    let generic = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        generic_pointer,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    assert!(generic.verify(&ctx).is_err());

    let signed_u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
    let signed_storage = shared_pointer(&mut ctx, signed_u64, true);
    let signed_storage = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        signed_storage,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    assert!(signed_storage.verify(&ctx).is_err());
}

/// Lifecycle effects cannot mix producer/consumer roles, ring sizes, or cursor scalar types.
#[test]
fn effects_reject_role_stage_and_scalar_drift() {
    let Setup {
        mut ctx,
        base,
        u32_value,
        u64_value,
    } = setup();
    let make = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        base,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    let pipeline = make.pipeline(&ctx);

    let wrong_role = CutePipelineProducerAcquireOp::new(
        &mut ctx,
        pipeline,
        u32_value,
        u32_value,
        CutePipelineStateAttr::consumer(STAGES),
    );
    assert!(
        wrong_role
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("Producer")
    );

    let wrong_stages = CutePipelineConsumerWaitOp::new(
        &mut ctx,
        pipeline,
        u32_value,
        u32_value,
        CutePipelineStateAttr::consumer(2),
    );
    assert!(
        wrong_stages
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("3")
    );

    let wrong_scalar = CutePipelineConsumerReleaseOp::new(
        &mut ctx,
        pipeline,
        u64_value,
        CutePipelineStateAttr::consumer(STAGES),
    );
    assert!(
        wrong_scalar
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("32-bit")
    );
}

/// The expectation result must remain the shared u64 barrier pointer consumed by TMA copies.
#[test]
fn expect_tx_rejects_barrier_pointer_type_drift() {
    let Setup {
        mut ctx,
        base,
        u32_value,
        ..
    } = setup();
    let make = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        base,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    let pipeline = make.pipeline(&ctx);
    let base_pointer_ty = base.get_type(&ctx);
    let wrong_result: TypeHandle = MirPtrType::get_generic(&mut ctx, base_pointer_ty, true).into();
    let raw = Operation::new(
        &mut ctx,
        CutePipelineProducerExpectTxOp::get_concrete_op_info(),
        vec![wrong_result],
        vec![pipeline, u32_value],
        vec![],
        0,
    );
    let expect = CutePipelineProducerExpectTxOp::wrap(raw);
    expect.set_attr_pipeline_producer_expect_state(&ctx, CutePipelineStateAttr::producer(STAGES));

    assert!(
        expect
            .verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("must match")
    );
}

/// The compiler-only handle cannot escape through an ordinary non-lifecycle operation.
#[test]
fn ghost_pipeline_rejects_an_unrelated_consumer() {
    let Setup { mut ctx, base, .. } = setup();
    let make = CuteTmaLoadPipelineMakeOp::new(
        &mut ctx,
        base,
        STAGES,
        CONSUMER_WARPS,
        TRANSACTION_BYTES,
        8,
    );
    let pipeline = make.pipeline(&ctx);
    let _unrelated =
        CutePipelineStateSlotOp::new(&mut ctx, pipeline, CutePipelineStateAttr::producer(STAGES));

    assert!(
        make.verify(&ctx)
            .unwrap_err()
            .to_string()
            .contains("lifecycle operations")
    );
}
