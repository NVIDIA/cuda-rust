/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 operation definitions, local contracts, and registration.

use super::*;

/// Runtime scalar or pointer contract required at one positional operand.
#[derive(Clone, Copy)]
enum Carrier {
    /// Unsigned 32-bit index or encoded tensor-memory address.
    U32,
    /// One-bit predicate selecting whether MMA retains the previous accumulator.
    Bool,
    /// FP32 scalar storage or accumulation.
    F32,
    /// Writable shared pointer to a 32-bit allocation token.
    SharedU32,
    /// Writable shared pointer to a 64-bit hardware barrier.
    SharedBarrier,
    /// Writable shared FP16 tile pointer.
    SharedF16,
    /// Readable shared FP16 tile pointer.
    ReadSharedF16,
    /// Immutable tensor-map pointer in generic, global, or constant space.
    Descriptor,
}
/// Check an operand's scalar width or pointer storage/address-space/mutability contract.
fn carrier_matches(ctx: &Context, value: Value, carrier: Carrier) -> bool {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let integer = |width| {
        ty.downcast_ref::<IntegerType>()
            .is_some_and(|i| i.width() == width && (width == 1 || i.is_unsigned()))
    };
    match carrier {
        Carrier::U32 => integer(32),
        Carrier::Bool => integer(1),
        Carrier::F32 => ty.downcast_ref::<FP32Type>().is_some(),
        _ => {
            let Some(ptr) = ty.downcast_ref::<MirPtrType>() else {
                return false;
            };
            if matches!(carrier, Carrier::Descriptor) {
                return !ptr.is_mutable
                    && matches!(
                        ptr.address_space,
                        address_space::GENERIC | address_space::GLOBAL | address_space::CONSTANT
                    );
            }
            if !matches!(
                ptr.address_space,
                address_space::SHARED | address_space::GENERIC
            ) {
                return false;
            }
            if !matches!(carrier, Carrier::ReadSharedF16) && !ptr.is_mutable {
                return false;
            }
            let pointee = ptr.pointee.deref(ctx);
            match carrier {
                Carrier::SharedU32 => pointee
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|i| i.width() == 32 && i.is_unsigned()),
                Carrier::SharedBarrier => pointee
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|i| i.width() == 64 && i.is_unsigned()),
                Carrier::SharedF16 | Carrier::ReadSharedF16 => {
                    pointee.downcast_ref::<MirFP16Type>().is_some()
                }
                _ => false,
            }
        }
    }
}

/// Generate zero-result operations with a typed plan and local carrier verification.
macro_rules! semantic_op {
    ($(#[$doc:meta])* $ty:ident, $name:literal, $key:ident, $plan:ty, [$($carrier:ident),*], $check:expr) => {
        #[pliron_op(name = $name, format, attributes = ($key: $plan))]
        pub struct $ty;
        // Keep contracts on the impl and constructor: pliron's generated
        // attribute-table links do not compose correctly with local struct docs.
        $(#[$doc])*
        impl $ty {
            $(#[$doc])*
            pub fn new(ctx: &mut Context, operands: Vec<Value>, plan: $plan) -> Self {
                let op = Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], operands, vec![], 0) };
                op.get_operation().deref_mut(ctx).attributes.set(stringify!($key).try_into().expect("plan key"), plan);
                op
            }
            /// View an existing operation without allocating or verifying it.
            pub fn wrap(op: Ptr<Operation>) -> Self { Self { op } }
            /// Borrow the typed plan while retaining the operation's context borrow.
            pub fn get_attr_plan<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, $plan>> {
                std::cell::Ref::filter_map(self.get_operation().deref(ctx), |op|
                    op.attributes.get::<$plan>(&stringify!($key).try_into().expect("plan key"))).ok()
            }
            /// Clone the typed plan, or return `None` when the attribute is missing.
            pub fn plan(&self, ctx: &Context) -> Option<$plan> { self.get_attr_plan(ctx).map(|attr| (*attr).clone()) }
        }
        impl Verify for $ty {
            fn verify(&self, ctx: &Context) -> Result<(), Error> {
                let op = self.get_operation().deref(ctx);
                let expected: &[Carrier] = &[$(Carrier::$carrier),*];
                if op.get_num_operands() != expected.len() || op.get_num_results() != 0 {
                    return verify_err!(op.loc(), "{} needs {} operands and no results", $name, expected.len());
                }
                let Some(plan) = self.plan(ctx) else { return verify_err!(op.loc(), "{} requires its typed plan", $name); };
                plan.verify(ctx)?;
                if !($check)(&plan) { return verify_err!(op.loc(), "{} has the wrong pipeline role", $name); }
                for (index, carrier) in expected.iter().enumerate() {
                    if !carrier_matches(ctx, op.get_operand(index), *carrier) {
                        return verify_err!(op.loc(), "{} operand {} has an invalid scalar/pointer carrier", $name, index);
                    }
                }
                Ok(())
            }
        }
    };
}
semantic_op!(
    /// Allocate TMEM columns and relinquish the allocation permit.
    ///
    /// Operands: `[token]`, a shared `u32` destination for the encoded address.
    /// All lanes of the allocating warp in both CTAs participate; synchronize
    /// readers before they load the token.
    CuteSm100TmemAllocOp,
    "cute.sm100_tmem_alloc",
    sm100_tmem_alloc_plan,
    CuteSm100TmemPlanAttr,
    [SharedU32],
    |_| true
);
semantic_op!(
    /// Free a completed two-CTA TMEM allocation.
    ///
    /// Operands: `[tmem]`, the encoded base address. Both allocating warps
    /// participate after all asynchronous users finish and the CTAs rendezvous.
    CuteSm100TmemDeallocOp,
    "cute.sm100_tmem_dealloc",
    sm100_tmem_dealloc_plan,
    CuteSm100TmemPlanAttr,
    [U32],
    |_| true
);
semantic_op!(
    /// Issue the paired A×B0 and A×B1 products into TMEM.
    ///
    /// Operands: `[tmem, a, b0, b1, accumulate]`; A/B pointers address shared
    /// FP16 tiles. One elected leader lane issues after input completion.
    /// `accumulate` controls the first K step; subsequent steps accumulate.
    /// All source tiles and destination TMEM stay live until MMA completion.
    CuteSm100TiledMmaOp,
    "cute.sm100_tiled_mma",
    sm100_tiled_mma_plan,
    CuteSm100MmaPlanAttr,
    [U32, ReadSharedF16, ReadSharedF16, ReadSharedF16, Bool],
    |_| true
);
semantic_op!(
    /// Add a row bias to FP32 TMEM values and place FP16 results in shared memory.
    ///
    /// Operands: `[tmem_slice, destination, tid, bias]`. The encoded slice
    /// selects the caller's 32 TMEM rows and 32 columns; `tid` is in `0..128`.
    /// All lanes of four epilogue warps participate after accumulator wait.
    /// Publish their shared writes before issuing a TMA store.
    CuteSm100TmemEpilogueOp,
    "cute.sm100_tmem_epilogue",
    sm100_tmem_epilogue_plan,
    CuteSm100EpiloguePlanAttr,
    [U32, SharedF16, U32, F32],
    |_| true
);
semantic_op!(
    /// Start one CTA's contribution to a cluster input transfer.
    ///
    /// Operands: `[descriptor, destination, row, column, barrier, rank]`.
    /// Coordinates count global FP16 elements; rank is the local CTA rank.
    /// All producer-warp lanes participate. Each CTA's destination covers its
    /// local tile; the leader's expected byte count covers both CTAs.
    CuteSm100ClusterTmaLoadOp,
    "cute.sm100_cluster_tma_load",
    sm100_cluster_tma_load_plan,
    CuteSm100ClusterCopyPlanAttr,
    [Descriptor, SharedF16, U32, U32, SharedBarrier, U32],
    |_| true
);
semantic_op!(
    /// Initialize input barriers from one elected lane per CTA.
    ///
    /// Operands: `[full, empty]`, shared arrays of `plan.stages` barriers.
    /// Publish initialization and synchronize the cluster before use.
    CuteSm100PipelineInitOp,
    "cute.sm100_pipeline_init",
    sm100_pipeline_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Wait until the producer may reuse an input stage.
    ///
    /// Operands: `[empty, stage, phase]`. The producer starts with phase one;
    /// the stage must be inside the shared empty-barrier array.
    CuteSm100PipelineAcquireOp,
    "cute.sm100_pipeline_acquire",
    sm100_pipeline_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Set the full barrier's expected transaction bytes for one input stage.
    ///
    /// Operands: `[full, stage]`. One elected rank-zero lane calls after
    /// acquire; the plan counts the copies from both CTAs.
    CuteSm100PipelineExpectOp,
    "cute.sm100_pipeline_expect",
    sm100_pipeline_expect_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Wait for an input stage and order subsequent tensor-core reads.
    ///
    /// Operands: `[full, stage, phase]`. Rank zero's MMA warp supplies its
    /// current consumer stage and phase.
    CuteSm100PipelineWaitOp,
    "cute.sm100_pipeline_wait",
    sm100_pipeline_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Release an input stage in both CTAs after pending MMA reads complete.
    ///
    /// Operands: `[empty, stage]`. One elected leader lane calls after all
    /// MMAs reading that stage have been issued.
    CuteSm100PipelineReleaseOp,
    "cute.sm100_pipeline_release",
    sm100_pipeline_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    /// Initialize the single accumulator slot from one elected lane per CTA.
    ///
    /// Operands: `[empty, full]`, two distinct shared barriers. Publish their
    /// initialization cluster-wide before either side uses the slot.
    CuteSm100AccumulatorInitOp,
    "cute.sm100_accumulator_init",
    sm100_accumulator_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Wait until epilogue warps have released the previous accumulator tile.
    ///
    /// Operands: `[empty, phase]`. The leader CTA's MMA warp acquires with
    /// initial phase one.
    CuteSm100AccumulatorAcquireOp,
    "cute.sm100_accumulator_acquire",
    sm100_accumulator_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Publish the accumulator to both CTAs after pending MMA completes.
    ///
    /// Operands: `[full]`. One elected leader lane calls after the last K stage.
    CuteSm100AccumulatorCommitOp,
    "cute.sm100_accumulator_commit",
    sm100_accumulator_commit_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Wait for an accumulator tile and order subsequent TMEM loads.
    ///
    /// Operands: `[full, phase]`. All consumer-warp lanes supply their current
    /// phase, initially zero.
    CuteSm100AccumulatorWaitOp,
    "cute.sm100_accumulator_wait",
    sm100_accumulator_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Release one consumer warp's accumulator portion to the leader CTA.
    ///
    /// Operands: `[empty]`. All epilogue-warp lanes call after their TMEM reads;
    /// lowering fences every lane and elects one arrival per warp.
    CuteSm100AccumulatorReleaseOp,
    "cute.sm100_accumulator_release",
    sm100_accumulator_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    /// Start an asynchronous shared-to-global FP16 output transfer.
    ///
    /// Operands: `[descriptor, source, row, column]`; coordinates count global
    /// elements. One elected issuer calls after writers publish and synchronize.
    /// Keep the source unchanged until its committed group finishes reading it.
    CuteSm100TmaStoreOp,
    "cute.sm100_tma_store",
    sm100_tma_store_plan,
    CuteSm100EpiloguePlanAttr,
    [Descriptor, ReadSharedF16, U32, U32],
    |_| true
);
semantic_op!(
    /// Commit this issuer's pending TMA stores as one hardware group.
    ///
    /// No operands. The same issuer performs the matching acquire and tail.
    CuteSm100StoreCommitOp,
    "cute.sm100_store_commit",
    sm100_store_commit_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    /// Wait until at most `plan.stages - 1` store groups still read shared memory.
    ///
    /// No operands. The store issuer may then reuse the oldest buffer in its
    /// ring; the newest group can still be pending.
    CuteSm100StoreAcquireOp,
    "cute.sm100_store_acquire",
    sm100_store_acquire_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    /// Wait until all committed stores stop reading shared memory.
    ///
    /// No operands. The store issuer calls before releasing its output buffers.
    CuteSm100StoreTailOp,
    "cute.sm100_store_tail",
    sm100_store_tail_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);

/// Apply a callback to the closed paired-SM100 operation set.
macro_rules! for_all_ops {
    ($callback:ident) => {
        $callback!(
            CuteSm100TmemAllocOp,
            CuteSm100TmemDeallocOp,
            CuteSm100TiledMmaOp,
            CuteSm100TmemEpilogueOp,
            CuteSm100ClusterTmaLoadOp,
            CuteSm100PipelineInitOp,
            CuteSm100PipelineAcquireOp,
            CuteSm100PipelineExpectOp,
            CuteSm100PipelineWaitOp,
            CuteSm100PipelineReleaseOp,
            CuteSm100AccumulatorInitOp,
            CuteSm100AccumulatorAcquireOp,
            CuteSm100AccumulatorCommitOp,
            CuteSm100AccumulatorWaitOp,
            CuteSm100AccumulatorReleaseOp,
            CuteSm100TmaStoreOp,
            CuteSm100StoreCommitOp,
            CuteSm100StoreAcquireOp,
            CuteSm100StoreTailOp
        )
    };
}
/// All operations belonging to this semantic family.
pub fn semantic_ids() -> Vec<OpId> {
    /// Collect the registered operation IDs from the supplied type list.
    macro_rules! ids { ($($ty:ty),*) => { vec![$(<$ty>::get_opid_static()),*] }; }
    for_all_ops!(ids)
}
/// Register the SM100 operations and their typed plans.
pub fn register(ctx: &mut Context) {
    /// Register every listed semantic operation in the supplied IR context.
    macro_rules! register_ops { ($($ty:ty),*) => { $(<$ty>::register(ctx);)* }; }
    for_all_ops!(register_ops);
    CuteSm100CollectorAttr::register(ctx);
    CuteSm100TmemPlanAttr::register(ctx);
    CuteSm100MmaPlanAttr::register(ctx);
    CuteSm100ClusterCopyPlanAttr::register(ctx);
    CuteSm100EpiloguePlanAttr::register(ctx);
    CuteSm100PipelinePlanAttr::register(ctx);
}
/// Locally verify a recognized operation; `None` means another family.
pub fn verify_local(ctx: &Context, op: Ptr<Operation>) -> Option<Result<(), Error>> {
    let id = Operation::get_opid(op, ctx);
    /// Select and run the local verifier for one recognized SM100 operation.
    macro_rules! verify_ops { ($($ty:ty),*) => { $(if id == <$ty>::get_opid_static() { return Some(<$ty>::wrap(op).verify(ctx)); })* }; }
    for_all_ops!(verify_ops);
    None
}
