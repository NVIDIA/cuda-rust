/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM100 tensor-memory, collector-aware tiled MMA, and cluster pipelines.
//!
//! These operations keep ordinary MIR scalar carriers and retain the static
//! layout and collective contracts as typed attributes. A paired MMA is one
//! semantic operation: its A collector is filled and last-used within each
//! K=16 step, so a backend cannot accidentally carry it across K iterations.
//!
//! Cluster TMA loads are collective across all 32 producer-warp lanes. Native
//! CuTe lowering owns lane election and warp synchronization; callers must
//! not place a load inside an elected-lane branch. TMA stores retain a single
//! issuer, and accumulator release fences all lanes before electing an arrival.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::attribute::Attribute;
use pliron::builtin::types::{FP32Type, IntegerType};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::location::Located;
use pliron::op::{Op, OpId};
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::Typed;
use pliron::value::Value;
use pliron::{verify_err, verify_err_noloc};
use pliron_derive::{pliron_attr, pliron_op};

use crate::attributes::{CuteComposedLayoutAttr, CuteTmaStorePipelineAttr};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};

/// The first N partition fills A, and the second consumes its last use.
#[pliron_attr(name = "cute.sm100_collector", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteSm100CollectorAttr {
    AFillLastUse,
}

#[pliron_attr(
    name = "cute.sm100_tmem_plan",
    format = "`<` $columns `,` $cta_group `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100TmemPlanAttr {
    pub columns: u32,
    pub cta_group: u32,
}
impl Verify for CuteSm100TmemPlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.cta_group != 2
            || !(32..=512).contains(&self.columns)
            || !self.columns.is_power_of_two()
        {
            return verify_err_noloc!(
                "SM100 tensor memory requires CTA group 2 and power-of-two columns in 32..=512"
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "cute.sm100_mma_plan",
    format = "`<` $m `,` $n0 `,` $n1 `,` $k `,` $tmem_columns `,` $cta_group `,` $collector `,` $a_layout `,` $b0_layout `,` $b1_layout `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100MmaPlanAttr {
    pub m: u32,
    pub n0: u32,
    pub n1: u32,
    pub k: u32,
    pub tmem_columns: u32,
    pub cta_group: u32,
    pub collector: CuteSm100CollectorAttr,
    pub a_layout: CuteComposedLayoutAttr,
    pub b0_layout: CuteComposedLayoutAttr,
    pub b1_layout: CuteComposedLayoutAttr,
}

fn row_major(rows: u32, columns: u32, swizzle_bits: u32) -> ComposedLayout {
    let inner: Layout = format!("({rows},{columns}):({columns},1)")
        .parse()
        .expect("fixed rank-two layout");
    ComposedLayout::new(
        Swizzle::new(swizzle_bits, 3, 3),
        0,
        inner,
        OffsetUnit::Elements,
    )
    .expect("fixed SM100 FP16 swizzle")
}

impl CuteSm100MmaPlanAttr {
    /// Bytes copied by both CTAs for one complete A/B0/B1 stage.
    pub fn transaction_bytes(&self) -> Option<u32> {
        self.m
            .checked_add(self.n0)?
            .checked_add(self.n1)?
            .checked_mul(self.k)?
            .checked_mul(2)
    }
}
impl Verify for CuteSm100MmaPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        CuteSm100TmemPlanAttr {
            columns: self.tmem_columns,
            cta_group: self.cta_group,
        }
        .verify(ctx)?;
        if self.m != 256
            || self.k != 64
            || self.tmem_columns != 512
            || !(16..=256).contains(&self.n0)
            || !(16..=256).contains(&self.n1)
            || !self.n0.is_multiple_of(16)
            || !self.n1.is_multiple_of(16)
            || self.n0 + self.n1 > self.tmem_columns
        {
            return verify_err_noloc!(
                "SM100 paired FP16 MMA requires M256/K64, two N partitions in 16..=256 divisible by 16, and 512 TMEM columns"
            );
        }
        for (name, layout, rows) in [
            ("A", &self.a_layout, 128),
            ("B0", &self.b0_layout, self.n0 / 2),
            ("B1", &self.b1_layout, self.n1 / 2),
        ] {
            layout.verify(ctx)?;
            if layout.0 != row_major(rows, self.k, 3) {
                return verify_err_noloc!(
                    "SM100 {name} MMA layout must be the matching K-major FP16 B128 shared tile"
                );
            }
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "cute.sm100_cluster_copy_plan",
    format = "`<` $layout `,` $cta_group `,` $leader_rank `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100ClusterCopyPlanAttr {
    pub layout: CuteComposedLayoutAttr,
    pub cta_group: u32,
    pub leader_rank: u32,
}
impl Verify for CuteSm100ClusterCopyPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        let dimensions = self.layout.0.inner().shape.leaves();
        if self.cta_group != 2
            || self.leader_rank != 0
            || dimensions.len() != 2
            || dimensions[0] < 8
            || dimensions[0] > 128
            || dimensions[0] % 8 != 0
            || dimensions[1] != 64
            || self.layout.0 != row_major(dimensions[0] as u32, 64, 3)
        {
            return verify_err_noloc!(
                "SM100 cluster TMA requires CTA group 2, leader 0, and a K64 FP16 B128 tile with 8..=128 rows divisible by 8"
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "cute.sm100_epilogue_plan",
    format = "`<` $layout `,` $rows `,` $columns `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteSm100EpiloguePlanAttr {
    pub layout: CuteComposedLayoutAttr,
    pub rows: u32,
    pub columns: u32,
}
impl Verify for CuteSm100EpiloguePlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.layout.verify(ctx)?;
        if self.rows != 128 || self.columns != 32 || self.layout.0 != row_major(128, 32, 2) {
            return verify_err_noloc!("SM100 epilogue requires the 128x32 FP16 B64 shared layout");
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "cute.sm100_pipeline_plan",
    format = "`<` $stages `,` $transaction_bytes `,` $consumer_arrivals `,` $cta_group `,` $leader_rank `,` $multicast_mask `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteSm100PipelinePlanAttr {
    pub stages: u32,
    pub transaction_bytes: u32,
    pub consumer_arrivals: u32,
    pub cta_group: u32,
    pub leader_rank: u32,
    pub multicast_mask: u32,
}
impl Verify for CuteSm100PipelinePlanAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0
            || self.cta_group != 2
            || self.leader_rank != 0
            || self.multicast_mask != 3
        {
            return verify_err_noloc!(
                "SM100 pipeline requires positive stages, CTA group 2, leader 0, and multicast mask 3"
            );
        }
        if self.transaction_bytes == 0 {
            if self.stages != 1
                || self.consumer_arrivals == 0
                || self.consumer_arrivals > 64
                || !self.consumer_arrivals.is_multiple_of(2)
            {
                return verify_err_noloc!(
                    "SM100 accumulator pipeline requires one stage and an even number of consumer arrivals in 2..=64"
                );
            }
        } else if self.consumer_arrivals != 1 || !self.transaction_bytes.is_multiple_of(16) {
            return verify_err_noloc!(
                "SM100 input pipeline requires one MMA completion arrival and a transaction size divisible by 16"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Carrier {
    U32,
    Bool,
    F32,
    SharedU32,
    SharedBarrier,
    SharedF16,
    ReadSharedF16,
    Descriptor,
}
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

macro_rules! semantic_op {
    ($ty:ident, $name:literal, $key:ident, $plan:ty, [$($carrier:ident),*], $check:expr) => {
        #[pliron_op(name = $name, format, attributes = ($key: $plan))]
        pub struct $ty;
        impl $ty {
            pub fn new(ctx: &mut Context, operands: Vec<Value>, plan: $plan) -> Self {
                let op = Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], operands, vec![], 0) };
                op.get_operation().deref_mut(ctx).attributes.set(stringify!($key).try_into().expect("plan key"), plan);
                op
            }
            pub fn wrap(op: Ptr<Operation>) -> Self { Self { op } }
            pub fn get_attr_plan<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, $plan>> {
                std::cell::Ref::filter_map(self.get_operation().deref(ctx), |op|
                    op.attributes.get::<$plan>(&stringify!($key).try_into().expect("plan key"))).ok()
            }
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
    CuteSm100TmemAllocOp,
    "cute.sm100_tmem_alloc",
    sm100_tmem_alloc_plan,
    CuteSm100TmemPlanAttr,
    [SharedU32],
    |_| true
);
semantic_op!(
    CuteSm100TmemDeallocOp,
    "cute.sm100_tmem_dealloc",
    sm100_tmem_dealloc_plan,
    CuteSm100TmemPlanAttr,
    [U32],
    |_| true
);
semantic_op!(
    CuteSm100TiledMmaOp,
    "cute.sm100_tiled_mma",
    sm100_tiled_mma_plan,
    CuteSm100MmaPlanAttr,
    [U32, ReadSharedF16, ReadSharedF16, ReadSharedF16, Bool],
    |_| true
);
semantic_op!(
    CuteSm100TmemEpilogueOp,
    "cute.sm100_tmem_epilogue",
    sm100_tmem_epilogue_plan,
    CuteSm100EpiloguePlanAttr,
    [U32, SharedF16, U32, F32],
    |_| true
);
semantic_op!(
    CuteSm100ClusterTmaLoadOp,
    "cute.sm100_cluster_tma_load",
    sm100_cluster_tma_load_plan,
    CuteSm100ClusterCopyPlanAttr,
    [Descriptor, SharedF16, U32, U32, SharedBarrier, U32],
    |_| true
);
semantic_op!(
    CuteSm100PipelineInitOp,
    "cute.sm100_pipeline_init",
    sm100_pipeline_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    CuteSm100PipelineAcquireOp,
    "cute.sm100_pipeline_acquire",
    sm100_pipeline_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    CuteSm100PipelineExpectOp,
    "cute.sm100_pipeline_expect",
    sm100_pipeline_expect_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    CuteSm100PipelineWaitOp,
    "cute.sm100_pipeline_wait",
    sm100_pipeline_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    CuteSm100PipelineReleaseOp,
    "cute.sm100_pipeline_release",
    sm100_pipeline_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes > 0
);
semantic_op!(
    CuteSm100AccumulatorInitOp,
    "cute.sm100_accumulator_init",
    sm100_accumulator_init_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    CuteSm100AccumulatorAcquireOp,
    "cute.sm100_accumulator_acquire",
    sm100_accumulator_acquire_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    CuteSm100AccumulatorCommitOp,
    "cute.sm100_accumulator_commit",
    sm100_accumulator_commit_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    CuteSm100AccumulatorWaitOp,
    "cute.sm100_accumulator_wait",
    sm100_accumulator_wait_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier, U32],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    CuteSm100AccumulatorReleaseOp,
    "cute.sm100_accumulator_release",
    sm100_accumulator_release_plan,
    CuteSm100PipelinePlanAttr,
    [SharedBarrier],
    |p: &CuteSm100PipelinePlanAttr| p.transaction_bytes == 0
);
semantic_op!(
    CuteSm100TmaStoreOp,
    "cute.sm100_tma_store",
    sm100_tma_store_plan,
    CuteSm100EpiloguePlanAttr,
    [Descriptor, ReadSharedF16, U32, U32],
    |_| true
);
semantic_op!(
    CuteSm100StoreCommitOp,
    "cute.sm100_store_commit",
    sm100_store_commit_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    CuteSm100StoreAcquireOp,
    "cute.sm100_store_acquire",
    sm100_store_acquire_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);
semantic_op!(
    CuteSm100StoreTailOp,
    "cute.sm100_store_tail",
    sm100_store_tail_plan,
    CuteTmaStorePipelineAttr,
    [],
    |_| true
);

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
    macro_rules! ids { ($($ty:ty),*) => { vec![$(<$ty>::get_opid_static()),*] }; }
    for_all_ops!(ids)
}
/// Register the SM100 operations and their typed plans.
pub fn register(ctx: &mut Context) {
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
    macro_rules! verify_ops { ($($ty:ty),*) => { $(if id == <$ty>::get_opid_static() { return Some(<$ty>::wrap(op).verify(ctx)); })* }; }
    for_all_ops!(verify_ops);
    None
}

fn static_plan<T: Attribute + Clone>(ctx: &Context, op: Ptr<Operation>) -> Option<T> {
    op.deref(ctx)
        .attributes
        .0
        .values()
        .find_map(|attr| attr.downcast_ref::<T>().cloned())
}

/// Check static contracts across the scalar-carrier SM100 graph in one function.
///
/// This checks matching tile layouts, copy transaction sizes, TMEM capacity,
/// and complete lifecycle membership. Dynamic phase arithmetic, lane election,
/// and pointer aliasing remain the documented unsafe Rust preconditions.
pub fn verify_story(ctx: &Context, ops: &[Ptr<Operation>]) -> Result<(), String> {
    let id = |op| Operation::get_opid(op, ctx);
    let find = |wanted| ops.iter().copied().find(|op| id(*op) == wanted);
    let Some(mma_op) = find(CuteSm100TiledMmaOp::get_opid_static()) else {
        return Ok(());
    };
    let mma = CuteSm100TiledMmaOp::wrap(mma_op)
        .plan(ctx)
        .ok_or("SM100 MMA plan missing")?;
    for needed in [
        CuteSm100TmemAllocOp::get_opid_static(),
        CuteSm100TmemDeallocOp::get_opid_static(),
        CuteSm100PipelineInitOp::get_opid_static(),
        CuteSm100PipelineAcquireOp::get_opid_static(),
        CuteSm100PipelineExpectOp::get_opid_static(),
        CuteSm100PipelineWaitOp::get_opid_static(),
        CuteSm100PipelineReleaseOp::get_opid_static(),
        CuteSm100AccumulatorInitOp::get_opid_static(),
        CuteSm100AccumulatorAcquireOp::get_opid_static(),
        CuteSm100AccumulatorCommitOp::get_opid_static(),
        CuteSm100AccumulatorWaitOp::get_opid_static(),
        CuteSm100AccumulatorReleaseOp::get_opid_static(),
        CuteSm100TmemEpilogueOp::get_opid_static(),
        CuteSm100TmaStoreOp::get_opid_static(),
        CuteSm100StoreCommitOp::get_opid_static(),
        CuteSm100StoreAcquireOp::get_opid_static(),
        CuteSm100StoreTailOp::get_opid_static(),
    ] {
        if find(needed.clone()).is_none() {
            return Err(format!(
                "SM100 tiled MMA is missing required lifecycle operation {needed}"
            ));
        }
    }
    let mut copied_layouts = Vec::new();
    let mut input_plan = None;
    let mut accumulator_plan = None;
    let mut store_plan = None;
    let input_ids = [
        CuteSm100PipelineInitOp::get_opid_static(),
        CuteSm100PipelineAcquireOp::get_opid_static(),
        CuteSm100PipelineExpectOp::get_opid_static(),
        CuteSm100PipelineWaitOp::get_opid_static(),
        CuteSm100PipelineReleaseOp::get_opid_static(),
    ];
    let accumulator_ids = [
        CuteSm100AccumulatorInitOp::get_opid_static(),
        CuteSm100AccumulatorAcquireOp::get_opid_static(),
        CuteSm100AccumulatorCommitOp::get_opid_static(),
        CuteSm100AccumulatorWaitOp::get_opid_static(),
        CuteSm100AccumulatorReleaseOp::get_opid_static(),
    ];
    for &op in ops {
        let opid = id(op);
        if opid == CuteSm100TiledMmaOp::get_opid_static()
            && CuteSm100TiledMmaOp::wrap(op).plan(ctx).as_ref() != Some(&mma)
        {
            return Err("SM100 tiled MMA plans in one function must agree".into());
        }
        if opid == CuteSm100TmemAllocOp::get_opid_static()
            || opid == CuteSm100TmemDeallocOp::get_opid_static()
        {
            let plan =
                static_plan::<CuteSm100TmemPlanAttr>(ctx, op).ok_or("SM100 TMEM plan missing")?;
            if plan.columns != mma.tmem_columns || plan.cta_group != mma.cta_group {
                return Err("SM100 allocation capacity disagrees with tiled MMA".into());
            }
        }
        if input_ids.contains(&opid) || accumulator_ids.contains(&opid) {
            let plan = static_plan::<CuteSm100PipelinePlanAttr>(ctx, op)
                .ok_or("SM100 pipeline plan missing")?;
            let target = if input_ids.contains(&opid) {
                &mut input_plan
            } else {
                &mut accumulator_plan
            };
            if target.is_some_and(|previous| previous != plan) {
                return Err("SM100 pipeline lifecycle plans disagree".into());
            }
            *target = Some(plan);
            if plan.transaction_bytes > 0 && Some(plan.transaction_bytes) != mma.transaction_bytes()
            {
                return Err(format!(
                    "SM100 pipeline promises {} bytes but the MMA operand tiles need {}",
                    plan.transaction_bytes,
                    mma.transaction_bytes().unwrap()
                ));
            }
            if plan.transaction_bytes == 0 && plan.consumer_arrivals != 8 {
                return Err(
                    "SM100 128x32 epilogue requires four consumer warps in each CTA (8 arrivals)"
                        .into(),
                );
            }
        }
        if opid == CuteSm100ClusterTmaLoadOp::get_opid_static() {
            copied_layouts.push(
                CuteSm100ClusterTmaLoadOp::wrap(op)
                    .plan(ctx)
                    .ok_or("SM100 copy plan missing")?
                    .layout,
            );
        }
        if [
            CuteSm100StoreCommitOp::get_opid_static(),
            CuteSm100StoreAcquireOp::get_opid_static(),
            CuteSm100StoreTailOp::get_opid_static(),
        ]
        .contains(&opid)
        {
            let plan = static_plan::<CuteTmaStorePipelineAttr>(ctx, op)
                .ok_or("SM100 store plan missing")?;
            if store_plan.is_some_and(|previous| previous != plan) {
                return Err("SM100 store lifecycle stage counts disagree".into());
            }
            store_plan = Some(plan);
        }
    }
    for required in [&mma.a_layout, &mma.b0_layout, &mma.b1_layout] {
        if !copied_layouts.contains(required) {
            return Err("SM100 MMA operand layout has no matching cluster TMA copy".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirUndefOp;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::Signedness;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::r#type::TypeHandle;

    fn mma_plan() -> CuteSm100MmaPlanAttr {
        CuteSm100MmaPlanAttr {
            m: 256,
            n0: 192,
            n1: 160,
            k: 64,
            tmem_columns: 512,
            cta_group: 2,
            collector: CuteSm100CollectorAttr::AFillLastUse,
            a_layout: CuteComposedLayoutAttr(row_major(128, 64, 3)),
            b0_layout: CuteComposedLayoutAttr(row_major(96, 64, 3)),
            b1_layout: CuteComposedLayoutAttr(row_major(80, 64, 3)),
        }
    }
    fn pipeline(bytes: u32, arrivals: u32) -> CuteSm100PipelinePlanAttr {
        CuteSm100PipelinePlanAttr {
            stages: if bytes == 0 { 1 } else { 5 },
            transaction_bytes: bytes,
            consumer_arrivals: arrivals,
            cta_group: 2,
            leader_rank: 0,
            multicast_mask: 3,
        }
    }
    fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
        let op = MirUndefOp::new(ctx, ty).get_operation();
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }
    fn complete_story(ctx: &mut Context, bytes: u32) -> (Ptr<Operation>, Vec<Ptr<Operation>>) {
        dialect_mir::register(ctx);
        crate::register(ctx);
        let module = ModuleOp::new(ctx, "sm100_test".try_into().unwrap()).get_operation();
        let block = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .iter(ctx)
            .next()
            .unwrap();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(ctx).into();
        let token_ty: TypeHandle = MirPtrType::get(ctx, u32_ty, true, address_space::SHARED).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(ctx, u64_ty, true, address_space::SHARED).into();
        let shared_ty: TypeHandle =
            MirPtrType::get(ctx, f16_ty, true, address_space::SHARED).into();
        let desc_ty: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
        let scalar = undef(ctx, block, u32_ty);
        let condition = undef(ctx, block, bool_ty);
        let bias = undef(ctx, block, f32_ty);
        let token = undef(ctx, block, token_ty);
        let full = undef(ctx, block, barrier_ty);
        let empty = undef(ctx, block, barrier_ty);
        let afull = undef(ctx, block, barrier_ty);
        let aempty = undef(ctx, block, barrier_ty);
        let shared = undef(ctx, block, shared_ty);
        let desc = undef(ctx, block, desc_ty);
        let mut ops = Vec::new();
        macro_rules! append { ($ty:ident,[$($arg:expr),*],$plan:expr) => {{
            let operation=$ty::new(ctx,vec![$($arg),*],$plan).get_operation();
            operation.insert_at_back(block,ctx);ops.push(operation);
        }}; }
        let tmem = CuteSm100TmemPlanAttr {
            columns: 512,
            cta_group: 2,
        };
        let input = pipeline(bytes, 1);
        let acc = pipeline(0, 8);
        let epi = CuteSm100EpiloguePlanAttr {
            layout: CuteComposedLayoutAttr(row_major(128, 32, 2)),
            rows: 128,
            columns: 32,
        };
        append!(CuteSm100TmemAllocOp, [token], tmem);
        append!(CuteSm100PipelineInitOp, [full, empty], input);
        append!(CuteSm100AccumulatorInitOp, [aempty, afull], acc);
        append!(CuteSm100PipelineAcquireOp, [empty, scalar, scalar], input);
        append!(CuteSm100PipelineExpectOp, [full, scalar], input);
        for rows in [128, 96, 80] {
            append!(
                CuteSm100ClusterTmaLoadOp,
                [desc, shared, scalar, scalar, full, scalar],
                CuteSm100ClusterCopyPlanAttr {
                    layout: CuteComposedLayoutAttr(row_major(rows, 64, 3)),
                    cta_group: 2,
                    leader_rank: 0
                }
            );
        }
        append!(CuteSm100AccumulatorAcquireOp, [aempty, scalar], acc);
        append!(CuteSm100PipelineWaitOp, [full, scalar, scalar], input);
        append!(
            CuteSm100TiledMmaOp,
            [scalar, shared, shared, shared, condition],
            mma_plan()
        );
        append!(CuteSm100PipelineReleaseOp, [empty, scalar], input);
        append!(CuteSm100AccumulatorCommitOp, [afull], acc);
        append!(CuteSm100AccumulatorWaitOp, [afull, scalar], acc);
        append!(
            CuteSm100TmemEpilogueOp,
            [scalar, shared, scalar, bias],
            epi.clone()
        );
        append!(CuteSm100TmaStoreOp, [desc, shared, scalar, scalar], epi);
        append!(CuteSm100StoreCommitOp, [], CuteTmaStorePipelineAttr::new(2));
        append!(
            CuteSm100StoreAcquireOp,
            [],
            CuteTmaStorePipelineAttr::new(2)
        );
        append!(CuteSm100AccumulatorReleaseOp, [aempty], acc);
        append!(CuteSm100StoreTailOp, [], CuteTmaStorePipelineAttr::new(2));
        append!(CuteSm100TmemDeallocOp, [scalar], tmem);
        (module, ops)
    }

    #[test]
    fn paired_plan_derives_complete_two_cta_transaction_bytes() {
        let ctx = Context::new();
        let plan = mma_plan();
        plan.verify(&ctx).unwrap();
        assert_eq!(plan.transaction_bytes(), Some(77_824));
        let mut bad = plan.clone();
        bad.b0_layout = plan.b1_layout.clone();
        assert!(bad.verify(&ctx).is_err());
        let mut bad = plan.clone();
        bad.n1 = 176;
        assert!(bad.verify(&ctx).is_err());
        let mut bad = plan;
        bad.cta_group = 1;
        assert!(bad.verify(&ctx).is_err());
    }
    #[test]
    fn tmem_and_cluster_collective_contracts_fail_closed() {
        let ctx = Context::new();
        assert!(
            CuteSm100TmemPlanAttr {
                columns: 48,
                cta_group: 2
            }
            .verify(&ctx)
            .is_err()
        );
        assert!(
            CuteSm100TmemPlanAttr {
                columns: 512,
                cta_group: 1
            }
            .verify(&ctx)
            .is_err()
        );
        let mut plan = pipeline(77_824, 1);
        plan.multicast_mask = 1;
        assert!(plan.verify(&ctx).is_err());
        let mut plan = pipeline(0, 8);
        plan.stages = 2;
        assert!(plan.verify(&ctx).is_err());
    }
    #[test]
    fn complete_sm100_lifecycle_is_verified_by_shared_entrypoint() {
        let mut ctx = Context::new();
        let (module, _) = complete_story(&mut ctx, 77_824);
        crate::verify::verify_cute_semantics(&ctx, module).unwrap();
    }
    #[test]
    fn rejects_only_one_ctas_expected_transaction_bytes() {
        let mut ctx = Context::new();
        let (module, _) = complete_story(&mut ctx, 38_912);
        let error = crate::verify::verify_cute_semantics(&ctx, module)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("promises 38912 bytes") && error.contains("77824"),
            "{error}"
        );
    }
    #[test]
    fn rejects_mma_input_layout_without_matching_cluster_copy() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        ops.retain(|op| {
            Operation::get_opid(*op, &ctx) != CuteSm100ClusterTmaLoadOp::get_opid_static()
        });
        assert!(
            verify_story(&ctx, &ops)
                .unwrap_err()
                .contains("no matching cluster TMA copy")
        );
    }
    #[test]
    fn rejects_missing_accumulator_release() {
        let mut ctx = Context::new();
        let (_, mut ops) = complete_story(&mut ctx, 77_824);
        ops.retain(|op| {
            Operation::get_opid(*op, &ctx) != CuteSm100AccumulatorReleaseOp::get_opid_static()
        });
        assert!(
            verify_story(&ctx, &ops)
                .unwrap_err()
                .contains("sm100_accumulator_release")
        );
    }
    #[test]
    fn rejects_bad_scalar_carriers_before_backend_lowering() {
        let mut ctx = Context::new();
        let (_, ops) = complete_story(&mut ctx, 77_824);
        let mma = ops
            .iter()
            .find(|op| Operation::get_opid(**op, &ctx) == CuteSm100TiledMmaOp::get_opid_static())
            .copied()
            .unwrap();
        let scalar = mma.deref(&ctx).get_operand(0);
        let malformed = CuteSm100TiledMmaOp::new(&mut ctx, vec![scalar; 5], mma_plan());
        assert!(malformed.verify(&ctx).is_err());
        let malformed = CuteSm100TmemAllocOp::new(
            &mut ctx,
            vec![],
            CuteSm100TmemPlanAttr {
                columns: 512,
                cta_group: 2,
            },
        );
        assert!(malformed.verify(&ctx).is_err());
    }
}
