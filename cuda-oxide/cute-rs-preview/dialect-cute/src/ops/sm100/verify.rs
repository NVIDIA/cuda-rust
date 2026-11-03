/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-operation layout, capacity, and collective lifecycle verification.

use super::*;

/// Find and clone the operation attribute of the requested static plan type.
fn static_plan<T: Attribute + Clone>(ctx: &Context, op: Ptr<Operation>) -> Option<T> {
    op.deref(ctx)
        .attributes
        .0
        .values()
        .find_map(|attr| attr.downcast_ref::<T>().cloned())
}

// CTM lowers its typed group-2 TMEM guard into these canonical NVVM ops.
// Read the column count when MIR still carries it as a direct constant; the
// group-2 op name is the static group contract in all cases.
/// Read a direct literal column count from a canonical CTM/NVVM allocation operation.
fn nvvm_tmem_columns(ctx: &Context, op: Ptr<Operation>) -> Option<u64> {
    let value = op.deref(ctx).get_operand(1);
    let constant = value.defining_op()?;
    if Operation::get_opid(constant, ctx) != MirConstantOp::get_opid_static() {
        return None;
    }
    let attribute = MirConstantOp::new(constant).get_attr_value(ctx)?;
    (attribute.value().bw() <= 64).then(|| attribute.value().to_u64())
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
    let nvvm = |name: &str| {
        ops.iter()
            .copied()
            .filter(|op| id(*op).to_string() == name)
            .collect::<Vec<_>>()
    };
    let ctm_alloc = nvvm("nvvm.tcgen05_alloc_cg2");
    let ctm_dealloc = nvvm("nvvm.tcgen05_dealloc_cg2");
    if find(CuteSm100TmemAllocOp::get_opid_static()).is_none() && ctm_alloc.is_empty() {
        if !nvvm("nvvm.tcgen05_alloc").is_empty() {
            return Err("SM100 tiled MMA requires CTA group 2 TMEM allocation".into());
        }
        return Err(
            "SM100 tiled MMA is missing required lifecycle operation cute.sm100_tmem_alloc or nvvm.tcgen05_alloc_cg2"
                .into(),
        );
    }
    if find(CuteSm100TmemDeallocOp::get_opid_static()).is_none() && ctm_dealloc.is_empty() {
        if !nvvm("nvvm.tcgen05_dealloc").is_empty() {
            return Err("SM100 tiled MMA requires CTA group 2 TMEM deallocation".into());
        }
        return Err(
            "SM100 tiled MMA is missing required lifecycle operation cute.sm100_tmem_dealloc or nvvm.tcgen05_dealloc_cg2"
                .into(),
        );
    }
    if !ctm_alloc.is_empty() && nvvm("nvvm.tcgen05_relinquish_alloc_permit_cg2").is_empty() {
        return Err(
            "SM100 CTM group-2 TMEM allocation is missing its allocation permit relinquish".into(),
        );
    }
    for op in ctm_alloc.into_iter().chain(ctm_dealloc) {
        if op.deref(ctx).get_num_operands() != 2 {
            return Err("SM100 CTM TMEM lifecycle operation must have two operands".into());
        }
        if nvvm_tmem_columns(ctx, op).is_some_and(|columns| columns != mma.tmem_columns as u64) {
            return Err("SM100 allocation capacity disagrees with tiled MMA".into());
        }
    }
    for needed in [
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
