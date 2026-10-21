/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression fixtures for this operation family.

use super::*;
use dialect_mir::ops::{MirConstantOp, MirUndefOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::{NOpdsInterface, NResultsInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::Signedness;
use pliron::linked_list::ContainsLinkedList;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use std::num::NonZeroUsize;

// Match importer-produced NVVM op IDs without a production dependency.
/// Test-only canonical two-CTA allocation recognized by the semantic lifecycle audit.
#[pliron_op(name = "nvvm.tcgen05_alloc_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
struct TestAllocCg2Op;
/// Test-only canonical two-CTA deallocation recognized by the lifecycle audit.
#[pliron_op(name = "nvvm.tcgen05_dealloc_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
struct TestDeallocCg2Op;
/// Test-only marker proving the two-CTA allocation permit was relinquished.
#[pliron_op(name = "nvvm.tcgen05_relinquish_alloc_permit_cg2", format, verifier = "succ", interfaces = [NOpdsInterface<0>, NResultsInterface<0>])]
struct TestRelinquishCg2Op;
/// Test-only one-CTA allocation used to reject the wrong collective mode.
#[pliron_op(name = "nvvm.tcgen05_alloc", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
struct TestAllocCg1Op;
/// Test-only one-CTA deallocation used to reject a mismatched collective mode.
#[pliron_op(name = "nvvm.tcgen05_dealloc", format, verifier = "succ", interfaces = [NOpdsInterface<2>, NResultsInterface<0>])]
struct TestDeallocCg1Op;

/// Append a direct unsigned 32-bit MIR literal for static-capacity verification tests.
fn literal_u32(ctx: &mut Context, block: Ptr<BasicBlock>, value: u32) -> Value {
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(ty, APInt::from_u32(value, NonZeroUsize::new(32).unwrap())),
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Append a test-only canonical CTM/NVVM operation and include it in the audited operation list.
fn append_ctm_op<T: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    ops: &mut Vec<Ptr<Operation>>,
    operands: Vec<Value>,
) {
    let op = Operation::new(ctx, T::get_concrete_op_info(), vec![], operands, vec![], 0);
    op.insert_at_back(block, ctx);
    ops.push(op);
}

/// Replace fixture allocation markers with configurable CTM lifecycle sites for compatibility and failure cases.
fn replace_cute_tmem_with_ctm(
    ctx: &mut Context,
    ops: &mut Vec<Ptr<Operation>>,
    columns: u32,
    group: u32,
    dealloc: bool,
    relinquish: bool,
) {
    let block = ops[0].deref(ctx).get_parent_block().unwrap();
    let shared = ops[0].deref(ctx).get_operand(0);
    let scalar = ops
        .iter()
        .find(|op| Operation::get_opid(**op, ctx) == CuteSm100TmemDeallocOp::get_opid_static())
        .unwrap()
        .deref(ctx)
        .get_operand(0);
    let is_cute_tmem = |op| {
        [
            CuteSm100TmemAllocOp::get_opid_static(),
            CuteSm100TmemDeallocOp::get_opid_static(),
        ]
        .contains(&Operation::get_opid(op, ctx))
    };
    for op in ops.iter().copied().filter(|op| is_cute_tmem(*op)) {
        op.unlink(ctx);
    }
    ops.retain(|op| !is_cute_tmem(*op));
    let count = literal_u32(ctx, block, columns);
    if group == 2 {
        append_ctm_op::<TestAllocCg2Op>(ctx, block, ops, vec![shared, count]);
        if relinquish {
            append_ctm_op::<TestRelinquishCg2Op>(ctx, block, ops, vec![]);
        }
        if dealloc {
            append_ctm_op::<TestDeallocCg2Op>(ctx, block, ops, vec![scalar, count]);
        }
    } else {
        append_ctm_op::<TestAllocCg1Op>(ctx, block, ops, vec![shared, count]);
        if dealloc {
            append_ctm_op::<TestDeallocCg1Op>(ctx, block, ops, vec![scalar, count]);
        }
    }
}

/// Build the supported paired M256/N192+160/K64 plan with canonical shared layouts.
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
/// Build an input or accumulator pipeline fixture from expected bytes and arrival count.
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
/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
    let op = MirUndefOp::new(ctx, ty).get_operation();
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}
/// Construct one complete paired-SM100 lifecycle, returning the module and all semantic operation sites.
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
    let barrier_ty: TypeHandle = MirPtrType::get(ctx, u64_ty, true, address_space::SHARED).into();
    let shared_ty: TypeHandle = MirPtrType::get(ctx, f16_ty, true, address_space::SHARED).into();
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
    /// Append one typed semantic operation to the test block and collected operation list.
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

/// The stage-byte promise includes A and both B partitions across both CTAs.
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
/// Unsupported allocation capacities and collective routing configurations fail static verification.
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
/// The backend-neutral entry point accepts a complete matching paired-SM100 lifecycle.
#[test]
fn complete_sm100_lifecycle_is_verified_by_shared_entrypoint() {
    let mut ctx = Context::new();
    let (module, _) = complete_story(&mut ctx, 77_824);
    crate::verify::verify_cute_semantics(&ctx, module).unwrap();
}
/// Canonical CTM two-CTA allocation/deallocation may supply storage for semantic CuTe MMA.
#[test]
fn accepts_ctm_group2_tmem_lifecycle_with_cute_mma() {
    let mut ctx = Context::new();
    let (module, mut ops) = complete_story(&mut ctx, 77_824);
    replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, true, true);
    verify_story(&ctx, &ops).unwrap();
    crate::verify::verify_cute_semantics(&ctx, module).unwrap();
}
/// A recognized allocation cannot omit the matching release of TMEM capacity.
#[test]
fn rejects_ctm_tmem_lifecycle_missing_deallocation() {
    let mut ctx = Context::new();
    let (_, mut ops) = complete_story(&mut ctx, 77_824);
    replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, false, true);
    let error = verify_story(&ctx, &ops).unwrap_err();
    assert!(error.contains("dealloc"), "{error}");
}
/// CTM allocation must relinquish its allocation permit before the lifecycle is accepted.
#[test]
fn rejects_ctm_tmem_lifecycle_missing_allocation_permit() {
    let mut ctx = Context::new();
    let (_, mut ops) = complete_story(&mut ctx, 77_824);
    replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 2, true, false);
    let error = verify_story(&ctx, &ops).unwrap_err();
    assert!(error.contains("permit"), "{error}");
}
/// One-CTA allocation/deallocation cannot satisfy a two-CTA MMA storage contract.
#[test]
fn rejects_ctm_tmem_lifecycle_with_group1_operations() {
    let mut ctx = Context::new();
    let (_, mut ops) = complete_story(&mut ctx, 77_824);
    replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 512, 1, true, false);
    let error = verify_story(&ctx, &ops).unwrap_err();
    assert!(error.contains("CTA group 2"), "{error}");
}
/// A directly known CTM column count must match the paired MMA's TMEM capacity.
#[test]
fn rejects_ctm_tmem_lifecycle_with_known_wrong_capacity() {
    let mut ctx = Context::new();
    let (_, mut ops) = complete_story(&mut ctx, 77_824);
    replace_cute_tmem_with_ctm(&mut ctx, &mut ops, 256, 2, true, true);
    let error = verify_story(&ctx, &ops).unwrap_err();
    assert!(error.contains("capacity"), "{error}");
}
/// The input barrier must expect both CTAs' copies, not only the local tile bytes.
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
/// Every shared MMA operand placement must have a corresponding cluster input-copy layout.
#[test]
fn rejects_mma_input_layout_without_matching_cluster_copy() {
    let mut ctx = Context::new();
    let (_, mut ops) = complete_story(&mut ctx, 77_824);
    ops.retain(|op| Operation::get_opid(*op, &ctx) != CuteSm100ClusterTmaLoadOp::get_opid_static());
    assert!(
        verify_story(&ctx, &ops)
            .unwrap_err()
            .contains("no matching cluster TMA copy")
    );
}
/// The epilogue must release its accumulator slot before the paired lifecycle is complete.
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
/// Malformed positional scalar types are rejected by the shared verifier before backend emission.
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
