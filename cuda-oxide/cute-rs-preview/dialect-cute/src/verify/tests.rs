/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for CuTe operation, type, and semantic verification contracts.

use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{
        MirAllocaOp, MirCastOp, MirGotoOp, MirLoadOp, MirNotOp, MirReturnOp, MirStoreOp,
        MirUndefOp, control_flow::MirCondBranchOp, function::MirFuncOp,
    },
    types::{MirPtrType, address_space},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OperandSegmentInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::Value,
};

use super::{
    epilogue::require_epilogue_cfg_links,
    mma::{build_mma_facts, require_smem_overlay_pair, verify_mma_use_closure},
    pipeline::require_producer_pipeline_order,
    provenance::{
        is_local_cell_pointer_alias_cast, is_semantic_pointer_carrier_cast, local_cell_is_closed,
    },
    scheduler::{build_scheduler_facts, scheduler_root},
    *,
};
use crate::{
    attributes::{
        CutePipelineStateAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr,
        CuteTensorLayoutAttr, CuteTileGridAttr, CuteTiledMmaPlanAttr,
    },
    layout::{ComposedLayout, OffsetUnit},
    ops::gemm_tma::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp},
    ops::mma::{CuteMmaPartitionBOp, CuteSmemTensorOverlayOp},
    ops::pipeline::{
        CutePipelineConsumerWaitOp, CutePipelineProducerExpectTxOp, CutePipelineStateNewOp,
        CuteTmaLoadPipelineMakeOp,
    },
    ops::scheduler::{
        CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp,
        CuteSchedulerNew1dOp, CuteWorkTileCoordinatesOp,
    },
    types::CuteTensorViewType,
};

/// Create a test module and return its entry block for constructing semantic IR directly.
fn module_top(ctx: &mut Context) -> (Ptr<Operation>, Ptr<BasicBlock>) {
    dialect_mir::register(ctx);
    crate::register(ctx);
    let module = ModuleOp::new(ctx, "cute_verify_test".try_into().unwrap());
    let module = module.get_operation();
    let region = module.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next().unwrap();
    (module, block)
}

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
    let operation = MirUndefOp::new(ctx, ty).get_operation();
    operation.insert_at_back(block, ctx);
    operation.deref(ctx).get_result(0)
}

/// Construct a valid compiler-only contiguous tensor type for escape-audit tests.
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

/// Append a compiler-owned local allocation used to test closed carrier storage.
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

/// Append a non-volatile MIR store into the fixture's local cell.
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

/// Append a non-volatile MIR load and return its SSA result.
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

/// Append a pointer cast with controlled pointee, mutability, and address-space facts.
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

/// Append an unconditional successor edge with explicit block-argument operands.
fn goto(ctx: &mut Context, block: Ptr<BasicBlock>, target: Ptr<BasicBlock>, operands: Vec<Value>) {
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

/// Append a conditional branch with distinct successor operands for CFG/provenance tests.
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

/// Construct one valid persistent-scheduler function with configurable body/exit placement.
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

/// The escape audit must inspect ordinary IR types even when no semantic CuTe operation exposes the ghost.
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

/// Function scopes isolate scheduler origins, and verification must leave the input module unchanged.
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

/// CFG work guards, rather than textual traversal order, determine whether a selected tile is valid.
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

/// Non-escaping compiler-owned loads/stores may forward a scheduler origin without requiring full SSA promotion.
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

/// One unrelated incoming store must invalidate a cell even when another store has valid scheduler provenance.
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

/// Casts that change pointee, address space, or write capability cannot establish closed-cell aliases.
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
    let different_pointee_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, true).into();
    for result_type in [shared_pointer_ty, different_pointee_ty] {
        let cast = pointer_cast(&mut ctx, block, pointer, result_type);
        let alias = cast.deref(&ctx).get_result(0);
        assert!(!is_local_cell_pointer_alias_cast(&ctx, cast));
        assert!(super::provenance::local_cell(&ctx, alias).is_none());
    }
    let read_only_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, false).into();
    let read_only = pointer_cast(&mut ctx, block, pointer, read_only_type);
    let read_only_alias = read_only.deref(&ctx).get_result(0);
    assert!(is_local_cell_pointer_alias_cast(&ctx, read_only));
    assert!(super::provenance::local_cell(&ctx, read_only_alias).is_some());

    let mutable_type: TypeHandle = MirPtrType::get_generic(&mut ctx, u64_ty, true).into();
    let escalation = pointer_cast(&mut ctx, block, read_only_alias, mutable_type);
    let escalated_alias = escalation.deref(&ctx).get_result(0);
    assert!(!is_local_cell_pointer_alias_cast(&ctx, escalation));
    assert!(super::provenance::local_cell(&ctx, escalated_alias).is_none());
    assert!(!local_cell_is_closed(&ctx, pointer));
}

/// Semantic shared carriers may survive the exact generic/shared round-trip while other address-space casts are rejected.
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

/// Matching scalar types cannot substitute a consumer-origin cursor for producer lifecycle state.
#[test]
fn rejects_pipeline_lifecycle_state_from_the_wrong_role_root() {
    let mut ctx = Context::new();
    let (module, block) = module_top(&mut ctx);
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let barrier_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
    let barrier_base = undef(&mut ctx, block, barrier_ty);
    let make = CuteTmaLoadPipelineMakeOp::new(&mut ctx, barrier_base, 3, 8, 8, 8).get_operation();
    make.insert_at_back(block, &ctx);
    let pipeline = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(&ctx);
    let producer =
        CutePipelineStateNewOp::new(&mut ctx, CutePipelineStateAttr::producer(3)).get_operation();
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

/// Producer drain must follow the cursor's acquire/expect/advance chain in the CFG.
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

/// A correctly typed barrier still needs provenance from the matching transaction-expectation operation.
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

/// Shared base and capacity must share one overlay origin, even when both overlays have identical types.
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
    let first =
        CuteSmemTensorOverlayOp::new(&mut ctx, first_base, first_capacity, u64_ty).get_operation();
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

/// One valid semantic use does not permit another use to escape the overlay's closed carrier flow.
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
    let overlay = CuteSmemTensorOverlayOp::new(&mut ctx, base, capacity, u64_ty).get_operation();
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

/// A plausible traversal order cannot replace a reachable writer-to-issuer synchronization path.
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

    let error = require_epilogue_cfg_links(&ctx, acquire, reusable, ready, half_zero, commit, tail)
        .unwrap_err()
        .to_string();
    assert!(error.contains("epilogue CFG"), "{error}");
}

/// The expectation promise must equal the sum of its direct TMA users' static tile bytes.
#[test]
fn rejects_expect_tx_when_direct_tma_bytes_do_not_match() {
    let mut ctx = Context::new();
    let (module, block) = module_top(&mut ctx);
    let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let barrier_ty: TypeHandle =
        MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
    let barrier_base = undef(&mut ctx, block, barrier_ty);
    let make = CuteTmaLoadPipelineMakeOp::new(&mut ctx, barrier_base, 3, 8, 8, 8).get_operation();
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
