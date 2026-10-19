/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    ops::{MirGlobalAllocOp, MirLoadOp, MirPtrOffsetOp, MirStoreOp},
    types::{MirPointerKind, MirPtrType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    opts::mem2reg::{AllocInfo, PromotableOpInterface, PromotableOpKind},
};

#[test]
fn test_mir_load_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);

    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    let mir_load = MirLoadOp::new(op);
    assert!(mir_load.verify(&ctx).is_ok(), "Valid MirLoadOp");

    let block_i32 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let i32_val = block_i32.deref(&ctx).get_argument(0);

    let op_fail_operand = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_val],
        vec![],
        0,
    );
    let mir_load_fail_operand = MirLoadOp::new(op_fail_operand);
    assert!(
        mir_load_fail_operand.verify(&ctx).is_err(),
        "MirLoadOp non-ptr operand"
    );

    let f32_ty = FP32Type::get(&ctx);
    let op_fail_res = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    let mir_load_fail_res = MirLoadOp::new(op_fail_res);
    assert!(
        mir_load_fail_res.verify(&ctx).is_err(),
        "MirLoadOp result mismatch"
    );
}

#[test]
fn test_mir_load_volatile_is_not_promotable() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    let mir_load = MirLoadOp::new(op);
    let alloc_info = AllocInfo {
        ptr: ptr_val,
        ty: i32_ty.into(),
    };

    assert!(!mir_load.is_volatile(&ctx));
    assert!(matches!(
        mir_load.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::Load
    ));

    mir_load.set_volatile(&mut ctx, true);

    assert!(mir_load.is_volatile(&ctx));
    assert!(matches!(
        mir_load.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::NonPromotableUse
    ));
}

#[test]
fn test_mir_ptr_offset_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into(), usize_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let idx_val = block.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![ptr_val, idx_val],
        vec![],
        0,
    );
    let offset_op = MirPtrOffsetOp::new(op);
    assert!(offset_op.verify(&ctx).is_ok(), "Valid MirPtrOffsetOp");
    assert!(
        offset_op.is_inbounds(&ctx),
        "ordinary pointer offsets default to inbounds"
    );
    offset_op.set_inbounds(&mut ctx, false);
    assert!(
        !offset_op.is_inbounds(&ctx),
        "wrapping pointer offsets retain their explicit semantics"
    );

    let block2 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), usize_ty.into()]);
    let i32_val = block2.deref(&ctx).get_argument(0);
    let idx_val2 = block2.deref(&ctx).get_argument(1);

    let op_bad_base = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![i32_val, idx_val2],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(op_bad_base).verify(&ctx).is_err());

    let f32_ty = FP32Type::get(&ctx);
    let ptr_f32_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), false);
    let op_bad_res = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_f32_ty.into()],
        vec![ptr_val, idx_val],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(op_bad_res).verify(&ctx).is_err());
}

#[test]
fn test_mir_store_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into(), i32_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    assert!(MirStoreOp::new(op).verify(&ctx).is_ok(), "Valid MirStoreOp");

    // Invalid: store to non-ptr
    let op_bad_ptr = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![val, val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_bad_ptr).verify(&ctx).is_err(),
        "MirStoreOp non-ptr dest"
    );

    // Invalid: type mismatch
    let f32_ty = FP32Type::get(&ctx);
    let block2 = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    let f32_val = block2.deref(&ctx).get_argument(0);
    let op_bad_type = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, f32_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_bad_type).verify(&ctx).is_err(),
        "MirStoreOp type mismatch"
    );
}

#[test]
fn test_mir_store_volatile_is_not_promotable() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into(), i32_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    let mir_store = MirStoreOp::new(op);
    let alloc_info = AllocInfo {
        ptr: ptr_val,
        ty: i32_ty.into(),
    };

    assert!(!mir_store.is_volatile(&ctx));
    match mir_store.promotion_kind(&ctx, &alloc_info) {
        PromotableOpKind::Store(stored) => assert!(stored == val),
        _ => panic!("non-volatile store should be promotable"),
    }

    mir_store.set_volatile(&mut ctx, true);

    assert!(mir_store.is_volatile(&ctx));
    assert!(matches!(
        mir_store.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::NonPromotableUse
    ));
}

#[test]
fn test_mir_global_alloc_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);

    // Helper: build a MirGlobalAllocOp whose result pointer is in `ptr_ty`
    // address space, with valid attributes.
    let build = |ctx: &mut Context, ptr_ty: pliron::r#type::TypedHandle<MirPtrType>| {
        let op = Operation::new(
            ctx,
            MirGlobalAllocOp::get_concrete_op_info(),
            vec![ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(ctx, TypeAttr::new(f32_ty.into()));
        alloc.set_attr_global_key(ctx, StringAttr::new("k".to_string()));
        alloc
    };

    // Global memory (addrspace 1) — the original allowed space.
    let ptr_global = MirPtrType::get_global(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_global).verify(&ctx).is_ok(),
        "global addrspace accepted"
    );

    // Constant memory (addrspace 4) — added for `#[constant]` support.
    let ptr_const = MirPtrType::get_constant(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_const).verify(&ctx).is_ok(),
        "constant addrspace accepted"
    );

    // Shared memory (addrspace 3) — must be rejected.
    let ptr_shared = MirPtrType::get_shared(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_shared).verify(&ctx).is_err(),
        "shared addrspace rejected"
    );

    let ptr_unique_global = MirPtrType::get_with_kind(
        &mut ctx,
        f32_ty.into(),
        true,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::UniqueRef,
    );
    assert!(
        build(&mut ctx, ptr_unique_global).verify(&ctx).is_err(),
        "a storage allocation cannot directly claim UniqueRef"
    );

    // Missing required attributes.
    let no_attrs = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![ptr_global.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirGlobalAllocOp::new(no_attrs).verify(&ctx).is_err(),
        "missing attributes rejected"
    );
}
