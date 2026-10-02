/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    op_interfaces::CastOpInterface,
    ops::{AddressOfOp, BitcastOp, BrOp, CondBrOp, ReturnOp, UndefOp},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IdentifierAttr,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_llvm_control_flow_verify() {
    let mut ctx = Context::new();

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // 1. BrOp
    let target_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let src_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg_val = src_block.deref(&ctx).get_argument(0);

    // Valid creation using wrapper (BrOp::new is available in ops.rs)
    let br_op = BrOp::new(&mut ctx, target_block, vec![arg_val]);
    assert!(br_op.verify(&ctx).is_ok(), "Valid BrOp");

    // Invalid BrOp: Operand count mismatch (manual construction)
    let br_op_bad = Operation::new(
        &mut ctx,
        BrOp::get_concrete_op_info(),
        vec![],
        vec![], // Missing operand
        vec![target_block],
        0,
    );
    assert!(
        br_op_bad.verify(&ctx).is_err(),
        "BrOp missing operand should fail verification"
    );

    // 2. CondBrOp
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let cond_block = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let cond_val = cond_block.deref(&ctx).get_argument(0);
    let val = cond_block.deref(&ctx).get_argument(1);

    let cond_br = CondBrOp::new(
        &mut ctx,
        cond_val,
        true_block,
        vec![val],
        false_block,
        vec![],
    );
    assert!(cond_br.verify(&ctx).is_ok(), "Valid CondBrOp");

    // 3. ReturnOp
    let ret_op = ReturnOp::new(&mut ctx, Some(val));
    assert!(ret_op.verify(&ctx).is_ok(), "Valid ReturnOp");
}

#[test]
fn test_llvm_arithmetic_verify() {
    let mut ctx = Context::new();

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let _block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);

    // Helper to check integer binops
    let check_int_bin_op = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                            name: &str,
                            needs_flags: bool| {
        let mut context = Context::new();
        let ty = IntegerType::get(&context, 32, Signedness::Signless);
        let blk = BasicBlock::new(&mut context, None, vec![ty.into(), ty.into()]);
        let l = blk.deref(&context).get_argument(0);
        let r = blk.deref(&context).get_argument(1);

        let op = Operation::new(&mut context, opid, vec![ty.into()], vec![l, r], vec![], 0);

        if needs_flags {
            let flags = llvm_export::attributes::IntegerOverflowFlagsAttr::default();
            op.deref_mut(&context).attributes.set(
                llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
                flags,
            );
        }

        assert!(op.verify(&context).is_ok(), "Valid {}", name);

        // Mismatch types
        let ty64 = IntegerType::get(&context, 64, Signedness::Signless);
        let blk64 = BasicBlock::new(&mut context, None, vec![ty64.into()]);
        let l64 = blk64.deref(&context).get_argument(0);

        let op_bad = Operation::new(&mut context, opid, vec![ty.into()], vec![l, l64], vec![], 0);

        if needs_flags {
            let flags = llvm_export::attributes::IntegerOverflowFlagsAttr::default();
            op_bad.deref_mut(&context).attributes.set(
                llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
                flags,
            );
        }

        assert!(op_bad.verify(&context).is_err(), "Type mismatch {}", name);
    };

    check_int_bin_op(
        llvm_export::ops::AddOp::get_concrete_op_info(),
        "AddOp",
        true,
    );
    check_int_bin_op(
        llvm_export::ops::SubOp::get_concrete_op_info(),
        "SubOp",
        true,
    );
    check_int_bin_op(
        llvm_export::ops::MulOp::get_concrete_op_info(),
        "MulOp",
        true,
    );
    check_int_bin_op(
        llvm_export::ops::ShlOp::get_concrete_op_info(),
        "ShlOp",
        true,
    );
    check_int_bin_op(
        llvm_export::ops::UDivOp::get_concrete_op_info(),
        "UDivOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::SDivOp::get_concrete_op_info(),
        "SDivOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::URemOp::get_concrete_op_info(),
        "URemOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::SRemOp::get_concrete_op_info(),
        "SRemOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::AndOp::get_concrete_op_info(),
        "AndOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::OrOp::get_concrete_op_info(),
        "OrOp",
        false,
    );
    check_int_bin_op(
        llvm_export::ops::XorOp::get_concrete_op_info(),
        "XorOp",
        false,
    );
}

#[test]
fn test_llvm_misc_verify() {
    let mut ctx = Context::new();

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let ptr_ty = llvm_export::types::PointerType::get(&ctx, 0);

    // 1. BitcastOp
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg = block.deref(&ctx).get_argument(0);

    // Valid case using interface
    let bitcast_op = BitcastOp::new(&mut ctx, arg, i64_ty.into());
    assert!(bitcast_op.verify(&ctx).is_ok(), "Valid BitcastOp");

    // Invalid case: Missing operand
    let op_bad = Operation::new(
        &mut ctx,
        BitcastOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![], // Missing operand
        vec![],
        0,
    );
    assert!(op_bad.verify(&ctx).is_err(), "BitcastOp missing operand");

    // 2. UndefOp
    let undef_op = UndefOp::new(&mut ctx, i32_ty.into());
    assert!(undef_op.verify(&ctx).is_ok(), "Valid UndefOp");

    let op_undef_bad = Operation::new(
        &mut ctx,
        UndefOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![arg], // Extra operand
        vec![],
        0,
    );
    assert!(op_undef_bad.verify(&ctx).is_err(), "UndefOp with operand");

    // 3. AddressOfOp
    let global_name = IdentifierAttr::new("my_global".try_into().unwrap());
    // Use generic address space (0) for the test
    let addr_op = AddressOfOp::new(&mut ctx, "my_global".try_into().unwrap(), 0);
    assert!(addr_op.verify(&ctx).is_ok(), "Valid AddressOfOp");

    // Get the key used for global name
    let key = addr_op
        .get_operation()
        .deref(&ctx)
        .attributes
        .0
        .keys()
        .next()
        .unwrap()
        .clone();

    // Missing attribute
    let op_addr_no_attr = Operation::new(
        &mut ctx,
        AddressOfOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        op_addr_no_attr.verify(&ctx).is_err(),
        "AddressOfOp missing global_name"
    );

    // Wrong result type
    let op_addr_bad_res = Operation::new(
        &mut ctx,
        AddressOfOp::get_concrete_op_info(),
        vec![i32_ty.into()], // Not pointer
        vec![],
        vec![],
        0,
    );
    op_addr_bad_res
        .deref_mut(&ctx)
        .attributes
        .set(key, global_name);
    assert!(
        op_addr_bad_res.verify(&ctx).is_err(),
        "AddressOfOp result not pointer"
    );
}
