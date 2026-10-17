/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops::{MirAssertOp, MirCondBranchOp, MirFuncOp, MirGotoOp, MirReturnOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{OperandSegmentSizesAttr, StringAttr, TypeAttr},
        op_interfaces::{
            ATTR_KEY_OPERAND_SEGMENT_SIZES, BranchOpInterface, OperandSegmentInterface,
        },
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
};

#[test]
fn test_mir_control_flow_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // 1. MirGotoOp
    let target_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let src_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg_val = src_block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![arg_val],
        vec![target_block],
        0,
    );
    let goto_op = MirGotoOp::new(op);
    assert!(goto_op.verify(&ctx).is_ok(), "Valid Goto");

    let op_bad = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    assert!(
        MirGotoOp::new(op_bad).verify(&ctx).is_err(),
        "Goto missing operand"
    );

    // 2. MirCondBranchOp
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let cond_block = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let cond_val = cond_block.deref(&ctx).get_argument(0);
    let val = cond_block.deref(&ctx).get_argument(1);

    let (cond_flat, cond_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![val], vec![]]);
    let op_cond = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond).set_operand_segment_sizes(&ctx, cond_sizes);
    let cond_br = MirCondBranchOp::new(op_cond);
    assert!(cond_br.verify(&ctx).is_ok(), "Valid CondBranch");

    let (cond_bad_flat, cond_bad_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![], vec![]]);
    let op_cond_bad = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_bad_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond_bad).set_operand_segment_sizes(&ctx, cond_bad_sizes);
    assert!(
        MirCondBranchOp::new(op_cond_bad).verify(&ctx).is_err(),
        "CondBranch missing operand"
    );

    // 3. MirReturnOp
    let func_ty = FunctionType::get(&ctx, vec![], vec![i32_ty.into()]);
    let func_ty_attr = TypeAttr::new(func_ty.into());

    let func_op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    let entry_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let ret_val = entry_block.deref(&ctx).get_argument(0);

    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    entry_block.insert_at_front(region, &ctx);

    let ret_op = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![ret_val],
        vec![],
        0,
    );
    ret_op.insert_at_back(entry_block, &ctx);

    let mir_ret = MirReturnOp::new(ret_op);
    assert!(mir_ret.verify(&ctx).is_ok(), "Valid Return");

    let f32_ty = FP32Type::get(&ctx);
    let f32_block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    let f32_val = f32_block.deref(&ctx).get_argument(0);

    let ret_op_bad = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![f32_val],
        vec![],
        0,
    );
    ret_op_bad.insert_at_back(entry_block, &ctx);

    let mir_ret_bad = MirReturnOp::new(ret_op_bad);
    assert!(mir_ret_bad.verify(&ctx).is_err(), "Return type mismatch");

    // 4. MirAssertOp
    let assert_op = MirAssertOp::new(&mut ctx, cond_val);
    assert!(assert_op.verify(&ctx).is_ok(), "Valid Assert");
    assert!(
        MirAssertOp::new(&mut ctx, val).verify(&ctx).is_err(),
        "Assert cond type mismatch"
    );

    let assert_succ = BasicBlock::new(&mut ctx, None, vec![]);
    for (operands, successors) in [
        (vec![], vec![]),
        (vec![cond_val, cond_val], vec![]),
        (vec![cond_val], vec![assert_succ]),
    ] {
        let malformed = Operation::new(
            &mut ctx,
            MirAssertOp::get_concrete_op_info(),
            vec![],
            operands,
            successors,
            0,
        );
        assert!(
            Operation::get_op::<MirAssertOp>(malformed, &ctx)
                .unwrap()
                .verify(&ctx)
                .is_err(),
            "Assertions must have exactly one condition and no successors"
        );
    }
}

/// Conditional branches must reject malformed segment metadata before indexing it.
#[test]
fn test_mir_cond_branch_rejects_malformed_segments() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let source = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let condition = source.deref(&ctx).get_argument(0);
    let value = source.deref(&ctx).get_argument(1);
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let branch = MirCondBranchOp::new(Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        vec![condition, value],
        vec![true_block, false_block],
        0,
    ));

    assert!(branch.verify(&ctx).is_err(), "Missing segment attribute");
    assert!(branch.verify_interfaces(&ctx).is_err());
    branch.get_operation().deref_mut(&ctx).attributes.set(
        ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
        StringAttr::new("wrong attribute type".to_owned()),
    );
    assert!(branch.verify(&ctx).is_err(), "Wrong segment attribute type");
    assert!(branch.verify_interfaces(&ctx).is_err());

    for sizes in [
        vec![],
        vec![2],
        vec![1, 1],
        vec![1, 1, 0, 0],
        vec![1, 0, 0],
        vec![0, 2, 0],
        vec![2, 0, 0],
        vec![1, 0, 1],
    ] {
        branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(sizes.clone()));
        assert!(branch.verify(&ctx).is_err(), "Accepted malformed {sizes:?}");
        assert!(
            branch.verify_interfaces(&ctx).is_err(),
            "Interfaces accepted {sizes:?}"
        );
    }

    branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(vec![1, 1, 0]));
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
}

/// Segment totals must not wrap to the operand count or panic during verification.
#[test]
fn test_mir_cond_branch_rejects_overflowing_segment_totals() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let source = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let condition = source.deref(&ctx).get_argument(0);
    let value = source.deref(&ctx).get_argument(1);
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let branch = MirCondBranchOp::new(Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        vec![condition, value],
        vec![true_block, false_block],
        0,
    ));

    // Both totals equal two modulo 2^32, matching the actual operand count.
    for sizes in [vec![1, u32::MAX, 2], vec![u32::MAX, 1, 2]] {
        branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(sizes));
        assert!(branch.verify(&ctx).is_err());
        assert!(branch.verify_successor_operand_layout(&ctx).is_err());
        assert!(<MirCondBranchOp as OperandSegmentInterface>::verify(&branch, &ctx).is_err());
        assert!(branch.verify_interfaces(&ctx).is_err());
    }
}

/// Invalid successor counts must be rejected without asking for an invalid range.
#[test]
fn test_mir_branches_reject_wrong_successor_counts() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let source = BasicBlock::new(&mut ctx, None, vec![i1_ty.into()]);
    let condition = source.deref(&ctx).get_argument(0);
    let target = BasicBlock::new(&mut ctx, None, vec![]);

    for count in [0, 2] {
        let branch = MirGotoOp::new(Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![target; count],
            0,
        ));
        assert!(branch.verify(&ctx).is_err());
        assert!(branch.verify_interfaces(&ctx).is_err());
    }
    for count in [0, 1, 3] {
        let branch = MirCondBranchOp::new(Operation::new(
            &mut ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![target; count],
            0,
        ));
        branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(vec![1, 0, 0]));
        assert!(branch.verify(&ctx).is_err());
        assert!(branch.verify_interfaces(&ctx).is_err());
    }
}

/// Range-based mutation must preserve the condition and the other successor's operands.
#[test]
fn test_mir_cond_branch_successor_operand_mutation() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let f32_ty = FP32Type::get(&ctx);
    let source = BasicBlock::new(
        &mut ctx,
        None,
        vec![i1_ty.into(), i32_ty.into(), i32_ty.into(), f32_ty.into()],
    );
    let condition = source.deref(&ctx).get_argument(0);
    let first = source.deref(&ctx).get_argument(1);
    let replacement = source.deref(&ctx).get_argument(2);
    let second = source.deref(&ctx).get_argument(3);
    let true_block = BasicBlock::new(&mut ctx, None, vec![]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let branch = MirCondBranchOp::new(Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        vec![condition],
        vec![true_block, false_block],
        0,
    ));
    branch.set_operand_segment_sizes(&ctx, OperandSegmentSizesAttr(vec![1, 0, 0]));
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
    assert_eq!(branch.successor_operand_range(&ctx, 0), 1..1);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 1..1);

    // Add false first: inserting true later must move, rather than overwrite, false.
    assert_eq!(branch.add_successor_operand(&mut ctx, 1, second), 0);
    BasicBlock::push_argument(false_block, &ctx, f32_ty.into());
    assert_eq!(branch.add_successor_operand(&mut ctx, 0, first), 0);
    BasicBlock::push_argument(true_block, &ctx, i32_ty.into());
    assert_eq!(branch.successor_operand_range(&ctx, 0), 1..2);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 2..3);
    assert_eq!(branch.successor_operands(&ctx, 0), vec![first]);
    assert_eq!(branch.successor_operands(&ctx, 1), vec![second]);
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());

    branch.set_successor_operand(&ctx, 0, 0, second);
    assert!(branch.verify(&ctx).is_err(), "Wrong true argument type");
    assert!(branch.verify_interfaces(&ctx).is_err());
    branch.set_successor_operand(&ctx, 0, 0, replacement);
    branch.set_successor_operand(&ctx, 1, 0, first);
    assert!(branch.verify(&ctx).is_err(), "Wrong false argument type");
    assert!(branch.verify_interfaces(&ctx).is_err());
    branch.set_successor_operand(&ctx, 1, 0, second);
    assert_eq!(branch.successor_operands(&ctx, 0), vec![replacement]);
    assert_eq!(branch.successor_operands(&ctx, 1), vec![second]);
    assert_eq!(branch.get_operation().deref(&ctx).get_operand(0), condition);

    assert_eq!(branch.remove_successor_operand(&mut ctx, 0, 0), replacement);
    BasicBlock::pop_argument(true_block, &ctx);
    assert_eq!(branch.successor_operand_range(&ctx, 1), 1..2);
    assert_eq!(branch.successor_operands(&ctx, 1), vec![second]);
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
    assert_eq!(branch.remove_successor_operand(&mut ctx, 1, 0), second);
    BasicBlock::pop_argument(false_block, &ctx);
    assert_eq!(
        branch
            .get_operation()
            .deref(&ctx)
            .operands()
            .collect::<Vec<_>>(),
        vec![condition]
    );
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
}

/// Goto ranges must support empty and nonempty successors and type-checked mutation.
#[test]
fn test_mir_goto_successor_operand_mutation() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let f32_ty = FP32Type::get(&ctx);
    let source = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i32_ty.into(), f32_ty.into()],
    );
    let first = source.deref(&ctx).get_argument(0);
    let replacement = source.deref(&ctx).get_argument(1);
    let wrong_type = source.deref(&ctx).get_argument(2);
    let target = BasicBlock::new(&mut ctx, None, vec![]);
    let branch = MirGotoOp::new(Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target],
        0,
    ));
    assert_eq!(branch.successor_operand_range(&ctx, 0), 0..0);
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
    assert_eq!(branch.add_successor_operand(&mut ctx, 0, first), 0);
    assert!(
        branch.verify(&ctx).is_err(),
        "Successor argument is missing"
    );
    BasicBlock::push_argument(target, &ctx, i32_ty.into());
    assert_eq!(branch.successor_operand_range(&ctx, 0), 0..1);
    branch.set_successor_operand(&ctx, 0, 0, wrong_type);
    assert!(branch.verify(&ctx).is_err());
    assert!(branch.verify_interfaces(&ctx).is_err());
    branch.set_successor_operand(&ctx, 0, 0, replacement);
    assert_eq!(branch.successor_operands(&ctx, 0), vec![replacement]);
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
    assert_eq!(branch.remove_successor_operand(&mut ctx, 0, 0), replacement);
    BasicBlock::pop_argument(target, &ctx);
    assert!(branch.verify(&ctx).is_ok());
    assert!(branch.verify_interfaces(&ctx).is_ok());
}
