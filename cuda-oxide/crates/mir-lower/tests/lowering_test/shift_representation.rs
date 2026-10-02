/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer shift lowering: the count-width conversion, the `bit_width - 1`
//! count mask, and the operand-type agreement `llvm.shl` / `llvm.lshr` /
//! `llvm.ashr` require.
//!
//! #1328 was an invalid same-width `llvm.trunc` on the shift count: a
//! `redux.sync` result reached the shift as `ui32` while the literal count was
//! signless, so `convert_shift` took its "types differ, so cast the count"
//! path and narrowed a 32-bit value to 32 bits. LLVM rejects that (`trunc`
//! needs a strictly smaller result) and the whole device module failed
//! verification.
//!
//! Reduction results are signless now, which is what keeps that state out of
//! the current pipeline, but the shift shape itself had no regression: the
//! redux tests cover returning a reduction directly, and no shift test runs
//! an intrinsic result into a shift. `redux_max_u32_shift_lowers` fails on the
//! tree the report was filed against and passes here.
//!
//! Every case asserts the whole chain rather than the first op: the count
//! conversion, the mask, and the shift all have to agree on one type, because
//! `llvm.and` and the shift itself are `SameOperandsAndResultType` ops.
//! Repairing only the count cast leaves the mask mismatched.

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

use crate::common::{lowered_kernel_body, make_test_ctx};

const KERNEL: &str = "kernel_func";

fn integer(ctx: &mut Context, width: u32, signedness: Signedness) -> TypeHandle {
    IntegerType::get(ctx, width, signedness).into()
}

/// A MIR function `kernel_func(args) -> (ret)` with one entry block.
fn build_returning_kernel(
    ctx: &mut Context,
    arg_tys: Vec<TypeHandle>,
    ret_tys: Vec<TypeHandle>,
) -> (Ptr<Operation>, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "shift_representation".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(ctx, arg_tys.clone(), ret_tys);
    let func_ptr = Operation::new(
        ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(ctx, func_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, KERNEL.try_into().unwrap());

    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, arg_tys);
    entry.insert_at_back(region, ctx);

    let module_block = module_ptr
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .unwrap();
    func.get_operation().insert_at_back(module_block, ctx);

    (module_ptr, entry)
}

fn mir_integer_constant(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    width: u32,
    signedness: Signedness,
    value: u32,
) -> Value {
    let ty = IntegerType::get(ctx, width, signedness);
    let constant = Operation::new(
        ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(constant).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u32(value, NonZeroUsize::new(width as usize).unwrap()),
        ),
    );
    constant.insert_at_back(block, ctx);
    constant.deref(ctx).get_result(0)
}

fn insert_shift(ctx: &mut Context, block: Ptr<BasicBlock>, op: Ptr<Operation>) -> Value {
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build a `mir.shl` whose result carries `value_ty`, as Rust's `Shl` impls do.
fn mir_shl(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    value_ty: TypeHandle,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = Operation::new(
        ctx,
        mir::MirShlOp::get_concrete_op_info(),
        vec![value_ty],
        vec![lhs, rhs],
        vec![],
        0,
    );
    insert_shift(ctx, block, op)
}

fn mir_shr(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    value_ty: TypeHandle,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = Operation::new(
        ctx,
        mir::MirShrOp::get_concrete_op_info(),
        vec![value_ty],
        vec![lhs, rhs],
        vec![],
        0,
    );
    insert_shift(ctx, block, op)
}

fn mir_return(ctx: &mut Context, block: Ptr<BasicBlock>, values: Vec<Value>) {
    let op = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        values,
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
}

/// Lower, verify, and hand back the kernel body for inspection.
fn lower_and_verify(ctx: &mut Context, module_ptr: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    mir_lower::lower_mir_to_llvm(ctx, module_ptr).expect("shift lowering failed");
    module_ptr
        .deref(ctx)
        .verify(ctx)
        .expect("lowered module must verify");
    lowered_kernel_body(ctx, module_ptr)
}

fn count_of<T: Op>(ctx: &Context, body: &[Ptr<Operation>]) -> usize {
    body.iter()
        .filter(|op| Operation::get_op::<T>(**op, ctx).is_some())
        .count()
}

fn find<T: Op>(ctx: &Context, body: &[Ptr<Operation>]) -> T {
    body.iter()
        .find_map(|op| Operation::get_op::<T>(*op, ctx))
        .expect("expected op in lowered kernel body")
}

fn operand_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    op.deref(ctx).operands().map(|v| v.get_type(ctx)).collect()
}

fn result_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    (0..op.deref(ctx).get_num_results())
        .map(|i| op.deref(ctx).get_result(i).get_type(ctx))
        .collect()
}

/// The reported #1328 shape: an unsigned 32-bit warp reduction shifted left by
/// a literal. The reduction result and the count share a width, so the count
/// must reach `llvm.shl` unconverted; a same-width `trunc` here is the bug.
#[test]
fn redux_max_u32_shift_lowers() {
    let mut ctx = make_test_ctx();
    let u32_ty = integer(&mut ctx, 32, Signedness::Unsigned);

    let (module_ptr, block) = build_returning_kernel(&mut ctx, vec![u32_ty, u32_ty], vec![u32_ty]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    let redux = nvvm::ReduxSyncUmaxOp::build(&mut ctx, mask, value);
    redux.insert_at_back(block, &ctx);
    let maximum = redux.deref(&ctx).get_result(0);

    let count = mir_integer_constant(&mut ctx, block, 32, Signedness::Unsigned, 8);
    let shifted = mir_shl(&mut ctx, block, u32_ty, maximum, count);
    mir_return(&mut ctx, block, vec![shifted]);

    let body = lower_and_verify(&mut ctx, module_ptr);

    assert_eq!(
        count_of::<llvm::TruncOp>(&ctx, &body),
        0,
        "no count narrowing"
    );
    assert_eq!(
        count_of::<llvm::ZExtOp>(&ctx, &body),
        0,
        "no count widening"
    );

    let shl = find::<llvm::ShlOp>(&ctx, &body);
    let shl_op = shl.get_operation();
    let operands = operand_types(&ctx, shl_op);
    assert_eq!(operands.len(), 2);
    assert_eq!(
        operands[0],
        operands[1],
        "llvm.shl requires one operand type, got {} and {}",
        operands[0].disp(&ctx),
        operands[1].disp(&ctx)
    );
    assert_eq!(
        result_types(&ctx, shl_op),
        vec![operands[0]],
        "llvm.shl result must match its operands"
    );
}

/// The count mask is part of the same chain: `llvm.and` is a
/// `SameOperandsAndResultType` op, so a mask built in a different
/// representation than the count fails verification just as the shift would.
#[test]
fn shift_count_mask_shares_the_shift_operand_type() {
    let mut ctx = make_test_ctx();
    let u32_ty = integer(&mut ctx, 32, Signedness::Unsigned);

    let (module_ptr, block) = build_returning_kernel(&mut ctx, vec![u32_ty, u32_ty], vec![u32_ty]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    let redux = nvvm::ReduxSyncUmaxOp::build(&mut ctx, mask, value);
    redux.insert_at_back(block, &ctx);
    let maximum = redux.deref(&ctx).get_result(0);

    let count = mir_integer_constant(&mut ctx, block, 32, Signedness::Unsigned, 8);
    let shifted = mir_shl(&mut ctx, block, u32_ty, maximum, count);
    mir_return(&mut ctx, block, vec![shifted]);

    let body = lower_and_verify(&mut ctx, module_ptr);

    let and = find::<llvm::AndOp>(&ctx, &body);
    let and_op = and.get_operation();
    let and_operands = operand_types(&ctx, and_op);
    assert_eq!(and_operands.len(), 2, "one mask and per shift");
    assert_eq!(
        and_operands[0], and_operands[1],
        "llvm.and requires one operand type"
    );
    assert_eq!(result_types(&ctx, and_op), vec![and_operands[0]]);

    // The masked count is what the shift consumes, so both chains agree on one
    // type end to end.
    let shl = find::<llvm::ShlOp>(&ctx, &body);
    assert_eq!(
        shl.get_operation().deref(&ctx).get_operand(1),
        and_op.deref(&ctx).get_result(0),
        "the shift consumes the masked count"
    );
    assert_eq!(operand_types(&ctx, shl.get_operation())[1], and_operands[0]);
}

/// A count narrower than the value widens; a count wider than the value
/// narrows. Both land the count on the value's type without a representation
/// change, and the mask follows.
#[test]
fn shift_count_widths_convert_in_the_documented_direction() {
    for (value_width, count_width, expects_zext) in [(32u32, 8u32, true), (8, 32, false)] {
        let mut ctx = make_test_ctx();
        let value_ty = integer(&mut ctx, value_width, Signedness::Unsigned);
        let count_ty = integer(&mut ctx, count_width, Signedness::Unsigned);

        let (module_ptr, block) =
            build_returning_kernel(&mut ctx, vec![value_ty, count_ty], vec![value_ty]);
        let value = block.deref(&ctx).get_argument(0);
        let count = block.deref(&ctx).get_argument(1);

        let shifted = mir_shl(&mut ctx, block, value_ty, value, count);
        mir_return(&mut ctx, block, vec![shifted]);

        let body = lower_and_verify(&mut ctx, module_ptr);

        let case = format!("value {value_width}, count {count_width}");
        assert_eq!(
            count_of::<llvm::ZExtOp>(&ctx, &body),
            usize::from(expects_zext),
            "{case}: zero-extension count"
        );
        assert_eq!(
            count_of::<llvm::TruncOp>(&ctx, &body),
            usize::from(!expects_zext),
            "{case}: truncating count"
        );
        assert_eq!(
            count_of::<llvm::BitcastOp>(&ctx, &body),
            0,
            "{case}: a width change is never a representation change"
        );

        // The count lands on the value's width in the lowered signless
        // representation, which is the type the whole chain has to share.
        let shl = find::<llvm::ShlOp>(&ctx, &body);
        let operands = operand_types(&ctx, shl.get_operation());
        assert_eq!(operands[0], operands[1], "{case}: one shift operand type");
        let lowered_width = operands[0]
            .deref(&ctx)
            .downcast_ref::<IntegerType>()
            .expect("shift operands are integers")
            .width();
        assert_eq!(
            lowered_width, value_width,
            "{case}: count lands on value width"
        );
    }
}

/// Signedness of a right shift is a property of the original MIR operation,
/// not of the lowered signless type: a signed value takes `ashr`, an unsigned
/// one takes `lshr`, and neither skips the count mask.
#[test]
fn right_shift_selects_arithmetic_or_logical_from_mir_signedness() {
    for (signedness, expects_arithmetic) in
        [(Signedness::Signed, true), (Signedness::Unsigned, false)]
    {
        let mut ctx = make_test_ctx();
        let value_ty = integer(&mut ctx, 32, signedness);

        let (module_ptr, block) =
            build_returning_kernel(&mut ctx, vec![value_ty, value_ty], vec![value_ty]);
        let value = block.deref(&ctx).get_argument(0);
        let count = block.deref(&ctx).get_argument(1);

        let shifted = mir_shr(&mut ctx, block, value_ty, value, count);
        mir_return(&mut ctx, block, vec![shifted]);

        let body = lower_and_verify(&mut ctx, module_ptr);

        let case = format!("{signedness:?} value");
        assert_eq!(
            count_of::<llvm::AShrOp>(&ctx, &body),
            usize::from(expects_arithmetic),
            "{case}: arithmetic form"
        );
        assert_eq!(
            count_of::<llvm::LShrOp>(&ctx, &body),
            usize::from(!expects_arithmetic),
            "{case}: logical form"
        );
        assert_eq!(count_of::<llvm::ZExtOp>(&ctx, &body), 0, "{case}");
        assert_eq!(count_of::<llvm::TruncOp>(&ctx, &body), 0, "{case}");
    }
}

/// An equal-width count is left alone (no cast at all) but is still masked with
/// `bit_width - 1`, and the shift consumes the mask rather than the raw count.
#[test]
fn matching_count_width_is_masked_without_a_cast() {
    let mut ctx = make_test_ctx();
    let u8_ty = integer(&mut ctx, 8, Signedness::Unsigned);

    let (module_ptr, block) = build_returning_kernel(&mut ctx, vec![u8_ty, u8_ty], vec![u8_ty]);
    let value = block.deref(&ctx).get_argument(0);
    let count = block.deref(&ctx).get_argument(1);

    let shifted = mir_shl(&mut ctx, block, u8_ty, value, count);
    mir_return(&mut ctx, block, vec![shifted]);

    let body = lower_and_verify(&mut ctx, module_ptr);

    assert_eq!(count_of::<llvm::ZExtOp>(&ctx, &body), 0);
    assert_eq!(count_of::<llvm::TruncOp>(&ctx, &body), 0);
    assert_eq!(count_of::<llvm::BitcastOp>(&ctx, &body), 0);

    let and = find::<llvm::AndOp>(&ctx, &body);
    let and_op = and.get_operation();
    let mask_operand = and_op.deref(&ctx).get_operand(1);
    let constant = mask_operand
        .defining_op()
        .and_then(|op| Operation::get_op::<llvm::ConstantOp>(op, &ctx))
        .expect("the mask operand is a constant");
    let attr = constant
        .get_attr_builtin_constant_value(&ctx)
        .expect("the mask constant carries a value");
    let integer_attr = (&**attr as &dyn pliron::attribute::Attribute)
        .downcast_ref::<IntegerAttr>()
        .expect("the mask constant is an integer");
    assert_eq!(
        integer_attr.value(),
        APInt::from_u32(7, NonZeroUsize::new(8).unwrap()),
        "the mask is bit_width - 1"
    );

    let shl = find::<llvm::ShlOp>(&ctx, &body);
    assert_eq!(
        shl.get_operation().deref(&ctx).get_operand(1),
        and_op.deref(&ctx).get_result(0)
    );
}
