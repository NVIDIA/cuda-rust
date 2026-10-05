/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for `Module::verify_dominance()`.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

// =========================================================================
// Helpers
// =========================================================================

/// Build a module with a single entry function containing the given ops.
fn build_kernel(
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build_body(&mut module, block_id, &args);
    // Append return.
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

fn i32_ty() -> Type {
    Type::Scalar(ScalarType::I32)
}

// =========================================================================
// Valid IR should pass
// =========================================================================

#[test]
fn valid_simple_ops() {
    let module = build_kernel("valid_simple", &[i32_ty(), i32_ty()], |m, blk, args| {
        // %2 = addi %0, %1
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(i32_ty())
            .build(m);
        append_op(m, blk, add);
    });
    module.verify_dominance().expect("valid IR should pass");
}

#[test]
fn valid_nested_region_sees_parent() {
    // Parent scope values are visible inside for-loop body.
    let module = build_kernel("valid_nested", &[i32_ty()], |m, blk, args| {
        let parent_val = args[0];

        // Build for-loop: for %iv = 0 to parent_val step 1
        let lb = {
            let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                .attr("value", Attribute::Integer(0, i32_ty()))
                .result(i32_ty())
                .build(m);
            append_op(m, blk, op);
            res[0]
        };
        let step = {
            let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                .attr("value", Attribute::Integer(1, i32_ty()))
                .result(i32_ty())
                .build(m);
            append_op(m, blk, op);
            res[0]
        };

        // Body block with induction variable
        let (body_region, body_blk, body_args) = build_single_block_region(m, &[i32_ty()]);

        // Inside body: use parent_val (from outer scope) + body_args[0] (iv)
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(parent_val)
            .operand(body_args[0])
            .result(i32_ty())
            .build(m);
        append_op(m, body_blk, add);

        // Yield from body
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(m);
        append_op(m, body_blk, yld);

        // For op
        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(parent_val) // ub
            .operand(step)
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    module
        .verify_dominance()
        .expect("nested region with parent values should pass");
}

#[test]
fn valid_if_with_results() {
    let module = build_kernel("valid_if", &[i32_ty()], |m, blk, args| {
        // condition: constant true (i1)
        let i1_ty = Type::Scalar(ScalarType::I1);
        let (cond_op, cond_res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
            .attr("value", Attribute::Integer(1, i1_ty.clone()))
            .result(i1_ty)
            .build(m);
        append_op(m, blk, cond_op);

        // then region
        let (then_region, then_blk, _) = build_single_block_region(m, &[]);
        let (add, add_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[0])
            .result(i32_ty())
            .build(m);
        append_op(m, then_blk, add);
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, then_blk, yld);

        // else region
        let (else_region, else_blk, _) = build_single_block_region(m, &[]);
        let (yld2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(args[0]) // parent scope value visible in else
            .build(m);
        append_op(m, else_blk, yld2);

        // if op
        let (if_op, _if_res) = OpBuilder::new(Opcode::If, Location::Unknown)
            .operand(cond_res[0])
            .result(i32_ty())
            .region(then_region)
            .region(else_region)
            .build(m);
        append_op(m, blk, if_op);
    });
    module.verify_dominance().expect("valid if should pass");
}

// =========================================================================
// Invalid IR should fail
// =========================================================================

#[test]
fn invalid_use_before_def() {
    // Build two ops but append them in reverse order, so the first op in the
    // block uses a value produced by the second op (use-before-def).
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);

    // Build the producer first (allocates its result value) but DON'T append yet.
    let (producer_op, produced) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr("value", Attribute::Integer(42, i32_ty()))
        .result(i32_ty())
        .build(&mut module);

    // Build the consumer that uses the producer's result value.
    let (consumer_op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
        .operand(produced[0])
        .operand(produced[0])
        .result(i32_ty())
        .build(&mut module);

    // Append in WRONG order: consumer first, then producer.
    append_op(&mut module, block_id, consumer_op);
    append_op(&mut module, block_id, producer_op);

    // Return
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);

    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("bad".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);

    let err = module
        .verify_dominance()
        .expect_err("use-before-def should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("does not dominate"),
        "error should mention dominance: {msg}"
    );
}

#[test]
fn invalid_cross_region_not_visible() {
    // A value defined in a nested region should NOT be visible in the parent.
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);

    // Build an if with a then-region that defines a value
    let i1_ty = Type::Scalar(ScalarType::I1);
    let (cond_op, cond_res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr("value", Attribute::Integer(1, i1_ty.clone()))
        .result(i1_ty)
        .build(&mut module);
    append_op(&mut module, block_id, cond_op);

    // then region: defines a constant
    let (then_region, then_blk, _) = build_single_block_region(&mut module, &[]);
    let (inner_const, inner_res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr("value", Attribute::Integer(99, i32_ty()))
        .result(i32_ty())
        .build(&mut module);
    append_op(&mut module, then_blk, inner_const);
    let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(&mut module);
    append_op(&mut module, then_blk, yld);

    // else region (empty body)
    let (else_region, else_blk, _) = build_single_block_region(&mut module, &[]);
    let (yld2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(&mut module);
    append_op(&mut module, else_blk, yld2);

    // if op (no results)
    let (if_op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
        .operand(cond_res[0])
        .region(then_region)
        .region(else_region)
        .build(&mut module);
    append_op(&mut module, block_id, if_op);

    // Now try to use inner_res[0] in the parent scope — this is invalid!
    let (bad_op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
        .operand(inner_res[0])
        .operand(inner_res[0])
        .result(i32_ty())
        .build(&mut module);
    append_op(&mut module, block_id, bad_op);

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);

    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("bad_cross_region".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);

    let err = module
        .verify_dominance()
        .expect_err("cross-region ref should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("does not dominate"),
        "error should mention dominance: {msg}"
    );
}

#[test]
fn valid_empty_function() {
    let module = build_kernel("empty", &[], |_, _, _| {});
    module
        .verify_dominance()
        .expect("empty function should pass");
}
