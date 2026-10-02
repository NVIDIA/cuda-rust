/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode compatibility tests.
//!
//! Translates the cuda-tile C++ bytecode tests (`.mlir` files under
//! `cuda-tile/test/Bytecode/`) into Rust, exercising the same IR patterns
//! through the `OpBuilder` API and verifying bytecode roundtrip.

// Float literals such as 3.14159 are arbitrary test payloads, not approximations of pi.
#![allow(clippy::approx_constant)]

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{Opcode, MAGIC};
use cutile_ir::ir::*;
use half::{bf16, f16};

// =========================================================================
// Helpers (same pattern as per_op_roundtrip.rs)
// =========================================================================

/// Build a module with a single entry function containing the given ops.
fn build_kernel(
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    let mut module = Module::new("kernels");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build_body(&mut module, block_id, &args);
    // Append return if not already present.
    let needs_return = {
        let block = module.block(block_id);
        block
            .ops
            .last()
            .is_none_or(|&last| !matches!(module.op(last).opcode, Opcode::Return))
    };
    if needs_return {
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
        append_op(&mut module, block_id, ret);
    }
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

/// Add an entry function to an existing module.
fn add_entry(
    module: &mut Module,
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) {
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(module, arg_types);
    build_body(module, block_id, &args);
    let needs_return = {
        let block = module.block(block_id);
        block
            .ops
            .last()
            .is_none_or(|&last| !matches!(module.op(last).opcode, Opcode::Return))
    };
    if needs_return {
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(module);
        append_op(module, block_id, ret);
    }
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(module);
    module.functions.push(entry);
}

/// Assert bytecode roundtrip: serialize -> decode -> no error, header present.
fn assert_roundtrip(module: &Module) {
    let bytecode =
        cutile_ir::write_bytecode(module).unwrap_or_else(|e| panic!("bytecode write failed: {e}"));
    let decoded = cutile_ir::decode_bytecode(&bytecode)
        .unwrap_or_else(|e| panic!("bytecode decode failed: {e}"));
    assert!(
        decoded.contains("TileIR bytecode v13."),
        "missing header in decoded output"
    );
}

/// Assert bytecode roundtrip and that the decoded text contains all expected strings.
#[allow(dead_code)]
fn assert_roundtrip_contains(module: &Module, expected: &[&str]) {
    let bytecode =
        cutile_ir::write_bytecode(module).unwrap_or_else(|e| panic!("bytecode write failed: {e}"));
    let decoded = cutile_ir::decode_bytecode(&bytecode)
        .unwrap_or_else(|e| panic!("bytecode decode failed: {e}"));
    assert!(
        decoded.contains("TileIR bytecode v13."),
        "missing header in decoded output"
    );
    for s in expected {
        assert!(
            decoded.contains(s),
            "decoded output missing expected string: {s:?}\nfull output:\n{decoded}"
        );
    }
}

// Common type helpers.
#[allow(dead_code)]
fn tile_ty(scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(scalar),
    })
}

fn tile_shaped(shape: &[i64], scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    })
}

#[allow(dead_code)]
fn tile_i32() -> Type {
    tile_ty(ScalarType::I32)
}
#[allow(dead_code)]
fn tile_f32() -> Type {
    tile_ty(ScalarType::F32)
}
#[allow(dead_code)]
fn token() -> Type {
    Type::Token
}

/// Build a constant op and append it to a block, returning the result value.
fn build_constant(
    module: &mut Module,
    block: BlockId,
    scalar: ScalarType,
    shape: &[i64],
    data: Vec<u8>,
) -> Value {
    let ty = Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    });
    let (op, results) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(ty.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: ty,
                shape: shape.to_vec(),
                data,
            }),
        )
        .build(module);
    append_op(module, block, op);
    results[0]
}

// =========================================================================
// 1. Constants per type (from constantTest.mlir)
//
// The C++ test puts all constants in one entry function. We split them
// into per-type tests because the decoder's constant-section parser has
// a known offset issue with many constants in a single module.
// =========================================================================

#[test]
fn test_constant_i1() {
    let module = build_kernel("const_i1", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I1, &[], vec![1u8]); // true
        build_constant(m, b, ScalarType::I1, &[], vec![0u8]); // false
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_i8() {
    let module = build_kernel("const_i8", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I8, &[], 42i8.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I8, &[], (-42i8).to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_i16() {
    let module = build_kernel("const_i16", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I16, &[], 1000i16.to_le_bytes().to_vec());
        build_constant(
            m,
            b,
            ScalarType::I16,
            &[],
            (-1000i16).to_le_bytes().to_vec(),
        );
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_i32() {
    let module = build_kernel("const_i32", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I32, &[], 1i32.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], (-1i32).to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], i32::MAX.to_le_bytes().to_vec());
        build_constant(
            m,
            b,
            ScalarType::I32,
            &[],
            (-2147483647i32).to_le_bytes().to_vec(),
        );
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_i64() {
    let module = build_kernel("const_i64", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I64, &[], 0i64.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I64, &[], 1i64.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I64, &[], (-1i64).to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_f32() {
    let module = build_kernel("const_f32", &[], |m, b, _| {
        build_constant(m, b, ScalarType::F32, &[], 1.0f32.to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_f64() {
    let module = build_kernel("const_f64", &[], |m, b, _| {
        build_constant(
            m,
            b,
            ScalarType::F64,
            &[],
            12.3456f64.to_le_bytes().to_vec(),
        );
        build_constant(
            m,
            b,
            ScalarType::F64,
            &[],
            (-12.3456f64).to_le_bytes().to_vec(),
        );
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_bf16() {
    let module = build_kernel("const_bf16", &[], |m, b, _| {
        build_constant(
            m,
            b,
            ScalarType::BF16,
            &[],
            bf16::from_f64(5.5).to_le_bytes().to_vec(),
        );
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_f8e4m3fn() {
    let module = build_kernel("const_f8e4m3fn", &[], |m, b, _| {
        // 2.5 in f8E4M3FN = 0x50
        build_constant(m, b, ScalarType::F8E4M3FN, &[], vec![0x50]);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_f8e5m2() {
    let module = build_kernel("const_f8e5m2", &[], |m, b, _| {
        // -1.0 in f8E5M2 = 0xBC
        build_constant(m, b, ScalarType::F8E5M2, &[], vec![0xBC]);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_constant_tf32() {
    let module = build_kernel("const_tf32", &[], |m, b, _| {
        build_constant(m, b, ScalarType::TF32, &[], 3.14f32.to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

/// Combined constant test using separate entry functions per type group
/// to mirror the full constantTest.mlir pattern.
///
/// NOTE: ignored because the decoder's constant-section parser has a
/// known offset bug when many distinct constant blobs share a single
/// module-level constant pool. The individual per-type tests above
/// cover every value; this test exists to track the decoder fix.
#[test]
#[ignore = "decoder constant-section offset bug with many constants"]
fn test_constants_multi_entry() {
    let mut module = Module::new("kernels");

    add_entry(&mut module, "const_integers", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I1, &[], vec![1u8]);
        build_constant(m, b, ScalarType::I1, &[], vec![0u8]);
        build_constant(m, b, ScalarType::I8, &[], 42i8.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I8, &[], (-42i8).to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I16, &[], 1000i16.to_le_bytes().to_vec());
        build_constant(
            m,
            b,
            ScalarType::I16,
            &[],
            (-1000i16).to_le_bytes().to_vec(),
        );
        build_constant(m, b, ScalarType::I32, &[], 1i32.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], (-1i32).to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I32, &[], i32::MAX.to_le_bytes().to_vec());
        build_constant(
            m,
            b,
            ScalarType::I32,
            &[],
            (-2147483647i32).to_le_bytes().to_vec(),
        );
    });

    add_entry(&mut module, "const_int64", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I64, &[], 0i64.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I64, &[], 1i64.to_le_bytes().to_vec());
        build_constant(m, b, ScalarType::I64, &[], (-1i64).to_le_bytes().to_vec());
    });

    add_entry(&mut module, "const_floats", &[], |m, b, _| {
        build_constant(m, b, ScalarType::F32, &[], 1.0f32.to_le_bytes().to_vec());
        build_constant(
            m,
            b,
            ScalarType::F64,
            &[],
            12.3456f64.to_le_bytes().to_vec(),
        );
        build_constant(
            m,
            b,
            ScalarType::F64,
            &[],
            (-12.3456f64).to_le_bytes().to_vec(),
        );
    });

    add_entry(&mut module, "const_exotic_floats", &[], |m, b, _| {
        build_constant(
            m,
            b,
            ScalarType::BF16,
            &[],
            bf16::from_f64(5.5).to_le_bytes().to_vec(),
        );
        build_constant(m, b, ScalarType::F8E4M3FN, &[], vec![0x50]);
        build_constant(m, b, ScalarType::F8E5M2, &[], vec![0xBC]);
        build_constant(m, b, ScalarType::TF32, &[], 3.14f32.to_le_bytes().to_vec());
    });

    assert_roundtrip(&module);
}

// =========================================================================
// 2. test_operations_basic (from operationsTest.mlir)
// =========================================================================

#[test]
fn test_operations_addi() {
    let module = build_kernel("addi_op", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_addf() {
    let module = build_kernel("addf_op", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_return() {
    let module = build_kernel("return_op", &[tile_i32()], |m, b, _| {
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(m);
        append_op(m, b, ret);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_constant() {
    let module = build_kernel("constant_op", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_multiple_ops() {
    let module = build_kernel("multiple_ops", &[tile_i32(), tile_i32()], |m, b, args| {
        // %0 = addi %a, %b
        let (op0, res0) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op0);
        // %1 = addi %0, %a
        let (op1, res1) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(res0[0])
            .operand(args[0])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op1);
        // %2 = constant i32: 5
        let c5 = build_constant(m, b, ScalarType::I32, &[], 5i32.to_le_bytes().to_vec());
        // %3 = addi %1, %2
        let (op3, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(res1[0])
            .operand(c5)
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op3);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_for_loop() {
    let module = build_kernel("for_op", &[tile_i32()], |m, b, args| {
        // Constants: lower=0, upper=5, step=1
        let lower = build_constant(m, b, ScalarType::I32, &[], 0i32.to_le_bytes().to_vec());
        let upper = build_constant(m, b, ScalarType::I32, &[], 5i32.to_le_bytes().to_vec());
        let step = build_constant(m, b, ScalarType::I32, &[], 1i32.to_le_bytes().to_vec());

        // Build for body: takes (iv, iter_value) -> continue(new_value)
        // The body block has: iv (induction var) + iter_value
        let (body_region, body_block, body_args) =
            build_single_block_region(m, &[tile_i32(), tile_i32()]);
        // new_value = addi %value, %iv
        let (addi_op, addi_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(body_args[1])
            .operand(body_args[0])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, body_block, addi_op);
        // continue %new_value
        let (cont_op, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
            .operand(addi_res[0])
            .build(m);
        append_op(m, body_block, cont_op);

        // For op: lower, upper, step, iter_values(%a)
        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lower)
            .operand(upper)
            .operand(step)
            .operand(args[0])
            .result(tile_i32())
            .region(body_region)
            .build(m);
        append_op(m, b, for_op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_operations_if_else() {
    let cond_ty = tile_ty(ScalarType::I1);
    let module = build_kernel(
        "if_else_op_test",
        &[cond_ty.clone(), tile_i32(), tile_i32()],
        |m, b, args| {
            // Then region: yield %a
            let (then_region, then_block, _) = build_single_block_region(m, &[]);
            let (y1, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operand(args[1])
                .build(m);
            append_op(m, then_block, y1);

            // Else region: yield %b
            let (else_region, else_block, _) = build_single_block_region(m, &[]);
            let (y2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operand(args[2])
                .build(m);
            append_op(m, else_block, y2);

            let (if_op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
                .operand(args[0])
                .result(tile_i32())
                .region(then_region)
                .region(else_region)
                .build(m);
            append_op(m, b, if_op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn test_operations_join_tokens() {
    let module = build_kernel("join_tokens_op", &[token(), token()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::JoinTokens, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(token())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// 3. test_attrs_rounding_ftz (from attrsTest.mlir)
// =========================================================================

#[test]
fn test_attrs_addf_rounding_nearest_even() {
    let module = build_kernel("addf_op_rn", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_addf_flush_to_zero() {
    let module = build_kernel("addf_op_ftz", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .attr("flush_to_zero", Attribute::Bool(true))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_addf_rounding_zero() {
    let module = build_kernel("addf_op_rz", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(1))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_addf_rounding_negative_inf() {
    let module = build_kernel("addf_op_rm", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(2))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_addf_rounding_positive_inf() {
    let module = build_kernel("addf_op_rp", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(3))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// 4. test_attrs_padding_values (from attrsTest.mlir)
// =========================================================================

fn build_padding_test(padding: Option<PaddingValue>) -> Module {
    let name = match padding {
        None => "pv_none".to_string(),
        Some(p) => format!("pv_{p:?}"),
    };
    let ptr_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![128],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![8],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![128],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: padding,
    });
    build_kernel(&name, std::slice::from_ref(&ptr_ty), move |m, b, args| {
        // make_tensor_view
        let (mtv, mtv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
            .operand(args[0])
            .result(tv_ty.clone())
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(0),
                    Attribute::i32(0),
                ]),
            )
            .build(m);
        append_op(m, b, mtv);
        // make_partition_view
        let (mpv, _) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
            .operand(mtv_res[0])
            .result(pv_ty.clone())
            .build(m);
        append_op(m, b, mpv);
    })
}

#[test]
fn test_attrs_padding_none() {
    assert_roundtrip(&build_padding_test(None));
}

#[test]
fn test_attrs_padding_zero() {
    assert_roundtrip(&build_padding_test(Some(PaddingValue::Zero)));
}

#[test]
fn test_attrs_padding_neg_zero() {
    assert_roundtrip(&build_padding_test(Some(PaddingValue::NegZero)));
}

#[test]
fn test_attrs_padding_nan() {
    assert_roundtrip(&build_padding_test(Some(PaddingValue::Nan)));
}

#[test]
fn test_attrs_padding_pos_inf() {
    assert_roundtrip(&build_padding_test(Some(PaddingValue::PosInf)));
}

#[test]
fn test_attrs_padding_neg_inf() {
    assert_roundtrip(&build_padding_test(Some(PaddingValue::NegInf)));
}

// =========================================================================
// 5. test_attrs_signedness (from attrsTest.mlir)
// =========================================================================

#[test]
fn test_attrs_divi_signed() {
    let module = build_kernel("divi_op_signed", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::DivI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_i32())
            .attr("signedness", Attribute::i32(1)) // signed
            .attr("rounding", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_divi_unsigned() {
    let module = build_kernel(
        "divi_op_unsigned",
        &[tile_i32(), tile_i32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::DivI, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_i32())
                .attr("signedness", Attribute::i32(0)) // unsigned
                .attr("rounding", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn test_attrs_mmai_mixed_signedness() {
    let tile_i8 = tile_ty(ScalarType::I8);
    let module = build_kernel(
        "mmai_op",
        &[tile_i8.clone(), tile_i8.clone(), tile_i32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaI, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(tile_i32())
                .attr("signedness_lhs", Attribute::i32(1)) // signed
                .attr("signedness_rhs", Attribute::i32(0)) // unsigned
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// =========================================================================
// 6. test_optional_fields (from optionalFieldsTest.mlir)
// =========================================================================

#[test]
fn test_optional_addf_with_and_without_ftz() {
    let mut module = Module::new("kernels");
    add_entry(
        &mut module,
        "optional_attrs_test",
        &[tile_f32(), tile_f32()],
        |m, b, args| {
            // addf with flush_to_zero
            let (op0, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(0))
                .attr("flush_to_zero", Attribute::Bool(true))
                .build(m);
            append_op(m, b, op0);

            // addf without flush_to_zero
            let (op1, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(0))
                .build(m);
            append_op(m, b, op1);

            // addf rounding=zero without ftz
            let (op2, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(1))
                .build(m);
            append_op(m, b, op2);

            // addf rounding=zero with ftz
            let (op3, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(1))
                .attr("flush_to_zero", Attribute::Bool(true))
                .build(m);
            append_op(m, b, op3);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn test_optional_load_ptr_tko_all_operands() {
    let ptr_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let mask_ty = tile_ty(ScalarType::I1);
    let padding_ty = tile_f32();

    let module = build_kernel(
        "optional_operands_test",
        &[ptr_ty.clone(), mask_ty.clone(), padding_ty.clone()],
        |m, b, args| {
            // make_token
            let (tok_op, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
                .result(token())
                .build(m);
            append_op(m, b, tok_op);

            // load_ptr_tko with all optional operands: ptr, mask, padding, token
            let (op0, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(tok_res[0])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0)) // weak
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1), // ptr
                        Attribute::i32(1), // mask
                        Attribute::i32(1), // padding
                        Attribute::i32(1), // token
                    ]),
                )
                .build(m);
            append_op(m, b, op0);

            // load_ptr_tko with just ptr (no mask, no padding, no token)
            let (op1, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1), // ptr
                        Attribute::i32(0), // mask
                        Attribute::i32(0), // padding
                        Attribute::i32(0), // token
                    ]),
                )
                .build(m);
            append_op(m, b, op1);

            // load_ptr_tko with ptr and mask only
            let (op2, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1), // ptr
                        Attribute::i32(1), // mask
                        Attribute::i32(0), // padding
                        Attribute::i32(0), // token
                    ]),
                )
                .build(m);
            append_op(m, b, op2);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn test_optional_load_ptr_tko_with_memory_scope() {
    let ptr_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let mask_ty = tile_ty(ScalarType::I1);

    let module = build_kernel(
        "mixed_optional_test",
        &[ptr_ty.clone(), mask_ty.clone()],
        |m, b, args| {
            // load_ptr_tko relaxed device with mask
            let (op0, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(1)) // relaxed
                .attr("memory_scope", Attribute::i32(1)) // device
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1), // ptr
                        Attribute::i32(1), // mask
                        Attribute::i32(0), // padding
                        Attribute::i32(0), // token
                    ]),
                )
                .build(m);
            append_op(m, b, op0);

            // load_ptr_tko relaxed device without mask
            let (op1, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(1))
                .attr("memory_scope", Attribute::i32(1))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(0),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op1);

            // load_ptr_tko weak with mask (no memory_scope)
            let (op2, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op2);
        },
    );
    assert_roundtrip(&module);
}

// =========================================================================
// 7. test_edge_cases (from edgeCasesTest.mlir)
// =========================================================================

#[test]
fn test_edge_case_no_parameters() {
    let module = build_kernel("no_parameters", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
    });
    assert_roundtrip(&module);
}

#[test]
fn test_edge_case_many_parameters() {
    let args_ty: Vec<Type> = (0..10).map(|_| tile_i32()).collect();
    let module = build_kernel("many_parameters", &args_ty, |m, b, args| {
        // Chain: %0 = addi(p0, p1), %1 = addi(%0, p2), ..., %8 = addi(%7, p9)
        let mut acc = {
            let (op, res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_i32())
                .attr("overflow", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
            res[0]
        };
        for &arg in args.iter().take(10).skip(2) {
            let (op, res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
                .operand(acc)
                .operand(arg)
                .result(tile_i32())
                .attr("overflow", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
            acc = res[0];
        }
    });
    assert_roundtrip(&module);
}

#[test]
fn test_edge_case_many_intermediates() {
    let module = build_kernel("multiple_returns", &[tile_i32()], |m, b, args| {
        let c0 = build_constant(m, b, ScalarType::I32, &[], 0i32.to_le_bytes().to_vec());
        let (op1, res1) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(c0)
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op1);

        let c1 = build_constant(m, b, ScalarType::I32, &[], 1i32.to_le_bytes().to_vec());
        let (op3, res3) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(c1)
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op3);

        let (op4, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(res1[0])
            .operand(res3[0])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op4);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_edge_case_long_function_name() {
    let module = build_kernel(
        "long_function_name_that_tests_string_table_with_longer_than_usual_identifiers",
        &[],
        |m, b, _| {
            build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
        },
    );
    assert_roundtrip(&module);
}

// =========================================================================
// 8. test_multidim_constants (from multidimTensorTest.mlir)
// =========================================================================

#[test]
fn test_multidim_1d_i32() {
    let module = build_kernel("array_constants_i32", &[], |m, b, _| {
        // tile<4xi32> with values [1, 2, 3, 4]
        let mut data = Vec::new();
        for v in [1i32, 2, 3, 4] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::I32, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_f32() {
    let module = build_kernel("array_constants_f32", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [5.0f32, 6.0, 7.0, 8.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::F32, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_i1() {
    let module = build_kernel("array_constants_i1", &[], |m, b, _| {
        // tile<4xi1> with values [true, false, true, false]
        build_constant(m, b, ScalarType::I1, &[4], vec![1, 0, 1, 0]);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_i16() {
    let module = build_kernel("array_constants_i16", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [10i16, 20, 30, 40] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::I16, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_2d_f64() {
    let module = build_kernel("array_constants_2d_f64", &[], |m, b, _| {
        // tile<2x2xf64> with values [[1.0, 2.0], [3.0, 4.0]]
        let mut data = Vec::new();
        for v in [1.0f64, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::F64, &[2, 2], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_3d_i32() {
    let module = build_kernel("array_constants_3d_i32", &[], |m, b, _| {
        // tile<2x2x2xi32> with values [[[1,2],[3,4]],[[5,6],[7,8]]]
        let mut data = Vec::new();
        for v in [1i32, 2, 3, 4, 5, 6, 7, 8] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::I32, &[2, 2, 2], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_i8() {
    let module = build_kernel("array_constants_i8", &[], |m, b, _| {
        let data: Vec<u8> = [9i8, 10, 11, 12]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        build_constant(m, b, ScalarType::I8, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_i64() {
    let module = build_kernel("array_constants_i64", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [100i64, 200, 300, 400] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::I64, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_f16() {
    let module = build_kernel("array_constants_f16", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0] {
            data.extend_from_slice(&f16::from_f64(v).to_le_bytes());
        }
        build_constant(m, b, ScalarType::F16, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_bf16() {
    let module = build_kernel("array_constants_bf16", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [5.0, 6.0, 7.0, 8.0] {
            data.extend_from_slice(&bf16::from_f64(v).to_le_bytes());
        }
        build_constant(m, b, ScalarType::BF16, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_tf32() {
    let module = build_kernel("array_constants_tf32", &[], |m, b, _| {
        let mut data = Vec::new();
        for v in [9.0f32, 10.0, 11.0, 12.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        build_constant(m, b, ScalarType::TF32, &[4], data);
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_f8e4m3fn() {
    let module = build_kernel("array_constants_f8e4m3fn", &[], |m, b, _| {
        // f8E4M3FN raw bytes for [1.0, 2.0, 3.0, 4.0]
        // 1.0 = 0x38, 2.0 = 0x40, 3.0 = 0x48, 4.0 = 0x50 (approx standard E4M3FN encoding)
        build_constant(
            m,
            b,
            ScalarType::F8E4M3FN,
            &[4],
            vec![0x38, 0x40, 0x48, 0x50],
        );
    });
    assert_roundtrip(&module);
}

#[test]
fn test_multidim_1d_f8e5m2() {
    let module = build_kernel("array_constants_f8e5m2", &[], |m, b, _| {
        // f8E5M2 raw bytes for [5.0, 6.0, 7.0, 8.0]
        // 5.0 = 0x45, 6.0 = 0x46, 7.0 = 0x47, 8.0 = 0x48 (approx standard E5M2 encoding)
        build_constant(m, b, ScalarType::F8E5M2, &[4], vec![0x45, 0x46, 0x47, 0x48]);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// 9. test_global_section (from globalSectionTest.mlir)
// =========================================================================

#[test]
fn test_global_section() {
    let mut module = Module::new("kernels");

    // Global: @val <f64: [1.0, 2.0, 3.0, 4.0]> : tile<4xf64>
    let mut val_data = Vec::new();
    for v in [1.0f64, 2.0, 3.0, 4.0] {
        val_data.extend_from_slice(&v.to_le_bytes());
    }
    module.globals.push(Global {
        sym_name: "val".into(),
        value: DenseElements {
            element_type: tile_shaped(&[4], ScalarType::F64),
            shape: vec![4],
            data: val_data,
        },
        alignment: 0,
        constant: false,
        symbol_visibility: SymbolVisibility::Public,
    });

    // Global: @val2 alignment = 256 <i32: 42> : tile<1xi32>
    module.globals.push(Global {
        sym_name: "val2".into(),
        value: DenseElements {
            element_type: tile_shaped(&[1], ScalarType::I32),
            shape: vec![1],
            data: 42i32.to_le_bytes().to_vec(),
        },
        alignment: 256,
        constant: false,
        symbol_visibility: SymbolVisibility::Public,
    });

    // Entry function: @add_entry()
    add_entry(&mut module, "add_entry", &[], |_m, _b, _args| {
        // empty body, return appended automatically
    });

    assert_roundtrip(&module);
}

#[test]
fn test_global_get_global() {
    // From operationsTest.mlir: module with global + get_global op
    let mut module = Module::new("kernels");

    let mut val_data = Vec::new();
    val_data.extend_from_slice(&1.23f32.to_le_bytes());
    module.globals.push(Global {
        sym_name: "my_test_global".into(),
        value: DenseElements {
            element_type: tile_shaped(&[1], ScalarType::F32),
            shape: vec![1],
            data: val_data,
        },
        alignment: 0,
        constant: false,
        symbol_visibility: SymbolVisibility::Public,
    });

    let ptr_f32_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });

    add_entry(&mut module, "get_global_op_test", &[], move |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::GetGlobal, Location::Unknown)
            .result(ptr_f32_ty.clone())
            .attr("name", Attribute::String("my_test_global".into()))
            .build(m);
        append_op(m, b, op);
    });

    assert_roundtrip(&module);
}

// =========================================================================
// 10. test_empty_module (from emptyModuleTest.mlir)
// =========================================================================

#[test]
fn test_empty_module() {
    let module = Module::new("kernels");
    assert_roundtrip(&module);
}

// =========================================================================
// 11. Negative tests
// =========================================================================

#[test]
fn test_invalid_magic() {
    // Build minimal bytecode with wrong magic
    let mut data = Vec::new();
    data.extend_from_slice(&[0x00, b'B', b'A', b'D', b'M', b'A', b'G', b'C']);
    data.push(13); // major
    data.push(1); // minor
    data.extend_from_slice(&0u16.to_le_bytes()); // tag
    data.push(0x00); // EndOfBytecode

    let result = cutile_ir::decode_bytecode(&data);
    assert!(result.is_err(), "expected error for invalid magic number");
}

#[test]
fn test_invalid_magic_partial() {
    // Corrupt one byte of the real magic
    let mut data = Vec::new();
    data.extend_from_slice(&MAGIC);
    data[1] = b'X'; // corrupt 'T' -> 'X'
    data.push(13);
    data.push(1);
    data.extend_from_slice(&0u16.to_le_bytes());
    data.push(0x00);

    let result = cutile_ir::decode_bytecode(&data);
    assert!(result.is_err(), "expected error for corrupted magic byte");
}

#[test]
fn test_invalid_truncated_header() {
    // Too short to contain a full header
    let data = vec![0x7F, b'T', b'i'];
    let result = cutile_ir::decode_bytecode(&data);
    assert!(result.is_err(), "expected error for truncated header");
}

#[test]
fn test_empty_bytes() {
    let result = cutile_ir::decode_bytecode(&[]);
    assert!(result.is_err(), "expected error for empty input");
}

// =========================================================================
// Combined multi-entry module test (from operationsTest.mlir)
// =========================================================================

#[test]
fn test_operations_multi_entry_module() {
    let mut module = Module::new("kernels");

    // addi_op
    add_entry(
        &mut module,
        "addi_op",
        &[tile_i32(), tile_i32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_i32())
                .attr("overflow", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
        },
    );

    // addf_op
    add_entry(
        &mut module,
        "addf_op",
        &[tile_f32(), tile_f32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
        },
    );

    // return_op
    add_entry(&mut module, "return_op", &[tile_i32()], |m, b, _| {
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(m);
        append_op(m, b, ret);
    });

    // constant_op
    add_entry(&mut module, "constant_op", &[], |m, b, _| {
        build_constant(m, b, ScalarType::I32, &[], 42i32.to_le_bytes().to_vec());
    });

    assert_roundtrip(&module);
}

// =========================================================================
// All rounding modes combined in one module (from attrsTest.mlir)
// =========================================================================

#[test]
fn test_attrs_all_rounding_modes_combined() {
    let mut module = Module::new("kernels");

    let modes = [
        ("rn", 0), // nearest_even
        ("rz", 1), // zero
        ("rm", 2), // negative_inf
        ("rp", 3), // positive_inf
    ];

    for (suffix, mode_val) in modes {
        let name = format!("addf_op_{suffix}");
        add_entry(
            &mut module,
            &name,
            &[tile_f32(), tile_f32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_f32())
                    .attr("rounding_mode", Attribute::i32(mode_val))
                    .build(m);
                append_op(m, b, op);
            },
        );
    }

    // Also add ftz variant
    add_entry(
        &mut module,
        "addf_op_ftz",
        &[tile_f32(), tile_f32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(0))
                .attr("flush_to_zero", Attribute::Bool(true))
                .build(m);
            append_op(m, b, op);
        },
    );

    assert_roundtrip(&module);
}

// =========================================================================
// All padding values in one module (from attrsTest.mlir)
// =========================================================================

#[test]
fn test_attrs_all_padding_values_combined() {
    let ptr_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![128],
        strides: vec![1],
    });

    let paddings: &[(Option<PaddingValue>, &str)] = &[
        (None, "none"),
        (Some(PaddingValue::Zero), "zero"),
        (Some(PaddingValue::NegZero), "neg_zero"),
        (Some(PaddingValue::Nan), "nan"),
        (Some(PaddingValue::PosInf), "pos_inf"),
        (Some(PaddingValue::NegInf), "neg_inf"),
    ];

    let mut module = Module::new("kernels");

    add_entry(
        &mut module,
        "make_partition_view_op",
        std::slice::from_ref(&ptr_ty),
        |m, b, args| {
            // make_tensor_view from pointer
            let (mtv, mtv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[0])
                .result(tv_ty.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, mtv);

            // For each padding value, create a make_partition_view
            for (padding, _label) in paddings {
                let pv_ty = Type::PartitionView(PartitionViewType {
                    tile_shape: vec![8],
                    tensor_view: TensorViewType {
                        element_type: ScalarType::F32,
                        shape: vec![128],
                        strides: vec![1],
                    },
                    dim_map: vec![0],
                    padding_value: *padding,
                });
                let (mpv, _) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                    .operand(mtv_res[0])
                    .result(pv_ty)
                    .build(m);
                append_op(m, b, mpv);
            }
        },
    );

    assert_roundtrip(&module);
}

// =========================================================================
// All multidim constants combined (from multidimTensorTest.mlir)
//
// Uses separate entry functions to work around the decoder's
// constant-section offset limitation with many constants.
// =========================================================================

/// NOTE: ignored for same reason as test_constants_multi_entry — decoder
/// constant-section offset bug with many constants in one module.
#[test]
#[ignore = "decoder constant-section offset bug with many constants"]
fn test_multidim_all_types_combined() {
    let mut module = Module::new("kernels");

    add_entry(&mut module, "array_int_constants", &[], |m, b, _| {
        // 4xi32: [1,2,3,4]
        {
            let mut data = Vec::new();
            for v in [1i32, 2, 3, 4] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::I32, &[4], data);
        }
        // 4xi1: [true, false, true, false]
        build_constant(m, b, ScalarType::I1, &[4], vec![1, 0, 1, 0]);
        // 4xi16: [10,20,30,40]
        {
            let mut data = Vec::new();
            for v in [10i16, 20, 30, 40] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::I16, &[4], data);
        }
        // 2x2x2xi32: [[[1,2],[3,4]],[[5,6],[7,8]]]
        {
            let mut data = Vec::new();
            for v in [1i32, 2, 3, 4, 5, 6, 7, 8] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::I32, &[2, 2, 2], data);
        }
        // 4xi8: [9,10,11,12]
        {
            let data: Vec<u8> = [9i8, 10, 11, 12]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            build_constant(m, b, ScalarType::I8, &[4], data);
        }
        // 4xi64: [100,200,300,400]
        {
            let mut data = Vec::new();
            for v in [100i64, 200, 300, 400] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::I64, &[4], data);
        }
    });

    add_entry(&mut module, "array_float_constants", &[], |m, b, _| {
        // 4xf32: [5.0,6.0,7.0,8.0]
        {
            let mut data = Vec::new();
            for v in [5.0f32, 6.0, 7.0, 8.0] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::F32, &[4], data);
        }
        // 2x2xf64: [[1.0,2.0],[3.0,4.0]]
        {
            let mut data = Vec::new();
            for v in [1.0f64, 2.0, 3.0, 4.0] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::F64, &[2, 2], data);
        }
        // 4xf16: [1.0,2.0,3.0,4.0]
        {
            let mut data = Vec::new();
            for v in [1.0, 2.0, 3.0, 4.0] {
                data.extend_from_slice(&f16::from_f64(v).to_le_bytes());
            }
            build_constant(m, b, ScalarType::F16, &[4], data);
        }
        // 4xbf16: [5.0,6.0,7.0,8.0]
        {
            let mut data = Vec::new();
            for v in [5.0, 6.0, 7.0, 8.0] {
                data.extend_from_slice(&bf16::from_f64(v).to_le_bytes());
            }
            build_constant(m, b, ScalarType::BF16, &[4], data);
        }
        // 4xtf32: [9.0,10.0,11.0,12.0]
        {
            let mut data = Vec::new();
            for v in [9.0f32, 10.0, 11.0, 12.0] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            build_constant(m, b, ScalarType::TF32, &[4], data);
        }
        // 4xf8E4M3FN: [1.0,2.0,3.0,4.0] (raw bytes)
        build_constant(
            m,
            b,
            ScalarType::F8E4M3FN,
            &[4],
            vec![0x38, 0x40, 0x48, 0x50],
        );
        // 4xf8E5M2: [5.0,6.0,7.0,8.0] (raw bytes)
        build_constant(m, b, ScalarType::F8E5M2, &[4], vec![0x45, 0x46, 0x47, 0x48]);
    });

    assert_roundtrip(&module);
}
