/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-op bytecode roundtrip tests.
//!
//! For each operation pattern, builds a minimal kernel, serializes to
//! bytecode, and decodes to verify no errors. These are cheap (no GPU)
//! and run with every `cargo test`.

// Float literals such as 3.14159 are arbitrary test payloads, not approximations of pi.
#![allow(clippy::approx_constant)]

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

/// Helper: build a module with a single entry function containing the given ops.
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

/// Assert bytecode roundtrip: serialize → decode → no error.
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

// Common types used across tests.
fn tile_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F32),
    })
}
fn tile_f32_64x64() -> Type {
    Type::Tile(TileType {
        shape: vec![64, 64],
        element_type: TileElementType::Scalar(ScalarType::F32),
    })
}
fn tile_i32() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::I32),
    })
}
fn tile_i8_32() -> Type {
    Type::Tile(TileType {
        shape: vec![32],
        element_type: TileElementType::Scalar(ScalarType::I8),
    })
}
fn tile_f4_64() -> Type {
    Type::Tile(TileType {
        shape: vec![64],
        element_type: TileElementType::Scalar(ScalarType::F4E2M1FN),
    })
}
fn scalar_i32() -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    })
}
fn scalar_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::F32),
    })
}
fn tile_i1() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::I1),
    })
}
fn scalar_i1() -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I1),
    })
}
fn ptr_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    })
}
fn tile_ptr_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    })
}
fn token() -> Type {
    Type::Token
}

fn tensor_view_f32_2d() -> TensorViewType {
    TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![8192, 128],
        strides: vec![128, 1],
    }
}

fn gather_scatter_view_f32_2d() -> Type {
    Type::GatherScatterView(GatherScatterViewType {
        tile_shape: vec![64, 64],
        tensor_view: tensor_view_f32_2d(),
        sparse_dim: 0,
        padding_value: None,
    })
}

fn strided_view_f32_2d() -> Type {
    Type::StridedView(StridedViewType {
        tile_shape: vec![64, 64],
        traversal_strides: vec![2, 1],
        tensor_view: tensor_view_f32_2d(),
        dim_map: vec![0, 1],
        padding_value: None,
    })
}

// =========================================================================
// Simple ops: result types + operands (no size)
// =========================================================================

#[test]
fn roundtrip_simple_unary_ops() {
    // Sqrt has rounding_mode attr — tested separately in roundtrip_exp2_rsqrt_sqrt.
    for opcode in [
        Opcode::AbsF,
        Opcode::Ceil,
        Opcode::Cos,
        Opcode::CosH,
        Opcode::Exp,
        Opcode::Floor,
        Opcode::Log,
        Opcode::Log2,
        Opcode::NegF,
        Opcode::Pow,
        Opcode::Sin,
        Opcode::SinH,
        Opcode::Tan,
        Opcode::TanH,
    ] {
        let module = build_kernel(&format!("{opcode:?}"), &[tile_f32()], |m, b, args| {
            // Sqrt/Exp2/Rsqrt have flags — but we test those separately.
            // Simple unary: result type + 1 operand.
            let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                .operand(args[0])
                .result(tile_f32())
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_simple_binary_ops() {
    for opcode in [Opcode::AndI, Opcode::OrI, Opcode::XOrI, Opcode::RemF] {
        let ty = if opcode == Opcode::RemF {
            tile_f32()
        } else {
            tile_i32()
        };
        let module = build_kernel(
            &format!("{opcode:?}"),
            &[ty.clone(), ty.clone()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(ty.clone())
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

// =========================================================================
// Ops with required attributes
// =========================================================================

#[test]
fn roundtrip_overflow_ops() {
    for opcode in [
        Opcode::AddI,
        Opcode::MulI,
        Opcode::SubI,
        Opcode::ShLI,
        Opcode::TruncI,
    ] {
        let module = build_kernel(
            &format!("{opcode:?}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let mut builder = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .result(tile_i32())
                    .attr("overflow", Attribute::i32(0));
                if !matches!(opcode, Opcode::TruncI) {
                    builder = builder.operand(args[1]);
                }
                let (op, _) = builder.build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_signedness_ops() {
    for opcode in [
        Opcode::MaxI,
        Opcode::MinI,
        Opcode::RemI,
        Opcode::ShRI,
        Opcode::ExtI,
    ] {
        let module = build_kernel(
            &format!("{opcode:?}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let mut builder = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .result(tile_i32())
                    .attr("signedness", Attribute::i32(1));
                if !matches!(opcode, Opcode::ExtI) {
                    builder = builder.operand(args[1]);
                }
                let (op, _) = builder.build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_cmp_ops() {
    // CmpF
    let module = build_kernel("cmpf", &[tile_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::CmpF, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(Type::Tile(TileType {
                shape: vec![128],
                element_type: TileElementType::Scalar(ScalarType::I1),
            }))
            .attr("comparison_predicate", Attribute::i32(0))
            .attr("comparison_ordering", Attribute::i32(1))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);

    // CmpI
    let module = build_kernel("cmpi", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(Type::Tile(TileType {
                shape: vec![128],
                element_type: TileElementType::Scalar(ScalarType::I1),
            }))
            .attr("comparison_predicate", Attribute::i32(0))
            .attr("signedness", Attribute::i32(1))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// Ops with flags (optional attributes)
// =========================================================================

#[test]
fn roundtrip_float_arith_with_flags() {
    for opcode in [
        Opcode::AddF,
        Opcode::SubF,
        Opcode::MulF,
        Opcode::DivF,
        Opcode::Fma,
    ] {
        let num_operands = if opcode == Opcode::Fma { 3 } else { 2 };
        let args_ty: Vec<Type> = (0..num_operands).map(|_| tile_f32()).collect();
        let module = build_kernel(&format!("{opcode:?}"), &args_ty, |m, b, args| {
            let mut builder = OpBuilder::new(opcode, Location::Unknown)
                .result(tile_f32())
                .attr("rounding_mode", Attribute::i32(0));
            for &a in args {
                builder = builder.operand(a);
            }
            let (op, _) = builder.build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_exp2_rsqrt_sqrt() {
    for opcode in [Opcode::Exp2, Opcode::Rsqrt] {
        let module = build_kernel(&format!("{opcode:?}"), &[tile_f32()], |m, b, args| {
            let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                .operand(args[0])
                .result(tile_f32())
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
    // Sqrt has rounding_mode attr too
    let module = build_kernel("sqrt", &[tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Sqrt, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// Variadic result ops
// =========================================================================

#[test]
fn roundtrip_variadic_result_ops() {
    // Break, Continue, Return, Yield — no results, with operands
    for opcode in [
        Opcode::Break,
        Opcode::Continue,
        Opcode::Return,
        Opcode::Yield,
    ] {
        let module = build_kernel(&format!("{opcode:?}"), &[], |m, b, _| {
            let (op, _) = OpBuilder::new(opcode, Location::Unknown).build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_get_tile_block_id() {
    let module = build_kernel("get_tile_block_id", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
            .result(scalar_i32())
            .result(scalar_i32())
            .result(scalar_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_make_token() {
    let module = build_kernel("make_token", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(token())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_constant() {
    let module = build_kernel("constant", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::Constant, Location::Unknown)
            .result(scalar_i32())
            .attr(
                "value",
                Attribute::DenseElements(DenseElements {
                    element_type: scalar_i32(),
                    shape: vec![],
                    data: 42i32.to_le_bytes().to_vec(),
                }),
            )
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// =========================================================================
// View construction ops
// =========================================================================

#[test]
fn roundtrip_make_tensor_view() {
    let ptr_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC],
        strides: vec![1],
    });
    let module = build_kernel(
        "make_tensor_view",
        &[ptr_ty.clone(), scalar_i32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tv_ty.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_make_partition_view() {
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let module = build_kernel(
        "make_partition_view",
        std::slice::from_ref(&tv_ty),
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                .operand(args[0])
                .result(pv_ty.clone())
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// =========================================================================
// Control flow ops (with regions)
// =========================================================================

#[test]
fn roundtrip_if_op() {
    let cond_ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I1),
    });
    let module = build_kernel("if_op", std::slice::from_ref(&cond_ty), |m, b, args| {
        // Build then region.
        let (then_region, then_block, _) = build_single_block_region(m, &[]);
        let (ret, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(m);
        append_op(m, then_block, ret);
        // Build else region.
        let (else_region, else_block, _) = build_single_block_region(m, &[]);
        let (ret, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(m);
        append_op(m, else_block, ret);

        let (op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
            .operand(args[0])
            .region(then_region)
            .region(else_region)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_for_op() {
    let module = build_kernel(
        "for_op",
        &[scalar_i32(), scalar_i32(), scalar_i32()],
        |m, b, args| {
            let (body_region, body_block, _) = build_single_block_region(m, &[scalar_i32()]);
            let (yield_op, _) = OpBuilder::new(Opcode::Yield, Location::Unknown).build(m);
            append_op(m, body_block, yield_op);

            let (op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .region(body_region)
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// =========================================================================
// P0 AUDIT: Remaining ops not covered above
// =========================================================================

// --- Simple ops (result types + operands, no size) ---

#[test]
fn roundtrip_simple_ops_remaining() {
    for (opcode, arg_tys, result_ty) in [
        (Opcode::Bitcast, vec![tile_f32()], tile_i32()),
        (Opcode::Broadcast, vec![scalar_f32()], tile_f32()),
        (Opcode::Reshape, vec![tile_f32()], tile_f32()),
        (
            Opcode::Select,
            vec![scalar_i1(), scalar_f32(), scalar_f32()],
            scalar_f32(),
        ),
        (
            Opcode::Offset,
            vec![tile_ptr_f32(), tile_i32()],
            tile_ptr_f32(),
        ),
        (Opcode::MulhiI, vec![tile_i32(), tile_i32()], tile_i32()),
        (Opcode::IntToPtr, vec![tile_i32()], tile_ptr_f32()),
        (Opcode::PtrToInt, vec![tile_ptr_f32()], tile_i32()),
        (Opcode::PtrToPtr, vec![tile_ptr_f32()], tile_ptr_f32()),
    ] {
        let module = build_kernel(&format!("{opcode:?}"), &arg_tys, |m, b, args| {
            let mut builder = OpBuilder::new(opcode, Location::Unknown).result(result_ty.clone());
            for &a in args {
                builder = builder.operand(a);
            }
            let (op, _) = builder.build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_absi() {
    let module = build_kernel("absi", &[tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AbsI, Location::Unknown)
            .operand(args[0])
            .result(tile_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_negi_with_overflow() {
    let module = build_kernel("negi", &[tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::NegI, Location::Unknown)
            .operand(args[0])
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_tanh_with_rounding() {
    let module = build_kernel("tanh_rm", &[tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::TanH, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(3))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_iota() {
    let module = build_kernel("iota", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::Iota, Location::Unknown)
            .result(tile_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// --- Required attribute ops ---

#[test]
fn roundtrip_cat() {
    let big = Type::Tile(TileType {
        shape: vec![256],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let module = build_kernel("cat", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Cat, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(big.clone())
            .attr("dim", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_permute() {
    let ty = Type::Tile(TileType {
        shape: vec![64, 128],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let ty_t = Type::Tile(TileType {
        shape: vec![128, 64],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let module = build_kernel("permute", std::slice::from_ref(&ty), |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Permute, Location::Unknown)
            .operand(args[0])
            .result(ty_t.clone())
            .attr("permutation", Attribute::DenseI32Array(vec![1, 0]))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_assert() {
    let module = build_kernel("assert_op", &[scalar_i1()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Assert, Location::Unknown)
            .operand(args[0])
            .attr("message", Attribute::String("assertion failed".into()))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_assume() {
    let module = build_kernel("assume_op", &[tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Assume, Location::Unknown)
            .operand(args[0])
            .result(tile_i32())
            .attr(
                "predicate",
                Attribute::Dictionary(vec![
                    ("kind".into(), Attribute::String("bounded".into())),
                    ("lower".into(), Attribute::i32(0)),
                    ("upper".into(), Attribute::i32(1024)),
                ]),
            )
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_divi() {
    let module = build_kernel("divi", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::DivI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_i32())
            .attr("signedness", Attribute::i32(1))
            .attr("rounding", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_mmai() {
    let ty = Type::Tile(TileType {
        shape: vec![16, 16],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let module = build_kernel(
        "mmai",
        &[ty.clone(), ty.clone(), ty.clone()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaI, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(ty.clone())
                .attr("signedness_lhs", Attribute::i32(1))
                .attr("signedness_rhs", Attribute::i32(1))
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_mmaf() {
    let ty_in = Type::Tile(TileType {
        shape: vec![16, 16],
        element_type: TileElementType::Scalar(ScalarType::F16),
    });
    let ty_acc = Type::Tile(TileType {
        shape: vec![16, 16],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let module = build_kernel(
        "mmaf",
        &[ty_in.clone(), ty_in.clone(), ty_acc.clone()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(ty_acc.clone())
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_pack_unpack() {
    let packed = tile_i8_32();
    let unpacked = tile_f4_64();
    let module = build_kernel(
        "pack_unpack",
        &[unpacked.clone(), packed.clone()],
        |m, b, args| {
            let (pack_op, _) = OpBuilder::new(Opcode::Pack, Location::Unknown)
                .operand(args[0])
                .result(packed.clone())
                .build(m);
            append_op(m, b, pack_op);

            let (unpack_op, _) = OpBuilder::new(Opcode::Unpack, Location::Unknown)
                .operand(args[1])
                .result(unpacked.clone())
                .build(m);
            append_op(m, b, unpack_op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_mmaf_scaled() {
    let ty_in = Type::Tile(TileType {
        shape: vec![16, 16],
        element_type: TileElementType::Scalar(ScalarType::F4E2M1FN),
    });
    let ty_acc = Type::Tile(TileType {
        shape: vec![16, 16],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let lhs_scale = Type::Tile(TileType {
        shape: vec![16, 1],
        element_type: TileElementType::Scalar(ScalarType::F8E8M0FNU),
    });
    let rhs_scale = Type::Tile(TileType {
        shape: vec![1, 16],
        element_type: TileElementType::Scalar(ScalarType::F8E8M0FNU),
    });
    let module = build_kernel(
        "mmaf_scaled",
        &[
            ty_in.clone(),
            ty_in.clone(),
            ty_acc.clone(),
            lhs_scale.clone(),
            rhs_scale.clone(),
        ],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaFScaled, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .operand(args[4])
                .result(ty_acc.clone())
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// --- Conversion ops ---

#[test]
fn roundtrip_ftof() {
    let f16 = Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F16),
    });
    let module = build_kernel("ftof", &[tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::FToF, Location::Unknown)
            .operand(args[0])
            .result(f16.clone())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_ftoi() {
    let module = build_kernel("ftoi", &[tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::FToI, Location::Unknown)
            .operand(args[0])
            .result(tile_i32())
            .attr("signedness", Attribute::i32(1))
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_itof() {
    let module = build_kernel("itof", &[tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::IToF, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .attr("signedness", Attribute::i32(1))
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// --- Float flags variants ---

#[test]
fn roundtrip_maxf_minf_with_flags() {
    for opcode in [Opcode::MaxF, Opcode::MinF] {
        let module = build_kernel(
            &format!("{opcode:?}"),
            &[tile_f32(), tile_f32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_f32())
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
        let module = build_kernel(
            &format!("{opcode:?}_flags"),
            &[tile_f32(), tile_f32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_f32())
                    .attr("propagate_nan", Attribute::i32(1))
                    .attr("flush_to_zero", Attribute::i32(1))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_float_arith_with_ftz() {
    for opcode in [Opcode::AddF, Opcode::SubF, Opcode::MulF, Opcode::DivF] {
        let module = build_kernel(
            &format!("{opcode:?}_ftz"),
            &[tile_f32(), tile_f32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_f32())
                    .attr("rounding_mode", Attribute::i32(0))
                    .attr("flush_to_zero", Attribute::i32(1))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_exp2_rsqrt_with_ftz() {
    for opcode in [Opcode::Exp2, Opcode::Rsqrt] {
        let module = build_kernel(&format!("{opcode:?}_ftz"), &[tile_f32()], |m, b, args| {
            let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                .operand(args[0])
                .result(tile_f32())
                .attr("flush_to_zero", Attribute::i32(1))
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

// --- Query ops ---

#[test]
fn roundtrip_get_num_tile_blocks() {
    let module = build_kernel("gnb", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::GetNumTileBlocks, Location::Unknown)
            .result(scalar_i32())
            .result(scalar_i32())
            .result(scalar_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_get_tensor_shape() {
    let tv = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC, DYNAMIC],
        strides: vec![DYNAMIC, 1],
    });
    let module = build_kernel("gts", std::slice::from_ref(&tv), |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::GetTensorShape, Location::Unknown)
            .operand(args[0])
            .result(scalar_i32())
            .result(scalar_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_get_index_space_shape() {
    let pv = Type::PartitionView(PartitionViewType {
        tile_shape: vec![64],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let module = build_kernel("giss", std::slice::from_ref(&pv), |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::GetIndexSpaceShape, Location::Unknown)
            .operand(args[0])
            .result(scalar_i32())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// --- Memory ops ---

#[test]
fn roundtrip_alloca() {
    let module = build_kernel("alloca", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::Alloca, Location::Unknown)
            .result(ptr_f32())
            .attr("num_elem", Attribute::i32(64))
            .attr("alignment", Attribute::i32(128))
            .attr("global", Attribute::Bool(true))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_make_gather_scatter_view() {
    let tv = Type::TensorView(tensor_view_f32_2d());
    let gsv = gather_scatter_view_f32_2d();
    let module = build_kernel("make_gather_scatter_view", &[tv], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::MakeGatherScatterView, Location::Unknown)
            .operand(args[0])
            .result(gsv)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_make_strided_view() {
    let tv = Type::TensorView(tensor_view_f32_2d());
    let sv = strided_view_f32_2d();
    let module = build_kernel("make_strided_view", &[tv], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::MakeStridedView, Location::Unknown)
            .operand(args[0])
            .result(sv)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_load_ptr_tko_minimal() {
    let module = build_kernel("lpt_min", &[tile_ptr_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .result(token())
            .attr("memory_ordering_semantics", Attribute::i32(0))
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
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_load_ptr_tko_with_mask_and_token() {
    let module = build_kernel(
        "lpt_mt",
        &[tile_ptr_f32(), tile_i1(), token()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_store_ptr_tko() {
    let module = build_kernel(
        "spt",
        &[tile_ptr_f32(), tile_f32(), token()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::StorePtrTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_load_view_tko() {
    let pv = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let module = build_kernel("lvt", &[pv.clone(), scalar_i32(), token()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .operand(args[2])
            .result(tile_f32())
            .result(token())
            .attr("memory_ordering_semantics", Attribute::i32(0))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(1),
                    Attribute::i32(1),
                ]),
            )
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_store_view_tko() {
    let pv = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let module = build_kernel(
        "svt",
        &[tile_f32(), pv.clone(), scalar_i32(), token()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_gather_scatter_view_load_store_tko() {
    let gsv = gather_scatter_view_f32_2d();
    let sparse_index = Type::Tile(TileType {
        shape: vec![8],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let module = build_kernel(
        "gsv_load_store",
        &[
            gsv.clone(),
            sparse_index,
            scalar_i32(),
            tile_f32_64x64(),
            token(),
        ],
        |m, b, args| {
            let (load, load_res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[4])
                .result(tile_f32_64x64())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(2),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, load);

            let (store, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
                .operand(args[3])
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(load_res[1])
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(2),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, store);
        },
    );
    assert_roundtrip(&module);
}

// --- Atomic ops ---

#[test]
fn roundtrip_atomic_rmw() {
    let module = build_kernel("armw", &[tile_ptr_f32(), tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(tile_f32())
            .result(token())
            .attr("memory_ordering_semantics", Attribute::i32(0))
            .attr("memory_scope", Attribute::i32(0))
            .attr("mode", Attribute::i32(0))
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
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_atomic_cas() {
    let module = build_kernel(
        "acas",
        &[tile_ptr_f32(), tile_f32(), tile_f32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AtomicCAS, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(tile_f32())
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr("memory_scope", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_atomic_red_view_tko() {
    let gsv = gather_scatter_view_f32_2d();
    let sparse_index = Type::Tile(TileType {
        shape: vec![8],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let module = build_kernel(
        "atomic_red_view_tko",
        &[gsv, sparse_index, scalar_i32(), tile_f32_64x64(), token()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AtomicRedViewTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .operand(args[4])
                .result(token())
                .attr("memory_ordering_semantics", Attribute::i32(1))
                .attr("memory_scope", Attribute::i32(1))
                .attr("mode", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(2),
                        Attribute::i32(1),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// --- Reduce and Scan ---

#[test]
fn roundtrip_reduce() {
    let module = build_kernel("reduce", &[tile_f32()], |m, b, args| {
        let (region, body, ba) = build_single_block_region(m, &[scalar_f32(), scalar_f32()]);
        let (add, ar) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(ba[0])
            .operand(ba[1])
            .result(scalar_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, body, add);
        let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(ar[0])
            .build(m);
        append_op(m, body, y);
        let (op, _) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
            .operand(args[0])
            .result(scalar_f32())
            .attr("dim", Attribute::i32(0))
            .attr("identities", Attribute::String("0.0".into()))
            .region(region)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_scan_forward() {
    let module = build_kernel("scan_fwd", &[tile_f32()], |m, b, args| {
        let (region, body, ba) = build_single_block_region(m, &[scalar_f32(), scalar_f32()]);
        let (add, ar) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(ba[0])
            .operand(ba[1])
            .result(scalar_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, body, add);
        let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(ar[0])
            .build(m);
        append_op(m, body, y);
        let (op, _) = OpBuilder::new(Opcode::Scan, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .attr("dim", Attribute::i32(0))
            .attr("reverse", Attribute::i32(0))
            .attr("identities", Attribute::String("0.0".into()))
            .region(region)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_scan_reverse() {
    let module = build_kernel("scan_rev", &[tile_f32()], |m, b, args| {
        let (region, body, ba) = build_single_block_region(m, &[scalar_f32(), scalar_f32()]);
        let (add, ar) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(ba[0])
            .operand(ba[1])
            .result(scalar_f32())
            .attr("rounding_mode", Attribute::i32(0))
            .build(m);
        append_op(m, body, add);
        let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(ar[0])
            .build(m);
        append_op(m, body, y);
        let (op, _) = OpBuilder::new(Opcode::Scan, Location::Unknown)
            .operand(args[0])
            .result(tile_f32())
            .attr("dim", Attribute::i32(0))
            .attr("reverse", Attribute::i32(1))
            .attr("identities", Attribute::String("0.0".into()))
            .region(region)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

// --- Control flow: Loop, If with results, For with iter args ---

#[test]
fn roundtrip_loop_op() {
    let module = build_kernel("loop_op", &[], |m, b, _| {
        let (body_region, body_block, _) = build_single_block_region(m, &[]);
        let (c, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, body_block, c);
        let (op, _) = OpBuilder::new(Opcode::Loop, Location::Unknown)
            .region(body_region)
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_if_with_results() {
    let module = build_kernel(
        "if_res",
        &[scalar_i1(), scalar_f32(), scalar_f32()],
        |m, b, args| {
            let (tr, tb, _) = build_single_block_region(m, &[]);
            let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operand(args[1])
                .build(m);
            append_op(m, tb, y);
            let (er, eb, _) = build_single_block_region(m, &[]);
            let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operand(args[2])
                .build(m);
            append_op(m, eb, y);
            let (op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
                .operand(args[0])
                .result(scalar_f32())
                .region(tr)
                .region(er)
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_for_with_iter_args() {
    let module = build_kernel(
        "for_ia",
        &[scalar_i32(), scalar_i32(), scalar_i32(), scalar_f32()],
        |m, b, args| {
            let (br, bb, ba) = build_single_block_region(m, &[scalar_i32(), scalar_f32()]);
            let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
                .operand(ba[1])
                .build(m);
            append_op(m, bb, y);
            let (op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .result(scalar_f32())
                .region(br)
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}

// --- Misc: JoinTokens, Extract, Print, Constants ---

#[test]
fn roundtrip_join_tokens() {
    let module = build_kernel("jt", &[token(), token()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::JoinTokens, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(token())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_extract() {
    let ty2d = Type::Tile(TileType {
        shape: vec![64, 128],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let row = Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let module = build_kernel("extract", &[ty2d.clone(), scalar_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Extract, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .result(row.clone())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_print() {
    let module = build_kernel("print_op", &[tile_f32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Print, Location::Unknown)
            .operand(args[0])
            .result(scalar_i32())
            .attr("str", Attribute::String("value = %f\n".into()))
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_constants_all_scalar_types() {
    let cases: Vec<(&str, ScalarType, Vec<u8>)> = vec![
        ("i1", ScalarType::I1, vec![0x01]),
        ("i8", ScalarType::I8, 42i8.to_le_bytes().to_vec()),
        ("i16", ScalarType::I16, 1000i16.to_le_bytes().to_vec()),
        ("i32", ScalarType::I32, 42i32.to_le_bytes().to_vec()),
        ("i64", ScalarType::I64, 123456i64.to_le_bytes().to_vec()),
        (
            "f16",
            ScalarType::F16,
            half::f16::from_f32(3.14).to_le_bytes().to_vec(),
        ),
        (
            "bf16",
            ScalarType::BF16,
            half::bf16::from_f32(3.14).to_le_bytes().to_vec(),
        ),
        ("f32", ScalarType::F32, 3.14f32.to_le_bytes().to_vec()),
        ("f64", ScalarType::F64, 3.14f64.to_le_bytes().to_vec()),
    ];
    for (name, scalar, data) in cases {
        let ty = Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(scalar),
        });
        let module = build_kernel(&format!("const_{name}"), &[], |m, b, _| {
            let (op, _) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                .result(ty.clone())
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: ty.clone(),
                        shape: vec![],
                        data: data.clone(),
                    }),
                )
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

// =========================================================================
// Comprehensive attribute variation tests
// =========================================================================

// Additional type helpers.
fn tile_f16() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F16),
    })
}
// ---- 1. Rounding mode variations for float arith ops ----

#[test]
fn roundtrip_addf_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("addf_rm{rm}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::AddF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_subf_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("subf_rm{rm}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::SubF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_mulf_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("mulf_rm{rm}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::MulF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_divf_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("divf_rm{rm}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::DivF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_fma_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("fma_rm{rm}_ftz{ftz}"),
                &[tile_f32(), tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::Fma, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .operand(args[2])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_sqrt_all_rounding_modes() {
    for rm in 0..=4 {
        for ftz in [false, true] {
            let module = build_kernel(
                &format!("sqrt_rm{rm}_ftz{ftz}"),
                &[tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::Sqrt, Location::Unknown)
                        .operand(args[0])
                        .result(tile_f32())
                        .attr("rounding_mode", Attribute::i32(rm));
                    if ftz {
                        builder = builder.attr("flush_to_zero", Attribute::i32(1));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

// ---- 2. Rounding mode variations for conversion ops ----

#[test]
fn roundtrip_ftof_all_rounding_modes() {
    for rm in 0..=4 {
        let module = build_kernel(&format!("ftof_rm{rm}"), &[tile_f32()], |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::FToF, Location::Unknown)
                .operand(args[0])
                .result(tile_f16())
                .attr("rounding_mode", Attribute::i32(rm))
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_ftoi_all_rounding_modes() {
    for rm in 0..=4 {
        for sign in 0..=1 {
            let module = build_kernel(
                &format!("ftoi_rm{rm}_s{sign}"),
                &[tile_f32()],
                |m, b, args| {
                    let (op, _) = OpBuilder::new(Opcode::FToI, Location::Unknown)
                        .operand(args[0])
                        .result(tile_i32())
                        .attr("signedness", Attribute::i32(sign))
                        .attr("rounding_mode", Attribute::i32(rm))
                        .build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_itof_all_rounding_modes() {
    for rm in 0..=4 {
        for sign in 0..=1 {
            let module = build_kernel(
                &format!("itof_rm{rm}_s{sign}"),
                &[tile_i32()],
                |m, b, args| {
                    let (op, _) = OpBuilder::new(Opcode::IToF, Location::Unknown)
                        .operand(args[0])
                        .result(tile_f32())
                        .attr("signedness", Attribute::i32(sign))
                        .attr("rounding_mode", Attribute::i32(rm))
                        .build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

// ---- 3. Overflow variations for integer ops ----

#[test]
fn roundtrip_addi_all_overflow() {
    for ov in 0..=3 {
        let module = build_kernel(
            &format!("addi_ov{ov}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("overflow", Attribute::i32(ov))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_subi_all_overflow() {
    for ov in 0..=3 {
        let module = build_kernel(
            &format!("subi_ov{ov}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::SubI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("overflow", Attribute::i32(ov))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_muli_all_overflow() {
    for ov in 0..=3 {
        let module = build_kernel(
            &format!("muli_ov{ov}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::MulI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("overflow", Attribute::i32(ov))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_shli_all_overflow() {
    for ov in 0..=3 {
        let module = build_kernel(
            &format!("shli_ov{ov}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::ShLI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("overflow", Attribute::i32(ov))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

// ---- 4. Signedness variations ----

#[test]
fn roundtrip_maxi_all_signedness() {
    for sign in 0..=1 {
        let module = build_kernel(
            &format!("maxi_s{sign}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::MaxI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("signedness", Attribute::i32(sign))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_mini_all_signedness() {
    for sign in 0..=1 {
        let module = build_kernel(
            &format!("mini_s{sign}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::MinI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("signedness", Attribute::i32(sign))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_remi_all_signedness() {
    for sign in 0..=1 {
        let module = build_kernel(
            &format!("remi_s{sign}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::RemI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("signedness", Attribute::i32(sign))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_shri_all_signedness() {
    for sign in 0..=1 {
        let module = build_kernel(
            &format!("shri_s{sign}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::ShRI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i32())
                    .attr("signedness", Attribute::i32(sign))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_exti_all_signedness() {
    for sign in 0..=1 {
        let tile_i64 = Type::Tile(TileType {
            shape: vec![128],
            element_type: TileElementType::Scalar(ScalarType::I64),
        });
        let module = build_kernel(&format!("exti_s{sign}"), &[tile_i32()], |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::ExtI, Location::Unknown)
                .operand(args[0])
                .result(tile_i64.clone())
                .attr("signedness", Attribute::i32(sign))
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_cmpi_all_signedness() {
    for sign in 0..=1 {
        let module = build_kernel(
            &format!("cmpi_s{sign}"),
            &[tile_i32(), tile_i32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_i1())
                    .attr("comparison_predicate", Attribute::i32(0))
                    .attr("signedness", Attribute::i32(sign))
                    .build(m);
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

// ---- 5. DivI rounding x signedness matrix ----

#[test]
fn roundtrip_divi_all_rounding_signedness() {
    for rounding in 0..=3 {
        for sign in 0..=1 {
            let module = build_kernel(
                &format!("divi_r{rounding}_s{sign}"),
                &[tile_i32(), tile_i32()],
                |m, b, args| {
                    let (op, _) = OpBuilder::new(Opcode::DivI, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_i32())
                        .attr("signedness", Attribute::i32(sign))
                        .attr("rounding", Attribute::i32(rounding))
                        .build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

// ---- 6. CmpF ordering x predicate variations ----

#[test]
fn roundtrip_cmpf_ordering_predicate_matrix() {
    // predicates: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
    // orderings: 0=ordered, 1=unordered
    for ordering in 0..=1 {
        for pred in 0..=5 {
            let module = build_kernel(
                &format!("cmpf_o{ordering}_p{pred}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let (op, _) = OpBuilder::new(Opcode::CmpF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_i1())
                        .attr("comparison_predicate", Attribute::i32(pred))
                        .attr("comparison_ordering", Attribute::i32(ordering))
                        .build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

// ---- 7. MaxF/MinF propagate_nan x flush_to_zero matrix ----

#[test]
fn roundtrip_maxf_propagate_nan_ftz_matrix() {
    for pn in 0..=1 {
        for ftz in 0..=1 {
            let module = build_kernel(
                &format!("maxf_pn{pn}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::MaxF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32());
                    if pn != 0 {
                        builder = builder.attr("propagate_nan", Attribute::i32(pn));
                    }
                    if ftz != 0 {
                        builder = builder.attr("flush_to_zero", Attribute::i32(ftz));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

#[test]
fn roundtrip_minf_propagate_nan_ftz_matrix() {
    for pn in 0..=1 {
        for ftz in 0..=1 {
            let module = build_kernel(
                &format!("minf_pn{pn}_ftz{ftz}"),
                &[tile_f32(), tile_f32()],
                |m, b, args| {
                    let mut builder = OpBuilder::new(Opcode::MinF, Location::Unknown)
                        .operand(args[0])
                        .operand(args[1])
                        .result(tile_f32());
                    if pn != 0 {
                        builder = builder.attr("propagate_nan", Attribute::i32(pn));
                    }
                    if ftz != 0 {
                        builder = builder.attr("flush_to_zero", Attribute::i32(ftz));
                    }
                    let (op, _) = builder.build(m);
                    append_op(m, b, op);
                },
            );
            assert_roundtrip(&module);
        }
    }
}

// ---- 8. AtomicRMW mode variations ----

#[test]
fn roundtrip_atomic_rmw_all_modes() {
    // Modes: 0=add, 1=and, 2=max, 3=min, 4=or, 5=xor, 6=fadd, 7=fmax, 8=xchg
    for mode in 0..=8 {
        let module = build_kernel(
            &format!("armw_mode{mode}"),
            &[tile_ptr_f32(), tile_f32()],
            |m, b, args| {
                let (op, _) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
                    .operand(args[0])
                    .operand(args[1])
                    .result(tile_f32())
                    .result(token())
                    .attr("memory_ordering_semantics", Attribute::i32(0))
                    .attr("memory_scope", Attribute::i32(0))
                    .attr("mode", Attribute::i32(mode))
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
                append_op(m, b, op);
            },
        );
        assert_roundtrip(&module);
    }
}

// ---- 9. Exp2/Rsqrt flush_to_zero on and off ----

#[test]
fn roundtrip_exp2_ftz_on_and_off() {
    for ftz in [false, true] {
        let module = build_kernel(&format!("exp2_ftz{ftz}"), &[tile_f32()], |m, b, args| {
            let mut builder = OpBuilder::new(Opcode::Exp2, Location::Unknown)
                .operand(args[0])
                .result(tile_f32());
            if ftz {
                builder = builder.attr("flush_to_zero", Attribute::i32(1));
            }
            let (op, _) = builder.build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

#[test]
fn roundtrip_rsqrt_ftz_on_and_off() {
    for ftz in [false, true] {
        let module = build_kernel(&format!("rsqrt_ftz{ftz}"), &[tile_f32()], |m, b, args| {
            let mut builder = OpBuilder::new(Opcode::Rsqrt, Location::Unknown)
                .operand(args[0])
                .result(tile_f32());
            if ftz {
                builder = builder.attr("flush_to_zero", Attribute::i32(1));
            }
            let (op, _) = builder.build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

// ---- 10. Constants across ALL scalar types including f8 and tf32 ----

#[test]
fn roundtrip_constants_f8_and_tf32_types() {
    let cases: Vec<(&str, ScalarType, Vec<u8>)> = vec![
        // F8E4M3FN: 1-byte float, use raw byte 0x3C (1.0 in e4m3fn)
        ("f8e4m3fn", ScalarType::F8E4M3FN, vec![0x3C]),
        // F8E5M2: 1-byte float, use raw byte 0x3C (1.0 in e5m2)
        ("f8e5m2", ScalarType::F8E5M2, vec![0x3C]),
        // TF32: 4-byte float (same width as f32), use 3.14f32 bytes
        ("tf32", ScalarType::TF32, 3.14f32.to_le_bytes().to_vec()),
    ];
    for (name, scalar, data) in cases {
        let ty = Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(scalar),
        });
        let module = build_kernel(&format!("const_{name}"), &[], |m, b, _| {
            let (op, _) = OpBuilder::new(Opcode::Constant, Location::Unknown)
                .result(ty.clone())
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: ty.clone(),
                        shape: vec![],
                        data: data.clone(),
                    }),
                )
                .build(m);
            append_op(m, b, op);
        });
        assert_roundtrip(&module);
    }
}

// --- Partition view with padding + 2D tensor view ---

#[test]
fn roundtrip_make_partition_view_with_padding() {
    let tv = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC],
        strides: vec![1],
    });
    let pv = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: Some(PaddingValue::Zero),
    });
    let module = build_kernel("pv_pad", std::slice::from_ref(&tv), |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
            .operand(args[0])
            .result(pv.clone())
            .build(m);
        append_op(m, b, op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_make_tensor_view_2d() {
    let tv = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC, DYNAMIC],
        strides: vec![DYNAMIC, 1],
    });
    let module = build_kernel(
        "tv2d",
        &[ptr_f32(), scalar_i32(), scalar_i32(), scalar_i32()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .result(tv.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(2),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
        },
    );
    assert_roundtrip(&module);
}
