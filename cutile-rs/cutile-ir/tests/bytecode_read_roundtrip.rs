/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode read round-trip tests.
//!
//! `write(m) -> read -> m' -> write(m')` must be byte-identical, and `m'` must
//! pass the same verifiers the writer runs before handing bytes to `tileiras`.

#![allow(clippy::approx_constant)]

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion, Opcode};
use cutile_ir::ir::{
    Attribute, BlockId, DenseElements, FuncType, Global, Location, Module, OptimizationHints,
    PaddingValue, PartitionViewType, PointerType, ScalarType, SymbolVisibility, TensorViewType,
    TileElementType, TileType, Type, Value, DYNAMIC,
};
use cutile_ir::read_bytecode_versioned;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert the full read/write round-trip at every supported bytecode version.
/// Bytes in should be identical to bytes out, with the rebuilt module passing
/// the standard verifiers.
fn assert_roundtrip(module: &Module) {
    for version in BytecodeVersion::SUPPORTED {
        assert_roundtrip_at(module, version);
    }
}

/// Assert the full read/write round-trip at `version`.
fn assert_roundtrip_at(module: &Module, version: BytecodeVersion) {
    let bytes = write_bytecode_version(module, version)
        .unwrap_or_else(|e| panic!("bytecode write at {version} failed: {e}"));
    let (rebuilt, read_version) = read_bytecode_versioned(&bytes)
        .unwrap_or_else(|e| panic!("bytecode read at {version} failed: {e}"));
    assert_eq!(
        read_version, version,
        "reader must report the header version ({version})"
    );
    rebuilt
        .verify_dominance()
        .unwrap_or_else(|e| panic!("rebuilt module dominance at {version}: {e}"));
    rebuilt
        .verify_bytecode_indices()
        .unwrap_or_else(|e| panic!("rebuilt module bytecode indices at {version}: {e}"));
    let bytes2 = write_bytecode_version(&rebuilt, version)
        .unwrap_or_else(|e| panic!("re-write of rebuilt module at {version} failed: {e}"));
    assert_eq!(
        bytes2.len(),
        bytes.len(),
        "round-trip bytes at {version} diverged in length ({} vs {})\n--- original mlir ---\n{}\n--- rebuilt mlir ---\n{}",
        bytes2.len(),
        bytes.len(),
        module.to_mlir_text(),
        rebuilt.to_mlir_text()
    );
    if bytes2 != bytes {
        // Find the first divergent byte for a useful diagnostic.
        let diff = bytes.iter().zip(&bytes2).position(|(a, b)| a != b).unwrap();
        let lo = diff.saturating_sub(8);
        let hi = (diff + 16).min(bytes.len()).min(bytes2.len());
        let hex = |v: &[u8]| {
            v[lo..hi]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        panic!(
            "round-trip bytes at {version} diverged at byte {diff} of {}/{}\norig[{}..{}]: {}\nre  [{}..{}]: {}\n--- original mlir ---\n{}\n--- rebuilt mlir ---\n{}",
            bytes.len(),
            bytes2.len(),
            lo,
            hi,
            hex(&bytes),
            lo,
            hi,
            hex(&bytes2),
            module.to_mlir_text(),
            rebuilt.to_mlir_text()
        );
    }
}

/// Build a module with a single entry function containing the given ops.
fn build_kernel(
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    build_kernel_with_entry(name, arg_types, &Location::Unknown, None, None, build_body)
}

/// Build a single-entry module, attaching `di_name` and `optimization_hints`
/// in the frontend's attribute order
/// `[sym_name, di_name?, function_type, optimization_hints?]` so that
/// string-interning order is kept correctly.
fn build_kernel_with_entry(
    name: &str,
    arg_types: &[Type],
    entry_loc: &Location,
    di_name: Option<String>,
    hints: Option<Attribute>,
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build_body(&mut module, block_id, &args);
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
    let mut builder = OpBuilder::new(Opcode::Entry, entry_loc.clone())
        .attr("sym_name", Attribute::String(name.into()));
    if let Some(dn) = di_name {
        builder = builder.attr("di_name", Attribute::String(dn));
    }
    builder = builder.attr("function_type", Attribute::Type(func_type));
    if let Some(h) = hints {
        builder = builder.attr("optimization_hints", h);
    }
    builder = builder.region(region_id);
    let (entry, _) = builder.build(&mut module);
    module.functions.push(entry);
    module
}

// ---------------------------------------------------------------------------
// Type builders
// ---------------------------------------------------------------------------

fn tile_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F32),
    })
}

fn tile_i32() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::I32),
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

fn tile_ptr_f32() -> Type {
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    })
}

fn tile_i1() -> Type {
    Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::I1),
    })
}

fn token() -> Type {
    Type::Token
}

fn tensor_view_f32() -> Type {
    Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC, 128],
        strides: vec![128, 1],
    })
}

fn partition_view_f32() -> Type {
    Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC, 128],
            strides: vec![128, 1],
        },
        dim_map: vec![0, 1],
        padding_value: Some(PaddingValue::Nan),
    })
}

/// `cuda_tile.constant` with dense data (the frontend's form).
fn dense_const(module: &mut Module, block: BlockId, ty: Type, data: Vec<u8>) -> Value {
    let (op, results) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(ty.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: ty,
                shape: vec![],
                data,
            }),
        )
        .build(module);
    append_op(module, block, op);
    results[0]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_empty_module() {
    let module = Module::new("empty");
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_basic_arith_kernel() {
    let module = build_kernel("basic", &[tile_f32(), tile_i32()], |m, b, a| {
        let zero = dense_const(m, b, scalar_i32(), 0i32.to_le_bytes().to_vec());
        let (cmp, cmp_res) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .result(scalar_i32())
            .attr("comparison_predicate", Attribute::i32(2))
            .attr("signedness", Attribute::i32(1))
            .operand(a[1])
            .operand(zero)
            .build(m);
        append_op(m, b, cmp);
        let (sel, sel_res) = OpBuilder::new(Opcode::Select, Location::Unknown)
            .result(tile_f32())
            .operand(cmp_res[0])
            .operand(a[0])
            .operand(a[0])
            .build(m);
        append_op(m, b, sel);
        let (neg, neg_res) = OpBuilder::new(Opcode::NegI, Location::Unknown)
            .result(tile_i32())
            .attr("overflow", Attribute::i32(3))
            .operand(a[1])
            .build(m);
        append_op(m, b, neg);
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .result(tile_i32())
            .attr("overflow", Attribute::i32(0))
            .operand(a[1])
            .operand(neg_res[0])
            .build(m);
        append_op(m, b, add);
        let (fadd, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(5))
            .operand(a[0])
            .operand(sel_res[0])
            .build(m);
        append_op(m, b, fadd);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_for_loop_with_carry() {
    let module = build_kernel("for_loop", &[tile_f32()], |m, b, a| {
        let zero_i = dense_const(m, b, scalar_i32(), 0i32.to_le_bytes().to_vec());
        let one_i = dense_const(m, b, scalar_i32(), 1i32.to_le_bytes().to_vec());
        let bound = dense_const(m, b, scalar_i32(), 16i32.to_le_bytes().to_vec());
        // for(%iv, %acc = %iv0, %acc0 in 0..16 step 1)
        let (loop_region, loop_block, loop_args) =
            build_single_block_region(m, &[tile_i32(), tile_f32()]);
        let (body_add, body_add_res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(5))
            .operand(a[0])
            .operand(loop_args[1])
            .build(m);
        append_op(m, loop_block, body_add);
        let (yield_op, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operands([loop_args[0], body_add_res[0]].iter().copied())
            .build(m);
        append_op(m, loop_block, yield_op);
        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .result(tile_f32())
            .operands([zero_i, bound, one_i, a[0]].iter().copied())
            .region(loop_region)
            .build(m);
        append_op(m, b, for_op);
        let _ = for_op;
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_if_else_and_loop() {
    let module = build_kernel("if_loop", &[tile_i32()], |m, b, a| {
        let zero = dense_const(m, b, scalar_i32(), 0i32.to_le_bytes().to_vec());
        let one = dense_const(m, b, scalar_i32(), 1i32.to_le_bytes().to_vec());
        let (cmp, cmp_res) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .result(scalar_i32())
            .attr("comparison_predicate", Attribute::i32(0))
            .attr("signedness", Attribute::i32(0))
            .operand(a[0])
            .operand(zero)
            .build(m);
        append_op(m, b, cmp);

        // if (%cond) -> tile<i32>
        let (if_region, if_block, if_args) = build_single_block_region(m, &[tile_i32()]);
        let (y, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operands(if_args.iter().copied())
            .build(m);
        append_op(m, if_block, y);
        let (else_region, else_block, _) = build_single_block_region(m, &[tile_i32()]);
        let (y2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(one)
            .build(m);
        append_op(m, else_block, y2);
        let (if_op, if_res) = OpBuilder::new(Opcode::If, Location::Unknown)
            .result(tile_i32())
            .operand(cmp_res[0])
            .region(if_region)
            .region(else_region)
            .build(m);
        append_op(m, b, if_op);

        // loop with carry + break
        let (loop_region, loop_block, loop_args) = build_single_block_region(m, &[tile_i32()]);
        let (cmp2, cmp2_res) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .result(scalar_i32())
            .attr("comparison_predicate", Attribute::i32(2))
            .attr("signedness", Attribute::i32(0))
            .operand(loop_args[0])
            .operand(one)
            .build(m);
        append_op(m, loop_block, cmp2);
        let (break_op, _) = OpBuilder::new(Opcode::Break, Location::Unknown)
            .operand(if_res[0])
            .build(m);
        append_op(m, loop_block, break_op);
        let (continue_op, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
            .operand(one)
            .build(m);
        append_op(m, loop_block, continue_op);
        let (loop_op, _) = OpBuilder::new(Opcode::Loop, Location::Unknown)
            .result(tile_i32())
            .operand(if_res[0])
            .region(loop_region)
            .build(m);
        append_op(m, b, loop_op);
        let _ = (cmp2_res, break_op, continue_op, loop_op);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_entry_with_optimization_hints() {
    // This test takes some extra care wrt optimization hints decoding.

    // Hint-carrying kernel.
    let hints = Attribute::OptimizationHints(OptimizationHints {
        entries: vec![(
            "sm_100a".to_owned(),
            vec![
                ("num_cta_in_cga".to_owned(), Attribute::i32(2)),
                (
                    "max_dynamic_shared_memory_bytes".to_owned(),
                    Attribute::i32(232448),
                ),
            ],
        )],
    });
    let entry_loc = Location::FileLineCol {
        filename: "kernel.cu".to_owned(),
        line: 42,
        column: 5,
    };
    let mut module = build_kernel_with_entry(
        "hinted",
        &[],
        &entry_loc,
        Some("hinted_kernel".to_owned()),
        Some(hints),
        |m, b, _| {
            // Empty body: the entry has no args, so the auto-return handles it.
            let _ = (m, b);
        },
    );
    // Plain function.
    let func_type = Type::Func(FuncType {
        inputs: vec![tile_i32()],
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, &[tile_i32()]);
    let (use_op, use_res) = OpBuilder::new(Opcode::NegI, Location::Unknown)
        .result(tile_i32())
        .operand(args[0])
        .build(&mut module);
    append_op(&mut module, block_id, use_op);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown)
        .operand(use_res[0])
        .build(&mut module);
    append_op(&mut module, block_id, ret);
    let plain_loc = Location::FileLineCol {
        filename: "kernel.cu".to_owned(),
        line: 99,
        column: 1,
    };
    let (entry, _) = OpBuilder::new(Opcode::Entry, plain_loc)
        .attr("sym_name", Attribute::String("plain".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_views_and_reduce() {
    let module = build_kernel("views", &[tensor_view_f32()], |m, b, a| {
        let (mv, mv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
            .result(tensor_view_f32())
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(0),
                    Attribute::i32(0),
                ]),
            )
            .operand(a[0])
            .build(m);
        append_op(m, b, mv);
        let (pv, pv_res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
            .result(partition_view_f32())
            .operand(mv_res[0])
            .build(m);
        append_op(m, b, pv);

        // load_view_tko with an explicit token operand
        let (tok, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(Type::Token)
            .build(m);
        append_op(m, b, tok);
        let c0 = dense_const(m, b, scalar_i32(), 0i32.to_le_bytes().to_vec());
        let (ld, ld_res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
            .result(tile_f32())
            .result(Type::Token)
            .attr(
                "memory_ordering_semantics",
                Attribute::i32(2), // relaxed
            )
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::i32(1),
                    Attribute::i32(2),
                    Attribute::i32(1),
                ]),
            )
            .operand(pv_res[0])
            .operand(c0)
            .operand(c0)
            .operand(tok_res[0])
            .build(m);
        append_op(m, b, ld);

        // reduce (sum) over the last dim
        let (reduce_region, reduce_block, reduce_args) =
            build_single_block_region(m, &[tile_f32(), tile_f32()]);
        let (acc_add, acc_add_res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(5))
            .operand(reduce_args[0])
            .operand(reduce_args[1])
            .build(m);
        append_op(m, reduce_block, acc_add);
        let (ry, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(acc_add_res[0])
            .build(m);
        append_op(m, reduce_block, ry);
        let (reduce, _) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
            .result(scalar_f32())
            .attr("dim", Attribute::i32(1))
            .attr(
                "identities",
                Attribute::Array(vec![Attribute::Float(0.0, Type::Scalar(ScalarType::F32))]),
            )
            .operand(ld_res[0])
            .region(reduce_region)
            .build(m);
        append_op(m, b, reduce);
    });
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_v13_2() {
    let module = build_kernel("v132", &[tile_f32()], |m, b, a| {
        let (exp, _) = OpBuilder::new(Opcode::Exp, Location::Unknown)
            .result(tile_f32())
            .operand(a[0])
            .build(m);
        append_op(m, b, exp);
    });
    assert_roundtrip(&module);
}

// ---------------------------------------------------------------------------
// Globals, pointer ops, atomics, tokens
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_globals_and_get_global() {
    let mut module = Module::new("test");
    // A frontend-shaped global: tile<1 x f32>, shape [1].
    let value_ty = Type::Tile(TileType {
        shape: vec![1],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    module.globals.push(Global {
        sym_name: "my_global".to_owned(),
        value: DenseElements {
            element_type: value_ty,
            shape: vec![1],
            data: 1.5f32.to_le_bytes().to_vec(),
        },
        alignment: 4,
        constant: false,
        symbol_visibility: SymbolVisibility::Public,
    });

    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);
    // get_global -> ptr<f32>, then load it.
    let (gg, gg_res) = OpBuilder::new(Opcode::GetGlobal, Location::Unknown)
        .result(tile_ptr_f32())
        .attr("name", Attribute::String("my_global".into()))
        .build(&mut module);
    append_op(&mut module, block_id, gg);
    let (ld, _) = OpBuilder::new(Opcode::LoadPtrTko, Location::Unknown)
        .operand(gg_res[0])
        .result(scalar_f32())
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
        .build(&mut module);
    append_op(&mut module, block_id, ld);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("global_kernel".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);

    assert_roundtrip(&module);
}

#[test]
fn roundtrip_ptr_store_and_atomics() {
    let module = build_kernel(
        "ptr_atomic",
        &[tile_ptr_f32(), tile_f32(), tile_i1(), token()],
        |m, b, a| {
            // store_ptr_tko with a mask
            let (st, st_res) = OpBuilder::new(Opcode::StorePtrTko, Location::Unknown)
                .operand(a[0])
                .operand(a[1])
                .operand(a[2])
                .operand(a[3])
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
            append_op(m, b, st);
            // atomic_rmw_tko (add)
            let (rmw, _) = OpBuilder::new(Opcode::AtomicRMW, Location::Unknown)
                .operand(a[0])
                .operand(a[1])
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
            append_op(m, b, rmw);
            let _ = st_res;
        },
    );
    assert_roundtrip(&module);
}

#[test]
fn roundtrip_join_tokens() {
    let module = build_kernel("jtok", &[token()], |m, b, a| {
        let (t2, t2_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(token())
            .build(m);
        append_op(m, b, t2);
        let (join, _) = OpBuilder::new(Opcode::JoinTokens, Location::Unknown)
            .result(token())
            .operand(a[0])
            .operand(t2_res[0])
            .build(m);
        append_op(m, b, join);
    });
    assert_roundtrip(&module);
}

// ---------------------------------------------------------------------------
// Real source locations on every op (the frontend always assigns them)
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_ops_with_file_line_col_locations() {
    let loc = |line: u32| Location::FileLineCol {
        filename: "kernel.cu".to_owned(),
        line,
        column: 3,
    };
    let entry_loc = loc(10);
    let module = build_kernel_with_entry(
        "located",
        &[tile_f32()],
        &entry_loc,
        Some("located_kernel".to_owned()),
        None,
        |m, b, a| {
            let (c, c_res) = OpBuilder::new(Opcode::Constant, loc(11))
                .result(scalar_f32())
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: scalar_f32(),
                        shape: vec![],
                        data: 2.0f32.to_le_bytes().to_vec(),
                    }),
                )
                .build(m);
            append_op(m, b, c);
            let (cmp, cmp_res) = OpBuilder::new(Opcode::CmpF, loc(12))
                .result(tile_i1())
                .attr("comparison_predicate", Attribute::i32(0))
                .attr("comparison_ordering", Attribute::i32(0))
                .operand(a[0])
                .operand(a[0])
                .build(m);
            append_op(m, b, cmp);
            let _ = (c_res, cmp_res);
        },
    );
    assert_roundtrip(&module);
}

// ---------------------------------------------------------------------------
// Robustness: malformed input must error, never panic
// ---------------------------------------------------------------------------

#[test]
fn reader_rejects_malformed_input() {
    // Empty input.
    assert!(cutile_ir::read_bytecode(&[]).is_err());
    // Bad magic.
    let bad = vec![0u8; 16];
    assert!(cutile_ir::read_bytecode(&bad).is_err());
    // Truncated header (fewer than 12 bytes).
    assert!(cutile_ir::read_bytecode(&[0x7F, b'T', b'i']).is_err());
    // A valid header but no EndOfBytecode / garbage sections.
    let mut v = vec![
        0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00, 13, 3, 0, 0,    // header
        0x7F, // unknown/oversized section id byte
    ];
    v.resize(32, 0);
    assert!(cutile_ir::read_bytecode(&v).is_err());
    // Random non-deterministic garbage: must not panic.
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;
    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as u8
    };
    for _ in 0..50 {
        let garbage: Vec<u8> = (0..64).map(|_| rand()).collect();
        let _ = cutile_ir::read_bytecode(&garbage); // just must not panic
    }
}

#[test]
fn reader_roundtrip_is_stable_under_repeated_reads() {
    let module = build_kernel("stable", &[tile_f32()], |m, b, a| {
        let (add, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .result(tile_f32())
            .attr("rounding_mode", Attribute::i32(5))
            .operand(a[0])
            .operand(a[0])
            .build(m);
        append_op(m, b, add);
    });
    let b1 = write_bytecode_version(&module, BytecodeVersion::V13_3).unwrap();
    for _ in 0..3 {
        let (m2, ver) = read_bytecode_versioned(&b1).unwrap();
        let b2 = write_bytecode_version(&m2, ver).unwrap();
        assert_eq!(b2, b1);
    }
}

#[test]
fn roundtrip_v13_4_features() {
    let f8_tile = Type::Tile(TileType {
        shape: vec![128],
        element_type: TileElementType::Scalar(ScalarType::F8E5M3FNU),
    });
    let mut module = build_kernel(
        "v134",
        &[tile_f32(), tile_f32(), tile_i32(), token(), f8_tile],
        |m, b, a| {
            let (fpowi, _) = OpBuilder::new(Opcode::FPowI, Location::Unknown)
                .result(tile_f32())
                .operand(a[0])
                .operand(a[2])
                .build(m);
            append_op(m, b, fpowi);
            let (insert, _) = OpBuilder::new(Opcode::Insert, Location::Unknown)
                .result(tile_f32())
                .operand(a[0])
                .operand(a[1])
                .operand(a[2])
                .build(m);
            append_op(m, b, insert);
            let (gdc, _) = OpBuilder::new(Opcode::GdcLaunchDependentsTko, Location::Unknown)
                .result(token())
                .operand(a[3])
                .build(m);
            append_op(m, b, gdc);
            let (gdc_wait, _) = OpBuilder::new(Opcode::GdcWaitTko, Location::Unknown)
                .result(token())
                .build(m);
            append_op(m, b, gdc_wait);
            let (fence, _) = OpBuilder::new(Opcode::MemoryFenceAliasTko, Location::Unknown)
                .result(token())
                .operand(a[3])
                .build(m);
            append_op(m, b, fence);
        },
    );
    module.producer = Some("cutile-ir 13.4 roundtrip".into());
    assert_roundtrip_at(&module, BytecodeVersion::V13_4);
}
