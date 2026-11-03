/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode structural validation tests.
//!
//! These tests build IR with various patterns, write bytecode, and validate
//! the output using:
//! 1. Module-level `verify_bytecode_indices()` (value numbering consistency)
//! 2. tileiras parsing (byte-level format correctness)
//!
//! Run with: cargo test -p tile-ir --test bytecode_validate -- --nocapture

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

// =========================================================================
// Helpers
// =========================================================================

fn i32_ty() -> Type {
    Type::Scalar(ScalarType::I32)
}

fn tile_i32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I32),
        shape: vec![],
    })
}

fn tile_f32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![],
    })
}

fn tile_i1() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I1),
        shape: vec![],
    })
}

fn token_ty() -> Type {
    Type::Token
}

fn const_i32(module: &mut Module, block: BlockId, val: i64) -> Value {
    let data = (val as i32).to_le_bytes().to_vec();
    let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: tile_i32(),
                shape: vec![],
                data,
            }),
        )
        .result(tile_i32())
        .build(module);
    append_op(module, block, op);
    res[0]
}

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

/// Validate a module: dominance, bytecode indices, write bytecode, and
/// optionally run tileiras to check byte-level format.
fn validate_module(module: &Module) {
    module.verify_dominance().expect("dominance check failed");
    module
        .verify_bytecode_indices()
        .expect("bytecode index check failed");

    let bc = write_test_bytecode(module).expect("write_bytecode failed");

    // Verify our own decoder can parse it.
    cutile_ir::decode_bytecode(&bc).expect("our decoder rejected the bytecode");

    // Try tileiras if available (it may not be installed in CI).
    run_tileiras(&bc, &module.name);
}

fn run_tileiras(bc: &[u8], name: &str) {
    let tmp = std::env::temp_dir().join(format!(
        "tile_ir_test_{}_{:?}.bc",
        name,
        std::thread::current().id()
    ));
    std::fs::write(&tmp, bc).expect("write bc file");
    match std::process::Command::new(tileiras_binary())
        .arg("--gpu-name")
        .arg("sm_120")
        .arg("-o")
        .arg("/dev/null")
        .arg(tmp.to_str().unwrap())
        .output()
    {
        Ok(out) => {
            std::fs::remove_file(&tmp).ok();
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                panic!("tileiras rejected bytecode for module '{name}':\n{stderr}");
            }
        }
        Err(e)
            if e.kind() == std::io::ErrorKind::NotFound
                && std::env::var_os("CUTILE_TILEIRAS_PATH")
                    .filter(|v| !v.is_empty())
                    .is_none() =>
        {
            // tileiras not available -- skip byte-level check
            std::fs::remove_file(&tmp).ok();
        }
        Err(e) => panic!("cannot run tileiras: {e}"),
    }
}

fn tileiras_binary() -> std::path::PathBuf {
    cutile_ir::toolchain::tileiras_from_env()
}

// Send the same format the JIT would emit, including on pre--list-versions
// assemblers. A writer-only run may omit tileiras; a broken explicit override
// must not quietly turn assembler validation off.
fn write_test_bytecode(module: &Module) -> cutile_ir::Result<Vec<u8>> {
    use cutile_ir::bytecode::BytecodeVersion;
    static VERSION: std::sync::OnceLock<BytecodeVersion> = std::sync::OnceLock::new();
    let version = *VERSION.get_or_init(|| {
        let requested = std::env::var_os("CUTILE_BYTECODE_VERSION").filter(|v| !v.is_empty());
        match cutile_ir::toolchain::negotiate_bytecode_version(
            &tileiras_binary(),
            requested.as_deref(),
        ) {
            Ok(version) => version,
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && std::env::var_os("CUTILE_TILEIRAS_PATH")
                        .filter(|v| !v.is_empty())
                        .is_none()
                    && requested.is_none() =>
            {
                BytecodeVersion::CURRENT
            }
            Err(e) => panic!("cannot select tileiras bytecode version: {e}"),
        }
    });
    cutile_ir::bytecode::write_bytecode_version(module, version)
}

// =========================================================================
// Test cases
// =========================================================================

#[test]
fn simple_arithmetic() {
    let module = build_kernel("simple_arith", &[tile_i32(), tile_i32()], |m, blk, args| {
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, add);
    });
    validate_module(&module);
}

#[test]
fn for_loop_with_parent_scope_refs() {
    // A for-loop whose body uses values from the parent scope.
    let module = build_kernel("for_parent_ref", &[tile_i32()], |m, blk, args| {
        let parent_val = args[0];
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // for %iv = lb to ub step step { addi parent_val, %iv }
        let (body_region, body_blk, body_args) = build_single_block_region(m, &[tile_i32()]);

        // body: use parent_val + iv
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(parent_val)
            .operand(body_args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add);

        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    validate_module(&module);
}

#[test]
fn for_loop_with_iter_args() {
    // for-loop with carried values (iter args).
    let module = build_kernel("for_iter_args", &[tile_i32()], |m, blk, args| {
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);
        let init = args[0];

        // for %iv, %acc = lb to ub step step iter(%init) { continue %acc + %iv }
        let (body_region, body_blk, body_args) =
            build_single_block_region(m, &[tile_i32(), tile_i32()]); // iv, acc

        let (add, add_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(body_args[1]) // acc
            .operand(body_args[0]) // iv
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add);

        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .operand(init) // init values
            .result(tile_i32()) // carried result
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    validate_module(&module);
}

#[test]
fn for_loop_with_token_iter_arg() {
    // Feasibility probe for token-ordering loop-carry: a for-loop that carries a
    // `!cuda_tile.token` as an iter-arg (seeded by make_token, yielded back). If
    // this validates through tileiras, a token is a legal loop-carried value —
    // which is what the mutable-store token threading needs (carry the token, not
    // the partition_view, which is a restricted iter-arg type).
    let module = build_kernel("for_token_iter_arg", &[], |m, blk, _args| {
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // Seed token.
        let (tok_op, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(token_ty())
            .build(m);
        append_op(m, blk, tok_op);
        let init = tok_res[0];

        // for %iv, %tok = lb to ub step step iter(%init) { continue %tok }
        // The body threads the carried token straight through (identity), the
        // minimal shape that puts a token on the loop's block-arg + result.
        let (body_region, body_blk, body_args) =
            build_single_block_region(m, &[tile_i32(), token_ty()]); // iv, tok

        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
            .operand(body_args[1]) // yield the carried token
            .build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .operand(init)
            .result(token_ty()) // carried token result
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    validate_module(&module);
}

#[test]
fn nested_for_loops() {
    // Outer for containing an inner for.
    let module = build_kernel("nested_for", &[tile_i32()], |m, blk, args| {
        let parent_val = args[0];
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // Outer for
        let (outer_region, outer_blk, outer_args) = build_single_block_region(m, &[tile_i32()]);

        // Inner for (uses outer iv and parent_val)
        let (inner_region, inner_blk, inner_args) = build_single_block_region(m, &[tile_i32()]);

        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(parent_val) // from grandparent
            .operand(outer_args[0]) // from outer loop
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, inner_blk, add);

        let (add2, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(inner_args[0]) // inner iv
            .operand(outer_args[0]) // outer iv
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, inner_blk, add2);

        let (cont_inner, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, inner_blk, cont_inner);

        let (inner_for, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(inner_region)
            .build(m);
        append_op(m, outer_blk, inner_for);

        let (cont_outer, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, outer_blk, cont_outer);

        let (outer_for, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(outer_region)
            .build(m);
        append_op(m, blk, outer_for);
    });
    validate_module(&module);
}

#[test]
fn if_else_with_yields() {
    // if-else that yields values.
    let module = build_kernel("if_yields", &[tile_i32()], |m, blk, args| {
        // condition
        let one = const_i32(m, blk, 1);
        let (cmp, cmp_res) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .operand(args[0])
            .operand(one)
            .attr("comparison_predicate", Attribute::Integer(0, i32_ty()))
            .attr("signedness", Attribute::Integer(0, i32_ty()))
            .result(tile_i1())
            .build(m);
        append_op(m, blk, cmp);

        // then
        let (then_region, then_blk, _) = build_single_block_region(m, &[]);
        let (add, add_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, then_blk, add);
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, then_blk, yld);

        // else
        let (else_region, else_blk, _) = build_single_block_region(m, &[]);
        let (yld2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(args[0])
            .build(m);
        append_op(m, else_blk, yld2);

        // if
        let (if_op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
            .operand(cmp_res[0])
            .result(tile_i32())
            .region(then_region)
            .region(else_region)
            .build(m);
        append_op(m, blk, if_op);
    });
    validate_module(&module);
}

#[test]
fn reduce_with_combiner() {
    // Reduce op with a combiner region.
    // Entry args must be scalar tiles, so we reshape + broadcast.
    let tile_1_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![1],
    });
    let tile_8_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![8],
    });

    let module = build_kernel("reduce_test", &[tile_f32()], |m, blk, args| {
        // Reshape scalar to tile<1xf32>, then broadcast to tile<8xf32>
        let (rs_op, rs_res) = OpBuilder::new(Opcode::Reshape, Location::Unknown)
            .operand(args[0])
            .result(tile_1_f32.clone())
            .build(m);
        append_op(m, blk, rs_op);
        let (bc_op, bc_res) = OpBuilder::new(Opcode::Broadcast, Location::Unknown)
            .operand(rs_res[0])
            .result(tile_8_f32.clone())
            .build(m);
        append_op(m, blk, bc_op);
        let args = &[bc_res[0]];
        // Combiner region: two scalar args -> yield their sum
        let (combiner_region, combiner_blk, combiner_args) =
            build_single_block_region(m, &[tile_f32(), tile_f32()]);
        let (add, add_res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(combiner_args[0])
            .operand(combiner_args[1])
            .attr("rounding_mode", Attribute::Integer(0, i32_ty()))
            .result(tile_f32())
            .build(m);
        append_op(m, combiner_blk, add);
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, combiner_blk, yld);

        // reduce
        let (red, _) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
            .operand(args[0])
            .attr("dim", Attribute::Integer(0, i32_ty()))
            .attr(
                "identities",
                Attribute::Array(vec![Attribute::Float(0.0, Type::Scalar(ScalarType::F32))]),
            )
            .result(tile_f32())
            .region(combiner_region)
            .build(m);
        append_op(m, blk, red);
    });
    validate_module(&module);
}

#[test]
fn many_ops_after_nested_region() {
    // After a for-loop, subsequent ops must use correct value indices.
    // This pattern catches rollback bugs.
    let module = build_kernel("after_nested", &[tile_i32()], |m, blk, args| {
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // for-loop
        let (body_region, body_blk, body_args) = build_single_block_region(m, &[tile_i32()]);
        let (add_inner, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(body_args[0])
            .operand(args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add_inner);
        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);

        // After the for-loop: operations that use pre-loop values.
        // If rollback is wrong, these operand indices will be off.
        let c1 = const_i32(m, blk, 42);
        let (add_after, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(c1)
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, add_after);

        let (mul_after, _) = OpBuilder::new(Opcode::MulI, Location::Unknown)
            .operand(args[0])
            .operand(c1)
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, mul_after);
    });
    validate_module(&module);
}

#[test]
fn load_store_with_tokens() {
    // LoadViewTko and StoreViewTko with token threading.
    let tile_ptr_f32 = Type::Tile(TileType {
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
        shape: vec![],
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![128],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![128],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let tile_128_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![128],
    });

    let module = build_kernel(
        "load_store",
        std::slice::from_ref(&tile_ptr_f32),
        |m, blk, args| {
            // make_token
            let (tok_op, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
                .result(token_ty())
                .build(m);
            append_op(m, blk, tok_op);

            // make_tensor_view
            let (mtv, mtv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[0])
                .result(tv_ty.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(0, i32_ty()),
                        Attribute::Integer(0, i32_ty()),
                    ]),
                )
                .build(m);
            append_op(m, blk, mtv);

            // make_partition_view
            let (mpv, mpv_res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                .operand(mtv_res[0])
                .result(pv_ty.clone())
                .build(m);
            append_op(m, blk, mpv);

            // load_view_tko
            let idx = const_i32(m, blk, 0);
            let (load, load_res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
                .operand(mpv_res[0]) // view
                .operand(idx) // index
                .operand(tok_res[0]) // token
                .attr("memory_ordering_semantics", Attribute::Integer(0, i32_ty()))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(1, i32_ty()),
                    ]),
                )
                .result(tile_128_f32.clone())
                .result(token_ty())
                .build(m);
            append_op(m, blk, load);

            // store_view_tko
            let (store, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
                .operand(load_res[0]) // tile
                .operand(mpv_res[0]) // view
                .operand(idx) // index
                .operand(load_res[1]) // token
                .attr("memory_ordering_semantics", Attribute::Integer(0, i32_ty()))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(1, i32_ty()),
                        Attribute::Integer(1, i32_ty()),
                    ]),
                )
                .result(token_ty())
                .build(m);
            append_op(m, blk, store);
        },
    );
    validate_module(&module);
}

// ── Debug-location variants: what does tileiras -G accept? ────────────────
// Bisection scaffolding for the debug section: each variant differs only in
// the Location structures attached to ops.

fn floc(file: &str, line: u32, col: u32) -> Location {
    Location::FileLineCol {
        filename: file.to_string(),
        line,
        column: col,
    }
}

fn sub(file: &str, line: u32, name: &str, linkage: &str) -> DISubprogram {
    let f = DIFile {
        name: file.rsplit('/').next().unwrap().to_string(),
        directory: "src".to_string(),
    };
    DISubprogram {
        file: f.clone(),
        line,
        name: name.to_string(),
        linkage_name: linkage.to_string(),
        compile_unit: DICompileUnit { file: f },
        scope_line: line,
    }
}

fn debug_variant(name: &str, op_locs: [Location; 2], entry_loc: Location) -> Module {
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: vec![tile_i32(), tile_i32()],
        results: vec![],
    });
    let (region_id, block_id, args) =
        build_single_block_region(&mut module, &[tile_i32(), tile_i32()]);
    let [l1, l2] = op_locs;
    let (add, add_res) = OpBuilder::new(Opcode::AddI, l1)
        .operand(args[0])
        .operand(args[1])
        .attr("overflow", Attribute::Integer(0, i32_ty()))
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, add);
    let (add2, _) = OpBuilder::new(Opcode::AddI, l2)
        .operand(add_res[0])
        .operand(args[1])
        .attr("overflow", Attribute::Integer(0, i32_ty()))
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, add2);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, entry_loc)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

#[test]
fn dbg_v1_entry_scoped_lines() {
    // Pre-S1 shape: bare file:line:col ops scoped to the entry subprogram.
    let m = debug_variant(
        "dbg_v1",
        [floc("src/k.rs", 7, 4), floc("src/k.rs", 8, 4)],
        floc("src/k.rs", 5, 0),
    );
    validate_module(&m);
}

#[test]
fn dbg_v2_foreign_scope_requires_call_site() {
    // The toolchain's verifier rule this suite exists to document: an op
    // scoped to a subprogram other than its function's MUST say where it
    // was inlined (a call-site chain). Bare foreign scopes are rejected.
    let helper = sub("src/k.rs", 20, "helper", "helper@k:20:4");
    let m = debug_variant(
        "dbg_v2",
        [
            Location::DebugInfo(DebugInfoLoc {
                filename: "src/k.rs".into(),
                line: 21,
                column: 8,
                scope: DebugScope::Subprogram(helper),
            }),
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    m.verify_dominance().expect("dominance");
    let bc = write_test_bytecode(&m).expect("write");
    cutile_ir::decode_bytecode(&bc).expect("our decoder accepts it");
    expect_tileiras_rejects(&bc, "dbg_v2", "debug info scope");
}

/// Asserts tileiras rejects the bytecode with a message containing
/// `needle`. Skips silently when tileiras is unavailable (CI without the
/// toolchain), like `run_tileiras`.
fn expect_tileiras_rejects(bc: &[u8], name: &str, needle: &str) {
    let tmp =
        std::env::temp_dir().join(format!("cutile_ir_reject_{name}_{}.bc", std::process::id()));
    std::fs::write(&tmp, bc).unwrap();
    let out = std::process::Command::new(tileiras_binary())
        .arg("--gpu-name")
        .arg("sm_120")
        .arg("-o")
        .arg(std::env::temp_dir().join(format!("cutile_ir_reject_{name}.cubin")))
        .arg(&tmp)
        .output();
    let _ = std::fs::remove_file(&tmp);
    // `Err` means tileiras is not installed; the check is skipped.
    if let Ok(out) = out {
        assert!(
            !out.status.success(),
            "{name}: tileiras unexpectedly accepted invalid debug info"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(needle),
            "{name}: rejection reason changed:\n{stderr}"
        );
    }
}

#[test]
fn dbg_v3_callee_scope_with_callsite() {
    // The S1 shape: callee-scoped DILoc wrapped in a CallSite.
    let helper = sub("src/k.rs", 20, "helper", "helper@k:20:4");
    let m = debug_variant(
        "dbg_v3",
        [
            Location::CallSite {
                callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                    filename: "src/k.rs".into(),
                    line: 21,
                    column: 8,
                    scope: DebugScope::Subprogram(helper),
                })),
                caller: Box::new(floc("src/k.rs", 7, 4)),
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    validate_module(&m);
}

#[test]
fn dbg_v4_callsite_bare_callee() {
    // Pre-S1 call-site shape: bare FileLineCol callee (entry-scoped).
    let m = debug_variant(
        "dbg_v4",
        [
            Location::CallSite {
                callee: Box::new(floc("src/k.rs", 21, 8)),
                caller: Box::new(floc("src/k.rs", 7, 4)),
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    validate_module(&m);
}

#[test]
fn dbg_v5_nested_callsite_chain() {
    let h1 = sub("src/k.rs", 20, "helper1", "helper1@k:20:4");
    let h2 = sub("src/k.rs", 30, "helper2", "helper2@k:30:4");
    let inner = Location::CallSite {
        callee: Box::new(Location::DebugInfo(DebugInfoLoc {
            filename: "src/k.rs".into(),
            line: 31,
            column: 8,
            scope: DebugScope::Subprogram(h2),
        })),
        caller: Box::new(Location::CallSite {
            callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                filename: "src/k.rs".into(),
                line: 21,
                column: 8,
                scope: DebugScope::Subprogram(h1),
            })),
            caller: Box::new(floc("src/k.rs", 7, 4)),
        }),
    };
    let m = debug_variant(
        "dbg_v5",
        [inner, floc("src/k.rs", 8, 4)],
        floc("src/k.rs", 5, 0),
    );
    validate_module(&m);
}

#[test]
fn dbg_v6_line_before_decl() {
    let helper = sub("src/k.rs", 20, "helper", "helper@k:20:4");
    let m = debug_variant(
        "dbg_v6",
        [
            Location::CallSite {
                callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                    filename: "src/k.rs".into(),
                    line: 10, // before the subprogram's decl line 20
                    column: 1,
                    scope: DebugScope::Subprogram(helper),
                })),
                caller: Box::new(floc("src/k.rs", 7, 4)),
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(format!("{dir}/v6.bc"), write_test_bytecode(&m).unwrap()).unwrap();
    }
    validate_module(&m);
}

#[test]
fn dbg_v7_duplicate_subprogram_name() {
    // Entry subprogram is named after sym_name ("dbg_v7"); a second
    // subprogram shares that NAME with a different linkage, like the
    // generated entry wrapper inlining the user impl of the same kernel.
    let twin = sub("src/k.rs", 11, "dbg_v7", "dbg_v7@k:11:7");
    let m = debug_variant(
        "dbg_v7",
        [
            Location::CallSite {
                callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                    filename: "src/k.rs".into(),
                    line: 12,
                    column: 8,
                    scope: DebugScope::Subprogram(twin),
                })),
                caller: Box::new(floc("src/k.rs", 5, 1)),
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(format!("{dir}/v7.bc"), write_test_bytecode(&m).unwrap()).unwrap();
    }
    validate_module(&m);
}

#[test]
fn dbg_v8_cross_file_subprogram() {
    // The inlined helper lives in a DIFFERENT file: second DIFile + second
    // DICompileUnit in one function's debug info.
    let helper = sub("src/other.rs", 20, "helper", "helper@other:20:4");
    let m = debug_variant(
        "dbg_v8",
        [
            Location::CallSite {
                callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                    filename: "src/other.rs".into(),
                    line: 21,
                    column: 8,
                    scope: DebugScope::Subprogram(helper),
                })),
                caller: Box::new(floc("src/k.rs", 7, 4)),
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    validate_module(&m);
}

#[test]
fn dbg_dump_variants_for_external_probe() {
    // Writes each debug-location variant's bytecode to $CUTILE_DBG_DUMP_DIR
    // for external tool probing (e.g. tileiras --device-debug).
    let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") else {
        return;
    };
    let h = |n: &str| sub("src/k.rs", 20, n, "h@k:20:4");
    let di = |line: u32, s: DISubprogram| {
        Location::DebugInfo(DebugInfoLoc {
            filename: "src/k.rs".into(),
            line,
            column: 8,
            scope: DebugScope::Subprogram(s),
        })
    };
    let cs = |callee: Location, caller: Location| Location::CallSite {
        callee: Box::new(callee),
        caller: Box::new(caller),
    };
    let variants: Vec<(&str, [Location; 2])> = vec![
        ("v1", [floc("src/k.rs", 7, 4), floc("src/k.rs", 8, 4)]),
        (
            "v3",
            [
                cs(di(21, h("helper")), floc("src/k.rs", 7, 4)),
                floc("src/k.rs", 8, 4),
            ],
        ),
        (
            "v5",
            [
                cs(
                    di(31, sub("src/k.rs", 30, "h2", "h2@k:30:4")),
                    cs(di(21, h("h1")), floc("src/k.rs", 7, 4)),
                ),
                floc("src/k.rs", 8, 4),
            ],
        ),
        (
            "v8",
            [
                cs(
                    Location::DebugInfo(DebugInfoLoc {
                        filename: "src/other.rs".into(),
                        line: 21,
                        column: 8,
                        scope: DebugScope::Subprogram(sub(
                            "src/other.rs",
                            20,
                            "helper",
                            "helper@other:20:4",
                        )),
                    }),
                    floc("src/k.rs", 7, 4),
                ),
                floc("src/k.rs", 8, 4),
            ],
        ),
    ];
    for (name, locs) in variants {
        let m = debug_variant(name, locs, floc("src/k.rs", 5, 0));
        let bc = write_test_bytecode(&m).unwrap();
        std::fs::write(format!("{dir}/{name}.bc"), bc).unwrap();
    }
}

#[test]
fn dbg_v9_deep_mixed_chain() {
    // Depth-4 inlined-at chain mixing two files, replicating the real
    // compiler output shape that fails under --device-debug.
    let user = sub("src/k.rs", 11, "dbg_v9", "dbg_v9@k:11:7");
    let store = sub(
        "src/core.rs",
        3271,
        "store_tile_1d",
        "store_tile_1d@core:3271:7",
    );
    let shape = sub("src/core.rs", 706, "shape", "shape@core:706:15");
    let di = |file: &str, line: u32, s: &DISubprogram| {
        Location::DebugInfo(DebugInfoLoc {
            filename: file.into(),
            line,
            column: 8,
            scope: DebugScope::Subprogram(s.clone()),
        })
    };
    let cs = |callee: Location, caller: Location| Location::CallSite {
        callee: Box::new(callee),
        caller: Box::new(caller),
    };
    // shape() inlined into store_tile_1d, inlined into the user fn,
    // inlined into the entry.
    let chain = cs(
        di("src/core.rs", 260, &shape),
        cs(
            di("src/core.rs", 268, &store),
            cs(di("src/k.rs", 13, &user), floc("src/k.rs", 5, 0)),
        ),
    );
    let m = debug_variant(
        "dbg_v9",
        [chain, floc("src/k.rs", 8, 4)],
        floc("src/k.rs", 5, 0),
    );
    let bc = write_test_bytecode(&m).unwrap();
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(format!("{dir}/v9.bc"), &bc).unwrap();
    }
    validate_module(&m);
}

#[test]
fn dbg_v11_chain_on_special_ops() {
    // Deep inlined-at chains attached to ops that dissolve or expand during
    // lowering (assume, get_tile_block_id). Repro hunt for the -G failure.
    let user = sub("src/k.rs", 11, "dbg_v11", "dbg_v11@k:11:7");
    let store = sub(
        "src/core.rs",
        3271,
        "store_tile_1d",
        "store_tile_1d@core:3271:7",
    );
    let di = |file: &str, line: u32, s: &DISubprogram| {
        Location::DebugInfo(DebugInfoLoc {
            filename: file.into(),
            line,
            column: 8,
            scope: DebugScope::Subprogram(s.clone()),
        })
    };
    let cs = |callee: Location, caller: Location| Location::CallSite {
        callee: Box::new(callee),
        caller: Box::new(caller),
    };
    let chain = || {
        cs(
            di("src/core.rs", 260, &store),
            cs(di("src/k.rs", 13, &user), floc("src/k.rs", 5, 0)),
        )
    };

    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: vec![tile_i32()],
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, &[tile_i32()]);
    let (bid, bid_res) = OpBuilder::new(Opcode::GetTileBlockId, chain())
        .result(tile_i32())
        .result(tile_i32())
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, bid);
    let (add, _) = OpBuilder::new(Opcode::AddI, chain())
        .operand(args[0])
        .operand(bid_res[0])
        .attr("overflow", Attribute::Integer(0, i32_ty()))
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, add);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, floc("src/k.rs", 5, 0))
        .attr("sym_name", Attribute::String("dbg_v11".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(
            format!("{dir}/v11.bc"),
            write_test_bytecode(&module).unwrap(),
        )
        .unwrap();
    }
    validate_module(&module);
}

#[test]
fn dbg_v12_caller_equals_function_attr() {
    // The inlined-at caller location is IDENTICAL to the function's own
    // record location (same interned attr) — the generated-entry pattern.
    let helper = sub("src/other.rs", 20, "helper", "helper@other:20:4");
    let m = debug_variant(
        "dbg_v12",
        [
            Location::CallSite {
                callee: Box::new(Location::DebugInfo(DebugInfoLoc {
                    filename: "src/other.rs".into(),
                    line: 21,
                    column: 8,
                    scope: DebugScope::Subprogram(helper),
                })),
                caller: Box::new(floc("src/k.rs", 5, 0)), // == entry loc
            },
            floc("src/k.rs", 8, 4),
        ],
        floc("src/k.rs", 5, 0),
    );
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(format!("{dir}/v12.bc"), write_test_bytecode(&m).unwrap()).unwrap();
    }
    validate_module(&m);
}

#[test]
fn dbg_v13_dissolved_op_with_record_caller() {
    // get_tile_block_id carrying a call-site whose caller attr equals the
    // function's own record attr — the exact real-compiler combination.
    let store = sub(
        "src/core.rs",
        3271,
        "store_tile_1d",
        "store_tile_1d@core:3271:7",
    );
    let chain = Location::CallSite {
        callee: Box::new(Location::DebugInfo(DebugInfoLoc {
            filename: "src/core.rs".into(),
            line: 260,
            column: 1,
            scope: DebugScope::Subprogram(store),
        })),
        caller: Box::new(floc("src/k.rs", 5, 0)), // == entry loc attr
    };
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: vec![tile_i32()],
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, &[tile_i32()]);
    let (bid, bid_res) = OpBuilder::new(Opcode::GetTileBlockId, chain)
        .result(tile_i32())
        .result(tile_i32())
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, bid);
    let (add, _) = OpBuilder::new(Opcode::AddI, floc("src/k.rs", 8, 4))
        .operand(args[0])
        .operand(bid_res[0])
        .attr("overflow", Attribute::Integer(0, i32_ty()))
        .result(tile_i32())
        .build(&mut module);
    append_op(&mut module, block_id, add);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, floc("src/k.rs", 5, 0))
        .attr("sym_name", Attribute::String("dbg_v13".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    if let Ok(dir) = std::env::var("CUTILE_DBG_DUMP_DIR") {
        std::fs::write(
            format!("{dir}/v13.bc"),
            write_test_bytecode(&module).unwrap(),
        )
        .unwrap();
    }
    validate_module(&module);
}
