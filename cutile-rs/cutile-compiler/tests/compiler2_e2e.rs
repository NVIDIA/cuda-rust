/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end test for compiler2: build IR → bytecode → tileiras → cubin.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{BytecodeVersion, Opcode};
use cutile_ir::ir::{
    Attribute, FuncType, Location, Module, PartitionViewType, ScalarType, TensorViewType,
    TileElementType, TileType, Type, DYNAMIC,
};

use cutile_compiler::cuda_tile_runtime_utils::{compile_tile_ir_module, tileiras_binary};

const DEFAULT_GPU_NAME: &str = "sm_120";

/// Build the simplest possible kernel: an entry that just returns.
fn build_empty_kernel() -> Module {
    let mut module = Module::new("test_module");

    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });

    let (region_id, block_id, _) = build_single_block_region(&mut module, &[]);

    let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret_id);

    let (entry_id, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("empty_kernel".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry_id);

    module
}

#[test]
fn test_empty_kernel_mlir_text() {
    let module = build_empty_kernel();
    let text = module.to_mlir_text();
    println!("{text}");
    assert!(text.contains("entry"), "expected 'entry' in MLIR text");
    assert!(
        text.contains("@empty_kernel"),
        "expected kernel name in MLIR text"
    );
    assert!(text.contains("return"), "expected 'return' in MLIR text");
}

#[test]
fn test_empty_kernel_bytecode_roundtrip() {
    let module = build_empty_kernel();

    // Serialize to bytecode.
    let bytecode = cutile_ir::write_bytecode(&module).expect("bytecode serialization failed");
    println!("bytecode length: {}", bytecode.len());
    println!(
        "bytecode hex (first 128 bytes): {:02x?}",
        &bytecode[..std::cmp::min(bytecode.len(), 128)]
    );
    assert!(bytecode.len() > 12, "bytecode too short");

    // Verify magic number.
    assert_eq!(
        &bytecode[0..8],
        &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]
    );

    // Decode and check.
    match cutile_ir::decode_bytecode(&bytecode) {
        Ok(decoded) => {
            println!("=== decoded ===\n{decoded}");
            assert!(decoded.contains(&format!("TileIR bytecode v{}", BytecodeVersion::CURRENT)));
            assert!(decoded.contains("empty_kernel"));
        }
        Err(e) => {
            // Print bytecode for debugging.
            for (i, chunk) in bytecode.chunks(16).enumerate() {
                let hex = chunk
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                let ascii = chunk
                    .iter()
                    .map(|b| {
                        if b.is_ascii_graphic() {
                            *b as char
                        } else {
                            '.'
                        }
                    })
                    .collect::<String>();
                println!("{:04x}: {hex:<48} {ascii}", i * 16);
            }
            panic!("decode failed: {e}");
        }
    }
}

#[test]
fn test_empty_kernel_tileiras() {
    // Skip if tileiras is not available. Lowering to a cubin does not need a CUDA device.
    if std::process::Command::new(tileiras_binary())
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping: tileiras not available");
        return;
    }
    let gpu_name = DEFAULT_GPU_NAME;

    let module = build_empty_kernel();
    println!("Target: {gpu_name}");

    // Print MLIR text for debugging.
    println!("=== MLIR text ===\n{}", module.to_mlir_text());

    // Print bytecode decode for debugging.
    let bytecode = cutile_ir::write_bytecode(&module).unwrap();
    println!(
        "=== Bytecode decode ===\n{}",
        cutile_ir::decode_bytecode(&bytecode).unwrap()
    );

    // Run through tileiras.
    let cubin = compile_tile_ir_module(&module, gpu_name).unwrap();
    println!("cubin: {} bytes", cubin.len());
    assert!(!cubin.is_empty(), "cubin image should be non-empty");
}

// =========================================================================
// Helper to skip tests when no GPU is available
// =========================================================================

fn tileiras_target() -> Option<&'static str> {
    if std::process::Command::new(tileiras_binary())
        .arg("--version")
        .output()
        .is_err()
    {
        return None;
    }
    Some(DEFAULT_GPU_NAME)
}

fn assert_tileiras_accepts(module: &Module) {
    let Some(gpu_name) = tileiras_target() else {
        eprintln!("skipping tileiras: tileiras not available");
        return;
    };

    println!("=== MLIR text ===\n{}", module.to_mlir_text());
    let bytecode = cutile_ir::write_bytecode(module).unwrap();
    println!(
        "=== Bytecode decode ===\n{}",
        cutile_ir::decode_bytecode(&bytecode).unwrap()
    );

    let cubin = compile_tile_ir_module(module, gpu_name).unwrap();
    assert!(!cubin.is_empty(), "cubin image should be non-empty");
    println!("tileiras accepted ✓");
}

// =========================================================================
// Kernel: add two tiles
// =========================================================================

/// Build a kernel matching the real compiler's output pattern:
/// - Args are raw pointers + shape/stride scalars (not tensor_view)
/// - tensor_view and partition_view built internally
/// - Uses make_tensor_view, make_partition_view, load_view_tko, addf, store_view_tko
fn build_add_kernel() -> Module {
    let mut module = Module::new("add_module");

    // Types
    let tile_ptr_f32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(cutile_ir::ir::PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let tile_i32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let tile_4xf32 = Type::Tile(TileType {
        shape: vec![4],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let token_ty = Type::Token;
    let f32_tv_dyn = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC],
        strides: vec![1],
    });
    let f32_tv_static = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![4],
        strides: vec![1],
    });
    let f32_pv_dyn = Type::PartitionView(PartitionViewType {
        tile_shape: vec![4],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let f32_pv_static = Type::PartitionView(PartitionViewType {
        tile_shape: vec![4],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![4],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });

    // Function signature: raw pointers + shapes (matching real compiler output)
    // entry @add(%ptr_a: tile<ptr<f32>>, %shape_a: tile<i32>,
    //            %ptr_b: tile<ptr<f32>>, %shape_b: tile<i32>,
    //            %ptr_c: tile<ptr<f32>>)
    let func_type = Type::Func(FuncType {
        inputs: vec![
            tile_ptr_f32.clone(),
            tile_i32.clone(), // shape_a
            tile_ptr_f32.clone(),
            tile_i32.clone(), // shape_b
            tile_ptr_f32.clone(),
        ],
        results: vec![],
    });

    let (region_id, block_id, args) = build_single_block_region(
        &mut module,
        &[
            tile_ptr_f32.clone(),
            tile_i32.clone(),
            tile_ptr_f32.clone(),
            tile_i32.clone(),
            tile_ptr_f32.clone(),
        ],
    );
    let ptr_a = args[0];
    let shape_a = args[1];
    let ptr_b = args[2];
    let shape_b = args[3];
    let ptr_c = args[4];

    // %tok_a = make_token
    let (op, results) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
        .result(token_ty.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tok_a = results[0];

    // %tv_a = make_tensor_view %ptr_a, shape=[%shape_a], strides=[]
    let (op, results) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
        .operand(ptr_a)
        .operand(shape_a) // dynamicShape
        .result(f32_tv_dyn.clone())
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1), // base
                Attribute::i32(1), // dynamicShape
                Attribute::i32(0), // dynamicStrides
            ]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tv_a = results[0];

    // %tok_b = make_token
    let (op, results) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
        .result(token_ty.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tok_b = results[0];

    // %tv_b = make_tensor_view %ptr_b, shape=[%shape_b], strides=[]
    let (op, results) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
        .operand(ptr_b)
        .operand(shape_b)
        .result(f32_tv_dyn.clone())
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(0),
            ]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tv_b = results[0];

    // %tok_c = make_token
    let (op, results) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
        .result(token_ty.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tok_c = results[0];

    // %tv_c = make_tensor_view %ptr_c, shape=[], strides=[] (static shape)
    let (op, results) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
        .operand(ptr_c)
        .result(f32_tv_static.clone())
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(0),
                Attribute::i32(0),
            ]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tv_c = results[0];

    // %pid_x, %pid_y, %pid_z = get_tile_block_id
    let (op, results) = OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
        .result(tile_i32.clone())
        .result(tile_i32.clone())
        .result(tile_i32.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let pid = results[0];

    // %pv_a = make_partition_view %tv_a
    let (op, results) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
        .operand(tv_a)
        .result(f32_pv_dyn.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let pv_a = results[0];

    // %tile_a, %tok1 = load_view_tko weak %pv_a[%pid] token=%tok_a
    let (op, results) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
        .operand(pv_a)
        .operand(pid)
        .operand(tok_a)
        .result(tile_4xf32.clone())
        .result(token_ty.clone())
        .attr("memory_ordering_semantics", Attribute::i32(0))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(1),
            ]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tile_a = results[0];

    // %pv_b = make_partition_view %tv_b
    let (op, results) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
        .operand(tv_b)
        .result(f32_pv_dyn.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let pv_b = results[0];

    // %tile_b, %tok2 = load_view_tko weak %pv_b[%pid] token=%tok_b
    let (op, results) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
        .operand(pv_b)
        .operand(pid)
        .operand(tok_b)
        .result(tile_4xf32.clone())
        .result(token_ty.clone())
        .attr("memory_ordering_semantics", Attribute::i32(0))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![
                Attribute::i32(1),
                Attribute::i32(1),
                Attribute::i32(1),
            ]),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let tile_b = results[0];

    // %sum = addf %tile_a, %tile_b
    let (op, results) = OpBuilder::new(Opcode::AddF, Location::Unknown)
        .operand(tile_a)
        .operand(tile_b)
        .result(tile_4xf32.clone())
        .attr("rounding_mode", Attribute::i32(0))
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let sum = results[0];

    // %pv_c = make_partition_view %tv_c
    let (op, results) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
        .operand(tv_c)
        .result(f32_pv_static.clone())
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let pv_c = results[0];

    // %cst_0 = constant <i32: 0>
    let (op, results) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .result(tile_i32.clone())
        .attr(
            "value",
            Attribute::DenseElements(cutile_ir::ir::DenseElements {
                element_type: tile_i32.clone(),
                shape: vec![],
                data: 0i32.to_le_bytes().to_vec(),
            }),
        )
        .build(&mut module);
    append_op(&mut module, block_id, op);
    let cst_0 = results[0];

    // store_view_tko weak %sum, %pv_c[%cst_0] token=%tok_c
    let (op, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
        .operand(sum)
        .operand(pv_c)
        .operand(cst_0)
        .operand(tok_c)
        .result(token_ty.clone())
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
        .build(&mut module);
    append_op(&mut module, block_id, op);

    // return
    let (op, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, op);

    // entry op
    let (entry_id, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("add_kernel".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry_id);

    module
}

#[test]
fn test_add_kernel_mlir_text() {
    let module = build_add_kernel();
    let text = module.to_mlir_text();
    println!("{text}");
    assert!(text.contains("addf"), "expected addf");
    assert!(text.contains("load_view_tko"), "expected load_view_tko");
    assert!(text.contains("store_view_tko"), "expected store_view_tko");
    assert!(
        text.contains("get_tile_block_id"),
        "expected get_tile_block_id"
    );
    assert!(text.contains("make_token"), "expected make_token");
    assert!(
        text.contains("make_tensor_view"),
        "expected make_tensor_view"
    );
    assert!(
        text.contains("make_partition_view"),
        "expected make_partition_view"
    );
    assert!(text.contains("constant"), "expected constant");
    assert!(text.contains("return"), "expected return");
    assert!(text.contains("@add_kernel"), "expected kernel name");
    assert!(text.contains("ptr<f32>"), "expected ptr<f32> in args");
}

#[test]
fn test_add_kernel_bytecode_roundtrip() {
    let module = build_add_kernel();
    let bytecode = cutile_ir::write_bytecode(&module).expect("bytecode serialization failed");
    let decoded = cutile_ir::decode_bytecode(&bytecode).expect("decode failed");
    println!("{decoded}");
    assert!(decoded.contains("add_kernel"));
    assert!(decoded.contains("tensor_view"));
}

#[test]
fn test_add_kernel_tileiras() {
    let module = build_add_kernel();
    assert_tileiras_accepts(&module);
}

/// Writes the add kernel bytecode for manual tileiras testing.
#[test]
fn test_add_kernel_save_bytecode() {
    let module = build_add_kernel();
    let dir = std::env::temp_dir();
    let path = dir.join("add_kernel_debug.bc");
    cutile_ir::write_bytecode_to_file(&module, path.to_str().unwrap())
        .expect("failed to write bytecode");
    println!("Bytecode written to {}", path.display());
    println!(
        "Run: tileiras --gpu-name sm_120 --opt-level 0 -o /tmp/test.cubin {}",
        path.display()
    );
}

// =========================================================================
// Kernel: print_tko with and without an ordering token
// =========================================================================

/// An entry that prints its scalar argument. With `with_token`, the print
/// takes an ordering token and its token result is consumed by a `return`
/// path: the encoding then needs the v13.2 flags bit and the token written
/// after the sized `args` group, which the writer once got wrong (flag
/// clear, token folded into the args) so every toolkit rejected it.
fn build_print_kernel(with_token: bool) -> Module {
    let mut module = Module::new("print_module");
    let tile_f32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let (region_id, block_id, args) =
        build_single_block_region(&mut module, std::slice::from_ref(&tile_f32));
    let mut operands = vec![args[0]];
    let mut token_count = 0;
    if with_token {
        let (tok, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(Type::Token)
            .build(&mut module);
        append_op(&mut module, block_id, tok);
        operands.push(tok_res[0]);
        token_count = 1;
    }
    let (print, _) = OpBuilder::new(Opcode::Print, Location::Unknown)
        .attr("str", Attribute::String("value = %f\n".into()))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(vec![Attribute::i32(1), Attribute::i32(token_count)]),
        )
        .operands(operands.iter().copied())
        .result(Type::Token)
        .build(&mut module);
    append_op(&mut module, block_id, print);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("print_kernel".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![tile_f32],
                results: vec![],
            })),
        )
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

/// `print_tko` round-trips through `tileiras` at every emittable version the
/// installed assembler accepts (the negotiated version and everything older
/// — a 13.2 `tileiras` cannot read 13.3 images), with and without the
/// optional token operand.
#[test]
fn test_print_kernel_tileiras_at_every_version() {
    use cutile_compiler::cuda_tile_runtime_utils::{
        run_tileiras, serialize_tile_ir_bytecode, TileirasOptions,
    };
    use cutile_ir::bytecode::write_bytecode_version;

    let Some(gpu_name) = tileiras_target() else {
        eprintln!("skipping tileiras: tileiras not available");
        return;
    };
    for with_token in [false, true] {
        let module = build_print_kernel(with_token);
        module.verify_dominance().expect("dominance");
        module.verify_bytecode_indices().expect("indices");
        let (_, negotiated) =
            serialize_tile_ir_bytecode(&module).expect("negotiate a bytecode version");
        for &version in BytecodeVersion::SUPPORTED
            .iter()
            .filter(|&&v| v <= negotiated)
        {
            let bytes = write_bytecode_version(&module, version)
                .unwrap_or_else(|e| panic!("print kernel must encode at {version}: {e}"));
            let cubin =
                run_tileiras(&bytes, gpu_name, &TileirasOptions::default()).unwrap_or_else(|e| {
                    panic!("tileiras rejected print_tko (token: {with_token}) at {version}: {e}")
                });
            assert!(!cubin.is_empty(), "empty cubin at {version}");
        }
    }
}
