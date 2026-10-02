/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Builds a real tile-ir kernel and serializes it to bytecode.
//!
//! The kernel adds two 4-element f32 tiles and stores the result:
//!   c[i] = a[i] + b[i]
//!
//! Run with: cargo run -p cutile-ir --example build_basic

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion, Opcode};
use cutile_ir::ir::*;
use std::process::Command;

/// Shorthand: build a scalar i32 constant and append it to a block.
fn cst_i32(m: &mut Module, blk: BlockId, val: i32) -> Value {
    let ty = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let (op, r) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: ty.clone(),
                shape: vec![],
                data: val.to_le_bytes().to_vec(),
            }),
        )
        .result(ty)
        .build(m);
    append_op(m, blk, op);
    r[0]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut m = Module::new("add_module");

    // -- Types --
    let tile_ptr_f32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
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
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![4],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![4],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![4],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });

    // Entry: add_kernel(%ptr_a, %ptr_b, %ptr_c)
    let func_type = FuncType {
        inputs: vec![
            tile_ptr_f32.clone(),
            tile_ptr_f32.clone(),
            tile_ptr_f32.clone(),
        ],
        results: vec![],
    };
    let (region, blk, args) = build_single_block_region(
        &mut m,
        &[
            tile_ptr_f32.clone(),
            tile_ptr_f32.clone(),
            tile_ptr_f32.clone(),
        ],
    );
    let (ptr_a, ptr_b, ptr_c) = (args[0], args[1], args[2]);

    // Tokens for memory ordering.
    let mk = |m: &mut Module, blk| {
        let (op, r) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(Type::Token)
            .build(m);
        append_op(m, blk, op);
        r[0]
    };
    let tok_a = mk(&mut m, blk);
    let tok_b = mk(&mut m, blk);
    let tok_c = mk(&mut m, blk);

    // Build tensor views (static shape, no dynamic operands).
    let seg_base_only = Attribute::Array(vec![
        Attribute::i32(1),
        Attribute::i32(0),
        Attribute::i32(0),
    ]);
    let build_tv = |m: &mut Module, blk, ptr| {
        let (op, r) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
            .operand(ptr)
            .attr("operandSegmentSizes", seg_base_only.clone())
            .result(tv_ty.clone())
            .build(m);
        append_op(m, blk, op);
        r[0]
    };
    let tv_a = build_tv(&mut m, blk, ptr_a);
    let tv_b = build_tv(&mut m, blk, ptr_b);
    let tv_c = build_tv(&mut m, blk, ptr_c);

    // get_tile_block_id → %pid
    let (op, pids) = OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
        .result(tile_i32.clone())
        .result(tile_i32.clone())
        .result(tile_i32.clone())
        .build(&mut m);
    append_op(&mut m, blk, op);
    let pid = pids[0];

    // Partition views.
    let build_pv = |m: &mut Module, blk, tv| {
        let (op, r) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
            .operand(tv)
            .result(pv_ty.clone())
            .build(m);
        append_op(m, blk, op);
        r[0]
    };
    let pv_a = build_pv(&mut m, blk, tv_a);
    let pv_b = build_pv(&mut m, blk, tv_b);
    let pv_c = build_pv(&mut m, blk, tv_c);

    // Load tiles.
    let seg_load = Attribute::Array(vec![
        Attribute::i32(1),
        Attribute::i32(1),
        Attribute::i32(1),
    ]);
    let load = |m: &mut Module, blk, pv, tok| {
        let (op, r) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
            .operand(pv)
            .operand(pid)
            .operand(tok)
            .attr("memory_ordering_semantics", Attribute::i32(0))
            .attr("operandSegmentSizes", seg_load.clone())
            .result(tile_4xf32.clone())
            .result(token_ty.clone())
            .build(m);
        append_op(m, blk, op);
        r[0]
    };
    let tile_a = load(&mut m, blk, pv_a, tok_a);
    let tile_b = load(&mut m, blk, pv_b, tok_b);

    // %sum = addf %tile_a, %tile_b
    let (op, res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
        .operand(tile_a)
        .operand(tile_b)
        .attr("rounding_mode", Attribute::i32(0))
        .result(tile_4xf32.clone())
        .build(&mut m);
    append_op(&mut m, blk, op);
    let sum = res[0];

    // Store result.
    let idx = cst_i32(&mut m, blk, 0);
    let seg_store = Attribute::Array(vec![
        Attribute::i32(1),
        Attribute::i32(1),
        Attribute::i32(1),
        Attribute::i32(1),
    ]);
    let (op, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
        .operand(sum)
        .operand(pv_c)
        .operand(idx)
        .operand(tok_c)
        .attr("memory_ordering_semantics", Attribute::i32(0))
        .attr("operandSegmentSizes", seg_store)
        .result(token_ty)
        .build(&mut m);
    append_op(&mut m, blk, op);

    // Return.
    let (op, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
    append_op(&mut m, blk, op);

    // Entry op.
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("add_kernel".into()))
        .attr("function_type", Attribute::Type(Type::Func(func_type)))
        .region(region)
        .build(&mut m);
    m.functions.push(entry);

    // -- Print IR --
    println!("{}", m.to_mlir_text());

    // -- Write bytecode --
    let tmp = std::env::temp_dir();
    let bc = tmp.join("add_kernel.bc");
    let bc = bc.to_str().unwrap();
    let tileiras = tileiras_binary();
    let version = bytecode_version(&tileiras)?;
    std::fs::write(bc, write_bytecode_version(&m, version)?)?;
    println!("Wrote Tile IR {version} bytecode to {bc}");

    // -- Compile with tileiras --
    let cubin = tmp.join("add_kernel.cubin");
    let cubin = cubin.to_str().unwrap();
    let output = Command::new(&tileiras)
        .args(["--gpu-name", "sm_120", "--opt-level", "3", "-o", cubin, bc])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "tileiras failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    println!("Compiled to {cubin}");
    Ok(())
}

fn tileiras_binary() -> std::path::PathBuf {
    cutile_ir::toolchain::tileiras_from_env()
}

fn bytecode_version(tileiras: &std::path::Path) -> std::io::Result<BytecodeVersion> {
    let requested = std::env::var_os("CUTILE_BYTECODE_VERSION").filter(|v| !v.is_empty());
    cutile_ir::toolchain::negotiate_bytecode_version(tileiras, requested.as_deref())
}
