/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Serialization must be deterministic: the same module serialized twice must
//! produce byte-identical bytecode.
//!
//! This is the load-bearing assumption of the on-disk cubin cache.
//! The disk cache key is a SHA-256 over the `.bc` bytes, so if serialization
//! were non-deterministic the cache could never hit across processes — a writer
//! process would store under key A while a reader computed key B for the same
//! kernel, and every cross-process lookup would miss (safe, but useless).
//!
//! The writer keeps this property by construction: the string pool is an
//! `indexmap::IndexMap` (insertion order), and every other index (types,
//! values, function-location entries) is assigned by insertion order too. The
//! `std::collections::HashMap`s in the writer are lookup tables only, never
//! iterated to produce output.
//!
//! This test fails the moment any `std::collections::HashMap` is iterated into
//! the output: each `write_bytecode` call builds fresh managers, and two
//! `HashMap`s in one process iterate in different orders (`RandomState` is
//! seeded per instance), so a HashMap-ordered section would differ between the
//! two serializations below.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

fn tile_i32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I32),
        shape: vec![],
    })
}

/// A module deliberately rich in the things that live in the writer's maps:
/// many distinct strings (attribute keys and values), several values, and
/// repeated types. If any of these sections were emitted in HashMap order,
/// two independent serializations would disagree.
fn build_string_heavy_module() -> Module {
    let mut module = Module::new("determinism_test_module");
    let arg_types = vec![tile_i32(), tile_i32(), tile_i32()];
    let func_type = Type::Func(FuncType {
        inputs: arg_types.clone(),
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &arg_types);

    // Several constants: multiple values and repeated type references.
    for v in 0..8i32 {
        // Distinct locations: the debug section (attr table + per-op ids)
        // must serialize deterministically too.
        let loc = Location::FileLineCol {
            filename: "src/determinism.rs".to_string(),
            line: 10 + v as u32,
            column: 4,
        };
        let (op, _res) = OpBuilder::new(Opcode::Constant, loc)
            .attr(
                "value",
                Attribute::DenseElements(DenseElements {
                    element_type: tile_i32(),
                    shape: vec![],
                    data: v.to_le_bytes().to_vec(),
                }),
            )
            .result(tile_i32())
            .build(&mut module);
        append_op(&mut module, block_id, op);
    }

    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);

    // Entry with many distinct string attributes: ~24 extra strings in the
    // pool (12 keys + 12 values), enough that a HashMap-ordered pool would
    // almost certainly differ between two instances.
    let mut entry = OpBuilder::new(
        Opcode::Entry,
        Location::FileLineCol {
            filename: "src/determinism.rs".to_string(),
            line: 5,
            column: 0,
        },
    )
    .attr("sym_name", Attribute::String("determinism_kernel".into()))
    .attr("function_type", Attribute::Type(func_type))
    .region(region_id);
    for i in 0..12 {
        entry = entry.attr(
            format!("str_attr_key_{i:02}"),
            Attribute::String(format!("str_attr_val_{i:02}")),
        );
    }
    let (entry, _) = entry.build(&mut module);
    module.functions.push(entry);

    module
}

#[test]
fn same_module_serializes_to_identical_bytes() {
    let module = build_string_heavy_module();

    let first = cutile_ir::write_bytecode(&module).expect("write_bytecode failed");
    let second = cutile_ir::write_bytecode(&module).expect("write_bytecode failed");

    assert_eq!(
        first,
        second,
        "serializing the same module twice produced different bytes ({} vs {} bytes); \
         a section is being emitted in HashMap order, which breaks the content-addressed \
         disk cache key across processes",
        first.len(),
        second.len(),
    );
}

/// The module must exercise the string pool for the test above to be meaningful:
/// if there were no strings, HashMap-order non-determinism could hide.
#[test]
fn test_module_actually_has_strings() {
    let module = build_string_heavy_module();
    let bc = cutile_ir::write_bytecode(&module).expect("write_bytecode failed");
    // The distinct attribute values are UTF-8 in the string section; find one.
    let needle = b"str_attr_val_07";
    assert!(
        bc.windows(needle.len()).any(|w| w == needle),
        "expected the string pool to contain the attribute values that make \
         the determinism test meaningful"
    );
}
