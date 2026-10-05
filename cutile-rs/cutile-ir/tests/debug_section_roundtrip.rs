/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The Debug section must roundtrip: locations attached to ops survive
//! serialization and come back out of the decoder — the historical failure
//! mode of this codec is silently dropping fields.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

fn tile_i32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I32),
        shape: vec![],
    })
}

fn floc(line: u32, column: u32) -> Location {
    Location::FileLineCol {
        filename: "src/kernels/scale.rs".to_string(),
        line,
        column,
    }
}

fn constant(module: &mut Module, block: BlockId, v: i32, loc: Location) {
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
        .build(module);
    append_op(module, block, op);
}

fn build_module() -> Module {
    let mut module = Module::new("debug_roundtrip_module");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);

    // Two ops on the same source line: their DILoc must intern to ONE attr.
    constant(&mut module, block_id, 1, floc(7, 4));
    constant(&mut module, block_id, 2, floc(7, 4));
    // A distinct line.
    constant(&mut module, block_id, 3, floc(9, 8));
    // An op with no location: id 0.
    constant(&mut module, block_id, 4, Location::Unknown);
    // An inlined op: callee location wrapped in the caller's call site.
    constant(
        &mut module,
        block_id,
        5,
        Location::CallSite {
            callee: Box::new(Location::FileLineCol {
                filename: "src/kernels/helper.rs".to_string(),
                line: 3,
                column: 12,
            }),
            caller: Box::new(floc(11, 2)),
        },
    );

    let (ret, _) = OpBuilder::new(Opcode::Return, floc(12, 0)).build(&mut module);
    append_op(&mut module, block_id, ret);

    let (entry, _) = OpBuilder::new(Opcode::Entry, floc(5, 3))
        .attr("sym_name", Attribute::String("scale_entry".into()))
        .attr("di_name", Attribute::String("scale".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

#[test]
fn debug_section_roundtrips_locations() {
    let module = build_module();
    let bytecode = cutile_ir::write_bytecode(&module).expect("serialize");
    let dump = cutile_ir::bytecode::decoder::decode_bytecode(&bytecode).expect("decode");

    // Subprogram: user-facing name + linkage symbol + declaration line.
    assert!(
        dump.contains(r#"name="scale", linkage="scale_entry""#),
        "subprogram must pair the DI name with the linkage symbol:\n{dump}"
    );
    assert!(
        dump.contains("line=5"),
        "subprogram must carry the function's declaration line:\n{dump}"
    );
    // File split into basename + directory.
    assert!(
        dump.contains(r#"DIFile(name="scale.rs", dir="src/kernels")"#),
        "file attr must split path into name and directory:\n{dump}"
    );
    // Per-op locations, scoped to the subprogram.
    assert!(
        dump.contains(r#""src/kernels/scale.rs":7:4"#),
        "op location must roundtrip file:line:col:\n{dump}"
    );
    assert!(
        dump.contains(r#""src/kernels/scale.rs":9:8"#),
        "second line must roundtrip:\n{dump}"
    );
    // Inlined op: a CallSite linking callee and caller locs.
    assert!(
        dump.contains("CallSite(callee=di["),
        "call-site chain must be interned:\n{dump}"
    );
    assert!(
        dump.contains(r#""src/kernels/helper.rs":3:12"#),
        "callee location must roundtrip:\n{dump}"
    );

    // The per-function id list: [func, op1..op6]. Ops 1 and 2 share one id
    // (interned), op 4 is 0 (unknown).
    let fn_line = dump
        .lines()
        .find(|l| l.trim_start().starts_with("fn 0:"))
        .expect("per-function id list");
    let ids: Vec<u64> = fn_line[fn_line.find('[').unwrap() + 1..fn_line.find(']').unwrap()]
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    assert_eq!(ids.len(), 7, "1 function attr + 6 ops: {fn_line}");
    assert_ne!(ids[0], 0, "function attr present");
    assert_eq!(ids[1], ids[2], "same-line ops intern to one attr");
    assert_ne!(ids[1], ids[3], "distinct lines get distinct attrs");
    assert_eq!(ids[4], 0, "unknown location is id 0");
    assert_ne!(ids[5], 0, "call-site op has an attr");
}

#[test]
fn location_free_module_still_carries_a_debug_section() {
    // The section is always written; a module with no locations gets the
    // empty-table workaround and all-zero ids rather than no section.
    let mut module = Module::new("no_locs");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("k_entry".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);

    let bytecode = cutile_ir::write_bytecode(&module).expect("serialize");
    let dump = cutile_ir::bytecode::decoder::decode_bytecode(&bytecode).expect("decode");
    assert!(
        dump.contains("=== Debug (1 functions) ==="),
        "section must exist:\n{dump}"
    );
    assert!(
        dump.contains("fn 0: [0, 0]"),
        "function and op ids must be 0 (no info):\n{dump}"
    );
}
