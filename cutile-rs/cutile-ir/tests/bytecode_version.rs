/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode version handling tests.
//!
//! Verifies that version information is correctly written, decoded, and
//! that `write_bytecode_version` rejects unsupported versions.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{BytecodeVersion, Opcode};
use cutile_ir::ir::*;
use cutile_ir::{decode_bytecode, write_bytecode};

// =========================================================================
// Helpers
// =========================================================================

/// Build a minimal module with a single empty kernel.
fn build_kernel(name: &str) -> Module {
    let mut module = Module::new("version_test");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _args) = build_single_block_region(&mut module, &[]);
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

fn build_kernel_with_worker_warps(name: &str, num_worker_warps_per_cta: i64) -> Module {
    let mut module = build_kernel(name);
    let entry = module.functions[0];
    module.op_mut(entry).attributes.push((
        "optimization_hints".into(),
        Attribute::OptimizationHints(OptimizationHints {
            entries: vec![(
                "sm_120".into(),
                vec![(
                    "num_worker_warps_per_cta".into(),
                    Attribute::i32(num_worker_warps_per_cta),
                )],
            )],
        }),
    ));
    module
}

// =========================================================================
// Tests
// =========================================================================

#[test]
fn current_version_roundtrip() {
    let module = build_kernel("ver_roundtrip");
    let bytecode = write_bytecode(&module).expect("write_bytecode failed");
    let decoded = decode_bytecode(&bytecode).expect("decode_bytecode failed");

    // The current version is BytecodeVersion::CURRENT.
    let expected = format!("TileIR bytecode v{}", BytecodeVersion::CURRENT);
    assert!(
        decoded.contains(&expected),
        "decoded output should contain '{expected}', got:\n{decoded}"
    );
}

#[test]
fn decoded_output_contains_version_prefix() {
    let module = build_kernel("ver_prefix");
    let bytecode = write_bytecode(&module).expect("write_bytecode failed");
    let decoded = decode_bytecode(&bytecode).expect("decode_bytecode failed");

    // The decoded text must start with the version line.
    assert!(
        decoded.contains("TileIR bytecode v13."),
        "decoded output should contain 'TileIR bytecode v13.', got:\n{decoded}"
    );
}

#[test]
fn write_with_explicit_version() {
    use cutile_ir::bytecode::write_bytecode_version;

    let module = build_kernel("ver_explicit");

    // Write with the current version explicitly.
    let bytecode = write_bytecode_version(&module, BytecodeVersion::CURRENT).expect("write failed");
    let decoded = decode_bytecode(&bytecode).expect("decode failed");

    let expected = format!("TileIR bytecode v{}", BytecodeVersion::CURRENT);
    assert!(
        decoded.contains(&expected),
        "explicit version write should produce '{expected}' in output"
    );
}

#[test]
fn write_with_min_supported_version() {
    use cutile_ir::bytecode::write_bytecode_version;

    let module = build_kernel("ver_min");

    // The minimum supported version should be accepted.
    let bytecode =
        write_bytecode_version(&module, BytecodeVersion::MIN_SUPPORTED).expect("write failed");
    let decoded = decode_bytecode(&bytecode).expect("decode failed");

    let expected = format!("TileIR bytecode v{}", BytecodeVersion::MIN_SUPPORTED);
    assert!(
        decoded.contains(&expected),
        "min supported version should produce '{expected}' in output"
    );
}

#[test]
fn reject_unsupported_version() {
    use cutile_ir::bytecode::write_bytecode_version;

    let module = build_kernel("ver_reject");

    // A version below MIN_SUPPORTED should be rejected.
    let old_version = BytecodeVersion {
        major: 1,
        minor: 0,
        tag: 0,
    };
    let result = write_bytecode_version(&module, old_version);
    assert!(result.is_err(), "should reject version below MIN_SUPPORTED");
}

#[test]
fn worker_warps_hint_requires_version_13_3() {
    use cutile_ir::bytecode::write_bytecode_version;

    let module = build_kernel_with_worker_warps("ver_worker_warps_reject", 4);
    let error = write_bytecode_version(&module, BytecodeVersion::V13_2)
        .expect_err("worker warps hint should require bytecode version 13.3");
    assert!(
        error.to_string().contains(
            "optimization hint 'num_worker_warps_per_cta' requires bytecode version 13.3 or newer"
        ),
        "unexpected error: {error}"
    );
}

#[test]
fn worker_warps_hint_is_supported_in_version_13_3() {
    use cutile_ir::bytecode::write_bytecode_version;

    let module = build_kernel_with_worker_warps("ver_worker_warps_accept", 4);
    write_bytecode_version(&module, BytecodeVersion::V13_3)
        .expect("worker warps hint should be supported in bytecode version 13.3");
}

#[test]
fn version_display_with_tag() {
    // Verify the Display impl includes the tag when non-zero.
    let v = BytecodeVersion {
        major: 13,
        minor: 2,
        tag: 5,
    };
    assert_eq!(format!("{v}"), "13.2.5");
}

#[test]
fn version_display_without_tag() {
    // Tag == 0 should omit the tag component.
    let v = BytecodeVersion {
        major: 13,
        minor: 2,
        tag: 0,
    };
    assert_eq!(format!("{v}"), "13.2");
}

#[test]
fn version_ordering() {
    let v1 = BytecodeVersion {
        major: 13,
        minor: 1,
        tag: 0,
    };
    let v2 = BytecodeVersion {
        major: 13,
        minor: 2,
        tag: 0,
    };
    let v3 = BytecodeVersion {
        major: 13,
        minor: 3,
        tag: 0,
    };
    assert!(v1 < v2);
    assert!(v2 < v3);
    assert!(v1 < v3);
}
