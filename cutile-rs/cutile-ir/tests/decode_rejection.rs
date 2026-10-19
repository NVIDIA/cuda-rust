/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode decoder rejection tests.
//!
//! Verifies that `decode_bytecode` returns errors on malformed input
//! (bad magic, truncated headers/sections, garbage bytes) and succeeds
//! on valid minimal modules.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{BytecodeVersion, Opcode, Section, MAGIC};
use cutile_ir::ir::*;
use cutile_ir::{decode_bytecode, write_bytecode};

// =========================================================================
// Helpers
// =========================================================================

/// Build a minimal valid bytecode buffer: header + EndOfBytecode marker, no sections.
fn minimal_bytecode() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.push(BytecodeVersion::CURRENT.major);
    buf.push(BytecodeVersion::CURRENT.minor);
    buf.extend_from_slice(&BytecodeVersion::CURRENT.tag.to_le_bytes());
    buf.push(Section::EndOfBytecode as u8);
    buf
}

/// Build a module with a single kernel containing a return op.
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

// =========================================================================
// Rejection tests
// =========================================================================

#[test]
fn reject_bad_magic() {
    let bad = vec![0x00; 16];
    let result = decode_bytecode(&bad);
    assert!(result.is_err(), "should reject bad magic");
}

#[test]
fn reject_truncated_header() {
    // Only 4 bytes -- too short to contain the 8-byte magic.
    let short = vec![0x42; 4];
    let result = decode_bytecode(&short);
    assert!(result.is_err(), "should reject truncated input (4 bytes)");
}

#[test]
fn reject_zero_length_bytecode() {
    let empty: &[u8] = &[];
    let result = decode_bytecode(empty);
    assert!(result.is_err(), "should reject zero-length bytecode");
}

#[test]
fn accept_header_only() {
    // Valid header followed immediately by EndOfBytecode -- no real sections.
    let buf = minimal_bytecode();
    let result = decode_bytecode(&buf);
    assert!(
        result.is_ok(),
        "header + EndOfBytecode should decode: {result:?}"
    );
}

#[test]
fn reject_invalid_section_marker() {
    // Valid header, then a section-id byte that is above the max known id,
    // followed by garbage. The decoder should fail when trying to parse
    // sections.
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC);
    buf.push(BytecodeVersion::CURRENT.major);
    buf.push(BytecodeVersion::CURRENT.minor);
    buf.extend_from_slice(&BytecodeVersion::CURRENT.tag.to_le_bytes());
    // A section byte with id=0x7F (well above NUM_SECTIONS) and no alignment
    // bit, followed by a nonsense varint length and trailing garbage.
    buf.push(0x7F);
    // Nonsense varint that claims a huge payload length.
    buf.push(0xFF);
    buf.push(0xFF);
    buf.push(0xFF);
    buf.push(0x07);
    let result = decode_bytecode(&buf);
    assert!(
        result.is_err(),
        "should reject invalid/oversized section marker"
    );
}

#[test]
fn reject_truncated_type_section() {
    // Build a valid module so we get correct bytecode, then find and truncate
    // inside the type section.
    let module = build_kernel("trunc_type", &[], |_m, _b, _a| {});
    let bytes = write_bytecode(&module).expect("write should succeed");

    // The type section must exist because the entry function has a function type.
    // Truncate the bytecode roughly in the middle.
    let mid = bytes.len() / 2;
    let truncated = &bytes[..mid];
    let result = decode_bytecode(truncated);
    assert!(
        result.is_err(),
        "should reject bytecode truncated at midpoint ({mid}/{} bytes)",
        bytes.len()
    );
}

#[test]
fn reject_truncated_string_section() {
    let module = build_kernel("trunc_string", &[], |_m, _b, _a| {});
    let bytes = write_bytecode(&module).expect("write should succeed");

    // Truncate just past the header (12 bytes: 8 magic + 4 version).
    // This removes all section data, causing the decoder to hit EOF mid-section.
    let header_plus = 14; // just past the header, into first section header
    if bytes.len() > header_plus {
        let truncated = &bytes[..header_plus];
        let result = decode_bytecode(truncated);
        assert!(
            result.is_err(),
            "should reject bytecode truncated right after header"
        );
    }
}

#[test]
fn reject_truncated_function_section() {
    let module = build_kernel("trunc_func", &[], |_m, _b, _a| {});
    let bytes = write_bytecode(&module).expect("write should succeed");

    // Truncate a few bytes before the end -- this should clip into either
    // the function, type, or string section payload.
    let near_end = bytes.len().saturating_sub(3);
    if near_end > 12 {
        let truncated = &bytes[..near_end];
        let result = decode_bytecode(truncated);
        assert!(
            result.is_err(),
            "should reject bytecode truncated near end ({near_end}/{} bytes)",
            bytes.len()
        );
    }
}

#[test]
fn valid_minimal_module_roundtrip() {
    let module = build_kernel("sanity", &[], |_m, _b, _a| {});
    let bytecode = write_bytecode(&module).expect("bytecode write failed");
    let decoded = decode_bytecode(&bytecode).expect("bytecode decode failed");
    assert!(
        decoded.contains("TileIR bytecode v13."),
        "decoded output should contain the version header"
    );
    assert!(
        decoded.contains("sanity"),
        "decoded output should contain the kernel name"
    );
}
