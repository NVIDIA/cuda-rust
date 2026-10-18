/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile library descriptions (public preview `simt`, Tile IR 13.5): the
//! 13.5 function table layout against hand-built bytes, the version lock
//! and header tag, decoder round trips, and the textual form against the
//! fixtures `tileiras` consumes in the Tile IR repository.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{
    write_bytecode_version, BytecodeVersion, Opcode, Section, MAGIC, VERSION_TAG_CONTAINS_PREVIEW,
};
use cutile_ir::capabilities::TargetCapabilities;
use cutile_ir::ir::*;
use cutile_ir::tilelib::{
    add_extern_func, implementation_module, interface_module, preview_call, verify_library,
    ExternFunc, ExternImpl, FragmentArg, TileArg,
};
use cutile_ir::{decode_bytecode, module_preview_content, write_bytecode};

// =========================================================================
// Library fixtures (the DKG tilelib test inputs)
// =========================================================================

fn tile_f32(n: i64) -> TileType {
    TileType {
        shape: vec![n],
        element_type: TileElementType::Scalar(ScalarType::F32),
    }
}

/// `tile_ir/compiler/test/tileiras/Inputs/tilelib_interface.mlir`
fn tilelib_interface() -> Module {
    interface_module(
        "tilelib",
        &[ExternFunc::new(
            "tilelib_scale",
            vec![TileArg::new("arg0", tile_f32(128), Access::InOut)],
        )],
    )
}

fn scale_fragment() -> FragmentType {
    FragmentType::new(
        vec![1],
        ScalarType::F32,
        MemorySpace::Local,
        Layout::contiguous_mem(1),
        Layout::elementwise(128),
    )
}

/// `tile_ir/compiler/test/tileiras/Inputs/tilelib_implementation.mlir`
fn tilelib_implementation() -> Module {
    implementation_module(
        "tilelib",
        &[ExternImpl::new(
            "tilelib_scale_impl",
            "tilelib_scale",
            vec![FragmentArg::new("arg0", scale_fragment(), Access::InOut)],
        )],
    )
}

/// A kernel calling the library: `preview$call @tilelib_scale(%x)`.
fn caller_program() -> Module {
    caller_program_with(Vec::new())
}

/// The caller kernel with constant arguments on its call.
fn caller_program_with(const_args: Vec<SymExpr>) -> Module {
    let mut module = Module::new("program");
    let tile = Type::Tile(tile_f32(128));
    let (region, block, args) = build_single_block_region(&mut module, std::slice::from_ref(&tile));
    let (_, results) = preview_call(
        &mut module,
        block,
        "tilelib_scale",
        const_args,
        &args,
        std::slice::from_ref(&tile),
    );
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("caller".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![tile],
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    assert_eq!(results.len(), 1);
    module
}

/// A plain kernel with an empty body, as the existing version tests build.
fn plain_kernel(name: &str) -> Module {
    let mut module = Module::new("m");
    let (region, block, _) = build_single_block_region(&mut module, &[]);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![],
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    module
}

// =========================================================================
// Section framing helpers
// =========================================================================

fn read_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result = 0u64;
    let mut shift = 0;
    loop {
        let byte = data[*pos];
        *pos += 1;
        result |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}

/// `(section id, payload)` in file order; header framing per spec 4.1.
fn sections(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&bytes[..8], &MAGIC);
    let mut pos = 12;
    let mut out = Vec::new();
    loop {
        let id_byte = bytes[pos];
        pos += 1;
        let id = id_byte & 0x7F;
        if id == Section::EndOfBytecode as u8 {
            assert_eq!(pos, bytes.len());
            return out;
        }
        let len = read_varint(bytes, &mut pos) as usize;
        if id_byte & 0x80 != 0 {
            let alignment = read_varint(bytes, &mut pos) as usize;
            while pos % alignment != 0 {
                assert_eq!(bytes[pos], 0xCB);
                pos += 1;
            }
        }
        out.push((id, bytes[pos..pos + len].to_vec()));
        pos += len;
    }
}

fn section(bytes: &[u8], id: Section) -> Vec<u8> {
    sections(bytes)
        .into_iter()
        .find(|(s, _)| *s == id as u8)
        .map(|(_, payload)| payload)
        .unwrap_or_else(|| panic!("no section {id:?}"))
}

fn strings(bytes: &[u8]) -> Vec<String> {
    let payload = section(bytes, Section::String);
    let mut pos = 0;
    let count = read_varint(&payload, &mut pos) as usize;
    while pos % 4 != 0 {
        pos += 1;
    }
    let offsets: Vec<usize> = (0..count)
        .map(|i| {
            let o = pos + 4 * i;
            u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize
        })
        .collect();
    let data = &payload[pos + 4 * count..];
    (0..count)
        .map(|i| {
            let end = offsets.get(i + 1).copied().unwrap_or(data.len());
            String::from_utf8(data[offsets[i]..end].to_vec()).unwrap()
        })
        .collect()
}

fn string_index(table: &[String], s: &str) -> u8 {
    u8::try_from(table.iter().position(|t| t == s).expect(s)).unwrap()
}

fn header(bytes: &[u8]) -> (u8, u8, u16) {
    (
        bytes[8],
        bytes[9],
        u16::from_le_bytes([bytes[10], bytes[11]]),
    )
}

// =========================================================================
// Version selection and header tag (spec 4.1)
// =========================================================================

#[test]
fn preview_content_is_detected() {
    assert!(module_preview_content(&tilelib_interface()));
    assert!(module_preview_content(&tilelib_implementation()));
    assert!(module_preview_content(&caller_program()));
    assert!(!module_preview_content(&plain_kernel("k")));
}

#[test]
fn default_write_selects_13_5_and_sets_the_preview_tag() {
    for module in [
        tilelib_interface(),
        tilelib_implementation(),
        caller_program(),
    ] {
        let bytes = write_bytecode(&module).unwrap();
        assert_eq!(header(&bytes), (13, 5, VERSION_TAG_CONTAINS_PREVIEW));
        // The decoder reads the tag back and keeps the version proper.
        let dump = decode_bytecode(&bytes).unwrap();
        assert!(dump.contains("TileIR bytecode v13.5.1"), "{dump}");
        assert!(dump.contains("contains public-preview content"), "{dump}");
    }
    // A module without preview content keeps the default version and a
    // clear tag.
    let bytes = write_bytecode(&plain_kernel("k")).unwrap();
    assert_eq!(header(&bytes), (13, 4, 0));
}

#[test]
fn preview_content_is_version_locked() {
    let module = tilelib_interface();
    for version in [
        BytecodeVersion::V13_2,
        BytecodeVersion::V13_3,
        BytecodeVersion::V13_4,
    ] {
        let error = write_bytecode_version(&module, version).unwrap_err();
        assert!(
            error.to_string().contains("public-preview content")
                && error.to_string().contains("13.5"),
            "{error}"
        );
    }
    let bytes = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
    assert_eq!(header(&bytes), (13, 5, VERSION_TAG_CONTAINS_PREVIEW));
    // An assembler reads preview content only at the version it calls
    // current, so a newer version is written on request with the same tag.
    let newer = BytecodeVersion {
        major: 13,
        minor: 6,
        tag: 0,
    };
    let bytes = write_bytecode_version(&module, newer).unwrap();
    assert_eq!(header(&bytes), (13, 6, VERSION_TAG_CONTAINS_PREVIEW));
    // 13.5 without preview content is an ordinary header.
    let bytes = write_bytecode_version(&plain_kernel("k"), BytecodeVersion::V13_5).unwrap();
    assert_eq!(header(&bytes), (13, 5, 0));
    assert!(BytecodeVersion::SUPPORTED.contains(&BytecodeVersion::V13_5));
    assert_eq!(BytecodeVersion::CURRENT, BytecodeVersion::V13_4);
}

// =========================================================================
// Function table layout (spec 4.5, 4.6)
// =========================================================================

/// Kernel entry at 13.5: hints always present (empty), then the positional
/// `arg_attrs` array, then the body. Below 13.5 the entry is unchanged.
#[test]
fn kernel_entry_layout_at_13_5_and_before() {
    let module = plain_kernel("k");
    let bytes = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
    let k = string_index(&strings(&bytes), "k");
    // The only type is the signature `() -> ()`, at index 0.
    assert_eq!(
        section(&bytes, Section::Func),
        vec![
            0x01, // one function
            k,    // name
            0x00, // signature type index
            0x02, // flags: kernel, form normal
            0x01, // debug list index
            0x0B, 0x00, // optimization hints: tag 11, empty dictionary
            0x06, 0x00, // arg_attrs: empty array
            0x03, 0x5C, 0x00, 0x00, // body: `return` with no results or operands
            0xCB, 0xCB, 0xCB, // align to 8
        ]
    );
    let bytes = write_bytecode_version(&module, BytecodeVersion::V13_4).unwrap();
    let k = string_index(&strings(&bytes), "k");
    let mut expected = vec![0x01, k, 0x00, 0x02, 0x01, 0x03, 0x5C, 0x00, 0x00];
    expected.resize(16, 0xCB);
    assert_eq!(section(&bytes, Section::Func), expected);
}

/// A kernel's hints are written when present at every version; the 13.5
/// layout drops the presence flag bit.
#[test]
fn kernel_hints_flag_bit_is_reclaimed_at_13_5() {
    let mut module = plain_kernel("k");
    let entry = module.functions[0];
    module.op_mut(entry).attributes.push((
        "optimization_hints".into(),
        Attribute::OptimizationHints(OptimizationHints {
            entries: vec![(
                "sm_120".into(),
                vec![("occupancy".into(), Attribute::i32(2))],
            )],
        }),
    ));
    let old = write_bytecode_version(&module, BytecodeVersion::V13_4).unwrap();
    assert_eq!(section(&old, Section::Func)[3], 0x02 | 0x04);
    let new = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
    assert_eq!(section(&new, Section::Func)[3], 0x02);
    let dump = decode_bytecode(&new).unwrap();
    assert!(
        dump.contains(
            "optimization_hints: #cuda_tile.optimization_hints<sm_120 = {occupancy = 2}>"
        ),
        "{dump}"
    );
}

/// Extern declaration: normal form, body length 0, const_params (spec 4.5).
#[test]
fn extern_func_entry_layout() {
    let bytes = write_bytecode(&tilelib_interface()).unwrap();
    let table = strings(&bytes);
    let name = string_index(&table, "tilelib_scale");
    let access = string_index(&table, "cuda_tile.preview$access");
    // Types: f32, tile<128xf32>, then the signature.
    assert_eq!(
        section(&bytes, Section::Func),
        vec![
            0x01, name, 0x02, // one function, name, signature
            0x01, // flags: private (a declaration), not a kernel, form normal
            0x01, // debug list index
            0x06, 0x01, 0x0A, 0x01, access, 0xDF, 0x01, 0x02, // arg_attrs: [{access = inout}]
            0x00, // body length 0: an extern declaration
            0x06, 0x00, // const_params: empty array
        ]
    );
    let dump = decode_bytecode(&bytes).unwrap();
    assert!(
        dump.contains("preview$func extern @tilelib_scale : (tile<128xf32>) -> (tile<128xf32>)"),
        "{dump}"
    );
    assert!(
        dump.contains("arg_attrs: [{cuda_tile.preview$access = #cuda_tile.preview$access<inout>}]"),
        "{dump}"
    );
    assert!(dump.contains("const_params: []"), "{dump}");
}

/// Implementation: form bits 01, the op payload (flags, symbols, type,
/// arg_attrs, one empty region), body length 0 (spec 4.6). Without a
/// `synchronization` clause (`None`) nothing is attached, so bit 8 is clear
/// and no value is written, as for an upstream module parsed without the
/// clause; the explicit default clause of the scratch fixtures
/// (`Some(Synchronization::None)`) sets the bit and writes the value.
#[test]
fn extern_impl_entry_layout() {
    let module = tilelib_implementation();
    let bytes = write_bytecode(&module).unwrap();
    let table = strings(&bytes);
    let symbol = string_index(&table, "tilelib_scale_impl");
    let implements = string_index(&table, "tilelib_scale");
    let access = string_index(&table, "cuda_tile.preview$access");
    // Types: f32, the fragment, then the signature `(fragment) -> ()`.
    let mut expected = vec![
        0x01, symbol, 0x02, // one function, symbol, signature
        0x05, // flags: private (a declaration), form extern impl
        0x01, // debug list index
        0x04, // payload flags: arg_attrs (bit 2) only
        symbol, implements, // sym_name, implements
        0x02,       // function_type index
        0x01, 0x0A, 0x01, access, 0xDF, 0x01, 0x02, // arg_attrs inline: [{access = inout}]
        0x01, 0x00, // one region, zero blocks
        0x00, // body length 0
    ];
    expected.resize(24, 0xCB);
    assert_eq!(section(&bytes, Section::Func), expected);
    let dump = decode_bytecode(&bytes).unwrap();
    assert!(
        dump.contains("preview$extern_impl @tilelib_scale_impl implements @tilelib_scale"),
        "{dump}"
    );
    assert!(!dump.contains("synchronization"), "{dump}");
    assert!(
        dump.contains(
            "preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<(1):(1)>, \
             fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>"
        ),
        "{dump}"
    );

    let mut explicit = ExternImpl::new(
        "tilelib_scale_impl",
        "tilelib_scale",
        vec![FragmentArg::new("arg0", scale_fragment(), Access::InOut)],
    );
    explicit.synchronization = Some(Synchronization::None);
    let module = implementation_module("tilelib", &[explicit]);
    let bytes = write_bytecode(&module).unwrap();
    let table = strings(&bytes);
    let symbol = string_index(&table, "tilelib_scale_impl");
    let implements = string_index(&table, "tilelib_scale");
    let access = string_index(&table, "cuda_tile.preview$access");
    let mut expected = vec![
        0x01, symbol, 0x02, 0x05, 0x01, // as above
        0x84, 0x02, // payload flags: arg_attrs (bit 2) and synchronization (bit 8)
        symbol, implements, 0x02, // sym_name, implements, function_type index
        0x01, 0x0A, 0x01, access, 0xDF, 0x01, 0x02, // arg_attrs inline
        0x00, // synchronization = none, written because the clause is present
        0x01, 0x00, // one region, zero blocks
        0x00, // body length 0
    ];
    expected.resize(24, 0xCB);
    assert_eq!(section(&bytes, Section::Func), expected);
    let dump = decode_bytecode(&bytes).unwrap();
    assert!(
        dump.contains("synchronization: #cuda_tile.preview$synchronization<none>"),
        "{dump}"
    );
    let text = module.to_mlir_text();
    assert!(
        text.contains("synchronization = #cuda_tile.preview$synchronization<none>"),
        "{text}"
    );
}

/// Every optional implementation clause gets its flag bit and payload, and
/// the decoder reads them all back.
#[test]
fn extern_impl_optional_clauses_roundtrip_through_the_decoder() {
    let mut implementation = ExternImpl::new(
        "transform_sm90",
        "transform",
        vec![FragmentArg::new(
            "value",
            FragmentType::new(
                vec![4],
                ScalarType::F32,
                MemorySpace::Local,
                Layout::contiguous_mem(4),
                Layout::flat(&[1, 32, 4], &[0, 0, 1]),
            ),
            Access::InOut,
        )],
    );
    implementation.synchronization = Some(Synchronization::Warp);
    implementation.arch = vec!["sm90".into(), "sm100".into()];
    implementation.scratch = Some(ScratchRequest::new(SymExpr::Int(128), SymExpr::Int(16)));
    implementation.frontend_abi = vec![("f32".into(), "i32".into())];
    implementation.const_params = vec![
        ConstParam::wildcard(),
        ConstParam::wildcard(),
        ConstParam::wildcard(),
    ];
    implementation.const_args = vec![
        SymExpr::Int(4),
        SymExpr::concrete_type(Type::Scalar(ScalarType::F32)),
        SymExpr::const_str("nvrtc"),
    ];
    let module = implementation_module("lib", &[implementation]);
    let bytes = write_bytecode(&module).unwrap();
    let func = section(&bytes, Section::Func);
    // Flags: const_params, const_args, arg_attrs, scratch, arch, frontendABI,
    // synchronization = bits 0, 1, 2, 4, 6, 7, 8 = 0x1D7.
    assert_eq!(&func[5..7], &[0xD7, 0x03]);
    let dump = decode_bytecode(&bytes).unwrap();
    for expected in [
        "const_params: [#cuda_tile.preview$const_param<name = \"?\", type = none>, ",
        "const_args: [#cuda_tile.preview$expr_int<4>, #cuda_tile.preview$concrete_type<f32>, #cuda_tile.preview$expr_const_str<\"nvrtc\">]",
        "scratch_smem: #cuda_tile.preview$scratch_request<smem.size = #cuda_tile.preview$expr_int<128>, smem.alignment = #cuda_tile.preview$expr_int<16>>",
        "arch: [\"sm90\", \"sm100\"]",
        "frontendABI: {f32 = \"i32\"}",
        "synchronization: #cuda_tile.preview$synchronization<warp>",
    ] {
        assert!(dump.contains(expected), "missing {expected:?} in\n{dump}");
    }
    assert!(
        module.to_mlir_text().contains(
            "cuda_tile.preview$extern_impl @transform_sm90<[?, ?, ?]> implements \
             @transform<[4, f32, \"nvrtc\"]>("
        ),
        "{}",
        module.to_mlir_text()
    );
}

// =========================================================================
// Program modules
// =========================================================================

#[test]
fn program_with_preview_call_validates_writes_and_decodes() {
    let module = caller_program();
    module.verify_dominance().unwrap();
    module.verify_bytecode_indices().unwrap();
    TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120")
        .validate_module(&module)
        .unwrap();
    let error = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120")
        .validate_module(&module)
        .unwrap_err();
    assert!(error.to_string().contains("13.5"), "{error}");

    let bytes = write_bytecode(&module).unwrap();
    assert_eq!(header(&bytes), (13, 5, VERSION_TAG_CONTAINS_PREVIEW));
    let dump = decode_bytecode(&bytes).unwrap();
    assert!(dump.contains("entry @caller"), "{dump}");
    assert!(
        dump.contains("optimization_hints: #cuda_tile.optimization_hints<>"),
        "{dump}"
    );
    assert!(dump.contains("arg_attrs: []"), "{dump}");

    let text = module.to_mlir_text();
    assert!(
        text.contains(
            "%1 = preview$call @tilelib_scale(%0) : (!cuda_tile.tile<128xf32>) -> !cuda_tile.tile<128xf32>"
        ),
        "{text}"
    );
}

#[test]
fn preview_call_prints_compact_constant_arguments() {
    let mut module = Module::new("m");
    let tile = Type::Tile(tile_f32(4));
    let (region, block, args) = build_single_block_region(&mut module, std::slice::from_ref(&tile));
    preview_call(
        &mut module,
        block,
        "f",
        vec![
            SymExpr::tuple([SymExpr::Int(1)]),
            SymExpr::Int(1),
            SymExpr::plus(SymExpr::Int(1), SymExpr::Int(3)),
            SymExpr::times(
                SymExpr::plus(SymExpr::Int(1), SymExpr::Int(3)),
                SymExpr::Int(4),
            ),
            SymExpr::minus(
                SymExpr::Int(8),
                SymExpr::minus(SymExpr::Int(4), SymExpr::Int(1)),
            ),
        ],
        &args,
        &[tile.clone(), Type::Token],
    );
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("k".into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: vec![tile],
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    let text = module.to_mlir_text();
    // The compact constant-argument form of preview_ops.mlir: trailing
    // comma on one-element tuples, no spaces around operators, parentheses
    // only where precedence needs them.
    assert!(
        text.contains("preview$call @f<[(1,), 1, 1+3, (1+3)*4, 8-(4-1)]>(%0) : (!cuda_tile.tile<4xf32>) -> (!cuda_tile.tile<4xf32>, !cuda_tile.token)"),
        "{text}"
    );

    // A call binds statically known values only, so the negated-variable
    // form is exercised on an implementation, whose constant arguments
    // may forward its own parameters.
    let mut implementation = ExternImpl::new(
        "g",
        "f",
        vec![FragmentArg::new("x", scale_fragment(), Access::InOut)],
    );
    implementation.const_params = vec![ConstParam::index("M"), ConstParam::wildcard()];
    implementation.const_args = vec![SymExpr::var("M"), SymExpr::negate(SymExpr::var("M"))];
    let text = implementation_module("m", &[implementation]).to_mlir_text();
    assert!(
        text.contains("cuda_tile.preview$extern_impl @g<[M: index, ?]> implements @f<[M, -M]>("),
        "{text}"
    );
}

// =========================================================================
// Textual fixtures
// =========================================================================

/// Whitespace-insensitive comparison with three rewrites, each applied to
/// both sides. `//` comment lines are dropped, because the fixtures carry
/// prose that no printer reproduces. A whitespace run is kept (as one
/// space) only between two word characters, so layout and line breaks never
/// matter while a missing space between two words still does. An empty
/// result list `-> ()` is removed, because the fixtures write it on one
/// implementation and omit it on the next while the upstream printer never
/// emits it, and the printer here follows upstream.
fn normalize(text: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || "_$.@%!#\"?".contains(c);
    let mut out = String::new();
    for line in text.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    let mut collapsed = String::new();
    let mut pending_space = false;
    for c in out.chars() {
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space {
            if let Some(prev) = collapsed.chars().last() {
                if is_word(prev) && is_word(c) {
                    collapsed.push(' ');
                }
            }
            pending_space = false;
        }
        collapsed.push(c);
    }
    collapsed.replace("->()", "")
}

fn assert_matches_fixture(module: &Module, fixture: &str) {
    let printed = module.to_mlir_text();
    assert_eq!(
        normalize(&printed),
        normalize(fixture),
        "printed:\n{printed}\nfixture:\n{fixture}"
    );
}

// tile_ir/compiler/test/tileiras/Inputs/tilelib_interface.mlir
const TILELIB_INTERFACE: &str = r#"
// The interface a tile library publishes to its consumers: the extern foreign
// functions that may be called. `tileiras --preview-create-tile-lib
// --preview-interface=` stores the serialized form of this module in the
// `.nvtlifc` section of the .tilelib, and `tileiras -l=<lib>` splices it into
// the program being compiled so that calls to @tilelib_scale resolve without
// the program declaring it itself.
//
// This module and tilelib_implementation.mlir are the two halves of one library
// and deliberately share the module name @tilelib. Only the contained symbols
// survive: both halves are merged into the consumer's own cuda_tile.module and
// the source module name is discarded.
cuda_tile.module @tilelib {
  cuda_tile.preview$func extern @tilelib_scale(
      %arg0: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>
}
"#;

// tile_ir/compiler/test/tileiras/Inputs/tilelib_implementation.mlir
const TILELIB_IMPLEMENTATION: &str = r#"
// The implementation catalog of a tile library: the candidate implementations of
// the interface's extern functions. `extern_impl` has no body -- it names the
// symbol that the accompanying code payload (an LTOIR section, or a provider's
// instantiate() result) is expected to export.
//
// This is the "library use case" from
// tile_ir/test/Dialect/CudaTile/preview_ops.mlir: the impl
// deliberately does not declare @tilelib_scale, because the declaration arrives
// separately from the interface.
//
// The result of @tilelib_scale travels back through the `inout` fragment: a
// simt-path extern_impl may not declare results, and its per-argument access
// must match the declaration's.
//
// `arch` is deliberately omitted so the impl matches every compute capability;
// arch strings are matched exactly after stripping '_', so `arch = ["sm100"]`
// would match sm_100 but not sm_100f.
cuda_tile.module @tilelib {
  cuda_tile.preview$extern_impl @tilelib_scale_impl
    implements @tilelib_scale
    (%arg0: !cuda_tile.preview$fragment<1xf32, local,
       mem_layout = #cuda_tile.preview$layout<(1):(1)>,
       fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
     {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}) -> ()
}
"#;

// tile_ir/compiler/test/python/Integration/CudaTile/Inputs/scratch_tilelib_interface.mlir
const SCRATCH_TILELIB_INTERFACE: &str = r#"
cuda_tile.module @scratch_tilelib {
  // Declare the call backed by a fixed-size scratch allocation.
  cuda_tile.preview$func extern @scratch_fixed(
      %input: !cuda_tile.tile<128xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<128xf32>

  // Declare the call whose scratch size depends on parameter N.
  cuda_tile.preview$func extern @scratch_symbolic<[N: index]>(
      %input: !cuda_tile.tile<[N]xf32>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    -> !cuda_tile.tile<[N]xf32>
}
"#;

// tile_ir/compiler/test/python/Integration/CudaTile/Inputs/scratch_tilelib_implementation.mlir
const SCRATCH_TILELIB_IMPLEMENTATION: &str = r#"
cuda_tile.module @scratch_tilelib {
  // Use a fixed 512-byte, 128-byte-aligned scratch allocation.
  cuda_tile.preview$extern_impl @scratch_fixed_impl implements @scratch_fixed(
      %input: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    synchronization = #cuda_tile.preview$synchronization<none>
    scratch {
      smem.size = 512,
      smem.alignment = 128
    }
    {frontendABI = {"!llvm.ptr<3>" = "!llvm.ptr"}}

  // Compute the scratch size from the concretized parameter N.
  cuda_tile.preview$extern_impl @scratch_symbolic_impl<[N: index]>
    implements @scratch_symbolic<[N]>(
      %input: !cuda_tile.preview$fragment<1xf32, local,
        mem_layout = #cuda_tile.preview$layout<(1):(1)>,
        fragmap = #cuda_tile.preview$layout<(N, 1, 1):(1, 0, 0)>>
        {cuda_tile.preview$access = #cuda_tile.preview$access<inout>})
    synchronization = #cuda_tile.preview$synchronization<none>
    scratch {
      smem.size = N * 4,
      smem.alignment = 128
    }
    {frontendABI = {"!llvm.ptr<3>" = "!llvm.ptr"}}
}
"#;

/// The concrete half of a scratch fixture: `@scratch_symbolic` declares a
/// `tile<[N]xf32>`, a symbolic tile shape this writer does not model (v1
/// emits concrete tiles only), so the comparison stops at that entry.
fn concrete_entries(fixture: &str, symbolic_marker: &str) -> String {
    let end = fixture.find(symbolic_marker).expect("symbolic entry");
    format!("{}}}\n", &fixture[..end])
}

#[test]
fn interface_prints_like_the_tileiras_fixture() {
    assert_matches_fixture(&tilelib_interface(), TILELIB_INTERFACE);
}

#[test]
fn implementation_prints_like_the_tileiras_fixture() {
    assert_matches_fixture(&tilelib_implementation(), TILELIB_IMPLEMENTATION);
}

#[test]
fn scratch_interface_prints_like_the_python_integration_fixture() {
    let module = interface_module(
        "scratch_tilelib",
        &[ExternFunc::new(
            "scratch_fixed",
            vec![TileArg::new("input", tile_f32(128), Access::InOut)],
        )],
    );
    assert_matches_fixture(
        &module,
        &concrete_entries(
            SCRATCH_TILELIB_INTERFACE,
            "  // Declare the call whose scratch size depends on parameter N.",
        ),
    );
}

#[test]
fn scratch_implementation_prints_like_the_python_integration_fixture() {
    let mut implementation = ExternImpl::new(
        "scratch_fixed_impl",
        "scratch_fixed",
        vec![FragmentArg::new("input", scale_fragment(), Access::InOut)],
    );
    implementation.scratch = Some(ScratchRequest::bytes(512));
    implementation.frontend_abi = vec![("!llvm.ptr<3>".into(), "!llvm.ptr".into())];
    // The fixture writes the default `synchronization` clause out, and
    // upstream keeps that as a present attribute (bit 8 on the wire):
    // `Some(Synchronization::None)`, as opposed to an absent clause.
    implementation.synchronization = Some(Synchronization::None);
    let module = implementation_module("scratch_tilelib", &[implementation]);
    assert_matches_fixture(
        &module,
        &concrete_entries(
            SCRATCH_TILELIB_IMPLEMENTATION,
            "  // Compute the scratch size from the concretized parameter N.",
        ),
    );
    // The symbolic entry's pieces that are modelled print the same way.
    let mut symbolic = ExternImpl::new(
        "scratch_symbolic_impl",
        "scratch_symbolic",
        vec![FragmentArg::new(
            "input",
            FragmentType::new(
                vec![1],
                ScalarType::F32,
                MemorySpace::Local,
                Layout::contiguous_mem(1),
                Layout {
                    shapes: vec![SymExpr::var("N"), SymExpr::Int(1), SymExpr::Int(1)],
                    strides: vec![SymExpr::Int(1), SymExpr::Int(0), SymExpr::Int(0)],
                },
            ),
            Access::InOut,
        )],
    );
    symbolic.const_params = vec![ConstParam::index("N")];
    symbolic.const_args = vec![SymExpr::var("N")];
    symbolic.synchronization = Some(Synchronization::None);
    symbolic.scratch = Some(ScratchRequest::new(
        SymExpr::times(SymExpr::var("N"), SymExpr::Int(4)),
        SymExpr::Int(128),
    ));
    symbolic.frontend_abi = vec![("!llvm.ptr<3>".into(), "!llvm.ptr".into())];
    let text = implementation_module("scratch_tilelib", &[symbolic]).to_mlir_text();
    for expected in [
        "cuda_tile.preview$extern_impl @scratch_symbolic_impl<[N: index]> implements @scratch_symbolic<[N]>(",
        "fragmap = #cuda_tile.preview$layout<(N, 1, 1):(1, 0, 0)>>",
        "synchronization = #cuda_tile.preview$synchronization<none>",
        "scratch {smem.size = N * 4, smem.alignment = 128}",
        "{frontendABI = {\"!llvm.ptr<3>\" = \"!llvm.ptr\"}}",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in\n{text}");
    }
}

// =========================================================================
// Verification
// =========================================================================

#[test]
fn library_verification_accepts_the_fixture_pair() {
    let funcs = [ExternFunc::new(
        "tilelib_scale",
        vec![TileArg::new("arg0", tile_f32(128), Access::InOut)],
    )];
    let impls = [ExternImpl::new(
        "tilelib_scale_impl",
        "tilelib_scale",
        vec![FragmentArg::new("arg0", scale_fragment(), Access::InOut)],
    )];
    verify_library(&funcs, &impls).unwrap();
    for module in [tilelib_interface(), tilelib_implementation()] {
        TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120")
            .validate_module(&module)
            .unwrap();
        let error = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120")
            .validate_module(&module)
            .unwrap_err();
        assert!(error.to_string().contains("13.5"), "{error}");
    }
}

#[test]
fn structural_checks_run_on_hand_built_modules() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120");

    // A declaration without access attributes.
    let mut module = Module::new("bad");
    let op = add_extern_func(
        &mut module,
        &ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::InOut)]),
    );
    module
        .op_mut(op)
        .attributes
        .retain(|(k, _)| k != "arg_attrs");
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("cuda_tile.preview$access"));

    // A declaration whose results do not match its bound outputs.
    let mut module = Module::new("bad");
    let op = add_extern_func(
        &mut module,
        &ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::In)]),
    );
    for (name, value) in &mut module.op_mut(op).attributes {
        if name == "function_type" {
            *value = Attribute::Type(Type::Func(FuncType {
                inputs: vec![Type::Tile(tile_f32(128))],
                results: vec![Type::Tile(tile_f32(128))],
            }));
        }
    }
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("out/inout"));

    // A fragment whose mem_layout is not a permutation of its footprint.
    let module = implementation_module(
        "bad",
        &[ExternImpl::new(
            "i",
            "f",
            vec![FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![2],
                    ScalarType::F32,
                    MemorySpace::Local,
                    Layout::flat(&[2], &[2]),
                    Layout::footprint_blocks(128, 2),
                ),
                Access::InOut,
            )],
        )],
    );
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("mem_layout"));

    // A fragmap without three modes.
    let module = implementation_module(
        "bad",
        &[ExternImpl::new(
            "i",
            "f",
            vec![FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![1],
                    ScalarType::F32,
                    MemorySpace::Local,
                    Layout::contiguous_mem(1),
                    Layout::flat(&[128], &[1]),
                ),
                Access::InOut,
            )],
        )],
    );
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("3 modes"));

    // Warp synchronization with the elementwise map (one participant).
    let mut warp = ExternImpl::new(
        "i",
        "f",
        vec![FragmentArg::new("x", scale_fragment(), Access::InOut)],
    );
    warp.synchronization = Some(Synchronization::Warp);
    let module = implementation_module("bad", &[warp]);
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("multiple of 32"));

    // Shared-memory fragments need a broadcast participant mode.
    let module = implementation_module(
        "bad",
        &[ExternImpl::new(
            "i",
            "f",
            vec![FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![128],
                    ScalarType::F32,
                    MemorySpace::Smem,
                    Layout::contiguous_mem(128),
                    Layout::flat(&[1, 32, 128], &[0, 1, 1]),
                ),
                Access::In,
            )],
        )],
    );
    assert!(caps
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("zero stride"));
}

#[test]
fn layout_constructors_cover_their_tiles() {
    let decl = ExternFunc::new(
        "f",
        vec![
            TileArg::new("a", tile_f32(256), Access::In),
            TileArg::new("b", tile_f32(256), Access::InOut),
        ],
    );
    for (fragmap, footprint, memory_space, sync) in [
        (Layout::elementwise(256), 1, MemorySpace::Local, None),
        (
            Layout::warp_lanes(256),
            1,
            MemorySpace::Local,
            Some(Synchronization::Warp),
        ),
        (
            Layout::footprint_blocks(256, 4),
            4,
            MemorySpace::Local,
            Some(Synchronization::None),
        ),
        (
            Layout::shared_whole(128, 256),
            256,
            MemorySpace::Smem,
            Some(Synchronization::Warpgroup),
        ),
    ] {
        let fragment = |access: Access| {
            FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![footprint],
                    ScalarType::F32,
                    memory_space,
                    Layout::contiguous_mem(footprint),
                    fragmap.clone(),
                ),
                access,
            )
        };
        let mut implementation = ExternImpl::new(
            "i",
            "f",
            vec![fragment(Access::In), fragment(Access::InOut)],
        );
        implementation.synchronization = sync;
        verify_library(std::slice::from_ref(&decl), &[implementation]).unwrap();
    }
    // A by-value scalar: rank-0 mem_layout, `in`, local.
    let scalar = ExternFunc::new("s", vec![TileArg::new("v", tile_f32(1), Access::In)]);
    let by_value = ExternImpl::new(
        "s_impl",
        "s",
        vec![FragmentArg::new(
            "v",
            FragmentType::new(
                vec![1],
                ScalarType::F32,
                MemorySpace::Local,
                Layout::scalar_mem(),
                Layout::elementwise(1),
            ),
            Access::In,
        )],
    );
    verify_library(
        std::slice::from_ref(&scalar),
        std::slice::from_ref(&by_value),
    )
    .unwrap();
    let mut by_pointer = by_value;
    by_pointer.args[0].access = Access::InOut;
    let mut decl = scalar;
    decl.args[0].access = Access::InOut;
    assert!(verify_library(&[decl], &[by_pointer])
        .unwrap_err()
        .to_string()
        .contains("rank-0"));
}

#[test]
#[should_panic(expected = "must divide the tile size")]
fn footprint_blocks_rejects_a_footprint_that_does_not_divide_the_tile() {
    let _ = Layout::footprint_blocks(128, 3);
}

#[test]
#[should_panic(expected = "must be positive")]
fn footprint_blocks_rejects_an_empty_footprint() {
    let _ = Layout::footprint_blocks(128, 0);
}

#[test]
#[should_panic(expected = "positive multiple of 32")]
fn warp_lanes_rejects_a_partial_warp() {
    let _ = Layout::warp_lanes(48);
}

/// Symbolic libraries: every variable a fragment or scratch request uses is
/// a named constant parameter of the implementation, bindings have the kind
/// of the declaration parameter, and a named implementation parameter keeps
/// the declaration's name and is forwarded as itself.
#[test]
fn symbolic_libraries_are_checked_against_their_parameters() {
    let mut decl = ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::InOut)]);
    decl.const_params = vec![ConstParam::index("N")];
    let over = |variable: &str| Layout {
        shapes: vec![SymExpr::var(variable), SymExpr::Int(1), SymExpr::Int(1)],
        strides: vec![SymExpr::Int(1), SymExpr::Int(0), SymExpr::Int(0)],
    };
    let implementation = |fragmap: Layout| {
        ExternImpl::new(
            "i",
            "f",
            vec![FragmentArg::new(
                "x",
                FragmentType::new(
                    vec![1],
                    ScalarType::F32,
                    MemorySpace::Local,
                    Layout::contiguous_mem(1),
                    fragmap,
                ),
                Access::InOut,
            )],
        )
    };
    let error = |implementation: ExternImpl| {
        verify_library(std::slice::from_ref(&decl), &[implementation])
            .unwrap_err()
            .to_string()
    };

    // The identity binding of the named parameter verifies.
    let mut good = implementation(over("N"));
    good.const_params = vec![ConstParam::index("N")];
    good.const_args = vec![SymExpr::var("N")];
    verify_library(std::slice::from_ref(&decl), std::slice::from_ref(&good)).unwrap();

    // A wildcard declares nothing, so the fragmap's N is undeclared.
    let mut undeclared = implementation(over("N"));
    undeclared.const_params = vec![ConstParam::wildcard()];
    undeclared.const_args = vec![SymExpr::var("N")];
    assert!(
        error(undeclared.clone()).contains("undeclared symbolic variable 'N'"),
        "{}",
        error(undeclared.clone())
    );
    let module = implementation_module("bad", &[undeclared]);
    let module_error = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120")
        .validate_module(&module)
        .unwrap_err()
        .to_string();
    assert!(
        module_error.contains("undeclared symbolic variable 'N'"),
        "{module_error}"
    );

    // A scratch request is held to the same rule.
    let mut scratch = good.clone();
    scratch.scratch = Some(ScratchRequest::new(
        SymExpr::times(SymExpr::var("T"), SymExpr::Int(4)),
        SymExpr::Int(128),
    ));
    assert!(
        error(scratch.clone())
            .contains("scratch size references an undeclared symbolic variable 'T'"),
        "{}",
        error(scratch)
    );

    // A named implementation parameter keeps the declaration's name...
    let mut renamed = implementation(over("K"));
    renamed.const_params = vec![ConstParam::index("K")];
    renamed.const_args = vec![SymExpr::var("K")];
    assert!(
        error(renamed.clone())
            .contains("must match the implemented function's constant parameter name 'N'"),
        "{}",
        error(renamed)
    );
    // ...and is forwarded as itself.
    let mut derived = good.clone();
    derived.const_args = vec![SymExpr::plus(SymExpr::var("N"), SymExpr::Int(1))];
    assert!(
        error(derived.clone()).contains("identity reference"),
        "{}",
        error(derived)
    );
    // Only wildcards may be bound to anything, and the binding must have the
    // declared kind: a tuple cannot bind an index parameter.
    let mut tuple = implementation(Layout::elementwise(128));
    tuple.const_params = vec![ConstParam::wildcard()];
    tuple.const_args = vec![SymExpr::tuple([SymExpr::Int(128)])];
    assert!(
        error(tuple.clone()).contains("must be an integer expression, got a tuple"),
        "{}",
        error(tuple)
    );

    // Datatype, string and shape parameters take a type, a string constant
    // and a (nested) tuple of integer expressions.
    let mut typed = ExternFunc::new("g", vec![TileArg::new("x", tile_f32(128), Access::InOut)]);
    typed.const_params = vec![
        ConstParam::new("T", Type::PreviewParam(PreviewParamType::DType)),
        ConstParam::new("S", Type::PreviewParam(PreviewParamType::String)),
        ConstParam::new("Shape", Type::PreviewParam(PreviewParamType::Shape)),
    ];
    let mut bound = ExternImpl::new(
        "g_impl",
        "g",
        vec![FragmentArg::new("x", scale_fragment(), Access::InOut)],
    );
    bound.const_params = vec![ConstParam::wildcard(); 3];
    bound.const_args = vec![
        SymExpr::concrete_type(Type::Scalar(ScalarType::F32)),
        SymExpr::const_str("nvrtc"),
        SymExpr::tuple([
            SymExpr::Int(128),
            SymExpr::tuple([SymExpr::Int(2), SymExpr::Int(4)]),
        ]),
    ];
    verify_library(std::slice::from_ref(&typed), std::slice::from_ref(&bound)).unwrap();
    bound.const_args[0] = SymExpr::Int(32);
    let kind_error = verify_library(&[typed], &[bound]).unwrap_err().to_string();
    assert!(kind_error.contains("datatype expression"), "{kind_error}");
}

/// Constant parameter lists: the `?` wildcard belongs to implementations
/// only, and names are unique on either half.
#[test]
fn const_param_lists_are_checked() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120");
    let decl = || ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::InOut)]);

    let mut wildcard = decl();
    wildcard.const_params = vec![ConstParam::wildcard()];
    let error = verify_library(&[wildcard.clone()], &[])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("'?' wildcard is only valid in extern_impl const_params"),
        "{error}"
    );
    let error = caps
        .validate_module(&interface_module("bad", &[wildcard]))
        .unwrap_err()
        .to_string();
    assert!(error.contains("'?' wildcard"), "{error}");

    let mut duplicate = decl();
    duplicate.const_params = vec![ConstParam::index("N"), ConstParam::index("N")];
    let error = verify_library(&[duplicate], &[]).unwrap_err().to_string();
    assert!(
        error.contains("duplicate constant parameter name: 'N'"),
        "{error}"
    );
    let mut implementation = ExternImpl::new(
        "i",
        "f",
        vec![FragmentArg::new("x", scale_fragment(), Access::InOut)],
    );
    implementation.const_params = vec![ConstParam::index("N"), ConstParam::index("N")];
    let error = caps
        .validate_module(&implementation_module("bad", &[implementation]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("duplicate constant parameter name: 'N'"),
        "{error}"
    );
}

/// A `mem_layout` whose domain size and storage span both equal the
/// footprint can still send two elements to one offset; only the offset
/// enumeration sees that.
#[test]
fn overlapping_mem_layout_is_rejected() {
    // (2, 2, 2):(1, 1, 5): domain 8, span 1 + 1 + 1 + 5 = 8, but the
    // elements (1, 0, 0) and (0, 1, 0) both land on offset 1.
    let fragment = FragmentType::new(
        vec![8],
        ScalarType::F32,
        MemorySpace::Local,
        Layout::flat(&[2, 2, 2], &[1, 1, 5]),
        Layout::footprint_blocks(128, 8),
    );
    let implementation = ExternImpl::new(
        "i",
        "f",
        vec![FragmentArg::new("x", fragment, Access::InOut)],
    );
    let error = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120")
        .validate_module(&implementation_module(
            "bad",
            std::slice::from_ref(&implementation),
        ))
        .unwrap_err()
        .to_string();
    assert!(error.contains("modes overlap or leave gaps"), "{error}");
    let decl = ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::InOut)]);
    let error = verify_library(&[decl], &[implementation])
        .unwrap_err()
        .to_string();
    assert!(error.contains("modes overlap or leave gaps"), "{error}");
}

/// The alignment is held to the same rule as the layouts: every variable
/// it names is a declared parameter of the implementation.
#[test]
fn undeclared_alignment_variable_is_rejected() {
    let mut fragment = scale_fragment();
    fragment.alignment = SymExpr::var("A");
    let implementation = ExternImpl::new(
        "i",
        "f",
        vec![FragmentArg::new("x", fragment, Access::InOut)],
    );
    let expected = "field alignment references an undeclared symbolic variable 'A'";
    let error = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120")
        .validate_module(&implementation_module(
            "bad",
            std::slice::from_ref(&implementation),
        ))
        .unwrap_err()
        .to_string();
    assert!(error.contains(expected), "{error}");
    let decl = ExternFunc::new("f", vec![TileArg::new("x", tile_f32(128), Access::InOut)]);
    let error = verify_library(&[decl], &[implementation])
        .unwrap_err()
        .to_string();
    assert!(error.contains(expected), "{error}");
}

/// Integer contexts reject tuples, strings, types and toward-zero division
/// at any depth: layout leaves, the fragment alignment and both scratch
/// fields go through one walker.
#[test]
fn integer_contexts_are_checked_at_depth() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120");
    let nested_div = SymExpr::plus(
        SymExpr::Div(Box::new(SymExpr::Int(8)), Box::new(SymExpr::Int(2))),
        SymExpr::Int(1),
    );
    let nested_tuple = SymExpr::times(SymExpr::tuple([SymExpr::Int(1)]), SymExpr::Int(2));
    let implementation = |fragment: FragmentType| {
        ExternImpl::new(
            "i",
            "f",
            vec![FragmentArg::new("x", fragment, Access::InOut)],
        )
    };

    let mut aligned = scale_fragment();
    aligned.alignment = nested_div.clone();
    let error = caps
        .validate_module(&implementation_module("bad", &[implementation(aligned)]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("fragment alignment must be an integer expression"),
        "{error}"
    );

    let mut leaf = scale_fragment();
    leaf.fragmap.strides[0] = nested_tuple.clone();
    let error = caps
        .validate_module(&implementation_module("bad", &[implementation(leaf)]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("fragmap stride leaf must be an integer expression, got a tuple"),
        "{error}"
    );

    let mut scratch = implementation(scale_fragment());
    scratch.scratch = Some(ScratchRequest::new(nested_tuple, SymExpr::Int(128)));
    let error = caps
        .validate_module(&implementation_module("bad", &[scratch.clone()]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("scratch size must be an integer expression"),
        "{error}"
    );
    scratch.scratch = Some(ScratchRequest::new(SymExpr::Int(512), nested_div));
    let funcs = [ExternFunc::new(
        "f",
        vec![TileArg::new("x", tile_f32(128), Access::InOut)],
    )];
    let error = verify_library(&funcs, &[scratch]).unwrap_err().to_string();
    assert!(
        error.contains("scratch alignment must be an integer expression"),
        "{error}"
    );
}

/// A call binds statically known values: no symbolic variable at any depth
/// and no empty tuple; context builtins stay allowed.
#[test]
fn preview_call_constant_arguments_must_be_static() {
    let caps = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_120");
    caps.validate_module(&caller_program_with(vec![
        SymExpr::builtin("context.num_threads"),
        SymExpr::tuple([SymExpr::Int(4)]),
    ]))
    .unwrap();
    let error = caps
        .validate_module(&caller_program_with(vec![SymExpr::negate(SymExpr::var(
            "M",
        ))]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("statically known") && error.contains("'M'"),
        "{error}"
    );
    let error = caps
        .validate_module(&caller_program_with(vec![SymExpr::tuple([])]))
        .unwrap_err()
        .to_string();
    assert!(error.contains("empty tuple"), "{error}");
}

// =========================================================================
// Determinism and decoder robustness
// =========================================================================

/// Argument names are a formatter-only convention (`ARG_NAMES_ATTR`): the
/// writer skips the attribute, so neither its key nor the names reach the
/// string table, and renaming the arguments leaves the bytes unchanged.
#[test]
fn argument_names_are_not_interned() {
    let named = |decl_arg: &str, impl_arg: &str| {
        let decl = ExternFunc::new(
            "named",
            vec![TileArg::new(decl_arg, tile_f32(128), Access::InOut)],
        );
        let implementation = ExternImpl::new(
            "named_impl",
            "named",
            vec![FragmentArg::new(impl_arg, scale_fragment(), Access::InOut)],
        );
        (
            interface_module("lib", &[decl]),
            implementation_module("lib", &[implementation]),
        )
    };
    let (interface, catalog) = named("left_operand", "right_operand");
    for (module, name) in [(&interface, "left_operand"), (&catalog, "right_operand")] {
        assert!(module.to_mlir_text().contains(&format!("%{name}:")));
        let table = strings(&write_bytecode(module).unwrap());
        assert!(
            !table.iter().any(|s| s == ARG_NAMES_ATTR || s == name),
            "{table:?}"
        );
    }
    let (renamed_interface, renamed_catalog) = named("a", "b");
    assert_eq!(
        write_bytecode(&interface).unwrap(),
        write_bytecode(&renamed_interface).unwrap()
    );
    assert_eq!(
        write_bytecode(&catalog).unwrap(),
        write_bytecode(&renamed_catalog).unwrap()
    );
}

#[test]
fn library_modules_serialize_deterministically() {
    for module in [
        tilelib_interface(),
        tilelib_implementation(),
        caller_program(),
    ] {
        assert_eq!(
            write_bytecode(&module).unwrap(),
            write_bytecode(&module).unwrap()
        );
    }
}

#[test]
fn truncated_library_bytecode_is_rejected() {
    let bytes = write_bytecode(&tilelib_implementation()).unwrap();
    for cut in [16, bytes.len() / 2, bytes.len() - 3] {
        assert!(decode_bytecode(&bytes[..cut]).is_err(), "cut at {cut}");
    }
}
