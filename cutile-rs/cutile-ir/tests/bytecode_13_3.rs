/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused CUDA Tile IR 13.3 bytecode encoding tests.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{Opcode, Section, MAGIC};
use cutile_ir::ir::*;

fn build_kernel(
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    let mut module = Module::new("bytecode_13_3");
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

fn tile(shape: &[i64], scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    })
}

fn tensor_view_f32() -> TensorViewType {
    TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![8192, 128],
        strides: vec![128, 1],
    }
}

fn gather_scatter_view_f32() -> Type {
    Type::GatherScatterView(GatherScatterViewType {
        tile_shape: vec![64, 64],
        tensor_view: tensor_view_f32(),
        sparse_dim: 0,
        padding_value: None,
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

#[derive(Clone, Copy)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn byte(&mut self) -> u8 {
        let byte = self.data[self.pos];
        self.pos += 1;
        byte
    }

    fn bytes(&mut self, len: usize) -> &'a [u8] {
        let start = self.pos;
        self.pos += len;
        &self.data[start..start + len]
    }

    fn varint(&mut self) -> u64 {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = self.byte();
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return result;
            }
            shift += 7;
        }
    }

    fn align_to(&mut self, alignment: usize) {
        let padding = (alignment - (self.pos % alignment)) % alignment;
        self.pos += padding;
    }
}

fn section_payload(bytecode: &[u8], wanted: Section) -> &[u8] {
    let mut r = Reader::new(bytecode);
    assert_eq!(r.bytes(MAGIC.len()), MAGIC);
    let _major = r.byte();
    let _minor = r.byte();
    let _tag_lo = r.byte();
    let _tag_hi = r.byte();

    loop {
        let id_with_align = r.byte();
        let id = id_with_align & 0x7f;
        if id == Section::EndOfBytecode as u8 {
            panic!("section {:?} not found", wanted);
        }
        let len = r.varint() as usize;
        if id_with_align & 0x80 != 0 {
            let alignment = r.varint() as usize;
            r.align_to(alignment);
        }
        let payload = r.bytes(len);
        if id == wanted as u8 {
            return payload;
        }
    }
}

fn first_function_body(bytecode: &[u8]) -> Vec<u8> {
    let payload = section_payload(bytecode, Section::Func);
    let mut r = Reader::new(payload);
    assert_eq!(r.varint(), 1);
    let _name = r.varint();
    let _sig = r.varint();
    let flags = r.byte();
    assert_eq!(
        flags & 0x04,
        0,
        "test helpers do not decode optimization hints"
    );
    let _loc = r.varint();
    let body_len = r.varint() as usize;
    r.bytes(body_len).to_vec()
}

fn body_reader(module: &Module) -> Reader<'static> {
    let bytecode = cutile_ir::write_bytecode(module).expect("write bytecode");
    let body = first_function_body(&bytecode);
    Reader::new(Box::leak(body.into_boxed_slice()))
}

#[test]
fn pack_unpack_use_13_3_opcodes_and_operand_order() {
    let packed = tile(&[32], ScalarType::I8);
    let unpacked = tile(&[64], ScalarType::F4E2M1FN);
    let module = build_kernel(
        "pack_unpack",
        &[unpacked.clone(), packed.clone()],
        |m, b, args| {
            let (pack, _) = OpBuilder::new(Opcode::Pack, Location::Unknown)
                .operand(args[0])
                .result(packed)
                .build(m);
            append_op(m, b, pack);
            let (unpack, _) = OpBuilder::new(Opcode::Unpack, Location::Unknown)
                .operand(args[1])
                .result(unpacked)
                .build(m);
            append_op(m, b, unpack);
        },
    );

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::Pack.as_u16() as u64);
    let _result_type = r.varint();
    assert_eq!(r.varint(), 0);
    assert_eq!(r.varint(), Opcode::Unpack.as_u16() as u64);
    let _result_type = r.varint();
    assert_eq!(r.varint(), 1);
}

#[test]
fn mmaf_scaled_uses_13_3_opcode_and_python_operand_order() {
    let input = tile(&[16, 16], ScalarType::F4E2M1FN);
    let acc = tile(&[16, 16], ScalarType::F32);
    let lhs_scale = tile(&[16, 1], ScalarType::F8E8M0FNU);
    let rhs_scale = tile(&[1, 16], ScalarType::F8E8M0FNU);
    let module = build_kernel(
        "mmaf_scaled",
        &[input.clone(), input, acc.clone(), lhs_scale, rhs_scale],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaFScaled, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .operand(args[4])
                .result(acc)
                .build(m);
            append_op(m, b, op);
        },
    );

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::MmaFScaled.as_u16() as u64);
    let _result_type = r.varint();
    for expected in 0..=4 {
        assert_eq!(r.varint(), expected);
    }
}

#[test]
fn alloca_encodes_flags_and_inline_integer_attributes() {
    let module = build_kernel("alloca", &[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::Alloca, Location::Unknown)
            .result(ptr_f32())
            .attr("num_elem", Attribute::i32(64))
            .attr("alignment", Attribute::i32(128))
            .attr("global", Attribute::Bool(true))
            .build(m);
        append_op(m, b, op);
    });

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::Alloca.as_u16() as u64);
    let _result_type = r.varint();
    assert_eq!(r.varint(), 1);
    assert_eq!(r.varint(), 64);
    assert_eq!(r.varint(), 128);
}

#[test]
fn mmaf_fast_acc_flag_tracks_boolean_value() {
    let input = tile(&[16, 16], ScalarType::F8E4M3FN);
    let acc = tile(&[16, 16], ScalarType::F32);
    let module = build_kernel(
        "mmaf_fast_acc",
        &[input.clone(), input, acc.clone()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MmaF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .result(acc)
                .attr("fast_acc", Attribute::Bool(false))
                .build(m);
            append_op(m, b, op);
        },
    );

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::MmaF.as_u16() as u64);
    let _result_type = r.varint();
    assert_eq!(r.varint(), 0);
}

#[test]
fn exp_encodes_13_3_default_rounding_mode() {
    let f32_tile = tile(&[4], ScalarType::F32);
    let module = build_kernel("exp", std::slice::from_ref(&f32_tile), |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Exp, Location::Unknown)
            .operand(args[0])
            .result(f32_tile.clone())
            .build(m);
        append_op(m, b, op);
    });

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::Exp.as_u16() as u64);
    let _result_type = r.varint();
    assert_eq!(r.varint(), 5);
    assert_eq!(r.varint(), 0);
}

#[test]
fn atomic_red_view_tko_encoding_matches_python_order() {
    let gsv = gather_scatter_view_f32();
    let sparse_index = tile(&[8], ScalarType::I32);
    let col_index = tile(&[], ScalarType::I32);
    let value = tile(&[64, 64], ScalarType::F32);
    let module = build_kernel(
        "atomic_red_view_tko",
        &[gsv, sparse_index, col_index, value, Type::Token],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AtomicRedViewTko, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .operand(args[2])
                .operand(args[3])
                .operand(args[4])
                .result(Type::Token)
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

    let mut r = body_reader(&module);
    assert_eq!(r.varint(), Opcode::AtomicRedViewTko.as_u16() as u64);
    assert_eq!(r.varint(), 1);
    let _token_type = r.varint();
    assert_eq!(r.varint(), 1);
    assert_eq!(r.varint(), 1);
    assert_eq!(r.varint(), 1);
    assert_eq!(r.varint(), 0);
    assert_eq!(r.varint(), 0);
    assert_eq!(r.varint(), 2);
    assert_eq!(r.varint(), 1);
    assert_eq!(r.varint(), 2);
    assert_eq!(r.varint(), 3);
    assert_eq!(r.varint(), 4);
}

#[test]
fn global_section_encodes_13_3_visibility_and_constant_flag() {
    let mut module = Module::new("globals");
    module.globals.push(Global {
        sym_name: "device_only".into(),
        value: DenseElements {
            element_type: tile(&[1], ScalarType::I32),
            shape: vec![1],
            data: 42i32.to_le_bytes().to_vec(),
        },
        alignment: 128,
        constant: true,
        symbol_visibility: SymbolVisibility::Private,
    });
    let bytecode = cutile_ir::write_bytecode(&module).expect("write bytecode");
    let decoded = cutile_ir::decode_bytecode(&bytecode).expect("decode bytecode");
    assert!(decoded.contains("private"));
    assert!(decoded.contains("constant=true"));
}
