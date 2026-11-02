/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA 13.5 multicast contracts and versioned function/debug records.
use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion, Opcode};
use cutile_ir::capabilities::TargetCapabilities;
use cutile_ir::ir::*;
#[path = "../../cutile/tests/support/tile_ir_reference.rs"]
mod reference;

fn tile(shape: &[i64], scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    })
}

fn multicast(
    opcode: Opcode,
    element: ScalarType,
    reduction: i64,
    with_token: bool,
) -> (Module, OpId) {
    let mut module = Module::new("multicast_13_5");
    let view = Type::PartitionView(PartitionViewType {
        tile_shape: vec![4],
        dim_map: vec![0],
        padding_value: None,
        tensor_view: TensorViewType {
            shape: vec![8],
            strides: vec![1],
            element_type: element,
        },
    })
    .with_pointer_attribute(PointerAttribute::Multicast);
    let load = opcode == Opcode::MultimemLoadReduceViewTko;
    let value = tile(&[4], element);
    let mut types = vec![view];
    if !load {
        types.push(value.clone());
    }
    types.push(tile(&[], ScalarType::I32));
    if with_token {
        types.push(Type::Token);
    }
    let (region, block, args) = build_single_block_region(&mut module, &types);
    let mut segments = vec![1];
    if !load {
        segments.push(1);
    }
    segments.extend([1, i64::from(with_token)]);
    let mut op = OpBuilder::new(opcode, Location::Unknown)
        .operands(args.iter().copied())
        .attr(
            "memory_ordering_semantics",
            Attribute::i32(if load { 2 } else { 3 }),
        )
        .attr("memory_scope", Attribute::i32(2))
        .attr("inbounds", Attribute::Array(vec![Attribute::Bool(true)]))
        .attr(
            "operandSegmentSizes",
            Attribute::Array(segments.into_iter().map(Attribute::i32).collect()),
        );
    if opcode != Opcode::MultimemStoreViewTko {
        op = op.attr("reduction", Attribute::i32(reduction));
    }
    if load {
        op = op.result(value);
    }
    let (op, _) = op.result(Type::Token).build(&mut module);
    append_op(&mut module, block, op);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(
        Opcode::Entry,
        Location::FileLineCol {
            filename: "multicast.rs".into(),
            line: 17,
            column: 4,
        },
    )
    .attr("sym_name", Attribute::String("multicast".into()))
    .attr(
        "function_type",
        Attribute::Type(Type::Func(FuncType {
            inputs: types,
            results: vec![],
        })),
    )
    .region(region)
    .build(&mut module);
    module.functions.push(entry);
    (module, op)
}

fn validate(module: &Module) -> Result<(), cutile_ir::capabilities::CapabilityError> {
    TargetCapabilities::new(BytecodeVersion::V13_5, "sm_100").validate_module(module)
}

#[test]
fn multicast_operations_support_tokens_and_reject_old_wire_versions() {
    for opcode in [
        Opcode::MultimemLoadReduceViewTko,
        Opcode::MultimemReduceViewTko,
        Opcode::MultimemStoreViewTko,
    ] {
        for with_token in [false, true] {
            let (module, _) = multicast(opcode, ScalarType::I32, 0, with_token);
            validate(&module).unwrap();
            let bytes = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
            let decoded = cutile_ir::decode_bytecode(&bytes).unwrap();
            assert!(decoded.contains("v13.5"));
            for old in [
                BytecodeVersion::V13_2,
                BytecodeVersion::V13_3,
                BytecodeVersion::V13_4,
            ] {
                assert!(write_bytecode_version(&module, old)
                    .unwrap_err()
                    .to_string()
                    .contains("13.5"));
            }
            for sm in ["sm_100", "sm_100a", "sm_103", "sm_107"] {
                TargetCapabilities::new(BytecodeVersion::V13_5, sm)
                    .validate_module(&module)
                    .unwrap();
            }
            for sm in ["sm_90", "sm_110", "sm_120", "sm_121"] {
                assert!(TargetCapabilities::new(BytecodeVersion::V13_5, sm)
                    .validate_module(&module)
                    .is_err());
            }
        }
    }
}

#[test]
fn multicast_reduction_type_contracts() {
    for opcode in [
        Opcode::MultimemLoadReduceViewTko,
        Opcode::MultimemReduceViewTko,
    ] {
        for element in [
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F16,
            ScalarType::BF16,
            ScalarType::F32,
            ScalarType::F64,
        ] {
            for reduction in 0..=8 {
                let (module, _) = multicast(opcode, element, reduction, false);
                let expected = if element.is_integer() {
                    reduction <= 7
                } else {
                    reduction == 0
                        || (opcode == Opcode::MultimemLoadReduceViewTko
                            && matches!(element, ScalarType::F16 | ScalarType::BF16)
                            && matches!(reduction, 1 | 2))
                };
                assert_eq!(
                    validate(&module).is_ok(),
                    expected,
                    "{opcode:?} {element:?} {reduction}"
                );
            }
        }
    }
}

#[test]
fn invalid_multicast_ordering_bounds_and_token_segments_are_rejected() {
    for (name, value, diagnostic) in [
        ("memory_ordering_semantics", Attribute::i32(3), "ordering"),
        ("memory_scope", Attribute::i32(8), "scope"),
        ("inbounds", Attribute::Array(vec![]), "inbounds"),
        ("operandSegmentSizes", Attribute::Array(vec![]), "segment"),
    ] {
        let (mut module, op) =
            multicast(Opcode::MultimemLoadReduceViewTko, ScalarType::I32, 0, true);
        module
            .op_mut(op)
            .attributes
            .iter_mut()
            .find(|(key, _)| key == name)
            .unwrap()
            .1 = value;
        assert!(validate(&module)
            .unwrap_err()
            .to_string()
            .contains(diagnostic));
    }
    let (mut module, op) = multicast(Opcode::MultimemReduceViewTko, ScalarType::I32, 0, true);
    module
        .op_mut(op)
        .attributes
        .iter_mut()
        .find(|(key, _)| key == "memory_ordering_semantics")
        .unwrap()
        .1 = Attribute::i32(0);
    assert!(validate(&module).is_err());
}

#[test]
fn pointer_classification_roundtrips_in_every_pointer_and_view_form() {
    for name in ["none", "unicast", "multicast"] {
        for text in [
            format!("!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<{name}>>"),
            format!("!cuda_tile.tile<4x!cuda_tile.ptr<i32, #cuda_tile.ptr_attr<{name}>>>"),
            format!("!cuda_tile.tensor_view<8xi32, strides=[1], #cuda_tile.ptr_attr<{name}>>"),
        ] {
            let ty = Type::parse(&text).unwrap();
            assert_eq!(ty.pointer_attribute().unwrap().name(), name);
        }
    }
}

#[test]
fn pointer_shape_and_offset_operations_cannot_drop_classification() {
    for classification in [PointerAttribute::Unicast, PointerAttribute::Multicast] {
        for opcode in [Opcode::Reshape, Opcode::Broadcast, Opcode::Offset] {
            let mut module = Module::new("classified_pointer_shape");
            let pointer = |shape: Vec<i64>| {
                Type::Tile(TileType {
                    shape,
                    element_type: TileElementType::Pointer(Box::new(PointerType {
                        pointee: ScalarType::I32,
                    })),
                })
            };
            let source = pointer(vec![1]).with_pointer_attribute(classification);
            let destination = pointer(if opcode == Opcode::Broadcast {
                vec![4]
            } else {
                vec![1]
            });
            let mut arguments = vec![source];
            if opcode == Opcode::Offset {
                arguments.push(tile(&[], ScalarType::I32));
            }
            let (region, block, values) = build_single_block_region(&mut module, &arguments);
            let (operation, _) = OpBuilder::new(opcode, Location::Unknown)
                .operands(values)
                .result(destination.clone().with_pointer_attribute(classification))
                .build(&mut module);
            append_op(&mut module, block, operation);
            let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
            append_op(&mut module, block, ret);
            let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
                .attr("sym_name", Attribute::String("shape".into()))
                .attr(
                    "function_type",
                    Attribute::Type(Type::Func(FuncType {
                        inputs: arguments,
                        results: vec![],
                    })),
                )
                .region(region)
                .build(&mut module);
            module.functions.push(entry);
            validate(&module).unwrap();
            module.op_mut(operation).result_types[0] = destination;
            assert!(validate(&module)
                .unwrap_err()
                .to_string()
                .contains("preserve ptr_attr"));
        }
    }
}

#[test]
fn function_argument_attributes_have_a_positional_13_5_field() {
    let mut module = Module::new("arg_attrs");
    let (region, block, _) = build_single_block_region(&mut module, &[tile(&[], ScalarType::I32)]);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(
        Opcode::Entry,
        Location::FileLineCol {
            filename: "argument_attributes.rs".into(),
            line: 9,
            column: 2,
        },
    )
    .attr("sym_name", Attribute::String("entry".into()))
    .attr(
        "function_type",
        Attribute::Type(Type::Func(FuncType {
            inputs: vec![tile(&[], ScalarType::I32)],
            results: vec![],
        })),
    )
    .region(region)
    .build(&mut module);
    module.functions.push(entry);
    for version in [BytecodeVersion::V13_4, BytecodeVersion::V13_5] {
        let absent = write_bytecode_version(&module, version).unwrap();
        module.op_mut(entry).attributes.push((
            "arg_attrs".into(),
            Attribute::Array(vec![Attribute::Dictionary(vec![])]),
        ));
        assert_eq!(
            write_bytecode_version(&module, version).unwrap(),
            absent,
            "all-empty attrs normalize away"
        );
        module.op_mut(entry).attributes.pop();
        assert!(cutile_ir::decode_bytecode(&absent)
            .unwrap()
            .contains("DISubprogram"));
    }
    module.op_mut(entry).attributes.push((
        "arg_attrs".into(),
        Attribute::Array(vec![Attribute::Dictionary(vec![(
            "cuda_tile.div_by".into(),
            Attribute::DivBy(DivBy {
                divisor: 16,
                every: None,
                along: None,
            }),
        )])]),
    ));
    let bytes = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
    reference::verify_if_configured(&bytes);
    assert!(cutile_ir::decode_bytecode(&bytes)
        .unwrap()
        .contains("cuda_tile.div_by"));
    assert!(write_bytecode_version(&module, BytecodeVersion::V13_4)
        .unwrap_err()
        .to_string()
        .contains("13.5"));
    module.op_mut(entry).attributes.last_mut().unwrap().1 =
        Attribute::Array(vec![Attribute::Bool(true)]);
    assert!(write_bytecode_version(&module, BytecodeVersion::V13_5)
        .unwrap_err()
        .to_string()
        .contains("dictionary"));
}

#[test]
fn debug_declaration_flags_require_13_5() {
    let mut module = Module::new("debug_declaration");
    let (region, block, _) = build_single_block_region(&mut module, &[]);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("entry".into()))
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
    let file = DIFile {
        name: "declaration.rs".into(),
        directory: "src".into(),
    };
    module.op_mut(entry).location = Location::DebugInfo(DebugInfoLoc {
        filename: "src/declaration.rs".into(),
        line: 1,
        column: 1,
        scope: DebugScope::Subprogram(DISubprogram {
            file: file.clone(),
            line: 1,
            name: "entry".into(),
            linkage_name: "entry".into(),
            compile_unit: DICompileUnit { file },
            scope_line: 1,
            is_declaration: true,
        }),
    });
    let bytes = write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
    reference::verify_if_configured(&bytes);
    assert!(cutile_ir::decode_bytecode(&bytes)
        .unwrap()
        .contains("flags=1"));
    for version in [
        BytecodeVersion::V13_2,
        BytecodeVersion::V13_3,
        BytecodeVersion::V13_4,
    ] {
        let error = write_bytecode_version(&module, version)
            .unwrap_err()
            .to_string();
        assert!(error.contains("declaration flags"), "{error}");
    }
}
