/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR 13.5: multicast contracts, versioned function and debug records,
//! the public-preview `preview$call` (foreign tile-library calls) and the
//! preview header bit.
//!
//! The translator checks use `CUTILE_REFERENCE_TRANSLATE` when configured,
//! otherwise a `cuda-tile-translate` next to the selected 13.5 `tileiras`.
//! They never need a GPU.

use std::path::{Path, PathBuf};
use std::process::Command;

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

fn floc(line: u32, column: u32) -> Location {
    Location::FileLineCol {
        filename: "src/kernels/foreign.rs".to_string(),
        line,
        column,
    }
}

fn kernel(name: &str, args: &[Type], body: impl FnOnce(&mut Module, BlockId, &[Value])) -> Module {
    let mut module = Module::new("coverage_13_5");
    let (region, block, values) = build_single_block_region(&mut module, args);
    body(&mut module, block, &values);
    let (ret, _) = OpBuilder::new(Opcode::Return, floc(9, 1)).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, floc(3, 1))
        .attr("sym_name", Attribute::String(name.into()))
        .attr(
            "function_type",
            Attribute::Type(Type::Func(FuncType {
                inputs: args.to_vec(),
                results: vec![],
            })),
        )
        .region(region)
        .build(&mut module);
    module.functions.push(entry);
    module
}

fn zeros(module: &mut Module, block: BlockId, ty: &Type) -> Value {
    let Type::Tile(t) = ty else {
        panic!("zeros of a non-tile type")
    };
    let TileElementType::Scalar(scalar) = t.element_type else {
        panic!("zeros of a pointer tile")
    };
    let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: Type::Tile(TileType {
                    shape: vec![],
                    element_type: TileElementType::Scalar(scalar),
                }),
                shape: t.shape.clone(),
                data: vec![0; scalar.byte_width()],
            }),
        )
        .result(ty.clone())
        .build(module);
    append_op(module, block, op);
    res[0]
}

/// One `preview$call` with constant arguments, an implicit out operand and an
/// ordering token, as the compiler emits for a foreign `cholesky` call.
fn call(
    module: &mut Module,
    block: BlockId,
    callee: &str,
    const_args: Vec<PreviewConstArg>,
    operands: &[Value],
    results: &[Type],
) -> Vec<Value> {
    let (op, res) = OpBuilder::new(Opcode::PreviewCall, floc(5, 7))
        .operands(operands.iter().copied())
        .results(results.iter().cloned())
        .attr("callee", Attribute::String(callee.into()))
        .attr(
            "const_args",
            Attribute::Array(
                const_args
                    .into_iter()
                    .map(Attribute::PreviewConstArg)
                    .collect(),
            ),
        )
        .build(module);
    append_op(module, block, op);
    res
}

fn cholesky_module() -> Module {
    let a = tile(&[1, 32, 32], ScalarType::F64);
    let info = tile(&[1], ScalarType::I32);
    kernel("cholesky_entry", &[], |m, b, _| {
        let a_val = zeros(m, b, &a);
        let info_val = zeros(m, b, &info);
        let (tok, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(Type::Token)
            .build(m);
        append_op(m, b, tok);
        call(
            m,
            b,
            "cholesky",
            vec![
                PreviewConstArg::Str("potrf".into()),
                PreviewConstArg::Str("lower".into()),
                PreviewConstArg::Int(1),
                PreviewConstArg::Int(32),
                PreviewConstArg::Datatype(Type::Scalar(ScalarType::F64)),
            ],
            &[a_val, info_val, tok_res[0]],
            &[a.clone(), info.clone(), Type::Token],
        );
    })
}

fn write(module: &Module, version: BytecodeVersion) -> Result<Vec<u8>, cutile_ir::Error> {
    cutile_ir::bytecode::write_bytecode_version(module, version)
}

#[test]
fn preview_call_requires_13_5_and_sets_the_header_preview_bit() {
    let module = cholesky_module();
    assert!(module.uses_preview_ops());
    let bytes = write(&module, BytecodeVersion::V13_5).unwrap();
    assert_eq!(&bytes[..12], b"\x7fTileIR\x00\x0d\x05\x01\x00");
    assert!(cutile_ir::decode_bytecode(&bytes)
        .unwrap()
        .contains("v13.5 (preview)"));
    for old in [
        BytecodeVersion::V13_2,
        BytecodeVersion::V13_3,
        BytecodeVersion::V13_4,
    ] {
        let message = write(&module, old).unwrap_err().to_string();
        assert!(message.contains("13.5"), "{message}");
    }
}

#[test]
fn plain_13_5_modules_have_no_preview_bit_and_decode() {
    let t = tile(&[4], ScalarType::F32);
    let module = kernel("plain", &[t.clone(), t.clone()], |m, b, v| {
        let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operands(v.iter().copied())
            .attr("rounding_mode", Attribute::i32(0))
            .result(t.clone())
            .build(m);
        append_op(m, b, op);
    });
    assert!(!module.uses_preview_ops());
    let bytes = write(&module, BytecodeVersion::V13_5).unwrap();
    assert_eq!(&bytes[..12], b"\x7fTileIR\x00\x0d\x05\x00\x00");
    let decoded = cutile_ir::decode_bytecode(&bytes).unwrap();
    assert!(decoded.contains("v13.5\n"), "{decoded}");
    assert!(decoded.contains("entry @plain"), "{decoded}");
}

/// A foreign call is checked against the selected version, negotiated or
/// pinned with `CUTILE_BYTECODE_VERSION`, like every other feature: below
/// 13.5 it is an error that names the requirement.
#[test]
fn foreign_calls_require_a_13_5_target() {
    use cutile_ir::requirements::Feature;
    let module = cholesky_module();
    let target = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_80");
    let message = Feature::ForeignCall
        .check(&target, &Location::Unknown)
        .unwrap_err()
        .message;
    assert_eq!(
        message,
        "preview$call (foreign tile-library call) requires Tile IR 13.5 or newer; \
         selected Tile IR 13.4, target sm_80"
    );
    let message = target.validate_module(&module).unwrap_err().message;
    assert!(
        message.contains("preview$call requires Tile IR 13.5 or newer; selected Tile IR 13.4"),
        "{message}"
    );
    TargetCapabilities::new(BytecodeVersion::V13_5, "sm_80")
        .validate_module(&module)
        .unwrap();
}

#[test]
fn preview_call_operands_must_be_tiles_or_tokens() {
    let ptr = Type::Pointer(PointerType {
        pointee: ScalarType::F32,
    });
    let module = kernel("bad", std::slice::from_ref(&ptr), |m, b, v| {
        call(m, b, "f", vec![], v, &[tile(&[], ScalarType::F32)]);
    });
    let target = TargetCapabilities::new(BytecodeVersion::V13_5, "sm_80");
    let message = target.validate_module(&module).unwrap_err().message;
    assert!(message.contains("tiles or tokens"), "{message}");
}

/// Prefer an explicitly configured translator, including one built by
/// cuda-tile-rs without a toolkit. Explicitly selected tools must pass the test;
/// a missing or incompatible executable is an error rather than a skip.
fn translator() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("CUTILE_REFERENCE_TRANSLATE") {
        return Some(path.into());
    }
    let tileiras = cutile_ir::toolchain::tileiras_from_env();
    let tileiras = if tileiras.components().count() == 1 {
        which(&tileiras)?
    } else {
        tileiras
    };
    let translate = tileiras.parent()?.join("cuda-tile-translate");
    if !translate.is_file()
        || cutile_ir::toolchain::negotiate_bytecode_version(&tileiras, Some("13.5".as_ref()))
            .is_err()
    {
        eprintln!("skipped: no cuda-tile-translate accepting Tile IR 13.5 next to {tileiras:?}");
        return None;
    }
    Some(translate)
}

fn which(program: &Path) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(program))
        .find(|path| path.is_file())
}

fn to_mlir(translate: &Path, bytes: &[u8]) -> String {
    let dir = std::env::temp_dir().join(format!("cutile-13-5-{}", uuid_like()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("module.tilebc");
    std::fs::write(&file, bytes).unwrap();
    let output = Command::new(translate)
        .arg("--cudatilebc-to-mlir")
        .arg("--preview=simt")
        .arg(&file)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

#[test]
fn official_translator_reads_13_5_preview_calls() {
    let Some(translate) = translator() else {
        return;
    };
    let module = cholesky_module();
    let mlir = to_mlir(&translate, &write(&module, BytecodeVersion::V13_5).unwrap());
    assert!(
        mlir.contains(r#"preview$call @cholesky<["potrf", "lower", 1, 32, f64]>"#),
        "{mlir}"
    );
    assert!(mlir.contains("!cuda_tile.token"), "{mlir}");
}
