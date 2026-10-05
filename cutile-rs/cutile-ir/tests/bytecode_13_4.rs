/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Versioned wire roundtrips and static capability/configuration refusals.

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::{BytecodeVersion, Opcode};
use cutile_ir::capabilities::TargetCapabilities;
use cutile_ir::ir::*;

fn tile(shape: &[i64], scalar: ScalarType) -> Type {
    Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    })
}

fn kernel(args: &[Type], body: impl FnOnce(&mut Module, BlockId, &[Value])) -> Module {
    let mut module = Module::new("coverage_13_4");
    let (region, block, values) = build_single_block_region(&mut module, args);
    body(&mut module, block, &values);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("coverage".into()))
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

fn write(module: &Module, version: BytecodeVersion) -> Result<Vec<u8>, cutile_ir::Error> {
    cutile_ir::bytecode::write_bytecode_version(module, version)
}

#[test]
fn all_five_new_ops_roundtrip_and_reject_older_wire_versions() {
    for opcode in [
        Opcode::Insert,
        Opcode::FPowI,
        Opcode::GdcLaunchDependentsTko,
        Opcode::GdcWaitTko,
        Opcode::MemoryFenceAliasTko,
    ] {
        let args = match opcode {
            Opcode::Insert => vec![
                tile(&[4], ScalarType::F32),
                tile(&[8], ScalarType::F32),
                tile(&[], ScalarType::I32),
            ],
            Opcode::FPowI => vec![tile(&[4], ScalarType::F32), tile(&[4], ScalarType::I32)],
            _ => vec![Type::Token],
        };
        let module = kernel(&args, |m, b, v| {
            let result = match opcode {
                Opcode::Insert => args[1].clone(),
                Opcode::FPowI => args[0].clone(),
                _ => Type::Token,
            };
            let (op, _) = OpBuilder::new(opcode, Location::Unknown)
                .operands(v.iter().copied())
                .result(result)
                .build(m);
            append_op(m, b, op);
        });
        TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120")
            .validate_module(&module)
            .unwrap();
        let bytes = write(&module, BytecodeVersion::V13_4).unwrap();
        assert!(cutile_ir::decode_bytecode(&bytes)
            .unwrap()
            .contains("v13.4"));
        for old in [BytecodeVersion::V13_2, BytecodeVersion::V13_3] {
            assert!(write(&module, old)
                .unwrap_err()
                .to_string()
                .contains("13.4"));
        }
    }
}

#[test]
fn atan2_roundtrips_on_every_supported_version() {
    let t = tile(&[4], ScalarType::F32);
    let module = kernel(&[t.clone(), t.clone()], |m, b, v| {
        let (op, _) = OpBuilder::new(Opcode::Atan2, Location::Unknown)
            .operands(v.iter().copied())
            .result(t.clone())
            .build(m);
        append_op(m, b, op);
    });
    assert_eq!(Opcode::Atan2 as u32, 0x6e);
    for version in BytecodeVersion::SUPPORTED {
        assert!(cutile_ir::decode_bytecode(&write(&module, version).unwrap()).is_ok());
    }
}

#[test]
fn float_widening_nearest_away_requires_13_4() {
    let module = kernel(&[tile(&[4], ScalarType::F16)], |m, b, v| {
        let (op, _) = OpBuilder::new(Opcode::FToF, Location::Unknown)
            .operand(v[0])
            .attr("rounding_mode", Attribute::i32(7))
            .result(tile(&[4], ScalarType::F32))
            .build(m);
        append_op(m, b, op);
    });
    TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120")
        .validate_module(&module)
        .unwrap();
    assert!(write(&module, BytecodeVersion::V13_4).is_ok());
    assert!(write(&module, BytecodeVersion::V13_3).is_err());
}

#[test]
fn arch_and_version_are_conjunctive_not_numeric_generation_shortcuts() {
    let fp4 = kernel(&[tile(&[2], ScalarType::F4E2M1FN)], |_, _, _| {});
    for sm in ["sm_100", "sm_107", "sm_110", "sm_120", "sm_121"] {
        TargetCapabilities::new(BytecodeVersion::V13_4, sm)
            .validate_module(&fp4)
            .unwrap();
    }
    assert!(TargetCapabilities::new(BytecodeVersion::V13_4, "sm_90")
        .validate_module(&fp4)
        .is_err());
    assert!(TargetCapabilities::new(BytecodeVersion::V13_2, "sm_120")
        .validate_module(&fp4)
        .is_err());
    let new_scale = kernel(&[tile(&[2], ScalarType::F8E5M3FNU)], |_, _, _| {});
    assert!(TargetCapabilities::new(BytecodeVersion::V13_4, "sm_100")
        .validate_module(&new_scale)
        .is_err());
    assert!(TargetCapabilities::new(BytecodeVersion::V13_3, "sm_120")
        .validate_module(&new_scale)
        .is_err());
}

#[test]
fn rejects_invalid_shapes_allocations_and_power_exponents_before_assembly() {
    let shape = kernel(&[tile(&[3], ScalarType::F32)], |_, _, _| {});
    let caps = TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120");
    assert!(caps
        .validate_module(&shape)
        .unwrap_err()
        .to_string()
        .contains("powers of two"));
    let allocation = kernel(&[], |m, b, _| {
        let (op, _) = OpBuilder::new(Opcode::Alloca, Location::Unknown)
            .attr("num_elem", Attribute::int(4, ScalarType::I64))
            .attr("alignment", Attribute::int(3, ScalarType::I64))
            .result(Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Pointer(Box::new(PointerType {
                    pointee: ScalarType::F32,
                })),
            }))
            .build(m);
        append_op(m, b, op);
    });
    assert!(caps
        .validate_module(&allocation)
        .unwrap_err()
        .to_string()
        .contains("alignment"));
    let exponent = kernel(
        &[tile(&[4], ScalarType::F32), tile(&[4], ScalarType::I64)],
        |m, b, v| {
            let (op, _) = OpBuilder::new(Opcode::FPowI, Location::Unknown)
                .operands(v.iter().copied())
                .result(tile(&[4], ScalarType::F32))
                .build(m);
            append_op(m, b, op);
        },
    );
    assert!(caps
        .validate_module(&exponent)
        .unwrap_err()
        .to_string()
        .contains("exponent"));
}

#[test]
fn worker_hint_range_tracks_the_selected_dialect() {
    let mut module = kernel(&[], |_, _, _| {});
    let entry = module.functions[0];
    module.op_mut(entry).attributes.push((
        "optimization_hints".into(),
        Attribute::OptimizationHints(OptimizationHints {
            entries: vec![(
                "sm_120".into(),
                vec![("num_worker_warps_per_cta".into(), Attribute::i32(16))],
            )],
        }),
    ));
    TargetCapabilities::new(BytecodeVersion::V13_3, "sm_120")
        .validate_module(&module)
        .unwrap();
    assert!(TargetCapabilities::new(BytecodeVersion::V13_4, "sm_120")
        .validate_module(&module)
        .unwrap_err()
        .to_string()
        .contains("num_worker_warps_per_cta"));
}
