/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::render_module;
use crate::{
    MlirAttribute, MlirBlock, MlirBlockArgument, MlirBlockId, MlirFloatType, MlirLocation,
    MlirModule, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueId, MlirValueUse,
};

fn module(root: MlirOperation) -> MlirModule {
    MlirModule {
        root,
        profile: "interop-test".into(),
    }
}

fn argument(id: u64, location: MlirLocation) -> MlirBlockArgument {
    MlirBlockArgument {
        id: MlirValueId(id),
        ty: MlirType::Integer(32),
        location,
    }
}

#[test]
fn generic_syntax_is_stable() {
    let mut add = MlirOperation::new("arith.addi").unwrap();
    add.results.push(MlirResult {
        id: MlirValueId(2),
        ty: MlirType::Integer(32),
    });
    add.operands = [0, 1]
        .map(|id| MlirValueUse {
            id: MlirValueId(id),
            ty: MlirType::Integer(32),
        })
        .into();
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(0),
            arguments: vec![
                argument(0, MlirLocation::Unknown),
                argument(1, MlirLocation::Unknown),
            ],
            operations: vec![add],
        }],
    });
    let plan = module(root);
    let text = render_module(&plan).unwrap();
    assert_eq!(text, render_module(&plan).unwrap());
    assert!(
        text.contains("%v2 = \"arith.addi\"(%v0, %v1) : (i32, i32) -> (i32)"),
        "{text}"
    );
}

#[test]
fn operation_properties_use_mlir_generic_syntax() {
    let mut branch = MlirOperation::new("probe.branch").unwrap();
    branch.operands.push(MlirValueUse {
        id: MlirValueId(0),
        ty: MlirType::Integer(32),
    });
    branch.successors = vec![MlirBlockId(1), MlirBlockId(2)];
    branch.properties.insert(
        "operandSegmentSizes".into(),
        MlirAttribute::DenseI32Array(vec![1, 0]),
    );
    branch
        .attributes
        .insert("probe.marker".into(), MlirAttribute::Unit);
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.regions.push(MlirRegion {
        blocks: vec![
            MlirBlock {
                id: MlirBlockId(0),
                arguments: vec![argument(0, MlirLocation::Unknown)],
                operations: vec![branch],
            },
            MlirBlock {
                id: MlirBlockId(1),
                arguments: vec![],
                operations: vec![],
            },
            MlirBlock {
                id: MlirBlockId(2),
                arguments: vec![],
                operations: vec![],
            },
        ],
    });
    let text = render_module(&module(root)).unwrap();
    assert!(text.contains("\"probe.branch\"(%v0) [^"), "{text}");
    assert!(
        text.contains("<{operandSegmentSizes = array<i32: 1, 0>}> {probe.marker = unit}"),
        "{text}"
    );
}

#[test]
fn dense_i64_properties_render_without_stringly_typed_escapes() {
    let mut root = MlirOperation::new("probe.indices").unwrap();
    root.properties
        .insert("position".into(), MlirAttribute::DenseI64Array(vec![0, 2]));
    let text = render_module(&module(root)).unwrap();
    assert!(text.contains("<{position = array<i64: 0, 2>}>"), "{text}");
}

#[test]
fn individual_argument_locations_and_nested_operation_locations_survive() {
    let first = MlirLocation::Named {
        name: "first".into(),
        child: Box::new(MlirLocation::Unknown),
    };
    let second = MlirLocation::FileLineCol {
        file: "kernel.rs".into(),
        line: 17,
        column: 4,
    };
    let mut child = MlirOperation::new("probe.child").unwrap();
    child.location = MlirLocation::CallSite {
        callee: Box::new(first.clone()),
        caller: Box::new(second.clone()),
    };
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(0),
            arguments: vec![argument(0, first), argument(1, second)],
            operations: vec![child],
        }],
    });
    let text = render_module(&module(root)).unwrap();
    assert!(
        text.contains("%v0: i32 loc(\"first\"), %v1: i32 loc(\"kernel.rs\":17:4)"),
        "{text}"
    );
    assert!(
        text.contains("loc(callsite(\"first\" at \"kernel.rs\":17:4))"),
        "{text}"
    );
}

#[test]
fn target_types_and_exact_float_bits_survive() {
    let mut root = MlirOperation::new("probe.types").unwrap();
    root.properties.insert(
        "types".into(),
        MlirAttribute::Array(vec![
            MlirAttribute::Type(MlirType::Tuple(vec![])),
            MlirAttribute::Type(MlirType::Index),
            MlirAttribute::Type(MlirType::Vector {
                shape: vec![2, 4],
                element: Box::new(MlirType::Float(MlirFloatType::F32)),
            }),
            MlirAttribute::Type(MlirType::Float(MlirFloatType::Other("f8E4M3FN".into()))),
            MlirAttribute::Type(MlirType::dialect("!cute.layout<\"(4,8):(1,4)\">").unwrap()),
        ]),
    );
    root.properties.insert(
        "value".into(),
        MlirAttribute::Float {
            value: "0xFFC01234".into(),
            ty: MlirType::Float(MlirFloatType::F32),
        },
    );
    let text = render_module(&module(root)).unwrap();
    assert!(
        text.contains("[tuple<>, index, vector<2x4xf32>, f8E4M3FN, !cute.layout<\"(4,8):(1,4)\">]"),
        "{text}"
    );
    assert!(text.contains("value = 0xFFC01234 : f32"), "{text}");
}

#[test]
fn upstream_string_escaping_handles_carriage_return_unicode_and_quotes() {
    let mut root = MlirOperation::new("probe.strings").unwrap();
    root.attributes
        .insert("probe.text".into(), MlirAttribute::String("a\r\"é".into()));
    let text = render_module(&module(root)).unwrap();
    assert!(text.contains(r#"probe.text = "a\0D\"\C3\A9""#), "{text}");
}

#[test]
fn hyphenated_dictionary_keys_and_symbol_names_are_quoted() {
    let mut root = MlirOperation::new("probe.keys").unwrap();
    root.properties
        .insert("property-key".into(), MlirAttribute::Unit);
    root.attributes.insert(
        "probe.dictionary-key".into(),
        MlirAttribute::Dictionary(
            [(
                "nested-key".into(),
                MlirAttribute::SymbolRef("symbol-name".into()),
            )]
            .into_iter()
            .collect(),
        ),
    );
    let text = render_module(&module(root)).unwrap();
    assert!(text.contains(r#"<{"property-key" = unit}>"#), "{text}");
    assert!(
        text.contains(r#""probe.dictionary-key" = {"nested-key" = @"symbol-name"}"#),
        "{text}"
    );
}

#[test]
fn undefined_operands_return_a_diagnostic() {
    let mut root = MlirOperation::new("probe.invalid").unwrap();
    root.operands.push(MlirValueUse {
        id: MlirValueId(77),
        ty: MlirType::Integer(32),
    });
    assert!(
        render_module(&module(root))
            .unwrap_err()
            .to_string()
            .contains("has no definition")
    );
}
