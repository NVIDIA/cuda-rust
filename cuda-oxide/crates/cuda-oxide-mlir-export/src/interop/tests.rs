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
    child.operands = [0, 1]
        .map(|id| MlirValueUse {
            id: MlirValueId(id),
            ty: MlirType::Integer(32),
        })
        .into();
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
    assert!(
        text.contains("\"probe.child\"(%v0, %v1) : (i32, i32) -> () loc(callsite("),
        "argument locations must not change operand types: {text}"
    );
}

#[test]
fn nested_blocks_preserve_known_and_unknown_argument_locations() {
    let mut nested = MlirOperation::new("probe.nested").unwrap();
    nested.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(1),
            arguments: vec![
                argument(1, MlirLocation::Unknown),
                argument(
                    2,
                    MlirLocation::Named {
                        name: "inner".into(),
                        child: Box::new(MlirLocation::Unknown),
                    },
                ),
            ],
            operations: vec![],
        }],
    });
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(0),
            arguments: vec![argument(
                0,
                MlirLocation::Named {
                    name: "outer".into(),
                    child: Box::new(MlirLocation::Unknown),
                },
            )],
            operations: vec![nested],
        }],
    });
    let text = render_module(&module(root)).unwrap();
    assert!(text.contains("%v0: i32 loc(\"outer\")):"), "{text}");
    assert!(
        text.contains("%v1: i32, %v2: i32 loc(\"inner\")):"),
        "{text}"
    );
    assert!(!text.contains("loc(unknown)"), "{text}");
}

#[test]
fn block_header_text_inside_a_string_attribute_is_unchanged() {
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.attributes.insert(
        "probe.text".into(),
        MlirAttribute::String("^block1v1(%v0: i32):\nnext".into()),
    );
    root.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(0),
            arguments: vec![argument(
                0,
                MlirLocation::Named {
                    name: "argument".into(),
                    child: Box::new(MlirLocation::Unknown),
                },
            )],
            operations: vec![],
        }],
    });
    let text = render_module(&module(root)).unwrap();
    assert!(
        text.contains("^block1v1(%v0: i32 loc(\"argument\")):"),
        "{text}"
    );
    assert!(
        text.contains(r#"probe.text = "^block1v1(%v0: i32):\nnext""#),
        "{text}"
    );
}

#[test]
fn argument_locations_preserve_fused_metadata_and_callsite_escaping() {
    let mut root = MlirOperation::new("probe.container").unwrap();
    root.regions.push(MlirRegion {
        blocks: vec![MlirBlock {
            id: MlirBlockId(0),
            arguments: vec![argument(
                0,
                MlirLocation::Fused {
                    metadata: Some(Box::new(MlirAttribute::String("a\r\"é".into()))),
                    locations: vec![MlirLocation::CallSite {
                        callee: Box::new(MlirLocation::Named {
                            name: "first".into(),
                            child: Box::new(MlirLocation::Unknown),
                        }),
                        caller: Box::new(MlirLocation::FileLineCol {
                            file: "kernel.rs".into(),
                            line: 17,
                            column: 4,
                        }),
                    }],
                },
            )],
            operations: vec![],
        }],
    });
    let text = render_module(&module(root)).unwrap();
    assert!(
        text.contains(
            r#"%v0: i32 loc(fused<"a\0D\"\C3\A9">[callsite("first" at "kernel.rs":17:4)])"#
        ),
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
fn raw_type_spellings_reject_controls_when_enum_constructors_are_bypassed() {
    for ty in [
        MlirType::Dialect("!probe.type<\n i32>".into()),
        MlirType::Float(MlirFloatType::Other("f32\0".into())),
        MlirType::Tuple(vec![MlirType::Function {
            inputs: vec![MlirType::Vector {
                shape: vec![2],
                element: Box::new(MlirType::Dialect("!probe.type<\t i32>".into())),
            }],
            results: vec![],
        }]),
    ] {
        let mut root = MlirOperation::new("probe.types").unwrap();
        root.results.push(MlirResult {
            id: MlirValueId(0),
            ty: ty.clone(),
        });
        let error = render_module(&module(root)).expect_err("raw types must remain control-free");
        assert!(
            error.to_string().contains("control character"),
            "{ty:?}: {error}"
        );
    }
}

#[test]
fn raw_attribute_spellings_reject_controls_when_enum_constructors_are_bypassed() {
    for attr in [
        MlirAttribute::Dialect("#probe.attr<\n^block1v1(%v0: i32):\n>".into()),
        MlirAttribute::Float {
            value: "0x7FC00000\r".into(),
            ty: MlirType::Float(MlirFloatType::F32),
        },
        MlirAttribute::Array(vec![MlirAttribute::Dictionary(
            [(
                "nested".into(),
                MlirAttribute::Dialect("#probe.attr<\t>".into()),
            )]
            .into(),
        )]),
        MlirAttribute::Array(vec![MlirAttribute::Type(MlirType::Dialect(
            "!probe.type<\u{85}>".into(),
        ))]),
    ] {
        let mut root = MlirOperation::new("probe.attributes").unwrap();
        root.attributes.insert("probe.value".into(), attr.clone());
        let error =
            render_module(&module(root)).expect_err("raw attributes must remain control-free");
        assert!(
            error.to_string().contains("control character"),
            "{attr:?}: {error}"
        );
    }
}

#[test]
fn fused_location_metadata_rejects_raw_control_characters() {
    let mut root = MlirOperation::new("probe.location").unwrap();
    root.location = MlirLocation::Fused {
        metadata: Some(Box::new(MlirAttribute::Array(vec![
            MlirAttribute::Dialect("#probe.metadata<\n>".into()),
        ]))),
        locations: vec![MlirLocation::Unknown],
    };
    let error =
        render_module(&module(root)).expect_err("raw location metadata must remain control-free");
    assert!(error.to_string().contains("control character"), "{error}");
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
