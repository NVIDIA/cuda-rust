/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fmt::Write;

use crate::{
    MlirAttribute, MlirBlock, MlirFloatType, MlirLocation, MlirModule, MlirOperation, MlirRegion,
    MlirType,
};

/// Render one module using MLIR's generic operation syntax.
pub fn render_module(module: &MlirModule) -> String {
    let mut output = String::new();
    render_operation(&mut output, &module.root, 0);
    output
}

fn render_operation(output: &mut String, operation: &MlirOperation, indent: usize) {
    write_indent(output, indent);
    if !operation.results.is_empty() {
        for (index, result) in operation.results.iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            let _ = write!(output, "%v{}", result.id.0);
        }
        output.push_str(" = ");
    }
    let _ = write!(output, "\"{}\"(", operation.name);
    for (index, operand) in operation.operands.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        let _ = write!(output, "%v{}", operand.id.0);
    }
    output.push(')');

    if !operation.successors.is_empty() {
        output.push_str(" [");
        for (index, successor) in operation.successors.iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            let _ = write!(output, "^bb{}", successor.0);
        }
        output.push(']');
    }

    if !operation.properties.is_empty() {
        output.push_str(" <{");
        render_named_attributes(output, &operation.properties);
        output.push_str("}>");
    }

    if !operation.regions.is_empty() {
        output.push_str(" (");
        for (index, region) in operation.regions.iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            render_region(output, region, indent);
        }
        output.push(')');
    }

    if !operation.attributes.is_empty() {
        output.push_str(" {");
        render_named_attributes(output, &operation.attributes);
        output.push('}');
    }

    output.push_str(" : (");
    for (index, operand) in operation.operands.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        render_type(output, &operand.ty);
    }
    output.push_str(") -> ");
    render_result_types(output, operation);
    if operation.location != MlirLocation::Unknown {
        output.push(' ');
        render_location(output, &operation.location);
    }
    output.push('\n');
}

fn render_named_attributes(
    output: &mut String,
    attributes: &std::collections::BTreeMap<String, MlirAttribute>,
) {
    for (index, (name, attribute)) in attributes.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        let _ = write!(output, "{name} = ");
        render_attribute(output, attribute);
    }
}

fn render_result_types(output: &mut String, operation: &MlirOperation) {
    match operation.results.as_slice() {
        [] => output.push_str("()"),
        [result] => render_type(output, &result.ty),
        results => {
            output.push('(');
            for (index, result) in results.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                render_type(output, &result.ty);
            }
            output.push(')');
        }
    }
}

fn render_region(output: &mut String, region: &MlirRegion, indent: usize) {
    output.push_str("{\n");
    for block in &region.blocks {
        render_block(output, block, indent + 1);
    }
    write_indent(output, indent);
    output.push('}');
}

fn render_block(output: &mut String, block: &MlirBlock, indent: usize) {
    write_indent(output, indent);
    let _ = write!(output, "^bb{}", block.id.0);
    if !block.arguments.is_empty() {
        output.push('(');
        for (index, argument) in block.arguments.iter().enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            let _ = write!(output, "%v{}: ", argument.id.0);
            render_type(output, &argument.ty);
            if argument.location != MlirLocation::Unknown {
                output.push(' ');
                render_location(output, &argument.location);
            }
        }
        output.push(')');
    }
    output.push_str(":\n");
    for operation in &block.operations {
        render_operation(output, operation, indent + 1);
    }
}

fn render_type(output: &mut String, ty: &MlirType) {
    match ty {
        MlirType::Index => output.push_str("index"),
        MlirType::Integer(width) => {
            let _ = write!(output, "i{width}");
        }
        MlirType::Float(kind) => output.push_str(match kind {
            MlirFloatType::F16 => "f16",
            MlirFloatType::BF16 => "bf16",
            MlirFloatType::F32 => "f32",
            MlirFloatType::F64 => "f64",
            MlirFloatType::Other(spelling) => spelling,
        }),
        MlirType::Vector { shape, element } => {
            output.push_str("vector<");
            for dimension in shape {
                let _ = write!(output, "{dimension}x");
            }
            render_type(output, element);
            output.push('>');
        }
        MlirType::Function { inputs, results } => {
            render_type_list(output, inputs);
            output.push_str(" -> ");
            if results.len() == 1 {
                render_type(output, &results[0]);
            } else {
                render_type_list(output, results);
            }
        }
        MlirType::Tuple(elements) => {
            output.push_str("tuple<");
            for (index, element) in elements.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                render_type(output, element);
            }
            output.push('>');
        }
        MlirType::Dialect(spelling) => output.push_str(spelling),
    }
}

fn render_type_list(output: &mut String, types: &[MlirType]) {
    output.push('(');
    for (index, ty) in types.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        render_type(output, ty);
    }
    output.push(')');
}

fn render_attribute(output: &mut String, attribute: &MlirAttribute) {
    match attribute {
        MlirAttribute::Unit => output.push_str("unit"),
        MlirAttribute::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        MlirAttribute::Integer { value, ty } => {
            let _ = write!(output, "{value} : ");
            render_type(output, ty);
        }
        MlirAttribute::Float { value, ty } => {
            output.push_str(value);
            output.push_str(" : ");
            render_type(output, ty);
        }
        MlirAttribute::String(value) => render_string(output, value),
        MlirAttribute::Type(ty) => render_type(output, ty),
        MlirAttribute::Array(elements) => {
            output.push('[');
            for (index, element) in elements.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                render_attribute(output, element);
            }
            output.push(']');
        }
        MlirAttribute::DenseI32Array(elements) => {
            output.push_str("array<i32");
            if !elements.is_empty() {
                output.push_str(": ");
                for (index, value) in elements.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    let _ = write!(output, "{value}");
                }
            }
            output.push('>');
        }
        MlirAttribute::DenseI64Array(elements) => {
            output.push_str("array<i64");
            if !elements.is_empty() {
                output.push_str(": ");
                for (index, value) in elements.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    let _ = write!(output, "{value}");
                }
            }
            output.push('>');
        }
        MlirAttribute::Dictionary(entries) => {
            output.push('{');
            for (index, (name, value)) in entries.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                let _ = write!(output, "{name} = ");
                render_attribute(output, value);
            }
            output.push('}');
        }
        MlirAttribute::SymbolRef(symbol) => {
            output.push('@');
            output.push_str(symbol);
        }
        MlirAttribute::Dialect(spelling) => output.push_str(spelling),
    }
}

fn render_location(output: &mut String, location: &MlirLocation) {
    output.push_str("loc(");
    render_location_instance(output, location);
    output.push(')');
}

/// A bare location instance, as MLIR's grammar nests them: only the
/// outermost location carries the `loc(...)` wrapper, so a named or fused
/// child is spelled `"name"(unknown)`, never `"name"(loc(unknown))`.
fn render_location_instance(output: &mut String, location: &MlirLocation) {
    match location {
        MlirLocation::Unknown => output.push_str("unknown"),
        MlirLocation::FileLineCol { file, line, column } => {
            render_string(output, file);
            let _ = write!(output, ":{line}:{column}");
        }
        MlirLocation::Fused {
            metadata,
            locations,
        } => {
            output.push_str("fused");
            if let Some(metadata) = metadata {
                output.push('<');
                render_attribute(output, metadata);
                output.push('>');
            }
            output.push('[');
            for (index, child) in locations.iter().enumerate() {
                if index != 0 {
                    output.push_str(", ");
                }
                render_location_instance(output, child);
            }
            output.push(']');
        }
        MlirLocation::Named { name, child } => {
            render_string(output, name);
            // MLIR elides an unknown child: `loc("name")`.
            if !matches!(**child, MlirLocation::Unknown) {
                output.push('(');
                render_location_instance(output, child);
                output.push(')');
            }
        }
        MlirLocation::CallSite { callee, caller } => {
            output.push_str("callsite(");
            render_location_instance(output, callee);
            output.push_str(" at ");
            render_location_instance(output, caller);
            output.push(')');
        }
    }
}

fn render_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character => output.push(character),
        }
    }
    output.push('"');
}

fn write_indent(output: &mut String, indent: usize) {
    for _ in 0..indent {
        output.push_str("  ");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MlirBlockArgument, MlirBlockId, MlirResult, MlirValueId, MlirValueUse};

    #[test]
    fn generic_syntax_is_stable() {
        let i32_ty = MlirType::Integer(32);
        let add = MlirOperation {
            name: "arith.addi".into(),
            results: vec![MlirResult {
                id: MlirValueId(2),
                ty: i32_ty.clone(),
            }],
            operands: vec![
                MlirValueUse {
                    id: MlirValueId(0),
                    ty: i32_ty.clone(),
                },
                MlirValueUse {
                    id: MlirValueId(1),
                    ty: i32_ty.clone(),
                },
            ],
            successors: vec![],
            properties: Default::default(),
            regions: vec![],
            attributes: Default::default(),
            location: MlirLocation::Unknown,
        };
        let mut root = MlirOperation::new("builtin.module").unwrap();
        root.regions.push(MlirRegion {
            blocks: vec![MlirBlock {
                id: MlirBlockId(0),
                arguments: vec![
                    MlirBlockArgument {
                        id: MlirValueId(0),
                        ty: i32_ty.clone(),
                        location: MlirLocation::Unknown,
                    },
                    MlirBlockArgument {
                        id: MlirValueId(1),
                        ty: i32_ty,
                        location: MlirLocation::Unknown,
                    },
                ],
                operations: vec![add],
            }],
        });
        let text = render_module(&MlirModule {
            root,
            profile: "test".into(),
        });
        assert_eq!(
            text,
            "\"builtin.module\"() ({\n  ^bb0(%v0: i32, %v1: i32):\n    %v2 = \"arith.addi\"(%v0, %v1) : (i32, i32) -> i32\n}) : () -> ()\n"
        );
    }

    #[test]
    fn operation_properties_use_mlir_generic_syntax() {
        let i1_ty = MlirType::Integer(1);
        let i32_ty = MlirType::Integer(32);
        let mut branch = MlirOperation::new("cf.cond_br").unwrap();
        branch.operands = vec![
            MlirValueUse {
                id: MlirValueId(0),
                ty: i1_ty.clone(),
            },
            MlirValueUse {
                id: MlirValueId(1),
                ty: i32_ty.clone(),
            },
            MlirValueUse {
                id: MlirValueId(1),
                ty: i32_ty,
            },
        ];
        branch.successors = vec![MlirBlockId(1), MlirBlockId(2)];
        branch.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 1]),
        );

        let mut root = MlirOperation::new("builtin.module").unwrap();
        root.regions.push(MlirRegion {
            blocks: vec![MlirBlock {
                id: MlirBlockId(0),
                arguments: vec![
                    MlirBlockArgument {
                        id: MlirValueId(0),
                        ty: i1_ty,
                        location: MlirLocation::Unknown,
                    },
                    MlirBlockArgument {
                        id: MlirValueId(1),
                        ty: MlirType::Integer(32),
                        location: MlirLocation::Unknown,
                    },
                ],
                operations: vec![branch],
            }],
        });
        let text = render_module(&MlirModule {
            root,
            profile: "test".into(),
        });

        assert!(text.contains(
            "\"cf.cond_br\"(%v0, %v1, %v1) [^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 1, 1>}> : (i1, i32, i32) -> ()"
        ));
    }

    #[test]
    fn dense_i64_properties_render_without_stringly_typed_escapes() {
        let mut text = String::new();
        render_attribute(&mut text, &MlirAttribute::DenseI64Array(vec![0, 2]));
        assert_eq!(text, "array<i64: 0, 2>");
    }
}
