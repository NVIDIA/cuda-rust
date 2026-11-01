/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact origins for short-circuit, folded, and const-bool binding constants.
mod common;
#[path = "fixtures/constant_debug_helpers.rs"]
mod helpers;

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, DebugScope, Location, Module, OpId, Operation};
use std::collections::{HashMap, HashSet};

const SOURCE: &str = include_str!("constant_source_locations.rs");
const HELPERS: &str = include_str!("fixtures/constant_debug_helpers.rs");

#[cutile::module]
mod constants_kernel {
    use super::helpers::constant_helpers::{nested, nested_bindings};
    use cutile::core::*;

    #[cutile::entry]
    fn expressions(out: &mut Tensor<i32, { [32] }>, a: bool, b: bool) {
        let value = nested(a, b);
        out.store(value.broadcast(out.shape()));
    }

    #[cutile::entry]
    fn bindings<const ENTRY: bool>(out: &mut Tensor<i32, { [32] }>, x: i32) {
        let result = nested_bindings(x);
        let value = if ENTRY { result } else { result + 3 };
        out.store(value.broadcast(out.shape()));
    }
}

fn line(source: &str, marker: &str) -> u32 {
    (source
        .lines()
        .position(|s| s.trim() == marker)
        .expect(marker)
        + 1) as u32
}

fn visit(module: &Module, ids: &[OpId], f: &mut impl FnMut(&Operation)) {
    for &id in ids {
        let op = module.op(id);
        f(op);
        for &region in &op.regions {
            for &block in &module.region(region).blocks {
                visit(module, &module.block(block).ops, f);
            }
        }
    }
}

fn describe(location: &Location) -> String {
    match location {
        Location::CallSite { callee, caller } => {
            format!("{} <- {}", describe(callee), describe(caller))
        }
        Location::DebugInfo(info) => {
            let DebugScope::Subprogram(scope) = &info.scope else {
                panic!("{info:?}");
            };
            format!(
                "{}@{}:{}",
                scope.name,
                info.filename.rsplit('/').next().unwrap(),
                info.line
            )
        }
        Location::FileLineCol { filename, line, .. } => {
            format!("{}:{line}", filename.rsplit('/').next().unwrap())
        }
        other => panic!("missing location: {other:?}"),
    }
}

// Independently follow serialized references, so losing or shortening an inline
// chain in the encoder fails even if all of its individual lines are present.
fn serialized_locations(bytecode: &[u8]) -> HashSet<String> {
    fn field<'a>(entry: &'a str, key: &str) -> &'a str {
        let rest = &entry[entry.find(&format!("{key}=")).unwrap() + key.len() + 1..];
        if let Some(rest) = rest.strip_prefix('"') {
            &rest[..rest.find('"').unwrap()]
        } else {
            &rest[..rest.find([',', ')']).unwrap()]
        }
    }
    fn render(id: &str, entries: &HashMap<&str, &str>) -> String {
        let body = entries[id];
        if body.starts_with("CallSite(") {
            return format!(
                "{} <- {}",
                render(field(body, "callee"), entries),
                render(field(body, "caller"), entries)
            );
        }
        let scope = entries[field(body, "scope")];
        let quoted = body.split('"').nth(1).unwrap();
        let file = quoted.rsplit('/').next().unwrap();
        let line = body.split('"').nth(2).unwrap()[1..]
            .split(':')
            .next()
            .unwrap();
        if field(scope, "linkage").ends_with("_entry") {
            format!("{file}:{line}")
        } else {
            format!("{}@{file}:{line}", field(scope, "name"))
        }
    }
    let dump = cutile_ir::decode_bytecode(bytecode).unwrap();
    let debug = dump.split("=== Debug").nth(1).unwrap();
    let entries: HashMap<_, _> = debug
        .lines()
        .map(str::trim_start)
        .filter(|s| s.starts_with("di["))
        .map(|s| s.split_once(" = ").unwrap())
        .collect();
    entries
        .iter()
        .filter(|(_, body)| body.starts_with("DILoc(") || body.starts_with("CallSite("))
        .map(|(id, _)| render(id, &entries))
        .collect()
}

fn check(name: &str, level: DebugInfoLevel, expected: &[(Vec<u8>, String)]) {
    let mut compiler = KernelCompiler::new(
        constants_kernel::__module_ast_self,
        "constants_kernel",
        name,
    )
    .target("sm_120")
    .strides(&[("out", &[1])])
    .options(CompileOptions::new().debug_info(level));
    if name == "bindings" {
        compiler = compiler.generics(vec!["false".into()]);
    }
    let artifacts = compiler.compile().unwrap();
    let mut constants = Vec::new();
    visit(
        artifacts.module(),
        &artifacts.module().functions,
        &mut |op| {
            if op.opcode == Opcode::Constant {
                let Some((_, Attribute::DenseElements(value))) =
                    op.attributes.iter().find(|(key, _)| key == "value")
                else {
                    panic!("{op:?}");
                };
                constants.push((value.data.clone(), describe(&op.location)));
            }
        },
    );
    let serialized = serialized_locations(&artifacts.bytecode().unwrap());
    // Entry-wrapper policy is owned by #1385/#1480. Accept either its old
    // module origin or its new signature origin; every helper and call-site
    // frame above that root is checked exactly, before and after serialization.
    let module = line(SOURCE, "mod constants_kernel {");
    let signature = SOURCE
        .lines()
        .position(|s| s.trim().starts_with(&format!("fn {name}")))
        .unwrap()
        + 1;
    let old_root = format!(" <- constant_source_locations.rs:{module}");
    let new_root = format!(" <- constant_source_locations.rs:{signature}");
    for (value, location) in expected {
        let updated_root = location
            .strip_suffix(&old_root)
            .map(|s| format!("{s}{new_root}"))
            .unwrap();
        let found = constants.iter().find(|(bytes, actual)| {
            bytes == value && (actual == location || actual == &updated_root)
        });
        let (_, actual) = found.unwrap_or_else(|| {
            panic!("{level:?}: missing {value:?} at {location}\n{constants:#?}")
        });
        assert!(
            serialized.contains(actual),
            "{level:?}: lost serialized inline chain {actual}\n{serialized:#?}"
        );
    }
}

#[test]
fn short_circuit_constants_keep_nested_cross_file_origins() {
    common::with_test_stack(|| {
        let definition = line(
            SOURCE,
            "fn expressions(out: &mut Tensor<i32, { [32] }>, a: bool, b: bool) {",
        );
        let call = line(SOURCE, "let value = nested(a, b);");
        let module = line(SOURCE, "mod constants_kernel {");
        let caller = format!("expressions@constant_source_locations.rs:{call} <- constant_source_locations.rs:{module}");
        assert!(definition > module);
        let expected: Vec<_> = [
            ("logical_and", "a && b", "let both = logical_and(a, b);", vec![0]),
            ("logical_or", "a || b", "let either = logical_or(a, b);", vec![255]),
            ("folded", "left + right", "let sum = folded();", 46i32.to_le_bytes().to_vec()),
        ].into_iter().map(|(name, expr, call, bytes)| {
            (bytes, format!("{name}@constant_debug_helpers.rs:{} <- nested@constant_debug_helpers.rs:{} <- {caller}", line(HELPERS, expr), line(HELPERS, call)))
        }).collect();
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            check("expressions", level, &expected[..2]);
        }
    });
}

#[test]
fn folded_expression_retains_its_line_instead_of_an_operand_definition() {
    common::with_test_stack(|| {
        let module = line(SOURCE, "mod constants_kernel {");
        let caller = line(SOURCE, "let value = nested(a, b);");
        let call = line(HELPERS, "let sum = folded();");
        let folded = line(HELPERS, "left + right");
        let expected = [(46i32.to_le_bytes().to_vec(), format!(
            "folded@constant_debug_helpers.rs:{folded} <- nested@constant_debug_helpers.rs:{call} <- expressions@constant_source_locations.rs:{caller} <- constant_source_locations.rs:{module}"
        ))];
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            check("expressions", level, &expected);
        }
    });
}

#[test]
fn const_bool_bindings_use_entry_function_and_method_definitions() {
    common::with_test_stack(|| {
        let definition = line(
            SOURCE,
            "fn bindings<const ENTRY: bool>(out: &mut Tensor<i32, { [32] }>, x: i32) {",
        );
        let call = line(SOURCE, "let result = nested_bindings(x);");
        let module = line(SOURCE, "mod constants_kernel {");
        let root = format!("constant_source_locations.rs:{module}");
        let caller = format!("bindings@constant_source_locations.rs:{call} <- {root}");
        let expected = [
            (vec![0], format!("bindings@constant_source_locations.rs:{definition} <- {root}")),
            (vec![255], format!("bool_function@constant_debug_helpers.rs:{} <- nested_bindings@constant_debug_helpers.rs:{} <- {caller}", line(HELPERS, "pub fn bool_function<const FLAG: bool>(x: i32) -> i32 {"), line(HELPERS, "let first = bool_function::<true>(x);"))),
            (vec![0], format!("bool_method@constant_debug_helpers.rs:{} <- nested_bindings@constant_debug_helpers.rs:{} <- {caller}", line(HELPERS, "fn bool_method<const FLAG: bool>(self) -> i32 {"), line(HELPERS, "first.bool_method::<false>()"))),
        ];
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            check("bindings", level, &expected);
        }
    });
}
