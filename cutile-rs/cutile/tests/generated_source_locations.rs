/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independent follow-ups to the generated-location audit (#1475).

mod common;
#[path = "fixtures/generated_debug_helpers.rs"]
mod helpers;

use cutile::compile_api::{CompileArtifacts, KernelCompiler};
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, DebugScope, Location, Module, OpId, Operation};

const HELPERS: &str = include_str!("fixtures/generated_debug_helpers.rs");
const SOURCE: &str = include_str!("generated_source_locations.rs");

#[cutile::module]
mod generated_kernels {
    use super::helpers::generated_debug_helpers::*;
    use cutile::core::*;

    #[cutile::entry]
    fn arrays<const VALUES: [i32; 2]>(out: &mut Tensor<i32, { [32] }>, x: i32) {
        let value = nested_arrays(x) + VALUES[0] + VALUES[1];
        out.store(broadcast_scalar(value, out.shape()));
    }

    #[cutile::entry]
    fn reduction_source(out: &mut Tensor<f32, { [1] }>, x: &Tensor<f32, { [32] }>) {
        let value = reductions(x.partition(shape![32]).load([0]));
        out.store(value.reshape(shape![1]));
    }

    #[cutile::entry]
    fn loop_source(out: &mut Tensor<f32, { [32] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        for j in 0i32..n {
            let tile = checked_load(x, j);
            out.store(tile);
        }
    }

    #[cutile::entry]
    fn assume_source(out: &mut Tensor<i32, { [32] }>, n: i32) {
        for k in 0i32..n {
            // SAFETY: the loop's lower bound establishes k >= 0.
            let nonnegative = unsafe { assume_bounds_lower::<i32, 0>(k) };
            out.store(broadcast_scalar(nonnegative, out.shape()));
        }
    }

    #[cutile::entry]
    fn launch_source(out: &mut Tensor<f32, { [32] }>, x: &Tensor<f32, { [-1] }>) {
        let a = nested_first_tile(x);
        let b = nested_first_tile(x);
        out.store(a + b);
    }
}

fn line(source: &str, text: &str) -> u32 {
    (source
        .lines()
        .position(|line| line.trim() == text)
        .expect(text)
        + 1) as u32
}

fn compile(name: &str, level: DebugInfoLevel) -> CompileArtifacts {
    let mut compiler = KernelCompiler::new(
        generated_kernels::__module_ast_self,
        "generated_kernels",
        name,
    )
    .target("sm_120")
    .strides(&[("out", &[1]), ("x", &[1])])
    .options(CompileOptions::new().debug_info(level));
    if name == "arrays" {
        compiler = compiler.generics(vec!["41".into(), "43".into()]);
    }
    compiler
        .compile()
        .unwrap_or_else(|error| panic!("{name}: {error}"))
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

fn frames(location: &Location) -> Vec<(&str, u32, Option<&str>)> {
    match location {
        Location::CallSite { callee, caller } => {
            let mut chain = frames(callee);
            chain.extend(frames(caller));
            chain
        }
        Location::DebugInfo(info) => vec![(
            &info.filename,
            info.line,
            match &info.scope {
                DebugScope::Subprogram(scope) => Some(scope.name.as_str()),
                _ => None,
            },
        )],
        Location::FileLineCol { filename, line, .. } => vec![(filename, *line, None)],
        Location::Unknown => panic!("missing source origin"),
    }
}

fn assert_helper(location: &Location, function: &str, expected_line: u32, caller_line: u32) {
    let chain = frames(location);
    assert!(
        chain[0].0.ends_with("fixtures/generated_debug_helpers.rs"),
        "{chain:?}"
    );
    assert_eq!(chain[0].1, expected_line, "{chain:?}");
    assert_eq!(chain[0].2, Some(function), "{chain:?}");
    assert_eq!(chain[1].1, caller_line, "{chain:?}");
    if function == "checked_load" {
        assert_eq!(chain.len(), 3, "{chain:?}");
        assert!(chain[1].0.ends_with("generated_source_locations.rs"));
        assert_eq!(chain[1].2, Some("loop_source"));
    } else {
        assert_eq!(chain.len(), 4, "{chain:?}");
        assert!(chain[1].0.ends_with("fixtures/generated_debug_helpers.rs"));
        assert_eq!(chain[1].2, Some("nested_arrays"));
        assert!(chain[2].0.ends_with("generated_source_locations.rs"));
        assert_eq!(chain[2].2, Some("arrays"));
        assert_eq!(
            chain[2].1,
            line(
                SOURCE,
                "let value = nested_arrays(x) + VALUES[0] + VALUES[1];"
            )
        );
    }
    assert!(chain
        .last()
        .unwrap()
        .0
        .ends_with("generated_source_locations.rs"));
}

#[test]
fn array_bindings_use_entry_function_and_method_definitions() {
    common::with_test_stack(|| {
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            let artifacts = compile("arrays", level);
            let mut seen = std::collections::HashSet::new();
            visit(
                artifacts.module(),
                &artifacts.module().functions,
                &mut |op| {
                    if op.opcode != Opcode::Constant {
                        return;
                    }
                    let Some((_, Attribute::DenseElements(value))) =
                        op.attributes.iter().find(|(key, _)| key == "value")
                    else {
                        return;
                    };
                    let Ok(bytes) = <[u8; 4]>::try_from(value.data.as_slice()) else {
                        return;
                    };
                    let number = i32::from_le_bytes(bytes);
                    match number {
                        -17 | 23 => assert_helper(
                            &op.location,
                            "array_function",
                            line(
                                HELPERS,
                                "pub fn array_function<const VALUES: [i32; 2]>(x: i32) -> i32 {",
                            ),
                            line(HELPERS, "let a = array_function::<{ [-17, 23] }>(x);"),
                        ),
                        31 | 37 => assert_helper(
                            &op.location,
                            "array_method",
                            line(
                                HELPERS,
                                "fn array_method(self, _shape: Shape<VALUES>) -> i32 {",
                            ),
                            line(HELPERS, "a.array_method(shape![31, 37])"),
                        ),
                        41 | 43 => {
                            let chain = frames(&op.location);
                            assert!(chain[0].0.ends_with("generated_source_locations.rs"));
                            assert_eq!(
                                chain[0].1,
                                line(
                                    SOURCE,
                                    "fn arrays<const VALUES: [i32; 2]>(out: &mut Tensor<i32, { [32] }>, x: i32) {",
                                )
                            );
                        }
                        _ => return,
                    }
                    seen.insert(number);
                },
            );
            assert_eq!(seen, [-17, 23, 31, 37, 41, 43].into_iter().collect());
            let dump = cutile_ir::decode_bytecode(&artifacts.bytecode().unwrap()).unwrap();
            for signature in [
                "pub fn array_function<const VALUES: [i32; 2]>(x: i32) -> i32 {",
                "fn array_method(self, _shape: Shape<VALUES>) -> i32 {",
            ] {
                assert!(
                    dump.contains(&format!(
                        "generated_debug_helpers.rs\":{}:",
                        line(HELPERS, signature)
                    )),
                    "{dump}"
                );
            }
        }
    });
}

#[test]
fn generated_reduction_regions_keep_the_call_location() {
    common::with_test_stack(|| {
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            let artifacts = compile("reduction_source", level);
            let mut reductions = 0;
            visit(
                artifacts.module(),
                &artifacts.module().functions,
                &mut |op| {
                    if op.opcode != Opcode::Reduce {
                        return;
                    }
                    reductions += 1;
                    let origin = &op.location;
                    let chain = frames(origin);
                    assert_eq!(chain.len(), 3, "{chain:?}");
                    assert!(chain[0].0.ends_with("fixtures/generated_debug_helpers.rs"));
                    assert_eq!(chain[0].2, Some("reductions"));
                    assert!(chain[1].0.ends_with("generated_source_locations.rs"));
                    assert_eq!(chain[1].2, Some("reduction_source"));
                    assert_eq!(
                        chain[1].1,
                        line(
                            SOURCE,
                            "let value = reductions(x.partition(shape![32]).load([0]));"
                        )
                    );
                    let lines = [
                        "let sum: Tile<f32, { [] }> = reduce_sum(x, 0);",
                        "let product: Tile<f32, { [] }> = reduce_prod(x, 0);",
                        "let minimum: Tile<f32, { [] }> = reduce_min(x, 0);",
                        "let maximum: Tile<f32, { [] }> = reduce_max(x, 0);",
                    ];
                    assert!(lines.iter().any(|text| line(HELPERS, text) == chain[0].1));
                    for &region in &op.regions {
                        for &block in &artifacts.module().region(region).blocks {
                            visit(
                                artifacts.module(),
                                &artifacts.module().block(block).ops,
                                &mut |body_op| assert_eq!(&body_op.location, origin),
                            );
                        }
                    }
                },
            );
            assert_eq!(reductions, 4);
            artifacts.bytecode().unwrap();
        }
    });
}

#[test]
fn placed_checks_keep_the_access_origin_on_their_generated_operands() {
    common::with_test_stack(|| {
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            let artifacts = compile("loop_source", level);
            let counts = artifacts.check_counts();
            assert_eq!(
                (counts.hoisted, counts.in_place),
                if level == DebugInfoLevel::Full {
                    (0, 1)
                } else {
                    (1, 0)
                }
            );
            let module = artifacts.module();
            let mut ops = Vec::new();
            visit(module, &module.functions, &mut |op| ops.push(op.clone()));
            let check = ops
                .iter()
                .find(|op| op.opcode == Opcode::Assert)
                .expect("device check");
            assert_helper(
                &check.location,
                "checked_load",
                line(HELPERS, "p.load([index])"),
                line(SOURCE, "let tile = checked_load(x, j);"),
            );
            // Walk the check's operand graph. Constants and comparisons in
            // the generated chain must retain the same origin after folding.
            let mut pending = check.operands.clone();
            let mut seen = std::collections::HashSet::new();
            let mut constants = 0;
            let mut comparisons = 0;
            while let Some(value) = pending.pop() {
                if !seen.insert(value) {
                    continue;
                }
                let cutile_ir::ir::ValueProducer::OpResult { op, .. } =
                    module.value_data(value).producer
                else {
                    continue;
                };
                let producer = module.op(op);
                if producer.opcode == Opcode::Constant {
                    if frames(&producer.location)[0]
                        .0
                        .ends_with("generated_source_locations.rs")
                        && frames(&producer.location)[0].1 == line(SOURCE, "for j in 0i32..n {")
                    {
                        continue;
                    }
                    constants += 1;
                    assert_eq!(producer.location, check.location);
                } else if producer.opcode == Opcode::CmpI {
                    comparisons += 1;
                    assert_eq!(producer.location, check.location);
                }
                // Pre-existing loop bounds and view/parameter computations
                // have their own legitimate source origins.
                if producer.location == check.location {
                    pending.extend_from_slice(&producer.operands);
                }
            }
            assert!(constants > 0 && comparisons > 0);
            artifacts.bytecode().unwrap();
        }
    });
}

#[test]
fn host_check_errors_keep_all_deduplicated_inline_origins() {
    common::with_test_stack(|| {
        for level in [
            DebugInfoLevel::None,
            DebugInfoLevel::Line,
            DebugInfoLevel::Full,
        ] {
            let artifacts = compile("launch_source", level);
            assert_eq!(
                artifacts.launch_checks().len(),
                1,
                "one comparison for repeated accesses"
            );
            assert_eq!(
                artifacts.launch_checks()[0]
                    .cause
                    .matches("\n  at ")
                    .count(),
                2,
                "both distinct access chains belong to the same host diagnostic"
            );
            let message = cutile::tile_kernel::validate_launch_checks(
                artifacts.launch_checks(),
                &[vec![32], vec![0]],
                &[vec![32], vec![0]],
                (1, 1, 1),
            )
            .unwrap_err()
            .to_string();
            for (file, expected) in [
                ("generated_debug_helpers.rs", line(HELPERS, "p.load([0])")),
                ("generated_debug_helpers.rs", line(HELPERS, "first_tile(x)")),
                (
                    "generated_source_locations.rs",
                    line(SOURCE, "let a = nested_first_tile(x);"),
                ),
                (
                    "generated_source_locations.rs",
                    line(SOURCE, "let b = nested_first_tile(x);"),
                ),
            ] {
                assert!(
                    message.contains(&format!("{file}:{expected}:")),
                    "{message}"
                );
            }
            assert!(
                message.contains("first_tile") && message.contains("nested_first_tile"),
                "{message}"
            );
            cutile::tile_kernel::validate_launch_checks(
                artifacts.launch_checks(),
                &[vec![32], vec![32]],
                &[vec![32], vec![32]],
                (1, 1, 1),
            )
            .unwrap();
            let unresolved = cutile::tile_kernel::validate_launch_checks(
                artifacts.launch_checks(),
                &[],
                &[],
                (1, 1, 1),
            )
            .unwrap_err()
            .to_string();
            assert!(unresolved.contains("unresolved operands"));
            assert!(unresolved.contains(&format!(
                "generated_debug_helpers.rs:{}:",
                line(HELPERS, "p.load([0])")
            )));
        }
    });
}

#[test]
fn loop_and_explicit_assumptions_keep_their_own_origins() {
    common::with_test_stack(|| {
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            let artifacts = compile("assume_source", level);
            let loop_line = line(SOURCE, "for k in 0i32..n {");
            let explicit_line = line(
                SOURCE,
                "let nonnegative = unsafe { assume_bounds_lower::<i32, 0>(k) };",
            );
            let mut seen = std::collections::HashSet::new();
            visit(
                artifacts.module(),
                &artifacts.module().functions,
                &mut |op| {
                    if op.opcode != Opcode::Assume {
                        return;
                    }
                    let chain = frames(&op.location);
                    if chain[0].2 == Some("assume_source") {
                        assert!(chain[0].0.ends_with("generated_source_locations.rs"));
                        assert!(
                            [
                                loop_line,
                                explicit_line,
                                line(
                                    SOURCE,
                                    "out.store(broadcast_scalar(nonnegative, out.shape()));"
                                )
                            ]
                            .contains(&chain[0].1),
                            "{chain:?}"
                        );
                        seen.insert(chain[0].1);
                    }
                },
            );
            assert!(seen.contains(&loop_line) && seen.contains(&explicit_line));
            let dump = cutile_ir::decode_bytecode(&artifacts.bytecode().unwrap()).unwrap();
            for source_line in seen {
                assert!(dump.contains(&format!("generated_source_locations.rs\":{source_line}:")));
            }
        }
    });
}

#[test]
#[ignore = "requires tileiras and nvdisasm"]
fn generated_locations_survive_assembly() {
    common::with_test_stack(|| {
        use cutile_compiler::cuda_tile_runtime_utils::{run_tileiras, TileirasOptions};
        use std::process::Command;
        struct Directory(std::path::PathBuf);
        impl Drop for Directory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let directory = Directory(
            std::env::temp_dir().join(format!("cutile-generated-locations-{}", std::process::id())),
        );
        std::fs::create_dir(&directory.0).unwrap();
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            for name in ["arrays", "reduction_source", "loop_source"] {
                let artifacts = compile(name, level);
                let flags =
                    TileirasOptions::from_compile_options(&CompileOptions::new().debug_info(level));
                let cubin = run_tileiras(&artifacts.bytecode().unwrap(), "sm_120", &flags).unwrap();
                let path = directory.0.join(format!("{name}-{level:?}.cubin"));
                std::fs::write(&path, cubin).unwrap();
                let output = Command::new("nvdisasm")
                    .arg("--print-line-info-inline")
                    .arg(&path)
                    .output()
                    .unwrap();
                assert!(output.status.success());
                let lines = String::from_utf8(output.stdout).unwrap();
                assert!(
                    lines.contains(if name == "arrays" {
                        "generated_source_locations.rs"
                    } else {
                        "generated_debug_helpers.rs"
                    }),
                    "{name} {level:?}: {lines}"
                );
                // Full-debug tileiras may coalesce adjacent reduction locations;
                // exact operation origins are checked before assembly above.
                if name == "reduction_source" && level == DebugInfoLevel::Line {
                    for text in [
                        "let sum: Tile<f32, { [] }> = reduce_sum(x, 0);",
                        "let product: Tile<f32, { [] }> = reduce_prod(x, 0);",
                        "let minimum: Tile<f32, { [] }> = reduce_min(x, 0);",
                        "let maximum: Tile<f32, { [] }> = reduce_max(x, 0);",
                    ] {
                        let expected = line(HELPERS, text).to_string();
                        assert!(
                            lines.lines().any(|row| {
                                row.contains("generated_debug_helpers.rs")
                                    && row
                                        .split(" inlined at")
                                        .next()
                                        .unwrap()
                                        .trim_end()
                                        .ends_with(&format!("line {expected}"))
                            }),
                            "{name} {level:?}: missing line {expected}\n{lines}"
                        );
                    }
                }
            }
        }
    });
}
