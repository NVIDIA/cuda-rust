/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Source provenance through user helpers and core operations, including
//! the real device compiler's line tables and inline scopes.

mod common;
#[path = "fixtures/debug_helpers.rs"]
mod debug_helpers_file;

use cutile::compile_api::{CompileArtifacts, KernelCompiler};
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_ir::ir::{DebugScope, Location, Module, OpId, Operation};
use std::path::PathBuf;
use std::process::Command;

#[cutile::module]
mod debug_kernel {
    use super::debug_helpers_file::debug_helpers::{nested, Bump};
    use cutile::core::*;

    #[cutile::entry()]
    fn source_kernel<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {
        let value = nested(x);
        let adjusted = value.bump();
        let tile = broadcast_scalar(adjusted, out.shape());
        out.store(tile);
    }
}

#[cutile::module]
mod dispatch_kernel {
    use cutile::core::*;

    // `load_tile_like(x, out)` is a trait-dispatch wrapper: type inference
    // lowers the free-function call to `x.load_tile_like(out)` before the
    // inline compiler runs. The lowered call must keep this call site.
    #[cutile::entry()]
    fn dispatch_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: &Tensor<i32, { [-1] }>) {
        let tile = load_tile_like(x, out);
        out.store(tile);
    }
}

fn compile(level: DebugInfoLevel) -> CompileArtifacts {
    KernelCompiler::new(
        debug_kernel::__module_ast_self,
        "debug_kernel",
        "source_kernel",
    )
    .target("sm_120")
    .generics(vec!["32".into()])
    .strides(&[("out", &[1])])
    .options(CompileOptions::new().debug_info(level))
    .compile()
    .expect("compile source kernel")
}

fn source_line(source: &str, text: &str) -> u32 {
    (source
        .lines()
        .position(|line| line.trim() == text)
        .expect("source marker")
        + 1) as u32
}

fn visit_ops(module: &Module, ops: &[OpId], visitor: &mut impl FnMut(&Operation)) {
    for &id in ops {
        let op = module.op(id);
        visitor(op);
        for &region in &op.regions {
            for &block in &module.region(region).blocks {
                visit_ops(module, &module.block(block).ops, visitor);
            }
        }
    }
}

#[test]
fn helper_instructions_keep_exact_lines_and_call_sites() {
    common::with_test_stack(|| {
        let artifacts = compile(DebugInfoLevel::Line);
        let helpers = include_str!("fixtures/debug_helpers.rs");
        for (name, expression) in [
            ("offset", "x + 7"),
            ("nested", "offset(x) * 2"),
            ("bump", "self + 3"),
        ] {
            let expected_line = source_line(helpers, expression);
            let mut found = false;
            visit_ops(
                artifacts.module(),
                &artifacts.module().functions,
                &mut |op| {
                    if let Location::CallSite { callee, caller } = &op.location {
                        if let Location::DebugInfo(info) = &**callee {
                            if let DebugScope::Subprogram(scope) = &info.scope {
                                if scope.name == name && info.line == expected_line {
                                    assert!(
                                        info.filename.ends_with("fixtures/debug_helpers.rs"),
                                        "{info:?}"
                                    );
                                    assert!(!matches!(&**caller, Location::Unknown));
                                    found = true;
                                }
                            }
                        }
                    }
                },
            );
            assert!(found, "missing instruction at {name}:{expected_line}");
        }
        let dump = cutile_ir::decode_bytecode(&artifacts.bytecode().unwrap()).unwrap();
        for expression in ["x + 7", "offset(x) * 2", "self + 3"] {
            let line = source_line(helpers, expression);
            assert!(
                dump.contains(&format!("debug_helpers.rs\":{line}:")),
                "{dump}"
            );
        }
    });
}

#[test]
fn explicit_debug_modes_override_build_default_and_change_cache_keys() {
    use cutile_compiler::cuda_tile_runtime_utils::TileirasOptions;
    use std::collections::HashSet;
    let mut l1 = HashSet::new();
    let mut l2 = HashSet::new();
    for (level, opt, debug, line) in [
        (DebugInfoLevel::None, 3, false, false),
        (DebugInfoLevel::Line, 3, false, true),
        (DebugInfoLevel::Full, 0, true, false),
    ] {
        let options = CompileOptions::new()
            .debug_info(DebugInfoLevel::Full)
            .debug_info(level)
            .occupancy(2)
            .sanitize_memcheck(true);
        assert_eq!(options.occupancy, Some(2));
        let flags = TileirasOptions::from_compile_options(&options);
        assert_eq!(
            (flags.opt_level, flags.device_debug, flags.lineinfo),
            (opt, debug, line)
        );
        assert!(flags.sanitize_memcheck);
        assert!(l1.insert(
            cutile::tile_kernel::TileFunctionKey::builder("m", "k")
                .compile_options(options)
                .build()
        ));
        assert!(l2.insert(cutile_compiler::jit_cache::l2_key(
            b"same bytecode",
            cutile_ir::bytecode::BytecodeVersion::CURRENT,
            "sm_120",
            &flags,
            "same tileiras"
        )));
    }
}

#[test]
fn build_time_default() {
    // The script supplies an independent oracle for each Cargo profile and
    // build override, then runs the same executable under different runtime
    // environments. It must not derive the expected value from our default.
    // For ordinary test runs, independently capture any explicit override.
    let expected = std::env::var("CUTILE_TEST_EXPECT_DEBUG_INFO")
        .ok()
        .or_else(|| option_env!("CUDA_RUST_DEBUG").map(str::to_owned));
    let options = CompileOptions::new();
    assert!(!(options.device_debug && options.lineinfo));
    if let Some(expected) = expected {
        let level = match expected.as_str() {
            "none" => DebugInfoLevel::None,
            "line" => DebugInfoLevel::Line,
            "full" => DebugInfoLevel::Full,
            other => panic!("invalid expected debug mode {other:?}"),
        };
        assert_eq!(options, CompileOptions::new().debug_info(level));
    }
    let explicit = options;
    assert_eq!(CompileOptions::new(), explicit);
    assert_eq!(CompileOptions::default(), explicit);
    assert_eq!(
        cutile::tile_kernel::TileFunctionKey::builder("m", "k").build(),
        cutile::tile_kernel::TileFunctionKey::builder("m", "k")
            .compile_options(explicit)
            .build()
    );
}

// Separate opt-in test: requires the actual NVIDIA assembler and disassembler.
// No GPU or profiler counter permissions are needed.
#[test]
#[ignore = "requires tileiras, nvdisasm, and readelf"]
fn cubin_has_rust_lines_and_inline_frames() {
    common::with_test_stack(|| {
        use cutile_compiler::cuda_tile_runtime_utils::{run_tileiras, TileirasOptions};
        let directory = CubinDirectory::new();
        for level in [
            DebugInfoLevel::None,
            DebugInfoLevel::Line,
            DebugInfoLevel::Full,
        ] {
            let artifacts = compile(level);
            let bytecode = artifacts.bytecode().unwrap();
            let flags =
                TileirasOptions::from_compile_options(&CompileOptions::new().debug_info(level));
            let cubin = run_tileiras(&bytecode, "sm_120", &flags)
                .expect("tileiras must accept inline scopes");
            let path = directory.0.join(format!("{level:?}.cubin"));
            std::fs::write(&path, cubin).unwrap();
            let sections = output(Command::new("readelf").arg("-SW").arg(&path));
            if level == DebugInfoLevel::None {
                assert!(!sections.contains(".debug_line"), "{sections}");
                continue;
            }
            assert!(sections.contains(".debug_line"), "{sections}");
            let sass = output(Command::new("nvdisasm").arg("--print-line-info").arg(&path));
            assert!(sass.contains("debug_helpers.rs"), "{sass}");
            assert!(sass.contains("debug_info.rs"), "{sass}");
            assert!(!sass.contains("_core.rs"), "{sass}");
            if level == DebugInfoLevel::Full {
                let dwarf = output(Command::new("readelf").arg("--debug-dump=info").arg(&path));
                assert!(dwarf.contains("DW_TAG_inlined_subroutine"), "{dwarf}");
                for name in ["source_kernel", "nested", "offset", "bump"] {
                    assert!(dwarf.contains(name), "missing {name} in {dwarf}");
                }
            }
        }
    });
}

fn output(command: &mut Command) -> String {
    let result = command
        .output()
        .unwrap_or_else(|error| panic!("{command:?}: {error}"));
    assert!(
        result.status.success(),
        "{command:?}:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

#[test]
#[ignore = "requires a CUDA Tile-capable GPU and tileiras"]
fn kernel_executes_in_all_debug_modes() {
    common::with_test_stack(|| {
        for level in [
            DebugInfoLevel::None,
            DebugInfoLevel::Line,
            DebugInfoLevel::Full,
        ] {
            run_kernel(level);
        }
    });
}

fn run_kernel(level: DebugInfoLevel) {
    use cutile::prelude::*;
    let values = debug_kernel::source_kernel(api::zeros::<i32>(&[32]).partition([32]), 1)
        .compile_options(CompileOptions::new().debug_info(level))
        .first()
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("source kernel must run");
    assert_eq!(values, vec![19; 32]);
}

#[test]
#[ignore = "GPU subprocess target for cuda_gdb_stops_in_cross_file_helper"]
fn debugger_target() {
    common::with_test_stack(|| run_kernel(DebugInfoLevel::Full));
}

#[test]
#[ignore = "requires a CUDA Tile-capable GPU, tileiras, cuda-gdb, and timeout"]
fn cuda_gdb_stops_in_cross_file_helper() {
    let line = source_line(include_str!("fixtures/debug_helpers.rs"), "x + 7");
    let transcript = output(
        Command::new("timeout")
            .args(["120", "cuda-gdb", "--batch", "-nx"])
            .args(["-ex", "set pagination off"])
            .args(["-ex", "set breakpoint pending on"])
            .args(["-ex", &format!("break debug_helpers.rs:{line}")])
            .args(["-ex", "run", "-ex", "bt", "-ex", "continue"])
            .arg("--args")
            .arg(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "debugger_target", "--nocapture"]),
    );
    for (frame, name) in [("#0", "offset"), ("#1", "nested"), ("#2", "source_kernel")] {
        assert!(
            transcript
                .lines()
                .any(|line| line.starts_with(frame) && line.contains(name)),
            "missing {frame} {name}:\n{transcript}"
        );
    }
    assert!(
        transcript.contains(&format!("debug_helpers.rs:{line}")),
        "{transcript}"
    );
    assert!(transcript.contains("1 passed; 0 failed"), "{transcript}");
}

struct CubinDirectory(PathBuf);

impl CubinDirectory {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("cutile-debug-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for CubinDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn cross_file_functions_and_methods_keep_inline_scopes() {
    common::with_test_stack(|| {
        let artifacts = compile(DebugInfoLevel::Line);
        let bytecode = artifacts.bytecode().expect("serialize");
        let dump = cutile_ir::decode_bytecode(&bytecode).expect("decode");
        for name in ["nested", "offset", "bump"] {
            assert!(
                dump.contains(&format!("name=\"{name}\", linkage=\"{name}@")),
                "missing inline scope for {name}:\n{dump}"
            );
        }
        assert!(dump.contains("debug_helpers.rs"), "{dump}");
        assert_eq!(dump.matches("DICompileUnit(").count(), 1, "{dump}");
        assert!(dump.contains("CallSite(callee=di["), "{dump}");
        // Builtin expansions should not displace the user's source with _core.rs.
        assert!(!dump.contains("_core.rs"), "{dump}");
    });
}

#[test]
fn lowered_dispatch_call_keeps_its_call_site() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            dispatch_kernel::__module_ast_self,
            "dispatch_kernel",
            "dispatch_source",
        )
        .target("sm_120")
        .generics(vec!["32".into()])
        .strides(&[("out", &[1]), ("x", &[1])])
        .options(CompileOptions::new().debug_info(DebugInfoLevel::Line))
        .compile()
        .expect("compile dispatch kernel");
        let source = include_str!("debug_info.rs");
        let call_line = source_line(source, "let tile = load_tile_like(x, out);");
        let module_line = source_line(source, "mod dispatch_kernel {");
        let mut load_lines = Vec::new();
        visit_ops(
            artifacts.module(),
            &artifacts.module().functions,
            &mut |op| {
                if op.opcode != cutile_ir::bytecode::Opcode::LoadViewTko {
                    return;
                }
                let mut location = &op.location;
                while let Location::CallSite { callee, .. } = location {
                    location = callee;
                }
                match location {
                    Location::DebugInfo(info) => load_lines.push(info.line),
                    Location::FileLineCol { line, .. } => load_lines.push(*line),
                    other => panic!("load without a source location: {other:?}"),
                }
            },
        );
        assert!(!load_lines.is_empty(), "no load_view_tko op");
        assert!(
            load_lines.iter().all(|&line| line == call_line),
            "inlined dispatch body must sit on the call line {call_line}, not the module line {module_line}: {load_lines:?}"
        );
    });
}

#[cutile::module]
mod assertion_kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn checked_scale<const S: [i32; 1]>(
        out: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        factor: f32,
    ) {
        validate_factor(factor);
        let tile_x = x.load_like(out);
        out.store(tile_x * factor.broadcast(out.shape()));
    }

    fn validate_factor(factor: f32) {
        assert_in_range(factor, 0.0f32, 1000.0f32);
    }

    #[allow(unreachable_code, unused_variables)]
    fn assert_in_range(value: f32, lo: f32, hi: f32) {
        cuda_tile_assert!(value >= lo, "checked_scale: factor below the allowed range");
        cuda_tile_assert!(value <= hi, "checked_scale: factor above the allowed range");
    }
}

/// Renders a location as `callee <- caller`, each frame as
/// `subprogram[definition line]@file:line`. The outermost frame of a kernel is
/// rendered as `file:line`.
fn describe(location: &Location) -> String {
    let base = |file: &str| file.rsplit('/').next().unwrap().to_string();
    match location {
        Location::FileLineCol { filename, line, .. } => format!("{}:{line}", base(filename)),
        Location::DebugInfo(info) => match &info.scope {
            DebugScope::Subprogram(scope) => format!(
                "{}[{}]@{}:{}",
                scope.name,
                scope.line,
                base(&info.filename),
                info.line
            ),
            other => panic!("unexpected scope {other:?}"),
        },
        Location::CallSite { callee, caller } => {
            format!("{} <- {}", describe(callee), describe(caller))
        }
        other => panic!("unexpected location {other:?}"),
    }
}

/// Every constant in the module as (little-endian value bytes, described location).
fn constant_locations(artifacts: &CompileArtifacts) -> Vec<(Vec<u8>, String)> {
    let mut constants = Vec::new();
    visit_ops(
        artifacts.module(),
        &artifacts.module().functions,
        &mut |op| {
            if op.opcode != cutile_ir::bytecode::Opcode::Constant {
                return;
            }
            let Some((_, cutile_ir::ir::Attribute::DenseElements(value))) =
                op.attributes.iter().find(|(name, _)| name == "value")
            else {
                panic!("constant without a dense value: {op:?}");
            };
            constants.push((value.data.clone(), describe(&op.location)));
        },
    );
    constants
}

/// The set of locations stored in the serialized debug section, rendered like `describe`.
fn decoded_locations(bytecode: &[u8]) -> std::collections::HashSet<String> {
    use std::collections::HashMap;
    fn field<'a>(entry: &'a str, key: &str) -> &'a str {
        let start = entry.find(&format!("{key}=")).unwrap() + key.len() + 1;
        let rest = &entry[start..];
        match rest.strip_prefix('"') {
            Some(quoted) => &quoted[..quoted.find('"').unwrap()],
            None => &rest[..rest.find([',', ')']).unwrap()],
        }
    }
    let dump = cutile_ir::decode_bytecode(bytecode).expect("decode");
    let debug = dump.split("=== Debug").nth(1).expect("debug section");
    let mut entries: HashMap<&str, &str> = HashMap::new();
    for line in debug
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("di["))
    {
        let (id, body) = line.split_once(" = ").unwrap();
        entries.insert(id, body);
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
        let line = body.split('"').nth(2).unwrap()[1..]
            .split(':')
            .next()
            .unwrap();
        let file = quoted.rsplit('/').next().unwrap();
        if field(scope, "linkage").ends_with("_entry") {
            format!("{file}:{line}")
        } else {
            format!(
                "{}[{}]@{file}:{line}",
                field(scope, "name"),
                field(scope, "line")
            )
        }
    }
    entries
        .iter()
        .filter(|(_, body)| body.starts_with("DILoc(") || body.starts_with("CallSite("))
        .map(|(id, _)| render(id, &entries))
        .collect()
}

/// Checks the exact location of the selected constants, rejects any location
/// outside the allowed set, and checks the same locations after serialization.
fn check_constant_locations(
    artifacts: &CompileArtifacts,
    allowed: &[String],
    required: &[(Vec<u8>, String)],
) {
    let constants = constant_locations(artifacts);
    assert!(!constants.is_empty(), "expected generated constants");
    let unexpected: Vec<_> = constants
        .iter()
        .filter(|(_, location)| !allowed.contains(location))
        .collect();
    assert!(
        unexpected.is_empty(),
        "unexpected locations: {unexpected:#?}"
    );
    for (value, location) in required {
        assert!(
            constants
                .iter()
                .any(|(bytes, found)| bytes == value && found == location),
            "no constant {value:?} at {location}, found {constants:#?}"
        );
    }
    let decoded = decoded_locations(&artifacts.bytecode().expect("serialize"));
    for (value, location) in &constants {
        assert!(
            decoded.contains(location),
            "location of constant {value:?} lost in serialization: {location}\n{decoded:#?}"
        );
    }
}

fn check_generated_constant_locations(level: DebugInfoLevel) {
    common::with_test_stack(move || {
        let artifacts = KernelCompiler::new(
            assertion_kernel::__module_ast_self,
            "assertion_kernel",
            "checked_scale",
        )
        .target("sm_120")
        .generics(vec!["128".into()])
        .strides(&[("out", &[1]), ("x", &[1])])
        .options(CompileOptions::new().debug_info(level))
        .compile()
        .expect("compile assertion kernel");
        let source = include_str!("debug_info.rs");
        let kernel = source_line(source, "fn checked_scale<const S: [i32; 1]>(");
        let validator = source_line(source, "fn validate_factor(factor: f32) {");
        let validate_call = source_line(source, "validate_factor(factor);");
        let range_call = source_line(source, "assert_in_range(factor, 0.0f32, 1000.0f32);");
        let load = source_line(source, "let tile_x = x.load_like(out);");
        let store = source_line(source, "out.store(tile_x * factor.broadcast(out.shape()));");
        let root = format!("debug_info.rs:{kernel}");
        // Generated constants without a source span fall back to the kernel definition.
        let fallback = format!("checked_scale[{kernel}]@debug_info.rs:{kernel} <- {root}");
        let statement =
            |line: u32| format!("checked_scale[{kernel}]@debug_info.rs:{line} <- {root}");
        // The literals written in the user's helper keep their own line and caller chain.
        let literal = format!(
            "validate_factor[{validator}]@debug_info.rs:{range_call} <- \
             checked_scale[{kernel}]@debug_info.rs:{validate_call} <- {root}"
        );
        let allowed = [
            root.clone(),
            fallback.clone(),
            statement(load),
            statement(store),
            literal.clone(),
        ];
        check_constant_locations(
            &artifacts,
            &allowed,
            &[
                (128i32.to_le_bytes().to_vec(), fallback),
                (0.0f32.to_le_bytes().to_vec(), literal.clone()),
                (1000.0f32.to_le_bytes().to_vec(), literal.clone()),
            ],
        );
        for (value, location) in constant_locations(&artifacts) {
            if value == 1000.0f32.to_le_bytes() {
                assert_eq!(location, literal, "user literal replaced by a fallback");
            }
        }
    });
}

#[test]
fn generated_constants_keep_function_locations_line() {
    check_generated_constant_locations(DebugInfoLevel::Line);
}

#[test]
fn generated_constants_keep_function_locations_full() {
    check_generated_constant_locations(DebugInfoLevel::Full);
}

#[cutile::module]
mod generated_helper_kernel {
    use super::debug_helpers_file::debug_helpers::generated;
    use cutile::core::*;

    #[cutile::entry()]
    fn generated_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {
        let value = generated::<{ [-17, 23] }, true>(x);
        out.store(value.broadcast(out.shape()));
    }
}

#[cutile::module]
mod nested_helper_kernel {
    use super::debug_helpers_file::debug_helpers::generated_nested;
    use cutile::core::*;

    #[cutile::entry()]
    fn nested_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {
        let total = generated_nested::<{ [-17, 23] }>(x);
        out.store(total.broadcast(out.shape()));
    }
}

fn check_cross_file_constant_locations(level: DebugInfoLevel, nested: bool) {
    common::with_test_stack(move || {
        macro_rules! compile_helper {
            ($module:ident, $entry:literal) => {
                KernelCompiler::new($module::__module_ast_self, stringify!($module), $entry)
                    .target("sm_120")
                    .generics(vec!["32".into()])
                    .strides(&[("out", &[1])])
                    .options(CompileOptions::new().debug_info(level))
                    .compile()
                    .expect("compile cross-file helper")
            };
        }
        let artifacts = if nested {
            compile_helper!(nested_helper_kernel, "nested_source")
        } else {
            compile_helper!(generated_helper_kernel, "generated_source")
        };
        let source = include_str!("debug_info.rs");
        let helpers = include_str!("fixtures/debug_helpers.rs");
        let (name, signature, call, store) = if nested {
            (
                "nested_source",
                "fn nested_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {",
                "let total = generated_nested::<{ [-17, 23] }>(x);",
                "out.store(total.broadcast(out.shape()));",
            )
        } else {
            (
                "generated_source",
                "fn generated_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {",
                "let value = generated::<{ [-17, 23] }, true>(x);",
                "out.store(value.broadcast(out.shape()));",
            )
        };
        let kernel = source_line(source, signature);
        let call_line = source_line(source, call);
        let store = source_line(source, store);
        let definition = source_line(
            helpers,
            "pub fn generated<const VALUES: [i32; 2], const FLAG: bool>(x: i32) -> i32 {",
        );
        let indexing = source_line(helpers, "x + VALUES[0] + VALUES[1]");
        let root = format!("debug_info.rs:{kernel}");
        let fallback = format!("{name}[{kernel}]@debug_info.rs:{kernel} <- {root}");
        let call_site = format!("{name}[{kernel}]@debug_info.rs:{call_line} <- {root}");
        let store_site = format!("{name}[{kernel}]@debug_info.rs:{store} <- {root}");
        let mut allowed = vec![root, fallback, store_site];
        let mut required = Vec::new();
        // Caller chain of the inner `generated` call, as seen by its constants.
        let inner_caller = if nested {
            let nested_definition = source_line(
                helpers,
                "pub fn generated_nested<const VALUES: [i32; 2]>(x: i32) -> i32 {",
            );
            let nested_call = source_line(helpers, "generated::<VALUES, true>(x) + 1");
            let frame = |line: u32| {
                format!(
                    "generated_nested[{nested_definition}]@debug_helpers.rs:{line} <- {call_site}"
                )
            };
            let outer_values = frame(nested_definition);
            let outer_call = frame(nested_call);
            for value in [-17i32, 23] {
                required.push((value.to_le_bytes().to_vec(), outer_values.clone()));
            }
            // The `+ 1` written in the outer helper keeps the outer helper's own line.
            required.push((1i32.to_le_bytes().to_vec(), outer_call.clone()));
            allowed.push(outer_values);
            allowed.push(outer_call.clone());
            outer_call
        } else {
            call_site
        };
        let values =
            format!("generated[{definition}]@debug_helpers.rs:{definition} <- {inner_caller}");
        let index =
            format!("generated[{definition}]@debug_helpers.rs:{indexing} <- {inner_caller}");
        for value in [-17i32, 23] {
            required.push((value.to_le_bytes().to_vec(), values.clone()));
        }
        for value in [0i32, 1] {
            required.push((value.to_le_bytes().to_vec(), index.clone()));
        }
        allowed.push(values);
        allowed.push(index);
        check_constant_locations(&artifacts, &allowed, &required);
    });
}

#[test]
fn generated_helper_constants_keep_definition_at_line_level() {
    check_cross_file_constant_locations(DebugInfoLevel::Line, false);
}

#[test]
fn generated_helper_constants_keep_definition_at_full_level() {
    check_cross_file_constant_locations(DebugInfoLevel::Full, false);
}

#[test]
fn nested_helper_constants_keep_caller_chain_at_line_level() {
    check_cross_file_constant_locations(DebugInfoLevel::Line, true);
}

#[test]
fn nested_helper_constants_keep_caller_chain_at_full_level() {
    check_cross_file_constant_locations(DebugInfoLevel::Full, true);
}

// Restrict the backend exception to the exact releases reproduced in #1385.
// In particular, 13.5 must establish its own profiling/debugging behavior.
fn has_known_inline_debug_loss(fingerprint: &str) -> bool {
    fingerprint.lines().any(|line| {
        matches!(
            line.trim(),
            "Cuda compilation tools, release 13.4, V13.4.59"
                | "Cuda compilation tools, release 13.4, V13.4.92"
        )
    })
}

#[test]
fn inline_debug_exception_does_not_cover_other_toolkit_versions() {
    for (version, expected) in [
        ("13.3, V13.3.36", false),
        ("13.4, V13.4.59", true),
        ("13.4, V13.4.60", false),
        ("13.4, V13.4.92", true),
        ("13.4, V13.4.93", false),
        ("13.5, V13.5.92", false),
    ] {
        let banner = format!("tileiras\nCuda compilation tools, release {version}\n");
        assert_eq!(has_known_inline_debug_loss(&banner), expected);
    }
    assert!(!has_known_inline_debug_loss("unknown assembler"));
}

// Opt-in, like `cubin_has_rust_lines_and_inline_frames`: the line table the
// device compiler emits for generated constants must stay inside the kernel
// and the helpers it calls. The exact IR/bytecode origins are checked above
// on every toolkit; only the known 13.4.59/13.4.92 backend loss is accounted for here.
#[test]
#[ignore = "requires tileiras, nvdisasm, and readelf"]
fn nested_helper_cubin_lines_stay_inside_their_functions() {
    common::with_test_stack(|| {
        use cutile_compiler::cuda_tile_runtime_utils::{
            run_tileiras, tileiras_fingerprint, TileirasOptions,
        };
        let fingerprint = tileiras_fingerprint();
        eprintln!("Assembler for debug provenance checks:\n{fingerprint}");
        let known_debug_loss = has_known_inline_debug_loss(fingerprint);
        let source = include_str!("debug_info.rs");
        let helpers = include_str!("fixtures/debug_helpers.rs");
        let kernel = source_line(
            source,
            "fn nested_source<const S: [i32; 1]>(out: &mut Tensor<i32, S>, x: i32) {",
        );
        let store = source_line(source, "out.store(total.broadcast(out.shape()));");
        let first_helper = source_line(
            helpers,
            "pub fn generated<const VALUES: [i32; 2], const FLAG: bool>(x: i32) -> i32 {",
        );
        let last_helper = source_line(helpers, "generated::<VALUES, true>(x) + 1");
        let directory = CubinDirectory::new();
        for level in [DebugInfoLevel::Line, DebugInfoLevel::Full] {
            let artifacts = KernelCompiler::new(
                nested_helper_kernel::__module_ast_self,
                "nested_helper_kernel",
                "nested_source",
            )
            .target("sm_120")
            .generics(vec!["32".into()])
            .strides(&[("out", &[1])])
            .options(CompileOptions::new().debug_info(level))
            .compile()
            .expect("compile nested helper kernel");
            let flags =
                TileirasOptions::from_compile_options(&CompileOptions::new().debug_info(level));
            let cubin = run_tileiras(&artifacts.bytecode().unwrap(), "sm_120", &flags)
                .expect("tileiras must accept the generated locations");
            let path = directory.0.join(format!("{level:?}.cubin"));
            std::fs::write(&path, cubin).unwrap();
            let sass = output(Command::new("nvdisasm").arg("--print-line-info").arg(&path));
            let mut kernel_lines = Vec::new();
            let mut helper_lines = Vec::new();
            let mut generated_lines = Vec::new();
            for entry in sass
                .lines()
                .filter_map(|line| line.trim().strip_prefix("//## File \""))
            {
                let (file, line) = entry.split_once("\", line ").unwrap();
                let line: u32 = line.trim().parse().unwrap();
                if file.ends_with("fixtures/debug_helpers.rs") {
                    helper_lines.push(line);
                } else if file.ends_with("tests/debug_info.rs") {
                    kernel_lines.push(line);
                } else if known_debug_loss && file.starts_with("./tile-ir-") {
                    // These assembler builds substitute internal Tile IR locations
                    // for Rust helper locations. Do not accept arbitrary foreign files.
                    generated_lines.push(line);
                } else {
                    panic!("line entry outside the kernel and its helpers: {entry}");
                }
            }
            assert!(
                !kernel_lines.is_empty(),
                "{level:?}: no kernel lines\n{sass}"
            );
            assert!(
                !helper_lines.is_empty() || known_debug_loss,
                "{level:?}: no helper lines\n{sass}"
            );
            assert!(
                kernel_lines
                    .iter()
                    .all(|line| (kernel..=store).contains(line)),
                "{level:?}: kernel lines {kernel_lines:?} outside {kernel}..={store}\n{sass}"
            );
            assert!(
                helper_lines
                    .iter()
                    .all(|line| (first_helper..=last_helper).contains(line)),
                "{level:?}: helper lines {helper_lines:?} outside {first_helper}..={last_helper}\n{sass}"
            );
            let mut missing_inline_scope = false;
            if level == DebugInfoLevel::Full {
                let dwarf = output(Command::new("readelf").arg("--debug-dump=info").arg(&path));
                assert!(dwarf.contains("nested_source"), "{dwarf}");
                missing_inline_scope = !dwarf.contains("generated_nested");
                assert!(!missing_inline_scope || known_debug_loss, "{dwarf}");
            }
            if known_debug_loss
                && (helper_lines.is_empty() || !generated_lines.is_empty() || missing_inline_scope)
            {
                eprintln!(
                    "::warning::the active tileiras has known inline-debug loss ({level:?}); \
                     surviving Rust locations were checked and frontend/bytecode checks remain strict, \
                     but helper locations/frames are not established by this backend. \
                     See NVIDIA/cuda-rust#1385. Other versions remain strict."
                );
            }
        }
    });
}

#[cutile::module]
mod short_circuit_kernel {
    use cutile::core::*;

    fn inlined_logic(a: bool, b: bool) -> bool {
        a && b
    }

    #[cutile::entry()]
    fn logic_source(out: &mut Tensor<i32, { [32] }>, x: i32, y: i32) {
        let a = x > 0;
        let b = y > 0;
        let c = a || b;
        let d = inlined_logic(a, b);
        let val = if c {
            1i32
        } else if d {
            2i32
        } else {
            0i32
        };
        let tile: Tile<i32, { [32] }> = broadcast_scalar(val, shape![32]);
        out.store(tile);
    }
}

#[test]
fn short_circuit_boolean_constants_keep_expression_locations_and_inline_chain() {
    common::with_test_stack(|| {
        use cutile_ir::ir::{ScalarType, TileElementType, Type};

        let artifacts = KernelCompiler::new(
            short_circuit_kernel::__module_ast_self,
            "short_circuit_kernel",
            "logic_source",
        )
        .target("sm_120")
        .generics(vec![])
        .strides(&[("out", &[1])])
        .options(CompileOptions::new().debug_info(DebugInfoLevel::Full))
        .compile()
        .expect("compile short circuit kernel");

        let source = include_str!("debug_info.rs");
        let or_line = source_line(source, "let c = a || b;");
        let inlined_and_line = source_line(source, "a && b");
        let helper_call_line = source_line(source, "let d = inlined_logic(a, b);");
        let module_line = source_line(source, "mod short_circuit_kernel {");

        let mut direct_or_constant_found = false;
        let mut inlined_and_constant_found = false;

        visit_ops(
            artifacts.module(),
            &artifacts.module().functions,
            &mut |op| {
                if op.opcode != cutile_ir::bytecode::Opcode::Constant {
                    return;
                }
                let is_bool = matches!(
                    op.result_types.first(),
                    Some(Type::Tile(t)) if matches!(t.element_type, TileElementType::Scalar(ScalarType::I1))
                );
                if !is_bool {
                    return;
                }
                if let Location::CallSite { callee, caller } = &op.location {
                    let callee_line = match &**callee {
                        Location::DebugInfo(info) => info.line,
                        Location::FileLineCol { line, .. } => *line,
                        _ => 0,
                    };
                    assert_ne!(
                        callee_line, module_line,
                        "callee of boolean constant must not point to module header: {:?}",
                        op.location
                    );
                    if callee_line == or_line {
                        direct_or_constant_found = true;
                    } else if callee_line == inlined_and_line {
                        let helper_inlined = match &**caller {
                            Location::CallSite {
                                callee: inner_callee,
                                ..
                            } => match &**inner_callee {
                                Location::DebugInfo(info) => info.line == helper_call_line,
                                Location::FileLineCol { line, .. } => *line == helper_call_line,
                                _ => false,
                            },
                            Location::DebugInfo(info) => info.line == helper_call_line,
                            Location::FileLineCol { line, .. } => *line == helper_call_line,
                            _ => false,
                        };
                        if helper_inlined {
                            inlined_and_constant_found = true;
                        }
                    }
                }
            },
        );

        assert!(
            direct_or_constant_found,
            "missing direct short-circuit boolean constant at line {or_line}"
        );
        assert!(
            inlined_and_constant_found,
            "missing inlined short-circuit boolean constant at line {inlined_and_line} called from {helper_call_line}"
        );
    });
}
