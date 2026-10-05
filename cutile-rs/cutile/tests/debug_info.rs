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
