/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CUTILE_JIT_*` overrides reach both compile stages: the frontend's check
//! placement and the cache key that names the `tileiras` flags. Every case
//! runs in a child process that inherits no `CUTILE_JIT_*` setting.

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_compiler::jit_env::{JIT_DEBUG_ENV, JIT_OPT_LEVEL_ENV, JIT_SANITIZE_ENV};
use std::collections::BTreeMap;

mod common;

#[cutile::module]
mod env_module {
    use cutile::core::*;

    /// Release hoists this kernel's one check to the loop preheader.
    #[cutile::entry]
    fn dynamic_loop<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        let p = x.partition(shape![B]);
        for j in 0i32..n {
            let t = p.load([j]);
            z.store(t);
        }
    }
}

const HOISTED: (u32, u32) = (1, 0);
const IN_PLACE: (u32, u32) = (0, 1);

fn code(name: &str) -> CompileOptions {
    let release = CompileOptions::new().debug_info(DebugInfoLevel::None);
    match name {
        "release" => release.opt_level(3),
        "line" => release.debug_info(DebugInfoLevel::Line).opt_level(3),
        "line_o1" => release.debug_info(DebugInfoLevel::Line).opt_level(1),
        "full" => release.debug_info(DebugInfoLevel::Full),
        "memcheck" => release.opt_level(3).sanitize_memcheck(true),
        _ => panic!("unknown code options {name:?}"),
    }
}

fn compiler(
    options: CompileOptions,
) -> KernelCompiler<fn() -> cutile::cutile_compiler::ast::Module> {
    KernelCompiler::new(
        env_module::__module_ast_self as fn() -> cutile::cutile_compiler::ast::Module,
        "env_module",
        "dynamic_loop",
    )
    .target("sm_120")
    .generics(vec!["64".into()])
    .strides(&[("z", &[1]), ("x", &[1])])
    .options(options)
}

fn placement(options: CompileOptions) -> (u32, u32) {
    let counts = compiler(options).compile().unwrap().check_counts();
    (counts.hoisted, counts.in_place)
}

fn compile_error(options: CompileOptions) -> String {
    match compiler(options).compile() {
        Ok(_) => panic!("compilation must fail"),
        Err(error) => error.to_string(),
    }
}

/// Runs `body` when this process is the child for `test`; otherwise spawns
/// that child with `vars`.
fn isolated(test: &str, vars: &[(&str, &str)], body: impl FnOnce() + Send + 'static) {
    if common::jit_env_case().is_some() {
        common::with_test_stack(body);
    } else {
        common::run_jit_env_case(test, "run", vars);
    }
}

#[test]
fn env_none_turns_off_debug_in_code() {
    isolated(
        "env_none_turns_off_debug_in_code",
        &[(JIT_DEBUG_ENV, "none")],
        || assert_eq!(placement(code("full")), HOISTED),
    );
}

#[test]
fn env_full_keeps_checks_at_their_access_site() {
    isolated(
        "env_full_keeps_checks_at_their_access_site",
        &[(JIT_DEBUG_ENV, "full")],
        || assert_eq!(placement(code("release")), IN_PLACE),
    );
}

#[test]
fn env_line_with_an_opt_level_stays_optimized() {
    isolated(
        "env_line_with_an_opt_level_stays_optimized",
        &[(JIT_DEBUG_ENV, "line"), (JIT_OPT_LEVEL_ENV, "1")],
        || assert_eq!(placement(code("full")), HOISTED),
    );
}

#[test]
fn invalid_value_fails_compilation() {
    isolated(
        "invalid_value_fails_compilation",
        &[(JIT_DEBUG_ENV, "FULL")],
        || {
            let error = compile_error(code("release"));
            assert!(error.contains(JIT_DEBUG_ENV), "{error}");
        },
    );
}

#[test]
fn full_debug_with_an_opt_level_fails_compilation() {
    isolated(
        "full_debug_with_an_opt_level_fails_compilation",
        &[(JIT_DEBUG_ENV, "full"), (JIT_OPT_LEVEL_ENV, "2")],
        || {
            let error = compile_error(code("release"));
            assert!(
                error.contains(JIT_DEBUG_ENV) && error.contains(JIT_OPT_LEVEL_ENV),
                "{error}"
            );
        },
    );
}

#[test]
fn overrides_are_read_once_per_process() {
    isolated("overrides_are_read_once_per_process", &[], || {
        assert_eq!(placement(code("release")), HOISTED);
        std::env::set_var(JIT_DEBUG_ENV, "full");
        assert_eq!(placement(code("release")), HOISTED);
    });
}

/// One compile per process, so the key a process reports is computed under
/// that process's overrides alone. Equal keys in two processes mean one
/// would serve the other's disk-cache entry.
#[test]
fn cache_key_follows_the_effective_mode() {
    const TEST: &str = "cache_key_follows_the_effective_mode";
    if let Some(name) = common::jit_env_case() {
        let options = code(&name);
        let (hoisted, in_place) = placement(options.clone());
        let key = compiler(options).l2_cache_key().unwrap();
        println!("PROBE key={key} placement={hoisted},{in_place}");
        return;
    }
    let probe = |name: &str, vars: &[(&str, &str)]| {
        let output = common::run_jit_env_case(TEST, name, vars);
        let line = output
            .lines()
            .find_map(|line| line.split_once("PROBE ").map(|(_, probe)| probe))
            .unwrap_or_else(|| panic!("no probe line:\n{output}"));
        let fields: BTreeMap<_, _> = line
            .split(' ')
            .filter_map(|field| field.split_once('='))
            .collect();
        (fields["key"].to_string(), fields["placement"].to_string())
    };

    let release = probe("release", &[]);
    let line = probe("line", &[]);
    let line_o1 = probe("line_o1", &[]);
    let full = probe("full", &[]);
    let memcheck = probe("memcheck", &[]);
    let keys = [&release.0, &line.0, &line_o1.0, &full.0, &memcheck.0];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b, "distinct modes must not share a cache entry");
        }
    }
    assert_eq!(release.1, "1,0");
    assert_eq!(full.1, "0,1");

    for (expected, name, vars) in [
        (&release, "full", &[(JIT_DEBUG_ENV, "none")][..]),
        (&line, "release", &[(JIT_DEBUG_ENV, "line")]),
        (
            &line_o1,
            "full",
            &[(JIT_DEBUG_ENV, "line"), (JIT_OPT_LEVEL_ENV, "1")],
        ),
        (&full, "release", &[(JIT_DEBUG_ENV, "full")]),
        (&memcheck, "release", &[(JIT_SANITIZE_ENV, "memcheck")]),
    ] {
        assert_eq!(
            &probe(name, vars),
            expected,
            "{vars:?} over {name:?} must match the same mode set in code"
        );
    }
}
