/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CUTILE_JIT_DEBUG=full` must reach both compile stages: the frontend stops
//! hoisting checks, and the cache key names the `-O0 --device-debug` flags,
//! even when the code asks for an optimized build.

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_compiler::jit_env::JIT_DEBUG_ENV;
use std::process::Command;

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

#[test]
fn env_full_debug_reaches_both_stages() {
    // The overrides are read once per process, so the assertions run in a
    // child that starts with the variable set.
    if std::env::var_os(JIT_DEBUG_ENV).is_none() {
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "env_full_debug_reaches_both_stages",
                "--nocapture",
            ])
            .env(JIT_DEBUG_ENV, "full")
            .status()
            .unwrap();
        assert!(status.success(), "child run failed: {status}");
        return;
    }
    assert_eq!(std::env::var(JIT_DEBUG_ENV).as_deref(), Ok("full"));
    common::with_test_stack(|| {
        let optimized = || {
            compiler(
                CompileOptions::new()
                    .debug_info(DebugInfoLevel::None)
                    .opt_level(3),
            )
        };
        let counts = optimized().compile().unwrap().check_counts();
        assert_eq!(
            (counts.hoisted, counts.in_place),
            (0, 1),
            "the frontend must keep the check at its access site"
        );
        assert_eq!(
            optimized().l2_cache_key().unwrap(),
            compiler(CompileOptions::new().debug_info(DebugInfoLevel::Full))
                .l2_cache_key()
                .unwrap(),
            "the cache key must name the full-debug tileiras flags"
        );
    });
}
