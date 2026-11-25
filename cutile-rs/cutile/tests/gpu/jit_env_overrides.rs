/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The launcher applies `CUTILE_JIT_*` before both compile stages and keys
//! its disk cache by the effective mode. Each case launches in a fresh
//! process against one shared store, so a cubin served from disk can only
//! have come from an earlier process with an equivalent mode.

use crate::common;
use cutile::api;
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use cutile_compiler::jit_env::{JIT_DEBUG_ENV, JIT_OPT_LEVEL_ENV, JIT_SANITIZE_ENV};
use std::collections::BTreeMap;
use std::sync::Arc;

#[cutile::module]
mod launch_env_module {
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

const B: usize = 64;
const TILES: usize = 4;
const STORE_ENV: &str = "CUTILE_TEST_JIT_ENV_STORE";

fn code(name: &str) -> CompileOptions {
    let release = CompileOptions::new()
        .debug_info(DebugInfoLevel::None)
        .opt_level(3);
    match name {
        "release" => release,
        "line" => release.debug_info(DebugInfoLevel::Line),
        "full" => release.debug_info(DebugInfoLevel::Full),
        _ => panic!("unknown code options {name:?}"),
    }
}

/// The loop stores every tile of `x` in turn, so `z` ends as the last one.
fn launch_and_check(options: CompileOptions) {
    let x: Vec<f32> = (0..B * TILES).map(|v| v as f32).collect();
    let x: Arc<Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(x))
        .sync()
        .expect("upload x")
        .into();
    let (z, _x, _n) =
        launch_env_module::dynamic_loop(api::zeros::<f32>(&[B]).partition([B]), x, TILES as i32)
            .generics(vec![B.to_string()])
            .compile_options(options)
            .sync()
            .expect("dynamic_loop");
    let z = z.unpartition().to_host_vec().sync().expect("download z");
    let last: Vec<f32> = ((TILES - 1) * B..TILES * B).map(|v| v as f32).collect();
    assert_eq!(z, last);
}

#[test]
fn disk_cache_follows_the_effective_mode() {
    const TEST: &str = "jit_env_overrides::disk_cache_follows_the_effective_mode";
    if let Some(name) = common::jit_env_case() {
        let store = std::env::var_os(STORE_ENV).expect("store directory");
        common::with_test_stack(move || {
            jit_cache::enable(Arc::new(
                FileSystemJitStore::new(&store).expect("open store"),
            ));
            launch_and_check(code(&name));
        });
        return;
    }

    let store = std::env::temp_dir().join(format!("cutile_jit_env_store_{}", std::process::id()));
    std::fs::create_dir_all(&store).unwrap();
    let store_dir = store.to_str().unwrap();
    let launch = |name: &str, vars: &[(&str, &str)]| {
        let vars = [vars, &[(STORE_ENV, store_dir), ("CUTILE_JIT_TIMING", "1")]].concat();
        let output = common::run_jit_env_case(TEST, name, &vars);
        let timing = output
            .lines()
            .find(|line| {
                line.starts_with("CUTILE_JIT_TIMING") && line.contains(" function=dynamic_loop ")
            })
            .unwrap_or_else(|| panic!("no timing line for dynamic_loop:\n{output}"));
        let fields: BTreeMap<_, _> = timing
            .split(' ')
            .filter_map(|field| field.split_once('='))
            .collect();
        (
            fields["stage2_source"].to_string(),
            format!("{},{}", fields["checks_hoisted"], fields["checks_in_place"]),
        )
    };

    for (name, vars, source, placement) in [
        ("release", &[(JIT_DEBUG_ENV, "full")][..], "tileiras", "0,1"),
        ("full", &[], "disk", "0,1"),
        ("release", &[], "tileiras", "1,0"),
        ("full", &[(JIT_DEBUG_ENV, "none")], "disk", "1,0"),
        ("release", &[(JIT_DEBUG_ENV, "line")], "tileiras", "1,0"),
        ("line", &[], "disk", "1,0"),
        ("line", &[(JIT_OPT_LEVEL_ENV, "1")], "tileiras", "1,0"),
        (
            "release",
            &[(JIT_SANITIZE_ENV, "memcheck")],
            "tileiras",
            "1,0",
        ),
    ] {
        assert_eq!(
            launch(name, vars),
            (source.to_string(), placement.to_string()),
            "{vars:?} over {name:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&store);
}
