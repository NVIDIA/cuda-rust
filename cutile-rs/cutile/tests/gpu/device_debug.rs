/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end: kernels compiled with the debugging/profiling compile options
//! (`--device-debug`, `--lineinfo`) JIT through the real `tileiras` and
//! launch correctly. Each option is part of the cache key, so these runs
//! cannot be served a release cubin (or serve one to anybody else).

use cutile::prelude::*;
use cutile::tile_kernel::{CompileOptions, DebugInfoLevel};
use my_module::scale_add;

use crate::common;

#[cutile::module]
mod my_module {
    use cutile::core::*;
    #[cutile::entry()]
    fn scale_add<const S: [i32; 1]>(
        c: &mut Tensor<f32, S>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(shape!(S), [pid]);
        let tile_b = b.load_tile(shape!(S), [pid]);
        c.store(tile_a + tile_a + tile_b);
    }
}

fn run_with(options: CompileOptions) -> Vec<f32> {
    scale_add(
        api::zeros(&[32]).partition([4]),
        api::ones(&[32]),
        api::ones(&[32]),
    )
    .grid((8, 1, 1))
    .compile_options(options)
    .first()
    .unpartition()
    .to_host_vec()
    .sync()
    .expect("kernel should run")
}

#[test]
fn device_debug_kernel_compiles_and_runs() {
    common::with_test_stack(|| {
        // --device-debug (implies --opt-level 0): the debug cubin must load
        // and produce the same results as release.
        let debug = run_with(CompileOptions::new().debug_info(DebugInfoLevel::Full));
        assert!(debug.iter().all(|&v| (v - 3.0f32).abs() < 1e-6));
    });
}

#[test]
fn lineinfo_kernel_compiles_and_runs() {
    common::with_test_stack(|| {
        let lineinfo = run_with(CompileOptions::new().debug_info(DebugInfoLevel::Line));
        assert!(lineinfo.iter().all(|&v| (v - 3.0f32).abs() < 1e-6));
    });
}

#[test]
fn sanitize_memcheck_kernel_compiles_and_runs() {
    common::with_test_stack(|| {
        // The instrumented cubin must still run correctly outside the
        // sanitizer; running under compute-sanitizer is a manual workflow.
        let sanitized = run_with(CompileOptions::new().sanitize_memcheck(true));
        assert!(sanitized.iter().all(|&v| (v - 3.0f32).abs() < 1e-6));
    });
}
