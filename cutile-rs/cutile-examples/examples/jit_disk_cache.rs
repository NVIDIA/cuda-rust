/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in persistent on-disk cubin cache.
//!
//! The in-memory kernel cache dies with the process; the disk cache survives
//! it. **Run this example twice** to see the difference:
//!
//! - first run: every kernel is compiled by `tileiras` and its cubin stored on
//!   disk (`disk hits: 0`; `backend compiles` is the number of distinct kernels
//!   — the `add` kernel plus the fill kernel behind `ones`/`zeros`);
//! - second run: a fresh process finds those cubins on disk and skips `tileiras`
//!   entirely (`backend compiles: 0`, and `disk hits` equals the first run's
//!   backend compiles).
//!
//! Disk persistence is off by default and has no environment-variable switch;
//! it starts only when the program calls `jit_cache::enable*`. This example
//! uses [`FileSystemJitStore::default_location`], which resolves to
//! `$XDG_CACHE_HOME/cutile/kernels` or `~/.cache/cutile/kernels` on Unix,
//! created with strict per-user permissions. Delete that directory to start cold again.
//!
//! `CUTILE_JIT_TIMING=1` prints per-stage timings including
//! `stage2_source=disk|tileiras`.

use cutile::api::{ones, zeros};
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod disk_cache_example_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

fn main() {
    if !cutile_examples::requirements::BASELINE
        .check("jit_disk_cache", 0)
        .expect("query example capabilities")
    {
        return;
    }

    let store = FileSystemJitStore::default_location().expect("open default cache directory");
    println!("cache directory: {}", store.root().display());
    jit_cache::enable(Arc::new(store));

    let x: Arc<Tensor<f32>> = ones(&[1024]).sync().expect("ones").into();
    let y: Arc<Tensor<f32>> = ones(&[1024]).sync().expect("ones").into();
    let z = zeros(&[1024]).sync().expect("zeros").partition([128]);

    let t0 = Instant::now();
    let z_host = disk_cache_example_module::add(z, x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add kernel");
    println!("first launch (includes JIT): {:?}", t0.elapsed());
    assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));

    let stats = jit_cache::stats();
    println!(
        "backend compiles: {}, disk hits: {}, entries written: {}, io errors: {}",
        jit_cache::jit_backend_compile_count(),
        jit_cache::jit_disk_hit_count(),
        stats.puts,
        stats.io_errors,
    );
    if jit_cache::jit_disk_hit_count() > 0 {
        println!("the cubin came from the disk cache; tileiras never ran in this process");
    } else {
        println!("cold run: the cubin was compiled and stored — run this example again");
    }
}
