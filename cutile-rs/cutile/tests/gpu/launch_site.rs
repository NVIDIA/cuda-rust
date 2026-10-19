/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The per-launch-site resolution cache: steady-state launches skip key
//! construction entirely, so these tests pin the two ways a site must NOT
//! serve a stale resolution — a changed specialization (alternating
//! generics through one call site) and an evicted global cache.

use cutile::prelude::*;
use cutile::tile_kernel::contains_cuda_function;

use crate::common;

#[cutile::module]
mod site_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn scale<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>) {
        let tile = x.load_like(z);
        z.store(tile + tile);
    }
}

use site_module::scale;

fn run(len: usize, tile: usize) -> Vec<f32> {
    scale(
        api::arange::<f32>(len).partition([tile]),
        api::arange::<f32>(len),
    )
    .grid(((len / tile) as u32, 1, 1))
    .first()
    .unpartition()
    .to_host_vec()
    .sync()
    .expect("scale kernel")
}

#[test]
fn alternating_specializations_through_one_site_stay_correct() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Two tile sizes = two specializations through the same call site.
        // The single-entry site cache thrashes; results must stay correct.
        for _ in 0..3 {
            for tile in [4usize, 8] {
                let host = run(32, tile);
                for (i, v) in host.iter().enumerate() {
                    assert_eq!(*v, 2.0 * i as f32, "tile {tile}, index {i}");
                }
            }
        }
    });
}

#[test]
fn cache_eviction_invalidates_hot_launch_sites() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Key-scoped observations: concurrent tests in this binary compile
        // their own kernels, so global compile counts race, but this key is
        // ours alone.
        let key = scale(
            api::arange::<f32>(32).partition([16]),
            api::arange::<f32>(32),
        )
        .generics(vec!["16".to_string()])
        .l1_cache_key()
        .expect("key");
        run(32, 16); // fill the site (and the global cache)
        assert!(contains_cuda_function(&key), "filled");
        // Quiesced (nothing of ours in flight after sync): evicting must
        // force the hot site to re-resolve, not serve its stale
        // Arc<Function> — the epoch check.
        unsafe {
            cutile::tile_kernel::clear_kernel_cache_for_tests();
        }
        assert!(!contains_cuda_function(&key), "evicted");
        let host = run(32, 16);
        assert!(
            contains_cuda_function(&key),
            "a launch after the clear re-resolved through the global cache"
        );
        for (i, v) in host.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}
