/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The per-launch-site resolution cache. A site keeps every specialization
//! it resolved, so a kernel launched at alternating shapes hits on every
//! launch, and an eviction empties every site, so no site serves or keeps
//! loaded a function the eviction removed.
//!
//! Every test here takes `cache_test_lock` and evicts only `site_module`'s
//! kernel (or only bumps the epoch), so no other test's module is unloaded.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cuda_async::device_context::Validator;
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Device, Function};
use cutile::prelude::*;
use cutile::tile_kernel::{
    compile_from_context, contains_cuda_function, evict_kernel, retain_kernels, CacheEpoch,
    CompileOptions, LaunchSite, SiteResolution, TileFunctionKey,
};

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

/// Evicts every specialization of `site_module`'s kernel, and nothing else,
/// through `retain_kernels`; returns how many were removed.
///
/// # Safety
/// No launch of `site_module`'s kernel may be in flight.
unsafe fn evict_site_module() -> usize {
    unsafe { retain_kernels(|key| key.module_name() != "site_module") }
}

/// The kernel-cache keys of `site_module`'s kernel, read without removing
/// anything.
fn site_module_keys() -> Vec<TileFunctionKey> {
    let mut keys = Vec::new();
    // SAFETY: the predicate keeps every key, so nothing is evicted.
    unsafe {
        retain_kernels(|key| {
            if key.module_name() == "site_module" {
                keys.push(key.clone());
            }
            true
        })
    };
    keys
}

#[test]
fn alternating_specializations_through_one_site_write_correct_results() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        // Two tile sizes = two specializations through the same call site.
        // This checks results only; `a_site_serves_every_specialization_it_stored`
        // checks that both specializations hit.
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
        // Quiesced (nothing of ours in flight after sync). Evicting this key
        // empties the launcher's site, so the next launch must resolve
        // through the global cache again instead of reusing the evicted
        // function.
        assert!(unsafe { evict_kernel(&key) }, "evicted");
        assert!(!contains_cuda_function(&key), "evicted");
        let host = run(32, 16);
        assert!(
            contains_cuda_function(&key),
            "a launch after the eviction re-resolved through the global cache"
        );
        for (i, v) in host.iter().enumerate() {
            assert_eq!(*v, 2.0 * i as f32, "index {i}");
        }
    });
}

/// Resolves `scale` for `B = b` as the generated launcher does: the epoch
/// is read before the function is.
fn resolve(ctx: &ExecutionContext, b: usize) -> (CacheEpoch, Arc<Function>, Arc<Validator>) {
    let epoch = CacheEpoch::now();
    let (function, validator) = compile_from_context(
        ctx,
        site_module::__module_ast_self,
        "site_module",
        "scale",
        "scale_entry",
        vec![b.to_string()],
        vec![("z".to_string(), vec![1]), ("x".to_string(), vec![1])],
        vec![],
        vec![],
        None,
        CompileOptions::default(),
        site_module::_SOURCE_HASH,
    )
    .expect("compile");
    (epoch, function, validator)
}

/// The site entry for a [`resolve`] result for `B = b`.
fn resolution(
    epoch: CacheEpoch,
    b: usize,
    function: &Arc<Function>,
    validator: &Arc<Validator>,
) -> SiteResolution {
    SiteResolution::new(
        epoch,
        0,
        vec![b.to_string()],
        vec![],
        vec![],
        None,
        CompileOptions::default(),
        Arc::clone(function),
        Arc::clone(validator),
    )
}

/// The probe of a [`resolution`] for `B = b`, as `LaunchSite::get` takes it.
fn probe(site: &LaunchSite, b: usize) -> Option<(Arc<Function>, Arc<Validator>)> {
    site.get(
        0,
        &[b.to_string()],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
}

fn context() -> ExecutionContext {
    let stream = Device::new(0)
        .expect("device")
        .new_stream()
        .expect("stream");
    ExecutionContext::new(stream)
}

#[test]
fn a_site_serves_every_specialization_it_stored() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static SITE: LaunchSite = LaunchSite::new();
        let ctx = context();
        let (epoch16, f16, v16) = resolve(&ctx, 16);
        let (epoch8, f8, v8) = resolve(&ctx, 8);
        assert!(!Arc::ptr_eq(&f16, &f8), "two specializations");
        let hit = |b: usize, want: &Arc<Function>, path: &str| {
            let (f, _) = probe(&SITE, b).unwrap_or_else(|| panic!("B = {b} misses ({path})"));
            assert!(
                Arc::ptr_eq(&f, want),
                "B = {b} serves its own function ({path})"
            );
        };

        SITE.store(resolution(epoch16, 16, &f16, &v16));
        SITE.store(resolution(epoch8, 8, &f8, &v8));
        hit(8, &f8, "newest entry");
        hit(16, &f16, "hash lookup");
        assert!(
            probe(&SITE, 4).is_none(),
            "an unstored specialization misses"
        );

        // Storing B = 16 again makes it the newest; B = 8 now takes the hash
        // lookup. The replaced entry is dropped, not kept beside the new one.
        SITE.store(resolution(epoch16, 16, &f16, &v16));
        hit(16, &f16, "newest entry");
        hit(8, &f8, "hash lookup");
        assert_eq!(
            SITE.entry_epochs_for_tests().len(),
            2,
            "one entry per probe"
        );

        // SAFETY: nothing has launched these resolutions.
        unsafe { evict_site_module() };
        assert!(
            SITE.entry_epochs_for_tests().is_empty(),
            "emptied by the eviction"
        );
    });
}

#[test]
fn launch_sites_never_keep_an_evicted_function() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static SITE: LaunchSite = LaunchSite::new();
        let ctx = context();

        // An eviction between resolving and storing: the function it removed
        // must not be stored, or the site would keep its module loaded.
        let (epoch, function, validator) = resolve(&ctx, 16);
        // SAFETY: nothing has launched this kernel.
        assert!(unsafe { evict_site_module() } > 0, "evicted");
        assert_ne!(epoch, CacheEpoch::now());
        SITE.store(resolution(epoch, 16, &function, &validator));
        assert_eq!(
            Arc::strong_count(&function),
            1,
            "stale resolution discarded"
        );

        // Stored while current, then evicted through `retain_kernels`: the
        // eviction empties the site.
        let (epoch, function, validator) = resolve(&ctx, 16);
        SITE.store(resolution(epoch, 16, &function, &validator));
        assert_eq!(
            Arc::strong_count(&function),
            3,
            "held by the site and the cache"
        );
        // SAFETY: as above.
        unsafe { evict_site_module() };
        assert_eq!(
            Arc::strong_count(&function),
            1,
            "site emptied by retain_kernels"
        );

        // The same through `evict_kernel`, one key at a time.
        let (epoch, function, validator) = resolve(&ctx, 16);
        SITE.store(resolution(epoch, 16, &function, &validator));
        assert_eq!(Arc::strong_count(&function), 3);
        for key in site_module_keys() {
            // SAFETY: as above.
            unsafe { evict_kernel(&key) };
        }
        assert_eq!(
            Arc::strong_count(&function),
            1,
            "site emptied by evict_kernel"
        );
    });
}

#[test]
fn launch_site_stores_racing_evictions_keep_only_current_entries() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        static SITE: LaunchSite = LaunchSite::new();
        let ctx = context();
        let (_, function, validator) = resolve(&ctx, 16);
        // This test's handle and the kernel cache's. The evictions below only
        // bump the epoch and empty sites, so the cache keeps its share.
        let baseline = Arc::strong_count(&function);

        for round in 0..20 {
            let storing = AtomicUsize::new(4);
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| {
                        for _ in 0..20 {
                            // Stamped as a launcher stamps it; the pause
                            // stands in for resolving, so an eviction often
                            // lands between the stamp and the store.
                            let epoch = CacheEpoch::now();
                            std::thread::sleep(Duration::from_micros(50));
                            SITE.store(resolution(epoch, 16, &function, &validator));
                        }
                        storing.fetch_sub(1, Ordering::Release);
                    });
                }
                // Evict until every store is in, so the last stores race an
                // eviction with no later one to clean up after them.
                scope.spawn(|| {
                    while storing.load(Ordering::Acquire) > 0 {
                        // SAFETY: every site keeps only functions the kernel
                        // cache also holds (this one included), so emptying
                        // them unloads no module; the cache is not touched.
                        unsafe { cutile::tile_kernel::bump_kernel_cache_epoch_for_tests() };
                        std::thread::sleep(Duration::from_micros(10));
                    }
                });
            });
            // Checked on the entries themselves, not through `get`, which
            // skips stale ones: every entry left must be current, and the
            // site's share of the function must be exactly its entries.
            let epochs = SITE.entry_epochs_for_tests();
            let now = CacheEpoch::now();
            assert!(
                epochs.iter().all(|&e| e == now),
                "round {round}: a stale entry survived the evictions"
            );
            assert!(epochs.len() <= 1, "round {round}: one entry per probe");
            assert_eq!(
                Arc::strong_count(&function),
                baseline + epochs.len(),
                "round {round}: the site holds only its entries"
            );
            assert_eq!(probe(&SITE, 16).is_some(), !epochs.is_empty());
        }
        // SAFETY: as above.
        unsafe { cutile::tile_kernel::bump_kernel_cache_epoch_for_tests() };
        assert_eq!(Arc::strong_count(&function), baseline, "emptied");
    });
}
