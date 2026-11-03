/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The compile-only terminal at a REAL dispatch call expression: a composed
//! builder (generics + compile options + a mapped output binding), warmed
//! with `api::meta` placeholders — no launch, no device allocation — must
//! populate the cache the subsequent real dispatch hits. Pins grout's
//! warmup contract (autotuning review, ask #8): warming through the
//! dispatch expression itself, so the warmed specialization cannot drift
//! from the dispatched one.

use cutile::prelude::*;
use cutile::tile_kernel::jit_compile_count;
use cutile_compiler::compiler::utils::CompileOptions;

use crate::common;

#[cutile::module]
mod composed_warm_module {
    use cutile::core::*;

    /// Mapped output + foreign inputs: the persistent_gemm shape, the same
    /// binding forms grout's dispatch sites compose.
    #[cutile::entry]
    fn mapped_madd<const BM: i32, const BN: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(shape![BM, BN]);
        for idx in z.iter_indices() {
            let (bm, bn) = idx.components();
            let t = px.load([bm, bn]);
            z.store(t + t, idx);
        }
    }
}

#[test]
fn composed_builder_compile_terminal_warms_the_dispatched_specialization() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let generics = || {
            vec![
                "16".to_string(),
                "16".to_string(),
                "4".to_string(),
                "1".to_string(),
            ]
        };
        let opts = CompileOptions::default();

        // Prime the tensor-creation kernels the real dispatch will allocate
        // with, so the counter isolates the kernel under test.
        let _pz = api::zeros::<f32>(&[64, 64]).sync().expect("prime zeros");
        let _px = api::ones::<f32>(&[64, 64]).sync().expect("prime ones");

        // Warm via meta placeholders through the SAME call expression the
        // real dispatch uses. No launch, no device allocation.
        let before = jit_compile_count();
        let zm = api::meta::<f32>(&[64, 64])
            .sync()
            .expect("meta tensor")
            .partition([16, 16])
            .map([4, 1], 4);
        let xm = api::meta::<f32>(&[64, 64]).sync().expect("meta tensor");
        composed_warm_module::mapped_madd(zm, std::sync::Arc::new(xm))
            .generics(generics())
            .compile_options(opts.clone())
            .compile()
            .expect("compile-only terminal on the composed builder");
        assert_eq!(
            jit_compile_count(),
            before + 1,
            "the compile terminal must JIT exactly this specialization"
        );

        // The real dispatch, identical expression with real tensors: must be
        // a pure cache hit — zero new JIT compiles.
        let z = api::zeros::<f32>(&[64, 64])
            .sync()
            .expect("alloc z")
            .partition([16, 16])
            .map([4, 1], 4);
        let x: std::sync::Arc<Tensor<f32>> =
            api::ones::<f32>(&[64, 64]).sync().expect("alloc x").into();
        let (z, _x) = composed_warm_module::mapped_madd(z, x)
            .generics(generics())
            .compile_options(opts)
            .sync()
            .expect("real dispatch after warmup");
        assert_eq!(
            jit_compile_count(),
            before + 1,
            "the warmed specialization must be a cache hit at real dispatch"
        );
        let out = z.unpartition().to_host_vec().sync().expect("read back");
        assert!(out.iter().all(|&v| v == 2.0), "kernel must compute x + x");
    });
}

#[test]
fn cache_management_evicts_and_recompiles() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        use cutile::tile_kernel::{
            clear_kernel_cache, contains_cuda_function, evict_kernel, retain_kernels,
        };
        let generics = || {
            vec![
                "16".to_string(),
                "16".to_string(),
                "4".to_string(),
                "1".to_string(),
            ]
        };
        let make = || {
            let zm = api::meta::<f32>(&[64, 64])
                .sync()
                .expect("meta")
                .partition([16, 16])
                .map([4, 1], 4);
            let xm = api::meta::<f32>(&[64, 64]).sync().expect("meta");
            composed_warm_module::mapped_madd(zm, std::sync::Arc::new(xm)).generics(generics())
        };
        // Nothing is executing (no launches in this test), so the quiesce
        // obligation is met trivially.
        let key = make().l1_cache_key().expect("key");
        make().compile().expect("compile");
        assert!(contains_cuda_function(&key), "compiled into the cache");

        // Targeted eviction removes exactly this specialization. SAFETY:
        // nothing is executing (no launches in this test), so the quiesce
        // obligation on the eviction APIs is met trivially.
        assert!(unsafe { evict_kernel(&key) }, "the key was cached");
        assert!(
            !unsafe { evict_kernel(&key) },
            "eviction is not idempotent-true"
        );
        assert!(!contains_cuda_function(&key), "evicted");

        // Recompile, then retain(false) and clear() both empty it.
        let before = jit_compile_count();
        make().compile().expect("recompile");
        assert_eq!(
            jit_compile_count(),
            before + 1,
            "eviction forces a real recompile"
        );
        assert!(
            unsafe { retain_kernels(|_| false) } >= 1,
            "retain reports removals"
        );
        assert!(
            !contains_cuda_function(&key),
            "retain(false) drops everything"
        );
        make().compile().expect("compile again");
        assert!(
            unsafe { clear_kernel_cache() } >= 1,
            "clear reports removals"
        );
        assert!(!contains_cuda_function(&key), "clear drops everything");
    });
}

/// `retain_kernels`' predicate must be able to re-enter the cache — the
/// natural "keep only my kernel's specializations" shape queries it. This
/// pins the deadlock fix: the predicate runs outside any DashMap shard lock,
/// so a lookup from inside it returns instead of self-deadlocking that shard
/// (which would wedge every JIT lookup in the process). If this regressed —
/// e.g. back to `cache.retain(|k, _| pred(k))` with the predicate under the
/// shard lock — this test would hang forever rather than fail.
#[test]
fn retain_predicate_may_reenter_the_cache_without_deadlock() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        use cutile::tile_kernel::{contains_cuda_function, retain_kernels};
        let make = || {
            let zm = api::meta::<f32>(&[64, 64])
                .sync()
                .expect("meta")
                .partition([16, 16])
                .map([4, 1], 4);
            let xm = api::meta::<f32>(&[64, 64]).sync().expect("meta");
            composed_warm_module::mapped_madd(zm, std::sync::Arc::new(xm)).generics(vec![
                "16".to_string(),
                "16".to_string(),
                "4".to_string(),
                "1".to_string(),
            ])
        };
        let key = make().l1_cache_key().expect("key");
        make().compile().expect("compile");
        assert!(contains_cuda_function(&key), "seeded");

        // SAFETY: no launch is in flight in this test. The predicate re-enters
        // the cache (a read lookup) and keeps everything; it must complete.
        let removed = unsafe {
            retain_kernels(|k| {
                let _present = contains_cuda_function(k);
                true
            })
        };
        assert_eq!(removed, 0, "kept everything");
        assert!(
            contains_cuda_function(&key),
            "still cached after re-entrant retain"
        );
    });
}
