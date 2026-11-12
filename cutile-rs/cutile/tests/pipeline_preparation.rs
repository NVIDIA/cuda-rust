/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU tests isolated from other binaries that change process-wide JIT counters.
#![cfg(feature = "experimental-tune")]

mod common;
use cuda_core::{Context, IntoResult};
use cutile::prelude::*;
use cutile::tile_kernel::{
    contains_cuda_function, jit_compile_count, kernel_cache_slot, KernelCompileJob, PreparedKernel,
};
use cutile_compiler::compile_control::CompileControl;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry]
    fn double<const N: i32>(out: &mut Tensor<f32, { [N] }>, x: &Tensor<f32, { [-1] }>) {
        let value = load_tile_like(x, out);
        out.store(value + value);
    }
    #[cutile::entry]
    fn checked_walk<const B: i32>(
        z: &mut Tensor<f32, { [B, B] }>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(shape![B, B]);
        let py = y.partition(shape![B, B]);
        let mut acc: Tile<f32, { [B, B] }> = constant(0.0, shape![B, B]);
        for k in 0i32..num_tiles(&px, 1) {
            acc = acc + py.load([k, 0i32]);
        }
        z.store(acc);
    }
}

fn prepare_on_cpu(job: KernelCompileJob) -> PreparedKernel {
    std::thread::Builder::new()
        .stack_size(common::TEST_STACK_SIZE)
        .spawn(move || {
            let mut before = std::ptr::null_mut();
            // SAFETY: initialized writable context output; does not create a context.
            unsafe { cuda_core::sys::cuCtxGetCurrent(&mut before) }
                .result()
                .unwrap();
            assert!(before.is_null());
            let prepared = job.compile(&CompileControl::default()).unwrap();
            let mut after = std::ptr::null_mut();
            // SAFETY: same current-thread context query as above.
            unsafe { cuda_core::sys::cuCtxGetCurrent(&mut after) }
                .result()
                .unwrap();
            assert!(after.is_null(), "preparation must not bind a CUDA context");
            prepared
        })
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn prepared_binary_is_not_loaded_until_install_and_first_launch_reuses_it() {
    let _lock = common::cache_test_lock();
    let stream = Context::new(0).unwrap().new_stream().unwrap();
    let x: Arc<Tensor<f32>> = api::ones(&[64]).sync_on(&stream).unwrap().into();
    let mut out = api::zeros::<f32>(&[64]).sync_on(&stream).unwrap();
    let job = kernels::double((&mut out).partition([32]), x.clone())
        .specialize_on(&stream)
        .unwrap()
        .into_compile_job()
        .unwrap();
    let key = job.key().clone();
    let count = jit_compile_count();
    let prepared = prepare_on_cpu(job);
    assert!(!contains_cuda_function(&key));
    assert_eq!(jit_compile_count(), count);
    assert!(!prepared.install(&CompileControl::default()).unwrap());
    assert!(contains_cuda_function(&key));
    assert_eq!(jit_compile_count(), count + 1);
    kernels::double((&mut out).partition([32]), x)
        .sync_on(&stream)
        .unwrap();
    assert_eq!(
        jit_compile_count(),
        count + 1,
        "launch must reuse prepared code"
    );
    assert_eq!(out.to_host_vec().sync_on(&stream).unwrap(), vec![2.0; 64]);
}

#[test]
fn prepared_binary_preserves_hoisted_launch_checks_on_cache_hits() {
    let _lock = common::cache_test_lock();
    common::with_test_stack(|| {
        let stream = Context::new(0).unwrap().new_stream().unwrap();
        let x: Arc<Tensor<f32>> = api::ones(&[16, 64]).sync_on(&stream).unwrap().into();
        let y: Arc<Tensor<f32>> = api::ones(&[64, 16]).sync_on(&stream).unwrap().into();
        let short_y: Arc<Tensor<f32>> = api::ones(&[48, 16]).sync_on(&stream).unwrap().into();
        let mut out = api::zeros::<f32>(&[16, 16]).sync_on(&stream).unwrap();
        let specialization =
            kernels::checked_walk((&mut out).partition([16, 16]), x.clone(), y.clone())
                .specialize_on(&stream)
                .unwrap();
        let key = specialization.l1_cache_key().clone();
        let bad = kernels::checked_walk((&mut out).partition([16, 16]), x.clone(), short_y.clone())
            .specialize_on(&stream)
            .unwrap();
        assert_eq!(
            &key,
            bad.l1_cache_key(),
            "bad launch must exercise the same cached specialization"
        );
        let prepared = prepare_on_cpu(specialization.into_compile_job().unwrap());
        prepared.install(&CompileControl::default()).unwrap();
        let slot = kernel_cache_slot(&key);
        assert!(
            !slot.get().unwrap().0.validator.launch_checks.is_empty(),
            "test requires a host-side bounds obligation"
        );
        kernels::checked_walk((&mut out).partition([16, 16]), x.clone(), y)
            .sync_on(&stream)
            .unwrap();
        let error = kernels::checked_walk((&mut out).partition([16, 16]), x, short_y)
            .sync_on(&stream)
            .err()
            .expect("short input must be rejected before launch");
        assert!(error.to_string().contains("launch check failed"), "{error}");
    });
}
