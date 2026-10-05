/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive control for the collector's host-CPU guard: a closure defined
//! inside a `#[target_feature]` host function, passed to a generic kernel.
//!
//! rustc copies the enclosing function's target features onto every closure
//! defined in it, so the closure below carries `avx2` (or `neon`) in its
//! codegen attributes even though its body is plain arithmetic. The guard
//! that refuses `#[target_feature]` functions reachable from a kernel must
//! not fire on the closure itself; only a genuinely host-only call inside
//! its body would be refused. This example has to keep compiling and
//! producing correct results.
//!
//! Usage:
//!   cargo oxide run host_closure_target_feature
//!
//! Expected: prints SUCCESS. On an x86_64 host without AVX2 the launch is
//! skipped and the example still prints SUCCESS, because the build is the
//! property under test.

use std::sync::Arc;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_module, load_kernel_module};

const N: usize = 1024;

#[cuda_module]
mod kernels {
    use super::*;

    /// Generic map kernel. `F` is the anonymous closure type from the host side.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("this example needs an x86_64 or aarch64 host");

/// Launches `map` with a closure written directly inside a feature-gated
/// host function. The attribute is the point of the example: the closure
/// inherits it, and must still be accepted as device code.
///
/// # Safety
/// The caller must have checked that the host CPU supports the feature.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn launch_from_feature_gated_host(
    ctx: &Arc<CudaContext>,
    input: &[f32],
    scale: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let input_dev = DeviceBuffer::from_host(&stream, input)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let module = load_kernel_module(ctx, "host_closure_target_feature")?;
    let typed = kernels::from_module(module)?;

    // The closure under test. Its body is portable; its attributes are not.
    let f = move |x: f32| x * scale + 1.0;

    // SAFETY: a 1D launch; the kernel guards its index against the output
    // length, and both buffers are live on this stream.
    unsafe {
        typed.map::<f32, _>(
            stream.as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            f,
            &input_dev,
            &mut out_dev,
        )
    }?;
    Ok(out_dev.to_host_vec(&stream)?)
}

fn host_has_feature() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== host_closure_target_feature ===");

    if !host_has_feature() {
        println!("skipping: host CPU lacks the feature this example enables");
        println!("SUCCESS (build is the property under test)");
        return Ok(());
    }

    let ctx = CudaContext::new(0)?;
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let scale = 3.0f32;

    // SAFETY: the feature check above passed.
    let output = unsafe { launch_from_feature_gated_host(&ctx, &input, scale) }?;

    let errors = output
        .iter()
        .zip(&input)
        .filter(|(got, x)| (**got - (**x * scale + 1.0)).abs() > 1e-5)
        .count();
    if errors == 0 {
        println!("SUCCESS: closure from a #[target_feature] host fn ran on the GPU ({N} elements)");
        Ok(())
    } else {
        println!("FAILED: {errors} mismatches");
        std::process::exit(1)
    }
}
