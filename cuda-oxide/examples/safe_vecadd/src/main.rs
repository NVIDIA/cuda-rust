/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Provably safe 1D vector addition (`c[i] = a[i] + b[i]`) using Lean 4 safety proofs.
//!
//! Build and run with:
//!   cargo oxide run safe_vecadd

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchContractError};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_contract, thread, verify};

/// Provably safe 1D vector addition module.
#[cuda_module]
mod kernels {
    use super::*;

    /// Provably safe vector addition kernel checked by Lean 4 at `PostMem2Reg`:
    /// - `no_oob_idx(a, b, c)`: every slice access stays within `0..len`
    /// - `disjoint_writes(c)`: distinct 1D threads never write the same slot
    /// - `writes_all_elems(c)`: every element in `0..c.len()` has a writer
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= c.len(), b.len() >= c.len()),
    )]
    #[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    /// Plain `&mut [f32]` vector addition kernel checked by Lean 4 at `PostMem2Reg`
    /// without `DisjointSlice` or `ThreadIndex` wrapper types:
    /// - `no_oob_idx(a, b, c)` + `disjoint_writes(c)` prove per-thread exclusivity on `c`
    ///   and elide all 3 slice bounds checks (`a[idx]`, `b[idx]`, `c[idx]`).
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= c.len(), b.len() >= c.len()),
    )]
    #[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
    #[kernel]
    pub fn vecadd_mut_slice(a: &[f32], b: &[f32], c: &mut [f32]) {
        let idx = (thread::blockIdx_x() as usize) * (thread::blockDim_x() as usize)
            + (thread::threadIdx_x() as usize);
        if idx < c.len() {
            c[idx] = a[idx] + b[idx];
        }
    }

    /// Reverse-indexed `&mut [f32]` vector addition (`c[c.len() - 1 - idx] = a[idx] + b[idx]`):
    /// - Lean 4 proves `no_oob_idx(a, b, c)`, `disjoint_writes(c)`, and `writes_all_elems(c)`
    ///   using the involutive witness thread `c.len() - 1 - elemIdx`, eliding all 3 bounds checks.
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= c.len(), b.len() >= c.len()),
    )]
    #[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
    #[kernel]
    pub fn vecadd_reverse(a: &[f32], b: &[f32], c: &mut [f32]) {
        let idx = thread::index_1d().get();
        if idx < c.len() {
            c[c.len() - 1 - idx] = a[idx] + b[idx];
        }
    }

    /// Equal-length `requires = (a.len() == c.len(), b.len() == c.len())` with
    /// scalar float math intrinsics (`a[idx].sin() + b[idx].cos()`):
    /// - `cuda-macros` forwards both directions of `==` to Lean 4 while pure
    ///   float math intrinsics export as opaque `.unk` values.
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() == c.len(), b.len() == c.len()),
    )]
    #[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
    #[kernel]
    pub fn vecadd_trig_eq(a: &[f32], b: &[f32], c: &mut [f32]) {
        let idx = thread::index_1d().get();
        if idx < c.len() {
            c[idx] = a[idx].sin() + b[idx].cos();
        }
    }

    /// Scalar-extent `requires = (a.len() >= n, c.len() >= n)` with IEEE-754
    /// `.rn` float intrinsics (`add_rn_f32(mul_rn_f32(a[idx], scale), bias)`):
    /// - Proves `no_oob_idx(a, c)` and `disjoint_writes(c)` on oversized scratch
    ///   buffers when only the active prefix `0..n` is processed.
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= n, c.len() >= n),
    )]
    #[verify(no_oob_idx(a, c), disjoint_writes(c))]
    #[kernel]
    pub fn scale_bias_prefix(a: &[f32], c: &mut [f32], n: u32, scale: f32, bias: f32) {
        let idx = thread::index_1d().get();
        if idx < n as usize {
            c[idx] = add_rn_f32(mul_rn_f32(a[idx], scale), bias);
        }
    }

    /// Scaled slice-extent `requires = (x.len() >= 4 * out.len())`:
    /// - Proves `4 * idx + 3 < x.len()` (and 64-bit non-overflow) for every
    ///   `idx < out.len()` without needing `u32_coordinates`.
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= 4 * out.len()),
    )]
    #[verify(no_oob_idx(x, out), disjoint_writes(out), writes_all_elems(out))]
    #[kernel]
    pub fn gather_quad_last(x: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = x[4 * i + 3];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provably Safe Vector Addition (`safe_vecadd`) ===\n");
    println!("Contract & Lean 4 Proof Summary:");
    println!("  • Launch contract: domain = 1, block = (256, 1, 1)");
    println!("  • Host requires:   a.len() >= c.len(), b.len() >= c.len()");
    println!("  • Lean 4 proved:   no_oob_idx(a, b, c)      [CudaOxide.Contracts.NoOobIdx]");
    println!("                     disjoint_writes(c)       [CudaOxide.Contracts.DisjointWrites]");
    println!("                     writes_all_elems(c)      [CudaOxide.Contracts.WritesAllElems]");
    println!("  • Codegen payoff:  a[idx] & b[idx] bounds checks elided before PTX lowering\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1000;
    const BLOCK_SIZE: u32 = 256;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut c_mut_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut c_rev_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut c_trig_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut c_scratch_dev = DeviceBuffer::<f32>::zeroed(&stream, N + 128)?;
    let x_quad_host: Vec<f32> = (0..(4 * N)).map(|i| i as f32).collect();
    let x_quad_dev = DeviceBuffer::from_host(&stream, &x_quad_host)?;
    let mut out_quad_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this binary embeds the single `safe_vecadd` PTX module.
    let module = unsafe { kernels::load(&ctx)? };
    let blocks = (N as u32).div_ceil(BLOCK_SIZE);
    let cfg = LaunchConfig1D::new(blocks, BLOCK_SIZE, 0);
    let launch = module.prepare_vecadd(cfg)?;
    let launch_mut = module.prepare_vecadd_mut_slice(cfg)?;
    let launch_rev = module.prepare_vecadd_reverse(cfg)?;
    let launch_trig = module.prepare_vecadd_trig_eq(cfg)?;
    let launch_scale = module.prepare_scale_bias_prefix(cfg)?;
    let launch_quad = module.prepare_gather_quad_last(cfg)?;

    // Safe, contract-checked launches:
    module.vecadd(&stream, &launch, &a_dev, &b_dev, &mut c_dev)?;
    module.vecadd_mut_slice(&stream, &launch_mut, &a_dev, &b_dev, &mut c_mut_dev)?;
    module.vecadd_reverse(&stream, &launch_rev, &a_dev, &b_dev, &mut c_rev_dev)?;
    module.vecadd_trig_eq(&stream, &launch_trig, &a_dev, &b_dev, &mut c_trig_dev)?;
    module.scale_bias_prefix(
        &stream,
        &launch_scale,
        &a_dev,
        &mut c_scratch_dev,
        (N - 64) as u32,
        2.5,
        1.0,
    )?;
    module.gather_quad_last(&stream, &launch_quad, &x_quad_dev, &mut out_quad_dev)?;

    // Host `requires` guard rejects an out-of-range scalar count (`n > a_dev.len()`) before launch:
    let bad_scale = module.scale_bias_prefix(
        &stream,
        &launch_scale,
        &a_dev,
        &mut c_scratch_dev,
        (N + 1) as u32,
        2.5,
        1.0,
    );
    assert!(
        matches!(
            bad_scale,
            Err(LaunchContractError::SizeRequirementViolated { .. })
        ),
        "expected SizeRequirementViolated when n > a.len(), got {bad_scale:?}"
    );

    let c_host = c_dev.to_host_vec(&stream)?;
    let c_mut_host = c_mut_dev.to_host_vec(&stream)?;
    let c_rev_host = c_rev_dev.to_host_vec(&stream)?;
    let c_trig_host = c_trig_dev.to_host_vec(&stream)?;
    let c_scratch_host = c_scratch_dev.to_host_vec(&stream)?;
    let out_quad_host = out_quad_dev.to_host_vec(&stream)?;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        let expected_trig = a_host[i].sin() + b_host[i].cos();
        let expected_scale = if i < N - 64 {
            a_host[i] * 2.5 + 1.0
        } else {
            0.0
        };
        let expected_quad = x_quad_host[4 * i + 3];
        assert!(
            (c_host[i] - expected).abs() <= 1e-5
                && c_host[i] == c_mut_host[i]
                && c_rev_host[N - 1 - i] == c_host[i]
                && (c_trig_host[i] - expected_trig).abs() <= 1e-4
                && (c_scratch_host[i] - expected_scale).abs() <= 1e-5
                && out_quad_host[i] == expected_quad,
            "mismatch at index {i}: expected {expected} (trig {expected_trig}), got {} / {} / rev {} / trig {} / scale {} / quad {}",
            c_host[i],
            c_mut_host[i],
            c_rev_host[N - 1 - i],
            c_trig_host[i],
            c_scratch_host[i],
            out_quad_host[i],
        );
    }

    println!(
        "Verified launch succeeded (`DisjointSlice`, `&mut [f32]`, reverse, `==` + trig, scalar `n` + `.rn`, and `4 * out.len()`): all {N} elements match!"
    );
    Ok(())
}
