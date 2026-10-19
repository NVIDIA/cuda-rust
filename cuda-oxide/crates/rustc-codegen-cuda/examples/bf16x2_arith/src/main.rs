// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end example exercising all eight packed bf16x2 arithmetic intrinsics.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::bf16x2::{
    abs_bf16x2, add_bf16x2, fma_relu_bf16x2, max_bf16x2, min_bf16x2, mul_bf16x2, neg_bf16x2,
    sub_bf16x2,
};
use cuda_device::convert::cvt_bf16x2_f32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Number of result slots written by the kernel (one per operation).
const NUM_OPS: usize = 8;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_bf16x2_arith(mut out: DisjointSlice<[u32; NUM_OPS]>) {
        let idx = thread::index_1d();
        if let Some(row) = out.get_mut(idx) {
            // Pack known f32 pairs into bf16x2.
            let a = cvt_bf16x2_f32(2.0, 4.0);
            let b = cvt_bf16x2_f32(3.0, 5.0);
            let neg_one = cvt_bf16x2_f32(-1.0, -1.0);
            let zero = cvt_bf16x2_f32(0.0, 0.0);

            // One thread owns one complete result row, so every write remains
            // within the element selected by its typed ThreadIndex witness.
            row[0] = add_bf16x2(a, b); // (5, 9)
            row[1] = sub_bf16x2(a, b); // (-1, -1)
            row[2] = mul_bf16x2(a, b); // (6, 20)
            row[3] = min_bf16x2(a, b); // (2, 4)
            row[4] = max_bf16x2(a, b); // (3, 5)
            let negated = neg_bf16x2(a);
            row[5] = negated; // (-2, -4)
            row[6] = abs_bf16x2(negated); // (2, 4)
            // relu(a * -1 + 0) = (0, 0)
            row[7] = fma_relu_bf16x2(a, neg_one, zero);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side bf16 helpers
// ---------------------------------------------------------------------------

fn unpack_bf16x2(packed: u32) -> (f32, f32) {
    let lo_bits = packed & 0xFFFF;
    let hi_bits = packed >> 16;
    let lo = f32::from_bits(lo_bits << 16); // bf16 is top 16 bits of f32
    let hi = f32::from_bits(hi_bits << 16);
    (lo, hi)
}

/// Approximate equality suitable for bf16 (~3 decimal digits of precision).
fn bf16_approx_eq(got: f32, expected: f32) -> bool {
    (got - expected).abs() < expected.abs() * 0.02 + 0.01
}

/// Verify a single result slot.  Returns `true` on success.
fn check(label: &str, packed: u32, expected_lo: f32, expected_hi: f32) -> bool {
    let (lo, hi) = unpack_bf16x2(packed);
    let ok = bf16_approx_eq(lo, expected_lo) && bf16_approx_eq(hi, expected_hi);
    if ok {
        println!("  {label}: ok  (lo={lo}, hi={hi})");
    } else {
        println!(
            "  {label}: FAIL  got ({lo}, {hi}), expected ({expected_lo}, {expected_hi})  [0x{packed:08x}]"
        );
    }
    ok
}

fn main() {
    println!("=== bf16x2_arith ===");

    let ctx = CudaContext::new(0).expect("CUDA init");

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!(
            "skipping: this all-operations example uses sm_90+ add/sub/mul (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut out = DeviceBuffer::<[u32; NUM_OPS]>::zeroed(&stream, 1).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_bf16x2_arith(&stream, LaunchConfig::for_num_elems(1), &mut out) }
        .expect("launch test_bf16x2_arith");

    let rows = out.to_host_vec(&stream).unwrap();
    assert_eq!(rows.len(), 1, "unexpected result-row count");
    let results = &rows[0];

    println!("verifying {NUM_OPS} operations:");

    let mut pass = true;
    pass &= check("add", results[0], 5.0, 9.0);
    pass &= check("sub", results[1], -1.0, -1.0);
    pass &= check("mul", results[2], 6.0, 20.0);
    pass &= check("min", results[3], 2.0, 4.0);
    pass &= check("max", results[4], 3.0, 5.0);
    pass &= check("neg", results[5], -2.0, -4.0);
    pass &= check("abs", results[6], 2.0, 4.0);
    pass &= check("fma_relu", results[7], 0.0, 0.0);

    if !pass {
        println!("FAIL: bf16x2_arith, one or more checks failed");
        std::process::exit(1);
    }
    println!("PASS: bf16x2_arith, all 8 packed bf16x2 operations verified on sm_{major}{minor}");
}
