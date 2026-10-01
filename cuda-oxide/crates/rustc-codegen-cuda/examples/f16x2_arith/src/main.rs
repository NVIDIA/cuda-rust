// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end example for packed f16x2 arithmetic intrinsics.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::f16x2::{
    abs_f16x2, add_f16x2, fma_f16x2, fma_ftz_f16x2, fma_ftz_relu_f16x2, fma_ftz_sat_f16x2,
    fma_relu_f16x2, fma_sat_f16x2, max_f16x2, min_f16x2, mul_f16x2, neg_f16x2, sub_f16x2,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const NUM_OPS: usize = 14;

// Packed lane pairs used by both the kernel and host oracle.
const A: u32 = 0x4400_4000; // (2, 4)
const B: u32 = 0x4500_4200; // (3, 5)
const C: u32 = 0x4980_4700; // (7, 11)
const NEG_ONE: u32 = 0xbc00_bc00; // (-1, -1)
// Two subnormal halves (2^-24 and 2^-23). Multiplying them by 1.0 keeps the
// product subnormal, which is what separates the ftz forms from the plain ones:
// ftz flushes a subnormal result to zero, the plain form preserves it.
const SUBNORMAL: u32 = 0x0002_0001;
const ONE: u32 = 0x3c00_3c00; // (1, 1)

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_f16x2_arith(mut out: DisjointSlice<[u32; NUM_OPS]>) {
        let idx = thread::index_1d();
        if let Some(row) = out.get_mut(idx) {
            row[0] = add_f16x2(A, B);
            row[1] = sub_f16x2(A, B);
            row[2] = mul_f16x2(A, B);
            row[3] = fma_f16x2(A, B, C);
            row[4] = min_f16x2(A, B);
            row[5] = max_f16x2(A, B);
            let negated = neg_f16x2(A);
            row[6] = negated;
            row[7] = abs_f16x2(negated);
            row[8] = fma_relu_f16x2(A, NEG_ONE, 0);
            // ftz differs from plain fma only when a subnormal is involved.
            row[9] = fma_f16x2(SUBNORMAL, ONE, 0);
            row[10] = fma_ftz_f16x2(SUBNORMAL, ONE, 0);
            // sat clamps the result into [0.0, 1.0]; 2*3+7 = 13 and 4*5+11 = 31.
            row[11] = fma_sat_f16x2(A, B, C);
            row[12] = fma_ftz_sat_f16x2(A, B, C);
            // ReLU keeps a positive subnormal; the ftz form flushes it first.
            row[13] = fma_ftz_relu_f16x2(SUBNORMAL, ONE, 0);
        }
    }
}

fn main() {
    println!("=== f16x2_arith ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 8 {
        println!(
            "skipping: min, max, and FMA with ReLU require sm_80+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut out = DeviceBuffer::<[u32; NUM_OPS]>::zeroed(&stream, 1).unwrap();

    // SAFETY: only index 0 reaches the one allocated result row.
    unsafe { module.test_f16x2_arith(&stream, LaunchConfig::for_num_elems(1), &mut out) }
        .expect("launch test_f16x2_arith");

    let rows = out.to_host_vec(&stream).unwrap();
    assert_eq!(rows.len(), 1, "unexpected result-row count");

    let expected = [
        ("add", 0x4880_4500),
        ("sub", 0xbc00_bc00),
        ("mul", 0x4d00_4600),
        ("fma", 0x4fc0_4a80),
        ("min", A),
        ("max", B),
        ("neg", 0xc400_c000),
        ("abs", A),
        ("fma_relu", 0x0000_0000),
        ("fma (subnormal preserved)", SUBNORMAL),
        ("fma_ftz (subnormal flushed)", 0x0000_0000),
        ("fma_sat", ONE),
        ("fma_ftz_sat", ONE),
        ("fma_ftz_relu (subnormal flushed)", 0x0000_0000),
    ];

    let mut passed = true;
    println!("verifying {NUM_OPS} operations:");
    for ((label, want), got) in expected.iter().zip(rows[0].iter()) {
        if got == want {
            println!("  {label}: ok  (0x{got:08x})");
        } else {
            println!("  {label}: FAIL  got 0x{got:08x}, expected 0x{want:08x}");
            passed = false;
        }
    }

    if !passed {
        println!("FAIL: f16x2_arith, one or more checks failed");
        std::process::exit(1);
    }
    println!(
        "PASS: f16x2_arith, all {NUM_OPS} packed f16x2 operations verified on sm_{major}{minor}"
    );
}
