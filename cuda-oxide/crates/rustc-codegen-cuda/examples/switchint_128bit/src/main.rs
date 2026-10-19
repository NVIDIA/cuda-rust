/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `SwitchInt` on a 128-bit scrutinee.
//!
//! `match` on an integer reaches MIR as a `SwitchInt` terminator whose arm
//! values are `u128` bit patterns at the scrutinee's width. The importer
//! builds one comparison constant per arm at that width, which it used to do
//! through a checked `u128 -> u64` narrowing, so any arm above `u64::MAX`
//! failed with "SwitchInt value ... does not fit in 64 bits".
//!
//! The importer has two paths for this, and both are exercised here:
//!   - one arm plus a default, comparing against the scrutinee directly,
//!   - several arms, built as a chain of comparison blocks.
//!
//! Signedness is covered too, since the comparison constant takes the
//! discriminant's signedness and a negative `i128` arm sets the top bit.
//!
//! Every scrutinee is assembled from runtime halves so the match survives to
//! MIR instead of being folded at compile time, and the host checks the code
//! each thread wrote rather than only that codegen succeeded.
//!
//! Usage:
//!   cargo oxide run switchint_128bit

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel};

/// Arm values that need more than 64 bits.
const V1: u128 = 1 << 100;
const V2: u128 = (1 << 127) - 1;
const V3: u128 = u128::MAX;

/// A negative `i128` arm, so the comparison constant carries a set top bit.
const N1: i128 = -(1 << 100);

#[cuda_module]
mod kernels {
    use super::*;

    /// Several arms, so the importer builds a comparison chain.
    #[kernel]
    pub fn classify_u128(lo: &[u64], hi: &[u64], mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            let value = (u128::from(hi[i]) << 64) | u128::from(lo[i]);
            *slot = match value {
                V1 => 1,
                V2 => 2,
                V3 => 3,
                _ => 0,
            };
        }
    }

    /// One arm plus a default, so the importer compares against the
    /// scrutinee directly rather than building a chain.
    #[kernel]
    pub fn is_v1(lo: &[u64], hi: &[u64], mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            let value = (u128::from(hi[i]) << 64) | u128::from(lo[i]);
            *slot = match value {
                V1 => 7,
                _ => 0,
            };
        }
    }

    /// A signed scrutinee with a negative arm.
    #[kernel]
    pub fn classify_i128(lo: &[u64], hi: &[u64], mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            let value = ((u128::from(hi[i]) << 64) | u128::from(lo[i])) as i128;
            *slot = match value {
                N1 => 5,
                -1 => 6,
                _ => 0,
            };
        }
    }
}

/// Split a `u128` into the halves the kernels reassemble.
fn halves(value: u128) -> (u64, u64) {
    (value as u64, (value >> 64) as u64)
}

fn main() {
    println!("=== switchint_128bit ===");

    // One case per arm, plus values that must fall through to the default.
    // The last two differ from V1 in only the low or only the high half, so a
    // comparison that dropped either half would misclassify them.
    let cases: [u128; 7] = [V1, V2, V3, 0, 42, V1 | 1, V1 >> 1];
    let n = cases.len();

    let lo_host: Vec<u64> = cases.iter().map(|value| halves(*value).0).collect();
    let hi_host: Vec<u64> = cases.iter().map(|value| halves(*value).1).collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let lo = DeviceBuffer::from_host(&stream, &lo_host).unwrap();
    let hi = DeviceBuffer::from_host(&stream, &hi_host).unwrap();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.classify_u128(&stream, cfg, &lo, &hi, &mut out) }
        .expect("classify_u128 launch");
    assert_eq!(
        out.to_host_vec(&stream).unwrap(),
        vec![1, 2, 3, 0, 0, 0, 0],
        "classify_u128"
    );

    let mut out_single = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.is_v1(&stream, cfg, &lo, &hi, &mut out_single) }.expect("is_v1 launch");
    assert_eq!(
        out_single.to_host_vec(&stream).unwrap(),
        vec![7, 0, 0, 0, 0, 0, 0],
        "is_v1"
    );

    // Signed cases: N1, then -1, which is V3 reinterpreted.
    let signed: [i128; 4] = [N1, -1, 0, 1 << 100];
    let signed_lo: Vec<u64> = signed.iter().map(|v| halves(*v as u128).0).collect();
    let signed_hi: Vec<u64> = signed.iter().map(|v| halves(*v as u128).1).collect();
    let slo = DeviceBuffer::from_host(&stream, &signed_lo).unwrap();
    let shi = DeviceBuffer::from_host(&stream, &signed_hi).unwrap();
    let signed_cfg = LaunchConfig::for_num_elems(signed.len() as u32);

    let mut out_signed = DeviceBuffer::<u32>::zeroed(&stream, signed.len()).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.classify_i128(&stream, signed_cfg, &slo, &shi, &mut out_signed) }
        .expect("classify_i128 launch");
    assert_eq!(
        out_signed.to_host_vec(&stream).unwrap(),
        vec![5, 6, 0, 0],
        "classify_i128"
    );

    println!("PASS");
}
