/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end packed f16x2 and bf16x2 global atomic-add example.
//!
//! The native bf16x2 instruction makes this combined example require sm_90.
//! Each thread adds `(low = 1, high = 2)`, and the host checks the returned
//! old values for each 16-bit half independently because PTX does not promise
//! that the two returned halves form one coherent 32-bit snapshot.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::atomic::{atom_add_bf16x2, atom_add_f16x2};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
use half::{bf16, f16};

const THREADS: u32 = 32;
// Packed as high-half `2.0`, low-half `1.0`.
const PACKED_F16_ONE_TWO: u32 = 0x4000_3c00;
const PACKED_BF16_ONE_TWO: u32 = 0x4000_3f80;
const COUNTERS: usize = 2;
const OLD_VALUES: usize = THREADS as usize;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_packed(base: *mut u32, old_f16: *mut u32, old_bf16: *mut u32) {
        let lane = thread::index_1d().get();
        if lane >= THREADS as usize {
            return;
        }

        unsafe {
            // SAFETY: The host passes two naturally aligned `u32` counters and
            // two disjoint, `THREADS`-element output buffers. Active threads
            // access the counters only through these device-scope atomics, and
            // each thread writes its returned values to its unique lane slot.
            let previous_f16 = atom_add_f16x2(base, PACKED_F16_ONE_TWO);
            let previous_bf16 = atom_add_bf16x2(base.add(1), PACKED_BF16_ONE_TWO);
            old_f16.add(lane).write(previous_f16);
            old_bf16.add(lane).write(previous_bf16);
        }
    }
}

fn unpack_f16x2(value: u32) -> (f32, f32) {
    (
        f16::from_bits(value as u16).to_f32(),
        f16::from_bits((value >> 16) as u16).to_f32(),
    )
}

fn unpack_bf16x2(value: u32) -> (f32, f32) {
    (
        bf16::from_bits(value as u16).to_f32(),
        bf16::from_bits((value >> 16) as u16).to_f32(),
    )
}

fn verify_old_values(label: &str, values: &[u32], encode: fn(f32) -> u16) {
    assert_eq!(values.len(), OLD_VALUES, "{label} old-value count");

    let mut low: Vec<_> = values.iter().map(|value| *value as u16).collect();
    let mut high: Vec<_> = values.iter().map(|value| (*value >> 16) as u16).collect();
    low.sort_unstable();
    high.sort_unstable();

    let expected_low: Vec<_> = (0..THREADS).map(|value| encode(value as f32)).collect();
    let expected_high: Vec<_> = (0..THREADS)
        .map(|value| encode((2 * value) as f32))
        .collect();

    // A packed atomic is two independently atomic 16-bit updates. The halves
    // of one returned `u32` need not come from one coherent old snapshot, so
    // validate the two old-value permutations independently.
    assert_eq!(low, expected_low, "{label} low-half old values");
    assert_eq!(high, expected_high, "{label} high-half old values");
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!(
            "skipping: native bf16x2 atomic add requires sm_90+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let counters = DeviceBuffer::<u32>::zeroed(&stream, COUNTERS).expect("allocate counters");
    let old_f16 = DeviceBuffer::<u32>::zeroed(&stream, OLD_VALUES).expect("allocate f16 results");
    let old_bf16 = DeviceBuffer::<u32>::zeroed(&stream, OLD_VALUES).expect("allocate bf16 results");
    let counters_ptr = counters.cu_deviceptr() as *mut u32;
    let old_f16_ptr = old_f16.cu_deviceptr() as *mut u32;
    let old_bf16_ptr = old_bf16.cu_deviceptr() as *mut u32;

    // SAFETY: the launch covers exactly THREADS lanes. `counters_ptr`
    // addresses two aligned counters, while `old_f16_ptr` and `old_bf16_ptr`
    // address disjoint THREADS-element output buffers. All three allocations
    // remain alive until the stream is synchronized.
    unsafe {
        module
            .add_packed(
                &stream,
                LaunchConfig::for_num_elems(THREADS),
                counters_ptr,
                old_f16_ptr,
                old_bf16_ptr,
            )
            .expect("launch add_packed");
    }
    stream.synchronize().expect("synchronize");

    let final_counters = counters.to_host_vec(&stream).expect("copy counters");
    let old_f16 = old_f16.to_host_vec(&stream).expect("copy f16 old values");
    let old_bf16 = old_bf16.to_host_vec(&stream).expect("copy bf16 old values");

    verify_old_values("f16x2", &old_f16, |value| f16::from_f32(value).to_bits());
    verify_old_values("bf16x2", &old_bf16, |value| bf16::from_f32(value).to_bits());

    let expected_low = THREADS as f32;
    let expected_high = (2 * THREADS) as f32;
    assert_eq!(
        unpack_f16x2(final_counters[0]),
        (expected_low, expected_high)
    );
    assert_eq!(
        unpack_bf16x2(final_counters[1]),
        (expected_low, expected_high)
    );
    println!(
        "PASS: independent old-value halves and final ({expected_low}, {expected_high}) counters verified on sm_{major}{minor}"
    );
}
