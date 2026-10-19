// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Volatile device memory access through `core::ptr::{read,write}_volatile`.
//!
//! Run: `cargo oxide run volatile_memory`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const MASK: u32 = 0x5a5a_5a5a;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn round_trip(input: &[u32], mut scratch: DisjointSlice<u32>, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let Some(slot) = scratch.get_mut(idx) else {
            return;
        };

        let ptr = slot as *mut u32;
        unsafe {
            core::ptr::write_volatile(ptr, input[i].wrapping_mul(3).wrapping_add(11));
            let first = core::ptr::read_volatile(ptr as *const u32);
            core::ptr::write_volatile(ptr, first ^ MASK);
            let second = core::ptr::read_volatile(ptr as *const u32);

            if let Some(out_slot) = out.get_mut(thread::index_1d()) {
                *out_slot = second;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 128;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let input: Vec<u32> = (0..N as u32)
        .map(|i| i.wrapping_mul(17).wrapping_add(5))
        .collect();
    let expected: Vec<u32> = input
        .iter()
        .map(|&x| x.wrapping_mul(3).wrapping_add(11) ^ MASK)
        .collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input)?;
    let mut scratch_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.round_trip(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &mut scratch_dev,
            &mut out_dev,
        )
    }?;

    let out = out_dev.to_host_vec(&stream)?;
    assert_eq!(out, expected, "volatile_memory: kernel output mismatch");

    println!("SUCCESS: volatile load/store round trip matched {N} values");
    Ok(())
}
