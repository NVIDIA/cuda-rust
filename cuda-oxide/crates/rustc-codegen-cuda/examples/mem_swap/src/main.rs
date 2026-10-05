/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-side regression coverage for memory swapping and replacement.
//!
//! `mem::swap` lowers to `core::intrinsics::typed_swap_nonoverlapping`, which
//! cuda-oxide handles as a load/load/store/store crossover.
//!
//! `mem::replace` follows the ordinary read/write replacement path on the pinned
//! Rust toolchain (`read_via_copy` + `write_via_move`, lowered before backend
//! codegen).
//!
//! This example also covers the public `core::ptr::swap_nonoverlapping` API.
//! That path is distinct from `mem::swap`: libcore performs a runtime byte-count
//! calculation and swaps non-overlapping regions in aligned chunks.
//!
//! Usage:
//!   cargo oxide run mem_swap
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run mem_swap

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct AlignedBlock {
        words: [u32; 4],
    }

    #[kernel]
    pub fn swap_kernel(a: &[i32], b: &[i32], mut out: DisjointSlice<i32>) {
        let t = thread::index_1d();
        let i = t.get();
        let mut x = a[i];
        let mut y = b[i];

        // `mem::swap` reaches `typed_swap_nonoverlapping`.
        core::mem::swap(&mut x, &mut y); // now x = b[i], y = a[i]

        // `mem::replace` exercises the ordinary read/write replacement path.
        let old = core::mem::replace(&mut y, 7); // old = a[i], y = 7

        if let Some(slot) = out.get_mut(t) {
            // Encode both: x (= b[i]) and old (= a[i]).
            *slot = x * 1000 + old;
        }
    }

    /// Swap two non-overlapping regions containing two 16-byte-aligned blocks.
    #[kernel]
    pub fn swap_nonoverlapping_kernel(seed: u32, mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let mut left = [
            AlignedBlock {
                words: [seed + 1, seed + 2, seed + 3, seed + 4],
            },
            AlignedBlock {
                words: [seed + 5, seed + 6, seed + 7, seed + 8],
            },
        ];
        let mut right = [
            AlignedBlock {
                words: [seed + 101, seed + 102, seed + 103, seed + 104],
            },
            AlignedBlock {
                words: [seed + 105, seed + 106, seed + 107, seed + 108],
            },
        ];

        unsafe {
            // SAFETY: `left` and `right` are separate live stack allocations.
            // Their ranges are non-overlapping, each base pointer is aligned for
            // `AlignedBlock`, and both contain exactly two initialized elements.
            core::ptr::swap_nonoverlapping(left.as_mut_ptr(), right.as_mut_ptr(), 2);

            // One active thread owns the complete output region. Keep
            // result export explicit so this regression does not depend on
            // array/slice iterator lowering.
            let ptr = out.as_mut_ptr();

            ptr.write(left[0].words[0]);
            ptr.add(1).write(left[0].words[1]);
            ptr.add(2).write(left[0].words[2]);
            ptr.add(3).write(left[0].words[3]);
            ptr.add(4).write(left[1].words[0]);
            ptr.add(5).write(left[1].words[1]);
            ptr.add(6).write(left[1].words[2]);
            ptr.add(7).write(left[1].words[3]);

            ptr.add(8).write(right[0].words[0]);
            ptr.add(9).write(right[0].words[1]);
            ptr.add(10).write(right[0].words[2]);
            ptr.add(11).write(right[0].words[3]);
            ptr.add(12).write(right[1].words[0]);
            ptr.add(13).write(right[1].words[1]);
            ptr.add(14).write(right[1].words[2]);
            ptr.add(15).write(right[1].words[3]);
        }
    }
}

const N: usize = 64;

fn main() {
    println!("=== core memory swap primitives on device ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    let a_init: Vec<i32> = (0..N as i32).collect(); // a[i] = i
    let b_init: Vec<i32> = (0..N as i32).map(|i| 100 + i).collect(); // b[i] = 100 + i
    let d_a = DeviceBuffer::from_host(&stream, &a_init).unwrap();
    let d_b = DeviceBuffer::from_host(&stream, &b_init).unwrap();
    let mut d_out = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.swap_kernel(stream.as_ref(), config, &d_a, &d_b, &mut d_out) }
        .expect("swap_kernel launch");

    let out = d_out.to_host_vec(&stream).unwrap();
    for (i, got) in out.iter().enumerate() {
        let want = (100 + i as i32) * 1000 + i as i32; // b[i]*1000 + a[i]
        assert_eq!(*got, want, "swap_kernel lane {i}");
    }
    println!("PASS: mem::swap");
    println!("PASS: mem::replace");

    const SEED: u32 = 17;
    const SWAP_WORDS: usize = 16;

    let mut swap_out = DeviceBuffer::<u32>::zeroed(&stream, SWAP_WORDS).unwrap();
    let single_thread = LaunchConfig::for_num_elems(1);

    // SAFETY: one thread is sufficient and the output buffer has all 16 words.
    unsafe {
        module.swap_nonoverlapping_kernel(stream.as_ref(), single_thread, SEED, &mut swap_out)
    }
    .expect("swap_nonoverlapping_kernel launch");

    let got = swap_out.to_host_vec(&stream).unwrap();
    let expected: Vec<u32> = (101..=108)
        .map(|offset| SEED + offset)
        .chain((1..=8).map(|offset| SEED + offset))
        .collect();

    assert_eq!(
        got, expected,
        "ptr::swap_nonoverlapping should exchange both aligned regions"
    );
    println!("PASS: ptr::swap_nonoverlapping (2 x 16-byte aligned blocks)");

    println!("PASS: mem_swap");
}
