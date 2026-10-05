/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `core::ptr::copy_nonoverlapping` lowering.
//!
//! `copy_nonoverlapping` reaches MIR as `StatementKind::Intrinsic(
//! NonDivergingIntrinsic::CopyNonOverlapping(_))`. The importer lowers it to a
//! `mir.memcpy` op, and mir-lower emits an `llvm.memcpy` intrinsic with the
//! element count scaled to bytes for the pointee type.
//!
//! This exercises the branches the scaling actually has:
//!   - one element per thread (count == 1),
//!   - many elements in one call (count > 1, so bytes = count * size_of),
//!   - a `u8` copy (size_of == 1, the no-scaling fast path),
//!   - a copy whose destination is shared memory.
//!
//! Usage:
//!   cargo oxide run copy_nonoverlapping

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// One `u32` per thread (count == 1).
    #[kernel]
    pub fn copy_each(input: &[u32], mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            unsafe {
                let src = input.as_ptr().add(idx.get());
                core::ptr::copy_nonoverlapping(src, slot as *mut u32, 1);
            }
        }
    }

    /// Whole buffer in one call (count > 1, so bytes = count * 4). Thread 0
    /// copies all `len` elements; the byte-scaling branch must be correct or
    /// the tail is left untouched.
    #[kernel]
    pub fn copy_block_u32(input: &[u32], mut out: DisjointSlice<u32>, len: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), len);
            }
        }
    }

    /// `u8` copy (size_of == 1, the no-scaling fast path).
    #[kernel]
    pub fn copy_block_u8(input: &[u8], mut out: DisjointSlice<u8>, len: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(input.as_ptr(), out.as_mut_ptr(), len);
            }
        }
    }

    /// Destination in shared memory. The raw pointer is normalized to the
    /// generic address space when it is formed, so this stays a `p0.p0` copy,
    /// but it keeps a copy that touches shared memory under test.
    #[kernel]
    pub fn copy_into_shared(input: &[u32], mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 128> = SharedArray::UNINIT;
        let i = thread::index_1d().get();
        if i >= 128 {
            return;
        }
        unsafe {
            let src = input.as_ptr().add(i);
            let dst = (core::ptr::addr_of_mut!(SMEM) as *mut u32).add(i);
            core::ptr::copy_nonoverlapping(src, dst, 1);
            if let Some(slot) = out.get_mut(thread::index_1d()) {
                *slot = (core::ptr::addr_of!(SMEM) as *const u32).add(i).read();
            }
        }
    }
}

fn main() {
    println!("=== copy_nonoverlapping ===");

    const N: usize = 128;
    let input_u32: Vec<u32> = (0..N as u32)
        .map(|i| i.wrapping_mul(17) ^ 0x55aa_1234)
        .collect();
    let input_u8: Vec<u8> = (0..N)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
        .collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let in_u32 = DeviceBuffer::from_host(&stream, &input_u32).unwrap();
    let in_u8 = DeviceBuffer::from_host(&stream, &input_u8).unwrap();
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // count == 1 per thread.
    let mut out_each = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.copy_each(&stream, cfg, &in_u32, &mut out_each) }.expect("copy_each launch");
    assert_eq!(
        out_each.to_host_vec(&stream).unwrap(),
        input_u32,
        "copy_each"
    );

    // count == N in one call (byte scaling).
    let mut out_block = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.copy_block_u32(&stream, cfg, &in_u32, &mut out_block, N) }
        .expect("copy_block_u32 launch");
    assert_eq!(
        out_block.to_host_vec(&stream).unwrap(),
        input_u32,
        "copy_block_u32"
    );

    // u8 (no scaling).
    let mut out_u8 = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.copy_block_u8(&stream, cfg, &in_u8, &mut out_u8, N) }
        .expect("copy_block_u8 launch");
    assert_eq!(
        out_u8.to_host_vec(&stream).unwrap(),
        input_u8,
        "copy_block_u8"
    );

    // shared destination.
    let mut out_shared = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.copy_into_shared(&stream, cfg, &in_u32, &mut out_shared) }
        .expect("copy_into_shared launch");
    assert_eq!(
        out_shared.to_host_vec(&stream).unwrap(),
        input_u32,
        "copy_into_shared"
    );

    println!("PASS: copy_nonoverlapping (per-element, block u32, block u8, shared dest)");
}
