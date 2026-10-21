/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared memory is accessed by thread-derived index, not an iterator.
#![allow(clippy::needless_range_loop)]

//! Unified Shared Memory Example
//!
//! Demonstrates SharedArray for block-level cooperation:
//! 1. shared_test - single SharedArray: loads data, syncs, reads neighbor
//! 2. shared_dual - two SharedArrays: loads from a and b, syncs, adds neighbors
//! 3. shared_u64_align1 - a `u64` array requested at 1-byte alignment is
//!    still declared at 8 bytes
//! 4. shared_v4_natural - a `#[repr(C, align(16))]` element at natural
//!    alignment is declared at 16 bytes
//!
//! The last two are regressions for the storage alignment contract: every
//! shared array is declared at the larger of the requested `ALIGN` and the
//! element's ABI alignment. The host checks the kernel values and the
//! `.align` declarations in the generated PTX.
//!
//! Build and run with:
//!   cargo oxide run sharedmem

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

/// A 16-byte aligned aggregate. The LLVM struct type cannot carry the
/// `align(16)` raise; it lives on the Rust type and must reach the storage
/// declaration.
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct V4(pub [f32; 4]);

/// Marker value written by the `u64` alignment kernel.
const U64_BASE: u64 = 0x1122_3344_5566_7788;

/// Threads in the block of the two alignment kernels.
const ALIGN_THREADS: usize = 32;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test kernel for shared memory with single array
    #[kernel]
    pub fn shared_test(data: &[f32], mut out: DisjointSlice<f32>) {
        static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Write to shared memory
        unsafe {
            TILE[tid] = data[gid];
        }

        thread::sync_threads();

        // Read from shared memory (neighbor)
        unsafe {
            let neighbor_idx = (tid + 1) % 256;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE[neighbor_idx];
            }
        }
    }

    /// Test kernel with TWO shared arrays - tests multiple shared allocations
    #[kernel]
    pub fn shared_dual(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        // Two separate shared memory allocations
        static mut TILE_A: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        // Load both arrays into shared memory
        unsafe {
            TILE_A[tid] = a[gid];
            TILE_B[tid] = b[gid];
        }

        thread::sync_threads();

        // Read neighbor's value from both tiles and add them
        unsafe {
            let neighbor_idx = (tid + 1) % 256;
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = TILE_A[neighbor_idx] + TILE_B[neighbor_idx];
            }
        }
    }

    /// `DATA` asks for 1-byte alignment but holds `u64` values.
    ///
    /// It is touched before `PAD` on purpose. Shared globals are inserted at
    /// the front of the module, so the first array a kernel references lands
    /// last in the PTX, right behind the 33-byte pad. Declared at 1 byte,
    /// ptxas packs it at an odd offset and `st.shared.b64` faults with
    /// `misaligned address`. Declared at the `u64` ABI alignment, it is
    /// placed on an 8-byte boundary.
    #[kernel]
    pub fn shared_u64_align1(mut out: DisjointSlice<u64>) {
        static mut DATA: SharedArray<u64, 2, 1> = SharedArray::UNINIT;
        static mut PAD: SharedArray<u8, 33> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        unsafe {
            if tid < 2 {
                DATA[tid] = U64_BASE + tid as u64;
            }
            if tid < 33 {
                PAD[tid] = tid as u8;
            }
        }

        thread::sync_threads();

        unsafe {
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = DATA[tid % 2] + PAD[tid % 33] as u64;
            }
        }
    }

    /// `VS` holds a `#[repr(C, align(16))]` element at natural alignment.
    ///
    /// Same ordering as `shared_u64_align1`: `VS` is touched before the
    /// 8-byte `PAD`, so it lands behind the pad in the PTX. Its loads and
    /// stores are 16-byte vector accesses, so the storage must be declared
    /// at 16 bytes even though no `ALIGN` was requested.
    #[kernel]
    pub fn shared_v4_natural(mut out: DisjointSlice<f32>) {
        static mut VS: SharedArray<V4, 4> = SharedArray::UNINIT;
        static mut PAD: SharedArray<u64, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let t = tid as f32;
        unsafe {
            if tid < 4 {
                VS[tid] = V4([t, t + 0.5, -t, 2.0 * t]);
            }
            if tid == 0 {
                PAD[0] = 3;
            }
        }

        thread::sync_threads();

        unsafe {
            let v = VS[tid % 4];
            if let Some(out_elem) = out.get_mut(thread::index_1d()) {
                *out_elem = v.0[0] + v.0[1] + v.0[2] + v.0[3] + PAD[0] as f32;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

/// Check the two declarations the alignment kernels rely on in the PTX that
/// `cargo oxide` writes next to this manifest. The byte sizes identify the
/// arrays: `DATA` is 16 bytes and `VS` is 64 bytes.
fn verify_shared_alignment_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("sharedmem.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    for (align, bytes, array) in [
        (8, 16, "SharedArray<u64, 2, 1>"),
        (16, 64, "SharedArray<V4, 4>"),
    ] {
        let directive = format!(".shared .align {align} .b8 ");
        let extent = format!("[{bytes}];");
        let count = ptx
            .lines()
            .filter(|line| line.contains(&directive) && line.trim_end().ends_with(&extent))
            .count();
        if count != 1 {
            return Err(format!(
                "{}: expected one `{directive}...{extent}` declaration for {array}, found {count}",
                path.display()
            )
            .into());
        }
    }
    Ok(())
}

fn main() {
    println!("=== Unified Shared Memory Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test size - must match TILE size (256 elements)
    const N: usize = 256;

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // Launch config for shared memory kernels
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: Single SharedArray =====
    println!("=== Test 1: Single SharedArray ===");
    {
        let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

        println!("Input data[0..5] = {:?}", &data_host[0..5]);

        let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.shared_test((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = data[(i + 1) % 256]
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = data_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (data[{}]), got {}",
                    i, expected, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Single SharedArray: correct neighbor read\n");
    }

    // ===== Test 2: Dual SharedArray =====
    println!("=== Test 2: Dual SharedArray ===");
    {
        let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..N).map(|i| (i + 100) as f32).collect();

        println!("Input a[0..5] = {:?}", &a_host[0..5]);
        println!("Input b[0..5] = {:?}", &b_host[0..5]);

        let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.shared_dual((stream).as_ref(), cfg, &a_dev, &b_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..5] = {:?}", &out_result[0..5]);

        // Verify: out[i] = a[(i+1)%256] + b[(i+1)%256]
        for i in 0..N {
            let neighbor_idx = (i + 1) % N;
            let expected = a_host[neighbor_idx] + b_host[neighbor_idx];
            if (out_result[i] - expected).abs() > 1e-5 {
                eprintln!(
                    "Mismatch at {}: expected {} (a[{}]+b[{}]), got {}",
                    i, expected, neighbor_idx, neighbor_idx, out_result[i]
                );
                std::process::exit(1);
            }
        }
        println!("✓ Dual SharedArray: correct neighbor sum from both tiles\n");
    }

    // One block of ALIGN_THREADS threads for the two alignment kernels.
    let align_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (ALIGN_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 3: u64 array requested at 1-byte alignment =====
    println!("=== Test 3: SharedArray<u64, 2, 1> behind a byte pad ===");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, ALIGN_THREADS).unwrap();

        // SAFETY: one block of ALIGN_THREADS threads; the buffer holds one element per thread.
        unsafe { module.shared_u64_align1(stream.as_ref(), align_cfg, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..4] = {:x?}", &out_result[0..4]);

        // Verify: out[i] = DATA[i % 2] + PAD[i % 33]
        for (i, &got) in out_result.iter().enumerate() {
            let expected = U64_BASE + (i % 2) as u64 + (i % 33) as u64;
            if got != expected {
                eprintln!("Mismatch at {i}: expected {expected:#x}, got {got:#x}");
                std::process::exit(1);
            }
        }
        println!("✓ u64 storage requested at 1 byte reads back intact\n");
    }

    // ===== Test 4: repr(align(16)) element at natural alignment =====
    println!("=== Test 4: SharedArray<V4, 4> behind a u64 pad ===");
    {
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, ALIGN_THREADS).unwrap();

        // SAFETY: one block of ALIGN_THREADS threads; the buffer holds one element per thread.
        unsafe { module.shared_v4_natural(stream.as_ref(), align_cfg, &mut out_dev) }
            .expect("Kernel launch failed");

        let out_result = out_dev.to_host_vec(&stream).unwrap();
        println!("Output out[0..4] = {:?}", &out_result[0..4]);

        // Verify: VS[s] sums to 3s + 0.5 for s = i % 4, plus PAD[0] = 3.
        for (i, &got) in out_result.iter().enumerate() {
            let s = (i % 4) as f32;
            let expected = 3.0 * s + 0.5 + 3.0;
            if got != expected {
                eprintln!("Mismatch at {i}: expected {expected}, got {got}");
                std::process::exit(1);
            }
        }
        println!("✓ align(16) element at natural alignment reads back intact\n");
    }

    // ===== Test 5: PTX declarations =====
    println!("=== Test 5: PTX storage declarations ===");
    if let Err(err) = verify_shared_alignment_ptx() {
        eprintln!("PTX declaration check failed: {err}");
        std::process::exit(1);
    }
    println!("✓ PTX declares the u64 array at .align 8 and the V4 array at .align 16\n");

    println!("✓ SUCCESS: All shared memory tests passed!");
}
