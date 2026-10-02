/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Elementwise addition with a short final tile, in both f32 and f16.
//!
//! Each thread owns one contiguous tile (4 x f32 or 8 x f16, both 16
//! bytes). The kernel spells the same view flow as CuTeDSL:
//!
//! ```text
//! Tensor -> zipped_divide -> slice -> load / add / store
//! ```
//!
//! Full tiles use one vector load or store. The final, shorter tile uses an
//! ordinary Rust branch and scalar loop.
//!
//! PASS requires bit-exact correctness for both element types on the GPU.
//! Backend-specific branches add their own code-shape and artifact gates.

#![feature(f16)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Deliberately not divisible by the tiles: f32 tail of 3, f16 tail of 5.
const N_F32: usize = 1003;
const N_F16: usize = 2005;
const TILE_F32: usize = 4; // 16 bytes
const TILE_F16: usize = 8; // 16 bytes

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_f32(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let tid = thread::index_1d().get();
        let g_a = cute_rs::Tensor::from_slice(a).zipped_divide::<TILE_F32>();
        let g_b = cute_rs::Tensor::from_slice(b).zipped_divide::<TILE_F32>();
        let g_c = cute_rs::TensorMut::from_disjoint_slice(&mut c).zipped_divide::<TILE_F32>();

        let t_a = g_a.slice(tid);
        let t_b = g_b.slice(tid);
        let mut t_c = g_c.slice(tid);

        if t_a.is_full() {
            let a_values = unsafe { t_a.load() };
            let b_values = unsafe { t_b.load() };
            unsafe { t_c.store(a_values + b_values) };
        } else {
            // Ragged tail: plain Rust scalar loop, boundary thread only.
            let mut k = t_a.base();
            while k < a.len() {
                unsafe { t_c.store_linear(k, a[k] + b[k]) };
                k += 1;
            }
        }
    }

    #[kernel]
    pub fn add_f16(a: &[f16], b: &[f16], mut c: DisjointSlice<f16>) {
        let tid = thread::index_1d().get();
        let g_a = cute_rs::Tensor::from_slice(a).zipped_divide::<TILE_F16>();
        let g_b = cute_rs::Tensor::from_slice(b).zipped_divide::<TILE_F16>();
        let g_c = cute_rs::TensorMut::from_disjoint_slice(&mut c).zipped_divide::<TILE_F16>();

        let t_a = g_a.slice(tid);
        let t_b = g_b.slice(tid);
        let mut t_c = g_c.slice(tid);

        if t_a.is_full() {
            let a_values = unsafe { t_a.load() };
            let b_values = unsafe { t_b.load() };
            unsafe { t_c.store(a_values + b_values) };
        } else {
            let mut k = t_a.base();
            while k < a.len() {
                unsafe { t_c.store_linear(k, a[k] + b[k]) };
                k += 1;
            }
        }
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

fn main() {
    println!("=== CuTe-style elementwise add, ragged edges, f32 + f16 ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // f32
    let a32: Vec<f32> = (0..N_F32).map(|i| i as f32).collect();
    let b32: Vec<f32> = (0..N_F32).map(|i| (i * 2) as f32).collect();
    let a32_dev = DeviceBuffer::from_host(&stream, &a32).unwrap();
    let b32_dev = DeviceBuffer::from_host(&stream, &b32).unwrap();
    let mut c32_dev = DeviceBuffer::<f32>::zeroed(&stream, N_F32).unwrap();
    let threads32 = ceil_div(N_F32, TILE_F32) as u32;
    // SAFETY: each thread owns one disjoint tile (or the scalar tail).
    unsafe {
        module.add_f32(
            &stream,
            LaunchConfig::for_num_elems(threads32),
            &a32_dev,
            &b32_dev,
            &mut c32_dev,
        )
    }
    .expect("f32 kernel launch failed");
    let c32 = c32_dev.to_host_vec(&stream).unwrap();
    for i in 0..N_F32 {
        let expected = a32[i] + b32[i];
        assert_eq!(
            c32[i].to_bits(),
            expected.to_bits(),
            "f32 bit mismatch at {i}: expected {expected}, got {}",
            c32[i],
        );
    }
    println!(
        "f32 numeric check: all {N_F32} elements correct (tail of {})",
        N_F32 % TILE_F32
    );

    // f16
    let a16: Vec<f16> = (0..N_F16)
        .map(|i| ((i % 100) as f32 / 8.0) as f16)
        .collect();
    let b16: Vec<f16> = (0..N_F16).map(|i| ((i % 50) as f32 / 4.0) as f16).collect();
    let a16_dev = DeviceBuffer::from_host(&stream, &a16).unwrap();
    let b16_dev = DeviceBuffer::from_host(&stream, &b16).unwrap();
    let mut c16_dev = DeviceBuffer::<f16>::zeroed(&stream, N_F16).unwrap();
    let threads16 = ceil_div(N_F16, TILE_F16) as u32;
    // SAFETY: as above.
    unsafe {
        module.add_f16(
            &stream,
            LaunchConfig::for_num_elems(threads16),
            &a16_dev,
            &b16_dev,
            &mut c16_dev,
        )
    }
    .expect("f16 kernel launch failed");
    let c16 = c16_dev.to_host_vec(&stream).unwrap();
    for i in 0..N_F16 {
        let expected = a16[i] + b16[i];
        assert_eq!(
            c16[i].to_bits(),
            expected.to_bits(),
            "f16 bit mismatch at {i}: expected {}, got {}",
            expected as f32,
            c16[i] as f32,
        );
    }
    println!(
        "f16 numeric check: all {N_F16} elements correct (tail of {})",
        N_F16 % TILE_F16
    );

    println!("\n✓ SUCCESS: f32 and f16 elementwise add are bit-exact!");
}
