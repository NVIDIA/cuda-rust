/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::cache::{load_l2_only, load_streaming, store_l2_only, store_streaming};
use cuda_device::cusimd::{Float2, Float4, Uint2, Uint4};
use cuda_device::{DisjointSlice, cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    // --- Scalar Typed Test Kernels ---

    #[kernel]
    pub fn test_u8_kernel(input: &[u8], mut output: DisjointSlice<u8>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_add(10));
            }
        }
    }

    #[kernel]
    pub fn test_i8_kernel(input: &[i8], mut output: DisjointSlice<i8>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_sub(5));
            }
        }
    }

    #[kernel]
    pub fn test_u16_kernel(input: &[u16], mut output: DisjointSlice<u16>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_add(100));
            }
        }
    }

    #[kernel]
    pub fn test_i16_kernel(input: &[i16], mut output: DisjointSlice<i16>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_sub(50));
            }
        }
    }

    #[kernel]
    pub fn test_u32_kernel(input: &[u32], mut output: DisjointSlice<u32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_add(1000));
            }
        }
    }

    #[kernel]
    pub fn test_i32_kernel(input: &[i32], mut output: DisjointSlice<i32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_sub(500));
            }
        }
    }

    #[kernel]
    pub fn test_u64_kernel(input: &[u64], mut output: DisjointSlice<u64>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_add(100_000));
            }
        }
    }

    #[kernel]
    pub fn test_i64_kernel(input: &[i64], mut output: DisjointSlice<i64>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val.wrapping_sub(50_000));
            }
        }
    }

    #[kernel]
    pub fn test_f32_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val * 2.0 + 1.0);
            }
        }
    }

    #[kernel]
    pub fn test_f64_kernel(input: &[f64], mut output: DisjointSlice<f64>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                store_streaming(slot, val * 3.0 + 2.0);
            }
        }
    }

    // --- Vector Test Kernels (Vectorized 64-bit / 128-bit memory transactions) ---

    #[kernel]
    pub fn test_float2_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((_slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            // Thread i processes Float2 pair at offset i * 2
            let in_ptr = input.as_ptr().cast::<Float2>();
            let out_ptr = output.as_mut_ptr().cast::<Float2>();
            unsafe {
                let val = load_streaming(in_ptr.add(i));
                let res = Float2::new([val.x() + 1.0, val.y() + 2.0]);
                store_streaming(out_ptr.add(i), res);
            }
        }
    }

    #[kernel]
    pub fn test_float4_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((_slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            // Thread i processes Float4 quad at offset i * 4
            let in_ptr = input.as_ptr().cast::<Float4>();
            let out_ptr = output.as_mut_ptr().cast::<Float4>();
            unsafe {
                let val = load_streaming(in_ptr.add(i));
                let res = Float4::new([val.x() + 1.0, val.y() + 2.0, val.z() + 3.0, val.w() + 4.0]);
                store_streaming(out_ptr.add(i), res);
            }
        }
    }

    #[kernel]
    pub fn test_uint2_kernel(input: &[u32], mut output: DisjointSlice<u32>) {
        if let Some((_slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            let in_ptr = input.as_ptr().cast::<Uint2>();
            let out_ptr = output.as_mut_ptr().cast::<Uint2>();
            unsafe {
                let val = load_l2_only(in_ptr.add(i));
                let res = Uint2::new([val.x().wrapping_add(10), val.y().wrapping_add(20)]);
                store_l2_only(out_ptr.add(i), res);
            }
        }
    }

    #[kernel]
    pub fn test_uint4_kernel(input: &[u32], mut output: DisjointSlice<u32>) {
        if let Some((_slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            let in_ptr = input.as_ptr().cast::<Uint4>();
            let out_ptr = output.as_mut_ptr().cast::<Uint4>();
            unsafe {
                let val = load_l2_only(in_ptr.add(i));
                let res = Uint4::new([
                    val.x().wrapping_add(10),
                    val.y().wrapping_add(20),
                    val.z().wrapping_add(30),
                    val.w().wrapping_add(40),
                ]);
                store_l2_only(out_ptr.add(i), res);
            }
        }
    }

    // --- Benchmarking Sweep Kernels ---

    #[kernel]
    pub fn kernel_baseline_streaming(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            let val = input[i];
            *slot = val * 2.0 + 1.0;
        }
    }

    #[kernel]
    pub fn kernel_streaming_cs(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_streaming(input.as_ptr().add(i));
                let res = val * 2.0 + 1.0;
                store_streaming(slot, res);
            }
        }
    }

    #[kernel]
    pub fn kernel_streaming_cg(input: &[f32], mut output: DisjointSlice<f32>) {
        if let Some((slot, idx)) = output.get_mut_indexed() {
            let i = idx.get();
            unsafe {
                let val = load_l2_only(input.as_ptr().add(i));
                let res = val * 2.0 + 1.0;
                store_l2_only(slot, res);
            }
        }
    }
}

fn main() {
    println!("=== Comprehensive Multi-DataType Cache Qualifier Hardware Tests ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const N: usize = 1024;
    let launch_config = LaunchConfig::for_num_elems(N as u32);

    // 1. u8
    {
        let host_in = vec![5u8; N];
        let mut dev_in = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_u8_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 15u8));
        println!("✓ u8 test: PASSED");
    }

    // 2. i8
    {
        let host_in = vec![20i8; N];
        let mut dev_in = DeviceBuffer::<i8>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<i8>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_i8_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 15i8));
        println!("✓ i8 test: PASSED");
    }

    // 3. u16
    {
        let host_in = vec![300u16; N];
        let mut dev_in = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<u16>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_u16_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 400u16));
        println!("✓ u16 test: PASSED");
    }

    // 4. i16
    {
        let host_in = vec![150i16; N];
        let mut dev_in = DeviceBuffer::<i16>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<i16>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_i16_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 100i16));
        println!("✓ i16 test: PASSED");
    }

    // 5. u32
    {
        let host_in = vec![5000u32; N];
        let mut dev_in = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_u32_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 6000u32));
        println!("✓ u32 test: PASSED");
    }

    // 6. i32
    {
        let host_in = vec![2500i32; N];
        let mut dev_in = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_i32_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 2000i32));
        println!("✓ i32 test: PASSED");
    }

    // 7. u64
    {
        let host_in = vec![1_000_000u64; N];
        let mut dev_in = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_u64_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 1_100_000u64));
        println!("✓ u64 test: PASSED");
    }

    // 8. i64
    {
        let host_in = vec![500_000i64; N];
        let mut dev_in = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_i64_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| x == 450_000i64));
        println!("✓ i64 test: PASSED");
    }

    // 9. f32
    {
        let host_in = vec![3.5f32; N];
        let mut dev_in = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_f32_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| (x - 8.0f32).abs() < 1e-5));
        println!("✓ f32 test: PASSED");
    }

    // 10. f64
    {
        let host_in = vec![4.0f64; N];
        let mut dev_in = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();
        let mut dev_out = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_f64_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        assert!(host_out.iter().all(|&x| (x - 14.0f64).abs() < 1e-9));
        println!("✓ f64 test: PASSED");
    }

    // 11. Float2 (Vector2 load_streaming / store_streaming)
    {
        let mut host_in = Vec::with_capacity(N * 2);
        for _ in 0..N {
            host_in.push(1.0f32);
            host_in.push(2.0f32);
        }
        let mut dev_in = DeviceBuffer::<f32>::zeroed(&stream, N * 2).unwrap();
        let mut dev_out = DeviceBuffer::<f32>::zeroed(&stream, N * 2).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_float2_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        for chunk in host_out.chunks(2) {
            assert!((chunk[0] - 2.0f32).abs() < 1e-5);
            assert!((chunk[1] - 4.0f32).abs() < 1e-5);
        }
        println!("✓ Float2 (64-bit vector) test: PASSED");
    }

    // 12. Float4 (Vector4 load_streaming / store_streaming)
    {
        let mut host_in = Vec::with_capacity(N * 4);
        for _ in 0..N {
            host_in.push(1.0f32);
            host_in.push(2.0f32);
            host_in.push(3.0f32);
            host_in.push(4.0f32);
        }
        let mut dev_in = DeviceBuffer::<f32>::zeroed(&stream, N * 4).unwrap();
        let mut dev_out = DeviceBuffer::<f32>::zeroed(&stream, N * 4).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_float4_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        for chunk in host_out.chunks(4) {
            assert!((chunk[0] - 2.0f32).abs() < 1e-5);
            assert!((chunk[1] - 4.0f32).abs() < 1e-5);
            assert!((chunk[2] - 6.0f32).abs() < 1e-5);
            assert!((chunk[3] - 8.0f32).abs() < 1e-5);
        }
        println!("✓ Float4 (128-bit vector) test: PASSED");
    }

    // 13. Uint2 (Vector2 load_l2_only / store_l2_only)
    {
        let mut host_in = Vec::with_capacity(N * 2);
        for _ in 0..N {
            host_in.push(100u32);
            host_in.push(200u32);
        }
        let mut dev_in = DeviceBuffer::<u32>::zeroed(&stream, N * 2).unwrap();
        let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, N * 2).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_uint2_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        for chunk in host_out.chunks(2) {
            assert_eq!(chunk[0], 110u32);
            assert_eq!(chunk[1], 220u32);
        }
        println!("✓ Uint2 (64-bit vector) test: PASSED");
    }

    // 14. Uint4 (Vector4 load_l2_only / store_l2_only)
    {
        let mut host_in = Vec::with_capacity(N * 4);
        for _ in 0..N {
            host_in.push(10u32);
            host_in.push(20u32);
            host_in.push(30u32);
            host_in.push(40u32);
        }
        let mut dev_in = DeviceBuffer::<u32>::zeroed(&stream, N * 4).unwrap();
        let mut dev_out = DeviceBuffer::<u32>::zeroed(&stream, N * 4).unwrap();
        dev_in.copy_from_host(&stream, &host_in).unwrap();
        unsafe {
            module.test_uint4_kernel(&stream, launch_config, &dev_in, &mut dev_out).unwrap();
        }
        let host_out = dev_out.to_host_vec(&stream).unwrap();
        for chunk in host_out.chunks(4) {
            assert_eq!(chunk[0], 20u32);
            assert_eq!(chunk[1], 40u32);
            assert_eq!(chunk[2], 60u32);
            assert_eq!(chunk[3], 80u32);
        }
        println!("✓ Uint4 (128-bit vector) test: PASSED");
    }

    println!("\nALL DATA TYPES AND VECTOR TRANSACTIONS VERIFIED ON HARDWARE SUCCESSFULLY!");
}

