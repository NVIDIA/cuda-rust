/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn inline_ptx_kernel(mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get() as u32;
            let rust_before = i.wrapping_add(3);
            let doubled: u32;
            let lane: u32;

            unsafe {
                ptx_asm!(
                    "add.u32 %0, %1, %1;",
                    out("=r") doubled,
                    in("r") rust_before,
                    options(register_only),
                );
                ptx_asm!("mov.u32 %0, %%laneid;", out("=r") lane);
                ptx_asm!("membar.gl;", clobber("memory"));
            }

            let rust_after = doubled.wrapping_sub(3).wrapping_add(lane);
            *slot = rust_after;
        }
    }

    #[kernel]
    pub fn inline_ptx_c_constraint_kernel(mut out: DisjointSlice<u64>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            const MODE: &[u8; 6] = b".wide\0";

            let x = idx.get() as u32 + 65_537;
            let y = idx.get() as u32 + 3;
            let product: u64;

            unsafe {
                ptx_asm!(
                    "mul%1.u32 %0, %2, %3;",
                    out("=l") product,
                    in("C") MODE,
                    in("r") x,
                    in("r") y,
                    options(register_only),
                );
            }

            *slot = product;
        }
    }

    /// Multi-output `ptx_asm!`: one asm block yields both the sum and the
    /// product of two thread-dependent values. The asymmetric data flow
    /// (sum != product) catches swapped register binding between the two
    /// `=r` outputs.
    #[kernel]
    pub fn inline_ptx_multi_out_kernel(
        mut sums: DisjointSlice<u32>,
        mut prods: DisjointSlice<u32>,
    ) {
        if let Some((sum_slot, idx)) = sums.get_mut_indexed()
            && let Some((prod_slot, _)) = prods.get_mut_indexed()
        {
            let i = idx.get() as u32;
            let x = i.wrapping_add(1);
            let y = i.wrapping_add(2);
            let sum: u32;
            let prod: u32;

            unsafe {
                ptx_asm!(
                    "add.u32 %0, %2, %3; mul.lo.u32 %1, %2, %3;",
                    out("=r") sum,
                    out("=r") prod,
                    in("r") x,
                    in("r") y,
                    options(register_only),
                );
            }

            *sum_slot = sum;
            *prod_slot = prod;
        }
    }

    /// Read-write `ptx_asm!`: the accumulator initializes `%0`, PTX updates
    /// the same operand, and the final value is written back to Rust.
    #[kernel]
    pub fn inline_ptx_inout_kernel(mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get() as u32;
            let mut accumulator = i.wrapping_add(10);
            let increment = i.wrapping_mul(3).wrapping_add(1);

            unsafe {
                ptx_asm!(
                    "add.u32 %0, %0, %1;",
                    inout("+r") accumulator,
                    in("r") increment,
                    options(register_only),
                );
            }

            *slot = accumulator;
        }
    }

    /// Multi-`inout` with mixed widths: one asm block updates a 32-bit
    /// counter (`+r`), a 64-bit accumulator (`+l`), and an `f32` scale
    /// (`+f`) in place, next to a plain `out`. Distinct update formulas per
    /// operand catch swapped tied-register bindings across widths.
    #[kernel]
    pub fn inline_ptx_multi_inout_kernel(
        mut counts: DisjointSlice<u32>,
        mut wides: DisjointSlice<u64>,
        mut scales: DisjointSlice<f32>,
        mut checks: DisjointSlice<u32>,
    ) {
        if let Some((count_slot, idx)) = counts.get_mut_indexed()
            && let Some((wide_slot, _)) = wides.get_mut_indexed()
            && let Some((scale_slot, _)) = scales.get_mut_indexed()
            && let Some((check_slot, _)) = checks.get_mut_indexed()
        {
            let i = idx.get() as u32;
            let mut count = i.wrapping_add(7);
            let mut wide = (i as u64).wrapping_mul(0x0000_0001_0000_0003);
            let mut scale = i as f32 * 0.5;
            let check: u32;

            unsafe {
                ptx_asm!(
                    "add.u32 %0, %0, %4;
                     add.u64 %1, %1, %5;
                     add.f32 %2, %2, %6;
                     mul.lo.u32 %3, %4, %4;",
                    inout("+r") count,
                    inout("+l") wide,
                    inout("+f") scale,
                    out("=r") check,
                    in("r") i,
                    in("l") 11u64,
                    in("f") 0.25f32,
                    options(register_only),
                );
            }

            *count_slot = count;
            *wide_slot = wide;
            *scale_slot = scale;
            *check_slot = check;
        }
    }
}

fn main() {
    println!("=== Inline PTX Example ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 128;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.inline_ptx_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
    }
    .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, got) in out.iter().copied().enumerate() {
        let expected = (i as u32 * 2) + 3 + (i as u32 % 32);
        if got != expected {
            eprintln!("Mismatch at {i}: expected {expected}, got {got}");
            std::process::exit(1);
        }
    }

    let mut sums_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut prods_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.inline_ptx_multi_out_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut sums_dev,
            &mut prods_dev,
        )
    }
    .expect("Kernel launch failed");

    let sums = sums_dev.to_host_vec(&stream).unwrap();
    let prods = prods_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let x = (i as u32).wrapping_add(1);
        let y = (i as u32).wrapping_add(2);
        let (got_sum, got_prod) = (sums[i], prods[i]);
        let (want_sum, want_prod) = (x.wrapping_add(y), x.wrapping_mul(y));
        if got_sum != want_sum || got_prod != want_prod {
            eprintln!(
                "Multi-output mismatch at {i}: expected ({want_sum}, {want_prod}), \
                 got ({got_sum}, {got_prod})"
            );
            std::process::exit(1);
        }
    }

    let mut products_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; the output buffer covers
    // every thread-dependent access.
    unsafe {
        module.inline_ptx_c_constraint_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut products_dev,
        )
    }
    .expect("C-constraint kernel launch failed");

    let products = products_dev.to_host_vec(&stream).unwrap();

    for (i, got) in products.iter().copied().enumerate() {
        let x = i as u64 + 65_537;
        let y = i as u64 + 3;
        let expected = x * y;

        assert_eq!(got, expected, "C-constraint mismatch at {i}");
    }

    let mut inout_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe {
        module.inline_ptx_inout_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut inout_dev,
        )
    }
    .expect("Kernel launch failed");

    let inout = inout_dev.to_host_vec(&stream).unwrap();
    for (i, got) in inout.iter().copied().enumerate() {
        let i = i as u32;
        let expected = i
            .wrapping_add(10)
            .wrapping_add(i.wrapping_mul(3).wrapping_add(1));
        if got != expected {
            eprintln!("Read-write mismatch at {i}: expected {expected}, got {got}");
            std::process::exit(1);
        }
    }

    let mut counts_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut wides_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut scales_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut checks_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.inline_ptx_multi_inout_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut counts_dev,
            &mut wides_dev,
            &mut scales_dev,
            &mut checks_dev,
        )
    }
    .expect("Kernel launch failed");

    let counts = counts_dev.to_host_vec(&stream).unwrap();
    let wides = wides_dev.to_host_vec(&stream).unwrap();
    let scales = scales_dev.to_host_vec(&stream).unwrap();
    let checks = checks_dev.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let iu = i as u32;
        let want_count = iu.wrapping_add(7).wrapping_add(iu);
        let want_wide = (iu as u64)
            .wrapping_mul(0x0000_0001_0000_0003)
            .wrapping_add(11);
        // Exact in f32: both terms are small dyadic rationals.
        let want_scale = iu as f32 * 0.5 + 0.25;
        let want_check = iu.wrapping_mul(iu);
        if counts[i] != want_count
            || wides[i] != want_wide
            || scales[i] != want_scale
            || checks[i] != want_check
        {
            eprintln!(
                "Multi-inout mismatch at {i}: expected ({want_count}, {want_wide}, \
                 {want_scale}, {want_check}), got ({}, {}, {}, {})",
                counts[i], wides[i], scales[i], checks[i]
            );
            std::process::exit(1);
        }
    }

    println!("SUCCESS: inline PTX results are correct");
}
