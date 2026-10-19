/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked and overflowing integer arithmetic conformance test.
//!
//! The historical regression verifies that `overflowing_add`,
//! `overflowing_sub`, and `overflowing_mul` return the correct
//! `(wrapping_result, overflow)` pair. Before the fix, the overflow flag was
//! hardcoded to `false`; it is now computed by the proper LLVM overflow
//! intrinsics.
//!
//! This example also exercises the public `checked_add`, `checked_sub`, and
//! `checked_mul` APIs. Those operations share Rust's checked binary arithmetic
//! machinery but expose overflow as `None` rather than as a boolean flag.
//! Unsigned `u8` and signed `i8` cases cover both overflow and non-overflow
//! paths, including `Some(0)` so it cannot be confused with `None`.
//!
//! Run:
//!   cargo oxide run checked_arith
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run checked_arith

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// out[i] = (a[i].overflowing_add(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_add(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_add(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }

    /// out[i] = (a[i].overflowing_sub(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_sub(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_sub(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }

    /// out[i] = (a[i].overflowing_mul(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_mul(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_mul(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }

    /// Exercise unsigned `checked_{add,sub,mul}` and encode each `Option<u8>`.
    ///
    /// Encoding:
    /// - `None` -> 0
    /// - `Some(value)` -> bit 8 set, payload in bits 0..7
    #[kernel]
    pub fn checked_u8(
        a: &[u8],
        b: &[u8],
        mut add_out: DisjointSlice<u32>,
        mut sub_out: DisjointSlice<u32>,
        mut mul_out: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();

        if let Some(o) = add_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_add(b[i]) {
                Some(value) => 0x100 | value as u32,
                None => 0,
            };
        }

        if let Some(o) = sub_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_sub(b[i]) {
                Some(value) => 0x100 | value as u32,
                None => 0,
            };
        }

        if let Some(o) = mul_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_mul(b[i]) {
                Some(value) => 0x100 | value as u32,
                None => 0,
            };
        }
    }

    /// Exercise signed `checked_{add,sub,mul}` and encode each `Option<i8>`.
    ///
    /// Encoding:
    /// - `None` -> 0
    /// - `Some(value)` -> bit 8 set, two's-complement payload in bits 0..7
    #[kernel]
    pub fn checked_i8(
        a: &[i8],
        b: &[i8],
        mut add_out: DisjointSlice<u32>,
        mut sub_out: DisjointSlice<u32>,
        mut mul_out: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();

        if let Some(o) = add_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_add(b[i]) {
                Some(value) => 0x100 | value as u8 as u32,
                None => 0,
            };
        }

        if let Some(o) = sub_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_sub(b[i]) {
                Some(value) => 0x100 | value as u8 as u32,
                None => 0,
            };
        }

        if let Some(o) = mul_out.get_mut(thread::index_1d()) {
            *o = match a[i].checked_mul(b[i]) {
                Some(value) => 0x100 | value as u8 as u32,
                None => 0,
            };
        }
    }
}

fn check(label: &str, got: u32, expected_result: u8, expected_overflow: bool) -> bool {
    let got_result = (got & 0xff) as u8;
    let got_overflow = (got >> 8) & 1 == 1;
    if got_result != expected_result || got_overflow != expected_overflow {
        eprintln!(
            "  FAIL {label}: result={got_result} (want {expected_result}), \
             overflow={got_overflow} (want {expected_overflow})"
        );
        false
    } else {
        true
    }
}

fn check_checked_u8(label: &str, got: u32, expected: Option<u8>) -> bool {
    let want = match expected {
        Some(value) => 0x100 | value as u32,
        None => 0,
    };

    if got != want {
        eprintln!("  FAIL {label}: encoded=0x{got:03x} (want 0x{want:03x}, {expected:?})");
        false
    } else {
        true
    }
}

fn check_checked_i8(label: &str, got: u32, expected: Option<i8>) -> bool {
    let want = match expected {
        Some(value) => 0x100 | value as u8 as u32,
        None => 0,
    };

    if got != want {
        eprintln!("  FAIL {label}: encoded=0x{got:03x} (want 0x{want:03x}, {expected:?})");
        false
    } else {
        true
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let cfg = LaunchConfig::for_num_elems(4);

    // --- overflowing_add ---
    // 200 + 100 = 300 -> wraps to 44, overflow
    // 100 + 50  = 150 -> no overflow
    // 255 + 1   = 256 -> wraps to 0, overflow
    // 0   + 0   = 0   -> no overflow
    let a_add: Vec<u8> = vec![200, 100, 255, 0];
    let b_add: Vec<u8> = vec![100, 50, 1, 0];
    let a_dev = DeviceBuffer::from_host(&stream, &a_add).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_add).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_add(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_add launch");
    let out_add = out_dev.to_host_vec(&stream).unwrap();

    // --- overflowing_sub ---
    // 100 - 200 = -100 -> wraps to 156, overflow
    // 200 - 100 = 100  -> no overflow
    // 0   - 1   = -1   -> wraps to 255, overflow
    // 50  - 50  = 0    -> no overflow
    let a_sub: Vec<u8> = vec![100, 200, 0, 50];
    let b_sub: Vec<u8> = vec![200, 100, 1, 50];
    let a_dev = DeviceBuffer::from_host(&stream, &a_sub).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_sub).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_sub(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_sub launch");
    let out_sub = out_dev.to_host_vec(&stream).unwrap();

    // --- overflowing_mul ---
    // 20  * 10  = 200   -> no overflow
    // 20  * 20  = 400   -> wraps to 144, overflow
    // 255 * 2   = 510   -> wraps to 254, overflow
    // 1   * 1   = 1     -> no overflow
    let a_mul: Vec<u8> = vec![20, 20, 255, 1];
    let b_mul: Vec<u8> = vec![10, 20, 2, 1];
    let a_dev = DeviceBuffer::from_host(&stream, &a_mul).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_mul).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_mul(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_mul launch");
    let out_mul = out_dev.to_host_vec(&stream).unwrap();

    // --- checked_{add,sub,mul} for unsigned u8 ---
    //
    // The four pairs deliberately include overflow, non-overflow, and Some(0):
    //   200, 100 -> add None;     sub Some(100); mul None
    //   100, 200 -> add None;     sub None;      mul None
    //    20,  10 -> add Some(30); sub Some(10);  mul Some(200)
    //     0,   0 -> add Some(0);  sub Some(0);   mul Some(0)
    let checked_u8_a: Vec<u8> = vec![200, 100, 20, 0];
    let checked_u8_b: Vec<u8> = vec![100, 200, 10, 0];
    let checked_u8_a_dev = DeviceBuffer::from_host(&stream, &checked_u8_a).unwrap();
    let checked_u8_b_dev = DeviceBuffer::from_host(&stream, &checked_u8_b).unwrap();
    let mut checked_u8_add_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    let mut checked_u8_sub_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    let mut checked_u8_mul_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.checked_u8(
            &stream,
            cfg,
            &checked_u8_a_dev,
            &checked_u8_b_dev,
            &mut checked_u8_add_dev,
            &mut checked_u8_sub_dev,
            &mut checked_u8_mul_dev,
        )
    }
    .expect("checked_u8 launch");

    let checked_u8_add = checked_u8_add_dev.to_host_vec(&stream).unwrap();
    let checked_u8_sub = checked_u8_sub_dev.to_host_vec(&stream).unwrap();
    let checked_u8_mul = checked_u8_mul_dev.to_host_vec(&stream).unwrap();

    // --- checked_{add,sub,mul} for signed i8 ---
    //
    // These values force the signed LLVM overflow paths in both directions
    // while retaining ordinary and Some(0) results:
    //    120,  10 -> add None;       sub Some(110);  mul None
    //   -120, -20 -> add None;       sub Some(-100); mul None
    //    120, -20 -> add Some(100);  sub None;       mul None
    //   -120,  20 -> add Some(-100); sub None;       mul None
    //     10,  10 -> add Some(20);   sub Some(0);    mul Some(100)
    //    -10,  10 -> add Some(0);    sub Some(-20);  mul Some(-100)
    let checked_i8_a: Vec<i8> = vec![120, -120, 120, -120, 10, -10];
    let checked_i8_b: Vec<i8> = vec![10, -20, -20, 20, 10, 10];
    let checked_i8_a_dev = DeviceBuffer::from_host(&stream, &checked_i8_a).unwrap();
    let checked_i8_b_dev = DeviceBuffer::from_host(&stream, &checked_i8_b).unwrap();
    let mut checked_i8_add_dev = DeviceBuffer::<u32>::zeroed(&stream, 6).unwrap();
    let mut checked_i8_sub_dev = DeviceBuffer::<u32>::zeroed(&stream, 6).unwrap();
    let mut checked_i8_mul_dev = DeviceBuffer::<u32>::zeroed(&stream, 6).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.checked_i8(
            &stream,
            LaunchConfig::for_num_elems(6),
            &checked_i8_a_dev,
            &checked_i8_b_dev,
            &mut checked_i8_add_dev,
            &mut checked_i8_sub_dev,
            &mut checked_i8_mul_dev,
        )
    }
    .expect("checked_i8 launch");

    let checked_i8_add = checked_i8_add_dev.to_host_vec(&stream).unwrap();
    let checked_i8_sub = checked_i8_sub_dev.to_host_vec(&stream).unwrap();
    let checked_i8_mul = checked_i8_mul_dev.to_host_vec(&stream).unwrap();

    let mut ok = true;

    // Historical overflowing arithmetic regression.
    ok &= check("add[0] 200+100", out_add[0], 44, true);
    ok &= check("add[1] 100+50", out_add[1], 150, false);
    ok &= check("add[2] 255+1", out_add[2], 0, true);
    ok &= check("add[3] 0+0", out_add[3], 0, false);

    ok &= check("sub[0] 100-200", out_sub[0], 156, true);
    ok &= check("sub[1] 200-100", out_sub[1], 100, false);
    ok &= check("sub[2] 0-1", out_sub[2], 255, true);
    ok &= check("sub[3] 50-50", out_sub[3], 0, false);

    ok &= check("mul[0] 20*10", out_mul[0], 200, false);
    ok &= check("mul[1] 20*20", out_mul[1], 144, true);
    ok &= check("mul[2] 255*2", out_mul[2], 254, true);
    ok &= check("mul[3] 1*1", out_mul[3], 1, false);

    // Unsigned checked arithmetic.
    let expected_u8_add = [None, None, Some(30), Some(0)];
    let expected_u8_sub = [Some(100), None, Some(10), Some(0)];
    let expected_u8_mul = [None, None, Some(200), Some(0)];

    for i in 0..4 {
        ok &= check_checked_u8(
            &format!("checked_add<u8>[{i}]"),
            checked_u8_add[i],
            expected_u8_add[i],
        );
        ok &= check_checked_u8(
            &format!("checked_sub<u8>[{i}]"),
            checked_u8_sub[i],
            expected_u8_sub[i],
        );
        ok &= check_checked_u8(
            &format!("checked_mul<u8>[{i}]"),
            checked_u8_mul[i],
            expected_u8_mul[i],
        );
    }

    // Signed checked arithmetic.
    let expected_i8_add = [None, None, Some(100), Some(-100), Some(20), Some(0)];
    let expected_i8_sub = [Some(110), Some(-100), None, None, Some(0), Some(-20)];
    let expected_i8_mul = [None, None, None, None, Some(100), Some(-100)];

    for i in 0..6 {
        ok &= check_checked_i8(
            &format!("checked_add<i8>[{i}]"),
            checked_i8_add[i],
            expected_i8_add[i],
        );
        ok &= check_checked_i8(
            &format!("checked_sub<i8>[{i}]"),
            checked_i8_sub[i],
            expected_i8_sub[i],
        );
        ok &= check_checked_i8(
            &format!("checked_mul<i8>[{i}]"),
            checked_i8_mul[i],
            expected_i8_mul[i],
        );
    }

    if ok {
        println!("SUCCESS: all overflowing_{{add,sub,mul}} results correct");
        println!("PASS: checked_add/sub/mul (unsigned u8)");
        println!("PASS: checked_add/sub/mul (signed i8)");
        println!("PASS: checked_arith");
    } else {
        std::process::exit(1);
    }
}
