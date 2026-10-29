/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! f32 `max` / `min` on the device (#1594).
//!
//! `f32::max`, `f32::min`, `libm::fmaxf` and `libm::fminf` lower to the native
//! PTX `max.f32` / `min.f32` (through `llvm.nvvm.fmax.f` / `llvm.nvvm.fmin.f`)
//! instead of a `setp` + `selp` pair. `cargo oxide inspect float_minmax` shows
//! the PTX.
//!
//! The host runs every pair of a value set that covers quiet and signaling
//! NaNs of both signs, infinities, signed zeros and subnormals, so each NaN
//! lands in both operand positions, and checks Rust's contract:
//!
//! - one operand is NaN: the other operand is returned bit for bit;
//! - both are NaN: the result is a NaN;
//! - otherwise: the larger (smaller) value, subnormals preserved.
//!
//! `max(-0.0, +0.0)` is unspecified in Rust, so the signed-zero results are
//! printed and only required to be a zero.
//!
//! Run with:
//!   cargo oxide run float_minmax

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn std_max(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i].max(b[i]);
        }
    }

    #[kernel]
    pub fn std_min(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i].min(b[i]);
        }
    }

    #[kernel]
    pub fn libm_max(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = libm::fmaxf(a[i], b[i]);
        }
    }

    #[kernel]
    pub fn libm_min(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = libm::fminf(a[i], b[i]);
        }
    }

    /// Both operations in one kernel, as in the report for #1594.
    #[kernel]
    pub fn std_minmax(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = a[i].max(b[i]) + a[i].min(b[i]);
        }
    }

    #[kernel]
    pub fn libm_minmax(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = libm::fmaxf(a[i], b[i]) + libm::fminf(a[i], b[i]);
        }
    }
}

/// Bit patterns: numbers, infinities, signed zeros, subnormals, and quiet and
/// signaling NaNs of both signs.
const VALUES: [u32; 16] = [
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7f7f_ffff, // f32::MAX
    0x0080_0000, // smallest normal
    0x0000_0001, // smallest subnormal
    0x8000_0001, // -smallest subnormal
    0x007f_ffff, // largest subnormal
    0x7fc0_0000, // +qNaN
    0xffc0_0000, // -qNaN
    0x7fa0_0000, // +sNaN
    0xffa0_0000, // -sNaN
    0x7fc0_1234, // qNaN with a payload
];

fn is_zero(x: f32) -> bool {
    x == 0.0
}

/// Check one result against Rust's contract. Returns an error description.
fn check(is_max: bool, x: f32, y: f32, r: f32) -> Result<(), String> {
    let (xb, yb, rb) = (x.to_bits(), y.to_bits(), r.to_bits());
    if x.is_nan() && y.is_nan() {
        return if r.is_nan() {
            Ok(())
        } else {
            Err(format!("both NaN, got {rb:#010x}"))
        };
    }
    if x.is_nan() || y.is_nan() {
        let want = if x.is_nan() { yb } else { xb };
        return if rb == want {
            Ok(())
        } else {
            Err(format!("one NaN, want {want:#010x}, got {rb:#010x}"))
        };
    }
    if is_zero(x) && is_zero(y) {
        return if is_zero(r) {
            Ok(())
        } else {
            Err(format!("zeros, got non-zero {rb:#010x}"))
        };
    }
    let want = match (is_max, x > y) {
        (true, true) | (false, false) => xb,
        _ => yb,
    };
    if rb == want {
        Ok(())
    } else {
        Err(format!("want {want:#010x}, got {rb:#010x}"))
    }
}

fn main() {
    println!("=== float_minmax: f32 max/min on the device ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in &VALUES {
        for &y in &VALUES {
            a.push(f32::from_bits(x));
            b.push(f32::from_bits(y));
        }
    }
    let n = a.len();

    let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(n as u32);

    macro_rules! run {
        ($kernel:ident) => {{
            let mut out = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
            // SAFETY: the launch is 1-D, one thread per element, and the
            // buffers cover the kernel's accesses.
            unsafe { module.$kernel(&stream, cfg, &a_dev, &b_dev, &mut out) }
                .expect(concat!(stringify!($kernel), " launch failed"));
            out.to_host_vec(&stream).unwrap()
        }};
    }

    let mut ok = true;
    for (name, is_max, out) in [
        ("std_max", true, run!(std_max)),
        ("std_min", false, run!(std_min)),
        ("libm_max", true, run!(libm_max)),
        ("libm_min", false, run!(libm_min)),
    ] {
        let mut bad = 0;
        for i in 0..n {
            if let Err(why) = check(is_max, a[i], b[i], out[i]) {
                if bad < 5 {
                    println!(
                        "{name}: x={:#010x} y={:#010x}: {why}",
                        a[i].to_bits(),
                        b[i].to_bits()
                    );
                }
                bad += 1;
            }
        }
        println!("{name}: {} of {n} pairs match", n - bad);
        ok &= bad == 0;

        if name.starts_with("std") {
            // Pair (x, y) is stored at index xi * VALUES.len() + yi.
            let at = |xi: usize, yi: usize| out[xi * VALUES.len() + yi].to_bits();
            println!(
                "  signed zero: ({name})(-0.0, +0.0) = {:#010x}, (+0.0, -0.0) = {:#010x}",
                at(3, 2),
                at(2, 3)
            );
        }
    }

    // The combined kernels: `max + min`. Skip pairs with a NaN; infinities of
    // opposite sign make the sum NaN on host and device alike.
    for (name, out) in [
        ("std_minmax", run!(std_minmax)),
        ("libm_minmax", run!(libm_minmax)),
    ] {
        let mut bad = 0;
        for i in 0..n {
            let (x, y) = (a[i], b[i]);
            if x.is_nan() || y.is_nan() {
                continue;
            }
            let want = x + y;
            let same = if want.is_nan() {
                out[i].is_nan()
            } else {
                out[i] == want
            };
            if !same {
                bad += 1;
            }
        }
        println!("{name}: {bad} mismatches among the non-NaN pairs");
        ok &= bad == 0;
    }

    if ok {
        println!("SUCCESS: f32 max/min follow Rust's NaN contract");
    } else {
        println!("FAIL: device max/min diverge from Rust's contract");
        std::process::exit(1);
    }
}
