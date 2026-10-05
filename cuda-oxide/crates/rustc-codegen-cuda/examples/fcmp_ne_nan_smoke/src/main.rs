/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal repro: float `!=` should be UNORDERED (x != x is TRUE for NaN).
// Rust PartialEq::ne on floats is unordered. cuda-oxide lowered it to fcmp ONE
// (ordered, FALSE for NaN), so x != x folded to false -> NaN handling broken.
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    // The self-comparison IS the test subject: it must lower to `fcmp une`,
    // not be rewritten as `.is_nan()` (which takes a different code path).
    #[allow(clippy::eq_op)]
    #[kernel]
    pub fn is_nan(x: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(ce) = c.get_mut(idx) {
            let v = x[i];
            // x != x is the canonical NaN check; must be 1.0 for NaN, 0.0 otherwise.
            *ce = if v != v { 1.0 } else { 0.0 };
        }
    }

    #[kernel]
    pub fn float_ne(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(ce) = c.get_mut(idx) {
            // Two-operand `!=`: unordered, so any NaN operand makes it true.
            *ce = if a[i] != b[i] { 1.0 } else { 0.0 };
        }
    }
}

fn check(name: &str, got: &[f32], expect: &[f32]) -> bool {
    let ok = got == expect;
    println!(
        "{name}: {:?}  (expect {:?})  {}",
        got,
        expect,
        if ok { "ok" } else { "MISMATCH" }
    );
    ok
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let m = kernels::load(&ctx).expect("load");

    // Self-comparison: x != x is the canonical NaN check.
    let x = vec![f32::NAN, 1.0, 0.0, -1.0, f32::INFINITY];
    let xd = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut cd = DeviceBuffer::<f32>::zeroed(&stream, x.len()).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        m.is_nan(
            &stream,
            LaunchConfig::for_num_elems(x.len() as u32),
            &xd,
            &mut cd,
        )
    }
    .expect("launch is_nan");
    let is_nan = cd.to_host_vec(&stream).unwrap();
    println!("input : {:?}", x);
    let ok_self = check("x!=x ", &is_nan, &[1.0, 0.0, 0.0, 0.0, 0.0]);

    // Two-operand `!=`. Locks the lanes where une and one differ (NaN on
    // either or both sides) and the IEEE lanes where they agree
    // (-0.0 == 0.0, inf == inf).
    let a = vec![f32::NAN, f32::NAN, 1.0, -0.0, f32::INFINITY, 2.0];
    let b = vec![1.0, f32::NAN, 1.0, 0.0, f32::INFINITY, 3.0];
    let ad = DeviceBuffer::from_host(&stream, &a).unwrap();
    let bd = DeviceBuffer::from_host(&stream, &b).unwrap();
    let mut nd = DeviceBuffer::<f32>::zeroed(&stream, a.len()).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        m.float_ne(
            &stream,
            LaunchConfig::for_num_elems(a.len() as u32),
            &ad,
            &bd,
            &mut nd,
        )
    }
    .expect("launch float_ne");
    let ne = nd.to_host_vec(&stream).unwrap();
    println!("a     : {:?}", a);
    println!("b     : {:?}", b);
    let ok_two = check("a!=b ", &ne, &[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    let ok = ok_self && ok_two;
    println!(
        "{}",
        if ok {
            "SUCCESS: float != is unordered (NaN detected)"
        } else {
            "FAILURE: float != mishandles NaN (ordered fcmp?)"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
