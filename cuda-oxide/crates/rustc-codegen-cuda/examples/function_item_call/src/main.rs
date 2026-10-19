/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for lowering function-item receivers in rust-call paths.
//!
//! Passing a function item to a generic `FnOnce` helper makes MIR call
//! `<fn item as FnOnce>::call_once`. The importer must resolve the receiver
//! back to the concrete function body instead of emitting a dangling trait-shim
//! callee symbol. The regression also covers `Fn`/`FnMut`, references,
//! multiple and nested-tuple arguments, tuple returns, and type/const-generic
//! function items. Diverging function items and closures also verify that a
//! direct Rust call returning `!` ends its MIR block with `unreachable`.
//! `call_once_decoy` proves that an ordinary function whose name contains
//! trait-like text is not mistaken for a callable-trait shim.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[inline(never)]
#[device]
fn plus_seven(x: u32) -> u32 {
    x + 7
}

#[device]
fn add_pair(a: u32, b: u32) -> u32 {
    a + 2 * b
}

#[device]
fn split(x: u32) -> (u32, u32) {
    (x, x + 1)
}

fn add_const<const N: u32>(x: u32) -> u32 {
    x + N
}

#[device]
fn identity<T: Copy>(x: T) -> T {
    x
}

fn nested_sum((a, (b, c)): (u32, (u32, u32))) -> u32 {
    a + b + c
}

// An empty loop gives this device function a `!` return without introducing
// panic lowering or another callee, which would test a different code path.
#[allow(clippy::empty_loop)]
#[inline(never)]
fn spin_forever(_value: u32) -> ! {
    loop {}
}

#[device]
fn apply_once<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[device]
fn apply_ref<F: Fn(u32) -> u32>(f: &F, x: u32) -> u32 {
    f(x)
}

#[device]
fn apply_mut<F: FnMut(u32) -> u32>(f: &mut F, x: u32) -> u32 {
    f(x)
}

#[device]
fn apply_two<F: FnOnce(u32, u32) -> u32>(f: F, a: u32, b: u32) -> u32 {
    f(a, b)
}

#[device]
fn apply_value<T, R, F>(f: F, value: T) -> R
where
    F: FnOnce(T) -> R,
{
    f(value)
}

#[device]
fn apply_never<F: FnOnce(u32) -> !>(f: F, value: u32) -> ! {
    f(value)
}

/// This is an ordinary Rust function, despite the text `call_once` in its
/// name. It guards against treating function-name substrings as callable-trait
/// shims and silently calling `f` instead of this body.
#[device]
fn call_once_decoy<F>(_f: F, args: (u32,)) -> u32 {
    args.0 + 1_000
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn function_item_call(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get() as u32;
        if let Some(slot) = out.get_mut(idx) {
            let by_once = apply_once(plus_seven, raw);
            let by_ref = apply_ref(&plus_seven, raw);
            let mut callable = plus_seven;
            let by_mut = apply_mut(&mut callable, raw);
            let ordinary_call = call_once_decoy(plus_seven, (raw,));
            let multi_arg = apply_two(add_pair, raw, 3);
            let pair = apply_value(split, raw);
            let const_generic = apply_once(add_const::<9>, raw);
            let type_generic = apply_value(identity::<u32>, raw);
            let nested_tuple = apply_value(nested_sum, (raw, (2, 3)));
            let callable_ref = &plus_seven;
            let through_reference = apply_ref(&callable_ref, raw);
            *slot = by_once
                + by_ref
                + by_mut
                + ordinary_call
                + multi_arg
                + pair.0
                + pair.1
                + const_generic
                + type_generic
                + nested_tuple
                + through_reference;
        }
    }

    /// Compile both diverging callable paths without entering them on-device.
    /// A Rust call returning `!` has no successor block, so the importer must
    /// emit the direct call followed by `mir.unreachable`.
    #[kernel]
    pub fn diverging_callable(flag: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if flag == 1 {
            apply_never(spin_forever, flag);
        }
        if flag == 2 {
            // Keep the closure body as the minimal non-returning expression;
            // panic or thread sleeping would exercise unrelated lowering.
            #[allow(clippy::empty_loop)]
            apply_never(|_| loop {}, flag);
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = 0xd1ce;
        }
    }
}

fn main() {
    const N: usize = 16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.function_item_call(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
    }
    .expect("kernel launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    let expected: Vec<u32> = (0..N).map(|i| 11 * i as u32 + 1_049).collect();
    if out != expected {
        eprintln!("FAIL: got {out:?}, expected {expected:?}");
        std::process::exit(1);
    }

    let mut diverging_out = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.diverging_callable(
            &stream,
            LaunchConfig::for_num_elems(1),
            0,
            &mut diverging_out,
        )
    }
    .expect("diverging callable regression launch");
    if diverging_out.to_host_vec(&stream).unwrap() != [0xd1ce] {
        eprintln!("FAIL: diverging callable regression took the wrong path");
        std::process::exit(1);
    }

    println!("function_item_call: PASS");
}
