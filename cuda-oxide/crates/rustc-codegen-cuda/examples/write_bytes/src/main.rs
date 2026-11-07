/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for `core::intrinsics::write_bytes` on the device.
//!
//! `ptr::write_bytes`, `mem::zeroed`, `MaybeUninit::zeroed` and `<[T]>::fill`
//! all reach the `write_bytes` intrinsic. For `fill` it sits in a one-byte
//! branch of the generic body, so every element type reaches it. cuda-oxide
//! lowers it to `mir.memset` and then `llvm.memset`, scaling the element count
//! to bytes.
//!
//! Each kernel writes a thread-local buffer through one of these routes and
//! stores a checksum of it; the host computes the same checksums.
//!
//! Usage:
//!   cargo oxide run write_bytes

use core::mem::MaybeUninit;

use cuda_core::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;

/// Accumulator that starts from `mem::zeroed`.
struct Stats {
    sum: u32,
    count: u32,
    max: u32,
}

/// Order-sensitive checksum, shared by the kernels and the host.
fn checksum(values: impl IntoIterator<Item = u32>) -> u32 {
    values
        .into_iter()
        .fold(0u32, |acc, value| acc.rotate_left(5) ^ value)
}

/// One byte repeated across a `u32`, as `write_bytes` sets each element.
fn splat(byte: u8) -> u32 {
    u32::from_ne_bytes([byte; 4])
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `ptr::write_bytes` with a runtime byte and a runtime element count:
    /// elements `start..8` of a local `[u32; 8]` become `splat(byte)`.
    #[kernel]
    pub fn write_bytes_tail(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let byte = input[i] as u8;
        let start = i % 8;
        let mut words = [0u32; 8];
        // SAFETY: `start < 8`, so the `8 - start` elements written are all in `words`.
        unsafe { core::ptr::write_bytes(words.as_mut_ptr().add(start), byte, 8 - start) };
        if let Some(o) = out.get_mut(idx) {
            *o = checksum(words);
        }
    }

    /// `mem::zeroed` builds an all-zero struct, which then accumulates a
    /// runtime number of inputs.
    #[kernel]
    pub fn zeroed_struct(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // SAFETY: all-zero bytes are a valid `Stats`.
        let mut stats: Stats = unsafe { core::mem::zeroed() };
        for k in 0..i % 4 {
            let value = input[(i + k) % input.len()];
            stats.sum = stats.sum.wrapping_add(value);
            stats.count += 1;
            stats.max = stats.max.max(value);
        }
        if let Some(o) = out.get_mut(idx) {
            *o = checksum([stats.sum, stats.count, stats.max]);
        }
    }

    /// `MaybeUninit::zeroed` for an array, with one element overwritten.
    #[kernel]
    pub fn maybe_uninit_zeroed(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // SAFETY: all-zero bytes are valid `u32`s.
        let mut words = unsafe { MaybeUninit::<[u32; 8]>::zeroed().assume_init() };
        words[i % 8] = input[i];
        if let Some(o) = out.get_mut(idx) {
            *o = checksum(words);
        }
    }

    /// `<[u8]>::fill`, which takes `fill`'s `write_bytes` branch.
    #[kernel]
    pub fn fill_bytes(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let mut bytes = [0u8; 16];
        bytes[i % 16..].fill(input[i] as u8);
        if let Some(o) = out.get_mut(idx) {
            *o = checksum(bytes.map(|b| b as u32));
        }
    }

    /// `<[u32]>::fill`, whose generic body still contains the `write_bytes`
    /// branch even though a four-byte element never takes it.
    #[kernel]
    pub fn fill_words(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let mut words = [0u32; 8];
        words[i % 8..].fill(input[i]);
        if let Some(o) = out.get_mut(idx) {
            *o = checksum(words);
        }
    }
}

fn main() {
    println!("=== write_bytes ===");

    let input: Vec<u32> = (0..N as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ 0x5A5A_1234)
        .collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let din = DeviceBuffer::from_host(&stream, &input).unwrap();
    let cfg = || LaunchConfig::for_num_elems(N as u32);

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.write_bytes_tail(&stream, cfg(), &din, &mut out) }
        .expect("write_bytes_tail launch");
    let want: Vec<u32> = (0..N)
        .map(|i| {
            let fill = splat(input[i] as u8);
            checksum((0..8).map(|k| if k >= i % 8 { fill } else { 0 }))
        })
        .collect();
    assert_eq!(out.to_host_vec(&stream).unwrap(), want, "ptr::write_bytes");
    println!("PASS: ptr::write_bytes (runtime byte and element count)");

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.zeroed_struct(&stream, cfg(), &din, &mut out) }.expect("zeroed_struct launch");
    let want: Vec<u32> = (0..N)
        .map(|i| {
            let values: Vec<u32> = (0..i % 4).map(|k| input[(i + k) % N]).collect();
            let sum = values.iter().fold(0u32, |acc, &v| acc.wrapping_add(v));
            let max = values.iter().copied().max().unwrap_or(0);
            checksum([sum, values.len() as u32, max])
        })
        .collect();
    assert_eq!(out.to_host_vec(&stream).unwrap(), want, "mem::zeroed");
    println!("PASS: mem::zeroed");

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.maybe_uninit_zeroed(&stream, cfg(), &din, &mut out) }
        .expect("maybe_uninit_zeroed launch");
    let want: Vec<u32> = (0..N)
        .map(|i| checksum((0..8).map(|k| if k == i % 8 { input[i] } else { 0 })))
        .collect();
    assert_eq!(
        out.to_host_vec(&stream).unwrap(),
        want,
        "MaybeUninit::zeroed"
    );
    println!("PASS: MaybeUninit::zeroed");

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.fill_bytes(&stream, cfg(), &din, &mut out) }.expect("fill_bytes launch");
    let want: Vec<u32> = (0..N)
        .map(|i| {
            let byte = input[i] as u8 as u32;
            checksum((0..16).map(|k| if k >= i % 16 { byte } else { 0 }))
        })
        .collect();
    assert_eq!(out.to_host_vec(&stream).unwrap(), want, "<[u8]>::fill");
    println!("PASS: <[u8]>::fill");

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.fill_words(&stream, cfg(), &din, &mut out) }.expect("fill_words launch");
    let want: Vec<u32> = (0..N)
        .map(|i| checksum((0..8).map(|k| if k >= i % 8 { input[i] } else { 0 })))
        .collect();
    assert_eq!(out.to_host_vec(&stream).unwrap(), want, "<[u32]>::fill");
    println!("PASS: <[u32]>::fill");

    println!("PASS: write_bytes");
}
