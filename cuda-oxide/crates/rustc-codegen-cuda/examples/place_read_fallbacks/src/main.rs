/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for place-read fallback cases.
//!
//! `Copy(place)` / `Move(place)` reads prefer address lowering when the place
//! can be represented as a real final load address. These kernels cover cases
//! that must stay on the explicit value fallback path:
//!
//! - enum downcast + payload field reads, which still use value-space enum
//!   payload extraction;
//! - ZST local and ZST field reads, where emitting a final load would be
//!   meaningless;
//! - tuple field reads, because `mir.field_addr` currently verifies struct
//!   pointees only while value-space `mir.extract_field` supports tuples.
//!
//! Run: cargo oxide run place_read_fallbacks

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy)]
pub enum Payload {
    A(u32),
    B(u32),
    C,
}

#[derive(Clone, Copy)]
pub struct Marker;

#[derive(Clone, Copy)]
pub struct WithZst {
    tag: Marker,
    value: u32,
}

#[device]
#[inline(never)]
pub fn score_marker(_marker: Marker) -> u32 {
    17
}

#[device]
#[inline(never)]
pub fn read_pair(pair: &(u32, u32), choose_first: bool) -> u32 {
    if choose_first { pair.0 } else { pair.1 }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Projection chain: [Index, Downcast, Field].
    ///
    /// Addressing enum payload fields needs variant-layout support that the
    /// address walker deliberately does not implement yet. This read should
    /// therefore fall back to the value path and use enum payload extraction.
    #[kernel]
    pub fn enum_payload_read(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() & 3;

        let xs = [Payload::A(7), Payload::B(8), Payload::C, Payload::A(11)];
        let value = match xs[i] {
            Payload::A(x) => x,
            Payload::B(y) => y + 100,
            Payload::C => 999,
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }

    /// Bare ZST read plus struct-field ZST read.
    ///
    /// The `marker` local has no runtime storage, and `wrapper.tag` has no
    /// meaningful final load address. Both should use the synthetic/value
    /// fallback path, while `wrapper.value` can still be read normally.
    #[kernel]
    pub fn zst_reads(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }

        let marker = Marker;
        let wrapper = WithZst {
            tag: marker,
            value: 25,
        };

        let value = score_marker(marker) + score_marker(wrapper.tag) + wrapper.value;

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }

    /// Tuple fields intentionally remain on the value fallback path.
    ///
    /// `mir.field_addr` currently verifies `MirStructType` pointees only, so
    /// forcing tuple fields through address lowering would fail verification.
    #[kernel]
    pub fn tuple_field_read(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;

        let pair = (i + 3, i + 40);
        let value = read_pair(&pair, (i & 1) == 0);

        if let Some(slot) = out.get_mut(idx) {
            *slot = value;
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    const N: usize = 4;
    let cfg = LaunchConfig::for_num_elems(N as u32);
    let mut failures = 0usize;

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.enum_payload_read(&stream, cfg, &mut out_dev) }
        .expect("enum_payload_read launch");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let expected = [7, 108, 999, 11];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        if got != want {
            eprintln!("FAIL enum_payload_read[{i}]: got {got} want {want}");
            failures += 1;
        }
    }

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.zst_reads(&stream, LaunchConfig::for_num_elems(1), &mut out_dev) }
        .expect("zst_reads launch");
    let out = out_dev.to_host_vec(&stream).unwrap();
    if out[0] != 59 {
        eprintln!("FAIL zst_reads: got {} want 59", out[0]);
        failures += 1;
    }

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.tuple_field_read(&stream, cfg, &mut out_dev) }
        .expect("tuple_field_read launch");
    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, &got) in out.iter().enumerate() {
        let want = if (i & 1) == 0 {
            i as u32 + 3
        } else {
            i as u32 + 40
        };
        if got != want {
            eprintln!("FAIL tuple_field_read[{i}]: got {got} want {want}");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("SUCCESS: place-read fallbacks produce correct results");
    } else {
        eprintln!("FAIL: {failures} place-read fallback mismatches");
        std::process::exit(1);
    }
}
