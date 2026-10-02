/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Multi-payload enums across the host/device ABI, in both directions.
//!
//! `E` below has two payload variants whose fields OVERLAP in rustc's
//! layout (`A`'s `u32` and `B`'s `f32` both live at byte 4) plus sparse
//! explicit discriminants. Its device lowering used to concatenate the
//! payloads (12 bytes vs rustc's 8), so kernels taking `E` behind a
//! pointer or slice were rejected at the ABI boundary with "not yet
//! field-faithful". The enum slot map now places every field at its
//! exact rustc byte offset (`{ i32, i32 }`, `A.0` in a typed slot,
//! `B.0` filler-resident behind a memory reinterpretation), so host and
//! device agree byte-for-byte:
//!
//! ```text
//! byte:      0        4
//! rustc:   [ tag    | A.0 (u32) / B.0 (f32) ]   8 bytes, overlapped
//! device:  { i32    , i32 }                     8 bytes, B.0 via spill
//! ```
//!
//! Two kernels prove the two ABI directions:
//!
//! | Kernel | Direction      | Exercises                                 |
//! |--------|----------------|-------------------------------------------|
//! | decode | host -> device | match + payload reads over host bytes     |
//! | encode | device -> host | construct + whole-enum stores, host reads |
//!
//! Run with:
//!   cargo oxide run enum_abi_multi_payload

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Two overlapping payload variants (different types at byte 4) plus a
/// fieldless variant, with sparse explicit discriminants so the tag holds
/// declared VALUES, never variant indices.
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum E {
    A(u32) = 7,
    B(f32) = 9,
    C = 4096,
}

// Sound because the device lowering is byte-identical to rustc's layout
// (which is exactly what this example proves end-to-end).
unsafe impl DeviceCopy for E {}

#[cuda_module]
mod kernels {
    use super::*;

    /// host -> device: match host-written enum bytes and decode each
    /// variant's payload. `B`'s payload read goes through the
    /// filler-resident memory path.
    #[kernel]
    pub fn decode(input: &[E], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let (Some(v), Some(out_elem)) = (input.get(idx.get()), out.get_mut(idx)) {
            *out_elem = match *v {
                E::A(x) => x,
                E::B(f) => (f as u32) + 100_000,
                E::C => 4096,
            };
        }
    }

    /// device -> host: construct every variant on the device and store
    /// whole enum values into host-visible memory. `B`'s construction
    /// writes its payload through the filler-resident memory path.
    #[kernel]
    pub fn encode(src: &[u32], mut out: DisjointSlice<E>) {
        let idx = thread::index_1d();
        if let (Some(&s), Some(out_elem)) = (src.get(idx.get()), out.get_mut(idx)) {
            *out_elem = match s % 3 {
                0 => E::A(s),
                1 => E::B(s as f32),
                _ => E::C,
            };
        }
    }
}

/// Host-side mirror of `decode`.
fn expected_decode(v: &E) -> u32 {
    match *v {
        E::A(x) => x,
        E::B(f) => (f as u32) + 100_000,
        E::C => 4096,
    }
}

fn main() {
    println!("=== enum_abi_multi_payload: field-faithful enum ABI ===\n");

    // The premise of the whole example: rustc overlaps the payloads.
    assert_eq!(
        std::mem::size_of::<E>(),
        8,
        "rustc layout: tag + overlapped payloads"
    );

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 64;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failures = 0usize;

    // ---- host -> device: the kernel decodes host-written enum bytes ----
    let h_input: Vec<E> = (0..N as u32)
        .map(|i| match i % 3 {
            0 => E::A(i),
            1 => E::B(i as f32),
            _ => E::C,
        })
        .collect();
    let d_input = DeviceBuffer::from_host(&stream, &h_input).unwrap();
    let mut d_decoded = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.decode(stream.as_ref(), cfg, &d_input, &mut d_decoded) }
        .expect("launch decode");
    let decoded = d_decoded.to_host_vec(&stream).unwrap();
    for (i, (&got, want)) in decoded
        .iter()
        .zip(h_input.iter().map(expected_decode))
        .enumerate()
    {
        if got != want {
            println!("FAIL decode i={i}: got {got}, want {want}");
            failures += 1;
        }
    }

    // ---- device -> host: the host decodes device-constructed enums ----
    let h_src: Vec<u32> = (0..N as u32).collect();
    let d_src = DeviceBuffer::from_host(&stream, &h_src).unwrap();
    let mut d_encoded = DeviceBuffer::<E>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.encode(stream.as_ref(), cfg, &d_src, &mut d_encoded) }.expect("launch encode");
    let encoded = d_encoded.to_host_vec(&stream).unwrap();
    for (i, v) in encoded.iter().enumerate() {
        let s = h_src[i];
        let ok = match (s % 3, v) {
            (0, E::A(x)) => *x == s,
            (1, E::B(f)) => *f == s as f32,
            (2, E::C) => true,
            _ => false,
        };
        if !ok {
            println!(
                "FAIL encode i={i}: variant/payload mismatch (tag bytes {:?})",
                &unsafe { std::mem::transmute::<E, [u8; 8]>(*v) }[..4]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("enum_abi_multi_payload: PASS ({N} threads; both ABI directions byte-faithful)");
    } else {
        println!("enum_abi_multi_payload: {failures} FAILURES");
        std::process::exit(1);
    }
}
