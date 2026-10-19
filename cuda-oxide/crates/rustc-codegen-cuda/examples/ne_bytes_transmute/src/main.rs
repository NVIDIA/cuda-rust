/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression test for issue #125: `u32::from_ne_bytes` in a kernel is a
// MIR Transmute from `[u8; 4]` to `u32`. The cast lowering used to fall
// through to `bitcast [4 x i8] %v to i32`, which LLVM rejects because
// bitcast is only defined between non-aggregate first-class types. The
// fix lowers any array-involving transmute through a stack slot
// (alloca + store + load), which the optimizer folds away.
//
// Kernels lock in both directions for u32 and f32, plus `f32::to_bits`,
// a same-width scalar transmute that must stay on the plain bitcast path.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// `[u8; 4]` -> `u32`: the exact shape from issue #125.
    /// Builds the byte array from a slice, then transmutes it to a word.
    #[kernel]
    pub fn u32_from_ne_bytes(bytes: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let base = 4 * i;
            let b = [
                bytes[base],
                bytes[base + 1],
                bytes[base + 2],
                bytes[base + 3],
            ];
            *out_elem = u32::from_ne_bytes(b);
        }
    }

    /// `u32` -> `[u8; 4]`: the reverse direction. Each byte is widened
    /// and repacked so the host can verify all four lanes in one u32.
    #[kernel]
    pub fn u32_to_ne_bytes(words: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let b = words[i].to_ne_bytes();
            *out_elem = (b[0] as u32)
                | ((b[1] as u32) << 8)
                | ((b[2] as u32) << 16)
                | ((b[3] as u32) << 24);
        }
    }

    /// `[u8; 4]` -> `f32`: same aggregate-to-scalar transmute with a
    /// float destination (`f32::from_ne_bytes` goes through `u32` and a
    /// scalar int-to-float bitcast).
    #[kernel]
    pub fn f32_from_ne_bytes(bytes: &[u8], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let base = 4 * i;
            let b = [
                bytes[base],
                bytes[base + 1],
                bytes[base + 2],
                bytes[base + 3],
            ];
            *out_elem = f32::from_ne_bytes(b);
        }
    }

    /// `f32` -> `[u8; 4]`, repacked the same way as `u32_to_ne_bytes`.
    #[kernel]
    pub fn f32_to_ne_bytes(vals: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let b = vals[i].to_ne_bytes();
            *out_elem = (b[0] as u32)
                | ((b[1] as u32) << 8)
                | ((b[2] as u32) << 16)
                | ((b[3] as u32) << 24);
        }
    }

    /// `f32` -> `u32` via `to_bits`: a same-width scalar transmute that
    /// needs no memory round-trip. Guards against the aggregate handling
    /// over-matching and degrading the plain bitcast path.
    #[kernel]
    pub fn f32_to_bits(vals: &[f32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = vals[i].to_bits();
        }
    }
}

/// Host-side reference for the device repack formula: widen each native-
/// endian byte and OR it into place. Matching formulas on both sides keep
/// the check endianness-agnostic.
fn repack(b: [u8; 4]) -> u32 {
    (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24)
}

fn check(name: &str, got: &[u32], expect: &[u32]) -> bool {
    let ok = got == expect;
    println!(
        "{name}: got {:08X?}  expect {:08X?}  {}",
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

    let words: Vec<u32> = vec![0xDEAD_BEEF, 0x0000_0001, 0x1234_5678, 0xFFFF_FFFF];
    let floats: Vec<f32> = vec![1.0, -2.5, 6.022e23, f32::MIN_POSITIVE];
    let n = words.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    // Flatten the native-endian bytes of each input on the host.
    let word_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let float_bytes: Vec<u8> = floats.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let mut ok = true;

    // [u8; 4] -> u32 (issue #125's failing shape).
    {
        let bytes_d = DeviceBuffer::from_host(&stream, &word_bytes).unwrap();
        let mut out_d = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { m.u32_from_ne_bytes(&stream, cfg, &bytes_d, &mut out_d) }
            .expect("launch u32_from_ne_bytes");
        let got = out_d.to_host_vec(&stream).unwrap();
        ok &= check("u32::from_ne_bytes", &got, &words);
    }

    // u32 -> [u8; 4].
    {
        let words_d = DeviceBuffer::from_host(&stream, &words).unwrap();
        let mut out_d = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { m.u32_to_ne_bytes(&stream, cfg, &words_d, &mut out_d) }
            .expect("launch u32_to_ne_bytes");
        let got = out_d.to_host_vec(&stream).unwrap();
        let expect: Vec<u32> = words.iter().map(|w| repack(w.to_ne_bytes())).collect();
        ok &= check("u32::to_ne_bytes  ", &got, &expect);
    }

    // [u8; 4] -> f32 (compare bit patterns to avoid float-eq pitfalls).
    {
        let bytes_d = DeviceBuffer::from_host(&stream, &float_bytes).unwrap();
        let mut out_d = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { m.f32_from_ne_bytes(&stream, cfg, &bytes_d, &mut out_d) }
            .expect("launch f32_from_ne_bytes");
        let got: Vec<u32> = out_d
            .to_host_vec(&stream)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let expect: Vec<u32> = floats.iter().map(|v| v.to_bits()).collect();
        ok &= check("f32::from_ne_bytes", &got, &expect);
    }

    // f32 -> [u8; 4].
    {
        let floats_d = DeviceBuffer::from_host(&stream, &floats).unwrap();
        let mut out_d = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { m.f32_to_ne_bytes(&stream, cfg, &floats_d, &mut out_d) }
            .expect("launch f32_to_ne_bytes");
        let got = out_d.to_host_vec(&stream).unwrap();
        let expect: Vec<u32> = floats.iter().map(|v| repack(v.to_ne_bytes())).collect();
        ok &= check("f32::to_ne_bytes  ", &got, &expect);
    }

    // f32 -> u32 scalar bitcast path (must keep working untouched).
    {
        let floats_d = DeviceBuffer::from_host(&stream, &floats).unwrap();
        let mut out_d = DeviceBuffer::<u32>::zeroed(&stream, n).unwrap();
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { m.f32_to_bits(&stream, cfg, &floats_d, &mut out_d) }.expect("launch f32_to_bits");
        let got = out_d.to_host_vec(&stream).unwrap();
        let expect: Vec<u32> = floats.iter().map(|v| v.to_bits()).collect();
        ok &= check("f32::to_bits      ", &got, &expect);
    }

    println!(
        "{}",
        if ok {
            "SUCCESS: ne_bytes transmutes round-trip correctly"
        } else {
            "FAILURE: ne_bytes transmute produced wrong bytes"
        }
    );
    std::process::exit(if ok { 0 } else { 1 });
}
