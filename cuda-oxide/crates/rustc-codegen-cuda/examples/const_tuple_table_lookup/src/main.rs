/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reading one field of a runtime-indexed tuple-array constant must stay
//! address-based to the final scalar load, not copy the whole table.
//!
//! `const_table_lookup` fixed this for a *scalar* table (`const T: [f32; N]`,
//! #684). This is the array-of-**tuple** shape: `const PAIRS: [(u8, u32);
//! 256]`, read as `let (a, b) = PAIRS[i]`. Before this fix, `mir.field_addr`
//! verified only struct, union and enum pointees, so a tuple field
//! projection fell back to the value path: the whole 256-entry table was
//! loaded as one first-class-aggregate value (which LLVM splits back into a
//! per-element store), **once per field projected**. Reading both `a` and
//! `b` from the same element cost two independent whole-table copies.
//!
//! Measured on an RTX 5060 (sm_120), same example, `cargo oxide inspect`,
//! before and after this diff with nothing else changed: `st.local` count
//! 878 -> 512, local depot 4096 -> 2048 bytes. Both fields now resolve
//! through one shared `mir.field_addr`-computed element address; the
//! remaining 512 stores are the table's own base-array materialization into
//! the depot -- a separate, pre-existing limit of #684's byte-image path,
//! which only trusts primitive-scalar or nested-array elements, so a
//! tuple-element table keeps its own per-thread copy until that is extended.
//!
//! `tuple_field_store` covers the WRITE side the same verifier change
//! unlocks: `arr[j].1 = x` through a runtime index and a write through a
//! `&mut` tuple-field borrow both previously failed dialect verification
//! loudly. The reordered `(u8, u32)` element makes the check bit-exact on
//! the memory-slot vs declaration-index distinction for stores.
//!
//! `sum_lookup` reads a table this fix does NOT change (`ROW: [u32; 4]`,
//! scalar elements, single index) as a same-run contrast: its lowering is
//! untouched by this diff, and its correctness check rules out an unrelated
//! regression in the ordinary array-constant path.
//!
//! Run: `cargo oxide run const_tuple_table_lookup`

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: u32 = 1 << 14;
const BLOCK: u32 = 256;
const TABLE_LEN: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    const PAIRS: [(u8, u32); TABLE_LEN] = {
        let mut t = [(0u8, 0u32); TABLE_LEN];
        let mut i = 0;
        while i < TABLE_LEN {
            t[i] = (i as u8, (i as u32).wrapping_mul(3).wrapping_add(1));
            i += 1;
        }
        t
    };

    const ROW: [u32; 4] = [11, 22, 33, 44];

    /// `let (a, b) = PAIRS[idx]` -- the shape this fix addresses.
    #[kernel]
    pub fn tuple_field_lookup(indices: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= indices.len() {
            return;
        }
        let (a, b) = PAIRS[(indices[i] as usize) & (TABLE_LEN - 1)];
        if let Some(o) = out.get_mut(idx) {
            *o = a as u32 + b;
        }
    }

    /// The WRITE side of the same verifier unlock: `arr[j].1 = x` and a
    /// `&mut` tuple-field borrow both lower to `mir.field_addr` + `mir.store`
    /// on a tuple pointee, which the verifier rejected before this fix. The
    /// `(u8, u32)` element is rustc-reordered (u32 first in memory), so a
    /// store that confused the declaration index with the memory slot would
    /// garble the neighbouring field and fail the bit-exact check below.
    #[kernel]
    pub fn tuple_field_store(indices: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= indices.len() {
            return;
        }
        let t = (indices[i] as usize) & (TABLE_LEN - 1);
        // Copy two neighbouring table entries into a local tuple array so
        // both a written and an untouched element are checked.
        let mut arr = [PAIRS[t], PAIRS[(t + 1) & (TABLE_LEN - 1)]];
        let j = t & 1;
        // Tuple-field write through a runtime index: `arr[j].1 = ...` with a
        // thread-unique value.
        arr[j].1 = indices[i].wrapping_mul(2_246_822_519);
        // Tuple-field write through a `&mut` borrow.
        let first = &mut arr[j].0;
        *first = (indices[i] >> 24) as u8;
        if let Some(o) = out.get_mut(idx) {
            *o = arr[j].0 as u32 + arr[j].1 + arr[1 - j].0 as u32 + arr[1 - j].1;
        }
    }

    /// A single-index scalar-array lookup, untouched by this diff, run
    /// alongside as a same-run contrast.
    #[kernel]
    pub fn sum_lookup(indices: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= indices.len() {
            return;
        }
        let value = ROW[(indices[i] as usize) & 3];
        if let Some(o) = out.get_mut(idx) {
            *o = value;
        }
    }
}

fn cpu_pairs() -> [(u8, u32); TABLE_LEN] {
    let mut t = [(0u8, 0u32); TABLE_LEN];
    for (i, entry) in t.iter_mut().enumerate() {
        *entry = (i as u8, (i as u32).wrapping_mul(3).wrapping_add(1));
    }
    t
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    // Deterministic, lane-divergent index spread (a multiplicative hash),
    // not a uniform pattern every warp lane would share.
    let indices_host: Vec<u32> = (0..N).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let indices = DeviceBuffer::from_host(&stream, &indices_host)?;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N as usize)?;
    let config = LaunchConfig {
        grid_dim: (N.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the grid covers exactly N elements and both buffers hold them.
    unsafe {
        module.tuple_field_lookup(&stream, config, &indices, &mut out)?;
    }
    stream.synchronize()?;

    let pairs = cpu_pairs();
    let got = out.to_host_vec(&stream)?;
    for (i, value) in got.iter().enumerate() {
        let (a, b) = pairs[(indices_host[i] as usize) & (TABLE_LEN - 1)];
        let expected = a as u32 + b;
        if *value != expected {
            return Err(
                format!("tuple_field_lookup: element {i} is {value}, expected {expected}").into(),
            );
        }
    }
    println!("tuple_field_lookup: {N} elements, exact match");

    let mut store_out = DeviceBuffer::<u32>::zeroed(&stream, N as usize)?;
    // SAFETY: the grid covers exactly N elements and both buffers hold them.
    unsafe {
        module.tuple_field_store(&stream, config, &indices, &mut store_out)?;
    }
    stream.synchronize()?;

    let store_got = store_out.to_host_vec(&stream)?;
    for (i, value) in store_got.iter().enumerate() {
        let t = (indices_host[i] as usize) & (TABLE_LEN - 1);
        let mut arr = [pairs[t], pairs[(t + 1) & (TABLE_LEN - 1)]];
        let j = t & 1;
        arr[j].1 = indices_host[i].wrapping_mul(2_246_822_519);
        arr[j].0 = (indices_host[i] >> 24) as u8;
        let expected = arr[j].0 as u32 + arr[j].1 + arr[1 - j].0 as u32 + arr[1 - j].1;
        if *value != expected {
            return Err(
                format!("tuple_field_store: element {i} is {value}, expected {expected}").into(),
            );
        }
    }
    println!("tuple_field_store: {N} elements, exact match");

    let mut sum_out = DeviceBuffer::<u32>::zeroed(&stream, N as usize)?;
    // SAFETY: the grid covers exactly N elements and both buffers hold them.
    unsafe {
        module.sum_lookup(&stream, config, &indices, &mut sum_out)?;
    }
    stream.synchronize()?;

    let row = [11u32, 22, 33, 44];
    let sum_got = sum_out.to_host_vec(&stream)?;
    for (i, value) in sum_got.iter().enumerate() {
        let expected = row[(indices_host[i] as usize) & 3];
        if *value != expected {
            return Err(format!("sum_lookup: element {i} is {value}, expected {expected}").into());
        }
    }
    println!("sum_lookup: {N} elements, exact match");

    println!("SUCCESS: tuple-field table lookups match the CPU reference");
    Ok(())
}
