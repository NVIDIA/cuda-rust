/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mutating an enum payload in place, through `&mut` and through assignment.
//!
//! Both forms need the address of the payload. Without enum payload addressing
//! the importer had no way to compute it: a mutable borrow was refused outright
//! rather than silently copied, and `(x as Variant).field = v` was rejected as
//! an unimplemented projection pair.
//!
//! The kernels below cover the paths that differ in lowering:
//!
//! - `assign_payload` writes through `(x as Variant).field = v`.
//! - `borrow_payload` takes `&mut` and hands it to a `#[device]` helper, so the
//!   borrow survives into a call and cannot fold into a direct store.
//! - `shared_bytes` uses an enum whose two payload variants hold different
//!   types at the same offset, so at most one of them has an LLVM slot of its
//!   own and the other is addressed by byte offset.
//! - `shared_bytes_no_slot` mutates the variant WITHOUT a slot of its own
//!   (`Bits` shares `Real`'s bytes), so the byte-offset addressing path runs
//!   against the original storage.
//! - `mutate_indexed_payload` mutates a runtime-indexed element inside an enum
//!   array payload, covering the composed `Downcast -> Field -> Index` store path.
//! - `shared_borrow_bool_payload` takes a SHARED borrow of a `bool` payload
//!   and passes it to a `#[device]` helper. A bool payload's bytes use
//!   canonical i8 storage, so no raw payload address can be handed out; the
//!   importer must fall back to a sound value copy for the read, exactly as
//!   it did before payload addressing existed. (Mutable borrows of such
//!   payloads stay rejected; see the `error_enum_bool_payload_addr` fixture.)
//! - `rebuild_payload` is the workaround this replaces, kept as a baseline.
//!
//! Each kernel reads its value back after mutating, so a write that landed in
//! a copy shows up as an unchanged element rather than passing quietly.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};
use std::time::Instant;

const LEN: u32 = 1 << 20;
const BLOCK: u32 = 256;
const RUNS: u32 = 20;

#[cuda_module]
mod kernels {
    use super::*;

    pub enum Slot {
        Occupied(f32),
        Empty,
    }

    /// Two payloads of different types sharing the same bytes, so at most one
    /// of them gets an LLVM slot and the other is addressed by offset.
    pub enum Either {
        Real(f32),
        Bits(u32),
    }

    /// A bool payload: semantically i1, physically a canonical i8 byte
    /// inside enum storage, so no raw payload address can represent it.
    pub enum Flag {
        On(bool),
        Off,
    }

    /// A variant of two fields, one of them canonical storage. Rebuilding
    /// around the bool has to carry the `f32` through untouched.
    pub enum Pair {
        Both(bool, f32),
        #[allow(dead_code)]
        Neither,
    }

    /// Array payload used to cover a nested `Downcast -> Field -> Index` store.
    pub enum Bucket {
        Data([u32; 4]),
        #[allow(dead_code)]
        Empty,
    }

    /// Scale a borrowed payload. Taking `&mut f32` across a call boundary
    /// keeps the borrow from folding into a plain store.
    #[device]
    pub fn scale_in_place(value: &mut f32, factor: f32) {
        *value *= factor;
    }

    /// Read a borrowed bool. Taking `&bool` across a call boundary keeps
    /// the shared borrow from folding into a direct payload read.
    #[device]
    pub fn read_bool(b: &bool) -> u32 {
        if *b { 1 } else { 0 }
    }

    /// Write through `(slot as Occupied).0 = v`.
    #[kernel]
    pub fn assign_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut slot = Slot::Occupied(0.0);
        if let Slot::Occupied(value) = &mut slot {
            *value = input[i] * 2.0;
        }
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Pass `&mut` to the payload into a helper.
    #[kernel]
    pub fn borrow_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut slot = Slot::Occupied(input[i]);
        if let Slot::Occupied(value) = &mut slot {
            scale_in_place(value, 2.0);
        }
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Mutate the payload of an enum whose variants share bytes.
    #[kernel]
    pub fn shared_bytes(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut either = Either::Real(input[i]);
        if let Either::Real(value) = &mut either {
            *value *= 2.0;
        }
        let result = match either {
            Either::Real(value) => value,
            Either::Bits(bits) => f32::from_bits(bits),
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Mutate the payload that has NO slot of its own. `Real`'s f32 claims
    /// the shared bytes first, so `Bits` is addressed by byte offset off the
    /// original enum storage; a write landing in a copy (or at the wrong
    /// offset) shows up as an unchanged or corrupted element.
    #[kernel]
    pub fn shared_bytes_no_slot(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut either = Either::Bits(input[i].to_bits());
        if let Either::Bits(bits) = &mut either {
            *bits = (f32::from_bits(*bits) * 2.0).to_bits();
        }
        let result = match either {
            Either::Real(value) => value,
            Either::Bits(bits) => f32::from_bits(bits),
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// SHARED borrow of a bool payload, handed to a `#[device]` helper so it
    /// survives MIR optimization into the importer. Canonical i8 storage
    /// means the payload has no honest raw address; the importer's address
    /// walker must punt and read through a sound value copy instead. Each
    /// input maps to a distinct output (On(true) doubles, On(false) triples,
    /// Off quadruples), so reading the wrong byte or the wrong variant shows
    /// up as a mismatched element.
    #[kernel]
    pub fn shared_borrow_bool_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let flag = if input[i] >= 100.0 {
            Flag::On(input[i] >= 250.0)
        } else {
            Flag::Off
        };
        let result = if let Flag::On(b) = &flag {
            if read_bool(b) == 1 {
                input[i] * 2.0
            } else {
                input[i] * 3.0
            }
        } else {
            input[i] * 4.0
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Write a bool payload without the borrow leaving the kernel.
    ///
    /// The borrow folds into a store to the payload place, which has no
    /// address to write through: the byte is canonical `i8` storage while the
    /// value is `i1`. The importer rebuilds the enum around the new payload
    /// instead, so the write lands in the enum itself. Each input maps to a
    /// distinct output, so a dropped write shows up as an unchanged element.
    #[kernel]
    pub fn mutate_bool_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut flag = Flag::On(false);
        if let Flag::On(value) = &mut flag {
            *value = true;
        }
        let result = match flag {
            // Only the written value gives the expected element, so a write
            // that landed in a copy reports a mismatch.
            Flag::On(true) => input[i] * 2.0,
            Flag::On(false) => -input[i],
            Flag::Off => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Write one field of a two-field variant, where the other must survive.
    ///
    /// The rebuild reads the sibling `f32` back out of the current value and
    /// passes it through, so a rebuild that dropped it would return the
    /// variant's default rather than the input.
    #[kernel]
    pub fn mutate_multi_field_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let mut pair = Pair::Both(false, input[i] * 2.0);
        if let Pair::Both(flag, _) = &mut pair {
            *flag = true;
        }
        let result = match pair {
            // The bool must have been written and the f32 must have survived.
            Pair::Both(true, carried) => carried,
            Pair::Both(false, _) => f32::NAN,
            Pair::Neither => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// Mutate a runtime-indexed element inside an enum's array payload.
    ///
    /// The indexed store exercises the composed `Downcast -> Field -> Index`
    /// place path.
    #[kernel]
    pub fn mutate_indexed_payload(mut out: DisjointSlice<u32>, n: u32, column: u32, base: u32) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= n as usize {
            return;
        }

        let c = (column as usize) & 3;
        let mut bucket = Bucket::Data([0x10, 0x20, 0x30, 0x40]);

        if let Bucket::Data(values) = &mut bucket {
            values[c] = base.wrapping_add(i as u32);
        }

        let result = match bucket {
            Bucket::Data(values) => values[c],
            Bucket::Empty => u32::MAX,
        };

        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }

    /// The workaround this replaces: rebuild the enum from a matched copy.
    #[kernel]
    pub fn rebuild_payload(input: &[f32], mut out: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= input.len() {
            return;
        }
        let value = input[i];
        let slot = if value.is_nan() {
            Slot::Empty
        } else {
            Slot::Occupied(value)
        };
        let slot = match slot {
            Slot::Occupied(value) => Slot::Occupied(value * 2.0),
            Slot::Empty => Slot::Empty,
        };
        let result = match slot {
            Slot::Occupied(value) => value,
            Slot::Empty => f32::NAN,
        };
        if let Some(cell) = out.get_mut(index) {
            *cell = result;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let host: Vec<f32> = (0..LEN).map(|i| (i % 1000) as f32 * 0.5).collect();
    let input = DeviceBuffer::from_host(&stream, &host)?;
    let config = LaunchConfig {
        grid_dim: (LEN.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let check = |name: &str,
                 run: &dyn Fn(&mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>|
     -> Result<(), Box<dyn std::error::Error>> {
        // Fill with a sentinel, so a kernel that never wrote is not mistaken
        // for one that wrote the right answer.
        let mut out = DeviceBuffer::from_host(&stream, &vec![f32::MIN; LEN as usize])?;
        run(&mut out)?;
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for (i, value) in got.iter().enumerate() {
            let expected = host[i] * 2.0;
            if (value - expected).abs() > 1e-6 {
                return Err(format!("{name}: element {i} is {value}, expected {expected}").into());
            }
        }
        println!("{name}: {LEN} payloads mutated in place, exact match");
        Ok(())
    };

    // SAFETY for each launch below: the grid covers exactly `LEN` elements and
    // both buffers hold that many.
    check("assign_payload", &|out| unsafe {
        module.assign_payload(&stream, config, &input, out)
    })?;
    check("borrow_payload", &|out| unsafe {
        module.borrow_payload(&stream, config, &input, out)
    })?;
    check("shared_bytes", &|out| unsafe {
        module.shared_bytes(&stream, config, &input, out)
    })?;
    check("shared_bytes_no_slot", &|out| unsafe {
        module.shared_bytes_no_slot(&stream, config, &input, out)
    })?;
    check("mutate_bool_payload", &|out| unsafe {
        module.mutate_bool_payload(&stream, config, &input, out)
    })?;
    check("mutate_multi_field_payload", &|out| unsafe {
        module.mutate_multi_field_payload(&stream, config, &input, out)
    })?;
    check("rebuild_payload", &|out| unsafe {
        module.rebuild_payload(&stream, config, &input, out)
    })?;

    // Focused nested enum-payload probe. The source is intentionally separate
    // from the f32 checks because its payload and oracle are u32.
    {
        const COLUMN: u32 = 3;
        const BASE: u32 = 0x1234_0000;
        let mut out = DeviceBuffer::from_host(&stream, &vec![u32::MAX; LEN as usize])?;
        // SAFETY: the grid covers exactly `LEN` elements, the output buffer has
        // that many entries, and the kernel masks COLUMN to the payload bounds.
        unsafe { module.mutate_indexed_payload(&stream, config, &mut out, LEN, COLUMN, BASE)? };
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for (i, value) in got.iter().enumerate() {
            let expected = BASE.wrapping_add(i as u32);
            if *value != expected {
                return Err(format!(
                    "mutate_indexed_payload: element {i} is {value}, expected {expected}"
                )
                .into());
            }
        }
        println!(
            "mutate_indexed_payload: {LEN} indexed enum payload elements mutated, exact match"
        );
    }

    // The bool-payload kernel maps each input to a variant-dependent output,
    // so it gets its own expectation instead of the uniform `* 2.0` check.
    {
        let mut out = DeviceBuffer::from_host(&stream, &vec![f32::MIN; LEN as usize])?;
        // SAFETY: the grid covers exactly `LEN` elements and both buffers
        // hold that many.
        unsafe { module.shared_borrow_bool_payload(&stream, config, &input, &mut out)? };
        stream.synchronize()?;
        let got = out.to_host_vec(&stream)?;
        for (i, value) in got.iter().enumerate() {
            let x = host[i];
            let expected = if x >= 100.0 {
                if x >= 250.0 { x * 2.0 } else { x * 3.0 }
            } else {
                x * 4.0
            };
            if (value - expected).abs() > 1e-6 {
                return Err(format!(
                    "shared_borrow_bool_payload: element {i} is {value}, expected {expected}"
                )
                .into());
            }
        }
        println!(
            "shared_borrow_bool_payload: {LEN} bool payloads read through a copy, exact match"
        );
    }

    // In-place mutation against the rebuild-from-a-copy workaround.
    let mut out = DeviceBuffer::from_host(&stream, &vec![0.0f32; LEN as usize])?;
    let mut time = |label: &str,
                    run: &dyn Fn(&mut DeviceBuffer<f32>) -> Result<(), cuda_core::DriverError>|
     -> Result<f64, Box<dyn std::error::Error>> {
        run(&mut out)?;
        stream.synchronize()?;
        let start = Instant::now();
        for _ in 0..RUNS {
            run(&mut out)?;
        }
        stream.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        println!("  {label:<18} {ms:7.4} ms");
        Ok(ms)
    };

    println!("\n{LEN} elements, {RUNS} timed runs:");
    let in_place = time("borrow in place", &|out| unsafe {
        module.borrow_payload(&stream, config, &input, out)
    })?;
    let rebuilt = time("rebuild from copy", &|out| unsafe {
        module.rebuild_payload(&stream, config, &input, out)
    })?;
    println!(
        "  ratio in-place / rebuild: {:.3}",
        in_place / rebuilt.max(f64::MIN_POSITIVE)
    );

    println!("\nSUCCESS");
    Ok(())
}
