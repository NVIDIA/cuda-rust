/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for MIR import of aggregate constants that rustc pads or
//! reorders.
//!
//! A constant's bytes are the memory image of its type, so each field starts at
//! its layout offset. `{ u8, u64 }` puts the `u64` at offset 8, not offset 1.
//! Reading the fields off as if they were packed in declaration order silently
//! yields the wrong values: the device saw `Padded { a: 200, b: 0x7d25933d707303e5 }`
//! as `a = 229` (the low byte of `b`) and `b = 0xc87d25933d707303` (`b` shifted by
//! a byte), with no diagnostic.
//!
//! Covered shapes:
//! - a struct constant with padding between its fields,
//! - the same shape as a tuple constant,
//! - a three-width struct whose offsets follow from neither order nor size,
//! - a nested padded struct,
//! - a struct holding an array of padded structs (element stride is the
//!   padded size, not the field sum),
//! - a `#[repr(C)]` struct with interior padding and a trailing field after it,
//! - a padded struct with a float field, which lowers on a separate path,
//! - a descending-alignment struct, which needs no padding and passed already.
//!
//! Run with:
//!   cargo oxide run padded_constants

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Ascending alignment, so rustc leaves 7 bytes of padding after `a`.
#[derive(Clone, Copy)]
pub struct Padded {
    pub a: u8,
    pub b: u64,
}

/// Three fields of three widths, so no two offsets follow from the sizes alone.
#[derive(Clone, Copy)]
pub struct Wide {
    pub a: u16,
    pub b: u32,
    pub c: u8,
}

/// A padded struct inside another, so the inner offsets are relative to a base
/// that is itself not zero.
#[derive(Clone, Copy)]
pub struct Nested {
    pub tag: u8,
    pub inner: Padded,
}

/// The float field lowers through its own branch of the constant walker.
#[derive(Clone, Copy)]
pub struct Floaty {
    pub a: u8,
    pub f: f32,
}

/// Descending alignment: no padding, no reordering. This shape was already
/// correct, and is here so a fix cannot regress it.
#[derive(Clone, Copy)]
pub struct Descending {
    pub b: u64,
    pub a: u8,
}

/// An array of padded structs. The element stride is the struct's recorded size,
/// so a stride taken from the field sum reads every element after the first from
/// the wrong offset. `Padded` is 16 bytes wide and its fields sum to 9.
#[derive(Clone, Copy)]
pub struct Batch {
    pub arr: [Padded; 2],
}

/// `#[repr(C)]` fixes declaration order, so the padding follows the C rule rather
/// than rustc's own choice. The offsets still have to come from the layout.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ReprC {
    pub a: u8,
    pub b: u32,
    pub c: u8,
}

const B: u64 = 9017775720968160229; // 0x7d25933d707303e5

const PADDED: Padded = Padded { a: 200, b: B };
const WIDE: Wide = Wide {
    a: 41292,
    b: 305419896,
    c: 200,
};
const NESTED: Nested = Nested {
    tag: 7,
    inner: Padded { a: 201, b: B },
};
const FLOATY: Floaty = Floaty { a: 202, f: 1.25 };
const DESCENDING: Descending = Descending { b: B, a: 203 };
const TUPLE: (u8, u64) = (204, B);
const BATCH: Batch = Batch {
    arr: [Padded { a: 205, b: B }, Padded { a: 206, b: !B }],
};
const REPR_C: ReprC = ReprC {
    a: 207,
    b: 305419897,
    c: 208,
};

/// One field per thread, so every field is read on the device rather than folded
/// into a single host-computed answer.
const FIELDS: usize = 21;

#[cuda_module]
mod kernels {
    use super::*;

    // Each constant is passed *by value* into an `#[inline(never)]` function.
    // Reading a field of a constant directly (`PADDED.b`) is const-folded before
    // it ever reaches the importer, so it would exercise nothing: the aggregate
    // has to be materialised, which is what crossing this call boundary forces.

    #[inline(never)]
    fn take_padded(s: Padded, sel: usize) -> u64 {
        if sel == 0 { s.a as u64 } else { s.b }
    }

    #[inline(never)]
    fn take_wide(s: Wide, sel: usize) -> u64 {
        match sel {
            0 => s.a as u64,
            1 => s.b as u64,
            _ => s.c as u64,
        }
    }

    #[inline(never)]
    fn take_nested(s: Nested, sel: usize) -> u64 {
        match sel {
            0 => s.tag as u64,
            1 => s.inner.b,
            _ => s.inner.a as u64,
        }
    }

    #[inline(never)]
    fn take_batch(s: Batch, sel: usize) -> u64 {
        match sel {
            0 => s.arr[0].a as u64,
            1 => s.arr[0].b,
            2 => s.arr[1].a as u64,
            _ => s.arr[1].b,
        }
    }

    /// Promoted reference-to-struct constant: `&(8..16)` reaches the importer
    /// as a thin reference whose provenance names the allocation holding the
    /// Range's bytes. Pins the by-ref decode path, which must follow the
    /// indirection and query layout on the pointee, not the reference.
    #[inline(never)]
    fn take_promoted_range(sel: usize) -> u64 {
        let r: &core::ops::Range<u64> = &(8..16);
        if sel == 0 { r.start } else { r.end }
    }

    #[inline(never)]
    fn take_repr_c(s: ReprC, sel: usize) -> u64 {
        match sel {
            0 => s.a as u64,
            1 => s.b as u64,
            _ => s.c as u64,
        }
    }

    #[inline(never)]
    fn take_descending(s: Descending, sel: usize) -> u64 {
        if sel == 0 { s.a as u64 } else { s.b }
    }

    #[inline(never)]
    fn take_tuple(t: (u8, u64), sel: usize) -> u64 {
        if sel == 0 { t.0 as u64 } else { t.1 }
    }

    #[inline(never)]
    fn take_floaty(s: Floaty, sel: usize) -> f32 {
        if sel == 0 { s.f } else { s.a as f32 }
    }

    #[inline(never)]
    fn field_u64(i: usize) -> u64 {
        match i % FIELDS {
            0 => take_padded(PADDED, 0),
            1 => take_padded(PADDED, 1),
            2 => take_wide(WIDE, 0),
            3 => take_wide(WIDE, 1),
            4 => take_wide(WIDE, 2),
            5 => take_nested(NESTED, 0),
            6 => take_nested(NESTED, 1),
            7 => take_descending(DESCENDING, 1),
            8 => take_tuple(TUPLE, 0),
            9 => take_tuple(TUPLE, 1),
            10 => take_nested(NESTED, 2),
            11 => take_descending(DESCENDING, 0),
            12 => take_batch(BATCH, 0),
            13 => take_batch(BATCH, 1),
            14 => take_batch(BATCH, 2),
            15 => take_batch(BATCH, 3),
            16 => take_repr_c(REPR_C, 0),
            17 => take_repr_c(REPR_C, 1),
            18 => take_repr_c(REPR_C, 2),
            19 => take_promoted_range(0),
            _ => take_promoted_range(1),
        }
    }

    #[inline(never)]
    fn field_f32(i: usize) -> f32 {
        take_floaty(FLOATY, i % 2)
    }

    #[kernel]
    pub fn check_padded_constants(
        mut out_u64: DisjointSlice<u64>,
        mut out_f32: DisjointSlice<f32>,
    ) {
        let tid = thread::index_1d();
        let i = tid.get();

        if let Some(slot) = out_u64.get_mut(tid) {
            *slot = field_u64(i);
        }

        let tid_f32 = thread::index_1d();
        if let Some(slot) = out_f32.get_mut(tid_f32) {
            *slot = field_f32(i);
        }
    }
}

fn expected_u64(i: usize) -> u64 {
    match i % FIELDS {
        0 => PADDED.a as u64,
        1 => PADDED.b,
        2 => WIDE.a as u64,
        3 => WIDE.b as u64,
        4 => WIDE.c as u64,
        5 => NESTED.tag as u64,
        6 => NESTED.inner.b,
        7 => DESCENDING.b,
        8 => TUPLE.0 as u64,
        9 => TUPLE.1,
        10 => NESTED.inner.a as u64,
        11 => DESCENDING.a as u64,
        12 => BATCH.arr[0].a as u64,
        13 => BATCH.arr[0].b,
        14 => BATCH.arr[1].a as u64,
        15 => BATCH.arr[1].b,
        16 => REPR_C.a as u64,
        17 => REPR_C.b as u64,
        18 => REPR_C.c as u64,
        19 => 8,
        _ => 16,
    }
}

fn expected_f32(i: usize) -> f32 {
    if i.is_multiple_of(2) {
        FLOATY.f
    } else {
        FLOATY.a as f32
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== padded_constants regression ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = FIELDS * 3;
    let mut out_u64 = DeviceBuffer::<u64>::zeroed(&stream, N)?;
    let mut out_f32 = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this is a 1D launch and the kernel bounds-checks each output
    // access against the corresponding slice length.
    unsafe {
        module.check_padded_constants(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out_u64,
            &mut out_f32,
        )
    }?;

    let got_u64 = out_u64.to_host_vec(&stream)?;
    let got_f32 = out_f32.to_host_vec(&stream)?;

    let mut failures = 0usize;
    for i in 0..N {
        let want_u64 = expected_u64(i);
        if got_u64[i] != want_u64 {
            println!(
                "FAIL integer field tid={i}: got={:#x} expected={:#x}",
                got_u64[i], want_u64
            );
            failures += 1;
        }

        let want_f32 = expected_f32(i);
        if got_f32[i] != want_f32 {
            println!(
                "FAIL float field tid={i}: got={} expected={}",
                got_f32[i], want_f32
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!(
            "padded_constants: PASS ({N} threads; padded struct, tuple, nested, float, descending, array-of-padded, repr(C))"
        );
        Ok(())
    } else {
        println!("padded_constants: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
