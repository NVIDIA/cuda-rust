// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Minimal reproduction for a rustc-codegen-cuda miscompile: reading from
 * `*const E` where `E` is a fieldless `#[repr(u32)] enum` strides by
 * 1 byte instead of the expected 4 bytes.
 *
 * Symptom:
 *   The kernel buffer at offset 4 reads as `1` via `*const u32`, but the
 *   same bytes read through a pointer to a fieldless `#[repr(u32)]` enum do
 *   not produce the slot-1 discriminant. The failure pattern matches 1-byte
 *   pointer stride instead of the expected 4-byte stride.
 *
 * Test design:
 *   Host buffer = [0u32, 1, 2, 3] - chosen so each slot's `as u32` value
 *   *is* a valid `Tag` discriminant. The kernel receives a single pointer
 *   and reads slot 0..=3 two ways:
 *
 *     a) `*const u32`  - control. Always reads stride 4. Should give 0..=3.
 *     b) `*const Tag`  - under test. If stride is correctly 4, gives
 *                        Tag::Foo, Bar, Baz, Qux (= 0,1,2,3
 *                        when cast back to u32). If stride is buggy (1),
 *                        slots 1..=3 read zero bytes inside the first u32.
 *
 * Output (current cuda-oxide spike/applied, expected):
 *   control_u32  [0, 1, 2, 3]  PASS
 *   enum_ptr     [0, 0, 0, 0]  FAIL (1-byte stride)
 *
 * After fix:
 *   control_u32  [0, 1, 2, 3]  PASS
 *   enum_ptr     [0, 1, 2, 3]  PASS
 *
 * The fix sources the discriminant from rustc's LAYOUT (not just the
 * variant count or the repr attribute), so this example also covers the
 * other tag shapes that fall out of the same layout query:
 *
 *   repr_c_enum    `#[repr(C)]` tag is C `int` (4 bytes on nvptx64)
 *   sparse_enum    default repr, `B = 1_000_000` forces a u32 tag
 *   neg_enum       default repr, `N = -1` forces a SIGNED i8 tag, so a
 *                  memory-loaded `e as i32` must sign-extend to -1
 *   usize_enum     `#[repr(usize)]` tag is pointer-width (8 bytes)
 *
 * Build:
 *   cargo oxide run repr_u32_enum_stride
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::fmt::Debug;
use std::sync::Arc;

/// Fieldless `#[repr(u32)]` enum used as a tag-like device-buffer element.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Tag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

// SAFETY: trivial repr(u32) POD, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for Tag {}

/// Fieldless `#[repr(C)]` enum: the tag is the C `int` type (4 bytes,
/// signed, on nvptx64). `repr().int` reports None for `#[repr(C)]`, so only
/// the layout query sees the 4-byte tag.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum CTag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

// SAFETY: trivial repr(C) POD, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for CTag {}

/// Default-repr enum with a sparse discriminant: 1_000_000 needs 32 bits,
/// so rustc picks a u32 tag where a variant-count guess picks u8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sparse {
    A = 0,
    B = 1_000_000,
}

// SAFETY: fieldless POD enum, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for Sparse {}

/// Default-repr enum with a negative discriminant: rustc picks a SIGNED
/// i8 tag, so a memory-loaded `e as i32` must sign-extend (-1, not 255).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Neg {
    N = -1,
    Z = 0,
}

// SAFETY: fieldless POD enum, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for Neg {}

/// Fieldless `#[repr(usize)]` enum: the tag is pointer-width (8 bytes on
/// nvptx64), so `*const UTag` must stride by 8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum UTag {
    Foo = 0,
    Bar = 1,
    Baz = 2,
    Qux = 3,
}

// SAFETY: trivial repr(usize) POD, safe to copy device↔host.
unsafe impl cuda_core::DeviceCopy for UTag {}

const N: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Control: read via `*const u32` with `add(i)`. Stride should be 4.
    /// Writes input[i] for i in 0..N.
    #[kernel]
    pub fn read_via_u32(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Read through a raw u32 pointer with arithmetic.
        let base: *const u32 = input.as_ptr();
        let v = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// Test: read via `*const Tag` with `add(i)`, then cast the discriminant
    /// back to u32. If stride is correctly 4 the output matches the u32
    /// control. If stride is buggy (1), slots 1..=3 read bytes inside the
    /// first u32 and come back as variant 0.
    #[kernel]
    pub fn read_via_enum(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        // Reinterpret the input buffer as `*const Tag`. The bytes are
        // identical (both u32-sized, repr(u32)); only pointer-arithmetic
        // stride is under test.
        let base: *const Tag = input.as_ptr() as *const Tag;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }

    /// `#[repr(C)]` stride test: the tag is the C `int` (4 bytes), so
    /// `*const CTag` must stride by 4 exactly like the u32 control.
    #[kernel]
    pub fn read_via_repr_c_enum(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        let base: *const CTag = input.as_ptr() as *const CTag;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }

    /// Default-repr sparse stride test: discriminant 1_000_000 forces a
    /// u32 tag, so `*const Sparse` must stride by 4 (a variant-count guess
    /// strides by 1 and also truncates the discriminant value).
    #[kernel]
    pub fn read_via_sparse_enum(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        let base: *const Sparse = input.as_ptr() as *const Sparse;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }

    /// Negative-discriminant sign test: `Neg` has a SIGNED i8 tag, so
    /// casting a memory-loaded value to i32 must sign-extend. With an
    /// unsigned-tag model the 0xFF byte zero-extends to 255 instead of -1.
    #[kernel]
    pub fn read_via_neg_enum(input: &[u8], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        let base: *const Neg = input.as_ptr() as *const Neg;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as i32;
        }
    }

    /// `#[repr(usize)]` stride test: the tag is pointer-width (8 bytes),
    /// so `*const UTag` must stride by 8 over a u64 buffer.
    #[kernel]
    pub fn read_via_usize_enum(input: &[u64], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= N {
            return;
        }
        let base: *const UTag = input.as_ptr() as *const UTag;
        let tag = unsafe { *base.add(i) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = tag as u32;
        }
    }
}

/// Run one kernel over `input`, compare against `expected`, report PASS/FAIL.
fn check<I, O, F>(
    name: &str,
    stream: &Arc<CudaStream>,
    input: &[I],
    expected: &[O],
    launch: F,
) -> bool
where
    I: DeviceCopy,
    O: DeviceCopy + PartialEq + Debug,
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &DeviceBuffer<I>, &mut DeviceBuffer<O>),
{
    let dev_in = DeviceBuffer::from_host(stream, input).unwrap();
    let mut dev_out = DeviceBuffer::<O>::zeroed(stream, expected.len()).unwrap();

    launch(
        stream,
        LaunchConfig::for_num_elems(expected.len() as u32),
        &dev_in,
        &mut dev_out,
    );

    let host_out = dev_out.to_host_vec(stream).unwrap();
    let pass = host_out == expected;
    let verdict = if pass { "PASS" } else { "FAIL" };
    println!("  {name:<14}  {host_out:?}   {verdict}");
    pass
}

fn main() {
    println!("=== enum discriminant layout (stride + signedness) repro ===\n");
    println!("Each kernel reads a device buffer through `*const E` with `add(i)`.");
    println!("Stride and signedness must come from rustc's layout for E.\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");

    let seq: Vec<u32> = (0..N as u32).collect();
    let mut all_pass = true;

    // Control: plain u32 reads, stride 4 regardless of the fix.
    all_pass &= check("control_u32", &stream, &seq, &seq, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.read_via_u32(s, cfg, i, o) }.expect("launch")
    });

    // #[repr(u32)]: the original issue #118 shape. Buggy stride was 1.
    all_pass &= check("enum_ptr", &stream, &seq, &seq, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.read_via_enum(s, cfg, i, o) }.expect("launch")
    });

    // #[repr(C)]: tag is C int (4 bytes); repr().int cannot see this one.
    all_pass &= check("repr_c_enum", &stream, &seq, &seq, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.read_via_repr_c_enum(s, cfg, i, o) }.expect("launch")
    });

    // Default repr, sparse discriminants: rustc picks a u32 tag.
    let sparse: Vec<u32> = vec![0, 1_000_000, 1_000_000, 0];
    all_pass &= check("sparse_enum", &stream, &sparse, &sparse, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.read_via_sparse_enum(s, cfg, i, o) }.expect("launch")
    });

    // Default repr, negative discriminant: SIGNED i8 tag; `as i32` must
    // sign-extend the 0xFF byte to -1, not zero-extend it to 255.
    let neg_in: Vec<u8> = vec![0xFF, 0x00, 0xFF, 0x00];
    let neg_expected: Vec<i32> = vec![-1, 0, -1, 0];
    all_pass &= check(
        "neg_enum",
        &stream,
        &neg_in,
        &neg_expected,
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        |s, cfg, i, o| unsafe { module.read_via_neg_enum(s, cfg, i, o) }.expect("launch"),
    );

    // #[repr(usize)]: pointer-width tag; stride 8 over a u64 buffer.
    let useq: Vec<u64> = (0..N as u64).collect();
    all_pass &= check("usize_enum", &stream, &useq, &seq, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.read_via_usize_enum(s, cfg, i, o) }.expect("launch")
    });

    println!();
    if all_pass {
        println!("RESULT: PASS - every enum shape strides and sign-extends correctly.");
        std::process::exit(0);
    } else {
        println!("RESULT: FAIL - at least one enum shape miscompiled (see above).");
        std::process::exit(1);
    }
}
