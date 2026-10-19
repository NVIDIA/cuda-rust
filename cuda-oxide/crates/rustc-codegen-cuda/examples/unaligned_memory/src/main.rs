/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for unaligned memory access.
//!
//! Covers both manually offset raw pointers and `#[repr(packed)]` field
//! projections formed with `addr_of!` / `addr_of_mut!`.
//!
//! Usage:
//!   cargo oxide run unaligned_memory
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run unaligned_memory

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[repr(C, packed)]
    struct PackedPacket {
        tag: u8,
        value: u32,
    }

    #[repr(C, packed(2))]
    struct PackedPacket2 {
        tag: u8,
        value: u32,
    }

    #[kernel]
    pub fn read_unaligned_u32(input: &[u8], mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        if let Some(slot) = out.get_mut(thread::index_1d()) {
            unsafe {
                // `input.as_ptr().add(1)` is byte-aligned only. Casting it to
                // `*const u32` deliberately creates an under-aligned pointer.
                let ptr = input.as_ptr().add(1).cast::<u32>();
                *slot = core::ptr::read_unaligned(ptr);
            }
        }
    }

    #[kernel]
    pub fn write_unaligned_u32(value: u32, mut out: DisjointSlice<u8>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        unsafe {
            // Preserve byte 0 and byte 5 as guards. The `u32` write occupies
            // bytes 1..=4 and therefore starts at an address not aligned to 4.
            let ptr = out.as_mut_ptr().add(1).cast::<u32>();
            core::ptr::write_unaligned(ptr, value);
        }
    }

    #[kernel]
    pub fn read_packed_field(input: &[u8], mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        if let Some(slot) = out.get_mut(thread::index_1d()) {
            unsafe {
                let packet = input.as_ptr().cast::<PackedPacket>();
                let value_ptr = core::ptr::addr_of!((*packet).value);
                *slot = core::ptr::read_unaligned(value_ptr);
            }
        }
    }

    #[kernel]
    pub fn write_packed_field(value: u32, mut out: DisjointSlice<u8>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        unsafe {
            let packet = out.as_mut_ptr().cast::<PackedPacket>();
            let value_ptr = core::ptr::addr_of_mut!((*packet).value);
            core::ptr::write_unaligned(value_ptr, value);
        }
    }

    #[kernel]
    pub fn read_packed_two_field(input: &[u8], mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        if let Some(slot) = out.get_mut(thread::index_1d()) {
            unsafe {
                let packet = input.as_ptr().cast::<PackedPacket2>();
                let value_ptr = core::ptr::addr_of!((*packet).value);
                *slot = core::ptr::read_unaligned(value_ptr);
            }
        }
    }
}

fn main() {
    println!("=== unaligned_memory ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    // ---------------------------------------------------------------------
    // Manual read_unaligned
    // ---------------------------------------------------------------------

    const READ_BYTES: [u8; 6] = [0xA5, 0x12, 0x34, 0x56, 0x78, 0x5A];
    let expected_read =
        u32::from_ne_bytes([READ_BYTES[1], READ_BYTES[2], READ_BYTES[3], READ_BYTES[4]]);

    let input = DeviceBuffer::from_host(&stream, &READ_BYTES).expect("read input allocation");
    let mut read_out = DeviceBuffer::<u32>::zeroed(&stream, 1).expect("read output allocation");

    // SAFETY: one thread executes the kernel; `input` contains at least five
    // bytes from the base, and `read_out` contains one writable `u32`.
    unsafe { module.read_unaligned_u32(&stream, cfg, &input, &mut read_out) }
        .expect("read_unaligned_u32 launch");

    let read_result = read_out.to_host_vec(&stream).expect("copy read result");
    assert_eq!(read_result, vec![expected_read], "read_unaligned result");
    println!("PASS: read_unaligned from base + 1");

    // ---------------------------------------------------------------------
    // Manual write_unaligned
    // ---------------------------------------------------------------------

    const WRITE_VALUE: u32 = 0x7856_3412;
    const LEFT_GUARD: u8 = 0xA5;
    const RIGHT_GUARD: u8 = 0x5A;

    let initial_write = [LEFT_GUARD, 0, 0, 0, 0, RIGHT_GUARD];
    let mut write_out =
        DeviceBuffer::from_host(&stream, &initial_write).expect("write output allocation");

    // SAFETY: one thread executes the kernel; `write_out` provides six writable
    // bytes, so the four-byte write beginning at byte 1 is fully in bounds.
    unsafe { module.write_unaligned_u32(&stream, cfg, WRITE_VALUE, &mut write_out) }
        .expect("write_unaligned_u32 launch");

    let write_result = write_out.to_host_vec(&stream).expect("copy write result");
    let expected_bytes = WRITE_VALUE.to_ne_bytes();

    assert_eq!(
        &write_result[1..5],
        &expected_bytes,
        "write_unaligned payload bytes"
    );
    println!("PASS: write_unaligned to base + 1");

    assert_eq!(write_result[0], LEFT_GUARD, "left guard byte");
    assert_eq!(write_result[5], RIGHT_GUARD, "right guard byte");
    println!("PASS: guard bytes preserved");

    // ---------------------------------------------------------------------
    // #[repr(C, packed)] field read
    // ---------------------------------------------------------------------

    let mut packed_read_out =
        DeviceBuffer::<u32>::zeroed(&stream, 1).expect("packed read output allocation");

    // SAFETY: `READ_BYTES` contains at least the five bytes occupied by
    // `PackedPacket`; the kernel only forms the raw field address and performs
    // an unaligned four-byte read from bytes 1..=4.
    unsafe { module.read_packed_field(&stream, cfg, &input, &mut packed_read_out) }
        .expect("read_packed_field launch");

    let packed_read_result = packed_read_out
        .to_host_vec(&stream)
        .expect("copy packed read result");
    assert_eq!(
        packed_read_result,
        vec![expected_read],
        "packed field read result"
    );
    println!("PASS: repr(packed) field read at rustc offset 1");

    // ---------------------------------------------------------------------
    // #[repr(C, packed)] field write
    // ---------------------------------------------------------------------

    let mut packed_write_out =
        DeviceBuffer::from_host(&stream, &initial_write).expect("packed write output allocation");

    // SAFETY: `packed_write_out` contains six writable bytes. `PackedPacket`
    // occupies bytes 0..=4, and its `value` field occupies bytes 1..=4.
    unsafe { module.write_packed_field(&stream, cfg, WRITE_VALUE, &mut packed_write_out) }
        .expect("write_packed_field launch");

    let packed_write_result = packed_write_out
        .to_host_vec(&stream)
        .expect("copy packed write result");
    assert_eq!(
        &packed_write_result[1..5],
        &expected_bytes,
        "packed field write payload bytes"
    );
    assert_eq!(
        packed_write_result[0], LEFT_GUARD,
        "packed write left guard byte"
    );
    assert_eq!(
        packed_write_result[5], RIGHT_GUARD,
        "packed write right guard byte"
    );
    println!("PASS: repr(packed) field write at rustc offset 1");

    // ---------------------------------------------------------------------
    // #[repr(C, packed(2))] field read
    // ---------------------------------------------------------------------

    const PACKED_TWO_BYTES: [u8; 7] = [0xA5, 0xCC, 0x12, 0x34, 0x56, 0x78, 0x5A];
    let expected_packed_two = u32::from_ne_bytes([
        PACKED_TWO_BYTES[2],
        PACKED_TWO_BYTES[3],
        PACKED_TWO_BYTES[4],
        PACKED_TWO_BYTES[5],
    ]);
    let packed_two_input =
        DeviceBuffer::from_host(&stream, &PACKED_TWO_BYTES).expect("packed(2) input allocation");
    let mut packed_two_out =
        DeviceBuffer::<u32>::zeroed(&stream, 1).expect("packed(2) output allocation");

    // SAFETY: `PackedPacket2::value` begins at byte 2 and occupies bytes 2..=5.
    unsafe { module.read_packed_two_field(&stream, cfg, &packed_two_input, &mut packed_two_out) }
        .expect("read_packed_two_field launch");

    let packed_two_result = packed_two_out
        .to_host_vec(&stream)
        .expect("copy packed(2) result");
    assert_eq!(
        packed_two_result,
        vec![expected_packed_two],
        "packed(2) field read result"
    );
    println!("PASS: repr(packed(2)) field read at rustc offset 2");

    println!("PASS: unaligned_memory");
}
