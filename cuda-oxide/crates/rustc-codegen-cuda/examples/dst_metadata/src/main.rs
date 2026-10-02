/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for DST `size_of_val` / `align_of_val` lowering.
//!
//! The slice case exercises runtime fat-pointer length metadata with a
//! non-byte element type. The `str` case reinterprets a runtime UTF-8 byte
//! slice as `str`, preserving the same `(data, byte_len)` fat-pointer layout.
//! The slice-tailed struct case exercises Rust's aggregate DST layout rule:
//! a sized prefix followed inline by a runtime-length `[T]` tail.
//!
//! Usage:
//!   cargo oxide run dst_metadata

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[repr(C)]
struct Header<T: ?Sized> {
    head: u64,
    tag: u8,
    tail: T,
}

const HEADER_HEAD: u64 = 0x1122_3344_5566_7788;
const HEADER_TAG: u8 = 0x5a;
const HEADER_TAIL: [u16; 4] = [11, 22, 33, 44];

#[cuda_module]
mod kernels {
    use super::*;

    /// Report `size_of_val` and `align_of_val` for a runtime `&[u32]`.
    #[kernel]
    pub fn slice_metadata(input: &[u32], mut out: DisjointSlice<usize>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let size = core::mem::size_of_val(input);
        let align = core::mem::align_of_val(input);

        unsafe {
            let out_ptr = out.as_mut_ptr();
            out_ptr.write(size);
            out_ptr.add(1).write(align);
        }
    }

    /// Report DST layout for a runtime `str` built from UTF-8 bytes.
    #[kernel]
    pub fn str_metadata(input: &[u8], mut out: DisjointSlice<usize>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        // SAFETY: the host passes the UTF-8 bytes of a Rust string literal.
        let text = unsafe { core::str::from_utf8_unchecked(input) };
        let size = core::mem::size_of_val(text);
        let align = core::mem::align_of_val(text);

        unsafe {
            let out_ptr = out.as_mut_ptr();
            out_ptr.write(size);
            out_ptr.add(1).write(align);
        }
    }

    /// Report layout and projection behavior for `Header<[u16]>`.
    #[kernel]
    pub fn struct_tail_metadata(mut out: DisjointSlice<usize>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let concrete = Header {
            head: HEADER_HEAD,
            tag: HEADER_TAG,
            tail: HEADER_TAIL,
        };
        let value: &Header<[u16]> = &concrete;

        let size = core::mem::size_of_val(value);
        let align = core::mem::align_of_val(value);

        unsafe {
            let out_ptr = out.as_mut_ptr();
            out_ptr.write(size);
            out_ptr.add(1).write(align);
            out_ptr.add(2).write(value.tail.len());
            out_ptr.add(3).write(value.tag as usize);
            out_ptr.add(4).write(value.head as usize);
        }
    }
}

fn main() {
    println!("=== dst_metadata ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    let slice_input = [11_u32, 22, 33, 44, 55, 66, 77];
    let device_slice = DeviceBuffer::from_host(&stream, &slice_input).unwrap();
    let mut slice_out = DeviceBuffer::<usize>::zeroed(&stream, 2).unwrap();

    // SAFETY: one thread executes the metadata query and the two-element output
    // buffer covers both stores.
    unsafe { module.slice_metadata(&stream, cfg, &device_slice, &mut slice_out) }
        .expect("slice_metadata launch");

    let slice_result = slice_out.to_host_vec(&stream).unwrap();
    assert_eq!(
        slice_result[0],
        slice_input.len() * core::mem::size_of::<u32>(),
        "size_of_val(&[u32])"
    );
    assert_eq!(
        slice_result[1],
        core::mem::align_of::<u32>(),
        "align_of_val(&[u32])"
    );
    println!("PASS: size_of_val on &[u32]");
    println!("PASS: align_of_val on &[u32]");

    // `oxide` is five bytes and `✓` is three UTF-8 bytes. This deliberately
    // distinguishes str byte-length metadata (8) from character count (6).
    let text_bytes = "oxide✓".as_bytes();
    let device_text = DeviceBuffer::from_host(&stream, text_bytes).unwrap();
    let mut str_out = DeviceBuffer::<usize>::zeroed(&stream, 2).unwrap();

    // SAFETY: `text_bytes` is valid UTF-8 and the output has two elements.
    unsafe { module.str_metadata(&stream, cfg, &device_text, &mut str_out) }
        .expect("str_metadata launch");

    let str_result = str_out.to_host_vec(&stream).unwrap();
    assert_eq!(str_result[0], text_bytes.len(), "size_of_val(str)");
    assert_eq!(
        str_result[1],
        core::mem::align_of::<u8>(),
        "align_of_val(str)"
    );
    println!("PASS: size_of_val on str");
    println!("PASS: align_of_val on str");

    let concrete = Header {
        head: HEADER_HEAD,
        tag: HEADER_TAG,
        tail: HEADER_TAIL,
    };
    let value: &Header<[u16]> = &concrete;
    let expected_size = core::mem::size_of_val(value);
    let expected_align = core::mem::align_of_val(value);
    let mut struct_out = DeviceBuffer::<usize>::zeroed(&stream, 5).unwrap();

    // SAFETY: one thread writes five `usize` values to a five-element output.
    unsafe { module.struct_tail_metadata(&stream, cfg, &mut struct_out) }
        .expect("struct_tail_metadata launch");

    let struct_result = struct_out.to_host_vec(&stream).unwrap();
    assert_eq!(
        struct_result[0], expected_size,
        "size_of_val(&Header<[u16]>)"
    );
    assert_eq!(
        struct_result[1], expected_align,
        "align_of_val(&Header<[u16]>)"
    );
    assert_eq!(struct_result[2], HEADER_TAIL.len(), "tail metadata");
    assert_eq!(struct_result[3], HEADER_TAG as usize, "tag projection");
    assert_eq!(struct_result[4], HEADER_HEAD as usize, "head projection");
    println!("PASS: size_of_val on &Header<[u16]>");
    println!("PASS: align_of_val on &Header<[u16]>");
    println!("PASS: unsized-tail metadata and prefix fields");
    println!("PASS: dst_metadata");
}
