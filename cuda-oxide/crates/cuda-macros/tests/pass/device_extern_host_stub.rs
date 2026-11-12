// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// `#[device]` extern symbols exist only in device LTOIR. A host binary that
// keeps a wrapper (as a cdylib keeps every `#[no_mangle]` device host copy
// that calls one) must still link, and the user's `improper_ctypes` levels,
// on the block or on the item, must keep covering the generated host side.
#![deny(warnings)]

use cuda_macros::device;

#[device]
unsafe extern "C" {
    fn device_only_symbol(x: f32) -> f32;
    #[allow(improper_ctypes)]
    fn device_only_char(c: char) -> char;
}

#[device]
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn device_only_block_char(c: char) -> char;
}

fn main() {
    std::hint::black_box(device_only_symbol as unsafe fn(f32) -> f32);
    std::hint::black_box(device_only_char as unsafe fn(char) -> char);
    std::hint::black_box(device_only_block_char as unsafe fn(char) -> char);
}
