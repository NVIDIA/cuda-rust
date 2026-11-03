// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_variables)]

use cuda_macros::ptx_asm;

fn missing_read_write_prefix() {
    let mut value = 1u32;

    unsafe {
        ptx_asm!("mov.u32 %0, %0;", inout("r") value);
    }
}

fn unsupported_read_write_constraint() {
    let mut value = 1u32;

    unsafe {
        ptx_asm!("mov.u32 %0, %0;", inout("+n") value);
    }
}

fn main() {}
