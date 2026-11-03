// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `in("C")` operands must be compile-time byte strings (`&'static [u8; N]`).
//! A bare integer constant must be rejected instead of silently splicing its
//! raw little-endian bytes into the PTX template.

use cuda_macros::ptx_asm;

fn main() {
    let x = 7u32;
    let y: u64;

    unsafe {
        ptx_asm!(
            "mul%1.u32 %0, %2, %2;",
            out("=l") y,
            in("C") 42,
            in("r") x,
        );
    }

    let _ = y;
}
