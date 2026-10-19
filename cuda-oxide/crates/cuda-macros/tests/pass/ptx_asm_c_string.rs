// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validates the host-side expansion of `in("C")` compile-time string
//! operands against the real `cuda_device::ptx` marker surface, including
//! the typed `__ptx_asm_c` helper the macro wraps every `C` operand in.

#![allow(dead_code, unused_variables)]

use cuda_macros::ptx_asm;

/// Byte-string literal operand.
fn c_string_literal(lhs: u32, rhs: u32) {
    let product: u64;

    unsafe {
        ptx_asm!(
            "mul%1.u32 %0, %2, %3;",
            out("=l") product,
            in("C") b".wide\0",
            in("r") lhs,
            in("r") rhs,
            options(register_only),
        );
    }

    let _ = product;
}

/// Named-const operand, matching the CUDA C++ `"C"` idiom.
fn c_string_named_const(lhs: u32, rhs: u32) {
    const MODE: &[u8; 6] = b".wide\0";
    let product: u64;

    unsafe {
        ptx_asm!(
            "mul%1.u32 %0, %2, %3;",
            out("=l") product,
            in("C") MODE,
            in("r") lhs,
            in("r") rhs,
            options(register_only),
        );
    }

    let _ = product;
}

fn main() {}
