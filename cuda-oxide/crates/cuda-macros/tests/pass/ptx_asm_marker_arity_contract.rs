// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validates that every output marker arity reachable from `ptx_asm!` exists
//! in the real `cuda_device::ptx` marker surface.

#![allow(dead_code, unused_imports, unused_variables)]

use cuda_macros::ptx_asm;
use cuda_device::ptx::{
    __ptx_asm_out_25, __ptx_asm_out_26, __ptx_asm_out_27, __ptx_asm_out_28,
    __ptx_asm_out_29, __ptx_asm_out_30, __ptx_asm_out_31, __ptx_asm_out_32,
};

fn maximum_marker_arity() {
    let mut o0 = 0u32;
    let mut o1 = 1u32;
    let mut o2 = 2u32;
    let mut o3 = 3u32;
    let mut o4 = 4u32;
    let mut o5 = 5u32;
    let mut o6 = 6u32;
    let mut o7 = 7u32;
    let mut o8 = 8u32;
    let mut o9 = 9u32;
    let mut o10 = 10u32;
    let mut o11 = 11u32;
    let mut o12 = 12u32;
    let mut o13 = 13u32;
    let mut o14 = 14u32;
    let mut o15 = 15u32;

    unsafe {
        ptx_asm!(
            "nop;",
            inout("+r") o0,
            inout("+r") o1,
            inout("+r") o2,
            inout("+r") o3,
            inout("+r") o4,
            inout("+r") o5,
            inout("+r") o6,
            inout("+r") o7,
            inout("+r") o8,
            inout("+r") o9,
            inout("+r") o10,
            inout("+r") o11,
            inout("+r") o12,
            inout("+r") o13,
            inout("+r") o14,
            inout("+r") o15,
            in("r") 0u32,
            in("r") 1u32,
            in("r") 2u32,
            in("r") 3u32,
            in("r") 4u32,
            in("r") 5u32,
            in("r") 6u32,
            in("r") 7u32,
            in("r") 8u32,
            in("r") 9u32,
            in("r") 10u32,
            in("r") 11u32,
            in("r") 12u32,
            in("r") 13u32,
            in("r") 14u32,
            in("r") 15u32,
            options(register_only),
        );
    }

    let _ = (
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    );
}

fn main() {}
