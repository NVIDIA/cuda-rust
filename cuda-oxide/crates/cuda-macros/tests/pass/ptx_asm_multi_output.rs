// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validates that `ptx_asm!` supports multiple `out` operands (2, 4, 8, and 16).

#![allow(dead_code, unused_variables)]

use cuda_macros::ptx_asm;

mod cuda_device {
    pub mod ptx {
        pub unsafe fn __ptx_asm_out_0<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_out_1<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            A0,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            _a0: A0,
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_out_2<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            A0,
            A1,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            _a0: A0,
            _a1: A1,
        ) -> T {
            panic!("test marker")
        }

        pub unsafe fn __ptx_asm_out_4<
            T,
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
            A0,
            A1,
            A2,
            A3,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
            _a0: A0,
            _a1: A1,
            _a2: A2,
            _a3: A3,
        ) -> T {
            panic!("test marker")
        }
    }
}

/// Two outputs with two inputs.
fn two_outputs() {
    let a: u32;
    let b: u32;

    unsafe {
        ptx_asm!(
            "mov.b32 %0, %2; mov.b32 %1, %3;",
            out("=r") a, out("=r") b,
            in("r") 1u32, in("r") 2u32,
            options(register_only),
        );
    }

    let _ = (a, b);
}

/// Four outputs with four inputs (covers MMA-style results).
fn four_outputs() {
    let a: u32;
    let b: u32;
    let c: u32;
    let d: u32;

    unsafe {
        ptx_asm!(
            "mov.b32 %0, %4; mov.b32 %1, %5; mov.b32 %2, %6; mov.b32 %3, %7;",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d,
            in("r") 1u32, in("r") 2u32, in("r") 3u32, in("r") 4u32,
            options(register_only),
        );
    }

    let _ = (a, b, c, d);
}

/// Eight outputs with one input (maximum supported count).
fn eight_outputs() {
    let a: u32;
    let b: u32;
    let c: u32;
    let d: u32;
    let e: u32;
    let f: u32;
    let g: u32;
    let h: u32;

    unsafe {
        ptx_asm!(
            "mov.b32 %0, %8; mov.b32 %1, %8; mov.b32 %2, %8; mov.b32 %3, %8; mov.b32 %4, %8; mov.b32 %5, %8; mov.b32 %6, %8; mov.b32 %7, %8;",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d,
            out("=r") e, out("=r") f, out("=r") g, out("=r") h,
            in("r") 42u32,
            options(register_only),
        );
    }

    let _ = (a, b, c, d, e, f, g, h);
}

/// Sixteen outputs with one input (maximum supported count).
fn sixteen_outputs() {
    let a: u32;
    let b: u32;
    let c: u32;
    let d: u32;
    let e: u32;
    let f: u32;
    let g: u32;
    let h: u32;
    let i: u32;
    let j: u32;
    let k: u32;
    let l: u32;
    let m: u32;
    let n: u32;
    let o: u32;
    let p: u32;

    unsafe {
        ptx_asm!(
            "mov.b32 %0, %16;
             mov.b32 %1, %16;
             mov.b32 %2, %16;
             mov.b32 %3, %16;
             mov.b32 %4, %16;
             mov.b32 %5, %16;
             mov.b32 %6, %16;
             mov.b32 %7, %16;
             mov.b32 %8, %16;
             mov.b32 %9, %16;
             mov.b32 %10, %16;
             mov.b32 %11, %16;
             mov.b32 %12, %16;
             mov.b32 %13, %16;
             mov.b32 %14, %16;
             mov.b32 %15, %16;",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d,
            out("=r") e, out("=r") f, out("=r") g, out("=r") h,
            out("=r") i, out("=r") j, out("=r") k, out("=r") l,
            out("=r") m, out("=r") n, out("=r") o, out("=r") p,
            in("r") 42u32,
            options(register_only),
        );
    }

    let _ = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

/// Mixed output types (integer and floating point).
fn mixed_output_types() {
    let a: u32;
    let b: f32;

    unsafe {
        ptx_asm!(
            "mov.b32 %0, %2; mov.f32 %1, %3;",
            out("=r") a, out("=f") b,
            in("r") 1u32, in("f") 2.0f32,
            options(register_only),
        );
    }

    let _ = (a, b);
}

/// Multi-output with no inputs.
fn multi_output_no_inputs() {
    let a: u32;
    let b: u32;

    unsafe {
        ptx_asm!(
            "mov.u32 %0, %%laneid; mov.u32 %1, %%laneid;",
            out("=r") a, out("=r") b,
        );
    }

    let _ = (a, b);
}

fn main() {}
