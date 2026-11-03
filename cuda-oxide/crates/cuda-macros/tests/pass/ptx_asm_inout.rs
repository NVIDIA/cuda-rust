// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time coverage for `ptx_asm!` read-write operands.

#![allow(dead_code, unused_assignments, unused_variables)]

use cuda_macros::ptx_asm;

mod cuda_device {
    pub mod ptx {
        macro_rules! define_ptx_asm_out {
            ($name:ident; $($arg:ident : $ty:ident),*) => {
                pub unsafe fn $name<
                    T,
                    const TEMPLATE_LEN: usize,
                    const CONSTRAINTS_LEN: usize,
                    const OPTIONS_LEN: usize,
                    $($ty,)*
                >(
                    _template: &'static [u8; TEMPLATE_LEN],
                    _constraints: &'static [u8; CONSTRAINTS_LEN],
                    _options: &'static [u8; OPTIONS_LEN],
                    $($arg: $ty,)*
                ) -> T {
                    panic!("test marker")
                }
            };
        }

        define_ptx_asm_out!(__ptx_asm_out_1; a0: A0);
        define_ptx_asm_out!(__ptx_asm_out_2; a0: A0, a1: A1);
        define_ptx_asm_out!(
            __ptx_asm_out_24;
            a0: A0, a1: A1, a2: A2, a3: A3, a4: A4, a5: A5,
            a6: A6, a7: A7, a8: A8, a9: A9, a10: A10, a11: A11,
            a12: A12, a13: A13, a14: A14, a15: A15, a16: A16, a17: A17,
            a18: A18, a19: A19, a20: A20, a21: A21, a22: A22, a23: A23
        );
    }
}

fn single_inout() {
    let mut value = 3u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %0, %1;",
            inout("+r") value,
            in("r") 4u32,
            options(register_only),
        );
    }

    let _ = value;
}

fn register_only_accepts_inout() {
    let mut value = 1u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %0, 1;",
            inout("+r") value,
            options(register_only),
        );
    }

    let _ = value;
}

fn multiple_inouts() {
    let mut x = 1u32;
    let mut y = 2u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %0, 1; add.u32 %1, %1, 2;",
            inout("+r") x,
            inout("+r") y,
            options(register_only),
        );
    }

    let _ = (x, y);
}

fn mixed_out_then_inout() {
    let output: u32;
    let mut accumulator = 5u32;

    unsafe {
        ptx_asm!(
            "mov.u32 %0, %2; add.u32 %1, %1, %2;",
            out("=r") output,
            inout("+r") accumulator,
            in("r") 7u32,
            options(register_only),
        );
    }

    let _ = (output, accumulator);
}

fn mixed_inout_then_out() {
    let mut accumulator = 5u32;
    let output: u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %0, %2; mov.u32 %1, %2;",
            inout("+r") accumulator,
            out("=r") output,
            in("r") 7u32,
            options(register_only),
        );
    }

    let _ = (accumulator, output);
}

fn supported_inout_constraints() {
    let mut h = 1u16;
    let mut r = 2u32;
    let mut l = 3u64;
    let mut q = 4u128;
    let mut f = 5.0f32;
    let mut d = 6.0f64;

    unsafe {
        ptx_asm!("mov.b16 %0, %0;", inout("+h") h);
        ptx_asm!("mov.b32 %0, %0;", inout("+r") r);
        ptx_asm!("mov.b64 %0, %0;", inout("+l") l);
        ptx_asm!("mov.b128 %0, %0;", inout("+q") q);
        ptx_asm!("mov.f32 %0, %0;", inout("+f") f);
        ptx_asm!("mov.f64 %0, %0;", inout("+d") d);
    }

    let _ = (h, r, l, q, f, d);
}

fn evaluates_complex_place() {
    let mut values = [1u32];

    unsafe {
        ptx_asm!(
            "add.u32 %0, %0, 1;",
            inout("+r") values[0],
            options(register_only),
        );
    }

    let _ = values;
}

// Not the reachable maximum (that is 16 inout + 16 in = arity 32, exercised
// against the real crate in ptx_asm_marker_arity_contract.rs); this file's
// mock marker surface stops at 24.
fn eight_inouts_with_full_input_block() {
    let mut o0 = 0u32;
    let mut o1 = 1u32;
    let mut o2 = 2u32;
    let mut o3 = 3u32;
    let mut o4 = 4u32;
    let mut o5 = 5u32;
    let mut o6 = 6u32;
    let mut o7 = 7u32;

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

    let _ = (o0, o1, o2, o3, o4, o5, o6, o7);
}

fn main() {}
