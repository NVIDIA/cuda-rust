// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

        pub unsafe fn __ptx_asm_void_0<
            const TEMPLATE_LEN: usize,
            const CONSTRAINTS_LEN: usize,
            const OPTIONS_LEN: usize,
        >(
            _template: &'static [u8; TEMPLATE_LEN],
            _constraints: &'static [u8; CONSTRAINTS_LEN],
            _options: &'static [u8; OPTIONS_LEN],
        ) {
        }
    }
}

fn accepts_cuda_doc_shape() {
    let x = 1u32;
    let z = 2u32;
    let y: u32;
    let reg_only: u32;
    let may_diverge: u32;
    let lane: u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %1, %2;",
            out("=r") y,
            in("r") x,
            in("r") z,
        );
        ptx_asm!(
            "mul.lo.u32 %0, %1, %1;",
            out("=r") reg_only,
            in("r") x,
            options(register_only),
        );
        ptx_asm!(
            "mov.u32 %0, %1;",
            out("=r") may_diverge,
            in("r") x,
            options(may_diverge, register_only),
        );
        ptx_asm!("mov.u32 %0, %%laneid;", out("=r") lane);
        ptx_asm!("membar.gl;", clobber("memory"));
    }

    let _ = (y, reg_only, may_diverge, lane);
}

fn accepts_supported_constraints() {
    let h_in = 1u16;
    let r_in = 2u32;
    let l_in = 3u64;
    let q_in = 4u128;
    let f_in = 5.0f32;
    let d_in = 6.0f64;

    let h_out: u16;
    let r_out: u32;
    let l_out: u64;
    let q_out: u128;
    let f_out: f32;
    let d_out: f64;
    let n_out: u32;

    unsafe {
        ptx_asm!(
            "mov.u16 %0, %1;",
            out("=h") h_out,
            in("h") h_in,
            options(register_only),
        );
        ptx_asm!(
            "mov.u32 %0, %1;",
            out("=r") r_out,
            in("r") r_in,
            options(register_only),
        );
        ptx_asm!(
            "mov.u64 %0, %1;",
            out("=l") l_out,
            in("l") l_in,
            options(register_only),
        );
        ptx_asm!(
            "mov.b128 %0, %1;",
            out("=q") q_out,
            in("q") q_in,
            options(register_only),
        );
        ptx_asm!(
            "mov.f32 %0, %1;",
            out("=f") f_out,
            in("f") f_in,
            options(register_only),
        );
        ptx_asm!(
            "mov.f64 %0, %1;",
            out("=d") d_out,
            in("d") d_in,
            options(register_only),
        );
        ptx_asm!(
            "add.u32 %0, %1, %2;",
            out("=r") n_out,
            in("r") r_in,
            in("n") 7u32,
            options(register_only),
        );
    }

    let _ = (h_out, r_out, l_out, q_out, f_out, d_out, n_out);
}

fn main() {}
