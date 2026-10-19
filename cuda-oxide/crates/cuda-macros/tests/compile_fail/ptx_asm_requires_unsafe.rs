// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unused_variables)]

use cuda_macros::ptx_asm;

mod cuda_device {
    pub mod ptx {
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
    }
}

fn main() {
    let x = 1u32;
    let y: u32;

    ptx_asm!("add.u32 %0, %1, %1;", out("=r") y, in("r") x);

    let _ = y;
}
