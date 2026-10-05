// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let out: u32;
    let x = 1u32;

    unsafe {
        ptx_asm!(
            "add.u32 %0, %1, %1;",
            out("=r") out,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
            in("r") x,
        );
    }
}
