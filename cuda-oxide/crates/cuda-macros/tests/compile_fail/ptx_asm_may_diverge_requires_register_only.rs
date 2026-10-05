// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let x = 1u32;
    let y: u32;

    unsafe {
        ptx_asm!(
            "mov.u32 %0, %1;",
            out("=r") y,
            in("r") x,
            options(may_diverge),
        );
    }

    let _ = y;
}
