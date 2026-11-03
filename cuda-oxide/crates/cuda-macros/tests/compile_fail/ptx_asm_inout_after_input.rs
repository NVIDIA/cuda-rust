// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let mut value = 1u32;

    unsafe {
        ptx_asm!(
            "add.u32 %1, %1, %0;",
            in("r") 2u32,
            inout("+r") value,
        );
    }

    let _ = value;
}
