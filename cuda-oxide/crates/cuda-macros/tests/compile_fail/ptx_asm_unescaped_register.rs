// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let lane: u32;

    unsafe {
        ptx_asm!("mov.u32 %0, %laneid;", out("=r") lane);
    }
}
