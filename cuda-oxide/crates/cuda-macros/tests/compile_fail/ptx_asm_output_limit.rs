// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::ptx_asm;

fn main() {
    let o00: u32;
    let o01: u32;
    let o02: u32;
    let o03: u32;
    let o04: u32;
    let o05: u32;
    let o06: u32;
    let o07: u32;
    let o08: u32;
    let o09: u32;
    let o10: u32;
    let o11: u32;
    let o12: u32;
    let o13: u32;
    let o14: u32;
    let o15: u32;
    let o16: u32;
    let o17: u32;
    let o18: u32;
    let o19: u32;
    let o20: u32;
    let o21: u32;
    let o22: u32;
    let o23: u32;
    let o24: u32;
    let o25: u32;
    let o26: u32;
    let o27: u32;
    let o28: u32;
    let o29: u32;
    let o30: u32;
    let o31: u32;
    let o32: u32;
    let o33: u32;
    let o34: u32;
    let o35: u32;
    let o36: u32;
    let o37: u32;
    let o38: u32;
    let o39: u32;
    let o40: u32;
    let o41: u32;
    let o42: u32;
    let o43: u32;
    let o44: u32;
    let o45: u32;
    let o46: u32;
    let o47: u32;
    let o48: u32;
    let o49: u32;
    let o50: u32;
    let o51: u32;
    let o52: u32;
    let o53: u32;
    let o54: u32;
    let o55: u32;
    let o56: u32;
    let o57: u32;
    let o58: u32;
    let o59: u32;
    let o60: u32;
    let o61: u32;
    let o62: u32;
    let o63: u32;
    let o64: u32;

    unsafe {
        ptx_asm!(
            "nop;",
            out("=r") o00, out("=r") o01, out("=r") o02, out("=r") o03, out("=r") o04, out("=r") o05, out("=r") o06, out("=r") o07, out("=r") o08, out("=r") o09, out("=r") o10, out("=r") o11, out("=r") o12, out("=r") o13, out("=r") o14, out("=r") o15, out("=r") o16, out("=r") o17, out("=r") o18, out("=r") o19, out("=r") o20, out("=r") o21, out("=r") o22, out("=r") o23, out("=r") o24, out("=r") o25, out("=r") o26, out("=r") o27, out("=r") o28, out("=r") o29, out("=r") o30, out("=r") o31, out("=r") o32, out("=r") o33, out("=r") o34, out("=r") o35, out("=r") o36, out("=r") o37, out("=r") o38, out("=r") o39, out("=r") o40, out("=r") o41, out("=r") o42, out("=r") o43, out("=r") o44, out("=r") o45, out("=r") o46, out("=r") o47, out("=r") o48, out("=r") o49, out("=r") o50, out("=r") o51, out("=r") o52, out("=r") o53, out("=r") o54, out("=r") o55, out("=r") o56, out("=r") o57, out("=r") o58, out("=r") o59, out("=r") o60, out("=r") o61, out("=r") o62, out("=r") o63, out("=r") o64,
        );
    }
}
