/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

static mut VALUE: i32 = 0;

#[inline(never)]
fn callee(child: &mut i32, root: *mut i32) {
    unsafe {
        *root = 1;
    }

    *child = 2;
}

fn main() {
    let root = &raw mut VALUE;
    let child = unsafe { &mut *root };

    callee(child, root);
}
