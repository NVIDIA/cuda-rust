// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A literal `#[unroll(N)]` expands to a marker call whose const argument sits
// in braces. Those braces must keep the macro's own span. On the attribute's
// span rustc treats them as written by the user, and its `unused_braces` lint
// then reports every literal factor in the user's crate, with a fix-it that
// deletes the attribute. This crate denies that lint, so the build fails if
// the braces ever take the attribute's span again.
#![deny(unused_braces)]

use cuda_device::{device, kernel};

#[device]
fn count_down(mut remaining: u32) -> u32 {
    let mut steps = 0;
    #[unroll]
    loop {
        if remaining == 0 {
            break;
        }
        remaining -= 1;
        steps += 1;
    }
    steps
}

#[kernel]
fn literal_factors(limit: u32) {
    let mut total = 0;
    #[unroll(4)]
    for i in 0..8 {
        total += i;
    }
    let mut index = 0;
    #[unroll(2)]
    while index < limit {
        index += 1;
    }
    let _ = (total, count_down(limit));
}

fn main() {
    assert_eq!(count_down(3), 3);
}
