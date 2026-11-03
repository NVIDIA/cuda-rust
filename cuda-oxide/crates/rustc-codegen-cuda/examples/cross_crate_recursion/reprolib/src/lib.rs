// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A recursive helper in a dependency crate. `rec1` is recursive, not `#[inline]`, not generic —
//! so rustc does not encode its MIR cross-crate, and (being recursive) it cannot be inlined away.
//! That is exactly the shape that regressed without `-Zalways-encode-mir`.
#![no_std]

pub fn rec1(a: &[u64]) -> u64 {
    if a.is_empty() {
        return 0;
    }
    a[0].wrapping_add(rec1(&a[1..]))
}
