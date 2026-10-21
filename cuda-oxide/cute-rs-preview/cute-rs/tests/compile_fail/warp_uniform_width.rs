// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject a 64-bit value where the uniformity marker accepts only 32 bits.
/// Reject a 64-bit value where the uniformity marker accepts only 32 bits.
fn main() {
    let _ = unsafe { cute_rs::arch::assume_warp_uniform(0u64) };
}
