// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: require an unsafe assertion before treating a value as warp-uniform.
/// Require an unsafe assertion before treating a value as warp-uniform.
fn main() {
    let _ = cute_rs::arch::assume_warp_uniform(0);
}
