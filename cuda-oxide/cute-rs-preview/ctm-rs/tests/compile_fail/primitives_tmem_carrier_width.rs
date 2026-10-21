/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: TMEM transfers retain their exact canonical register count.
use ctm_rs::{memory::CuSimd, tcgen05::tmem::store_32x32b_x4};
/// Reject a store with too few register operands.
unsafe fn wrong(address: u32, data: CuSimd<u32, 2>) {
    unsafe { store_32x32b_x4(address, data) };
}
/// Unused host entry point.
fn main() {}
