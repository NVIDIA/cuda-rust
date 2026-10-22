/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{
    Bf16,
    bf16_mma::{Bf16Mma16x8x16, FragmentC},
};
fn main() {
    let mma = Bf16Mma16x8x16::get_slice(0);
    let a = mma.partition_a().load_with(|_, _| Bf16::from_bits(0));
    let b = mma.partition_b().load_with(|_, _| Bf16::from_bits(0));
    let _ = unsafe { mma.mma_atom_call(FragmentC::ZERO, b, a) };
}
