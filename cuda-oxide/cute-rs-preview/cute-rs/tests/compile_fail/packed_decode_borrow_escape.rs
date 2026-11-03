/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{BFloat16, mma::bf16::PackedDecode};
fn escape<'a>() -> Option<PackedDecode<'a>> {
    let storage = [BFloat16::from_bits(0); 128];
    PackedDecode::new(&storage, &storage, 1, 16, 8, 16, 16)
}
fn main() {
    let _ = escape();
}
