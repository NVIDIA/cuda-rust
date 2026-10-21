/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{Bf16, bf16_mma::PackedDecode};
fn main() {
    let storage = [Bf16::from_bits(0); 128];
    let mut escaped = None;
    PackedDecode::with_validated(&storage, &storage, 1, 16, 8, 16, 16, |proof| {
        escaped = Some(proof);
    });
    let _ = escaped;
}
