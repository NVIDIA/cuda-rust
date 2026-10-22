/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{BFloat16, bf16_mma::PackedDecode};
fn main() {
    let mut storage = [BFloat16::from_bits(0); 128];
    PackedDecode::with_validated(&storage, &storage, 1, 16, 8, 16, 16, |_| {
        storage[0] = BFloat16::from_bits(1);
    });
}
