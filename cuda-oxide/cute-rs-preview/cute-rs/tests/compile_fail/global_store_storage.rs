// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject FP64 carriers in the supported global-store tile formats.
/// Reject FP64 carriers in the supported global-store tile formats.
fn main() {
    let _ = cute_rs::GlobalStoreTile([0.0f64; 4]);
}
