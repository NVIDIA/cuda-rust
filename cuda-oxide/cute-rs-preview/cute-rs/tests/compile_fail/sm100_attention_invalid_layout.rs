// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject a K16 shared/shared attention configuration.
use cute_rs::sm100_attention::{MmaSs, SharedLayout};
/// K-major shared layout too narrow for this attention MMA path.
type UnsupportedK16 = SharedLayout<128, 16>;
/// Reject a K16 shared/shared attention configuration.
const _: MmaSs<UnsupportedK16, UnsupportedK16, 128, 128, 16, 1> = MmaSs::new();
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
