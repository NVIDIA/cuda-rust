// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Compile-fail probe: reject K16 TMEM/shared MMA even when the B layout itself is valid.
use cute_rs::sm100_attention::{MmaTs, SharedLayout};
// MN-major shared B alone permits K16, but the TMEM-source importer does not.
/// MN-major B layout valid on its own but unsupported by TMEM/shared K16 MMA.
type Value = SharedLayout<128, 16, true>;
/// Reject K16 TMEM/shared MMA even when the B layout itself is valid.
const _: MmaTs<Value, 128, 128, 16, 1> = MmaTs::new();
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
