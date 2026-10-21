/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: keep local shared addresses distinct from cluster shared addresses.
use ctm_rs::memory::{SharedAddress,ClusterSharedAddress};
/// Accept only a cluster address, making a local-address substitution fail.
fn cluster(_: ClusterSharedAddress<'_,u32>) {}
/// Keep local shared addresses distinct from cluster shared addresses.
fn wrong(address: SharedAddress<'_,u32>) { cluster(address); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
