/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject a local shared address passed to a relaxed remote arrival.
use ctm_rs::{cluster::arrive_remote_relaxed, memory::SharedAddress};
use cuda_device::barrier::Barrier;
/// Reject a local shared address passed to a relaxed remote arrival.
fn wrong(local: SharedAddress<Barrier>) {
    unsafe { arrive_remote_relaxed(local) };
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
