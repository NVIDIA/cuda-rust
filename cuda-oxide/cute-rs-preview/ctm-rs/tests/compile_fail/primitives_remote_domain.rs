/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject a local shared address passed to a remote barrier arrival.
use cuda_device::barrier::Barrier;
use ctm_rs::{memory::SharedAddress,cluster::arrive_remote};
/// Reject a local shared address passed to a remote barrier arrival.
fn wrong(local:SharedAddress<Barrier>) { unsafe { arrive_remote(local) }; }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
