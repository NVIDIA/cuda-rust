/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: prevent extracting an instruction address before the barrier is initialized.
use cuda_device::barrier::{ManagedBarrier,Uninit,TmaBarrier};
use ctm_rs::mbarrier::BarrierAddressExt;
/// Prevent extracting an instruction address before the barrier is initialized.
fn wrong(b:ManagedBarrier<Uninit,TmaBarrier>) { let _ = b.shared_address(); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
