/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject a 33-column tensor-memory allocation.
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemUninit},groups::CtaGroup2};
/// Reject a 33-column tensor-memory allocation.
fn wrong(slot:*mut u32) { let _ = unsafe { TmemGuard::<TmemUninit,33,CtaGroup2>::from_raw_slot(slot) }; }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
