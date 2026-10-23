/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: prevent access to a tensor-memory guard after deallocation consumes it.
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemReady},groups::CtaGroup2};
/// Prevent access to a tensor-memory guard after deallocation consumes it.
fn wrong(g:TmemGuard<TmemReady,512,CtaGroup2>) { let _ = unsafe { g.dealloc_collective() }; let _ = g.raw_address(); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
