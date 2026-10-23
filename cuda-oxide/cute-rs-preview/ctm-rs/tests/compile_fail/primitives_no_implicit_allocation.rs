/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: require explicit collective allocation rather than an implicit guard allocation method.
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemUninit},groups::{CtaGroup1,CtaGroup2}};
/// Reject implicit allocation for a group-one TMEM guard.
fn group1(g:TmemGuard<TmemUninit,512,CtaGroup1>) { let _ = unsafe { g.alloc() }; }
/// Reject implicit allocation for a group-two TMEM guard.
fn group2(g:TmemGuard<TmemUninit,512,CtaGroup2>) { let _ = unsafe { g.alloc() }; }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
