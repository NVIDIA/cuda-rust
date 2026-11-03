/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: keep a tensor-memory allocation borrowed until its last view use.
use ctm_rs::{groups::CtaGroup2,tcgen05::tmem::{TmemGuard,TmemReady,TmemF32,ColumnOffset,TmemOperand}};
/// Keep a TMEM view live after the attempted deallocation.
fn consume_view(_:impl TmemOperand) {}
/// Keep a tensor-memory allocation borrowed until its last view use.
fn wrong(g:TmemGuard<TmemReady,512,CtaGroup2>) {
    let view = g.view::<TmemF32<256,192>>(ColumnOffset::new(0)).unwrap();
    let _dead = unsafe { g.dealloc_collective() };
    consume_view(view);
}
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
