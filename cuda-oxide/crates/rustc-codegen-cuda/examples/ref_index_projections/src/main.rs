// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * References and raw pointers into indexed places.
 *
 * Kernels take addresses of runtime-indexed elements through every shape
 * the address walker must handle:
 *
 *     let compute = |idx: usize| { ... arr[idx] ... };   // closure captures
 *     &p.0[k]                                            // field + runtime index
 *     &(*p).0[1]                                         // deref + const index
 *     addr_of! / addr_of_mut!                            // raw-pointer paths
 *     &mut local.field[k]                                // write-through
 *     struct Holder<'a>(&'a [f32; 2]); &h.0[k]           // deref in the tail
 *     value.tail[1]                                      // DST slice tail + const index
 *     value.tail[k]                                      // DST slice tail + runtime index
 *     padded.tail[k]                                     // DST slice tail behind padding
 *     outer.inner.tail[1]                                // nested DST tail + const index
 *     outer.inner.tail[k]                                // nested DST tail + runtime index
 *     value.tail[k] = x                                  // DST tail element write
 *     &value.tail[k]                                     // DST tail element borrow
 *
 * Each kernel writes the difference compute(1) - compute(0), with inputs
 * designed so the two slots differ: a dropped index reads slot 0 twice
 * and reports 0 instead of 5.
 *
 * Build:
 *     cargo oxide run ref_index_projections
 *
 * Guards the fix for issue #120 (Rvalue::Ref dropped runtime Index
 * projections, returning the field's base address).
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// 2-element struct wrapper. `#[repr(transparent)]` keeps the inner array
/// layout preserved while still exercising a field projection before the
/// indexed access.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Pair(pub [f32; 2]);

impl Pair {
    /// Returns the inner element by value. Matches the `Pair`-style API
    /// used in the original "simple" repro attempts.
    #[inline(always)]
    pub fn get(&self, i: usize) -> f32 {
        self.0[i]
    }

    /// Returns a reference to the inner element. This exercises the
    /// `wrapper.node(slot) -> &T` API shape that exposed the importer bug.
    #[inline(always)]
    pub const fn node(&self, slot: usize) -> &f32 {
        &self.0[slot]
    }
}

/// Same as `Pair`, but WITHOUT `#[repr(transparent)]`. Used to check
/// whether the miscompile is triggered specifically by `repr(transparent)`
/// or by any newtype wrapper over `[f32; 2]`.
#[derive(Copy, Clone)]
pub struct PairNoTransparent(pub [f32; 2]);

impl PairNoTransparent {
    #[inline(always)]
    pub const fn node(&self, slot: usize) -> &f32 {
        &self.0[slot]
    }
}

/// Out-of-line accessor reproducing the EXACT issue #120 MIR shape: its own
/// body contains `_0 = &(((*_1).0)[_2])`, i.e. `Rvalue::Ref` of a place with
/// projections `[Deref, Field(0), Index(_2)]`. `#[inline(never)]` keeps the
/// shape from being dissolved by MIR opts before the importer sees it.
#[inline(never)]
fn node(p: &Pair, k: usize) -> &f32 {
    &p.0[k]
}

/// Struct holding a REFERENCE to an array, so that `&h.0[k]` carries a
/// `Deref` in the middle of the projection tail:
/// `[Field(0), Deref, Index(k)]` (or `[Deref, Field(0), Deref, Index(k)]`
/// when `h` is itself behind a reference). Exercises the address walker's
/// Deref arm.
pub struct Holder<'a>(pub &'a [f32; 2]);

/// Struct with an unsized slice tail used to pin slice-value projections.
#[repr(C)]
pub struct SliceTail<T: ?Sized> {
    pub head: u32,
    pub tail: T,
}

/// The exact issue #870 repro layout: `head: u64` plus `tag: u8` put the
/// `[u16]` tail at byte offset 10, behind one padding byte. `SliceTail`
/// places its tail at offset 4 with no padding, so only this struct can
/// catch a wrong-tail-offset bug in the slice-tail Field lowering.
#[repr(C)]
pub struct PaddedTail<T: ?Sized> {
    pub head: u64,
    pub tag: u8,
    pub tail: T,
}

/// Slice-tailed inner struct used to pin issue #880's nested projection:
/// `Deref -> Field(inner) -> Field(tail) -> Index`.
#[repr(C)]
pub struct NestedInner<T: ?Sized> {
    pub head: u32,
    pub tail: T,
}

/// A DST whose unsized tail is inherited through its final `NestedInner<T>`
/// field. The outer fat pointer carries the metadata needed by `inner.tail`.
#[repr(C)]
pub struct NestedOuter<T: ?Sized> {
    pub id: u64,
    pub inner: NestedInner<T>,
}

/// Nested padded variant. `PaddedTail<[u16]>` places its tail at byte offset
/// 10 inside `inner`; `marker` also forces `inner` away from offset zero.
/// Together they catch both lost metadata and a naive nested field offset.
#[repr(C)]
pub struct NestedPaddedOuter<T: ?Sized> {
    pub marker: u32,
    pub inner: PaddedTail<T>,
}

/// Out-of-line accessor pinning the `[Deref, Field(0), Deref, Index(k)]`
/// projection chain in its own MIR body.
#[inline(never)]
fn holder_node<'a>(h: &'a Holder<'a>, k: usize) -> &'a f32 {
    &h.0[k]
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Baseline: explicit slot-0 / slot-1 unroll, no closure. Always works.
    #[kernel]
    pub fn test_unrolled_baseline(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = input[i];

        let r0 = pair[0] + 1.0;
        let r1 = pair[1] + 1.0;

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// THE BUG: closure indexes `arr[idx]` with `idx` being the closure
    /// parameter. Called twice with literal 0 and 1.
    #[kernel]
    pub fn test_closure_indexes_into_array(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 { pair[k] + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Variation: closure does an `if k == 0` match instead of indexing.
    /// Tests whether the bug is specifically about the indexed GEP path or
    /// about closure-with-param in general. Expected: PASS — the match
    /// path doesn't go through the GEP-pinned-to-0 codegen.
    #[kernel]
    pub fn test_closure_indexes_via_match(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { pair[0] } else { pair[1] };
            v + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Indexed access through a `#[repr(transparent)]` struct wrapping
    /// `[f32; 2]`, with the indexing hidden behind a method call.
    #[kernel]
    pub fn test_closure_into_struct_wrapper(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 { pair.get(k) + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Indexed access goes through `pair.node(k)` returning `&f32`, then
    /// dereferences the result. Same as `test_closure_into_struct_wrapper`
    /// but the accessor returns a reference instead of by-value.
    #[kernel]
    pub fn test_closure_node_ref_access(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 { *pair.node(k) + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Closer to the actual failing case: closure body contains a warp
    /// shuffle, mimicking `tmp0 = warp::shuffle_xor_f32(v, ...)`. The
    /// hypothesis is that the buggy GEP-pin-to-zero only fires when the
    /// closure body contains certain other ops (shuffle intrinsic call,
    /// etc.) that block inlining/SROA from cleaning up the bad GEP.
    #[kernel]
    pub fn test_closure_with_shuffle(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let shuf = warp::shuffle_xor_f32(v, 1);
            v + shuf * 0.0 + 1.0 // shuf contribution = 0 so per-thread answer is still v + 1
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles, NO captures. Same straight-line topology
    /// as `test_closure_with_shuffle` but with a *second* shuffle call.
    #[kernel]
    pub fn test_closure_two_shuffles(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1); // = v (XOR with same lane back)
            // Per-thread answer ends up being v + 0 + 1 = v + 1.
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles but operand is `1.0` constant — NOT loaded
    /// from `pair.node(k)`. Does the closure parameter even get used?
    #[kernel]
    pub fn test_two_shuffles_no_indexed_load(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // The closure parameter `k` is used ONLY in this `if` to pick a
        // pre-loaded value. No GEP indexed by `k`.
        let a = pair.0[0];
        let b = pair.0[1];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { a } else { b };
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles, indexed load goes through a *bare array*
    /// `[f32; 2]` rather than the `Pair` newtype. Tests whether the bug
    /// is wrapper-specific or also reproduces on raw arrays.
    #[kernel]
    pub fn test_two_shuffles_raw_array(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let arr: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 {
            let v = arr[k];
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    // (test_two_sqrt_calls removed — it caused PTX gen to silently fail.)

    /// Bisection: 2 shuffles, indexed load through a non-`repr(transparent)`
    /// newtype. Tests whether the bug is specifically about `repr(transparent)`.
    #[kernel]
    pub fn test_two_shuffles_no_transparent(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = PairNoTransparent(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: like `test_closure_two_shuffles` but the closure is
    /// INLINED at the call site (no closure binding). Tests whether the
    /// bug is about the closure body or about the indexed access in any
    /// caller-with-param pattern.
    #[kernel]
    pub fn test_two_shuffles_inlined(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // No closure: two straight-line bodies, but each does indexed access
        // through pair.node(k) where k is a *runtime* value coming from a
        // local var (to defeat constant propagation at the access site).
        let mut k = 0usize;
        let v0 = *pair.node(k);
        let tmp0_0 = warp::shuffle_xor_f32(v0, 1);
        let tmp1_0 = warp::shuffle_xor_f32(tmp0_0, 1);
        let r0 = v0 + tmp1_0 * 0.0 + 1.0;

        k = 1;
        let v1 = *pair.node(k);
        let tmp0_1 = warp::shuffle_xor_f32(v1, 1);
        let tmp1_1 = warp::shuffle_xor_f32(tmp0_1, 1);
        let r1 = v1 + tmp1_1 * 0.0 + 1.0;

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Stronger trigger: closure body has a shuffle plus several constants
    /// captured from the enclosing scope. This keeps the same straight-line
    /// topology as the reduced failure while staying domain-neutral.
    #[kernel]
    pub fn test_closure_shuffle_with_captures(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // Captures with non-trivial use of the indexed load.
        let w_0: f32 = 0.0;
        let w_1: f32 = 1.0;
        let w2_0: f32 = 0.0;
        let w2_1: f32 = 0.0;

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let out0 = v * w_1 + tmp0 * w_0;
            let to_shfl = v * w2_1 + tmp0 * w2_0;
            let tmp1 = warp::shuffle_xor_f32(to_shfl, 1);
            out0 + tmp1 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Like `test_closure_shuffle_with_captures` but with the call-site
    /// idiom `[compute(0), compute(1)]` (constructing an array literal).
    #[kernel]
    pub fn test_closure_via_array_literal(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let w_0: f32 = 0.0;
        let w_1: f32 = 1.0;
        let w2_0: f32 = 0.0;
        let w2_1: f32 = 0.0;

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let out0 = v * w_1 + tmp0 * w_0;
            let to_shfl = v * w2_1 + tmp0 * w2_0;
            let tmp1 = warp::shuffle_xor_f32(to_shfl, 1);
            out0 + tmp1 + 1.0
        };

        let pair_out: [f32; 2] = [compute(0), compute(1)];

        if let Some(slot) = out.get_mut(idx) {
            *slot = pair_out[1] - pair_out[0];
        }
    }

    /// Control: closure captures the pre-loaded values, not the array.
    /// No GEP indexed by closure param. Expected: PASS.
    #[kernel]
    pub fn test_closure_pre_loaded_outside(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];
        let a = pair[0];
        let b = pair[1];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { a } else { b };
            v + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    // ── Address-walker regression kernels (PR #121 hardening) ──────────
    //
    // The remaining kernels pin the borrow/raw-pointer address shapes that
    // the unified `translate_place_address` walker must lower correctly:
    // mutable write-through, ConstantIndex tails, `addr_of!` raw pointers,
    // the exact issue #120 MIR shape, and Deref-in-the-tail projections.
    // Each writes a diff that is `+5.0` only when the computed address
    // points at the real element (not slot 0 and not a copy).

    /// (a) `&mut local.field[k]` write-through: both lanes start EQUAL, a
    /// `+5.0` is stored through the mutable borrow of lane `k = 1`, then
    /// the ORIGINAL local is read back. diff = 5 only if the write landed
    /// in the real `pair.0[1]`: a dropped index writes lane 0 (diff = -5),
    /// a value-copy borrow writes a temporary (diff = 0).
    #[kernel]
    pub fn test_mut_ref_writethrough(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let base = input[i][0];
        let mut pair = Pair([base, base]);
        // Runtime index (data-derived so it cannot const-fold):
        // input[0][0] == 0.0, so k == 1 at runtime.
        let k = ((input[0][0] as usize) & 1) + 1;
        {
            let lane: &mut f32 = &mut pair.0[k];
            *lane += 5.0;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = pair.0[1] - pair.0[0];
        }
    }

    /// (b) ConstantIndex tail: `&(*pr).0[1]` with LITERAL indices, i.e.
    /// projection `[Deref, Field(0), ConstantIndex{1}]`. The pre-#121 code
    /// dropped this tail exactly like the runtime-Index one.
    #[kernel]
    pub fn test_constant_index_tail(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);
        let pr: &Pair = &pair;
        let r1: &f32 = &pr.0[1];
        let r0: &f32 = &pr.0[0];
        if let Some(slot) = out.get_mut(idx) {
            *slot = *r1 - *r0;
        }
    }

    /// (c) Raw-pointer reads via `core::ptr::addr_of!`: exercises the
    /// `Rvalue::AddressOf` path (`&raw const (*pr).0[k]`) which had its own
    /// copy of the Field-only tail walk before the hardening.
    #[kernel]
    pub fn test_raw_addr_of_const(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);
        let pr: &Pair = &pair;
        // Runtime index: input[0][0] == 0.0, so k == 0 at runtime.
        let k = (input[0][0] as usize) & 1;
        let p1: *const f32 = core::ptr::addr_of!(pr.0[k + 1]);
        let p0: *const f32 = core::ptr::addr_of!(pr.0[k]);
        let (r1, r0) = unsafe { (*p1, *p0) };
        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// (c') Raw-pointer WRITE via `core::ptr::addr_of_mut!`
    /// (`&raw mut local[k]`): both lanes start equal, lane `k = 1` gets
    /// `+5.0` through the raw pointer, and the original local is read back.
    #[kernel]
    pub fn test_raw_addr_of_mut_writethrough(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let base = input[i][0];
        let mut acc = [base, base];
        let k = ((input[0][0] as usize) & 1) + 1; // == 1 at runtime
        let pm: *mut f32 = core::ptr::addr_of_mut!(acc[k]);
        unsafe {
            *pm += 5.0;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = acc[1] - acc[0];
        }
    }

    /// (d) The exact issue #120 shape: an `#[inline(never)]` accessor
    /// `fn node(p: &Pair, k: usize) -> &f32 { &p.0[k] }` whose own MIR is
    /// `_0 = &(((*_1).0)[_2])`.
    #[kernel]
    pub fn test_inline_never_node_fn(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);
        let k = (input[0][0] as usize) & 1; // == 0 at runtime
        let r1 = *node(&pair, k + 1);
        let r0 = *node(&pair, k);
        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// (e) Deref INSIDE the projection tail: `Holder<'a>(&'a [f32; 2])`,
    /// so `&hr.0[k]` is `[Deref, Field(0), Deref, Index(k)]` -- the walker
    /// must load the array reference out of the field before indexing.
    /// Uses both the pinned `#[inline(never)]` accessor and the direct
    /// in-kernel form.
    #[kernel]
    pub fn test_holder_deref_tail(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let arr: [f32; 2] = input[i];
        let h = Holder(&arr);
        let hr: &Holder = &h;
        let k = (input[0][0] as usize) & 1; // == 0 at runtime
        let r1: &f32 = &hr.0[k + 1];
        let r0: &f32 = holder_node(hr, k);
        if let Some(slot) = out.get_mut(idx) {
            *slot = *r1 - *r0;
        }
    }

    // ── Slice-value projection regressions ───────────────────────────────

    /// ConstantIndex directly on the `MirSliceType` value produced by
    /// projecting the unsized slice tail of a DST struct (issue #870).
    #[kernel]
    pub fn test_slice_tail_constant_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let concrete = SliceTail {
            head: 0,
            tail: input[i],
        };
        let value: &SliceTail<[f32]> = &concrete;

        let r0 = value.tail[0];
        let r1 = value.tail[1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Runtime Index directly on the `MirSliceType` value produced by
    /// projecting the unsized slice tail of a DST struct.
    #[kernel]
    pub fn test_slice_tail_runtime_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let concrete = SliceTail {
            head: 0,
            tail: input[i],
        };
        let value: &SliceTail<[f32]> = &concrete;

        // Data-derived so rustc keeps a runtime Index projection. The test
        // input makes k == 0 at runtime, preserving the +5.0 harness oracle.
        let k = (input[0][0] as usize) & 1;
        let r0 = value.tail[k];
        let r1 = value.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Padded tail offset (issue #870 repro layout): the `[u16]` tail sits
    /// at byte offset 10, behind `head: u64`, `tag: u8`, and one padding
    /// byte. Reads the tail with BOTH constant and runtime indices; the
    /// all-ones sentinels in `head` and `tag` make any wrong tail offset
    /// read 0xFF bytes (or shifted elements) and break the +5.0 oracle.
    #[kernel]
    pub fn test_slice_tail_padded_offset(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let a = input[i][0] as u16;
        let b = input[i][1] as u16;
        let concrete = PaddedTail {
            head: u64::MAX,
            tag: 0xFF,
            tail: [a, b, b, b],
        };
        let value: &PaddedTail<[u16]> = &concrete;

        // Constant indices into the padded tail: c1 - c0 == +5.
        let c0 = value.tail[0];
        let c1 = value.tail[1];

        // Data-derived so rustc keeps a runtime Index projection. The test
        // input makes k == 2 at runtime; tail[2] == tail[3], so the runtime
        // pair contributes 0 and the +5.0 harness oracle is preserved.
        let k = ((input[0][0] as usize) & 1) + 2;
        let r0 = value.tail[k];
        let r1 = value.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = (c1 as f32 - c0 as f32) + (r1 as f32 - r0 as f32);
        }
    }

    /// Issue #880 constant-index repro. The fat reference belongs to
    /// `NestedOuter<[f32]>`, while the indexed slice lives one struct field
    /// deeper at `inner.tail`. Metadata must survive `Field(inner)`.
    #[kernel]
    pub fn test_nested_slice_tail_constant_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let concrete = NestedOuter {
            id: u64::MAX,
            inner: NestedInner {
                head: u32::MAX,
                tail: input[i],
            },
        };
        let value: &NestedOuter<[f32]> = &concrete;

        let r0 = value.inner.tail[0];
        let r1 = value.inner.tail[1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Issue #880 runtime-index repro. The data-derived index prevents MIR
    /// from folding the final projection into a ConstantIndex.
    #[kernel]
    pub fn test_nested_slice_tail_runtime_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let concrete = NestedOuter {
            id: u64::MAX,
            inner: NestedInner {
                head: u32::MAX,
                tail: input[i],
            },
        };
        let value: &NestedOuter<[f32]> = &concrete;

        let k = (input[0][0] as usize) & 1;
        let r0 = value.inner.tail[k];
        let r1 = value.inner.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Nested padded issue #880 regression. The `[u16]` tail is ten bytes
    /// into `PaddedTail`, and that whole DST is itself the final field of
    /// `NestedPaddedOuter`. Both constant and runtime indexing must retain
    /// the outer fat pointer's length while computing the two field offsets.
    #[kernel]
    pub fn test_nested_slice_tail_padded_offset(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let a = input[i][0] as u16;
        let b = input[i][1] as u16;
        let concrete = NestedPaddedOuter {
            marker: u32::MAX,
            inner: PaddedTail {
                head: u64::MAX,
                tag: 0xFF,
                tail: [a, b, b, b],
            },
        };
        let value: &NestedPaddedOuter<[u16]> = &concrete;

        let c0 = value.inner.tail[0];
        let c1 = value.inner.tail[1];

        let k = ((input[0][0] as usize) & 1) + 2;
        let r0 = value.inner.tail[k];
        let r1 = value.inner.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = (c1 as f32 - c0 as f32) + (r1 as f32 - r0 as f32);
        }
    }

    // ── DST slice-tail address regressions (issue #881) ────────────────

    /// Constant-index write through a mutable DST slice tail. The store must
    /// target the original tail element, not a materialized copy.
    #[kernel]
    pub fn test_slice_tail_write_constant_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let base = input[i][0];
        let mut concrete = SliceTail {
            head: u32::MAX,
            tail: [base, base],
        };
        {
            let value: &mut SliceTail<[f32]> = &mut concrete;
            value.tail[1] += 5.0;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = concrete.tail[1] - concrete.tail[0];
        }
    }

    /// Runtime-index write through a mutable DST slice tail.
    #[kernel]
    pub fn test_slice_tail_write_runtime_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let base = input[i][0];
        let mut concrete = SliceTail {
            head: u32::MAX,
            tail: [base, base],
        };
        let k = ((input[0][0] as usize) & 1) + 1;
        {
            let value: &mut SliceTail<[f32]> = &mut concrete;
            value.tail[k] += 5.0;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = concrete.tail[1] - concrete.tail[0];
        }
    }

    /// Constant-index shared borrow of a DST slice-tail element.
    #[kernel]
    pub fn test_slice_tail_borrow_constant_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let mut concrete = SliceTail {
            head: u32::MAX,
            tail: input[i],
        };
        // Reborrow shared references from a writable fat-DST base. Internal
        // tail reconstruction keeps the base carrier writable; only the
        // final Reborrow establishes SharedRef/immutable.
        let value: &mut SliceTail<[f32]> = &mut concrete;

        let r0: &f32 = &value.tail[0];
        let r1: &f32 = &value.tail[1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = *r1 - *r0;
        }
    }

    /// Runtime-index shared borrow of a DST slice-tail element.
    #[kernel]
    pub fn test_slice_tail_borrow_runtime_index(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let concrete = SliceTail {
            head: u32::MAX,
            tail: input[i],
        };
        let value: &SliceTail<[f32]> = &concrete;

        let k = (input[0][0] as usize) & 1;
        let r0: &f32 = &value.tail[k];
        let r1: &f32 = &value.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = *r1 - *r0;
        }
    }

    /// Padded DST write regression. A literal index and a runtime index both
    /// target tail[1]; together their +2/+3 writes preserve the +5 oracle.
    #[kernel]
    pub fn test_slice_tail_write_padded(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let base = input[i][0] as u16;
        let mut concrete = PaddedTail {
            head: u64::MAX,
            tag: 0xFF,
            tail: [base, base, base, base],
        };
        {
            let value: &mut PaddedTail<[u16]> = &mut concrete;
            value.tail[1] += 2;
            let k = ((input[0][0] as usize) & 1) + 1;
            value.tail[k] += 3;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = (concrete.tail[1] as f32) - (concrete.tail[0] as f32);
        }
    }

    /// Padded DST borrow regression. Constant borrows contribute +5 while
    /// runtime borrows of equal tail elements contribute zero.
    #[kernel]
    pub fn test_slice_tail_borrow_padded(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }

        let a = input[i][0] as u16;
        let b = input[i][1] as u16;
        let concrete = PaddedTail {
            head: u64::MAX,
            tag: 0xFF,
            tail: [a, b, b, b],
        };
        let value: &PaddedTail<[u16]> = &concrete;

        let c0: &u16 = &value.tail[0];
        let c1: &u16 = &value.tail[1];

        let k = ((input[0][0] as usize) & 1) + 2;
        let r0: &u16 = &value.tail[k];
        let r1: &u16 = &value.tail[k + 1];

        if let Some(slot) = out.get_mut(idx) {
            *slot = (*c1 as f32 - *c0 as f32) + (*r1 as f32 - *r0 as f32);
        }
    }
}

const N: usize = 4;

fn make_input() -> Vec<[f32; 2]> {
    // Design: arr[1] - arr[0] = +5.0 for every thread, so any correct
    // kernel reports diff = 5.0. The miscompile reports diff = 0.0.
    (0..N).map(|i| [i as f32, i as f32 + 5.0]).collect()
}

/// Launches one kernel and prints its verdict. Returns `true` on PASS so
/// `main` can track failures and exit non-zero on any mismatch.
fn run_and_report<F>(name: &str, stream: &Arc<CudaStream>, launch: F) -> bool
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &DeviceBuffer<[f32; 2]>, &mut DeviceBuffer<f32>),
{
    let input = make_input();
    let dev_in = DeviceBuffer::from_host(stream, &input).unwrap();
    let mut dev_out = DeviceBuffer::<f32>::zeroed(stream, N).unwrap();

    launch(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        &dev_in,
        &mut dev_out,
    );

    let host_out = dev_out.to_host_vec(stream).unwrap();
    let all_correct = host_out.iter().all(|&d| (d - 5.0).abs() < 1e-6);
    let any_zero = host_out.iter().any(|&d| d.abs() < 1e-6);

    let verdict = if all_correct {
        "PASS"
    } else if any_zero {
        "FAIL (miscompile: closure idx ignored)"
    } else {
        "FAIL (unexpected)"
    };
    println!(
        "  {name:<40}  {verdict}    (diff[0..{}] = {:?})",
        N.min(4),
        &host_out[..N.min(4)]
    );
    all_correct
}

fn main() {
    println!("=== ref_index_projections: addresses of indexed places ===\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Load embedded PTX");

    println!("Each kernel computes r1 - r0 where r0 reads arr[0] and r1 reads arr[1].");
    println!("Input designed so the correct diff is +5.0 for every element.\n");

    let mut all_pass = true;

    all_pass &= run_and_report("test_unrolled_baseline", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_unrolled_baseline(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_closure_indexes_into_array",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_closure_indexes_into_array(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_closure_indexes_via_match", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_indexes_via_match(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_closure_into_struct_wrapper",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_closure_into_struct_wrapper(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_closure_pre_loaded_outside",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_closure_pre_loaded_outside(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_closure_node_ref_access", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_node_ref_access(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_closure_with_shuffle", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_with_shuffle(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_closure_two_shuffles", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_two_shuffles(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_two_shuffles_no_indexed_load",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_two_shuffles_no_indexed_load(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_two_shuffles_raw_array", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_two_shuffles_raw_array(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_two_shuffles_no_transparent",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_two_shuffles_no_transparent(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_two_shuffles_inlined", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_two_shuffles_inlined(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_closure_shuffle_with_captures",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_closure_shuffle_with_captures(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_closure_via_array_literal", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_closure_via_array_literal(s, cfg, i, o) }.expect("launch")
    });

    // Address-walker regression kernels (PR #121 hardening).
    all_pass &= run_and_report("test_mut_ref_writethrough", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_mut_ref_writethrough(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_constant_index_tail", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_constant_index_tail(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_raw_addr_of_const", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_raw_addr_of_const(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_raw_addr_of_mut_writethrough",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_raw_addr_of_mut_writethrough(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_inline_never_node_fn", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_inline_never_node_fn(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_holder_deref_tail", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_holder_deref_tail(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_slice_tail_constant_index", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_tail_constant_index(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_slice_tail_runtime_index", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_tail_runtime_index(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_slice_tail_padded_offset", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_tail_padded_offset(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report(
        "test_nested_slice_tail_constant_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_nested_slice_tail_constant_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_nested_slice_tail_runtime_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_nested_slice_tail_runtime_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_nested_slice_tail_padded_offset",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_nested_slice_tail_padded_offset(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_slice_tail_write_constant_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_tail_write_constant_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_slice_tail_write_runtime_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_tail_write_runtime_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_slice_tail_borrow_constant_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_tail_borrow_constant_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report(
        "test_slice_tail_borrow_runtime_index",
        &stream,
        |s, cfg, i, o| {
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe { module.test_slice_tail_borrow_runtime_index(s, cfg, i, o) }.expect("launch")
        },
    );
    all_pass &= run_and_report("test_slice_tail_write_padded", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_tail_write_padded(s, cfg, i, o) }.expect("launch")
    });
    all_pass &= run_and_report("test_slice_tail_borrow_padded", &stream, |s, cfg, i, o| {
        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.test_slice_tail_borrow_padded(s, cfg, i, o) }.expect("launch")
    });

    if all_pass {
        println!("\nSUCCESS: all kernels passed");
    } else {
        println!("\nFAILURE: at least one kernel returned a wrong diff");
        std::process::exit(1);
    }
}
