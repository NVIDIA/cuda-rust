/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kani proofs of the witness arithmetic in [`crate::thread`] and
//! [`crate::disjoint`]. Compiled only under `cargo kani`.
//!
//! Run from `cuda-oxide/`: `scripts/kani.sh [--harness <filter>]`, which is
//! `cargo kani -p cuda-device -Z stubbing` plus a check that every
//! `kani::cover!` was reachable.
//!
//! Every harness calls the shipped `__internal` mint functions and the shipped
//! `DisjointSlice::get_mut` on hardware registers replaced by the stubs in
//! [`hw`]. A `p1_*` or `p3_*` harness only compares two witnesses, or two
//! resolved pointers, minted by two threads of one launch. A `p2_*` harness
//! compares one witness against the exact formula evaluated in `u128`, which
//! is the one place a formula appears a second time.
//!
//! # Trusted base
//!
//! Assumed, not proven. Each item names the one place it is encoded.
//!
//! - **T1 hardware invariants.** For every axis `threadIdx < blockDim` and
//!   `blockIdx < gridDim`, and `blockDim.x * blockDim.y * blockDim.z <= 1024`.
//!   All threads of a launch read the same `blockDim` and `gridDim`.
//!   Encoded in [`hw::any_thread`] and nowhere else.
//!   `p2_index_1d_unchecked_coords` is the one harness that does not assume it.
//! - **T2 distinct threads.** Two distinct threads of one launch differ in at
//!   least one of the six index registers. Encoded in [`hw::any_thread_pair`].
//! - **T3 domain markers.** A `LaunchContext<Domain, _>` is minted only for a
//!   launch whose rank is at most `Domain::MAX_DIMENSIONS`, which host
//!   preparation checks. Encoded in [`hw::assume_domain`].
//! - **T4 intrinsic lowering.** The rustc backend replaces each `sreg.rs`
//!   function and `warp::lane_id` with the matching NVVM register read. The
//!   stubs in [`hw`] stand in for those reads, one stub per function.
//!   `lane_id` is modelled as the block-linear thread id modulo the warp
//!   size, which is how the hardware numbers lanes. `gridid`, `nsmid` and
//!   `smid` are stubbed to constants; no mint function reads them.
//! - **T5 Euclidean uniqueness.** For `t1, t2 < d`, `b1 * d + t1 == b2 * d +
//!   t2` implies `(b1, t1) == (b2, t2)`. Proof: with `b1 < b2`, `b1 * d + t1
//!   < (b1 + 1) * d <= b2 * d <= b2 * d + t2`. Kani discharges it at full width
//!   for every concrete `blockDim.x` and row width in the harness matrix, and
//!   exhaustively for every 8-bit row width and 8-bit row in
//!   `p3_resolve_symbolic_w_u8`.
//!   With `blockDim.x` symbolic, even bounded to the hardware limit
//!   `1..=1024` and with `y` and `z` fixed to one, `p1_index_1d` does not
//!   finish in 300 s under cadical or kissat (Kani 0.68, CBMC 6.11), so
//!   injectivity for block widths outside the matrix rests on this lemma.
//! - **T6 `U32Coordinates` contract.** Host preparation proved the active axis
//!   products fit `u32`. Encoded in [`hw::assume_u32_axis`], used only by the
//!   `*_u32_matches_math_*` harnesses.
//! - **T7 affine use.** `ThreadIndex` is `!Copy`, `!Clone`, `!Send`, `!Sync`
//!   and `get_mut` consumes it, so one witness yields one `&mut`. This is a
//!   type-system property, not a Kani theorem.
//! - **T8 stateless registers.** A mint function reads the registers and
//!   keeps no state between calls, so two mints run one after the other
//!   against one register file, with [`hw::enter`] switching the thread in
//!   between, model two threads of one launch. Encoded in
//!   [`fixture::two_threads`].
//! - **T9 target width and heap.** `usize` is 64 bits, as the crate already
//!   requires, and a resolving slice holds at most 2^38 elements, the heap
//!   Kani models. Encoded in [`fixture::any_len`].
//!
//! # Harness names
//!
//! The matrix in [`harnesses!`] is the only list; the mutant script names
//! harnesses from it. Suffixes: `_<D>` is the launch domain (`unknown`, `d1`,
//! `d2`, `d3`), `_bx<N>` a concrete `blockDim.x`, `_b<X>x<Y>` a concrete
//! `blockDim.x` and `.y`, `_s<S>` a const row stride (`s2p31` is `1 << 31`,
//! `smax` is `usize::MAX`), `_w<W>` a concrete runtime row width.
//!
//! - `p1_*` injectivity: two distinct threads never resolve to one element.
//!   `p1_index_1d_*`, `p1_warp_index_*`, `p1_index_2d_*` and
//!   `p1_index_2d_runtime_witnesses_distinct_*`.
//! - `p2_*` sentinel and bounds, one thread: `p2_index_1d_sentinel_*` and
//!   `p2_index_1d_unchecked_coords` (the `is_valid` biconditional),
//!   `p2_warp_index_no_wrap`, `p2_index_2d_*_no_wrap` and
//!   `p2_index_2d_runtime_coords` (`Some` exactly when the launch rank and
//!   the exact formula allow it, and then it is the exact formula),
//!   `p2_get_mut_flat`, `p2_get_mut_indexed_*` and `p2_get_mut_runtime_2d_*`
//!   (`Some` exactly when in bounds, and the pointer is `base + i`),
//!   `p2_zst_resolves_nothing`, `p2_zero_width_resolves_nothing`, and the
//!   `*_u32_matches_math_*` pair (the wrapping `u32` result is the exact
//!   value under T6).
//! - `p3_*` per-slice `Runtime2DIndex` resolution: `p3_resolve_*`,
//!   `p3_cross_slice_no_alias_*` (the historical bug: witnesses minted from a
//!   width-5 and a width-100 slice and resolved through either), and
//!   `p3_witness_width_independent` (a witness carries coordinates, never the
//!   minting slice's width).
//!
//! `p3_resolve_*` run with `blockDim = [1, 1]`, so `row` and `col` are block
//! indices and the solver spends its budget on `row * width` for a wide or
//! symbolic width. The in-block half of the coordinate arithmetic is proven
//! for `Runtime2DIndex` by `p2_index_2d_runtime_coords` under a symbolic
//! geometry and by the `_b32x8` and `_b16x16` harnesses.

#![allow(non_snake_case)]

extern crate std;

use core::ptr::{NonNull, from_mut};
use core::sync::atomic::{AtomicU32, Ordering::Relaxed};

use crate::disjoint::DisjointSlice;
use crate::thread::__internal::{
    self as mint, Domain1, Domain2, Domain3, LaunchDomain, NativeCoordinates, U32Coordinates,
    UnknownDomain,
};
use crate::thread::{Index1D, Index2D, Runtime2DIndex, ThreadIndex};

/// One launch and one current thread, held in statics because a Kani stub is
/// a global function replacement.
// The register stubs are reached only through `#[kani::stub]`, which rustc
// cannot see as a use.
#[allow(dead_code)]
mod hw {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct Geometry {
        pub block_dim: [u32; 3],
        pub grid_dim: [u32; 3],
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct Thread {
        pub thread_idx: [u32; 3],
        pub block_idx: [u32; 3],
    }

    /// Threads per block the hardware allows (T1).
    pub const MAX_BLOCK_THREADS: u32 = 1024;

    const ZERO: AtomicU32 = AtomicU32::new(0);
    static BLOCK_DIM: [AtomicU32; 3] = [ZERO, ZERO, ZERO];
    static GRID_DIM: [AtomicU32; 3] = [ZERO, ZERO, ZERO];
    static THREAD_IDX: [AtomicU32; 3] = [ZERO, ZERO, ZERO];
    static BLOCK_IDX: [AtomicU32; 3] = [ZERO, ZERO, ZERO];

    fn store(regs: &[AtomicU32; 3], values: [u32; 3]) {
        for (reg, value) in regs.iter().zip(values) {
            reg.store(value, Relaxed);
        }
    }

    impl Geometry {
        pub fn any() -> Self {
            Geometry {
                block_dim: kani::any(),
                grid_dim: kani::any(),
            }
        }

        pub fn block_x(block_x: u32) -> Self {
            Self::block_xy([block_x, kani::any()])
        }

        pub fn block_xy([block_x, block_y]: [u32; 2]) -> Self {
            let mut g = Self::any();
            g.block_dim[0] = block_x;
            g.block_dim[1] = block_y;
            g
        }

        pub fn unit(&self, axis: usize) -> bool {
            self.block_dim[axis] == 1 && self.grid_dim[axis] == 1
        }

        pub fn one_dimensional(&self) -> bool {
            self.unit(1) && self.unit(2)
        }
    }

    pub fn install(g: &Geometry) {
        store(&BLOCK_DIM, g.block_dim);
        store(&GRID_DIM, g.grid_dim);
    }

    pub fn enter(t: &Thread) {
        store(&THREAD_IDX, t.thread_idx);
        store(&BLOCK_IDX, t.block_idx);
    }

    pub fn run(g: &Geometry, t: &Thread) {
        install(g);
        enter(t);
    }

    /// A symbolic thread of `g` under T1.
    pub fn any_thread(g: &Geometry) -> Thread {
        let [bx, by, bz] = g.block_dim;
        kani::assume(bx as u128 * by as u128 * bz as u128 <= MAX_BLOCK_THREADS as u128);
        let t = Thread {
            thread_idx: kani::any(),
            block_idx: kani::any(),
        };
        for axis in 0..3 {
            kani::assume(t.thread_idx[axis] < g.block_dim[axis]);
            kani::assume(t.block_idx[axis] < g.grid_dim[axis]);
        }
        t
    }

    /// Two distinct threads of `g` (T2).
    pub fn any_thread_pair(g: &Geometry) -> (Thread, Thread) {
        let (a, b) = (any_thread(g), any_thread(g));
        kani::assume(a != b);
        (a, b)
    }

    /// T3: the launch rank the host checked before minting a `D` scope.
    pub fn assume_domain<D: LaunchDomain>(g: &Geometry) {
        match D::MAX_DIMENSIONS {
            1 => kani::assume(g.unit(1) && g.unit(2)),
            2 => kani::assume(g.unit(2)),
            _ => {}
        }
    }

    /// T6: host preparation proved `gridDim * blockDim <= 2^32` on `axis`.
    pub fn assume_u32_axis(g: &Geometry, axis: usize) {
        kani::assume(g.grid_dim[axis] as u64 * g.block_dim[axis] as u64 <= 1 << 32);
    }

    pub fn threadIdx_x() -> u32 {
        THREAD_IDX[0].load(Relaxed)
    }
    pub fn threadIdx_y() -> u32 {
        THREAD_IDX[1].load(Relaxed)
    }
    pub fn threadIdx_z() -> u32 {
        THREAD_IDX[2].load(Relaxed)
    }
    pub fn blockIdx_x() -> u32 {
        BLOCK_IDX[0].load(Relaxed)
    }
    pub fn blockIdx_y() -> u32 {
        BLOCK_IDX[1].load(Relaxed)
    }
    pub fn blockIdx_z() -> u32 {
        BLOCK_IDX[2].load(Relaxed)
    }
    pub fn blockDim_x() -> u32 {
        BLOCK_DIM[0].load(Relaxed)
    }
    pub fn blockDim_y() -> u32 {
        BLOCK_DIM[1].load(Relaxed)
    }
    pub fn blockDim_z() -> u32 {
        BLOCK_DIM[2].load(Relaxed)
    }
    pub fn gridDim_x() -> u32 {
        GRID_DIM[0].load(Relaxed)
    }
    pub fn gridDim_y() -> u32 {
        GRID_DIM[1].load(Relaxed)
    }
    pub fn gridDim_z() -> u32 {
        GRID_DIM[2].load(Relaxed)
    }
    pub fn gridid() -> u64 {
        0
    }
    pub fn nsmid() -> u32 {
        1
    }
    pub fn smid() -> u32 {
        0
    }
    /// T4: `%laneid` is the block-linear thread id modulo the warp size.
    pub fn lane_id() -> u32 {
        let [tx, ty, tz] = [threadIdx_x(), threadIdx_y(), threadIdx_z()];
        let [bx, by, _] = [blockDim_x(), blockDim_y(), blockDim_z()];
        (tx + bx * (ty + by * tz)) % 32
    }
}

macro_rules! hw_proof {
    ($(#[$m:meta])* fn $name:ident() $body:block) => {
        $(#[$m])*
        #[kani::proof]
        #[kani::stub(crate::thread::threadIdx_x, hw::threadIdx_x)]
        #[kani::stub(crate::thread::threadIdx_y, hw::threadIdx_y)]
        #[kani::stub(crate::thread::threadIdx_z, hw::threadIdx_z)]
        #[kani::stub(crate::thread::blockIdx_x, hw::blockIdx_x)]
        #[kani::stub(crate::thread::blockIdx_y, hw::blockIdx_y)]
        #[kani::stub(crate::thread::blockIdx_z, hw::blockIdx_z)]
        #[kani::stub(crate::thread::blockDim_x, hw::blockDim_x)]
        #[kani::stub(crate::thread::blockDim_y, hw::blockDim_y)]
        #[kani::stub(crate::thread::blockDim_z, hw::blockDim_z)]
        #[kani::stub(crate::thread::gridDim_x, hw::gridDim_x)]
        #[kani::stub(crate::thread::gridDim_y, hw::gridDim_y)]
        #[kani::stub(crate::thread::gridDim_z, hw::gridDim_z)]
        #[kani::stub(crate::thread::gridid, hw::gridid)]
        #[kani::stub(crate::thread::nsmid, hw::nsmid)]
        #[kani::stub(crate::thread::smid, hw::smid)]
        #[kani::stub(crate::warp::lane_id, hw::lane_id)]
        fn $name() $body
    };
}

macro_rules! harnesses {
    ($($(#[$m:meta])* $name:ident => $body:expr;)*) => {
        $(hw_proof! { $(#[$m])* fn $name() { $body; } })*
    };
}

mod fixture {
    use super::*;
    use std::alloc::{Layout, alloc};

    pub fn two_threads<W>(
        g: &hw::Geometry,
        a: &hw::Thread,
        b: &hw::Thread,
        mut mint: impl FnMut() -> W,
    ) -> (W, W) {
        hw::run(g, a);
        let wa = mint();
        hw::enter(b);
        let wb = mint();
        (wa, wb)
    }

    /// A symbolic element count a resolving slice can have. Kani's allocator
    /// rejects zero-sized requests (an empty slice resolves nothing anyway),
    /// and 2^38 `u32`s is the 2^40-byte ceiling Kani's heap models.
    pub fn any_len() -> usize {
        let len: usize = kani::any();
        kani::assume(1 <= len && len <= 1 << 38);
        len
    }

    /// A live, exclusively owned buffer of `len` `T`s on Kani's heap, as
    /// `from_raw_parts*` requires. `get_mut` derives `&mut *ptr.add(i)`, and
    /// Kani rejects that on a dangling base, so the allocation is real.
    pub fn heap<T>(len: usize) -> *mut T {
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { alloc(layout) }.cast::<T>();
        kani::assume(!ptr.is_null());
        ptr
    }

    pub fn flat_slice<'a>(len: usize) -> DisjointSlice<'a, u32> {
        unsafe { DisjointSlice::from_raw_parts(heap(len), len) }
    }

    pub fn runtime_2d_slice<'a>(len: usize, width: u32) -> DisjointSlice<'a, u32, Runtime2DIndex> {
        unsafe { DisjointSlice::from_raw_parts_with_space(heap(len), len, width) }
    }

    pub fn axis(block: u32, size: u32, thread: u32) -> u128 {
        block as u128 * size as u128 + thread as u128
    }

    pub fn valid_index<S: crate::thread::FlatIndexSpace>(w: &ThreadIndex<'_, S>) -> Option<usize> {
        w.is_valid().then(|| w.get())
    }

    pub fn coords(w: &ThreadIndex<'_, Runtime2DIndex>) -> (usize, usize) {
        (w.row(), w.col())
    }

    pub fn assert_no_alias<T: PartialEq>(a: Option<T>, b: Option<T>) {
        if let (Some(a), Some(b)) = (a, b) {
            assert!(a != b, "two threads resolved to one element");
        }
    }
}

mod prop {
    use super::*;

    pub fn p1_index_1d<D: LaunchDomain>(g: hw::Geometry) {
        hw::assume_domain::<D>(&g);
        let (a, b) = hw::any_thread_pair(&g);
        let scope = unsafe { mint::make_kernel_scope::<D, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || mint::index_1d(&scope));
        kani::cover!(wa.is_valid() && wb.is_valid());
        // Kani reports a cover in a dead branch as unreachable, so the
        // geometries where y cannot differ discharge it by implication.
        let y_can_differ = D::MAX_DIMENSIONS != 1 && g.block_dim[0] <= hw::MAX_BLOCK_THREADS / 2;
        kani::cover!(!y_can_differ || a.thread_idx[1] != b.thread_idx[1]);
        fixture::assert_no_alias(fixture::valid_index(&wa), fixture::valid_index(&wb));
    }

    pub fn p1_warp_index(g: hw::Geometry) {
        let (a, b) = hw::any_thread_pair(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || mint::warp_index(&scope));
        let (ia, ib) = (wa.map(|w| w.get()), wb.map(|w| w.get()));
        kani::cover!(ia.is_some() && ib.is_some() && a.block_idx[0] != b.block_idx[0]);
        kani::cover!(a.thread_idx[0] % 32 != 0 && ia.is_none());
        fixture::assert_no_alias(ia, ib);
    }

    pub fn p1_index_2d<D: LaunchDomain, const S: usize>(block: [u32; 2]) {
        let g = hw::Geometry::block_xy(block);
        hw::assume_domain::<D>(&g);
        let (a, b) = hw::any_thread_pair(&g);
        let scope = unsafe { mint::make_kernel_scope::<D, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || mint::index_2d::<S>(&scope));
        let ia = wa.as_ref().and_then(fixture::valid_index);
        let ib = wb.as_ref().and_then(fixture::valid_index);
        kani::cover!(ia.is_some() && ib.is_some() && a.block_idx != b.block_idx);
        kani::cover!(ia.is_none());
        fixture::assert_no_alias(ia, ib);
    }

    pub fn p1_index_2d_runtime_witnesses_distinct(block: [u32; 2]) {
        let g = hw::Geometry::block_xy(block);
        let (a, b) = hw::any_thread_pair(&g);
        let slice = fixture::runtime_2d_slice(1, kani::any());
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || mint::index_2d_runtime(&scope, &slice));
        let (ca, cb) = (
            wa.as_ref().map(fixture::coords),
            wb.as_ref().map(fixture::coords),
        );
        kani::cover!(ca.is_some() && cb.is_some() && a.block_idx != b.block_idx);
        kani::cover!(
            block == [1, 1] || (ca.is_some() && cb.is_some() && a.thread_idx != b.thread_idx)
        );
        fixture::assert_no_alias(ca, cb);
    }

    pub fn p2_index_1d_sentinel<D: LaunchDomain>(g: hw::Geometry) {
        hw::assume_domain::<D>(&g);
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<D, NativeCoordinates>() };
        hw::run(&g, &t);
        let w = mint::index_1d(&scope);
        kani::cover!(w.is_valid());
        kani::cover!(D::MAX_DIMENSIONS == 1 || !w.is_valid());
        assert!(
            w.is_valid() == (w.get() != usize::MAX),
            "is_valid disagrees with the sentinel"
        );
        assert!(
            w.is_valid() == g.one_dimensional(),
            "rank check disagrees with the launch rank"
        );
    }

    pub fn p2_index_1d_unchecked_coords() {
        let g = hw::Geometry::any();
        let t = hw::Thread {
            thread_idx: kani::any(),
            block_idx: kani::any(),
        };
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        hw::run(&g, &t);
        let w = mint::index_1d(&scope);
        let exact = fixture::axis(t.block_idx[0], g.block_dim[0], t.thread_idx[0]);
        kani::cover!(w.is_valid() && t.thread_idx[0] >= g.block_dim[0]);
        kani::cover!(!w.is_valid() && g.one_dimensional());
        assert!(
            w.is_valid() == (w.get() != usize::MAX),
            "is_valid disagrees with the sentinel"
        );
        assert!(
            w.is_valid()
                == (g.one_dimensional() && g.block_dim[0] != 0 && exact < usize::MAX as u128),
            "validity disagrees with the launch rank and the arithmetic"
        );
        assert!(
            !w.is_valid() || w.get() as u128 == exact,
            "valid witness is not the exact index"
        );
    }

    pub fn p2_warp_index_no_wrap(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        hw::run(&g, &t);
        let w = mint::warp_index(&scope).map(|w| w.get());
        let exact = t.block_idx[0] as u128 * g.block_dim[0].div_ceil(32) as u128
            + (t.thread_idx[0] / 32) as u128;
        kani::cover!(w.is_some_and(|i| i > 0));
        kani::cover!(g.one_dimensional() && w.is_none());
        assert!(
            w.is_some() == (g.one_dimensional() && t.thread_idx[0] % 32 == 0),
            "warp witness granted to the wrong lanes"
        );
        assert!(
            w.is_none_or(|i| i as u128 == exact),
            "warp index is not the exact formula"
        );
    }

    pub fn p2_index_2d_no_wrap<const S: usize>(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        hw::run(&g, &t);
        let w = mint::index_2d::<S>(&scope);
        let row = fixture::axis(t.block_idx[1], g.block_dim[1], t.thread_idx[1]);
        let col = fixture::axis(t.block_idx[0], g.block_dim[0], t.thread_idx[0]);
        kani::cover!(w.is_some() && row > 0);
        kani::cover!(w.as_ref().is_some_and(|w| w.is_valid()));
        kani::cover!(w.is_none() && g.unit(2));
        let exact = row * S as u128 + col;
        assert!(
            w.is_some() == (g.unit(2) && col < S as u128 && exact <= usize::MAX as u128),
            "Some disagrees with the launch rank, the column bound and the arithmetic"
        );
        if let Some(w) = w {
            assert!(
                w.get() as u128 == exact,
                "flat index is not the exact formula"
            );
            assert!(
                w.is_valid() == (exact != usize::MAX as u128),
                "validity disagrees with the sentinel"
            );
        }
    }

    pub fn p2_index_2d_runtime_coords(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let width: u32 = kani::any();
        let slice = fixture::runtime_2d_slice(1, width);
        hw::run(&g, &t);
        let w = mint::index_2d_runtime(&scope, &slice);
        let c = w.as_ref().map(fixture::coords);
        let row = fixture::axis(t.block_idx[1], g.block_dim[1], t.thread_idx[1]);
        let col = fixture::axis(t.block_idx[0], g.block_dim[0], t.thread_idx[0]);
        kani::cover!(c.is_some() && t.thread_idx[1] > 0 && t.block_idx[1] > 0);
        kani::cover!(c.is_none() && g.unit(2) && width > 0 && row <= u32::MAX as u128);
        kani::cover!(c.is_none() && g.unit(2) && col < width as u128);
        assert!(
            c.is_some()
                == (width != 0 && g.unit(2) && col < width as u128 && row <= u32::MAX as u128),
            "Some disagrees with the width, the launch rank and the exact coordinates"
        );
        if let Some((r, c)) = c {
            assert!(
                (r as u128, c as u128) == (row, col),
                "coordinates are not the exact formula"
            );
        }
    }

    pub fn p2_get_mut_flat(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let len = fixture::any_len();
        let mut slice = fixture::flat_slice(len);
        let base = slice.as_mut_ptr();
        hw::run(&g, &t);
        let w = mint::index_1d(&scope);
        let i = fixture::valid_index(&w);
        let p = slice.get_mut(w).map(from_mut);
        kani::cover!(p.is_some() && i.is_some_and(|i| i > 0));
        kani::cover!(i.is_some_and(|i| i >= len));
        assert!(
            p.is_some() == i.is_some_and(|i| i < len),
            "flat get_mut disagrees with is_valid && i < len"
        );
        if let (Some(p), Some(i)) = (p, i) {
            assert!(p == unsafe { base.add(i) }, "flat get_mut is not base + i");
        }
    }

    pub fn p2_get_mut_indexed_flat(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let len = fixture::any_len();
        let mut slice = fixture::flat_slice(len);
        let base = slice.as_mut_ptr();
        hw::run(&g, &t);
        let i = fixture::valid_index(&mint::index_1d(&scope));
        let got = slice
            .get_mut_indexed(&scope)
            .map(|(p, w)| (from_mut(p), w.get()));
        kani::cover!(got.is_some() && i.is_some_and(|i| i > 0));
        kani::cover!(i.is_some_and(|i| i >= len));
        assert!(
            got.is_some() == i.is_some_and(|i| i < len),
            "flat get_mut_indexed disagrees with is_valid && i < len"
        );
        if let (Some((p, j)), Some(i)) = (got, i) {
            assert!(j == i, "flat get_mut_indexed returned another index");
            assert!(
                p == unsafe { base.add(i) },
                "flat get_mut_indexed is not base + i"
            );
        }
    }

    pub fn p2_get_mut_indexed_2d<const S: usize>(block: [u32; 2]) {
        let g = hw::Geometry::block_xy(block);
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let len = fixture::any_len();
        let mut slice =
            unsafe { DisjointSlice::<u32, Index2D<S>>::from_raw_parts(fixture::heap(len), len) };
        let base = slice.as_mut_ptr();
        hw::run(&g, &t);
        let i = mint::index_2d::<S>(&scope)
            .as_ref()
            .and_then(fixture::valid_index);
        let got = slice
            .get_mut_indexed(&scope)
            .map(|(p, w)| (from_mut(p), w.get()));
        kani::cover!(got.is_some() && i.is_some_and(|i| i >= S));
        kani::cover!(i.is_some_and(|i| i >= len));
        kani::cover!(i.is_none() && g.unit(2));
        assert!(
            got.is_some() == i.is_some_and(|i| i < len),
            "2D get_mut_indexed disagrees with index_2d and i < len"
        );
        if let (Some((p, j)), Some(i)) = (got, i) {
            assert!(j == i, "2D get_mut_indexed returned another index");
            assert!(
                p == unsafe { base.add(i) },
                "2D get_mut_indexed is not base + i"
            );
        }
    }

    pub fn p2_get_mut_runtime_2d(width: u32, block: [u32; 2]) {
        let g = hw::Geometry::block_xy(block);
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let len = fixture::any_len();
        let mut slice = fixture::runtime_2d_slice(len, width);
        let base = slice.as_mut_ptr();
        hw::run(&g, &t);
        let w = mint::index_2d_runtime(&scope, &slice);
        let c = w.as_ref().map(fixture::coords);
        let p = w.and_then(|w| slice.get_mut(w)).map(from_mut);
        let flat = c.map(|(row, col)| row as u128 * width as u128 + col as u128);
        kani::cover!(p.is_some() && c.is_some_and(|(row, _)| row > 0));
        kani::cover!(c.is_some() && p.is_none());
        assert!(
            c.is_none_or(|(_, col)| col < width as usize),
            "witness column outside the row width"
        );
        assert!(
            p.is_some() == flat.is_some_and(|f| f < len as u128),
            "runtime 2D get_mut disagrees with row * width + col < len"
        );
        if let (Some(p), Some(flat)) = (p, flat) {
            assert!(
                p == unsafe { base.add(flat as usize) },
                "runtime 2D get_mut is not base + row * width + col"
            );
        }
    }

    pub fn p2_zst_resolves_nothing(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let len = fixture::any_len();
        let base = NonNull::<()>::dangling().as_ptr();
        let mut flat = unsafe { DisjointSlice::<(), Index1D>::from_raw_parts(base, len) };
        let mut grid = unsafe {
            DisjointSlice::<(), Runtime2DIndex>::from_raw_parts_with_space(base, len, 100)
        };
        hw::run(&g, &t);
        let w1 = mint::index_1d(&scope);
        let w2 = mint::index_2d_runtime(&scope, &grid);
        kani::cover!(w1.in_bounds(len));
        kani::cover!(w2.is_some());
        assert!(
            flat.get_mut(w1).is_none(),
            "flat get_mut granted a zero-sized element"
        );
        assert!(
            w2.and_then(|w| grid.get_mut(w)).is_none(),
            "runtime 2D get_mut granted a zero-sized element"
        );
    }

    pub fn p2_zero_width_resolves_nothing(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let slice = fixture::runtime_2d_slice(fixture::any_len(), 0);
        hw::run(&g, &t);
        kani::cover!(g.unit(2));
        assert!(
            mint::index_2d_runtime(&scope, &slice).is_none(),
            "zero row width minted a witness"
        );
    }

    pub fn p2_index_1d_u32_matches_math(g: hw::Geometry) {
        hw::assume_domain::<Domain1>(&g);
        hw::assume_u32_axis(&g, 0);
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<Domain1, U32Coordinates>() };
        hw::run(&g, &t);
        let w = mint::index_1d_u32(&scope);
        kani::cover!(w.get() > u32::MAX / 2);
        assert!(
            w.get() as u128 == fixture::axis(t.block_idx[0], g.block_dim[0], t.thread_idx[0]),
            "u32 index wrapped"
        );
    }

    pub fn p2_coord_2d_u32_matches_math(block: [u32; 2]) {
        let g = hw::Geometry::block_xy(block);
        hw::assume_domain::<Domain2>(&g);
        hw::assume_u32_axis(&g, 0);
        hw::assume_u32_axis(&g, 1);
        let t = hw::any_thread(&g);
        let scope = unsafe { mint::make_kernel_scope::<Domain2, U32Coordinates>() };
        hw::run(&g, &t);
        let w = mint::coord_2d_u32(&scope);
        kani::cover!(w.row() > u32::MAX / 2 && w.col() > u32::MAX / 2);
        assert!(
            w.row() as u128 == fixture::axis(t.block_idx[1], g.block_dim[1], t.thread_idx[1]),
            "u32 row wrapped"
        );
        assert!(
            w.col() as u128 == fixture::axis(t.block_idx[0], g.block_dim[0], t.thread_idx[0]),
            "u32 column wrapped"
        );
    }

    pub fn p3_resolve<D: LaunchDomain>(width: u32) {
        p3_resolve_capped::<D>(width, 1 << 38, u32::MAX);
    }

    /// [`p3_resolve`] with the slice length and `gridDim.y` capped. The solver
    /// cost of `row * width` grows with the one bits of `width` and the bits of
    /// `row`, so two harnesses shrink `row`:
    ///
    /// - `p3_resolve_w65535`: 16 one bits. Uncapped it does not finish in 900 s
    ///   under cadical or kissat, while `65536` and `65537` take 5 s. Capping
    ///   the length at 2^32 (rows below 2^16) takes 74 s, 2^30 (rows below
    ///   2^14) 20 s, 2^24 6 s. The harness runs at 2^30.
    /// - `p3_resolve_symbolic_w_u8`: a symbolic width. Uncapped it does not
    ///   finish in 300 s under either solver. With `gridDim.y <= 256`, so
    ///   `row` is an 8-bit value, it takes 28 s.
    pub fn p3_resolve_capped<D: LaunchDomain>(width: u32, max_len: usize, max_grid_y: u32) {
        let g = hw::Geometry::block_xy([1, 1]);
        hw::assume_domain::<D>(&g);
        kani::assume(g.grid_dim[1] <= max_grid_y);
        let (a, b) = hw::any_thread_pair(&g);
        let len = fixture::any_len();
        kani::assume(len <= max_len);
        let mut slice = fixture::runtime_2d_slice(len, width);
        let scope = unsafe { mint::make_kernel_scope::<D, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || mint::index_2d_runtime(&scope, &slice));
        let row_a = wa.as_ref().map(|w| w.row());
        let pa = wa.and_then(|w| slice.get_mut(w)).map(from_mut);
        let pb = wb.and_then(|w| slice.get_mut(w)).map(from_mut);
        kani::cover!(pa.is_some() && pb.is_some() && row_a.is_some_and(|r| r > 0));
        fixture::assert_no_alias(pa, pb);
    }

    pub fn p3_resolve_symbolic_w_u8() {
        let width: u32 = kani::any();
        kani::assume(1 <= width && width <= 255);
        p3_resolve_capped::<UnknownDomain>(width, 1 << 38, 256);
    }

    pub fn p3_cross_slice_no_alias(block: [u32; 2]) {
        const NARROW: u32 = 5;
        const WIDE: u32 = 100;
        let g = hw::Geometry::block_xy(block);
        let (a, b) = hw::any_thread_pair(&g);
        let mut narrow = fixture::runtime_2d_slice(fixture::any_len(), NARROW);
        let mut wide = fixture::runtime_2d_slice(fixture::any_len(), WIDE);
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        let (wa, wb) = fixture::two_threads(&g, &a, &b, || {
            let from_narrow: bool = kani::any();
            let w = if from_narrow {
                mint::index_2d_runtime(&scope, &narrow)
            } else {
                mint::index_2d_runtime(&scope, &wide)
            };
            (w, from_narrow)
        });
        let (ca, cb) = (
            wa.0.as_ref().map(fixture::coords),
            wb.0.as_ref().map(fixture::coords),
        );
        let through_wide: bool = kani::any();
        let (pa, pb) = if through_wide {
            (
                wa.0.and_then(|w| wide.get_mut(w)).map(from_mut),
                wb.0.and_then(|w| wide.get_mut(w)).map(from_mut),
            )
        } else {
            (
                wa.0.and_then(|w| narrow.get_mut(w)).map(from_mut),
                wb.0.and_then(|w| narrow.get_mut(w)).map(from_mut),
            )
        };
        kani::cover!(
            through_wide
                && wa.1
                && !wb.1
                && ca == Some((1, 0))
                && cb == Some((0, NARROW as usize))
                && pa.is_some()
                && pb.is_some()
        );
        kani::cover!(
            !through_wide
                && !wa.1
                && wb.1
                && ca == Some((0, NARROW as usize + 2))
                && cb == Some((1, 2))
                && pb.is_some()
        );
        fixture::assert_no_alias(pa, pb);
    }

    pub fn p3_witness_width_independent(g: hw::Geometry) {
        let t = hw::any_thread(&g);
        let (w1, w2): (u32, u32) = (kani::any(), kani::any());
        kani::assume(w1 >= 1 && w2 >= 1);
        let (s1, s2) = (
            fixture::runtime_2d_slice(1, w1),
            fixture::runtime_2d_slice(1, w2),
        );
        let scope = unsafe { mint::make_kernel_scope::<UnknownDomain, NativeCoordinates>() };
        hw::run(&g, &t);
        let c1 = mint::index_2d_runtime(&scope, &s1)
            .as_ref()
            .map(fixture::coords);
        let c2 = mint::index_2d_runtime(&scope, &s2)
            .as_ref()
            .map(fixture::coords);
        kani::cover!(c1.is_some() && c2.is_some() && w1 != w2);
        kani::cover!(c1.is_some() && c2.is_none());
        if let (Some(c1), Some(c2)) = (c1, c2) {
            assert!(
                c1 == c2,
                "witness coordinates depend on the minting slice's width"
            );
        }
    }
}

harnesses! {
    p1_index_1d_unknown_bx1 => prop::p1_index_1d::<UnknownDomain>(hw::Geometry::block_x(1));
    p1_index_1d_unknown_bx32 => prop::p1_index_1d::<UnknownDomain>(hw::Geometry::block_x(32));
    p1_index_1d_unknown_bx48 => prop::p1_index_1d::<UnknownDomain>(hw::Geometry::block_x(48));
    p1_index_1d_unknown_bx256 => prop::p1_index_1d::<UnknownDomain>(hw::Geometry::block_x(256));
    p1_index_1d_unknown_bx1024 => prop::p1_index_1d::<UnknownDomain>(hw::Geometry::block_x(1024));
    p1_index_1d_d1_bx1 => prop::p1_index_1d::<Domain1>(hw::Geometry::block_x(1));
    p1_index_1d_d1_bx32 => prop::p1_index_1d::<Domain1>(hw::Geometry::block_x(32));
    p1_index_1d_d1_bx48 => prop::p1_index_1d::<Domain1>(hw::Geometry::block_x(48));
    p1_index_1d_d1_bx256 => prop::p1_index_1d::<Domain1>(hw::Geometry::block_x(256));
    p1_index_1d_d1_bx1024 => prop::p1_index_1d::<Domain1>(hw::Geometry::block_x(1024));
    p1_index_1d_d2_bx1 => prop::p1_index_1d::<Domain2>(hw::Geometry::block_x(1));
    p1_index_1d_d2_bx32 => prop::p1_index_1d::<Domain2>(hw::Geometry::block_x(32));
    p1_index_1d_d2_bx48 => prop::p1_index_1d::<Domain2>(hw::Geometry::block_x(48));
    p1_index_1d_d2_bx256 => prop::p1_index_1d::<Domain2>(hw::Geometry::block_x(256));
    p1_index_1d_d2_bx1024 => prop::p1_index_1d::<Domain2>(hw::Geometry::block_x(1024));
    p1_index_1d_d3_bx1 => prop::p1_index_1d::<Domain3>(hw::Geometry::block_x(1));
    p1_index_1d_d3_bx32 => prop::p1_index_1d::<Domain3>(hw::Geometry::block_x(32));
    p1_index_1d_d3_bx48 => prop::p1_index_1d::<Domain3>(hw::Geometry::block_x(48));
    p1_index_1d_d3_bx256 => prop::p1_index_1d::<Domain3>(hw::Geometry::block_x(256));
    p1_index_1d_d3_bx1024 => prop::p1_index_1d::<Domain3>(hw::Geometry::block_x(1024));

    p1_warp_index_bx32 => prop::p1_warp_index(hw::Geometry::block_x(32));
    p1_warp_index_bx48 => prop::p1_warp_index(hw::Geometry::block_x(48));
    p1_warp_index_bx1000 => prop::p1_warp_index(hw::Geometry::block_x(1000));
    p1_warp_index_bx1024 => prop::p1_warp_index(hw::Geometry::block_x(1024));

    p1_index_2d_s1_d2_b1x1 => prop::p1_index_2d::<Domain2, 1>([1, 1]);
    p1_index_2d_s1_d2_b32x8 => prop::p1_index_2d::<Domain2, 1>([32, 8]);
    p1_index_2d_s1_d2_b16x16 => prop::p1_index_2d::<Domain2, 1>([16, 16]);
    p1_index_2d_s1_d3_b1x1 => prop::p1_index_2d::<Domain3, 1>([1, 1]);
    p1_index_2d_s1_d3_b32x8 => prop::p1_index_2d::<Domain3, 1>([32, 8]);
    p1_index_2d_s1_d3_b16x16 => prop::p1_index_2d::<Domain3, 1>([16, 16]);
    p1_index_2d_s5_d2_b1x1 => prop::p1_index_2d::<Domain2, 5>([1, 1]);
    p1_index_2d_s5_d2_b32x8 => prop::p1_index_2d::<Domain2, 5>([32, 8]);
    p1_index_2d_s5_d2_b16x16 => prop::p1_index_2d::<Domain2, 5>([16, 16]);
    p1_index_2d_s5_d3_b1x1 => prop::p1_index_2d::<Domain3, 5>([1, 1]);
    p1_index_2d_s5_d3_b32x8 => prop::p1_index_2d::<Domain3, 5>([32, 8]);
    p1_index_2d_s5_d3_b16x16 => prop::p1_index_2d::<Domain3, 5>([16, 16]);
    p1_index_2d_s5_unknown_b32x8 => prop::p1_index_2d::<UnknownDomain, 5>([32, 8]);
    p1_index_2d_s32_d2_b1x1 => prop::p1_index_2d::<Domain2, 32>([1, 1]);
    p1_index_2d_s32_d2_b32x8 => prop::p1_index_2d::<Domain2, 32>([32, 8]);
    p1_index_2d_s32_d2_b16x16 => prop::p1_index_2d::<Domain2, 32>([16, 16]);
    p1_index_2d_s32_d3_b1x1 => prop::p1_index_2d::<Domain3, 32>([1, 1]);
    p1_index_2d_s32_d3_b32x8 => prop::p1_index_2d::<Domain3, 32>([32, 8]);
    p1_index_2d_s32_d3_b16x16 => prop::p1_index_2d::<Domain3, 32>([16, 16]);
    p1_index_2d_s1024_d2_b1x1 => prop::p1_index_2d::<Domain2, 1024>([1, 1]);
    p1_index_2d_s1024_d2_b32x8 => prop::p1_index_2d::<Domain2, 1024>([32, 8]);
    p1_index_2d_s1024_d2_b16x16 => prop::p1_index_2d::<Domain2, 1024>([16, 16]);
    p1_index_2d_s1024_d3_b1x1 => prop::p1_index_2d::<Domain3, 1024>([1, 1]);
    p1_index_2d_s1024_d3_b32x8 => prop::p1_index_2d::<Domain3, 1024>([32, 8]);
    p1_index_2d_s1024_d3_b16x16 => prop::p1_index_2d::<Domain3, 1024>([16, 16]);
    p1_index_2d_s2p31_d2_b1x1 => prop::p1_index_2d::<Domain2, { 1 << 31 }>([1, 1]);
    p1_index_2d_s2p31_d2_b32x8 => prop::p1_index_2d::<Domain2, { 1 << 31 }>([32, 8]);
    p1_index_2d_s2p31_d2_b16x16 => prop::p1_index_2d::<Domain2, { 1 << 31 }>([16, 16]);
    p1_index_2d_s2p31_d3_b1x1 => prop::p1_index_2d::<Domain3, { 1 << 31 }>([1, 1]);
    p1_index_2d_s2p31_d3_b32x8 => prop::p1_index_2d::<Domain3, { 1 << 31 }>([32, 8]);
    p1_index_2d_s2p31_d3_b16x16 => prop::p1_index_2d::<Domain3, { 1 << 31 }>([16, 16]);
    p1_index_2d_s2p63_d2_b1x1 => prop::p1_index_2d::<Domain2, { 1 << 63 }>([1, 1]);
    p1_index_2d_s2p63_d2_b32x8 => prop::p1_index_2d::<Domain2, { 1 << 63 }>([32, 8]);
    p1_index_2d_s2p63_d2_b16x16 => prop::p1_index_2d::<Domain2, { 1 << 63 }>([16, 16]);
    p1_index_2d_s2p63_d3_b1x1 => prop::p1_index_2d::<Domain3, { 1 << 63 }>([1, 1]);
    p1_index_2d_s2p63_d3_b32x8 => prop::p1_index_2d::<Domain3, { 1 << 63 }>([32, 8]);
    p1_index_2d_s2p63_d3_b16x16 => prop::p1_index_2d::<Domain3, { 1 << 63 }>([16, 16]);
    p1_index_2d_smax_d2_b1x1 => prop::p1_index_2d::<Domain2, { usize::MAX }>([1, 1]);
    p1_index_2d_smax_d2_b32x8 => prop::p1_index_2d::<Domain2, { usize::MAX }>([32, 8]);
    p1_index_2d_smax_d2_b16x16 => prop::p1_index_2d::<Domain2, { usize::MAX }>([16, 16]);
    p1_index_2d_smax_d3_b1x1 => prop::p1_index_2d::<Domain3, { usize::MAX }>([1, 1]);
    p1_index_2d_smax_d3_b32x8 => prop::p1_index_2d::<Domain3, { usize::MAX }>([32, 8]);
    p1_index_2d_smax_d3_b16x16 => prop::p1_index_2d::<Domain3, { usize::MAX }>([16, 16]);
    p1_index_2d_smax_unknown_b32x8 => prop::p1_index_2d::<UnknownDomain, { usize::MAX }>([32, 8]);
    p1_index_2d_runtime_witnesses_distinct_b1x1 => prop::p1_index_2d_runtime_witnesses_distinct([1, 1]);
    p1_index_2d_runtime_witnesses_distinct_b32x8 => prop::p1_index_2d_runtime_witnesses_distinct([32, 8]);
    p1_index_2d_runtime_witnesses_distinct_b16x16 => prop::p1_index_2d_runtime_witnesses_distinct([16, 16]);

    p2_index_1d_sentinel_unknown => prop::p2_index_1d_sentinel::<UnknownDomain>(hw::Geometry::any());
    p2_index_1d_sentinel_d1 => prop::p2_index_1d_sentinel::<Domain1>(hw::Geometry::any());
    p2_index_1d_sentinel_d2 => prop::p2_index_1d_sentinel::<Domain2>(hw::Geometry::any());
    p2_index_1d_sentinel_d3 => prop::p2_index_1d_sentinel::<Domain3>(hw::Geometry::any());
    p2_index_1d_unchecked_coords => prop::p2_index_1d_unchecked_coords();
    p2_warp_index_no_wrap => prop::p2_warp_index_no_wrap(hw::Geometry::any());
    p2_index_2d_s1_no_wrap => prop::p2_index_2d_no_wrap::<1>(hw::Geometry::any());
    p2_index_2d_s5_no_wrap => prop::p2_index_2d_no_wrap::<5>(hw::Geometry::any());
    p2_index_2d_s32_no_wrap => prop::p2_index_2d_no_wrap::<32>(hw::Geometry::any());
    p2_index_2d_s1024_no_wrap => prop::p2_index_2d_no_wrap::<1024>(hw::Geometry::any());
    p2_index_2d_s2p31_no_wrap => prop::p2_index_2d_no_wrap::<{ 1 << 31 }>(hw::Geometry::any());
    p2_index_2d_s2p63_no_wrap => prop::p2_index_2d_no_wrap::<{ 1 << 63 }>(hw::Geometry::any());
    p2_index_2d_smax_no_wrap => prop::p2_index_2d_no_wrap::<{ usize::MAX }>(hw::Geometry::any());
    p2_index_2d_runtime_coords => prop::p2_index_2d_runtime_coords(hw::Geometry::any());
    p2_get_mut_flat => prop::p2_get_mut_flat(hw::Geometry::block_x(256));
    p2_get_mut_indexed_flat => prop::p2_get_mut_indexed_flat(hw::Geometry::block_x(256));
    p2_get_mut_indexed_2d_s5_b32x8 => prop::p2_get_mut_indexed_2d::<5>([32, 8]);
    p2_get_mut_indexed_2d_s1024_b16x16 => prop::p2_get_mut_indexed_2d::<1024>([16, 16]);
    p2_get_mut_runtime_2d_b1x1 => prop::p2_get_mut_runtime_2d(100, [1, 1]);
    p2_get_mut_runtime_2d_b32x8 => prop::p2_get_mut_runtime_2d(100, [32, 8]);
    p2_zst_resolves_nothing => prop::p2_zst_resolves_nothing(hw::Geometry::block_x(32));
    p2_zero_width_resolves_nothing => prop::p2_zero_width_resolves_nothing(hw::Geometry::any());
    p2_index_1d_u32_matches_math_bx1 => prop::p2_index_1d_u32_matches_math(hw::Geometry::block_x(1));
    p2_index_1d_u32_matches_math_bx32 => prop::p2_index_1d_u32_matches_math(hw::Geometry::block_x(32));
    p2_index_1d_u32_matches_math_bx48 => prop::p2_index_1d_u32_matches_math(hw::Geometry::block_x(48));
    p2_index_1d_u32_matches_math_bx256 => prop::p2_index_1d_u32_matches_math(hw::Geometry::block_x(256));
    p2_index_1d_u32_matches_math_bx1024 => prop::p2_index_1d_u32_matches_math(hw::Geometry::block_x(1024));
    p2_coord_2d_u32_matches_math_b1x1 => prop::p2_coord_2d_u32_matches_math([1, 1]);
    p2_coord_2d_u32_matches_math_b32x8 => prop::p2_coord_2d_u32_matches_math([32, 8]);
    p2_coord_2d_u32_matches_math_b16x16 => prop::p2_coord_2d_u32_matches_math([16, 16]);

    p3_resolve_w1 => prop::p3_resolve::<Domain3>(1);
    p3_resolve_w2 => prop::p3_resolve::<Domain3>(2);
    p3_resolve_w3 => prop::p3_resolve::<Domain3>(3);
    p3_resolve_w5 => prop::p3_resolve::<Domain3>(5);
    p3_resolve_w32 => prop::p3_resolve::<Domain3>(32);
    p3_resolve_w100 => prop::p3_resolve::<Domain3>(100);
    p3_resolve_w1024 => prop::p3_resolve::<Domain3>(1024);
    p3_resolve_w65535 => prop::p3_resolve_capped::<Domain3>(65535, 1 << 30, u32::MAX);
    p3_resolve_w2p31 => prop::p3_resolve::<Domain3>(1 << 31);
    p3_resolve_wmax => prop::p3_resolve::<Domain3>(u32::MAX);
    p3_resolve_symbolic_w_u8 => prop::p3_resolve_symbolic_w_u8();
    p3_cross_slice_no_alias_b1x1 => prop::p3_cross_slice_no_alias([1, 1]);
    p3_cross_slice_no_alias_b32x8 => prop::p3_cross_slice_no_alias([32, 8]);
    p3_cross_slice_no_alias_b16x16 => prop::p3_cross_slice_no_alias([16, 16]);
    p3_witness_width_independent => prop::p3_witness_width_independent(hw::Geometry::any());
}
