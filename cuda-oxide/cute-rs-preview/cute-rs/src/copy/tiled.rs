/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed, block-wide copies from global memory to shared memory.
//!
//! Every thread in a block helps move part of one tile:
//!
//! ```text
//! global tile: [thread 0 part][thread 1 part][...]
//!                            │
//!       TiledCopy::get_slice(thread_idx) ──► ThrCopy::copy
//!                            │
//!                            ▼
//! shared tile: [thread 0 part][thread 1 part][...]
//! ```
//!
//! [`GlobalCopyTensor`] describes the source. [`SharedTensor`] describes the
//! destination. [`TiledCopy`] describes how the work is split across threads,
//! and [`TiledCopy::get_slice`] selects one thread's part of that plan as a
//! [`ThrCopy`], following CuTeDSL's `TiledCopy.get_slice` and `ThrCopy`.
//! These wrappers add no runtime storage beyond the source and destination
//! views and the thread index. The compiler inlines them down to
//! [`crate::copy_g2s`].

use core::marker::PhantomData;

use crate::copy::cooperative::{GmemMatrix, SmemTile, copy_g2s};
use crate::markers::{C, CopyUniversal, Layout, LeadingDimMarker, ReifySmem2D, RowMajor};
use crate::tensor::TensorElement;
use ctm_rs::memory::DisjointSlice;

/// A typed global-memory source for a block-wide copy.
///
/// `Element` says what the bits mean. `T` says how those bits are physically
/// stored and copied. For example:
///
/// ```text
/// Element = Mxf4E2M1   four logical FP4 values
/// T       = f16        one 16-bit storage box
/// ```
///
/// `Role` can mark the source as A or B so typed operations cannot swap them.
pub struct GlobalCopyTensor<Element: TensorElement<T>, T: Copy, LD, Role = ()> {
    /// Global pointer, matrix bounds and row-pitch promise used by cooperative lowering.
    carrier: GmemMatrix<T, LD>,
    /// Logical element encoding and A/B role attached to the global storage.
    element: PhantomData<(Element, Role)>,
}

impl<Element: TensorElement<T>, T: Copy, LD, Role> GlobalCopyTensor<Element, T, LD, Role> {
    /// Attach an element meaning and role to a global-memory matrix view.
    #[must_use]
    #[inline(always)]
    pub const fn from_matrix(carrier: GmemMatrix<T, LD>) -> Self {
        Self {
            carrier,
            element: PhantomData,
        }
    }

    /// Attach an element meaning and role to a global-memory matrix view.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn from_carrier(carrier: GmemMatrix<T, LD>) -> Self {
        Self::from_matrix(carrier)
    }

    /// Return the underlying matrix view used by the compiler copy operation.
    #[inline(always)]
    pub(crate) const fn carrier(&self) -> &GmemMatrix<T, LD> {
        &self.carrier
    }
}

/// A typed view of an existing shared-memory tile.
///
/// `Element` and `T` have the same meaning as in [`GlobalCopyTensor`]. `Layout`
/// maps tile coordinates to shared-memory offsets. `Role` can distinguish A
/// storage from B storage.
///
/// This view does not allocate shared memory.
pub struct SharedTensor<Element: TensorElement<T>, T: Copy, Layout, Role = ()> {
    /// Shared allocation base, capacity and physical layout marker.
    carrier: SmemTile<T, Layout>,
    /// Logical element encoding and A/B role attached to the shared storage.
    element: PhantomData<(Element, Role)>,
}

impl<Element: TensorElement<T>, T: Copy, Layout, Role> SharedTensor<Element, T, Layout, Role> {
    /// View an existing shared-memory allocation as a typed tile.
    ///
    /// # Safety
    ///
    /// - `base` must point to shared memory and be aligned for `T`.
    /// - At least `capacity` elements must remain live while the view is used.
    /// - Every offset produced by `Layout` must be below `capacity`.
    /// - Threads must synchronize before reading data written by other
    ///   threads or by an asynchronous copy.
    /// - Reads and writes must not break Rust's aliasing rules.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_parts(base: *mut T, capacity: usize) -> Self {
        unsafe { __compiler::shared_tensor_overlay(base, capacity) }
    }

    /// Return the read-only shared-memory view used by compiler operations.
    #[inline(always)]
    pub(crate) const fn carrier(&self) -> &SmemTile<T, Layout> {
        &self.carrier
    }

    /// Return the writable shared-memory view used by compiler operations.
    #[inline(always)]
    const fn carrier_mut(&mut self) -> &mut SmemTile<T, Layout> {
        &mut self.carrier
    }
}

/// Stable compiler boundary for attaching tensor meaning to shared storage.
///
/// Kernel code uses [`SharedTensor::from_raw_parts`]. The importer recognizes
/// this exact helper when rustc inlines that small public wrapper.
#[doc(hidden)]
pub mod __compiler {
    use super::*;

    /// Compiler boundary attaching element and operand-role meaning to a shared-memory carrier.
    #[inline(never)]
    pub const unsafe fn shared_tensor_overlay<Element: TensorElement<T>, T: Copy, Layout, Role>(
        base: *mut T,
        capacity: usize,
    ) -> SharedTensor<Element, T, Layout, Role> {
        SharedTensor {
            carrier: SmemTile {
                base,
                capacity,
                layout: PhantomData,
            },
            element: PhantomData,
        }
    }
}

/// A complete plan for how one block copies one tile.
///
/// The type parameters describe the plan:
///
/// ```text
/// Atom          smallest hardware copy
/// ThreadLayout  which thread copies which part
/// ValueLayout   which values that thread copies
/// TileLayout    shape of the complete source tile
/// ```
///
/// The value contains no runtime data. [`Self::get_slice`] selects one
/// thread's part of the plan; the destination layout comes from the
/// [`SharedTensor`] passed to [`ThrCopy::copy`].
pub struct TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout> {
    /// Records the atom and thread/value/tile maps without runtime storage.
    config: PhantomData<(Atom, ThreadLayout, ValueLayout, TileLayout)>,
}

impl<Atom, ThreadLayout, ValueLayout, TileLayout>
    TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout>
{
    /// Create the zero-sized copy plan.
    #[must_use]
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            config: PhantomData,
        }
    }

    /// Select the part of the plan one thread carries out.
    ///
    /// This is CuTeDSL's `TiledCopy.get_slice(thr_idx)`. Device use must pass
    /// the calling thread's own index in the block, as `ThreadLayout` numbers
    /// the threads; [`ThrCopy::copy`] then issues that thread's part.
    #[must_use]
    #[inline(always)]
    pub const fn get_slice(
        &self,
        thread_idx: u32,
    ) -> ThrCopy<Atom, ThreadLayout, ValueLayout, TileLayout> {
        ThrCopy {
            thread_idx,
            plan: PhantomData,
        }
    }
}

impl<Atom, ThreadLayout, ValueLayout, TileLayout> Default
    for TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout>
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// One thread's slice of a [`TiledCopy`] plan.
///
/// This is CuTeDSL's `ThrCopy`, returned by [`TiledCopy::get_slice`]. The
/// plan has no runtime data, so the slice holds only the thread index and
/// the plan's type parameters:
///
/// ```text
/// TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout>
///                   │ get_slice(thread_idx)
///                   ▼
/// ThrCopy { thread_idx }   one per thread, copies its own part
/// ```
pub struct ThrCopy<Atom, ThreadLayout, ValueLayout, TileLayout> {
    /// Index of the thread this slice belongs to, as `ThreadLayout` numbers the threads.
    thread_idx: u32,
    /// Records the atom and thread/value/tile maps without runtime storage.
    plan: PhantomData<(Atom, ThreadLayout, ValueLayout, TileLayout)>,
}

impl<Atom, ThreadLayout, ValueLayout, TileLayout>
    ThrCopy<Atom, ThreadLayout, ValueLayout, TileLayout>
{
    /// Copy this thread's part of one global tile into shared memory.
    ///
    /// `tile_coord` selects the source tile. The thread index given to
    /// [`TiledCopy::get_slice`] selects this thread's assigned part of that
    /// tile. In CuTeDSL this is `cute.copy(tiled_copy,
    /// thr_copy.partition_S(src), thr_copy.partition_D(dst))`; the Rust form
    /// covers one two-dimensional global tile and one shared destination
    /// tile, with both partitions derived from the plan's layouts inside the
    /// compiler's cooperative-copy operation, [`crate::copy_g2s`].
    ///
    /// # Safety
    ///
    /// - Every thread named by `ThreadLayout` must call this operation.
    /// - The launched thread count must match `ThreadLayout`.
    /// - Source and destination pointers, sizes, layouts, row pitches, and
    ///   copy alignment must all match their typed descriptions.
    /// - Threads must wait for the copy before reading the destination.
    #[inline(always)]
    pub unsafe fn copy<
        Element: TensorElement<T>,
        T: Copy,
        LD: LeadingDimMarker,
        SharedLayout,
        Role,
    >(
        &self,
        source: &GlobalCopyTensor<Element, T, LD, Role>,
        tile_coord: (usize, usize),
        destination: &mut SharedTensor<Element, T, SharedLayout, Role>,
    ) {
        unsafe {
            copy_g2s::<Atom, ThreadLayout, ValueLayout, TileLayout, SharedLayout, T>(
                source.carrier(),
                tile_coord,
                destination.carrier_mut(),
                self.thread_idx,
            )
        }
    }

    /// Copy this thread's part of one global matrix tile into a shared tile.
    ///
    /// Forwards directly to [`crate::copy_g2s`] using the plan's `Atom`,
    /// `ThreadLayout`, `ValueLayout`, and `TileLayout` type arguments.
    ///
    /// # Safety
    /// The caller must satisfy all participation, alignment, pitch, and bounds
    /// requirements of [`crate::copy_g2s`].
    #[inline(always)]
    pub unsafe fn copy_g2s<T: Copy, LD: LeadingDimMarker, SharedLayout>(
        &self,
        source: &GmemMatrix<T, LD>,
        tile_coord: (usize, usize),
        destination: &mut SmemTile<T, SharedLayout>,
    ) {
        unsafe {
            copy_g2s::<Atom, ThreadLayout, ValueLayout, TileLayout, SharedLayout, T>(
                source,
                tile_coord,
                destination,
                self.thread_idx,
            )
        }
    }

    /// Convert row band `ROW` (`0` or `1`) of an `m16n8k16` [`crate::mma::FragmentC`] to `f16x2`
    /// and store it into `destination` at `(warp_m, tile_n)`.
    ///
    /// Mirrors CuTeDSL's `cute.copy(smem_copy_atom_O, taccOrO, taccOsO)` using
    /// the static [`crate::mma::bf16::CLayout`] and [`SmemTile::store_packed_f16x2`].
    ///
    /// # Safety
    /// `self.thread_idx % 32` must be this thread's hardware lane, `(warp_m * 16 + row, tile_n * 8 + col)`
    /// must lie inside `destination`, and no conflicting shared-memory access may race this store.
    #[inline(always)]
    pub unsafe fn copy_r2s<const ROW: usize, SharedLayout: ReifySmem2D>(
        &self,
        source: &crate::mma::FragmentC,
        destination: &SmemTile<f16, SharedLayout>,
        warp_m: usize,
        tile_n: usize,
    ) {
        const { assert!(ROW < 2) };
        let lane = self.thread_idx % 32;
        let row = warp_m * 16 + crate::mma::FragmentC::row_coord::<ROW>(lane) as usize;
        let col = crate::mma::FragmentC::col_coord::<0>(lane, tile_n) as usize;
        unsafe {
            destination.store_packed_f16x2(row, col, source.0[ROW * 2], source.0[ROW * 2 + 1]);
        }
    }
}

impl<const THR_R: i64, const THR_C: i64, const TILE_R: i64, const TILE_C: i64>
    ThrCopy<
        CopyUniversal<16>,
        Layout<(C<THR_R>, C<THR_C>), (C<THR_C>, C<1>)>,
        Layout<(C<1>, C<8>), (C<8>, C<1>)>,
        RowMajor<TILE_R, TILE_C>,
    >
{
    /// Copy this thread's 16-byte (8-element `f16`) slice from shared memory to global memory.
    ///
    /// Mirrors CuTeDSL's predicated `cute.copy(gmem_tiled_copy_O, tOsO, tOgO)`
    /// in FlashAttention-2 epilogues. Threads whose global row is at or beyond
    /// `total_rows` do not write.
    ///
    /// # Safety
    /// - `self.thread_idx` must be in `0 .. (THR_R * THR_C) as u32`.
    /// - `source` must be CTA-synchronized, 16-byte aligned, and valid for `SharedLayout`.
    /// - `destination` must be 16-byte aligned and contain at least `total_rows * row_stride`
    ///   elements, with `col_offset` and `row_stride` divisible by 8.
    /// - Participating threads must map to disjoint global output slices.
    #[inline(always)]
    pub unsafe fn copy_s2g<SharedLayout: ReifySmem2D>(
        &self,
        source: &SmemTile<f16, SharedLayout>,
        destination: &mut DisjointSlice<f16>,
        tile_row: usize,
        total_rows: usize,
        row_stride: usize,
        col_offset: usize,
    ) {
        const {
            assert!(THR_R > 0 && THR_C > 0);
            assert!(TILE_R == THR_R && TILE_C == THR_C * 8);
            assert!(SharedLayout::SHAPE.0 >= TILE_R && SharedLayout::SHAPE.1 == TILE_C);
            assert!(SharedLayout::STRIDE.1 == 1);
            assert!(SharedLayout::NUM_BITS == 0 || SharedLayout::NUM_BASE >= 3);
        }
        let tid = self.thread_idx as usize;
        let thr_cols = THR_C as usize;
        let local_row = tid / thr_cols;
        let local_col = (tid % thr_cols) * 8;
        let global_row = tile_row * (TILE_R as usize) + local_row;
        if global_row < total_rows {
            let smem_offset =
                crate::static_layout::crd2idx_2d::<SharedLayout>(local_row, local_col);
            let dst_offset = global_row * row_stride + col_offset + local_col;
            unsafe {
                let packed = source.base.add(smem_offset).cast::<u128>().read();
                destination
                    .as_mut_ptr()
                    .add(dst_offset)
                    .cast::<u128>()
                    .write(packed);
            }
        }
    }
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Copy plans must be zero-sized and thread slices must hold only the thread index,
    /// while typed source/destination views retain their carrier sizes.
    #[test]
    fn tiled_copy_is_static_and_typed_views_preserve_carrier_size() {
        /// Marker-only copy plan used to check zero runtime size.
        type Copy = TiledCopy<(), (), (), ()>;
        assert_eq!(core::mem::size_of::<Copy>(), 0);
        /// One thread's slice of the marker-only plan.
        type Slice = ThrCopy<(), (), (), ()>;
        assert_eq!(core::mem::size_of::<Slice>(), core::mem::size_of::<u32>());
        /// Distinct logical element marker used to test typed copy carrier size.
        #[derive(Clone, Copy)]
        struct Element;
        impl TensorElement<u32> for Element {}
        assert_eq!(
            core::mem::size_of::<GlobalCopyTensor<Element, u32, (), ()>>(),
            core::mem::size_of::<GmemMatrix<u32, ()>>()
        );
        assert_eq!(
            core::mem::size_of::<SharedTensor<Element, u32, (), ()>>(),
            core::mem::size_of::<SmemTile<u32, ()>>()
        );
    }

    /// The thread slice must carry the index it was selected with.
    #[test]
    fn get_slice_records_the_thread_index() {
        let plan = TiledCopy::<(), (), (), ()>::new();
        assert_eq!(plan.get_slice(0).thread_idx, 0);
        assert_eq!(plan.get_slice(127).thread_idx, 127);
    }
}
