/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative copies from global memory to shared memory.
//!
//! CUDA groups its workers like this:
//!
//! ```text
//! thread = one worker
//! warp   = 32 threads that run together
//! CTA    = one block of threads that can share memory
//! ```
//!
//! [`copy_g2s`] lets every thread in a CTA copy one piece of a tile:
//!
//! ```text
//! global matrix ── all CTA threads ──► shared tile
//! ```
//!
//! Types hold facts fixed before the kernel starts, such as layouts and
//! alignment. Function arguments hold values known only while it runs, such
//! as pointers, sizes, the tile coordinate, and the thread number.

use core::marker::PhantomData;

use crate::markers::LeadingDimMarker;

/// Commit each participating thread's asynchronous copies and expose the tile.
///
/// This single-stage protocol waits for all committed copy groups before the
/// typed CTA barrier. It does not implement a multistage producer pipeline.
///
/// # Safety
/// Every thread participating in the preceding cooperative copies must call
/// this operation convergently, with exactly `THREADS` complete-warp
/// participants. Destination storage must remain live through the wait, and
/// `BARRIER` must satisfy `NamedBarrier::sync`'s phase and exclusivity rules.
#[inline(always)]
pub unsafe fn finish_async_copies<const BARRIER: u32, const THREADS: u32>() {
    // SAFETY: every participating thread retains its copy destinations and
    // executes the matching completion protocol under the caller contract.
    unsafe {
        ctm_rs::async_copy::cp_async_commit_group();
        ctm_rs::async_copy::cp_async_wait_group::<0>();
        crate::arch::NamedBarrier::<BARRIER, THREADS>::sync();
    }
}

/// A row-major matrix stored in global GPU memory.
///
/// `leading_dim.elements` is the row pitch: the number of elements from the
/// start of one row to the start of the next. It can be larger than `cols`
/// when rows contain padding.
///
/// ```text
/// row 0: [ data data data | padding ]
/// row 1: [ data data data | padding ]
///          <---- row pitch ---->
/// ```
///
/// `LD` records a compile-time divisibility promise about that pitch. Build
/// this plain data value with a struct literal. [`copy_g2s`] checks the static
/// layout contract; its caller must satisfy the documented pointer, capacity,
/// alignment, and pitch requirements.
#[repr(C)]
pub struct GmemMatrix<T, LD> {
    /// Start of the global-memory allocation.
    pub base: *const T,
    /// Logical matrix rows.
    pub rows: usize,
    /// Logical matrix columns, excluding row padding.
    pub cols: usize,
    /// Row pitch plus its compile-time divisibility marker.
    pub leading_dim: LD,
}

/// A tile in memory shared by all threads in one CTA.
///
/// `Layout` has no runtime bytes. It tells the compiler how logical rows and
/// columns map to physical addresses:
///
/// ```text
/// base pointer + capacity + zero-byte Layout marker
///                         │
///                         └── visible at the copy call
/// ```
///
/// Build this plain data value with a struct literal.
#[repr(C)]
pub struct SmemTile<T, Layout> {
    /// Start of the shared-memory allocation.
    pub base: *mut T,
    /// Number of `T` elements available from `base`.
    pub capacity: usize,
    /// Zero-byte marker that keeps the shared layout in the type.
    pub layout: PhantomData<Layout>,
}

impl<T, Layout> Copy for SmemTile<T, Layout> {}
impl<T, Layout> Clone for SmemTile<T, Layout> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// A borrowed fixed-size logical row in a typed shared-memory tile.
///
/// This view contains coordinates and a borrow, not loaded values. Reductions
/// can consume one element at a time without retaining a full register tile.
pub struct SharedRow<'a, T: Copy, Layout, const N: usize> {
    tile: &'a SmemTile<T, Layout>,
    row: usize,
    column: usize,
}

impl<T: Copy, Layout: crate::markers::ReifySmem2D, const N: usize> SharedRow<'_, T, Layout, N> {
    /// Reduce corresponding shared/global logical coordinates in increasing order.
    ///
    /// Each pair is loaded immediately before its callback. The operation does
    /// not create intermediate register tiles or reassociate the accumulation.
    /// The other tensor may use a different physical layout and must contain
    /// at least `N` logical elements. Empty rows never load or call `f`.
    #[inline(always)]
    pub fn zip_fold<Element, OtherLayout, Storage, Accumulator>(
        self,
        other: crate::tensor::partition::StridedTensor<'_, Element, OtherLayout, Storage>,
        initial: Accumulator,
        f: impl FnMut(Accumulator, T, Storage) -> Accumulator,
    ) -> Accumulator
    where
        Element: crate::tensor::TensorElement<Storage>,
        OtherLayout: crate::tensor::partition::IndexLayout,
        Storage: Copy,
    {
        assert!(N <= other.len());
        unsafe { self.zip_fold_unchecked(other, initial, f) }
    }

    /// Reduce corresponding shared/global logical coordinates after an enclosing extent check.
    ///
    /// # Safety
    /// `other` must contain at least `N` logical elements, and this shared row
    /// must satisfy `row_view`'s bounds, initialization, and synchronization contract.
    #[inline(always)]
    pub unsafe fn zip_fold_unchecked<Element, OtherLayout, Storage, Accumulator>(
        self,
        other: crate::tensor::partition::StridedTensor<'_, Element, OtherLayout, Storage>,
        initial: Accumulator,
        mut f: impl FnMut(Accumulator, T, Storage) -> Accumulator,
    ) -> Accumulator
    where
        Element: crate::tensor::TensorElement<Storage>,
        OtherLayout: crate::tensor::partition::IndexLayout,
        Storage: Copy,
    {
        let mut accumulator = initial;
        let mut i = 0;
        while i < N {
            ctm_rs::thread::__unroll_config::<0>();
            let (global, shared) = unsafe {
                (
                    other.get_unchecked(i),
                    self.tile.get(self.row, self.column + i),
                )
            };
            accumulator = f(accumulator, shared, global);
            i += 1;
        }
        accumulator
    }
}

impl<T: Copy, Layout> SmemTile<T, Layout> {
    /// Wrap a shared-memory allocation as a typed tile with `capacity` elements.
    ///
    /// # Safety
    /// `base` must point to live, suitably aligned shared-memory storage for
    /// at least `capacity` elements of `T`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_parts(base: *mut T, capacity: usize) -> Self {
        Self {
            base,
            capacity,
            layout: PhantomData,
        }
    }

    /// Return the adjacent shared-memory tile of the same type and capacity immediately after `self`.
    ///
    /// # Safety
    /// The underlying shared allocation must contain at least `self.capacity` additional
    /// suitably aligned `T` elements immediately following `self`.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn next_tile(self) -> Self {
        Self {
            base: unsafe { self.base.add(self.capacity) },
            capacity: self.capacity,
            layout: PhantomData,
        }
    }

    /// View `len` contiguous elements of `U` starting `offset_elements` of `U` after `self`.
    ///
    /// Returns both a read-only [`crate::tensor::partition::StridedTensor`] and a
    /// writable [`crate::tensor::partition::StridedTensorMut`] over the shared region;
    /// callers must synchronize between reads and writes with a CTA barrier.
    ///
    /// # Safety
    /// The region starting at `(self.base.add(self.capacity) as *mut U).add(offset_elements)`
    /// must be aligned for `U` and contain at least `len` live shared elements.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn trailing_views<'a, U: Copy + crate::tensor::TensorElement<U>>(
        self,
        offset_elements: usize,
        len: usize,
    ) -> (
        crate::tensor::partition::StridedTensor<'a, U>,
        crate::tensor::partition::StridedTensorMut<'a, U>,
    ) {
        let ptr = unsafe {
            self.base
                .add(self.capacity)
                .cast::<U>()
                .add(offset_elements)
        };
        unsafe {
            (
                crate::tensor::partition::StridedTensor::from_raw_parts(ptr, len, 1),
                crate::tensor::partition::StridedTensorMut::from_raw_parts(ptr, len, 1),
            )
        }
    }
}

impl<T: Copy, Layout: crate::markers::ReifySmem2D> SmemTile<T, Layout> {
    /// Partition this shared tile along the row mode and select subtile `tile_row`.
    ///
    /// `SubLayout` must have the same column count, row/column strides, and
    /// swizzle parameters as `Layout`, with a positive row extent that divides
    /// the parent tile into contiguous row groups.
    ///
    /// # Safety
    /// `tile_row` must select a complete subtile within `self.capacity`, and
    /// the underlying shared allocation must remain live while the view is used.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn row_tile<SubLayout: crate::markers::ReifySmem2D>(
        &self,
        tile_row: usize,
    ) -> SmemTile<T, SubLayout> {
        const {
            assert!(SubLayout::SHAPE.0 > 0 && SubLayout::SHAPE.1 == Layout::SHAPE.1);
            assert!(
                SubLayout::STRIDE.0 == Layout::STRIDE.0 && SubLayout::STRIDE.1 == Layout::STRIDE.1
            );
            assert!(SubLayout::OFFSET == 0 && Layout::OFFSET == 0);
            assert!(
                SubLayout::NUM_BITS == Layout::NUM_BITS
                    && SubLayout::NUM_BASE == Layout::NUM_BASE
                    && SubLayout::NUM_SHIFT == Layout::NUM_SHIFT
            );
        }
        let sub_elements = (SubLayout::SHAPE.0 as usize) * (Layout::STRIDE.0 as usize);
        SmemTile {
            base: unsafe { self.base.add(tile_row * sub_elements) },
            capacity: sub_elements,
            layout: PhantomData,
        }
    }

    /// Partition this shared tile along the row mode and select subtile `tile_row`
    /// (CuTeDSL `cute.local_tile` along the row mode).
    ///
    /// # Safety
    /// Same bounds and lifetime requirements as [`Self::row_tile`].
    #[must_use]
    #[inline(always)]
    pub const unsafe fn local_tile<SubLayout: crate::markers::ReifySmem2D>(
        &self,
        tile_row: usize,
    ) -> SmemTile<T, SubLayout> {
        unsafe { self.row_tile::<SubLayout>(tile_row) }
    }

    /// Borrow `N` consecutive logical row coordinates without loading them.
    ///
    /// # Safety
    /// For the entire returned view's lifetime, every `(row, column + i)`,
    /// `i < N`, must be in the layout domain, map below `capacity`, and refer
    /// to live, aligned, initialized readable storage. All asynchronous writes
    /// must have completed and the necessary synchronization must have occurred.
    /// No write may conflict with the borrowed row during that lifetime. Empty
    /// rows contain no coordinates and do not require an accessible pointer.
    #[inline(always)]
    pub unsafe fn row_view<const N: usize>(
        &self,
        row: usize,
        column: usize,
    ) -> SharedRow<'_, T, Layout, N> {
        SharedRow {
            tile: self,
            row,
            column,
        }
    }

    /// Read one logical shared-memory coordinate through its declared layout.
    ///
    /// This is the single-coordinate read `tensor[(row, column)]` of CuTe's
    /// `Tensor.__getitem__`.
    ///
    /// # Safety
    /// The coordinate must lie within the layout, its mapped offset must be
    /// below `capacity`, and the element must be initialized and readable.
    /// Pending asynchronous writes must be completed and appropriate thread
    /// synchronization performed; no active write may conflict with this read.
    #[inline(always)]
    pub unsafe fn get(self, row: usize, column: usize) -> T {
        let (stride_r, stride_c, offset, y_mask, right_shift, left_shift) = const {
            let _ = crate::static_layout::max_offset_2d::<Layout>();
            let mask = ((1usize << Layout::NUM_BITS) - 1)
                << (Layout::NUM_BASE as i32
                    + if Layout::NUM_SHIFT > 0 {
                        Layout::NUM_SHIFT
                    } else {
                        0
                    }) as u32;
            let r_shift = if Layout::NUM_SHIFT >= 0 {
                Layout::NUM_SHIFT as u32
            } else {
                0
            };
            let l_shift = if Layout::NUM_SHIFT < 0 {
                (-Layout::NUM_SHIFT) as u32
            } else {
                0
            };
            (
                Layout::STRIDE.0 as usize,
                Layout::STRIDE.1 as usize,
                Layout::OFFSET as usize,
                mask,
                r_shift,
                l_shift,
            )
        };
        let plain = row * stride_r + column * stride_c + offset;
        let idx = plain ^ (((plain & y_mask) >> right_shift) << left_shift);
        unsafe { self.base.add(idx).read() }
    }

    /// Gather N consecutive logical row coordinates into a register tile.
    ///
    /// Consecutive logical coordinates may be physically swizzled; the layout
    /// determines each address rather than assuming contiguous physical bytes.
    /// This is CuTe's `Tensor.load` over the `N` coordinates of one row.
    ///
    /// # Safety
    /// Every `(row, column + i)` for `i < N` must satisfy `get`'s bounds,
    /// initialization, completion and synchronization requirements.
    #[inline(always)]
    pub unsafe fn load<const N: usize>(
        &self,
        row: usize,
        column: usize,
    ) -> crate::tensor::RegisterTile<T, N> {
        crate::tensor::RegisterTile::from_fn(|i| unsafe { self.get(row, column + i) })
    }

    /// Write one logical shared-memory coordinate through its declared layout.
    ///
    /// This is the single-coordinate store `tensor[(row, column)] = value` of
    /// CuTe's `Tensor.__setitem__`.
    ///
    /// # Safety
    /// The coordinate must lie within the layout, its mapped offset must be
    /// below `capacity`, and no conflicting read or write may race this store.
    #[inline(always)]
    pub unsafe fn store_at(&self, row: usize, column: usize, value: T) {
        let offset = crate::static_layout::crd2idx_2d::<Layout>(row, column);
        unsafe { self.base.add(offset).write(value) };
    }

    /// Scatter `N` consecutive logical row coordinates from a register tile.
    ///
    /// Consecutive logical coordinates are mapped through `Layout`, including
    /// any shared-memory swizzle. This is CuTe's `Tensor.store` over one row.
    ///
    /// # Safety
    /// Every `(row, column + i)` for `i < N` must satisfy `store_at`'s bounds
    /// and non-conflict requirements.
    #[inline(always)]
    pub unsafe fn store<const N: usize>(
        &self,
        row: usize,
        column: usize,
        values: crate::tensor::RegisterTile<T, N>,
    ) {
        let arr = values.into_array();
        let mut i = 0;
        while i < N {
            ctm_rs::thread::__unroll_config::<0>();
            unsafe { self.store_at(row, column + i, arr[i]) };
            i += 1;
        }
    }
}

impl<Layout: crate::markers::ReifySmem2D> SmemTile<f16, Layout> {
    /// Convert two FP32 values to `f16x2` and store them at an aligned pair `(row, column)`.
    ///
    /// Requires unit column stride and a swizzle base of at least two elements so
    /// even-aligned column pairs map to contiguous 4-byte shared words.
    ///
    /// # Safety
    /// `column` must be even, `(row, column)` and `(row, column + 1)` must lie
    /// within the layout and `capacity`, `self.base` must be 4-byte aligned,
    /// and no conflicting access may race this store.
    #[inline(always)]
    pub unsafe fn store_packed_f16x2(&self, row: usize, column: usize, low: f32, high: f32) {
        const {
            assert!(Layout::STRIDE.1 == 1);
            assert!(Layout::NUM_BITS == 0 || Layout::NUM_BASE >= 1);
        }
        let offset = crate::static_layout::crd2idx_2d::<Layout>(row, column);
        unsafe {
            self.base
                .add(offset)
                .cast::<u32>()
                .write(crate::arch::cvt_f16x2_f32(low, high));
        }
    }
}

/// Copy one global-memory tile into shared memory using the whole CTA.
///
/// `tile_coord = (r, c)` counts tiles, not elements. For a 16x8 tile,
/// `(2, 3)` starts at element `(32, 24)`.
///
/// The compiler retains this call as one typed cooperative-copy operation.
/// The selected backend chooses its realization. On a matrix edge, every
/// thread still follows the same path; a copy reports how many source bytes
/// are valid, and missing bytes become zero. The caller must still complete
/// the copy protocol and synchronize the CTA.
///
/// # Safety
///
/// The CTA must contain exactly the threads described by `ThrL`. Every one of
/// those threads must call this function on the same control-flow path, and
/// their `tidx` values must cover `ThrL` exactly once. Both raw pointers must
/// be valid. Each `Atom`, meaning one fixed-size copy piece, must be aligned.
/// The destination must have the capacity required by `SmemL`.
///
/// `src.leading_dim.elements` must be at least `src.cols` and divisible by
/// the divisor carried by its `LeadingDim<D>` type. The compiler checks that
/// the thread/value layout covers the tile exactly, that each atom maps to a
/// valid destination, and that `D * size_of::<T>()` is at least one atom.
/// Edge tiles are safe because out-of-bounds source bytes are replaced by
/// zero instead of being read.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn copy_g2s<Atom, ThrL, ValL, TileL, SmemL, T: Copy>(
    src: &GmemMatrix<T, impl LeadingDimMarker>,
    tile_coord: (usize, usize),
    dst_smem_tile: &mut SmemTile<T, SmemL>,
    tidx: u32,
) {
    let _ = (
        src,
        tile_coord,
        dst_smem_tile,
        tidx,
        PhantomData::<(Atom, ThrL, ValL, TileL)>,
    );
    unreachable!("cute-rs `copy_g2s` executed outside device compilation")
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Layout markers must add no bytes to global and shared copy carriers.
    #[test]
    fn runtime_carriers_are_only_the_visible_abi_fields() {
        // The layout marker occupies no bytes:
        //
        // SmemTile = [ pointer ][ capacity ][ zero-byte static marker ]
        assert_eq!(
            core::mem::size_of::<GmemMatrix<u32, crate::markers::LeadingDim<4>>>(),
            4 * core::mem::size_of::<usize>()
        );
        assert_eq!(
            core::mem::size_of::<SmemTile<u32, ()>>(),
            2 * core::mem::size_of::<usize>()
        );

        let tile = SmemTile::<u32, ()> {
            base: core::ptr::null_mut(),
            capacity: 24,
            layout: PhantomData,
        };
        assert!(tile.base.is_null());
        assert_eq!(tile.capacity, 24);
    }
}
