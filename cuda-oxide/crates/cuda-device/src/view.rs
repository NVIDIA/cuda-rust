/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked-once views for 32-bit kernel indexing.
//!
//! A launch contract proves that thread coordinates fit in `u32`. These views
//! then check one complete element or tile before exposing check-free interior
//! accesses:
//!
//! ```text
//! thread 0 owns [0 .. N)
//! thread 1 owns [N .. 2N)
//! thread 2 owns [2N .. 3N)
//! ```
//!
//! The pointer fields are private. Safe code can obtain a view only through a
//! checked slice constructor or a `DisjointSlice` method that consumes a
//! thread-unique [`crate::thread::ThreadIndex32`].
//!
//! # Initialization
//!
//! The by-value readers ([`InBounds32::read`], [`InBoundsMut32::read`],
//! [`RowView32::get`], [`ColView32::get`], [`RowViewIter32::next`],
//! [`ColViewIter32::next`] and [`ZipView32::next`]) copy the element out of
//! the underlying buffer, so every slot they touch must already be
//! **initialized**. Reading a slot that was never written is undefined
//! behavior for any element type: Rust treats even integers and floats read
//! from uninitialized memory as invalid. Beyond being written at all, the
//! slot must hold a valid value of the type, which zeroed memory does not
//! guarantee for types with invalid bit patterns such as references or enums
//! without a zero variant.
//!
//! In practice: buffers from `DeviceBuffer::zeroed` or
//! `DeviceBuffer::from_host` arrive fully initialized, while one from
//! `DeviceBuffer::uninitialized_async` must not be read until a kernel or
//! copy has written it, exactly as that constructor's safety contract says.

use crate::DisjointSlice;
use crate::thread::{Index1D, ThreadCoord2D32, ThreadIndex32};
use core::marker::PhantomData;
use core::mem::size_of;

/// Index-space marker where thread `t` owns `N` consecutive elements.
///
/// ```text
/// tile_start = t * N
/// tile_end   = tile_start + N
/// ```
pub enum LinearTiles<const N: usize> {}

/// Index-space marker for a per-thread `ROWS × COLS` row-major tile.
///
/// Thread coordinate `(y, x)` owns rows `y * ROWS..(y + 1) * ROWS` and
/// columns `x * COLS..(x + 1) * COLS`.
///
/// `ROW_STRIDE` is the caller-declared logical row width: the number of
/// elements from the start of one row to the start of the next. It must match
/// the buffer's layout. It is encoded in the type, so two layouts with
/// different row widths cannot exchange tile proofs. The final row may be partial; each
/// requested tile is checked against the slice length.
pub enum RowMajorTiles<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> {}

/// Index-space marker for a per-thread `ROWS × COLS` tile in a matrix whose
/// row width is known only at runtime.
///
/// Thread coordinate `(y, x)` owns rows `y * ROWS..(y + 1) * ROWS` and columns
/// `x * COLS..(x + 1) * COLS`, exactly like [`RowMajorTiles`]. The difference:
/// [`RowMajorTiles`] bakes the matrix's row width into the type as a
/// compile-time constant, while here it is an ordinary runtime value (for
/// example a GEMM dimension `n`) passed to [`DisjointSlice::tile_2d32_rt`].
///
/// Because the row width is not in the type, it travels as data: the host
/// writes it into the launch packet beside the slice's pointer and length, and
/// [`DisjointSlice::row_width`] reads it back. One row width per slice for the
/// slice's whole lifetime is what tile disjointness needs, and a value the
/// host wrote is one device code cannot vary between threads, so
/// [`tile_2d32_rt`](DisjointSlice::tile_2d32_rt) is safe.
///
/// [`DisjointSlice::row_width`]: crate::disjoint::DisjointSlice::row_width
pub enum RuntimeRowMajorTiles<const ROWS: usize, const COLS: usize> {}

/// A checked local index into a static `N`-element view.
///
/// The private field prevents safe code from inventing an out-of-range value.
/// Use [`new`](Self::new) for a runtime index or [`constant`](Self::constant)
/// for an index that should fold at compile time.
#[must_use]
pub struct LocalIndex32<const N: usize> {
    raw: u32,
}

impl<const N: usize> LocalIndex32<N> {
    /// Check a runtime local index.
    #[inline(always)]
    pub const fn new(raw: u32) -> Option<Self> {
        if N != 0 && N <= u32::MAX as usize && (raw as usize) < N {
            Some(Self { raw })
        } else {
            None
        }
    }

    /// Construct a compile-time local index.
    ///
    /// An invalid constant fails compilation: `N` must be non-zero, `N` must
    /// fit in `u32`, and `I` must be less than `N`. A valid monomorphized call
    /// folds to the immediate index.
    #[inline(always)]
    pub const fn constant<const I: u32>() -> Self {
        const {
            assert!(N != 0, "a static view cannot have zero elements");
            assert!(N <= u32::MAX as usize, "a static view must fit in u32");
            assert!((I as usize) < N, "local index is outside the static view");
        }
        Self { raw: I }
    }

    /// Return the local index.
    #[inline(always)]
    pub const fn get(&self) -> u32 {
        self.raw
    }
}

/// A parent-bound proof that one immutable element is in bounds.
///
/// The proof stores the resolved pointer rather than a free-standing numeric
/// index, so it cannot be applied to an unrelated shorter slice.
#[must_use]
pub struct InBounds32<'a, T> {
    ptr: *const T,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T> InBounds32<'a, T> {
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
        }
    }

    /// Borrow the proven element.
    #[inline(always)]
    pub fn get(&self) -> &T {
        // SAFETY: constructors check the whole parent view before resolving
        // this pointer, and the lifetime is tied to that parent borrow.
        unsafe { &*self.ptr }
    }

    /// Load a `Copy` value from the proven element.
    ///
    /// The element must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        *self.get()
    }
}

/// A parent-bound proof that one mutable element is in bounds.
///
/// This capability owns the exclusive borrow of its parent view. It is neither
/// `Copy` nor `Clone`, and its pointer is private.
#[must_use]
pub struct InBoundsMut32<'a, T> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut T>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T> InBoundsMut32<'a, T> {
    #[inline(always)]
    unsafe fn from_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Borrow the proven element for reading.
    #[inline(always)]
    pub fn get(&self) -> &T {
        // SAFETY: the capability carries the exclusive parent borrow.
        unsafe { &*self.ptr }
    }

    /// Borrow the proven element for writing.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: the capability is unique and carries the exclusive borrow.
        unsafe { &mut *self.ptr }
    }

    /// Load a `Copy` value from the proven element.
    ///
    /// The element must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        *self.get()
    }

    /// Store a value into the proven element.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        *self.get_mut() = value;
    }
}

/// An immutable `N`-element view checked once at construction.
#[must_use]
pub struct StaticView32<'a, T, const N: usize> {
    ptr: *const T,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T, const N: usize> StaticView32<'a, T, N> {
    /// Create a view when the slice contains exactly `N` elements and `N` fits
    /// in a 32-bit local index. Zero-length static views are rejected.
    #[inline(always)]
    pub fn from_slice(slice: &'a [T]) -> Option<Self> {
        if N == 0 || N > u32::MAX as usize || slice.len() != N {
            return None;
        }
        Some(Self {
            ptr: slice.as_ptr(),
            _borrow: PhantomData,
        })
    }

    /// Resolve a checked local index with no further bounds branch.
    #[inline(always)]
    pub fn at(&self, index: LocalIndex32<N>) -> InBounds32<'_, T> {
        // SAFETY: LocalIndex32<N> proves index < N, and construction proved
        // that the parent contains exactly N elements.
        unsafe { InBounds32::from_ptr(self.ptr.add(index.get() as usize)) }
    }

    /// Resolve a compile-time local index.
    #[inline(always)]
    pub fn at_const<const I: u32>(&self) -> InBounds32<'_, T> {
        self.at(LocalIndex32::constant::<I>())
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        N as u32
    }

    /// Static views are never empty: `N == 0` is rejected at construction.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// A mutable `N`-element view checked once at construction.
///
/// After construction, [`at`](Self::at) and [`at_const`](Self::at_const) use a
/// `LocalIndex32<N>` proof and emit no dynamic bounds check.
#[must_use]
pub struct StaticViewMut32<'a, T, const N: usize> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

/// A checked `ROWS × COLS` mutable tile in a row-major parent allocation.
///
/// The runtime representation is one pointer. Dimensions and row stride live
/// in the type, and construction checks the whole rectangle once.
/// `ROW_STRIDE` is the caller-declared logical row width and must match the
/// parent buffer's layout:
///
/// ```text
/// base ── row 0: [ COLS elements ] ... stride gap
///         row 1: [ COLS elements ] ... stride gap
///         ...
/// ```
///
/// Interior [`at`](Self::at) calls use only `LocalIndex32` proofs and perform
/// no bounds branch.
#[must_use]
pub struct StaticTileMut32<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    StaticTileMut32<'a, T, ROWS, COLS, ROW_STRIDE>
{
    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve checked local row/column indices without another bounds branch.
    #[inline(always)]
    pub fn at(&mut self, row: LocalIndex32<ROWS>, col: LocalIndex32<COLS>) -> InBoundsMut32<'_, T> {
        // Tile construction proved that even the largest local offset fits in
        // u32. Wrapping operations state that proof without adding overflow
        // branches; only ptr.add widens the final offset.
        let offset = row
            .get()
            .wrapping_mul(ROW_STRIDE as u32)
            .wrapping_add(col.get());
        // SAFETY: both local indices are in range, and the complete rectangle
        // was checked by tile_2d32 before this wrapper was constructed.
        unsafe { InBoundsMut32::from_ptr(self.ptr.add(offset as usize)) }
    }

    /// Resolve compile-time row/column indices.
    #[inline(always)]
    pub fn at_const<const ROW: u32, const COL: u32>(&mut self) -> InBoundsMut32<'_, T> {
        self.at(
            LocalIndex32::constant::<ROW>(),
            LocalIndex32::constant::<COL>(),
        )
    }

    /// Number of logical rows in the tile.
    #[inline(always)]
    pub const fn rows(&self) -> u32 {
        ROWS as u32
    }

    /// Number of logical columns in the tile.
    #[inline(always)]
    pub const fn cols(&self) -> u32 {
        COLS as u32
    }
}

/// A writable `ROWS × COLS` tile in a matrix whose row width is a runtime
/// value.
///
/// At runtime this is just one pointer plus the row width; the tile
/// dimensions live in the type. [`DisjointSlice::tile_2d32_rt`] already
/// checked the complete rectangle (no arithmetic overflow, the tile does not
/// spill into the next row, and its bottom-right corner is inside the slice)
/// before constructing this, so interior [`at`](Self::at) calls take
/// compile-time-checked local indices and perform no bounds branch at all.
#[must_use]
pub struct RuntimeTileMut32<'a, T, const ROWS: usize, const COLS: usize> {
    ptr: *mut T,
    stride: u32,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T, const ROWS: usize, const COLS: usize> RuntimeTileMut32<'a, T, ROWS, COLS> {
    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T, stride: u32) -> Self {
        Self {
            ptr,
            stride,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve checked local row/column indices without another bounds branch.
    #[inline(always)]
    pub fn at(&mut self, row: LocalIndex32<ROWS>, col: LocalIndex32<COLS>) -> InBoundsMut32<'_, T> {
        // The rectangle proof bounded every flat offset by the parent length,
        // which can exceed u32, so the local offset math widens to u64 first.
        // Neither operation can wrap u64 (both factors fit in u32); wrapping
        // operations state that proof without adding overflow branches.
        let offset = u64::from(row.get())
            .wrapping_mul(u64::from(self.stride))
            .wrapping_add(u64::from(col.get()));
        // SAFETY: both local indices are in range, and the complete rectangle
        // was checked by tile_2d32_rt before this wrapper was constructed.
        unsafe { InBoundsMut32::from_ptr(self.ptr.add(offset as usize)) }
    }

    /// Resolve compile-time row/column indices.
    #[inline(always)]
    pub fn at_const<const ROW: u32, const COL: u32>(&mut self) -> InBoundsMut32<'_, T> {
        self.at(
            LocalIndex32::constant::<ROW>(),
            LocalIndex32::constant::<COL>(),
        )
    }

    /// Number of logical rows in the tile.
    #[inline(always)]
    pub const fn rows(&self) -> u32 {
        ROWS as u32
    }

    /// Number of logical columns in the tile.
    #[inline(always)]
    pub const fn cols(&self) -> u32 {
        COLS as u32
    }

    /// The validated row width of the parent matrix.
    #[inline(always)]
    pub const fn stride(&self) -> u32 {
        self.stride
    }
}

impl<'a, T, const N: usize> StaticViewMut32<'a, T, N> {
    /// Create a mutable view over exactly `N` elements.
    ///
    /// Zero-length views, widths larger than `u32`, and zero-sized element
    /// types are rejected. Zero-sized mutable tiles would give different GPU
    /// threads the same address, which cannot support exclusive references.
    #[inline(always)]
    pub fn from_slice(slice: &'a mut [T]) -> Option<Self> {
        if !valid_mutable_extent::<T, N>() || slice.len() != N {
            return None;
        }
        Some(Self {
            ptr: slice.as_mut_ptr(),
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve a checked local index with no further bounds branch.
    #[inline(always)]
    pub fn at(&mut self, index: LocalIndex32<N>) -> InBoundsMut32<'_, T> {
        // SAFETY: LocalIndex32<N> proves index < N, and construction proved
        // the complete N-element range in bounds.
        unsafe { InBoundsMut32::from_ptr(self.ptr.add(index.get() as usize)) }
    }

    /// Resolve a compile-time local index.
    #[inline(always)]
    pub fn at_const<const I: u32>(&mut self) -> InBoundsMut32<'_, T> {
        self.at(LocalIndex32::constant::<I>())
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        N as u32
    }

    /// Static views are never empty: `N == 0` is rejected at construction.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// A mutable run of runtime length, checked once at construction.
///
/// The 1D counterpart of [`RuntimeTileMut32`], for the clipped tail of a run
/// whose full extent would leave the parent. Its length is a runtime value, so
/// [`at`](Self::at) keeps one bounds branch where [`StaticViewMut32::at`] has
/// none.
#[must_use]
pub struct RuntimeViewMut32<'a, T> {
    ptr: *mut T,
    len: u32,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T> RuntimeViewMut32<'a, T> {
    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T, len: u32) -> Self {
        Self {
            ptr,
            len,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve a local index, `None` when it is past this run's end.
    #[inline(always)]
    pub fn at(&mut self, index: u32) -> Option<InBoundsMut32<'_, T>> {
        if index >= self.len {
            return None;
        }
        // SAFETY: the index is below the length construction checked against
        // the parent, so the element is inside the parent allocation.
        Some(unsafe { InBoundsMut32::from_ptr(self.ptr.add(index as usize)) })
    }

    /// Number of elements in this run.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the run is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One thread's run of contiguous elements, whole or clipped by the parent end.
///
/// Real launches round the grid up, so the last thread with any work usually
/// owns fewer than `N` elements. Returning `None` for that thread, as
/// [`DisjointSlice::tile_thread32`] does, pushes the tail back to raw indexing
/// and gives up the proof exactly where it is most fiddly to reason about.
///
/// Splitting the two cases keeps the check-free [`StaticViewMut32`] for every
/// thread whose run is whole, which is all but one of them, and still hands the
/// tail a checked view rather than nothing.
///
/// Matching on `&mut` and calling through the borrowed view compiles for a
/// kernel: the payload is a view of two scalars, so the importer reads it out
/// of the enum by value and no payload address is ever taken. Reaching for the
/// pointer and length by hand instead would duplicate the bounds argument each
/// view already carries.
#[must_use]
pub enum ThreadRunMut32<'a, T, const N: usize> {
    /// The thread owns all `N` elements.
    Full(StaticViewMut32<'a, T, N>),
    /// The parent ended part-way through the run.
    Clipped(RuntimeViewMut32<'a, T>),
}

impl<'a, T, const N: usize> ThreadRunMut32<'a, T, N> {
    /// Number of elements this thread owns, `N` unless the run was clipped.
    #[inline(always)]
    pub fn len(&self) -> u32 {
        match self {
            ThreadRunMut32::Full(_) => N as u32,
            ThreadRunMut32::Clipped(view) => view.len(),
        }
    }

    /// Whether the run is empty. Construction never produces an empty run.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether the parent ended part-way through this run.
    #[inline(always)]
    pub fn is_clipped(&self) -> bool {
        matches!(self, ThreadRunMut32::Clipped(_))
    }

    /// Resolve a local index, `None` when it is past this run's end.
    ///
    /// The uniform accessor, for code that treats both cases alike. Match on
    /// the variant instead when the whole-run case should keep
    /// [`StaticViewMut32::at_const`] and its absent bounds branch.
    ///
    /// Each arm defers to the view it holds, so the bounds argument lives in
    /// one place per view kind: [`LocalIndex32`] proves `index < N` for a
    /// whole run, and [`RuntimeViewMut32::at`] checks the clipped length that
    /// construction measured against the parent. This method therefore owns
    /// no `unsafe` of its own.
    #[inline(always)]
    pub fn at(&mut self, index: u32) -> Option<InBoundsMut32<'_, T>> {
        match self {
            ThreadRunMut32::Full(view) => LocalIndex32::<N>::new(index).map(|local| view.at(local)),
            ThreadRunMut32::Clipped(view) => view.at(index),
        }
    }
}

// =============================================================================
// Read-side views with runtime extents
// =============================================================================
//
// Read views need no thread token and no disjointness story: any number of
// threads may read the same element concurrently, so overlapping immutable
// views are sound by construction. The constructors are therefore plain safe
// functions over `&[T]` that prove a complete band once (in u64, so the index
// arithmetic itself cannot wrap) and return a view whose interior accesses are
// check-free. The only remaining runtime compare is the one that defines the
// access: `get`'s single bound, or the loop bound inside an iterator's
// `next`.
//
// These types are constructed inside kernels from ordinary `&[T]` parameters
// and must never appear in a kernel signature. Every method is
// `#[inline(always)]` and every proof lives in a scalar-only helper, so the
// pointer-producing wrappers MIR-inline completely and the kernel parameter's
// global-pointer provenance survives to the final loads (the same pattern as
// `checked_row_major_tile_start` below).

/// Lets a kernel read a matrix that arrived as a plain `&[T]`.
///
/// A matrix is stored as one long array, row after row. `stride` is the
/// **row width**: how many elements one row occupies, which is also the jump
/// from the start of one row to the start of the next.
///
/// ```text
/// 3 x 4 matrix, stride = 4, stored as one flat array of 12 elements:
///
///  col:      0   1   2   3
///  row 0:  [ 0   1   2   3 ]     element (row, col) lives at
///  row 1:  [ 4   5   6   7 ]     flat index  row * stride + col
///  row 2:  [ 8   9  10  11 ]
/// ```
///
/// The adapter itself checks nothing. Each [`row`](Self::row) or
/// [`col`](Self::col) call checks **once** that every element it will ever
/// hand out lies inside the slice, then returns a view whose reads need no
/// further checks. One check up front instead of one check per element read;
/// that is the entire trick.
///
/// Build this inside the kernel from a slice parameter. It must not itself be
/// a kernel parameter.
#[must_use]
pub struct MatrixView32<'a, T> {
    ptr: *const T,
    len: usize,
    stride: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T> Clone for MatrixView32<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for MatrixView32<'a, T> {}

impl<'a, T> MatrixView32<'a, T> {
    /// Wrap a flat slice together with its row width.
    ///
    /// Nothing is checked here; each [`row`](Self::row) / [`col`](Self::col)
    /// call performs its own check. A zero `stride` simply makes every later
    /// check fail with `None`, so construction itself cannot go wrong.
    #[inline(always)]
    pub fn new(slice: &'a [T], stride: u32) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            stride,
            _borrow: PhantomData,
        }
    }

    /// Check once that row `row`'s first `cols` elements all lie inside the
    /// slice, and return a view that reads them check-free.
    ///
    /// In the diagram on [`MatrixView32`], `row(1, 4)` covers flat indices
    /// 4, 5, 6, 7 (consecutive elements). The check runs in u64, so the index
    /// arithmetic itself cannot overflow:
    ///
    /// ```text
    /// cols != 0  and  cols <= stride      the strip stays inside one row
    /// row * stride + (cols - 1) < len     its last element is inside the
    ///                                     slice (and therefore all of them)
    /// ```
    ///
    /// If any part fails, the result is `None` and no pointer is created.
    #[inline(always)]
    pub fn row(&self, row: u32, cols: u32) -> Option<RowView32<'a, T>> {
        let start = checked_row_band_start(row, self.stride, cols, self.len);
        if start == INVALID_LINEAR_2D {
            return None;
        }
        // SAFETY: the scalar helper proved the complete `cols`-element band in
        // bounds, so `start` and every interior offset resolve inside the
        // parent slice. Shared reads need no disjointness.
        Some(RowView32 {
            ptr: unsafe { self.ptr.add(start as usize) },
            len: cols,
            _borrow: PhantomData,
        })
    }

    /// Check once that column `col`'s first `rows` elements all lie inside
    /// the slice, and return a view that reads them check-free.
    ///
    /// A column's elements are *not* next to each other in the flat array;
    /// each one sits a full row width after the previous. In the diagram on
    /// [`MatrixView32`], `col(2, 3)` covers flat indices 2, 6, 10, a jump of
    /// `stride = 4` per step. The check runs in u64:
    ///
    /// ```text
    /// rows != 0  and  col < stride        the column exists in this layout
    /// (rows - 1) * stride + col < len     its last element is inside the
    ///                                     slice (and therefore all of them)
    /// ```
    #[inline(always)]
    pub fn col(&self, col: u32, rows: u32) -> Option<ColView32<'a, T>> {
        let start = checked_col_band_start(col, self.stride, rows, self.len);
        if start == INVALID_LINEAR_2D {
            return None;
        }
        // SAFETY: the scalar helper proved every element `i * stride` for
        // `i < rows` in bounds of the parent slice.
        Some(ColView32 {
            ptr: unsafe { self.ptr.add(start as usize) },
            stride: self.stride,
            len: rows,
            _borrow: PhantomData,
        })
    }

    /// The row width this adapter was constructed with.
    #[inline(always)]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    /// Length of the underlying slice.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the underlying slice is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A read-only view of one row strip: consecutive elements, checked once.
///
/// Created by [`MatrixView32::row`], which verifies the whole strip is inside
/// the slice before creating any pointer. Because this view only reads, any
/// number of threads may hold overlapping ones; that is why no thread token
/// is needed here, unlike the mutable views.
///
/// [`get`](Self::get) costs exactly one compare; [`iter`](Self::iter) costs
/// nothing beyond the loop's own "am I done" compare.
#[must_use]
pub struct RowView32<'a, T> {
    ptr: *const T,
    len: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T> Clone for RowView32<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for RowView32<'a, T> {}

impl<'a, T> RowView32<'a, T> {
    /// A view of zero elements.
    ///
    /// This exists for kernels with barriers, where every thread must keep
    /// running even if its own view check failed (a thread that returns early
    /// never reaches the barrier and the block hangs). Fall back to an empty
    /// view instead: `get` on it always returns `None`, so
    /// `.get(i).unwrap_or(0.0)` turns into the fill value with no early
    /// return.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            // Never dereferenced: every accessor first requires `i < len` and
            // `len` is zero.
            ptr: core::ptr::null(),
            len: 0,
            _borrow: PhantomData,
        }
    }

    /// Load element `i` with a single bounds compare.
    ///
    /// The element must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    pub fn get(&self, i: u32) -> Option<T>
    where
        T: Copy,
    {
        if i < self.len {
            // SAFETY: construction proved the complete band, and `i` is
            // inside it.
            Some(unsafe { *self.ptr.add(i as usize) })
        } else {
            None
        }
    }

    /// Iterate over the elements in order.
    ///
    /// An iterator must ask "is there a next element?" once per step; that
    /// single compare *is* the loop's exit condition, which even a raw
    /// pointer loop needs. So iteration adds zero checks on top of the loop
    /// itself.
    #[inline(always)]
    pub fn iter(&self) -> RowViewIter32<'a, T> {
        RowViewIter32 {
            ptr: self.ptr,
            len: self.len,
            next: 0,
            _borrow: PhantomData,
        }
    }

    /// Pair this row with an equal-length column for a dot product.
    ///
    /// Zipping two iterators normally asks "does each side have a next
    /// element?", two compares per step. This checks **once, here** that both
    /// sides have the same length; the returned iterator then advances one
    /// shared counter and yields `(row[i], col[i])`, so the loop keeps a
    /// single compare per step: its own exit condition.
    #[inline(always)]
    pub fn zip_exact(self, col: ColView32<'a, T>) -> Option<ZipView32<'a, T>> {
        if self.len != col.len {
            return None;
        }
        Some(ZipView32 {
            row_ptr: self.ptr,
            col_ptr: col.ptr,
            col_stride: col.stride,
            len: self.len,
            next: 0,
            _borrow: PhantomData,
        })
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the view has zero elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A read-only view of one column: elements one row width apart, checked
/// once.
///
/// Created by [`MatrixView32::col`], which verifies the whole column strip is
/// inside the slice before creating any pointer. Element `i` of the view
/// lives at flat offset `i * stride` from the column's top. Like
/// [`RowView32`], reading needs no thread token: overlapping reads between
/// threads are harmless.
#[must_use]
pub struct ColView32<'a, T> {
    ptr: *const T,
    stride: u32,
    len: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T> Clone for ColView32<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for ColView32<'a, T> {}

impl<'a, T> ColView32<'a, T> {
    /// A view of zero elements. See [`RowView32::empty`].
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            // Never dereferenced: every accessor first requires `i < len` and
            // `len` is zero.
            ptr: core::ptr::null(),
            stride: 0,
            len: 0,
            _borrow: PhantomData,
        }
    }

    /// Load element `i` (flat offset `i * stride`) with a single bounds
    /// compare.
    ///
    /// The element must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    pub fn get(&self, i: u32) -> Option<T>
    where
        T: Copy,
    {
        if i < self.len {
            // Construction proved the last element's flat offset in bounds of
            // the parent, which can exceed u32, so widen before multiplying.
            // u32 * u32 cannot wrap u64.
            let offset = u64::from(i).wrapping_mul(u64::from(self.stride));
            // SAFETY: `i < len` and the complete band was proven at
            // construction.
            Some(unsafe { *self.ptr.add(offset as usize) })
        } else {
            None
        }
    }

    /// Iterate over the elements in order.
    ///
    /// An iterator must ask "is there a next element?" once per step; that
    /// single compare *is* the loop's exit condition, which even a raw
    /// pointer loop needs. So iteration adds zero checks on top of the loop
    /// itself.
    #[inline(always)]
    pub fn iter(&self) -> ColViewIter32<'a, T> {
        ColViewIter32 {
            ptr: self.ptr,
            stride: self.stride,
            len: self.len,
            next: 0,
            _borrow: PhantomData,
        }
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        self.len
    }

    /// Whether the view has zero elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The flat distance between consecutive elements of the view.
    #[inline(always)]
    pub const fn stride(&self) -> u32 {
        self.stride
    }
}

/// Sequential iterator over a [`RowView32`].
#[must_use]
pub struct RowViewIter32<'a, T> {
    ptr: *const T,
    len: u32,
    next: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T: Copy> Iterator for RowViewIter32<'a, T> {
    type Item = T;

    /// Every element yielded must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        if self.next < self.len {
            // SAFETY: the view constructor proved the complete band and
            // `next < len` holds here.
            let value = unsafe { *self.ptr.add(self.next as usize) };
            // next < len <= u32::MAX, so the increment cannot wrap; the
            // wrapping form states that without an overflow branch.
            self.next = self.next.wrapping_add(1);
            Some(value)
        } else {
            None
        }
    }
}

/// Sequential iterator over a [`ColView32`].
#[must_use]
pub struct ColViewIter32<'a, T> {
    ptr: *const T,
    stride: u32,
    len: u32,
    next: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T: Copy> Iterator for ColViewIter32<'a, T> {
    type Item = T;

    /// Every element yielded must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        if self.next < self.len {
            let offset = u64::from(self.next).wrapping_mul(u64::from(self.stride));
            // SAFETY: the view constructor proved the complete band and
            // `next < len` holds here.
            let value = unsafe { *self.ptr.add(offset as usize) };
            self.next = self.next.wrapping_add(1);
            Some(value)
        } else {
            None
        }
    }
}

/// A row and a column fused into one iterator of `(row[i], col[i])` pairs.
///
/// Built by [`RowView32::zip_exact`] after a single length-equality check.
/// One shared counter drives both sides, so a dot-product loop compiles to
/// exactly what a hand-written pointer loop would be:
/// load, load, multiply-add, advance, compare, branch. No bounds checks
/// inside the loop.
#[must_use]
pub struct ZipView32<'a, T> {
    row_ptr: *const T,
    col_ptr: *const T,
    col_stride: u32,
    len: u32,
    next: u32,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T: Copy> Iterator for ZipView32<'a, T> {
    type Item = (T, T);

    /// Every element of both bands must already be initialized; see
    /// [Initialization](self#initialization) in the module docs.
    #[inline(always)]
    fn next(&mut self) -> Option<(T, T)> {
        if self.next < self.len {
            let i = self.next;
            let col_offset = u64::from(i).wrapping_mul(u64::from(self.col_stride));
            // SAFETY: both band proofs ran at view construction, the length
            // equality ran in zip_exact, and `i < len` holds here.
            let row_value = unsafe { *self.row_ptr.add(i as usize) };
            let col_value = unsafe { *self.col_ptr.add(col_offset as usize) };
            self.next = i.wrapping_add(1);
            Some((row_value, col_value))
        } else {
            None
        }
    }
}

#[inline(always)]
const fn valid_mutable_extent<T, const N: usize>() -> bool {
    N != 0 && N <= u32::MAX as usize && size_of::<T>() != 0
}

#[inline(always)]
const fn valid_row_major_shape<T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>()
-> bool {
    ROWS != 0
        && COLS != 0
        && ROW_STRIDE != 0
        && ROWS <= u32::MAX as usize
        && COLS <= u32::MAX as usize
        && ROW_STRIDE <= u32::MAX as usize
        && ROW_STRIDE >= COLS
        && size_of::<T>() != 0
}

const INVALID_LINEAR_2D: u64 = u64::MAX;

/// Compute `row * stride + col` without an `Option` aggregate. Returning the
/// sentinel keeps the device representation scalar; callers reject it before
/// converting back to a pointer offset.
#[inline(always)]
fn checked_linear_2d(row: u32, stride: u32, col: u32) -> u64 {
    if stride == 0 || row > (u32::MAX - col) / stride {
        INVALID_LINEAR_2D
    } else {
        u64::from(row * stride + col)
    }
}

#[inline(always)]
fn scaled_tile_axis_fits(origin: u32, width: u32) -> bool {
    width != 0 && origin <= (u32::MAX - (width - 1)) / width
}

/// Prove a consecutive row strip `[row * stride, row * stride + cols)` in
/// bounds, entirely in u64.
///
/// Widening every u32 input first means `row * stride + (cols - 1)` cannot
/// wrap: it is at most `(2^32 - 1)^2 + (2^32 - 1) < 2^64`. The sentinel keeps
/// this proof scalar-only (no `Option` aggregate), so the pointer-producing
/// wrapper above it MIR-inlines and keeps the kernel parameter's provenance.
#[inline(always)]
fn checked_row_band_start(row: u32, stride: u32, cols: u32, len: usize) -> u64 {
    if cols == 0 || cols > stride {
        return INVALID_LINEAR_2D;
    }
    let start = u64::from(row) * u64::from(stride);
    let last = start + u64::from(cols - 1);
    if last < len as u64 {
        // start <= last < len <= u64::MAX, so start never equals the sentinel.
        start
    } else {
        INVALID_LINEAR_2D
    }
}

/// Prove a column strip (`rows` elements, `stride` apart, first at
/// flat index `col`) in bounds, entirely in u64.
#[inline(always)]
fn checked_col_band_start(col: u32, stride: u32, rows: u32, len: usize) -> u64 {
    if rows == 0 || col >= stride {
        return INVALID_LINEAR_2D;
    }
    let last = u64::from(rows - 1) * u64::from(stride) + u64::from(col);
    if last < len as u64 {
        // col < 2^32 <= u64::MAX, so the start never equals the sentinel.
        u64::from(col)
    } else {
        INVALID_LINEAR_2D
    }
}

impl<'a, T> DisjointSlice<'a, T, Index1D> {
    /// Check this thread's 32-bit element index once and return a parent-bound
    /// read/write capability.
    #[inline(always)]
    pub fn element_thread32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> Option<InBoundsMut32<'_, T>> {
        if size_of::<T>() == 0 {
            return None;
        }
        let index = thread.get() as usize;
        if index >= self.len() {
            return None;
        }
        // SAFETY: the single bounds check above covers the resolved element.
        // ThreadIndex32 is unique for the validated 1D launch and is consumed.
        Some(unsafe { InBoundsMut32::from_ptr(self.as_mut_ptr().add(index)) })
    }
}

impl<'a, T, const N: usize> DisjointSlice<'a, T, LinearTiles<N>> {
    /// Check one complete per-thread tile and return a check-free static view.
    ///
    /// ```text
    /// prove thread * N + (N - 1) <= u32::MAX
    /// start = thread * N
    /// last  = start + (N - 1)
    /// accept only when last < slice.len()
    /// ```
    #[inline(always)]
    pub fn tile_thread32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> Option<StaticViewMut32<'_, T, N>> {
        if size_of::<T>() == 0 {
            return None;
        }
        let start = checked_linear_tile_start::<N>(thread.get(), self.len());
        if start == u64::MAX {
            return None;
        }
        let start = start as u32;
        // SAFETY: start..end was computed without overflow and checked as one
        // complete range. Consuming the unique thread index makes tiles
        // disjoint, and the non-ZST check gives each element an address.
        Some(unsafe { StaticViewMut32::from_checked_ptr(self.as_mut_ptr().add(start as usize)) })
    }

    /// This thread's run of up to `N` contiguous elements, clipped at the end
    /// of the parent rather than refused.
    ///
    /// [`tile_thread32`](Self::tile_thread32) accepts only a whole tile, so the
    /// one thread whose run straddles the end gets `None` and has to fall back
    /// to raw indexing. Since launches round the grid up, that thread almost
    /// always exists. This returns a [`ThreadRunMut32`] instead: `Full` for
    /// every thread whose `N` elements fit, `Clipped` for the one that
    /// straddles the end, and `None` for a thread whose run starts at or past
    /// the end and therefore owns nothing.
    ///
    /// Like the rest of the 32-suffix family, run starts live in `u32`: a run
    /// that would start past element offset `u32::MAX` is also `None`, and a
    /// width `N` of zero or above `u32::MAX` is rejected outright, exactly as
    /// [`tile_thread32`](Self::tile_thread32) rejects it.
    ///
    /// Disjointness is unchanged. Thread `t` owns `t * N .. t * N + N`, runs of
    /// distinct threads do not overlap, and clipping only shortens the last
    /// one.
    #[inline(always)]
    pub fn thread_run32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> Option<ThreadRunMut32<'_, T, N>> {
        if size_of::<T>() == 0 {
            return None;
        }
        let bounds = checked_run_bounds::<N>(thread.get() as u64, self.len());
        if bounds == NO_RUN {
            return None;
        }
        let start = run_bounds_start(bounds) as usize;
        let count = run_bounds_count(bounds);
        let ptr = self.as_mut_ptr();

        if count as usize == N {
            // SAFETY: the complete N-element range starts inside the parent and
            // ends at or before its end. The consumed unique thread index makes
            // runs of distinct threads disjoint.
            return Some(ThreadRunMut32::Full(unsafe {
                StaticViewMut32::from_checked_ptr(ptr.add(start))
            }));
        }

        // SAFETY: `count` is what remains of the parent from `start`, proven
        // by `checked_run_bounds`, so the clipped range stays inside the
        // parent allocation.
        Some(ThreadRunMut32::Clipped(unsafe {
            RuntimeViewMut32::from_checked_ptr(ptr.add(start), count)
        }))
    }

    /// Walk every run this thread owns under a grid-stride loop.
    ///
    /// The period is read from the launch geometry rather than taken as an
    /// argument. A passed stride is the defect described in #583: one that
    /// disagrees with the actual launch makes threads overlap silently, and
    /// nothing in the signature would catch it.
    ///
    /// # Disjointness
    ///
    /// With `period = gridDim.x * blockDim.x`, thread `t` owns tiles `t`,
    /// `t + period`, `t + 2 * period`, and so on. Every tile index this thread
    /// takes is congruent to `t` modulo `period`, and `t < period` holds for a
    /// 1D launch because `t = blockIdx.x * blockDim.x + threadIdx.x`, so two
    /// distinct threads never take the same tile. Clipping only shortens the
    /// final run.
    ///
    /// # The `u32::MAX` offset cutoff
    ///
    /// Run starts live in `u32`, like everything in the 32-suffix family. A
    /// run whose start element offset would exceed `u32::MAX` is never
    /// yielded: for a parent longer than 2^32 elements the walk ends at that
    /// boundary rather than continuing into the region past it. A run that
    /// starts at or below the boundary may still extend past it, because each
    /// element is proven against the parent length in full u64 width.
    #[inline(always)]
    pub fn grid_stride_runs32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> GridStrideRuns32<'_, 'a, T, N> {
        let period = (crate::thread::gridDim_x() as u64) * (crate::thread::blockDim_x() as u64);
        GridStrideRuns32 {
            slice: self,
            next_tile: thread.get() as u64,
            period,
        }
    }
}

/// The runs one thread owns under a grid-stride loop.
///
/// Produced by [`DisjointSlice::grid_stride_runs32`]. Each run is borrowed from
/// the parent slice for as long as it is held, so runs are taken one at a time
/// rather than through [`Iterator`], which cannot express that.
///
/// ```rust,ignore
/// let mut runs = data.grid_stride_runs32(thread);
/// while let Some(mut run) = runs.next_run() {
///     for k in 0..run.len() {
///         if let Some(mut slot) = run.at(k) {
///             slot.write(slot.read() * 2.0);
///         }
///     }
/// }
/// ```
#[must_use]
pub struct GridStrideRuns32<'slice, 'a, T, const N: usize> {
    slice: &'slice mut DisjointSlice<'a, T, LinearTiles<N>>,
    next_tile: u64,
    period: u64,
}

impl<'slice, 'a, T, const N: usize> GridStrideRuns32<'slice, 'a, T, N> {
    /// The next run this thread owns, or `None` once they are exhausted.
    ///
    /// Exhaustion includes the `u32::MAX` start cutoff described on
    /// [`DisjointSlice::grid_stride_runs32`]: for a parent longer than 2^32
    /// elements the walk stops at that boundary.
    #[inline(always)]
    pub fn next_run(&mut self) -> Option<ThreadRunMut32<'_, T, N>> {
        // A zero period would revisit the same tile forever; a launch always
        // has at least one thread, so refusing it costs nothing real.
        if size_of::<T>() == 0 || self.period == 0 {
            return None;
        }
        let bounds = checked_run_bounds::<N>(self.next_tile, self.slice.len());
        if bounds == NO_RUN {
            return None;
        }
        self.next_tile = self.next_tile.checked_add(self.period)?;

        let start = run_bounds_start(bounds) as usize;
        let count = run_bounds_count(bounds);
        let ptr = self.slice.as_mut_ptr();
        if count as usize == N {
            // SAFETY: the complete N-element range lies inside the parent, and
            // every tile index this thread takes is congruent to its own index
            // modulo the launch period, so runs never overlap another thread's.
            return Some(ThreadRunMut32::Full(unsafe {
                StaticViewMut32::from_checked_ptr(ptr.add(start))
            }));
        }
        // SAFETY: `count` is what remains of the parent from `start`, proven
        // by `checked_run_bounds`, so the clipped range stays inside the
        // parent allocation.
        Some(ThreadRunMut32::Clipped(unsafe {
            RuntimeViewMut32::from_checked_ptr(ptr.add(start), count)
        }))
    }

    /// The grid-stride period this walk uses, in tiles.
    #[inline(always)]
    pub const fn period(&self) -> u64 {
        self.period
    }
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    DisjointSlice<'a, T, RowMajorTiles<ROWS, COLS, ROW_STRIDE>>
{
    /// Check one complete rectangular tile and return a check-free static view.
    ///
    /// `ROW_STRIDE` is the caller-declared logical row width and must match
    /// the buffer's layout. The slice length does not have to be a multiple of
    /// the row width; a tile is returned only when its complete rectangle fits.
    ///
    /// Construction proves the following before creating a pointer:
    ///
    /// ```text
    /// start_row = thread.row * ROWS
    /// start_col = thread.col * COLS
    /// last_col  < ROW_STRIDE       (the tile cannot wrap into the next row)
    /// last_row * ROW_STRIDE + last_col < parent.len()
    /// ```
    #[inline(always)]
    pub fn tile_2d32<'kernel>(
        &mut self,
        thread: ThreadCoord2D32<'kernel>,
    ) -> Option<StaticTileMut32<'_, T, ROWS, COLS, ROW_STRIDE>> {
        let start = checked_row_major_tile_start::<T, ROWS, COLS, ROW_STRIDE>(
            thread.row(),
            thread.col(),
            self.len(),
        );
        if start == INVALID_LINEAR_2D {
            return None;
        }

        // SAFETY: the scalar-only helper checked the complete rectangle.
        // Distinct 2D thread coordinates map to disjoint row and column bands.
        Some(unsafe { StaticTileMut32::from_checked_ptr(self.as_mut_ptr().add(start as usize)) })
    }
}

impl<'a, T, const ROWS: usize, const COLS: usize>
    DisjointSlice<'a, T, RuntimeRowMajorTiles<ROWS, COLS>>
{
    /// Check one complete rectangular tile against a **runtime** row width
    /// and return a tile whose accesses need no further checks.
    ///
    /// This is [`tile_2d32`] for matrices whose row width is a runtime value
    /// (like a GEMM dimension `n`) instead of a compile-time constant. It
    /// consumes a [`ThreadCoord2D32`] carrying the thread's real hardware
    /// coordinates and mutably borrows the slice, so at most one tile is
    /// live per thread at a time (re-minting the coordinate after dropping a
    /// tile re-derives the same rectangle, never a second aliasing one).
    /// Before creating any pointer it checks, entirely in u64:
    ///
    /// ```text
    /// start_row = thread.row * ROWS        (widened; cannot wrap u64)
    /// start_col = thread.col * COLS
    /// last_col  < stride                   the tile stays inside one row, so
    ///                                      it cannot overlap its left/right
    ///                                      neighbor (given a uniform stride)
    /// last_row * stride + last_col < len   the bottom-right corner, and
    ///                                      therefore every element, is
    ///                                      inside the slice
    /// ```
    ///
    /// Zero-sized element types are rejected, as for every mutable view:
    /// distinct threads would otherwise share one address.
    ///
    /// # Why the stride comes from the slice
    ///
    /// Two threads that disagree about the row width describe two different
    /// grids over the same memory, and their "disjoint" tiles can land on the
    /// same elements. With 1x1 tiles: thread `(1, 0)` with stride 5 resolves
    /// flat index `1*5 + 0 = 5`, while thread `(0, 5)` with stride 100
    /// resolves `0*100 + 5 = 5`. Same element, two `&mut`, a data race.
    ///
    /// The row width is therefore a property of the slice rather than of the
    /// call. The host writes it into the launch packet beside the pointer and
    /// length, every thread reads that one word, and device code has no step
    /// at which it could substitute another. Under a width fixed this way the
    /// `last_col < stride` check below keeps each tile inside one logical row,
    /// which makes the coordinate-to-offset mapping injective and the tiles of
    /// distinct threads disjoint.
    ///
    /// A per-call width cannot give that guarantee even when each candidate
    /// value is itself launch-uniform, because safe code can select between two
    /// such values under a thread-varying condition, or pass different ones at
    /// two call sites on one slice.
    ///
    /// The bound width still has to be the row width this slice actually uses
    /// for the kernel to compute the intended result. That is a correctness
    /// requirement rather than a safety one: a width that misdescribes the
    /// layout gives wrong answers inside the slice, never aliasing or an
    /// out-of-bounds access.
    ///
    /// When the row width is known at compile time, prefer [`tile_2d32`],
    /// which makes a mismatch a type error.
    ///
    /// [`tile_2d32`]: DisjointSlice::tile_2d32
    #[inline(always)]
    pub fn tile_2d32_rt<'kernel>(
        &mut self,
        thread: ThreadCoord2D32<'kernel>,
    ) -> Option<RuntimeTileMut32<'_, T, ROWS, COLS>> {
        let stride = self.row_width();
        let start = checked_runtime_tile_start::<T, ROWS, COLS>(
            thread.row(),
            thread.col(),
            stride,
            self.len(),
        );
        if start == INVALID_LINEAR_2D {
            return None;
        }

        // SAFETY: the scalar-only helper checked the complete rectangle.
        // The `&mut self` borrow keeps at most one tile live per thread, and
        // a re-minted coordinate re-derives the same rectangle. Across
        // threads, the row width is the one the host bound to this slice, under
        // which distinct hardware coordinates map to disjoint row and column
        // bands.
        Some(unsafe {
            RuntimeTileMut32::from_checked_ptr(self.as_mut_ptr().add(start as usize), stride)
        })
    }
}

/// Check a complete row-major tile without accepting or returning a pointer.
///
/// Keeping this proof scalar-only lets `tile_2d32` MIR-inline as a tiny pointer
/// wrapper. The kernel's original global pointer provenance therefore reaches
/// the final loads and stores before LLVM capture/address-space inference.
#[inline(always)]
fn checked_row_major_tile_start<
    T,
    const ROWS: usize,
    const COLS: usize,
    const ROW_STRIDE: usize,
>(
    thread_row: u32,
    thread_col: u32,
    len: usize,
) -> u64 {
    if !valid_row_major_shape::<T, ROWS, COLS, ROW_STRIDE>() {
        return INVALID_LINEAR_2D;
    }

    let rows = ROWS as u32;
    let cols = COLS as u32;
    let stride = ROW_STRIDE as u32;
    if !scaled_tile_axis_fits(thread_row, rows) || !scaled_tile_axis_fits(thread_col, cols) {
        return INVALID_LINEAR_2D;
    }

    let start_row = thread_row * rows;
    let start_col = thread_col * cols;
    let last_row = start_row + (rows - 1);
    let last_col = start_col + (cols - 1);

    // The X tile must remain inside one logical row. This also makes tiles
    // owned by adjacent X threads disjoint.
    if last_col >= stride {
        return INVALID_LINEAR_2D;
    }

    let start = checked_linear_2d(start_row, stride, start_col);
    let last = checked_linear_2d(last_row, stride, last_col);
    if start == INVALID_LINEAR_2D || last == INVALID_LINEAR_2D || (last as usize) >= len {
        INVALID_LINEAR_2D
    } else {
        start
    }
}

#[inline(always)]
const fn valid_runtime_tile_shape<T, const ROWS: usize, const COLS: usize>() -> bool {
    ROWS != 0
        && COLS != 0
        && ROWS <= u32::MAX as usize
        && COLS <= u32::MAX as usize
        && size_of::<T>() != 0
}

/// Check a complete runtime-row-width tile without accepting or returning a
/// pointer, keeping `tile_2d32_rt` a tiny inlinable pointer wrapper.
///
/// All arithmetic is u64 over widened u32 inputs. The only product that could
/// wrap u64 is `last_row * stride` (both factors up to about 2^64 / 2^32), so
/// it is guarded by a division compare before being performed.
#[inline(always)]
fn checked_runtime_tile_start<T, const ROWS: usize, const COLS: usize>(
    thread_row: u32,
    thread_col: u32,
    stride: u32,
    len: usize,
) -> u64 {
    if !valid_runtime_tile_shape::<T, ROWS, COLS>() || stride == 0 {
        return INVALID_LINEAR_2D;
    }

    let rows = ROWS as u64;
    let cols = COLS as u64;
    let stride = u64::from(stride);
    // u32-range factor products cannot wrap u64, and neither can adding a
    // value below 2^32 afterwards.
    let start_row = u64::from(thread_row) * rows;
    let start_col = u64::from(thread_col) * cols;
    let last_row = start_row + (rows - 1);
    let last_col = start_col + (cols - 1);

    // The X tile must remain inside one logical row. Under a launch-uniform
    // row width this also makes tiles owned by adjacent X threads disjoint.
    if last_col >= stride {
        return INVALID_LINEAR_2D;
    }
    if last_row > (u64::MAX - last_col) / stride {
        return INVALID_LINEAR_2D;
    }
    let last = last_row * stride + last_col;
    if last >= len as u64 {
        return INVALID_LINEAR_2D;
    }
    // start <= last < len <= u64::MAX, so the start never equals the sentinel.
    start_row * stride + start_col
}

/// Keep the range proof scalar-only so the pointer-bearing wrapper above stays
/// small enough to inline without losing the kernel parameter's global-memory
/// provenance.
#[inline(always)]
fn checked_linear_tile_start<const N: usize>(thread: u32, len: usize) -> u64 {
    if N == 0 || N > u32::MAX as usize {
        return u64::MAX;
    }
    let width = N as u32;
    let last_offset = width - 1;
    // Prove both arithmetic operations before performing either one. We check
    // the inclusive last element so a one-element tile at u32::MAX remains
    // representable.
    if thread > (u32::MAX - last_offset) / width {
        return u64::MAX;
    }
    let start = thread * width;
    let last = start + last_offset;
    if (last as usize) < len {
        u64::from(start)
    } else {
        u64::MAX
    }
}

/// The reserved no-run value for [`checked_run_bounds`]. A run that exists has
/// at least one element, so its packed count field is nonzero and a valid
/// packed word can never be zero.
const NO_RUN: u64 = 0;

/// Prove one clip-tolerant run of a `LinearTiles<N>` walk in bounds, entirely
/// in u64.
///
/// The run for tile index `tile` covers `tile * N .. tile * N + N`, shortened
/// at `len`. On success the element count is packed above the start offset,
/// `(count << 32) | start`, with `start <= u32::MAX` and
/// `1 <= count <= N <= u32::MAX`, so the packing is lossless and never
/// produces the reserved [`NO_RUN`] zero. Like [`checked_linear_tile_start`],
/// the proof stays scalar-only so the pointer-bearing wrappers above it
/// MIR-inline without losing the kernel parameter's provenance.
///
/// [`NO_RUN`] when `N` is zero or exceeds `u32::MAX` (the same widths
/// [`checked_linear_tile_start`] and `valid_mutable_extent` reject), when
/// `tile * N` overflows, or when the run would start at or past `len` or past
/// the `u32::MAX` element offset the 32-suffix family stops at.
#[inline(always)]
fn checked_run_bounds<const N: usize>(tile: u64, len: usize) -> u64 {
    if N == 0 || N > u32::MAX as usize {
        return NO_RUN;
    }
    let start = match tile.checked_mul(N as u64) {
        Some(start) => start,
        None => return NO_RUN,
    };
    if start >= len as u64 || start > u32::MAX as u64 {
        return NO_RUN;
    }
    let available = len as u64 - start;
    let count = if available < N as u64 {
        available
    } else {
        N as u64
    };
    (count << 32) | start
}

/// The start offset packed into a non-[`NO_RUN`] [`checked_run_bounds`] word.
#[inline(always)]
const fn run_bounds_start(bounds: u64) -> u32 {
    bounds as u32
}

/// The element count packed into a non-[`NO_RUN`] [`checked_run_bounds`] word.
#[inline(always)]
const fn run_bounds_count(bounds: u64) -> u32 {
    (bounds >> 32) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_indices_reject_invalid_extents_and_offsets() {
        assert!(LocalIndex32::<0>::new(0).is_none());
        assert!(LocalIndex32::<4>::new(3).is_some());
        assert!(LocalIndex32::<4>::new(4).is_none());
    }

    #[test]
    fn immutable_static_view_reads_after_one_shape_check() {
        let values = [10_u32, 20, 30, 40];
        let view = StaticView32::<_, 4>::from_slice(&values).unwrap();
        assert_eq!(view.at_const::<2>().read(), 30);
    }

    #[test]
    fn mutable_static_view_writes_after_one_shape_check() {
        let mut values = [0_u32; 4];
        let mut view = StaticViewMut32::<_, 4>::from_slice(&mut values).unwrap();
        view.at_const::<3>().write(9);
        assert_eq!(values[3], 9);
    }

    #[test]
    fn mutable_static_view_rejects_zero_sized_elements() {
        let mut values = [(); 4];
        assert!(StaticViewMut32::<_, 4>::from_slice(&mut values).is_none());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn tile_range_keeps_the_exact_u32_boundary_and_rejects_overflow() {
        let full_u32_len = u32::MAX as usize + 1;
        assert_eq!(
            checked_linear_tile_start::<1>(u32::MAX, full_u32_len),
            u64::from(u32::MAX)
        );
        assert_eq!(
            checked_linear_tile_start::<1>(u32::MAX, u32::MAX as usize),
            u64::MAX
        );
        assert_eq!(
            checked_linear_tile_start::<2>(u32::MAX, full_u32_len),
            u64::MAX
        );
    }

    #[test]
    fn run_bounds_reject_empty_and_oversized_widths() {
        assert_eq!(checked_run_bounds::<0>(0, 16), NO_RUN);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            checked_run_bounds::<{ u32::MAX as usize + 1 }>(0, usize::MAX),
            NO_RUN
        );
    }

    #[test]
    fn run_bounds_stop_at_the_parent_end() {
        // The first tile at or past the data owns nothing: start == len...
        assert_eq!(checked_run_bounds::<4>(2, 8), NO_RUN);
        // ...and start > len.
        assert_eq!(checked_run_bounds::<4>(3, 8), NO_RUN);
        // An empty parent owns no runs at all.
        assert_eq!(checked_run_bounds::<4>(0, 0), NO_RUN);
    }

    #[test]
    fn run_bounds_decide_full_against_clipped_at_the_exact_boundary() {
        // available == N: the run is still whole.
        let full = checked_run_bounds::<4>(1, 8);
        assert_eq!(run_bounds_start(full), 4);
        assert_eq!(run_bounds_count(full), 4);
        // available == N - 1: one element short, so the run clips.
        let clipped = checked_run_bounds::<4>(1, 7);
        assert_eq!(run_bounds_start(clipped), 4);
        assert_eq!(run_bounds_count(clipped), 3);
        // available == 1: the shortest run that still exists.
        let last = checked_run_bounds::<4>(2, 9);
        assert_eq!(run_bounds_start(last), 8);
        assert_eq!(run_bounds_count(last), 1);
    }

    #[test]
    fn run_bounds_reject_tile_scale_overflow() {
        // tile * N overflows u64 outright.
        assert_eq!(checked_run_bounds::<2>(u64::MAX, usize::MAX), NO_RUN);
        // The product fits u64 but lands past the u32::MAX start offset.
        assert_eq!(
            checked_run_bounds::<{ u32::MAX as usize }>(2, usize::MAX),
            NO_RUN
        );
        // 2^33 * 2^31 == 2^64 wraps to a start of zero, which the start
        // guard would wave through; only the checked multiply rejects it.
        assert_eq!(checked_run_bounds::<{ 1usize << 31 }>(1 << 33, 8), NO_RUN);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn run_bounds_let_a_boundary_run_extend_past_the_u32_offset_space() {
        // The u32::MAX cutoff bounds where a run may start, not where it may
        // end: tile 1431655765 of width 3 starts exactly at u32::MAX, and
        // its remaining two elements sit at offsets 2^32 and 2^32 + 1.
        let len = u32::MAX as usize + 4; // 2^32 + 3, room past the run's end
        let run = checked_run_bounds::<3>(1431655765, len);
        assert_eq!(run_bounds_start(run), u32::MAX);
        assert_eq!(run_bounds_count(run), 3);
        // The very next tile would start at 2^32 + 2, past the cutoff, and
        // is refused even though the parent still has an element there.
        assert_eq!(checked_run_bounds::<3>(1431655766, len), NO_RUN);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn run_bounds_keep_the_exact_u32_boundary_and_stop_past_it() {
        let long = u32::MAX as usize + 2;
        // A run may start exactly at element offset u32::MAX...
        let edge = checked_run_bounds::<1>(u64::from(u32::MAX), long);
        assert_eq!(run_bounds_start(edge), u32::MAX);
        assert_eq!(run_bounds_count(edge), 1);
        // ...but never past it, even with parent elements remaining. This is
        // the documented cutoff of the 32-suffix family, not the end of data.
        assert_eq!(
            checked_run_bounds::<1>(u64::from(u32::MAX) + 1, long),
            NO_RUN
        );
    }

    #[test]
    fn grid_stride_walk_guards_a_zero_period_and_covers_runs_exactly_once() {
        let mut data = [0u32; 11];
        // A zero period would revisit tile 0 forever; the guard refuses to
        // yield anything instead.
        {
            let mut slice =
                unsafe { DisjointSlice::<u32, LinearTiles<4>>::from_mut_slice(&mut data) };
            let mut runs = GridStrideRuns32 {
                slice: &mut slice,
                next_tile: 0,
                period: 0,
            };
            assert!(runs.next_run().is_none());
        }
        // Two host-simulated threads with period 2: thread 0 takes tiles 0 and
        // 2 (the last clipped to 3 elements), thread 1 takes tile 1 and then
        // terminates at start >= len. Every element is touched exactly once.
        for thread in 0..2u64 {
            let mut slice =
                unsafe { DisjointSlice::<u32, LinearTiles<4>>::from_mut_slice(&mut data) };
            let mut runs = GridStrideRuns32 {
                slice: &mut slice,
                next_tile: thread,
                period: 2,
            };
            while let Some(mut run) = runs.next_run() {
                for k in 0..run.len() {
                    let mut slot = run.at(k).expect("k < len must resolve");
                    let value = slot.read();
                    slot.write(value + 1);
                }
            }
        }
        assert!(data.iter().all(|&touched| touched == 1));
    }

    #[test]
    fn row_major_shape_rejects_empty_wide_and_zero_sized_tiles() {
        assert!(!valid_row_major_shape::<u32, 0, 4, 8>());
        assert!(!valid_row_major_shape::<u32, 2, 9, 8>());
        assert!(!valid_row_major_shape::<(), 2, 4, 8>());
        assert!(valid_row_major_shape::<u32, 2, 4, 8>());
        #[cfg(target_pointer_width = "64")]
        assert!(!valid_row_major_shape::<u32, { u32::MAX as usize + 1 }, 1, 1>());
    }

    #[test]
    fn scalar_linear_helper_uses_a_reserved_failure_value() {
        assert_eq!(checked_linear_2d(2, 10, 3), 23);
        assert_eq!(checked_linear_2d(1, 0, 0), INVALID_LINEAR_2D);
        assert_eq!(checked_linear_2d(u32::MAX, 1, 0), u32::MAX as u64);
        assert_eq!(
            checked_linear_2d(u32::MAX, u32::MAX, u32::MAX),
            INVALID_LINEAR_2D
        );
    }

    #[test]
    fn row_major_tile_helper_checks_the_complete_rectangle() {
        assert_eq!(checked_row_major_tile_start::<u32, 2, 4, 16>(1, 1, 56), 36);
        assert_eq!(
            checked_row_major_tile_start::<u32, 2, 4, 16>(1, 1, 55),
            INVALID_LINEAR_2D
        );
        assert_eq!(checked_row_major_tile_start::<u32, 1, 2, 64>(0, 31, 64), 62);
        assert_eq!(
            checked_row_major_tile_start::<u32, 1, 2, 64>(0, 32, 128),
            INVALID_LINEAR_2D
        );
        assert_eq!(
            checked_row_major_tile_start::<u32, 2, 1, 1>(u32::MAX, 0, usize::MAX),
            INVALID_LINEAR_2D
        );
        assert_eq!(
            checked_row_major_tile_start::<(), 1, 1, 1>(0, 0, 1),
            INVALID_LINEAR_2D
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn row_major_tile_helper_keeps_the_exact_u32_linear_boundary() {
        assert_eq!(
            checked_row_major_tile_start::<u32, 1, 1, 1>(u32::MAX, 0, u32::MAX as usize + 1,),
            u32::MAX as u64
        );
    }

    #[test]
    fn tile_axis_precheck_keeps_the_exact_u32_boundary() {
        assert!(scaled_tile_axis_fits(u32::MAX, 1));
        assert!(scaled_tile_axis_fits(u32::MAX / 2, 2));
        assert!(!scaled_tile_axis_fits(u32::MAX / 2 + 1, 2));
        assert!(!scaled_tile_axis_fits(0, 0));
    }

    #[test]
    fn static_tile_wrapper_is_one_pointer() {
        assert_eq!(
            size_of::<StaticTileMut32<'_, u32, 2, 4, 16>>(),
            size_of::<*mut u32>()
        );
    }

    #[test]
    fn row_band_helper_checks_the_complete_band() {
        // 3 x 8 matrix, full rows.
        assert_eq!(checked_row_band_start(0, 8, 8, 24), 0);
        assert_eq!(checked_row_band_start(2, 8, 8, 24), 16);
        assert_eq!(checked_row_band_start(3, 8, 8, 24), INVALID_LINEAR_2D);
        // A prefix of a row is fine; spilling past the row width is not.
        assert_eq!(checked_row_band_start(1, 8, 3, 24), 8);
        assert_eq!(checked_row_band_start(1, 8, 9, 24), INVALID_LINEAR_2D);
        // Degenerate shapes.
        assert_eq!(checked_row_band_start(0, 8, 0, 24), INVALID_LINEAR_2D);
        assert_eq!(checked_row_band_start(0, 0, 1, 24), INVALID_LINEAR_2D);
        // A short final row rejects the whole band.
        assert_eq!(checked_row_band_start(2, 8, 8, 23), INVALID_LINEAR_2D);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn row_band_helper_widens_before_multiplying() {
        // row * stride overflows u32 but not u64; the proof must survive.
        let len = (u32::MAX as usize) * 16 + 16;
        assert_eq!(
            checked_row_band_start(u32::MAX, 16, 16, len),
            u64::from(u32::MAX) * 16
        );
        assert_eq!(
            checked_row_band_start(u32::MAX, 16, 16, len - 1),
            INVALID_LINEAR_2D
        );
    }

    #[test]
    fn col_band_helper_checks_the_complete_band() {
        // 3 x 8 matrix, full columns.
        assert_eq!(checked_col_band_start(0, 8, 3, 24), 0);
        assert_eq!(checked_col_band_start(7, 8, 3, 24), 7);
        assert_eq!(checked_col_band_start(8, 8, 3, 24), INVALID_LINEAR_2D);
        assert_eq!(checked_col_band_start(0, 8, 4, 24), INVALID_LINEAR_2D);
        assert_eq!(checked_col_band_start(0, 8, 0, 24), INVALID_LINEAR_2D);
        assert_eq!(checked_col_band_start(0, 0, 1, 24), INVALID_LINEAR_2D);
        // The bottom element is the binding constraint.
        assert_eq!(checked_col_band_start(7, 8, 3, 23), INVALID_LINEAR_2D);
    }

    #[test]
    fn matrix_read_views_read_after_one_band_proof() {
        // 3 x 4 row-major matrix.
        let values: [u32; 12] = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let matrix = MatrixView32::new(&values, 4);

        let row = matrix.row(1, 4).unwrap();
        assert_eq!(row.len(), 4);
        assert_eq!(row.get(0), Some(10));
        assert_eq!(row.get(3), Some(13));
        assert_eq!(row.get(4), None);
        let mut row_iter = row.iter();
        assert_eq!(row_iter.next(), Some(10));
        assert_eq!(row_iter.next(), Some(11));
        assert_eq!(row_iter.next(), Some(12));
        assert_eq!(row_iter.next(), Some(13));
        assert_eq!(row_iter.next(), None);

        let col = matrix.col(2, 3).unwrap();
        assert_eq!(col.len(), 3);
        assert_eq!(col.get(0), Some(2));
        assert_eq!(col.get(2), Some(22));
        assert_eq!(col.get(3), None);
        let mut iter = col.iter();
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(12));
        assert_eq!(iter.next(), Some(22));
        assert_eq!(iter.next(), None);

        assert!(matrix.row(3, 4).is_none());
        assert!(matrix.row(0, 5).is_none());
        assert!(matrix.col(4, 3).is_none());
        assert!(matrix.col(0, 4).is_none());
    }

    #[test]
    fn zip_exact_pairs_row_and_column_after_one_length_check() {
        // A: 2 x 3 (stride 3), B: 3 x 2 (stride 2). Dot A row 1 with B col 0.
        let a: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a_row = MatrixView32::new(&a, 3).row(1, 3).unwrap();
        let b_col = MatrixView32::new(&b, 2).col(0, 3).unwrap();

        let mut sum = 0.0f32;
        for (x, y) in a_row.zip_exact(b_col).unwrap() {
            sum += x * y;
        }
        assert_eq!(sum, 4.0 * 10.0 + 5.0 * 30.0 + 6.0 * 50.0);

        let b_short = MatrixView32::new(&b, 2).col(0, 2).unwrap();
        assert!(a_row.zip_exact(b_short).is_none());
    }

    #[test]
    fn empty_read_views_always_decline() {
        let row = RowView32::<u32>::empty();
        assert!(row.is_empty());
        assert_eq!(row.get(0), None);
        assert_eq!(row.iter().next(), None);

        let col = ColView32::<u32>::empty();
        assert!(col.is_empty());
        assert_eq!(col.get(0), None);
        assert_eq!(col.iter().next(), None);
    }

    #[test]
    fn runtime_tile_helper_checks_the_complete_rectangle() {
        // 2 x 4 tiles in a matrix with runtime row width 16 (same shapes as the
        // static helper test above).
        assert_eq!(checked_runtime_tile_start::<u32, 2, 4>(1, 1, 16, 56), 36);
        assert_eq!(
            checked_runtime_tile_start::<u32, 2, 4>(1, 1, 16, 55),
            INVALID_LINEAR_2D
        );
        // A tile may not wrap into the next logical row.
        assert_eq!(checked_runtime_tile_start::<u32, 1, 2>(0, 31, 64, 64), 62);
        assert_eq!(
            checked_runtime_tile_start::<u32, 1, 2>(0, 32, 64, 128),
            INVALID_LINEAR_2D
        );
        // Degenerate row widths and zero-sized elements are rejected.
        assert_eq!(
            checked_runtime_tile_start::<u32, 1, 1>(0, 0, 0, 16),
            INVALID_LINEAR_2D
        );
        assert_eq!(
            checked_runtime_tile_start::<(), 1, 1>(0, 0, 16, 16),
            INVALID_LINEAR_2D
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn runtime_tile_helper_guards_the_u64_corner_product() {
        // last_row * stride would wrap u64 without the division guard.
        assert_eq!(
            checked_runtime_tile_start::<u32, { u32::MAX as usize }, 1>(
                u32::MAX,
                0,
                u32::MAX,
                usize::MAX,
            ),
            INVALID_LINEAR_2D
        );
        // The exact u32 linear boundary stays representable.
        assert_eq!(
            checked_runtime_tile_start::<u32, 1, 1>(u32::MAX, 0, 1, u32::MAX as usize + 1),
            u64::from(u32::MAX)
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn runtime_tile_wrapper_is_pointer_plus_row_width() {
        assert_eq!(
            size_of::<RuntimeTileMut32<'_, u32, 1, 1>>(),
            2 * size_of::<usize>()
        );
    }
}
