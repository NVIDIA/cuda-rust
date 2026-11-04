/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Borrowed tensor storage with composable, typed cyclic coordinate layouts.
//!
//! `local_partition::<P>` retains `Cyclic<Parent, P>` in the result type. Subranges
//! retain `Subrange<Parent>`; nested operations compose logical coordinates
//! before the original storage layout maps them to physical offsets. No
//! transformed pointer or erased multiplied stride defines partition semantics.
//!
//! Raw strided views may have uninitialized holes between logical elements.
//! Unlike a contiguous tensor slice, their storage proof covers only the
//! coordinates described by the layout; no reference to those holes is formed.
use crate::tensor::{RegisterTile, TensorElement};
use core::marker::PhantomData;
use cuda_device::DisjointSlice;

/// Sealed supertrait module preventing external crates from implementing [`IndexLayout`].
/// Ensures every [`IndexLayout`] upholds the originating storage bounds proof.
mod sealed {
    /// Marker trait implemented only by verified coordinate maps in this module.
    /// Guarantees that `crd2idx` only produces offsets covered by the parent view.
    pub trait Sealed {}
}

/// A library-defined logical coordinate map bound to a borrowed tensor.
///
/// The sealed implementations preserve the originating storage proof through
/// each transformation. An arbitrary external map cannot make safe reads reach
/// an unvalidated address.
pub trait IndexLayout: sealed::Sealed + Copy {
    /// Number of logical elements selected by the layout.
    fn len(&self) -> usize;
    /// Whether the layout selects no elements.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Map an in-domain logical coordinate to an originating storage offset
    /// (CuTeDSL `cute.crd2idx`). This computes an integer only; it does not
    /// read or write storage.
    fn crd2idx(&self, index: usize) -> usize;
}

/// One-dimensional shape and positive physical element stride.
#[derive(Clone, Copy)]
pub struct Strided1D {
    /// Number of logical coordinates (`0..len`) in this 1D view.
    /// Bounds all safe indexing and subview operations.
    len: usize,
    /// Physical storage elements skipped when the logical index increases by `1`.
    /// Equal to `1` for contiguous slices.
    stride: usize,
}
impl sealed::Sealed for Strided1D {}
impl IndexLayout for Strided1D {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn crd2idx(&self, index: usize) -> usize {
        index * self.stride
    }
}

/// A contiguous logical interval of a parent coordinate map.
#[derive(Clone, Copy)]
pub struct Subrange<Parent: IndexLayout> {
    /// Underlying coordinate map whose logical domain is being sliced.
    /// Translates `start + index` to a physical storage offset.
    parent: Parent,
    /// Starting logical coordinate in `parent` corresponding to index `0`.
    /// Added before delegating to `parent.crd2idx`.
    start: usize,
    /// Number of logical elements exposed by this subrange.
    /// Satisfies `start + len <= parent.len()`.
    len: usize,
}
impl<Parent: IndexLayout> sealed::Sealed for Subrange<Parent> {}
impl<Parent: IndexLayout> IndexLayout for Subrange<Parent> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn crd2idx(&self, index: usize) -> usize {
        self.parent.crd2idx(self.start + index)
    }
}

/// One of `PARTS` cyclic partitions of a parent coordinate map.
///
/// Logical `i` maps to parent coordinate `part + i * PARTS`. The partition
/// count remains in the type even after further slicing or partitioning.
#[derive(Clone, Copy)]
pub struct Cyclic<Parent: IndexLayout, const PARTS: usize> {
    /// Parent coordinate map being split across `PARTS` cyclic lanes.
    ///Evaluates `part + index * PARTS` to a physical storage offset.
    parent: Parent,
    /// Lane or partition index in `0..PARTS` owned by the caller.
    /// Selects the starting offset of the cyclic sequence.
    part: usize,
    /// Number of elements in this partition (`0` when `part >= parent.len()`).
    /// Handles ragged tails without reading past the parent view.
    len: usize,
}
impl<Parent: IndexLayout, const PARTS: usize> sealed::Sealed for Cyclic<Parent, PARTS> {}
impl<Parent: IndexLayout, const PARTS: usize> IndexLayout for Cyclic<Parent, PARTS> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn crd2idx(&self, index: usize) -> usize {
        self.parent.crd2idx(self.part + index * PARTS)
    }
}

/// Construct cyclic partition `part` (`0..PARTS`) of `layout` and compute its exact element count.
/// Ragged tails give higher `part` indices one fewer element instead of reading out of bounds.
#[inline(always)]
fn partition<Layout: IndexLayout, const PARTS: usize>(
    layout: Layout,
    part: usize,
) -> Cyclic<Layout, PARTS> {
    const {
        assert!(PARTS > 0);
    }
    assert!(part < PARTS);
    let len = if part >= layout.len() {
        0
    } else {
        1 + (layout.len() - 1 - part) / PARTS
    };
    Cyclic {
        parent: layout,
        part,
        len,
    }
}

/// Read-only borrowed storage and its typed logical coordinate map.
///
/// `Element: TensorElement<Storage>` preserves the same logical/storage identity
/// contract as `Tensor`. Scalar access returns the physical storage value;
/// numerical conversion remains explicit.
pub struct StridedTensor<'a, Element, Layout = Strided1D, Storage = Element>
where
    Element: TensorElement<Storage>,
    Layout: IndexLayout,
    Storage: Copy,
{
    /// Non-owning read-only base pointer to the underlying storage buffer.
    /// Indexed only at offsets produced by `layout.crd2idx`.
    base: *const Storage,
    /// Composable logical-to-physical coordinate map for this tensor view.
    /// Tracks length, stride, subranges, and cyclic thread partitions.
    layout: Layout,
    /// Zero-sized lifetime and logical-element marker tying the view to `'a`.
    /// Prevents the borrowed storage from being mutated or dropped while live.
    borrow: PhantomData<(&'a [Storage], Element)>,
}
impl<Element: TensorElement<Storage>, Layout: IndexLayout, Storage: Copy> Copy
    for StridedTensor<'_, Element, Layout, Storage>
{
}
impl<Element: TensorElement<Storage>, Layout: IndexLayout, Storage: Copy> Clone
    for StridedTensor<'_, Element, Layout, Storage>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: Copy + TensorElement<T>> StridedTensor<'a, T> {
    /// Borrow an ordinary slice with its unchanged scalar element identity.
    #[inline(always)]
    pub fn from_slice(storage: &'a [T]) -> Self {
        let _ = crate::tensor::Tensor::from_slice(storage)
            .local_tile::<1>(0)
            .base();
        unsafe { Self::from_raw_parts(storage.as_ptr(), storage.len(), 1) }
    }
    /// Reinterpret a borrowed slice `&[S]` as a contiguous `StridedTensor<'a, T>` of
    /// `(storage.len() * size_of::<S>()) / size_of::<T>()` elements (for example, viewing
    /// `&[u16]` or `&[BFloat16]` as `StridedTensor<u128>` for vectorized 128-bit loads).
    ///
    /// # Safety
    /// `storage.as_ptr()` must be aligned to `align_of::<T>()` and the total byte length
    /// must be an exact multiple of `size_of::<T>()`.
    #[allow(clippy::manual_slice_size_calculation)]
    #[inline(always)]
    pub unsafe fn from_slice_as<S: Copy>(storage: &'a [S]) -> Self {
        let len = (storage.len() * core::mem::size_of::<S>()) / core::mem::size_of::<T>();
        unsafe { Self::from_raw_parts(storage.as_ptr().cast::<T>(), len, 1) }
    }
    /// Borrow the logical elements of a strided storage range.
    ///
    /// # Safety
    /// `stride` is positive. Every `base.add(i * stride)`, `i < len`, must be
    /// live, aligned and initialized for `'a`, with representable offsets and
    /// no conflicting mutation. Holes need not be initialized. Empty views
    /// may use a dangling aligned pointer and never perform pointer arithmetic.
    #[inline(always)]
    pub const unsafe fn from_raw_parts(base: *const T, len: usize, stride: usize) -> Self {
        Self {
            base,
            layout: Strided1D { len, stride },
            borrow: PhantomData,
        }
    }

    /// Borrow a contiguous `len`-element read-only shared-memory view starting at `smem_base + byte_offset`.
    ///
    /// # Safety
    /// `smem_base.add(byte_offset)` must be aligned to `align_of::<T>()` and cover `len` initialized `T` elements.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_shared_bytes(
        smem_base: *const u8,
        byte_offset: usize,
        len: usize,
    ) -> Self {
        unsafe { Self::from_raw_parts(smem_base.add(byte_offset).cast::<T>(), len, 1) }
    }
}
impl<'a, Element: TensorElement<Storage>, Storage: Copy>
    StridedTensor<'a, Element, Strided1D, Storage>
{
    /// Borrow contiguous encoded storage without changing its logical identity.
    #[inline(always)]
    pub fn from_storage(storage: &'a [Storage]) -> Self {
        Self {
            base: storage.as_ptr(),
            layout: Strided1D {
                len: storage.len(),
                stride: 1,
            },
            borrow: PhantomData,
        }
    }
}
impl<'a, T: Copy + TensorElement<T>> StridedTensor<'a, T, Strided1D, T> {
    /// Advance the base pointer by `start * stride` elements and bound the view to `len` elements.
    ///
    /// Preserves the flat `Strided1D` layout so loop-invariant or induction-step
    /// slices become pointer induction variables in LLVM/MLIR.
    ///
    /// # Safety
    /// `start <= self.len()` and `len <= self.len() - start` must hold.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn offset_unchecked(self, start: usize, len: usize) -> Self {
        Self {
            base: unsafe { self.base.add(start * self.layout.stride) },
            layout: Strided1D {
                len,
                stride: self.layout.stride,
            },
            borrow: PhantomData,
        }
    }

    /// Read a logical coordinate from a 1D strided view without a nested trait call.
    ///
    /// # Safety
    /// `index < self.len()` must hold.
    #[inline(always)]
    pub unsafe fn at_unchecked(self, index: usize) -> T {
        unsafe { self.base.add(index * self.layout.stride).read() }
    }
}

impl<'a> StridedTensor<'a, u128, Strided1D, u128> {
    /// Load one 16-byte (`128`-bit) element at `index` as a 4-lane [`super::RegisterTile<f32, 4>`].
    ///
    /// # Safety
    /// `index < self.len()` must hold and the underlying pointer must be 16-byte aligned.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_f32x4(&self, index: usize) -> super::RegisterTile<f32, 4> {
        let q = unsafe { self.get_unchecked(index) };
        super::RegisterTile::from_array([
            f32::from_bits((q & 0xFFFF_FFFF) as u32),
            f32::from_bits(((q >> 32) & 0xFFFF_FFFF) as u32),
            f32::from_bits(((q >> 64) & 0xFFFF_FFFF) as u32),
            f32::from_bits((q >> 96) as u32),
        ])
    }
}

impl<'a, T: Copy + TensorElement<T>> StridedTensorMut<'a, T, Strided1D, T> {
    /// Advance the mutable base pointer by `start * stride` elements and bound the view to `len` elements.
    ///
    /// # Safety
    /// `start <= self.len()` and `len <= self.len() - start` must hold.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn offset_unchecked(self, start: usize, len: usize) -> Self {
        Self {
            base: unsafe { self.base.add(start * self.layout.stride) },
            layout: Strided1D {
                len,
                stride: self.layout.stride,
            },
            borrow: PhantomData,
        }
    }

    /// Store one logical coordinate into a 1D strided view without a nested trait call.
    ///
    /// # Safety
    /// `index < self.len()` and no conflicting access may occur.
    #[inline(always)]
    pub unsafe fn write_at(&self, index: usize, value: T) {
        unsafe { self.base.add(index * self.layout.stride).write(value) }
    }
}

impl<'a> StridedTensorMut<'a, crate::numeric::BFloat16, Strided1D, crate::numeric::BFloat16> {
    /// Atomically add a packed `(lo, hi)` `BFloat16` pair into `(row, col)` of a 2D row-major matrix.
    ///
    /// # Safety
    /// `row * row_stride + col + 1 < self.len()` must hold and the target address must be 4-byte aligned.
    #[inline(always)]
    pub unsafe fn atomic_add_bf16x2_at_2d(
        &self,
        row: usize,
        col: usize,
        row_stride: usize,
        lo: crate::numeric::BFloat16,
        hi: crate::numeric::BFloat16,
    ) {
        let ptr = unsafe { self.ptr_at_2d_unchecked(row, col, row_stride) };
        unsafe { crate::arch::atomic_add_bf16x2(ptr, lo, hi) };
    }
}

impl<'a, Element: TensorElement<Storage>, Layout: IndexLayout, Storage: Copy>
    StridedTensor<'a, Element, Layout, Storage>
{
    /// Number of logical storage elements selected by this view.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.layout.len()
    }
    /// Whether this view contains no logical elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }
    /// Retain the coordinate map for composition or inspection.
    #[inline(always)]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }
    /// Select a checked interval of logical coordinates, retaining its parent map.
    #[inline(always)]
    pub fn subview(
        self,
        start: usize,
        len: usize,
    ) -> StridedTensor<'a, Element, Subrange<Layout>, Storage> {
        assert!(start <= self.len() && len <= self.len() - start);
        unsafe { self.subview_unchecked(start, len) }
    }
    /// Select a logical interval after an enclosing extent check.
    /// # Safety
    /// `start <= self.len()` and `len <= self.len() - start` must hold.
    #[inline(always)]
    pub unsafe fn subview_unchecked(
        self,
        start: usize,
        len: usize,
    ) -> StridedTensor<'a, Element, Subrange<Layout>, Storage> {
        StridedTensor {
            base: self.base,
            layout: Subrange {
                parent: self.layout,
                start,
                len,
            },
            borrow: PhantomData,
        }
    }
    /// Select partition `index` of a `PARTS`-way cyclic split, preserving the
    /// count in the returned type.
    ///
    /// Logical coordinate `i` of the result is parent coordinate
    /// `index + i * PARTS`. This is CuTe's `local_partition` with a `PARTS:1`
    /// thread layout: the thread count is the const generic and the thread
    /// index is the argument. A ragged tail leaves the later partitions
    /// shorter instead of reading past the parent.
    #[inline(always)]
    pub fn local_partition<const PARTS: usize>(
        self,
        index: usize,
    ) -> StridedTensor<'a, Element, Cyclic<Layout, PARTS>, Storage> {
        StridedTensor {
            base: self.base,
            layout: partition::<Layout, PARTS>(self.layout, index),
            borrow: PhantomData,
        }
    }
    /// Read a checked logical coordinate through its composed layout.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Storage {
        assert!(index < self.len());
        unsafe { self.get_unchecked(index) }
    }
    /// Read a logical coordinate after an enclosing bounds check.
    /// # Safety
    /// `index < self.len()` must hold.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> Storage {
        unsafe { self.base.add(self.layout.crd2idx(index)).read() }
    }
    /// Return a const pointer to the storage element at logical coordinate `index`
    /// using the view's `crd2idx` coordinate map.
    /// # Safety
    /// `index < self.len()` and the addressed storage must be live for `'a`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn ptr_at_unchecked(&self, index: usize) -> *const Storage {
        unsafe { self.base.add(self.layout.crd2idx(index)) }
    }
    /// Reduce logical coordinates in increasing order without reassociation.
    #[inline(always)]
    pub fn fold<U>(self, initial: U, mut f: impl FnMut(U, Storage) -> U) -> U {
        let mut result = initial;
        let mut index = 0;
        while index < self.len() {
            result = f(result, unsafe { self.get_unchecked(index) });
            index += 1;
        }
        result
    }
    /// Gather the first `N` logical values into an ordered register tile.
    #[inline(always)]
    pub fn load<const N: usize>(self) -> RegisterTile<Storage, N> {
        assert!(N <= self.len());
        unsafe { self.load_unchecked() }
    }
    /// Gather an ordered register tile after an enclosing extent check.
    /// # Safety
    /// `N <= self.len()` must hold. Vector alignment is not required.
    #[inline(always)]
    pub unsafe fn load_unchecked<const N: usize>(self) -> RegisterTile<Storage, N> {
        RegisterTile::from_fn(|i| unsafe { self.get_unchecked(i) })
    }
    /// Gather `N` values spaced by `STRIDE` starting at `lane` (`lane + i * STRIDE`) without bounds checks.
    ///
    /// # Safety
    /// Every `lane + i * STRIDE` for `i < N` must be `< self.len()`.
    #[inline(always)]
    pub unsafe fn load_strided_unchecked<const STRIDE: usize, const N: usize>(
        self,
        lane: usize,
    ) -> RegisterTile<Storage, N> {
        RegisterTile::from_fn(|i| unsafe { self.get_unchecked(lane + i * STRIDE) })
    }
    /// Transform equally sized logical views, even when their layouts differ.
    /// # Safety
    /// Destination coordinates must have one writer and no conflicting access.
    /// Destination writes cannot change input values read by another iteration
    /// or thread. Both views must have valid storage throughout the operation.
    #[inline(always)]
    pub unsafe fn map_to<E: TensorElement<S>, L: IndexLayout, S: Copy>(
        self,
        output: &mut StridedTensorMut<'_, E, L, S>,
        mut f: impl FnMut(Storage) -> S,
    ) {
        assert!(self.len() == output.len());
        let mut i = 0;
        while i < self.len() {
            unsafe {
                output.store_at(i, f(self.get_unchecked(i)));
            }
            i += 1;
        }
    }
    /// Transform corresponding logical coordinates, preserving evaluation order.
    /// # Safety
    /// The destination has `map_to`'s non-overlap requirements for both inputs.
    /// All views must have equal logical lengths.
    #[inline(always)]
    pub unsafe fn zip_map_to<
        E: TensorElement<S>,
        L: IndexLayout,
        S: Copy,
        OE: TensorElement<OS>,
        OL: IndexLayout,
        OS: Copy,
    >(
        self,
        other: StridedTensor<'_, E, L, S>,
        output: &mut StridedTensorMut<'_, OE, OL, OS>,
        mut f: impl FnMut(Storage, S) -> OS,
    ) {
        assert!(self.len() == other.len() && self.len() == output.len());
        let mut i = 0;
        while i < self.len() {
            unsafe {
                output.store_at(i, f(self.get_unchecked(i), other.get_unchecked(i)));
            }
            i += 1;
        }
    }
}

/// A uniquely owned writable view and its typed logical coordinate map.
///
/// This type is neither Copy nor Clone. GPU stores still require unsafe because
/// Rust cannot prove that different hardware threads chose disjoint coordinates.
pub struct StridedTensorMut<'a, Element, Layout = Strided1D, Storage = Element>
where
    Element: TensorElement<Storage>,
    Layout: IndexLayout,
    Storage: Copy,
{
    /// Non-owning mutable base pointer to the underlying storage buffer.
    /// Indexed only at offsets produced by `layout.crd2idx`.
    base: *mut Storage,
    /// Composable logical-to-physical coordinate map for this writable view.
    /// Tracks length, stride, subranges, and cyclic thread partitions.
    layout: Layout,
    /// Zero-sized exclusive borrow and logical-element marker for `'a`.
    /// Prevents aliasing mutable views in safe Rust code.
    borrow: PhantomData<(&'a mut [Storage], Element)>,
}
impl<'a, T: Copy + TensorElement<T>> StridedTensorMut<'a, T> {
    /// Borrow a mutable slice without changing its scalar element identity.
    #[inline(always)]
    pub fn from_mut_slice(storage: &'a mut [T]) -> Self {
        unsafe { Self::from_raw_parts(storage.as_mut_ptr(), storage.len(), 1) }
    }
    /// Borrow writable device storage while retaining its original alias rules.
    #[inline(always)]
    pub fn from_disjoint_slice(storage: &'a mut DisjointSlice<'_, T>) -> Self {
        let ptr = storage.as_mut_ptr();
        let len = storage.len();
        let view = unsafe { core::slice::from_raw_parts(ptr.cast_const(), len) };
        let _ = crate::tensor::Tensor::from_slice(view)
            .local_tile::<1>(0)
            .base();
        unsafe { Self::from_raw_parts(ptr, len, 1) }
    }
    /// Reinterpret a writable `DisjointSlice<'_, S>` as a contiguous `StridedTensorMut<'a, T>` of
    /// `(storage.len() * size_of::<S>()) / size_of::<T>()` elements (for example, viewing
    /// `DisjointSlice<u8>` as `StridedTensorMut<u64>` or `DisjointSlice<BFloat16>` as `StridedTensorMut<u128>`).
    ///
    /// # Safety
    /// `storage.as_mut_ptr()` must be aligned to `align_of::<T>()` and the total byte length
    /// must be an exact multiple of `size_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_disjoint_slice_as<S: Copy>(storage: &'a mut DisjointSlice<'_, S>) -> Self {
        let len = (storage.len() * core::mem::size_of::<S>()) / core::mem::size_of::<T>();
        unsafe { Self::from_raw_parts(storage.as_mut_ptr().cast::<T>(), len, 1) }
    }
    /// Borrow writable strided storage with an externally established extent.
    /// # Safety
    /// `stride` is positive. Each `base.add(i*stride)`, `i < len`, must be live,
    /// aligned and writable for `'a`, with representable pointer offsets.
    /// Stores separately require disjoint active access; reads additionally
    /// require initialization. Holes need not contain initialized values.
    #[inline(always)]
    pub const unsafe fn from_raw_parts(base: *mut T, len: usize, stride: usize) -> Self {
        Self {
            base,
            layout: Strided1D { len, stride },
            borrow: PhantomData,
        }
    }

    /// Borrow a contiguous `len`-element writable shared-memory view starting at `smem_base + byte_offset`.
    ///
    /// # Safety
    /// `smem_base.add(byte_offset)` must be aligned to `align_of::<T>()` and cover `len` writable `T` elements.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_shared_bytes(
        smem_base: *mut u8,
        byte_offset: usize,
        len: usize,
    ) -> Self {
        unsafe { Self::from_raw_parts(smem_base.add(byte_offset).cast::<T>(), len, 1) }
    }
}
impl<'a, Element: TensorElement<Storage>, Layout: IndexLayout, Storage: Copy>
    StridedTensorMut<'a, Element, Layout, Storage>
{
    /// Number of selected logical elements.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.layout.len()
    }
    /// Whether this writable view selects no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }
    /// Inspect the retained coordinate map without granting storage access.
    #[inline(always)]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }
    /// Select a checked logical interval, consuming the original writable view.
    #[inline(always)]
    pub fn subview(
        self,
        start: usize,
        len: usize,
    ) -> StridedTensorMut<'a, Element, Subrange<Layout>, Storage> {
        assert!(start <= self.len() && len <= self.len() - start);
        unsafe { self.subview_unchecked(start, len) }
    }
    /// Reborrow the same typed view; the parent cannot be used during this borrow.
    #[inline(always)]
    pub fn reborrow(&mut self) -> StridedTensorMut<'_, Element, Layout, Storage> {
        StridedTensorMut {
            base: self.base,
            layout: self.layout,
            borrow: PhantomData,
        }
    }
    /// Select a logical interval after an enclosing extent check.
    /// # Safety
    /// `start <= self.len()` and `len <= self.len() - start` must hold.
    #[inline(always)]
    pub unsafe fn subview_unchecked(
        self,
        start: usize,
        len: usize,
    ) -> StridedTensorMut<'a, Element, Subrange<Layout>, Storage> {
        StridedTensorMut {
            base: self.base,
            layout: Subrange {
                parent: self.layout,
                start,
                len,
            },
            borrow: PhantomData,
        }
    }
    /// Select partition `index` of a `PARTS`-way cyclic split, consuming the
    /// original writable view.
    ///
    /// Logical coordinate `i` of the result is parent coordinate
    /// `index + i * PARTS`, as CuTe's `local_partition` maps it with a
    /// `PARTS:1` thread layout. A ragged tail leaves the later partitions
    /// shorter instead of reaching past the parent.
    #[inline(always)]
    pub fn local_partition<const PARTS: usize>(
        self,
        index: usize,
    ) -> StridedTensorMut<'a, Element, Cyclic<Layout, PARTS>, Storage> {
        StridedTensorMut {
            base: self.base,
            layout: partition::<Layout, PARTS>(self.layout, index),
            borrow: PhantomData,
        }
    }
    /// Return a mutable pointer to the storage element at logical coordinate `index`
    /// using the view's `crd2idx` coordinate map.
    ///
    /// # Safety
    /// `index < self.len()` and the caller must uphold exclusive access for the
    /// addressed storage slice.
    #[must_use]
    #[inline(always)]
    pub unsafe fn ptr_at_unchecked(&self, index: usize) -> *mut Storage {
        unsafe { self.base.add(self.layout.crd2idx(index)) }
    }

    /// Return a mutable pointer to the storage element at 2D row-major coordinate
    /// `(row, col)` (with `cols` columns per row), composed with this view's `crd2idx`.
    ///
    /// # Safety
    /// `row * cols + col < self.len()` and the caller must uphold exclusive access
    /// for the addressed storage slice.
    #[must_use]
    #[inline(always)]
    pub unsafe fn ptr_at_2d_unchecked(&self, row: usize, col: usize, cols: usize) -> *mut Storage {
        let flat = cute_layout::crd2idx_row_major_2d(row, col, cols);
        unsafe { self.ptr_at_unchecked(flat) }
    }

    /// Store one logical coordinate using the composed coordinate map.
    /// # Safety
    /// `index < self.len()` and no conflicting access to this coordinate may
    /// occur, including access from another GPU thread.
    #[inline(always)]
    pub unsafe fn store_at(&mut self, index: usize, value: Storage) {
        unsafe {
            self.base.add(self.layout.crd2idx(index)).write(value);
        }
    }
    /// Read, transform and replace one initialized logical coordinate.
    /// # Safety
    /// `index < self.len()`, the element is initialized, and no conflicting
    /// active read or write may occur. This is not an atomic operation.
    #[inline(always)]
    pub unsafe fn update_at(&mut self, index: usize, f: impl FnOnce(Storage) -> Storage) {
        let pointer = unsafe { self.base.add(self.layout.crd2idx(index)) };
        unsafe {
            pointer.write(f(pointer.read()));
        }
    }
    /// Scatter an ordered register tile through this view's coordinate map.
    /// # Safety
    /// No other active access may conflict with the selected destinations.
    /// This method checks that `N` logical coordinates are present.
    #[inline(always)]
    pub unsafe fn store<const N: usize>(&mut self, values: RegisterTile<Storage, N>) {
        assert!(N <= self.len());
        unsafe { self.store_unchecked(values) };
    }

    /// Scatter an ordered register tile after an enclosing extent check.
    ///
    /// # Safety
    /// `N <= self.len()` must hold, and no other active access may conflict
    /// with the selected destinations.
    #[inline(always)]
    pub unsafe fn store_unchecked<const N: usize>(&mut self, values: RegisterTile<Storage, N>) {
        let values = values.into_array();
        let mut i = 0;
        while i < N {
            cuda_device::thread::__unroll_config::<0>();
            unsafe {
                self.store_at(i, values[i]);
            }
            i += 1;
        }
    }

    /// Scatter `N` values spaced by `STRIDE` starting at `lane` (`lane + i * STRIDE`) without bounds checks.
    ///
    /// # Safety
    /// Every `lane + i * STRIDE` for `i < N` must be `< self.len()` with no conflicting access.
    #[inline(always)]
    pub unsafe fn store_strided_unchecked<const STRIDE: usize, const N: usize>(
        &mut self,
        lane: usize,
        values: RegisterTile<Storage, N>,
    ) {
        let values = values.into_array();
        let mut i = 0;
        while i < N {
            cuda_device::thread::__unroll_config::<0>();
            unsafe {
                self.store_at(lane + i * STRIDE, values[i]);
            }
            i += 1;
        }
    }

    /// Store an aligned `N`-element [`GlobalStoreTile`](crate::atom::vector::GlobalStoreTile) at
    /// logical coordinate `index` without allocating in L1 cache.
    ///
    /// # Safety
    /// `self.layout.crd2idx(index)` must start a contiguous, aligned `N`-element
    /// range within the underlying global allocation with no conflicting access.
    #[inline(always)]
    pub unsafe fn store_global_no_allocate_at<const N: usize>(
        &mut self,
        index: usize,
        values: &crate::atom::vector::GlobalStoreTile<Storage, N>,
    ) where
        Storage: crate::atom::vector::GlobalStoreElement,
    {
        unsafe {
            crate::atom::vector::store_global_no_allocate(
                self.base.add(self.layout.crd2idx(index)),
                values,
            );
        }
    }

    /// Store packed `u32` words (such as `f16x2` pairs) at logical coordinate `index`
    /// without allocating in L1 cache.
    ///
    /// # Safety
    /// `self.layout.crd2idx(index)` must be `32`-byte aligned and cover `N`
    /// `u32` words (`2 * N` `f16` elements) within the underlying global allocation
    /// with no conflicting access.
    #[inline(always)]
    pub unsafe fn store_global_no_allocate_words_at<const N: usize>(
        &mut self,
        index: usize,
        values: &crate::atom::vector::GlobalStoreTile<u32, N>,
    ) {
        unsafe {
            crate::atom::vector::store_global_no_allocate(
                self.base.add(self.layout.crd2idx(index)).cast::<u32>(),
                values,
            );
        }
    }
}
