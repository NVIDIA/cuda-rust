/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Borrowed tensor storage with composable, typed cyclic coordinate layouts.
//!
//! `partition::<P>` retains `Cyclic<Parent, P>` in the result type. Subranges
//! retain `Subrange<Parent>`; nested operations compose logical coordinates
//! before the original storage layout maps them to physical offsets. No
//! transformed pointer or erased multiplied stride defines partition semantics.
//!
//! Raw strided views may have uninitialized holes between logical elements.
//! Unlike a contiguous tensor slice, their storage proof covers only the
//! coordinates described by the layout; no reference to those holes is formed.
use crate::tensor::{RegisterTile, TensorElement};
use core::marker::PhantomData;
use ctm_rs::memory::DisjointSlice;

mod sealed {
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
    /// Map an in-domain logical coordinate to an originating storage offset.
    /// This computes an integer only; it does not read or write storage.
    fn offset(&self, index: usize) -> usize;
}

/// One-dimensional shape and positive physical element stride.
#[derive(Clone, Copy)]
pub struct Strided1D {
    len: usize,
    stride: usize,
}
impl sealed::Sealed for Strided1D {}
impl IndexLayout for Strided1D {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn offset(&self, index: usize) -> usize {
        index * self.stride
    }
}

/// A contiguous logical interval of a parent coordinate map.
#[derive(Clone, Copy)]
pub struct Subrange<Parent: IndexLayout> {
    parent: Parent,
    start: usize,
    len: usize,
}
impl<Parent: IndexLayout> sealed::Sealed for Subrange<Parent> {}
impl<Parent: IndexLayout> IndexLayout for Subrange<Parent> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn offset(&self, index: usize) -> usize {
        self.parent.offset(self.start + index)
    }
}

/// One of `PARTS` cyclic partitions of a parent coordinate map.
///
/// Logical `i` maps to parent coordinate `part + i * PARTS`. The partition
/// count remains in the type even after further slicing or partitioning.
#[derive(Clone, Copy)]
pub struct Cyclic<Parent: IndexLayout, const PARTS: usize> {
    parent: Parent,
    part: usize,
    len: usize,
}
impl<Parent: IndexLayout, const PARTS: usize> sealed::Sealed for Cyclic<Parent, PARTS> {}
impl<Parent: IndexLayout, const PARTS: usize> IndexLayout for Cyclic<Parent, PARTS> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    fn offset(&self, index: usize) -> usize {
        self.parent.offset(self.part + index * PARTS)
    }
}

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
    base: *const Storage,
    layout: Layout,
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
        unsafe { Self::from_raw_parts(storage.as_ptr(), storage.len(), 1) }
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
    /// Select a cyclic partition, preserving its count in the returned type.
    #[inline(always)]
    pub fn partition<const PARTS: usize>(
        self,
        part: usize,
    ) -> StridedTensor<'a, Element, Cyclic<Layout, PARTS>, Storage> {
        StridedTensor {
            base: self.base,
            layout: partition::<Layout, PARTS>(self.layout, part),
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
        unsafe { self.base.add(self.layout.offset(index)).read() }
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
    base: *mut Storage,
    layout: Layout,
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
        unsafe { Self::from_raw_parts(storage.as_mut_ptr(), storage.len(), 1) }
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
    /// Select one typed cyclic partition, consuming the original writable view.
    #[inline(always)]
    pub fn partition<const PARTS: usize>(
        self,
        part: usize,
    ) -> StridedTensorMut<'a, Element, Cyclic<Layout, PARTS>, Storage> {
        StridedTensorMut {
            base: self.base,
            layout: partition::<Layout, PARTS>(self.layout, part),
            borrow: PhantomData,
        }
    }
    /// Store one logical coordinate using the composed coordinate map.
    /// # Safety
    /// `index < self.len()` and no conflicting access to this coordinate may
    /// occur, including access from another GPU thread.
    #[inline(always)]
    pub unsafe fn store_at(&mut self, index: usize, value: Storage) {
        unsafe {
            self.base.add(self.layout.offset(index)).write(value);
        }
    }
    /// Read, transform and replace one initialized logical coordinate.
    /// # Safety
    /// `index < self.len()`, the element is initialized, and no conflicting
    /// active read or write may occur. This is not an atomic operation.
    #[inline(always)]
    pub unsafe fn update_at(&mut self, index: usize, f: impl FnOnce(Storage) -> Storage) {
        let pointer = unsafe { self.base.add(self.layout.offset(index)) };
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
        let values = values.into_array();
        let mut i = 0;
        while i < N {
            ctm_rs::thread::__unroll_config::<0>();
            unsafe {
                self.store_at(i, values[i]);
            }
            i += 1;
        }
    }
}
