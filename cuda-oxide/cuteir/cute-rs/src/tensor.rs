/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed views of existing tensor memory.
//!
//! A [`Tensor`] combines three things:
//!
//! ```text
//! storage: where the bits are
//! layout:  how coordinates find those bits
//! element: what the bits mean
//!              │
//!              ▼
//!            Tensor
//! ```
//!
//! A tensor view does not allocate, copy, or load memory. Operations such as
//! [`Tensor::zipped_divide`] and [`Tensor::slice`] only change how the same
//! storage is viewed. [`Tensor::load`] is the first operation that reads it.

use core::marker::PhantomData;
use core::ops::Add;

use cuda_device::DisjointSlice;

use crate::{assume_div, load_tile, store_tile};

/// Says which Rust storage type may hold one or more logical elements.
///
/// For example, `E2M1: TensorElement<u8>` means an `u8` may carry two packed
/// FP4 values. A missing implementation rejects an invalid pair at compile
/// time, such as E2M1 data stored in `f32`.
pub trait TensorElement<Storage: Copy> {}

// Plain f16 and f32 values use the same type for meaning and storage.
impl TensorElement<f16> for f16 {}
impl TensorElement<f32> for f32 {}

/// One line of adjacent elements.
///
/// ```text
/// index:   0   1   2   3
/// memory: [a] [b] [c] [d]
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contiguous1D {
    len: usize,
}

impl Contiguous1D {
    /// Create a layout containing `len` adjacent elements.
    #[must_use]
    #[inline(always)]
    pub const fn new(len: usize) -> Self {
        Self { len }
    }

    /// Return the number of elements.
    #[must_use]
    #[inline(always)]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return `true` when the length is zero.
    #[must_use]
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

/// A one-dimensional tensor split into fixed-size tiles.
///
/// For `TILE = 4`:
///
/// ```text
/// original: [0 1 2 3 | 4 5 6 7 | 8 9]
/// tile:          0          1        2
/// inside:    0 1 2 3    0 1 2 3    0 1
/// ```
///
/// The final tile may be shorter than `TILE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zipped1D<const TILE: usize> {
    len: usize,
}

impl<const TILE: usize> Zipped1D<TILE> {
    /// Return the number of tiles, including a short final tile.
    #[must_use]
    #[inline(always)]
    pub const fn tile_count(self) -> usize {
        const { assert!(TILE > 0) };
        self.len.div_ceil(TILE)
    }
}

/// One selected tile from a [`Zipped1D`] view.
///
/// It stores the tile's first index and the full tensor length. It does not
/// contain or load the tile values.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile1D<const TILE: usize> {
    base: usize,
    len: usize,
}

impl<const TILE: usize> Tile1D<TILE> {
    /// Return the tile's first index in the original tensor.
    #[must_use]
    #[inline(always)]
    pub const fn base(self) -> usize {
        self.base
    }

    /// Return `true` when all `TILE` positions are in bounds.
    #[must_use]
    #[inline(always)]
    pub const fn is_full(self) -> bool {
        self.valid_len() == TILE
    }

    /// Return the number of positions that are in bounds.
    #[must_use]
    #[inline(always)]
    pub const fn valid_len(self) -> usize {
        if self.base >= self.len {
            0
        } else {
            let remaining = self.len - self.base;
            if remaining < TILE { remaining } else { TILE }
        }
    }
}

/// The `N` values from one tile, held in thread-local registers.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegisterTile<T: Copy, const N: usize>([T; N]);

impl<T: Copy, const N: usize> RegisterTile<T, N> {
    /// Borrow the values as a fixed-size array.
    #[must_use]
    #[inline(always)]
    pub const fn as_array(&self) -> &[T; N] {
        &self.0
    }

    /// Consume the tile and return its fixed-size array.
    #[must_use]
    #[inline(always)]
    pub fn into_array(self) -> [T; N] {
        self.0
    }
}

impl<T: Copy + Add<Output = T>, const N: usize> Add for RegisterTile<T, N> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self::Output {
        let mut i = 0;
        while i < N {
            self.0[i] = self.0[i] + rhs.0[i];
            i += 1;
        }
        self
    }
}

/// A read-only view of storage with a typed element format and layout.
///
/// The view owns no memory and performs no allocation:
///
/// ```text
/// Tensor<'a, Element, Layout, Storage>
///             │        │        │
///             │        │        └─ Rust type holding the bits
///             │        └────────── coordinate-to-offset rule
///             └─────────────────── meaning of the bits
/// ```
///
/// For ordinary values, `Storage` defaults to `Element`, so
/// `Tensor<f32, Layout>` is enough. Packed data names both types; for example,
/// `Tensor<E2M1, Layout, u8>` means two logical FP4 values share each byte.
/// [`TensorElement`] checks that the pair is allowed.
///
/// Copying this value copies only the view. The underlying storage is still
/// borrowed. This view is used inside Rust kernel code; it is not passed
/// directly as a raw kernel argument.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a, Element, Layout, Storage = Element>
where
    Element: TensorElement<Storage>,
    Storage: Copy,
{
    pub(crate) storage: &'a [Storage],
    pub(crate) layout: Layout,
    element: PhantomData<Element>,
}

impl<'a, Element: TensorElement<Storage>, Layout, Storage: Copy>
    Tensor<'a, Element, Layout, Storage>
{
    /// View `storage` through `layout` without copying or loading it.
    #[must_use]
    #[inline(always)]
    pub const fn from_storage(storage: &'a [Storage], layout: Layout) -> Self {
        Self {
            storage,
            layout,
            element: PhantomData,
        }
    }

    /// Return the slice that holds the physical bits.
    #[must_use]
    #[inline(always)]
    pub const fn storage(&self) -> &'a [Storage] {
        self.storage
    }

    /// Return the layout value stored by this view.
    #[must_use]
    #[inline(always)]
    pub const fn layout(&self) -> Layout
    where
        Layout: Copy,
    {
        self.layout
    }
}

impl<'a, T: Copy + TensorElement<T>> Tensor<'a, T, Contiguous1D, T> {
    /// View a slice as one contiguous row.
    #[must_use]
    #[inline(always)]
    pub const fn from_slice(storage: &'a [T]) -> Self {
        make_tensor_read(storage)
    }

    /// Split the row into `(position_in_tile, tile_number)` coordinates.
    ///
    /// ```text
    /// [0 1 2 3 | 4 5 6 7]  ── zipped_divide::<4>() ──► 2 tiles of 4
    /// ```
    ///
    /// This only changes the view. It does not read memory.
    #[must_use]
    #[inline(always)]
    pub const fn zipped_divide<const TILE: usize>(self) -> Tensor<'a, T, Zipped1D<TILE>, T> {
        zipped_divide_read::<T, TILE>(self)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> Tensor<'a, T, Zipped1D<TILE>, T> {
    /// Select tile number `tile`.
    ///
    /// This only computes the tile's starting index. Call [`Tensor::load`] to
    /// read its values.
    #[must_use]
    #[inline(always)]
    pub const fn slice(self, tile: usize) -> Tensor<'a, T, Tile1D<TILE>, T> {
        slice_read::<T, TILE>(self, tile)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> Tensor<'a, T, Tile1D<TILE>, T> {
    /// Return `true` when every position in this tile is in bounds.
    ///
    /// This is the tensor-level form of [`Tile1D::is_full`]. Keeping the
    /// question on the view makes the complete CuTe flow visible to the
    /// compiler:
    ///
    /// ```text
    /// Tensor -> zipped_divide -> slice -> is_full
    /// ```
    #[must_use]
    #[inline(always)]
    pub const fn is_full(self) -> bool {
        tensor_tile_is_full::<T, TILE>(self)
    }

    /// Return this tile's first index in the original tensor.
    ///
    /// This is useful for a short final tile, where the kernel falls back to
    /// scalar loads and stores.
    #[must_use]
    #[inline(always)]
    pub const fn base(self) -> usize {
        tensor_tile_base::<T, TILE>(self)
    }

    /// Load the full tile into one [`RegisterTile`] using one vector load.
    ///
    /// # Safety
    ///
    /// - [`Tile1D::is_full`] must be `true`.
    /// - The tile's first address must be aligned to
    ///   `TILE * size_of::<T>()` bytes.
    /// - That total byte size must be a supported power of two from 4 through
    ///   16.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load(self) -> RegisterTile<T, TILE> {
        unsafe { tensor_load_tile::<T, TILE>(self) }
    }
}

/// A writable view of storage with a typed element format and layout.
///
/// This is separate from [`Tensor`] so a read-only slice can never gain a
/// store method. `TensorMut` is not `Copy` or `Clone`: there should be one
/// owner of the writable view.
///
/// GPU threads may still calculate overlapping indices, so store methods are
/// unsafe and state the non-overlap rules. This view is used inside Rust
/// kernel code; it is not passed directly as a raw kernel argument.
#[derive(Debug)]
pub struct TensorMut<'a, Element, Layout, Storage = Element>
where
    Element: TensorElement<Storage>,
    Storage: Copy,
{
    storage: *mut Storage,
    storage_len: usize,
    layout: Layout,
    element: PhantomData<Element>,
    borrow: PhantomData<&'a mut [Storage]>,
}

impl<'a, Element: TensorElement<Storage>, Layout, Storage: Copy>
    TensorMut<'a, Element, Layout, Storage>
{
    /// Build a writable tensor view from a raw pointer, length, and layout.
    ///
    /// # Safety
    ///
    /// - `storage` must remain valid and correctly aligned for `storage_len`
    ///   elements throughout `'a`.
    /// - Every offset produced by `layout` must be less than `storage_len`.
    /// - No other live reference may read or write the same memory in a way
    ///   that breaks Rust's exclusive-access rules.
    #[must_use]
    #[inline(always)]
    pub const unsafe fn from_raw_parts(
        storage: *mut Storage,
        storage_len: usize,
        layout: Layout,
    ) -> Self {
        Self {
            storage,
            storage_len,
            layout,
            element: PhantomData,
            borrow: PhantomData,
        }
    }

    /// Return the layout value stored by this view.
    #[must_use]
    #[inline(always)]
    pub const fn layout(&self) -> Layout
    where
        Layout: Copy,
    {
        self.layout
    }
}

impl<'a, T: Copy + TensorElement<T>> TensorMut<'a, T, Contiguous1D, T> {
    /// View a kernel output slice as one contiguous row.
    #[must_use]
    #[inline(always)]
    pub fn from_disjoint_slice(storage: &'a mut DisjointSlice<'_, T>) -> Self {
        let len = storage.len();
        unsafe { make_tensor_write(storage.as_mut_ptr(), len) }
    }

    /// Write one value at its index in the original contiguous tensor.
    ///
    /// # Safety
    ///
    /// - `index` must be less than the tensor length.
    /// - No concurrently running thread may access the same element in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store_at(&mut self, index: usize, value: T) {
        unsafe { *self.storage.add(index) = value };
    }

    /// Split the row into `(position_in_tile, tile_number)` coordinates.
    ///
    /// This only changes the view. It does not write memory.
    #[must_use]
    #[inline(always)]
    pub fn zipped_divide<const TILE: usize>(self) -> TensorMut<'a, T, Zipped1D<TILE>, T> {
        zipped_divide_write::<T, TILE>(self)
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> TensorMut<'a, T, Zipped1D<TILE>, T> {
    /// Select writable tile number `tile`.
    ///
    /// This only computes the tile's starting index. Call [`TensorMut::store`]
    /// to write it.
    #[must_use]
    #[inline(always)]
    pub fn slice(self, tile: usize) -> TensorMut<'a, T, Tile1D<TILE>, T> {
        slice_write::<T, TILE>(self, tile)
    }
}

/// Compiler boundary for a read-only contiguous tensor view.
///
/// This stays public only so device compilation can identify the exact
/// definition across crate boundaries. Use [`Tensor::from_slice`] in kernels.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn make_tensor_read<'a, T: Copy + TensorElement<T>>(
    storage: &'a [T],
) -> Tensor<'a, T, Contiguous1D, T> {
    Tensor::from_storage(storage, Contiguous1D::new(storage.len()))
}

/// Compiler boundary for a writable contiguous tensor view.
///
/// Use [`TensorMut::from_disjoint_slice`] instead. The safety rules are the
/// same as [`TensorMut::from_raw_parts`].
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const unsafe fn make_tensor_write<'a, T: Copy + TensorElement<T>>(
    storage: *mut T,
    storage_len: usize,
) -> TensorMut<'a, T, Contiguous1D, T> {
    unsafe { TensorMut::from_raw_parts(storage, storage_len, Contiguous1D::new(storage_len)) }
}

/// Compiler boundary for splitting a read-only row into fixed-size tiles.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn zipped_divide_read<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Contiguous1D, T>,
) -> Tensor<'a, T, Zipped1D<TILE>, T> {
    const { assert!(TILE > 0) };
    Tensor::from_storage(
        tensor.storage,
        Zipped1D {
            len: tensor.layout.len,
        },
    )
}

/// Compiler boundary for splitting a writable row into fixed-size tiles.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub fn zipped_divide_write<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Contiguous1D, T>,
) -> TensorMut<'a, T, Zipped1D<TILE>, T> {
    const { assert!(TILE > 0) };
    TensorMut {
        storage: tensor.storage,
        storage_len: tensor.storage_len,
        layout: Zipped1D {
            len: tensor.layout.len,
        },
        element: PhantomData,
        borrow: PhantomData,
    }
}

/// Compiler boundary for selecting one read-only tile.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn slice_read<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Zipped1D<TILE>, T>,
    tile: usize,
) -> Tensor<'a, T, Tile1D<TILE>, T> {
    Tensor::from_storage(
        tensor.storage,
        Tile1D {
            base: tile.saturating_mul(TILE),
            len: tensor.layout.len,
        },
    )
}

/// Compiler boundary for selecting one writable tile.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub fn slice_write<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Zipped1D<TILE>, T>,
    tile: usize,
) -> TensorMut<'a, T, Tile1D<TILE>, T> {
    TensorMut {
        storage: tensor.storage,
        storage_len: tensor.storage_len,
        layout: Tile1D {
            base: tile.saturating_mul(TILE),
            len: tensor.layout.len,
        },
        element: PhantomData,
        borrow: PhantomData,
    }
}

impl<'a, T: Copy + TensorElement<T>, const TILE: usize> TensorMut<'a, T, Tile1D<TILE>, T> {
    /// Write one value using its absolute index in the original tensor.
    ///
    /// Use this for a short final tile. Use [`TensorMut::store`] for a full
    /// tile so all values are written by one vector store.
    ///
    /// # Safety
    ///
    /// - `index` must be between this tile's [`Tile1D::base`] and the end of
    ///   its valid range.
    /// - No concurrently running thread may access the same element in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store_linear(&mut self, index: usize, value: T) {
        unsafe { tensor_store_element_abs::<T, TILE>(self, index, value) };
    }

    /// Write a complete [`RegisterTile`] using one vector store.
    ///
    /// # Safety
    ///
    /// - [`Tile1D::is_full`] must be `true`.
    /// - The tile's first address must meet the alignment required by
    ///   [`crate::store_tile`].
    /// - `TILE * size_of::<T>()` must be a supported power of two from 4
    ///   through 16.
    /// - No concurrently running thread may access the same elements in a
    ///   conflicting way.
    #[inline(always)]
    pub unsafe fn store(self, values: RegisterTile<T, TILE>) {
        unsafe { tensor_store_tile::<T, TILE>(self, values) };
    }
}

/// Compiler boundary for asking whether a selected tile is complete.
///
/// This stays public only so device compilation can identify the exact
/// definition across crate boundaries. Use [`Tensor::is_full`] in kernels.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn tensor_tile_is_full<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> bool {
    tensor.layout.is_full()
}

/// Compiler boundary for reading a selected tile's absolute base index.
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub const fn tensor_tile_base<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> usize {
    tensor.layout.base()
}

/// Compiler boundary for loading one complete selected tile.
///
/// # Safety
///
/// The rules are the same as [`Tensor::load`].
#[doc(hidden)]
#[must_use]
#[inline(never)]
pub unsafe fn tensor_load_tile<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: Tensor<'a, T, Tile1D<TILE>, T>,
) -> RegisterTile<T, TILE> {
    let base = unsafe { assume_div::<TILE>(tensor.layout.base) };
    RegisterTile(unsafe { load_tile::<T, TILE>(tensor.storage, base) })
}

/// Compiler boundary for storing one complete selected tile.
///
/// # Safety
///
/// The rules are the same as [`TensorMut::store`].
#[doc(hidden)]
#[inline(never)]
pub unsafe fn tensor_store_tile<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: TensorMut<'a, T, Tile1D<TILE>, T>,
    values: RegisterTile<T, TILE>,
) {
    let base = unsafe { assume_div::<TILE>(tensor.layout.base) };
    unsafe { store_tile::<T, TILE>(tensor.storage, base, values.as_array()) };
}

/// Compiler boundary for one scalar store in a short final tile.
///
/// # Safety
///
/// The rules are the same as [`TensorMut::store_linear`].
#[doc(hidden)]
#[inline(never)]
pub unsafe fn tensor_store_element_abs<'a, T: Copy + TensorElement<T>, const TILE: usize>(
    tensor: &mut TensorMut<'a, T, Tile1D<TILE>, T>,
    index: usize,
    value: T,
) {
    unsafe { *tensor.storage.add(index) = value };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestElement;

    impl TensorElement<u32> for TestElement {}

    #[test]
    fn tensor_view_borrows_storage_without_allocating() {
        let storage = [0u32; 8];
        let tensor = Tensor::<TestElement, _, u32>::from_storage(&storage, (2usize, 4usize));

        assert_eq!(tensor.storage().as_ptr(), storage.as_ptr());
        assert_eq!(tensor.storage().len(), storage.len());
        assert_eq!(tensor.layout(), (2, 4));
    }

    #[test]
    fn zipped_divide_and_slice_are_views_over_the_same_storage() {
        let storage = [0.0f32; 10];
        let tensor: Tensor<'_, f32, Contiguous1D> = Tensor::from_slice(&storage);
        let tiled = tensor.zipped_divide::<4>();

        assert_eq!(tiled.layout().tile_count(), 3);

        let middle = tiled.slice(1);
        assert_eq!(middle.storage().as_ptr(), storage.as_ptr());
        assert_eq!(middle.layout().base(), 4);
        assert!(middle.layout().is_full());

        let tail = tiled.slice(2);
        assert_eq!(tail.layout().base(), 8);
        assert!(!tail.layout().is_full());
    }

    #[test]
    fn register_tiles_add_elementwise() {
        let a = RegisterTile([1.0f32, 2.0, 3.0, 4.0]);
        let b = RegisterTile([10.0f32, 20.0, 30.0, 40.0]);

        assert_eq!((a + b).into_array(), [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn mutable_tensor_is_not_larger_than_its_runtime_fields() {
        assert_eq!(
            core::mem::size_of::<TensorMut<'static, f32, Contiguous1D, f32>>(),
            3 * core::mem::size_of::<usize>()
        );
    }
}
