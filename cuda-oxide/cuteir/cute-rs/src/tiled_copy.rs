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
//!                     TiledCopy::copy
//!                            │
//!                            ▼
//! shared tile: [thread 0 part][thread 1 part][...]
//! ```
//!
//! [`GlobalCopyTensor`] describes the source. [`SharedTensor`] describes the
//! destination. [`TiledCopy`] describes how the work is split across threads.
//! These wrappers add no runtime storage beyond the source and destination
//! views. The compiler inlines them down to [`crate::copy_g2s`].

use core::marker::PhantomData;

use crate::cooperative::{GmemMatrix, SmemTile, copy_g2s};
use crate::markers::LeadingDimMarker;
use crate::tensor::TensorElement;

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
    carrier: GmemMatrix<T, LD>,
    element: PhantomData<(Element, Role)>,
}

impl<Element: TensorElement<T>, T: Copy, LD, Role> GlobalCopyTensor<Element, T, LD, Role> {
    /// Attach an element meaning and role to a global-memory matrix view.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn from_carrier(carrier: GmemMatrix<T, LD>) -> Self {
        Self {
            carrier,
            element: PhantomData,
        }
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
    carrier: SmemTile<T, Layout>,
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
/// The value contains no runtime data. The destination layout comes from the
/// [`SharedTensor`] passed to [`Self::copy`].
pub struct TiledCopy<Atom, ThreadLayout, ValueLayout, TileLayout> {
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

    /// Let the whole block copy one global tile into shared memory.
    ///
    /// `tile_coord` selects the source tile. `thread_idx` selects this thread's
    /// assigned part of that tile.
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
        thread_idx: u32,
    ) {
        unsafe {
            copy_g2s::<Atom, ThreadLayout, ValueLayout, TileLayout, SharedLayout, T>(
                source.carrier(),
                tile_coord,
                destination.carrier_mut(),
                thread_idx,
            )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_copy_is_static_and_typed_views_preserve_carrier_size() {
        type Copy = TiledCopy<(), (), (), ()>;
        assert_eq!(core::mem::size_of::<Copy>(), 0);
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
}
