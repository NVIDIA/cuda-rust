/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DisjointSlice - a type-safe abstraction for parallel GPU writes.
//!
//! This module provides `DisjointSlice<T>`, which guarantees that each thread
//! accesses a unique element, preventing data races.
//!
//! # Safety Model
//!
//! Safety is enforced through the type system and bounds checking:
//!
//! 1. **ThreadIndex**: Can only be constructed by `index_1d`,
//!    `index_2d::<S>`, or `index_2d_runtime(&slice)`, which derive
//!    the index from hardware built-in variables (`threadIdx`,
//!    `blockIdx`, `blockDim`) -- read-only special registers assigned
//!    by the runtime at kernel launch. Flat spaces combine these into
//!    a scalar index per thread, with the 2D const stride encoded in
//!    the index space so mixing const strides is rejected at compile
//!    time. A `Runtime2DIndex` witness instead carries the raw
//!    `(row, col)` coordinates, and the addressed slice resolves them
//!    against its own host-bound row width at the access site.
//!
//! 2. **`get_mut(idx)`**: Bounds-checked access via an explicit
//!    `ThreadIndex`. Returns `Option<&mut T>` — `None` for out-of-bounds
//!    threads.
//!
//! 3. **`get_mut_indexed()`**: One-call form that mints the witness and
//!    resolves it in a single shot. Available when the index space implements
//!    [`crate::thread::IndexFormula`] (i.e. `Index1D` or `Index2D<S>`).
//!
//! 4. **`get_unchecked_mut()`**: Unsafe escape hatch for performance-critical
//!    paths where bounds have been validated by other means.
//!
//! The unsafe boundary is pushed away from each access. A prepared launch proves
//! that the kernel's declared index domain matches its geometry; an unprepared
//! raw launch is unsafe and leaves that proof to the caller. Constructing a
//! `DisjointSlice` from raw memory is also unsafe.

use crate::thread::{
    FlatIndexSpace, Index1D, Index2D, IndexFormula, LaunchContext, Runtime2DIndex, ThreadIndex,
    WarpIndex,
};
use crate::view::{LinearTiles, RowMajorTiles, RuntimeRowMajorTiles};
use core::marker::PhantomData;
use core::mem::size_of;

mod space_layout_sealed {
    pub trait Sealed {}
}

/// The layout data an index space needs at runtime, carried in the slice.
///
/// An index space that fixes its geometry in the type needs nothing here, so
/// [`Data`](Self::Data) is `()` and the slice keeps its `{ ptr, len }` layout.
/// A space whose row width is a runtime value sets `Data = u32`, and the host
/// marshals that width into the launch packet beside the pointer and length.
///
/// # Why the row width belongs here and not at the access site
///
/// Tile disjointness needs one row width per slice for the slice's whole
/// lifetime. A width supplied per call cannot give that, even when each
/// candidate value is itself launch-uniform: safe code can select between two
/// [`Uniform<u32>`] values under a thread-varying condition, and the selected
/// width then differs between threads. Two threads that disagree about the row
/// width describe two grids over one buffer, and their "disjoint" tiles
/// collide. With 1x1 tiles, thread `(1, 0)` at width 5 and thread `(0, 5)` at
/// width 100 both resolve flat index 5.
///
/// Binding the row width at the host boundary removes the choice. The host writes
/// one value into the launch packet, every thread reads that word, and device
/// code has no step at which it could substitute another.
///
/// [`Uniform<u32>`]: crate::uniform::Uniform
pub trait SpaceLayout: space_layout_sealed::Sealed {
    /// Runtime layout data marshalled with the slice, or `()` when the index
    /// space is fully described by its type.
    type Data: Copy;
}

impl space_layout_sealed::Sealed for Index1D {}
impl SpaceLayout for Index1D {
    type Data = ();
}

impl<const ROW_STRIDE: usize> space_layout_sealed::Sealed for Index2D<ROW_STRIDE> {}
impl<const ROW_STRIDE: usize> SpaceLayout for Index2D<ROW_STRIDE> {
    type Data = ();
}

impl<const N: usize> space_layout_sealed::Sealed for LinearTiles<N> {}
impl<const N: usize> SpaceLayout for LinearTiles<N> {
    type Data = ();
}

impl<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> space_layout_sealed::Sealed
    for RowMajorTiles<ROWS, COLS, ROW_STRIDE>
{
}
impl<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> SpaceLayout
    for RowMajorTiles<ROWS, COLS, ROW_STRIDE>
{
    type Data = ();
}

impl space_layout_sealed::Sealed for Runtime2DIndex {}
impl SpaceLayout for Runtime2DIndex {
    type Data = u32;
}

impl space_layout_sealed::Sealed for WarpIndex {}
impl SpaceLayout for WarpIndex {
    type Data = ();
}

impl<const ROWS: usize, const COLS: usize> space_layout_sealed::Sealed
    for RuntimeRowMajorTiles<ROWS, COLS>
{
}
impl<const ROWS: usize, const COLS: usize> SpaceLayout for RuntimeRowMajorTiles<ROWS, COLS> {
    /// The matrix row width, in elements.
    type Data = u32;
}

/// A slice-like type that can only be accessed with thread-local indices.
///
/// # Safety Invariants
///
/// The type system enforces these invariants:
/// 1. Default access via `get_mut(ThreadIndex)` is bounds-checked and sound.
/// 2. `ThreadIndex` can only be created by trusted index functions
///    (`index_1d`, `index_2d::<S>`, `index_2d_runtime(&slice)`), which
///    derive the index from hardware built-in variables -- read-only
///    special registers assigned by the runtime at launch.
/// 3. Each thread's `ThreadIndex` is unique within its index space when the
///    prepared launch contract matches that space. Raw launch callers must
///    uphold the same condition explicitly.
///
/// Each thread accesses a unique element, making parallel writes safe without
/// synchronization.
///
/// # Memory Layout
///
/// Internally, this is identical to a slice: `{ ptr: *mut T, len: usize }`
/// The safety comes from type-level enforcement and bounds checking.
///
/// # Soundness
///
/// `get_mut()` returns `Option<&mut T>`, making out-of-bounds access
/// impossible in safe code. The previous API (`get() -> &mut T`) relied on
/// the caller to check bounds externally; in release builds this was UB for
/// out-of-bounds indices — a soundness hole. The current design follows
/// `slice::get_mut()` / `slice::get_unchecked_mut()` from std: the safe
/// path is sound when reached through a prepared launch. Raw launch geometry
/// and the unchecked element accessor are explicit unsafe escape hatches.
///
/// The type is `Send` but NOT `Sync`: each GPU thread gets its own copy of
/// the struct (with the same backing pointer), then uses its unique
/// `ThreadIndex` to access a different element. Sharing `&DisjointSlice`
/// across threads is not meaningful.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::{thread, DisjointSlice};
///
/// #[kernel]
/// pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = thread::index_1d();
///     let i = idx.get();
///     if let Some(c_elem) = c.get_mut(idx) {
///         *c_elem = a[i] + b[i];
///     }
/// }
/// ```
#[repr(C)]
pub struct DisjointSlice<'a, T, IndexSpace: SpaceLayout = Index1D> {
    ptr: *mut T,
    len: usize,
    /// Runtime layout for the index space, written once by the host. A
    /// zero-sized `()` for every space whose geometry is in its type, so the
    /// struct keeps its `{ ptr, len }` layout and ABI in that case.
    space: IndexSpace::Data,
    _marker: PhantomData<&'a mut [T]>,
    _space: PhantomData<fn() -> IndexSpace>,
}

/// The `space` field must not disturb the kernel ABI of any index space that
/// carries no runtime layout. `()` is zero-sized, so a `#[repr(C)]` slice over
/// a static space stays exactly `{ ptr, len }` and every existing kernel keeps
/// its two-word parameter lowering. A runtime row width adds one word.
const _: () = {
    assert!(
        size_of::<DisjointSlice<'static, f32, Index1D>>()
            == size_of::<*mut f32>() + size_of::<usize>()
    );
    assert!(
        size_of::<DisjointSlice<'static, f32, RowMajorTiles<2, 2, 64>>>()
            == size_of::<*mut f32>() + size_of::<usize>()
    );
    assert!(
        size_of::<DisjointSlice<'static, f32, RuntimeRowMajorTiles<1, 1>>>()
            == size_of::<*mut f32>() + size_of::<usize>() + size_of::<u64>()
    );
};

mod launch_contract_sealed {
    pub trait Sealed {}
}

impl<'a, T, IndexSpace: SpaceLayout> launch_contract_sealed::Sealed
    for DisjointSlice<'a, T, IndexSpace>
{
}

/// Compiler-facing proof that a `DisjointSlice` has the expected element type
/// and supports a launch domain.
///
/// This trait is sealed: only the genuine [`DisjointSlice`] type can implement
/// it. `#[cuda_module]` adds this bound to contracted kernels so Rust resolves
/// type aliases before checking the declared launch domain.
///
/// A 2D index space also supports a 1D launch (all Y dimensions are one), but
/// a 1D index space cannot support a 2D launch.
#[doc(hidden)]
pub trait __LaunchContractDisjointSlice<Element, const DOMAIN: u8>:
    launch_contract_sealed::Sealed
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 1> for DisjointSlice<'a, T, Index1D> {}

impl<'a, T, const N: usize> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, LinearTiles<N>>
{
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, RowMajorTiles<ROWS, COLS, ROW_STRIDE>>
{
}

impl<'a, T, const ROWS: usize, const COLS: usize> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, RuntimeRowMajorTiles<ROWS, COLS>>
{
}

impl<'a, T, const ROW_STRIDE: usize> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, crate::thread::Index2D<ROW_STRIDE>>
{
}

impl<'a, T, const ROW_STRIDE: usize> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, crate::thread::Index2D<ROW_STRIDE>>
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 1> for DisjointSlice<'a, T, crate::thread::WarpIndex> {}

impl<'a, T> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, crate::thread::Runtime2DIndex>
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, crate::thread::Runtime2DIndex>
{
}

/// Compiler-facing proof that a `DisjointSlice`'s launch-packet shape matches
/// the host ABI `#[cuda_module]` chose for it.
///
/// The macro selects between the two-word `(ptr, len)` and three-word
/// `(ptr, len, width)` host marshalling from the *spelling* of the index
/// space, because a proc macro cannot resolve types. Spelling alone is
/// forgeable: `type Rt = RuntimeRowMajorTiles<1, 1>;` names a runtime-width
/// space without saying so, and a host launch that pushed two parameters for a
/// three-parameter kernel would make the driver read past the argument array.
///
/// This sealed trait is the semantic authority behind that fast path. The
/// macro bounds every `DisjointSlice` parameter by the packet shape it chose
/// (`HAS_ROW_WIDTH = true` or `false`), and only the genuine `DisjointSlice` whose
/// [`SpaceLayout::Data`] actually matches that packet shape implements the
/// corresponding instantiation. An alias that hides a runtime row width, or a
/// look-alike space that fakes one, fails the bound before any launch code is
/// generated. If a future index space carries layout data of another shape,
/// it implements neither instantiation and every kernel using it fails
/// closed until the host marshalling learns the new shape.
#[doc(hidden)]
pub trait __LaunchContractDisjointSliceAbi<Element, const HAS_ROW_WIDTH: bool>:
    launch_contract_sealed::Sealed
{
}

impl<'a, T, IndexSpace: SpaceLayout<Data = ()>> __LaunchContractDisjointSliceAbi<T, false>
    for DisjointSlice<'a, T, IndexSpace>
{
}

impl<'a, T, IndexSpace: SpaceLayout<Data = u32>> __LaunchContractDisjointSliceAbi<T, true>
    for DisjointSlice<'a, T, IndexSpace>
{
}

impl<'a, T, IndexSpace: SpaceLayout> DisjointSlice<'a, T, IndexSpace> {
    /// Create a `DisjointSlice` from a raw pointer, a length and the runtime
    /// layout its index space needs.
    ///
    /// This is the general form. For an index space whose geometry is fixed in
    /// its type, [`from_raw_parts`](Self::from_raw_parts) says the same thing
    /// without the `()`.
    ///
    /// # Safety
    ///
    /// Carries every obligation of [`from_raw_parts`](Self::from_raw_parts),
    /// and one more: `space` must describe the buffer's real layout, and must
    /// be the same value for every thread of the launch. The host writes it
    /// once into the launch packet, which is what discharges the second half.
    #[inline]
    pub unsafe fn from_raw_parts_with_space(
        ptr: *mut T,
        len: usize,
        space: IndexSpace::Data,
    ) -> Self {
        DisjointSlice {
            ptr,
            len,
            space,
            _marker: PhantomData,
            _space: PhantomData,
        }
    }
}

impl<'a, T, IndexSpace: SpaceLayout<Data = ()>> DisjointSlice<'a, T, IndexSpace> {
    /// Create a DisjointSlice from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to valid, aligned memory for `len` elements of type `T`
    /// - The memory will remain valid and not be deallocated for lifetime `'a`
    /// - **Exclusive access**: no other live `DisjointSlice<T>` (or `&mut [T]`,
    ///   `&[T]`, raw read/write) covers any byte of
    ///   `ptr..ptr + len * size_of::<T>()` for the duration of `'a`. Two
    ///   `DisjointSlice` over the same range gives every thread two `&mut T`
    ///   to the same slot, which is UB regardless of the witness-type story.
    /// - The kernel launch configuration ensures threads access disjoint elements
    ///   (i.e., grid dimensions match the data dimensions)
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        DisjointSlice {
            ptr,
            len,
            space: (),
            _marker: PhantomData,
            _space: PhantomData,
        }
    }

    /// Create a DisjointSlice from a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The kernel launch configuration ensures threads access disjoint elements
    /// - No other code accesses the slice during kernel execution
    #[inline]
    pub unsafe fn from_mut_slice(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw_parts(slice.as_mut_ptr(), slice.len()) }
    }
}

impl<'a, T, IndexSpace: SpaceLayout> DisjointSlice<'a, T, IndexSpace> {
    /// Get the length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a mutable reference to an element at a raw index, without
    /// bounds checking.
    ///
    /// This is an escape hatch for paths where bounds have been validated by
    /// other means. It gives up the bounds check as well as the disjointness
    /// proof, so reach for it only when no index space describes the access.
    ///
    /// One shape that used to need it no longer does: a warp reduction where
    /// only lane 0 writes is [`WarpIndex`]. The warp
    /// is the index space, [`thread::warp_index`](crate::thread::warp_index)
    /// mints the witness for lane 0 alone, and the write goes through
    /// [`get_mut`](Self::get_mut) with its bounds check intact.
    ///
    /// What genuinely remains here is a destination drawn from data, such as a
    /// scatter through a permutation. That the permutation is a bijection is a
    /// fact about buffer contents, so no device-side type can carry it, and
    /// both the bounds and the disjointness stay the caller's obligation.
    ///
    /// Other uses:
    /// - Histogram updates with atomic operations
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `idx < self.len()` (bounds are valid)
    /// - No two threads write to the same index simultaneously
    /// - The uniqueness guarantee comes from the algorithm (document it!)
    ///
    /// # Example: Warp Reduction
    ///
    /// ```rust,ignore
    /// // SAFETY: Only lane 0 of each warp writes, and warp indices are unique
    /// if warp::lane_id() == 0 {
    ///     let warp_idx = gid.get() / 32;
    ///     unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {
        debug_assert!(
            idx < self.len,
            "Index out of bounds: {} >= {}",
            idx,
            self.len
        );
        unsafe { &mut *self.ptr.add(idx) }
    }
}

impl<'a, T, IndexSpace: FlatIndexSpace + SpaceLayout> DisjointSlice<'a, T, IndexSpace> {
    /// Get a mutable reference to an element at a thread-local index,
    /// returning `None` if the index is out of bounds.
    ///
    /// This is the default, sound access method. Mirrors `slice::get_mut()`.
    ///
    /// # Safety Argument
    ///
    /// This method is safe (not marked `unsafe`) because:
    ///
    /// 1. **Bounds checked**: Returns `None` for out-of-bounds indices.
    ///
    /// 2. **Unique access**: `ThreadIndex` can only be constructed by
    ///    `index_1d()`, `index_2d::<S>()`, or `index_2d_runtime()`, which
    ///    derive the index from hardware built-in variables (`threadIdx`,
    ///    `blockIdx`, `blockDim`) -- read-only special registers assigned
    ///    by the runtime at kernel launch. 2D stride is carried in the
    ///    index space, so a slice can only be indexed by a matching witness.
    ///
    /// 3. **No data races**: Given the constraint above, each thread's
    ///    `ThreadIndex` is unique under the prepared launch's matching index
    ///    domain. An unsafe raw launch takes responsibility for the same
    ///    geometry invariant, so each thread accesses a different location.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let idx = thread::index_1d();
    /// let i = idx.get();
    /// if let Some(elem) = c.get_mut(idx) {
    ///     *elem = a[i] + b[i];
    /// }
    /// ```
    // Full-debug frame shape:
    // caller PC -> kernel
    // helper PC -> get_mut -> kernel
    //
    // NVPTX otherwise covers discontiguous inlined instructions with one DWARF
    // envelope, so CUDA-GDB can report `get_mut` at caller-only PCs. Outline this
    // helper only in full debug; other modes keep normal inlining.
    #[cfg_attr(cuda_oxide_internal_outline_disjoint_get_mut_v1, inline(never))]
    #[cfg_attr(not(cuda_oxide_internal_outline_disjoint_get_mut_v1), inline)]
    pub fn get_mut<'kernel>(&mut self, idx: ThreadIndex<'kernel, IndexSpace>) -> Option<&mut T> {
        let i = idx.get();
        if size_of::<T>() != 0 && idx.is_valid() && i < self.len {
            // SAFETY:
            // - Bounds check passed above.
            // - `idx` is a ThreadIndex derived from hardware built-in variables.
            //   The prepared launch contract, or an unsafe raw launch caller,
            //   guarantees that its index space is unique for this geometry.
            // - The DisjointSlice was constructed with valid memory (from_raw_parts safety).
            Some(unsafe { &mut *self.ptr.add(i) })
        } else {
            None
        }
    }
}

impl<'a, T, const ROWS: usize, const COLS: usize>
    DisjointSlice<'a, T, RuntimeRowMajorTiles<ROWS, COLS>>
{
    /// The row width this slice was built with, in elements.
    ///
    /// Written once by the host into the launch packet, so every thread of the
    /// launch reads the same value. That is what lets
    /// [`tile_2d32_rt`](Self::tile_2d32_rt) be safe.
    #[inline(always)]
    pub fn row_width(&self) -> u32 {
        self.space
    }
}

impl<'a, T> DisjointSlice<'a, T, Runtime2DIndex> {
    /// The row width this slice was built with, in elements.
    ///
    /// Written once by the host into the launch packet, so every thread of the
    /// launch reads the same value.
    #[inline(always)]
    pub fn row_width(&self) -> u32 {
        self.space
    }

    /// Get a mutable reference to the element this thread's witness selects,
    /// returning `None` if the thread owns no cell of this slice's grid.
    ///
    /// # Safety Argument
    ///
    /// A [`Runtime2DIndex`] witness stores the thread's raw `(row, col)`
    /// coordinates, and this method resolves them against **this** slice's
    /// own host-bound row width (`row * width + col`, requiring
    /// `col < width`). That per-slice resolution is what makes the method
    /// safe:
    ///
    /// 1. **Bounds checked**: `col < width` and `flat < len` both return
    ///    `None` on failure.
    ///
    /// 2. **Injective per slice**: with one row width for the whole launch
    ///    (written once by the host) and `col < width`, distinct `(row, col)`
    ///    pairs resolve to distinct flat indices. Distinct threads hold
    ///    distinct hardware coordinates, so no two threads reach the same
    ///    element through this slice.
    ///
    /// 3. **No witness mixing**: a witness minted from a slice with a
    ///    different row width carries only its coordinates, never the minting
    ///    slice's grid. Resolving here uses this slice's width, so handing
    ///    witnesses across slices of different widths cannot alias two
    ///    `&mut` onto one element the way flat-index witnesses could.
    #[inline]
    pub fn get_mut<'kernel>(
        &mut self,
        idx: ThreadIndex<'kernel, Runtime2DIndex>,
    ) -> Option<&mut T> {
        if size_of::<T>() == 0 || !idx.is_valid() {
            return None;
        }
        let row = idx.row();
        let col = idx.col();
        let width = self.space as usize;
        // `col < width` keeps the coordinate inside one logical row, which is
        // what makes `row * width + col` injective; it also rejects a zero
        // width. `row` and `col` each fit 32 bits (packed representation) and
        // `width <= u32::MAX`, so the flat index cannot wrap 64-bit `usize`.
        if col >= width {
            return None;
        }
        let i = row * width + col;
        if i < self.len {
            // SAFETY:
            // - Bounds check passed above.
            // - The per-slice resolution argument in the doc comment: one
            //   host-bound row width per launch plus `col < width` makes the
            //   coordinate-to-offset map injective across threads.
            // - The DisjointSlice was constructed with valid memory
            //   (from_raw_parts_with_space safety).
            Some(unsafe { &mut *self.ptr.add(i) })
        } else {
            None
        }
    }
}

impl<'a, T, IS: IndexFormula + SpaceLayout> DisjointSlice<'a, T, IS> {
    /// One-shot indexed access — mints this thread's witness and resolves
    /// it to a mutable reference in a single call.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// let idx = thread::index_*();      // matching the slice's index space
    /// let cell = slice.get_mut(idx);    // None if out of bounds
    /// // (cell, idx)                    // ThreadIndex still in hand
    /// ```
    ///
    /// but with one index computation instead of two, and a flatter
    /// match: out-of-grid threads (e.g. `col >= ROW_STRIDE` for 2D) and
    /// out-of-slice indices both fold into a single `None`.
    ///
    /// # Where you call it
    ///
    /// Inside `#[kernel]` / `#[device]` the macro splices in the kernel
    /// scope for you, so the call site reads:
    ///
    /// ```rust,ignore
    /// #[kernel]
    /// fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    ///     if let Some((c_elem, idx)) = c.get_mut_indexed() {
    ///         let i = idx.get();
    ///         *c_elem = a[i] + b[i];
    ///     }
    /// }
    /// ```
    ///
    /// `get_mut_indexed` is a reserved method name inside annotated
    /// bodies — see the [reserved names note on
    /// `ThreadIndex`](crate::thread::ThreadIndex#reserved-names-inside-kernel-and-device).
    ///
    /// # Index space coverage
    ///
    /// Available for slices whose index space implements [`IndexFormula`]:
    /// `Index1D` and `Index2D<ROW_STRIDE>`. For `Runtime2DIndex` the row
    /// stride is a runtime value read from the slice, so use the
    /// [`index_2d_runtime`](crate::thread::index_2d_runtime) and
    /// [`get_mut`](Self::get_mut) pair explicitly.
    ///
    /// # Safety Argument
    ///
    /// Same as [`get_mut`](Self::get_mut): the returned `ThreadIndex` is
    /// minted from hardware built-ins via the trusted `__internal::*`
    /// path, the bounds check is explicit, and the borrow of `&mut self`
    /// keeps the returned reference exclusive for its lifetime.
    #[inline]
    pub fn get_mut_indexed<'kernel, Domain, Coordinates>(
        &mut self,
        scope: &'kernel LaunchContext<'kernel, Domain, Coordinates>,
    ) -> Option<(&mut T, ThreadIndex<'kernel, IS>)>
    where
        Domain: crate::thread::__internal::LaunchDomain,
    {
        let idx = IS::from_scope(scope)?;
        let i = idx.get();
        if size_of::<T>() != 0 && i < self.len {
            // SAFETY:
            // - bounds check passed above
            // - idx is freshly minted from hardware special registers (no
            //   laundering — !Copy, !Send, `'kernel`-bound); the prepared
            //   launch, or unsafe raw caller, proves IS is unique here
            // - DisjointSlice was constructed with valid memory
            //   (from_raw_parts safety)
            Some((unsafe { &mut *self.ptr.add(i) }, idx))
        } else {
            None
        }
    }
}

// SAFETY: DisjointSlice can be sent between threads because:
// - A prepared launch, or unsafe raw caller, guarantees that each thread's
//   ThreadIndex selects a unique element for the active geometry
// - The pointer and length are just data, no thread affinity
// - T: Send means the elements themselves can be sent between threads
unsafe impl<'a, T: Send, IndexSpace: SpaceLayout> Send for DisjointSlice<'a, T, IndexSpace> {}

// DisjointSlice auto-trait summary:
//   Send: yes (explicit impl above, when T: Send)
//   Sync: NO (not implemented) — each GPU thread gets its own copy of the
//         struct, then uses its conditionally unique ThreadIndex to access a
//         different element. Sharing &DisjointSlice across threads would allow
//         multiple threads to call get_mut() on the same struct, which
//         would produce aliasing &mut T references — unsound.

#[cfg(test)]
mod tests {
    //! Host-side proofs of the `Runtime2DIndex` resolution semantics.
    //!
    //! The witness carries `(row, col)` and the ADDRESSED slice resolves it
    //! against its own row width. These tests replay the exact scenario that
    //! made flat-index witnesses unsound: witnesses minted for row widths 5
    //! and 100 both flattened to element 5 (from `(1, 0)` and `(0, 5)`), so two
    //! threads held `&mut` to one element. Per-slice resolution has to make
    //! those resolutions distinct for every slice they could be handed to.

    use super::DisjointSlice;
    use crate::thread::{Runtime2DIndex, ThreadIndex, pack_runtime_2d_coords};

    /// Resolve a witness against a freshly built slice and return the flat
    /// element index it lands on, via pointer arithmetic on a real buffer.
    fn resolved_flat_index(
        buffer: &mut [f32],
        width: u32,
        row: usize,
        col: usize,
    ) -> Option<usize> {
        let base = buffer.as_mut_ptr();
        let len = buffer.len();
        // SAFETY: `buffer` is exclusively borrowed for the call and `width`
        // describes the grid the assertions below expect.
        let mut slice: DisjointSlice<'_, f32, Runtime2DIndex> =
            unsafe { DisjointSlice::from_raw_parts_with_space(base, len, width) };
        let witness = ThreadIndex::<Runtime2DIndex>::for_test(pack_runtime_2d_coords(row, col));
        slice
            .get_mut(witness)
            .map(|elem| (core::ptr::from_mut(elem) as usize - base as usize) / size_of::<f32>())
    }

    #[test]
    fn witnesses_from_two_row_widths_cannot_alias_one_element() {
        let mut buffer = [0.0f32; 512];

        // The historical collision: flat witnesses gave (1, 0)@width5 -> 5
        // and (0, 5)@width100 -> 5, one element for two threads. Resolved
        // against the addressed slice instead, the two coordinate pairs must
        // never land on the same element of any one slice.
        //
        // Addressed at row width 5: (0, 5) has col >= width, no cell at all.
        assert_eq!(resolved_flat_index(&mut buffer, 5, 1, 0), Some(5));
        assert_eq!(resolved_flat_index(&mut buffer, 5, 0, 5), None);
        // Addressed at row width 100: distinct elements.
        assert_eq!(resolved_flat_index(&mut buffer, 100, 1, 0), Some(100));
        assert_eq!(resolved_flat_index(&mut buffer, 100, 0, 5), Some(5));
    }

    #[test]
    fn resolution_is_injective_per_slice() {
        let mut buffer = [0.0f32; 1024];
        for width in [1u32, 3, 7, 64] {
            let mut seen = [false; 1024];
            for row in 0..8usize {
                for col in 0..8usize {
                    if let Some(flat) = resolved_flat_index(&mut buffer, width, row, col) {
                        assert!(col < width as usize, "resolved a cell outside the row");
                        assert!(
                            !seen[flat],
                            "coordinates ({row}, {col}) aliased element {flat} at row width {width}"
                        );
                        seen[flat] = true;
                    }
                }
            }
        }
    }

    #[test]
    fn zero_row_width_resolves_nothing() {
        let mut buffer = [0.0f32; 16];
        assert_eq!(resolved_flat_index(&mut buffer, 0, 0, 0), None);
    }

    #[test]
    fn out_of_len_coordinates_resolve_to_none() {
        let mut buffer = [0.0f32; 10];
        // Row 3 at row width 4 starts at element 12, past len 10.
        assert_eq!(resolved_flat_index(&mut buffer, 4, 3, 0), None);
        // Last in-bounds cell still resolves.
        assert_eq!(resolved_flat_index(&mut buffer, 4, 2, 1), Some(9));
    }
}
