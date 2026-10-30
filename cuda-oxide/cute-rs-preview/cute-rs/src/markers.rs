/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Layout facts stored in Rust types.
//!
//! These marker types have no runtime data. They let the compiler see a
//! layout directly in a function's type arguments.
//!
//! Most code can use a short form such as `RowMajor<32, 8>`. The full form
//! stores the same shape and strides explicitly, with Rust tuples grouping
//! the modes:
//!
//! ```text
//! 32 rows × 8 columns, row-major
//!
//! Layout< (C<32>, C<8>), (C<8>, C<1>) >
//!         └── shape ──┘  └─ stride ─┘
//!                      row: 8
//!                         column: 1
//! ```

use core::marker::PhantomData;

/// One integer stored in a type, usually a size or stride.
pub struct C<const VALUE: i64>;

/// A layout made from a `Shape` and its `Stride`.
///
/// A stride is the number of elements to skip when a coordinate increases by
/// one. Nested modes are Rust tuples of [`C`] leaves, which can themselves be
/// nested: `(A, (B, C))` and `((A, B), C)` describe different layouts.
///
/// This zero-sized marker is read from a function's type arguments. The
/// runtime value with the same shape and stride is [`cute_layout::Layout`],
/// reachable as `cute_rs::layout::Layout`.
pub struct Layout<Shape, Stride>(
    #[doc = "Carries corresponding shape and stride descriptions."] PhantomData<(Shape, Stride)>,
);

/// A `ROWS × COLS` matrix stored one complete row after another.
///
/// ```text
/// [a00 a01 a02] [a10 a11 a12]
/// ```
pub struct RowMajor<const ROWS: i64, const COLS: i64>;

/// A `ROWS × COLS` matrix stored one complete column after another.
///
/// ```text
/// [a00 a10] [a01 a11] [a02 a12]
/// ```
pub struct ColMajor<const ROWS: i64, const COLS: i64>;

/// One global-to-shared copy atom of `BYTES` bytes, issued as `cp.async`
/// (CuTeDSL `cpasync.CopyG2SOp`).
pub struct CopyG2S<const BYTES: u32>;

/// One universal vector copy atom of `BYTES` bytes (`4`, `8`, or `16` bytes),
/// corresponding to CuTeDSL's `cute.nvgpu.CopyUniversalOp`.
pub struct CopyUniversal<const BYTES: u32>;

/// The distance between rows, with a compile-time alignment promise.
///
/// `elements` is known only while the kernel runs. The type promises that it
/// is divisible by `DIVISOR`:
///
/// ```text
/// LeadingDim<4> { elements: 132 }
///            │              │
///            │              └─ 132 elements between row starts
///            └──────────────── 132 % 4 == 0
/// ```
///
/// It intentionally has no `new` helper. Build it with a struct literal so
/// the promise stays visible at the copy call where the compiler needs it.
#[repr(transparent)]
pub struct LeadingDim<const DIVISOR: usize> {
    /// Number of elements from the start of one row to the next.
    pub elements: usize,
}

/// Internal link between [`LeadingDim`] and block-wide copy operations.
///
/// This trait is public only because it appears in a public function bound.
/// Users do not implement it; the compiler accepts [`LeadingDim`] only.
#[doc(hidden)]
pub trait LeadingDimMarker {}

impl<const DIVISOR: usize> LeadingDimMarker for LeadingDim<DIVISOR> {}

/// A CuTe XOR swizzle written as `Swizzle<B, M, S>`.
///
/// A swizzle changes shared-memory addresses so threads are less likely to
/// request the same memory bank at once and be forced to wait:
///
/// ```text
/// normal address ── XOR selected bits ──► swizzled address
/// ```
///
/// - `B`: number of bits XORed.
/// - `M`: low bits left unchanged.
/// - `S`: distance from those low bits to the second XOR field.
///
/// At this API boundary, the bit positions count elements. The compiler
/// converts them to byte positions after it knows the element type.
pub struct Swizzle<const B: u32, const M: u32, const S: i32>;

/// One address rule built by applying `Outer`, adding `OFFSET`, then applying
/// `Inner`: `Inner(OFFSET + Outer(coordinate))`.
///
/// ```text
/// coordinate ── Outer ──► element offset
///                          + OFFSET
///                              │
///                              ▼
///                            Inner ──► final address
/// ```
///
/// The names are CuTe's. `Inner` is the swizzle, the map on the outside of
/// the expression that runs last; `Outer` is the affine layout that runs
/// first. The type arguments sit in the order of CuTeDSL's
/// `make_composed_layout(inner, offset, outer)`, so
/// `ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>` reads like that
/// call.
///
/// `OFFSET` and `Outer` count elements here. The compiler converts the whole
/// rule to bytes after it knows the element size. The runtime value with the
/// same parts is [`cute_layout::ComposedLayout`], reachable as
/// `cute_rs::layout::ComposedLayout`.
pub struct ComposedLayout<Inner, const OFFSET: i64, Outer>(
    #[doc = "Carries the inner swizzle and the outer layout."] PhantomData<(Inner, Outer)>,
);

/// Makes a two-dimensional shared-memory layout readable by host code.
///
/// The device compiler reads the layout from its Rust type. Host code needs
/// ordinary constants to build a TMA descriptor. This trait gives both sides
/// the same values:
///
/// ```text
/// one layout type
///      ├──► device compiler
///      └──► host TMA descriptor
/// ```
///
/// The swizzle constants carry the field names of `cute_layout::Swizzle`,
/// which are CuTeDSL's `Swizzle.num_bits`, `num_base`, and `num_shift`.
pub trait ReifySmem2D {
    /// Tile extents as `(rows, columns)`.
    const SHAPE: (i64, i64);
    /// Elements skipped when the row, then the column, increases by one.
    const STRIDE: (i64, i64);
    /// Fixed starting offset in elements; `0` for a plain layout.
    const OFFSET: i64;
    /// Swizzle `B`, the number of XOR-ed bits, in element units; `0` means no
    /// swizzle.
    const NUM_BITS: u32;
    /// Swizzle `M`: the number of low address bits left unchanged.
    const NUM_BASE: u32;
    /// Swizzle `S`: the distance between the two address fields.
    const NUM_SHIFT: i32;
}

impl<const R: i64, const CC: i64, const SR: i64, const SC: i64> ReifySmem2D
    for Layout<(C<R>, C<CC>), (C<SR>, C<SC>)>
{
    const SHAPE: (i64, i64) = (R, CC);
    const STRIDE: (i64, i64) = (SR, SC);
    const OFFSET: i64 = 0;
    const NUM_BITS: u32 = 0;
    const NUM_BASE: u32 = 4;
    const NUM_SHIFT: i32 = 3;
}

impl<const R: i64, const CC: i64> ReifySmem2D for RowMajor<R, CC> {
    const SHAPE: (i64, i64) = (R, CC);
    const STRIDE: (i64, i64) = (CC, 1);
    const OFFSET: i64 = 0;
    const NUM_BITS: u32 = 0;
    const NUM_BASE: u32 = 4;
    const NUM_SHIFT: i32 = 3;
}

impl<const R: i64, const CC: i64> ReifySmem2D for ColMajor<R, CC> {
    const SHAPE: (i64, i64) = (R, CC);
    const STRIDE: (i64, i64) = (1, R);
    const OFFSET: i64 = 0;
    const NUM_BITS: u32 = 0;
    const NUM_BASE: u32 = 4;
    const NUM_SHIFT: i32 = 3;
}

/// Read one composed layout. Do not nest `ComposedLayout`: this implementation
/// reports only its own inner swizzle and would hide a nested one.
impl<const B: u32, const M: u32, const S: i32, const OFFSET: i64, Outer: ReifySmem2D> ReifySmem2D
    for ComposedLayout<Swizzle<B, M, S>, OFFSET, Outer>
{
    const SHAPE: (i64, i64) = Outer::SHAPE;
    const STRIDE: (i64, i64) = Outer::STRIDE;
    const OFFSET: i64 = OFFSET;
    const NUM_BITS: u32 = B;
    const NUM_BASE: u32 = M;
    const NUM_SHIFT: i32 = S;
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Nested layout, swizzle and copy markers must occupy zero runtime bytes.
    #[test]
    fn markers_are_compile_time_only() {
        /// Two-dimensional 32-by-8 marker shape used to test zero-sized composition.
        type Shape = (C<32>, C<8>);
        /// Row-major marker strides corresponding to the test shape.
        type Stride = (C<8>, C<1>);
        /// Pairs the test’s shape and strides without runtime storage.
        type Affine = Layout<Shape, Stride>;
        /// Adds a swizzle to verify composed markers also remain zero-sized.
        type Smem = ComposedLayout<Swizzle<3, 4, 3>, 0, Affine>;

        assert_eq!(core::mem::size_of::<CopyG2S<16>>(), 0);
        assert_eq!(core::mem::size_of::<Smem>(), 0);
        assert_eq!(core::mem::size_of::<RowMajor<32, 8>>(), 0);
        assert_eq!(
            core::mem::size_of::<LeadingDim<4>>(),
            core::mem::size_of::<usize>()
        );
    }
}
