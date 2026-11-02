/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zero-sized compile-time layout and swizzle markers stored in Rust types.
//!
//! These marker types carry no runtime bytes. They let the compiler and
//! [`crate::static_layout`] inspect a layout directly from a function's generic
//! type arguments:
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

/// One integer stored in a type, usually a shape extent or stride.
pub struct C<const VALUE: i64>;

/// A static layout made from a `Shape` and its `Stride`.
///
/// A stride is the number of elements to skip when a coordinate increases by
/// one. Nested modes are Rust tuples of [`C`] leaves, which can themselves be
/// nested: `(A, (B, C))` and `((A, B), C)` describe different layouts.
///
/// This zero-sized marker is read from a function's type arguments. The
/// runtime value with the same shape and stride is [`crate::Layout`].
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

/// A CuTe XOR swizzle written as `Swizzle<B, M, S>`.
///
/// A swizzle permutes shared-memory addresses so threads are less likely to
/// request the same memory bank at once:
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
/// The type arguments sit in the order of CuTeDSL's
/// `make_composed_layout(inner, offset, outer)`, so
/// `ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>` reads like that
/// call. The runtime value with the same parts is [`crate::ComposedLayout`].
pub struct ComposedLayout<Inner, const OFFSET: i64, Outer>(
    #[doc = "Carries the inner swizzle and the outer layout."] PhantomData<(Inner, Outer)>,
);

/// Makes a two-dimensional shared-memory layout marker readable as compile-time constants.
///
/// The device compiler reads the layout from its Rust type, while host and
/// `const` evaluators use these constants directly:
///
/// ```text
/// one layout type
///      ├──► device compiler
///      └──► host TMA descriptor & static_layout
/// ```
pub trait ReifySmem2D {
    /// Tile extents as `(rows, columns)`.
    const SHAPE: (i64, i64);
    /// Elements skipped when the row, then the column, increases by one.
    const STRIDE: (i64, i64);
    /// Fixed starting offset in elements; `0` for a plain layout.
    const OFFSET: i64;
    /// Swizzle `B`, the number of XOR-ed bits, in element units; `0` means no swizzle.
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

/// Regression tests for zero-sized layout markers.
#[cfg(test)]
mod tests {
    use super::*;

    /// Nested layout and swizzle markers must occupy zero runtime bytes.
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

        assert_eq!(core::mem::size_of::<Smem>(), 0);
        assert_eq!(core::mem::size_of::<RowMajor<32, 8>>(), 0);
    }
}
