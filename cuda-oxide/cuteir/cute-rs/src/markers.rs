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
//! stores the same shape and strides explicitly:
//!
//! ```text
//! 32 rows × 8 columns, row-major
//!
//! L< T2<C<32>, C<8>>, T2<C<8>, C<1>> >
//!    └── shape ─────┘   └─ strides ───┘
//!                           row: 8
//!                        column: 1
//! ```

use core::marker::PhantomData;

/// One integer stored in a type, usually a size or stride.
pub struct C<const VALUE: i64>;

/// Two type-level values grouped together.
///
/// Pairs can be nested. For example, `(A, B), C` and `A, (B, C)` describe
/// different layouts.
pub struct T2<A, B>(PhantomData<(A, B)>);

/// A layout made from a `Shape` and its `Stride`.
///
/// A stride is the number of elements to skip when a coordinate increases by
/// one.
pub struct L<Shape, Stride>(PhantomData<(Shape, Stride)>);

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

/// One `cp.async` copy of `BYTES` bytes from global to shared memory.
pub struct CpAsync<const BYTES: u32>;

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

/// One address rule built by applying `Inner`, adding `OFFSET`, then applying
/// `Outer`.
///
/// ```text
/// coordinate ── Inner ──► element offset
///                          + OFFSET
///                              │
///                              ▼
///                            Outer ──► final address
/// ```
///
/// `OFFSET` and `Inner` count elements here. The compiler converts the whole
/// rule to bytes after it knows the element size.
pub struct Composed<Outer, const OFFSET: i64, Inner>(PhantomData<(Outer, Inner)>);

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
pub trait ReifySmem2D {
    /// Number of rows.
    const ROWS: i64;
    /// Number of columns.
    const COLS: i64;
    /// Number of elements skipped when the row increases by one.
    const ROW_STRIDE: i64;
    /// Number of elements skipped when the column increases by one.
    const COL_STRIDE: i64;
    /// Fixed starting offset in elements; `0` for a plain layout.
    const OFFSET: i64;
    /// Swizzle settings in element units; `B = 0` means no swizzle.
    const SWIZZLE_B: u32;
    /// Number of low address bits left unchanged by the swizzle.
    const SWIZZLE_M: u32;
    /// Distance between the two address fields used by the swizzle.
    const SWIZZLE_S: i32;
}

impl<const R: i64, const CC: i64, const SR: i64, const SC: i64> ReifySmem2D
    for L<T2<C<R>, C<CC>>, T2<C<SR>, C<SC>>>
{
    const ROWS: i64 = R;
    const COLS: i64 = CC;
    const ROW_STRIDE: i64 = SR;
    const COL_STRIDE: i64 = SC;
    const OFFSET: i64 = 0;
    const SWIZZLE_B: u32 = 0;
    const SWIZZLE_M: u32 = 4;
    const SWIZZLE_S: i32 = 3;
}

impl<const R: i64, const CC: i64> ReifySmem2D for RowMajor<R, CC> {
    const ROWS: i64 = R;
    const COLS: i64 = CC;
    const ROW_STRIDE: i64 = CC;
    const COL_STRIDE: i64 = 1;
    const OFFSET: i64 = 0;
    const SWIZZLE_B: u32 = 0;
    const SWIZZLE_M: u32 = 4;
    const SWIZZLE_S: i32 = 3;
}

impl<const R: i64, const CC: i64> ReifySmem2D for ColMajor<R, CC> {
    const ROWS: i64 = R;
    const COLS: i64 = CC;
    const ROW_STRIDE: i64 = 1;
    const COL_STRIDE: i64 = R;
    const OFFSET: i64 = 0;
    const SWIZZLE_B: u32 = 0;
    const SWIZZLE_M: u32 = 4;
    const SWIZZLE_S: i32 = 3;
}

/// Read one composed layout. Do not nest `Composed`: this implementation
/// returns the outer swizzle and would hide the inner one.
impl<const B: u32, const M: u32, const S: i32, const OFFSET: i64, Inner: ReifySmem2D> ReifySmem2D
    for Composed<Swizzle<B, M, S>, OFFSET, Inner>
{
    const ROWS: i64 = Inner::ROWS;
    const COLS: i64 = Inner::COLS;
    const ROW_STRIDE: i64 = Inner::ROW_STRIDE;
    const COL_STRIDE: i64 = Inner::COL_STRIDE;
    const OFFSET: i64 = OFFSET;
    const SWIZZLE_B: u32 = B;
    const SWIZZLE_M: u32 = M;
    const SWIZZLE_S: i32 = S;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_compile_time_only() {
        type Shape = T2<C<32>, C<8>>;
        type Stride = T2<C<8>, C<1>>;
        type Affine = L<Shape, Stride>;
        type Smem = Composed<Swizzle<3, 4, 3>, 0, Affine>;

        assert_eq!(core::mem::size_of::<CpAsync<16>>(), 0);
        assert_eq!(core::mem::size_of::<Smem>(), 0);
        assert_eq!(core::mem::size_of::<RowMajor<32, 8>>(), 0);
        assert_eq!(
            core::mem::size_of::<LeadingDim<4>>(),
            core::mem::size_of::<usize>()
        );
    }
}
