/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Copy-atom and leading-dimension markers, plus re-exports of the compile-time
//! layout markers from [`cute_layout::markers`].

pub use cute_layout::markers::{
    C, ColMajor, ComposedLayout, Layout, ReifySmem2D, RowMajor, Swizzle,
};

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

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Copy-atom and re-exported layout markers must occupy zero runtime bytes.
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
