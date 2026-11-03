/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Allocation-free evaluation of static CuTe shape/stride markers.
//!
//! The leftmost shape mode changes fastest, as in [`crate::Layout`].
//! This bounded, nonnegative-stride subset is suitable for device address maps
//! and `const` contexts.

use crate::markers::{C, ColMajor, ComposedLayout, Layout, ReifySmem2D, RowMajor, Swizzle};

/// A static layout mapping a flat logical coordinate to an element offset.
pub trait StaticLayout {
    /// Logical element count. Invalid shapes are rejected at instantiation.
    const SIZE: usize;
    /// Upper bound on every in-domain physical element offset.
    ///
    /// Affine layouts have an exact maximum; swizzled layouts may conservatively
    /// include unused offsets. The bound fits both `usize` and positive `i64`.
    const MAX_OFFSET: usize;
    /// Map a flat coordinate in the layout's domain to its element offset
    /// without accessing memory (CuTeDSL `cute.crd2idx`). Callers that use
    /// the offset for memory access must bound the coordinate.
    fn crd2idx(flat: usize) -> usize;
}

/// Convert a nonnegative layout parameter without truncating it on narrow hosts.
const fn parameter(value: i64) -> usize {
    assert!(value >= 0 && value as u128 <= usize::MAX as u128);
    value as usize
}

/// Bound offsets by the signed representation consumed by the shared algebra.
const fn checked_offset_add(a: usize, b: usize) -> usize {
    let result = match a.checked_add(b) {
        Some(value) => value,
        None => panic!("static layout physical offset overflow"),
    };
    assert!(
        result as u128 <= i64::MAX as u128,
        "static layout offset exceeds i64"
    );
    result
}

/// Multiply two positive layout offset terms and verify the product fits in `i64`.
const fn checked_offset_mul(a: usize, b: usize) -> usize {
    let result = match a.checked_mul(b) {
        Some(value) => value,
        None => panic!("static layout physical offset overflow"),
    };
    assert!(
        result as u128 <= i64::MAX as u128,
        "static layout offset exceeds i64"
    );
    result
}

/// Swizzling changes only the target bit field. Fill all bits up through that
/// field to bound every offset below `maximum`, including holes in the layout.
const fn swizzled_bound(maximum: usize, bits: u32, base: u32, shift: i32) -> usize {
    let _ = crate::Swizzle::new(bits, base, shift);
    if bits == 0 {
        return maximum;
    }
    let target_end = base + bits + if shift < 0 { (-shift) as u32 } else { 0 };
    let lower_mask = (1_u64 << target_end) - 1;
    let result = maximum as u64 | lower_mask;
    assert!(result as u128 <= usize::MAX as u128 && result <= i64::MAX as u64);
    result as usize
}

impl<const N: i64, const S: i64> StaticLayout for Layout<C<N>, C<S>> {
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        parameter(N)
    };
    const MAX_OFFSET: usize = {
        assert!(N > 0 && S >= 0);
        checked_offset_mul(parameter(N - 1), parameter(S))
    };
    #[inline(always)]
    fn crd2idx(flat: usize) -> usize {
        let _ = Self::SIZE;
        flat * S as usize
    }
}

impl<A, B, SA, SB> StaticLayout for Layout<(A, B), (SA, SB)>
where
    Layout<A, SA>: StaticLayout,
    Layout<B, SB>: StaticLayout,
{
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        match Layout::<A, SA>::SIZE.checked_mul(Layout::<B, SB>::SIZE) {
            Some(n) => n,
            None => panic!("static layout extent overflow"),
        }
    };
    const MAX_OFFSET: usize =
        checked_offset_add(Layout::<A, SA>::MAX_OFFSET, Layout::<B, SB>::MAX_OFFSET);
    #[inline(always)]
    fn crd2idx(flat: usize) -> usize {
        let _ = Self::SIZE;
        Layout::<A, SA>::crd2idx(flat % Layout::<A, SA>::SIZE)
            + Layout::<B, SB>::crd2idx(flat / Layout::<A, SA>::SIZE)
    }
}

impl<const R: i64, const K: i64> StaticLayout for RowMajor<R, K> {
    const SIZE: usize = <Layout<(C<R>, C<K>), (C<K>, C<1>)>>::SIZE;
    const MAX_OFFSET: usize = <Layout<(C<R>, C<K>), (C<K>, C<1>)>>::MAX_OFFSET;
    #[inline(always)]
    fn crd2idx(flat: usize) -> usize {
        <Layout<(C<R>, C<K>), (C<K>, C<1>)>>::crd2idx(flat)
    }
}

impl<const R: i64, const K: i64> StaticLayout for ColMajor<R, K> {
    const SIZE: usize = <Layout<(C<R>, C<K>), (C<1>, C<R>)>>::SIZE;
    const MAX_OFFSET: usize = <Layout<(C<R>, C<K>), (C<1>, C<R>)>>::MAX_OFFSET;
    #[inline(always)]
    fn crd2idx(flat: usize) -> usize {
        <Layout<(C<R>, C<K>), (C<1>, C<R>)>>::crd2idx(flat)
    }
}

impl<const B: u32, const M: u32, const S: i32, const O: i64, Outer: StaticLayout> StaticLayout
    for ComposedLayout<Swizzle<B, M, S>, O, Outer>
{
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        Outer::SIZE
    };
    const MAX_OFFSET: usize =
        swizzled_bound(checked_offset_add(Outer::MAX_OFFSET, parameter(O)), B, M, S);
    #[inline(always)]
    fn crd2idx(flat: usize) -> usize {
        let _ = Self::SIZE;
        crate::Swizzle::new(B, M, S).apply((Outer::crd2idx(flat) + O as usize) as i64) as usize
    }
}

/// Map a `(row, column)` coordinate through a typed two-dimensional
/// shared-memory layout, including its swizzle (CuTeDSL `cute.crd2idx` on a
/// composed layout).
///
/// Offsets and strides count elements. The static offset is added before the
/// swizzle, matching [`crate::ComposedLayout`]. This computes an offset
/// only; callers that dereference it must establish coordinate bounds and
/// storage lifetime.
#[must_use]
#[inline(always)]
pub const fn crd2idx_2d<L: ReifySmem2D>(row: usize, column: usize) -> usize {
    let (stride_r, stride_c, offset, swizzle) = const {
        let _ = max_offset_2d::<L>();
        (
            L::STRIDE.0 as usize,
            L::STRIDE.1 as usize,
            L::OFFSET as usize,
            crate::Swizzle::new(L::NUM_BITS, L::NUM_BASE, L::NUM_SHIFT),
        )
    };
    let plain = row * stride_r + column * stride_c + offset;
    swizzle.apply(plain as i64) as usize
}

/// Unpack a flat 1D coordinate into `(row, column)` through a typed 2D layout
/// marker `L` (CuTeDSL `cute.idx2crd`).
///
/// Supports compact column-major (`ColMajor<R, C>` / `Layout<(C<R>, C<C>), (C<1>, C<R>)>`)
/// and compact row-major (`RowMajor<R, C>` / `Layout<(C<R>, C<C>), (C<C>, C<1>)>`)
/// 2D layouts, delegating the coordinate decomposition to `cute-layout`.
#[must_use]
#[inline(always)]
pub const fn idx2crd_2d<L: ReifySmem2D>(flat: usize) -> (usize, usize) {
    const {
        assert!(L::SHAPE.0 > 0 && L::SHAPE.1 > 0);
        assert!(
            (L::STRIDE.0 == 1 && L::STRIDE.1 == L::SHAPE.0)
                || (L::STRIDE.0 == L::SHAPE.1 && L::STRIDE.1 == 1),
            "idx2crd_2d requires a compact ColMajor or RowMajor 2D layout"
        );
    }
    if L::STRIDE.0 == 1 {
        let (r, c) = crate::idx2crd_col_major_2d(flat as u32, L::SHAPE.0 as u32);
        (r as usize, c as usize)
    } else {
        let (r, c) = crate::idx2crd_row_major_2d(flat as u32, L::SHAPE.1 as u32);
        (r as usize, c as usize)
    }
}

/// Bound the physical offsets of a reified two-dimensional layout at compile time.
///
/// Rejects invalid dimensions and offsets that cannot be represented by both
/// `usize` and the signed CuTe algebra. Swizzles can conservatively include
/// holes in this bound. `crd2idx_2d` instantiates this check before device code.
pub const fn max_offset_2d<L: ReifySmem2D>() -> usize {
    assert!(L::SHAPE.0 > 0 && L::SHAPE.1 > 0);
    assert!(L::STRIDE.0 >= 0 && L::STRIDE.1 >= 0 && L::OFFSET >= 0);
    let row_max = checked_offset_mul(parameter(L::SHAPE.0 - 1), parameter(L::STRIDE.0));
    let column_max = checked_offset_mul(parameter(L::SHAPE.1 - 1), parameter(L::STRIDE.1));
    let maximum = checked_offset_add(
        checked_offset_add(row_max, column_max),
        parameter(L::OFFSET),
    );
    swizzled_bound(maximum, L::NUM_BITS, L::NUM_BASE, L::NUM_SHIFT)
}
