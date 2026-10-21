/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Allocation-free evaluation of static CuTe shape/stride markers.
//!
//! The leftmost shape mode changes fastest, as in `cute_layout::Layout`.
//! This bounded, nonnegative-stride subset is suitable for device address maps;
//! the complete algebra and validation remain in `cute-layout`.
use crate::markers::{C, ColMajor, Composed, L, ReifySmem2D, RowMajor, Swizzle, T2};

/// A static layout mapping a flat logical coordinate to an element offset.
pub trait StaticLayout {
    /// Logical element count. Invalid shapes are rejected at instantiation.
    const SIZE: usize;
    /// Upper bound on every in-domain physical element offset.
    ///
    /// Affine layouts have an exact maximum; swizzled layouts may conservatively
    /// include unused offsets. The bound fits both `usize` and positive `i64`.
    const MAX_OFFSET: usize;
    /// Evaluate a coordinate in the layout's domain, without accessing memory.
    /// Callers that use the offset for memory access must bound the coordinate.
    fn offset(flat: usize) -> usize;
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
    let _ = cute_layout::Swizzle::new(bits, base, shift);
    if bits == 0 {
        return maximum;
    }
    let target_end = base + bits + if shift < 0 { (-shift) as u32 } else { 0 };
    let lower_mask = (1_u64 << target_end) - 1;
    let result = maximum as u64 | lower_mask;
    assert!(result as u128 <= usize::MAX as u128 && result <= i64::MAX as u64);
    result as usize
}

impl<const N: i64, const S: i64> StaticLayout for L<C<N>, C<S>> {
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        parameter(N)
    };
    const MAX_OFFSET: usize = {
        assert!(N > 0 && S >= 0);
        checked_offset_mul(parameter(N - 1), parameter(S))
    };
    #[inline(always)]
    fn offset(flat: usize) -> usize {
        let _ = Self::SIZE;
        flat * S as usize
    }
}
impl<A, B, SA, SB> StaticLayout for L<T2<A, B>, T2<SA, SB>>
where
    L<A, SA>: StaticLayout,
    L<B, SB>: StaticLayout,
{
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        match L::<A, SA>::SIZE.checked_mul(L::<B, SB>::SIZE) {
            Some(n) => n,
            None => panic!("static layout extent overflow"),
        }
    };
    const MAX_OFFSET: usize = checked_offset_add(L::<A, SA>::MAX_OFFSET, L::<B, SB>::MAX_OFFSET);
    #[inline(always)]
    fn offset(flat: usize) -> usize {
        let _ = Self::SIZE;
        L::<A, SA>::offset(flat % L::<A, SA>::SIZE) + L::<B, SB>::offset(flat / L::<A, SA>::SIZE)
    }
}
impl<const R: i64, const K: i64> StaticLayout for RowMajor<R, K> {
    const SIZE: usize = <L<T2<C<R>, C<K>>, T2<C<K>, C<1>>>>::SIZE;
    const MAX_OFFSET: usize = <L<T2<C<R>, C<K>>, T2<C<K>, C<1>>>>::MAX_OFFSET;
    #[inline(always)]
    fn offset(flat: usize) -> usize {
        <L<T2<C<R>, C<K>>, T2<C<K>, C<1>>>>::offset(flat)
    }
}
impl<const R: i64, const K: i64> StaticLayout for ColMajor<R, K> {
    const SIZE: usize = <L<T2<C<R>, C<K>>, T2<C<1>, C<R>>>>::SIZE;
    const MAX_OFFSET: usize = <L<T2<C<R>, C<K>>, T2<C<1>, C<R>>>>::MAX_OFFSET;
    #[inline(always)]
    fn offset(flat: usize) -> usize {
        <L<T2<C<R>, C<K>>, T2<C<1>, C<R>>>>::offset(flat)
    }
}
impl<const B: u32, const M: u32, const S: i32, const O: i64, Inner: StaticLayout> StaticLayout
    for Composed<Swizzle<B, M, S>, O, Inner>
{
    const SIZE: usize = {
        let _ = Self::MAX_OFFSET;
        Inner::SIZE
    };
    const MAX_OFFSET: usize =
        swizzled_bound(checked_offset_add(Inner::MAX_OFFSET, parameter(O)), B, M, S);
    #[inline(always)]
    fn offset(flat: usize) -> usize {
        let _ = Self::SIZE;
        cute_layout::Swizzle::new(B, M, S).apply((Inner::offset(flat) + O as usize) as i64) as usize
    }
}

/// Evaluate a typed two-dimensional shared-memory layout, including its swizzle.
///
/// Offsets and strides count elements. The static offset is added before the
/// swizzle, matching `ComposedLayout`. This computes an offset only; callers
/// that dereference it must establish coordinate bounds and storage lifetime.
#[inline(always)]
pub fn offset_2d<L: ReifySmem2D>(row: usize, column: usize) -> usize {
    const {
        let _ = max_offset_2d::<L>();
    }
    let plain = row * L::ROW_STRIDE as usize + column * L::COL_STRIDE as usize + L::OFFSET as usize;
    cute_layout::Swizzle::new(L::SWIZZLE_B, L::SWIZZLE_M, L::SWIZZLE_S).apply(plain as i64) as usize
}

/// Bound the physical offsets of a reified two-dimensional layout at compile time.
///
/// Rejects invalid dimensions and offsets that cannot be represented by both
/// `usize` and the signed CuTe algebra. Swizzles can conservatively include
/// holes in this bound. `offset_2d` instantiates this check before device code.
pub const fn max_offset_2d<L: ReifySmem2D>() -> usize {
    assert!(L::ROWS > 0 && L::COLS > 0);
    assert!(L::ROW_STRIDE >= 0 && L::COL_STRIDE >= 0 && L::OFFSET >= 0);
    let row_max = checked_offset_mul(parameter(L::ROWS - 1), parameter(L::ROW_STRIDE));
    let column_max = checked_offset_mul(parameter(L::COLS - 1), parameter(L::COL_STRIDE));
    let maximum = checked_offset_add(
        checked_offset_add(row_max, column_max),
        parameter(L::OFFSET),
    );
    swizzled_bound(maximum, L::SWIZZLE_B, L::SWIZZLE_M, L::SWIZZLE_S)
}
