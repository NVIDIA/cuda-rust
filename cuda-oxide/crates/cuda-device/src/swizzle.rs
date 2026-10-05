/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! XOR swizzles for shared memory, and the bank-conflict arithmetic to pick one.
//!
//! Shared memory is 32 banks of 4 bytes, and the bank is
//! `(byte_offset / 4) % 32`. A warp that reads one column of a row-major tile
//! whose row stride is a multiple of 32 elements has every lane land on the same
//! bank, which serialises the access 32 ways:
//!
//! ```text
//! column access, f32 tile          worst conflict   banks used
//! row stride 32                    32-way            1 of 32
//! row stride 64                    32-way            1 of 32
//! row stride 16                    16-way            2 of 32
//! row stride 33 (padded)            1-way           32 of 32
//! ```
//!
//! Padding the stride to 33 fixes it and wastes a column per row, which also
//! breaks the alignment that wide accesses need. The alternative is to permute
//! the offsets instead of spacing them out: XOR part of the address with bits
//! taken from higher up, so lanes that shared a bank no longer do.
//!
//! ```text
//! row stride 32 with a swizzle     worst conflict   banks used
//! Swizzle<5, 0, 5>                  1-way           32 of 32
//! Swizzle<3, 0, 5>                  4-way            8 of 32
//! Swizzle<2, 0, 5>                  8-way            4 of 32
//! ```
//!
//! Fewer scattered bits leaves proportionally more conflict, so five bits -
//! exactly the 32 banks - is the choice worth defaulting to. Combinations
//! that would not be self-inverse, such as `Swizzle<5, 0, 4>`, are rejected
//! at compile time rather than silently misbehaving.
//!
//! # Relation to the index algebra
//!
//! A swizzle is not an affine sum of digits, so it cannot be written as a
//! mixed-radix numeral and the compact-layout argument says nothing about it
//! directly. It composes with that argument anyway, for a simpler reason: a
//! swizzle is a **bijection** on the index space, and applying a bijection to a
//! set of pairwise-disjoint indices leaves them pairwise disjoint. So swizzling
//! a disjoint access pattern is always safe, and the only thing to get right is
//! that reads and writes use the same swizzle.
//!
//! This module is analysis and address arithmetic only. It does not allocate or
//! access shared memory; combine it with [`crate::shared::SharedArray`].

/// Shared memory banks.
pub const BANKS: usize = 32;
/// Bytes per bank in one cycle.
pub const BANK_WIDTH_BYTES: usize = 4;
/// Threads whose accesses are serviced together.
pub const WARP_LANES: usize = 32;

/// An XOR swizzle: `x ^ (((x >> S) & ((1 << B) - 1)) << M)`.
///
/// `B` bits are taken from bit `S` upward and XORed into the address starting at
/// bit `M`. Larger `B` scatters across more banks; `S` selects which part of the
/// address drives the scatter, and is normally the log2 of the row stride so
/// that the row index does the scattering.
///
/// # Choosing parameters
///
/// For a row-major tile of `f32` with a row stride of `2^S` elements, and a warp
/// reading a column, `Swizzle<5, 0, S>` is conflict-free: five bits is exactly
/// the 32 banks. Fewer bits leaves proportionally more conflict, which the table
/// in the module docs shows and the tests pin.
///
/// # Constraint
///
/// `M + B <= S` is required, and checked at compile time. Without it the XOR
/// target bits overlap the bits being read, and the swizzle stops being its own
/// inverse - so a store and a load through the same swizzle would disagree.
/// Some rejected combinations are not even bijections: at `M == S` the sourced
/// bits are XORed with themselves and zeroed outright, collapsing distinct
/// offsets. An involution is what makes the type usable as "apply on the way
/// in, apply again on the way out".
///
/// # Relation to CuTe's `Swizzle`
///
/// CuTe's `Swizzle<B, M, S>` measures its shift `S` relative to the base `M`
/// (source bits start at `M + S`), while this type's `S` is the absolute
/// source-bit position. A CuTe `Swizzle<B, M, S>` is this type's
/// `Swizzle<B, M, M + S>`; conversely, `Swizzle<B, M, S>` here corresponds to
/// CuTe's `Swizzle<B, M, S - M>`. The conventions coincide only at `M == 0`,
/// so translate the third parameter explicitly when porting a CuTe layout.
pub struct Swizzle<const B: usize, const M: usize, const S: usize>;

impl<const B: usize, const M: usize, const S: usize> Swizzle<B, M, S> {
    /// Compile-time check that the swizzle is an involution.
    const INVOLUTION: () = assert!(
        M + B <= S,
        "Swizzle<B, M, S> requires M + B <= S, otherwise applying it twice does \
         not restore the original offset"
    );

    /// Bits scattered.
    pub const BITS: usize = B;
    /// Distinct offsets the swizzle can map onto, from one source group.
    pub const SPREAD: usize = 1 << B;

    /// Apply the swizzle to an element offset.
    ///
    /// Self-inverse: `apply(apply(x)) == x`.
    #[must_use]
    #[inline(always)]
    pub const fn apply(offset: usize) -> usize {
        // Force the involution check; it is a compile error if M + B > S.
        let () = Self::INVOLUTION;
        offset ^ (((offset >> S) & ((1 << B) - 1)) << M)
    }

    /// Undo the swizzle. Identical to [`Self::apply`], since it is an involution.
    ///
    /// Provided under its own name so call sites read in the direction they mean.
    #[must_use]
    #[inline(always)]
    pub const fn undo(offset: usize) -> usize {
        Self::apply(offset)
    }
}

/// Bank an element offset lands in.
#[must_use]
#[inline(always)]
pub const fn bank_of(elem_offset: usize, elem_size: usize) -> usize {
    (elem_offset * elem_size / BANK_WIDTH_BYTES) % BANKS
}

/// Worst-case serialisation for the offsets one warp accesses.
///
/// Returns the largest number of lanes landing on a single bank: `1` is
/// conflict-free, `32` is fully serialised. Lanes hitting the *same address*
/// still count here, even though the hardware broadcasts them, so this is a
/// conservative bound rather than an exact cycle count.
#[must_use]
pub const fn conflict_degree(offsets: &[usize; WARP_LANES], elem_size: usize) -> usize {
    let mut counts = [0usize; BANKS];
    let mut lane = 0;
    while lane < WARP_LANES {
        let b = bank_of(offsets[lane], elem_size);
        counts[b] += 1;
        lane += 1;
    }
    let mut worst = 0;
    let mut b = 0;
    while b < BANKS {
        if counts[b] > worst {
            worst = counts[b];
        }
        b += 1;
    }
    worst
}

/// Distinct banks the offsets touch.
#[must_use]
pub const fn banks_used(offsets: &[usize; WARP_LANES], elem_size: usize) -> usize {
    let mut seen = [false; BANKS];
    let mut lane = 0;
    while lane < WARP_LANES {
        seen[bank_of(offsets[lane], elem_size)] = true;
        lane += 1;
    }
    let mut n = 0;
    let mut b = 0;
    while b < BANKS {
        if seen[b] {
            n += 1;
        }
        b += 1;
    }
    n
}

/// Offsets a warp touches reading one column of a row-major tile.
///
/// Lane `i` reads row `i`, the pattern that conflicts worst and the reason
/// swizzling exists.
#[must_use]
pub const fn column_offsets(row_stride_elems: usize, col: usize) -> [usize; WARP_LANES] {
    let mut out = [0usize; WARP_LANES];
    let mut lane = 0;
    while lane < WARP_LANES {
        out[lane] = lane * row_stride_elems + col;
        lane += 1;
    }
    out
}

/// Offsets a warp touches reading one row.
///
/// Contiguous, so it is conflict-free without a swizzle. Provided to check that
/// a swizzle chosen for the column case does not spoil the row case.
#[must_use]
pub const fn row_offsets(row: usize, row_stride_elems: usize) -> [usize; WARP_LANES] {
    let mut out = [0usize; WARP_LANES];
    let mut lane = 0;
    while lane < WARP_LANES {
        out[lane] = row * row_stride_elems + lane;
        lane += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: usize = 4;

    /// Apply a swizzle across a whole offset array.
    fn swizzled<const B: usize, const M: usize, const S: usize>(
        offsets: [usize; WARP_LANES],
    ) -> [usize; WARP_LANES] {
        let mut out = [0usize; WARP_LANES];
        for (o, &i) in out.iter_mut().zip(offsets.iter()) {
            *o = Swizzle::<B, M, S>::apply(i);
        }
        out
    }

    /// The problem, quantified. A column of a stride-32 f32 tile puts every lane
    /// on one bank.
    #[test]
    fn column_access_conflicts_without_a_swizzle() {
        for stride in [32usize, 64, 128] {
            let o = column_offsets(stride, 0);
            assert_eq!(
                conflict_degree(&o, F32),
                32,
                "stride {stride} should be fully serialised"
            );
            assert_eq!(banks_used(&o, F32), 1);
        }
        // A stride of 16 halves it, and 33 avoids it by wasting a column.
        assert_eq!(conflict_degree(&column_offsets(16, 0), F32), 16);
        assert_eq!(conflict_degree(&column_offsets(33, 0), F32), 1);
    }

    /// The fix, quantified, and the reason to prefer five bits.
    #[test]
    fn swizzle_removes_the_column_conflict() {
        let base = column_offsets(32, 0);
        assert_eq!(conflict_degree(&swizzled::<5, 0, 5>(base), F32), 1);
        assert_eq!(banks_used(&swizzled::<5, 0, 5>(base), F32), 32);
        // Fewer scattered bits leaves proportionally more conflict.
        assert_eq!(conflict_degree(&swizzled::<3, 0, 5>(base), F32), 4);
        assert_eq!(conflict_degree(&swizzled::<2, 0, 5>(base), F32), 8);
    }

    /// A swizzle must not spoil the access it was not chosen for.
    #[test]
    fn row_access_stays_conflict_free() {
        for row in [0usize, 1, 7, 31] {
            let o = row_offsets(row, 32);
            assert_eq!(conflict_degree(&o, F32), 1, "row {row} unswizzled");
            assert_eq!(
                conflict_degree(&swizzled::<5, 0, 5>(o), F32),
                1,
                "row {row} swizzled"
            );
        }
    }

    /// The property that lets a swizzle compose with a disjoint access pattern:
    /// it is a permutation, so it cannot make two distinct offsets collide.
    #[test]
    fn swizzle_is_a_bijection() {
        const SPAN: usize = 1 << 10;
        let mut seen = [false; SPAN];
        for x in 0..SPAN {
            let y = Swizzle::<5, 0, 5>::apply(x);
            assert!(y < SPAN, "{x} mapped outside the span, to {y}");
            assert!(!seen[y], "two offsets collided on {y}");
            seen[y] = true;
        }
        assert!(seen.iter().all(|&s| s), "the map must be onto");
    }

    /// Self-inverse, which is what makes "apply going in, apply coming out"
    /// correct. The `M + B <= S` constraint is what guarantees it, and it is
    /// enforced at compile time.
    #[test]
    fn swizzle_is_its_own_inverse() {
        for x in 0..(1usize << 10) {
            assert_eq!(Swizzle::<5, 0, 5>::undo(Swizzle::<5, 0, 5>::apply(x)), x);
            assert_eq!(Swizzle::<3, 0, 5>::undo(Swizzle::<3, 0, 5>::apply(x)), x);
            assert_eq!(Swizzle::<2, 1, 5>::undo(Swizzle::<2, 1, 5>::apply(x)), x);
        }
    }

    #[test]
    fn bank_arithmetic_wraps_at_thirty_two() {
        assert_eq!(bank_of(0, F32), 0);
        assert_eq!(bank_of(1, F32), 1);
        assert_eq!(bank_of(31, F32), 31);
        assert_eq!(bank_of(32, F32), 0, "wraps after 32 words");
        // An f64 element spans two words, so consecutive elements skip a bank.
        assert_eq!(bank_of(1, 8), 2);
        assert_eq!(bank_of(16, 8), 0);
    }

    /// Usable in const position, so a conflict-free choice can be asserted at
    /// compile time rather than measured in a profiler.
    #[test]
    fn analysis_runs_at_compile_time() {
        const OFFSETS: [usize; WARP_LANES] = column_offsets(32, 0);
        const UNSWIZZLED: usize = conflict_degree(&OFFSETS, F32);
        const _: () = assert!(UNSWIZZLED == 32);
        assert_eq!(UNSWIZZLED, 32);
    }
}
