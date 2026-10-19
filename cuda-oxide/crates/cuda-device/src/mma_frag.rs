/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fragment index algebra for the `m16n8k16` tensor-core accumulator.
//!
//! [`crate::wmma::mma_m16n8k16_f32_f16`] takes and returns `[f32; 4]`, and the
//! four values are *not* four adjacent elements of the output tile. Each lane
//! holds a scattered quarter of a 16x8 tile, and the required mapping is stated
//! in the intrinsic's docs as index arithmetic the caller has to reproduce
//! exactly. Getting it wrong does not fail: it computes a different, wrong
//! matrix product. This module derives the mapping once so call sites do not
//! have to.
//!
//! The mapping is not specific to the f16 instruction. Every `m16n8`
//! accumulator shape that hands each lane four 32-bit registers distributes
//! C/D the same way: [`crate::wmma::mma_m16n8k16_f32_bf16`],
//! [`crate::wmma::mma_m16n8k8_f32_tf32`], and
//! [`crate::wmma::mma_m16n8k32_s32_s8`] all document the identical C/D layout,
//! so this module's helpers apply to their accumulators unchanged.
//!
//! # The layout is a mixed-radix numeral
//!
//! For the `m16n8k16` f32 accumulator, PTX assigns lane `l` the elements
//!
//! ```text
//!     group = l / 4          tig = l % 4
//!     c[0], c[1]  ->  row = group,      col = 2*tig + {0, 1}
//!     c[2], c[3]  ->  row = group + 8,  col = 2*tig + {0, 1}
//! ```
//!
//! Substituting into the row-major offset `row * 8 + col` and grouping terms
//! shows this is a positional numeral:
//!
//! ```text
//!     offset = (group + 8*(j>>1)) * 8 + 2*tig + (j&1)
//!            = (j&1)*1 + tig*2 + group*8 + (j>>1)*64
//!
//!     digit    range   place value   source
//!     j & 1      2         1         fragment index, low bit
//!     tig        4         2         l % 4
//!     group      8         8         l / 4
//!     j >> 1     2        64         fragment index, high bit
//! ```
//!
//! Each place value is the product of the ranges of all lower digits: `1`, then
//! `1*2 = 2`, then `2*4 = 8`, then `8*8 = 64`, with total span `64*2 = 128`.
//! That is the condition for a mixed-radix system to be uniquely decodable, so
//! the map `(l, j) -> offset` is injective; since the domain has `32 * 4 = 128`
//! points and the tile has `16 * 8 = 128` elements, it is a bijection. The
//! warp's fragments therefore tile the output exactly, with no overlap and no
//! gap.
//!
//! The no-overlap half is the same guarantee that
//! [`crate::disjoint::DisjointSlice`] builds into the type system for parallel
//! writes; here it falls out of the numeral structure instead of thread
//! indexing. What makes this instance awkward to write by hand is that the
//! fragment index is *split across two non-adjacent digit positions*, at place
//! values 1 and 64, interleaved with the two lane digits.

/// Rows in the `m16n8k16` accumulator tile.
pub const ACC_ROWS: usize = 16;
/// Columns in the `m16n8k16` accumulator tile.
pub const ACC_COLS: usize = 8;
/// Accumulator registers each lane holds.
pub const ACC_PER_LANE: usize = 4;
/// Lanes participating in one `mma.sync`.
pub const WARP_LANES: usize = 32;

/// Row and column in the accumulator tile for one fragment register.
///
/// `lane` must be in `0..32` and `j` in `0..4`; both are reduced modulo their
/// range so an out-of-range argument stays inside the tile rather than producing
/// an offset past its end.
#[must_use]
#[inline(always)]
pub const fn acc_coords(lane: usize, j: usize) -> (usize, usize) {
    let lane = lane % WARP_LANES;
    let j = j % ACC_PER_LANE;
    let group = lane / 4;
    let tig = lane % 4;
    (group + 8 * (j >> 1), 2 * tig + (j & 1))
}

/// Row-major offset within the accumulator tile for one fragment register.
///
/// Equivalent to `row * ACC_COLS + col` from [`acc_coords`], written in the
/// numeral form derived in the module docs.
#[must_use]
#[inline(always)]
pub const fn acc_offset(lane: usize, j: usize) -> usize {
    let lane = lane % WARP_LANES;
    let j = j % ACC_PER_LANE;
    let group = lane / 4;
    let tig = lane % 4;
    // (j&1)*1 + tig*2 + group*8 + (j>>1)*64
    (j & 1) + tig * 2 + group * 8 + (j >> 1) * 64
}

/// Offset of a tile element within a larger row-major matrix.
///
/// `tile_row` and `tile_col` are the tile's origin in the matrix, and
/// `row_stride` the matrix's row length. This is the form GEMM epilogues need:
/// the accumulator tile is written into a strided destination rather than a
/// standalone 16x8 buffer.
///
/// Returns `None` on overflow so a bad stride cannot wrap into a plausible
/// offset.
#[must_use]
#[inline(always)]
pub fn acc_matrix_offset(
    lane: usize,
    j: usize,
    tile_row: usize,
    tile_col: usize,
    row_stride: usize,
) -> Option<usize> {
    let (row, col) = acc_coords(lane, j);
    row.checked_add(tile_row)?
        .checked_mul(row_stride)?
        .checked_add(col.checked_add(tile_col)?)
}

/// Recover `(lane, j)` from a tile offset.
///
/// The inverse of [`acc_offset`], and the reason the mapping is sound: it exists
/// precisely because the place values form the running product of the digit
/// ranges. Returns `None` for an offset outside the tile.
///
/// This is exported mainly so a caller can assert the round-trip in their own
/// tests, which is a cheap way to catch a hand-rolled layout that disagrees.
#[must_use]
#[inline(always)]
pub const fn acc_decode(offset: usize) -> Option<(usize, usize)> {
    if offset >= ACC_ROWS * ACC_COLS {
        return None;
    }
    // Read the digits back out, lowest place value first.
    let j_lo = offset % 2;
    let tig = (offset / 2) % 4;
    let group = (offset / 8) % 8;
    let j_hi = offset / 64;
    Some((group * 4 + tig, (j_hi << 1) | j_lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the module rests on: `(lane, j) -> offset` is a bijection onto
    /// the tile. Overlap would mean two lanes writing one element; a gap would
    /// mean an element never written.
    #[test]
    fn lane_fragment_map_is_a_bijection_onto_the_tile() {
        let mut owner = [None::<(usize, usize)>; ACC_ROWS * ACC_COLS];
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let offset = acc_offset(lane, j);
                assert!(
                    offset < owner.len(),
                    "offset {offset} outside tile for lane {lane}, j {j}"
                );
                assert_eq!(
                    owner[offset], None,
                    "element {offset} claimed by both {:?} and (lane {lane}, j {j})",
                    owner[offset]
                );
                owner[offset] = Some((lane, j));
            }
        }
        assert!(
            owner.iter().all(Option::is_some),
            "the warp's fragments must cover every element of the tile"
        );
    }

    /// `acc_offset` must agree with the row/col form, which is how the PTX docs
    /// state the layout. A disagreement means the numeral regrouping is wrong.
    #[test]
    fn numeral_form_agrees_with_row_column_form() {
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let (row, col) = acc_coords(lane, j);
                assert!(row < ACC_ROWS && col < ACC_COLS, "coords escape the tile");
                assert_eq!(
                    acc_offset(lane, j),
                    row * ACC_COLS + col,
                    "numeral and row-major forms disagree at lane {lane}, j {j}"
                );
            }
        }
    }

    /// Decoding is the inverse of encoding. This is the property that makes the
    /// place values correct: a mixed-radix numeral is decodable exactly when
    /// each place value is the running product of the lower ranges.
    #[test]
    fn decode_inverts_offset() {
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                assert_eq!(
                    acc_decode(acc_offset(lane, j)),
                    Some((lane, j)),
                    "round-trip failed for lane {lane}, j {j}"
                );
            }
        }
        assert_eq!(acc_decode(ACC_ROWS * ACC_COLS), None);
    }

    /// The two halves of the fragment index sit at place values 1 and 64, so
    /// `j` 0 and 1 differ by one column while `j` 0 and 2 differ by eight rows.
    /// Pinned because this is the part most easily written wrong by hand.
    #[test]
    fn fragment_index_splits_across_two_digit_positions() {
        for lane in 0..WARP_LANES {
            let (r0, c0) = acc_coords(lane, 0);
            let (r1, c1) = acc_coords(lane, 1);
            let (r2, c2) = acc_coords(lane, 2);
            // Low bit of j walks the column.
            assert_eq!((r1, c1), (r0, c0 + 1));
            // High bit of j jumps eight rows.
            assert_eq!((r2, c2), (r0 + 8, c0));
            assert_eq!(acc_offset(lane, 2) - acc_offset(lane, 0), 64);
        }
    }

    /// Placing the tile in a strided matrix must preserve disjointness, since
    /// that is how a GEMM epilogue actually writes it.
    #[test]
    fn matrix_placement_stays_disjoint() {
        let row_stride = 128;
        let (tile_row, tile_col) = (32, 16);
        let mut seen = [false; ACC_ROWS * 128];
        for lane in 0..WARP_LANES {
            for j in 0..ACC_PER_LANE {
                let off = acc_matrix_offset(lane, j, tile_row, tile_col, row_stride).unwrap();
                let local = off - tile_row * row_stride - tile_col;
                assert!(!seen[local], "collision in strided placement at {off}");
                seen[local] = true;
            }
        }
        assert_eq!(
            seen.iter().filter(|&&s| s).count(),
            WARP_LANES * ACC_PER_LANE
        );
    }

    #[test]
    fn matrix_offset_rejects_overflow() {
        assert_eq!(acc_matrix_offset(0, 0, usize::MAX, 0, 2), None);
        assert_eq!(acc_matrix_offset(0, 0, 1, usize::MAX, usize::MAX), None);
    }
}

// =============================================================================
// Predication against a logical tile extent
// =============================================================================
//
// A tile shape rarely divides the problem, and the accumulator makes the
// mismatch awkward: each lane's four values are scattered across the tile, so
// clipping is not a contiguous prefix the way it is for a linear tile. Lane 0
// may hold four valid elements while lane 4 holds two, at the same `j`.
//
// Concretely, at `M = 9` against the 16-row tile: `j` 0 and 1 land on row
// `group`, always inside; `j` 2 and 3 land on row `group + 8`, inside only for
// `group == 0`. So lanes 0..3 keep all four registers and every other lane
// keeps two - which is 4*4 + 28*2 = 72 = 9 * 8 elements, the whole logical tile
// and nothing more.
//
// The predicate is per register rather than per lane, so it is returned as a
// 4-bit mask. A kernel tests one mask instead of recomputing bounds per store.
// Computing the mask adds no divergent control flow (every lane evaluates the
// same straight-line expressions), but the masks themselves differ across
// lanes, so stores guarded on mask bits can still diverge exactly as any
// bounds check would. The mask buys fewer tests and a warp-uniform fast path
// when `acc_mask_is_uniform` holds for every lane; it does not by itself keep
// the warp convergent.

/// Whether one accumulator register lies inside a logical extent.
///
/// `valid_rows` and `valid_cols` are the real problem dimensions, which may be
/// smaller than the `16 x 8` tile the instruction computes. Values larger than
/// the tile are treated as the tile, so an over-large extent cannot admit an
/// element that does not exist.
#[must_use]
#[inline(always)]
pub const fn acc_is_valid(lane: usize, j: usize, valid_rows: usize, valid_cols: usize) -> bool {
    let rows = if valid_rows < ACC_ROWS {
        valid_rows
    } else {
        ACC_ROWS
    };
    let cols = if valid_cols < ACC_COLS {
        valid_cols
    } else {
        ACC_COLS
    };
    let (row, col) = acc_coords(lane, j);
    row < rows && col < cols
}

/// Which of a lane's four accumulator registers lie inside a logical extent.
///
/// Bit `j` is set when register `j` is valid. A mask of `0` means the lane holds
/// nothing worth storing.
///
/// # Example
///
/// ```rust,ignore
/// // M = 9 rows of a 16-row tile, all 8 columns.
/// let mask = mma_frag::acc_valid_mask(lane, 9, 8);
/// for j in 0..4 {
///     if mask & (1 << j) != 0 {
///         let (row, col) = mma_frag::acc_coords(lane, j);
///         // store d[j] at (row, col)
///     }
/// }
/// ```
#[must_use]
#[inline(always)]
pub const fn acc_valid_mask(lane: usize, valid_rows: usize, valid_cols: usize) -> u32 {
    let mut mask = 0u32;
    let mut j = 0;
    while j < ACC_PER_LANE {
        if acc_is_valid(lane, j, valid_rows, valid_cols) {
            mask |= 1 << j;
        }
        j += 1;
    }
    mask
}

/// Whether a lane's registers are either all valid or all invalid.
///
/// Useful for picking a store path: when every lane in the warp is uniform, the
/// epilogue can use unpredicated stores for the lanes that are wholly inside and
/// skip the rest, rather than testing per register.
#[must_use]
#[inline(always)]
pub const fn acc_mask_is_uniform(mask: u32) -> bool {
    mask == 0 || mask == 0b1111
}

/// Valid registers a whole warp holds for a logical extent.
///
/// Equals `min(valid_rows, 16) * min(valid_cols, 8)`, because the layout is a
/// bijection: every element of the logical tile is held by exactly one lane. Kept
/// as a computed sum so a wrong predicate shows up as a wrong total rather than
/// as missing output.
#[must_use]
#[inline(always)]
pub const fn acc_valid_count(valid_rows: usize, valid_cols: usize) -> usize {
    let mut total = 0;
    let mut lane = 0;
    while lane < WARP_LANES {
        total += acc_valid_mask(lane, valid_rows, valid_cols).count_ones() as usize;
        lane += 1;
    }
    total
}

#[cfg(test)]
mod predication_tests {
    use super::*;

    /// The shape this fork runs: 9 rows of a 16-row tile.
    #[test]
    fn nine_rows_keeps_four_registers_only_in_the_first_group() {
        // group 0 is lanes 0..3; row group + 8 = 8, still inside 9.
        for lane in 0..4 {
            assert_eq!(
                acc_valid_mask(lane, 9, 8),
                0b1111,
                "lane {lane} is wholly inside at M=9"
            );
        }
        // Every other lane loses the two registers on row group + 8.
        for lane in 4..WARP_LANES {
            assert_eq!(
                acc_valid_mask(lane, 9, 8),
                0b0011,
                "lane {lane} keeps only j 0 and 1 at M=9"
            );
        }
    }

    /// The count must equal the logical tile exactly: no element dropped, none
    /// invented. This is the property the bijection buys.
    #[test]
    fn valid_count_matches_the_logical_tile() {
        assert_eq!(acc_valid_count(9, 8), 72, "9 x 8");
        assert_eq!(acc_valid_count(16, 8), 128, "the whole tile");
        assert_eq!(acc_valid_count(1, 1), 1);
        assert_eq!(acc_valid_count(0, 8), 0);
        assert_eq!(acc_valid_count(8, 8), 64, "exactly half the rows");
        // Sweep every extent.
        for rows in 0..=ACC_ROWS {
            for cols in 0..=ACC_COLS {
                assert_eq!(
                    acc_valid_count(rows, cols),
                    rows * cols,
                    "mismatch at {rows} x {cols}"
                );
            }
        }
    }

    /// An extent beyond the tile must clamp, not admit elements the instruction
    /// never computes.
    #[test]
    fn oversized_extent_clamps_to_the_tile() {
        assert_eq!(acc_valid_count(999, 999), 128);
        assert_eq!(acc_valid_count(16, 99), 128);
        assert_eq!(acc_valid_count(99, 8), 128);
    }

    /// Each valid register maps to a distinct element of the logical tile, which
    /// is what makes a predicated store race-free: masking removes elements from
    /// a bijection and cannot create an overlap.
    #[test]
    fn predicated_registers_stay_disjoint() {
        let (rows, cols) = (9usize, 8usize);
        let mut owner = [None::<(usize, usize)>; ACC_ROWS * ACC_COLS];
        for lane in 0..WARP_LANES {
            let mask = acc_valid_mask(lane, rows, cols);
            for j in 0..ACC_PER_LANE {
                if mask & (1 << j) == 0 {
                    continue;
                }
                let (row, col) = acc_coords(lane, j);
                assert!(row < rows && col < cols, "masked-in element is outside");
                let off = row * ACC_COLS + col;
                assert_eq!(owner[off], None, "element {off} claimed twice");
                owner[off] = Some((lane, j));
            }
        }
        let claimed = owner.iter().filter(|o| o.is_some()).count();
        assert_eq!(claimed, rows * cols);
    }

    /// Column clipping bites differently from row clipping: columns come from
    /// `tig`, so a narrow extent removes whole lanes rather than registers.
    #[test]
    fn column_clipping_removes_lanes_not_registers() {
        // 2 valid columns: only tig == 0 contributes (cols 0 and 1).
        for lane in 0..WARP_LANES {
            let mask = acc_valid_mask(lane, 16, 2);
            if lane % 4 == 0 {
                assert_eq!(mask, 0b1111, "lane {lane} has tig 0");
            } else {
                assert_eq!(mask, 0, "lane {lane} is outside 2 columns");
            }
        }
        assert_eq!(acc_valid_count(16, 2), 32);
    }

    #[test]
    fn uniformity_predicate_matches_the_mask() {
        assert!(acc_mask_is_uniform(0));
        assert!(acc_mask_is_uniform(0b1111));
        assert!(!acc_mask_is_uniform(0b0011));
        assert!(!acc_mask_is_uniform(0b0001));
        // At M=9 lanes disagree, so the epilogue cannot use one uniform path.
        assert!(acc_mask_is_uniform(acc_valid_mask(0, 9, 8)));
        assert!(!acc_mask_is_uniform(acc_valid_mask(4, 9, 8)));
        // At the full tile every lane is uniform.
        for lane in 0..WARP_LANES {
            assert!(acc_mask_is_uniform(acc_valid_mask(lane, 16, 8)));
        }
    }
}
