// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Explicit physical addresses for this kernel, in bytes unless named otherwise.

/// Logical N-column offsets of the eight accumulator fragment slots.
pub const N_OFFSETS: [usize; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
/// Return the packed u32 word index within a scale atom from a row's low seven bits.
#[inline(always)]
pub const fn scale_word(row: usize) -> usize {
    // Only the lowest seven bits participate, even for a full-matrix row.
    let row = row as u32;
    ((row & 31) * 4 + ((row >> 5) & 3)) as usize
}
/// Return a scale byte offset for a full-matrix row and K-group index in the blocked atom layout.
#[inline(always)]
pub const fn scale_offset(row: usize, group: usize, groups: usize) -> usize {
    (row / 128) * (groups / 4) * 512 + (group / 4) * 512 + scale_word(row) * 4 + group % 4
}
/// Byte offset for `atom < 8`, `half < 2`, and `lane < 32`.
#[inline(always)]
pub const fn matrix_row(atom: usize, half: usize, lane: usize) -> usize {
    let atom = atom as u32;
    let half = half as u32;
    let lane = lane as u32;
    let sub = lane / 8;
    let row = 16 * atom + 8 * (sub % 2) + lane % 8;
    let base = row * 64 + 16 * (sub / 2);
    // The K64 half selects bit 5 independently of the lane/atom swizzle.
    ((base ^ ((base & 0x180) >> 3)) ^ (half * 32)) as usize
}
/// Element offset for `row < 128` and `column < 128`.
#[inline(always)]
pub const fn output_element(row: usize, column: usize) -> usize {
    let row = row as u32;
    let column = column as u32;
    let plain = row * 64 + column % 64;
    ((column / 64) * 8192 + (plain ^ ((plain & 0x1c0) >> 3))) as usize
}
/// Element offset for `warp < 8`, `lane < 32`, `band < 2`, and `slot < 8`.
#[cfg(test)]
#[inline(always)]
pub const fn output_row(warp: usize, lane: usize, band: usize, slot: usize) -> usize {
    let warp = warp as u32;
    let lane = lane as u32;
    let band = band as u32;
    let row = 16 * (warp % 4) + 64 * band + (lane & 15);
    let column = 16 * (warp / 4) + N_OFFSETS[slot] as u32;
    output_element(row as usize, column as usize)
}

#[cfg(test)]
mod tests {
    //! Exhaustive CPU checks for bounded scale, matrix-load, and accumulator address calculations.

    use super::*;

    /// Check narrowed arithmetic matches the original usize layout throughout each legal coordinate domain.
    #[test]
    fn bounded_addresses_match_the_original_usize_mapping() {
        /// Independent usize output-element calculation used to validate the narrowed implementation.
        fn reference_output(row: usize, column: usize) -> usize {
            let plain = row * 64 + column % 64;
            (column / 64) * 8192 + (plain ^ ((plain & 0x1c0) >> 3))
        }

        for row in 0..2048 {
            assert_eq!(scale_word(row), (row & 31) * 4 + ((row >> 5) & 3));
        }
        // Truncation must also preserve scale_word's full usize domain.
        for low in 0..128 {
            let row = (usize::MAX & !127) | low;
            assert_eq!(scale_word(row), (row & 31) * 4 + ((row >> 5) & 3));
        }
        for atom in 0..8 {
            for half in 0..2 {
                for lane in 0..32 {
                    let sub = lane / 8;
                    let row = 16 * atom + 8 * (sub % 2) + lane % 8;
                    let column = 16 * half + 8 * (sub / 2);
                    let plain = row * 64 + column * 2;
                    assert_eq!(matrix_row(atom, half, lane), plain ^ ((plain & 0x180) >> 3));
                }
            }
        }
        for row in 0..128 {
            for column in 0..128 {
                assert_eq!(output_element(row, column), reference_output(row, column));
            }
        }
        for warp in 0..8 {
            for lane in 0..32 {
                for band in 0..2 {
                    for (slot, offset) in N_OFFSETS.into_iter().enumerate() {
                        let row = 16 * (warp % 4) + 64 * band + (lane & 15);
                        let column = 16 * (warp / 4) + offset;
                        assert_eq!(
                            output_row(warp, lane, band, slot),
                            reference_output(row, column)
                        );
                    }
                }
            }
        }
    }

    /// Check scale atoms are byte-complete bijections with adjacent bytes for four K groups.
    #[test]
    fn scale_atoms_cover_every_byte_once_and_keep_four_k_groups_adjacent() {
        let mut seen = [false; 512];
        for row in 0..128 {
            for group in 0..4 {
                let offset = scale_word(row) * 4 + group;
                assert!(offset < 512 && !seen[offset]);
                seen[offset] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
        assert_eq!((scale_word(0), scale_word(32), scale_word(1)), (0, 1, 4));
        assert_eq!(scale_offset(128, 4, 8), 1536);
    }
    /// Check each K64 half provides distinct in-bounds 16-byte-aligned matrix-load rows.
    #[test]
    fn ldmatrix_addresses_cover_each_k64_half_with_aligned_rows() {
        for atom in 0..8 {
            for half in 0..2 {
                let mut seen = [false; 512];
                for lane in 0..32 {
                    let offset = matrix_row(atom, half, lane);
                    assert!(offset < 8192 && offset.is_multiple_of(16));
                    assert!(!seen[offset / 16]);
                    seen[offset / 16] = true;
                }
            }
        }
        assert_eq!(matrix_row(0, 0, 1), 64);
        assert_eq!(matrix_row(0, 1, 0), 32);
    }
    /// Check all warp/lane/register fragments cover the full output tile exactly once.
    #[test]
    fn accumulator_mapping_covers_the_output_tile_exactly_once() {
        let mut seen = [false; 128 * 128];
        for warp in 0..8 {
            for lane in 0..32 {
                for band in 0..2 {
                    for (slot, offset) in N_OFFSETS.into_iter().enumerate() {
                        for register in 0..4 {
                            let row = 16 * (warp % 4) + 64 * band + lane / 4 + 8 * (register / 2);
                            let col = 16 * (warp / 4) + offset + 2 * (lane % 4) + register % 2;
                            assert!(row < 128 && col < 128);
                            let physical = output_element(row, col);
                            assert!(physical < seen.len() && !seen[physical]);
                            seen[physical] = true;
                        }
                        // STMATRIX address-provider lanes name the start of each
                        // eight-element row, in the independently computed swizzle.
                        let row = 16 * (warp % 4) + 64 * band + (lane & 15);
                        let col = 16 * (warp / 4) + offset;
                        assert_eq!(output_row(warp, lane, band, slot), output_element(row, col));
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }
}
