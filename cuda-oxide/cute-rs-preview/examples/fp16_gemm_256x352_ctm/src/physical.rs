// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! FP16 output swizzle shared by the device store path and exhaustive CPU layout checks.

#[cfg(test)]
mod tests {
    //! Enumerate output positions to check complete coverage and contiguous vector-store chunks.

    use super::*;
    /// Check every output element appears once and every eight-lane chunk stays contiguous under the swizzle.
    #[test]
    fn output_swizzle_covers_each_element_once_and_preserves_eight_element_chunks() {
        let mut seen = [false; 128 * 32];
        for row in 0..128 {
            for col in 0..32 {
                let offset = output_offset(row, col);
                assert!(offset < seen.len());
                assert!(!seen[offset], "duplicate location at ({row},{col})");
                seen[offset] = true;
                assert_eq!(offset % 8, col as usize % 8);
                assert_eq!(offset, output_offset(row, col / 8 * 8) + col as usize % 8);
            }
        }
        assert!(seen.into_iter().all(|used| used));
        assert_eq!(output_offset(2, 0), 72);
        assert_eq!(output_offset(7, 24), 224);
    }
}

/// B64 swizzle in FP16 element units. Kernel callers use row<128, column<32.
#[inline(always)]
pub const fn output_offset(row: u32, column: u32) -> usize {
    let plain = row * 32 + column;
    (plain ^ ((plain >> 3) & 0x18)) as usize
}
