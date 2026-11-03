/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native SM100 two-lane `f32` arithmetic.
//!
//! The generated raw operations carry each `f32x2` value in a `u64`. This
//! differs from [`crate::vector::F32x2`], which is an over-aligned memory
//! element. Pack/unpack helpers and [`Float2`] wrappers provide typed sugar on
//! top of that catalog-owned register representation.

use crate::Float2;

include!("generated/f32x2.rs");

/// Pack two scalar f32 values into the PTX f32x2 register representation.
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> u64 {
    u64::from(lo.to_bits()) | (u64::from(hi.to_bits()) << 32)
}

/// Return the low scalar lane from a packed PTX f32x2 register.
#[inline(always)]
pub fn unpack_f32x2_lo(value: u64) -> f32 {
    f32::from_bits(value as u32)
}

/// Return the high scalar lane from a packed PTX f32x2 register.
#[inline(always)]
pub fn unpack_f32x2_hi(value: u64) -> f32 {
    f32::from_bits((value >> 32) as u32)
}

/// Add the two scalar lanes of a packed PTX f32x2 register.
#[inline(always)]
pub fn horizontal_add_f32x2_packed(value: u64) -> f32 {
    unpack_f32x2_lo(value) + unpack_f32x2_hi(value)
}

/// Two-lane round-to-nearest-even fused multiply-add (`a * b + c`).
#[inline(always)]
pub fn fma_float2(a: Float2, b: Float2, c: Float2) -> Float2 {
    // PTX models f32x2 as one 64-bit register pair. Keep that representation
    // at the native arithmetic boundary so ptxas can reuse the pairs directly.
    let result = fma_f32x2(
        pack_f32x2(a.x(), a.y()),
        pack_f32x2(b.x(), b.y()),
        pack_f32x2(c.x(), c.y()),
    );
    Float2::new([unpack_f32x2_lo(result), unpack_f32x2_hi(result)])
}

/// Two-lane round-to-nearest-even addition.
#[inline(always)]
pub fn add_float2(a: Float2, b: Float2) -> Float2 {
    let result = add_f32x2(pack_f32x2(a.x(), a.y()), pack_f32x2(b.x(), b.y()));
    Float2::new([unpack_f32x2_lo(result), unpack_f32x2_hi(result)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_carrier_preserves_all_lane_bits() {
        // Include signed zero, subnormal boundaries, infinities, and NaN payloads.
        let patterns = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7f81_2345,
            0xffc5_4321,
        ];
        for lo in patterns {
            for hi in patterns {
                let packed = pack_f32x2(f32::from_bits(lo), f32::from_bits(hi));
                assert_eq!(packed, u64::from(lo) | (u64::from(hi) << 32));
                assert_eq!(unpack_f32x2_lo(packed).to_bits(), lo);
                assert_eq!(unpack_f32x2_hi(packed).to_bits(), hi);
            }
        }
    }

    #[test]
    fn horizontal_sum_uses_both_scalar_lanes() {
        assert_eq!(horizontal_add_f32x2_packed(pack_f32x2(1.25, -2.5)), -1.25);
        assert_eq!(
            horizontal_add_f32x2_packed(pack_f32x2(-0.0, -0.0)).to_bits(),
            0x8000_0000
        );
    }
}
