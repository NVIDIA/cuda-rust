/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsics.
//!
//! These intrinsics provide access to PTX type conversion instructions that
//! are more efficient than scalar Rust casts.

include!("generated/convert.rs");

// =============================================================================
// Scalar conversion and packing helpers
// =============================================================================

/// Converts `f32` to the raw bits of a `bf16` by truncating the low 16 bits.
///
/// This is a bit-level truncation rather than a rounding conversion. In
/// particular, an `f32` NaN whose payload exists only in the discarded bits
/// can become a `bf16` infinity. Use [`f32_to_bf16_rne`] when IEEE-style
/// round-to-nearest-even behavior is required.
#[must_use]
#[inline(always)]
pub fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Converts `f32` to the raw bits of a `bf16` using round-to-nearest-even.
///
/// NaNs remain NaNs even when their payload is confined to the low 16 bits of
/// the `f32` representation.
#[must_use]
#[inline(always)]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7F80_0000;
    let mantissa = bits & 0x007F_FFFF;
    if exponent == 0x7F80_0000 && mantissa != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }

    let retained_lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + retained_lsb) >> 16) as u16
}

/// Widens raw `bf16` bits to `f32` by placing them in the high 16 bits.
///
/// This is the inverse of [`f32_to_bf16`] for values that are already exact
/// in bf16: the low 16 bits of the `f32` become zero.
#[must_use]
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Packs two raw `bf16` bit patterns into a `u32`, low half first.
#[must_use]
#[inline(always)]
pub const fn pack_bf16_pair(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

/// Packs two raw `f16` bit patterns into a `u32`, low half first.
#[must_use]
#[inline(always)]
pub const fn pack_f16_pair(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}

/// Truncates two `f32` values to `bf16` and packs them low half first.
#[must_use]
#[inline(always)]
pub fn f32_pair_to_packed_bf16(lo: f32, hi: f32) -> u32 {
    pack_bf16_pair(f32_to_bf16(lo), f32_to_bf16(hi))
}

// =============================================================================
// Packed f16x2 unpacking
// =============================================================================
//
// The generated conversions above all run f32 -> packed narrow type. The
// inverse is hand-written because there is nothing to generate it from: the
// pinned LLVM metadata has no f16x2-to-f32x2 intrinsic (every `ff2f16x2_*`
// record is the packing direction), and PTX has no single unpacking
// instruction either. Splitting a packed pair is a `mov.b32` into two 16-bit
// registers followed by two `cvt.f32.f16`, which is what the casts below
// lower to.
//
// These matter because packed f16 is the layout that gives the only reliably
// wide global loads: reading weights as `u32`/`u64` and unpacking in registers
// beats four scalar `f16` loads. Doing that inline at every call site is where
// the manual `from_bits`/shift/mask arithmetic came from.
//
// Names follow this module's generated convention, `cvt_<dst>_<src>`: the
// destination comes first, so `cvt_f32x2_f16x2` reads "two f32 from a packed
// f16x2".

/// Unpacks a packed `f16x2` into its two `f32` values, low half first.
///
/// This is the inverse of [`cvt_f16x2_f32`].
///
/// # Example
///
/// ```rust,ignore
/// // One 32-bit load carrying two f16 weights.
/// let packed = unsafe { *(ptr as *const u32) };
/// let (w0, w1) = convert::cvt_f32x2_f16x2(packed);
/// ```
#[must_use]
#[inline(always)]
pub fn cvt_f32x2_f16x2(packed: u32) -> (f32, f32) {
    let lo = f16::from_bits(packed as u16);
    let hi = f16::from_bits((packed >> 16) as u16);
    (lo as f32, hi as f32)
}

/// Unpacks the low half of a packed `f16x2` to `f32`.
///
/// Prefer [`cvt_f32x2_f16x2`] when both halves are needed; this exists for the
/// case where only one is, so the other conversion is not emitted at all.
#[must_use]
#[inline(always)]
pub fn cvt_f32_f16x2_lo(packed: u32) -> f32 {
    f16::from_bits(packed as u16) as f32
}

/// Unpacks the high half of a packed `f16x2` to `f32`.
///
/// See [`cvt_f32_f16x2_lo`].
#[must_use]
#[inline(always)]
pub fn cvt_f32_f16x2_hi(packed: u32) -> f32 {
    f16::from_bits((packed >> 16) as u16) as f32
}

/// Unpacks a packed `bf16x2` into its two `f32` values, low half first.
///
/// This is the inverse of [`cvt_bf16x2_f32`] or [`cvt_rz_bf16x2_f32`]. `bf16` shares `f32`'s exponent
/// range, so widening is an exact shift into the high half of the mantissa
/// rather than a conversion.
#[must_use]
#[inline(always)]
pub fn cvt_f32x2_bf16x2(packed: u32) -> (f32, f32) {
    (
        f32::from_bits(packed << 16),
        f32::from_bits(packed & 0xFFFF_0000),
    )
}

/// Unpacks the low half of a packed `bf16x2` to `f32`.
///
/// Prefer [`cvt_f32x2_bf16x2`] when both halves are needed; this exists for the
/// case where only one is, so the other conversion is not emitted at all.
#[must_use]
#[inline(always)]
pub fn cvt_f32_bf16x2_lo(packed: u32) -> f32 {
    f32::from_bits(packed << 16)
}

/// Unpacks the high half of a packed `bf16x2` to `f32`.
///
/// See [`cvt_f32_bf16x2_lo`].
#[must_use]
#[inline(always)]
pub fn cvt_f32_bf16x2_hi(packed: u32) -> f32 {
    f32::from_bits(packed & 0xFFFF_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the half ordering so the halves cannot silently swap:
    /// `cvt_f16x2_f32` documents the first argument as the low half.
    ///
    /// The generated packer is device-only and panics on the host, so the
    /// packed words here are hand-written constants rather than packer output.
    #[test]
    fn unpacks_halves_in_the_documented_order() {
        // 1.0 = 0x3C00, 2.0 = 0x4000 in f16.
        let packed = 0x4000_3C00;
        let (lo, hi) = cvt_f32x2_f16x2(packed);
        assert_eq!(lo, 1.0, "low half must come from the low 16 bits");
        assert_eq!(hi, 2.0, "high half must come from the high 16 bits");
        assert_eq!(cvt_f32_f16x2_lo(packed), 1.0);
        assert_eq!(cvt_f32_f16x2_hi(packed), 2.0);
    }

    /// A negative value in one half must not bleed into the other, which is
    /// what an arithmetic shift instead of a logical one would do.
    #[test]
    fn sign_bits_stay_in_their_own_half() {
        // -1.0 = 0xBC00 in the high half, 1.0 = 0x3C00 in the low half.
        let (lo, hi) = cvt_f32x2_f16x2(0xBC00_3C00);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, -1.0);
    }

    #[test]
    fn unpacks_bf16_pairs() {
        // bf16 is the top 16 bits of the f32 bit pattern.
        let one = 1.0f32.to_bits() >> 16; // 0x3F80
        let two = 2.0f32.to_bits() >> 16; // 0x4000
        let packed = (two << 16) | one;
        let (lo, hi) = cvt_f32x2_bf16x2(packed);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 2.0);
        assert_eq!(cvt_f32_bf16x2_lo(packed), 1.0);
        assert_eq!(cvt_f32_bf16x2_hi(packed), 2.0);
    }

    /// Same guard as the f16 sign test: a negative half must not bleed into
    /// the other half through the shift or the mask.
    #[test]
    fn bf16_sign_bits_stay_in_their_own_half() {
        // -1.0 = 0xBF80 in the high half, 1.0 = 0x3F80 in the low half.
        let packed = 0xBF80_3F80;
        let (lo, hi) = cvt_f32x2_bf16x2(packed);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, -1.0);
        assert_eq!(cvt_f32_bf16x2_lo(packed), 1.0);
        assert_eq!(cvt_f32_bf16x2_hi(packed), -1.0);
    }

    #[test]
    fn scalar_bf16_conversion_covers_rounding_and_special_values() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);

        // Exact tie with an even retained LSB rounds down; an odd one rounds up.
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16_rne(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16_rne(0.0), 0x0000);
        assert_eq!(f32_to_bf16_rne(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_rne(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16_rne(f32::NEG_INFINITY), 0xFF80);

        // A payload entirely below the bf16 boundary must not become infinity.
        let positive_nan = f32_to_bf16_rne(f32::from_bits(0x7F80_0001));
        let negative_nan = f32_to_bf16_rne(f32::from_bits(0xFF80_0001));
        assert_eq!(positive_nan & 0x7F80, 0x7F80);
        assert_ne!(positive_nan & 0x007F, 0);
        assert_eq!(negative_nan & 0xFF80, 0xFF80);
        assert_ne!(negative_nan & 0x007F, 0);
    }

    #[test]
    fn packing_helpers_keep_the_first_value_in_the_low_half() {
        assert_eq!(pack_bf16_pair(0x3F80, 0x4000), 0x4000_3F80);
        assert_eq!(pack_f16_pair(0x3C00, 0x4000), 0x4000_3C00);
        assert_eq!(f32_pair_to_packed_bf16(1.0, 2.0), 0x4000_3F80);
    }

    /// Every f16 bit pattern that is not a NaN must survive the trip,
    /// including subnormals, both zeroes, and both infinities.
    #[test]
    fn every_non_nan_f16_pattern_round_trips() {
        for bits in 0u16..=u16::MAX {
            let expected = f16::from_bits(bits);
            if expected.is_nan() {
                continue;
            }
            let (lo, _) = cvt_f32x2_f16x2(bits as u32);
            assert_eq!(
                lo, expected as f32,
                "low-half mismatch for f16 bits {bits:#06x}"
            );
            let (_, hi) = cvt_f32x2_f16x2((bits as u32) << 16);
            assert_eq!(
                hi, expected as f32,
                "high-half mismatch for f16 bits {bits:#06x}"
            );
        }
    }

    /// Every bf16 bit pattern must survive the trip through the f32 widen,
    /// NaNs included: `bf16_to_f32` places the bits in the high half and
    /// `f32_to_bf16` truncates them back out bit-identically.
    #[test]
    fn every_bf16_pattern_round_trips() {
        for bits in 0u16..=u16::MAX {
            assert_eq!(
                f32_to_bf16(bf16_to_f32(bits)),
                bits,
                "round-trip mismatch for bf16 bits {bits:#06x}"
            );
        }
    }
}
