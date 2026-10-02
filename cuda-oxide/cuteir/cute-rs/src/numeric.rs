/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small number formats used by block-scaled kernels.
//!
//! A logical value and the bits that carry it are sometimes different. For
//! example, two FP4 values share one byte:
//!
//! ```text
//! one byte
//! ┌────────────┬────────────┐
//! │ high FP4   │ low FP4    │
//! │ bits 7..4  │ bits 3..0  │
//! └────────────┴────────────┘
//! ```
//!
//! These types keep that meaning visible:
//!
//! ```text
//! logical value       storage bits
//! E2M1                u8   holds 2 FP4 values
//! UE8M0               u8   holds 1 scale
//! Mxf4E2M1            f16  holds 4 FP4 values
//! UE8M0x4             u32  holds 4 scales
//! ```
//!
//! [`PackedE2M1x2`] and [`UE8M0x2`] represent values already loaded into a
//! register. Their conversion methods describe the numeric result without
//! naming a backend instruction. CuTe-aware compilers retain the enclosing
//! high-level operation and choose the implementation for their backend.

use crate::tensor::TensorElement;

/// One logical `Float4E2M1FN` value, also called E2M1 or FP4.
///
/// ```text
/// 4 bits: [sign][exponent exponent][fraction]
/// ```
///
/// Each value uses four bits, so one `u8` storage byte holds two values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct E2M1;

/// One unsigned `Float8E8M0FNU` scale, also called UE8M0.
///
/// ```text
/// 8 bits: [exponent only] ──► power-of-two scale
/// ```
///
/// It has no sign or fraction bits. The all-ones byte represents NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UE8M0;

/// Four neighboring [`UE8M0`] scale bytes carried in one `u32`.
///
/// ```text
/// u32: [scale 3][scale 2][scale 1][scale 0]
///                                      └─ lowest K group
/// ```
///
/// This type marks a memory view. [`UE8M0x2`] is the two-scale register value
/// used by one-thread conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UE8M0x4;

/// Four neighboring [`E2M1`] values packed for an MXFP4 MMA instruction.
///
/// The four nibbles use the bits of one `f16` storage value. It is only a
/// 16-bit box; its bits do **not** represent an FP16 number.
///
/// ```text
/// f16 bits: [FP4 3][FP4 2][FP4 1][FP4 0]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mxf4E2M1;

impl TensorElement<u8> for E2M1 {}
impl TensorElement<u8> for UE8M0 {}
impl TensorElement<f16> for Mxf4E2M1 {}
impl TensorElement<u32> for UE8M0x4 {}

/// Two [`E2M1`] values packed into one byte.
///
/// ```text
/// bits 7..4 = value 1
/// bits 3..0 = value 0
/// ```
///
/// Every bit pattern is valid, including positive and negative zero.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedE2M1x2(u8);

impl PackedE2M1x2 {
    /// Treat `bits` as two packed E2M1 values without changing them.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Return the original packed byte.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert both FP4 values to `f32`, returning the low nibble first.
    ///
    /// ```text
    /// packed E2M1 byte ──► decode two logical values ──► (f32, f32)
    /// ```
    ///
    /// Both conversion steps are exact.
    #[must_use]
    #[inline(always)]
    pub fn to_f32x2(self) -> (f32, f32) {
        (e2m1_to_f32(self.0 & 0x0f), e2m1_to_f32(self.0 >> 4))
    }
}

/// Two [`UE8M0`] scales packed into one 16-bit register.
///
/// ```text
/// bits 15..8 = scale 1
/// bits  7..0 = scale 0
/// ```
///
/// Every byte is valid. The byte `255` represents NaN.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UE8M0x2(u16);

impl UE8M0x2 {
    /// Pack `low` as scale 0 and `high` as scale 1.
    #[must_use]
    #[inline(always)]
    pub const fn from_bytes(low: u8, high: u8) -> Self {
        Self((low as u16) | ((high as u16) << 8))
    }

    /// Treat `bits` as two packed scales without changing them.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Return the original packed 16 bits.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert both scales to `f32`, returning the low byte first.
    ///
    /// ```text
    /// packed UE8M0 pair ──► decode two powers of two ──► (f32, f32)
    /// ```
    ///
    /// UE8M0 values are powers of two, so both steps are exact. NaN stays NaN.
    #[must_use]
    #[inline(always)]
    pub fn to_f32x2(self) -> (f32, f32) {
        (
            ue8m0_to_f32(self.0 as u8),
            ue8m0_to_f32((self.0 >> 8) as u8),
        )
    }
}

#[inline(always)]
fn e2m1_to_f32(nibble: u8) -> f32 {
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[(nibble & 0x07) as usize];
    if nibble & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

#[inline(always)]
fn ue8m0_to_f32(byte: u8) -> f32 {
    match byte {
        // UE8M0 code zero denotes 2^-127, exactly halfway through the first
        // f32 subnormal exponent bin.
        0 => f32::from_bits(0x0040_0000),
        // The all-ones code is NaN. Keep a stable quiet payload without
        // requiring a backend conversion intrinsic.
        0xff => f32::from_bits(0x7fff_ffff),
        exponent => f32::from_bits((exponent as u32) << 23),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_pair_preserves_all_bits() {
        for bits in 0u8..=u8::MAX {
            assert_eq!(PackedE2M1x2::from_bits(bits).to_bits(), bits);
        }
    }

    #[test]
    fn scale_pair_uses_low_byte_first() {
        let pair = UE8M0x2::from_bytes(0x7f, 0x80);
        assert_eq!(pair.to_bits(), 0x807f);
        assert_eq!(UE8M0x2::from_bits(0x807f), pair);
    }

    #[test]
    fn e2m1_conversion_covers_every_code_bit_exactly() {
        const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, expected) in MAGNITUDES.into_iter().enumerate() {
            assert_eq!(e2m1_to_f32(code as u8).to_bits(), expected.to_bits());
            assert_eq!(
                e2m1_to_f32(code as u8 | 0x08).to_bits(),
                (-expected).to_bits()
            );
        }
    }

    #[test]
    fn ue8m0_conversion_handles_floor_normal_and_nan_codes() {
        assert_eq!(ue8m0_to_f32(0).to_bits(), 0x0040_0000);
        assert_eq!(ue8m0_to_f32(1).to_bits(), 0x0080_0000);
        assert_eq!(ue8m0_to_f32(126), 0.5);
        assert_eq!(ue8m0_to_f32(127), 1.0);
        assert_eq!(ue8m0_to_f32(128), 2.0);
        assert_eq!(ue8m0_to_f32(254).to_bits(), 0x7f00_0000);
        assert!(ue8m0_to_f32(255).is_nan());
    }
}
