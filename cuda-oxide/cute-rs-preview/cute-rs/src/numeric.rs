/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit storage and conversions for reduced-precision number formats.
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
//! Float4E2M1FN        u8   holds 2 FP4 values
//! Float8E8M0FNU       u8   holds 1 scale
//! Mxf4E2M1            f16  holds 4 FP4 values
//! UE8M0x4             u32  holds 4 scales
//! ```
//!
//! [`Float4E2M1FNx2`] and [`UE8M0x2`] represent values already loaded into a
//! register. Their conversion methods describe the numeric result without
//! naming a backend instruction. CuTe-aware compilers retain the enclosing
//! high-level operation and choose the implementation for their backend.

use crate::tensor::TensorElement;

/// One BF16 value stored in sixteen bits, distinct from FP16 and integer data.
///
/// Arithmetic requires explicit [`Self::to_f32`] widening and
/// [`Self::from_f32`] rounding. No arithmetic operators or implicit conversions
/// are supplied. Every bit pattern is valid, including signed zero and NaNs.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct BFloat16(#[doc = "The unchanged IEEE-style BF16 storage bits."] u16);

impl BFloat16 {
    /// Preserve an existing BF16 bit pattern without conversion.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Return the unchanged BF16 storage bits.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Round FP32 to BF16 using round-to-nearest, ties-to-even.
    ///
    /// Preserves signed zero and infinities; NaNs remain NaNs.
    #[must_use]
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Self(__compiler::cvt_f32_to_bf16(value))
    }

    /// Widen to FP32 exactly, retaining all BF16 payload bits.
    #[must_use]
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        cuda_device::convert::bf16_to_f32(self.0)
    }
}

impl TensorElement<BFloat16> for BFloat16 {}

/// Evaluate SiLU using stable FP32 sigmoid branches, then round once to BF16.
///
/// Uses non-FTZ approximate base-two exponential and FTZ approximate reciprocal.
/// This is a device operation; callers retain an explicit BF16 boundary when
/// composing it with a later product (for example, a gated activation).
#[must_use]
#[inline(always)]
pub fn silu_bf16(value: BFloat16) -> BFloat16 {
    let value = value.to_f32();
    let sigmoid = if value >= 0.0 {
        let e = crate::arch::exp2_f32(-value * core::f32::consts::LOG2_E);
        crate::arch::reciprocal_ftz_f32(1.0 + e)
    } else {
        let e = crate::arch::exp2_f32(value * core::f32::consts::LOG2_E);
        e * crate::arch::reciprocal_ftz_f32(1.0 + e)
    };
    BFloat16::from_f32(value * sigmoid)
}

// SAFETY: repr(transparent) over u16, no padding, references or invalid patterns.
#[cfg(feature = "host")]
unsafe impl cuda_core::DeviceCopy for BFloat16 {}

/// One logical `Float4E2M1FN` value, also called E2M1 or FP4.
///
/// ```text
/// 4 bits: [sign][exponent exponent][fraction]
/// ```
///
/// Each value uses four bits, so one `u8` storage byte holds two values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float4E2M1FN;

/// One unsigned `Float8E8M0FNU` scale, also called UE8M0.
///
/// ```text
/// 8 bits: [exponent only] ──► power-of-two scale
/// ```
///
/// It has no sign or fraction bits. The all-ones byte represents NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float8E8M0FNU;

/// One `Float8E4M3FN` scale, also called E4M3 (used by NVFP4 block scaling).
///
/// ```text
/// 8 bits: [sign][4-bit exponent][3-bit mantissa]
/// ```
///
/// Unlike [`Float8E8M0FNU`], E4M3 carries three fraction bits so each 16-element NVFP4
/// block scale can represent fractional steps such as `1.5` (`max_val / 6.0`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float8E4M3FN(#[doc = "Raw 8-bit E4M3FN storage byte."] u8);

impl Float8E4M3FN {
    /// Preserve an existing E4M3FN byte without conversion.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Return the raw 8-bit E4M3FN storage byte.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Round an `f32` value to `Float8E4M3FN` using round-to-nearest-even with finite saturation
    /// (CuTeDSL `val.to(cutlass.Float8E4M3FN)`).
    #[must_use]
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Self(__compiler::cvt_f32_to_e4m3(value))
    }

    /// Widen this `Float8E4M3FN` value to `f32` (CuTeDSL `val.to(cutlass.Float32)`).
    #[must_use]
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        __compiler::cvt_e4m3_to_f32(self.0)
    }
}

impl TensorElement<Float8E4M3FN> for Float8E4M3FN {}

/// Four neighboring [`Float8E8M0FNU`] scale bytes carried in one `u32`.
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

/// Four neighboring [`Float8E4M3FN`] scale bytes carried in one `u32` (covering K=64 at SF=16).
///
/// ```text
/// u32: [scale 3][scale 2][scale 1][scale 0]
///       K48..63  K32..47  K16..31   K0..15
/// ```
///
/// This type marks a shared-memory or global-memory view of four 16-element
/// NVFP4 block scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct E4M3x4;

/// Four neighboring [`Float4E2M1FN`] values packed for an MXFP4 MMA instruction.
///
/// The four nibbles use the bits of one `f16` storage value. It is only a
/// 16-bit box; its bits do **not** represent an FP16 number.
///
/// ```text
/// f16 bits: [FP4 3][FP4 2][FP4 1][FP4 0]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mxf4E2M1;

impl TensorElement<u8> for Float4E2M1FN {}
impl TensorElement<u8> for Float8E8M0FNU {}
impl TensorElement<u8> for Float8E4M3FN {}
impl TensorElement<f16> for Mxf4E2M1 {}
impl TensorElement<u32> for UE8M0x4 {}
impl TensorElement<u32> for E4M3x4 {}

/// Eight neighboring [`Float4E2M1FN`] values packed into one 32-bit word (`u32`).
///
/// ```text
/// u32: [FP4 7][FP4 6][FP4 5][FP4 4][FP4 3][FP4 2][FP4 1][FP4 0]
/// ```
///
/// Serves as the 32-bit register carrier for 8-element and 16-element
/// `TensorSSA<Float4E2M1FN, N>` vector conversions.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float4E2M1FNx8(#[doc = "Eight packed 4-bit E2M1 nibbles in increasing K order."] u32);

impl Float4E2M1FNx8 {
    /// Wrap 32 packed bits as eight E2M1 values without changing them.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Return the underlying 32-bit packed representation.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }
}

impl TensorElement<u32> for Float4E2M1FNx8 {}
impl TensorElement<Float4E2M1FNx8> for Float4E2M1FNx8 {}

/// Two [`Float4E2M1FN`] values packed into one byte.
///
/// ```text
/// bits 7..4 = value 1
/// bits 3..0 = value 0
/// ```
///
/// Every bit pattern is valid, including positive and negative zero.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float4E2M1FNx2(#[doc = "Low and high FP4 nibbles preserved in one byte."] u8);

impl Float4E2M1FNx2 {
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
        __compiler::cvt_e2m1x2_to_f32x2(self.0)
    }
}

/// Two [`Float8E8M0FNU`] scales packed into one 16-bit register.
///
/// ```text
/// bits 15..8 = scale 1
/// bits  7..0 = scale 0
/// ```
///
/// Every byte is valid. The byte `255` represents NaN.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UE8M0x2(#[doc = "Two exponent-only scales, with the first scale in the low byte."] u16);

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
        __compiler::cvt_ue8m0x2_to_f32x2(self.0)
    }
}

/// Two [`Float8E4M3FN`] scales packed into one 16-bit register.
///
/// ```text
/// bits 15..8 = scale 1 (K group 1)
/// bits  7..0 = scale 0 (K group 0)
/// ```
///
/// Magnitude `0x7f` (`0x7f` and `0xff`) represents NaN; all other bit patterns
/// decode to exact finite `f32` values in `[-448.0, 448.0]`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct E4M3x2(#[doc = "Two E4M3FN scales, with the first scale in the low byte."] u16);

impl E4M3x2 {
    /// Pack `low` as scale 0 and `high` as scale 1.
    #[must_use]
    #[inline(always)]
    pub const fn from_bytes(low: u8, high: u8) -> Self {
        Self((low as u16) | ((high as u16) << 8))
    }

    /// Treat `bits` as two packed E4M3 scales without changing them.
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

    /// Convert both E4M3 scales to `f32`, returning the low byte first.
    ///
    /// ```text
    /// packed E4M3 pair ──► decode two fractional scales ──► (f32, f32)
    /// ```
    #[must_use]
    #[inline(always)]
    pub fn to_f32x2(self) -> (f32, f32) {
        (
            Float8E4M3FN::from_bits(self.0 as u8).to_f32(),
            Float8E4M3FN::from_bits((self.0 >> 8) as u8).to_f32(),
        )
    }
}

/// Compiler hooks for reduced-precision scalar and register-tile conversions.
#[doc(hidden)]
pub mod __compiler {
    /// Convert one `f32` scalar into one `BFloat16` `u16` (`arith.truncf : f32 to bf16`).
    #[must_use]
    #[inline(never)]
    pub fn cvt_f32_to_bf16(value: f32) -> u16 {
        cuda_device::convert::f32_to_bf16_rne(value)
    }

    /// Convert one `f32` scalar into one `Float8E4M3FN` byte (`arith.truncf : f32 to f8E4M3FN`).
    #[must_use]
    #[inline(never)]
    pub fn cvt_f32_to_e4m3(value: f32) -> u8 {
        let _ = value;
        unreachable!("CuTe numeric conversion marker requires device compilation")
    }

    /// Widen one `Float8E4M3FN` byte into `f32` (`arith.extf : f8E4M3FN to f32`).
    #[must_use]
    #[inline(never)]
    pub fn cvt_e4m3_to_f32(bits: u8) -> f32 {
        let _ = bits;
        unreachable!("CuTe numeric conversion marker requires device compilation")
    }

    /// Widen one packed `Float4E2M1FNx2` byte into `(f32, f32)`.
    #[must_use]
    #[inline(never)]
    pub fn cvt_e2m1x2_to_f32x2(bits: u8) -> (f32, f32) {
        let _ = bits;
        unreachable!("CuTe numeric conversion marker requires device compilation")
    }

    /// Widen one packed `UE8M0x2` `u16` pair into `(f32, f32)`.
    #[must_use]
    #[inline(never)]
    pub fn cvt_ue8m0x2_to_f32x2(bits: u16) -> (f32, f32) {
        let _ = bits;
        unreachable!("CuTe numeric conversion marker requires device compilation")
    }

    /// Convert 8 `f32` scalars into 8 packed 4-bit `Float4E2M1FN` nibbles (`arith.truncf : vector<8xf32> to vector<8xf4E2M1FN>`).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    #[inline(never)]
    pub fn cvt_f32x8_to_e2m1x8(
        v0: f32,
        v1: f32,
        v2: f32,
        v3: f32,
        v4: f32,
        v5: f32,
        v6: f32,
        v7: f32,
    ) -> u32 {
        let _ = (v0, v1, v2, v3, v4, v5, v6, v7);
        unreachable!("CuTe numeric conversion marker requires device compilation")
    }
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;

    /// Packing and unpacking FP4 pairs must preserve all eight storage bits.
    #[test]
    fn e2m1_pair_preserves_all_bits() {
        for bits in 0u8..=u8::MAX {
            assert_eq!(Float4E2M1FNx2::from_bits(bits).to_bits(), bits);
        }
    }

    /// The low UE8M0 byte must store in the low half and the high byte in the high half.
    #[test]
    fn scale_pair_uses_low_byte_first() {
        let pair = UE8M0x2::from_bytes(0x7f, 0x80);
        assert_eq!(pair.to_bits(), 0x807f);
        assert_eq!(UE8M0x2::from_bits(0x807f), pair);
        let e4m3 = E4M3x2::from_bytes(0x38, 0x3c);
        assert_eq!(e4m3.to_bits(), 0x3c38);
        assert_eq!(E4M3x2::from_bits(0x3c38), e4m3);
    }
}
