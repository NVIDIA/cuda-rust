/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar `bf16` min/max intrinsics.
//!
//! The raw functions carry each value as the `u16` bit pattern of one bfloat16,
//! matching how [`crate::convert`] already carries packed 16-bit conversion
//! operands. [`Bf16`] is the typed form of the same value: it wraps the `u16`
//! and has one method for each raw function, so a kernel can take `&[Bf16]`
//! instead of `&[u16]`. The raw functions stay available and return the same
//! bits. For the SIMD forms that operate on two bfloat16 values in one `u32`,
//! see [`crate::bf16x2`].
//!
//! ```rust,ignore
//! use cuda_device::bf16::Bf16;
//!
//! #[kernel]
//! pub fn scale(mut out: DisjointSlice<Bf16>, x: &[Bf16], s: f32) {
//!     let idx = thread::index_1d();
//!     let i = idx.get();
//!     if let Some(o) = out.get_mut(idx) {
//!         *o = Bf16::from_f32_rne(x[i].to_f32() * s);
//!     }
//! }
//! ```
//!
//! Only the forms without `ftz` are available. LLVM declares
//! `llvm.nvvm.fmin.ftz.bf16` and its relatives, but the NVPTX backend has no
//! selection pattern for them and instruction selection fails, so they are not
//! exposed here. The `NaN` modifier returns a canonical NaN when either input is
//! NaN instead of the numeric operand, and `xorsign.abs` compares magnitudes and
//! takes the sign from the XOR of the input signs. The plain and `NaN` forms
//! require `sm_80` and PTX ISA 7.0; the `xorsign.abs` forms require `sm_86` and
//! PTX ISA 7.2.
//!
//! The canonical NaN the `NaN` forms return is `0x7FFF`, not the IEEE default
//! quiet NaN `0x7FC0`: PTX canonicalizes to an all-ones exponent and mantissa
//! with a clear sign. Measured on `sm_86`; the `generated_intrinsics` example
//! asserts it.

/// One bfloat16 value.
///
/// `Bf16` has the size and alignment of `u16` and holds the bit pattern of one
/// bfloat16: the high 16 bits of the `f32` with the same sign and exponent.
/// Every bit pattern is a valid value. [`Default`] is positive zero.
///
/// The methods wrap the raw `u16` functions in this module and in
/// [`crate::convert`] one-to-one and return the same bits. The conversions and
/// the bit accessors are pure: they run on the host and on the device. The
/// min/max methods are device-only: each one calls a PTX instruction and
/// panics when executed on the host.
///
/// # Comparison
///
/// `Bf16` does not implement `PartialEq` or `PartialOrd`. A derived comparison
/// would compare bits, so a NaN would equal itself and `0.0` would differ from
/// `-0.0`. Compare [`to_f32`](Self::to_f32) results for numeric equality and
/// [`to_bits`](Self::to_bits) results for bit equality.
///
/// # Host buffers
///
/// `Bf16` implements [`DeviceRepr<u16>`](crate::DeviceRepr), so cuda-core's
/// `DeviceBuffer::wrap_elem` views a `DeviceBuffer<u16>` as a
/// `DeviceBuffer<Bf16>` and `peel_elem` turns it back.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Bf16(u16);

// SAFETY: `Bf16` is `#[repr(transparent)]` over `u16` with no other fields, so
// it has the layout and the valid values of `u16`. This lets a
// `DeviceBuffer<u16>` be viewed as `DeviceBuffer<Bf16>` with cuda-core's
// `DeviceBuffer::wrap_elem` and turned back with `peel_elem`. Neither type
// needs destruction or owns resources.
unsafe impl cuda_repr::DeviceRepr<u16> for Bf16 {}

// SAFETY: a transparent wrapper over one `u16`, so there is no padding and
// every bit pattern, all zeroes included, is a valid value.
unsafe impl bytemuck::Zeroable for Bf16 {}
unsafe impl bytemuck::Pod for Bf16 {}

impl Bf16 {
    /// Builds a value from its raw bfloat16 bit pattern.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bfloat16 bit pattern.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an `f32` by truncation: keeps the high 16 bits and discards
    /// the low 16 bits.
    ///
    /// Wraps [`crate::convert::f32_to_bf16`]. This is not a rounding
    /// conversion, and an `f32` NaN whose payload is only in the discarded bits
    /// becomes an infinity. Use [`from_f32_rne`](Self::from_f32_rne) for
    /// round-to-nearest-even. Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub fn from_f32(value: f32) -> Self {
        Self(crate::convert::f32_to_bf16(value))
    }

    /// Converts an `f32` with round-to-nearest-even. A NaN stays a NaN.
    ///
    /// Wraps [`crate::convert::f32_to_bf16_rne`]. Pure: runs on the host and on
    /// the device.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_rne(value: f32) -> Self {
        Self(crate::convert::f32_to_bf16_rne(value))
    }

    /// Widens the value to `f32`. The conversion is exact.
    ///
    /// Wraps [`crate::convert::bf16_to_f32`]. Pure: runs on the host and on the
    /// device.
    #[must_use]
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        crate::convert::bf16_to_f32(self.0)
    }

    /// Returns the smaller of `self` and `other`. If one input is NaN, the
    /// result is the other input.
    ///
    /// Wraps [`min_bf16`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. The result is one of the inputs,
    /// so there is no rounding.
    #[must_use]
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(min_bf16(self.0, other.0))
    }

    /// Returns the larger of `self` and `other`. If one input is NaN, the
    /// result is the other input.
    ///
    /// Wraps [`max_bf16`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. The result is one of the inputs,
    /// so there is no rounding.
    #[must_use]
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(max_bf16(self.0, other.0))
    }

    /// Returns the smaller of `self` and `other`, or the canonical NaN
    /// `0x7FFF` if either input is NaN.
    ///
    /// Wraps [`min_nan_bf16`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_nan(self, other: Self) -> Self {
        Self(min_nan_bf16(self.0, other.0))
    }

    /// Returns the larger of `self` and `other`, or the canonical NaN `0x7FFF`
    /// if either input is NaN.
    ///
    /// Wraps [`max_nan_bf16`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_nan(self, other: Self) -> Self {
        Self(max_nan_bf16(self.0, other.0))
    }

    /// Returns the smaller magnitude of `self` and `other`, with the sign set
    /// to the XOR of the input signs.
    ///
    /// Wraps [`min_xorsign_abs_bf16`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_xorsign_abs(self, other: Self) -> Self {
        Self(min_xorsign_abs_bf16(self.0, other.0))
    }

    /// Returns the larger magnitude of `self` and `other`, with the sign set
    /// to the XOR of the input signs.
    ///
    /// Wraps [`max_xorsign_abs_bf16`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_xorsign_abs(self, other: Self) -> Self {
        Self(max_xorsign_abs_bf16(self.0, other.0))
    }

    /// The [`min_xorsign_abs`](Self::min_xorsign_abs) operation with the NaN
    /// rule of [`min_nan`](Self::min_nan): a NaN input gives a NaN result.
    ///
    /// Wraps [`min_nan_xorsign_abs_bf16`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_nan_xorsign_abs(self, other: Self) -> Self {
        Self(min_nan_xorsign_abs_bf16(self.0, other.0))
    }

    /// The [`max_xorsign_abs`](Self::max_xorsign_abs) operation with the NaN
    /// rule of [`max_nan`](Self::max_nan): a NaN input gives a NaN result.
    ///
    /// Wraps [`max_nan_xorsign_abs_bf16`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_nan_xorsign_abs(self, other: Self) -> Self {
        Self(max_nan_xorsign_abs_bf16(self.0, other.0))
    }
}

include!("generated/bf16.rs");
