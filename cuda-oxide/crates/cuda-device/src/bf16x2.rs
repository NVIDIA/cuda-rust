// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two bf16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.
//!
//! [`Bf16x2`] is the typed form of that `u32`. It has one method for each raw
//! function here and for each packed bf16 conversion in [`crate::convert`], so
//! a kernel can take `&[Bf16x2]` instead of `&[u32]`. The raw functions stay
//! available and return the same bits.
//!
//! ```rust,ignore
//! use cuda_device::bf16x2::Bf16x2;
//!
//! #[kernel]
//! pub fn axpy(mut y: DisjointSlice<Bf16x2>, x: &[Bf16x2], a: f32) {
//!     let idx = thread::index_1d();
//!     let i = idx.get();
//!     if let Some(o) = y.get_mut(idx) {
//!         *o = Bf16x2::from_f32_pair_rne(a, a).fma(x[i], *o);
//!     }
//! }
//! ```
//!
//! The packing here is the ALU format for SIMD arithmetic. For over-aligned
//! *memory element* types that make loads and stores single wide transactions,
//! see [`crate::vector`]. For multi-register *value* groups, see
//! [`crate::cusimd::CuSimd`].
//!
//! See also [`crate::bf16`] for the scalar min/max forms that operate on a
//! single bfloat16 rather than a packed pair.
//!
//! # `fma` modifiers
//!
//! Only `fma_bf16x2` and `fma_relu_bf16x2` exist. The `ftz` and `sat` forms
//! that [`crate::f16x2`] provides are absent from the PTX ISA for `bf16x2`, not
//! merely unimplemented here: ptxas rejects `fma.rn.ftz.bf16x2` with "Illegal
//! modifier `.ftz` for instruction `fma`" while accepting
//! `fma.rn.relu.bf16x2`. LLVM declares the missing forms as intrinsics, so they
//! fail at instruction selection rather than at assembly.

use crate::bf16::Bf16;

/// Two bfloat16 values packed in one `u32`.
///
/// `Bf16x2` has the size and alignment of `u32`. The first value uses the low
/// 16 bits and the second value uses the high 16 bits, which is the packing
/// the raw `bf16x2` functions use. Every bit pattern is a valid value.
/// [`Default`] is two positive zeroes.
///
/// The methods wrap the raw `u32` functions in this module and in
/// [`crate::convert`] one-to-one and return the same bits. Methods marked pure
/// run on the host and on the device. Methods marked device-only call a PTX
/// instruction and panic when executed on the host.
///
/// There are no `Add`, `Sub`, `Mul`, or `Neg` operators. [`add`](Self::add),
/// [`sub`](Self::sub), and [`mul`](Self::mul) require `sm_90`, and an operator
/// would hide that requirement.
///
/// # Comparison
///
/// `Bf16x2` does not implement `PartialEq` or `PartialOrd`. A derived
/// comparison would compare bits, so a NaN would equal itself and `0.0` would
/// differ from `-0.0`. Compare [`to_f32_pair`](Self::to_f32_pair) results for
/// numeric equality and [`to_bits`](Self::to_bits) results for bit equality.
///
/// # Host buffers
///
/// `Bf16x2` implements [`DeviceRepr<u32>`](crate::DeviceRepr), so cuda-core's
/// `DeviceBuffer::wrap_elem` views a `DeviceBuffer<u32>` as a
/// `DeviceBuffer<Bf16x2>` and `peel_elem` turns it back.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Bf16x2(u32);

// SAFETY: `Bf16x2` is `#[repr(transparent)]` over `u32` with no other fields,
// so it has the layout and the valid values of `u32`. This lets a
// `DeviceBuffer<u32>` be viewed as `DeviceBuffer<Bf16x2>` with cuda-core's
// `DeviceBuffer::wrap_elem` and turned back with `peel_elem`. Neither type
// needs destruction or owns resources.
unsafe impl cuda_repr::DeviceRepr<u32> for Bf16x2 {}

// SAFETY: a transparent wrapper over one `u32`, so there is no padding and
// every bit pattern, all zeroes included, is a valid value.
unsafe impl bytemuck::Zeroable for Bf16x2 {}
unsafe impl bytemuck::Pod for Bf16x2 {}

// `add`, `sub`, `mul`, and `neg` are methods and not operator traits because
// an operator would hide the target requirement of the instruction.
#[allow(clippy::should_implement_trait)]
impl Bf16x2 {
    /// Builds a pair from its raw packed bit pattern.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw packed bit pattern.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Packs two values, `lo` in the low 16 bits and `hi` in the high 16 bits.
    ///
    /// Wraps [`crate::convert::pack_bf16_pair`]. Pure: runs on the host and on
    /// the device.
    #[must_use]
    #[inline(always)]
    pub const fn new(lo: Bf16, hi: Bf16) -> Self {
        Self(crate::convert::pack_bf16_pair(lo.to_bits(), hi.to_bits()))
    }

    /// Returns the first value, from the low 16 bits.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn lo(self) -> Bf16 {
        Bf16::from_bits(self.0 as u16)
    }

    /// Returns the second value, from the high 16 bits.
    ///
    /// Pure: runs on the host and on the device.
    #[must_use]
    #[inline(always)]
    pub const fn hi(self) -> Bf16 {
        Bf16::from_bits((self.0 >> 16) as u16)
    }

    /// Converts two `f32` values by truncation and packs them, `lo` first.
    ///
    /// Wraps [`crate::convert::f32_pair_to_packed_bf16`]. Each value keeps its
    /// high 16 bits; this is not a rounding conversion. Pure: runs on the host
    /// and on the device.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair(lo: f32, hi: f32) -> Self {
        Self(crate::convert::f32_pair_to_packed_bf16(lo, hi))
    }

    /// Converts two `f32` values with round-to-nearest-even and packs them,
    /// `lo` first.
    ///
    /// Wraps [`crate::convert::cvt_bf16x2_f32`]. Device-only: panics on the
    /// host. Requires `sm_80`.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_rne(lo: f32, hi: f32) -> Self {
        Self(crate::convert::cvt_bf16x2_f32(lo, hi))
    }

    /// Converts two `f32` values with round-toward-zero and packs them, `lo`
    /// first.
    ///
    /// Wraps [`crate::convert::cvt_rz_bf16x2_f32`]. Device-only: panics on the
    /// host. Requires `sm_80`.
    #[must_use]
    #[inline(always)]
    pub fn from_f32_pair_rz(lo: f32, hi: f32) -> Self {
        Self(crate::convert::cvt_rz_bf16x2_f32(lo, hi))
    }

    /// Widens both values to `f32`, first value first. The conversion is
    /// exact.
    ///
    /// Wraps [`crate::convert::cvt_f32x2_bf16x2`]. Pure: runs on the host and
    /// on the device.
    #[must_use]
    #[inline(always)]
    pub fn to_f32_pair(self) -> (f32, f32) {
        crate::convert::cvt_f32x2_bf16x2(self.0)
    }

    /// Takes the absolute value of both values.
    ///
    /// Wraps [`abs_bf16x2`]. Device-only: panics on the host. Requires `sm_80`.
    /// The operation clears the sign bits, so there is no rounding.
    #[must_use]
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(abs_bf16x2(self.0))
    }

    /// Negates both values.
    ///
    /// Wraps [`neg_bf16x2`]. Device-only: panics on the host. Requires `sm_80`.
    /// The operation flips the sign bits, so there is no rounding.
    #[must_use]
    #[inline(always)]
    pub fn neg(self) -> Self {
        Self(neg_bf16x2(self.0))
    }

    /// Adds `other` to `self` in each half, with round-to-nearest-even.
    ///
    /// Wraps [`add_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_90`.
    #[must_use]
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(add_bf16x2(self.0, other.0))
    }

    /// Subtracts `other` from `self` in each half, with round-to-nearest-even.
    ///
    /// Wraps [`sub_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_90`.
    #[must_use]
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(sub_bf16x2(self.0, other.0))
    }

    /// Multiplies `self` by `other` in each half, with round-to-nearest-even.
    ///
    /// Wraps [`mul_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_90`.
    #[must_use]
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        Self(mul_bf16x2(self.0, other.0))
    }

    /// Computes `self * b + c` in each half as a fused multiply-add, with one
    /// round-to-nearest-even at the end.
    ///
    /// Wraps [`fma_bf16x2`] as `fma_bf16x2(self, b, c)`. Device-only: panics on
    /// the host. Requires `sm_80`.
    #[must_use]
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        Self(fma_bf16x2(self.0, b.0, c.0))
    }

    /// Computes `self * b + c` in each half as a fused multiply-add with
    /// round-to-nearest-even, then replaces a negative result with zero.
    ///
    /// Wraps [`fma_relu_bf16x2`] as `fma_relu_bf16x2(self, b, c)`. Device-only:
    /// panics on the host. Requires `sm_80`.
    #[must_use]
    #[inline(always)]
    pub fn fma_relu(self, b: Self, c: Self) -> Self {
        Self(fma_relu_bf16x2(self.0, b.0, c.0))
    }

    /// Returns the smaller value in each half. If one input of a half is NaN, the
    /// result is the other input.
    ///
    /// Wraps [`min_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(min_bf16x2(self.0, other.0))
    }

    /// Returns the larger value in each half. If one input of a half is NaN, the
    /// result is the other input.
    ///
    /// Wraps [`max_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(max_bf16x2(self.0, other.0))
    }

    /// Returns the smaller value in each half, or a canonical NaN in a half
    /// where either input is NaN.
    ///
    /// Wraps [`min_nan_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_nan(self, other: Self) -> Self {
        Self(min_nan_bf16x2(self.0, other.0))
    }

    /// Returns the larger value in each half, or a canonical NaN in a half
    /// where either input is NaN.
    ///
    /// Wraps [`max_nan_bf16x2`] with `self` as the first argument. Device-only:
    /// panics on the host. Requires `sm_80`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_nan(self, other: Self) -> Self {
        Self(max_nan_bf16x2(self.0, other.0))
    }

    /// Returns the smaller magnitude in each half, with the sign set to the XOR
    /// of the input signs.
    ///
    /// Wraps [`min_xorsign_abs_bf16x2`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_xorsign_abs(self, other: Self) -> Self {
        Self(min_xorsign_abs_bf16x2(self.0, other.0))
    }

    /// Returns the larger magnitude in each half, with the sign set to the XOR
    /// of the input signs.
    ///
    /// Wraps [`max_xorsign_abs_bf16x2`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_xorsign_abs(self, other: Self) -> Self {
        Self(max_xorsign_abs_bf16x2(self.0, other.0))
    }

    /// The [`min_xorsign_abs`](Self::min_xorsign_abs) operation with the NaN
    /// rule of [`min_nan`](Self::min_nan): a NaN input gives a NaN result in
    /// that half.
    ///
    /// Wraps [`min_nan_xorsign_abs_bf16x2`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn min_nan_xorsign_abs(self, other: Self) -> Self {
        Self(min_nan_xorsign_abs_bf16x2(self.0, other.0))
    }

    /// The [`max_xorsign_abs`](Self::max_xorsign_abs) operation with the NaN
    /// rule of [`max_nan`](Self::max_nan): a NaN input gives a NaN result in
    /// that half.
    ///
    /// Wraps [`max_nan_xorsign_abs_bf16x2`] with `self` as the first argument.
    /// Device-only: panics on the host. Requires `sm_86`. No rounding.
    #[must_use]
    #[inline(always)]
    pub fn max_nan_xorsign_abs(self, other: Self) -> Self {
        Self(max_nan_xorsign_abs_bf16x2(self.0, other.0))
    }
}

include!("generated/bf16x2.rs");
