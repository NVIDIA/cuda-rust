/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed scales passed to block-scaled MMA.

use core::marker::PhantomData;

/// The two [`UE8M0`](crate::numeric::UE8M0) scales used by one K=64 MXFP4 MMA.
///
/// Each scale covers 32 K values:
///
/// ```text
/// K:      0 ................ 31 | 32 ............... 63
/// scale:              s0       |              s1
/// ```
///
/// `Mode` keeps A scales ([`Mkl`](super::Mkl)) separate from B scales ([`Nkl`](super::Nkl)) so the MMA
/// call cannot swap them accidentally.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaScalePair<Mode> {
    /// Two UE8M0 scale bytes in the low 16 bits, in increasing K order.
    pub(super) bits: u32,
    /// Distinguishes A-side and B-side scale providers.
    pub(super) mode: PhantomData<Mode>,
}

impl<Mode> MmaScalePair<Mode> {
    /// Return the two packed bytes passed to the MMA instruction.
    ///
    /// Bits 7..0 scale K=0..31. Bits 15..8 scale K=32..63.
    #[must_use]
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }
}

/// Four neighboring K-group scales packed in one register.
///
/// ```text
/// [s0][s1][s2][s3]
///  └ pair 0 ┘└ pair 1 ┘
/// ```
///
/// [`Self::pair`] selects the two scales needed by one K=64 MMA. `Mode` keeps
/// A and B scale packs distinct.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalePack4<Mode> {
    /// Four UE8M0 bytes in increasing K-group order.
    bits: u32,
    /// Distinguishes M-row and N-row scale packs.
    mode: PhantomData<Mode>,
}

impl<Mode> ScalePack4<Mode> {
    /// Wrap four packed scale bytes without changing them.
    #[inline(always)]
    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self {
            bits,
            mode: PhantomData,
        }
    }

    #[cfg(test)]
    /// Return the packed bytes for host-side layout tests.
    #[inline(always)]
    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }

    /// Select scales 0 and 1 for `HALF = 0`, or 2 and 3 for `HALF = 1`.
    #[must_use]
    #[inline(always)]
    pub const fn pair<const HALF: usize>(self) -> MmaScalePair<Mode> {
        const { assert!(HALF < 2) };
        MmaScalePair {
            bits: (self.bits >> (HALF * 16)) & 0xffff,
            mode: PhantomData,
        }
    }

    /// Select either two-scale pair using a runtime index and no bounds check.
    ///
    /// # Safety
    ///
    /// `half` must be exactly `0` or `1`.
    #[inline(always)]
    pub(crate) const unsafe fn pair_at_unchecked(self, half: usize) -> MmaScalePair<Mode> {
        MmaScalePair {
            bits: (self.bits >> (half * 16)) & 0xffff,
            mode: PhantomData,
        }
    }
}

/// Four [`E4M3`](crate::numeric::E4M3) scales used by one K=64 NVFP4 MMA (`SF_VEC = 16`).
///
/// Each scale covers 16 consecutive K values:
///
/// ```text
/// K:      0..15 | 16..31 | 32..47 | 48..63
/// scale:   s0   |   s1   |   s2   |   s3
/// ```
///
/// `Mode` keeps A scales ([`Mkl`](super::Mkl)) separate from B scales ([`Nkl`](super::Nkl)).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmaScaleQuad<Mode> {
    /// Four E4M3 scale bytes packed in increasing K-group order.
    pub(super) bits: u32,
    /// Distinguishes A-side and B-side scale providers.
    pub(super) mode: PhantomData<Mode>,
}

impl<Mode> MmaScaleQuad<Mode> {
    /// Wrap four packed E4M3 scale bytes in increasing K order.
    #[must_use]
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self {
            bits,
            mode: PhantomData,
        }
    }

    /// Return the 32-bit packed scale word passed to the NVFP4 MMA instruction.
    #[must_use]
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.bits
    }
}
