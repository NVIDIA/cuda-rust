/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Register reduction states with explicit tie and invalid-input semantics.

/// A maximum with its original coordinate, choosing the lowest index on ties.
///
/// Nonfinite observations make validity sticky-false. The numeric candidate may
/// still change, so consumers must check [`Self::is_valid`] before using it.
/// An empty state has index `u32::MAX`; callers must supply at least one value
/// when they require a valid in-bounds index.
#[derive(Clone, Copy, Debug)]
pub struct ArgMax {
    value: f32,
    index: u32,
    valid: bool,
}

impl ArgMax {
    /// Identity state for a finite-value argmax reduction.
    #[must_use]
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            value: f32::NEG_INFINITY,
            index: u32::MAX,
            valid: true,
        }
    }

    /// Include a value and its original coordinate without reassociation.
    #[must_use]
    #[inline(always)]
    pub fn observe(mut self, value: f32, index: u32) -> Self {
        self.valid = self.valid && value.to_bits() & 0x7f80_0000 != 0x7f80_0000;
        if value > self.value || (value == self.value && index < self.index) {
            self.value = value;
            self.index = index;
        }
        self
    }

    /// Combine two partitions, retaining original indices and invalid status.
    #[must_use]
    #[inline(always)]
    pub fn merge(mut self, other: Self) -> Self {
        self.valid = self.valid && other.valid;
        if other.value > self.value || (other.value == self.value && other.index < self.index) {
            self.value = other.value;
            self.index = other.index;
        }
        self
    }

    /// Selected numeric candidate; meaningful as a finite maximum when valid.
    #[must_use]
    #[inline(always)]
    pub const fn value(self) -> f32 {
        self.value
    }

    /// Original coordinate of the selected candidate, or `u32::MAX` if empty.
    #[must_use]
    #[inline(always)]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Whether every observed input was finite (also true for an empty state).
    #[must_use]
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        self.valid
    }

    /// Reduce one state per lane in XOR order 16, 8, 4, 2, 1.
    ///
    /// # Safety
    /// All 32 lanes must execute convergently, each passing its actual lane ID
    /// in `0..32`. No lane may exit before the final full-warp validity vote.
    #[must_use]
    #[inline(always)]
    pub unsafe fn reduce_warp(mut self, lane: u32) -> Self {
        let mut offset = 16;
        while offset != 0 {
            let value = ctm_rs::warp::shuffle_xor_f32(self.value, offset);
            // SAFETY: the full-warp caller contract includes every source lane.
            let index = unsafe { ctm_rs::warp::shuffle_sync(u32::MAX, self.index, lane ^ offset) };
            // Validity is reduced separately, exactly once, after candidates.
            self = self.merge(Self {
                value,
                index,
                valid: true,
            });
            offset /= 2;
        }
        // SAFETY: all lanes reach this vote under the caller contract.
        self.valid = unsafe { ctm_rs::warp::all(self.valid) };
        self
    }
}

/// Select the first maximum of a BF16 tensor using one complete warp.
///
/// Values are partitioned cyclically over 32 lanes; each lane visits its values
/// in ascending coordinate order before the fixed XOR reduction. All lanes
/// receive the same result. Any nonfinite input invalidates that result.
///
/// # Safety
/// All 32 lanes must execute convergently with the same input view and their
/// actual lane IDs. The tensor length must fit in `u32`; consumers must reject
/// an empty or invalid result before using its index as an address.
#[must_use]
#[inline(always)]
pub unsafe fn argmax_bf16<L: crate::tensor::partition::IndexLayout>(
    input: crate::tensor::partition::StridedTensor<'_, crate::numeric::Bf16, L>,
    lane: u32,
) -> ArgMax {
    let values = input.partition::<32>(lane as usize);
    let mut index = lane;
    let local = values.fold(ArgMax::empty(), |state, value| {
        let state = state.observe(value.to_f32(), index);
        // The terminal increment is not used as an address and may wrap for a
        // u32-sized input. Each observed coordinate itself remains in bounds.
        index = index.wrapping_add(32);
        state
    });
    // SAFETY: the caller provides the full-warp participation/lane contract.
    unsafe { local.reduce_warp(lane) }
}
