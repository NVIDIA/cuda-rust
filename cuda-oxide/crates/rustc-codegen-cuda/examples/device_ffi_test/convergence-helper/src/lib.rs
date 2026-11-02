/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cuda_device::device]
unsafe extern "C" {
    // Deliberately private: the downstream generic instantiation must recover
    // convergence from dependency metadata without an exported declaration.
    #[cuda_device::convergent]
    fn warp_reduce_sum(value: f32) -> f32;
}

/// Reduce a warp and scale its sum in a downstream device instantiation.
///
/// # Safety
///
/// All lanes in a full warp must participate in the reduction.
#[cuda_device::device]
pub unsafe fn scaled_warp_sum<const SCALE: u32>(value: f32) -> f32 {
    unsafe { warp_reduce_sum(value) * SCALE as f32 }
}
