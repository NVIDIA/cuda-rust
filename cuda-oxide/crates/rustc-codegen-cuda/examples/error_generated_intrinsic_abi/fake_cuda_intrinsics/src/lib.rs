/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]

/// Deliberately unsupported raw-intrinsic ABI used by the negative example.
pub mod __cuda_oxide_intrinsic_abi_v2 {
    #[inline(never)]
    pub fn i0001() -> u32 {
        unreachable!("fake ABI-v2 intrinsic executed")
    }
}
