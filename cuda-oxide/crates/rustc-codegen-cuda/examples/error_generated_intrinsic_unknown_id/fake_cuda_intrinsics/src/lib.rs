/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]

/// Deliberately unknown ID in the otherwise supported ABI-v1 namespace.
pub mod __cuda_oxide_intrinsic_abi_v1 {
    #[inline(never)]
    pub fn i9999() -> u32 {
        unreachable!("fake unknown intrinsic executed")
    }
}
