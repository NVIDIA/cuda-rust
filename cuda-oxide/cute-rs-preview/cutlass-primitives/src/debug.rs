/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit device execution failure for violated kernel contracts.

/// Abort device execution and report an interrupt to the host.
#[inline(always)]
pub fn trap() -> ! {
    cuda_device::debug::trap()
}
