/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-fail probe: require an unsafe block for device-only descriptor prefetch.
use cute_rs::TmaDesc;

/// Require an unsafe block for device-only descriptor prefetch.

fn prefetch<T, Layout>(descriptor: &TmaDesc<T, Layout>) {
    descriptor.prefetch();
}

/// Unused host entry point; the device or rejection probe is only type-checked.

fn main() {}
