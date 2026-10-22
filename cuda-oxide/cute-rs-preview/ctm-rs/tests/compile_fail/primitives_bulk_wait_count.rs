/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: reject a bulk-copy wait count above seven.
use ctm_rs::tma::cp_async_bulk_wait_group_read;
/// Reject a bulk-copy wait count above seven.
fn wrong() { cp_async_bulk_wait_group_read::<8>(); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
