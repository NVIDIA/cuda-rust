/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Compile-fail probe: require bulk-copy wait counts as const generics rather than runtime arguments.
use ctm_rs::tma::cp_async_bulk_wait_group_read;
/// Require bulk-copy wait counts as const generics rather than runtime arguments.
fn wrong(n:u32) { cp_async_bulk_wait_group_read(n); }
/// Unused host entry point; the device or rejection probe is only type-checked.
fn main() {}
