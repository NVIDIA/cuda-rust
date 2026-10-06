// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::kernel;

#[kernel]
pub fn static_lifetime(#[grid_constant] descriptor: &'static u32) {}

#[kernel]
pub fn named_lifetime<'a>(#[grid_constant] descriptor: &'a u32) {}

fn main() {}
