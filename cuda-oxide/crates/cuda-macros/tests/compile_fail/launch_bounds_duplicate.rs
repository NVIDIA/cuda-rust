// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A kernel may have only one `#[launch_bounds]`, whichever attribute
//! expands first, and even when both carry the same bounds.

use cuda_device::{kernel, launch_bounds};

#[kernel]
#[launch_bounds(128)]
#[launch_bounds(256)]
fn kernel_first() {}

#[launch_bounds(128)]
#[kernel]
#[launch_bounds(128)]
fn launch_bounds_first() {}

fn main() {}
