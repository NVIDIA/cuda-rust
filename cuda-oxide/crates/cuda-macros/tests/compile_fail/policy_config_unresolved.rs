// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_device::{kernel, launch_bounds};

#[kernel]
#[launch_bounds(P::MAX_THREADS)]
fn unresolved<P>() {}

fn main() {}
