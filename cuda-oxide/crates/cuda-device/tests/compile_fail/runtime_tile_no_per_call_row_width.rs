/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The row width belongs to the slice, so there is no per-call width to
//! disagree about. This is the case that made the earlier design unsound: two call sites
//! on one slice could pass different launch-uniform values, or one call site
//! could select between two of them under a thread-varying condition, and two
//! threads would then resolve the same element through "disjoint" tiles.
//!
//! Both spellings below are rejected: `tile_2d32_rt` takes only the thread
//! coordinate, and a slice built for one row width cannot be rebuilt for
//! another.

use cuda_device::thread::{LaunchContextRef, __internal};
use cuda_device::{DisjointSlice, RuntimeRowMajorTiles, Uniform};

fn cannot_pass_a_row_width_per_call<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
    mut c: DisjointSlice<'_, f32, RuntimeRowMajorTiles<1, 1>>,
    thread_varying: bool,
) {
    let coord = __internal::coord_2d_u32(launch_context);
    // SAFETY: a literal is trivially the same in every thread. The point of
    // the case is that selecting between two such values is what defeated the
    // old signature, not that either value is itself non-uniform.
    let wide = unsafe { Uniform::new_unchecked(100) };
    let narrow = unsafe { Uniform::new_unchecked(5) };
    let chosen = if thread_varying { wide } else { narrow };
    let _tile = c.tile_2d32_rt(coord, chosen);
}

fn main() {}
