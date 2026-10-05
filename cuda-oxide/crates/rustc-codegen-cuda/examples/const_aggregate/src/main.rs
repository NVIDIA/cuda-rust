/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Nested-aggregate constant materialization (regression).
//!
//! A `const` whose field is itself an aggregate (a struct, or an array of
//! structs) used to fail device codegen with
//! `Unsupported construct: Struct constant field N has unsupported type`.
//! The fix recursively rebuilds each nested field from its constant bytes.
//!
//! Three shapes are exercised:
//!   - `Mat3` : struct whose fields are `Vec3` structs (struct-of-struct).
//!   - `Mesh` : struct whose field is a `[Vec3; 3]` (struct-of-array-of-struct).
//!   - `Marker(())` : zero-sized struct that still carries a (ZST) field.
//!
//! Each const is passed by value to an `#[inline(never)]` device fn so rustc
//! cannot fold the field reads away: the whole aggregate must be materialized
//! as a `Const` operand, which is the path the fix repairs.
//!
//! Run: cargo oxide run const_aggregate

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct Vec3 {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    #[derive(Clone, Copy)]
    pub struct Mat3 {
        pub r0: Vec3,
        pub r1: Vec3,
        pub r2: Vec3,
    }

    #[derive(Clone, Copy)]
    pub struct Mesh {
        pub verts: [Vec3; 3],
    }

    /// Zero-sized struct that still has a (zero-sized) field.
    #[derive(Clone, Copy)]
    pub struct Marker(pub ());

    // struct-of-struct const: the 3x3 identity matrix.
    const IDENTITY: Mat3 = Mat3 {
        r0: Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
        r1: Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        r2: Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    };

    // struct-of-array-of-struct const.
    const MESH: Mesh = Mesh {
        verts: [
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 4.0,
            },
        ],
    };

    const MARK: Marker = Marker(());

    /// Sum of the diagonal: 1.0 + 1.0 + 1.0 = 3.0 for the identity.
    #[inline(never)]
    fn diag_sum(m: Mat3) -> f32 {
        m.r0.x + m.r1.y + m.r2.z
    }

    /// Sum of the per-vertex diagonal: 1.0 + 2.0 + 4.0 = 7.0.
    #[inline(never)]
    fn mesh_sum(me: Mesh) -> f32 {
        me.verts[0].x + me.verts[1].y + me.verts[2].z
    }

    /// The ZST field carries no data; the call still materializes the const.
    #[inline(never)]
    fn marker_value(_m: Marker) -> f32 {
        100.0
    }

    /// out[i] = diag_sum(IDENTITY) + mesh_sum(MESH) + marker_value(MARK)
    ///        = 3.0 + 7.0 + 100.0 = 110.0  for every lane.
    #[kernel]
    pub fn const_aggregate(mut out: DisjointSlice<f32>) {
        let t = thread::index_1d();
        if let Some(slot) = out.get_mut(t) {
            *slot = diag_sum(IDENTITY) + mesh_sum(MESH) + marker_value(MARK);
        }
    }
}

const N: usize = 64;
const EXPECTED: f32 = 110.0;

fn main() {
    println!("=== Nested-aggregate constant materialization ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.const_aggregate(stream.as_ref(), config, &mut d_out) }
        .expect("Kernel launch failed");

    let out = d_out.to_host_vec(&stream).unwrap();
    let mut bad = None;
    for (i, v) in out.iter().enumerate() {
        if (v - EXPECTED).abs() > 1e-6 {
            bad = Some((i, *v));
            break;
        }
    }

    match bad {
        None => println!("SUCCESS: all {N} lanes == {EXPECTED} (3.0 + 7.0 + 100.0)"),
        Some((i, v)) => {
            println!("FAIL: lane {i} = {v}, expected {EXPECTED}");
            std::process::exit(1);
        }
    }
}

#[allow(dead_code)]
fn _stream_marker(_s: &Arc<CudaStream>) {}
