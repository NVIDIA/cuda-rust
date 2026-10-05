/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end example for scalar and packed integer min/max intrinsics (sm_90+).

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::i16x2::{
    max_relu_s16x2, max_s16x2, max_u16x2, min_relu_s16x2, min_s16x2, min_u16x2,
};
use cuda_device::int::{max_relu_s32, min_relu_s32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const NUM_OPS: usize = 8;

// Each packed value is written as (high lane, low lane).
const PACKED_A: u32 = 0x000a_8000; // signed: (10, -32768), unsigned: (10, 32768)
const PACKED_B: u32 = 0xffff_0001; // signed: (-1, 1), unsigned: (65535, 1)
const RELU_A: u32 = 0xfffd_fff7; // (-3, -9)
const RELU_B: u32 = 0xffff_0004; // (-1, 4)

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn test_integer_minmax(mut out: DisjointSlice<[u32; NUM_OPS]>) {
        let idx = thread::index_1d();
        if let Some(row) = out.get_mut(idx) {
            row[0] = min_relu_s32(-7, 5) as u32;
            row[1] = max_relu_s32(-7, 5) as u32;
            row[2] = min_s16x2(PACKED_A, PACKED_B);
            row[3] = max_s16x2(PACKED_A, PACKED_B);
            row[4] = min_u16x2(PACKED_A, PACKED_B);
            row[5] = max_u16x2(PACKED_A, PACKED_B);
            row[6] = min_relu_s16x2(RELU_A, RELU_B);
            row[7] = max_relu_s16x2(RELU_A, RELU_B);
        }
    }
}

fn signed_lane(value: u32, shift: u32) -> i16 {
    (value >> shift) as u16 as i16
}

fn unsigned_lane(value: u32, shift: u32) -> u16 {
    (value >> shift) as u16
}

fn pack_lanes(low: i16, high: i16) -> u32 {
    u32::from(low as u16) | (u32::from(high as u16) << 16)
}

fn signed_reference(a: u32, b: u32, op: fn(i16, i16) -> i16) -> u32 {
    pack_lanes(
        op(signed_lane(a, 0), signed_lane(b, 0)),
        op(signed_lane(a, 16), signed_lane(b, 16)),
    )
}

fn unsigned_reference(a: u32, b: u32, op: fn(u16, u16) -> u16) -> u32 {
    u32::from(op(unsigned_lane(a, 0), unsigned_lane(b, 0)))
        | (u32::from(op(unsigned_lane(a, 16), unsigned_lane(b, 16))) << 16)
}

fn relu(value: i16) -> i16 {
    value.max(0)
}

fn scalar_relu_reference(a: i32, b: i32, op: fn(i32, i32) -> i32) -> u32 {
    op(a, b).max(0) as u32
}

fn main() {
    println!("=== integer_minmax (sm_90+) ===");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major * 10 + minor < 90 {
        println!(
            "skipping: integer min/max extensions require sm_90+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let mut out = DeviceBuffer::<[u32; NUM_OPS]>::zeroed(&stream, 1).unwrap();

    // SAFETY: only index 0 reaches the one allocated result row.
    unsafe { module.test_integer_minmax(&stream, LaunchConfig::for_num_elems(1), &mut out) }
        .expect("launch test_integer_minmax");

    let rows = out.to_host_vec(&stream).unwrap();
    assert_eq!(rows.len(), 1, "unexpected result-row count");

    let expected = [
        ("min_relu_s32", scalar_relu_reference(-7, 5, i32::min)),
        ("max_relu_s32", scalar_relu_reference(-7, 5, i32::max)),
        ("min_s16x2", signed_reference(PACKED_A, PACKED_B, i16::min)),
        ("max_s16x2", signed_reference(PACKED_A, PACKED_B, i16::max)),
        (
            "min_u16x2",
            unsigned_reference(PACKED_A, PACKED_B, u16::min),
        ),
        (
            "max_u16x2",
            unsigned_reference(PACKED_A, PACKED_B, u16::max),
        ),
        (
            "min_relu_s16x2",
            signed_reference(RELU_A, RELU_B, |a, b| relu(a.min(b))),
        ),
        (
            "max_relu_s16x2",
            signed_reference(RELU_A, RELU_B, |a, b| relu(a.max(b))),
        ),
    ];

    let mut passed = true;
    println!("verifying {NUM_OPS} operations against host references:");
    for ((label, want), got) in expected.iter().zip(rows[0].iter()) {
        if label.ends_with("s32") {
            if got == want {
                println!("  {label}: ok  ({})", *got as i32);
            } else {
                println!(
                    "  {label}: FAIL  got {}, expected {}",
                    *got as i32, *want as i32
                );
                passed = false;
            }
            continue;
        }

        let got_lanes = [unsigned_lane(*got, 0), unsigned_lane(*got, 16)];
        let want_lanes = [unsigned_lane(*want, 0), unsigned_lane(*want, 16)];
        for lane in 0..2 {
            if got_lanes[lane] == want_lanes[lane] {
                println!("  {label} lane {lane}: ok  (0x{:04x})", got_lanes[lane]);
            } else {
                println!(
                    "  {label} lane {lane}: FAIL  got 0x{:04x}, expected 0x{:04x}",
                    got_lanes[lane], want_lanes[lane]
                );
                passed = false;
            }
        }
    }

    if !passed {
        println!("FAIL: integer_minmax, one or more checks failed");
        std::process::exit(1);
    }
    println!("PASS: all {NUM_OPS} integer min/max intrinsics verified on sm_{major}{minor}");
}
