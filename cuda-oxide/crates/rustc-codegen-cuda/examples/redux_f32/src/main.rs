/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single-instruction warp f32 reductions (Blackwell family 10x).
//!
//! Companion to `redux_minmax` (the integer family): all eight f32 forms —
//! min/max × `.abs` × `.NaN` — each lowered to one `redux.sync.*.f32`
//! instruction. The NaN kernel feeds one lane a NaN so the two NaN policies
//! observably differ.
//!
//! Compile-only in CI, since it pins `sm_100a` and no runner has that device.
//! Run it on a B200/GB200 (sm_100) and the reductions are verified:
//!
//!   cargo oxide run redux_f32

use cuda_device::{DisjointSlice, device, kernel, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Reduces `v` through all eight forms; lane 0 writes them in the slot
    /// order the host's expected arrays assume.
    #[device]
    fn reduce_all_forms(v: f32, mut out: DisjointSlice<f32>) {
        let values = [
            warp::redux_sync_min_f32(FULL_MASK, v),
            warp::redux_sync_max_f32(FULL_MASK, v),
            warp::redux_sync_min_abs_f32(FULL_MASK, v),
            warp::redux_sync_max_abs_f32(FULL_MASK, v),
            warp::redux_sync_min_nan_f32(FULL_MASK, v),
            warp::redux_sync_max_nan_f32(FULL_MASK, v),
            warp::redux_sync_min_abs_nan_f32(FULL_MASK, v),
            warp::redux_sync_max_abs_nan_f32(FULL_MASK, v),
        ];

        if warp::lane_id() == 0 {
            for (slot, value) in values.into_iter().enumerate() {
                // SAFETY: the host buffer holds exactly these eight slots.
                unsafe { *out.get_unchecked_mut(slot) = value };
            }
        }
    }

    /// Lane `l` contributes `l as f32 - 15.5`; all inputs finite, so the
    /// `.NaN` variants agree with the base forms.
    #[kernel]
    pub fn redux_f32_finite(out: DisjointSlice<f32>) {
        let v = warp::lane_id() as f32 - 15.5;
        reduce_all_forms(v, out);
    }

    /// Lane 0 contributes NaN: the base forms ignore it, every `.NaN` form
    /// propagates it.
    #[kernel]
    pub fn redux_f32_nan_policy(out: DisjointSlice<f32>) {
        let lane = warp::lane_id();
        let v = if lane == 0 {
            f32::NAN
        } else {
            lane as f32 - 15.5
        };
        reduce_all_forms(v, out);
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== redux.sync f32 family (Blackwell family 10x) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // The device code is pinned to sm_100a, so only an exact sm_100 device
    // can load it; the f32 redux family itself exists only on family 10x.
    if (major, minor) != (10, 0) {
        println!("\nskipping: this build targets sm_100a (B200/GB200)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // A single warp is all we need to demonstrate the reduction semantics.
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut failed = false;

    // ===== Test 1: finite inputs, all eight forms =====
    println!("\n--- Test 1: finite inputs (NaN forms agree with base forms) ---");
    let mut finite_dev = DeviceBuffer::<f32>::zeroed(&stream, 8).unwrap();

    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.redux_f32_finite((stream).as_ref(), cfg, &mut finite_dev) }
        .expect("Kernel launch failed");

    let finite = finite_dev.to_host_vec(&stream).unwrap();
    let expected = [-15.5, 15.5, 0.5, 15.5, -15.5, 15.5, 0.5, 15.5];
    println!("[min, max, min.abs, max.abs, ...NaN forms] = {:?}", finite);
    println!(
        "expected                                   = {:?}",
        expected
    );

    if finite == expected {
        println!("✓ all eight reductions correct on finite inputs");
    } else {
        println!("✗ finite reduction mismatch!");
        failed = true;
    }

    // ===== Test 2: one NaN lane splits the two NaN policies =====
    println!("\n--- Test 2: lane 0 contributes NaN ---");
    let mut nan_dev = DeviceBuffer::<f32>::zeroed(&stream, 8).unwrap();

    // SAFETY: launch shape/resources match the kernel; the buffer covers its accesses.
    unsafe { module.redux_f32_nan_policy((stream).as_ref(), cfg, &mut nan_dev) }
        .expect("Kernel launch failed");

    let nan = nan_dev.to_host_vec(&stream).unwrap();
    println!("[min, max, min.abs, max.abs] = {:?}", &nan[..4]);
    println!("expected                     = [-14.5, 15.5, 0.5, 15.5]");
    println!("NaN forms                    = {:?}", &nan[4..]);

    if nan[..4] == [-14.5, 15.5, 0.5, 15.5] && nan[4..].iter().all(|value| value.is_nan()) {
        println!("✓ base forms ignore NaN; .NaN forms propagate it");
    } else {
        println!("✗ NaN policy mismatch!");
        failed = true;
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: redux.sync f32 family produced correct results");
}
