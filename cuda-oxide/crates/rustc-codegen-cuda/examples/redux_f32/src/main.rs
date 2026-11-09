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
//! Compile-only in CI, which builds it with `--arch sm_100a` and has no
//! family-10x device. `cargo oxide run` targets the attached GPU, so on a
//! B200/GB200 (sm_100) or B300/GB300 (sm_103) the reductions are verified:
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
    use cuda_core::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    println!("=== redux.sync f32 family (Blackwell family 10x) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // The f32 redux family exists only on family 10x, and the module's own
    // `.target` decides which GPUs can load it: an `sm_100a` build (the CI
    // build) runs only on sm_100, an `sm_103a` build only on sm_103.
    let target = ptx_target();
    if !module_target_runs_on(&target, major, minor) {
        println!("\nskipping: this module targets {target}, which sm_{major}{minor} cannot load");
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

/// The `.target` of the generated PTX (e.g. `sm_103a`). `cargo oxide run`
/// picks it from the attached GPU unless `--arch` overrides it, so it can
/// differ from the GPU this binary later runs on.
fn ptx_target() -> String {
    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("redux_f32.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ptx_path.display()));
    ptx.lines()
        .find_map(|line| line.strip_prefix(".target "))
        .map(|rest| rest.split([',', ' ']).next().unwrap_or(rest).to_string())
        .unwrap_or_else(|| panic!("no .target line in {}", ptx_path.display()))
}

/// Whether a GPU of compute capability `major.minor` can load a module built
/// for `target`: an `a` target needs that exact capability, an `f` target the
/// same major and at least its minor, a plain target at least its capability.
fn module_target_runs_on(target: &str, major: i32, minor: i32) -> bool {
    let Some(digits) = target.strip_prefix("sm_") else {
        return false;
    };
    let (number, suffix) = match digits.chars().last() {
        Some(c @ ('a' | 'f')) => (&digits[..digits.len() - 1], Some(c)),
        _ => (digits, None),
    };
    let Ok(capability) = number.parse::<i32>() else {
        return false;
    };
    let (t_major, t_minor) = (capability / 10, capability % 10);
    match suffix {
        Some('a') => (major, minor) == (t_major, t_minor),
        Some(_) => major == t_major && minor >= t_minor,
        None => (major, minor) >= (t_major, t_minor),
    }
}
