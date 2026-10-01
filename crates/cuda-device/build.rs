/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expose the target's minimum compute capability as device cfgs.
//!
//! An explicit target sets the floor; an advisory device hint can only lower
//! the default. The shared resolver keeps these cfgs consistent with backend
//! selection. Unsupported intrinsic branches disappear before MIR lowering.

fn main() {
    // The two variables the backend already reads to pick a target.
    println!("cargo::rerun-if-env-changed=CUDA_OXIDE_TARGET");
    println!("cargo::rerun-if-env-changed=CUDA_OXIDE_DEVICE_ARCH");

    // Declare the whole ladder, not just the rungs set for this build, so an
    // unset rung is a false `cfg` rather than an `unexpected_cfgs` warning.
    let values = cuda_target_spec::SM_FLOOR_LADDER
        .iter()
        .map(|floor| format!("\"{floor}\""))
        .collect::<Vec<_>>()
        .join(", ");
    println!("cargo::rustc-check-cfg=cfg(cuda_oxide_sm_at_least, values({values}))");

    let target = std::env::var("CUDA_OXIDE_TARGET").ok();
    let device_arch = std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok();
    match cuda_target_spec::resolve_sm_floor(target.as_deref(), device_arch.as_deref()) {
        Ok(floor) => {
            for rung in cuda_target_spec::sm_floors_at_most(floor) {
                println!("cargo::rustc-cfg=cuda_oxide_sm_at_least=\"{rung}\"");
            }
        }
        // Fail the build rather than guess. Guessing high emits instructions
        // the target cannot run; guessing low silently drops the faster form.
        Err(error) => println!("cargo::error={error}"),
    }
}
