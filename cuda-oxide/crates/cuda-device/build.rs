/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expose the compute-capability floor to device code as `cfg`s.
//!
//! Device code has no way to ask what target it is being compiled for, so an
//! API with a faster form on newer hardware -- `redux.sync` in place of a
//! shuffle tree, for one -- either always pays for the portable form or asks
//! every caller to gate the choice by hand. A `cfg` is decided before rustc
//! runs, so rustc drops the branch that does not apply and the requirement
//! scan never sees an instruction the target cannot run.
//!
//! Because a `cfg` is fixed before compilation it can only carry a *lower*
//! bound: the floor is the lowest capability the resulting PTX will be built
//! for, never the exact device. `CUDA_OXIDE_TARGET` sets it exactly, since
//! that is what gets built. `CUDA_OXIDE_DEVICE_ARCH` can only lower it: the
//! backend builds for the detected device when that device can run the kernel
//! and otherwise for the arch the kernel requires, so it bounds how low
//! selection may go, not how high. The rule lives in `cuda-target-spec` and is
//! shared with the backend's own target selection rather than restated here,
//! so the two cannot disagree about what the floor is.

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
    // Read the hint with the backend's grammar. One that does not parse is
    // left out: the backend rejects it, so nothing is built against it.
    let device_arch = std::env::var("CUDA_OXIDE_DEVICE_ARCH")
        .ok()
        .and_then(|value| value.parse::<cuda_target_spec::DeviceArch>().ok());
    match cuda_target_spec::resolve_sm_floor(
        target.as_deref(),
        device_arch.as_ref().map(AsRef::as_ref),
    ) {
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
