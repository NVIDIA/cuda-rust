// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, non_upper_case_globals)]

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaModule, CudaStream};
use cuda_macros::{cuda_launch, kernel};
use std::sync::Arc;

#[kernel]
pub fn collision_probe<
    const __cuda_oxide_arg_0: usize,
    const __cuda_oxide_kernel_hash: usize,
    const __cuda_oxide_kernel_ptr: usize,
    const __cuda_oxide_force_mono: usize,
>(value: u32) {
    let _ = (
        __cuda_oxide_arg_0,
        __cuda_oxide_kernel_hash,
        __cuda_oxide_kernel_ptr,
        __cuda_oxide_force_mono,
        value,
    );
}

fn low_level_launch<
    const __cuda_oxide_args: usize,
    const __cuda_oxide_arg_0: usize,
    const __cuda_oxide_ptx_name: usize,
    const __cuda_oxide_function: usize,
    const __cuda_oxide_config: usize,
    const __cuda_oxide_cooperative: usize,
    const __cuda_oxide_error: usize,
>(module: Arc<CudaModule>, stream: Arc<CudaStream>) {
    // SAFETY: this function is compiled to test macro hygiene and is never run.
    let _ = unsafe {
        cuda_launch! {
            kernel: collision_probe::<0, 0, 0, 0>,
            stream: stream,
            module: module,
            config: LaunchConfig::for_num_elems(1),
            cooperative: false,
            args: [1u32]
        }
    };
}

fn main() {
    let _ = collision_probe_ptx_name::<0, 0, 0, 0>();
    let _ = low_level_launch::<0, 0, 0, 0, 0, 0, 0>;
}
