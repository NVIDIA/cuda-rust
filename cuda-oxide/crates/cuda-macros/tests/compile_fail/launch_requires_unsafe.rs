// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bare `cuda_launch!` (no `unsafe { }` around it) must not compile.
//!
//! The macro expansion calls the unsafe `cuda_core` launch functions without
//! wrapping them, so the unsafe obligation surfaces to the caller (E0133).
//! Everything else in this file type-checks; the only error is the missing
//! `unsafe` block.

use cuda_macros::cuda_launch;

// Stand-in for the marker struct `#[kernel]` generates for a kernel
// named `dummy`.
#[allow(non_camel_case_types)]
struct __dummy_CudaKernel;
impl cuda_host::CudaKernel for __dummy_CudaKernel {
    const PTX_NAME: &'static str = "dummy";
}

fn launch_without_unsafe(
    stream: std::sync::Arc<cuda_core::CudaStream>,
    module: std::sync::Arc<cuda_core::CudaModule>,
) {
    let x = 1.0f32;

    cuda_launch! {
        kernel: dummy,
        stream: stream,
        module: module,
        config: cuda_core::LaunchConfig::for_num_elems(1),
        args: [x]
    }
    .unwrap();
}

fn main() {}
