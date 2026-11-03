// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use cuda_core::{CudaContext, CudaModule};
use cuda_device::{cuda_module, kernel, launch_contract};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn contracted(input: &[u32]) {
        let _ = input;
    }
}

fn bind_untrusted(module: Arc<CudaModule>, context: &Arc<CudaContext>) {
    let _ = kernels::load(context);
    let _ = kernels::from_module(module);
    let _ = kernels::load_named(context, "some-other-artifact");
}

fn main() {}
