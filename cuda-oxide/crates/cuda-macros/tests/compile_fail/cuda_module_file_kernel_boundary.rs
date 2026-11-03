// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(proc_macro_hygiene)]

use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    #[cuda_macros::kernel]
    pub fn root(value: u32) {
        let _ = value;
    }

    pub mod file_kernel;
}

fn undiscovered(_: &kernels::file_kernel::LoadedModule) {}

fn main() {}
