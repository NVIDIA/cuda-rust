// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    mod first {
        use super::*;

        #[cuda_macros::kernel]
        pub fn step(value: u32) {
            let _ = value;
        }
    }

    mod second {
        use super::*;

        #[cuda_macros::kernel]
        pub fn step(value: u32) {
            let _ = value;
        }
    }
}

fn main() {}
