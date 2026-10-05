// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::cuda_module;

#[cuda_module]
mod kernels {
    mod child {
        #[cuda_macros::kernel]
        pub fn from_parent() {}
    }
}

fn main() {}
