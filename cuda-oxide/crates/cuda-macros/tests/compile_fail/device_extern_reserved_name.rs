// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::device;

#[device]
unsafe extern "C" {
    fn cuda_oxide_device_extern_evil();
}

fn main() {}
