// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

#[derive(Copy, Clone, DeviceCopy)]
struct HostReference {
    value: &'static u32,
}

fn main() {}
