// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn configure<const X: u32, const Y: u32, const Z: u32>() {
    cuda_device::cluster::__cluster_config::<X, Y, Z>();
}

fn main() {
    configure::<1, 1, 1>();
}
