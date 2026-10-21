/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::partition::{Cyclic, Strided1D, StridedTensor};
fn warp_partition(_: StridedTensor<'_, f32, Cyclic<Strided1D, 32>>) {}
fn main() {
    let values = [0.0_f32; 128];
    warp_partition(StridedTensor::from_slice(&values).partition::<16>(0));
}
