/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::partition::StridedTensorMut;
fn main() {
    let mut values = [0.0_f32; 4];
    let mut first = StridedTensorMut::from_mut_slice(&mut values);
    let mut second = first;
    unsafe {
        first.store_at(0, 1.0);
    }
    unsafe {
        second.store_at(1, 2.0);
    }
}
