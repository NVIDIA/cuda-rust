/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::StaticViewMut32;

fn main() {
    let mut values = [0_u32; 4];
    let mut view = StaticViewMut32::<_, 4>::from_slice(&mut values).unwrap();
    let proof = view.at_const::<0>();
    let _moved = proof;
    let _ = proof.read();
}
