/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Print the tables checked into README.md and the compatibility reference.
//! Does not require a CUDA installation or driver.

#[path = "../tests/support/feature_matrix.rs"]
mod feature_matrix;

fn main() {
    println!("{}", feature_matrix::readme_targets());
    println!("{}", feature_matrix::readme_matrix());
    print!("{}", feature_matrix::full_matrix());
}
