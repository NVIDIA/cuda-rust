// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cuda_macros::cluster_launch(1, 1, 1)]
fn valid_cluster_kernel() {}

fn main() {}
