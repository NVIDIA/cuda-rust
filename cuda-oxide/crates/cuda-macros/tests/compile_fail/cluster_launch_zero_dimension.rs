// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cuda_macros::cluster_launch(1, 0, 1)]
fn invalid_cluster_kernel() {}

fn main() {}
