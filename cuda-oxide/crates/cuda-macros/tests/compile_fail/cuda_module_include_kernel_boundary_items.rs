// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cuda_macros::kernel]
pub fn from_include(value: u32) {
    let _ = value;
}
