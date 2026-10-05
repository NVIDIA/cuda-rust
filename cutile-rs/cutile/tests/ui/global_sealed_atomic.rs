/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::core::Atomic;
#[derive(Copy, Clone)]
struct UserAtomic;
impl Atomic for UserAtomic {
    type Value = i32;
}
fn main() {}
