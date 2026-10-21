/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{
    markers::{C, L},
    static_layout::StaticLayout,
};
type Invalid = L<C<4>, C<{ i64::MAX }>>;
const _: usize = Invalid::SIZE;
fn main() {}
