/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{
    markers::{C, Composed, L, Swizzle},
    static_layout::StaticLayout,
};
type Invalid = Composed<Swizzle<0, 0, 0>, 1, L<C<2>, C<{ i64::MAX }>>>;
const _: usize = Invalid::SIZE;
fn main() {}
