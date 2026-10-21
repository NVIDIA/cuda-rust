/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cute_rs::{
    markers::{C, L, T2},
    static_layout::StaticLayout,
};
type Invalid = L<T2<C<2>, C<2>>, T2<C<{ i64::MAX }>, C<1>>>;
const _: usize = Invalid::SIZE;
fn main() {}
