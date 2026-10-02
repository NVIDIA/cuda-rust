// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use core::{marker::PhantomData, ptr};

use cute_rs::{RowMajor, SmemTile, TmaDesc, copy_tma_s2g_2d};

type DescriptorLayout = RowMajor<8, 8>;
type SourceLayout = RowMajor<4, 16>;

fn main() {
    let descriptor = ptr::null::<TmaDesc<f16, DescriptorLayout>>();
    let source = SmemTile::<f16, SourceLayout> {
        base: ptr::null_mut(),
        capacity: 64,
        layout: PhantomData,
    };

    unsafe { copy_tma_s2g_2d(descriptor, (0, 0), source) };
}
