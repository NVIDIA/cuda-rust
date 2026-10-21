// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail probe: reject a TMA store whose descriptor and source have different logical shapes.
#![feature(f16)]

use core::{marker::PhantomData, ptr};

use cute_rs::{RowMajor, SmemTile, TmaDesc, copy_tma_s2g_2d};

/// Descriptor shape with eight rows and eight contiguous elements.

type DescriptorLayout = RowMajor<8, 8>;
/// Equal-capacity source with four rows and sixteen contiguous elements.
type SourceLayout = RowMajor<4, 16>;

/// Reject a TMA store whose descriptor and source have different logical shapes.

fn main() {
    let descriptor = ptr::null::<TmaDesc<f16, DescriptorLayout>>();
    let source = SmemTile::<f16, SourceLayout> {
        base: ptr::null_mut(),
        capacity: 64,
        layout: PhantomData,
    };

    unsafe { copy_tma_s2g_2d(descriptor, (0, 0), source) };
}
