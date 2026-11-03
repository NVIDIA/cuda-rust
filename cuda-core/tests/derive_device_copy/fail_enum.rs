// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

// Enums are rejected: a zeroed or device-written buffer could materialize an
// out-of-range discriminant, which is undefined behavior. Even an enum with a
// zero-valid first variant is refused, because per-field checking cannot prove
// every device-produced byte pattern is a valid variant.
#[derive(Copy, Clone, DeviceCopy)]
#[repr(u8)]
enum Tag {
    A = 1,
    B = 2,
}

fn main() {
    let _ = Tag::A;
}
