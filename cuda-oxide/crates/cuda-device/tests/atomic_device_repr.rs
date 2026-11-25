// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]

use cuda_device::DeviceRepr;
use cuda_device::atomic::*;

#[test]
fn atomic_representations_match_their_scalars() {
    fn check<Atomic: DeviceRepr<Inner>, Inner>() {
        assert_eq!(size_of::<Atomic>(), size_of::<Inner>());
        assert_eq!(align_of::<Atomic>(), align_of::<Inner>());
        assert!(!core::mem::needs_drop::<Atomic>());
        assert!(!core::mem::needs_drop::<Inner>());
    }

    macro_rules! check_atomics {
        ($($atomic:ty => $inner:ty),* $(,)?) => {
            $(check::<$atomic, $inner>();)*
        };
    }
    check_atomics!(
        DeviceAtomicU32 => u32, DeviceAtomicI32 => i32,
        DeviceAtomicU64 => u64, DeviceAtomicI64 => i64,
        DeviceAtomicF16 => f16, DeviceAtomicF32 => f32, DeviceAtomicF64 => f64,
        BlockAtomicU32 => u32, BlockAtomicI32 => i32,
        BlockAtomicU64 => u64, BlockAtomicI64 => i64,
        BlockAtomicF16 => f16, BlockAtomicF32 => f32, BlockAtomicF64 => f64,
        SystemAtomicU32 => u32, SystemAtomicI32 => i32,
        SystemAtomicU64 => u64, SystemAtomicI64 => i64,
        SystemAtomicF16 => f16, SystemAtomicF32 => f32, SystemAtomicF64 => f64,
    );
}

#[test]
fn atomic_shared_reference_conversions_are_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/atomic_shared_references.rs");
}
