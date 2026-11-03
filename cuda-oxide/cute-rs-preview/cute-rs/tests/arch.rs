// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = "Checks the unsafe u32 contract of the warp-uniform compiler hint."]
/// Compile checks must reject unchecked calls and non-u32 warp-uniform arguments.
#[test]
fn warp_uniform_hint_requires_an_explicit_unsafe_u32_contract() {
    let _: unsafe fn(u32) -> u32 = cute_rs::arch::make_warp_uniform;
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/warp_uniform_unsafe.rs");
    tests.compile_fail("tests/compile_fail/warp_uniform_width.rs");
}

/// The sync-restricted fence is an unsafe forwarder whose order is a type parameter.
#[test]
fn fence_sync_restrict_takes_its_order_as_a_type() {
    let _: unsafe fn() = cute_rs::arch::fence_sync_restrict::<cute_rs::arch::Acquire>;
}

/// The cluster coordinate triples are plain readers of the grid-level cluster registers.
#[test]
fn cluster_triples_are_plain_coordinate_readers() {
    let _: fn() -> (u32, u32, u32) = cute_rs::arch::cluster_idx;
    let _: fn() -> (u32, u32, u32) = cute_rs::arch::cluster_dim;
}
