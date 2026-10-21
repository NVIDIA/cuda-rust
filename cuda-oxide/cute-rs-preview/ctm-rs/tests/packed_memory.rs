/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Packed global words preserve payloads without numeric conversion.

#[test]
fn aligned_word_load_preserves_each_half_and_word_offset() {
    let words = [0x3f80_8000u32, 0x7fc1_0001, 0xffff_7f80];
    let halves = words.as_ptr().cast::<u16>();
    for (index, expected) in words.iter().enumerate() {
        // SAFETY: a u32 allocation guarantees alignment, initialization and
        // four readable bytes at each even u16 offset for this whole borrow.
        let actual = unsafe { ctm_rs::memory::load_aligned_u32(halves.add(index * 2).cast()) };
        assert_eq!(actual, *expected);
    }
}

#[test]
fn raw_word_load_requires_an_explicit_unsafe_boundary() {
    let _: unsafe fn(*const u32) -> u32 = ctm_rs::memory::load_aligned_u32;
}
