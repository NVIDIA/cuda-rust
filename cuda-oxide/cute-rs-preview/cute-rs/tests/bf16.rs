/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! BF16 storage and explicit conversion contracts.
use cute_rs::{BFloat16, RegisterTile, Tensor};

#[test]
fn storage_preserves_every_pattern_and_tensor_coordinates() {
    assert_eq!(core::mem::size_of::<BFloat16>(), 2);
    assert_eq!(core::mem::align_of::<BFloat16>(), 2);
    for bits in 0..=u16::MAX {
        let value = BFloat16::from_bits(bits);
        assert_eq!(value.to_bits(), bits);
        assert_eq!(value.to_f32().to_bits(), (bits as u32) << 16);
    }
    let data = [BFloat16::from_bits(0); 9];
    let tail = Tensor::from_slice(&data).zipped_divide::<8>().slice(1);
    assert_eq!(tail.base(), 8);
    assert!(!tail.is_full());
}

#[test]
fn conversion_rounds_ties_even_and_preserves_special_values() {
    for (input, expected) in [
        (0x3f80_8000, 0x3f80),
        (0x3f81_8000, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0xbf81_8000, 0xbf82),
        (0, 0),
        (0x8000_0000, 0x8000),
        (0x7f80_0000, 0x7f80),
        (0xff80_0000, 0xff80),
        (0x0000_8000, 0),
        (0x0001_8000, 2),
    ] {
        assert_eq!(
            BFloat16::from_f32(f32::from_bits(input)).to_bits(),
            expected
        );
    }
    for bits in [0x7f80_0001, 0xff80_0001] {
        assert!(BFloat16::from_f32(f32::from_bits(bits)).to_f32().is_nan());
    }
}

#[test]
fn register_arithmetic_requires_explicit_widening_and_narrowing() {
    let tile = RegisterTile::from_array([BFloat16::from_f32(1.0), BFloat16::from_f32(2.0)]);
    let output = tile
        .map(BFloat16::to_f32)
        .map(|v| v * 0.5)
        .map(BFloat16::from_f32);
    assert_eq!(output.into_array().map(BFloat16::to_f32), [0.5, 1.0]);
}
