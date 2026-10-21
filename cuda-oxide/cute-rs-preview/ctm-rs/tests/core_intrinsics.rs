/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Core facade contracts: bit conversions, fragment order and collective types.

use ctm_rs::{convert, debug, float, mma, thread, warp};

/// Macro rewriting must preserve the CTM prefix and the canonical scope witness.
#[ctm_rs::kernel]
pub fn scoped_index_probe(mut output: ctm_rs::memory::DisjointSlice<u32>) {
    if let Some(value) = output.get_mut(thread::index_1d()) {
        *value = 1;
    }
}

/// Conversions retain ties-to-even, signed zeros, infinities and NaNs.
#[test]
fn bf16_rounding_keeps_ties_even_and_special_values_distinct() {
    for (input, expected) in [
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x3f80_7fff, 0x3f80),
        (0x3f80_8000, 0x3f80),
        (0x3f80_8001, 0x3f81),
        (0x3f81_8000, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0xbf81_8000, 0xbf82),
        (0x7f80_0000, 0x7f80),
        (0xff80_0000, 0xff80),
        (0x7f80_0001, 0x7fc0),
        (0xff80_0001, 0xffc0),
    ] {
        assert_eq!(convert::f32_to_bf16_rne(f32::from_bits(input)), expected);
    }
}

/// Every sixteen-bit payload widens exactly, including noncanonical NaNs.
#[test]
fn bf16_widening_preserves_every_storage_pattern() {
    for bits in 0..=u16::MAX {
        assert_eq!(convert::bf16_to_f32(bits).to_bits(), u32::from(bits) << 16);
    }
}

/// Fragment pairs stay adjacent within each row and cover the whole MMA tile once.
#[test]
fn accumulator_coordinates_preserve_pair_order_and_bijection() {
    const LAST: (usize, usize) = mma::acc_coords(31, 3);
    assert_eq!(LAST, (15, 7));
    assert_eq!(mma::acc_coords(0, 0), (0, 0));
    assert_eq!(mma::acc_coords(0, 1), (0, 1));
    assert_eq!(mma::acc_coords(0, 2), (8, 0));
    assert_eq!(mma::acc_coords(0, 3), (8, 1));
    assert_eq!(mma::acc_coords(31, 0), (7, 6));
    assert_eq!(mma::acc_coords(32, 4), (0, 0));
    let mut visits = [[0u8; 8]; 16];
    for lane in 0..32 {
        for register in 0..4 {
            let (row, col) = mma::acc_coords(lane, register);
            visits[row][col] += 1;
        }
    }
    assert!(visits.into_iter().flatten().all(|count| count == 1));
}

/// Collective participation stays unsafe without executing GPU instructions on the host.
#[test]
fn collective_and_instruction_signatures_preserve_their_contracts() {
    let _: unsafe fn(u32, u32, u32) -> u32 = warp::shuffle_sync;
    let _: unsafe fn(f32) -> f32 = warp::reduce_sum_f32;
    let _: unsafe fn(bool) -> bool = warp::all;
    let _: unsafe fn(u32) = warp::sync_mask;
    let _: unsafe fn() = thread::setmaxnreg_inc::<24>;
    let _: unsafe fn() = thread::setmaxnreg_inc::<256>;
    let _: unsafe fn() = thread::setmaxnreg_dec::<24>;
    let _: unsafe fn() = thread::setmaxnreg_dec::<256>;
    let _: fn(f32) -> f32 = float::rsqrt_approx_f32;
    let _: fn(f32) -> f32 = float::lg2_approx_f32;
    let _: fn(f32) -> f32 = float::sin_approx_f32;
    let _: fn() -> ! = debug::trap;
    let _: fn() -> thread::ThreadIndex<'static> = thread::index_1d;
}
