// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![doc = "Checks SM120 register carriers and packed scale-byte order."]
use ctm_rs::mma::{Accumulator16x8, Mxf4A, Mxf4B, ScalePairA, ScalePairB};
/// Register and scale carriers must preserve exact instruction ordering and packed bytes.
#[test]
fn register_carriers_preserve_instruction_order_and_scale_bytes() {
    assert_eq!(
        Mxf4A::from_registers([1, 2, 3, 4]).registers(),
        [1, 2, 3, 4]
    );
    assert_eq!(Mxf4B::from_registers([5, 6]).registers(), [5, 6]);
    assert_eq!(Accumulator16x8::zero().registers(), [0.0; 4]);
    assert_eq!(
        Accumulator16x8::from_registers([1., 2., 3., 4.]).registers(),
        [1., 2., 3., 4.]
    );
    assert_eq!(ScalePairA::from_bits(0x807e).bits(), 0x807e);
    assert_eq!(ScalePairB::from_bits(0x7e80).bits(), 0x7e80);
}
