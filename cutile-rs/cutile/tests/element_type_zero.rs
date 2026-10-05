/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for the `ElementType::ZERO` associated constant.
//!
//! Covers: value correctness for every built-in element type, and usability in
//! `const` contexts (the primary motivation for making `ZERO` a const).

use cutile::core::{bf16, f16, f8e4m3fn, f8e5m2, tf32, ElementType};

#[test]
#[allow(clippy::assertions_on_constants)] // the constants are what is under test
fn zero_values_match_expected() {
    assert_eq!(<f16 as ElementType>::ZERO, f16::ZERO);
    assert_eq!(<bf16 as ElementType>::ZERO, bf16::ZERO);
    assert_eq!(<f32 as ElementType>::ZERO, 0.0f32);
    assert_eq!(<f64 as ElementType>::ZERO, 0.0f64);
    assert_eq!(<i8 as ElementType>::ZERO, 0i8);
    assert_eq!(<u8 as ElementType>::ZERO, 0u8);
    assert_eq!(<i16 as ElementType>::ZERO, 0i16);
    assert_eq!(<u16 as ElementType>::ZERO, 0u16);
    assert_eq!(<i32 as ElementType>::ZERO, 0i32);
    assert_eq!(<u32 as ElementType>::ZERO, 0u32);
    assert_eq!(<i64 as ElementType>::ZERO, 0i64);
    assert_eq!(<u64 as ElementType>::ZERO, 0u64);
    assert!(!<bool as ElementType>::ZERO);
    assert_eq!(<tf32 as ElementType>::ZERO, tf32(0));
    assert_eq!(<f8e4m3fn as ElementType>::ZERO, f8e4m3fn(0));
    assert_eq!(<f8e5m2 as ElementType>::ZERO, f8e5m2(0));
}

#[test]
#[allow(clippy::assertions_on_constants)] // the constants are what is under test
fn zero_usable_in_const_context() {
    const F32_ZERO: f32 = <f32 as ElementType>::ZERO;
    const I32_ZERO: i32 = <i32 as ElementType>::ZERO;
    const BOOL_ZERO: bool = <bool as ElementType>::ZERO;
    const F16_ZERO: f16 = <f16 as ElementType>::ZERO;
    const TF32_ZERO: tf32 = <tf32 as ElementType>::ZERO;

    assert_eq!(F32_ZERO, 0.0);
    assert_eq!(I32_ZERO, 0);
    assert!(!BOOL_ZERO);
    assert_eq!(F16_ZERO, f16::ZERO);
    assert_eq!(TF32_ZERO, tf32(0));
}

#[test]
fn zero_usable_in_generic_context() {
    fn zero<T: ElementType + PartialEq + std::fmt::Debug>(expected: T) {
        assert_eq!(T::ZERO, expected);
    }
    zero::<f32>(0.0);
    zero::<f64>(0.0);
    zero::<i32>(0);
    zero::<u64>(0);
    zero::<bool>(false);
    zero::<f16>(f16::ZERO);
    zero::<bf16>(bf16::ZERO);
}
