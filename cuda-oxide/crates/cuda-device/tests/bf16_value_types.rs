/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host contract of the typed bfloat16 values `Bf16` and `Bf16x2`.
//!
//! Only the methods that run on the host are called here. The methods that
//! wrap a generated intrinsic are device-only; the `bf16_value_types` example
//! checks them on a GPU.

use cuda_device::DeviceRepr;
use cuda_device::bf16::Bf16;
use cuda_device::bf16x2::Bf16x2;
use cuda_device::convert::{
    bf16_to_f32, cvt_f32x2_bf16x2, f32_pair_to_packed_bf16, f32_to_bf16, f32_to_bf16_rne,
    pack_bf16_pair,
};

/// Low 16 bits attached to every high half in the `f32` sweep: exact values,
/// the rounding tie, both neighbours of the tie, and arbitrary payloads.
///
/// Each entry fits in 16 bits. The narrowing conversions drop or round away
/// exactly these bits.
const LOW_HALVES: [u32; 10] = [
    0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x1234, 0xABCD, 0x4000, 0xC000,
];

/// 16-bit patterns used as the fixed half while the other half sweeps.
///
/// The set holds both zeros, plus and minus one, an infinity, a NaN with a payload,
/// all bits set, and the smallest subnormal. The packed tests place each
/// entry once in the low half and once in the high half.
const FIXED_HALVES: [u16; 8] = [
    0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0x7FC1, 0xFFFF, 0x0001,
];

/// Calls `visit` with every `f32` whose high 16 bits take each of the 65536
/// patterns, combined with each entry of [`LOW_HALVES`]. The high half covers
/// both zeros, every subnormal and normal exponent, both infinities, and every
/// NaN payload that bf16 can hold.
fn for_each_f32_pattern(mut visit: impl FnMut(f32)) {
    for high in 0..=u16::MAX {
        for low in LOW_HALVES {
            visit(f32::from_bits((u32::from(high) << 16) | low));
        }
    }
}

/// Round-to-nearest-even from `f32` to bf16 for a value that is not NaN,
/// derived from the two neighbouring bf16 values instead of from bit tricks.
///
/// The function measures the distance to both neighbours in `f64`, picks the
/// nearer one, and picks the neighbour with an even last bit on a tie. It
/// returns raw bf16 bits with the sign of `value`. Panics when `value` is
/// NaN.
fn reference_round_to_nearest_even(value: f32) -> u16 {
    assert!(!value.is_nan());
    let bits = value.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let magnitude = bits & 0x7FFF_FFFF;
    // bf16 neighbours of the magnitude. `above` can be the infinity pattern;
    // its distance is measured as if the exponent range continued.
    let below = magnitude & 0xFFFF_0000;
    if below == magnitude {
        return sign | (below >> 16) as u16;
    }
    let above = below + 0x0001_0000;
    let widen = |pattern: u32| {
        if pattern >= 0x7F80_0000 {
            2.0_f64.powi(128)
        } else {
            f64::from(f32::from_bits(pattern))
        }
    };
    let exact = f64::from(f32::from_bits(magnitude));
    let distance_below = exact - widen(below);
    let distance_above = widen(above) - exact;
    let below_is_even = (below >> 16) & 1 == 0;
    let chosen = if distance_below < distance_above {
        below
    } else if distance_below > distance_above {
        above
    } else if below_is_even {
        below
    } else {
        above
    };
    sign | (chosen >> 16) as u16
}

/// Fails to compile when the type implements the bound: the second blanket
/// implementation then also applies and the inferred parameter is ambiguous.
///
/// Use: `assert_not_implemented!(Type: Bound)`. The check happens at compile
/// time; the expansion does nothing at run time.
macro_rules! assert_not_implemented {
    ($ty:ty: $($bound:tt)+) => {{
        /// Helper trait with one implementation per marker type.
        /// The call below compiles only when exactly one marker applies.
        trait AmbiguousIfImplemented<Marker> {
            /// Empty function whose only job is to force marker inference.
            fn check() {}
        }
        /// Applies to every type, with the unit marker.
        impl<T> AmbiguousIfImplemented<()> for T {}
        /// Marker for the implementation that needs the bound.
        #[allow(dead_code)]
        struct Implemented;
        /// Applies only to types that implement the bound.
        impl<T: $($bound)+> AmbiguousIfImplemented<Implemented> for T {}
        <$ty as AmbiguousIfImplemented<_>>::check();
    }};
}

/// Pins the size and alignment of `Bf16` to `u16` and of `Bf16x2` to `u32`.
#[test]
fn layouts_match_the_raw_integers() {
    assert_eq!(size_of::<Bf16>(), size_of::<u16>());
    assert_eq!(align_of::<Bf16>(), align_of::<u16>());
    assert_eq!(size_of::<Bf16x2>(), size_of::<u32>());
    assert_eq!(align_of::<Bf16x2>(), align_of::<u32>());
}

/// Pins `Default` for both types to all-zero bits, which is positive zero.
#[test]
fn default_is_positive_zero() {
    assert_eq!(Bf16::default().to_bits(), 0x0000);
    assert_eq!(Bf16x2::default().to_bits(), 0x0000_0000);
    assert_eq!(Bf16::default().to_f32().to_bits(), 0.0_f32.to_bits());
}

/// Pins the marker traits of both types: `Copy`, `Clone`, `Default`,
/// `Debug`, `Send`, `Sync`, `'static`, and no drop glue.
#[test]
fn values_are_plain_copyable_data() {
    /// Compiles only when `T` has every listed trait; then checks that `T`
    /// needs no drop.
    fn check<T: Copy + Clone + Default + core::fmt::Debug + Send + Sync + 'static>() {
        assert!(!core::mem::needs_drop::<T>());
    }
    check::<Bf16>();
    check::<Bf16x2>();

    let one = Bf16::from_bits(0x3F80);
    let copy = one;
    assert_eq!(one.to_bits(), copy.to_bits());
    let pair = Bf16x2::from_bits(0x4000_3F80);
    let copy = pair;
    assert_eq!(pair.to_bits(), copy.to_bits());

    assert!(!format!("{one:?}").is_empty());
    assert!(!format!("{pair:?}").is_empty());
}

/// Pins `DeviceRepr<u16>` for `Bf16` and `DeviceRepr<u32>` for `Bf16x2`,
/// the bound that lets a raw integer buffer be retyped.
#[test]
fn device_repr_links_each_type_to_its_raw_integer() {
    /// Compiles only when `Wrapper: DeviceRepr<Inner>`; then checks that
    /// both types have the same size and alignment and need no drop.
    fn check<Wrapper: DeviceRepr<Inner>, Inner>() {
        assert_eq!(size_of::<Wrapper>(), size_of::<Inner>());
        assert_eq!(align_of::<Wrapper>(), align_of::<Inner>());
        assert!(!core::mem::needs_drop::<Wrapper>());
        assert!(!core::mem::needs_drop::<Inner>());
    }
    check::<Bf16, u16>();
    check::<Bf16x2, u32>();
}

/// Pins the traits that both types must not implement: comparison,
/// `From`/`Into` conversions, `Deref`, and the arithmetic operators.
#[test]
fn comparison_conversion_and_operator_traits_are_absent() {
    assert_not_implemented!(Bf16: PartialEq);
    assert_not_implemented!(Bf16: PartialOrd);
    assert_not_implemented!(Bf16: From<u16>);
    assert_not_implemented!(Bf16: From<f32>);
    assert_not_implemented!(Bf16: Into<u16>);
    assert_not_implemented!(Bf16: Into<f32>);
    assert_not_implemented!(Bf16: core::ops::Deref);

    assert_not_implemented!(Bf16x2: PartialEq);
    assert_not_implemented!(Bf16x2: PartialOrd);
    assert_not_implemented!(Bf16x2: From<u32>);
    assert_not_implemented!(Bf16x2: From<(f32, f32)>);
    assert_not_implemented!(Bf16x2: Into<u32>);
    assert_not_implemented!(Bf16x2: Into<(f32, f32)>);
    assert_not_implemented!(Bf16x2: core::ops::Deref);
    assert_not_implemented!(Bf16x2: core::ops::Add);
    assert_not_implemented!(Bf16x2: core::ops::Sub);
    assert_not_implemented!(Bf16x2: core::ops::Mul);
    assert_not_implemented!(Bf16x2: core::ops::Neg);
}

/// Pins `from_bits`, `to_bits`, `new`, `lo`, and `hi` as `const fn`: each
/// local constant below compiles only when its initializer is const.
#[test]
fn bit_and_packing_constructors_are_const() {
    /// bf16 1.0, built from bits in a const context.
    const ONE: Bf16 = Bf16::from_bits(0x3F80);
    /// bf16 2.0, built from bits in a const context.
    const TWO: Bf16 = Bf16::from_bits(0x4000);
    /// Bits of `ONE`, read back in a const context.
    const ONE_BITS: u16 = ONE.to_bits();
    /// `ONE` in the low half and `TWO` in the high half.
    const PAIR: Bf16x2 = Bf16x2::new(ONE, TWO);
    /// Packed bits of `PAIR`.
    const PAIR_BITS: u32 = PAIR.to_bits();
    /// Bits of the low half of `PAIR`.
    const PAIR_LO_BITS: u16 = PAIR.lo().to_bits();
    /// Bits of the high half of `PAIR`.
    const PAIR_HI_BITS: u16 = PAIR.hi().to_bits();
    /// Packed bits that pass through `Bf16x2::from_bits` in a const context.
    const RAW_PAIR_BITS: u32 = Bf16x2::from_bits(0xBF80_3F80).to_bits();

    assert_eq!(ONE_BITS, 0x3F80);
    assert_eq!(PAIR_BITS, 0x4000_3F80);
    assert_eq!(PAIR_LO_BITS, 0x3F80);
    assert_eq!(PAIR_HI_BITS, 0x4000);
    assert_eq!(RAW_PAIR_BITS, 0xBF80_3F80);
}

/// Pins `Bf16::from_bits` then `to_bits` as the identity on all 65536
/// patterns.
#[test]
fn bf16_bits_round_trip_for_every_pattern() {
    for bits in 0..=u16::MAX {
        assert_eq!(
            Bf16::from_bits(bits).to_bits(),
            bits,
            "Bf16 bits {bits:#06x} changed"
        );
    }
}

/// Pins `Bf16x2::from_bits` then `to_bits` as the identity while each half
/// sweeps all patterns against [`FIXED_HALVES`].
#[test]
fn bf16x2_bits_round_trip_across_both_halves() {
    for sweep in 0..=u16::MAX {
        for fixed in FIXED_HALVES {
            let sweep_low = u32::from(sweep) | (u32::from(fixed) << 16);
            let sweep_high = u32::from(fixed) | (u32::from(sweep) << 16);
            for bits in [sweep_low, sweep_high] {
                assert_eq!(
                    Bf16x2::from_bits(bits).to_bits(),
                    bits,
                    "Bf16x2 bits {bits:#010x} changed"
                );
            }
        }
    }
}

/// Pins `Bf16::from_f32` to `f32_to_bf16` and to the high 16 bits of the
/// `f32`, over the full pattern sweep.
#[test]
fn from_f32_truncates_like_the_convert_function() {
    for_each_f32_pattern(|value| {
        let bits = value.to_bits();
        let got = Bf16::from_f32(value).to_bits();
        assert_eq!(got, f32_to_bf16(value), "from_f32 of f32 bits {bits:#010x}");
        assert_eq!(
            got,
            (bits >> 16) as u16,
            "from_f32 of f32 bits {bits:#010x}"
        );
    });
}

/// Pins literal results of `Bf16::from_f32` for exact values, zeros,
/// infinities, values above the tie, subnormals, and NaN payloads.
#[test]
fn from_f32_truncation_special_values() {
    assert_eq!(Bf16::from_f32(1.0).to_bits(), 0x3F80);
    assert_eq!(Bf16::from_f32(-2.0).to_bits(), 0xC000);
    assert_eq!(Bf16::from_f32(0.0).to_bits(), 0x0000);
    assert_eq!(Bf16::from_f32(-0.0).to_bits(), 0x8000);
    assert_eq!(Bf16::from_f32(f32::INFINITY).to_bits(), 0x7F80);
    assert_eq!(Bf16::from_f32(f32::NEG_INFINITY).to_bits(), 0xFF80);
    // Truncation drops the low 16 bits even above the tie.
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0x3F80_FFFF)).to_bits(),
        0x3F80
    );
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0xBF81_8000)).to_bits(),
        0xBF81
    );
    // A subnormal f32 below the smallest bf16 step truncates to zero.
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0x0000_FFFF)).to_bits(),
        0x0000
    );
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0x0001_0000)).to_bits(),
        0x0001
    );
    // A NaN payload in the high half survives; one only in the low half is
    // dropped, which leaves the infinity pattern.
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0x7FC1_0000)).to_bits(),
        0x7FC1
    );
    assert_eq!(
        Bf16::from_f32(f32::from_bits(0x7F80_0001)).to_bits(),
        0x7F80
    );
}

/// Pins `Bf16::from_f32_rne` to `f32_to_bf16_rne` over the full pattern
/// sweep, NaN inputs included.
#[test]
fn from_f32_rne_matches_the_convert_function() {
    for_each_f32_pattern(|value| {
        let bits = value.to_bits();
        assert_eq!(
            Bf16::from_f32_rne(value).to_bits(),
            f32_to_bf16_rne(value),
            "from_f32_rne of f32 bits {bits:#010x}"
        );
    });
}

/// Pins `Bf16::from_f32_rne` to the independent reference rounding for
/// every swept value that is not NaN.
#[test]
fn from_f32_rne_rounds_to_nearest_even() {
    for_each_f32_pattern(|value| {
        if value.is_nan() {
            return;
        }
        let bits = value.to_bits();
        assert_eq!(
            Bf16::from_f32_rne(value).to_bits(),
            reference_round_to_nearest_even(value),
            "from_f32_rne of f32 bits {bits:#010x}"
        );
    });
}

/// Pins literal results of `Bf16::from_f32_rne` for ties, their neighbours,
/// exponent carries, overflow to infinity, and subnormals.
#[test]
fn from_f32_rne_ties_and_special_values() {
    let rne = |bits: u32| Bf16::from_f32_rne(f32::from_bits(bits)).to_bits();

    assert_eq!(Bf16::from_f32_rne(1.0).to_bits(), 0x3F80);
    assert_eq!(Bf16::from_f32_rne(0.0).to_bits(), 0x0000);
    assert_eq!(Bf16::from_f32_rne(-0.0).to_bits(), 0x8000);
    assert_eq!(Bf16::from_f32_rne(f32::INFINITY).to_bits(), 0x7F80);
    assert_eq!(Bf16::from_f32_rne(f32::NEG_INFINITY).to_bits(), 0xFF80);

    // An exact tie goes to the even neighbour, in both signs.
    assert_eq!(rne(0x3F80_8000), 0x3F80);
    assert_eq!(rne(0x3F81_8000), 0x3F82);
    assert_eq!(rne(0xBF80_8000), 0xBF80);
    assert_eq!(rne(0xBF81_8000), 0xBF82);
    // One unit either side of the tie goes to the nearer neighbour.
    assert_eq!(rne(0x3F80_7FFF), 0x3F80);
    assert_eq!(rne(0x3F80_8001), 0x3F81);
    assert_eq!(rne(0x3F81_7FFF), 0x3F81);
    assert_eq!(rne(0x3F81_8001), 0x3F82);
    // Rounding carries into the exponent and, at the top, into infinity.
    assert_eq!(rne(0x3FFF_C000), 0x4000);
    assert_eq!(Bf16::from_f32_rne(f32::MAX).to_bits(), 0x7F80);
    assert_eq!(Bf16::from_f32_rne(f32::MIN).to_bits(), 0xFF80);
    assert_eq!(rne(0x7F7F_7FFF), 0x7F7F);
    // Subnormals round on the same grid.
    assert_eq!(rne(0x0000_0001), 0x0000);
    assert_eq!(rne(0x0000_8000), 0x0000);
    assert_eq!(rne(0x0000_8001), 0x0001);
    assert_eq!(rne(0x0001_8000), 0x0002);
    assert_eq!(rne(0x8000_8001), 0x8001);
}

/// Pins that `Bf16::from_f32_rne` maps every swept NaN to a NaN with the
/// same sign bit.
#[test]
fn from_f32_rne_keeps_nan_a_nan() {
    for_each_f32_pattern(|value| {
        if !value.is_nan() {
            return;
        }
        let bits = value.to_bits();
        let got = Bf16::from_f32_rne(value);
        assert!(
            got.to_f32().is_nan(),
            "from_f32_rne turned NaN bits {bits:#010x} into {:#06x}",
            got.to_bits()
        );
        assert_eq!(
            got.to_bits() & 0x8000,
            ((bits >> 16) as u16) & 0x8000,
            "from_f32_rne changed the sign of NaN bits {bits:#010x}"
        );
    });
}

/// Pins `Bf16::to_f32` to `bf16_to_f32` and to the bits shifted left by 16,
/// for all 65536 patterns.
#[test]
fn to_f32_widens_every_pattern() {
    for bits in 0..=u16::MAX {
        let widened = Bf16::from_bits(bits).to_f32().to_bits();
        assert_eq!(
            widened,
            bf16_to_f32(bits).to_bits(),
            "to_f32 of bf16 bits {bits:#06x}"
        );
        assert_eq!(
            widened,
            u32::from(bits) << 16,
            "to_f32 of bf16 bits {bits:#06x}"
        );
    }
    assert_eq!(Bf16::from_bits(0x3F80).to_f32(), 1.0);
    assert_eq!(Bf16::from_bits(0xC000).to_f32(), -2.0);
    assert_eq!(Bf16::from_bits(0x7F80).to_f32(), f32::INFINITY);
    assert_eq!(Bf16::from_bits(0xFF80).to_f32(), f32::NEG_INFINITY);
    assert!(Bf16::from_bits(0x7FC0).to_f32().is_nan());
    assert!(Bf16::from_bits(0xFF81).to_f32().is_nan());
}

/// Pins `to_f32` followed by either narrowing as the identity on bits. For
/// NaN inputs the rounding path is pinned to `f32_to_bf16_rne` instead.
#[test]
fn widening_then_narrowing_returns_the_same_bits() {
    for bits in 0..=u16::MAX {
        let value = Bf16::from_bits(bits);
        assert_eq!(
            Bf16::from_f32(value.to_f32()).to_bits(),
            bits,
            "truncating round trip of bf16 bits {bits:#06x}"
        );
        let rounded = Bf16::from_f32_rne(value.to_f32()).to_bits();
        if value.to_f32().is_nan() {
            assert_eq!(
                rounded,
                f32_to_bf16_rne(bf16_to_f32(bits)),
                "rounding round trip of NaN bf16 bits {bits:#06x}"
            );
        } else {
            assert_eq!(
                rounded, bits,
                "rounding round trip of bf16 bits {bits:#06x}"
            );
        }
    }
}

/// Pins `Bf16x2::new(lo, hi)` to `lo | hi << 16` and to `pack_bf16_pair`,
/// and pins `lo()` / `hi()` to return the same two values.
#[test]
fn new_places_the_first_value_in_the_low_half() {
    let packed = Bf16x2::new(Bf16::from_bits(0x3F80), Bf16::from_bits(0x4000));
    assert_eq!(packed.to_bits(), 0x4000_3F80);
    assert_eq!(packed.lo().to_bits(), 0x3F80);
    assert_eq!(packed.hi().to_bits(), 0x4000);

    for sweep in 0..=u16::MAX {
        for fixed in FIXED_HALVES {
            for (lo, hi) in [(sweep, fixed), (fixed, sweep)] {
                let packed = Bf16x2::new(Bf16::from_bits(lo), Bf16::from_bits(hi));
                let expected = u32::from(lo) | (u32::from(hi) << 16);
                assert_eq!(
                    packed.to_bits(),
                    expected,
                    "new({lo:#06x}, {hi:#06x}) packed bits"
                );
                assert_eq!(
                    packed.to_bits(),
                    pack_bf16_pair(lo, hi),
                    "new({lo:#06x}, {hi:#06x}) against pack_bf16_pair"
                );
                assert_eq!(packed.lo().to_bits(), lo, "lo of new({lo:#06x}, {hi:#06x})");
                assert_eq!(packed.hi().to_bits(), hi, "hi of new({lo:#06x}, {hi:#06x})");
            }
        }
    }
}

/// Pins `lo()` to bits 0..16 and `hi()` to bits 16..32 of the raw value,
/// with no bit of one half visible in the other.
#[test]
fn lo_and_hi_split_raw_bits_without_mixing_halves() {
    // The sign bit of one half must not reach the other half.
    let mixed = Bf16x2::from_bits(0xBF80_3F80);
    assert_eq!(mixed.lo().to_bits(), 0x3F80);
    assert_eq!(mixed.hi().to_bits(), 0xBF80);
    let mixed = Bf16x2::from_bits(0x3F80_BF80);
    assert_eq!(mixed.lo().to_bits(), 0xBF80);
    assert_eq!(mixed.hi().to_bits(), 0x3F80);

    for sweep in 0..=u16::MAX {
        for fixed in FIXED_HALVES {
            for (lo, hi) in [(sweep, fixed), (fixed, sweep)] {
                let packed = Bf16x2::from_bits(u32::from(lo) | (u32::from(hi) << 16));
                assert_eq!(packed.lo().to_bits(), lo, "lo of {lo:#06x}/{hi:#06x}");
                assert_eq!(packed.hi().to_bits(), hi, "hi of {lo:#06x}/{hi:#06x}");
            }
        }
    }
}

/// Pins `Bf16x2::from_f32_pair` to `f32_pair_to_packed_bf16` and to
/// `Bf16::from_f32` applied to each half, first argument in the low half.
#[test]
fn from_f32_pair_truncates_and_packs_low_half_first() {
    assert_eq!(Bf16x2::from_f32_pair(1.0, 2.0).to_bits(), 0x4000_3F80);
    assert_eq!(Bf16x2::from_f32_pair(-1.0, 1.0).to_bits(), 0x3F80_BF80);
    assert_eq!(Bf16x2::from_f32_pair(-0.0, 0.0).to_bits(), 0x0000_8000);
    // Truncation, not rounding: the low 16 bits of each f32 are dropped.
    assert_eq!(
        Bf16x2::from_f32_pair(f32::from_bits(0x3F80_FFFF), f32::from_bits(0xC000_8001)).to_bits(),
        0xC000_3F80
    );

    let fixed_values = [
        1.0_f32,
        -0.0,
        f32::NEG_INFINITY,
        f32::from_bits(0x7FC1_2345),
        f32::from_bits(0x0000_8001),
    ];
    for_each_f32_pattern(|sweep| {
        for fixed in fixed_values {
            for (lo, hi) in [(sweep, fixed), (fixed, sweep)] {
                let got = Bf16x2::from_f32_pair(lo, hi);
                let (lo_bits, hi_bits) = (lo.to_bits(), hi.to_bits());
                assert_eq!(
                    got.to_bits(),
                    f32_pair_to_packed_bf16(lo, hi),
                    "from_f32_pair of f32 bits ({lo_bits:#010x}, {hi_bits:#010x})"
                );
                assert_eq!(
                    got.lo().to_bits(),
                    Bf16::from_f32(lo).to_bits(),
                    "low half of from_f32_pair of f32 bits ({lo_bits:#010x}, {hi_bits:#010x})"
                );
                assert_eq!(
                    got.hi().to_bits(),
                    Bf16::from_f32(hi).to_bits(),
                    "high half of from_f32_pair of f32 bits ({lo_bits:#010x}, {hi_bits:#010x})"
                );
            }
        }
    });
}

/// Pins `Bf16x2::to_f32_pair` to `cvt_f32x2_bf16x2`, to `lo()` / `hi()`
/// widened, and to a lossless round trip through `from_f32_pair`.
#[test]
fn to_f32_pair_returns_low_half_first() {
    let (lo, hi) = Bf16x2::from_bits(0x4000_3F80).to_f32_pair();
    assert_eq!((lo, hi), (1.0, 2.0));
    let (lo, hi) = Bf16x2::from_bits(0xBF80_3F80).to_f32_pair();
    assert_eq!((lo, hi), (1.0, -1.0));
    let (lo, hi) = Bf16x2::from_bits(0x0000_8000).to_f32_pair();
    assert_eq!((lo.to_bits(), hi.to_bits()), (0x8000_0000, 0x0000_0000));

    for sweep in 0..=u16::MAX {
        for fixed in FIXED_HALVES {
            for (lo, hi) in [(sweep, fixed), (fixed, sweep)] {
                let bits = u32::from(lo) | (u32::from(hi) << 16);
                let packed = Bf16x2::from_bits(bits);
                let (got_lo, got_hi) = packed.to_f32_pair();
                let (want_lo, want_hi) = cvt_f32x2_bf16x2(bits);
                assert_eq!(
                    (got_lo.to_bits(), got_hi.to_bits()),
                    (want_lo.to_bits(), want_hi.to_bits()),
                    "to_f32_pair of packed bits {bits:#010x}"
                );
                assert_eq!(
                    (got_lo.to_bits(), got_hi.to_bits()),
                    (u32::from(lo) << 16, u32::from(hi) << 16),
                    "to_f32_pair of packed bits {bits:#010x}"
                );
                assert_eq!(
                    (got_lo.to_bits(), got_hi.to_bits()),
                    (
                        packed.lo().to_f32().to_bits(),
                        packed.hi().to_f32().to_bits()
                    ),
                    "to_f32_pair against lo/hi of packed bits {bits:#010x}"
                );
                assert_eq!(
                    Bf16x2::from_f32_pair(got_lo, got_hi).to_bits(),
                    bits,
                    "pair round trip of packed bits {bits:#010x}"
                );
            }
        }
    }
}

/// Pins `bytemuck::Zeroable` for both types: the zeroed value has all-zero
/// bits.
#[test]
fn zeroed_values_are_positive_zero() {
    let zero: Bf16 = bytemuck::Zeroable::zeroed();
    assert_eq!(zero.to_bits(), 0x0000);
    let zero: Bf16x2 = bytemuck::Zeroable::zeroed();
    assert_eq!(zero.to_bits(), 0x0000_0000);
}

/// Pins `bytemuck` casts between `Bf16` and `u16`, for shared slices,
/// mutable slices, and single values, with bits unchanged.
#[test]
fn bf16_slices_cast_to_and_from_u16() {
    let raw: [u16; 6] = [0x0000, 0x8000, 0x3F80, 0x7F80, 0x7FC1, 0xFFFF];

    let typed: &[Bf16] = bytemuck::cast_slice(&raw);
    assert_eq!(typed.len(), raw.len());
    for (value, bits) in typed.iter().zip(raw) {
        assert_eq!(value.to_bits(), bits);
    }

    let values: Vec<Bf16> = raw.iter().map(|&bits| Bf16::from_bits(bits)).collect();
    let back: &[u16] = bytemuck::cast_slice(&values);
    assert_eq!(back, raw);

    let mut storage = raw;
    let writable: &mut [Bf16] = bytemuck::cast_slice_mut(&mut storage);
    writable[2] = Bf16::from_bits(0x4000);
    assert_eq!(storage, [0x0000, 0x8000, 0x4000, 0x7F80, 0x7FC1, 0xFFFF]);

    assert_eq!(bytemuck::cast::<u16, Bf16>(0xBF80).to_bits(), 0xBF80);
    assert_eq!(bytemuck::cast::<Bf16, u16>(Bf16::from_bits(0xBF80)), 0xBF80);
}

/// Pins `bytemuck` casts between `Bf16x2` and `u32`, for shared slices,
/// mutable slices, and single values, with bits unchanged.
#[test]
fn bf16x2_slices_cast_to_and_from_u32() {
    let raw: [u32; 5] = [
        0x0000_0000,
        0x4000_3F80,
        0xBF80_3F80,
        0x7FC1_FF80,
        0xFFFF_FFFF,
    ];

    let typed: &[Bf16x2] = bytemuck::cast_slice(&raw);
    assert_eq!(typed.len(), raw.len());
    for (value, bits) in typed.iter().zip(raw) {
        assert_eq!(value.to_bits(), bits);
    }

    let values: Vec<Bf16x2> = raw.iter().map(|&bits| Bf16x2::from_bits(bits)).collect();
    let back: &[u32] = bytemuck::cast_slice(&values);
    assert_eq!(back, raw);

    let mut storage = raw;
    let writable: &mut [Bf16x2] = bytemuck::cast_slice_mut(&mut storage);
    writable[0] = Bf16x2::new(Bf16::from_bits(0x3F80), Bf16::from_bits(0xC000));
    assert_eq!(storage[0], 0xC000_3F80);

    assert_eq!(
        bytemuck::cast::<u32, Bf16x2>(0xBF80_3F80).to_bits(),
        0xBF80_3F80
    );
    assert_eq!(
        bytemuck::cast::<Bf16x2, u32>(Bf16x2::from_bits(0xBF80_3F80)),
        0xBF80_3F80
    );
}

/// Pins the cast of a `Bf16x2` slice to a `Bf16` slice of twice the length,
/// with the half order that the target byte order gives.
#[test]
fn a_packed_slice_views_as_scalar_values_in_memory_order() {
    let packed = [
        Bf16x2::new(Bf16::from_bits(0x3F80), Bf16::from_bits(0x4000)),
        Bf16x2::new(Bf16::from_bits(0xBF80), Bf16::from_bits(0x7F80)),
    ];
    let scalars: &[Bf16] = bytemuck::cast_slice(&packed);
    let bits: Vec<u16> = scalars.iter().map(|value| value.to_bits()).collect();
    // The low half is the first value in memory on a little-endian target and
    // the second on a big-endian one.
    if cfg!(target_endian = "little") {
        assert_eq!(bits, [0x3F80, 0x4000, 0xBF80, 0x7F80]);
    } else {
        assert_eq!(bits, [0x4000, 0x3F80, 0x7F80, 0xBF80]);
    }
}
