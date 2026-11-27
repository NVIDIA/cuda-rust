/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device math that avoids local memory.
//!
//! `f32::sin` / `f32::cos` lower to libdevice `__nv_sinf` / `__nv_cosf`. Their
//! argument reduction has two paths: a three-constant Cody-Waite reduction for
//! `|x| < 105615`, and a Payne-Hanek reduction for larger arguments that keeps
//! a seven-word table in local memory. Every inlined call carries both, so a
//! kernel that only ever sees small angles still gets a local-memory frame,
//! local loads and stores, and several hundred instructions per call site.
//!
//! [`sin_cos`] is the first path alone: the same constants, the same
//! correctly-rounded operations in the same order, hence the same bits as
//! libdevice for every argument it accepts. One reduction serves both results.
//! It is plain Rust (`core` float math), so host code, such as a CPU twin of a
//! device function, computes the same bits instead of hitting a device-only
//! intrinsic.

use core::f32::math::{mul_add, round_ties_even};

/// `2/pi`, rounded (libdevice `0x3F22F983`).
const TWO_OVER_PI: f32 = f32::from_bits(0x3F22_F983);
/// `pi/2` split in three parts for the Cody-Waite reduction (negated, libdevice order).
const NEG_PI_2_HI: f32 = f32::from_bits(0xBFC9_0FDA);
const NEG_PI_2_MID: f32 = f32::from_bits(0xB3A2_2168);
const NEG_PI_2_LO: f32 = f32::from_bits(0xA7C2_34C5);
/// Largest argument magnitude the Cody-Waite path covers (libdevice switches at it).
pub const SIN_COS_MAX_ARG: f32 = 105_615.0;

/// `(sin x, cos x)`, bit-identical to libdevice `__nv_sinf` / `__nv_cosf` (non-FTZ)
/// for `|x| < SIN_COS_MAX_ARG`, without their local-memory slow path.
///
/// The caller guarantees the bound; every angle derived from a direction, a few
/// turns of rotation or a sampled `2 * pi * u` satisfies it. Beyond it the
/// reduction is not exact and accuracy degrades with `|x|` instead of switching
/// to the Payne-Hanek path. NaN maps to NaN.
///
/// Every step is a single correctly-rounded operation that the backend cannot
/// fuse differently from libdevice: the product `x * 2/pi` feeds a rounding,
/// not an addition, and every multiply-add is an explicit `mul_add`.
#[inline(always)]
pub fn sin_cos(x: f32) -> (f32, f32) {
    // libdevice: an integer q = cvt.rni(x * 2/pi) (ties to even), converted back to
    // f32. The round trip through i32 matters: it turns -0 into +0, which keeps the
    // sign of a zero x in t. |q| < 2^17 here, so both conversions are exact.
    let quadrant = round_ties_even(x * TWO_OVER_PI) as i32;
    let q = quadrant as f32;
    let t = mul_add(q, NEG_PI_2_HI, x);
    let t = mul_add(q, NEG_PI_2_MID, t);
    let t = mul_add(q, NEG_PI_2_LO, t);
    let quadrant = quadrant as u32;
    (
        sin_reduced(t, quadrant),
        sin_reduced(t, quadrant.wrapping_add(1)),
    )
}

/// libdevice `__internal_accurate_sinf` after the reduction: `sin(t + quadrant * pi/2)`
/// for `|t| <= pi/4`, an odd (sine) or even (cosine) minimax polynomial by the
/// quadrant's low bit, negated when its second bit is set.
#[inline(always)]
fn sin_reduced(t: f32, quadrant: u32) -> f32 {
    let s = t * t;
    let cosine = quadrant & 1 != 0;
    let lead = if cosine { 1.0 } else { t };
    let lead_s = mul_add(s, lead, 0.0);
    let p = if cosine {
        mul_add(f32::from_bits(0x37CB_AC00), s, f32::from_bits(0xBAB6_07ED))
    } else {
        f32::from_bits(0xB94D_4153)
    };
    let p = mul_add(
        p,
        s,
        f32::from_bits(if cosine { 0x3D2A_AABB } else { 0x3C08_85E4 }),
    );
    let p = mul_add(
        p,
        s,
        f32::from_bits(if cosine { 0xBEFF_FFFF } else { 0xBE2A_AAA8 }),
    );
    let r = mul_add(p, lead_s, lead);
    // libdevice negates with fma(r, -1, 0), which keeps +0 for r = +0.
    if quadrant & 2 != 0 {
        mul_add(r, -1.0, 0.0)
    } else {
        r
    }
}
