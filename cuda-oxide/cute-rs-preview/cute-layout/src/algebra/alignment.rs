/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static vector-access alignment derived from layout mappings and division.

use crate::Layout;

use super::mapping::{coalesce, right_inverse};
use super::tiling::logical_divide;
use super::{AlgebraError, Result};

/// Returns the alignment factor guaranteed by the layout's offset pattern.
///
/// This examines only the static strides; it does not inspect a memory
/// pointer. For `(8,16):(1,4)`, the first coordinate gives an eight-offset
/// unit-stride run, while changing the second coordinate shifts that run by
/// four:
///
/// ```text
/// layout coordinate       offset walk
/// (0 through 7, 0)        0 1 2 3 4 5 6 7       run length 8
/// (0 through 7, 1)        4 5 6 7 8 9 10 11     shifted by 4
///
/// greatest common divisor of 8 and 4 = 4 offset units
/// ```
///
/// The result is `4` because width 4 divides both the eight-unit run and its
/// four-unit shift. Four-unit vector chunks therefore have compatible starts.
/// This does **not** prove that the base pointer is aligned; the compiler must
/// check the pointer separately before selecting a vector instruction.
///
/// The unit is whatever the layout's strides count: elements for an
/// element-stride layout or bytes for a byte-stride layout. A negative stride
/// returns `1` because this analysis cannot prove a larger factor for a
/// backwards walk.
pub fn max_alignment(layout: &Layout) -> Result<i64> {
    let shapes = layout.shape.flatten();
    let strides = layout.stride.flatten();
    if shapes.iter().any(|&shape| shape <= 0) {
        return Err(AlgebraError(
            "max_alignment: every shape extent must be positive".into(),
        ));
    }
    if strides.iter().any(|&stride| stride < 0) {
        return Ok(1);
    }

    let flat = coalesce(layout);
    let inverse = right_inverse(&flat)?;
    let permuted = logical_divide(&flat, &inverse)?;
    let modes = permuted.modes();
    if modes.len() != 2 {
        return Err(AlgebraError(
            "max_alignment: expected vector and remainder modes".into(),
        ));
    }

    let vector_size = modes[0].size();
    let rest_stride_gcd = modes[1]
        .stride
        .flatten()
        .into_iter()
        .fold(0, gcd_nonnegative);
    Ok(gcd_nonnegative(vector_size, rest_stride_gcd).max(1))
}

/// Compute the greatest common divisor of nonnegative layout alignment factors.
fn gcd_nonnegative(a: i64, b: i64) -> i64 {
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    // Both inputs came from i64 layout values, so their greatest common
    // divisor also fits in i64.
    a as i64
}
