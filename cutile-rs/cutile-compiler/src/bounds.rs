/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer interval arithmetic for static bounds tracking.
//! Used by the compiler to propagate and check value ranges at compile time.

use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use syn::BinOp;

// ---------------------------------------------------------------------------
// TileBinaryOp — lives here so both old and new compiler can share it
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
/// Enumeration of all supported binary operations in the CUDA Tile IR.
pub enum TileBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    CeilDiv,
    TrueDiv,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Maps a string operation name (e.g. `"add"`, `"ceil_div"`) to a [`TileBinaryOp`].
pub fn get_binary_op_from_op_str(op_str: &str) -> Result<TileBinaryOp, JITError> {
    match op_str {
        "add" => Ok(TileBinaryOp::Add),
        "sub" => Ok(TileBinaryOp::Sub),
        "mul" => Ok(TileBinaryOp::Mul),
        "div" => Ok(TileBinaryOp::Div),
        "ceil_div" => Ok(TileBinaryOp::CeilDiv),
        "true_div" => Ok(TileBinaryOp::TrueDiv),
        "rem" => Ok(TileBinaryOp::Rem),
        "eq" => Ok(TileBinaryOp::Eq),
        "ne" => Ok(TileBinaryOp::Ne),
        "lt" => Ok(TileBinaryOp::Lt),
        "le" => Ok(TileBinaryOp::Le),
        "gt" => Ok(TileBinaryOp::Gt),
        "ge" => Ok(TileBinaryOp::Ge),
        "min" | "min_tile" => Ok(TileBinaryOp::Min),
        "max" | "max_tile" => Ok(TileBinaryOp::Max),
        "and" => Ok(TileBinaryOp::BitAnd),
        "or" => Ok(TileBinaryOp::BitOr),
        "xor" => Ok(TileBinaryOp::BitXor),
        "shl" => Ok(TileBinaryOp::Shl),
        "shr" => Ok(TileBinaryOp::Shr),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("unrecognized arithmetic operation `{op_str}`")),
    }
}

/// Converts a Rust `syn::BinOp` to the corresponding [`TileBinaryOp`].
pub fn get_tile_bop_from_rust_bop(rust_bin_op: &BinOp) -> Result<TileBinaryOp, JITError> {
    match rust_bin_op {
        BinOp::Add(_) => Ok(TileBinaryOp::Add),
        BinOp::Sub(_) => Ok(TileBinaryOp::Sub),
        BinOp::Mul(_) => Ok(TileBinaryOp::Mul),
        BinOp::Div(_) => Ok(TileBinaryOp::Div),
        BinOp::Rem(_) => Ok(TileBinaryOp::Rem),
        BinOp::Eq(_) => Ok(TileBinaryOp::Eq),
        BinOp::Ne(_) => Ok(TileBinaryOp::Ne),
        BinOp::Lt(_) => Ok(TileBinaryOp::Lt),
        BinOp::Le(_) => Ok(TileBinaryOp::Le),
        BinOp::Gt(_) => Ok(TileBinaryOp::Gt),
        BinOp::Ge(_) => Ok(TileBinaryOp::Ge),
        BinOp::BitAnd(_) => Ok(TileBinaryOp::BitAnd),
        BinOp::BitOr(_) => Ok(TileBinaryOp::BitOr),
        BinOp::BitXor(_) => Ok(TileBinaryOp::BitXor),
        BinOp::And(_) => Ok(TileBinaryOp::BitAnd),
        BinOp::Or(_) => Ok(TileBinaryOp::BitOr),
        BinOp::Shl(_) => Ok(TileBinaryOp::Shl),
        BinOp::Shr(_) => Ok(TileBinaryOp::Shr),
        _ => SourceLocation::unknown().jit_error_result("this binary operator is not supported"),
    }
}

// TODO (hme): Look into bounds for types other than i64.

fn div_ceil_i64(lhs: i64, rhs: i64) -> i64 {
    // i64::MIN / -1 overflows i64 unconditionally; saturate like the
    // `Div` impl does (checked_rem is None only in that same case, where
    // the remainder is 0).
    let quotient = lhs.saturating_div(rhs);
    let remainder = lhs.checked_rem(rhs).unwrap_or(0);
    if remainder == 0 {
        quotient
    } else if (lhs > 0) == (rhs > 0) {
        quotient.saturating_add(1)
    } else {
        quotient
    }
}

/// An inclusive interval `[start, end]` over a copyable type.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Bounds<T: Copy + PartialEq> {
    pub start: T, // Inclusive.
    pub end: T,   // Inclusive.
}

impl<T: Copy + PartialEq> Bounds<T> {
    /// Creates a new bounds interval from `start` to `end` (both inclusive).
    pub fn new(start: T, end: T) -> Bounds<T> {
        Self { start, end }
    }
    /// Creates an exact (single-value) bounds where `start == end`.
    pub fn exact(value: T) -> Bounds<T> {
        Self {
            start: value,
            end: value,
        }
    }
    /// Returns `true` if this interval represents a single known value.
    pub fn is_exact(&self) -> bool {
        self.end == self.start
    }
}

impl Add for Bounds<i64> {
    type Output = Bounds<i64>;
    fn add(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        // Saturating so overflow widens to the i64 range (a sound over-
        // approximation) rather than panicking (debug) or wrapping (release,
        // which would be an unsound bound).
        let possible_bounds = [
            a.start.saturating_add(b.start),
            a.start.saturating_add(b.end),
            a.end.saturating_add(b.start),
            a.end.saturating_add(b.end),
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Sub for Bounds<i64> {
    type Output = Bounds<i64>;
    fn sub(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        // Saturating: see `Add` — overflow widens to the i64 range soundly.
        let possible_bounds = [
            a.start.saturating_sub(b.start),
            a.start.saturating_sub(b.end),
            a.end.saturating_sub(b.start),
            a.end.saturating_sub(b.end),
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Mul for Bounds<i64> {
    type Output = Bounds<i64>;
    fn mul(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        // Saturating: see `Add` — products can overflow i64 for wide inputs;
        // widening to the i64 range keeps the bound sound.
        let possible_bounds = [
            a.start.saturating_mul(b.start),
            a.start.saturating_mul(b.end),
            a.end.saturating_mul(b.start),
            a.end.saturating_mul(b.end),
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Div for Bounds<i64> {
    type Output = Bounds<i64>;
    fn div(self, rhs: Bounds<i64>) -> Bounds<i64> {
        // For signed integer division:
        // - The minimum is when the numerator is smallest and divisor is largest and non-zero.
        // - The maximum is when the numerator is largest and divisor is smallest and non-zero.
        // If all values are non-zero and positive, the solution is the following
        // min = div(a.start, b.end)
        // max = div(a.start, b.start)
        // Since we permit signed values, it's easier to just take the min/max of all possible bounds.
        let a = self;
        let b = rhs;
        match (b.start, b.end) {
            (0, 0) => panic!("Division by zero"),
            (_, 0) => panic!("Division by zero"),
            (0, _) => panic!("Division by zero"),
            _ => {
                // Saturating: the only signed-division overflow is
                // `i64::MIN / -1`; saturating it to `i64::MAX` keeps the bound
                // sound instead of panicking. Divisors are non-zero here (the
                // caller of `bounds_from_bop` rejects zero divisors).
                let possible_bounds = [
                    a.start.saturating_div(b.start),
                    a.start.saturating_div(b.end),
                    a.end.saturating_div(b.start),
                    a.end.saturating_div(b.end),
                ];
                let start = *possible_bounds
                    .iter()
                    .min()
                    .expect("Unexpected failed min op.");
                let end = *possible_bounds
                    .iter()
                    .max()
                    .expect("Unexpected failed max op.");
                Bounds::new(start, end)
            }
        }
    }
}

impl Rem for Bounds<i64> {
    type Output = Bounds<i64>;
    /// The caller must exclude zero from `rhs` ([`bounds_from_bop`] rejects
    /// zero-straddling divisors before applying `%`).
    fn rem(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        // Corner sampling is wrong for `%`: extrema are not at the interval
        // corners (e.g. [0, 3] % [3, 3] corners to [0, 0], but 2 % 3 = 2).
        // Instead, bound by the residue-range facts of truncated remainder:
        // |a % b| <= |b| - 1, |a % b| <= |a|, and the result's sign follows
        // the dividend's sign.
        debug_assert!(
            !(b.start <= 0 && 0 <= b.end),
            "Rem bounds require a divisor interval that excludes zero"
        );
        if a.is_exact() && b.is_exact() {
            // checked_rem is None only for a zero divisor (excluded by the
            // caller) and for i64::MIN % -1, whose value is 0.
            return Bounds::exact(a.start.checked_rem(b.start).unwrap_or(0));
        }
        // unsigned_abs, not saturating_abs: |i64::MIN| - 1 == i64::MAX must
        // not be shrunk by saturation (i64::MAX % i64::MIN == i64::MAX, which
        // an m of i64::MAX - 1 would wrongly exclude). The divisor excludes
        // zero, so max unsigned_abs >= 1 and `- 1` cannot underflow.
        let m = (b.start.unsigned_abs().max(b.end.unsigned_abs()) - 1) as i64;
        let start = if a.start >= 0 { 0 } else { a.start.max(-m) };
        let end = if a.end <= 0 { 0 } else { a.end.min(m) };
        Bounds::new(start, end)
    }
}

/// Computes the output bounds of a binary operation `f` applied to two intervals.
pub fn bop_bounds<F: Fn(i64, i64) -> i64>(a: &Bounds<i64>, b: &Bounds<i64>, f: F) -> Bounds<i64> {
    // Compute bounds for various binary operations.
    // In general, the new bounds (for valid inputs) are:
    // start = min(op(a.start, b.start), op(a.start, b.end), op(a.end, b.start), op(a.end, b.end))
    // end = max(op(a.start, b.start), op(a.start, b.end), op(a.end, b.start), op(a.end, b.end))
    if a.is_exact() && b.is_exact() {
        return Bounds::exact(f(a.start, b.start));
    }
    let possible_bounds = [
        f(a.start, b.start),
        f(a.start, b.end),
        f(a.end, b.start),
        f(a.end, b.end),
    ];
    let start = *possible_bounds
        .iter()
        .min()
        .expect("Unexpected failed min op.");
    let end = *possible_bounds
        .iter()
        .max()
        .expect("Unexpected failed max op.");
    Bounds::new(start, end)
}

/// Computes sound output bounds for the bitwise ops (`&`, `|`, `^`).
///
/// Corner sampling is wrong for bitwise ops: extrema are not at the interval
/// corners. For a in [5, 8] and b in [6, 7], 6 & 6 = 6 exceeds every corner
/// sample (max 5); for a, b in [0, 2], 2 ^ 1 = 3 exceeds every corner sample
/// (max 2).
///
/// Instead, bound by the two's-complement envelope: if both operands fit in a
/// k-bit signed integer, so does the result, because every bit above the low
/// k is a copy of the sign bit, and a bitwise op maps uniform high bits to
/// uniform high bits. When both operands are non-negative the envelope
/// tightens per op: `a & b <= min(a, b)`, `a | b >= max(a, b)`, and `|` / `^`
/// cannot set a bit above the highest bit of either operand.
fn bitwise_bounds(op: &TileBinaryOp, a: &Bounds<i64>, b: &Bounds<i64>) -> Bounds<i64> {
    let f = |a: i64, b: i64| match op {
        TileBinaryOp::BitAnd => a & b,
        TileBinaryOp::BitOr => a | b,
        TileBinaryOp::BitXor => a ^ b,
        _ => unreachable!(),
    };
    if a.is_exact() && b.is_exact() {
        return Bounds::exact(f(a.start, b.start));
    }
    if a.start >= 0 && b.start >= 0 {
        // Smallest 2^n - 1 covering v, i.e. all-ones over v's highest set bit.
        let ones_over = |v: i64| match 64 - v.leading_zeros() {
            0 => 0,
            n if n >= 63 => i64::MAX,
            n => (1i64 << n) - 1,
        };
        return match op {
            TileBinaryOp::BitAnd => Bounds::new(0, a.end.min(b.end)),
            TileBinaryOp::BitOr => Bounds::new(a.start.max(b.start), ones_over(a.end | b.end)),
            TileBinaryOp::BitXor => Bounds::new(0, ones_over(a.end | b.end)),
            _ => unreachable!(),
        };
    }
    // Signed case: smallest k such that every endpoint fits in k-bit two's
    // complement, i.e. -2^(k-1) <= v <= 2^(k-1) - 1.
    let signed_bits = |v: i64| {
        if v >= 0 {
            65 - v.leading_zeros()
        } else {
            65 - v.leading_ones()
        }
    };
    let k = [a.start, a.end, b.start, b.end]
        .into_iter()
        .map(signed_bits)
        .max()
        .expect("Unexpected failed max op.");
    if k >= 64 {
        Bounds::new(i64::MIN, i64::MAX)
    } else {
        Bounds::new(-(1i64 << (k - 1)), (1i64 << (k - 1)) - 1)
    }
}

/// Returns the result bounds for a [`TileBinaryOp`], or `None` on division by zero.
pub fn bounds_from_bop(op: &TileBinaryOp, a: &Bounds<i64>, b: &Bounds<i64>) -> Option<Bounds<i64>> {
    match op {
        // Shift intervals are not corner-monotone for negative or oversized
        // operands; stay conservative.
        TileBinaryOp::Shl | TileBinaryOp::Shr => None,
        TileBinaryOp::CeilDiv | TileBinaryOp::Div | TileBinaryOp::TrueDiv | TileBinaryOp::Rem => {
            // Any op with a divisor is handled here so it can reject a zero
            // divisor. Reject when zero lies anywhere in the divisor's inclusive
            // range, not just at an endpoint: an interior zero (e.g. `b=[-1, 2]`)
            // is a genuine division by zero the caller must handle, and it also
            // makes the corner sampling unsound (quotients diverge as the divisor
            // approaches 0). For `Rem`, this guard is also what prevents a `% 0`
            // panic — the previous code applied `%` with no zero check.
            if b.start <= 0 && 0 <= b.end {
                None
            } else {
                Some(match op {
                    TileBinaryOp::Div | TileBinaryOp::TrueDiv => *a / *b,
                    TileBinaryOp::CeilDiv => bop_bounds(a, b, div_ceil_i64),
                    TileBinaryOp::Rem => *a % *b,
                    _ => unreachable!(),
                })
            }
        }
        _ => Some(match op {
            TileBinaryOp::Add => *a + *b,
            TileBinaryOp::Sub => *a - *b,
            TileBinaryOp::Mul => *a * *b,
            TileBinaryOp::Eq => {
                // Here we use overlap analysis instead of `bop_bounds`, because `f(x,y) = (x==y)` is
                // not monotone, which can lead to the condition being observed as statically false,
                // interpreted as dead code by the compiler and is silently removed.
                if a.is_exact() && b.is_exact() {
                    Bounds::exact((a.start == b.start) as i64)
                } else if a.end < b.start || b.end < a.start {
                    Bounds::exact(0) // disjoint - never equal
                } else {
                    Bounds::new(0, 1) // overlap - unknown
                }
            }
            TileBinaryOp::Ne => {
                // Same as above
                if a.is_exact() && b.is_exact() {
                    Bounds::exact((a.start != b.start) as i64)
                } else if a.end < b.start || b.end < a.start {
                    Bounds::exact(1) // disjoint - always not equal
                } else {
                    Bounds::new(0, 1) // overlap - unknown
                }
            }
            TileBinaryOp::Lt => bop_bounds(a, b, |a, b| (a < b) as i64),
            TileBinaryOp::Le => bop_bounds(a, b, |a, b| (a <= b) as i64),
            TileBinaryOp::Gt => bop_bounds(a, b, |a, b| (a > b) as i64),
            TileBinaryOp::Ge => bop_bounds(a, b, |a, b| (a >= b) as i64),
            TileBinaryOp::Min => bop_bounds(a, b, |a, b| a.min(b)),
            TileBinaryOp::Max => bop_bounds(a, b, |a, b| a.max(b)),
            TileBinaryOp::BitAnd | TileBinaryOp::BitOr | TileBinaryOp::BitXor => {
                bitwise_bounds(op, a, b)
            }
            _ => unreachable!(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bnd(start: i64, end: i64) -> Bounds<i64> {
        Bounds::new(start, end)
    }

    // --- Division-by-zero guard (div / ceil_div / true_div / rem) ------------

    #[test]
    fn div_rejects_zero_at_an_endpoint() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Div, &bnd(1, 10), &bnd(0, 5)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Div, &bnd(1, 10), &bnd(-5, 0)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Div, &bnd(1, 10), &bnd(0, 0)),
            None
        );
    }

    #[test]
    fn div_rejects_zero_in_the_interior() {
        // Previously accepted (endpoints are non-zero), which both missed a real
        // division by zero and produced an unsound bound: 100 / [-1, 2] can reach
        // 100 (at b = 1) but corner sampling gives only [-100, 50].
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Div, &bnd(100, 100), &bnd(-1, 2)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::CeilDiv, &bnd(1, 10), &bnd(-1, 3)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::TrueDiv, &bnd(1, 10), &bnd(-2, 4)),
            None
        );
    }

    #[test]
    fn div_accepts_sign_consistent_divisor() {
        // Divisor entirely positive: corner bounds are sound.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Div, &bnd(0, 10), &bnd(2, 2)),
            Some(bnd(0, 5))
        );
    }

    #[test]
    fn rem_rejects_zero_divisor_instead_of_panicking() {
        // Previously `a % 0` panicked the compiler; now it is rejected like div.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 10), &bnd(0, 3)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 10), &bnd(-1, 2)),
            None
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 10), &bnd(0, 0)),
            None
        );
    }

    #[test]
    fn rem_accepts_nonzero_divisor() {
        assert!(bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 3), &bnd(3, 3)).is_some());
    }

    // --- Rem range analysis --------------------------------------------------

    #[test]
    fn rem_covers_the_full_residue_range() {
        // Corner sampling gave [0, 0] here (0 % 3 = 3 % 3 = 0), dropping the
        // reachable 2 % 3 = 2.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 3), &bnd(3, 3)),
            Some(bnd(0, 2))
        );
    }

    #[test]
    fn rem_sign_follows_the_dividend() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(-7, -4), &bnd(3, 3)),
            Some(bnd(-2, 0))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(-3, 3), &bnd(2, 3)),
            Some(bnd(-2, 2))
        );
    }

    #[test]
    fn rem_is_clamped_by_a_small_dividend() {
        // |a % b| <= |a|: a tiny dividend can't produce a large residue.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(0, 1), &bnd(100, 100)),
            Some(bnd(0, 1))
        );
    }

    #[test]
    fn rem_exact_operands_stay_exact() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Rem, &bnd(7, 7), &bnd(3, 3)),
            Some(bnd(1, 1))
        );
    }

    #[test]
    fn rem_divisor_at_i64_min_covers_extreme_residues() {
        // |i64::MIN| - 1 == i64::MAX; a saturating_abs-based bound loses one
        // and wrongly excludes i64::MAX % i64::MIN == i64::MAX.
        assert_eq!(
            bounds_from_bop(
                &TileBinaryOp::Rem,
                &bnd(i64::MAX - 1, i64::MAX),
                &bnd(i64::MIN, i64::MIN)
            ),
            Some(bnd(0, i64::MAX))
        );
        assert_eq!(
            bounds_from_bop(
                &TileBinaryOp::Rem,
                &bnd(i64::MIN, i64::MIN + 1),
                &bnd(i64::MIN, i64::MIN)
            ),
            Some(bnd(i64::MIN + 1, 0))
        );
    }

    // --- Eq / Ne overlap analysis --------------------------------------------

    #[test]
    fn eq_disjoint_ranges_are_never_equal() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Eq, &bnd(0, 2), &bnd(5, 9)),
            Some(bnd(0, 0))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Ne, &bnd(0, 2), &bnd(5, 9)),
            Some(bnd(1, 1))
        );
    }

    #[test]
    fn eq_overlapping_ranges_are_unknown() {
        // Corner sampling gave an exact [0, 0] here (no corner pair is equal),
        // so `a == b` was treated as statically false and the branch was
        // silently dropped as dead code.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Eq, &bnd(0, 5), &bnd(3, 9)),
            Some(bnd(0, 1))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Ne, &bnd(0, 5), &bnd(3, 9)),
            Some(bnd(0, 1))
        );
        // Ranges touching at a single point still overlap.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Eq, &bnd(0, 3), &bnd(3, 5)),
            Some(bnd(0, 1))
        );
    }

    #[test]
    fn eq_exact_operands_are_decided() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Eq, &bnd(4, 4), &bnd(4, 4)),
            Some(bnd(1, 1))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Eq, &bnd(4, 4), &bnd(5, 5)),
            Some(bnd(0, 0))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Ne, &bnd(4, 4), &bnd(4, 4)),
            Some(bnd(0, 0))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Ne, &bnd(4, 4), &bnd(5, 5)),
            Some(bnd(1, 1))
        );
    }

    // --- Bitwise envelope -----------------------------------------------------

    #[test]
    fn xor_covers_interior_values() {
        // Corner sampling gave [0, 2] here, dropping the reachable 2 ^ 1 = 3.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitXor, &bnd(0, 2), &bnd(0, 2)),
            Some(bnd(0, 3))
        );
    }

    #[test]
    fn and_covers_interior_values() {
        // Corner sampling gave [0, 5] here, dropping the reachable 6 & 6 = 6.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitAnd, &bnd(5, 8), &bnd(6, 7)),
            Some(bnd(0, 7))
        );
    }

    #[test]
    fn or_is_bounded_below_by_its_operands() {
        // a | b >= max(a, b) for non-negative operands.
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitOr, &bnd(4, 6), &bnd(1, 3)),
            Some(bnd(4, 7))
        );
    }

    #[test]
    fn bitwise_signed_operands_use_the_twos_complement_envelope() {
        // Endpoints all fit in 3-bit two's complement, so results lie in
        // [-4, 3]; -2 ^ 2 = -4 shows the lower bound is reachable.
        for op in [
            TileBinaryOp::BitAnd,
            TileBinaryOp::BitOr,
            TileBinaryOp::BitXor,
        ] {
            assert_eq!(
                bounds_from_bop(&op, &bnd(-2, 2), &bnd(0, 3)),
                Some(bnd(-4, 3))
            );
        }
    }

    #[test]
    fn bitwise_exact_operands_stay_exact() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitXor, &bnd(6, 6), &bnd(3, 3)),
            Some(bnd(5, 5))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitAnd, &bnd(6, 6), &bnd(3, 3)),
            Some(bnd(2, 2))
        );
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::BitOr, &bnd(6, 6), &bnd(3, 3)),
            Some(bnd(7, 7))
        );
    }

    // --- Saturating arithmetic: overflow widens soundly, never panics -------

    #[test]
    fn add_saturates_on_overflow() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Add, &bnd(i64::MAX, i64::MAX), &bnd(1, 1)),
            Some(bnd(i64::MAX, i64::MAX))
        );
    }

    #[test]
    fn mul_saturates_on_overflow() {
        assert_eq!(
            bounds_from_bop(&TileBinaryOp::Mul, &bnd(0, i64::MAX), &bnd(2, 2)),
            Some(bnd(0, i64::MAX))
        );
    }

    #[test]
    fn ceil_div_saturates_i64_min_over_negative_one() {
        // i64::MIN / -1 overflows i64; the corner must saturate, not panic.
        assert_eq!(
            bounds_from_bop(
                &TileBinaryOp::CeilDiv,
                &bnd(i64::MIN, i64::MIN),
                &bnd(-1, -1)
            ),
            Some(bnd(i64::MAX, i64::MAX))
        );
        assert_eq!(
            bounds_from_bop(
                &TileBinaryOp::CeilDiv,
                &bnd(i64::MIN, i64::MAX),
                &bnd(-1, -1)
            ),
            Some(bnd(-i64::MAX, i64::MAX))
        );
    }
}
