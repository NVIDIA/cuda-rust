/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-value dataflow facts: the forward, monotone abstract-interpretation
//! lattices the compiler propagates over scalar values, gathered in one place.
//!
//! Two lattices, propagated together by [`transfer`]:
//! - **interval** — [`Bounds`] (`crate::bounds`), a concrete inclusive
//!   `[start, end]` range. The constant / static rung of bounds checks consumes
//!   it, and an *exact* range folds to a compile-time constant.
//! - **symbolic** — [`Term`] (`cuda_async::predicate`), a canonical linear form
//!   `sum(coeff·atom) + constant`. Loop check-hoisting consumes it via
//!   `Term::as_single_affine`; launch-time check hoisting consumes it via the
//!   obligation solver (`passes::obligation::resolve`).
//!
//! The two are not independent: **an interval is the range-abstraction of the
//! symbolic term.** [`term_range`] computes a `Bounds` from a `Term` given a
//! range per atom — the constant case (a term with no atoms) is just its
//! constant folded to an exact range. This is LLVM SCEV's `getRange` / MLIR's
//! `IntegerRangeAnalysis` reduction, and it is why the two lattices can share a
//! module rather than duplicate interval arithmetic.
//!
//! (Provenance brands — `PartitionAxisOrigin` / `DimOrigin` in
//! `compiler::_value` — are the third, origin, facet of a value's facts; they
//! propagate structurally rather than arithmetically and are categorized here
//! by reference until they move in.)

use crate::bounds::{bounds_from_bop, Bounds, TileBinaryOp};
use crate::compiler::_value::TileRustValue;
use cuda_async::predicate::{Atom, Term};

/// The arithmetic facts produced for a binary op's result value.
pub(crate) struct ScalarFacts {
    pub(crate) bounds: Option<Bounds<i64>>,
    pub(crate) term: Option<Term>,
    pub(crate) floor_div: Option<FloorDiv>,
}

/// `floor(numerator / divisor)` with a compile-time-constant `divisor`.
///
/// The device op behind `/` truncates toward zero (Rust semantics; see
/// `compile_binary_op`), which coincides with `floor` exactly when the
/// numerator is non-negative. Every consumer of this residue relates it to a
/// tensor extent — never negative — so the `floor` reading below is exact
/// there; the divisor is required positive at construction for the same
/// reason.
///
/// Integer division is not linear, so it cannot live in [`Term`]. It is kept
/// here, on the *analysis* side, instead of being added to the predicate
/// vocabulary as an opaque atom. That distinction matters: an opaque atom would
/// be evaluable but not reasonable-about — nothing could relate it to the
/// operands it was built from, so an obligation naming it could only ever be
/// shipped to the host, never discharged by a declared `preconditions` fact.
///
/// Keeping the residue here lets an obligation site reduce it to a predicate
/// over the *existing* vocabulary. The reduction that matters:
/// `floor(e/d) == ceil(e/d)` iff `d` divides `e`, so an equality against a tile
/// count becomes [`cuda_async::predicate::Predicate::divisible_by`] over `e` —
/// which a precondition can entail, and the host can decide. This mirrors MLIR,
/// where `floordiv`/`ceildiv` simplification is driven by known divisibility
/// rather than by treating the division as an opaque leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FloorDiv {
    pub(crate) numerator: Term,
    pub(crate) divisor: i64,
}

/// The value domain of a runtime integer type — the range every machine
/// value of that type inhabits. `None` means the domain is unknown (or not
/// an integer), and interval facts must not be kept for it.
///
/// `usize` is deliberately `None`: the DSL relabels it against `i32` and
/// `u32` freely, so no single domain is trustworthy. 64-bit types are also
/// `None`: interval composition saturates at `i64` while the machine wraps,
/// and saturated endpoints cannot be told apart from genuine ones.
pub(crate) fn int_value_domain(elem_ty: &str) -> Option<Bounds<i64>> {
    match elem_ty {
        "bool" | "i1" => Some(Bounds::new(0, 1)),
        "i32" => Some(Bounds::new(i32::MIN as i64, i32::MAX as i64)),
        "u32" => Some(Bounds::new(0, u32::MAX as i64)),
        _ => None,
    }
}

/// The arithmetic transfer function: given an op, its two operands, and the
/// result's value domain, produce the result value's facts (interval +
/// symbolic). One place for both lattices; the caller still owns IR-emission
/// control flow (the exact-range → constant fold and the operand-kind check).
///
/// The interval is kept only when it proves this op cannot wrap: interval
/// composition is mathematical, the machine op wraps at the type width, and
/// the two agree exactly when the mathematical result range fits the type's
/// value domain. A range that escapes the domain — or a domain we cannot
/// name — yields NO interval, because after a wrap the machine value can be
/// anything in the type. Checking only a downstream consumer's final range
/// is not enough: a later `max`/`%`/`min` narrows the mathematical range
/// back into the domain while the wrapped machine value stays outside it
/// (2026-08-12 review, S1).
pub(crate) fn transfer(
    op: &TileBinaryOp,
    lhs: &TileRustValue,
    rhs: &TileRustValue,
    result_domain: Option<Bounds<i64>>,
) -> ScalarFacts {
    let bounds = propagate_bounds(op, lhs, rhs).filter(|b| {
        result_domain.is_some_and(|domain| domain.start <= b.start && b.end <= domain.end)
    });
    ScalarFacts {
        bounds,
        term: propagate_term(op, lhs, rhs),
        floor_div: propagate_floor_div(op, lhs, rhs),
    }
}

/// Record an integer division by a compile-time constant as a [`FloorDiv`]
/// residue. Only a truncating/floor `Div` qualifies, and only when the
/// numerator has a symbolic term to name — otherwise there is nothing an
/// obligation could later reduce it against.
pub(crate) fn propagate_floor_div(
    op: &TileBinaryOp,
    lhs: &TileRustValue,
    rhs: &TileRustValue,
) -> Option<FloorDiv> {
    if !matches!(op, TileBinaryOp::Div) {
        return None;
    }
    let divisor = rhs.bounds.filter(|b| b.is_exact()).map(|b| b.start)?;
    if divisor <= 0 {
        return None;
    }
    Some(FloorDiv {
        numerator: lhs.term.clone()?,
        divisor,
    })
}

/// Interval transfer: propagate `[start, end]` ranges through the op. `None`
/// unless both operands carry a range.
pub(crate) fn propagate_bounds(
    op: &TileBinaryOp,
    lhs: &TileRustValue,
    rhs: &TileRustValue,
) -> Option<Bounds<i64>> {
    match (lhs.bounds, rhs.bounds) {
        (Some(a), Some(b)) => bounds_from_bop(op, &a, &b),
        _ => None,
    }
}

/// Symbolic transfer: propagate the canonical linear [`Term`] through the op.
///
/// Each operand contributes its own `term`, or a constant term synthesized from
/// an exact range (so `i + 3` keeps propagating). A product stays linear only
/// when one side is constant; other ops (division, remainder) and `i64`
/// overflow yield `None` — the term algebra's bail-to-weaker policy. The
/// single-variable affine fragment the loop hoister needs is recovered via
/// [`Term::as_single_affine`].
pub(crate) fn propagate_term(
    op: &TileBinaryOp,
    lhs: &TileRustValue,
    rhs: &TileRustValue,
) -> Option<Term> {
    // Each operand's term: an explicit symbolic form, or a constant from an
    // exact range.
    let term_of = |v: &TileRustValue| -> Option<Term> {
        v.term.clone().or_else(|| {
            v.bounds
                .filter(|b| b.is_exact())
                .map(|b| Term::constant(b.start))
        })
    };
    let lt = term_of(lhs)?;
    let rt = term_of(rhs)?;
    match op {
        TileBinaryOp::Add => lt.add(&rt),
        TileBinaryOp::Sub => lt.sub(&rt),
        TileBinaryOp::Mul => {
            // Linearity: a product stays affine only if one side is constant.
            if let Some(c) = rt.as_constant() {
                lt.mul_const(c)
            } else if let Some(c) = lt.as_constant() {
                rt.mul_const(c)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The interval-from-symbolic reduction (LLVM SCEV `getRange`): the range of an
/// affine `Term = sum(coeff·atom) + constant`, given a range per atom, is
/// `constant + sum(coeff · [lo, hi])` with interval arithmetic (a negative
/// coefficient swaps the endpoints). `None` if any atom lacks a range or the
/// arithmetic overflows `i64`.
///
/// This makes the interval derivable from the symbolic term wherever the term
/// is affine — the const-rung bounds we track today are the special case where
/// the term has no atoms.
///
/// The live consumer is the hoisting classifier: when a loop's induction range
/// is a compile-time constant, an affine index's static range follows from its
/// term via this reduction, so the check discharges as a constant instead of a
/// runtime strongest-instance substitution.
pub(crate) fn term_range(
    term: &Term,
    atom_range: &impl Fn(&Atom) -> Option<Bounds<i64>>,
) -> Option<Bounds<i64>> {
    let mut lo = term.constant_part();
    let mut hi = term.constant_part();
    for (atom, &coeff) in term.coeffs() {
        let r = atom_range(atom)?;
        let (a, b) = (coeff.checked_mul(r.start)?, coeff.checked_mul(r.end)?);
        let (add_lo, add_hi) = if a <= b { (a, b) } else { (b, a) };
        lo = lo.checked_add(add_lo)?;
        hi = hi.checked_add(add_hi)?;
    }
    Some(Bounds { start: lo, end: hi })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(param: usize, axis: usize) -> Atom {
        Atom::Dim { param, axis }
    }

    #[test]
    fn term_range_of_a_constant_is_the_exact_range() {
        let range = term_range(&Term::constant(7), &|_| None).unwrap();
        assert_eq!(range, Bounds { start: 7, end: 7 });
        assert!(range.is_exact());
    }

    #[test]
    fn term_range_of_affine_uses_atom_ranges() {
        // 2*i + 3, with i in [0, 4]  ->  [3, 11].
        let term = Term::affine(dim(0, 0), 2, 3);
        let env = |_: &Atom| Some(Bounds { start: 0, end: 4 });
        assert_eq!(term_range(&term, &env), Some(Bounds { start: 3, end: 11 }));
    }

    #[test]
    fn term_range_negative_coefficient_swaps_endpoints() {
        // -1*i + 10, with i in [0, 4]  ->  [6, 10].
        let term = Term::affine(dim(0, 0), -1, 10);
        let env = |_: &Atom| Some(Bounds { start: 0, end: 4 });
        assert_eq!(term_range(&term, &env), Some(Bounds { start: 6, end: 10 }));
    }

    #[test]
    fn term_range_is_none_when_an_atom_has_no_range() {
        let term = Term::atom(dim(9, 9));
        assert_eq!(term_range(&term, &|_| None), None);
    }

    #[test]
    fn runtime_domains_cover_boolean_and_32_bit_integer_results() {
        assert_eq!(int_value_domain("bool"), Some(Bounds::new(0, 1)));
        assert_eq!(
            int_value_domain("i32"),
            Some(Bounds::new(i32::MIN as i64, i32::MAX as i64))
        );
        assert_eq!(
            int_value_domain("u32"),
            Some(Bounds::new(0, u32::MAX as i64))
        );
        assert_eq!(int_value_domain("i64"), None);
        assert_eq!(int_value_domain("usize"), None);
    }
}
