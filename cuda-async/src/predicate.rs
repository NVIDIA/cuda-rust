/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical predicate / term algebra for staged obligation resolution.
//!
//! Plain data shared by the compiler (which builds and resolves obligations)
//! and the host launcher (which evaluates launch-stage predicates). It lives in
//! this low crate so a *single* representation serves all three stages — the
//! same [`Predicate`] is folded at JIT, evaluated against tensor extents at
//! launch, and (later) lowered to an in-kernel assert.
//!
//! ## Normal form (follows LLVM ScalarEvolution and MLIR `AffineExpr`)
//!
//! A [`Term`] is kept in a canonical form, so **semantic equality is structural
//! equality** (`PartialEq`/`Hash`). For the linear fragment we use a sparse
//! coefficient map rather than a rewritten expression tree: the map makes the
//! operand-ordering, constant-folding and like-term-combination rules those
//! systems apply by hand (e.g. MLIR `simplifyAdd`) automatic and confluent by
//! construction. Anything non-linear (a `mod`/`floordiv` residue, an opaque
//! product) enters as a leaf [`Atom`], exactly as SCEV treats `SCEVUnknown` /
//! `SCEVUDivExpr` as leaves.
//!
//! Overflow follows MLIR (`mlir/lib/IR/AffineExpr.cpp`): a construction that
//! overflows `i64` is *not canonicalizable* and returns `None`, so the caller
//! falls back to a weaker (later) stage. This keeps the algebra sound — an
//! un-formed term simply fails to discharge an obligation early; it never
//! produces a wrong equality.

use std::collections::BTreeMap;

/// The compilation / execution stage at which a value becomes available:
/// `Jit ⊑ Launch ⊑ Device` (earlier is cheaper — fewer device registers).
/// `Ord` is derived with `Jit` smallest, so `max` over a term's atoms yields
/// the earliest stage at which the whole term is evaluable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Jit,
    Launch,
    Device,
}

/// An indivisible term operand, tagged with the stage at which its value is
/// known.
///
/// Atoms are identified by *resolved* symbols, never source names: a tensor
/// axis extent by kernel-parameter position, an induction variable by its
/// opaque IR value id. This makes the launch-known classifier structural — a
/// term's stage is the `max` over its atoms (see [`Term::stage`]), so
/// liftability is computed, not judged per predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// `extent(param, axis)` — a tensor axis extent, identified by kernel
    /// parameter position. Known at [`Stage::Launch`]: the host has every
    /// extent before `cuLaunchKernel`.
    Dim { param: usize, axis: usize },
    /// A kernel-runtime induction variable, by opaque IR value id.
    /// [`Stage::Device`].
    Iv(u32),
    /// A special register: the tile-block id on grid axis `k`
    /// (`get_tile_block_id().k`). [`Stage::Device`] — a per-CTA runtime value.
    TileBlockId(usize),
    /// A special register: the number of tile blocks on grid axis `k`
    /// (`get_num_tile_blocks().k`, i.e. `gridDim.k`). [`Stage::Launch`] — the
    /// host fixes the grid before `cuLaunchKernel`.
    ///
    /// **This is the CTA-*slab* count, not an in-kernel tile count.** It equals
    /// `div_ceil(root_shape, slab_shape)` (`Partition::grid`) — the quantity
    /// `validate_grids` checks the launch grid against. A partition access needs
    /// the tile count *within* the kernel-visible view (`ceil(view_shape /
    /// tile)`), which for a `&mut Tensor` param is tiles inside one slab. **The
    /// two are unrelated**: never treat `NumTileBlocks(k)` as a partition's tile
    /// count without a *verified* `Launch` obligation equating them. Conflating
    /// them once admitted out-of-bounds stores (reverted).
    NumTileBlocks(usize),
    /// `view_extent(param, axis)` — the extent of the parameter's
    /// *kernel-visible view*, identified by kernel parameter position.
    /// [`Stage::Launch`]: the host chose the view when it partitioned.
    ///
    /// **Frame.** [`Atom::Dim`] is the *root* frame (the whole tensor, what
    /// `param_shapes` holds — and the frame declared `preconditions` are stated
    /// in). This atom is the *view* frame: for an immutable `&Tensor` the view
    /// is the whole tensor (same value as `Dim`), but for a slabbed
    /// `&mut Tensor` it is the per-work-item piece, a different quantity. The
    /// two frames are different atom variants precisely so that a root-frame
    /// fact can never entail a view-frame obligation (or vice versa) by
    /// structural equality — the frame confusion that once admitted
    /// out-of-bounds stores is unrepresentable rather than merely avoided.
    ViewExtent { param: usize, axis: usize },
    /// `ceil(extent(param, axis) / tile)` — the number of `tile`-sized tiles
    /// along a tensor axis, identified by kernel parameter position with the
    /// tile size baked into the atom's identity (the same axis at two tile
    /// sizes is two different quantities). [`Stage::Launch`]: the host holds
    /// the extent and the tile is a compile-time constant.
    ///
    /// **Frame: ROOT, like [`Atom::Dim`]** — the whole tensor's tile count,
    /// which is what an index walked from an immutable partition's axis is
    /// bounded by. A view-frame variant (tiles within a `&mut` slab) is the
    /// extension point if a consumer ever needs it; keeping the frames as
    /// separate variants is this module's standing rule.
    ///
    /// This is a NAME, not a fact: like every atom it is a free variable the
    /// host evaluates, and no relationship between `TileCount` and `Dim` is
    /// known to the solver — a goal stated over tile counts cannot be
    /// discharged by a believed extent fact except through explicit code.
    /// `tile` must be >= 1; minting sites enforce it.
    TileCount {
        param: usize,
        axis: usize,
        tile: i32,
    },
}

impl Atom {
    /// The stage at which this atom's value is known.
    pub fn stage(&self) -> Stage {
        match self {
            Atom::Dim { .. }
            | Atom::NumTileBlocks(_)
            | Atom::ViewExtent { .. }
            | Atom::TileCount { .. } => Stage::Launch,
            Atom::Iv(_) | Atom::TileBlockId(_) => Stage::Device,
        }
    }
}

/// A canonical linear integer term: `sum(coeff_i * atom_i) + constant`.
///
/// Normal-form invariant: `coeffs` holds no zero coefficients and is sorted by
/// [`Atom`] (a `BTreeMap`). Two terms are semantically equal iff structurally
/// equal, so `PartialEq`/`Hash` *are* the equality test. Build only through the
/// constructors below — each preserves the invariant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Term {
    coeffs: BTreeMap<Atom, i64>,
    constant: i64,
}

impl Term {
    /// The constant term `c`.
    pub fn constant(c: i64) -> Self {
        Term {
            coeffs: BTreeMap::new(),
            constant: c,
        }
    }

    /// The term `atom` (coefficient 1).
    pub fn atom(a: Atom) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(a, 1);
        Term {
            coeffs,
            constant: 0,
        }
    }

    /// The affine term `scale * atom + offset`. Consolidates the old
    /// `AffineForm { scale, var, offset }`. `None` iff `scale == 0` would be
    /// preferred as a plain constant — instead a zero scale folds to `offset`.
    pub fn affine(a: Atom, scale: i64, offset: i64) -> Self {
        if scale == 0 {
            return Term::constant(offset);
        }
        let mut coeffs = BTreeMap::new();
        coeffs.insert(a, scale);
        Term {
            coeffs,
            constant: offset,
        }
    }

    /// The whole-term constant, if it has no atoms.
    pub fn as_constant(&self) -> Option<i64> {
        self.coeffs.is_empty().then_some(self.constant)
    }

    /// Read access to the coefficient map (sorted, zero-free).
    pub fn coeffs(&self) -> &BTreeMap<Atom, i64> {
        &self.coeffs
    }

    /// The constant part (may be nonzero even when atoms are present).
    pub fn constant_part(&self) -> i64 {
        self.constant
    }

    /// The stage at which the whole term becomes evaluable: the `max` over its
    /// atoms; a constant term is [`Stage::Jit`]. A predicate over this term can
    /// be decided no earlier than this stage.
    pub fn stage(&self) -> Stage {
        self.coeffs
            .keys()
            .map(Atom::stage)
            .max()
            .unwrap_or(Stage::Jit)
    }

    /// `self + other`, or `None` on `i64` overflow (not canonicalizable — the
    /// caller sinks to a later stage). Follows MLIR `AffineExpr` overflow policy.
    pub fn add(&self, other: &Term) -> Option<Term> {
        let mut coeffs = self.coeffs.clone();
        for (atom, &c) in &other.coeffs {
            let entry = coeffs.entry(*atom).or_insert(0);
            *entry = entry.checked_add(c)?;
            if *entry == 0 {
                coeffs.remove(atom);
            }
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_add(other.constant)?,
        })
    }

    /// `-self`, or `None` on overflow.
    pub fn neg(&self) -> Option<Term> {
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &self.coeffs {
            coeffs.insert(*atom, c.checked_neg()?);
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_neg()?,
        })
    }

    /// `self - other`, or `None` on overflow.
    pub fn sub(&self, other: &Term) -> Option<Term> {
        self.add(&other.neg()?)
    }

    /// `k * self`, or `None` on overflow.
    pub fn mul_const(&self, k: i64) -> Option<Term> {
        if k == 0 {
            return Some(Term::constant(0));
        }
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &self.coeffs {
            coeffs.insert(*atom, c.checked_mul(k)?);
        }
        Some(Term {
            coeffs,
            constant: self.constant.checked_mul(k)?,
        })
    }

    /// A sign-canonical representative: negate iff the leading (least-[`Atom`])
    /// nonzero coefficient — or the constant, when there are no atoms — is
    /// negative. Used to give sign-invariant predicates (`== 0`, `!= 0`,
    /// `% d == 0`) one representative, so e.g. `a == b` and `b == a` unify.
    fn sign_canonical(&self) -> Term {
        let leading_negative = match self.coeffs.iter().next() {
            Some((_, &c)) => c < 0,
            None => self.constant < 0,
        };
        if leading_negative {
            // A canonical term's negation cannot overflow in practice; if it
            // somehow would, keep the original — still sound (equality may miss,
            // sinking a stage), never wrong.
            self.neg().unwrap_or_else(|| self.clone())
        } else {
            self.clone()
        }
    }

    /// If the term is `scale * atom + offset` — exactly one atom — return
    /// `(atom, scale, offset)`. This projects to the single-variable affine
    /// fragment used by loop check-hoisting (the strongest-instance
    /// substitution): a richer term (two IVs, a product) returns `None`, so the
    /// hoister declines it exactly as it declined a non-affine index before.
    pub fn as_single_affine(&self) -> Option<(Atom, i64, i64)> {
        if self.coeffs.len() != 1 {
            return None;
        }
        let (&atom, &scale) = self.coeffs.iter().next().unwrap();
        Some((atom, scale, self.constant))
    }

    /// Evaluate given a resolver from atom to concrete value. `None` if any atom
    /// is unresolved or arithmetic overflows.
    pub fn eval(&self, resolve_atom: &impl Fn(&Atom) -> Option<i64>) -> Option<i64> {
        let mut acc = self.constant;
        for (atom, &c) in &self.coeffs {
            let v = resolve_atom(atom)?;
            acc = acc.checked_add(c.checked_mul(v)?)?;
        }
        Some(acc)
    }
}

/// A relation over canonical [`Term`]s. Constructors put each predicate into a
/// canonical representative, so structurally-equal predicates compare equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// `term == 0`. Equalities `a == b` are stored as `Zero(a - b)`.
    Zero(Term),
    /// `term != 0`. For a non-negative operand this is `term > 0`.
    Nonzero(Term),
    /// `term > 0` (strict). The canonical form of an inequality `a < b`, stored
    /// as `Positive(b - a)`. Unlike `Zero`/`Nonzero`, this is *not*
    /// sign-invariant (`a < b` ≠ `b < a`), so no sign canonicalization is
    /// applied — the term `b - a` is already canonical.
    Positive(Term),
    /// `term % divisor == 0`, with `divisor >= 1`.
    DivisibleBy { term: Term, divisor: i64 },
}

impl Predicate {
    /// `a == b`, canonicalized as `Zero(sign_canonical(a - b))`. `None` on
    /// overflow forming `a - b`.
    pub fn eq(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Zero(a.sub(b)?.sign_canonical()))
    }

    /// `term != 0` (for a non-negative operand, `term > 0`).
    pub fn nonzero(term: Term) -> Predicate {
        Predicate::Nonzero(term.sign_canonical())
    }

    /// `a < b`, canonicalized as `Positive(b - a)` (i.e. `b - a > 0`). `None` on
    /// overflow forming `b - a`.
    pub fn lt(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Positive(b.sub(a)?))
    }

    /// `a <= b` over integers, canonicalized as `a < b + 1`, i.e.
    /// `Positive(b - a + 1)`. `None` on overflow.
    ///
    /// This is the natural form for an access-safety goal: an index drawn
    /// from `[0, ceil(a/t))` stays inside an axis of extent `b` exactly when
    /// `a <= b` bounds it from above (see the checker's linearization). An
    /// *equality* in that position is not the same claim — it also rejects
    /// every `b` strictly larger than `a`, which is safe.
    pub fn le(a: &Term, b: &Term) -> Option<Predicate> {
        Some(Predicate::Positive(b.add(&Term::constant(1))?.sub(a)?))
    }

    /// `term % divisor == 0`. Coefficients and constant are reduced modulo
    /// `divisor` into `[0, divisor)` — valid because `c * a ≡ (c mod d) * a
    /// (mod d)` for every integer `a` — giving a canonical residue term.
    /// `None` if `divisor < 1`.
    pub fn divisible_by(term: Term, divisor: i64) -> Option<Predicate> {
        if divisor < 1 {
            return None;
        }
        let mut coeffs = BTreeMap::new();
        for (atom, &c) in &term.coeffs {
            let r = c.rem_euclid(divisor);
            if r != 0 {
                coeffs.insert(*atom, r);
            }
        }
        let reduced = Term {
            coeffs,
            constant: term.constant.rem_euclid(divisor),
        };
        Some(Predicate::DivisibleBy {
            term: reduced.sign_canonical(),
            divisor,
        })
    }

    /// The stage at which this predicate can be decided (its term's stage).
    pub fn stage(&self) -> Stage {
        match self {
            Predicate::Zero(t) | Predicate::Nonzero(t) | Predicate::Positive(t) => t.stage(),
            Predicate::DivisibleBy { term, .. } => term.stage(),
        }
    }

    /// Evaluate the predicate against an atom resolver (used to fold at JIT and
    /// to check tensor extents at launch). `None` if any atom is unresolved.
    pub fn eval(&self, resolve_atom: &impl Fn(&Atom) -> Option<i64>) -> Option<bool> {
        match self {
            Predicate::Zero(t) => Some(t.eval(resolve_atom)? == 0),
            Predicate::Nonzero(t) => Some(t.eval(resolve_atom)? != 0),
            Predicate::Positive(t) => Some(t.eval(resolve_atom)? > 0),
            Predicate::DivisibleBy { term, divisor } => {
                Some(term.eval(resolve_atom)?.rem_euclid(*divisor) == 0)
            }
        }
    }
}

/// A predicate hoisted out of the device kernel to the host launcher, with a
/// diagnostic `cause` rendered if it fails. Carried on the compiled kernel
/// (`Validator`) and evaluated once per launch by the host's `validate_launch`.
/// This is the same [`Predicate`] the compiler resolved — one representation,
/// evaluated at launch instead of emitted in-kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchCheck {
    pub predicate: Predicate,
    pub cause: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(param: usize, axis: usize) -> Atom {
        Atom::Dim { param, axis }
    }

    // ── Normal form: canonical by construction, confluent ──────────────────

    #[test]
    fn like_terms_combine_and_order_is_irrelevant() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        // (a + b) and (b + a) reach the same normal form.
        assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
        // a + a == 2a.
        assert_eq!(a.add(&a).unwrap(), a.mul_const(2).unwrap());
    }

    #[test]
    fn like_terms_cancel_to_drop_zero_coefficients() {
        let a = Term::atom(dim(0, 0));
        let zero = a.sub(&a).unwrap();
        assert_eq!(zero, Term::constant(0));
        assert!(zero.coeffs().is_empty());
    }

    #[test]
    fn confluence_two_build_orders_agree() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 1));
        // (2a + b) + 3   vs   (b + 3) + 2a
        let lhs = a
            .mul_const(2)
            .unwrap()
            .add(&b)
            .unwrap()
            .add(&Term::constant(3))
            .unwrap();
        let rhs = b
            .add(&Term::constant(3))
            .unwrap()
            .add(&a.mul_const(2).unwrap())
            .unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn affine_consolidates_scale_var_offset() {
        let a = dim(0, 0);
        // 3 * a + 5 built two ways.
        assert_eq!(
            Term::affine(a, 3, 5),
            Term::atom(a)
                .mul_const(3)
                .unwrap()
                .add(&Term::constant(5))
                .unwrap()
        );
        // A zero scale folds to the constant.
        assert_eq!(Term::affine(a, 0, 7), Term::constant(7));
    }

    // ── Stage is the max over atoms ────────────────────────────────────────

    #[test]
    fn as_single_affine_projects_only_single_var_forms() {
        let a = dim(0, 0);
        // 3*a + 5 projects to (a, 3, 5).
        assert_eq!(Term::affine(a, 3, 5).as_single_affine(), Some((a, 3, 5)));
        // A constant (zero atoms) does not project.
        assert_eq!(Term::constant(7).as_single_affine(), None);
        // Two atoms do not project (richer than single-var affine).
        let two = Term::atom(dim(0, 0)).add(&Term::atom(dim(1, 0))).unwrap();
        assert_eq!(two.as_single_affine(), None);
    }

    #[test]
    fn stage_is_max_over_atoms() {
        assert_eq!(Term::constant(4).stage(), Stage::Jit);
        assert_eq!(Term::atom(dim(0, 0)).stage(), Stage::Launch);
        let mixed = Term::atom(dim(0, 0)).add(&Term::atom(Atom::Iv(7))).unwrap();
        assert_eq!(mixed.stage(), Stage::Device);
    }

    // ── Overflow bails (MLIR policy), never wraps ──────────────────────────

    #[test]
    fn overflow_bails_to_none() {
        let big = Term::constant(i64::MAX);
        assert!(big.add(&Term::constant(1)).is_none());
        assert!(Term::atom(dim(0, 0))
            .mul_const(2)
            .unwrap()
            .mul_const(i64::MAX)
            .is_none());
    }

    // ── Predicate canonicalization ─────────────────────────────────────────

    #[test]
    fn equality_is_symmetric_after_canonicalization() {
        let a = Term::atom(dim(0, 0));
        let b = Term::atom(dim(1, 0));
        // a == b and b == a produce the same canonical predicate.
        assert_eq!(
            Predicate::eq(&a, &b).unwrap(),
            Predicate::eq(&b, &a).unwrap()
        );
    }

    #[test]
    fn nonzero_is_sign_invariant() {
        let a = Term::atom(dim(0, 0));
        let neg_a = a.neg().unwrap();
        assert_eq!(Predicate::nonzero(a), Predicate::nonzero(neg_a));
    }

    #[test]
    fn divisibility_reduces_coefficients_mod_divisor() {
        let a = dim(0, 0);
        // 6a + 9  and  0a + 3 (== 3)  are both "divisible by 3": 6≡0, 9≡0 mod 3.
        let p1 = Predicate::divisible_by(Term::affine(a, 6, 9), 3).unwrap();
        // 6 ≡ 0 (mod 3) drops the atom; 9 ≡ 0 (mod 3): reduces to `0 % 3 == 0`.
        let expected = Predicate::divisible_by(Term::constant(0), 3).unwrap();
        assert_eq!(p1, expected);
        assert!(Predicate::divisible_by(Term::constant(0), 0).is_none());
    }

    // ── Evaluation (the launch / fold interpreter) ─────────────────────────

    #[test]
    fn eval_resolves_atoms_and_decides() {
        // extent(param 0, axis 0) = 128.
        let env = |atom: &Atom| match atom {
            Atom::Dim { param: 0, axis: 0 } => Some(128),
            _ => None,
        };
        let nonzero = Predicate::nonzero(Term::atom(dim(0, 0)));
        assert_eq!(nonzero.eval(&env), Some(true));

        let zero_env = |atom: &Atom| match atom {
            Atom::Dim { param: 0, axis: 0 } => Some(0),
            _ => None,
        };
        assert_eq!(nonzero.eval(&zero_env), Some(false));

        // Unresolved atom → cannot decide.
        let missing = Predicate::nonzero(Term::atom(dim(9, 9)));
        assert_eq!(missing.eval(&env), None);
    }

    // ── Special registers + Lt canonical inequality ───────────────────────

    #[test]
    fn special_register_stages() {
        // The tile-block id is a per-CTA runtime value (Device); the grid dim is
        // host-known before launch (Launch).
        assert_eq!(Atom::TileBlockId(0).stage(), Stage::Device);
        assert_eq!(Atom::NumTileBlocks(0).stage(), Stage::Launch);
        assert_eq!(Term::atom(Atom::TileBlockId(0)).stage(), Stage::Device);
        assert_eq!(Term::atom(Atom::NumTileBlocks(0)).stage(), Stage::Launch);
    }

    #[test]
    fn lt_is_the_hardware_axiom_and_directional() {
        let id = Term::atom(Atom::TileBlockId(0));
        let n = Term::atom(Atom::NumTileBlocks(0));
        // The universal axiom `TileBlockId(0) < NumTileBlocks(0)`.
        let axiom = Predicate::lt(&id, &n).unwrap();
        // Directional: `a < b` differs from `b < a` (no sign canonicalization).
        assert_ne!(axiom, Predicate::lt(&n, &id).unwrap());
        // Stage is Device (mixes a Device id with a Launch grid dim), so it can
        // only discharge at Jit via a matching assumption, never hoist.
        assert_eq!(axiom.stage(), Stage::Device);
    }

    #[test]
    fn lt_evaluates_strictly() {
        let env = |atom: &Atom| match atom {
            Atom::TileBlockId(0) => Some(3),
            Atom::NumTileBlocks(0) => Some(4),
            _ => None,
        };
        let lt = Predicate::lt(
            &Term::atom(Atom::TileBlockId(0)),
            &Term::atom(Atom::NumTileBlocks(0)),
        )
        .unwrap();
        assert_eq!(lt.eval(&env), Some(true)); // 3 < 4
        let eq_env = |atom: &Atom| match atom {
            Atom::TileBlockId(0) => Some(4),
            Atom::NumTileBlocks(0) => Some(4),
            _ => None,
        };
        assert_eq!(lt.eval(&eq_env), Some(false)); // 4 < 4 is false (strict)
    }

    #[test]
    fn tile_count_evaluates_ceil_div_in_the_root_frame() {
        let ta = Term::atom(Atom::TileCount {
            param: 0,
            axis: 1,
            tile: 32,
        });
        let tb = Term::atom(Atom::TileCount {
            param: 1,
            axis: 0,
            tile: 32,
        });
        let le = Predicate::le(&ta, &tb).unwrap();
        assert_eq!(le.stage(), Stage::Launch);
        let env = |a_extent: i64, b_extent: i64| {
            move |atom: &Atom| match atom {
                Atom::TileCount {
                    param: 0,
                    axis: 1,
                    tile: 32,
                } => Some((a_extent + 31) / 32),
                Atom::TileCount {
                    param: 1,
                    axis: 0,
                    tile: 32,
                } => Some((b_extent + 31) / 32),
                _ => None,
            }
        };
        // The sub-tile band: shorter in elements, equal in tiles — admitted.
        assert_eq!(le.eval(&env(128, 100)), Some(true));
        // Genuinely fewer tiles — rejected.
        assert_eq!(le.eval(&env(128, 96)), Some(false));
        // Distinct tile sizes are distinct atoms, not the same quantity.
        assert_ne!(
            Term::atom(Atom::TileCount {
                param: 0,
                axis: 1,
                tile: 32
            }),
            Term::atom(Atom::TileCount {
                param: 0,
                axis: 1,
                tile: 16
            })
        );
    }

    #[test]
    fn le_admits_equality_and_larger_and_rejects_smaller() {
        let a = Term::atom(Atom::Dim { param: 0, axis: 1 });
        let b = Term::atom(Atom::Dim { param: 1, axis: 0 });
        let le = Predicate::le(&a, &b).unwrap();
        let env = |x: i64, y: i64| {
            move |atom: &Atom| match atom {
                Atom::Dim { param: 0, axis: 1 } => Some(x),
                Atom::Dim { param: 1, axis: 0 } => Some(y),
                _ => None,
            }
        };
        assert_eq!(le.eval(&env(64, 64)), Some(true)); // equal: safe
        assert_eq!(le.eval(&env(64, 256)), Some(true)); // target larger: safe
        assert_eq!(le.eval(&env(128, 96)), Some(false)); // target short: reject
                                                         // Not the same claim as equality: eq rejects the larger target.
        let eq = Predicate::eq(&a, &b).unwrap();
        assert_eq!(eq.eval(&env(64, 256)), Some(false));
        // Self-comparison folds without an environment: `a <= a` is Positive(1).
        let refl = Predicate::le(&a, &a).unwrap();
        assert_eq!(refl.eval(&|_| None), Some(true));
    }

    #[test]
    fn eval_divisibility() {
        let env = |atom: &Atom| match atom {
            Atom::Dim { param: 0, axis: 0 } => Some(256),
            _ => None,
        };
        let div = Predicate::divisible_by(Term::atom(dim(0, 0)), 64).unwrap();
        assert_eq!(div.eval(&env), Some(true));
        let div2 = Predicate::divisible_by(Term::atom(dim(0, 0)), 100).unwrap();
        assert_eq!(div2.eval(&env), Some(false));
    }
}
