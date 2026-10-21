/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Staged obligation resolution: the solver seam for launch-time check hoisting.
//!
//! The predicate vocabulary itself lives in [`cuda_async::predicate`] (a
//! canonical [`Term`]/[`Predicate`] shared with the host launcher). This module
//! adds the *solver* around it:
//! - [`Obligation`] — a predicate to prove, with a diagnostic `cause` (rustc
//!   `Obligation`).
//! - [`Assumptions`] — the set of predicates believed in scope (rustc
//!   `ParamEnv.caller_bounds`).
//! - [`resolve`] — picks the earliest [`Stage`] that discharges an obligation:
//!   [`Resolution::Jit`] if a dominating assumption entails it, else
//!   [`Resolution::Launch`] if the predicate ranges only over launch-known
//!   operands (`predicate.stage() <= Launch`), else [`Resolution::Device`].
//!
//! Because [`Predicate`] is canonical (see `cuda_async::predicate`), assumption
//! entailment is set membership and the launch-known test is `stage()` over the
//! predicate's atoms — no per-variant special-casing.

use std::collections::HashSet;

use cuda_async::predicate::{LaunchCheck, Predicate, Stage};

use crate::passes::proof_analysis::ProofResults;

/// A predicate to prove, carrying a diagnostic `cause` for the eventual launch
/// or in-kernel failure message.
#[derive(Debug, Clone)]
pub(crate) struct Obligation {
    pub(crate) predicate: Predicate,
    pub(crate) cause: String,
}

impl Obligation {
    pub(crate) fn new(predicate: Predicate, cause: impl Into<String>) -> Self {
        Self {
            predicate,
            cause: cause.into(),
        }
    }
}

/// Where an obligation was discharged. [`Resolution::Jit`] costs zero registers
/// and zero host work; [`Resolution::Launch`] costs one host compare per launch;
/// [`Resolution::Device`] is an in-kernel assert (the existing path).
#[derive(Debug, Clone)]
pub(crate) enum Resolution {
    Jit,
    #[allow(dead_code)] // accepted once a client collects launch checks (RMSNorm wiring).
    Launch(LaunchCheck),
    Device,
}

/// The set of predicates believed in scope (rustc `ParamEnv.caller_bounds`).
/// Membership is exact because [`Predicate`] is canonical.
#[derive(Debug, Clone, Default)]
pub(crate) struct Assumptions {
    believed: HashSet<Predicate>,
}

impl Assumptions {
    /// Seed from declared `preconditions`: `dim(a, i) == dim(b, j)` becomes an
    /// entry-scoped `Predicate::eq` over the two axis extents, and
    /// `dim(t, k) % d == 0` becomes `Predicate::divisible_by` over one extent,
    /// resolving tensor names to param indices via `param_index`. Each fact's
    /// truth is verified by the generated launcher before the kernel runs — a
    /// violating launch is rejected — which is what makes believing it here
    /// sound. A fact naming a non-parameter (or that fails to canonicalize) is
    /// skipped — conservative.
    pub(crate) fn from_preconditions(
        proof: &ProofResults,
        param_index: &std::collections::HashMap<String, usize>,
    ) -> Self {
        use crate::passes::proof_analysis::MetadataExpr;
        use crate::passes::proof_analysis::MetadataFact;
        use cuda_async::predicate::{Atom, Term};
        let mut believed = HashSet::new();
        for fact in &proof.metadata_facts {
            match fact {
                MetadataFact::DimEq {
                    lhs:
                        MetadataExpr::Dim {
                            tensor: lt,
                            axis: la,
                        },
                    rhs:
                        MetadataExpr::Dim {
                            tensor: rt,
                            axis: ra,
                        },
                } => {
                    let (Some(&lp), Some(&rp)) = (param_index.get(lt), param_index.get(rt)) else {
                        continue;
                    };
                    let lhs = Term::atom(Atom::Dim {
                        param: lp,
                        axis: *la,
                    });
                    let rhs = Term::atom(Atom::Dim {
                        param: rp,
                        axis: *ra,
                    });
                    if let Some(pred) = Predicate::eq(&lhs, &rhs) {
                        believed.insert(pred);
                    }
                }
                MetadataFact::DimDivisible {
                    tensor,
                    axis,
                    divisor,
                } => {
                    let Some(&param) = param_index.get(tensor) else {
                        continue;
                    };
                    let term = Term::atom(Atom::Dim { param, axis: *axis });
                    if let Some(pred) = Predicate::divisible_by(term, *divisor) {
                        believed.insert(pred);
                    }
                }
            }
        }
        Self { believed }
    }

    // NOTE: the assumption set deliberately holds NOTHING besides declared,
    // launch-verified `preconditions`. It used to be seeded with the hardware
    // register axioms (`TileBlockId(k) < NumTileBlocks(k)`): universally true,
    // but their only consumer was a rung that formed the axiom as its own goal
    // and discharged it against itself — a tautology that admitted
    // out-of-bounds stores (reverted). Live-looking machinery with no sound
    // consumer is how that bug shipped, so the axioms are gone until a rung
    // exists that (a) forms the real proposition `TileBlockId(k) <
    // num_partitions(view, k)` and (b) obtains `NumTileBlocks(k) ==
    // num_partitions(view, k)` as a *verified* Launch obligation. Every set
    // member being launch-verified is also what keeps `entails` sound as plain
    // membership without a dominance test.

    /// Does an assumption in scope entail `predicate`? Exact set membership,
    /// since predicates are canonical.
    ///
    /// **There is no dominance check here.** The design (`staged dominance`)
    /// requires an assumption to dominate the obligation it discharges; this is
    /// vacuously satisfied today because the only assumption source is
    /// kernel-scoped and unconditional — declared entry `preconditions` hold at
    /// every program point. The moment a
    /// *flow-sensitive* assumption is minted (a loop-body fact, an `if`-arm
    /// fact, a point `require`), this must become a real dominance test or it
    /// will discharge obligations from facts that do not reach them.
    fn entails(&self, predicate: &Predicate) -> bool {
        self.believed.contains(predicate)
    }
}

/// The solver: pick the earliest [`Stage`] that discharges `obligation` given
/// the dominating `assumptions`.
///
/// - [`Resolution::Jit`] — a dominating assumption entails the predicate.
/// - [`Resolution::Launch`] — the predicate ranges only over launch-known
///   operands (`stage() <= Launch`), so the host decides it once per launch.
/// - [`Resolution::Device`] — otherwise; fall to the existing in-kernel check.
///
/// A client that has not opted into launch-check collection simply treats any
/// non-`Jit` resolution as "not discharged here" and keeps its existing
/// (device) path — so introducing the `Launch` tier is behavior-preserving
/// until a client collects it.
pub(crate) fn resolve(obligation: &Obligation, assumptions: &Assumptions) -> Resolution {
    // A predicate over no atoms is already decided: its term is a constant, so
    // evaluate it rather than shipping a host compare of two known numbers.
    // `dim(x, 1) == dim(x, 1)` canonicalizes to `Zero(0)` and lands here — a
    // goal worth recognising, since asking whether an axis matches *itself* is
    // the degenerate case of every cross-tensor query. A constant-false
    // predicate falls through to the stage rungs, which will place a check
    // that fails: conservative, and it keeps the reporting of a real violation
    // in one place.
    if obligation.predicate.eval(&|_| None) == Some(true) {
        return Resolution::Jit;
    }
    if assumptions.entails(&obligation.predicate) {
        Resolution::Jit
    } else if obligation.predicate.stage() <= Stage::Launch {
        Resolution::Launch(LaunchCheck {
            predicate: obligation.predicate.clone(),
            cause: obligation.cause.clone(),
        })
    } else {
        Resolution::Device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_async::predicate::{Atom, Term};

    fn dim_term(param: usize, axis: usize) -> Term {
        Term::atom(Atom::Dim { param, axis })
    }

    fn assumptions(preds: Vec<Predicate>) -> Assumptions {
        Assumptions {
            believed: preds.into_iter().collect(),
        }
    }

    #[test]
    fn dim_eq_entailed_resolves_at_jit() {
        let env = assumptions(vec![
            Predicate::eq(&dim_term(0, 0), &dim_term(1, 0)).unwrap()
        ]);
        // Symmetric: the goal states the equality the other way round.
        let obl = Obligation::new(
            Predicate::eq(&dim_term(1, 0), &dim_term(0, 0)).unwrap(),
            "test",
        );
        assert!(matches!(resolve(&obl, &env), Resolution::Jit));
    }

    #[test]
    fn dim_eq_unentailed_but_launch_known_resolves_at_launch() {
        // Both operands are tensor extents (Launch-known), so an unproven
        // equality is a launch check rather than a device assert.
        let env = assumptions(vec![]);
        let obl = Obligation::new(
            Predicate::eq(&dim_term(0, 0), &dim_term(1, 1)).unwrap(),
            "test",
        );
        assert!(matches!(resolve(&obl, &env), Resolution::Launch(_)));
    }

    #[test]
    fn device_stage_predicate_falls_to_device() {
        // An induction variable is Device-stage, so a predicate over it cannot
        // be hoisted.
        let env = assumptions(vec![]);
        let obl = Obligation::new(Predicate::nonzero(Term::atom(Atom::Iv(3))), "iv");
        assert!(matches!(resolve(&obl, &env), Resolution::Device));
    }

    #[test]
    fn dim_nonzero_resolves_at_launch() {
        let env = assumptions(vec![]);
        let obl = Obligation::new(Predicate::nonzero(dim_term(0, 0)), "extent > 0");
        assert!(matches!(resolve(&obl, &env), Resolution::Launch(_)));
    }

    #[test]
    fn no_axiom_is_believed_without_a_declared_precondition() {
        // The assumption set holds nothing but declared, launch-verified facts.
        // In particular the hardware register axioms are NOT seeded: their only
        // consumer was a rung that formed the axiom as its own goal and
        // discharged it against itself — a tautology that admitted
        // out-of-bounds stores (reverted). A block-id bound is Device-stage, so
        // with nothing believed it falls to Device — fail closed.
        let env = Assumptions::from_preconditions(
            &ProofResults::default(),
            &std::collections::HashMap::new(),
        );
        let goal = Predicate::lt(
            &Term::atom(Atom::TileBlockId(0)),
            &Term::atom(Atom::NumTileBlocks(0)),
        )
        .unwrap();
        let obl = Obligation::new(goal, "block id in grid");
        assert!(matches!(resolve(&obl, &env), Resolution::Device));
    }
}
