<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Testing Bayesian selection

Surveyed 2026-11-11. The sources below are pinned to the revisions inspected.
Our Rust tests are independently written around selected properties and failure
modes; upstream test code and fixtures were not copied. No Python, ML library,
network access or GPU is needed to execute these tests after building cuTile.
Building still requires the usual cuTile/shared-host prerequisites.

## Sources and selected coverage

| Implementation surveyed | Relevant upstream coverage | Selected cuTile checks |
| --- | --- | --- |
| [scikit-learn GP tests](https://github.com/scikit-learn/scikit-learn/blob/19ca9c861a7adb948a19484337159a748feede54/sklearn/gaussian_process/tests/test_gpr.py) and [kernel tests](https://github.com/scikit-learn/scikit-learn/blob/19ca9c861a7adb948a19484337159a748feede54/sklearn/gaussian_process/tests/test_kernels.py) | Prior and interpolation properties; normalization; equal and nearly equal inputs; constant targets; covariance consistency. | Interpolation and far-field prior recovery; stable predictions after reordering conflicting near-duplicate observations; acquisition invariance to latency units and category renaming/list permutation. Independent sklearn numerical fixtures were already present. |
| [BoTorch analytic acquisition tests](https://github.com/meta-pytorch/botorch/blob/c88de6c6e8bacc8189377342d96b8ff697344a15/test/acquisition/test_analytic.py) | Known EI values, minimization/maximization, posterior scaling, and extreme-value/monotonicity tests for log EI. | Check minimization direction, nonnegative finite EI, increasing value with uncertainty and scale equivariance, including tiny positive variance. Adapted to ordinary EI: this does not establish log-EI tail accuracy or gradient correctness. |
| [Optuna GP tests](https://github.com/optuna/optuna/blob/9e4d3502c85cb58c3e891a48e90ef9afae00acda/tests/samplers_tests/test_gp.py) and [sampler tests](https://github.com/optuna/optuna/blob/9e4d3502c85cb58c3e891a48e90ef9afae00acda/tests/samplers_tests/test_samplers.py) | Numerical stability near convergence; reproducible sequences with an explicit seed. | A cluster of repeated/nearly repeated observations retains finite bounded variance. Constant-latency searches reproduce the entire ask/tell sequence, including pending work, and exhaust without fallback or duplicate proposals. |
| [Helion search tests](https://github.com/pytorch/helion/blob/7b4a1cb7e9876fbffd68eff721aa4c340747431a/test/test_autotuner_search.py), [final selection](https://github.com/pytorch/helion/blob/7b4a1cb7e9876fbffd68eff721aa4c340747431a/test/test_autotuner_final_selection.py), and [rebenchmark tests](https://github.com/pytorch/helion/blob/7b4a1cb7e9876fbffd68eff721aa4c340747431a/test/test_autotuner_rebenchmark.py) | All-failed spaces; bad candidates between good ones; LFBO ranking against a reference; noisy winners; drift/order effects; pinned near ties; missing confirmation evidence. | All-invalid GP search returns no measured winner. Fresh paired confirmation recovers the stable alternative from an optimistic screening winner under common drift, verifies alternating finalist order, and retains the incumbent on near ties. |

The new tests live in [tests/invariants.rs](tests/invariants.rs) and the two
confirmation regressions in [../confirmation.rs](../confirmation.rs).
They use synthetic observations and paired samples, without sleeps or
wall-clock thresholds for correctness. Reproducibility assumes the same
completion order: different asynchronous outcomes supply different information.
The drift test scales both arms of each pair together; it does not simulate
every source of within-pair GPU timing bias.

The scale-equivariance regression exposed a fixed `1e-20` variance cutoff that
zeroed representable EI values. The implementation now uses the deterministic
limit only for zero variance. Ordinary EI can still underflow in extreme tails;
we do not implement BoTorch's log-EI or differentiable acquisition optimizer.

## Coverage retained instead of duplicated

Existing tests cover pending mean conditioning, explicit prior hints, malformed
spaces, typed failures/late measurements, cancellation, worker draining, budget
reservation, an unscreened incumbent, invalid paired samples, and refusing a
partial confirmation winner. Helion's analogous tests support keeping these
contracts; its fallback policy and timing aggregation are not identical to ours.
For example, interrupted confirmation can retain a fully confirmed alternative,
but never a partial leader. If none qualified it keeps the supplied incumbent
with `NotConfirmed`, without asserting that the incumbent was freshly validated.

Forest-specific leaf ranking, hyperparameter optimization, autograd, Monte Carlo
acquisition, multi-output GPs and distributed-rank coordination are outside the
implemented API. Their tests were not transplanted. Held-out GPU comparisons
remain necessary to establish sample efficiency; deterministic numerical and
protocol tests do not demonstrate that one selector is universally faster.

## Review regressions and remaining limits

A structured failing region exposed a selection failure: discarding invalid and
timed-out points left their neighborhoods at high prior variance, attracting
repeated exploration. The selector now applies predicted-mean conditioning to
those points, as it does for pending work. Tests verify that late latency values
cannot change that conditioning, while cancellation and infrastructure errors do
not penalize neighborhoods. A fixed 40-seed replay compares early failing share
against seeded random on 32-point 1-D and 64-point 2-D spaces for both invalid and
timed-out outcomes. This is a regression fixture, not held-out evidence that BO
beats random generally; it uses evaluation-count rather than wall-time budgets.

Negative-tail EI now uses the normal Mills-ratio continued fraction instead of
subtracting nearly equal approximate CDF/PDF terms. Independent 80-digit fixtures
check relative accuracy through z = -20, including either side of the old -8
cutoff. Central CDF values retain the documented approximation; extreme tails can
still underflow, and this is not log-EI.

Pipeline regressions check refreshed active-stage deadlines with 1/2/4 workers,
clamping to the overall deadline, selector interruption, and source-located
frontend rejection without discarding other valid candidates. They assert
ordering/deadline arithmetic rather than elapsed-time cutoffs. Separate tests
exercise simultaneous flight releases and assembler cancellation after startup,
including removal of partial outputs and compiler intermediates.

Follow-up work remains: fitting/scoring stays synchronous with cubic dense-GP
fitting cost, and can dominate short evaluations after hundreds of observations.
Incremental updates or a training-set cap need separate performance validation.
Integer axes still default to linear scaling; callers should select Log2 for
appropriate tile-size knobs.

Consensus regressions cover candidate-only rejection before/after a successful
round, local timeout recovery, retaining a fully confirmed winner when a later
comparison is interrupted, and aborting on infrastructure or incumbent/ambiguous
errors even after an alternative has qualified. An invalidity error must be
explicitly scoped with `Failure::for_candidate` to disqualify only that arm;
marking it cannot relabel infrastructure errors or interruptions. Infrastructure
failure takes precedence over cancellation/timeout in every screening stage and
worker draining, including after earlier successful measurements. Original
errors and interruption diagnostics are both retained. These are synthetic,
ordered tests; they do not depend on real-time speed thresholds.

## Running the focused checks

From `cutile-rs/`, with the normal toolkit/libclang environment:

```sh
cargo test -p cutile --features experimental-tune --lib tune::
cargo clippy -p cutile --lib --tests --features experimental-tune -- -D warnings
cargo fmt --check -p cutile
```

All four reviewed projects are evidence for choosing tests, not runtime or test
dependencies. Reference fixture versions and generation expressions remain next
to the existing numerical tests in [tests.rs](tests.rs).
