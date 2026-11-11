/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cheap screening followed by fresh paired comparisons with a fixed incumbent.

use super::*;
use crate::bench::{BenchOptions, Measurement};

/// Confirmation policy. Defaults are starting points to calibrate for a workload,
/// not statistical guarantees. No external statistics or ML dependency is used.
#[derive(Debug, Clone)]
pub struct ConfirmationOptions {
    pub incumbent: usize,
    /// Number of screened alternatives to compare with the incumbent.
    pub finalists: usize,
    /// Fraction of the total wall-clock budget reserved for fresh comparisons.
    pub reserved_fraction: f64,
    pub rounds: usize,
    /// Practical improvement floor, e.g. 0.02 for 2%. The lower quartile of paired
    /// improvements must exceed this in every round. This is a repeatability
    /// rule, not a confidence interval or a universal noise threshold.
    pub minimum_improvement: f64,
    pub screening: BenchOptions,
    pub confirmation: BenchOptions,
}

impl ConfirmationOptions {
    pub fn new(incumbent: usize) -> Self {
        Self {
            incumbent,
            finalists: 4,
            reserved_fraction: 0.25,
            rounds: 3,
            minimum_improvement: 0.02,
            screening: BenchOptions {
                warmup: Duration::from_millis(5),
                rep: Duration::from_millis(10),
                min_reps: 5,
                max_reps: 50,
                clear_l2: true,
            },
            confirmation: BenchOptions {
                min_reps: 20,
                max_reps: 200,
                ..Default::default()
            },
        }
    }

    fn validate(&self, count: usize) -> Result<(), Failure> {
        if self.incumbent >= count || self.finalists == 0 || self.rounds == 0 {
            return Err(Failure::infrastructure(
                "invalid incumbent, finalist count or round count",
            ));
        }
        if !self.reserved_fraction.is_finite()
            || !(0.0..1.0).contains(&self.reserved_fraction)
            || self.reserved_fraction == 0.0
            || !self.minimum_improvement.is_finite()
            || !(0.0..1.0).contains(&self.minimum_improvement)
        {
            return Err(Failure::infrastructure(
                "invalid confirmation budget or improvement fraction",
            ));
        }
        for opts in [&self.screening, &self.confirmation] {
            if opts.min_reps == 0 || opts.max_reps < opts.min_reps {
                return Err(Failure::infrastructure(
                    "invalid benchmark repetition bounds",
                ));
            }
        }
        if self.confirmation.min_reps < 3 {
            return Err(Failure::infrastructure(
                "confirmation needs at least three pairs per round",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedEvidence {
    pub incumbent_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
    pub median_improvement: f64,
    pub lower_quartile_improvement: f64,
    pub passes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmationRound {
    pub candidate_id: String,
    pub round: usize,
    pub elapsed: Duration,
    /// Raw samples remain evidence even if the callback overran its deadline.
    pub evidence: Option<PairedEvidence>,
    pub failure: Option<Failure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmationStatus {
    Improved,
    NoImprovement,
    /// No replacement is recommended; selected remains the supplied incumbent.
    /// This does not establish that the incumbent passed validation this run.
    NotConfirmed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfirmedOutput {
    pub search: PipelineOutput,
    pub incumbent: Config,
    pub selected: Config,
    pub confirmations: Vec<ConfirmationRound>,
    pub status: ConfirmationStatus,
    pub failure: Option<Failure>,
    pub elapsed: Duration,
}

fn evidence(
    a: Measurement,
    b: Measurement,
    options: &ConfirmationOptions,
) -> Result<PairedEvidence, Failure> {
    if a.reps() != b.reps()
        || a.reps() < options.confirmation.min_reps
        || a.times_ms()
            .iter()
            .chain(b.times_ms())
            .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(Failure::infrastructure(
            "invalid or insufficient paired measurements",
        ));
    }
    let mut improvements: Vec<_> = a
        .times_ms()
        .iter()
        .zip(b.times_ms())
        .map(|(&a, &b)| 1.0 - f64::from(b) / f64::from(a))
        .collect();
    improvements.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        let position = q * (improvements.len() - 1) as f64;
        let low = position.floor() as usize;
        let high = position.ceil() as usize;
        improvements[low] + (improvements[high] - improvements[low]) * position.fract()
    };
    let lower = quantile(0.25);
    Ok(PairedEvidence {
        incumbent_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
        median_improvement: quantile(0.5),
        lower_quartile_improvement: lower,
        passes: lower > options.minimum_improvement,
    })
}

impl Pipeline {
    /// Screen with the normal preparation pipeline, drain all its workers, then
    /// compare shortlisted configurations against the supplied incumbent.
    /// Requires a total `budget`; reserves part of it rather than extending it.
    ///
    /// `screen` must check correctness before timing. `compare` must freshly
    /// check both configurations and return aligned paired measurements, e.g.
    /// with `bench::do_bench_paired_controlled`. It must not launch background
    /// preparation. External GPU work is outside this scheduler's control.
    ///
    /// All shortlisted comparisons must finish successfully within the budget.
    /// A replacement must pass the repeatability rule in every fresh round;
    /// otherwise the incumbent is retained. Confirmation errors and overruns
    /// never promote a partial leader. Results include raw paired samples.
    #[allow(clippy::too_many_arguments)]
    pub fn run_confirmed<J: Send, P: Send>(
        mut self,
        configs: &[Config],
        strategy: &mut impl Strategy,
        options: &ConfirmationOptions,
        resolve: impl FnMut(&Config) -> Result<J, Failure>,
        prepare: impl Fn(J, &WorkControl) -> Result<P, Failure> + Sync,
        mut screen: impl FnMut(&Config, P, &BenchOptions, &WorkControl) -> Result<Trial, Failure>,
        mut compare: impl FnMut(
            &Config,
            &Config,
            &BenchOptions,
            &WorkControl,
        ) -> Result<(Measurement, Measurement), Failure>,
    ) -> Result<ConfirmedOutput, Failure> {
        options.validate(configs.len())?;
        let start = Instant::now();
        let total = self
            .budget
            .ok_or_else(|| Failure::infrastructure("confirmed tuning requires a total budget"))?;
        let deadline = start
            .checked_add(total)
            .ok_or_else(|| Failure::infrastructure("budget is too large"))?;
        let overall = self.control.with_deadline(deadline);
        let available = overall.deadline().unwrap().saturating_duration_since(start);
        let screen_duration = available.mul_f64(1.0 - options.reserved_fraction);
        self.control = overall.with_deadline(start + screen_duration);
        self.budget = Some(screen_duration);
        let trial_limit = self.trial_limit;
        let search = self.run(configs, strategy, resolve, prepare, |c, p, control| {
            screen(c, p, &options.screening, control)
        })?;
        // run() has drained its workers and dropped all queued artifacts here.
        let mut output = ConfirmedOutput {
            search,
            incumbent: configs[options.incumbent].clone(),
            selected: configs[options.incumbent].clone(),
            confirmations: Vec::new(),
            status: ConfirmationStatus::NotConfirmed,
            failure: None,
            elapsed: start.elapsed(),
        };
        if output.search.status == RunStatus::Incomplete {
            output.failure = Some(Failure::infrastructure(
                output
                    .search
                    .error
                    .clone()
                    .unwrap_or_else(|| "incomplete screening".into()),
            ));
            return Ok(output);
        }
        if let Err(error) = overall.check() {
            output.failure = Some(error.into());
            return Ok(output);
        }
        if let Some(failure) = output
            .search
            .observations
            .iter()
            .find(|o| o.trial_id == options.incumbent)
            .and_then(|o| o.failure.as_ref())
            .filter(|f| f.kind == FailureKind::Invalid)
        {
            output.failure = Some(failure.clone());
            return Ok(output);
        }
        let mut finalists: Vec<_> = output
            .search
            .observations
            .iter()
            .filter(|o| o.trial_id != options.incumbent)
            .filter_map(|o| o.score_ms().map(|score| (o.trial_id, score)))
            .collect();
        finalists.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        finalists.truncate(options.finalists);
        if finalists.is_empty() {
            return Ok(output);
        }
        let mut eligible = vec![true; finalists.len()];
        let mut worst_median = vec![f64::INFINITY; finalists.len()];
        for round in 0..options.rounds {
            let order: Vec<_> = if round % 2 == 0 {
                (0..finalists.len()).collect()
            } else {
                (0..finalists.len()).rev().collect()
            };
            for position in order {
                let candidate = &configs[finalists[position].0];
                let begun = Instant::now();
                let control = match trial_limit {
                    Some(limit) => overall.with_deadline(
                        begun
                            .checked_add(limit)
                            .ok_or_else(|| Failure::infrastructure("trial limit is too large"))?,
                    ),
                    None => overall.clone(),
                };
                let result = control
                    .check()
                    .map_err(Failure::from)
                    .and_then(|()| {
                        compare(
                            &output.incumbent,
                            candidate,
                            &options.confirmation,
                            &control,
                        )
                    })
                    .and_then(|(a, b)| evidence(a, b, options));
                let (evidence, mut failure) = match result {
                    Ok(evidence) => (Some(evidence), None),
                    Err(failure) => (None, Some(failure)),
                };
                if let Err(error) = control.check() {
                    failure = Some(Failure::interrupted(error, failure));
                }
                if let Some(evidence) = &evidence {
                    eligible[position] &= evidence.passes;
                    worst_median[position] =
                        worst_median[position].min(evidence.median_improvement);
                }
                output.confirmations.push(ConfirmationRound {
                    candidate_id: candidate.id.clone(),
                    round,
                    elapsed: begun.elapsed(),
                    evidence,
                    failure: failure.clone(),
                });
                if let Some(failure) = failure {
                    output.failure = Some(failure);
                    output.elapsed = start.elapsed();
                    return Ok(output);
                }
            }
        }
        if let Err(error) = overall.check() {
            output.failure = Some(error.into());
            output.elapsed = start.elapsed();
            return Ok(output);
        }
        output.status = ConfirmationStatus::NoImprovement;
        if let Some(position) = (0..finalists.len())
            .filter(|&i| eligible[i])
            .max_by(|&a, &b| worst_median[a].total_cmp(&worst_median[b]))
        {
            output.selected = configs[finalists[position].0].clone();
            output.status = ConfirmationStatus::Improved;
        }
        output.elapsed = start.elapsed();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::ParamValue;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn configs() -> Vec<Config> {
        (0..3)
            .map(|i| Config::new([("x", ParamValue::Int(i))]))
            .collect()
    }
    fn options() -> ConfirmationOptions {
        let mut o = ConfirmationOptions::new(0);
        o.rounds = 2;
        o.confirmation.min_reps = 4;
        o
    }
    fn samples(a: &[f32], b: &[f32]) -> (Measurement, Measurement) {
        (
            Measurement::from_times_ms(a.to_vec()),
            Measurement::from_times_ms(b.to_vec()),
        )
    }
    fn score(c: &Config) -> Trial {
        Trial::measured(&c.id, 3.0 - c.int("x").unwrap() as f32, 1.0, 5)
    }

    #[test]
    fn interrupted_confirmation_preserves_its_callback_failure() {
        for original in [
            Failure::invalid("pair failed correctness"),
            Failure::infrastructure("GPU disconnected"),
        ] {
            let cs = configs();
            let result = Pipeline::default()
                .budget(Duration::from_secs(30))
                .run_confirmed(
                    &cs,
                    &mut super::super::GridOrder,
                    &options(),
                    |_| Ok(()),
                    |_, _| Ok(()),
                    |c, _, _, _| Ok(score(c)),
                    |_, _, _, control| {
                        control.cancel();
                        Err(original.clone())
                    },
                )
                .unwrap();
            assert_eq!(result.status, ConfirmationStatus::NotConfirmed);
            assert_eq!(result.selected.id, cs[0].id);
            let failure = result.failure.as_ref().unwrap();
            assert_eq!(failure.kind, FailureKind::Cancelled);
            assert!(failure.reason.contains("work cancelled"));
            assert!(failure.reason.contains(&original.to_string()));
            assert_eq!(
                result.confirmations[0].failure.as_ref().unwrap().reason,
                failure.reason
            );
        }
    }

    #[test]
    fn confirmation_has_reserved_time_runs_without_workers_and_replaces_incumbent() {
        let live = AtomicUsize::new(0);
        struct Artifact<'a>(&'a AtomicUsize);
        impl Drop for Artifact<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let search_deadline = std::cell::Cell::new(None);
        let cs = configs();
        let result = Pipeline::default()
            .workers(2)
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(42),
                &options(),
                |_| Ok(()),
                |_, _| {
                    live.fetch_add(1, Ordering::SeqCst);
                    Ok(Artifact(&live))
                },
                |c, _, opts, control| {
                    assert_eq!(opts.max_reps, 50);
                    search_deadline.set(control.deadline());
                    Ok(score(c))
                },
                |incumbent, candidate, _, control| {
                    assert_eq!(live.load(Ordering::SeqCst), 0);
                    assert_eq!(incumbent.id, cs[0].id);
                    assert!(control.deadline() > search_deadline.get());
                    let latency = if candidate.int("x") == Some(2) {
                        0.7
                    } else {
                        0.9
                    };
                    Ok(samples(&[1.0; 4], &[latency; 4]))
                },
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::Improved);
        assert_eq!(result.selected.id, cs[2].id);
        assert_eq!(result.confirmations.len(), 4);
        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: ConfirmedOutput = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.confirmations[0]
                .evidence
                .as_ref()
                .unwrap()
                .candidate_ms
                .len(),
            4
        );
    }

    #[test]
    fn a_noisy_or_inconsistent_screening_winner_does_not_replace_incumbent() {
        let cs = configs();
        let mut opts = options();
        opts.finalists = 1;
        let mut comparisons = 0;
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &opts,
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| Ok(score(c)),
                |_, _, _, _| {
                    comparisons += 1;
                    Ok(if comparisons == 1 {
                        samples(&[1.0; 4], &[0.8; 4])
                    } else {
                        samples(&[1.0; 4], &[0.5, 0.9, 1.1, 1.2])
                    })
                },
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NoImprovement);
        assert_eq!(result.selected.id, cs[0].id);
    }

    #[test]
    fn fresh_paired_rounds_recover_the_stable_finalist_despite_drift() {
        // Inspired by Helion's final-selection and rebenchmark regressions;
        // see bayesian/TESTING.md. Our rule uses paired ratios, not its median
        // of absolute round timings. No sleep or real-time correctness check.
        let cs = configs();
        let mut opts = options();
        opts.rounds = 3;
        let mut order = Vec::new();
        let mut strategy = BayesianSearch::new(&cs, Default::default()).unwrap();
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut strategy,
                &opts,
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| {
                    let time = match c.int("x").unwrap() {
                        1 => 0.1,
                        2 => 0.5,
                        _ => 1.0,
                    };
                    Ok(Trial::measured(&c.id, time, time, 20))
                },
                |_, candidate, _, _| {
                    let index = candidate.int("x").unwrap();
                    order.push(index);
                    // Common clock drift between comparisons must not change
                    // the winner. One lucky sample must not rescue the loser.
                    let scale = 1.0 + order.len() as f32 * 0.1;
                    let a = [scale; 4];
                    let b = if index == 1 {
                        [0.1, 1.1, 1.1, 1.1]
                    } else {
                        [0.8; 4]
                    }
                    .map(|v| v * scale);
                    Ok(samples(&a, &b))
                },
            )
            .unwrap();
        assert_eq!(order, [1, 2, 2, 1, 1, 2]);
        assert_eq!(result.search.best_observed.unwrap().id, cs[1].id);
        assert_eq!(result.status, ConfirmationStatus::Improved);
        assert_eq!(result.selected.id, cs[2].id);
    }

    #[test]
    fn near_ties_keep_the_incumbent_despite_optimistic_screening() {
        // Helion tests retaining a pinned seed within tolerance. Here the
        // explicit incumbent remains unless the paired improvement clears 2%.
        let cs = configs();
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| Ok(score(c)),
                |_, _, _, _| Ok(samples(&[1.0; 4], &[0.995; 4])),
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NoImprovement);
        assert_eq!(result.selected.id, cs[0].id);
        assert_eq!(result.confirmations.len(), 4);
        assert!(result
            .confirmations
            .iter()
            .all(|r| !r.evidence.as_ref().unwrap().passes));
    }

    #[test]
    fn a_partial_confirmation_leader_never_wins_after_cancellation() {
        let cs = configs();
        let mut calls = 0;
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| Ok(score(c)),
                |_, _, _, control| {
                    calls += 1;
                    if calls == 2 {
                        control.cancel();
                    }
                    Ok(samples(&[1.0; 4], &[0.8; 4]))
                },
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NotConfirmed);
        assert_eq!(
            result.failure.as_ref().unwrap().kind,
            FailureKind::Cancelled
        );
        assert_eq!(result.selected.id, cs[0].id);
        assert!(result.confirmations.last().unwrap().evidence.is_some());
    }

    #[test]
    fn zero_total_budget_never_dispatches_or_confirms() {
        let cs = configs();
        let result = Pipeline::default()
            .budget(Duration::ZERO)
            .run_confirmed::<(), ()>(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| unreachable!(),
                |_, _| unreachable!(),
                |_, _, _, _| unreachable!(),
                |_, _, _, _| unreachable!(),
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NotConfirmed);
        assert_eq!(result.failure.as_ref().unwrap().kind, FailureKind::TimedOut);
        assert_eq!(result.selected.id, cs[0].id);
    }

    #[test]
    fn confirmation_failure_keeps_incumbent_and_preserves_reason() {
        let cs = configs();
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| Ok(score(c)),
                |_, _, _, _| Err(Failure::invalid("fresh numerical check failed")),
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NotConfirmed);
        assert_eq!(result.selected.id, cs[0].id);
        assert_eq!(result.failure.as_ref().unwrap().kind, FailureKind::Invalid);
    }

    #[test]
    fn unequal_invalid_or_too_few_pairs_are_not_evidence() {
        let opts = options();
        for (a, b) in [
            samples(&[1.0; 4], &[0.8; 3]),
            samples(&[1.0; 2], &[0.8; 2]),
            samples(&[1.0; 4], &[0.8, 0.8, f32::NAN, 0.8]),
            samples(&[0.0; 4], &[0.8; 4]),
        ] {
            assert_eq!(
                evidence(a, b, &opts).unwrap_err().kind,
                FailureKind::Infrastructure
            );
        }
        let (a, b) = samples(&[1.0; 4], &[0.1, 1.0, 1.0, 1.0]);
        assert!(!evidence(a, b, &opts).unwrap().passes);
    }

    #[test]
    fn known_invalid_incumbent_is_not_confirmed() {
        let cs = configs();
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| {
                    if c.id == cs[0].id {
                        Err(Failure::invalid("bad baseline"))
                    } else {
                        Ok(score(c))
                    }
                },
                |_, _, _, _| unreachable!("must not compare against a known invalid baseline"),
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::NotConfirmed);
        assert_eq!(result.failure.unwrap().kind, FailureKind::Invalid);
        assert!(result.confirmations.is_empty());
    }

    #[test]
    fn an_unscreened_incumbent_still_gets_fresh_comparisons() {
        struct OneCandidate(bool);
        impl Strategy for OneCandidate {
            fn suggest(&mut self, _: &SearchView<'_>) -> Option<usize> {
                if std::mem::replace(&mut self.0, true) {
                    None
                } else {
                    Some(1)
                }
            }
        }
        let cs = configs();
        let mut calls = 0;
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .run_confirmed(
                &cs,
                &mut OneCandidate(false),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, _| Ok(score(c)),
                |incumbent, candidate, _, _| {
                    assert_eq!(incumbent.id, cs[0].id);
                    assert_eq!(candidate.id, cs[1].id);
                    calls += 1;
                    Ok(samples(&[1.0; 4], &[0.8; 4]))
                },
            )
            .unwrap();
        assert_eq!(calls, options().rounds);
        assert!(result.search.unvisited.contains(&0));
        assert_eq!(result.status, ConfirmationStatus::Improved);
        assert_eq!(result.selected.id, cs[1].id);
    }

    #[test]
    fn external_deadline_caps_both_phases() {
        let cs = configs();
        let deadline = Instant::now() + Duration::from_secs(10);
        let control = WorkControl::default().with_deadline(deadline);
        let result = Pipeline::default()
            .budget(Duration::from_secs(60))
            .control(control)
            .run_confirmed(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _, control| {
                    assert!(control.deadline().unwrap() < deadline);
                    Ok(score(c))
                },
                |_, _, _, control| {
                    assert_eq!(control.deadline(), Some(deadline));
                    Ok(samples(&[1.0; 4], &[0.8; 4]))
                },
            )
            .unwrap();
        assert_eq!(result.status, ConfirmationStatus::Improved);
    }

    #[test]
    fn policy_rejects_invalid_options_and_requires_a_budget() {
        let cs = configs();
        let error = Pipeline::default()
            .run_confirmed::<(), ()>(
                &cs,
                &mut RandomOrder::new(0),
                &options(),
                |_| unreachable!(),
                |_, _| unreachable!(),
                |_, _, _, _| unreachable!(),
                |_, _, _, _| unreachable!(),
            )
            .unwrap_err();
        assert!(error.reason.contains("total budget"));
        for fraction in [0.0, 1.0, -1.0, f64::NAN] {
            let mut opts = options();
            opts.reserved_fraction = fraction;
            assert!(opts.validate(cs.len()).is_err());
        }
        let mut opts = options();
        opts.confirmation.min_reps = 2;
        assert!(opts.validate(cs.len()).is_err());
        opts = options();
        opts.incumbent = cs.len();
        assert!(opts.validate(cs.len()).is_err());
    }
}
