/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded CPU preparation with serialized foreground evaluation.
//!
//! This opt-in scheduler is separate from the existing synchronous `Objective`
//! API and resume-log schema. `run` returns provisional observations;
//! `run_confirmed` reserves time for fresh paired comparisons with an incumbent
//! after all preparation workers stop. GPU work belongs on the foreground lane.

use super::{Config, Trial, TrialState};
use crate::error::Error;
pub use cutile_compiler::compile_control::{CompileControl as WorkControl, CompileInterrupted};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

mod bayesian;
mod confirmation;
pub use bayesian::{BayesianOptions, BayesianSearch, ParameterScale};
pub use confirmation::{
    ConfirmationOptions, ConfirmationRound, ConfirmationStatus, ConfirmedOutput, PairedEvidence,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureKind {
    Invalid,
    TimedOut,
    Cancelled,
    Infrastructure,
}

/// Explicit validity checks and source-located frontend rejections can classify
/// a configuration as invalid. Unclassified compiler errors are infrastructure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub kind: FailureKind,
    /// Diagnostic text, including an earlier callback failure when interrupted.
    pub reason: String,
}

impl Failure {
    /// Interruption controls scoring, but must not erase a callback's diagnosis.
    /// Rechecking the same interruption keeps its existing diagnostic intact.
    fn interrupted(interruption: CompileInterrupted, earlier: Option<Self>) -> Self {
        let mut failure = Self::from(interruption);
        if let Some(earlier) = earlier {
            if earlier.kind == failure.kind {
                return earlier;
            }
            failure.reason = format!("{}; earlier failure: {earlier}", failure.reason);
        }
        failure
    }

    pub fn invalid(reason: impl Into<String>) -> Self {
        Self {
            kind: FailureKind::Invalid,
            reason: reason.into(),
        }
    }
    pub fn infrastructure(reason: impl Into<String>) -> Self {
        Self {
            kind: FailureKind::Infrastructure,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.reason)
    }
}

impl std::error::Error for Failure {}

impl From<CompileInterrupted> for Failure {
    fn from(value: CompileInterrupted) -> Self {
        Self {
            kind: match value {
                CompileInterrupted::Cancelled => FailureKind::Cancelled,
                CompileInterrupted::DeadlineExceeded => FailureKind::TimedOut,
            },
            reason: value.to_string(),
        }
    }
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        // Frontend/source-located rejections belong to this specialization.
        // Unclassified assembler, I/O and driver errors remain infrastructure.
        if matches!(
            &error,
            Error::JIT(cutile_compiler::error::JITError::Located(..))
        ) {
            return Self::invalid(error.to_string());
        }
        let inner = match &error {
            Error::Anyhow(error) => Some(error),
            Error::JIT(cutile_compiler::error::JITError::Anyhow(error)) => Some(error),
            _ => None,
        };
        if let Some(interrupted) = inner.and_then(|e| e.downcast_ref::<CompileInterrupted>()) {
            let mut failure = Self::from(*interrupted);
            failure.reason = error.to_string();
            return failure;
        }
        // Compiler errors can mean a missing executable or an I/O failure.
        // Do not silently teach a selector that those candidates are invalid.
        Self::infrastructure(error.to_string())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTimes {
    /// Callback duration; None if resolution was never invoked.
    pub resolve: Option<Duration>,
    /// Time waiting for a CPU worker; None if preparation was never queued.
    pub queue: Option<Duration>,
    /// CPU callback duration; None if preparation was never invoked.
    pub prepare: Option<Duration>,
    /// Includes loading, user checks, warmup and timing. None means not run.
    pub evaluate: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Candidate index. This first pipeline evaluates each candidate at most once.
    pub trial_id: usize,
    pub config_id: String,
    pub stages: StageTimes,
    /// Late measurements remain evidence, but `failure` excludes them from
    /// scoring. A prepared artifact alone never produces a measured trial.
    pub trial: Option<Trial>,
    pub failure: Option<Failure>,
}

impl Observation {
    pub fn score_ms(&self) -> Option<f32> {
        self.failure
            .is_none()
            .then(|| self.trial.as_ref()?.median_ms())
            .flatten()
    }
}

pub struct SearchView<'a> {
    pub configs: &'a [Config],
    pub completed: &'a [Observation],
    /// Includes preparation, ready artifacts, and the current GPU evaluation.
    pub pending: &'a [usize],
    pub remaining: Option<Duration>,
}

/// Ask/tell selection. The scheduler owns concurrency and pending state.
pub trait Strategy {
    /// Validate configuration-dependent state before starting workers.
    fn validate(&self, _configs: &[Config]) -> Result<(), Failure> {
        Ok(())
    }
    /// None means no suggestion now. Called again after pending work completes.
    fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize>;
    fn observe(&mut self, _observation: &Observation) {}
}

impl<T: Strategy + ?Sized> Strategy for Box<T> {
    fn validate(&self, configs: &[Config]) -> Result<(), Failure> {
        (**self).validate(configs)
    }
    fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize> {
        (**self).suggest(view)
    }
    fn observe(&mut self, observation: &Observation) {
        (**self).observe(observation)
    }
}

/// Declared-order enumeration for comparison with model-guided search.
#[derive(Default)]
pub struct GridOrder;
impl Strategy for GridOrder {
    fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize> {
        (0..view.configs.len())
            .find(|i| !view.pending.contains(i) && !view.completed.iter().any(|o| o.trial_id == *i))
    }
}

/// Reproducible random selection without replacement. Exhaustive when the
/// budget covers the space. Optional initial indices are visited first.
/// Create a fresh strategy for each run; initial indices are suggestions, not
/// a guarantee of incumbent coverage or confirmation.
pub struct RandomOrder {
    state: u64,
    initial: std::collections::VecDeque<usize>,
    order: Option<Vec<usize>>,
}

impl RandomOrder {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            initial: Default::default(),
            order: None,
        }
    }
    pub fn with_initial(mut self, indices: impl IntoIterator<Item = usize>) -> Self {
        self.initial.extend(indices);
        self
    }
    fn random(&mut self) -> u64 {
        // SplitMix64: deterministic integer arithmetic, no process-global RNG.
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

impl Strategy for RandomOrder {
    fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize> {
        if self.order.is_none() {
            let mut order: Vec<_> = (0..view.configs.len()).collect();
            for i in (1..order.len()).rev() {
                // Rejection sampling avoids modulo bias in Fisher-Yates.
                let n = (i + 1) as u64;
                let threshold = n.wrapping_neg() % n;
                let value = loop {
                    let value = self.random();
                    if value >= threshold {
                        break value;
                    }
                };
                order.swap(i, (value % n) as usize);
            }
            self.order = Some(order);
        }
        let available = |i: usize| {
            !view.pending.contains(&i) && !view.completed.iter().any(|o| o.trial_id == i)
        };
        while let Some(index) = self.initial.pop_front() {
            if available(index) {
                return Some(index);
            }
        }
        while let Some(index) = self.order.as_mut().unwrap().pop() {
            if available(index) {
                return Some(index);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// The strategy has no more suggestions and no work remains pending.
    Completed,
    BudgetExhausted,
    Cancelled,
    Incomplete,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub observations: Vec<Observation>,
    pub unvisited: Vec<usize>,
    pub status: RunStatus,
    pub error: Option<String>,
    /// Authoritative wall time, including draining workers. Stage durations
    /// overlap and must not be summed to reconstruct this value.
    pub elapsed: Duration,
    /// Provisional search leader only; no paired finalist confirmation yet.
    pub best_observed: Option<Config>,
}

/// Bounded preparation. One worker means one candidate ahead of the current
/// evaluation; increasing this also increases the maximum ready queue size.
pub struct Pipeline {
    workers: usize,
    overlap: bool,
    budget: Option<Duration>,
    trial_limit: Option<Duration>,
    control: WorkControl,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            workers: 1,
            overlap: true,
            budget: None,
            trial_limit: None,
            control: WorkControl::default(),
        }
    }
}

struct Job<J> {
    index: usize,
    input: J,
    queued: Instant,
    resolve: Option<Duration>,
    control: WorkControl,
}

struct Ready<P> {
    index: usize,
    result: Result<P, Failure>,
    stages: StageTimes,
    control: WorkControl,
}

impl Pipeline {
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }
    /// Disable overlap for a sequential comparison using the identical protocol.
    pub fn overlap(mut self, overlap: bool) -> Self {
        self.overlap = overlap;
        self
    }
    pub fn budget(mut self, budget: Duration) -> Self {
        self.budget = Some(budget);
        self
    }
    /// Limit each active stage (resolve, prepare, evaluate), excluding queue
    /// time. Every stage is also bounded by the overall run deadline.
    pub fn trial_limit(mut self, limit: Duration) -> Self {
        self.trial_limit = Some(limit);
        self
    }
    /// Share cancellation with this run. An infrastructure failure cancels the
    /// shared control to stop its remaining workers. Use a fresh control per run.
    pub fn control(mut self, control: WorkControl) -> Self {
        self.control = control;
        self
    }

    /// Resolve on the foreground thread, prepare on CPU workers, then evaluate
    /// on the foreground thread. A successful evaluation must include the
    /// caller's correctness gate. Use `Failure::invalid` or convert a located
    /// frontend rejection with `Failure::from` for invalidity evidence.
    /// Return a measured `Trial` on success; legacy invalid
    /// trials can also represent broken timing and are infrastructure errors.
    /// Per-trial timeouts are unscored;
    /// infrastructure errors stop the run.
    ///
    /// Deadlines stop dispatch and exclude late measurements. Workers are
    /// drained before returning, so arbitrary non-cooperative callbacks can
    /// overrun the budget. This API does not claim to preempt Rust or GPU code.
    /// No existing resume log is read or written; serialize the returned
    /// observations together with source/toolchain/workload provenance.
    #[allow(clippy::too_many_arguments)]
    pub fn run<J: Send, P: Send>(
        self,
        configs: &[Config],
        strategy: &mut impl Strategy,
        mut resolve: impl FnMut(&Config) -> Result<J, Failure>,
        prepare: impl Fn(J, &WorkControl) -> Result<P, Failure> + Sync,
        mut evaluate: impl FnMut(&Config, P, &WorkControl) -> Result<Trial, Failure>,
    ) -> Result<PipelineOutput, Failure> {
        if self.workers == 0 {
            return Err(Failure::infrastructure("workers must be positive"));
        }
        let ids: HashSet<_> = configs.iter().map(|c| &c.id).collect();
        if ids.len() != configs.len() {
            return Err(Failure::infrastructure("duplicate configuration IDs"));
        }
        strategy.validate(configs)?;
        let start = Instant::now();
        let control = match self.budget {
            Some(budget) => self.control.with_deadline(
                start
                    .checked_add(budget)
                    .ok_or_else(|| Failure::infrastructure("budget is too large"))?,
            ),
            None => self.control,
        };
        let mut observations = Vec::new();
        let mut pending = Vec::new();
        let mut status = RunStatus::Completed;
        let mut run_error = None;
        let worker_count = self.workers.min(configs.len());
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job<J>>();
        let jobs_rx = Mutex::new(jobs_rx);
        let (results_tx, results_rx) = mpsc::channel::<Ready<P>>();
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                let rx = &jobs_rx;
                let tx = results_tx.clone();
                let prepare = &prepare;
                std::thread::Builder::new()
                    .name("cutile-prepare".into())
                    .stack_size(8 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        loop {
                            // Release the receiver lock before compiling.
                            let job = rx.lock().unwrap().recv();
                            let Ok(job) = job else { break };
                            let begun = Instant::now();
                            let mut prepare_time = None;
                            let result = stage_control(&job.control, self.trial_limit, begun)
                                .and_then(|stage| {
                                    stage.check()?;
                                    let result = std::panic::catch_unwind(
                                        std::panic::AssertUnwindSafe(|| prepare(job.input, &stage)),
                                    )
                                    .unwrap_or_else(|_| {
                                        Err(Failure::infrastructure("preparation worker panicked"))
                                    });
                                    prepare_time = Some(begun.elapsed());
                                    checked_result(result, &stage)
                                });
                            let ready = Ready {
                                index: job.index,
                                result,
                                control: job.control,
                                stages: StageTimes {
                                    resolve: job.resolve,
                                    queue: Some(begun.duration_since(job.queued)),
                                    prepare: prepare_time,
                                    evaluate: None,
                                },
                            };
                            if tx.send(ready).is_err() {
                                break;
                            }
                        }
                    })
                    .map_err(|e| {
                        Failure::infrastructure(format!("cannot start preparation worker: {e}"))
                    })?;
            }
            drop(results_tx);
            let fill = |limit: usize,
                        pending: &mut Vec<usize>,
                        observations: &mut Vec<Observation>,
                        strategy: &mut dyn Strategy,
                        resolve: &mut dyn FnMut(&Config) -> Result<J, Failure>|
             -> Result<(), Failure> {
                while pending.len() < limit {
                    control.check()?;
                    let view = SearchView {
                        configs,
                        completed: observations,
                        pending,
                        remaining: control
                            .deadline()
                            .map(|end| end.saturating_duration_since(Instant::now())),
                    };
                    let Some(index) = strategy.suggest(&view) else {
                        // A selector may spend the remaining budget fitting its
                        // model and return None. That is not space exhaustion.
                        control.check()?;
                        break;
                    };
                    if index >= configs.len()
                        || pending.contains(&index)
                        || observations.iter().any(|o| o.trial_id == index)
                    {
                        return Err(Failure::infrastructure(
                            "strategy suggested an unavailable configuration",
                        ));
                    }
                    let begun = Instant::now();
                    let trial_control = stage_control(&control, self.trial_limit, begun)?;
                    let mut resolve_time = None;
                    let input = trial_control.check().map_err(Failure::from).and_then(|()| {
                        let start = Instant::now();
                        let result = resolve(&configs[index]);
                        resolve_time = Some(start.elapsed());
                        result
                    });
                    let input = checked_result(input, &trial_control);
                    match input {
                        Ok(input) => {
                            pending.push(index);
                            jobs_tx
                                .send(Job {
                                    index,
                                    input,
                                    resolve: resolve_time,
                                    queued: Instant::now(),
                                    control: control.clone(),
                                })
                                .map_err(|_| {
                                    Failure::infrastructure("preparation workers disconnected")
                                })?;
                        }
                        Err(failure) => {
                            let observation = Observation {
                                trial_id: index,
                                config_id: configs[index].id.clone(),
                                stages: StageTimes {
                                    resolve: resolve_time,
                                    ..Default::default()
                                },
                                trial: None,
                                failure: Some(failure.clone()),
                            };
                            strategy.observe(&observation);
                            observations.push(observation);
                            if failure.kind == FailureKind::Infrastructure {
                                return Err(failure);
                            }
                        }
                    }
                }
                Ok(())
            };
            loop {
                if let Err(interrupted) = control.check() {
                    let failure = Failure::from(interrupted);
                    status = status_for(&failure);
                    run_error = Some(failure.reason);
                    break;
                }
                let limit = if self.overlap { worker_count } else { 1 };
                if let Err(error) = fill(
                    limit,
                    &mut pending,
                    &mut observations,
                    strategy,
                    &mut resolve,
                ) {
                    status = status_for(&error);
                    run_error = Some(error.reason);
                    break;
                }
                if pending.is_empty() {
                    break;
                }
                let mut ready = match results_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(ready) => ready,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => {
                        status = RunStatus::Incomplete;
                        run_error = Some("preparation workers disconnected".into());
                        break;
                    }
                };
                // Keep this candidate pending while selecting its successor.
                // Its unknown outcome must be visible to a model-guided strategy.
                let refill = if self.overlap {
                    fill(
                        worker_count.saturating_add(1),
                        &mut pending,
                        &mut observations,
                        strategy,
                        &mut resolve,
                    )
                } else {
                    Ok(())
                };
                if let Err(error) = refill {
                    status = status_for(&error);
                    run_error = Some(error.reason);
                    ready.result = Err(Failure::interrupted(
                        control
                            .check()
                            .err()
                            .unwrap_or(CompileInterrupted::Cancelled),
                        ready.result.err(),
                    ));
                }
                let result = checked_result(ready.result, &ready.control);
                let evaluation_control = stage_control(&control, self.trial_limit, Instant::now())?;
                let result = result.and_then(|prepared| {
                    evaluation_control.check()?;
                    let begun = Instant::now();
                    let result = evaluate(&configs[ready.index], prepared, &evaluation_control);
                    ready.stages.evaluate = Some(begun.elapsed());
                    result
                });
                let observation = observation(
                    configs,
                    ready.index,
                    ready.stages,
                    result,
                    &evaluation_control,
                );
                pending.retain(|&index| index != ready.index);
                let fatal = observation
                    .failure
                    .as_ref()
                    .filter(|e| e.kind == FailureKind::Infrastructure)
                    .cloned();
                strategy.observe(&observation);
                observations.push(observation);
                if let Some(error) = fatal {
                    status = status_for(&error);
                    run_error = Some(error.reason);
                    break;
                }
                if status != RunStatus::Completed {
                    break;
                }
            }
            drop(jobs_tx);
            if status != RunStatus::Completed && status != RunStatus::BudgetExhausted {
                control.cancel()
            }
            // Draining is evidence collection only: no new GPU work or scoring.
            while !pending.is_empty() {
                let Ok(ready) = results_rx.recv() else {
                    status = RunStatus::Incomplete;
                    break;
                };
                pending.retain(|&index| index != ready.index);
                let interruption = ready
                    .control
                    .check()
                    .err()
                    .unwrap_or(CompileInterrupted::Cancelled);
                let failure = Failure::interrupted(interruption, ready.result.err());
                let observation = Observation {
                    trial_id: ready.index,
                    config_id: configs[ready.index].id.clone(),
                    stages: ready.stages,
                    trial: None,
                    failure: Some(failure),
                };
                strategy.observe(&observation);
                observations.push(observation);
            }
            Ok::<(), Failure>(())
        })?;
        let best_observed = if status == RunStatus::Incomplete {
            None
        } else {
            observations
                .iter()
                .filter_map(|o| o.score_ms().map(|score| (o.trial_id, score)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| configs[i].clone())
        };
        Ok(PipelineOutput {
            unvisited: (0..configs.len())
                .filter(|i| !observations.iter().any(|o| o.trial_id == *i))
                .collect(),
            observations,
            best_observed,
            status,
            error: run_error,
            elapsed: start.elapsed(),
        })
    }
}

fn stage_control(
    overall: &WorkControl,
    limit: Option<Duration>,
    begun: Instant,
) -> Result<WorkControl, Failure> {
    match limit {
        Some(limit) => Ok(overall.with_deadline(
            begun
                .checked_add(limit)
                .ok_or_else(|| Failure::infrastructure("trial limit is too large"))?,
        )),
        None => Ok(overall.clone()),
    }
}

/// Discard late values while retaining failures from non-cooperative callbacks.
fn checked_result<T>(result: Result<T, Failure>, control: &WorkControl) -> Result<T, Failure> {
    match control.check() {
        Ok(()) => result,
        Err(interruption) => Err(Failure::interrupted(interruption, result.err())),
    }
}

fn status_for(failure: &Failure) -> RunStatus {
    match failure.kind {
        FailureKind::TimedOut => RunStatus::BudgetExhausted,
        FailureKind::Cancelled => RunStatus::Cancelled,
        FailureKind::Invalid | FailureKind::Infrastructure => RunStatus::Incomplete,
    }
}

fn observation(
    configs: &[Config],
    index: usize,
    stages: StageTimes,
    result: Result<Trial, Failure>,
    control: &WorkControl,
) -> Observation {
    let (trial, mut failure) = match result {
        Ok(mut trial) => {
            // Identity is scheduler-owned, as in the synchronous Objective API.
            trial.config_id = configs[index].id.clone();
            let failure = match &trial.state {
                TrialState::Invalid { reason } => Some(Failure::infrastructure(format!(
                    "evaluation returned an unmeasured trial: {reason}; use Failure::invalid for a known validity failure"
                ))),
                TrialState::Measured {
                    median_ms,
                    min_ms,
                    reps,
                } if !median_ms.is_finite()
                    || *median_ms <= 0.0
                    || !min_ms.is_finite()
                    || *min_ms <= 0.0
                    || min_ms > median_ms
                    || *reps == 0 =>
                {
                    Some(Failure::infrastructure("invalid timing result"))
                }
                _ => None,
            };
            (Some(trial), failure)
        }
        Err(error) => (None, Some(error)),
    };
    if let Err(interrupted) = control.check() {
        failure = Some(Failure::interrupted(interrupted, failure))
    }
    Observation {
        trial_id: index,
        config_id: configs[index].id.clone(),
        stages,
        trial,
        failure,
    }
}

#[cfg(test)]
mod tests {
    use super::super::ParamValue;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn configs(count: usize) -> Vec<Config> {
        (0..count)
            .map(|i| Config::new([("x", ParamValue::Int(i as i64))]))
            .collect()
    }
    fn measured(config: &Config) -> Trial {
        Trial::measured(&config.id, 1.0, 0.9, 10)
    }

    #[test]
    fn successor_prepares_during_evaluation_and_observes_previous_result() {
        // Channels prove overlap; the timeout is only a deadlock watchdog, not
        // an assertion about how quickly either stage ought to complete.
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        struct Ordered {
            next: usize,
        }
        impl Strategy for Ordered {
            fn suggest(&mut self, view: &SearchView<'_>) -> Option<usize> {
                let i = self.next;
                if i >= view.configs.len() {
                    return None;
                }
                if i == 1 {
                    assert_eq!(view.pending, &[0]);
                    assert!(view.completed.is_empty());
                }
                if i == 2 {
                    assert_eq!(view.pending, &[1]);
                    assert_eq!(view.completed[0].trial_id, 0);
                }
                self.next += 1;
                Some(i)
            }
        }
        let result = Pipeline::default()
            .run(
                &configs(3),
                &mut Ordered { next: 0 },
                |c| Ok(c.int("x").unwrap()),
                |i, _| {
                    if i == 1 {
                        started_tx.send(()).unwrap();
                        release_rx
                            .lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(30))
                            .unwrap();
                    }
                    Ok(i)
                },
                |c, i, _| {
                    if i == 0 {
                        started_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                        release_tx.send(()).unwrap();
                    }
                    Ok(measured(c))
                },
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Completed);
        assert_eq!(result.observations.len(), 3);
    }

    #[test]
    fn workers_prepare_concurrently_and_ready_artifacts_are_bounded() {
        let (started_tx, started_rx) = mpsc::channel();
        let gate = (Mutex::new(false), std::sync::Condvar::new());
        let alive = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        struct Artifact<'a>(&'a AtomicUsize);
        impl Drop for Artifact<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        std::thread::scope(|scope| {
            let gate_ref = &gate;
            scope.spawn(move || {
                let mut received = 0;
                for _ in 0..4 {
                    if started_rx.recv_timeout(Duration::from_secs(30)).is_ok() {
                        received += 1;
                    } else {
                        break;
                    }
                }
                *gate_ref.0.lock().unwrap() = true;
                gate_ref.1.notify_all();
                assert_eq!(
                    received, 4,
                    "all four workers must start before any is released"
                );
            });
            let result = Pipeline::default()
                .workers(4)
                .run(
                    &configs(12),
                    &mut RandomOrder::new(7),
                    |c| Ok(c.clone()),
                    |_, _| {
                        let _ = started_tx.send(());
                        let mut released = gate.0.lock().unwrap();
                        while !*released {
                            released = gate.1.wait(released).unwrap();
                        }
                        let count = alive.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        Ok(Artifact(&alive))
                    },
                    |c, _artifact, _| Ok(measured(c)),
                )
                .unwrap();
            assert_eq!(result.observations.len(), 12);
        });
        assert!(
            peak.load(Ordering::SeqCst) <= 5,
            "four ahead plus current evaluation"
        );
        assert_eq!(alive.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn zero_budget_dispatches_nothing() {
        let result = Pipeline::default()
            .budget(Duration::ZERO)
            .run::<(), ()>(
                &configs(2),
                &mut RandomOrder::new(0),
                |_| panic!("must not resolve"),
                |_, _| panic!("must not prepare"),
                |_, _, _| panic!("must not evaluate"),
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::BudgetExhausted);
        assert!(result.observations.is_empty());
        assert_eq!(result.unvisited, [0, 1]);
    }

    #[test]
    fn stage_limits_start_at_execution_and_remain_bounded_by_run() {
        let start = Instant::now();
        let overall = WorkControl::default().with_deadline(start + Duration::from_secs(100));
        for (offset, expected) in [(0, 30), (40, 70), (90, 100)] {
            let stage = stage_control(
                &overall,
                Some(Duration::from_secs(30)),
                start + Duration::from_secs(offset),
            )
            .unwrap();
            assert_eq!(
                stage.deadline(),
                Some(start + Duration::from_secs(expected))
            );
        }
        for workers in [1, 2, 4] {
            let limit = Duration::from_secs(30);
            let result = Pipeline::default()
                .workers(workers)
                .trial_limit(limit)
                .run(
                    &configs(24),
                    &mut GridOrder,
                    |_| Ok(Instant::now()),
                    |resolved, control| {
                        // Deadline arithmetic verifies that queueing did not consume
                        // the stage allowance; no execution-speed assertion needed.
                        assert!(control.deadline().unwrap() >= resolved + limit);
                        Ok(Instant::now())
                    },
                    |c, prepared, control| {
                        assert!(control.deadline().unwrap() >= prepared + limit);
                        Ok(measured(c))
                    },
                )
                .unwrap();
            assert_eq!(result.status, RunStatus::Completed);
            assert_eq!(
                result
                    .observations
                    .iter()
                    .filter(|o| o.score_ms().is_some())
                    .count(),
                24
            );
        }
    }

    #[test]
    fn selector_interruption_is_not_reported_as_exhaustion() {
        struct CancelSelector(WorkControl);
        impl Strategy for CancelSelector {
            fn suggest(&mut self, _: &SearchView<'_>) -> Option<usize> {
                self.0.cancel();
                None
            }
        }
        let control = WorkControl::default();
        let mut selector = CancelSelector(control.clone());
        let result = Pipeline::default()
            .control(control)
            .run::<(), ()>(
                &configs(2),
                &mut selector,
                |_| panic!("no dispatch"),
                |_, _| unreachable!(),
                |_, _, _| unreachable!(),
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Cancelled);
        assert_eq!(result.unvisited.len(), 2);
    }

    #[test]
    fn frontend_rejection_skips_candidate_but_unclassified_errors_stop_run() {
        use cutile_compiler::{ast::SourceLocation, error::JITError};
        for located in [true, false] {
            let result = Pipeline::default()
                .overlap(false)
                .run(
                    &configs(3),
                    &mut GridOrder,
                    |c| Ok(c.int("x").unwrap()),
                    |i, _| {
                        if i == 1 {
                            let error = if located {
                                JITError::located(
                                    "tile size must be a power of two",
                                    SourceLocation::new("kernel.rs".into(), 7, 3),
                                )
                            } else {
                                JITError::Generic("cannot launch assembler".into())
                            };
                            Err(Failure::from(Error::JIT(error)))
                        } else {
                            Ok(())
                        }
                    },
                    |c, _, _| Ok(measured(c)),
                )
                .unwrap();
            let failure = result.observations[1].failure.as_ref().unwrap();
            if located {
                assert_eq!(failure.kind, FailureKind::Invalid);
                assert!(failure.reason.contains("kernel.rs:7:3"));
                assert_eq!(result.status, RunStatus::Completed);
                assert!(result.best_observed.is_some());
                assert_eq!(result.observations.len(), 3);
            } else {
                assert_eq!(failure.kind, FailureKind::Infrastructure);
                assert_eq!(result.status, RunStatus::Incomplete);
                assert!(result.best_observed.is_none());
            }
        }
    }

    #[test]
    fn expired_trial_is_not_invalid_and_does_not_run_resolver() {
        let result = Pipeline::default()
            .trial_limit(Duration::ZERO)
            .run::<(), ()>(
                &configs(2),
                &mut RandomOrder::new(0),
                |_| panic!("expired before resolver"),
                |_, _| unreachable!(),
                |_, _, _| unreachable!(),
            )
            .unwrap();
        assert_eq!(result.observations.len(), 2);
        assert!(result
            .observations
            .iter()
            .all(|o| o.failure.as_ref().unwrap().kind == FailureKind::TimedOut));
        assert!(result.best_observed.is_none());
        let encoded = serde_json::to_value(&result).unwrap();
        assert!(encoded["observations"][0]["stages"]["resolve"].is_null());
        assert!(encoded["observations"][0]["stages"]["prepare"].is_null());
        let decoded: PipelineOutput = serde_json::from_value(encoded).unwrap();
        assert!(decoded.observations.iter().all(|o| o.score_ms().is_none()));
    }

    #[test]
    fn cancellation_while_waiting_for_preparation_skips_evaluation() {
        let control = WorkControl::default();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        std::thread::scope(|scope| {
            let external_control = control.clone();
            scope.spawn(move || {
                started_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                external_control.cancel();
                release_tx.send(()).unwrap();
            });
            let result = Pipeline::default()
                .control(control)
                .run(
                    &configs(2),
                    &mut RandomOrder::new(0),
                    |_| Ok(()),
                    |_, _| {
                        started_tx.send(()).unwrap();
                        release_rx
                            .lock()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(30))
                            .unwrap();
                        // Simulate a non-cooperative callback returning an artifact.
                        Ok(())
                    },
                    |_, _, _| panic!("cancelled artifact must not reach the GPU"),
                )
                .unwrap();
            assert_eq!(result.status, RunStatus::Cancelled);
            assert_eq!(result.observations.len(), 1);
            let observation = &result.observations[0];
            assert_eq!(
                observation.failure.as_ref().unwrap().kind,
                FailureKind::Cancelled
            );
            assert!(observation.stages.evaluate.is_none());
            assert!(result.best_observed.is_none());
        });
    }

    #[test]
    fn preparation_panic_aborts_without_labeling_candidate_invalid() {
        let result = Pipeline::default()
            .run(
                &configs(1),
                &mut RandomOrder::new(0),
                |_| Ok(()),
                |_, _| -> Result<(), Failure> { panic!("broken worker") },
                |_, _, _| unreachable!(),
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Incomplete);
        assert_eq!(
            result.observations[0].failure.as_ref().unwrap().kind,
            FailureKind::Infrastructure
        );
        assert!(result.best_observed.is_none());
    }

    #[test]
    fn cancelled_measurement_is_evidence_but_cannot_win() {
        let result = Pipeline::default()
            .run(
                &configs(1),
                &mut RandomOrder::new(0),
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, control| {
                    control.cancel();
                    Ok(measured(c))
                },
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Cancelled);
        assert!(result.observations[0].trial.is_some());
        assert_eq!(
            result.observations[0].failure.as_ref().unwrap().kind,
            FailureKind::Cancelled
        );
        assert!(result.best_observed.is_none());
    }

    #[test]
    fn invalid_candidates_continue_but_infrastructure_failure_aborts() {
        let cs = configs(3);
        for infrastructure in [false, true] {
            let result = Pipeline::default()
                .overlap(false)
                .run(
                    &cs,
                    &mut RandomOrder::new(0).with_initial([0, 1, 2]),
                    |c| Ok(c.int("x").unwrap()),
                    |_, _| Ok(()),
                    |c, _, _| {
                        if c.int("x") == Some(1) {
                            Err(if infrastructure {
                                Failure::infrastructure("missing executable")
                            } else {
                                Failure::invalid("incorrect result")
                            })
                        } else {
                            Ok(measured(c))
                        }
                    },
                )
                .unwrap();
            if infrastructure {
                assert_eq!(result.status, RunStatus::Incomplete);
                assert!(result.best_observed.is_none());
                assert_eq!(result.unvisited, [2]);
            } else {
                assert_eq!(result.status, RunStatus::Completed);
                assert_eq!(result.observations.len(), 3);
                assert!(result.best_observed.is_some());
            }
        }
    }

    #[test]
    fn selector_cannot_resubmit_pending_or_completed_work() {
        struct Duplicate;
        impl Strategy for Duplicate {
            fn suggest(&mut self, _: &SearchView<'_>) -> Option<usize> {
                Some(0)
            }
        }
        let result = Pipeline::default()
            .run(
                &configs(2),
                &mut Duplicate,
                |_| Ok(()),
                |_, _| Ok(()),
                |c, _, _| Ok(measured(c)),
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Incomplete);
        assert!(result.error.unwrap().contains("unavailable"));
    }

    #[test]
    fn cancelled_callbacks_preserve_their_failure_in_every_stage() {
        for stage in 0..3 {
            for original in [
                Failure::invalid("bad output"),
                Failure::infrastructure("transport failed"),
            ] {
                let control = WorkControl::default();
                let external = control.clone();
                let result = Pipeline::default()
                    .overlap(false)
                    .control(control)
                    .run::<(), ()>(
                        &configs(1),
                        &mut GridOrder,
                        |_| {
                            if stage == 0 {
                                external.cancel();
                                Err(original.clone())
                            } else {
                                Ok(())
                            }
                        },
                        |_, control| {
                            if stage == 1 {
                                control.cancel();
                                Err(original.clone())
                            } else {
                                Ok(())
                            }
                        },
                        |_, _, control| {
                            assert_eq!(stage, 2);
                            control.cancel();
                            Err(original.clone())
                        },
                    )
                    .unwrap();
                assert_eq!(result.status, RunStatus::Cancelled);
                assert!(result.best_observed.is_none());
                assert_eq!(result.observations.len(), 1);
                let observation = &result.observations[0];
                assert!(observation.score_ms().is_none());
                assert!(observation.trial.is_none());
                assert_eq!(observation.stages.prepare.is_some(), stage >= 1);
                assert_eq!(observation.stages.evaluate.is_some(), stage == 2);
                let failure = observation.failure.as_ref().unwrap();
                assert_eq!(failure.kind, FailureKind::Cancelled);
                assert!(failure.reason.contains("work cancelled"));
                assert!(failure.reason.contains(&original.to_string()));
                let decoded: PipelineOutput =
                    serde_json::from_value(serde_json::to_value(&result).unwrap()).unwrap();
                assert_eq!(
                    decoded.observations[0].failure.as_ref().unwrap().reason,
                    failure.reason
                );
                assert!(decoded.observations[0].score_ms().is_none());
            }
        }
    }

    #[test]
    fn expired_callback_results_retain_diagnostics_without_scoring() {
        // Produce callback results before expiring the control. This tests the
        // completion boundary without racing a short budget against scheduling.
        for original in [
            Failure::invalid("wrong shape"),
            Failure::infrastructure("assembler failed"),
        ] {
            let resolved: Result<usize, _> = Err(original.clone());
            let prepared: Result<Vec<u8>, _> = Err(original.clone());
            let evaluated: Result<Trial, _> = Err(original.clone());
            let control = WorkControl::default().with_deadline(Instant::now());
            for failure in [
                checked_result(resolved, &control).unwrap_err(),
                checked_result(prepared, &control).unwrap_err(),
            ] {
                assert_eq!(failure.kind, FailureKind::TimedOut);
                assert!(failure.reason.contains("work deadline exceeded"));
                assert!(failure.reason.contains(&original.to_string()));
                let again = checked_result::<()>(Err(failure.clone()), &control).unwrap_err();
                assert_eq!(again.reason, failure.reason);
            }
            let observation =
                observation(&configs(1), 0, StageTimes::default(), evaluated, &control);
            assert!(observation.score_ms().is_none());
            let failure = observation.failure.unwrap();
            assert_eq!(failure.kind, FailureKind::TimedOut);
            assert!(failure.reason.contains("work deadline exceeded"));
            assert!(failure.reason.contains(&original.to_string()));
        }
    }

    #[test]
    fn drained_worker_failure_retains_cause_and_cancellation() {
        for original in [
            Failure::invalid("rejected config"),
            Failure::infrastructure("compiler crashed"),
        ] {
            let (started, waiting) = mpsc::channel();
            let (release, resume) = mpsc::channel();
            let resume = Mutex::new(resume);
            let result = Pipeline::default()
                .run(
                    &configs(2),
                    &mut GridOrder,
                    |c| Ok(c.int("x").unwrap()),
                    |index, _| {
                        if index == 1 {
                            started.send(()).unwrap();
                            resume
                                .lock()
                                .unwrap()
                                .recv_timeout(Duration::from_secs(30))
                                .unwrap();
                            Err(original.clone())
                        } else {
                            Ok(())
                        }
                    },
                    |c, _, control| {
                        assert_eq!(c.int("x"), Some(0));
                        waiting.recv_timeout(Duration::from_secs(30)).unwrap();
                        control.cancel();
                        release.send(()).unwrap();
                        Ok(measured(c))
                    },
                )
                .unwrap();
            assert_eq!(result.status, RunStatus::Cancelled);
            assert!(result.best_observed.is_none());
            assert_eq!(result.observations.len(), 2);
            let drained = result
                .observations
                .iter()
                .find(|o| o.trial_id == 1)
                .unwrap();
            assert!(drained.stages.evaluate.is_none());
            assert!(drained.score_ms().is_none());
            let failure = drained.failure.as_ref().unwrap();
            assert_eq!(failure.kind, FailureKind::Cancelled);
            assert!(failure.reason.contains(&original.to_string()));
        }
    }

    #[test]
    fn successor_resolution_error_does_not_erase_preparation_error() {
        let result = Pipeline::default()
            .run::<(), ()>(
                &configs(2),
                &mut GridOrder,
                |c| {
                    if c.int("x") == Some(0) {
                        Ok(())
                    } else {
                        Err(Failure::infrastructure("resolver broke"))
                    }
                },
                |_, _| Err(Failure::invalid("prepare rejected candidate")),
                |_, _, _| panic!("failed preparation cannot be evaluated"),
            )
            .unwrap();
        assert_eq!(result.status, RunStatus::Incomplete);
        assert!(result.best_observed.is_none());
        assert!(result.error.unwrap().contains("resolver broke"));
        let prepared = result
            .observations
            .iter()
            .find(|o| o.trial_id == 0)
            .unwrap();
        assert!(prepared.score_ms().is_none());
        let failure = prepared.failure.as_ref().unwrap();
        assert_eq!(failure.kind, FailureKind::Cancelled);
        assert!(failure
            .reason
            .contains("Invalid: prepare rejected candidate"));
    }

    #[test]
    fn typed_interruption_survives_compiler_error_context() {
        let error = Error::JIT(cutile_compiler::error::JITError::Anyhow(
            anyhow::Error::new(CompileInterrupted::DeadlineExceeded).context("assembler stopped"),
        ));
        assert_eq!(Failure::from(error).kind, FailureKind::TimedOut);
        assert_eq!(
            Failure::from(Error::JIT(cutile_compiler::error::JITError::Generic(
                "missing tileiras".into()
            )))
            .kind,
            FailureKind::Infrastructure
        );
    }

    #[test]
    fn broken_measurements_are_not_kernel_invalidity() {
        for median in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
            let result = Pipeline::default()
                .run(
                    &configs(1),
                    &mut RandomOrder::new(0),
                    |_| Ok(()),
                    |_, _| Ok(()),
                    |c, _, _| Ok(Trial::measured(&c.id, median, median, 1)),
                )
                .unwrap();
            assert_eq!(result.status, RunStatus::Incomplete);
            assert_eq!(
                result.observations[0].failure.as_ref().unwrap().kind,
                FailureKind::Infrastructure
            );
            assert!(result.best_observed.is_none());
        }
    }
}
