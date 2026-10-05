/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel autotuning: declared search spaces, pluggable searchers, and
//! persisted, provenance-checked results.
//!
//! **Experimental.** This module is gated behind the `experimental-tune`
//! Cargo feature; enabling it opts into an API that may change in breaking
//! ways between releases.
//!
//! The vocabulary follows the tools users already know: a [`Config`] is one
//! candidate configuration (Triton's `Config`), a [`Searcher`] decides the
//! visit order (Ray Tune's word), [`GridSearch`] is the default exhaustive
//! searcher, and each measured candidate produces a [`Trial`] (Optuna's
//! word). Measurement
//! runs through [`crate::bench::do_bench`] (CUDA events, warmup, L2
//! clearing, medians).
//!
//! Principles, in order:
//! - **Explicit opt-in, no magic.** Nothing tunes behind the programmer's
//!   back; the search space is declared, the objective is programmer-written,
//!   and results apply only when a program explicitly loads a record.
//! - **Invalid candidates are data.** A candidate rejected by launch checks
//!   or the correctness gate records [`TrialState::Invalid`] with its message;
//!   it never aborts the search.
//! - **Persistence is checked.** The trial log records the tuner's name, a
//!   hash of its search space, and — when supplied — the arch, the kernel's
//!   source hash, and the toolchain fingerprint ([`LogProvenance`]), and
//!   refuses to resume from a log that does not match; the record store
//!   built on top applies the same provenance to recorded winners.

use crate::bench::{do_bench, BenchOptions, Measurement};
use crate::error::Error;
use cuda_core::Stream;
use cutile_compiler::jit_cache::L2_KEY_SCHEMA;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── Config ──────────────────────────────────────────────────────────────────

/// One candidate configuration: named integer/string parameters.
///
/// Parameters cover both user axes (tile sizes, split counts — read by the
/// launch closure to pick a monomorphization) and compiler knobs (applied by
/// the launch closure via `CompileOptions`). Keeping them in one ordered map
/// makes a `Config` fully serializable for records and logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Config {
    /// Stable identity within a search space; record and log records key
    /// on it. [`Config::new`] derives it from the parameters — construct
    /// through `new` so the two can never disagree.
    pub id: String,
    /// Ordered parameter map.
    pub params: BTreeMap<String, ParamValue>,
}

/// A parameter value: integer or string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Int(i64),
    Str(String),
}

impl Config {
    /// Builds a config from parameters, deriving a stable id of the form
    /// `k1=v1,k2=v2` (keys in sorted order).
    pub fn new<I, K>(params: I) -> Self
    where
        I: IntoIterator<Item = (K, ParamValue)>,
        K: Into<String>,
    {
        let params: BTreeMap<String, ParamValue> =
            params.into_iter().map(|(k, v)| (k.into(), v)).collect();
        // String values are JSON-encoded (quoted + escaped) so ids cannot
        // alias: `A=1` (int) differs from `A="1"` (string), and a string
        // containing `,` or `=` cannot imitate additional parameters. Keys
        // holding a separator (or a quote) are JSON-encoded the same way; a
        // raw key can never start with `"`, so the two forms cannot collide.
        let id = params
            .iter()
            .map(|(k, v)| {
                let key = if k.contains(['=', ',', '"']) {
                    serde_json::to_string(k).unwrap_or_else(|_| format!("{k:?}"))
                } else {
                    k.clone()
                };
                match v {
                    ParamValue::Int(i) => format!("{key}={i}"),
                    ParamValue::Str(s) => format!(
                        "{key}={}",
                        serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}"))
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        Self { id, params }
    }

    /// Integer parameter accessor.
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.params.get(key) {
            Some(ParamValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// String parameter accessor.
    pub fn str(&self, key: &str) -> Option<&str> {
        match self.params.get(key) {
            Some(ParamValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Order-independent fingerprint of a candidate set. The trial log and
/// records record it so that resume/apply against a *different* search
/// space is detected instead of silently trusted.
pub fn space_hash(configs: &[Config]) -> String {
    // FNV-1a over the sorted ids; stability matters, cryptography does not.
    let mut ids: Vec<&str> = configs.iter().map(|c| c.id.as_str()).collect();
    ids.sort_unstable();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        for b in id.as_bytes().iter().chain(&[0u8]) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    format!("{h:016x}")
}

// ── Trial ───────────────────────────────────────────────────────────────────

/// The result of visiting one candidate.
///
/// **Serialization stability:** the serde form of `Trial`/[`TrialState`] is a
/// stable contract — resumable trial logs and restart-loop drivers depend on
/// it round-tripping across cutile upgrades. An incompatible change bumps the
/// [`TrialLog`] header's `log_schema`, and `TrialLog::open` refuses a
/// mismatched header rather than silently discarding resume state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Trial {
    pub config_id: String,
    pub state: TrialState,
}

/// What happened when a candidate was visited.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TrialState {
    /// Correctness gate passed; timing measured.
    Measured {
        median_ms: f32,
        min_ms: f32,
        reps: usize,
    },
    /// Rejected — by a launch check, the correctness gate, or a compile
    /// error. The message is recorded; the search continues.
    Invalid { reason: String },
}

impl TrialState {
    /// A `Measured` state, or `Invalid` if the median is non-finite. A NaN or
    /// infinite median serializes to `null` in the JSONL log (serde has no
    /// non-finite float form), which the resume reader then silently drops —
    /// so a non-finite timing must never enter the log as `Measured`.
    fn measured_or_invalid(median_ms: f32, min_ms: f32, reps: usize) -> Self {
        if !median_ms.is_finite() || !min_ms.is_finite() {
            return TrialState::Invalid {
                reason: format!("non-finite timing (median {median_ms}, min {min_ms})"),
            };
        }
        TrialState::Measured {
            median_ms,
            min_ms,
            reps,
        }
    }
}

impl Trial {
    /// Constructs a measured trial. This is the out-of-crate [`Objective`]
    /// implementor's path to a return value: the types are
    /// `#[non_exhaustive]`, so literal construction is crate-private by
    /// design, and these constructors are the public surface.
    ///
    /// A non-finite `median_ms` is recorded as [`TrialState::Invalid`] rather
    /// than `Measured` — a non-finite timing cannot round-trip through the log.
    pub fn measured(
        config_id: impl Into<String>,
        median_ms: f32,
        min_ms: f32,
        reps: usize,
    ) -> Self {
        Self {
            config_id: config_id.into(),
            state: TrialState::measured_or_invalid(median_ms, min_ms, reps),
        }
    }

    /// Constructs a rejected trial carrying the rejection reason.
    pub fn invalid(config_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            config_id: config_id.into(),
            state: TrialState::Invalid {
                reason: reason.into(),
            },
        }
    }

    /// The median time, if measured.
    pub fn median_ms(&self) -> Option<f32> {
        match &self.state {
            TrialState::Measured { median_ms, .. } => Some(*median_ms),
            TrialState::Invalid { .. } => None,
        }
    }
}

// ── Objective ──────────────────────────────────────────────────────────────────

/// What a [`Searcher`] searches through: a finite candidate list and a way to
/// measure one candidate.
///
/// For the common case the library implements this for you — supply launch and
/// gate closures to [`Autotuner::run`]. Implement it yourself (and pass it to
/// [`Autotuner::run_objective`]) when an engine already owns the launch,
/// correctness-gate, and timing machinery and just wants the search, logging,
/// resume, and required-config coverage on top.
pub trait Objective {
    /// The declared candidates, pruned, in declaration order.
    fn configs(&self) -> &[Config];
    /// Visits candidate `index`: correctness gate, then timing. Failures
    /// become [`TrialState::Invalid`]; this never panics for a bad candidate.
    /// The returned trial's `config_id` is authoritative-stamped from `index`
    /// by the library, so an implementation need not echo it back exactly.
    fn measure(&mut self, index: usize) -> Trial;
    /// Remaining wall-clock budget, if one was set.
    fn budget_remaining(&self) -> Option<Duration>;
}

// ── Searcher ─────────────────────────────────────────────────────────────────

/// Decides which candidates to visit and in what order.
///
/// Implementations must treat the objective's budget as authoritative and must
/// tolerate [`TrialState::Invalid`] trials. The library ships [`GridSearch`]
/// (the default); a TPE searcher is planned as an explicit opt-in.
pub trait Searcher {
    /// Runs the search, returning every trial visited (in visit order).
    fn search(&mut self, objective: &mut dyn Objective) -> Vec<Trial>;
}

/// Exhaustive searcher: visits every candidate once, in declaration order,
/// skipping candidates whose trials were supplied by [`resume`] and stopping
/// early only when the objective's budget runs out.
///
/// [`resume`]: GridSearch::resume
#[derive(Default)]
pub struct GridSearch {
    known: Vec<Trial>,
}

impl GridSearch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seeds already-known trials (e.g. parsed from a previous run's log);
    /// their candidates are not re-measured.
    pub fn resume(mut self, known: Vec<Trial>) -> Self {
        self.known = known;
        self
    }
}

impl Searcher for GridSearch {
    fn search(&mut self, objective: &mut dyn Objective) -> Vec<Trial> {
        // Resumed trials count only when (a) their config still exists in the
        // current space — a removed or renamed candidate's history must not
        // decide this search — and (b) they actually measured: an Invalid may
        // have been transient (poisoned context, OOM next door), so it is
        // retried; genuinely invalid candidates fail again cheaply.
        let current: std::collections::BTreeSet<&str> =
            objective.configs().iter().map(|c| c.id.as_str()).collect();
        let mut trials: Vec<Trial> = std::mem::take(&mut self.known)
            .into_iter()
            .filter(|t| current.contains(t.config_id.as_str()) && t.median_ms().is_some())
            .collect();
        let visited: std::collections::BTreeSet<String> =
            trials.iter().map(|t| t.config_id.clone()).collect();
        let todo: Vec<usize> = (0..objective.configs().len())
            .filter(|i| !visited.contains(&objective.configs()[*i].id))
            .collect();
        for index in todo {
            if objective.budget_remaining() == Some(Duration::ZERO) {
                break;
            }
            trials.push(objective.measure(index));
        }
        trials
    }
}

/// Selects the winner among trials: the measured candidate with the lowest
/// median. `None` when nothing measured successfully.
pub fn best_config<'a>(configs: &'a [Config], trials: &[Trial]) -> Option<&'a Config> {
    let mut best: Option<(&'a Config, f32)> = None;
    for t in trials {
        let Some(ms) = t.median_ms() else { continue };
        if !ms.is_finite() {
            continue;
        }
        // A trial whose config is no longer in the space (a stale resumed
        // record) must not decide the winner — and must not shadow a valid
        // one either, so it is skipped rather than looked up and lost.
        let Some(config) = configs.iter().find(|c| c.id == t.config_id) else {
            continue;
        };
        if best.is_none_or(|(_, b)| ms < b) {
            best = Some((config, ms));
        }
    }
    best.map(|(c, _)| c)
}

// ── Autotuner ───────────────────────────────────────────────────────────────

/// The assembled tuner for one kernel and one shape class.
///
/// ```rust,ignore
/// let output = Autotuner::new("fmha_decode")
///     .configs(configs)
///     .prune(|c| c.int("BN").unwrap() <= pp)
///     .budget(Duration::from_secs(300))
///     .run(&stream, |stream, config| {
///         // launch with `config`, verify against a reference, then return
///         // the closure do_bench will time. Err(..) => TrialState::Invalid.
///         ...
///     })?;
/// ```
/// A candidate-filter predicate (see [`Autotuner::prune`]).
type PrunePredicate = Box<dyn Fn(&Config) -> bool>;

pub struct Autotuner {
    /// Name recorded in the trial log header; a resume against a log written
    /// by a different tuner is refused.
    pub name: String,
    configs: Vec<Config>,
    prune: Vec<PrunePredicate>,
    required: Vec<Config>,
    budget: Option<Duration>,
    bench: BenchOptions,
    log_path: Option<PathBuf>,
    provenance: LogProvenance,
}

/// What a tuning run returns: every trial visited plus the winning config,
/// if any. Named like [`std::process::Output`], and — like it — `Debug` and
/// `Clone`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Output {
    pub trials: Vec<Trial>,
    pub best: Option<Config>,
}

impl Autotuner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            configs: Vec::new(),
            prune: Vec::new(),
            required: Vec::new(),
            budget: None,
            bench: BenchOptions::default(),
            log_path: None,
            provenance: LogProvenance::default(),
        }
    }

    /// Records the target architecture (e.g. `"sm_120"`) in the trial log
    /// header, so a resume from a log written on a different arch is refused.
    /// The caller resolves the arch up front (mirroring [`Workspace::arch`]);
    /// the tuner does not derive it from a device — the objective path has no
    /// device handle. Optional: a log written without an arch, or opened
    /// without one, skips the check (legacy logs stay readable).
    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.provenance.arch = Some(arch.into());
        self
    }

    /// Records the kernel module's `_SOURCE_HASH` in the trial log header, so
    /// a resume from a log written before the kernel was edited is refused
    /// (mirroring [`Workspace::source_hash`]). Optional, like [`arch`](Self::arch).
    pub fn source_hash(mut self, source_hash: impl Into<String>) -> Self {
        self.provenance.source_hash = Some(source_hash.into());
        self
    }

    /// Records the `tileiras --version` fingerprint in the trial log header,
    /// so a resume from a log written under another toolkit is refused
    /// (mirroring [`Workspace::tileiras_fingerprint`]). Optional, like
    /// [`arch`](Self::arch).
    pub fn tileiras_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.provenance.tileiras_fingerprint = Some(fingerprint.into());
        self
    }

    /// Sets every provenance axis at once — typically
    /// [`LogProvenance::from_workspace`] of the [`Workspace`] the results will
    /// be recorded against.
    pub fn provenance(mut self, provenance: LogProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Declares the candidate list (Triton's `configs=[...]`).
    pub fn configs(mut self, configs: Vec<Config>) -> Self {
        self.configs = configs;
        self
    }

    /// Declares incumbent configurations the search MUST cover: each must be
    /// a member of the declared space (checked by id after pruning — a
    /// missing incumbent is an error, not a silent omission), and each is
    /// measured before the searcher runs, so no budget cutoff can skip it.
    /// This is what keeps "the search optimized a menu that could not contain
    /// the config it had to beat" from happening: when every required config
    /// measures successfully, the winner beat (or is) each of them. A required
    /// config that fails to measure (a transient hiccup, a compile error) is
    /// not silently dropped — the run returns an error rather than crowning a
    /// winner that never faced it.
    ///
    /// Duplicate entries are de-duplicated; each required config is measured
    /// at most once.
    pub fn require(mut self, configs: Vec<Config>) -> Self {
        self.required.extend(configs);
        self
    }

    /// Filters candidates (Triton's `prune_configs_by`); rejected candidates
    /// are never visited. Predicates are applied when the search runs, so
    /// `.prune(..)` and `.configs(..)` compose in either order.
    ///
    /// Applies only to the library-run path ([`run`](Self::run) /
    /// [`run_with`](Self::run_with)); on the [`run_objective`](Self::run_objective)
    /// path the [`Objective`] owns its own candidate list.
    pub fn prune(mut self, keep: impl Fn(&Config) -> bool + 'static) -> Self {
        self.prune.push(Box::new(keep));
        self
    }

    /// Wall-clock budget for the whole search.
    pub fn budget(mut self, budget: Duration) -> Self {
        self.budget = Some(budget);
        self
    }

    /// Measurement options ([`BenchOptions`]: `warmup`/`rep` budgets, L2
    /// clearing).
    pub fn bench(mut self, bench: BenchOptions) -> Self {
        self.bench = bench;
        self
    }

    /// Appends every trial to a JSONL log at `path` (created if missing) and
    /// seeds the searcher from any trials already in it — which is what makes
    /// an interrupted exhaustive run resumable.
    pub fn log(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_path = Some(path.into());
        self
    }

    /// Runs the search with [`GridSearch`] (the default searcher), resuming
    /// from the trial log when one is configured.
    ///
    /// `setup` is called once per candidate. It applies the config (picks the
    /// monomorphization, builds `CompileOptions`), runs the programmer's
    /// correctness gate, and returns the closure to be timed — or an error,
    /// which records the candidate as [`TrialState::Invalid`] and moves on.
    ///
    /// After the search, the two best candidates are re-measured
    /// head-to-head with [`crate::bench::do_bench_paired`] and the winner is
    /// decided from that contemporaneous comparison — sequential (or resumed)
    /// medians never pick the winner on their own, since clock and thermal
    /// drift between measurements can exceed the margin between two good
    /// configurations. The runoff is skipped when the budget is already
    /// exhausted; a finalist that fails its runoff setup forfeits to the
    /// other; and if the paired measurement itself fails, the sequential
    /// medians decide.
    pub fn run<S, F>(mut self, stream: &Arc<Stream>, setup: S) -> Result<Output, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        self.apply_prune();
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(&self.configs),
            &self.provenance,
        )?;
        let searcher = GridSearch::new().resume(log.existing_trials());
        self.run_searcher(searcher, stream, setup, &mut log)
    }

    /// Runs the search with an explicit [`Searcher`]. The trial log still
    /// records every trial, but resume semantics are the searcher's concern.
    pub fn run_with<S, F>(
        mut self,
        searcher: impl Searcher,
        stream: &Arc<Stream>,
        setup: S,
    ) -> Result<Output, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        self.apply_prune();
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(&self.configs),
            &self.provenance,
        )?;
        self.run_searcher(searcher, stream, setup, &mut log)
    }

    /// Runs the search over a caller-implemented [`Objective`] — the
    /// engine-objective path — with [`GridSearch`], resuming from the trial
    /// log when one is configured. Every trial is appended to the log, and
    /// [`require`](Self::require)d configs are measured before the searcher
    /// runs.
    ///
    /// Unlike [`run`](Self::run), there is no paired runoff: the runoff
    /// re-times finalists with the library's bench closures, which an
    /// external objective does not expose. `Output::best` is the best
    /// sequential median; engine-scale winner selection (and its own
    /// contemporaneous re-measurement, if wanted) stays with the caller.
    pub fn run_objective(self, objective: &mut dyn Objective) -> Result<Output, Error> {
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(objective.configs()),
            &self.provenance,
        )?;
        let searcher = GridSearch::new().resume(log.existing_trials());
        self.run_objective_searcher(searcher, objective, &mut log)
    }

    /// [`run_objective`](Self::run_objective) with an explicit [`Searcher`].
    /// Logging and required-config coverage still apply; resume semantics
    /// are the searcher's concern.
    pub fn run_objective_with(
        self,
        searcher: impl Searcher,
        objective: &mut dyn Objective,
    ) -> Result<Output, Error> {
        let mut log = TrialLog::open(
            self.log_path.as_deref(),
            &self.name,
            &space_hash(objective.configs()),
            &self.provenance,
        )?;
        self.run_objective_searcher(searcher, objective, &mut log)
    }

    fn run_objective_searcher(
        self,
        mut searcher: impl Searcher,
        objective: &mut dyn Objective,
        log: &mut TrialLog,
    ) -> Result<Output, Error> {
        let required = required_indices(&self.required, objective.configs())?;
        let existing = log.existing_trials();
        let resumed: std::collections::BTreeSet<String> = existing
            .iter()
            .filter(|t| t.median_ms().is_some())
            .map(|t| t.config_id.clone())
            .collect();
        let deadline = self.budget.map(|b| Instant::now() + b);
        let mut logging = LoggingObjective {
            inner: objective,
            log,
            deadline,
        };
        let mut cache = std::collections::BTreeMap::new();
        for index in &required {
            let id = logging.configs()[*index].id.clone();
            if resumed.contains(&id) {
                // Already measured in a prior run: seed the RequiredFirst cache
                // with the resumed trial (keyed by index) so the searcher is
                // served the known-good measurement instead of re-measuring it.
                // A custom searcher may not resume from the log, and a re-measure
                // that returns Invalid would otherwise shadow the good trial in
                // merge_unclaimed, letting the winner be declared without ever
                // facing the incumbent.
                if let Some(t) = existing
                    .iter()
                    .find(|t| t.config_id == id && t.median_ms().is_some())
                {
                    cache.insert(*index, t.clone());
                }
            } else {
                cache.insert(*index, logging.measure(*index));
            }
        }
        require_measured(logging.configs(), &required, &cache)?;
        let pre_measured: Vec<Trial> = cache.values().cloned().collect();
        let mut trials = {
            let mut wrapped = RequiredFirst {
                inner: &mut logging,
                cache,
            };
            searcher.search(&mut wrapped)
        };
        merge_unclaimed(&mut trials, pre_measured);
        let best = best_config(objective.configs(), &trials).cloned();
        Ok(Output { trials, best })
    }

    fn apply_prune(&mut self) {
        let prune = std::mem::take(&mut self.prune);
        self.configs.retain(|c| prune.iter().all(|keep| keep(c)));
    }

    fn run_searcher<S, F>(
        mut self,
        mut searcher: impl Searcher,
        stream: &Arc<Stream>,
        setup: S,
        log: &mut TrialLog,
    ) -> Result<Output, Error>
    where
        S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
        F: FnMut(&Arc<Stream>) -> Result<(), Error>,
    {
        let required = std::mem::take(&mut self.required);
        let existing = log.existing_trials();
        let resumed: std::collections::BTreeSet<String> = existing
            .iter()
            .filter(|t| t.median_ms().is_some())
            .map(|t| t.config_id.clone())
            .collect();
        let mut objective = BenchObjective {
            configs: std::mem::take(&mut self.configs),
            stream: stream.clone(),
            setup,
            bench: self.bench.clone(),
            deadline: self.budget.map(|b| Instant::now() + b),
            log,
        };
        // Required configs: membership is an error to violate, and coverage
        // precedes the searcher so no budget cutoff can skip an incumbent.
        // Pre-measured trials are cache-served when the searcher visits the
        // same index, so nothing is measured (or logged) twice.
        let required = required_indices(&required, &objective.configs)?;
        let mut cache = std::collections::BTreeMap::new();
        for index in &required {
            let id = objective.configs[*index].id.clone();
            if resumed.contains(&id) {
                // Seed the RequiredFirst cache with the resumed measurement so a
                // searcher that revisits the index is served the known-good
                // trial instead of re-measuring it — a re-measure returning
                // Invalid would otherwise shadow it and drop the incumbent.
                if let Some(t) = existing
                    .iter()
                    .find(|t| t.config_id == id && t.median_ms().is_some())
                {
                    cache.insert(*index, t.clone());
                }
            } else {
                cache.insert(*index, objective.measure(*index));
            }
        }
        require_measured(&objective.configs, &required, &cache)?;
        let pre_measured: Vec<Trial> = cache.values().cloned().collect();
        let mut trials = {
            let mut wrapped = RequiredFirst {
                inner: &mut objective,
                cache,
            };
            searcher.search(&mut wrapped)
        };
        merge_unclaimed(&mut trials, pre_measured);

        // Paired runoff between the two best. Rationale in run()'s docs. An
        // exhausted budget skips it: the sequential medians decide, and no
        // further GPU work runs.
        let best = match top_two(&objective.configs, &trials) {
            None => None,
            Some((only, None)) => Some(only.clone()),
            Some((a, Some(b))) => {
                let (a, b) = (a.clone(), b.clone());
                if objective.budget_remaining() == Some(Duration::ZERO) {
                    Some(a)
                } else {
                    let result = objective.runoff(&a, &b);
                    Some(runoff_verdict(a, b, result, &mut trials, objective.log))
                }
            }
        };
        Ok(Output { trials, best })
    }
}

/// Resolves required configs to indices in `configs`, by id, de-duplicated
/// (first-occurrence order preserved). A required config absent from the space
/// is an error: the search would otherwise optimize a menu that cannot contain
/// the config it has to beat. Duplicate `.require(..)` entries collapse to one
/// index, so an incumbent is never measured (or logged) twice.
fn required_indices(required: &[Config], configs: &[Config]) -> Result<Vec<usize>, Error> {
    let mut indices = Vec::new();
    for r in required {
        let index = configs.iter().position(|c| c.id == r.id).ok_or_else(|| {
            crate::error::tensor_error(&format!(
                "required config `{}` is not in the declared space \
                     (after pruning); the search cannot cover it",
                r.id
            ))
        })?;
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    Ok(indices)
}

/// Fails the run if a freshly-measured required config did not measure. A
/// transient failure that turned a required incumbent [`TrialState::Invalid`]
/// must not be silently swallowed: the coverage guarantee (the winner beat
/// every required config) cannot hold, so the run errors rather than crowning a
/// winner that never faced it. Resumed required trials are already `Measured`
/// (filtered on median presence upstream), so only fresh entries are checked.
fn require_measured(
    configs: &[Config],
    required: &[usize],
    cache: &std::collections::BTreeMap<usize, Trial>,
) -> Result<(), Error> {
    for index in required {
        if let Some(trial) = cache.get(index) {
            if trial.median_ms().is_none() {
                let reason = match &trial.state {
                    TrialState::Invalid { reason } => reason.as_str(),
                    _ => "not measured",
                };
                return Err(crate::error::tensor_error(&format!(
                    "required config `{}` failed to measure ({reason}); \
                     cannot guarantee the winner beat it",
                    configs[*index].id
                )));
            }
        }
    }
    Ok(())
}

/// Folds pre-measured required trials the searcher never claimed into the
/// result list. A searcher may legitimately not visit a required index (an
/// exhausted budget, a non-exhaustive strategy); the coverage guarantee is
/// that the measurement HAPPENED and is reported, regardless.
fn merge_unclaimed(trials: &mut Vec<Trial>, pre_measured: Vec<Trial>) {
    for trial in pre_measured {
        match trials.iter_mut().find(|t| t.config_id == trial.config_id) {
            // A pre-measured `Measured` trial supersedes an `Invalid` the
            // searcher produced for the same config (e.g. a transient re-measure
            // failure): the pre-measurement is the coverage guarantee, so it
            // must not be shadowed by a later failed re-measure.
            Some(existing) if existing.median_ms().is_none() && trial.median_ms().is_some() => {
                *existing = trial;
            }
            Some(_) => {}
            None => trials.push(trial),
        }
    }
}

/// Objective adapter that serves pre-measured trials from a cache, so required
/// configs measured before the search are not re-measured when the searcher
/// visits them.
struct RequiredFirst<'a> {
    inner: &'a mut dyn Objective,
    cache: std::collections::BTreeMap<usize, Trial>,
}

impl Objective for RequiredFirst<'_> {
    fn configs(&self) -> &[Config] {
        self.inner.configs()
    }
    fn measure(&mut self, index: usize) -> Trial {
        match self.cache.remove(&index) {
            Some(trial) => trial,
            None => self.inner.measure(index),
        }
    }
    fn budget_remaining(&self) -> Option<Duration> {
        self.inner.budget_remaining()
    }
}

/// Objective adapter that appends every measured trial to a [`TrialLog`] — the
/// logging half of [`Autotuner::run`], hoisted so caller-implemented objectives
/// (engine-scale objectives) get the same logging and resume.
///
/// It also (a) stamps the authoritative `config_id` from the dispatched index,
/// so a caller-implemented `measure` that echoes the wrong id cannot break
/// winner selection or resume matching, and (b) folds `Autotuner::budget` into
/// the objective's own budget, so the wall-clock cap applies on the objective
/// path too.
struct LoggingObjective<'a> {
    inner: &'a mut dyn Objective,
    log: &'a mut TrialLog,
    deadline: Option<Instant>,
}

impl Objective for LoggingObjective<'_> {
    fn configs(&self) -> &[Config] {
        self.inner.configs()
    }
    fn measure(&mut self, index: usize) -> Trial {
        // Capture the authoritative id BEFORE dispatching: the library
        // dispatched `index`, so the id of the config there right now is
        // authoritative. Reading it after `measure` would stamp the wrong id if
        // the implementation reorders its config vector during the call. Don't
        // trust the implementation's echo either (a mismatch would yield no
        // winner and a resume that re-measures forever).
        let authoritative_id = self.inner.configs().get(index).map(|c| c.id.clone());
        let mut trial = self.inner.measure(index);
        if let Some(id) = authoritative_id {
            trial.config_id = id;
        }
        self.log.append(&trial);
        trial
    }
    fn budget_remaining(&self) -> Option<Duration> {
        let inner = self.inner.budget_remaining();
        let own = self
            .deadline
            .map(|d| d.saturating_duration_since(Instant::now()));
        match (own, inner) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, b) => b,
        }
    }
}

/// The two best measured candidates present in `configs`, by median.
fn top_two<'a>(
    configs: &'a [Config],
    trials: &[Trial],
) -> Option<(&'a Config, Option<&'a Config>)> {
    let mut ranked: Vec<(&Config, f32)> = Vec::new();
    for t in trials {
        let Some(ms) = t.median_ms() else { continue };
        if !ms.is_finite() {
            continue;
        }
        if let Some(c) = configs.iter().find(|c| c.id == t.config_id) {
            // Keep the best median per config (runoff entries may duplicate).
            match ranked.iter_mut().find(|(rc, _)| rc.id == c.id) {
                Some(entry) => entry.1 = entry.1.min(ms),
                None => ranked.push((c, ms)),
            }
        }
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut it = ranked.into_iter();
    let first = it.next()?.0;
    Some((first, it.next().map(|(c, _)| c)))
}

/// The library-owned objective: applies a config via the user's setup closure,
/// then times it with [`do_bench`].
struct BenchObjective<'l, S> {
    configs: Vec<Config>,
    stream: Arc<Stream>,
    setup: S,
    bench: BenchOptions,
    deadline: Option<Instant>,
    log: &'l mut TrialLog,
}

impl<S, F> Objective for BenchObjective<'_, S>
where
    S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    fn configs(&self) -> &[Config] {
        &self.configs
    }

    fn measure(&mut self, index: usize) -> Trial {
        let config = &self.configs[index];
        let state = match (self.setup)(&self.stream, config) {
            Err(e) => TrialState::Invalid {
                reason: e.to_string(),
            },
            Ok(mut f) => match do_bench(&self.stream, &self.bench, |s| f(s)) {
                Err(e) => TrialState::Invalid {
                    reason: e.to_string(),
                },
                Ok(m) => measured(&m),
            },
        };
        let trial = Trial {
            config_id: config.id.clone(),
            state,
        };
        self.log.append(&trial);
        trial
    }

    fn budget_remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}

impl<S, F> BenchObjective<'_, S>
where
    S: FnMut(&Arc<Stream>, &Config) -> Result<F, Error>,
    F: FnMut(&Arc<Stream>) -> Result<(), Error>,
{
    /// Contemporaneous A/B/A/B re-measurement of two finalists.
    fn runoff(
        &mut self,
        a: &Config,
        b: &Config,
    ) -> Result<(Measurement, Measurement), RunoffError> {
        let mut fa = (self.setup)(&self.stream, a).map_err(|error| RunoffError::Setup {
            b_failed: false,
            error,
        })?;
        let mut fb = (self.setup)(&self.stream, b).map_err(|error| RunoffError::Setup {
            b_failed: true,
            error,
        })?;
        crate::bench::do_bench_paired(&self.stream, &self.bench, |s| fa(s), |s| fb(s))
            .map_err(RunoffError::Bench)
    }
}

/// Why a runoff could not produce a verdict.
enum RunoffError {
    /// One finalist failed its runoff setup (compile, launch check, or the
    /// correctness gate): that finalist is invalid now, whatever its earlier
    /// median said.
    Setup { b_failed: bool, error: Error },
    /// The paired measurement failed after both setups succeeded —
    /// environmental, not attributable to either finalist.
    #[allow(dead_code)] // payload kept for debugging; not attributed to a config
    Bench(Error),
}

/// Decides the winner from a runoff attempt. `a` is the sequential leader:
/// it wins ties and unattributable failures. A finalist that fails its own
/// runoff setup forfeits to the other, and the failure is recorded (and
/// logged) against the finalist that actually failed.
fn runoff_verdict(
    a: Config,
    b: Config,
    result: Result<(Measurement, Measurement), RunoffError>,
    trials: &mut Vec<Trial>,
    log: &mut TrialLog,
) -> Config {
    match result {
        Err(RunoffError::Setup { b_failed, error }) => {
            let (loser, winner) = if b_failed { (b, a) } else { (a, b) };
            let t = Trial {
                config_id: loser.id.clone(),
                state: TrialState::Invalid {
                    reason: format!("runoff setup failed: {error}"),
                },
            };
            log.append(&t);
            trials.push(t);
            winner
        }
        // Neither finalist is marked Invalid: the failure cannot be pinned
        // on one of them, and the sequential medians remain the recorded
        // decision basis.
        Err(RunoffError::Bench(_)) => a,
        Ok((ma, mb)) => {
            let (oa, ob) = (measured(&ma), measured(&mb));
            for (cfg, o) in [(&a, &oa), (&b, &ob)] {
                let t = Trial {
                    config_id: cfg.id.clone(),
                    state: o.clone(),
                };
                log.append(&t);
                trials.push(t);
            }
            // An Invalid or non-finite runoff median can never win.
            let key = |o: &TrialState| match o {
                TrialState::Measured { median_ms, .. } if median_ms.is_finite() => *median_ms,
                _ => f32::INFINITY,
            };
            if key(&oa) <= key(&ob) {
                a
            } else {
                b
            }
        }
    }
}

fn measured(m: &Measurement) -> TrialState {
    if m.reps() == 0 {
        // Reachable via BenchOptions { min_reps: 0 } with a zero budget;
        // median of nothing would panic, and Objective::measure never panics.
        return TrialState::Invalid {
            reason: "no timed reps (check BenchOptions)".into(),
        };
    }
    TrialState::measured_or_invalid(m.median_ms(), m.min_ms(), m.reps())
}

// ── Trial log (JSONL) ───────────────────────────────────────────────────────

/// Append-only JSONL record of every trial, headed by an identity record;
/// parsing it back is what makes interrupted runs resumable.
///
/// The first line identifies the tuner and its search space. A log whose
/// header does not match the running search is REFUSED — resuming kernel B
/// from kernel A's log (or from a differently-shaped space with coincident
/// parameter names) silently adopts foreign timings, which is exactly the
/// under-keyed-persistence failure this module exists to prevent.
///
/// Because logging is explicitly requested (`.log(path)`), failures to open
/// or head the file are hard errors, not silent no-ops. Per-trial append
/// failures after a successful open are best-effort.
///
/// SINGLE WRITER PER PATH. Each record is one `write_all`, but the log is not
/// guarded against concurrent writers across processes: two tuners pointed at
/// the same path can both truncate a fresh log or interleave records. Give
/// each concurrent tuning run its own log file (a distinct `.log(path)`).
#[derive(Debug)]
pub struct TrialLog {
    file: Option<std::fs::File>,
    existing: Vec<Trial>,
}

/// Provenance a trial log is keyed on beyond the tuner name and search-space
/// hash: the target architecture, the kernel module's `_SOURCE_HASH`, and the
/// `tileiras --version` fingerprint — the same axes [`Workspace`] carries for
/// records. A measured trial is only comparable to a new one when all three
/// agree: a log written before a kernel edit or a toolkit change would
/// otherwise resume and let its stale timings decide `best` without
/// re-measurement.
///
/// Every field is optional so untagged callers and legacy logs keep working:
/// when BOTH the stored header and the running search name a value and they
/// differ, the resume is refused; `None` on either side skips that check
/// (with a warning when the log is the untagged side, since the caller
/// evidently cares). Fill it from a [`Workspace`] with
/// [`LogProvenance::from_workspace`] to get every check.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogProvenance {
    /// Target architecture, e.g. `"sm_120"`.
    pub arch: Option<String>,
    /// The kernel module's `_SOURCE_HASH`.
    pub source_hash: Option<String>,
    /// Output of `tileiras --version`.
    pub tileiras_fingerprint: Option<String>,
}

impl LogProvenance {
    /// Every axis of `ws`, so the trial log is checked as strictly as a
    /// [`Record`] loaded against the same workspace.
    pub fn from_workspace(ws: &Workspace) -> Self {
        Self {
            arch: Some(ws.arch.clone()),
            source_hash: Some(ws.source_hash.clone()),
            tileiras_fingerprint: Some(ws.tileiras_fingerprint.clone()),
        }
    }

    // (name, stored, running) for every axis, in the order they are reported.
    fn axes<'a>(
        &'a self,
        other: &'a LogProvenance,
    ) -> [(&'static str, &'a Option<String>, &'a Option<String>); 3] {
        [
            ("arch", &self.arch, &other.arch),
            ("source_hash", &self.source_hash, &other.source_hash),
            (
                "tileiras fingerprint",
                &self.tileiras_fingerprint,
                &other.tileiras_fingerprint,
            ),
        ]
    }
}

#[derive(Serialize, Deserialize)]
struct LogHeader {
    log_schema: u32,
    tuner: String,
    space: String,
    /// Provenance the caller supplied. `#[serde(default)]` on each field keeps
    /// logs written before it existed readable (they parse as `None`), and
    /// `None` on either side skips that axis' check (see [`LogProvenance`]).
    #[serde(default)]
    arch: Option<String>,
    #[serde(default)]
    source_hash: Option<String>,
    #[serde(default)]
    tileiras_fingerprint: Option<String>,
}

impl LogHeader {
    fn provenance(&self) -> LogProvenance {
        LogProvenance {
            arch: self.arch.clone(),
            source_hash: self.source_hash.clone(),
            tileiras_fingerprint: self.tileiras_fingerprint.clone(),
        }
    }
}

impl TrialLog {
    /// Opens (or heads) a trial log at `path` for tuner `tuner` over the
    /// space identified by `space` (see [`space_hash`]), scoped by
    /// `provenance` (arch, kernel source hash, tileiras fingerprint — each
    /// optional). `None` for `path` yields a no-op log. A log written by a
    /// different tuner or space is refused — adopting foreign timings is the
    /// under-keyed-persistence failure this module exists to prevent. Each
    /// provenance axis is refused the same way when both the stored header
    /// and `provenance` name a value and they differ; `None` on either side
    /// skips that axis (see [`LogProvenance`]).
    pub fn open(
        path: Option<&Path>,
        tuner: &str,
        space: &str,
        provenance: &LogProvenance,
    ) -> Result<Self, Error> {
        let Some(path) = path else {
            return Ok(Self {
                file: None,
                existing: Vec::new(),
            });
        };
        let expected = LogHeader {
            log_schema: 1,
            tuner: tuner.to_string(),
            space: space.to_string(),
            arch: provenance.arch.clone(),
            source_hash: provenance.source_hash.clone(),
            tileiras_fingerprint: provenance.tileiras_fingerprint.clone(),
        };
        let mut existing = Vec::new();
        let mut needs_newline = false;
        // Fresh means "no valid header on disk": a missing file, an empty
        // file, or a whitespace-only one. Fresh logs are truncated and headed;
        // anything else must present a matching header or be refused.
        let mut fresh = true;
        match std::fs::read_to_string(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(crate::error::tensor_error(&format!(
                    "trial log {} is unreadable: {e}",
                    path.display()
                )));
            }
            Ok(contents) if contents.trim().is_empty() => {}
            Ok(contents) => {
                let mut lines = contents.lines();
                let header: LogHeader = lines
                    .next()
                    .and_then(|l| serde_json::from_str(l).ok())
                    .ok_or_else(|| {
                        crate::error::tensor_error(&format!(
                            "trial log {} has no valid header; delete it or point .log() elsewhere",
                            path.display()
                        ))
                    })?;
                // Name the field that actually differs — otherwise a
                // schema-only mismatch prints identical tuner/space on both
                // sides and reads as nonsense. Arch refuses only when BOTH
                // sides name one (None on either side skips the check, so
                // legacy logs and arch-agnostic callers keep working).
                let mut diffs = Vec::new();
                if header.log_schema != expected.log_schema {
                    diffs.push(format!(
                        "log schema {} (this cutile writes {})",
                        header.log_schema, expected.log_schema
                    ));
                }
                if header.tuner != expected.tuner {
                    diffs.push(format!(
                        "tuner {:?} (running {:?})",
                        header.tuner, expected.tuner
                    ));
                }
                if header.space != expected.space {
                    diffs.push(format!(
                        "space {} (running {})",
                        header.space, expected.space
                    ));
                }
                for (axis, stored, running) in header.provenance().axes(provenance) {
                    match (stored, running) {
                        (Some(h), Some(e)) if h != e => {
                            diffs.push(format!("{axis} {h:?} (running {e:?})"));
                        }
                        // A provenance guard is only as strong as the log's
                        // first writer: an untagged log carries no value, so a
                        // caller that now supplies one gets no protection. Warn
                        // loudly rather than silently adopt possibly-foreign
                        // timings — the caller clearly cares about this axis.
                        (None, Some(e)) => {
                            eprintln!(
                                "cutile::tune: resuming trial log {} that records no {axis} \
                                 while this run has {e:?} — its timings are adopted \
                                 unchecked; re-tune from scratch if it may predate a \
                                 change on that axis",
                                path.display()
                            );
                        }
                        _ => {}
                    }
                }
                if !diffs.is_empty() {
                    return Err(crate::error::tensor_error(&format!(
                        "trial log {} belongs to a different search — {}; \
                         delete it or point .log() elsewhere",
                        path.display(),
                        diffs.join(", "),
                    )));
                }
                existing = lines
                    .enumerate()
                    .filter_map(|(i, l)| match serde_json::from_str::<Trial>(l) {
                        Ok(trial) => Some(trial),
                        Err(e) => {
                            // Skip a corrupted line rather than fail the whole
                            // resume, but say so — a silently dropped interior
                            // line forfeits that trial's resume invisibly.
                            eprintln!(
                                "cutile::tune: skipping unparseable trial-log line {} in {}: {e}",
                                i + 2, // +1 for the header line, +1 for 1-based
                                path.display(),
                            );
                            None
                        }
                    })
                    .collect();
                // A crash mid-write can leave a torn final line without a
                // newline; appending directly would corrupt the next record.
                needs_newline = !contents.ends_with('\n');
                fresh = false;
            }
        }
        let mut opts = std::fs::OpenOptions::new();
        if fresh {
            // Truncate so stray whitespace can't precede the header.
            opts.create(true).write(true).truncate(true);
        } else {
            opts.append(true);
        }
        let mut file = opts.open(path).map_err(|e| {
            crate::error::tensor_error(&format!(
                "trial log {} cannot be opened for append: {e}",
                path.display()
            ))
        })?;
        // Repairing a torn final line and heading a fresh log are part of a
        // successful open — a failure here means the log is not usable, so it
        // is an error, not a silent no-op (the doc promises this).
        if needs_newline {
            file.write_all(b"\n").map_err(|e| {
                crate::error::tensor_error(&format!(
                    "trial log {} could not be repaired for append: {e}",
                    path.display()
                ))
            })?;
        }
        if fresh {
            let line = serde_json::to_string(&expected).map_err(|e| {
                crate::error::tensor_error(&format!("trial log header is unserializable: {e}"))
            })?;
            // One write, newline included, so a header line is never torn.
            file.write_all(format!("{line}\n").as_bytes())
                .map_err(|e| {
                    crate::error::tensor_error(&format!(
                        "trial log {} could not be headed: {e}",
                        path.display()
                    ))
                })?;
        }
        Ok(Self {
            file: Some(file),
            existing,
        })
    }

    /// Trials parsed from the existing log, in file order — feed these to
    /// [`GridSearch::resume`].
    pub fn existing_trials(&self) -> Vec<Trial> {
        self.existing.clone()
    }

    /// Appends one trial. Best-effort after a successful open; a failed write
    /// is reported to stderr rather than silently dropped.
    pub fn append(&mut self, trial: &Trial) {
        if let (Some(file), Ok(line)) = (self.file.as_mut(), serde_json::to_string(trial)) {
            // Emit the record and its newline in ONE write: two writes let a
            // second process appending the same log (O_APPEND) interleave
            // between them and corrupt both records.
            if let Err(e) = file.write_all(format!("{line}\n").as_bytes()) {
                eprintln!("cutile::tune: failed to append trial to log: {e}");
            }
        }
    }
}

// ── Record ────────────────────────────────────────────────────────────────

/// On-disk format version; bump on breaking changes.
const RECORD_SCHEMA: u32 = 1;

/// A persisted, provenance-checked record of tuning winners: one entry per
/// shape-class bucket, serialized as human-diffable pretty JSON intended to
/// be committed next to the code it tunes.
///
/// Staleness is enforced at load, not documented: [`Record::load_verified`]
/// refuses entries whose provenance no longer matches the running workspace,
/// so a stale winner cannot silently apply. The strong check is the stored
/// winner's persistent-cache key: recomputed via
/// `Specialization::l2_cache_key()` (or `KernelCompiler::l2_cache_key()`),
/// it covers the serialized bytecode — dependencies included — and the
/// toolchain fingerprint, closing the known gap of `source_hash` (which
/// covers only the kernel's own module).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Record-format version; bump on breaking changes.
    pub schema: u32,
    /// Kernel (or tuner) name this record belongs to.
    pub kernel: String,
    /// The kernel module's `_SOURCE_HASH` at tune time.
    pub source_hash: String,
    /// cutile crate version at tune time.
    pub cutile_version: String,
    /// `tileiras --version` fingerprint at tune time.
    pub tileiras_fingerprint: String,
    /// Target architecture (e.g. `sm_120`). Records are per-arch; loading
    /// on a different arch is refused, never approximated.
    pub arch: String,
    /// Hostname the tuning ran on (informational).
    pub machine: String,
    /// Unix seconds at creation (informational).
    pub created_unix_secs: u64,
    /// Fingerprint of the candidate set the winners were chosen from.
    #[serde(default)]
    pub space_hash: Option<String>,
    /// Free-form revision tag for the correctness gate / objective; drift is
    /// surfaced as a warning (a stronger gate may invalidate old winners).
    #[serde(default)]
    pub gate: Option<String>,
    /// Winners, one per bucket.
    pub entries: Vec<RecordEntry>,
}

/// A stored L2 cache key and the key encoding that produced it.
///
/// The two travel as one value on purpose. A digest is comparable only
/// against another digest from the same encoding: cutile can change the key
/// preimage without touching the kernel or the toolchain, and a bare digest
/// cannot tell that migration apart from a stale winner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum L2Key {
    Tagged {
        /// [`L2_KEY_SCHEMA`] at the time the digest was computed.
        schema: u32,
        digest: String,
    },
    /// A key written before the encoding was recorded. It cannot claim an
    /// encoding, so it is never comparable.
    Untagged(String),
}

impl L2Key {
    /// Tags `digest` with the encoding this build computes keys under.
    pub fn current(digest: String) -> Self {
        Self::Tagged {
            schema: L2_KEY_SCHEMA,
            digest,
        }
    }

    /// The digest itself, discarding comparability. For display and tooling;
    /// [`comparable`](Self::comparable) is what a staleness check wants.
    pub fn digest(&self) -> &str {
        match self {
            Self::Tagged { digest, .. } | Self::Untagged(digest) => digest,
        }
    }

    /// The digest when this build would derive keys the same way, or why a
    /// comparison would be meaningless.
    pub fn comparable(&self) -> Result<&str, String> {
        match self {
            Self::Tagged { schema, digest } if *schema == L2_KEY_SCHEMA => Ok(digest),
            Self::Tagged { schema, .. } => Err(format!(
                "was computed under l2 key encoding {schema}, workspace uses {L2_KEY_SCHEMA}"
            )),
            Self::Untagged(_) => Err(format!(
                "predates l2 key encoding tags, workspace uses {L2_KEY_SCHEMA}"
            )),
        }
    }
}

/// One bucket's winner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordEntry {
    /// Shape-class bucket this winner applies to (e.g. `"tg<=512"`).
    pub bucket: String,
    /// The winning configuration.
    pub config: Config,
    /// Its measured median, milliseconds.
    pub median_ms: f32,
    /// Timed reps behind the median.
    pub samples: usize,
    /// The winner's persistent-cache (L2) key at tune time, when the caller
    /// recorded one. Enables the strong staleness check at load.
    pub l2_key: Option<L2Key>,
}

/// The provenance the loader checks a record against.
pub struct Workspace {
    /// Kernel/tuner name the caller is loading FOR. Two kernels in one
    /// module share a source hash, so this is its own refusal axis.
    pub kernel: String,
    pub source_hash: String,
    pub arch: String,
    pub tileiras_fingerprint: String,
    /// Fingerprint of the CURRENT candidate set (see [`space_hash`]); when
    /// both sides carry one, a mismatch refuses — a winner chosen from a
    /// different space is not a winner here.
    pub space_hash: Option<String>,
}

impl Record {
    /// Starts a record with the given provenance; machine name and
    /// timestamp are captured from the environment.
    pub fn new(ws: &Workspace) -> Self {
        Self {
            schema: RECORD_SCHEMA,
            kernel: ws.kernel.clone(),
            source_hash: ws.source_hash.clone(),
            cutile_version: env!("CARGO_PKG_VERSION").to_string(),
            tileiras_fingerprint: ws.tileiras_fingerprint.clone(),
            arch: ws.arch.clone(),
            machine: hostname(),
            created_unix_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            space_hash: ws.space_hash.clone(),
            gate: None,
            entries: Vec::new(),
        }
    }

    /// Inserts or replaces the winner for `bucket`. Replacement is in place,
    /// preserving entry order (committed records should diff cleanly).
    pub fn insert(&mut self, entry: RecordEntry) {
        match self.entries.iter_mut().find(|e| e.bucket == entry.bucket) {
            Some(slot) => *slot = entry,
            None => self.entries.push(entry),
        }
    }

    /// The winner for `bucket`, if recorded.
    pub fn get(&self, bucket: &str) -> Option<&RecordEntry> {
        self.entries.iter().find(|e| e.bucket == bucket)
    }

    /// Writes pretty JSON (stable field order — committed records should
    /// diff cleanly).
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::tensor_error(&format!("record serialize: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| crate::error::tensor_error(&format!("record write: {e}")))
    }

    /// Loads without verification. Prefer [`load_verified`](Self::load_verified)
    /// anywhere the entries will actually be applied.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| crate::error::tensor_error(&format!("record read: {e}")))?;
        serde_json::from_str(&contents)
            .map_err(|e| crate::error::tensor_error(&format!("record parse: {e}")))
    }

    /// Loads and verifies against the running workspace.
    ///
    /// REFUSES (errors) on: schema, kernel, arch, or `source_hash` mismatch;
    /// a `space_hash` mismatch when both sides carry one; duplicate buckets;
    /// and — when the toolchain fingerprints MATCH — a stored winner
    /// `l2_key` that differs from the recomputed one (the strong,
    /// dependency-inclusive check).
    ///
    /// WARNS (returned, never silent) on: toolchain-fingerprint drift and
    /// stored keys from another l2 key encoding (both make a recomputed key
    /// incomparable, so the check is skipped rather than read as staleness —
    /// configs remain valid, timings may not); gate tag drift; entries
    /// without a stored key; and entries whose key the verifier declined to
    /// recompute (`Ok(None)`).
    ///
    /// `verify_l2` receives each keyed entry and returns the key the CURRENT
    /// workspace derives for its config — typically
    /// `launcher.specialize()?.l2_cache_key()` or
    /// `KernelCompiler::...l2_cache_key()`.
    pub fn load_verified(
        path: &Path,
        ws: &Workspace,
        mut verify_l2: impl FnMut(&RecordEntry) -> Result<Option<String>, Error>,
    ) -> Result<(Self, Vec<String>), Error> {
        let record = Self::load(path)?;
        let refuse = |what: &str, stored: &str, current: &str| {
            Err(crate::error::tensor_error(&format!(
                "stale tuning record at {}: {what} mismatch (record: {stored}, workspace: {current}); re-tune or delete it",
                path.display(),
            )))
        };
        if record.schema != RECORD_SCHEMA {
            return refuse(
                "schema",
                &record.schema.to_string(),
                &RECORD_SCHEMA.to_string(),
            );
        }
        if record.kernel != ws.kernel {
            return refuse("kernel", &record.kernel, &ws.kernel);
        }
        if record.arch != ws.arch {
            return refuse("arch", &record.arch, &ws.arch);
        }
        if record.source_hash != ws.source_hash {
            return refuse("source_hash", &record.source_hash, &ws.source_hash);
        }
        if let (Some(stored), Some(current)) = (&record.space_hash, &ws.space_hash) {
            if stored != current {
                return refuse("space_hash", stored, current);
            }
        }
        {
            let mut seen = std::collections::BTreeSet::new();
            for e in &record.entries {
                if !seen.insert(e.bucket.as_str()) {
                    return Err(crate::error::tensor_error(&format!(
                        "tuning record at {} has duplicate entries for bucket {:?}; fix or re-tune it",
                        path.display(),
                        e.bucket,
                    )));
                }
                // A config's id is derived from its params; a stored entry
                // where the two disagree has been hand-edited or corrupted,
                // and downstream consumers key on the id.
                let derived = Config::new(e.config.params.clone()).id;
                if e.config.id != derived {
                    return refuse(
                        &format!("config id for bucket {:?}", e.bucket),
                        &e.config.id,
                        &derived,
                    );
                }
            }
        }

        let mut warnings = Vec::new();
        if record.space_hash.is_none() && ws.space_hash.is_some() {
            // The workspace records a space hash, so a record without one
            // predates the field or has had it stripped; either way the
            // same-space check silently cannot run.
            warnings.push(
                "tuning record carries no space_hash; the search-space match was not checked"
                    .to_string(),
            );
        }
        // The stored keys embed the record's fingerprint: under toolchain
        // drift recomputation CANNOT match, so the strong check is skipped
        // record-wide rather than misread as staleness. Ordering matters here.
        if record.tileiras_fingerprint != ws.tileiras_fingerprint {
            warnings.push(format!(
                "tuning record was produced by a different tileiras ({} vs {}); configs remain valid but timings may have shifted and per-entry key verification was skipped — consider re-tuning",
                record.tileiras_fingerprint, ws.tileiras_fingerprint,
            ));
        } else {
            for entry in &record.entries {
                // Three outcomes, not two: a key can match, contradict, or
                // never have been comparable in the first place.
                match entry.l2_key.as_ref().map(L2Key::comparable) {
                    None => warnings.push(format!(
                        "bucket {:?} carries no l2 key; only source-level staleness checks applied",
                        entry.bucket
                    )),
                    Some(Err(why)) => warnings.push(format!(
                        "bucket {:?}: stored l2 key {why}; the key check was skipped — configs remain valid, consider re-tuning",
                        entry.bucket
                    )),
                    Some(Ok(stored)) => match verify_l2(entry)? {
                        None => warnings.push(format!(
                            "bucket {:?}: verifier declined to recompute the l2 key; stored key not checked",
                            entry.bucket
                        )),
                        Some(current) => {
                            if current != stored {
                                return refuse(
                                    &format!("l2 key for bucket {:?}", entry.bucket),
                                    stored,
                                    &current,
                                );
                            }
                        }
                    },
                }
            }
        }
        if record.cutile_version != env!("CARGO_PKG_VERSION") {
            warnings.push(format!(
                "tuning record was produced by cutile {} (running {})",
                record.cutile_version,
                env!("CARGO_PKG_VERSION"),
            ));
        }
        Ok((record, warnings))
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(bn: i64, splits: i64) -> Config {
        Config::new([
            ("BN", ParamValue::Int(bn)),
            ("SPLITS", ParamValue::Int(splits)),
        ])
    }

    struct FakeObjective {
        configs: Vec<Config>,
        cost: fn(&Config) -> Option<f32>,
        measured: Vec<String>,
        budget: Option<Duration>,
    }

    impl Objective for FakeObjective {
        fn configs(&self) -> &[Config] {
            &self.configs
        }
        fn measure(&mut self, index: usize) -> Trial {
            let c = &self.configs[index];
            self.measured.push(c.id.clone());
            let state = match (self.cost)(c) {
                Some(ms) => TrialState::Measured {
                    median_ms: ms,
                    min_ms: ms,
                    reps: 3,
                },
                None => TrialState::Invalid {
                    reason: "gate failed".into(),
                },
            };
            Trial {
                config_id: c.id.clone(),
                state,
            }
        }
        fn budget_remaining(&self) -> Option<Duration> {
            self.budget
        }
    }

    #[test]
    fn config_ids_are_stable_and_param_order_independent() {
        let a = Config::new([("SPLITS", ParamValue::Int(8)), ("BN", ParamValue::Int(64))]);
        let b = cfg(64, 8);
        assert_eq!(a.id, b.id);
        assert_eq!(a.id, "BN=64,SPLITS=8");
        assert_eq!(a.int("BN"), Some(64));
        assert_eq!(a.int("missing"), None);
    }

    #[test]
    fn grid_search_visits_everything_once_and_picks_best() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let mut objective = FakeObjective {
            configs: configs.clone(),
            cost: |c| Some(c.int("BN").unwrap() as f32), // smaller BN = faster
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSearch::new().search(&mut objective);
        assert_eq!(objective.measured.len(), 3);
        assert_eq!(trials.len(), 3);
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(32));
    }

    #[test]
    fn invalid_candidates_are_recorded_not_fatal() {
        let configs = vec![cfg(32, 2), cfg(64, 4)];
        let mut objective = FakeObjective {
            configs: configs.clone(),
            cost: |c| (c.int("BN") != Some(32)).then_some(1.0), // 32 invalid
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSearch::new().search(&mut objective);
        assert_eq!(trials.len(), 2);
        assert!(matches!(trials[0].state, TrialState::Invalid { .. }));
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(64), "invalid one never wins");
    }

    #[test]
    fn resume_skips_known_trials() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let known = vec![Trial {
            config_id: configs[1].id.clone(),
            state: TrialState::Measured {
                median_ms: 0.5,
                min_ms: 0.5,
                reps: 3,
            },
        }];
        let mut objective = FakeObjective {
            configs: configs.clone(),
            cost: |_| Some(9.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSearch::new().resume(known).search(&mut objective);
        assert_eq!(
            objective.measured.len(),
            2,
            "known candidate not re-measured"
        );
        assert_eq!(trials.len(), 3, "known trial still in the result set");
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(64), "resumed trial can win");
    }

    #[test]
    fn exhausted_budget_stops_the_search() {
        let configs = vec![cfg(32, 2), cfg(64, 4), cfg(128, 8)];
        let mut objective = FakeObjective {
            configs,
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: Some(Duration::ZERO),
        };
        let trials = GridSearch::new().search(&mut objective);
        assert!(trials.is_empty(), "zero budget measures nothing");
    }

    fn ws() -> Workspace {
        Workspace {
            kernel: "fmha_decode".into(),
            source_hash: "abc123".into(),
            arch: "sm_120".into(),
            tileiras_fingerprint: "release 13.3, V13.3.36".into(),
            space_hash: None,
        }
    }

    fn record_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("cutile_record_{label}_{}.json", std::process::id()))
    }

    #[test]
    fn record_roundtrips_and_verifies() {
        let path = record_path("roundtrip");
        let mut a = Record::new(&ws());
        a.insert(RecordEntry {
            bucket: "tg<=512".into(),
            config: cfg(64, 8),
            median_ms: 1.25,
            samples: 12,
            l2_key: Some(L2Key::current("f".repeat(64))),
        });
        a.save(&path).unwrap();

        let (loaded, warnings) = Record::load_verified(&path, &ws(), |e| {
            Ok(e.l2_key.as_ref().map(|k| k.digest().to_string()))
        })
        .unwrap();
        assert!(warnings.is_empty());
        let entry = loaded.get("tg<=512").unwrap();
        assert_eq!(entry.config.int("BN"), Some(64));
        assert_eq!(entry.samples, 12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_refuses_source_hash_and_arch_mismatch() {
        let path = record_path("refuse");
        Record::new(&ws()).save(&path).unwrap();

        let mut other = ws();
        other.source_hash = "different".into();
        let err = Record::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("source_hash mismatch"));
        assert!(err.to_string().contains("re-tune"));

        let mut other = ws();
        other.arch = "sm_100".into();
        let err = Record::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("arch mismatch"));

        let mut other = ws();
        other.kernel = "other_kernel".into();
        let err = Record::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("kernel mismatch"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_refuses_space_mismatch_and_duplicate_buckets() {
        let path = record_path("space");
        let mut with_space = ws();
        with_space.space_hash = Some(space_hash(&[cfg(64, 8), cfg(128, 8)]));
        Record::new(&with_space).save(&path).unwrap();

        // Same kernel, different candidate set: refused when both carry one.
        let mut other = ws();
        other.space_hash = Some(space_hash(&[cfg(64, 8)]));
        let err = Record::load_verified(&path, &other, |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("space_hash mismatch"));
        // Loader without an expectation: accepted (no false refusals).
        let (_, _) = Record::load_verified(&path, &ws(), |_| Ok(None)).unwrap();

        // Duplicate buckets (hand-edited/merge-resolved record): refused.
        let mut dup = Record::new(&ws());
        for _ in 0..2 {
            dup.entries.push(RecordEntry {
                bucket: "b".into(),
                config: cfg(64, 8),
                median_ms: 1.0,
                samples: 3,
                l2_key: None,
            });
        }
        dup.save(&path).unwrap();
        let err = Record::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("duplicate entries for bucket"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_refuses_l2_key_drift_and_warns_on_fingerprint_drift() {
        let path = record_path("l2");
        let mut a = Record::new(&ws());
        a.insert(RecordEntry {
            bucket: "b".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some(L2Key::current("a".repeat(64))),
        });
        a.save(&path).unwrap();

        // Recomputed key differs => refuse (the dependency-inclusive check).
        let err = Record::load_verified(&path, &ws(), |_| Ok(Some("b".repeat(64)))).unwrap_err();
        assert!(err.to_string().contains("l2 key for bucket"));

        // Verifier declines (None) => provenance fields alone decide; a
        // fingerprint drift is a warning, not a refusal.
        let mut drifted = ws();
        drifted.tileiras_fingerprint = "release 13.4, V13.4.1".into();
        let (_, warnings) = Record::load_verified(&path, &drifted, |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("different tileiras"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn key_encoding_drift_skips_l2_verification_instead_of_refusing() {
        // A cutile upgrade can change the l2 key preimage without touching
        // the kernel or the toolchain. Such a key is not stale, it is
        // incomparable, and the strong check steps aside instead of refusing.
        let path = record_path("keyencoding");
        let mut a = Record::new(&ws());
        a.insert(RecordEntry {
            bucket: "older".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some(L2Key::Tagged {
                schema: L2_KEY_SCHEMA - 1,
                digest: "a".repeat(64),
            }),
        });
        // A key written before encodings were tagged: same treatment.
        a.insert(RecordEntry {
            bucket: "untagged".into(),
            config: cfg(128, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some(L2Key::Untagged("b".repeat(64))),
        });
        a.save(&path).unwrap();
        // Wire compatibility: an untagged key is a bare string on disk, so a
        // record written before this field existed still parses.
        let json = std::fs::read_to_string(&path).unwrap();
        assert!(json.contains(&format!("\"l2_key\": \"{}\"", "b".repeat(64))));

        let mut called = false;
        let (_, warnings) = Record::load_verified(&path, &ws(), |_| {
            called = true;
            Ok(Some("c".repeat(64))) // would refuse if it were consulted
        })
        .unwrap();
        assert!(!called, "verifier must not run on an incomparable key");
        assert!(warnings[0].contains("l2 key encoding"));
        assert!(warnings[1].contains("predates l2 key encoding tags"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn per_entry_encoding_keeps_mixed_records_honest() {
        // Entries are inserted independently, so one record can hold keys
        // from two encodings — re-tuning a single bucket does exactly that.
        // Tagging the record as a whole would mis-sentence one of them; the
        // tag rides with each key, so each is judged on its own.
        let path = record_path("mixed");
        let mut a = Record::new(&ws());
        a.insert(RecordEntry {
            bucket: "retuned".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some(L2Key::current("a".repeat(64))),
        });
        a.insert(RecordEntry {
            bucket: "carried over".into(),
            config: cfg(128, 8),
            median_ms: 2.0,
            samples: 5,
            l2_key: Some(L2Key::Untagged("b".repeat(64))),
        });
        a.save(&path).unwrap();

        let mut checked = Vec::new();
        let err = Record::load_verified(&path, &ws(), |e| {
            checked.push(e.bucket.clone());
            Ok(Some("z".repeat(64)))
        })
        .unwrap_err();
        assert_eq!(checked, ["retuned"], "only the comparable key is verified");
        assert!(err.to_string().contains("l2 key for bucket \"retuned\""));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_insert_replaces_bucket_winner() {
        let mut a = Record::new(&ws());
        for (bn, ms) in [(64, 2.0), (128, 1.0)] {
            a.insert(RecordEntry {
                bucket: "b".into(),
                config: cfg(bn, 4),
                median_ms: ms,
                samples: 3,
                l2_key: None,
            });
        }
        assert_eq!(a.entries.len(), 1, "one winner per bucket");
        assert_eq!(a.get("b").unwrap().config.int("BN"), Some(128));
    }

    #[test]
    fn record_refuses_schema_and_id_param_mismatch() {
        let path = record_path("schema");
        let mut a = Record::new(&ws());
        a.schema = RECORD_SCHEMA + 1;
        a.save(&path).unwrap();
        let err = Record::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("schema mismatch"));

        // A hand-edited entry whose id disagrees with its params: refused.
        let mut a = Record::new(&ws());
        let mut config = cfg(64, 8);
        config.id = "BN=128,SPLITS=8".into();
        a.insert(RecordEntry {
            bucket: "b".into(),
            config,
            median_ms: 1.0,
            samples: 3,
            l2_key: None,
        });
        a.save(&path).unwrap();
        let err = Record::load_verified(&path, &ws(), |_| Ok(None)).unwrap_err();
        assert!(err.to_string().contains("config id for bucket"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_without_space_hash_warns_when_workspace_expects_one() {
        let path = record_path("nospace");
        Record::new(&ws()).save(&path).unwrap(); // record: space_hash None
        let mut expecting = ws();
        expecting.space_hash = Some(space_hash(&[cfg(64, 8)]));
        let (_, warnings) = Record::load_verified(&path, &expecting, |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no space_hash"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fingerprint_drift_skips_l2_verification_instead_of_refusing() {
        // Ordering pin: under toolchain drift the recomputed key CANNOT match
        // the stored one, so a mismatching recomputation must be ignored
        // (drift warning), never misread as staleness.
        let path = record_path("driftorder");
        let mut a = Record::new(&ws());
        a.insert(RecordEntry {
            bucket: "b".into(),
            config: cfg(64, 8),
            median_ms: 1.0,
            samples: 5,
            l2_key: Some(L2Key::current("a".repeat(64))),
        });
        a.save(&path).unwrap();
        let mut drifted = ws();
        drifted.tileiras_fingerprint = "release 13.4, V13.4.1".into();
        let mut called = false;
        let (_, warnings) = Record::load_verified(&path, &drifted, |_| {
            called = true;
            Ok(Some("b".repeat(64))) // would refuse if it were consulted
        })
        .unwrap();
        assert!(!called, "verifier must not run under fingerprint drift");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("skipped"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_version_drift_warns() {
        let path = record_path("version");
        let mut a = Record::new(&ws());
        a.cutile_version = "0.0.0-elsewhere".into();
        a.save(&path).unwrap();
        let (_, warnings) = Record::load_verified(&path, &ws(), |_| Ok(None)).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("produced by cutile 0.0.0-elsewhere"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn trial_log_roundtrips_and_resumes() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_log_{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        {
            let mut log =
                TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
            log.append(&Trial {
                config_id: "BN=64".into(),
                state: TrialState::Measured {
                    median_ms: 1.5,
                    min_ms: 1.4,
                    reps: 5,
                },
            });
            log.append(&Trial {
                config_id: "BN=128".into(),
                state: TrialState::Invalid {
                    reason: "launch check".into(),
                },
            });
        }
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
        let existing = log.existing_trials();
        assert_eq!(existing.len(), 2);
        assert_eq!(existing[0].median_ms(), Some(1.5));
        assert!(existing[1].median_ms().is_none());

        // Wrong tuner or space: refused loudly, not silently adopted, and the
        // message names the field that differs.
        let err = TrialLog::open(Some(dir.as_path()), "other", "s", &LogProvenance::default())
            .unwrap_err();
        assert!(err.to_string().contains("different search"));
        assert!(err.to_string().contains("tuner"));
        let err = TrialLog::open(
            Some(dir.as_path()),
            "t",
            "different",
            &LogProvenance::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("different search"));
        assert!(err.to_string().contains("space"));

        // Torn final line: next append still yields a parseable record.
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new().append(true).open(&dir).unwrap();
            write!(f, "{{\"config_id\":\"torn").unwrap();
        }
        {
            let mut log =
                TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
            log.append(&Trial {
                config_id: "BN=256".into(),
                state: TrialState::Measured {
                    median_ms: 2.0,
                    min_ms: 2.0,
                    reps: 3,
                },
            });
        }
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
        assert_eq!(
            log.existing_trials().len(),
            3,
            "torn line dropped, new record intact"
        );
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn arch_mismatch_refuses_but_none_on_either_side_resumes() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_arch_{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);

        // Head a log tagged with arch "sm_120", with one measured trial.
        {
            let mut log =
                TrialLog::open(Some(dir.as_path()), "t", "s", &arch_only("sm_120")).unwrap();
            log.append(&Trial::measured("BN=64", 1.5, 1.4, 5));
        }

        // Reopening on a different arch is refused, and the message says so.
        let err = TrialLog::open(Some(dir.as_path()), "t", "s", &arch_only("sm_100")).unwrap_err();
        assert!(err.to_string().contains("different search"), "{err}");
        assert!(err.to_string().contains("arch"), "{err}");

        // Same arch resumes.
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &arch_only("sm_120")).unwrap();
        assert_eq!(log.existing_trials().len(), 1);

        // Caller supplies no arch: the check is skipped, so the sm_120 log
        // still resumes (arch-agnostic callers keep working).
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
        assert_eq!(log.existing_trials().len(), 1);

        // A legacy log with no arch recorded resumes even when the caller now
        // supplies one (None on the header side skips the check).
        let _ = std::fs::remove_file(&dir);
        {
            let mut log =
                TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
            log.append(&Trial::measured("BN=64", 1.5, 1.4, 5));
        }
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &arch_only("sm_100")).unwrap();
        assert_eq!(log.existing_trials().len(), 1);

        let _ = std::fs::remove_file(&dir);
    }

    fn arch_only(arch: &str) -> LogProvenance {
        LogProvenance {
            arch: Some(arch.to_string()),
            ..LogProvenance::default()
        }
    }

    #[test]
    fn source_hash_and_tileiras_mismatch_refuse_resume() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_prov_{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let tagged = LogProvenance {
            arch: Some("sm_120".into()),
            source_hash: Some("abc123".into()),
            tileiras_fingerprint: Some("release 13.3, V13.3.36".into()),
        };
        {
            let mut log = TrialLog::open(Some(dir.as_path()), "t", "s", &tagged).unwrap();
            log.append(&Trial::measured("BN=64", 1.5, 1.4, 5));
        }
        // Identical provenance resumes.
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &tagged).unwrap();
        assert_eq!(log.existing_trials().len(), 1);

        // The kernel was edited since the log was written: refused, naming the axis.
        let mut edited = tagged.clone();
        edited.source_hash = Some("def456".into());
        let err = TrialLog::open(Some(dir.as_path()), "t", "s", &edited).unwrap_err();
        assert!(err.to_string().contains("different search"), "{err}");
        assert!(err.to_string().contains("source_hash"), "{err}");

        // The toolkit changed: refused, naming the axis.
        let mut toolkit = tagged.clone();
        toolkit.tileiras_fingerprint = Some("release 13.4, V13.4.1".into());
        let err = TrialLog::open(Some(dir.as_path()), "t", "s", &toolkit).unwrap_err();
        assert!(err.to_string().contains("tileiras fingerprint"), "{err}");

        // Every differing axis is reported, not only the first.
        let mut both = edited.clone();
        both.tileiras_fingerprint = toolkit.tileiras_fingerprint.clone();
        let err = TrialLog::open(Some(dir.as_path()), "t", "s", &both).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("source_hash") && msg.contains("tileiras fingerprint"),
            "{msg}"
        );

        // An untagged caller skips every check (legacy behaviour).
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
        assert_eq!(log.existing_trials().len(), 1);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn legacy_header_without_provenance_still_resumes() {
        // A header written before source_hash / tileiras_fingerprint existed:
        // the fields deserialize as None, so a tagged caller resumes (with a
        // warning) instead of being refused.
        let dir = std::env::temp_dir().join(format!("cutile_tune_legacy_{}", std::process::id()));
        std::fs::write(&dir, "{\"log_schema\":1,\"tuner\":\"t\",\"space\":\"s\"}\n").unwrap();
        {
            let mut log =
                TrialLog::open(Some(dir.as_path()), "t", "s", &LogProvenance::default()).unwrap();
            log.append(&Trial::measured("BN=64", 1.5, 1.4, 5));
        }
        let tagged = LogProvenance {
            arch: Some("sm_120".into()),
            source_hash: Some("abc123".into()),
            tileiras_fingerprint: Some("release 13.3, V13.3.36".into()),
        };
        let log = TrialLog::open(Some(dir.as_path()), "t", "s", &tagged).unwrap();
        assert_eq!(
            log.existing_trials().len(),
            1,
            "None on the header side skips each axis"
        );
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn provenance_from_workspace_fills_every_axis() {
        let p = LogProvenance::from_workspace(&ws());
        assert_eq!(p.arch.as_deref(), Some("sm_120"));
        assert_eq!(p.source_hash.as_deref(), Some("abc123"));
        assert_eq!(
            p.tileiras_fingerprint.as_deref(),
            Some("release 13.3, V13.3.36")
        );
    }

    #[test]
    fn config_ids_do_not_alias_across_types_or_separators() {
        let int1 = Config::new([("A", ParamValue::Int(1))]);
        let str1 = Config::new([("A", ParamValue::Str("1".into()))]);
        assert_ne!(int1.id, str1.id, "int 1 and string \"1\" must differ");
        let sneaky = Config::new([("x", ParamValue::Str("1,y=2".into()))]);
        let honest = Config::new([
            ("x", ParamValue::Str("1".into())),
            ("y", ParamValue::Int(2)),
        ]);
        assert_ne!(sneaky.id, honest.id, "separator injection must not alias");
    }

    #[test]
    fn stale_resumed_trials_neither_win_nor_block_a_winner() {
        // A resumed trial for a config no longer in the space: dropped, and
        // the remaining valid winner is still selected (not None).
        let configs = vec![cfg(64, 4)];
        let stale = Trial {
            config_id: "BN=16,SPLITS=2".into(),
            state: TrialState::Measured {
                median_ms: 0.1,
                min_ms: 0.1,
                reps: 3,
            },
        };
        let mut objective = FakeObjective {
            configs: configs.clone(),
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSearch::new().resume(vec![stale]).search(&mut objective);
        assert_eq!(trials.len(), 1, "stale trial dropped from results");
        let best = best_config(&configs, &trials).expect("valid winner survives");
        assert_eq!(best.int("BN"), Some(64));
    }

    #[test]
    fn resumed_invalid_trials_are_retried() {
        let configs = vec![cfg(64, 4)];
        let invalid = Trial {
            config_id: configs[0].id.clone(),
            state: TrialState::Invalid {
                reason: "transient".into(),
            },
        };
        let mut objective = FakeObjective {
            configs: configs.clone(),
            cost: |_| Some(1.0),
            measured: Vec::new(),
            budget: None,
        };
        let trials = GridSearch::new()
            .resume(vec![invalid])
            .search(&mut objective);
        assert_eq!(objective.measured.len(), 1, "previously-Invalid retried");
        assert!(trials.iter().any(|t| t.median_ms() == Some(1.0)));
    }

    #[test]
    fn prune_composes_with_configs_in_either_order() {
        let mk = || vec![cfg(32, 2), cfg(64, 4)];
        // prune BEFORE configs — must still apply (predicates run at search).
        let mut a = Autotuner::new("t")
            .prune(|c| c.int("BN") != Some(32))
            .configs(mk());
        a.apply_prune();
        assert_eq!(a.configs.len(), 1);
        assert_eq!(a.configs[0].int("BN"), Some(64));
        // And after, identically.
        let mut b = Autotuner::new("t")
            .configs(mk())
            .prune(|c| c.int("BN") != Some(32));
        b.apply_prune();
        assert_eq!(b.configs.len(), 1);
        assert_eq!(b.configs[0].int("BN"), Some(64));
    }

    #[test]
    fn best_config_skips_non_finite_medians() {
        let configs = vec![cfg(64, 4), cfg(128, 8)];
        let trials = vec![
            Trial {
                config_id: configs[0].id.clone(),
                state: TrialState::Measured {
                    median_ms: f32::NAN,
                    min_ms: f32::NAN,
                    reps: 3,
                },
            },
            Trial {
                config_id: configs[1].id.clone(),
                state: TrialState::Measured {
                    median_ms: 2.0,
                    min_ms: 2.0,
                    reps: 3,
                },
            },
        ];
        let best = best_config(&configs, &trials).unwrap();
        assert_eq!(best.int("BN"), Some(128), "NaN never wins");
    }

    fn meas(times: &[f32]) -> Measurement {
        Measurement::from_times_ms(times.to_vec())
    }

    fn silent_log() -> TrialLog {
        TrialLog::open(None, "t", "s", &LogProvenance::default()).unwrap()
    }

    #[test]
    fn runoff_setup_failure_forfeits_to_the_other_finalist() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        for (b_failed, winner_id, loser_id) in [
            (false, b.id.clone(), a.id.clone()),
            (true, a.id.clone(), b.id.clone()),
        ] {
            let mut trials = Vec::new();
            let err = RunoffError::Setup {
                b_failed,
                error: crate::error::tensor_error("boom"),
            };
            let winner = runoff_verdict(
                a.clone(),
                b.clone(),
                Err(err),
                &mut trials,
                &mut silent_log(),
            );
            assert_eq!(winner.id, winner_id);
            assert_eq!(trials.len(), 1);
            assert_eq!(
                trials[0].config_id, loser_id,
                "failure blamed on the failing finalist"
            );
            assert!(matches!(
                &trials[0].state,
                TrialState::Invalid { reason } if reason.contains("runoff setup failed")
            ));
        }
    }

    #[test]
    fn runoff_bench_failure_keeps_the_sequential_leader() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        let mut trials = Vec::new();
        let winner = runoff_verdict(
            a.clone(),
            b,
            Err(RunoffError::Bench(crate::error::tensor_error(
                "stream died",
            ))),
            &mut trials,
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        assert!(
            trials.is_empty(),
            "unattributable failures mark nobody Invalid"
        );
    }

    #[test]
    fn non_finite_runoff_median_never_wins() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        // Under a plain `<=`, `ms_a <= NaN` is false and NaN-b would win.
        let winner = runoff_verdict(
            a.clone(),
            b.clone(),
            Ok((meas(&[2.0, 2.0, 2.0]), meas(&[f32::NAN, f32::NAN]))),
            &mut Vec::new(),
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        let winner = runoff_verdict(
            a,
            b.clone(),
            Ok((meas(&[f32::NAN]), meas(&[3.0]))),
            &mut Vec::new(),
            &mut silent_log(),
        );
        assert_eq!(winner.id, b.id);
    }

    #[test]
    fn runoff_trials_carry_real_measurement_metadata() {
        let (a, b) = (cfg(32, 2), cfg(64, 4));
        let mut trials = Vec::new();
        let winner = runoff_verdict(
            a.clone(),
            b,
            Ok((meas(&[1.0, 2.0, 3.0]), meas(&[4.0, 5.0, 6.0]))),
            &mut trials,
            &mut silent_log(),
        );
        assert_eq!(winner.id, a.id);
        assert_eq!(trials.len(), 2);
        match &trials[0].state {
            TrialState::Measured {
                median_ms,
                min_ms,
                reps,
            } => {
                assert_eq!(*median_ms, 2.0);
                assert_eq!(*min_ms, 1.0);
                assert_eq!(*reps, 3, "reps reflect the actual paired measurement");
            }
            other => panic!("expected Measured, got {other:?}"),
        }
    }

    #[test]
    fn config_ids_do_not_alias_across_key_separators() {
        // A key containing separators must not imitate two parameters.
        let smuggled = Config::new([("A=1,B", ParamValue::Int(2))]);
        let honest = Config::new([("A", ParamValue::Int(1)), ("B", ParamValue::Int(2))]);
        assert_ne!(smuggled.id, honest.id);
        // Nor can a key with literal quotes imitate an encoded one.
        let quoted = Config::new([("\"A\"", ParamValue::Int(1))]);
        let plain = Config::new([("A", ParamValue::Int(1))]);
        assert_ne!(quoted.id, plain.id);
    }

    #[test]
    fn whitespace_only_log_is_headed_and_resumable() {
        let dir = std::env::temp_dir().join(format!("cutile_tune_ws_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trials.jsonl");
        std::fs::write(&path, "\n").unwrap();
        {
            let mut log = TrialLog::open(Some(&path), "t", "s", &LogProvenance::default()).unwrap();
            log.append(&Trial {
                config_id: cfg(1, 1).id,
                state: TrialState::Invalid { reason: "x".into() },
            });
        }
        // Reopening must find a valid header, not refuse the log.
        let log = TrialLog::open(Some(&path), "t", "s", &LogProvenance::default()).unwrap();
        assert_eq!(log.existing_trials().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Objective-path API: constructors, logging, resume, required coverage ──

    /// Uniform-cost fake objective over `n` integer-parameterized configs, with
    /// per-index costs supplied through the config's own `i` parameter.
    fn indexed_objective(times: &[f32]) -> FakeObjective {
        // Cost is recovered from the config's index parameter at measure
        // time, so each candidate has a distinct, deterministic timing.
        fn cost(c: &Config) -> Option<f32> {
            c.int("i").map(|i| [3.0f32, 1.0, 2.0][i as usize])
        }
        let _ = times; // fixed table above; parameter kept for call-site clarity
        FakeObjective {
            configs: (0..3)
                .map(|i| Config::new([("i", ParamValue::Int(i))]))
                .collect(),
            cost,
            measured: Vec::new(),
            budget: None,
        }
    }

    #[test]
    fn trial_constructors_round_trip_through_serde() {
        let m = Trial::measured("c1", 1.5, 1.2, 7);
        let i = Trial::invalid("c2", "launch check failed");
        for t in [&m, &i] {
            let line = serde_json::to_string(t).unwrap();
            let back: Trial = serde_json::from_str(&line).unwrap();
            assert_eq!(back.config_id, t.config_id);
            assert_eq!(back.median_ms(), t.median_ms());
        }
        assert_eq!(m.median_ms(), Some(1.5));
        assert_eq!(i.median_ms(), None);
    }

    #[test]
    fn run_objective_logs_every_trial_and_resumes() {
        let dir =
            std::env::temp_dir().join(format!("cutile_tune_objective_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trials.jsonl");
        let _ = std::fs::remove_file(&path);

        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        let out = Autotuner::new("objective_test")
            .log(&path)
            .run_objective(&mut objective)
            .unwrap();
        assert_eq!(objective.measured.len(), 3);
        assert_eq!(out.trials.len(), 3);
        assert_eq!(out.best.as_ref().unwrap().int("i"), Some(1));

        // Second run over the same log: everything resumes, nothing measured.
        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        let out = Autotuner::new("objective_test")
            .log(&path)
            .run_objective(&mut objective)
            .unwrap();
        assert!(
            objective.measured.is_empty(),
            "resumed run must not re-measure"
        );
        assert_eq!(out.best.as_ref().unwrap().int("i"), Some(1));
    }

    #[test]
    fn require_rejects_a_config_outside_the_space() {
        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        let missing = Config::new([("i", ParamValue::Int(99))]);
        let err = match Autotuner::new("t")
            .require(vec![missing])
            .run_objective(&mut objective)
        {
            Ok(_) => panic!("a required config outside the space must error"),
            Err(err) => err,
        };
        assert!(
            format!("{err}").contains("not in the declared space"),
            "must name the coverage violation: {err}"
        );
        assert!(objective.measured.is_empty(), "no measurement on error");
    }

    #[test]
    fn required_configs_measure_first_and_only_once() {
        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        let incumbent = objective.configs[2].clone();
        let incumbent_id = incumbent.id.clone();
        let out = Autotuner::new("t")
            .require(vec![incumbent])
            .run_objective(&mut objective)
            .unwrap();
        assert_eq!(
            objective.measured[0], incumbent_id,
            "the incumbent must be visited first"
        );
        assert_eq!(objective.measured.len(), 3, "no candidate measured twice");
        assert_eq!(out.trials.len(), 3);
    }

    #[test]
    fn required_trial_survives_an_exhausted_budget() {
        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        objective.budget = Some(Duration::ZERO);
        let incumbent = objective.configs[2].clone();
        let incumbent_id = incumbent.id.clone();
        let out = Autotuner::new("t")
            .require(vec![incumbent])
            .run_objective(&mut objective)
            .unwrap();
        assert!(
            out.trials.iter().any(|t| t.config_id == incumbent_id),
            "the incumbent's trial must be reported even when the budget \
             stops the searcher: {:?}",
            out.trials
        );
    }

    #[test]
    fn duplicate_require_measures_the_incumbent_once() {
        let mut objective = indexed_objective(&[3.0, 1.0, 2.0]);
        let incumbent = objective.configs[2].clone();
        let incumbent_id = incumbent.id.clone();
        let out = Autotuner::new("t")
            .require(vec![incumbent.clone(), incumbent])
            .run_objective(&mut objective)
            .unwrap();
        assert_eq!(
            objective
                .measured
                .iter()
                .filter(|id| **id == incumbent_id)
                .count(),
            1,
            "a duplicated required config is measured only once: {:?}",
            objective.measured
        );
        assert_eq!(out.trials.len(), 3, "no duplicate trials");
    }

    #[test]
    fn required_measurement_failure_is_an_error() {
        // A required config whose measurement fails must fail the run, not
        // silently crown a winner that never faced it.
        fn cost(c: &Config) -> Option<f32> {
            c.int("i")
                .filter(|i| *i != 0)
                .map(|i| [0.0f32, 1.0, 2.0][i as usize])
        }
        let mut objective = FakeObjective {
            configs: (0..3)
                .map(|i| Config::new([("i", ParamValue::Int(i))]))
                .collect(),
            cost,
            measured: Vec::new(),
            budget: None,
        };
        let failing = objective.configs[0].clone();
        let err = Autotuner::new("t")
            .require(vec![failing])
            .run_objective(&mut objective)
            .unwrap_err();
        assert!(
            err.to_string().contains("failed to measure"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_mislabeled_config_id_is_stamped_from_the_dispatched_index() {
        // An Objective that echoes the wrong config_id must not break winner
        // selection: the library stamps the authoritative id from the index.
        struct Mislabel {
            configs: Vec<Config>,
        }
        impl Objective for Mislabel {
            fn configs(&self) -> &[Config] {
                &self.configs
            }
            fn measure(&mut self, index: usize) -> Trial {
                // Correct timing, but a bogus, non-matching id.
                Trial::measured(format!("BOGUS-{index}"), [3.0f32, 1.0, 2.0][index], 1.0, 3)
            }
            fn budget_remaining(&self) -> Option<Duration> {
                None
            }
        }
        let mut objective = Mislabel {
            configs: (0..3)
                .map(|i| Config::new([("i", ParamValue::Int(i))]))
                .collect(),
        };
        let want = objective.configs[1].id.clone(); // cost 1.0 is best
        let out = Autotuner::new("t").run_objective(&mut objective).unwrap();
        assert_eq!(
            out.best.as_ref().map(|c| c.id.clone()),
            Some(want),
            "winner should be the real best config despite mislabeled ids"
        );
        assert!(
            out.trials.iter().all(|t| !t.config_id.starts_with("BOGUS")),
            "trial ids should be stamped, not the bogus echoes: {:?}",
            out.trials
        );
    }

    #[test]
    fn a_pre_measured_trial_supersedes_a_searcher_invalid_of_the_same_config() {
        // The required-coverage backstop: on resume + a custom searcher, a
        // required incumbent is served from the cache, but if the searcher still
        // reports an Invalid for it (e.g. a transient re-measure), the known-good
        // pre-measurement must win — otherwise the winner is crowned without ever
        // facing the incumbent.
        let mut trials = vec![Trial::invalid("c", "transient re-measure")];
        let pre_measured = vec![Trial::measured("c", 1.0, 1.0, 3)];
        merge_unclaimed(&mut trials, pre_measured);
        assert_eq!(trials.len(), 1, "no duplicate trial for the same config");
        assert_eq!(
            trials[0].median_ms(),
            Some(1.0),
            "the Measured pre-measurement replaced the searcher's Invalid: {:?}",
            trials
        );
    }

    #[test]
    fn measured_with_a_non_finite_min_is_invalid() {
        // Both timings must be finite: a non-finite min serializes to `null` and
        // would be silently dropped when the log is reopened.
        assert!(
            Trial::measured("c", 1.0, f32::INFINITY, 3)
                .median_ms()
                .is_none(),
            "non-finite min must be recorded Invalid, not Measured"
        );
        assert!(
            Trial::measured("c", f32::NAN, 1.0, 3).median_ms().is_none(),
            "non-finite median must be recorded Invalid"
        );
        assert!(
            Trial::measured("c", 1.0, 0.5, 3).median_ms().is_some(),
            "finite timings remain Measured"
        );
    }
}
