# Autotuning

`cutile::tune` searches a declared space of kernel configurations, measures
each candidate with `cutile::bench`'s device-event timing, and persists both
the trials and the winner. It is gated behind the `experimental-tune`
feature, and the API may change between releases:

```toml
cutile = { version = "...", features = ["experimental-tune"] }
```

Use a closure to tune an individual kernel, or implement `Objective` to
measure a whole request. These synchronous interfaces and their resume logs
remain supported. The opt-in preparation pipeline below overlaps CPU
compilation with foreground GPU evaluation.

## Finding a good configuration under a time budget

Use Bayesian optimization (BO) with lookahead compilation when a full sweep
would take too long. BO uses measurements from this run to choose useful next
experiments; lookahead compiles future candidates while the GPU evaluates the
current one. With a budget shorter than a full sweep, this explores only part
of the declared space. The result is a tested candidate, not proof of the global
optimum. A small space may still be exhausted within the budget.

From `cutile-rs/`, try the RMSNorm example with a 30-second tuning budget and
one preparation worker (the default lookahead):

```sh
cargo run --release -p cutile-examples --example autotune --features experimental-tune -- --search bayes --workers 1 --budget-ms 30000 --seed 42
```

To try more parallel compilation, change `--workers 1` to `--workers 4`.
This bounds the number of preparing or ready candidates, not concurrent GPU
measurements. More lookahead can hide slow compilation, but those candidates
are chosen before the latest results arrive and some may never be measured.
Compare with `--serial` and with `--search random` under the same budget.

For your own workload, configure the pipeline and selector together:

```rust
use std::time::Duration;
use cutile::tune::pipeline::{BayesianOptions, BayesianSearch, Pipeline};

let mut strategy = BayesianSearch::new(&configs, BayesianOptions::default())?;
let pipeline = Pipeline::default()
    .workers(1)
    .budget(Duration::from_secs(30))
    .trial_limit(Duration::from_secs(10));
// Supply resolve/prepare/evaluate and paired-confirmation callbacks to
// pipeline.run_confirmed(...), as described below and in the example.
```

With one worker, B compiles while A is evaluated. A's result then informs the
choice of C, which compiles while B is evaluated. The strategy accounts for
pending candidates so it does not suggest the same experiment twice.

Use `run_confirmed` to compare finalists with a known incumbent; its default
reserves 25% of the same budget for fresh paired measurements. If confirmation
cannot finish, only an alternative that already completed every required round
can replace the incumbent. Otherwise `NotConfirmed` keeps the incumbent and does not
certify a replacement. Budgeting is cooperative: it stops new work and bounds
controlled stages, but an already-running GPU operation or user callback can
overrun. Application build time and the example's fixture setup are outside
the tuning budget.

## Preparing candidates in parallel

The `autotune` example checks RMSNorm outputs against a CPU reference and
verifies that its first launch reuses the prepared binary. Run it from the
`cutile-rs/` workspace:

```sh
cargo run --release -p cutile-examples --example autotune --features experimental-tune
# Compare more CPU workers with the same measurement protocol:
cargo run --release -p cutile-examples --example autotune --features experimental-tune -- --workers 4 --budget-ms 30000
cargo run --release -p cutile-examples --example autotune --features experimental-tune -- --serial
# Compare selection with the same worker count, budget and confirmation:
cargo run --release -p cutile-examples --example autotune --features experimental-tune -- --search random --seed 42
cargo run --release -p cutile-examples --example autotune --features experimental-tune -- --search grid
```

`tune::pipeline::Pipeline` owns scheduling. Its three callbacks have separate
roles:

1. `resolve` runs on the foreground thread. Resolve the actual launch metadata
   with `.specialize_on(&stream)`, then call `.into_compile_job()` to capture
   the target and compiler settings. This step can query the device and execute
   input operations; do not move it into a preparation worker.
2. `prepare` runs on CPU workers. `job.compile(control)` runs the Rust frontend
   and assembler, producing a host-owned `PreparedKernel`. It does not load a
   CUDA module or synchronize a CUDA stream.
3. `evaluate` runs on the foreground thread, one candidate at a time. Check
   the artifact's key against the actual launch, call `prepared.install(control)`,
   run a correctness check, then measure with `bench::do_bench_controlled`.

Explicit installation is optional: an ordinary launch of the same specialization
joins an in-progress CPU compilation and loads the resulting binary itself.
Keep the `PreparedKernel` handle alive until that launch if compilation has
already finished. The example installs explicitly so it can separately check
that the subsequent launch leaves the installation counter unchanged.

Preparation and ordinary cache-miss launches share the same CPU compilation
registry. Whichever begins compilation first produces the binary; other callers
wait without compiling it again. Loading still occurs on a foreground caller.
Cancelling a waiter does not cancel another caller's compilation. If the owner
is cancelled or panics, a waiting caller with a live budget can retry; compiler
errors are reported and later attempts remain retryable. Registry entries and
CPU artifacts disappear when their last caller releases them. This deduplicates
overlapping work; it does not add a permanent CPU binary cache.

One worker is the default: while A is evaluated, B compiles; after observing
A, the strategy selects C while B is evaluated. With `workers(k)`, at most k
candidates are preparing or ready in addition to the current evaluation.
More workers can improve throughput but also perturb short-kernel timings;
measure this tradeoff on your workload. `overlap(false)` supplies a sequential
comparison with the same callbacks.

`Strategy::suggest` sees completed observations, every pending index
(including the current evaluation), and remaining time. `observe` receives
each outcome. `RandomOrder::new(seed).with_initial([baseline_index])` provides
seeded sampling without replacement, exhausting the space when the budget
allows. Create a fresh strategy per run. Initial indices are suggestions,
not a guarantee of incumbent coverage. `GridOrder` visits the declared order.
The example defaults to the Bayesian selector described below; the synchronous
`Autotuner` still defaults to `GridSearch`.

The budget includes resolution, queueing, preparation, loading, correctness
checks and measurement. `trial_limit` gives resolution, preparation and evaluation
each a fresh active-stage allowance, always capped by the overall run deadline.
Time waiting in the scheduler's own queues counts against the run budget, but
not the stage limit; waiting inside a callback, including a remote service queue,
still consumes that callback's active-stage allowance.
The example's fixture and CPU-reference setup happen before the run clock
starts. Expired deadlines stop dispatch and exclude late measurements from
scoring. The assembler process can be killed and reaped; frontend passes and
benchmark repetitions check cancellation between stages. Arbitrary callbacks,
CUDA calls and running GPU kernels cannot be forcibly interrupted. Returning
waits for outstanding workers, so this is a cooperative deadline, not a hard
wall-clock guarantee. Interrupted assembler runs remove their bytecode and
temporary intermediates after reaping the process; ordinary compiler failures
still retain the bytecode named in their diagnostic.

Explicit `Failure::invalid` outcomes and source-located frontend rejections
(`JITError::Located`, converted through `Failure::from`) mark a candidate invalid.
Unclassified compiler/driver errors still stop the run. Return a
measured `Trial` on success: legacy invalid trials also represent non-finite
timings, so the pipeline treats them as infrastructure errors. Timeouts and
cancellations are unscored. Infrastructure errors remain `Infrastructure` and
stop the run even if a deadline or cancellation happened first, including when
collecting results from draining workers. Other late failures are classified as
`TimedOut` or `Cancelled`, with their original diagnostic retained in
`Failure::reason`. Neither outcome becomes a usable measurement. Generic
compiler/driver errors default to infrastructure errors because they may reflect
the environment rather than the candidate. The caller can
classify a known candidate-specific rejection explicitly. Observations include
separate stage durations; an unvisited stage is `None`, and overlapping stage
times must not be summed as elapsed wall time. Evaluation time includes loading,
checks and measurement, rather than pretending those costs were measured separately.

**`run().best_observed` is provisional.** Use `run_confirmed` below, or independently
compare finalists with your incumbent before applying a result. An incomplete
search has no best observed result. The synchronous resume-log schema is unchanged;
the pipeline does not read or write it. Store observations with workload, source
and toolchain provenance if you need an audit record.

Prepared artifacts are bounded by the pipeline, but installed modules retain
the existing global cache lifetime. Automatic eviction is deliberately avoided:
other streams may still be using a specialization. See cache management below.

## Bayesian selection with a neutral starting point

`BayesianSearch` fits a small Gaussian process to **current-run log latencies**
and scores every unvisited configuration by expected improvement. It can spend
a measurement on a promising uncertain region rather than simply trying the
lowest predicted latency. This is intended for small finite spaces with costly
evaluations, not millions of candidates. Fitting is cubic in the number of
observations, and scoring the finite list adds work for each candidate. Model
work is included in the budget and checks the deadline between fit rows and
candidate scores.

```rust
use cutile::tune::pipeline::{BayesianOptions, BayesianSearch, ParameterScale};

let options = BayesianOptions {
    seed: 42,
    scales: [("BLOCK_SIZE".into(), ParameterScale::Log2)].into(),
    ..Default::default()
};
let mut strategy = BayesianSearch::new(&configs, options)?.with_initial([0])?;
// Pass &mut strategy to Pipeline::run_confirmed as below.
```

Without `with_initial` or explicit prior offsets, every configuration has the
same prior mean and marginal uncertainty; a seeded tie-break selects the first
candidate. Integer axes default to linear normalization, positive tile sizes
can opt into logarithmic scaling, and strings are categorical. Integer category
IDs can also use `Categorical`. Every configuration must declare the same keys
and parameter types. The Matérn 5/2 kernel, length scale and observation noise
are fixed explicit settings in `BayesianOptions`, rather than hyperparameters
fit from a handful of samples. These defaults are starting assumptions, not a
claim that this model is best for every tuning space.

Optional `prior_log_offsets` give weak preferences by configuration ID. For
example, `-0.1` encodes an expected latency about 10% below the common baseline.
Offsets are limited to magnitude `ln(2)`, leave initial uncertainty unchanged,
and never add observations. Hardware-informed hints must be supplied explicitly;
the tuner does not invent architecture rules, call an LLM, or train offline.
The default remains neutral. If using previous tuning runs to construct hints
or initial candidate suggestions, verify the kernel, parameter meanings,
workload, GPU and compiler first. Do not insert old timings into current-run
observations. Automatic history retrieval and cross-kernel transfer are not
implemented, and resume provenance checks are unchanged.

Pending configurations are temporarily conditioned at their predicted means,
reducing redundant exploration while another candidate is compiling or being
measured. They cannot improve the measured incumbent. Expected improvement uses
the best posterior mean among valid measured points; this is a plug-in treatment
of noise, not an integrated noisy-EI algorithm. Invalid and timed-out points use
the same predicted-mean conditioning to reduce repeated exploration around
failures. This reduces acquisition uncertainty without changing the predicted
mean or incumbent; it is a search heuristic, not evidence of a measured latency
or a learned probability of validity. Cancelled and infrastructure-failed points
receive no such penalty. All failed trials remain excluded as latency targets.
Known rejected candidate indices are excluded; the first implementation does not
predict the validity of unseen configurations. A numerical failure falls back to seeded random and
increments `numerical_fallbacks()`.

The implementation uses standard GP regression equations with Cholesky solves,
checked against independent scikit-learn/SciPy fixtures. It adds no ML runtime
dependency. Compare it with seeded random and grid under the **same elapsed-time
budget**, including compilation, correctness checks and confirmation. Small
spaces that fit the budget can be exhausted by any strategy; there is no claim
that a model always wins. The existing `Strategy` interface also permits a
different selector without changing compilation or measurement.

## Screening and confirming a replacement

The RMSNorm example uses `Pipeline::run_confirmed` with
`ConfirmationOptions::new(incumbent_index)`. It requires a total `budget` and
adds a fourth, foreground callback to the preparation pipeline:

```rust
let policy = ConfirmationOptions::new(0);
let result = pipeline.run_confirmed(
    &configs, &mut strategy, &policy,
    resolve, prepare,
    |config, prepared, bench_options, control| { /* check, screen -> Trial */ },
    |incumbent, candidate, bench_options, control| {
        // Check the incumbent normally: an error here must stop confirmation.
        // Mark only candidate-specific invalidity with
        // .map_err(Failure::for_candidate), then return aligned measurements
        // from bench::do_bench_paired_controlled(...).
    },
)?;
```

The default policy screens with at most 50 timed repetitions and reserves 25% of the
total budget for confirmation. Screening may finish sooner if the declared space
is exhausted. After **all preparation workers have joined and queued artifacts
have been dropped**, it compares the four best measured alternatives with the
incumbent in three fresh rounds. The incumbent is compared even if it was never
screened. Both arms must pass fresh correctness checks in the callback, and a
known invalid incumbent is not used. Attribute candidate-only failures with
`.map_err(Failure::for_candidate)` around that arm's checks: this converts
`Invalid` into `InvalidCandidate`, without relabeling infrastructure errors or
interruptions. A generic `Invalid` from the paired callback remains ambiguous
and stops confirmation because it may concern the incumbent. Never wrap the
whole comparison or incumbent checks with `for_candidate`.

No background preparation runs during these comparisons; callers must also keep
unrelated GPU work away from the device.

`do_bench_paired_controlled` alternates A/B then B/A ordering, with aligned raw
samples and cancellation/deadline checks between arms. A comparison cannot return
an unmatched pair as a successful measurement. The pipeline's `trial_limit`
applies to each paired comparison, including its correctness checks. Confirmation
uses the original overall deadline, not a new budget. As with screening, an
uninterruptible callback or GPU operation can still overrun it.

For each pair the improvement is `1 - candidate_time / incumbent_time`. By
default, the lower quartile of those improvements must exceed 2% in **every**
round. Among qualifying candidates, the result selects the largest worst-round
median improvement. This is a conservative, configurable repeatability heuristic,
not a confidence interval or a universal 2% noise threshold. Calibrate the margin,
repetitions and round count for your kernels and hardware; the defaults do not
guarantee an optimal choice. Finalist order reverses on alternate rounds.

`ConfirmedOutput` includes screening observations, raw paired samples, failures,
the incumbent, the selected configuration and one of three statuses:

- `Improved`: a replacement passed the configured confirmation rule.
- `NoImprovement`: every shortlisted alternative completed its comparisons, but
  none passed the improvement rule.
- `NotConfirmed`: insufficient valid candidates, failed checks, cancellation or
  insufficient time. The selected configuration remains the supplied incumbent;
  this status does **not** certify that the incumbent passed validation.

Every selected replacement must complete all of its own required rounds, with
all correctness checks passing. An `InvalidCandidate` or candidate-local timeout
disqualifies only that candidate. Other finalists can continue, and a later
budget/cancellation cutoff can retain the best already fully confirmed candidate;
a partial leader never wins. `ConfirmedOutput.failure` may therefore contain a
timeout/cancellation diagnostic alongside `Improved`; per-candidate errors remain
in `confirmations`. Infrastructure errors and incumbent/ambiguous correctness
errors stop confirmation without recommending a replacement, even after another
candidate completed its rounds.

Previous screening times and model predictions never count as fresh confirmation
evidence. A questionable correctness test should be investigated separately;
these rules do not waive failed tests or select by correctness pass percentage.

To save the example's observations and raw confirmation samples, pass
`--evidence NEW_FILE`. It creates a new JSON file without overwriting an existing
one. It includes the strategy settings and declared configurations. Record source
revision, GPU and toolkit alongside it; this example
file is evidence, not a provenance-checked resume log.

## Declaring a space and running the search

A configuration is a set of named integer or string parameters; the space
is the list of configurations. Only `configs` and the terminal `run` are
required. The rest of the builder is optional: `prune` filters candidates,
`budget` bounds wall-clock, `log` makes the run resumable, and `require`
declares incumbents the search must cover.

```rust
let configs: Vec<Config> = [16i64, 32, 64, 128]
    .into_iter()
    .map(|bn| Config::new([("BN", ParamValue::Int(bn))]))
    .collect();

let output = Autotuner::new("fmha_decode")
    .configs(configs)
    .prune(|c| c.int("BN").unwrap() <= pp)
    .require(vec![shipping_config])
    .budget(Duration::from_secs(300))
    .arch("sm_100") // record the arch; a resume from another arch is refused
    .log("fmha_decode.trials.jsonl")
    .run(&stream, |stream, config| {
        // The setup closure is yours: read the config, build the kernel's
        // composed launcher for it, run it once as a correctness gate, and
        // return the closure to be timed. Err(..) records the candidate as
        // invalid and the search continues.
        let bn = config.int("BN").unwrap();
        let mut launch = move |stream: &Arc<Stream>| {
            fmha_decode(out.partition([1, bn as i32]), &q, &k, &v)
                .generics(generics_for(bn))
                .sync_on(stream)
        };
        launch(&stream)?;
        Ok(launch)
    })?;
let best: Config = output.best.expect("a winner");
```

The two best candidates are compared in a paired A/B runoff to account for
clock and temperature drift. A finalist that fails setup forfeits.

Every `require`d configuration must belong to the search space. These
configurations are measured first, before the budget can cut off the search.
The winner must match or beat each of them.

Trial logs record the tuner name and a hash of the search space. A log from
a different tuner or space is rejected.

(engine-scale-measurement-implements-objective)=
## Measuring a full request with `Objective`

When the objective is end-to-end (tokens per second, request latency), the
library cannot own the launch. Implement `Objective` instead: `configs()`
exposes the space, `measure(index)` applies one candidate at engine scope
and returns a `Trial` built with `Trial::measured` or `Trial::invalid`, and
`budget_remaining()` reports what is left. `Autotuner::run_objective` then
provides the same trial logging, resume, and `require` coverage as the
closure path:

```rust
let mut objective = EngineObjective::new(engine, configs);
let output = Autotuner::new("engine_prefill")
    .arch(current_arch) // e.g. "sm_100"; resume refuses a foreign-arch log
    .log("prefill.trials.jsonl")
    .require(vec![current_default])
    .run_objective(&mut objective)?;
```

This path has no paired runoff: the library cannot re-time an engine
objective through its bench closures. `Output::best` is the best sequential
median. The caller can re-measure finalists together to check for drift.

Serialized `Trial` records remain compatible across cuTile upgrades. An
incompatible change increments the log schema version, and `TrialLog::open`
rejects the old schema.


## Committing winners

A `tune::Record` persists winners so production loads them instead of
re-searching: one JSON file per kernel, written by `save` and committed
next to the code it tunes. The file holds a provenance header (kernel
name, source hash, cutile version, `tileiras` fingerprint, architecture,
search-space hash) and one entry per shape-class bucket.

```rust
use cutile::tune::{L2Key, Record, RecordEntry, Workspace};

// After tuning: `winner` is the winning Trial, `best` its Config
// (`output.best`), and `launcher` the kernel's composed builder applied
// with `best`'s values — the same call expression production dispatches.
let TrialState::Measured { median_ms, reps, .. } = winner.state else {
    unreachable!("the winner was measured");
};
let mut record = Record::new(&workspace);
record.insert(RecordEntry {
    bucket: "tg<=512".into(),
    config: best,
    median_ms,
    samples: reps,
    l2_key: Some(L2Key::current(launcher.l2_cache_key()?)),
});
record.save(&path)?;

// In production: load, verified against the running workspace.
let (record, warnings) = Record::load_verified(&path, &workspace, |entry| {
    Ok(Some(specialize_for(&entry.config)?.l2_cache_key()?))
})?;
let entry = record.get("tg<=512");
```

A bucket is a plain string label, matched exactly by `Record::get`; the
predicate that decides which bucket a runtime shape falls into lives in
the consumer's dispatch code, and the record only stores the label.
`median_ms` and `samples` are the winner's measured median latency and
the number of timed reps behind it — provenance for a human reading the
committed file, not inputs to verification. The `l2_key` is the winner's
persistent-cache key, computed from the composed builder without
compiling or launching (`.l2_cache_key()`).

`load_verified` rejects a record if its kernel, architecture, source hash,
or search space differs from the running workspace. It also recomputes each
entry's cache key to check the kernel's dependencies and toolchain. Changes
that affect only timing produce warnings.

## Warming and managing the kernel cache during sweeps

Call `.compile()` on the same builder used for `.execute` or `.sync_on` to
warm the specialization that will run. `api::meta` tensors supply shape and
dtype without device allocation or kernel launches:

```rust
my_module::my_kernel(api::meta::<f32>(&[64, 64]).sync()?.partition([16, 16]), ...)
    .generics(generics)
    .compile_options(opts)
    .compile()?;
```

Tuning sweeps and long-running serving engines can accumulate many
specializations, and each cached kernel holds device memory. The in-memory
kernel cache is intentionally unbounded. The `unsafe` functions
`clear_kernel_cache()`, `evict_kernel(&key)`, and `retain_kernels(pred)` in
`cutile::tile_kernel` are available without any Cargo feature. They remove
entries, releasing each module's device memory when its last holder drops.
These functions are `unsafe`: synchronize every stream that may still be
running a cached kernel before evicting it. Between tuning trials or while
a serving engine is idle,
callers can evict and then reload the needed specializations. The autotuning
API in `cutile::tune` still requires `experimental-tune`.
