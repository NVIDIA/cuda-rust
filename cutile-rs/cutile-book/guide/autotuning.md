# Autotuning

`cutile::tune` searches a declared space of kernel configurations, measures
each candidate with `cutile::bench`'s device-event timing, and persists both
the trials and the winner. It is gated behind the `experimental-tune`
feature, and the API may change between releases:

```toml
cutile = { version = "...", features = ["experimental-tune"] }
```

Use a closure to tune an individual kernel, or implement `Objective` to
measure a whole request. The `autotune` example uses a closure:
`cargo run -p cutile-examples --example autotune --features experimental-tune`.

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
