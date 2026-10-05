# ptx-schedule

Structural PTX schedule analysis and deterministic perturbation.

A kernel that races only under one interleaving passes every run until it does
not. This crate makes that interleaving reachable on purpose: it finds the
points in a PTX module where thread progress is observable, then inserts a
seeded `nanosleep.u32` at a subset of them so the same seed always produces
the same rewrite.

## What the analyzer deliberately does not do

`analyze_ptx` and `perturb_ptx` own neither CUDA execution nor an input
generator: they read PTX text and write PTX text. Building, launching,
watchdogging and verdict assignment live one layer up in `campaign.rs`
(described below), and the kernel inputs stay whatever the example already
uses. Static site discovery, mutation and triage therefore all read the same
source model, so a site named in a finding is the site the analyzer found.

## The model

`analyze_ptx` turns a module into an ordered list of `ScheduleSite`s -- the
places where another thread's progress can be observed:

| `SiteKind` | what it marks |
|:-----------|:--------------|
| `Atomic` | an atomic read-modify-write |
| `Reduction` | a `red.*` reduction with no returned value |
| `Barrier` | `bar.*` / `barrier.*` / `mbarrier.*` synchronization |
| `Fence` | `fence.*` / `membar.*` ordering |
| `OrderedMemory` | a load or store carrying an explicit ordering qualifier (`.volatile`, `.acquire`, `.relaxed`, ...) |
| `WarpCollective` | `shfl`, `vote`, `match`, `redux` and friends |
| `AsyncProxy` | the asynchronous proxy pipeline: `cp.async.*`, `cp.reduce.async.*`, `wgmma.*`, `tcgen05.*` and `clusterlaunchcontrol.*` -- the bulk-copy, bulk-reduce and matrix issues and the commit/wait pairs that order them |
| `GridDependency` | programmatic dependent-launch ordering: `griddepcontrol.launch_dependents` / `griddepcontrol.wait` |
| `TensorMapMutation` | `tensormap.replace.*` generic-proxy tensor-map descriptor mutation, published by `fence.proxy.tensormap::*` |
| `Backedge` | a `bra` back to the same or an earlier block -- a conservative loop detector |

Each site keeps its ordinal, enclosing callable, byte span, the instruction
text and any guarding predicate (a back-edge also records its block index),
so a rewrite can be described without re-parsing.

`perturb_ptx` then applies `InjectionOptions` and returns the rewritten module
plus an `InjectionDecision` per site:

- `seed` -- the whole rewrite is a deterministic function of it;
- `intensity` -- a probability dial, not a fraction: each site is touched
  with chance `0.75 × intensity`, and the same dial scales how long the
  inserted sleeps run;
- `max_sleep_ns` -- the delay ceiling (default `DEFAULT_MAX_SLEEP_NS`, 64 µs);
- `focus` -- an optional substring matched against each site's opcode and
  instruction text. Matching sites become very likely (`0.95 × intensity`),
  everything else very unlikely (`0.15 × intensity`), so a sweep leans toward
  one area without excluding the rest.

Nothing is inserted when the draw selects no site, and a seed that changes
nothing is reported rather than silently producing the original text.

## Structure

```text
src/
├── lib.rs       # site discovery (analyze_ptx) and injection (perturb_ptx)
├── campaign.rs  # the seed-sweep driver: build, run, watchdog, confirm
└── main.rs      # single-file CLI over one .ptx
```

## Campaign verdicts

`campaign::run_campaign` sweeps a seed range and classifies each run. A
finding is re-run (`confirm_runs` times in total) and reported with how many
of those runs reproduced it, so a one-off shows up as a one-off instead of
being mistaken for a reproducible schedule bug:

| `RunKind` | meaning |
|:----------|:--------|
| `Pass` | the perturbed build behaved like the baseline |
| `Skipped` | the example declined to run and printed a skip marker |
| `Hang` | the watchdog fired |
| `Crash` | the process died |
| `Mismatch` | the example reported its own failure |
| `OutputChanged` | stdout differed with no explicit failure marker (opt-in) |
| `GpuWedged` | the device stopped responding |
| `HarnessError` | the campaign itself failed, not the kernel |

A skipped baseline runs no seeds and exits successfully, including with
`--fail-on-finding`. That flag fails only when a campaign finds a real variant.

## Consumers

| Crate | Uses it for |
|:------|:------------|
| `cargo-oxide` | `cargo oxide fuzz-schedule <example>`, the user-facing campaign |

The crate also ships a `ptx-schedule` binary for one PTX file at a time:

```bash
ptx-schedule kernel.ptx --list-sites
ptx-schedule kernel.ptx --seed 7 --intensity 0.5 -o perturbed.ptx \
    --decisions-json decisions.json
```

## License

Apache-2.0. See [LICENSE](https://github.com/NVlabs/cuda-oxide/blob/main/LICENSE).
