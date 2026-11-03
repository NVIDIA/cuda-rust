# Experiment 2: Execution Mode Overhead

**Paper section**: Section 5.2

**Claim**: cuTile Rust exposes synchronous, asynchronous, and CUDA graph
execution through one `DeviceOp` model, letting callers trade latency,
host-thread occupancy, and launch overhead without changing the kernel.

This directory contains the harnesses and checked-in data for the two
execution-mode panels in `figures/generated/`:

- `exp2_execmode_latency.pdf`: latency vs. pipeline length.
- `exp2_async_throughput.pdf`: sync vs. async throughput while overlapping
  GPU work with configurable host work.

## Layout

```text
sec5_exp2_execution_mode_overhead/
  README.md
  run.sh                         # Part A sweep + plot
  run_part_b.sh                  # Part B sweep + plot
  plot_part_a.py                 # writes exp2_execmode_latency.pdf
  plot_part_b.py                 # writes exp2_async_throughput.pdf
  launch_overhead_rust/
    Cargo.toml                   # depends on published cuTile Rust crates 0.2.0
    src/main.rs                  # Part A: execution-mode schedules
    src/bin/async_throughput.rs  # Part B: sync/async overlap
    results_part_a.csv           # checked-in Part A data
    results_part_b.csv           # checked-in Part B data
```

Both Rust binaries use a small f16 elementwise kernel:

```text
k_scale(y, x, g): y = x * g, D = 2048
```

The benchmark intentionally uses tiny kernels so host-side scheduling and
launch costs are visible.

## Part A: Pipeline Latency

`launch_overhead_rust/src/main.rs` measures host-visible latency for an
N-kernel pipeline under sync, async, and graph execution. The sync mode
is split into two measured schedules to expose synchronization cost:

| Mode | Implementation | Host wait pattern |
|---|---|---|
| `sync-individual` | Launch each `k_scale` with `.sync_on(stream)` | Wait after every kernel |
| `sync-chained` | Submit N `k_scale` ops with `.async_on(stream)` | One final `stream.synchronize()` |
| `async` | Build an N-op `.then()` chain and `.await` once | One async completion callback |
| `graph` | Capture the N-op chain once with `CudaGraph::capture` | One graph replay per iteration |

The CSV schema is:

```text
mode,d,n_ops,median_us,min_us,p25_us,p75_us,max_us,n_samples
```

The plot uses `min_us` as the line value and shades `min_us..p75_us`,
matching the paper caption.

Run the full sweep:

```bash
cd cutile-benchmarks/paper/sec5_exp2_execution_mode_overhead
./run.sh
```

Useful overrides:

```bash
N_VALUES="1 2 3 5 10 30 100 300 1000" ./run.sh
RESULTS_DIR=/tmp/cutile-paper-exp2 FIGURES_DIR=/tmp/cutile-paper-figures ./run.sh
PIN_CPU=8 ./run.sh
PY=/path/to/python ./run.sh
```

## Part B: Async Throughput Under Host Work

`launch_overhead_rust/src/bin/async_throughput.rs` measures whether async
can overlap host work with GPU work. Each iteration runs:

- GPU work: graph replay of an N-op `k_scale` pipeline, default `N=300`.
- Host work: calibrated busy-spin for `W` microseconds.

Modes:

| Mode | Pattern |
|---|---|
| `sync` | `graph.launch().sync_on(stream); spin(W)` |
| `async` | `tokio::join!(DeviceFuture::scheduled(graph.launch(), stream), spin(W))` |

The CSV schema is:

```text
mode,n_ops,w_us,tp_median,tp_min,tp_p25,tp_p75,tp_max,n_samples
```

The plot uses `tp_max` as the line value and shades `tp_p25..tp_max`,
matching the paper caption.

Run the full sweep:

```bash
cd cutile-benchmarks/paper/sec5_exp2_execution_mode_overhead
./run_part_b.sh
```

Useful overrides:

```bash
W_VALUES="0 1 3 10 30 100 300 1000 3000 10000" ./run_part_b.sh
RESULTS_DIR=/tmp/cutile-paper-exp2 FIGURES_DIR=/tmp/cutile-paper-figures ./run_part_b.sh
N_OPS=300 ./run_part_b.sh
PIN_CPU=8 ./run_part_b.sh
```

## Reproducibility Notes

- The paper identifies the execution-mode measurement platform as an
  NVIDIA GeForce RTX 5090. The paper does not specify a clock lock for this
  result.
- `../lock_clocks.sh` is retained as a local reproduction helper; its default
  lock target is 2.4 GHz.
- The Rust crate depends on the published cuTile Rust crates at version
  `0.2.0`.
- The plot scripts default to `python3`. Set `PY=/path/to/python` if the
  plotting dependencies live in another environment.
- Use `RESULTS_DIR=/tmp/...` and `FIGURES_DIR=/tmp/...` for smoke tests that
  should not rewrite the committed CSVs or generated paper figures.
- `run.sh` and `run_part_b.sh` pin the benchmark process with `taskset`
  when possible. Override with `PIN_CPU=<id>`.
- Graph timings are replay timings. Capture happens once before the hot
  loop and is not charged to each iteration.

## Current State

- [x] Part A benchmark binary (`launch_overhead`) implements all measured schedules.
- [x] Part A CSV checked in as `launch_overhead_rust/results_part_a.csv`.
- [x] Part A plot script emits `figures/generated/exp2_execmode_latency.pdf`.
- [x] Part B benchmark binary (`async_throughput`) implements sync and async.
- [x] Part B CSV checked in as `launch_overhead_rust/results_part_b.csv`.
- [x] Part B plot script emits `figures/generated/exp2_async_throughput.pdf`.
