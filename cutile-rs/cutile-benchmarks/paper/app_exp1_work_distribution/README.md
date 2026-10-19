# Appendix Experiment 1 — GPU Work Distribution (bimodal GEMMs)

Evaluates §5.2 Part B2 of the paper: whether one async host thread can keep
multiple CUDA streams busy with mixed-size GEMMs.

## Claim under test

Given a queue of GEMMs of heterogeneous size (mix of small and large),
driven across $S$ CUDA streams on a single NVIDIA GeForce RTX 5090, three execution
strategies should behave as follows:

- **(A) Serial**: one host thread, one stream. Drains the queue
  strictly in order. Establishes a single-stream throughput ceiling.
- **(B) Threaded**: $S$ host threads, each owning one stream.
  Synchronous `.sync_on()` on each thread. Represents what you build
  *without* async — multi-stream concurrency via OS threads.
- **(C) Async**: one host thread running a `tokio` current-thread
  runtime, $S$ `spawn_local`'d tasks, each owning one stream. Async
  dispatch via `DeviceFuture::scheduled(op, ExecutionContext(stream))`
  so launches stay on the intended stream.

The paper tests whether async can match or beat threaded throughput while
using one host thread.

<a id="why-this-matters"></a>
## Workload motivation

Inference workloads mix small decodes with large prefills and varying batch
sizes. This experiment measures dispatch across streams on one host thread.
It does not evaluate a serving scheduler.

## Directory layout

```
app_exp1_work_distribution/
  README.md                        (this file)
  run_bimodal.sh                   (sweeps S; runs all three modes)
  plot_bimodal.py                  (renders figure)
  bimodal_gemms_rust/              (Rust benchmark crate)
    Cargo.toml
    .cargo/config.toml             (CUDA toolkit paths)
    src/main.rs                    (kernels, buffers, three modes, driver)
    results_bimodal.csv            (appended by each run)
```

## Kernel & workload

- **Kernel**: unchecked GEMM from cuTile Rust, tile block `(BM,BN,BK)
  = (128, 128, 64)`, K dimension runtime-valued so one kernel handles
  all sizes.
- **Workload unit**: one GEMM of Small (`M=N=K=256`) or Large
  (`M=N=K=2048`) size.
- **Mix**: deterministic interleaving of Small and Large controlled by
  `--small-ratio` (default 0.8 — inference-shaped, small-dominant).
- **Size counts**: `--n-work` units per sample (default 2000).

## Design notes

**Per-worker, pre-allocated buffers.** Each worker owns one set of
buffers sized for Small and one for Large. Buffers are reused across
work units via the `Option<Partition>` take/replace pattern so the
measurement is pure dispatch + kernel time, no allocation.

**Persistent worker lifecycle across samples.** `DEVICE_CONTEXTS` in
cuda-async is `thread_local!`, which means every fresh OS thread
starts with an empty kernel JIT cache. Without persistent workers,
Mode B would pay 100s of ms of JIT per sample. The benchmark spawns
the workers once and reuses them via command/done channels:

- Mode B: `std::sync::mpsc` channels between main and worker threads.
- Mode C: `tokio::sync::mpsc` channels between the main task and
  `spawn_local`'d worker tasks.

**Context binding.** In Mode B each worker calls
`stream.device().bind_to_thread()` on startup, binding the CUDA
context to that OS thread. Without this, driver calls serialize
against whichever thread currently owns the context.

**Stream creation in Mode C.** `Device` is constructed in `main()` and cloned
to each async worker. Each worker creates its own stream on the async thread.
Creating streams on the outer thread and moving them into tasks was slower
on this hardware.

**Stream binding in Mode C.** Each `DeviceOp` is awaited via
`DeviceFuture::scheduled(op, ExecutionContext::new(stream))` so
launches land on the worker's dedicated stream instead of the
default `StreamPoolRoundRobin` policy.

## Running

From this directory:

```bash
# Single config
cd bimodal_gemms_rust
cargo run --release --bin bimodal_gemms -- \
    --mode async --streams 8 --n-work 1000 \
    --samples 5 --warmup

# Full sweep (all modes × several S values, writes results_bimodal.csv)
cd ..
./run_bimodal.sh

# Non-destructive smoke/rerun output
RESULTS_DIR=/tmp/cutile-paper-workdist FIGURES_DIR=/tmp/cutile-paper-figures ./run_bimodal.sh

# Render figure
python3 plot_bimodal.py
```

<a id="known-characteristics-observed-so-far"></a>
## Results

- **Async scales with $S$** up to the GPU's stream-concurrency
  ceiling. On NVIDIA GeForce RTX 5090 with the workload above, async reaches
  ~77.6k WU/s at $S=8$ (~2.1× the single-stream serial baseline of
  ~36.9k WU/s).
- **Threaded also reaches the same ceiling** on this workload. The checked-in
  sweep reports ~77.4k WU/s at $S=8$. At moderate stream counts
  ($S=2$--$4$), thread-per-stream is ahead of single-thread async; by the
  stream-concurrency ceiling, async catches up while using one host thread.
- **Serial** is the flat baseline (single-stream throughput ceiling
  with zero scheduling overhead). Async at $S=1$ is slower than serial
  due to the async callback tax.

## Interpretation for the paper

Async reaches the same GPU throughput ceiling as thread-per-stream using one
host thread. Thread-per-stream is faster at moderate stream counts.
