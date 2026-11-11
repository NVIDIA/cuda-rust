# Concurrency-test audit

This audit accompanies the fixes for [#1680](https://github.com/NVIDIA/cuda-rust/issues/1680).
It covers the shared host, Tile and SIMT trees at main `d9430b0a1`, searching
Rust tests and embedded test modules for sleeps, elapsed-time assertions,
timeout-based non-completion, environment mutation, shared counters and
fixed temporary paths. Finding a pattern does not establish that every test
using it has failed. Benchmark timers and runtime polling are separate from
correctness assertions.

## Fixed in this change

| Area | Risk | Replacement |
| --- | --- | --- |
| `cuda-async/tests/drop_in_flight.rs` | A 200 ms automatic gate release and a <100 ms drop assertion race host scheduling; an unpolled-drop test also asserts <50 ms. | Explicit GPU release after drop returns, retained-owner checks before/after completion, and an execution counter for the unpolled case. |
| `cuda-core/tests/stream_query.rs` | A sleeping callback can finish before the query; latency is used as proof of nonblocking behavior. | Keep the callback gated until query returns and verify incomplete/complete state. |
| `cuda-core/tests/shared_allocation.rs` | Drop latency assertions; a callback timeout silently releases pending work. | Explicit gate release, watchdog failure, access exclusion and context-retirement checks. |
| `cuda-core/tests/device_buffer.rs` | A silent worker for 100 ms is treated as proof that initialization/free is fenced. Allocating behind a gate can itself synchronize the context. | Successful driver-fence traces in GPU unit tests plus real cross-stream readback/dependency tests. No scheduling-based proof of blocking. |
| Shared-core GPU tests in the preceding rows | Concurrent tests may run context-synchronizing allocation/setup while another test has gated work. | Serialize setup and gated regions within each test binary, including when using ordinary parallel `cargo test`. |
| `cuda-async/tests/concurrent_capture.rs` | 50/100 ms sleeps do not establish overlapping captures or stream creation during capture. | Entry/completion handshakes inside the capture scopes, with failing watchdogs. |
| `cuda-async/tests/execute_once.rs` | A 150 ms sleep claims to guarantee contention. | Keep barrier-started integration stress tests; add a unit test observing the actual `Running` wait branch before settling the producer. |
| `cuda-async/tests/reactor_correctness.rs` | GPU-duration assumptions, early waker completion, per-test environment mutation and guessed recycle/park delays. | Gates for pending/waker tests; explicit device completion after cancellation; remove process environment mutation. Keep exact interleaving claims in model tests. |
| `cutile/tests/gpu/submission_lifetimes.rs` | Redundant one-second latency limits; sleep-based backend warmup; timeout release can be overlooked. | Gated warmup, pending/owner observations, and acknowledgement that release was explicit. |
| `cuda-host/tests/shared_launch_lifetimes.rs` | A silently expired callback can invalidate the access-lifetime experiment. | Require acknowledgement of explicit release; retain serialization and access checks. |
| `cuda-artifact-finalizer/src/ptx.rs` | Releasing an executable's write descriptor after 30 ms may miss the retry path or exhaust retries under load. | Release after three observed, real ETXTBSY failures using a thread-local test hook. |

The hooks are compiled only under `cfg(test)`; there is no new public API or
production instrumentation. The GPU-only unit tests are explicitly included
in `cutile-rs/scripts/run_gpu_tests.sh`, with filters that exclude ignored
performance benchmarks. The CPU suite continues to skip them.

Three Tile fixtures (`autotune`, `foreign_calls`, and
`generated_source_locations`) also used PID-only temporary paths. Interrupted
runs and PID reuse could leave stale files or make directory creation fail.
They now exclusively create a fresh directory and remove it on unwind.

## Remaining broader risks

These are **not repaired or validated by this change**. They are recorded here
so #1680 remains open rather than implying that the whole repository is free
of scheduling-sensitive tests.

| Location | Follow-up |
| --- | --- |
| `cuda-oxide/cute-rs-preview/models/qwen_decoder_cute/src/model.rs`, `decode_linear_grouped.rs` | Manual/ignored GPU fixtures infer blocked return/projection from 100 ms without a message. They need an observation hook at the pool-completion/group-completion boundary, plus a negative control that removes that fence. A worker-start message alone is insufficient. |
| The same model's `talker_decode_validation.rs`, `predictor_graph_validation.rs` | Delayed foreign producers use 100 ms sleeps and silently bounded callback gates. Establish the consumer's readiness fence before releasing the producer, and fail on watchdog release. Validate the model on its supported GPU. |
| `cuda-oxide/cute-rs-preview/examples/ffn_cute`, `rope_cache_cute` | Manual validation uses 250 ms delayed producers. Exact numerical checks remain useful, but the intended interleaving is not guaranteed. Replace with readiness-boundary hooks and rerun the examples. |
| GPU lifetime/context tests generally | Context/module/allocator setup may synchronize even without explicit sleeps. Warm setup before gating and isolate contexts or serialize tests that share one. This change covers the identified shared-core binaries, not every GPU example. |
| Cached compilation and environment tests | Process-wide cache counters, default tile libraries and environment settings require serialization or subprocess isolation. Existing `cache_test_lock` and isolated environment tests should remain; new tests must participate rather than assume a clean cache. |

`cuda-host/src/ltoir_cache.rs` already uses lock-arrival handshakes from #1716;
its old sleep must not be restored. `warmup` timings are diagnostics, and
`warmup_bench`, launch-overhead and reactor A/B timings are benchmarks. They
are not latency assertions in a correctness suite. Bounded eventual-cleanup
polling (including allocation retirement and device-buffer leak checks) remains:
the observed cleanup state decides success, and the deadline only bounds a hang.

## Rules for future tests

- Observe return while GPU work is still explicitly gated; then release and
  verify eventual cleanup. Always release on unwinding.
- A watchdog expiry must fail, even if releasing the gate lets later checks pass.
- To prove blocking, observe the actual fence/wait boundary. Absence of a message
  proves nothing about whether another thread has run.
- Use handshakes for required interleavings and model tests for internal races.
  Barrier-started stress tests are useful, but do not guarantee contention.
- Keep compilation/allocation/module setup outside a gated region. Retain pinned
  gates until completion; freeing pinned memory can synchronize the context.
- Validate that deliberately removing a fence, duplicating execution or blocking
  a supposed nonblocking operation makes its test fail. Do not just widen a
  latency threshold or retry until green.
