# One Talker decode step

[Back to the model](../README.md). See [benchmarking](benchmark.md) for measured
results and [implementation and validation](implementation.md) for full contracts.

After prompt prefill, each Talker call consumes one BF16 row with 2,048 values.
Its 28 layers append one token to their K/V caches. Final normalization and the
codec head produce a hidden row and 3,072 logits; the host then selects the next
audio code.

## Where the CPU waits

The synchronous runner completes each operator or projection group before
returning its output to the next helper. This schematic omits individual kernels
and shows the repeated readiness and completion boundaries:

```text
CPU:  ready → submit norm → wait → ready → submit Q/K/V → wait → …
GPU:             norm                         Q/K/V            …
                        repeat through 28 layers, final norm and head
```

The persistent Talker plan owns the cache, scratch and private stream for the
request. It can submit dependent operations in stream order while the GPU is
still working:

```text
CPU:  input readiness wait
          │
          └─► submit layer 0 → layer 1 → … → layer 27 → final norm → head
                                                                         │
                                                   step completion wait ◄─┘
                                                             │
                                                   read guards, publish prefix
                                                             │
                                                   return hidden row and logits

GPU:       layer 0 ─────────► layer 1 ──► … ──► layer 27 ──► norm ──► head
```

The arrows show ordering, not overlapping layer computation. There is no host
wait between submitted layers. Input readiness still covers producers on other
streams; completion and guard readback still precede success. The context-wide
pool recycle fence also remains. Returned outputs own separate buffers, so the
next call can reuse scratch without changing earlier results. If submission or
completion fails, the plan is poisoned and the old cache prefix stays published;
cleanup attempts to finish pending work before owners drop.

Prompt prefill uses the synchronous runner. `QWEN_TALKER_MODE=sync` also selects
that runner for decode; `eager` selects the persistent plan and is the default.
The predictor independently uses its existing CUDA Graph for greedy requests.

## The two measured changes

| Change | What it changes | Warm synthesis p50 |
| --- | --- | ---: |
| Ordered projection prefetch | Stages 32 K16 operand pairs in a CuTe register tile, giving the compiler more loads to schedule | 0.922 → 0.693 s |
| Persistent Talker execution | Reuses owned scratch and submits the complete layer stack before waiting | 0.694 → 0.568 s |

These are consecutive paired experiments on RTX 5090 / SM120a, using the same
request and preserving its exact 3.92-second output. Each retains the original
MMA order and BF16 rounding boundaries. Separate profiled requests show projection
time falling from 580.4 to 353.0 ms in the first change. In the second, total
stream/context waits across the request fall from 68,378 to 3,528 and gaps without
recorded GPU activity fall from 211.0 to 73.8 ms. Profile intervals include
instrumentation and differ from the unprofiled synthesis timing boundary.

The latest reference p50 is 0.549 s for different, 3.76-second audio. These
measurements cover one request; the known Vivian greedy repetition failure and
broader quality/performance acceptance remain open. Exact evidence and reproduction
commands are in [the benchmark guide](benchmark.md), with raw measurements in
[projection-prefetch-results.json](projection-prefetch-results.json) and
[talker-execution-results.json](talker-execution-results.json).

## Follow the code

- [`talker_decode.rs`](../src/talker_decode.rs): `TalkerDecodePlan::run_into`
  checks readiness, submits all layers, waits for completion and publishes the
  cache prefix.
- [`decoder_execution.rs`](../src/decoder_execution.rs): `LayerScratch` owns
  reusable intermediates; `Modules::enqueue_layer` submits the shared layer
  schedule used by Talker and predictor execution.
- [`ffn_cute/src/kernel.rs`](../../../examples/ffn_cute/src/kernel.rs): the
  earlier projection-prefetch change stages operand tiles inside each projection
  kernel. This changes device load scheduling; the Talker change controls host
  submission and waits.
