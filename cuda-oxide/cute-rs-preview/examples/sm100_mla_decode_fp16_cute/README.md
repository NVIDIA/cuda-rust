# FP16 MLA decode with CuTe-RS — SM100a (B200)

This cuda-oxide example ports the public CUTLASS FP16 MLA decode kernel to
CuTe-RS for data center Blackwell. Two-CTA clusters process a paged latent KV
cache; a second kernel combines partial results when multiple splits are used.

The kernel uses `cute-rs` for typed layouts, structured MMA and TMA operations,
and `ctm-rs` for instruction primitives, kernel macros and storage types.
Packed FP32 registers have a distinct type; asynchronous page copies carry
a shared destination and a ready barrier.
The manifest retains `cuda-device` because macro expansions still refer to
that crate internally. CuTe operations retain their CuTe dialect lowering
through `cutlass-mlir`; the primitives-only compiler profile does not cover
this example.

Direct instruction calls and workload-local fragment handling remain legacy
migration work under the [CuTe kernel architecture](../../AGENTS.md). Tensor
computation belongs in reusable `cute-rs` operations.

The linked CTM migration reports record B200 validation. The measurements below
also validate both implementations on the current branch. This kernel requires
SM100 hardware; the RTX 5090 / SM120 cannot execute it.

- **128 heads**, one query, latent dimension **512**, RoPE dimension **64**,
  and **128-token pages**; the latent cache supplies both K and V.
- FP16 inputs, probabilities and output; FP32 QK/PV accumulation and base-2 LSE.
- Positive batch and uniform KV lengths, **1–256 splits**, permuted pages and
  partial final pages. Attention is noncausal with explicit length masking.
- Softmax scale defaults to **0.0416** and output scale to **1.0**.

From `cuda-oxide/`, with CUTLASS 4.7 installed (`cargo oxide toolchain install cutlass`):

```bash
# Compile without launching a GPU kernel
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide build sm100_mla_decode_fp16_cute --arch sm_100a

# Check both implementations against the independent oracle on B200
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 1 --kv-length 1024 --splits 1 --clusters 1 --check-only

# Small, unsplit decode
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 1 --kv-length 1024 --splits 1 --clusters 1 --warmup 300 --iters 100 --windows 5

# Batched, persistent split-KV decode
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_mla_decode_fp16_cute --arch sm_100a -- --batch 16 --kv-length 8192 --splits 8 --clusters 74 --warmup 300 --iters 100 --windows 5
```

Install the [Python reference dependencies](reference/README.md) before running.
The default runner compares Rust and pinned CuTeDSL on identical inputs and
checks both against an independent FP64 oracle before timing. It finishes with:

```text
CuTeDSL: <time> us
CuTe-RS: <time> us
```

Add `--json comparison.json` to save the full report or `--check-only` to
validate without timing. Pass `--help` after `--` for options.

B200 results at `029fda8e3` (2026-10-18), CUTLASS 4.7 at optimization level 2
on both sides. The CuTeDSL reference is the unmodified, pinned upstream kernel;
both implementations pass correctness checks. Each value is the median of five
window means, with **300 warmups** and 100 invocations per window, hot cache.
Times are microseconds:

| Batch / KV length / splits / clusters | CUDA events: CuTe-RS | CUDA events: CuTeDSL | Nsight kernels: CuTe-RS | Nsight kernels: CuTeDSL |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1024 / 1 / 1 | 16.769 | 27.827 | 16.959 | 18.093 |
| 16 / 8192 / 8 / 74 | 64.211 | 52.636 | 59.148 | 61.988 |

Both measurements include the split-reduction kernel when enabled. CUDA events
include gaps between launches; Nsight Systems sums the main and reduction GPU
execution durations, excluding those gaps. Kernel durations still include
device stalls. The profiler runs separately, so subtracting columns does not
isolate host overhead.

Clocks were unlocked and CuTeDSL ran first. Persistent decode drifted across
windows: Nsight kernel sums ranged **48.297–62.647 µs** for CuTe-RS and
**49.036–64.389 µs** for CuTeDSL. These medians do not establish a stable speed
ranking. Small CuTeDSL event windows also varied from **18.132–40.883 µs**.
[Nsight methodology, component timings and reproduction](reference/nsight-measurements.md)
retain the full evidence. Earlier measurements using both timing orders and
different warmups remain in the [historical notes](reference/measurement-notes.md).

- [CTM migration: 9 passing B200 cases, sanitizers and compiler comparison](reference/results/ctm-migration/summary.json)
- [CuTeDSL reference and setup](reference/README.md)
- [Validation, profiling, benchmark commands, and detailed results](reference/measurement-notes.md)
- [Historical correctness results](reference/results/optimization-b200/validation.json) · [Benchmark evidence](reference/results/optimization-b200/summary.json)
- [Shared attention feature-gap roadmap](reference/attention-feature-gaps.md)
