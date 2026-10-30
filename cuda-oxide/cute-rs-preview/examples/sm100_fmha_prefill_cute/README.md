# FP16 FMHA prefill with CuTe-RS — SM100a (B200)

This cuda-oxide example computes `softmax(Q @ K.T / sqrt(128)) @ V` on
data center Blackwell GPUs, using the public CUTLASS 4.7 FMHA schedule.

The kernel uses `cute-rs` for typed layouts, structured MMA and TMA operations
(`MmaSs`, `MmaTs`, `TmaCopy`, `SharedTile::store_packed_8`), single-CTA
`PipelineTmaUmma` / `PipelineUmmaAsync` / `PipelineAsyncUmma` / `PipelineAsync`
stage rings, `TmemAllocator<512, 1>`, `TmemFragment`, and `cute_rs::arch`
forwarders (`add_packed_f32x2`, `mul_packed_f32x2`, `fma_packed_f32x2`,
`cvt_f16x2_f32`, `fence_view_async_shared`, `NamedBarrier`), while `ctm-rs`
supplies kernel macros and shared-storage declarations.
The manifest retains `cuda-device` because macro expansions still refer to
that crate internally. CuTe operations retain their CuTe dialect lowering
through `cutlass-mlir`; the primitives-only compiler profile does not cover
this example.

The measurements below validate both implementations on the current branch.
This kernel requires
SM100 hardware; the RTX 5090 / SM120 cannot execute it.

- FP16 inputs/output, FP32 accumulation, and FP16 probabilities before PV.
- Contiguous BSHD tensors, head dimension **128**, runtime batch and lengths.
- MHA or GQA (`Hq % Hkv == 0`), dense attention (default) or top-left causal
  masking. Partial query and KV tiles are supported.

From `cuda-oxide/`, once per checkout:

```bash
cargo oxide toolchain install cutlass
python3 -m venv cute-rs-preview/.venv
cute-rs-preview/.venv/bin/pip install -r cute-rs-preview/requirements.txt
```

Then:

```bash
# Compile without launching a GPU kernel
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide build sm100_fmha_prefill_cute --arch sm_100a

# Check both implementations against the independent oracle on B200
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 1,257,385,4,2 --causal --check-only

# Dense
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 1,1024,1024,8,8 --warmup 100 --iters 100 --windows 5

# Dense persistent
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 4,4096,4096,8,8 --warmup 100 --iters 100 --windows 5

# Causal grouped-query attention (GQA)
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 cargo oxide run sm100_fmha_prefill_cute --arch sm_100a -- --shape 4,4096,4096,8,2 --causal --warmup 100 --iters 100 --windows 5
```

The default runner compares Rust and pinned CuTeDSL on identical inputs and
checks both against an independent FP32 reference before timing. It finishes with:

```text
CuTeDSL: <time> us
CuTe-RS: <time> us
```

Add `--json comparison.json` to save the full report.
`--shape` is `B,Sq,Sk,Hq,Hkv`; add `--causal` for causal attention or
`--check-only` to validate without timing. Pass `--help` after `--` for options.

B200 results at `029fda8e3` (2026-10-18), CUTLASS 4.7 at optimization level 2
on both sides. All cases pass correctness checks. Each value is the median of
five window means, with **100 warmups** and 100 invocations per window, hot cache.
Times are microseconds:

| Profile (B, Sq, Sk, Hq, Hkv) | CUDA events: CuTe-RS | CUDA events: CuTeDSL | Nsight kernel: CuTe-RS | Nsight kernel: CuTeDSL |
| --- | ---: | ---: | ---: | ---: |
| Dense (1, 1024, 1024, 8, 8) | 18.552 | 73.692 | 18.682 | 18.444 |
| Dense persistent (4, 4096, 4096, 8, 8) | 279.571 | 277.812 | 276.195 | 279.437 |
| Causal GQA (4, 4096, 4096, 8, 2) | 155.356 | 170.114 | 155.967 | 167.412 |

CUDA events measure complete invocation throughput, including gaps between
launches. Nsight Systems measures GPU kernel execution intervals in a separate
profiled run, excluding those gaps but including stalls inside the kernel.
The small-case event gap largely disappears in the kernel measurements.
Profiler overhead means the two columns cannot be subtracted to isolate host
overhead. Clocks were unlocked and CuTeDSL ran first; persistent cases drift
across windows, so these medians alone do not establish a speed ranking.

- [CuTeDSL reference, setup and validation commands](reference/README.md)
- [Shared attention functionality and compiler gaps](../sm100_mla_decode_fp16_cute/reference/attention-feature-gaps.md)
- [Dedicated MLA decode example](../sm100_mla_decode_fp16_cute/README.md)
