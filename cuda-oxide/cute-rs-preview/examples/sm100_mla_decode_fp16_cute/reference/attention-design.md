# B200 attention architecture

This design records the initial matched FP16 implementation. See each example
README for supported profiles and validation commands.

The separate FMHA prefill and FP16 MLA decode examples target B200/SM100a.
Kernel schedules and attention mathematics remain visible in Rust through
reusable CuTe types and NVVM operations. The selected profiles match the public
CUTLASS references in precision, layout, masking, schedule, and measurement.

## Source and compiler pairing

Both examples pin public CUTLASS v4.7.0, commit
`dcf215af68a2d08d305076c152a06f201728cd53`. This matches the existing pinned
CUTLASS 4.7 compiler and the installed `nvidia-cutlass-dsl==4.7.0` runtime.
The requested files have moved into `cute/blackwell/kernel/attention/`:

| Example | Source below `examples/python/CuTeDSL/` | Local helper |
| --- | --- | --- |
| Prefill | [`cute/blackwell/kernel/attention/fmha/fmha.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py) | `helpers/fmha_helpers.py`, `helpers/__init__.py` |
| Decode | [`cute/blackwell/kernel/attention/mla/mla_decode_fp16.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py) | `cute/blackwell/kernel/attention/mla/mla_helpers.py` |

Each example keeps the original files and BSD notices unchanged in its own
`reference/` tree. `SOURCE.md` records commit and SHA-256 provenance;
`requirements.txt` pins the Python runtime dependencies. Adapters remain
outside the copied sources.

Verified source digests:

```text
fmha.py           c11801367b5681e531b756238920124ba4d2ca7809bdb4f9aacd9aaa9ae3a955
fmha_helpers.py   cbe090eb5567b7cdbd8b4944cbd579084fd60379c463bcfb011e6d4b28ef20e1
mla_decode_fp16.py 82116d97e611366d3122473ada3fde7e2550f1e06914ba1834fc448954dfd106
mla_helpers.py    13b0b43a5417be6d79369354ab90dbaddd2c49ec49c982258bf90fe2f285a2ae
```

## Programming model and initial coverage

Typed tensors, layouts, MMA configurations, copy partitions, and pipelines
form the CuTe-RS API. Instructions without CuTe
semantics can use the existing NVVM layer. Raw descriptors and instruction
bits belong in the implementation, with typed host/device contracts at the
example boundary. `cute_ext` GQA is a separate LIR model and is outside this
implementation.

Implemented initial coverage:

- Prefill: dense FP16 Q/K/V and output, FP32 QK/PV accumulation, D=128,
  fixed sequence lengths, MHA and GQA, noncausal and top-left causal masks,
  persistent scheduling, and residual sequence tiles.
- Decode: FP16 MLA with H=128, latent dimension 512, RoPE dimension 64,
  query length 1 initially, paged KV with page size 128, fixed sequence length
  and uniform split count, persistent scheduling, output and log-sum-exp.
  Test both unsplit and split execution. Page lookup and split reduction are
  part of this first implementation, not optional approximations.
- Other dtypes, additional head dimensions/query lengths, variable sequence
  lengths or split counts, sliding windows, sinks, skipped softmax/correction,
  exp2 emulation, and PDL need explicit follow-on coverage. Until implemented,
  unsupported inputs must fail validation before allocation or launch.

The scope is a matched subset of each public example, not a claim that every
upstream flag is supported. Any expansion should preserve the same requirement
for a source-matched configuration and correctness before timing.

## Separate kernel schedules

### FMHA prefill

Compute `O = softmax(Q K^T / sqrt(D)) V`, with FP32 running maxima, sums,
accumulation, and correction. Round probabilities to FP16 at the reference's
conversion point before PV. Apply masks before softmax, including residual
keys and causal coordinates; zero-filling keys is not masking. GQA maps each
query head to its corresponding KV head without duplicating the cache.

Preserve the single-CTA, 512-thread schedule for the initial D128 profile:

| Warps | Role |
| --- | --- |
| 0–3, 4–7 | Two softmax groups |
| 8–11 | Output correction |
| 12 | MMA |
| 13 | TMA loads |
| 14 | Epilogue |
| 15 | Idle/control role from the reference |

Each CTA handles two 128-row query tiles with a 128-key QK tile. QK uses
shared-memory operands; PV reads its probability operand from TMEM and V
from shared memory. Q has two stages; the aliased K/V ring has three; O has
two. These data regions total 224 KiB before metadata and alignment. Preserve
the reference's register redistribution and the twelve pipeline instances
covering TMA-to-MMA, MMA-to-thread, thread-to-MMA, and thread-to-thread traffic.

Allocate 512 TMEM columns. S starts at columns 0 and 128, O at 256 and 384,
and the packed P views at 160 and 32 alias the opposite S buffers. This aliasing
requires explicit producer/consumer ordering. A different buffer schedule may
be useful for an isolated correctness probe, but cannot be the matched
performance implementation.

### MLA decode

The latent cache supplies both K and V. Scores combine
`Q_latent C_latent^T + Q_rope C_rope^T`; use the reference's default softmax
scale `0.0416`, not a substituted square-root scale. Output scale defaults
to 1.0. FP16 inputs and probabilities feed FP32 accumulators. LSE is base-2.
The selected decode is noncausal; length masks still exclude invalid keys.

Preserve the `(2,1,1)` cluster and 384 threads per CTA:

| Warps | Role |
| --- | --- |
| 0–3 | Softmax |
| 4–7 | Correction and epilogue |
| 8 | MMA |
| 9 | TMA loads |
| 10 | Page-table loads |
| 11 | Idle/control role from the reference |

Use group-two M128 MMA, QK `(128,128,64)` and PV `(128,256,32)`.
QK processes eight latent chunks and one RoPE chunk. PV uses FP16 probabilities
in shared memory and an MN-major V operand; it is not FMHA's TMEM-source MMA.
Preserve Q1, KV15, S2, P2, correction2, O1, and page-table4 staging. The MMA
order is QK0, then QK-next/PV-current, followed by the final PV.

Allocate 512 TMEM columns: S uses the first 128, O begins at 128 and occupies
256, and correction metadata begins at 384. Match the reference's register
redistribution, inter-CTA barriers, and page-table scheduler.

For split KV, retain FP32 normalized partial outputs and base-2 partial LSE.
Merge with weights `exp2(partial_lse - global_lse)` and cast the final output
to FP16. Include the reduction launch in every timed invocation. Allocate
`B * H * Sq * splits * (512 + 1) * sizeof(f32)` workspace with checked sizes.

## Reusable API and compiler changes

Add composable SM100 operations alongside the specialized paired GEMM family.
Keep the existing M256 paired-collector contract intact; broadening its shape
assertions would retain incorrect assumptions about operands and lifetimes.

1. **MMA and storage types.** Model CTA group, M/N/K shape, element/accumulator
   precision, operand role, major order, and shared versus TMEM operand source.
   Support group-one M128 SS and TS MMA for FMHA, and group-two M128 SS MMA
   for MLA. Provide layout-aware TMEM FP32 fragment loads/stores and packed
   FP16 probability stores, plus allocation/view contracts for both groups.
2. **Layouts and TMA.** Extend typed tensor-map construction to the source's
   strided/head-aware and paged tensors, hierarchical shared layouts, and
   K-major/MN-major operands. Existing row-major 2D construction requires pitch
   to equal the swizzle span; D128 FP16 and the attention layouts exceed that
   contract. Match the reference's layout algebra and TMA coordinates rather
   than manufacturing descriptors inside example code.
3. **Arithmetic.** Expose fragment row maxima/sums, shuffles, exponentials,
   reciprocal, packed conversion, and output rescaling with the reference's
   FP32 behavior and rounding points. Keep online-softmax loops in Rust;
   do not hide a complete FMHA or MLA kernel behind one compiler intrinsic.
4. **Pipelines.** Represent producer/consumer kind, stage count, transaction
   bytes, participant counts, cluster routing, and barrier identity. Support
   independent pipelines and named thread-group synchronization; record the
   aliased-buffer handoffs. Static verification checks associations and legal
   layouts/counts. Dynamic phases, lane participation, and pointer lifetime
   remain explicit unsafe caller obligations backed by GPU tests.
5. **Verification.** The present `verify_story` in
   `dialect-cute/src/sm100_ops.rs` selects one MMA plan for a whole function,
   rejects another plan, and assumes the GEMM pipeline/epilogue lifecycle.
   The new operations verify each independent MMA plan, canonical shared
   layout, pointer address space/mutability, and intra-function CTA-group
   consistency. QK and PV can coexist without weakening the existing GEMM
   verifier. Cross-helper CTA grouping, TMEM allocation/lifetimes, and
   score/probability alias handoffs remain documented unsafe caller
   obligations, exercised by the kernel correctness and sanitizer matrix.
6. **NVVM export.** Most underlying instructions exist in `cuda-device` and
   `dialect-nvvm`; the CUTLASS exporter has narrower coverage. Extend
   `nvvm_tcgen05` beyond Shared/F16/Cg2 and one TMEM load shape, and add the
   required TMEM stores and TMA ranks/forms. Move common packed conversions
   out of the SM120-only mapping pack so full CuTe can register them without
   architecture confusion. Audit actual reference requirements before adding
   any new intrinsic catalog entry.

Changes span `cute-rs`, `dialect-cute`, `mir-importer/src/translator/cute`,
`cuda-oxide-mlir-export`, and targeted `cuda-oxide-codegen` tests. New
first-party source files use the repository NVIDIA/Apache-2.0 SPDX headers.

See the [prefill reference guide](../../sm100_fmha_prefill_cute/reference/README.md)
and [decode reference guide](README.md) for input contracts and reproduction.
The [feature-gap roadmap](attention-feature-gaps.md) tracks supported extensions.
