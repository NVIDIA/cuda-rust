# Qwen implementation and validation

[Back to the model](../README.md).

This model workload implements `Qwen3-TTS-12Hz-1.7B-CustomVoice` on RTX 5090 / SM120
using `cute-rs` tensor, layout, partition and MMA operations, with CTM supplying
the underlying instructions. It includes typed Talker/predictor composition,
bounded checkpoint loading, CustomVoice prompt preparation, greedy code
generation and the GPU waveform decoder. Python supplies offline reference
fixtures and optional text tokenization; Rust executes all model tensor work.

```text
text IDs → embedding → biased linear → SiLU → biased linear
  → Talker: 28 decoder blocks → final norm → codec-0 head
  → predictor: [Talker last hidden, code-0 embedding]
      → biased 2048→1024 adapter → 5 blocks → codebook-specific head
      → repeat to obtain codes 1–15, using a fresh cache for each frame
  → sum all 16 code embeddings + next text/pad row → next Talker call
complete 16-code frames → codec embedding tables → transformer → upsampling → 24 kHz WAV
```

| Decoder contract | Talker | Predictor |
| --- | --- | --- |
| Residual / FFN width | 2048 / 6144 | 1024 / 3072 |
| Query / KV heads; head width | 16 / 8; 128 | 16 / 8; 128 |
| Q / K / V projection width | 2048 / 1024 / 1024 | 2048 / 1024 / 1024 |
| Output projection | 2048→2048 | 2048→1024 |
| Rotary policy / position limit | Interleaved `[24,20,20]` / 32768 | Standard / 65536 |
| Cache lifetime | Persists across frames | Independent per frame; 2 seed rows + 14 appends |

Each block executes input RMSNorm, Q/K/V projections, per-head Q/K RMSNorm,
RoPE and cache append, causal GQA, output projection, residual addition,
post-attention RMSNorm, SwiGLU FFN and another residual addition. Final RMSNorm
is outside the layer loop. All layer projections are bias-free; the text and
predictor input adapters apply bias in FP32 before narrowing to BF16.

`cute-rs::decoder` checks projection/norm roles, residual and head widths,
embedding extents and vocabulary bounds. `Batch<W>` owns its BF16 allocation;
`DecoderCache<H,I,K>` owns a separate cache for every layer. A failed stack call
poisons the cache until reset. The runner synchronizes before reading borrowed
buffers and on every launch-error path. These types do not identify checkpoint
contents or prove arbitrary raw-pointer/warp-participation safety.

Model/decode intermediates reuse initialized, exactly sized allocations through
exclusive leases. A context-wide completion fence precedes recycling, including
reads submitted on other streams. Checked-out outputs may outlive their runner.
Each pool retains at most 64 MiB idle storage, 4 MiB per allocation and sixteen
allocations per size; large or excess allocations are released normally.

The synchronous runner's one/two-row decoder calls submit Q/K/V together and
gate/up together, preserving typed projection roles and the original kernel order.
All launch preparation and output allocation precede submission. Each group
establishes context readiness
once and completes its stream before returning or releasing borrows, including
errors and unwinding. The pool's context-wide recycle fence remains unchanged.
Separate branded launch caches retain at most sixteen grids per projection
kernel; tensor checks still run on every call. Immutable `DeviceTables<D>` upload
the checked RoPE coefficients once per decoder call for reuse across its layers.

Source modules reuse the existing FFN and RoPE/attention examples and the shared
`examples/common/rmsnorm.rs`; they compile into this executable's own artifact.
Kernel computation is expressed through cute-rs. BF16 projections and codec
convolutions share role-typed MMA partitions derived from static thread/value
layouts. Normalization and attention use typed cyclic tensor partitions and
ordered register-tile operations; shared-memory reads use the copy layout.
CTM remains at kernel declaration/ABI boundaries and inside library instruction
implementations. The `cuda-device` dependency is required by macro expansion.

The [kernel architecture rule](../../../AGENTS.md) is mandatory. Host tests include
a source-boundary guard covering all eight GPU source modules used here. That
guard catches direct low-level instruction calls; review must also establish
that layouts and computations are meaningfully expressed through the library.

RMSNorm specializes widths 128, 1024 and 2048 with register preloads and unrolled
loops. It preserves the original per-lane FP32 accumulation, warp reduction,
division-before-epsilon rounding, approximate rsqrt and both BF16 casts. Other
widths retain the generic kernel. `--norm-specialized` compares exact bits with
that kernel across row tails, exponent ranges, epsilons and allocation guards;
`--norm-specialized-bench` measures paired launches with resident inputs/weights.

From `cuda-oxide/`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run qwen_decoder_cute --arch sm_120a
cargo test --offline --manifest-path cute-rs-preview/models/qwen_decoder_cute/Cargo.toml
cargo clippy --offline --manifest-path cute-rs-preview/models/qwen_decoder_cute/Cargo.toml --all-targets -- -D warnings
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute
```

Validation covers both real model widths, an irregular H17/I33 two-layer stack,
65-token tails, whole/chunked/single-token equivalence, cache prefix/guard
preservation, empty work, reset, capacity rejection and context failure recovery.
Independent FP64 references check each operation from its actual GPU input and
the complete graph from its original input. Per-operation tolerance is
`2e-5 + 0.008*abs(reference)`; propagated graph tolerance is
`0.002 + 0.025*abs(reference)`. These are synthetic tolerances, not measured
checkpoint acceptance thresholds.

The full-depth wiring fixture uses identity residual blocks, a nonzero predictor
input adapter, distinct codebook tables and distinct heads. Its deterministic
code sequence detects reversed seed rows and incorrect table/head selection;
two successive frames verify fresh predictor caches and persistent Talker state.
Nonzero decoder math is checked separately at the actual widths. Text tests use
a three-token vocabulary at width 2048 to avoid a large redundant fixture table;
the typed text interface defaults to the checkpoint vocabulary of 151936.

The original step 4c synthetic validation on SM120 checked 935,767
intermediate/output/cache comparisons and the full-depth wiring
fixtures pass. Maximum reference error is `1.52587891e-5`. The fresh-host-thread
allocation regression was observed failing before explicit context binding.
That synthetic suite also passed Compute Sanitizer memcheck, synccheck and
racecheck with zero errors/hazards; the existing FFN and RMSNorm GPU examples
passed after kernel reuse.

## Waveform decoder

`codec_model::Runner` owns 269 required decoder weight tensors, excluding the
encoder and the two unused quantizer input projections. The stored FP32 weights
are explicitly narrowed to BF16; codebook normalization then executes on the
GPU. Each operation borrows checked weights and owns its output `Batch<W>`.
Convolution input/output/kernel widths and linear projection roles are Rust
types. Ordinary and transposed checkpoint layouts remain distinct.

| Stage | Implementation contract |
| --- | --- |
| Codec lookup | 16 tables, vocabulary 2048, width 256; sequential BF16 residual-codebook sums; separate 256→512 projections for group 0 and groups 1–15 |
| Initial convolution | 512→1024, causal kernel 3 |
| Transformer | 8 layers, hidden 512, FFN 1024, 16 query/KV heads of width 64, causal 72-position window, RMS epsilon `1e-5` |
| First upsampling | Two factor-2 stages with ConvNeXt blocks, width 1024 |
| Waveform upsampling | Factors 8, 5, 4, 3; channels 1536→768→384→192→96; SnakeBeta and dilated residual convolutions |
| Output | One channel, clamp to `[-1,1]`, 1920 samples per code frame, 24 kHz |

The full decoder matches official 300-frame chunks with 25 frames of recomputed
left context. Positions restart for each chunk, and the duplicated waveform
prefix is cropped. Learned decoder operations run on the GPU. Host code builds
rotary tables and handles sample readback, cropping, exact BF16-to-FP32 widening
and WAV serialization.

Two observable rounding boundaries are explicit. BF16 convolution/transposed
convolution round the dot product before adding bias. The transformer's input
projection also rounds before bias because the official implementation feeds
PyTorch Linear a strided view of the convolution output. Other contiguous
linear projections use FP32 bias accumulation before BF16 narrowing. GPU
discriminators prevent a future fusion from accidentally changing these rules.

Run the pinned codec comparisons after provisioning the [official reference](README.md):

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run qwen_decoder_cute --arch sm_120a -- \
  --codec /tmp/qwen3-tts-acceptance/model /tmp/qwen3-tts-acceptance/reference-codec
```

The reference reader checks tensor shape, dtype, byte count, SHA-256 and path
containment. Checkpoint/config files are also hashed before loading. The codec
command writes JSON measurements to `/tmp/qwen-codec-metrics.json`; override
that path with `CUTE_CODEC_REPORT`. The recorded gate is relative RMS ≤ 0.05 for
every compared stage, with separate coverage flags for the attention window
and chunk boundary.

| Pinned fixture | Samples | Waveform relative RMS | Waveform SNR |
| --- | ---: | ---: | ---: |
| 2 deterministic frames | 3,840 | 0.01977 | 34.08 dB |
| 4 deterministic frames | 7,680 | 0.01216 | 38.30 dB |
| 3 generated frames | 5,760 | 0.01011 | 39.91 dB |
| 73 frames, crosses attention window | 140,160 | 0.02903 | 30.74 dB |
| 301 frames, crosses decoder chunk boundary | 577,920 | 0.03284 | 29.67 dB |

These five cases check 4,466,688 intermediate/waveform values. Quantizer output,
initial convolution and transformer input projection match the short fixtures
exactly. The full codec passes Compute Sanitizer memcheck; convolution/element
fixtures and 57,600 D64 attention outputs pass both memcheck and synccheck, with
zero reported errors. SNR here measures agreement with the reference waveform;
it is not a speech quality rating.

## Generation, sampling and measurement

Requests support greedy decoding or explicitly seeded host sampling.
`SamplingConfig<3072>` and `SamplingConfig<2048>` keep the Talker and
predictor vocabularies separate. Private validated parameters enforce finite
positive temperature, bounded top-k and top-p in `(0,1]`. Filtering follows
temperature → top-k → top-p, after Talker suppression/repetition handling.
Seeded streams are reproducible within this implementation, not bit-identical
to PyTorch's RNG. Omitting `sampling` from the request keeps greedy behavior.

Greedy requests execute the predictor's fifteen feedback steps through a CUDA
Graph by default. Set `QWEN_PREDICTOR_MODE=sync|eager|graph` to compare the original
synchronous runner, the persistent predictor plan with ordinary launches, or
graph replay. Sampled requests keep the predictor synchronous and retain its
host RNG draw order.

The predictor plan owns initialized scratch, five layer caches and an `Arc` to
checkpoint weights. Exclusive replay checks contexts, establishes readiness for
inputs produced on other streams, and verifies completion before returning owned
outputs successfully. Error and unwind paths attempt completion; cleanup cannot
report a CUDA failure from `Drop`. Graph handles are destroyed before their
buffers and modules. The existing pool's context-wide recycle fence remains.
Persistent storage and graph capture are prepared once during model loading,
outside warm request timings; reports record `predictor_prepare_seconds` as part
of `model_load_seconds`. `--predictor-graph MODEL_DIR` checks graph/eager/original
exact outputs, replay ownership, greedy selection, guards, foreign-stream
readiness and error/unwind completion against the pinned checkpoint.

After the ordinary prompt prefill, the Talker uses a private persistent eager
plan by default. `QWEN_TALKER_MODE=sync|eager` selects the original per-operator
path or one submission region across all 28 layers, final norm and codec head.
Both modes use identical CuTe computation and host sampling. The report records
`talker_execution`; per-request plan setup stays inside synthesis timing.

The plan consumes the completed typed cache, retains checkpoint ownership, and
reuses typed layer scratch on a private stream. It checks input readiness across
streams and verifies completion before returning independent hidden/logit owners;
error/unwind cleanup attempts completion. Attention
uses the actual live prefix and exact partial guard boundary at each split count.
Failures after submission poison the plan without publishing a new common prefix;
this is stronger than the original helper's head-failure publication behavior.
The public cache interfaces and pool recycle fence retain their existing rules.
The [execution walkthrough](execution.md) shows these wait boundaries before and
after the change.

`--talker-decode MODEL_DIR` compares exact hidden values, logits and full K/V
storage with the synchronous runner at prefixes 63/64/65 and 127/128/129. It also
checks distinct MRoPE axes, retained outputs, rejected inputs, delayed foreign
producers, guard poisoning, and completion on errors/unwinding. Removing the
plan's context-readiness fence makes the delayed-producer fixture fail.
Canonical/held-out `--checkpoint` checks still exercise the original runner;
they do not by themselves validate this new execution path.

```bash
/tmp/qwen3-tts-acceptance/venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/tokenize_request.py \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --text 'Hello. This is a cuda-oxide acceptance test.' \
  --speaker Vivian --sample --seed 0 --output /tmp/qwen-request.json

CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --generate /tmp/qwen3-tts-acceptance/model /tmp/qwen-request.json /tmp/qwen.wav
```

The 24 kHz mono WAV has an adjacent JSON report containing actual checkpoint
hashes, binary hash, request/policy/seeds, codec IDs, EOS/cap status, waveform
hash and timing boundaries. `--benchmark MODEL_DIR REQUEST_JSON OUTPUT_DIR 20`
retains weights across three warmups and twenty requests, requiring identical
codes and waveform hashes. The first delivered buffer is the full waveform;
incremental streaming is not implemented. The reference runners are described
in the [reference guide](README.md#reference-benchmark).

One and two row projections follow the reduction order of the reference's
cuBLAS kernel for that shape, chosen by `reference_linear.rs`: one-row shapes
use FP32 FMA chains over `k mod 16` (`k mod 32` for 1024→1024) merged by XOR
offsets, the one-row 6144→2048 and two-row 3072→1024 shapes use tensor-core
chains per K slice with FP32 slice sums in order, and the remaining two-row
shapes keep the original tensor-core chain, which already matched the reference
bit for bit. Bias is added in FP32 before the single BF16 rounding.
The tensor-core path uses aligned packed loads through CuTe partitions; a CuTe
register tile stages 32 K16 operand pairs, and its fold retains ascending MMA
accumulation and final BF16 rounding, with individual K16 steps for short
remainders.
Shapes outside the plan table retain the original implementation. The
`--decode-linear` regression checks 21,585 accuracy/bias values and 55,583
exact-order bit comparisons: FMA-chain outputs against a host emulation of the
probed order, tensor-core outputs against the original four-warp entry, and the
split-K entry with one slice against the ordinary chain, including strides,
K=16/32/48/80 tails, K496/512/528/1040 batch boundaries, varied BF16 values and
a packed-path bias rounding discriminator. It also checks 446
grouped-projection values, invalid
rows/contexts and a delayed foreign-stream producer; removing the readiness
fence makes that regression fail. `--projection-worker` replays saved inputs
through the production projection path; every saved reference activation
(2,053,120 values over all predictor and Talker decode shapes) matches exactly. The default GPU validation checks reusable
RoPE table ownership, contexts, token counts, exact rounding and cache guards.
Typed weight/bias roles and
synchronized borrows remain unchanged. Memcheck, synccheck and racecheck pass.
`--decode-linear-bench` provides event-timed repeated projections with resident
weights; these isolated measurements exclude model orchestration and allocations.

Greedy Vivian and the Ryan robot sentence pass complete-speech checks. The
greedy Ryan acceptance sentence runs to the 127-frame cap, exactly as the
official reference does with either SDPA backend; only the reference's optimized
variants finish it. The cause of the earlier Vivian repetition was the
tensor-core accumulation rounding toward zero at every MMA step, which pushed
one predictor projection output across a BF16 rounding boundary. Greedy
pass/fail on these sentences is
decided by rounding noise in all implementations, so further accuracy changes
are not a path to passing every sentence. Broad voice/style/language
quality, speaker consistency, incremental delivery and SM100 execution remain
separate acceptance work.

Contracts follow the [pinned model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/0c0e3051f131929182e2c023b9537f8b1c68adfe/config.json)
and [pinned model implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py).
The waveform graph follows the [pinned 12 Hz codec implementation](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py).
