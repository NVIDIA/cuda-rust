# Pinned Qwen correctness reference

See [implementation and validation](implementation.md) for Rust model contracts
and diagnostic commands, [the execution walkthrough](execution.md) for the
Talker's submission and wait boundaries, and the
[reference benchmark](#reference-benchmark) below for speed comparisons.

`export_reference.py` runs the official model and writes inputs, intermediate
results, caches, logits and generated codec IDs for the Rust implementation to
check. Python is only an offline correctness oracle; Rust inference does not call
it. The exporter does not download, install or update anything.

| Artifact | Pin |
| --- | --- |
| Official Qwen3-TTS source | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | `0c0e3051f131929182e2c023b9537f8b1c68adfe` |
| Source archive SHA-256 | `c7129dc1399dac840ccd99ef62ebef82849378b9a5486f553b1264e59e1012e1` |
| Main weights | 3,833,402,552 bytes; 404 BF16 tensors |
| Speech tokenizer weights | 682,293,092 bytes; 496 FP32 tensors, loaded as BF16 |

The script verifies all Python source files and pinned model/config/tokenizer
artifact hashes before loading the model. Weight files and generated fixtures
stay outside Git.

`artifacts.json` records the verified sizes and SHA-256 values for the eight
model/config/tokenizer files. Rust generation and benchmarking check these
actual local bytes once before loading the model; hashing is outside warm
timings. Checkpoint and codec comparisons verify their fixture's artifact
identities and record the actual hashes in their reports. A shape-compatible
checkpoint with different bytes is rejected.

## Run

The local acceptance environment is isolated at
`/tmp/qwen3-tts-acceptance/venv`. It reuses the existing CUDA-enabled PyTorch
installation without changing that installation. It has PyTorch
`2.14.1+cu130`, Transformers `4.57.3`, Accelerate `1.12.0`, Safetensors `0.8.0`
and NumPy `2.5.4`. The manifest records the runtime versions and GPU.

The official package also imports its unused 25 Hz implementation. Its import
requires `librosa`, `soundfile`, `sox`, `onnxruntime`, `einops` and `torchaudio`.
The isolated environment uses `torchaudio==2.11.0` installed with `--no-deps`:
there is no published matching `2.14.1` wheel in the available package index.
The selected 12 Hz inference path does not call torchaudio. Missing system SoX
and FlashAttention print upstream notices; neither is required by this path.

From the cuda-oxide root:

```bash
/tmp/qwen3-tts-acceptance/venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/export_reference.py \
  --source-dir /tmp/qwen3-tts-acceptance/source \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --output /tmp/qwen3-tts-acceptance/reference-sdpa \
  --codec-fixtures --codec-long-fixtures

/tmp/qwen3-tts-acceptance/venv/bin/python -m unittest discover \
  -s cute-rs-preview/models/qwen_decoder_cute/reference -p 'test_*.py' -v
```

Default input: “Hello. This is a cuda-oxide acceptance test.”, Ryan, English,
no style instruction, batch one. Streaming-text prompt preparation produces
10 prefill rows followed by one-row decode steps. `max_new_tokens=4` produces
four code-zero decisions and three completed 16-code frames. This short fixture
checks execution; it is not a complete speech quality or throughput benchmark.

The official Talker policy suppresses IDs 2048–3071 except EOS 2150, suppresses
EOS for the first two new tokens, and applies repetition penalty 1.05 to prior
code-zero IDs. The predictor uses greedy argmax. The manifest records the exact
arguments passed to the official generator and the actual chosen code IDs.

## Artifact layout

`manifest.json` maps tensor names to relative files, dtype, shape, byte count
and SHA-256. Raw files contain contiguous little-endian values. BF16 is stored
as its exact two-byte representation, including NaNs and signed zero.

| Names | Meaning |
| --- | --- |
| `prompt/*` | Text IDs, prepared Talker input, mask, trailing text, pad embedding |
| `talker/step_NNN/input` | Actual input for prefill or that decode call |
| `talker/step_NNN/layer_NN/output` | Output of each of 28 layers |
| `talker/step_NNN/layer_NN/cache_key`, `cache_value` | Completed cache in **[batch, token, KV head, lane]** order |
| `talker/step_NNN/layer_00/*` | Norm, projections, Q/K norm, attention, residual and FFN stages |
| `talker/step_NNN/hidden`, `logits` | Final norm and raw codec-head output for every input row |
| `predictor/step_NNN/*` | Same data for all 15 predictor calls of the selected frame and all five layers; `--predictor-frame` defaults to zero |
| `predictor/step_NNN/adapter_input` | Original 2048-wide input before the 1024-wide adapter |
| `generation/*` | Generated code-zero sequence, complete frames and Talker hidden rows |
| `codec/{two_frames,four_frames,generated}/*` | Codec inputs, selected decoder stages and FP32 exported waveform |
| `codec/{long_window,chunk_boundary}/*` | Optional 73/301-frame inputs and final waveform for sliding-window/chunk-boundary checks |

Tensor shapes retain the batch axis. Projection/norm tensors are token-major;
Q/K per-head norm outputs retain shape `[1, rows, heads, 128]`. `talker_steps`
and `predictor_steps` store position IDs and cache offsets in JSON.

## Numerical baseline

The canonical fixture uses official `attn_implementation="sdpa"` with PyTorch's
math backend, BF16 input/output and FP32 attention intermediates. TF32 and
reduced-precision BF16 GEMM reduction are disabled. Hooks observe the actual
module execution. They add copies and synchronization, so their timings must
not be used as performance measurements.

A second official run with `--attention eager` established that exact model-wide
BF16 equality across attention implementations is inappropriate:

| Same prefill input | Maximum absolute difference | Relative RMS difference |
| --- | ---: | ---: |
| Layer 0 attention | 0.0078125 | 0.00703 |
| Layer 0 output | 0.03125 | 0.00177 |
| Layer 27 output | 12.0 | 0.00535 |
| Final norm | 0.875 | 0.02282 |
| Raw logits | 0.625 | 0.02083 |

Both official implementations generate the same first 16-code frame. They
diverge during the second frame. Compare Rust on the saved reference inputs and
prefixes to isolate numerical errors; autonomous generation needs separate
code/logit-margin and audio-quality reporting. These observed differences are
context for choosing tolerances, not automatic Rust acceptance thresholds.

## Frozen transformer regression policy

The initial 3% relative-L2 pilot failed four of 937 comparisons: predictor
step 10 final norm and step 14 final layer, norm and logits. All four Talker
and 15 predictor decisions matched. The original failing report is retained as
`/tmp/qwen-stage4d-report.json`; no kernel was changed to mask those differences.

To distinguish local errors from accumulation, `QWEN_ISOLATE_LAYERS=1` supplies
each layer with the official preceding-layer output on every decode step while
building that layer's own cache. Its worst relative L2 was 0.007943. The full
composition's worst was 0.046185.

Before running a held-out prompt, the fixture regression limits are frozen at
**0.01 relative L2 for isolated layers and 0.05 for propagated graphs**. Both
runs must pass, all values must be finite, and the fixture's required code IDs
must match. The backend experiment explains why bitwise graph equality is not
required; it does not by itself establish correctness. Primitive tests,
isolated-layer checks and a held-out prompt provide separate evidence.
These are transformer fixture gates, not speech-quality acceptance or a general
promise of exact autoregressive codes for arbitrary prompts.

## Held-out transformer check

The limits above were frozen before exporting this second prompt:
**“The robot is ready. Please move the blue block to the left.”** Ryan, English,
BF16, SDPA math and `max_new_tokens=4` remain unchanged. The fixture contains
964 tensors and uses the same verified model/source revisions.

| Held-out execution | Comparisons | Largest relative L2 | Frozen limit | Required code decisions |
| --- | ---: | ---: | ---: | ---: |
| Propagated graph | 937 | 0.0425433 | 0.05 | 19/19 match |
| Isolated layers | 937 | 0.00517062 | 0.01 | 19/19 match |

The largest propagated difference is predictor step 11, layer 4 output; the
largest isolated difference is predictor step 10, final normalized hidden
state. Both runs pass without changing the limits. The 19 required decisions
are four Talker code-zero choices and all 15 predictor choices for the first
frame; they do not claim full autoregressive sequence identity.

Reproduce the official fixture with:

```bash
/tmp/qwen3-tts-acceptance/venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/export_reference.py \
  --source-dir /tmp/qwen3-tts-acceptance/source \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --output /tmp/qwen3-tts-acceptance/reference-heldout \
  --text 'The robot is ready. Please move the blue block to the left.' \
  --attention sdpa --max-new-tokens 4
```

Run the compiled example with `--checkpoint MODEL_DIR REFERENCE_DIR` once
normally and once with `QWEN_ISOLATE_LAYERS=1`; set `QWEN_REPORT` to distinct
paths. The recorded runs used physical GPU 1, an RTX 5090, selected with
`CUDA_VISIBLE_DEVICES=GPU-be79f07e-6c57-bdbd-b506-f92b8902c4aa`. Reports are
`/tmp/qwen-heldout-full.json` and `/tmp/qwen-heldout-isolated.json`; both contain
the eight actual artifact hashes.

This closes the held-out numerical requirement for stage 4D's teacher-forced
transformer checkpoint validation. Full speech quality, arbitrary-prompt
behavior and performance acceptance remain separate gates.

## Vivian second-frame diagnostic: strict decision gate remains failed

The same initial text with the Vivian preset exposes a broader-coverage failure:
the original Rust greedy rollout repeats “Hello” and reaches the 127-frame cap,
while the official math/automatic-attention runs reach EOS after 48 frames and
speak the full sentence. This predates the projection optimization. A separate
fixture captures the official predictor inputs and prefixes for frame 1:

```bash
/tmp/qwen3-tts-acceptance/venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/export_reference.py \
  --source-dir /tmp/qwen3-tts-acceptance/source \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --output /tmp/qwen3-tts-acceptance/reference-vivian-frame1 \
  --speaker Vivian --attention sdpa --max-new-tokens 4 --predictor-frame 1
```

`predictor_frame` records the selected zero-based frame in the manifest; the
15 saved calls remain numbered `step_000` through `step_014`. The exporter
rejects a requested frame that was not completed. Tensor and artifact checks
and the existing numerical/decision gates remain unchanged.

| Replay | Numerical comparisons | Largest relative L2 | Required decisions |
| --- | ---: | ---: | ---: |
| Rust propagated | 937/937 pass | 2.10571% | 18/19 match: **failed** |
| Rust isolated layers | 937/937 pass | 0.511888% | 19/19 match |

The propagated failure is predictor `step_013` (codec group 14). Official BF16
logits tie tokens 63 and 514 at 19.875, selecting the lower ID 63. Rust retains
514 at 19.875 but produces 19.75 for 63: one BF16 output step changes the winner.
The complete logit vector has 0.71555% relative L2 error; its hidden input has
1.61890%. Isolated-layer replay restores the official tie and token 63.

The original autonomous rollout first diverged earlier, at frame 1, codec
group 11 (606 versus 102). Supplying the saved official predictor input
restores the exact 19.625 tie between 102 and 606 and selects 102 in both Rust
modes. This rules out a different greedy tie-breaking rule at that observed
boundary; it does not establish where every upstream state difference began.
Controlled official SDPA replay is bitwise identical, while eager attention on
the same saved inputs changes another predictor decision (codec group 13) and
reaches 7.8812% hidden-state relative L2 error.

These results show that small propagated numerical changes can alter a
zero-margin decision even when tensor tolerances pass. They do **not** qualify
the repeated/truncated Vivian speech. Keep the strict decision failure and
full-rollout quality failure open; do not widen tolerances, add tie epsilons or
replace the failed greedy result with a sampling result. Seeded sampling is a
separate implemented policy with separate quality qualification.

Local diagnostic artifacts are `/tmp/qwen-vivian-frame1-rust.json`,
`/tmp/qwen-vivian-frame1-isolated.json`,
`/tmp/qwen-vivian-frame1-controlled.json` and
`/tmp/qwen-vivian-frame1-reference-margins.json`. These are diagnostic executions,
not performance measurements or evidence that all speaker presets pass.

## Reference benchmark

`benchmark.py` times the official PyTorch runner and the optimized third-party
runners with the same timing boundary as the Rust `--benchmark` mode. Run its
self-test first, then the official mode against the pinned checkpoint, source
and reference manifest:

```bash
python3 cute-rs-preview/models/qwen_decoder_cute/reference/benchmark.py --self-test

/path/to/benchmark-venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/benchmark.py \
  --mode official --sdpa-backend math \
  --model-dir /path/to/model \
  --official-source /path/to/source \
  --request /path/to/request.json \
  --reference-manifest /path/to/reference-sdpa/manifest.json \
  --warmups 3 --repeats 20 \
  --output /tmp/qwen3-tts-benchmarks/official.json
```

For the hybrid runner, use the same arguments plus `--mode hybrid`,
`--triton-source` and `--faster-source` pointing at the pinned third-party
checkouts that the script verifies. `--sdpa-backend auto` qualifies the fastest
available attention path; record it as a distinct configuration from the math
backend.

## Optional seeded sampling

`tokenize_request.py --sample --seed 0` emits the checkpoint's separate Talker
and predictor sampling defaults. Checked vocabulary-specific samplers preserve
greedy mode when the optional request field is absent. Reports record the RNG
version and SHA256-derived predictor seed; repeated-seed quality checks compare
both codec IDs and waveform hashes.

Sampling is tested separately from greedy correctness and performance.
