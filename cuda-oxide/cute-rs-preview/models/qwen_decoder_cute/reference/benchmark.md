# Qwen full-waveform benchmark

## Reference reduction order, 2026-10-17

The one and two row decode projections now follow the reduction order of the
reference's cuBLAS kernels, shape by shape; the
[bug note](bugs/decode-projection-rounding.md) records the cause, the kernels
and the checks. Warm medians, three warmups and twenty runs, same RTX 5090,
nothing else running, identical outputs within every batch:

| Request | cuda-oxide before | cuda-oxide now | Optimized reference (hybrid/math) |
| --- | ---: | ---: | ---: |
| Vivian | cap 127: 1.625 s for 10.16 s of repetition | **0.648 s** for 4.72 s, 0.1373 per audio second | 0.569 s for 3.92 s, 0.1453 |
| Ryan robot | 0.673 s for 4.72 s, 0.1425 | **0.608 s** for 4.40 s, 0.1381 | completes in 51 frames; no timing recorded |
| Ryan acceptance | 0.569 s for 3.92 s, 0.1451 | cap 127: 1.576 s for 10.16 s | 0.547 s for 3.76 s, 0.1456 |

Synthesis costs 11.0 ms per codec frame against 11.6 ms before. The greedy Ryan
acceptance sentence caps for cuda-oxide and for the official reference (`sdpa`
math and auto); only the reference's optimized hybrid and faster variants finish
it. The "before" rows were measured today with the preserved baseline binary.
The sections below are kept as the record of the investigation that led here.

## Rejected precision candidate, 2026-10-17

A reusable CuTe FP64 dot product replaced every predictor attention-output
projection, with direct BF16 rounding. It used live GPU inputs in synchronous,
eager and CUDA graph execution; no saved values or frame-specific correction.
Talker arithmetic and the other predictor projections stayed unchanged.

The candidate fixes the saved numerical case but **fails speech acceptance**:
Ryan repeats and reaches the generation limit; Vivian reaches EOS but its ASR
transcript has one word mismatch. The candidate is retained in
`/tmp/cuda-rust-vivian-fp64-candidate`. The working checkout's source, binary and
cubin were restored to their original versions.

| Implementation / voice | Generation p50 | p95 | Audio | Compute / audio second | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Original cuda-oxide / Ryan | 0.566396 s | 0.567098 s | 3.92 s | 0.144489 | EOS, original output unchanged |
| Precision candidate / Ryan | 1.686481 s | 1.743286 s | 10.16 s | 0.165992 | **Rejected: repeats, hits cap** |
| Precision candidate / Vivian | 0.685454 s | 0.691129 s | 4.64 s | 0.147727 | EOS, one ASR word mismatch |
| Optimized reference / Ryan | 0.547476 s | 0.548909 s | 3.76 s | 0.145605 | Completes before cap |
| Optimized reference / Vivian | 0.569402 s | 0.571002 s | 3.92 s | 0.145256 | Completes; exact ASR transcript |

All five batches ran serially on the same RTX 5090 with three warmups and twenty
measurements. Every measured output was deterministic within its batch. The
reference uses the pinned hybrid/math configuration; all requests use greedy
decoding. Timing covers prepared token IDs through the complete host waveform,
excluding model loading, tokenization, WAV writing and validation. Audio lengths
and codes differ, so these are equal-request comparisons, not identical work.
The capped Ryan candidate is not a complete-speech performance result.

Pinned ASR recognizes the original and reference Ryan sentences with the same
single spelling mismatch (`kuda` for `cuda`). Candidate Ryan yields only “Hello”.
Candidate Vivian yields `qdar` for `cuda`; reference Vivian has no word errors.
Both versions pass the separate Ryan robot sentence, with 4.72 s original audio
and 5.04 s candidate audio. ASR is transcript evidence, not human listening or
speaker-similarity acceptance.

[Machine-readable evidence](precision-candidate-results-20261017.json) records
provenance, complete timing summaries, quality outcomes and raw artifact hashes.
The [Vivian investigation](vivian.md) explains why the earlier one-time diagnostic
did not establish a successful general repair.

## Earlier Ryan-only recheck, 2026-10-17

The unchanged production binary completed three warmups and twenty Ryan
measurements on the same RTX 5090. Median synthesis was **0.566685 s**, p95
**0.567928 s**, for identical **3.92 s** audio: **0.144563 compute seconds per
audio second**. The median is 0.24% below the previous 0.568071 s result, consistent
with the same performance level. All measured codes and waveform hashes match
the preserved benchmark; the saved WAV is byte-identical. No other GPU work ran
during measurement. The reference's 0.549109 s result remains the 2026-10-16
measurement and was not rerun in that earlier recheck.

The successful [Vivian diagnostic](vivian.md) produces 4.16 s of complete speech.
It uses synchronous predictor execution, live hash checks and a one-time output
upload; its reports explicitly mark timings as non-performance data. It does
not establish the speed of a production repair. Vivian's optimized generation
time for an accepted production repair remains pending.

[Recheck evidence](performance-recheck-20261017.json) records the unchanged
binary, output checks, telemetry summary and raw artifact hashes.

## Persistent Talker execution

Measured on 2026-10-16, RTX 5090 / SM120a. Queuing a complete Talker decode step
over owned reusable buffers reduces synthesis from **0.694 to
0.568 seconds**, with identical 3.92-second audio: **18.1% less time**.
This builds on the ordered projection prefetch below.
The [execution walkthrough](execution.md) shows what changed at the submission
and wait boundaries.

| Runner | Synthesis p50 | Synthesis p95 | Audio duration |
| --- | ---: | ---: | ---: |
| cuda-oxide before this change | 0.693598 s | 0.694611 s | 3.92 s |
| cuda-oxide persistent Talker | **0.568071 s** | **0.576663 s** | **3.92 s** |
| Optimized reference, hybrid/math | 0.549109 s | 0.549646 s | 3.76 s |

The reference produces different codes and a shorter waveform. Time per audio
second is 0.144916 for cuda-oxide and 0.146040 for the reference.
This is one request, not broad performance or quality acceptance. Vivian's known
greedy repetition failure remains unchanged.

The new path submits all 28 layers, final normalization and the output projection
before completing the step. It owns its cache and scratch, checks external input
readiness, checks completion before success, and attempts cleanup on errors.
The live attention prefix, CuTe arithmetic and BF16 rounding stay unchanged. The
public synchronous APIs and context-wide pool recycle fence retain their safety
rules. Talker graph capture is not part of this change; the predictor still uses
its existing graph.

Two three-warmup/twenty-measurement batches per version ran in order **before,
after, after, before**, followed by a fresh three-warmup/twenty-measurement reference
batch. Every measured cuda-oxide output has the original codes and FP32 waveform
hash; both saved WAVs match byte for byte. A five-measurement control with the new
binary in synchronous Talker mode takes 0.694062 seconds.
All GPU work was serialized. Clocks were unlocked; batch median SM clocks were
2865–2880 MHz and maximum temperatures 50–58 °C. Telemetry includes loading and
warmups. The final forty-run p95 includes the slower second batch; no timing
samples were discarded. Per-request Talker plan setup is included. Timing ends
when the host waveform is available, before hashing, WAV writing and local
request-state teardown, as in the prior cache path; this is not full API-call
or sustained-throughput timing.

Separate Nsight traces show stream/context waits falling from **68,378 to
3,528**, and gaps without recorded GPU activity from
**211.0 to 73.8 ms**.
These are diagnostic first-to-last-kernel spans with graph nodes traced. They
exclude the final waveform copy and host tail and are not partitions of the
unprofiled synthesis latency.

Validation passes: 91 host tests, strict clippy, exact hidden/logit/full-cache
comparisons across attention prefixes 63/64/65 and 127/128/129, readiness and
ownership checks, failure poisoning, and five saved speech cases. Removing the
required input-readiness wait makes the delayed-producer test fail; restoring it
passes. Six memcheck/racecheck/synccheck runs cover Talker and predictor execution.
Predictor API-error reporting is disabled for its intentional capture-failure
probes; memory/race/synchronization checks remain enabled. Talker retains API-error
reporting. Both final builds have identical SASS for all thirty kernels.

After building, reproduce either mode from `cuda-oxide/`, using a fresh output directory:

```bash
QWEN_TALKER_MODE=eager QWEN_PREDICTOR_MODE=graph \
  cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --benchmark /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen-talker-rerun 20
```

Set `QWEN_TALKER_MODE=sync` for the original per-operator path; `eager` is now the
default. [Implementation and focused validation](implementation.md#generation-sampling-and-measurement)
describe the ownership rules and `--talker-decode` checks.
[Machine-readable evidence](talker-execution-results.json) records latencies,
hashes and profile counts. Raw artifacts remain in `/tmp/qwen-talker-experiment/`.

## Ordered projection prefetch

Measured on 2026-10-16, RTX 5090 / SM120a, from code base `1f87f3cf8`, before the
persistent Talker change. Ordered prefetch reduced the same request from
**0.922 to 0.693 seconds**: **24.8% less synthesis time**, a 1.33× speedup, for
identical 3.92-second audio.

| Runner | Synthesis p50 | Synthesis p95 | Audio duration |
| --- | ---: | ---: | ---: |
| cuda-oxide before | 0.921603 s | 0.922230 s | 3.92 s |
| cuda-oxide after | **0.693436 s** | **0.694813 s** | **3.92 s** |
| Optimized reference, hybrid/math | 0.548521 s | 0.549364 s | 3.76 s |

The reference uses the same request and timing boundary but produces different
codes and a shorter waveform. This is context for the remaining gap, not an
identical-work comparison. Broad model quality acceptance remains open, including
the existing Vivian greedy repetition failure.

The change stages 32 K16 operand pairs in an existing CuTe `RegisterTile`, up from
four explicitly loaded pairs. This gives the compiler more independent loads to
schedule. The ordered fold retains every MMA accumulation and rounding boundary.
Output tiles, bias handling, readiness, ownership and the pool recycle fence are
unchanged. Smaller output tiles gave little benefit and are not part of this patch.
Both projection kernels use 76 registers and no local or stack allocation.

For cuda-oxide, two batches each used three warmups and twenty measured requests,
run in order **before, after, after, before**. The table combines forty measured
requests per version using nearest observed ranks. Each new run has identical
codes and FP32 waveform hashes to the preserved output; both saved WAVs match
byte for byte. The reference was rerun with three warmups and twenty measurements.
All GPU work was serialized after compilation. Clocks were unlocked; batch median
SM clocks were 2865–2880 MHz, and sampled maximum temperatures were 53–59 °C.
These telemetry samples include loading and warmups as well as measurements.

Separate Nsight captures confirm projection time fell from **580.4 to 353.0 ms**.
The GPU gaps stayed near **211 ms**, and kernel/launch/synchronization counts were
unchanged. These are diagnostic profiled intervals, not partitions of the
unprofiled latency above. This motivated the persistent Talker follow-up above,
which retains cross-stream readiness and completion cleanup on errors.

Validation passed: 87 host tests (three separately gated tests ignored), strict
clippy, 19,537 FP64/bias checks, 27,935 exact original-MMA comparisons, grouped
cross-stream checks, canonical/held-out checkpoints in both propagation modes,
five saved speech outputs, and nine memcheck/racecheck/synccheck runs across
projection, synthetic-model and predictor-graph workloads. Six new fixtures cover
K496/512/528/1040 with both row counts, bias modes and even padded strides. The
checkpoint reports retain their existing numerical gates; matching report
statistics are not claimed as per-tensor bit identity.

The working-checkout build and the sanitized isolated build have identical SASS
for all thirty kernels. See [machine-readable evidence](projection-prefetch-results.json)
for binary/report hashes, individual latencies and profile counts. Raw logs,
binaries and captures are under `/tmp/qwen-next-experiment/` on the measured host.

To reproduce a timing batch after building, from `cuda-oxide/`:

```bash
CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
QWEN_TALKER_MODE=sync QWEN_PREDICTOR_MODE=graph \
  cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --benchmark /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen-prefetch-rerun 20
```

Use a fresh output directory and run without other GPU work. The command includes
three warmups; build/loading/graph preparation are outside measured synthesis.

`benchmark.py` measures the pinned CustomVoice checkpoint on the local RTX 5090,
batch 1, BF16 model/decoder weights and BF16 KV cache. It supports the official
runner, CUDA graphs (`faster`), Triton fusion (`triton`) and both (`hybrid`).
These are candidates for the optimized baseline; no runner is declared fastest
until its output and measurements pass qualification.

The timing boundary is **host token IDs → complete host waveform**. Text
tokenization, checkpoint loading, fixture hashing, output comparison and WAV
serialization are outside measured inference. Model loading and the first call
including graph capture/JIT are reported separately. The Rust benchmark must use
the same request and timing boundary.

| Measurement | Definition |
| --- | --- |
| Latency p50/p95 | Synchronized wall time, 3 warmups and 20 measured runs; nearest observed rank |
| RTF p50/p95 | Synthesis seconds / actual waveform seconds; lower is better |
| Codec throughput | Complete 16-code frames / synthesis seconds |
| Peak memory | PyTorch allocated/reserved bytes, including resident weights |
| Output checks | Finite 24 kHz waveform, deterministic codes/audio, reference code agreement |
| First playable audio | Same as complete-waveform latency; incremental streaming is disabled |

The three-frame acceptance fixture is a numerical smoke case, not a useful
speech-quality or representative throughput workload. Longer requests require
their own reference artifacts and enough generated audio to evaluate quality.

## Pinned candidates and policy adapters

- Official model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`;
  [official source](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e).
- [Qwen3-TTS-Triton](https://github.com/newgrit1004/qwen3-tts-triton/tree/c64873e6a792d4dabc7ae592933fefd740ff4d5f),
  version 0.3.0, commit `c64873e6a792d4dabc7ae592933fefd740ff4d5f`.
  Its lockfile specifies `faster-qwen3-tts==0.2.4`. Hybrid uses its default
  layer patch range `[0,24)` with fused normalization and no TurboQuant.
- [faster-qwen3-tts 0.2.4 wheel](https://files.pythonhosted.org/packages/53/3b/7bfd1959a182dc29782ff05d12eb0b587b119b6ea0db07524ab97b6cdbd7/faster_qwen3_tts-0.2.4-py3-none-any.whl),
  SHA256 `aede5398034dc063feec82fd54e5289f2d8fb308a6667e388b1dafeb87f6d9a0`.
  Current faster main uses a newer Transformers/Qwen fork, so it is a separate
  qualification candidate; existing compiler and reference environments were not upgraded.

For the locked faster runner, the adapter configures predictor greedy decoding
**before graph capture**, widens logits to FP32 before the existing repetition
penalty, and translates the common `max_new_tokens=N` into its native frame cap
`N-1`. The official API and Rust runtime leave the last sampled code-zero token
unconsumed at a token limit. Both logical/native caps are recorded. EOS suppression
for the first two predictions, EOS 2150 and repetition penalty 1.05 match the
official policy. Static KV capacity is the smallest power of two covering the
request plus prompt allowance, with a minimum of 128 and maximum of 32768.

All model artifacts and official/optimized Python source trees are checksum
verified. Fusion and attention backends can still change rounded values and
generated codes. The JSON records this directly; exact code identity is useful
numerical evidence but is not required for intelligible speech. The harness keeps
`ready_for_speedup_claim=false` until a separate review of output quality and
paired timing conditions. It records whether the full repetition count is met.

## Local commands

Source archives and the wheel were downloaded into
`/tmp/qwen3-tts-acceptance/benchmark-sources/`; no repository was cloned or pulled.
The separate `benchmark-venv` reads the existing Torch/reference dependencies
through a `.pth` file without changing them. Its imports were checked before GPU
execution. Archives, commit IDs and archive hashes remain alongside the sources.

```bash
python3 cute-rs-preview/models/qwen_decoder_cute/reference/benchmark.py --self-test

CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  /tmp/qwen3-tts-acceptance/benchmark-venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/benchmark.py \
  --mode official --sdpa-backend math \
  --model-dir /tmp/qwen3-tts-acceptance/model \
  --official-source /tmp/qwen3-tts-acceptance/source \
  --request /tmp/qwen3-tts-acceptance/request.json \
  --reference-manifest /tmp/qwen3-tts-acceptance/reference-sdpa/manifest.json \
  --warmups 3 --repeats 20 \
  --output /tmp/qwen3-tts-acceptance/benchmarks/official.json
```

For hybrid, use the same arguments plus:

```bash
--mode hybrid \
--triton-source /tmp/qwen3-tts-acceptance/benchmark-sources/qwen3-tts-triton-c64873e6a792d4dabc7ae592933fefd740ff4d5f \
--faster-source /tmp/qwen3-tts-acceptance/benchmark-sources/faster-qwen3-tts-0.2.4 \
--output /tmp/qwen3-tts-acceptance/benchmarks/hybrid.json
```

Use `--sdpa-backend auto` when qualifying the fastest available attention path.
Record it as a distinct configuration from the math backend used for checkpoint
correctness. A report and first measured waveform are saved together. Run each
GPU benchmark alone; concurrent GPU work invalidates latency comparisons.

## Historical full-speech qualification pilot

The measurements below predate ordered projection prefetch and persistent Talker
execution. They retain the original baseline and quality evidence; the latest
performance measurements are at the top of this guide.

The same 19 text tokens, Ryan/English, greedy policy and logical cap of 128 were
tested on GPU `GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72`. These are **one warmup
and one measured run**, with other development active on the machine. They are
diagnostics, not a final speed comparison. Every waveform is finite and mono
24 kHz; audio duration comes from its sample count.

| Runner / SDPA | Measured seconds | Audio seconds | Stop | ASR WER |
| --- | ---: | ---: | --- | ---: |
| Official / math | 6.212 | 10.16 | Cap | 0.875 |
| Official / auto | 4.796 | 10.16 | Cap | 0.875 |
| Faster / math | 1.836 | 10.16 | Cap | 0.875 |
| Faster / auto | 0.623 | 3.52 | EOS | 0.000 |
| Hybrid / math | 0.549 | 3.76 | EOS | 0.125 |
| Hybrid / auto | 1.334 | 10.16 | Cap | 0.875 |

The capped cases transcribe only the first greeting and are not qualified as
complete speech for this request. Faster/auto and hybrid/math produce the whole
sentence. Hybrid/math's sole ASR substitution is the initial letter in
`cuda-oxide`; CER is 0.0286. Rust's separately generated 3.92-second waveform has
the same normalized transcript and ASR scores as hybrid/math. This is useful
intelligibility evidence for one sentence, not a broad quality claim or speaker
similarity measurement. Listening review and a larger prompt set remain needed.

This pilot selected hybrid/math and faster/auto for the subsequent 3+20
comparisons below. Results for the incomplete greetings remain recorded because
their timings do not qualify as complete-speech performance.

Pilot JSON/WAVs are in `/tmp/qwen3-tts-acceptance/benchmarks/`, with full-speech
math reports named `*-full.json`, auto reports `*-auto.json`, and ASR summaries
`asr-baselines.json` and `asr-rust-auto.json`.

### Provisional formal baseline samples

Both qualified configurations subsequently completed 3 warmups and 20 measured
requests on the same GPU0 UUID. All runs ended at EOS and every measured waveform
and codec sequence was byte-identical to its ASR-assessed pilot.

| Configuration | Latency p50 / p95, seconds | RTF p50 / p95 | Torch peak allocated / reserved, bytes |
| --- | --- | --- | --- |
| Hybrid / math | 0.546174 / 0.546676 | 0.145259 / 0.145392 | 4,455,345,664 / 4,521,459,712 |
| Faster / auto | 0.622380 / 0.623022 | 0.176813 / 0.176995 | 4,448,713,728 / 4,494,196,736 |

Telemetry was sampled every 500 ms throughout each process. Maximum temperatures
were 50/52 °C; median SM clocks were 2872/2880 MHz. Host load averages ranged from
0.31 to 0.85 and average CPU utilization was about 6%. Measured latency ranges
were below 0.3%, with no observed contention. GPU1 correctness work was permitted
during this window, so the baseline samples were provisional pending the paired
Rust run and quiet-system check below. Raw observations, telemetry CSV,
host/environment snapshots and `summary.json` are in `benchmarks/formal-provisional/` under the
temporary acceptance directory. No speedup over Rust is claimed here.

The original JSON reports used interpolated percentiles. Their untouched raw
observations were re-aggregated with Rust's nearest-observed-rank estimator;
`*-normalized.json` and `summary-normalized.json` preserve original file hashes
and describe the adjustment. The table uses these normalized aggregates.
For this delivery mode, first playable audio is the complete-waveform latency;
this is recorded explicitly with `incremental_streaming=false`.

PyTorch allocator peaks are reset before each request. **Allocated** includes
live Torch-managed storage, resident weights and graph pools; **reserved** also
includes allocator caching slack. These metrics exclude some CUDA context and
library allocations. Sampled device-memory peaks were 5054/5034 MiB respectively;
they include driver-visible overhead but can miss short-lived peaks. Rust memory
comparisons must use the same accounting boundary or label the metrics separately.

The preserved pre-optimization Rust binary subsequently ran 3+20 on the same
GPU0. Nearest-rank latency was 4.986564 s p50 / 4.991908 s p95, with RTF p50
1.272083. All runs ended at EOS with 49 frames / 3.92 s audio and identical codec
hashes. Its first saved waveform exactly matches the previously ASR-assessed
Rust waveform; that binary did not retain per-run waveform hashes, so full audio
determinism across all 20 requests is unverified. Sampled device-memory peak was
4492 MiB, maximum temperature 51 °C, median SM clock 2880 MHz, and latency range
0.21%. The report, environment snapshots and telemetry are in `benchmarks/rust-preopt/`.
The copied binary is `/tmp/qwen3-tts-acceptance/qwen-preopt`, SHA256
`86dd29edbc8b0a5391e35620596091c5acab23a5445fcd970978af8ecbf25852`.
The next section records the correctness and measurement follow-up for that stage.

### Historical same-GPU comparison

The final Rust binary for that stage completed a separate quiet-system 3+20 run
on GPU0. Every measured code sequence and waveform hash matched, and the saved WAV is identical
to the pre-optimization output. Nearest-rank results are:

| Implementation | Latency p50 / p95, seconds | RTF p50 | Frames / audio seconds |
| --- | --- | ---: | --- |
| Rust preopt | 4.986564 / 4.991908 | 1.272083 | 49 / 3.92 |
| Rust final | 4.897868 / 4.906532 | 1.249456 | 49 / 3.92 |
| Hybrid / math | 0.546174 / 0.546676 | 0.145259 | 47 / 3.76 |
| Faster / auto | 0.622380 / 0.623022 | 0.176813 | 44 / 3.52 |

That identical-output Rust change reduced median latency by **1.78%**. At this
stage Rust was slower than real time on this request. Against hybrid/math, its
latency ratio was 8.97× and RTF ratio was 8.60×. These cross-runner measurements
use the same checkpoint, text, precision and greedy policy, but different rounded trajectories
and EOS lengths; they do not represent identical generated work. Performance
parity had not been reached at this stage.

Final telemetry showed a maximum 55 °C, median SM clock 2880 MHz, average host CPU
utilization 4.6%, and measured latency range 0.33%, with no observed contention.
The sampled device-memory peak was 4510 MiB. Raw reports, hashes, comparisons and
telemetry are in `benchmarks/rust-final/`; the pinned binary SHA256 is
`2441142c29cbfd0b638f5bdb45e502d6fd76c612ed04cd70fd5fefcb3b7e7144`.

### Additional quality cases

The preserved Rust binary and hybrid/math were also checked with two requests at
the same logical cap of 128. These are quality diagnostics, without timing claims.

| Request | Rust preopt | Hybrid/math |
| --- | --- | --- |
| Ryan: “The robot is ready. Please move the blue block to the left.” | EOS, 59 frames / 4.72 s; exact transcript, WER/CER 0 | EOS, 51 frames / 4.08 s; exact transcript, WER/CER 0 |
| Vivian: original acceptance sentence | Cap, 127 frames / 10.16 s; only the greeting, WER 0.875 | EOS, 49 frames / 3.92 s; exact normalized transcript, WER/CER 0 |

Vivian is present in the checkpoint configuration (speaker ID 3065). The Rust
failure predates the optimization attempt and remains an open quality gap;
successful Ryan cases do not establish all-speaker acceptance. ASR assesses
intelligibility here, not speaker identity or consistency. Requests, WAVs, exact
transcripts, hashes and `summary.json` are under `benchmarks/quality-cases/`.

Official Vivian diagnostics subsequently reached EOS at 48 frames / 3.84 s
under both math and auto attention. Both transcribe the whole sentence, with
one proper-name substitution (WER 0.125). Rust first differs from official/math
at frame 1, codebook 11, and from official/auto at frame 1, codebook 15; frame 0
matches. This comparison establishes a Rust quality gap but does not alone
classify it as numerical sensitivity or an implementation defect. Teacher-forced
predictor/logit comparisons remain the appropriate numerical investigation.

## Optional seeded sampling

Greedy remains the deterministic checkpoint and performance-comparison policy.
The checkpoint's normal sampling defaults are available as an explicit request:

```json
"sampling": {
  "seed": 0,
  "talker": {"temperature": 0.9, "top_k": 50, "top_p": 1.0},
  "predictor": {"temperature": 0.9, "top_k": 50, "top_p": 1.0}
}
```

`tokenize_request.py --sample --seed 0` writes this object; omitting `--sample`
preserves the existing greedy request. Talker repetition penalty 1.05 and its
special-token/EOS rules remain in place. The predictor has its own policy for
the fifteen remaining code groups and no repetition penalty.

`SamplingConfig<V>` validates the parameters before generation. `Sampler<3072>`
and `Sampler<2048>` keep the two vocabularies distinct. Selection uses FP32
temperature/top-k/top-p and softmax, then an FP64 cumulative probability sum.
Top-k boundary ties and the top-p crossing token are retained. Raw BF16 logits
must be finite; only explicitly processed Talker masks may contain negative
infinity. GPU model computation remains in Rust kernels; the sampler uses the
existing host-logit readback boundary.

The recorded RNG is `splitmix64-v1/top53-uniform`. Talker uses the request seed;
predictor uses the first eight little-endian bytes of
`SHA256("cuda-oxide/qwen-predictor/splitmix64-v1" || seed_le_u64)`. The RNG is
reproducible and noncryptographic, with distinct role seeds. Equal integer seeds
do not reproduce PyTorch's RNG stream. Tests therefore compare independently
exported Transformers 4.57.3 probabilities and shared uniform draws, alongside
same-seed repeatability, parameter bounds, masks and dtype checks.

Sampling quality is a separate acceptance case. It does not close the unresolved
greedy Vivian issue, and its timing is not substituted into the greedy benchmark.

Final sampling checks used binary
`2441142c29cbfd0b638f5bdb45e502d6fd76c612ed04cd70fd5fefcb3b7e7144` and the
SHA256 role-seed derivation above:

| Speaker / seed | Stop / duration | ASR result |
| --- | --- | --- |
| Ryan / 0 | EOS, 45 frames / 3.60 s | Complete sentence; one proper-name letter substitution, WER 0.125 |
| Vivian / 0 | EOS, 56 frames / 4.48 s | Exact normalized transcript, WER/CER 0 |
| Vivian / 1 | EOS, 52 frames / 4.16 s | Exact normalized transcript, WER/CER 0 |
| Vivian / 0 repeated | EOS, 56 frames / 4.48 s | Identical complete codec sequence, stop, sample count and waveform hash |

Results and ASR provenance are in `benchmarks/sampling-quality-final/`. Earlier
`sampling-quality/` outputs used a superseded seed derivation and are retained
only as diagnostics. These few cases establish seeded repeatability and
intelligibility evidence; they do not establish broad speech quality or speaker
consistency.

## Reproducible ASR assessment

`assess_audio.py` runs the official
[Whisper-base.en checkpoint](https://huggingface.co/openai/whisper-base.en/tree/911407f4214e0e1d82085af863093ec0b66f9cd6)
locally in FP32 with greedy decoding. Model revision, file hashes, waveform hashes,
ASR environment and resampling are included in its JSON. The checkpoint is held
under `/tmp/qwen3-tts-acceptance/asr`; no model weights or generated audio are
committed. This tool supports mono 24 kHz requests of at most 30 seconds.

WER uses Unicode normalization, lowercase letters/numbers and punctuation as
word boundaries. CER additionally removes spaces. Neither score corrects
proper nouns by hand. ASR can mishear speech and does not measure naturalness or
speaker consistency.

```bash
python3 cute-rs-preview/models/qwen_decoder_cute/reference/assess_audio.py --self-test

CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  /tmp/qwen3-tts-acceptance/benchmark-venv/bin/python \
  cute-rs-preview/models/qwen_decoder_cute/reference/assess_audio.py \
  --asr-dir /tmp/qwen3-tts-acceptance/asr \
  --request /tmp/qwen3-tts-acceptance/request-full.json \
  --wav /tmp/qwen3-tts-acceptance/rust-full.wav \
  --output /tmp/qwen3-tts-acceptance/benchmarks/asr-rust.json
```
