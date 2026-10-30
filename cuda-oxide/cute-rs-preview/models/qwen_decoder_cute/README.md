# Qwen text to speech

This model turns text into a 24 kHz speech WAV using
`Qwen3-TTS-12Hz-1.7B-CustomVoice`. It runs on RTX 5090 / SM120a.
Python prepares text tokens; Rust runs the model and generates the audio.

## How it works

```text
text tokens
    │
    ▼
Talker ──► first audio code
    │
    ▼
Predictor ──► fifteen more codes
    │
    └── repeat to build the speech sequence
                       │
                       ▼
                waveform decoder ──► speech.wav
```

The Talker reads the text and earlier audio. The smaller predictor completes
one group of sixteen audio codes. The waveform decoder turns those groups into
samples you can play.

Inside each GPU operation, `cute-rs` describes the data flow:

```text
BF16 buffers → tensor views → layout partitions → register tiles
                                                     │
output buffer ◄────────── copies ◄──── reductions or matrix multiplication
```

Typed dimensions keep model widths and weight roles separate. Owned buffers and
completion checks keep storage alive until GPU work finishes. The
[execution walkthrough](reference/execution.md) shows how a Talker step reuses
this storage and where the CPU waits.

## Run the checks

From `cuda-oxide/`, with the CUTLASS compiler installed (`cargo oxide toolchain install cutlass`):

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide run qwen_decoder_cute --arch sm_120a
```

Without arguments, the program checks synthetic decoder inputs, cache updates,
and the complete Talker/predictor wiring. A successful run prints `PASS` lines
and ends with `Qwen structured generation: ... passed`. It does not load a
checkpoint or generate speech in this mode.

## Generate speech

You need the pinned checkpoint and a tokenized request. The
[reference guide](reference/README.md) describes the required files;
[request preparation](reference/implementation.md#generation-sampling-and-measurement)
shows how to tokenize text and choose a voice.

Using the existing local acceptance files, after building above:

```bash
cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --generate /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen.wav
```

Listen to `/tmp/qwen.wav`. The adjacent JSON report records the generation time,
audio length and output checks. Replace the paths if your files live elsewhere.
Changing the request's text requires generating new token IDs too.

## Measure generation speed

```bash
cute-rs-preview/models/qwen_decoder_cute/target/release/qwen_decoder_cute \
  --benchmark /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-full.json /tmp/qwen-benchmark 20
```

This warms up three times and measures twenty complete generations. Each
printed time covers prepared text tokens through the complete waveform;
model loading, tokenization and WAV saving are excluded. The median is
`latency_p50_seconds` in `/tmp/qwen-benchmark/benchmark.json`. The generated
speech is saved there as `output.wav`.

Recorded results (RTX 5090, BF16; generation times are warm medians):

| Implementation / request | Generation time | Audio length | Compute seconds / audio second |
| --- | ---: | ---: | ---: |
| cuda-oxide / Vivian | **0.648 s** | 4.72 s | **0.1373** |
| Optimized reference (hybrid/math) / Vivian | 0.569 s | 3.92 s | 0.1453 |
| cuda-oxide / Ryan robot sentence | **0.608 s** | 4.40 s | **0.1381** |
| Optimized reference (hybrid/math) / Ryan | 0.547 s | 3.76 s | 0.1456 |

Each row uses 20 measured runs after three warmups on 2026-10-17 with the
decode projections in the reference's reduction order. The recordings differ.
Greedy decoding of the Ryan acceptance sentence runs to the 127-frame cap, for
cuda-oxide and for the official reference alike; only the reference's optimized
variants finish it.

The reference runners behind the comparison rows are documented in the
[reference guide](reference/README.md#reference-benchmark). Shapes, numerical
contracts, validation commands and recorded checks are in
[implementation and validation](reference/implementation.md).

Greedy Vivian completes since the decode projections follow the reference's
reduction order (`src/reference_linear.rs`). The greedy
Ryan acceptance sentence caps at 127 frames, as the official reference does.
Broader voice and language acceptance remains open. Audio is delivered as a
complete waveform; incremental playback is not implemented.
