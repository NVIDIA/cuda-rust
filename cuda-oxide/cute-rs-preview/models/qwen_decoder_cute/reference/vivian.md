# Vivian greedy speech investigation

Snapshot: 2026-10-17, RTX 5090. **Resolved.** The cause was the tensor-core
accumulation rounding toward zero at every MMA step; the decode projections now
follow the reference's cuBLAS reduction order shape by shape and match it bit
for bit. Greedy Vivian completes; greedy Ryan now caps, as the official
reference does. The [bug note](bugs/decode-projection-rounding.md) records the
cause, the fix and the checks. The sections below are the investigation that led
there, kept as written: a one-time diagnostic correction restored this sentence,
and the reusable but merely more accurate FP64 candidate made Ryan repeat.

## Reusable GPU candidate: numerical check passes, speech acceptance fails

The candidate applies the same CuTe FP64 dot and direct BF16 rounding to every
predictor attention-output projection, across all layers, rows, steps and voices.
It computes from live operands entirely on the GPU, including graph replay.
There are no request-specific conditions, saved replacements or forced codes.

The saved failing projection now matches all 1,024 independently certified BF16
results, including column 17. Another 76 synthetic results cover cancellation,
strides, multiple rows, tails and guards. The focused CuTe host suite passes 56
tests; the model host suite passes 97, with three GPU-only tests skipped.
Generated instructions confirm FP64 arithmetic. Predictor lifecycle validation
passes exact synchronous/eager/graph comparisons, ownership, guard rejection,
foreign-stream readiness and capture/error/unwind completion. Projection
memcheck, synccheck and racecheck report zero errors/hazards. These checks qualify
the numerical implementation; they do not overturn its speech-quality failures.
This is high-accuracy accumulation, not a proof of exact rounding for every
possible BF16 input.

| Greedy request | Original | Candidate |
| --- | --- | --- |
| Ryan acceptance sentence | EOS, 3.92 s; complete sentence with one ASR spelling mismatch | **Cap, 10.16 s; ASR only “Hello”** |
| Vivian acceptance sentence | Prior preserved run: cap, 10.16 s; only “Hello” | EOS, 4.64 s; one ASR word mismatch |
| Ryan robot sentence | EOS, 4.72 s; exact ASR transcript | EOS, 5.04 s; exact ASR transcript |

The candidate was rejected and isolated at `/tmp/cuda-rust-vivian-fp64-candidate`.
The original working source, binary and cubin were restored. Fresh measurements
put original Ryan at 0.566 s, candidate Vivian at 0.685 s, and the optimized
reference at 0.547 s for Ryan and 0.569 s for Vivian. Candidate Ryan takes 1.686 s
but never completes the requested sentence; that is not a valid speech speedup.
See [timing and quality evidence](benchmark.md#rejected-precision-candidate-2026-10-17).

This separates two findings: the earlier single correction causally rescued one
trajectory; the broader numerical policy changes many trajectories and fails a
regression request. More accurate arithmetic alone has not established a general
greedy repair. Before another arithmetic change, the next useful discriminator
is the first changed Ryan predictor state under this preserved candidate. Its
first observed code difference is frame 1 / group 4: original 1341 versus
candidate 1381 (zero-based indices). All preceding codec IDs agree. The first
Talker code difference follows at frame 2: 1824 versus 1014. This localizes the
next comparison; it does not yet identify the responsible layer or arithmetic
value. Raw evidence is `production-repair/fp64-candidate/ryan-first-code-difference.json`
under the local investigation directory.

## Free continuation: one corrected calculation restores this sentence

The native follow-up below identified a specific rounding difference. We then
corrected its **whole projection vector once**, at frame 1 / predictor step 6 /
layer 3, and let every subsequent code be selected normally. This is the actual
failing native input, checked against saved input, original-output and weight
hashes. No reference hidden values or selected token IDs are substituted.

| Same Vivian request | Unchanged control | One arithmetic correction |
| --- | --- | --- |
| Stop | Generation limit, 127 frames | End of speech, 52 frames |
| Audio length | 10.16 s | 4.16 s |
| Independent ASR | Only “Hello” | Complete requested sentence |
| Normalized word error rate | 0.875 | 0 |

The correction changes three BF16 projection values; only one survives the
residual addition. It changes the next layer's stored attention history, then
the natural group-11 choice from 606 to 102. Free continuation subsequently
chooses code 63 at group 14 and code 162 at group 15. The Talker's first changed
code-zero choice occurs at index 4. The final audio stops naturally; its longest
consecutive code-zero repetition drops from 47 to 3.

```text
One more accurate projection
          |
different retained attention history
          |
different natural greedy choice
          |
free continuation completes the requested sentence
```

This demonstrates that a numerical difference can redirect this failing run.
It does not establish a unique cause or justify a frame-specific production
patch. The reusable FP64 policy tested above fails the other-request regression gate.
Prior broad precision changes also regressed other requests, as recorded below.

The unchanged synchronous run and identical-output upload control both match
all saved codes and waveform bytes. Control and correction use the same binary
and GPU. Exactly one replacement is consumed; later steps use ordinary native
operators and natural token selection. The independent runtime audit passes
240 checks. The full diagnostic host suite passes 123 tests, with four opt-in
tests skipped. Pinned local Whisper provides intelligibility evidence; this is
not a human assessment of voice similarity or broad model acceptance.

Artifacts under `/tmp/qwen-vivian-investigation/feedback/`:
`native-free-none-v4.json`, `native-free-control-v4.json`,
`native-free-exact-v4.json`, their WAVs, `native-free-speech-audit.json`,
`native-free-asr-v4.json`, `native-free-quality-binding-audit.json`,
`native-projection-probe-sources-v4/manifest.json`,
and `native-projection-evidence-index.json`. The isolated diagnostic remains
outside the main checkout; production code and benchmark results are unchanged.

## Native replay: a rounding difference changes an earlier choice

The location is the **predictor's fourth decoder block** (layer index 3), in
`self_attn.o_proj`, the matrix multiplication after attention weighting and before
the residual addition. Its weight has shape `[1024, 2048]`; each output sums
2,048 products. This is separate from the Talker's attention and from softmax.

The reference projection was also rerun with the **exact same native input and
weight bits**. Its complete output matches the previously recorded reference
hash. At the sensitive column 17:

| Same-input projection result | BF16 value |
| --- | ---: |
| cuda-oxide | -0.10791015625 |
| Official reference | -0.1083984375 |
| Correct rounding of the exact dot product | -0.1083984375 |

The exact sum is `-0.10815433898824267`, only `4.21e-8` below the rounding
midpoint. The accumulator probe below lands on the other side before conversion.
Thus the reference avoids this specific rounding difference on the same operands;
it is not universally exact, differing from the oracle at one other output here.
Its full model also has different upstream states, so this local comparison
does not establish a unique explanation of every reference trajectory.
`reference-native-step6/report.json` and `reference_output.bf16le` under the
local evidence directory preserve the comparison and complete reference output.

The reference experiment below now has a native follow-up. Its two sensitive
coordinates are already correctly rounded on cuda-oxide's own inputs, so those
coordinates are not a demonstrated cause of our failure. An independent audit
instead finds 21 rounding differences across the same layer's 15 native output
projections. Only two survive the immediate residual addition.

Correcting the complete projection at **step 6, layer 3** changes a real native
choice through the next layer's retained cache:

```text
More accurate accumulated sum
          |
one BF16 difference survives the residual addition
          |
layer 3 output changes -> layer 4 stores different K/V at position 7
          |
group 11: score(102) rises 19.5 -> 19.625; score(606) stays 19.625
          |
ordinary greedy tie rule selects 102 instead of 606
```

The full corrected vector contains three changed values; only column 17 survives
the residual addition. The other surviving correction occurs at step 12,
column 87. Testing each step separately gives this native comparison:

| Corrected layer-3 projections | Natural group-11 choice | Final group-15 choice |
| --- | ---: | ---: |
| None | 606 | 278 |
| Step 6 only | 102 | 278 |
| Step 12 only | 606 | 278 |
| All 15 steps | 102 | 278 |

**This replay prescribes groups 1–14**, including code 606 at group 11. The
corrected run would naturally choose 102 there, so its final 278 does not tell
us what freely continued speech would do. Step 6 alone exactly reproduces the
full-correction run's downstream tensors through step 11. Step 12 contributes
a separate cache change at position 13.

The upstream input also matters. Starting our predictor with the official
Talker input and correcting this layer restores both the reference's group-11
choice and final code 162. With our own Talker input, the table above applies.
These controlled results do not establish a general accuracy policy or repair.

Validation: identity controls preserve all 3,330 native tensors and every code;
all 32,768 candidate projection outputs have certified BF16 rounding, including
exact integer checks for all 41 changed values across both input cases. Live
input, original output and weight hashes must match before every substitution.
The four-arm audit and separate cache/localization audit pass. A synchronous
full-speech control also preserves every saved code and all waveform bytes.

A separate CuTe accumulator probe reproduces all 1,966,080 saved projection
outputs. Its FP32-to-BF16 conversion agrees with independent rounding for every
value; the sensitive error is already in its accumulated sum. Compute Sanitizer
reports zero errors. This sibling kernel has the same arithmetic loop and a
different diagnostic output store; BF16 equality does not prove identical hidden
FP32 bits in the original binary.

Artifacts under `/tmp/qwen-vivian-investigation/feedback/`:
`native-layer3-projection-own-input-oracle.json`,
`native-projection-repair-inputs/certification.json`,
`native-projection-certificate-review.json`,
`native-projection-four-arm-audit.json`,
`native-projection-factorial-localization.json`,
`native-sync-baseline-audit.json`, and `accumulator-probe/audit.json`.
All runtime changes remain in the isolated diagnostic checkout.

## Reference hybrid: one projection value changes the disputed decision

A controlled substitution now reproduces the decision change using **one
cuda-oxide projection call**, with the same live input and weights as the
official reference. It leaves attention, normalization, activations and output
heads in the reference implementation. The worker uses the existing typed
`Runner::project` path and unchanged CuTe projection kernel.

The smallest tested substitution is predictor **step 3, layer 3** (zero-based)
at the attention output projection, `self_attn.o_proj`. Only column 721 differs:

| Result | Value |
| --- | ---: |
| Official reference | -0.01275634765625 |
| cuda-oxide | -0.0128173828125 |
| Independent FP64 dot product | -0.01278682037445833 |

The exact-input FP64 calculation rounds to the official BF16 value. The two
observed outputs are neighboring BF16 values; the exact dot lies approximately
`4.49e-8` from their rounding midpoint. This is a concrete accumulation/rounding
difference, without evidence of an incorrect weight, input or memory access.

```text
One changed projection value at step 3
                |
       layer 3 output changes
                |
   layer 4 retains different K/V at cache position 4
                |
       final attention result changes
                |
 code 162: 21.625 -> 21.5; code 278 stays 21.625
                |
         greedy choice: 162 -> 278
```

All fourteen preceding natural code choices remain unchanged. Their prescribed
prefix therefore overrides no choice in this case. Final-step layer inputs and
current Q/K/V are unchanged; the difference reaches final attention through the
retained earlier cache entry. The cache preserves its appended values exactly.
A separate substitution at step 2/layer 3 also changes the final choice to 278.
Both are sufficient paths, so neither is established as the unique cause.

The broader layer-3 substitution changes twelve projection values, but only
steps 2 and 3 change the layer's output; the other differences are absorbed by
subsequent rounding. Replacing only the final predictor layer's projections
keeps the official choice at 162. The existing single-code speech experiment
below remains essential: giving code 278 to the reference does not make its
whole utterance fail. This new result explains a specific decision divergence;
it does not close the autonomous repetition failure or qualify a repair.

Before substitution, all 1,050 saved projection calls (1,966,080 output values,
both hidden-input cases) match the worker bit-for-bit. Calling the worker while
retaining native reference results preserves all 2,385 saved tensors per case
and their natural decisions. The selective-step control also passes. The
diagnostic host suite passes 106 tests, with three opt-in tests skipped.

Local artifacts under `/tmp/qwen-vivian-investigation/feedback/`:
`projection-worker-fixtures.json`, `projection-worker-sources/`,
`projection-intervention-audit-eight-runs.json`,
`projection-step-intervention-audit.json`, `projection-evidence-index.json`,
`layer3-projection-own-input-oracle.json`, `official-projection-control/`,
`official-layer3-selected-control/`, `official-layer3-step2/`, and
`official-layer3-step3/`. Reports pin the worker binary, source, model, live
inputs and returned outputs. Production kernels and generation policy remain
unchanged; these diagnostic timings are not benchmark results.

## First isolated calculation difference

A replay now holds the predictor's input and preceding codes fixed and captures
every operation in its five layers. The first difference is **layer-zero
attention on the initial two-row predictor call**. The adapter output, input
normalization, Q/K/V projections, Q/K normalization, rotated query and complete
K/V caches are byte-identical up to this point. Attention differs in 385 of
4,096 values, all in the second row.

The reference selects PyTorch FlashAttention. Its result is reproduced exactly
by rounding the unnormalized attention weights to BF16 before mixing the value
vectors, while retaining the unrounded normalization denominator. cuda-oxide
keeps those weights in FP32. Its 4,096 results match both an independent FP64
calculation rounded to BF16 and a separate PyTorch math-attention replay.

```text
Identical query, key and value inputs
              |
      attention weighting
       /                \
reference: BF16       cuda-oxide: FP32
intermediate weights intermediate weights
       \                /
      different BF16 outputs
```

This identifies a concrete numerical difference, **not an incorrect attention
mask, cache or proven kernel defect**. Across 150 reference calls, an independent
FP64 oracle with this intermediate BF16 rounding reproduces 327,654 of 327,680
stored values; the remaining 26 still reflect arithmetic differences. Matching
one attention output does not establish complete speech quality.

Two counterfactual replays test whether this is the entire cause. Both retain the
same fourteen forced prefix codes and leave the final code free:

| Official predictor computation | Official hidden input | cuda-oxide hidden input |
| --- | ---: | ---: |
| Unchanged reference | 162 | 278 |
| Substitute only the first cuda-oxide attention output | 162 | 278 |
| All attention uses FP64 unrounded probabilities, final BF16 store | 162 | 162 |

The single substitution checks that the live query/key/value inputs exactly
match the saved cuda-oxide inputs before replacement. It does not change the
final choice. Removing intermediate probability rounding throughout the
reference predictor also does not reproduce cuda-oxide's final 278 from the
official input. The earliest split is therefore **not sufficient to explain
the final decision**. Other accumulated numerical differences remain relevant;
the FP64 attention control is not bit-identical to cuda-oxide's FP32 arithmetic.
Some natural prefix choices change in these counterfactuals, but the prescribed
prefix is deliberately retained. These are component diagnostics, not freely
generated speech or proposed production changes.

The capture preserves all 120 existing cuda-oxide replay tensors and all 1,170
existing reference replay tensors exactly. The original reference's 45 natural
boundary tensors also remain exact. The reference audit passes 4,835 checks,
including actual attention inputs, masks, grouped heads, rotary rounding, cache
updates, residual additions and layer connections. Own-input checks of all five
layers in the initial call find exact normalization and elementwise results;
matrix reductions have small rounding differences in both implementations.
The diagnostic host suite passes 101 tests, with three GPU opt-ins skipped.

The expanded own-input audit covers all 15 predictor steps and 75 layer calls
for the shared official seed. Both implementations' 409,600 residual-add and
product values match exactly. Of 409,600 normalization and 245,760 SiLU values,
cuda-oxide has one normalization and two SiLU differences from the higher
precision oracle; the reference has none. The normalization value matches the
pinned FP32 formula, and both SiLU differences occur at the same input almost
exactly on a BF16 rounding midpoint. Of 983,040 linear outputs, the reference
differs from the oracle at 157 values and cuda-oxide at 697. Errors larger than
one BF16 spacing occur at near-zero cancellation results; their maximum absolute
errors are approximately `3.73e-7` and `3.30e-6`, respectively, within conservative
FP32 accumulation bounds. This audit identifies no additional contract
violation; those bounds do not establish the actual hardware reduction order
or explain which accumulated differences cause the speech failure.

Rotary coefficients also match bit-for-bit over positions 0 through 137,
covering the entire failing Talker run: all 17,664 BF16 sine/cosine values are
identical. This compares the actual loaded reference rotary module on CUDA with
the unchanged Rust table source linked against the real cute-rs build; batch
and single-position Rust tables also agree. Host/CUDA trigonometry differences
are therefore not an explanation for this request's recorded trajectory.

There is also a separate, earlier difference in the actual text projection
before Talker: prompt relative L2 is approximately 0.0000414. The controlled
predictor replay removes that input difference from the attention comparison;
it does not establish equivalence of the entire autonomous model.

Artifacts under `/tmp/qwen-vivian-investigation/feedback/`:
`stage-boundary-comparison.json`, `attention-backend-probe.json`,
`probability-rounding-confirmation.json`, `attention-own-input-oracle.json`,
`full-path-contract-oracles.json`, `attention-intervention-audit.json`,
`rotary-full-range-comparison.json`,
`official-stage-audit.json`, `layer-trace-control.json`,
`prompt-projection-comparison.json`, `official-predictor-full-stages/`, and
`rust-predictor-layer-replay/`; the two counterfactuals are
`official-first-rust-attention/` and `official-unrounded-attention/`.
`layer-trace-sources/` preserves the diagnostic
sources, base commit and binary hash. Production kernels remain unchanged.

## Replay the failing history in the successful reference

The automatic-attention reference Talker was also given the exact captured
cuda-oxide input rows and greedy history, starting from a fresh persistent
reference cache. It matches **31 of 32 choices**, disagreeing only at step 9.
It selects repeated code 1524 at every tested step from 15 through 31, just as
cuda-oxide does. This extends the earlier math-attention replay to the same
automatic-attention policy used by the successful reference control.

The current-step calculation is therefore not sufficient to explain the loop:
the preceding generated inputs already make the reference repeat as well.
This replay does not replace those inputs with the reference's own choices,
and is not an autonomous reference failure or a proof that every earlier
cuda-oxide operation is correct. Artifact: `feedback/official-auto-on-rust/`
under the local investigation root.

## Single-code feedback test, 2026-10-17

The successful official automatic-SDPA run and baseline cuda-oxide agree on
their first 31 codes. Frame 1/group 15 (zero-based) is the first difference:
cuda-oxide selects 278, while the reference selects 162. Diagnostic runs changed
only this returned predictor code before embedding feedback, then resumed free
greedy generation. No kernel arithmetic or sampling setting changed.

| Runner and diagnostic intervention | Result | Complete sentence in ASR |
| --- | --- | --- |
| cuda-oxide, unchanged | 127-frame cap | No |
| cuda-oxide, reconstruct the same code 278 | Identical baseline output | No |
| cuda-oxide, replace 278 with 162 once | EOS, 49 frames / 3.92 s | Yes |
| Official auto, unchanged | EOS, 48 frames / 3.84 s | Yes |
| Official auto, replace 162 with 278 once | EOS, 55 frames / 4.40 s | Yes |

The reconstruction control matches all 516 captured tensors, codec IDs and
waveform bytes exactly. The unmodified reference control matches its saved codes
and waveform hash. Both changed-code runs transcribe the complete sentence with
one project-name substitution (WER .125). Changing this one choice is sufficient
to escape the cuda-oxide loop, but 278 is not intrinsically a failing code: the
reference continues successfully with it. Forced codes are diagnostic evidence,
not a proposed implementation or a closed greedy acceptance result.

At that decision, cuda-oxide scores 162 at 21.5 and 278 at 21.625; the reference
ties both at 21.625 and correctly selects the lower ID. Each implementation's
two scores match FP64 dot products on its own final hidden input after BF16
rounding. The disagreement therefore exists before the final output projection;
it is not explained by rounding in that projection alone. The two hidden inputs
have relative L2 difference .01905.

A second replay holds code zero and the preceding fourteen predictor codes fixed,
then leaves the final code free. Each case starts with a fresh predictor cache:

| Talker hidden input | Official predictor | cuda-oxide predictor |
| --- | ---: | ---: |
| Official | 162 | 278 |
| cuda-oxide | 278 | 278 |

The official control reproduces all 45 saved adapter/head boundary tensors
exactly. The cuda-oxide control reproduces its native graph's final hidden row
and all 2,048 logits byte-for-byte. Either the different Talker input or predictor
computation can change this close decision; neither needs a different preceding
code history in this replay. This does not identify a particular faulty operation
or establish that all observed numerical differences violate their contracts.

Artifacts: `/tmp/qwen-vivian-investigation/feedback/causal-summary.json`,
`head-oracle.json`, `replay-comparison.json`, `official-predictor-replay/`,
`rust-predictor-replay/`,
the `*-trace/` directories and the two `*-asr.json` reports.
The isolated diagnostic source and binaries are preserved in that directory;
production generation defaults and kernels remain unchanged.

## Earlier controls

The failing request is “Hello. This is a cuda-oxide acceptance test.”, Vivian,
English, `non_streaming_mode=false`, no instruction, and `max_new_tokens=128`.
Baseline cuda-oxide produces 127 completed frames (10.16 s), reaches the cap,
and transcribes only “Hello!” (WER .875). Official BF16 math and automatic SDPA
runs complete the sentence at EOS after 48 frames (3.84 s; WER .125).
Existing official greedy Ryan math/auto runs also fail this sentence; they are
not successful quality targets. Baseline cuda-oxide Ryan completes 49 frames.

`EOS N` below counts completed frames; `cap` means 127 frames without EOS.
Ryan robot means “The robot is ready. Please move the blue block to the left.”

| Experiment | Vivian sentence | Ryan sentence | Ryan robot | Why not ship |
| --- | --- | --- | --- | --- |
| Baseline | cap | EOS 49 | EOS 59 | Vivian truncates |
| Block-128 projection accumulation | EOS 51 | cap | EOS 53 | Ryan regresses |
| Atomwise MMA plus compensation | EOS 59 | cap | Not run | Ryan regresses |
| FP32 output heads | EOS 53 | cap | Not run | Ryan regresses |
| FP64 projection accumulation | EOS 49 | EOS 47 | cap | Robot regresses |
| Baseline, full text in initial prompt | cap | EOS 45 | EOS 46 | Vivian still fails |

EOS alone is not speech-quality acceptance. FP64 Vivian has WER 0, but Ryan has
WER .25 (project-name confusion), and robot says only “The robot is ready.”
(WER 2/3). Independent official diagnostics using FP64 Linear operations or
FP32 output heads complete both acceptance voices, but are altered-precision
controls, not canonical references or evidence of a robust cuda-oxide fix.

Controlled evidence narrows the cause without identifying one decisive repair:

- The 24-call official-input replay passes all 2,997 tensor limits, but baseline
  matches only 37/39 required decisions; block-128 matches 38/39 and still fails.
- Official Talker replay on captured Rust inputs/history matches 20/21 choices
  and also repeats the code. Earlier generated state already carries the failure;
  an isolated local Talker decision mismatch does not explain the entire rollout.
- CPU reconstruction reproduces all 23 saved official decode inputs bit-for-bit
  from embeddings and text rows. Actual Rust prompt projections still differ
  slightly; this control does not establish complete upstream equivalence.
- Early zero-margin decisions differ between successful official math/auto runs.
  Improved dot-product accuracy or a matching short fixture is insufficient to
  establish full-generation quality. No tie epsilon or relaxed gate is justified.

The pinned official checkpoint defaults both Talker and predictor to sampling:
temperature .9, top-k 50, top-p 1, and cap 8192. Talker repetition penalty is
1.05; the predictor has no repetition penalty. Its public
`generate_custom_voice` API defaults to full-text prompt mode and recommends
sampling. Explicit caller options override those defaults. Our benchmark and
reference exporter intentionally force both stages greedy with streaming-text
prompt preparation. Rust likewise selects greedy when `sampling` is absent.
Vivian is speaker embedding ID 3065, Ryan 3061; neither has a dialect override or
speaker-specific decoding policy. Sampling is a separate generation policy; a successful sampled result does
not resolve the failed greedy check.

The unchanged production binary also reproduced the successful seed-0 Vivian
sampling result exactly: 56 frames, 4.48 seconds of audio, identical codec IDs
and waveform hash. One invocation took 1.228 seconds including waveform
decoding; this is not a warm benchmark. Sampling currently uses synchronous
predictor execution, whereas greedy uses a CUDA Graph.

ASR uses pinned Whisper base.en with greedy decoding and fixed normalization.
It can mishear proper nouns and does not measure naturalness or speaker identity;
WER is intelligibility evidence alongside EOS/duration, not a listening verdict.

Local evidence root: `/tmp/qwen-vivian-investigation/`. See `findings.md` for
detail; `baseline.json`, `controlled-24.json`, `official-on-rust/manifest.json`,
`quality/`, `atomwise-*.json`, `fp32-heads-*.json`, `fp64-*.json`, `full-text/`,
`vivian-asr.json`, `quality-asr.json`, `precision-asr.json`, and `fp64-robot-asr.json`.
The robot baseline is `/tmp/qwen3-tts-acceptance/benchmarks/optimization2-quality/robot.json`.
Reports retain requests, hashes and execution policy; local artifacts are not
portable qualification claims, and these runs are not a performance benchmark.

To reproduce the preserved baseline and ASR, run from the repository root with
the pinned local model/environment available, choosing unused output names:

```bash
CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  QWEN_TALKER_MODE=eager QWEN_PREDICTOR_MODE=graph \
  /tmp/qwen-vivian-investigation/baseline-binary --generate \
  /tmp/qwen3-tts-acceptance/model \
  /tmp/qwen3-tts-acceptance/request-vivian-full.json /tmp/vivian-baseline-rerun.wav
CUDA_VISIBLE_DEVICES=GPU-04a6d226-f1d6-0fc8-a95a-3ede21981e72 \
  /tmp/qwen3-tts-acceptance/venv/bin/python \
  cuda-oxide/cute-rs-preview/models/qwen_decoder_cute/reference/assess_audio.py \
  --asr-dir /tmp/qwen3-tts-acceptance/asr \
  --request /tmp/qwen3-tts-acceptance/request-vivian-full.json \
  --wav /tmp/vivian-baseline-rerun.wav --output /tmp/vivian-baseline-rerun-asr.json
```

The generation command writes its report beside the WAV. Preserved baseline
binary SHA-256: `73534da1bb6a70cd5561d9edbfd4116b0b56b7dfa4baf9d4e06398e2d21d0be9`.
Model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`; official source:
`022e286b98fbec7e1e916cb940cdf532cd9f488e`. Source behavior is recorded in pinned
`qwen3_tts_model.py:287,732`, `modeling_qwen3_tts.py:2022,2089`, and repository
`src/tts.rs:60`, `reference/benchmark.py:192,221`, `reference/export_reference.py:308`.
