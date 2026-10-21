# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Same-checkpoint, B=1 BF16 full-waveform benchmark and runner qualification."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def percentile(values, percent):
    """Return the nearest observed rank for a percentile in (0, 100].

    Reject empty or nonfinite samples and invalid percentiles. No interpolation
    is used, matching the Rust benchmark's median and tail-latency estimator."""
    if not values or not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("percentile requires observations and a percentage in (0,100]")
    values = sorted(values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("measurements must be finite")
    return values[max(0, math.ceil(len(values) * percent / 100) - 1)]


def summarize(runs):
    """Aggregate latency, real-time factor and codec-frame throughput.

    Each run supplies positive seconds, audio_seconds and frames. Real-time
    factor is generation time divided by audio duration; lower is better."""
    if not runs or any(
        run["seconds"] <= 0 or run["audio_seconds"] <= 0 or run["frames"] <= 0
        for run in runs
    ):
        raise ValueError("benchmark needs nonempty finite audio and positive timings")
    latency = [run["seconds"] for run in runs]
    rtf = [run["seconds"] / run["audio_seconds"] for run in runs]
    fps = [run["frames"] / run["seconds"] for run in runs]
    return {
        "percentile_estimator": "nearest observed rank: ceil(N * p) - 1, zero-based",
        "latency_seconds_p50": percentile(latency, 50),
        "latency_seconds_p95": percentile(latency, 95),
        "first_playable_audio_seconds_p50": percentile(latency, 50),
        "first_playable_audio_seconds_p95": percentile(latency, 95),
        "real_time_factor_p50": percentile(rtf, 50),
        "real_time_factor_p95": percentile(rtf, 95),
        "codec_frames_per_second_p50": percentile(fps, 50),
        "real_time_factor_definition": "synthesis_seconds / actual_audio_seconds; lower is better",
    }


def version(name):
    """Return installed package metadata, or None when the package is absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_package_source(root, package, expected):
    """Hash sorted package Python paths and contents, rejecting a changed source tree."""
    digest = hashlib.sha256()
    for path in sorted((root / package).rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if digest.hexdigest() != expected:
        raise ValueError(f"{package} source differs from the pinned benchmark candidate")
    return digest.hexdigest()


def benchmark(args):
    """Measure a pinned runner from host token IDs through the complete waveform.

    Arguments select official, graph, Triton or hybrid execution and attention
    backend. Source/checkpoint verification and loading happen before warmups;
    output hashing and WAV/JSON writing happen after synchronized inference.
    The report records deterministic-output checks but does not certify quality."""
    from export_reference import (
        ARTIFACT_HASHES, MODEL_REVISION, SOURCE_HASHES, SOURCE_REVISION,
        SOURCE_TREE_SHA256, source_tree_sha256, verify_artifacts,
    )

    request = json.loads(args.request.read_text())
    logical_cap = int(request["max_new_tokens"])
    if not 2 <= logical_cap <= 4096:
        raise ValueError("max_new_tokens must be in [2,4096]")
    if request.get("non_streaming_mode", False) and args.mode in ("faster", "hybrid"):
        raise ValueError("locked faster0.2.4 custom-voice path only supports the streaming-style prompt")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups >= 0 and repeats >= 1 required")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(args.output.parent / "triton-cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(args.output.parent / "inductor-cache"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = verify_artifacts(args.model_dir, ARTIFACT_HASHES)
    source_hashes = verify_artifacts(args.official_source, SOURCE_HASHES)
    if source_tree_sha256(args.official_source) != SOURCE_TREE_SHA256:
        raise ValueError("official source tree differs from the pinned reference")
    sys.path.insert(0, str(args.official_source))
    optimized_hashes = {}
    if args.mode in ("faster", "hybrid"):
        optimized_hashes["faster_qwen3_tts"] = verify_package_source(
            args.faster_source, "faster_qwen3_tts",
            "2d997551012e86d993853f6fa6cbc0a518573bb68949d5420316969eeb5782a3",
        )
        sys.path.insert(0, str(args.faster_source))
    if args.mode in ("triton", "hybrid"):
        optimized_hashes["qwen3_tts_triton"] = verify_package_source(
            args.triton_source / "src", "qwen3_tts_triton",
            "dccb819750047f419306b9fa301893792fa01c2f50c391cd12d3443ba636955e",
        )
        sys.path.insert(0, str(args.triton_source / "src"))

    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    adaptations = []
    required_capacity = len(request["text_ids"]) + len(request.get("instruct_ids", [])) + logical_cap + 16
    static_capacity = max(128, 1 << (required_capacity - 1).bit_length())
    if static_capacity > 32768:
        raise ValueError("request exceeds the pinned Talker position capacity")
    if args.mode in ("faster", "hybrid"):
        from faster_qwen3_tts import FasterQwen3TTS
        import faster_qwen3_tts.generate as fast_module

        accelerated = FasterQwen3TTS.from_pretrained(
            str(args.model_dir), device="cuda", dtype=torch.bfloat16,
            attn_implementation="sdpa", max_seq_len=static_capacity,
        )
        wrapper = accelerated.model
        accelerated.predictor_graph.do_sample = False
        # The released faster policy operates directly on BF16 logits. The
        # official generation processor first widens logits to FP32.
        original_penalty = fast_module.apply_repetition_penalty

        def fp32_penalty(logits, history, penalty):
            """Apply the existing repetition penalty after widening logits to FP32."""
            return original_penalty(logits.float(), history, penalty)

        fast_module.apply_repetition_penalty = fp32_penalty
        adaptations.extend([
            "predictor_graph.do_sample=False before CUDA graph capture",
            "FP32 repetition-penalty input, matching official generation policy",
            "native fast_generate cap=logical max_new_tokens-1 to match official frame count",
        ])
    else:
        wrapper = Qwen3TTSModel.from_pretrained(
            str(args.model_dir), device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )
    model = wrapper.model.eval()
    wrapper._validate_languages([request["language"]])
    wrapper._validate_speakers([request["speaker"]])
    if args.mode in ("triton", "hybrid"):
        from qwen3_tts_triton.models.patching import apply_triton_kernels

        apply_triton_kernels(model, enable_fused_norm=True, patch_range=(0, 24))
        adaptations.append("upstream hybrid default patch_range=(0,24), fused norm enabled, TurboQuant disabled")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("benchmark requires BF16 model weights")
    if next(model.speech_tokenizer.model.decoder.parameters()).dtype != torch.bfloat16:
        raise ValueError("benchmark requires the same BF16 waveform decoder")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    def inference():
        """Build the prompt, generate one complete code sequence and decode host audio.

        Returns a flat FP32 waveform, sample rate and device codec tensor. The locked
        faster runner receives a cap one below the logical cap to match completed
        frame counting; its first invocation also prepares CUDA graphs."""
        ids = [torch.tensor([request["text_ids"]], dtype=torch.long, device="cuda")]
        instruct = request.get("instruct_ids")
        instructions = [
            torch.tensor([instruct], dtype=torch.long, device="cuda") if instruct else None
        ]
        common = dict(
            max_new_tokens=logical_cap, min_new_tokens=2, do_sample=False,
            temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
        )
        if args.mode in ("faster", "hybrid"):
            tie, tam, tth, tpe = accelerated._build_talker_inputs_local(
                m=model, input_ids=ids, ref_ids=[None], voice_clone_prompt=None,
                languages=[request["language"]], speakers=[request["speaker"]],
                non_streaming_mode=False, instruct_ids=instructions,
            )
            if not accelerated._warmed_up:
                accelerated._warmup(tie.shape[1])
            model.talker.rope_deltas = None
            common["max_new_tokens"] = logical_cap - 1
            codes, _ = fast_module.fast_generate(
                talker=model.talker, talker_input_embeds=tie, attention_mask=tam,
                trailing_text_hiddens=tth, tts_pad_embed=tpe,
                config=model.config.talker_config,
                predictor_graph=accelerated.predictor_graph,
                talker_graph=accelerated.talker_graph,
                **common,
            )
            if codes is None:
                raise ValueError("optimized runner returned no codec frames")
        else:
            codes, _ = model.generate(
                input_ids=ids, instruct_ids=instructions,
                languages=[request["language"]], speakers=[request["speaker"]],
                non_streaming_mode=request.get("non_streaming_mode", False),
                subtalker_dosample=False, subtalker_temperature=0.9,
                subtalker_top_k=50, subtalker_top_p=1.0,
                **{key: value for key, value in common.items() if key != "min_new_tokens"},
            )
            codes = codes[0]
        wavs, sample_rate = model.speech_tokenizer.decode([{"audio_codes": codes}])
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), sample_rate, codes

    def attention_context():
        """Select the explicit math SDPA backend or leave backend selection automatic."""
        return sdpa_kernel(SDPBackend.MATH) if args.sdpa_backend == "math" else nullcontext()

    warmups = []
    runs = []
    first_waveform = None
    first_codes = None
    with torch.inference_mode(), attention_context():
        for iteration in range(args.warmups + args.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            waveform, sample_rate, codes = inference()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # Fixture copies and WAV/JSON serialization happen outside timing.
            codes = codes.detach().cpu().tolist()
            if sample_rate != 24000 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError("runner must produce finite, nonempty 24kHz audio")
            observation = {
                "seconds": elapsed, "audio_seconds": len(waveform) / sample_rate,
                "first_playable_audio_seconds": elapsed,
                "delivery_mode": "complete_waveform", "incremental_streaming": False,
                "samples": len(waveform), "frames": len(codes),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "codec_sha256": hashlib.sha256(np.asarray(codes, dtype="<i8").tobytes()).hexdigest(),
                "waveform_sha256": hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest(),
                "stopped_before_cap": len(codes) < logical_cap - 1,
            }
            if iteration < args.warmups:
                warmups.append(observation)
            else:
                runs.append(observation)
                if first_waveform is None:
                    first_waveform, first_codes = waveform.copy(), codes
            print(f"{args.mode} {'warmup' if iteration < args.warmups else 'measure'} {iteration}: {elapsed:.4f}s, {len(codes)} frames, {len(waveform)/sample_rate:.4f}s audio", flush=True)
    qualification = {
        "deterministic_codec_outputs": len({run["codec_sha256"] for run in runs}) == 1,
        "deterministic_waveforms": len({run["waveform_sha256"] for run in runs}) == 1,
        "reference_code_agreement": None,
        "reference_comparison": "not provided",
    }
    if args.reference_manifest:
        reference = json.loads(args.reference_manifest.read_text())
        comparable = all([
            reference["generation"]["max_new_tokens"] == logical_cap,
            reference["speaker"].lower() == request["speaker"].lower(),
            reference["language"].lower() == request["language"].lower(),
            reference["text"] == request.get("text"),
        ])
        qualification["reference_comparison"] = "same request" if comparable else "request/cap differs"
        if comparable:
            qualification["reference_code_agreement"] = first_codes == reference["codes"]
    qualification["full_measurement_count"] = args.warmups >= 3 and args.repeats >= 20
    # Greedy trajectories can diverge after a small numerical difference. An
    # exact code match is useful correctness evidence, but full speech quality
    # and paired timing conditions need a separate assessment before comparison.
    qualification["output_quality_assessment"] = "pending separate ASR and listening review"
    qualification["ready_for_speedup_claim"] = False
    properties = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1, "mode": args.mode,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION, "official_source_revision": SOURCE_REVISION,
        "optimized_source_revision": "c64873e6a792d4dabc7ae592933fefd740ff4d5f" if args.mode in ("triton", "hybrid") else None,
        "faster_wheel": "0.2.4" if args.mode in ("faster", "hybrid") else None,
        "faster_wheel_sha256": "aede5398034dc063feec82fd54e5289f2d8fb308a6667e388b1dafeb87f6d9a0" if args.mode in ("faster", "hybrid") else None,
        "request": request, "logical_max_new_tokens": logical_cap,
        "native_max_new_tokens": logical_cap - 1 if args.mode in ("faster", "hybrid") else logical_cap,
        "static_kv_capacity": static_capacity if args.mode in ("faster", "hybrid") else None,
        "timing_boundary": "token IDs on host -> complete host waveform; excludes text tokenizer, model load and WAV writing",
        "delivery_mode": "complete_waveform", "incremental_streaming": False,
        "first_playable_audio_seconds": runs[0]["seconds"],
        "streaming_latency_note": "first measured request; complete waveform is first playable output; distribution in summary",
        "gpu": properties.name, "compute_capability": [properties.major, properties.minor],
        "gpu_uuid": str(getattr(properties, "uuid", os.environ.get("CUDA_VISIBLE_DEVICES", "unreported"))),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_bytes": properties.total_memory, "cuda": torch.version.cuda,
        "dtype": "BF16", "kv_dtype": "BF16", "batch_size": 1,
        "sdpa_backend": args.sdpa_backend, "torch": torch.__version__,
        "environment": {name: version(name) for name in ["transformers", "triton", "numpy", "safetensors"]},
        "artifacts": artifacts, "official_source_hashes": source_hashes,
        "optimized_source_hashes": optimized_hashes,
        "adaptations": adaptations, "model_load_seconds": load_seconds,
        "cold_first_request_seconds": warmups[0]["seconds"] if warmups else runs[0]["seconds"],
        "warmups": warmups, "runs": runs, "summary": summarize(runs),
        "qualification": qualification, "first_codec_frames": first_codes,
    }
    import soundfile as sf

    sf.write(args.output.with_suffix(".wav"), first_waveform, 24000, subtype="FLOAT")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"], "qualification": qualification}, indent=2))


def self_test():
    """Check observed-rank percentiles and metric rejection without importing Torch."""
    assert percentile([1, 3, 2], 50) == 2
    assert percentile([1, 3, 2], 95) == 3
    assert percentile([3, 1, 4, 2], 50) == 2
    stats = summarize([
        {"seconds": 2.0, "audio_seconds": 4.0, "frames": 50},
        {"seconds": 4.0, "audio_seconds": 4.0, "frames": 50},
    ])
    assert stats["latency_seconds_p50"] == 2.0
    assert stats["real_time_factor_p50"] == 0.5
    assert stats["codec_frames_per_second_p50"] == 12.5
    for runs in [[], [{"seconds": 1.0, "audio_seconds": 0.0, "frames": 0}]]:
        try:
            summarize(runs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid measurements accepted")
    print("Benchmark metric tests passed")


def main():
    """Parse runner inputs and dispatch a self-test or the timed benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=["official", "faster", "triton", "hybrid"], default="official")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--triton-source", type=Path)
    parser.add_argument("--faster-source", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sdpa-backend", choices=["auto", "math"], default="auto")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ["model_dir", "request", "official_source", "output"]:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.mode in ("faster", "hybrid") and args.faster_source is None:
        parser.error("--faster-source is required for faster/hybrid")
    if args.mode in ("triton", "hybrid") and args.triton_source is None:
        parser.error("--triton-source is required for triton/hybrid")
    benchmark(args)


if __name__ == "__main__":
    main()
