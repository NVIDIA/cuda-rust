# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned local ASR intelligibility evidence for generated acceptance WAVs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import unicodedata


#: Official Whisper revision used for repeatable English transcription.
ASR_REVISION = "911407f4214e0e1d82085af863093ec0b66f9cd6"
#: Expected local safetensors bytes, checked independently of provenance.json.
ASR_WEIGHT_SHA256 = "d4dd5542fd6a1d35639e21384238f3bfe6c557c849d392b5905d33ee29e71db5"


def normalized(text):
    """Normalize Unicode and case, replacing punctuation with single word separators."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def edit_distance(reference, hypothesis):
    """Compute Levenshtein distance between sequences using one previous DP row."""
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, 1):
        current = [row]
        for column, right in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1, current[-1] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def transcript_metrics(reference, hypothesis):
    """Return normalized word and character error counts and rates.

    Character errors exclude spaces. An empty normalized reference is rejected;
    proper nouns receive no special correction."""
    reference, hypothesis = normalized(reference), normalized(hypothesis)
    if not reference:
        raise ValueError("reference must contain letters or numbers")
    words = reference.split()
    characters = reference.replace(" ", "")
    word_errors = edit_distance(words, hypothesis.split())
    character_errors = edit_distance(characters, hypothesis.replace(" ", ""))
    return {
        "normalized_reference": reference, "normalized_transcript": hypothesis,
        "word_errors": word_errors, "reference_words": len(words),
        "word_error_rate": word_errors / len(words),
        "character_errors": character_errors, "reference_characters": len(characters),
        "character_error_rate": character_errors / len(characters),
    }


def sha256(path):
    """Return the SHA-256 of a local artifact or waveform file."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def assess(args):
    """Transcribe local WAVs with pinned Whisper weights and record intelligibility.

    Requires finite mono 24 kHz audio of at most 30 seconds. Audio is resampled
    to 16 kHz for deterministic FP32 ASR. The JSON includes input/model hashes
    and WER/CER; these measure transcript agreement, not speaker similarity."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    provenance = json.loads((args.asr_dir / "provenance.json").read_text())
    if provenance["model"] != "openai/whisper-base.en" or provenance["revision"] != ASR_REVISION:
        raise ValueError("ASR provenance differs from the pinned checkpoint")
    for artifact in provenance["artifacts"]:
        path = args.asr_dir / artifact["file"]
        if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
            raise ValueError(f"ASR artifact hash mismatch: {path}")
    if sha256(args.asr_dir / "model.safetensors") != ASR_WEIGHT_SHA256:
        raise ValueError("ASR checkpoint hash differs from the pinned source")
    request = json.loads(args.request.read_text())
    reference = request["text"]
    transcript_metrics(reference, "")

    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.asr_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.asr_dir, local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    results = []
    with torch.inference_mode():
        for path in args.wav:
            waveform, rate = sf.read(path, dtype="float32")
            if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise ValueError(f"expected finite, nonempty mono audio: {path}")
            if rate != 24000 or len(waveform) > 30 * rate:
                raise ValueError("this acceptance ASR supports mono 24kHz WAVs up to 30 seconds")
            samples = resample_poly(waveform, up=2, down=3)
            features = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            tokens = model.generate(
                input_features=features.input_features.to("cuda"),
                attention_mask=features.attention_mask.to("cuda"),
                do_sample=False, num_beams=1, max_new_tokens=256,
            )
            transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            result = {
                "wav": str(path.resolve()), "wav_sha256": sha256(path),
                "sample_rate": rate, "samples": len(waveform),
                "audio_seconds": len(waveform) / rate,
                "peak_amplitude": float(np.max(np.abs(waveform))),
                "rms_amplitude": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                "transcript": transcript, **transcript_metrics(reference, transcript),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {
        "schema_version": 1, "request": request, "request_sha256": sha256(args.request),
        "assessment_script_sha256": sha256(Path(__file__)),
        "asr": provenance, "asr_dtype": "FP32", "asr_decoding": "greedy, beam=1, max_new_tokens=256",
        "resampling": "scipy.signal.resample_poly, 24000 -> 16000 Hz, up=2/down=3",
        "normalization": "Unicode NFKC/casefold; non-alphanumeric -> word boundaries; CER excludes spaces",
        "limitations": "ASR agreement is intelligibility evidence, not a human quality or speaker-similarity score. Proper nouns can be mistranscribed.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def self_test():
    # Wrong normalization or edit-distance denominators must change these results.
    """Check transcript normalization and edit-distance metrics without running ASR."""
    exact = transcript_metrics("Hello, CUDA-oxide!", "hello cuda oxide")
    assert exact is not None and exact["word_error_rate"] == 0
    assert exact["character_error_rate"] == 0
    substitution = transcript_metrics("cat", "cut")
    assert substitution["word_error_rate"] == 1
    assert math.isclose(substitution["character_error_rate"], 1 / 3)
    deletion = transcript_metrics("one two", "one")
    assert deletion["word_error_rate"] == 0.5
    assert deletion["character_error_rate"] == 0.5
    assert transcript_metrics("cat", "cat cat cat")["word_error_rate"] == 2
    assert transcript_metrics("cat", "")["word_error_rate"] == 1
    try:
        transcript_metrics("!!!", "words")
    except ValueError:
        pass
    else:
        raise AssertionError("empty normalized reference accepted")
    print("ASR metric tests passed")


def main():
    """Parse checkpoint/request/WAV paths and run metric checks or ASR assessment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--asr-dir", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--wav", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, key) is None for key in ["asr_dir", "request", "wav", "output"]):
        parser.error("--asr-dir, --request, --wav and --output are required")
    assess(args)


if __name__ == "__main__":
    main()
