# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenize one local CustomVoice request; all model execution stays in Rust.

The wrapper strings match Qwen3TTSModel at source revision
022e286b98fbec7e1e916cb940cdf532cd9f488e. This helper needs transformers and
the local tokenizer files. It never loads or executes model weights.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    """Validate text and sampling options, then write a local tokenized request.

    The assistant/user wrappers match the pinned official implementation. Only
    tokenizer files are loaded; model weights and GPU inference are not used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="Ryan")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--non-streaming", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true", help="use the pinned Talker/predictor sampling defaults")
    parser.add_argument("--seed", type=int, help="u64 sampling seed; defaults to zero with --sample")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("text must not be empty")
    if not 2 <= args.max_new_tokens <= 4096:
        parser.error("max-new-tokens must be between 2 and 4096")
    if args.seed is not None and (not args.sample or not 0 <= args.seed < 2**64):
        parser.error("--seed requires --sample and a value in [0,2**64)")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    text = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    request = {
        "text": args.text,
        "text_ids": tokenizer(text, add_special_tokens=False)["input_ids"],
        "speaker": args.speaker,
        "language": args.language,
        "non_streaming_mode": args.non_streaming,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.sample:
        request["sampling"] = {
            "seed": 0 if args.seed is None else args.seed,
            "talker": {"temperature": 0.9, "top_k": 50, "top_p": 1.0},
            "predictor": {"temperature": 0.9, "top_k": 50, "top_p": 1.0},
        }
    if args.instruct:
        instruction = f"<|im_start|>user\n{args.instruct}<|im_end|>\n"
        request["instruct_ids"] = tokenizer(instruction, add_special_tokens=False)[
            "input_ids"
        ]
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(request['text_ids'])} text tokens)")


if __name__ == "__main__":
    main()
