/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CLI entry point for validating and benchmarking `Qwen/Qwen3-8B` (BF16 & NVFP4) on RTX 5090.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

use qwen3_8b_cute::benchmark::run_benchmark;
use qwen3_8b_cute::config::{HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, Q_HEADS};
use std::path::PathBuf;

/// Parse CLI arguments and execute the `Qwen3-8B` BF16 + NVFP4 validation and benchmark suite.
fn main() -> qwen3_8b_cute::operator::Result<()> {
    let mut num_layers = 4usize;
    let mut prompt_len = 16usize;
    let mut decode_steps = 8usize;
    let mut checkpoint: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--layers" => {
                num_layers = args
                    .next()
                    .ok_or("--layers requires a positive integer")?
                    .parse()?;
            }
            "--full" => {
                num_layers = LAYERS;
            }
            "--prompt-len" => {
                prompt_len = args
                    .next()
                    .ok_or("--prompt-len requires a positive integer")?
                    .parse()?;
            }
            "--decode-steps" => {
                decode_steps = args
                    .next()
                    .ok_or("--decode-steps requires a positive integer")?
                    .parse()?;
            }
            "--checkpoint" => {
                checkpoint = Some(PathBuf::from(
                    args.next().ok_or("--checkpoint requires a path")?,
                ));
            }
            other => {
                return Err(format!(
                    "unknown argument `{other}` (expected --layers N, --full, --prompt-len N, --decode-steps N, or --checkpoint PATH)"
                )
                .into());
            }
        }
    }

    println!(
        "Qwen/Qwen3-8B (cute-rs) | H={HIDDEN}, I={INTERMEDIATE}, Q={Q_HEADS}, KV={KV_HEADS}, layers={num_layers}/{LAYERS}"
    );
    let report = run_benchmark(num_layers, prompt_len, decode_steps, checkpoint.as_deref())?;

    for mode in [&report.bf16, &report.nvfp4] {
        println!(
            "  [{:<32}] weights: {:>7.2} MiB | prefill ({} tok): {:>7.2} ms ({:>7.1} tok/s) | decode ({} steps): {:>6.2} ms/tok ({:>6.1} tok/s)",
            mode.label,
            mode.weight_bytes as f64 / (1024.0 * 1024.0),
            mode.prompt_tokens,
            mode.prefill_ms,
            mode.prefill_tok_per_sec,
            mode.decode_steps,
            mode.decode_ms_per_tok,
            mode.decode_tok_per_sec,
        );
    }
    println!(
        "  weight compression: {:.2}x | prefill BF16-vs-NVFP4 cosine similarity: {:.6}",
        report.compression_ratio, report.cosine_similarity
    );
    Ok(())
}
