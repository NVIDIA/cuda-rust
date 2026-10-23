/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed Qwen Talker, code predictor and waveform decoder with reference checks.
// Share the already validated kernels and owned-cache implementation. These
// source modules compile into this binary's own embedded device artifact.
#[path = "../../../examples/rope_cache_cute/src/attention.rs"]
/// Checked attention launches over the completed key/value cache prefix.
mod attention;
#[path = "../../../examples/rope_cache_cute/src/attention_kernel.rs"]
/// Shared CuTe kernels for split attention and its stable merge.
mod attention_kernel;
#[path = "../../../examples/rope_cache_cute/src/attention_reference.rs"]
/// Independent CPU attention calculations used by numerical checks.
mod attention_reference;
/// Bounded checkpoint reads and owned, shape-checked model weights.
mod checkpoint;
/// Pinned artifact verification and stage-by-stage model comparisons.
mod checkpoint_validation;
/// Sliding-window attention used by the waveform decoder.
mod codec_attention;
/// CuTe convolution and elementwise kernels for waveform reconstruction.
mod codec_kernel;
/// Complete audio-code-to-waveform decoder and its checkpoint weights.
mod codec_model;
/// Typed waveform-decoder operators and their numerical fixtures.
mod codec_ops;
/// Comparison of waveform-decoder stages with saved reference tensors.
mod codec_validation;
#[cfg(test)]
/// Source checks that keep workload kernels within the CuTe abstraction boundary.
mod cute_boundary;
/// Width-aware projection dispatch and prepared decode launches.
mod decode_linear;
/// Resident-input projection microbenchmarks, separate from model timing.
mod decode_linear_bench;
/// Shared typed scratch and ordered layer submission for private execution regions.
mod decoder_execution;
/// CuTe kernels for embeddings, biased projections and row operations.
mod element_kernel;
#[path = "../../../examples/ffn_cute/src/kernel.rs"]
/// Shared CuTe projection and activation kernels from the feed-forward example.
mod ffn_kernel;
/// Typed Talker and predictor composition and audio-code selection.
mod generation;
/// Full-depth synthetic tests of text adapters and code-predictor wiring.
mod generation_validation;
#[path = "../../../examples/rope_cache_cute/src/kernel.rs"]
/// Shared rotary-position and cache-append kernels.
mod kernel;
/// Stage-level and complete-layer decoder comparisons at model and tail widths.
mod layer_validation;
/// Owned decoder batches, per-layer caches and synchronous model operations.
mod model;
#[path = "../../../examples/common/rmsnorm.rs"]
/// Shared generic and width-specialized BF16 normalization kernels.
mod norm_kernel;
/// Bit-preservation checks and isolated timings for specialized normalization.
mod norm_specialized;
#[path = "../../../examples/rope_cache_cute/src/operator.rs"]
/// Shared cache ownership, input readiness and checked launch helpers.
mod operator;
/// Persistent predictor storage and optional CUDA graph capture/replay.
mod predictor_graph;
/// CuTe kernels for predictor feedback, selection and pitched row storage.
mod predictor_graph_kernel;
/// Diagnostic JSON-line service exposing the decode projection kernels.
mod projection_worker;
/// Independent CPU decoder calculations for synthetic numerical checks.
mod reference;
/// Per-shape reduction plans mirroring the reference's cuBLAS kernel choice.
mod reference_linear;
/// CuTe kernels computing decode projections in the reference reduction order.
mod reference_linear_kernel;
#[path = "../../../examples/rope_cache_cute/src/rotary.rs"]
/// Validated host/device rotary tables with typed head widths.
mod rotary;
#[path = "../../../examples/rope_cache_cute/src/reference.rs"]
/// Independent CPU reference for rotary rounding and cache contents.
mod rotary_reference;
/// Validated greedy or seeded token selection with separate vocabulary types.
mod sampling;
/// Completed one-row Talker decode over private reusable typed storage.
mod talker_decode;
/// Request parsing, prompt preparation and complete-waveform generation or benchmarking.
mod tts;
/// Default synthetic GPU validation without checkpoint loading.
mod validation;
/// Mono float WAV serialization and its format checks.
mod wav;

/// Dispatch synthetic checks, pinned-reference validation, or speech generation.
///
/// No arguments run checkpoint-free GPU checks. `--generate` loads a model and
/// writes one complete waveform; `--benchmark` retains loaded weights across
/// warmups and measured requests. Invalid arguments and failed checks return an
/// error to the shell. See the usage string for each command's required paths.
fn main() -> operator::Result<()> {
    use std::path::Path;
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None=>validation::run(),
        Some("--decode-linear") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            decode_linear::validate(context.default_stream())
        }
        Some("--decode-linear-bench") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            decode_linear_bench::run(context.default_stream())
        }
        Some("--predictor-graph") if args.len()==3 => {
            let model_dir=Path::new(&args[2]);
            checkpoint_validation::verify_pinned_artifacts(model_dir)?;
            let context=cuda_core::CudaContext::new(0)?;
            let stream=context.default_stream();
            let weights=std::sync::Arc::new(checkpoint::QwenWeights::load(&stream,model_dir.join("model.safetensors"))?);
            predictor_graph::validate(stream,weights)
        }
        Some("--projection-worker") if args.len()==3 =>projection_worker::run(Path::new(&args[2])),
        Some("--talker-decode") if args.len()==3 => {
            let model_dir=Path::new(&args[2]);
            checkpoint_validation::verify_pinned_artifacts(model_dir)?;
            let context=cuda_core::CudaContext::new(0)?;
            let stream=context.default_stream();
            let weights=std::sync::Arc::new(checkpoint::QwenWeights::load(&stream,model_dir.join("model.safetensors"))?);
            talker_decode::validate(stream,weights)
        }
        Some("--norm-specialized") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            norm_specialized::validate(context.default_stream())
        }
        Some("--norm-specialized-bench") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            norm_specialized::benchmark(context.default_stream())
        }
        Some("--codec-ops") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            codec_ops::validate(context.default_stream())
        }
        Some("--codec-attention") if args.len()==2 => {
            let context=cuda_core::CudaContext::new(0)?;
            codec_attention::validate(context.default_stream())
        }
        Some("--checkpoint") if args.len()==4 =>checkpoint_validation::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("--codec") if args.len()==4 =>codec_validation::run(Path::new(&args[2]),Path::new(&args[3])),
        Some("--prompt") if args.len()==4 => {
            let reference=checkpoint_validation::Reference::open(Path::new(&args[3]))?;
            reference.verify_artifacts(Path::new(&args[2]))?;
            let context=cuda_core::CudaContext::new(0)?;
            let stream=context.default_stream();
            let weights=checkpoint::QwenWeights::load(&stream,Path::new(&args[2]).join("model.safetensors"))?;
            let runner=model::Runner::new(stream)?;
            tts::validate_prompt_with(&runner,&weights,&reference,Path::new(&args[2]))
        }
        Some("--generate") if args.len()==5 =>tts::run(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("--benchmark") if args.len()==5 || args.len()==6 => {
            let repetitions=if args.len()==6 {args[5].parse()?} else {20};
            tts::benchmark(Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),repetitions)
        }
        _=>Err("usage: qwen_decoder_cute [--decode-linear | --decode-linear-bench | --norm-specialized | --norm-specialized-bench | --predictor-graph MODEL_DIR | --projection-worker MODEL_DIR | --talker-decode MODEL_DIR | --codec-ops | --codec-attention | --checkpoint MODEL_DIR REFERENCE_DIR | --codec MODEL_DIR REFERENCE_DIR | --prompt MODEL_DIR REFERENCE_DIR | --generate MODEL_DIR REQUEST_JSON OUTPUT_WAV | --benchmark MODEL_DIR REQUEST_JSON OUTPUT_DIR [REPETITIONS]]".into()),
    }
}
