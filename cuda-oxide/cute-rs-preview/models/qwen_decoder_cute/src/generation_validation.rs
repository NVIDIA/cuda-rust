/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Structured full-depth fixture: isolates model wiring from learned weights.
use crate::{
    generation::*,
    layer_validation::DeviceLayer,
    model::{Batch, DecoderCache, Runner},
    operator::Result,
    reference::{self, LayerWeights},
};
use cuda_core::DeviceBuffer;
use cute_rs::{
    BFloat16,
    decoder::{BiasWeights, EmbeddingWeights, NormWeights, TokenIds},
    ffn::ProjectionWeights,
};

/// Construct positive-zero BF16 storage for sparse fixture matrices.
fn zeros(len: usize) -> Vec<BFloat16> {
    vec![BFloat16::from_bits(0); len]
}
/// Construct unit BF16 values for identity RMSNorm weights and expectations.
fn ones(len: usize) -> Vec<BFloat16> {
    vec![reference::round(1.0); len]
}
/// Generate a codebook value depending on stage, selected ID and channel.
///
/// Alternating first/second channels steer the next predictor choice, exposing
/// wrong table selection or stale feedback.
fn codec_value(stage: usize, id: usize, column: usize) -> BFloat16 {
    let base = (stage + 1 + id % 17) as f64 / 16.0;
    reference::round(match column {
        0 => base + if stage.is_multiple_of(2) { 2.0 } else { 0.0 },
        1 => base + if stage.is_multiple_of(2) { 0.0 } else { 2.0 },
        _ => base / 4.0,
    })
}
/// Build D128 Q16/KV8 weights with zero projections and identity residual paths.
///
/// Hidden and intermediate widths select Talker or predictor storage extents.
fn identity_layer(hidden: usize, intermediate: usize) -> LayerWeights {
    LayerWeights {
        input_norm: ones(hidden),
        query: zeros(2048 * hidden),
        key: zeros(1024 * hidden),
        value: zeros(1024 * hidden),
        output: zeros(hidden * 2048),
        q_norm: ones(128),
        k_norm: ones(128),
        post_norm: ones(hidden),
        gate: zeros(intermediate * hidden),
        up: zeros(intermediate * hidden),
        down: zeros(hidden * intermediate),
    }
}
/// Assert equal lengths and BF16 bits, reporting the named failing coordinate.
fn exact(actual: &[BFloat16], expected: &[BFloat16], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{label}[{i}]");
    }
}

/// Validate text adaptation, all 28 Talker layers and two 15-code predictor frames.
///
/// Structured identity-residual weights make expected outputs exact while
/// exercising full model wiring, fresh predictor caches, codebook feedback,
/// context rejection and Talker capacity rejection. CUDA errors propagate;
/// numerical and state invariants use assertions.
pub fn run(runner: &Runner) -> Result<()> {
    let stream = runner.stream();
    // Tiny vocabulary, actual text hidden width and both biased dense layers.
    let text_table: Vec<_> = (0..3)
        .flat_map(|row| vec![reference::round((row + 1) as f64 / 4.0); 2048])
        .collect();
    let text_table_dev = DeviceBuffer::from_host(stream, &text_table)?;
    let mut first = zeros(2048 * 2048);
    let mut second = zeros(2048 * 2048);
    for i in 0..2048 {
        first[i * 2048 + i] = reference::round(1.0);
        second[i * 2048 + i] = reference::round(2.0);
    }
    let first_dev = DeviceBuffer::from_host(stream, &first)?;
    let second_dev = DeviceBuffer::from_host(stream, &second)?;
    let first_bias = DeviceBuffer::from_host(stream, &ones(2048))?;
    let second_bias = DeviceBuffer::from_host(stream, &vec![reference::round(0.5); 2048])?;
    let text = TextWeights::<3> {
        embedding: EmbeddingWeights::new(&text_table_dev, 2048)?,
        first: ProjectionWeights::new(&first_dev, 2048)?,
        first_bias: BiasWeights::new(&first_bias)?,
        second: ProjectionWeights::new(&second_dev, 2048)?,
        second_bias: BiasWeights::new(&second_bias)?,
    };
    let prepared = text_input(runner, &text, TokenIds::new(&[2, 0])?)?;
    let expected: Vec<_> = [1.75, 1.25]
        .into_iter()
        .flat_map(|v| {
            let activated = reference::silu(&[reference::round(v)])[0];
            vec![reference::round(2.0 * reference::widen(activated) + 0.5); 2048]
        })
        .collect();
    exact(&prepared.read(stream)?, &expected, "text adapter");

    // A repeated identity-residual layer makes full 28/5 depth inexpensive to
    // provision and gives exact predictions. Nonzero decoder math is tested
    // independently by the full-width layer oracle in layer_validation.
    let talker_layer = DeviceLayer::new(stream, &identity_layer(2048, 6144))?;
    let talker_norm = DeviceBuffer::from_host(stream, &ones(2048))?;
    let mut codec_head = zeros(3072 * 2048);
    codec_head[7 * 2048] = reference::round(1.0);
    let codec_head_dev = DeviceBuffer::from_host(stream, &codec_head)?;
    let mut codec_table: Vec<_> = (0..3072)
        .flat_map(|row| vec![reference::round((row + 1) as f64 / 8.0); 2048])
        .collect();
    codec_table[7 * 2048 + 1] = reference::round(2.0);
    let codec_table_dev = DeviceBuffer::from_host(stream, &codec_table)?;
    let talker = TalkerWeights {
        layers: (0..TALKER_LAYERS)
            .map(|_| talker_layer.weights())
            .collect::<Result<_>>()?,
        norm: NormWeights::new(&talker_norm)?,
        codec_head: ProjectionWeights::new(&codec_head_dev, 2048)?,
        codec_embedding: EmbeddingWeights::new(&codec_table_dev, 2048)?,
    };
    let mut talker_cache = DecoderCache::new(stream, TALKER_LAYERS, 3)?;
    // Reject a mismatched output-head context before any of the 28 layers append.
    let other_context = cuda_core::CudaContext::new(0)?;
    let other_stream = other_context.default_stream();
    other_context.bind_to_thread()?;
    let other_head = DeviceBuffer::from_host(&other_stream, &codec_head)?;
    let invalid_talker = TalkerWeights {
        layers: (0..TALKER_LAYERS)
            .map(|_| talker_layer.weights())
            .collect::<Result<_>>()?,
        norm: NormWeights::new(&talker_norm)?,
        codec_head: ProjectionWeights::new(&other_head, 2048)?,
        codec_embedding: EmbeddingWeights::new(&codec_table_dev, 2048)?,
    };
    assert!(
        talker_step(
            runner,
            &invalid_talker,
            &mut talker_cache,
            &prepared,
            &[[0; 3], [1; 3]]
        )
        .is_err()
    );
    assert_eq!(talker_cache.len(), 0);
    assert!(!talker_cache.is_poisoned());
    let step = talker_step(
        runner,
        &talker,
        &mut talker_cache,
        &prepared,
        &[[0; 3], [1; 3]],
    )?;
    assert_eq!(talker_cache.len(), 2);
    let code = greedy_token::<3072>(&step.logits.read(stream)?)?;
    assert_eq!(code, 7);
    exact(
        &step.last_hidden.read(stream)?,
        &ones(2048),
        "full-depth Talker hidden",
    );
    let code_zero = code_zero_input(runner, &talker, code)?;
    let mut code_zero_values = ones(2048);
    code_zero_values[1] = reference::round(2.0);
    exact(
        &code_zero.read(stream)?,
        &code_zero_values,
        "code zero embedding",
    );

    let predictor_layer = DeviceLayer::new(stream, &identity_layer(1024, 3072))?;
    let predictor_norm = DeviceBuffer::from_host(stream, &ones(1024))?;
    let mut adapter_values = zeros(1024 * 2048);
    for column in 0..1024 {
        adapter_values[column * 2048 + column] = reference::round(1.0);
    }
    let adapter = DeviceBuffer::from_host(stream, &adapter_values)?;
    let adapter_bias = DeviceBuffer::from_host(stream, &vec![reference::round(0.25); 1024])?;
    let mut embedding_buffers = Vec::new();
    let mut head_buffers = Vec::new();
    for stage in 0..REMAINING_CODES {
        // Distinct tables and row-dependent values expose wrong table or ID
        // selection. Alternating first/second coordinates steer the next head.
        let embeddings: Vec<_> = (0..2048)
            .flat_map(|row| (0..2048).map(move |col| codec_value(stage, row, col)))
            .collect();
        embedding_buffers.push(DeviceBuffer::from_host(stream, &embeddings)?);
        let mut head = zeros(2048 * 1024);
        head[(stage + 1) * 1024] = reference::round(1.0);
        head[(stage + 101) * 1024 + 1] = reference::round(1.0);
        head_buffers.push(DeviceBuffer::from_host(stream, &head)?);
    }
    let predictor = PredictorWeights {
        layers: (0..PREDICTOR_LAYERS)
            .map(|_| predictor_layer.weights())
            .collect::<Result<_>>()?,
        norm: NormWeights::new(&predictor_norm)?,
        projection: ProjectionWeights::new(&adapter, 2048)?,
        bias: BiasWeights::new(&adapter_bias)?,
        embeddings: std::array::from_fn(|i| {
            EmbeddingWeights::new(&embedding_buffers[i], 2048).unwrap()
        }),
        heads: std::array::from_fn(|i| ProjectionWeights::new(&head_buffers[i], 1024).unwrap()),
    };
    let frame = predictor_greedy_frame(runner, &predictor, &step.last_hidden, &code_zero)?;
    // The first decision must use code zero (coordinate 1 > coordinate 0),
    // rather than the preceding uniform Talker row. Subsequent tables alternate.
    let expected_codes =
        std::array::from_fn::<_, 15, _>(|i| i as u32 + if i.is_multiple_of(2) { 101 } else { 1 });
    assert_eq!(frame.remaining_codes, expected_codes);
    let expected_sum: Vec<_> = (0..2048)
        .map(|col| {
            reference::round(
                reference::widen(code_zero_values[col])
                    + expected_codes
                        .iter()
                        .enumerate()
                        .map(|(stage, &id)| reference::widen(codec_value(stage, id as usize, col)))
                        .sum::<f64>(),
            )
        })
        .collect();
    exact(
        &frame.embedding_sum.read(stream)?,
        &expected_sum,
        "sixteen-code embedding sum",
    );
    let text_padding = Batch::<2048>::from_host(stream, 1, &vec![reference::round(0.5); 2048])?;
    let next = runner.add(&frame.embedding_sum, &text_padding)?;
    let next_step = talker_step(runner, &talker, &mut talker_cache, &next, &[[2; 3]])?;
    assert_eq!(talker_cache.len(), 3);
    assert!(!talker_cache.is_poisoned());
    assert_eq!(greedy_token::<3072>(&next_step.logits.read(stream)?)?, 7);
    let second_frame =
        predictor_greedy_frame(runner, &predictor, &next_step.last_hidden, &code_zero)?;
    assert_eq!(second_frame.remaining_codes, frame.remaining_codes);
    exact(
        &second_frame.embedding_sum.read(stream)?,
        &frame.embedding_sum.read(stream)?,
        "fresh predictor frame cache",
    );
    assert!(talker_step(runner, &talker, &mut talker_cache, &next, &[[3; 3]]).is_err());
    assert_eq!(talker_cache.len(), 3);
    assert!(!talker_cache.is_poisoned());
    println!(
        "Qwen structured generation: text adapter, 28-layer Talker prefill/decode, two fresh 5-layer/15-code predictor frames passed"
    );
    Ok(())
}
