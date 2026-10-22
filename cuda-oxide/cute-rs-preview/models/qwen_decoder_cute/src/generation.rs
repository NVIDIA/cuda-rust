/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed text, Talker and code-predictor boundaries for deterministic validation.
//!
//! Dimensions and operation order follow the pinned Qwen3-TTS CustomVoice
//! configuration. This module consumes loaded weights and prepared prompt rows;
//! it does not load a checkpoint or format a speech prompt. Host selection is
//! supplied by a separately checked, vocabulary-typed sampler.

use crate::model::{Batch, DecoderCache, Runner};
use crate::operator::Result;
use crate::rotary::RotarySpec;
use crate::sampling::{Sampler, SamplingConfig};
use cute_rs::BFloat16;
use cute_rs::decoder::{
    BiasWeights, DecoderWeights, EmbeddingWeights, FinalNorm, NormWeights, TokenIds,
};
use cute_rs::ffn::{InputProjection, Logits, ProjectionWeights, TextProjection1, TextProjection2};

/// Decoder layer count in the pinned 2048-wide Talker.
pub const TALKER_LAYERS: usize = 28;
/// Decoder layer count in the pinned 1024-wide code predictor.
pub const PREDICTOR_LAYERS: usize = 5;
/// Code groups predicted after code zero, producing sixteen codes per codec frame.
pub const REMAINING_CODES: usize = 15;

/// The Talker preserves 2048-wide hidden states across 28 decoder layers.
pub struct TalkerWeights<'a> {
    /// Ordered Talker layers; execution checks that all 28 are present.
    pub layers: Vec<DecoderWeights<'a, 2048, 6144, 2048, 1024, 128>>,
    /// Final RMS normalization scale, with one BF16 value per hidden channel.
    pub norm: NormWeights<'a, FinalNorm, 2048>,
    /// Dense `[3072, 2048]` projection from the final hidden row to code-zero logits.
    pub codec_head: ProjectionWeights<'a, Logits, 2048, 3072>,
    /// 3072-row codec table, each row containing 2048 BF16 hidden values.
    pub codec_embedding: EmbeddingWeights<'a, 3072, 2048>,
}

/// Predictor embeddings retain the Talker's width, then a biased projection
/// maps 2048 to 1024 before its five decoder layers. Each code group has its
/// own output head and embedding table.
pub struct PredictorWeights<'a> {
    /// Ordered predictor layers; execution requires all five layers.
    pub layers: Vec<DecoderWeights<'a, 1024, 3072, 2048, 1024, 128>>,
    /// Final predictor RMS normalization scale over 1024 hidden values.
    pub norm: NormWeights<'a, FinalNorm, 1024>,
    /// Dense `[1024, 2048]` adapter applied before the predictor stack.
    pub projection: ProjectionWeights<'a, InputProjection, 2048, 1024>,
    /// 1024-channel adapter bias, added before the projection result narrows to BF16.
    pub bias: BiasWeights<'a, InputProjection, 1024>,
    /// Per-group tables for codes one through fifteen, each `[2048, 2048]`.
    pub embeddings: [EmbeddingWeights<'a, 2048, 2048>; REMAINING_CODES],
    /// Per-group `[2048, 1024]` logits projections in code-generation order.
    pub heads: [ProjectionWeights<'a, Logits, 1024, 2048>; REMAINING_CODES],
}

/// Text tokenizer rows pass through a biased two-layer SiLU adapter before
/// participating in the Talker's prompt or per-frame input.
pub struct TextWeights<'a, const VOCAB: usize = 151936> {
    /// Token-indexed BF16 text rows with the vocabulary extent fixed by `VOCAB`.
    pub embedding: EmbeddingWeights<'a, VOCAB, 2048>,
    /// First 2048-to-2048 projection, followed by BF16 SiLU activation.
    pub first: ProjectionWeights<'a, TextProjection1, 2048, 2048>,
    /// First projection's 2048-channel bias, applied before BF16 output rounding.
    pub first_bias: BiasWeights<'a, TextProjection1, 2048>,
    /// Second 2048-to-2048 projection yielding Talker-compatible text rows.
    pub second: ProjectionWeights<'a, TextProjection2, 2048, 2048>,
    /// Second projection's 2048-channel bias, applied before BF16 output rounding.
    pub second_bias: BiasWeights<'a, TextProjection2, 2048>,
}

/// Only the final input row predicts the next code-zero token.
pub struct TalkerStep {
    /// One final-normalized 2048-value row, also used to seed the code predictor.
    pub last_hidden: Batch<2048>,
    /// One BF16 row of 3072 code-zero scores, before host suppression or sampling.
    pub logits: Batch<3072>,
}

/// The caller already supplied code zero. These are codes one through fifteen,
/// plus the sum of all sixteen embedding vectors for the next Talker input.
pub struct CodecFrame {
    /// Selected indices for code groups one through fifteen, each in `0..2048`.
    pub remaining_codes: [u32; REMAINING_CODES],
    /// One 2048-wide row: FP32 sum of sixteen BF16 embeddings, narrowed once to BF16.
    pub embedding_sum: Batch<2048>,
}

/// Reject incomplete or mismatched layer stacks before model execution.
fn check_layers(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(format!("model requires {expected} decoder layers, received {actual}").into());
    }
    Ok(())
}

/// Exact position schedule for a fresh code predictor invocation. Hidden state
/// and code-zero embedding occupy rows zero and one. Codes one through fourteen
/// are subsequently appended; code fifteen ends the frame without another append.
fn predictor_positions(step: usize) -> Result<Vec<[u32; 3]>> {
    if step >= REMAINING_CODES {
        return Err("predictor step exceeds the fifteen remaining code groups".into());
    }
    Ok(if step == 0 {
        vec![[0; 3], [1; 3]]
    } else {
        vec![[(step + 1) as u32; 3]]
    })
}

/// Deterministic validation policy. Nonfinite logits are errors, and the lowest
/// vocabulary index wins a tie. This does not reproduce stochastic sampling.
pub fn greedy_token<const VOCAB: usize>(logits: &[BFloat16]) -> Result<u32> {
    if VOCAB == 0 || VOCAB > u32::MAX as usize || logits.len() != VOCAB {
        return Err("greedy logits must match a nonempty u32-indexed vocabulary".into());
    }
    let mut best_value = f32::NEG_INFINITY;
    let mut best_index = 0;
    for (index, value) in logits.iter().enumerate() {
        let value = value.to_f32();
        if !value.is_finite() {
            return Err(format!("nonfinite logit at vocabulary index {index}").into());
        }
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    Ok(best_index as u32)
}

/// Embed checked text IDs and apply linear+bias, SiLU, then linear+bias.
///
/// Returns one dense 2048-wide BF16 row per token, retaining each intermediate
/// BF16 rounding boundary. Tensor validation and CUDA errors propagate.
pub fn text_input<const VOCAB: usize>(
    runner: &Runner,
    weights: &TextWeights<'_, VOCAB>,
    ids: TokenIds<'_, VOCAB>,
) -> Result<Batch<2048>> {
    let embedded = runner.embed(&weights.embedding, ids)?;
    let first = runner.project(&embedded, &weights.first, Some(&weights.first_bias))?;
    let activated = runner.silu(&first)?;
    runner.project(&activated, &weights.second, Some(&weights.second_bias))
}

/// Append prepared prompt or generation rows to the persistent Talker cache.
/// Explicit three-axis positions permit the pinned interleaved MRoPE policy;
/// prompt formatting and the choice of positions belong to the caller.
/// If the output projection fails after the decoder succeeds, the cache retains
/// the completed prefix but is invalidated until reset.
pub fn talker_step(
    runner: &Runner,
    weights: &TalkerWeights<'_>,
    cache: &mut DecoderCache<2048, 6144, 8>,
    input: &Batch<2048>,
    positions: &[[u32; 3]],
) -> Result<TalkerStep> {
    check_layers(weights.layers.len(), TALKER_LAYERS)?;
    if input.rows() == 0 {
        return Err("a Talker prediction requires at least one input row".into());
    }
    if !std::sync::Arc::ptr_eq(
        weights.codec_head.buffer().context(),
        runner.stream().context(),
    ) {
        return Err("Talker codec head must share the runner context".into());
    }
    let last_row = u32::try_from(input.rows() - 1)?;
    let tables = RotarySpec::interleaved(1_000_000.0, 32768, [24, 20, 20])?.tables(positions)?;
    let hidden = runner.forward::<2048, 6144, 16, 8, 2048, 1024>(
        input,
        &weights.layers,
        &weights.norm,
        cache,
        &tables,
    )?;
    let result = (|| -> Result<TalkerStep> {
        let last_hidden = runner.select(&hidden, &[last_row])?;
        let logits = runner.project(&last_hidden, &weights.codec_head, None)?;
        Ok(TalkerStep {
            last_hidden,
            logits,
        })
    })();
    if result.is_err() {
        cache.invalidate();
    }
    result
}

/// Embed a checked code-zero token using the Talker's vocabulary table.
pub fn code_zero_input(
    runner: &Runner,
    weights: &TalkerWeights<'_>,
    code: u32,
) -> Result<Batch<2048>> {
    runner.embed(&weights.codec_embedding, TokenIds::new(&[code])?)
}

/// Generate one complete set of remaining codec groups with a fresh predictor
/// cache. Host logits readback is intentional for this correctness baseline.
pub fn predictor_greedy_frame(
    runner: &Runner,
    weights: &PredictorWeights<'_>,
    talker_last: &Batch<2048>,
    code_zero_embedding: &Batch<2048>,
) -> Result<CodecFrame> {
    let mut sampler = Sampler::new(SamplingConfig::<2048>::greedy()?, 0);
    predictor_frame(
        runner,
        weights,
        talker_last,
        code_zero_embedding,
        &mut sampler,
    )
}

/// Select fifteen predictor codes with explicit, request-owned sampler state.
///
/// Both input batches must contain one 2048-wide row. A fresh sixteen-position
/// cache starts with the Talker hidden row and code-zero embedding; each selected
/// group supplies the next input embedding. The returned sum accumulates all
/// sixteen embeddings in FP32 and narrows once. Layer/shape, CUDA, and selection
/// failures propagate; sampler draws already consumed are not rolled back.
pub fn predictor_frame(
    runner: &Runner,
    weights: &PredictorWeights<'_>,
    talker_last: &Batch<2048>,
    code_zero_embedding: &Batch<2048>,
    sampler: &mut Sampler<2048>,
) -> Result<CodecFrame> {
    check_layers(weights.layers.len(), PREDICTOR_LAYERS)?;
    if talker_last.rows() != 1 || code_zero_embedding.rows() != 1 {
        return Err("predictor frame requires one Talker row and one code-zero row".into());
    }
    let mut input = runner.concat(talker_last, code_zero_embedding)?;
    let mut codec_rows = runner.select(code_zero_embedding, &[0])?;
    let mut cache =
        DecoderCache::<1024, 3072, 8>::new(runner.stream(), PREDICTOR_LAYERS, REMAINING_CODES + 1)?;
    let rotary = RotarySpec::standard(1_000_000.0, 65536)?;
    let mut remaining_codes = [0; REMAINING_CODES];

    for (step, code) in remaining_codes.iter_mut().enumerate() {
        let tables = rotary.tables(&predictor_positions(step)?)?;
        let projected = runner.project(&input, &weights.projection, Some(&weights.bias))?;
        let hidden = runner.forward::<1024, 3072, 16, 8, 2048, 1024>(
            &projected,
            &weights.layers,
            &weights.norm,
            &mut cache,
            &tables,
        )?;
        let last = runner.select(&hidden, &[(hidden.rows() - 1) as u32])?;
        let logits = runner.project(&last, &weights.heads[step], None)?;
        *code = sampler.select_bf16(&logits.read(runner.stream())?)?;
        let embedded = runner.embed(&weights.embeddings[step], TokenIds::new(&[*code])?)?;
        codec_rows = runner.concat(&codec_rows, &embedded)?;
        input = embedded;
    }
    debug_assert_eq!(cache.len(), REMAINING_CODES + 1);
    let embedding_sum = runner.sum_rows(&codec_rows)?;
    Ok(CodecFrame {
        remaining_codes,
        embedding_sum,
    })
}

/// CPU-only checks for predictor positions, layer counts, and deterministic token selection.
#[cfg(test)]
mod tests {
    use super::*;
    use cute_rs::BFloat16;

    /// Round a scalar to BF16 for host token-selection fixtures.
    fn bf16(value: f32) -> BFloat16 {
        BFloat16::from_f32(value)
    }

    /// Verify two initial rows plus fourteen appends fill exactly sixteen predictor positions.
    #[test]
    fn one_frame_predicts_fifteen_codes_from_sixteen_cached_rows() {
        let mut cached = 0;
        for step in 0..15 {
            let positions = predictor_positions(step).unwrap();
            assert_eq!(positions.len(), if step == 0 { 2 } else { 1 });
            for position in positions {
                assert_eq!(position, [cached; 3]);
                cached += 1;
            }
        }
        assert_eq!(cached, 16);
        assert!(predictor_positions(15).is_err());
        assert!(predictor_positions(usize::MAX).is_err());
    }

    /// Check vocabulary width validation and lowest-index tie breaking.
    #[test]
    fn greedy_selection_returns_exact_vocab_indices_and_resolves_ties() {
        assert_eq!(
            greedy_token::<4>(&[bf16(-2.0), bf16(3.0), bf16(3.0), bf16(0.5)]).unwrap(),
            1
        );
        assert_eq!(
            greedy_token::<4>(&[bf16(-2.0), bf16(-1.0), bf16(-3.0), bf16(0.5)]).unwrap(),
            3
        );
        assert_eq!(greedy_token::<1>(&[bf16(-5.0)]).unwrap(), 0);
        assert!(greedy_token::<4>(&[bf16(1.0); 3]).is_err());
        assert!(greedy_token::<4>(&[bf16(1.0); 5]).is_err());
        assert!(greedy_token::<0>(&[]).is_err());
    }

    /// Ensure all raw logits are finite, including candidates that would not win.
    #[test]
    fn greedy_selection_rejects_nonfinite_logits_even_when_not_winning() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(greedy_token::<3>(&[bf16(4.0), bf16(value), bf16(-1.0)]).is_err());
        }
    }

    /// Reject partial or cross-model decoder stacks before launching work.
    #[test]
    fn full_model_layer_counts_are_enforced_before_work() {
        assert!(check_layers(28, 28).is_ok());
        assert!(check_layers(5, 5).is_ok());
        assert!(check_layers(1, 28).is_err());
        assert!(check_layers(0, 5).is_err());
        assert!(check_layers(28, 5).is_err());
    }
}
