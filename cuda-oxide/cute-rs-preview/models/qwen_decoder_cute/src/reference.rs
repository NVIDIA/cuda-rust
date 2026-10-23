/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent FP64 decoder oracle with explicit BF16 storage boundaries.
use cute_rs::BFloat16;

/// Owned host weights for a D128 decoder-layer oracle.
///
/// Matrices use dense `[output, input]` storage; H is hidden width, I is
/// intermediate width, QW/KW are query/key-value head counts times 128.
#[derive(Clone)]
pub struct LayerWeights {
    /// H input RMSNorm weights.
    pub input_norm: Vec<BFloat16>,
    /// Query projection matrix `[QW, H]`.
    pub query: Vec<BFloat16>,
    /// Key projection matrix `[KW, H]`.
    pub key: Vec<BFloat16>,
    /// Value projection matrix `[KW, H]`.
    pub value: Vec<BFloat16>,
    /// Attention output projection matrix `[H, QW]`.
    pub output: Vec<BFloat16>,
    /// 128 RMSNorm weights shared across query heads.
    pub q_norm: Vec<BFloat16>,
    /// 128 RMSNorm weights shared across key heads.
    pub k_norm: Vec<BFloat16>,
    /// H post-attention RMSNorm weights.
    pub post_norm: Vec<BFloat16>,
    /// FFN gate matrix `[I, H]`.
    pub gate: Vec<BFloat16>,
    /// FFN expansion matrix `[I, H]`.
    pub up: Vec<BFloat16>,
    /// FFN contraction matrix `[H, I]`.
    pub down: Vec<BFloat16>,
}

/// Dense BF16 snapshots at every observable layer storage boundary.
///
/// All buffers are row-major; widths follow the associated layer weights.
pub struct LayerTrace {
    /// Input RMSNorm output `[rows, H]`.
    pub normalized: Vec<BFloat16>,
    /// Query projection output `[rows, QW]`.
    pub q: Vec<BFloat16>,
    /// Key projection output `[rows, KW]`.
    pub k: Vec<BFloat16>,
    /// Unrotated value projection output `[rows, KW]`.
    pub v: Vec<BFloat16>,
    /// Per-head query RMSNorm output `[rows, QW]`.
    pub q_normed: Vec<BFloat16>,
    /// Per-head key RMSNorm output `[rows, KW]`.
    pub k_normed: Vec<BFloat16>,
    /// Rotary query output `[rows, QW]`.
    pub rotated_q: Vec<BFloat16>,
    /// Rotary key output `[rows, KW]`, also the logical K cache.
    pub rotated_k: Vec<BFloat16>,
    /// Causal grouped-query attention output `[rows, QW]`.
    pub attention: Vec<BFloat16>,
    /// Attention output projected to `[rows, H]`.
    pub projected: Vec<BFloat16>,
    /// First residual addition `[rows, H]`.
    pub residual: Vec<BFloat16>,
    /// Post-attention RMSNorm output `[rows, H]`.
    pub post_normed: Vec<BFloat16>,
    /// FFN gate projection `[rows, I]`.
    pub gate: Vec<BFloat16>,
    /// FFN up projection `[rows, I]`.
    pub up: Vec<BFloat16>,
    /// BF16 SiLU of the gate `[rows, I]`.
    pub silu: Vec<BFloat16>,
    /// Rounded SiLU-times-up product `[rows, I]`.
    pub product: Vec<BFloat16>,
    /// FFN down projection `[rows, H]`.
    pub down: Vec<BFloat16>,
    /// Final residual addition `[rows, H]`.
    pub output: Vec<BFloat16>,
}

/// Convert FP64 to FP32, then round to BF16 using round-to-nearest, ties-to-even.
///
/// NaNs are quieted while retaining their upper payload bits.
pub fn round(value: f64) -> BFloat16 {
    let value = value as f32;
    let bits = value.to_bits();
    let retained = (bits >> 16) as u16;
    if value.is_nan() {
        return BFloat16::from_bits(retained | 0x0040);
    }
    let discarded = bits & 0xffff;
    let increment = discarded > 0x8000 || (discarded == 0x8000 && retained & 1 != 0);
    BFloat16::from_bits(retained.wrapping_add(u16::from(increment)))
}

/// Widen BF16 storage exactly through FP32 to FP64, preserving signed zero.
pub fn widen(value: BFloat16) -> f64 {
    f64::from(f32::from_bits(u32::from(value.to_bits()) << 16))
}

/// Multiply row and width counts, panicking if the reference extent overflows.
fn extent(rows: usize, width: usize) -> usize {
    rows.checked_mul(width).expect("reference extent overflow")
}

/// Compute row statistics in FP64, then narrow before multiplying the BF16 weight.
/// Input/output are dense `[rows, width]`; weight has `width` elements.
///
/// Panics for mismatched extents, zero width, extent overflow or a nonfinite
/// or negative epsilon. The normalized value and weighted output each round.
pub fn norm(
    input: &[BFloat16],
    weight: &[BFloat16],
    rows: usize,
    width: usize,
    epsilon: f32,
) -> Vec<BFloat16> {
    assert!(width > 0, "normalization width must be positive");
    assert!(
        epsilon.is_finite() && epsilon >= 0.0,
        "invalid normalization epsilon"
    );
    assert_eq!(
        input.len(),
        extent(rows, width),
        "input storage length mismatch"
    );
    assert_eq!(weight.len(), width, "normalization weight length mismatch");
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let sum: f64 = row.iter().map(|&x| widen(x).powi(2)).sum();
        let inverse = (sum / width as f64 + f64::from(epsilon)).sqrt().recip();
        for (&x, &weight) in row.iter().zip(weight) {
            let normalized = round(widen(x) * inverse);
            output.push(round(widen(normalized) * widen(weight)));
        }
    }
    output
}

/// Dense weights are stored [output, input]; optional bias precedes output narrowing.
/// Compute an FP64 dot product per dense output, then round to BF16.
///
/// Input is `[rows, input_width]`; output is `[rows, output_width]`. Panics
/// for zero widths, extent overflow, or mismatched input/weight/bias storage.
pub fn linear(
    input: &[BFloat16],
    weight: &[BFloat16],
    rows: usize,
    input_width: usize,
    output_width: usize,
    bias: Option<&[BFloat16]>,
) -> Vec<BFloat16> {
    assert!(
        input_width > 0 && output_width > 0,
        "linear widths must be positive"
    );
    assert_eq!(
        input.len(),
        extent(rows, input_width),
        "input storage length mismatch"
    );
    assert_eq!(
        weight.len(),
        extent(output_width, input_width),
        "linear weight length mismatch"
    );
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_width, "linear bias length mismatch");
    }
    let mut output = Vec::with_capacity(extent(rows, output_width));
    for row in input.chunks_exact(input_width) {
        for (index, weight_row) in weight.chunks_exact(input_width).enumerate() {
            let dot: f64 = row
                .iter()
                .zip(weight_row)
                .map(|(&x, &w)| widen(x) * widen(w))
                .sum();
            let value = dot + bias.map_or(0.0, |b| widen(b[index]));
            output.push(round(value));
        }
    }
    output
}

/// Add equal-length BF16 arrays in FP64 and round each sum once.
///
/// Panics when the storage lengths differ.
pub fn add(left: &[BFloat16], right: &[BFloat16]) -> Vec<BFloat16> {
    assert_eq!(left.len(), right.len(), "residual storage length mismatch");
    left.iter()
        .zip(right)
        .map(|(&x, &y)| round(widen(x) + widen(y)))
        .collect()
}

/// Evaluate sign-stable FP64 SiLU independently and round each output to BF16.
pub fn silu(input: &[BFloat16]) -> Vec<BFloat16> {
    input
        .iter()
        .map(|&x| {
            let x = widen(x);
            let sigmoid = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            round(x * sigmoid)
        })
        .collect()
}

/// Gather width-sized rows in ID order, preserving duplicates and exact bits.
///
/// Panics for zero width, partial table rows, out-of-range IDs or extent overflow.
pub fn embedding(table: &[BFloat16], ids: &[u32], width: usize) -> Vec<BFloat16> {
    assert!(width > 0, "embedding width must be positive");
    assert!(
        table.len().is_multiple_of(width),
        "embedding table has a partial row"
    );
    let vocabulary = table.len() / width;
    let mut output = Vec::with_capacity(extent(ids.len(), width));
    for &id in ids {
        let id = usize::try_from(id).expect("embedding ID exceeds host address space");
        assert!(id < vocabulary, "embedding ID exceeds vocabulary");
        let start = extent(id, width);
        output.extend_from_slice(&table[start..start + width]);
    }
    output
}

/// One complete layer over an initially empty cache, with causal visibility.
///
/// The final stack RMSNorm and vocabulary head are intentionally outside this layer.
/// FP64 arithmetic is an accuracy oracle, not a bitwise PyTorch execution model.
/// Uses D128 heads and epsilon 1e-6. Input is `[rows, hidden]`; cosine and
/// sine tables are `[rows, 64]`. Panics for invalid widths/head grouping,
/// partial rows or inconsistent weight/table extents.
#[allow(clippy::too_many_arguments)]
pub fn layer(
    input: &[BFloat16],
    weights: &LayerWeights,
    hidden: usize,
    intermediate: usize,
    query_heads: usize,
    kv_heads: usize,
    cos: &[BFloat16],
    sin: &[BFloat16],
) -> LayerTrace {
    /// Coordinates per query/key/value head in the language decoder.
    const DIM: usize = 128;
    /// RMSNorm stabilizer used by language decoder layers.
    const EPSILON: f32 = 1e-6;
    assert!(
        hidden > 0 && intermediate > 0,
        "layer widths must be positive"
    );
    assert!(
        query_heads > 0 && kv_heads > 0,
        "head counts must be positive"
    );
    assert!(
        query_heads.is_multiple_of(kv_heads),
        "query heads must be a multiple of KV heads"
    );
    assert!(
        input.len().is_multiple_of(hidden),
        "layer input has a partial row"
    );
    let rows = input.len() / hidden;
    let query_width = extent(query_heads, DIM);
    let kv_width = extent(kv_heads, DIM);
    assert_eq!(
        cos.len(),
        extent(rows, DIM / 2),
        "cosine table length mismatch"
    );
    assert_eq!(sin.len(), cos.len(), "sine table length mismatch");
    let normalized = norm(input, &weights.input_norm, rows, hidden, EPSILON);
    let q = linear(&normalized, &weights.query, rows, hidden, query_width, None);
    let k = linear(&normalized, &weights.key, rows, hidden, kv_width, None);
    let v = linear(&normalized, &weights.value, rows, hidden, kv_width, None);
    let q_normed = norm(&q, &weights.q_norm, extent(rows, query_heads), DIM, EPSILON);
    let k_normed = norm(&k, &weights.k_norm, extent(rows, kv_heads), DIM, EPSILON);
    let rotated_q = crate::rotary_reference::rotate(&q_normed, cos, sin, query_heads, DIM);
    let rotated_k = crate::rotary_reference::rotate(&k_normed, cos, sin, kv_heads, DIM);
    let attention = crate::attention_reference::attention(
        &rotated_q,
        &rotated_k,
        &v,
        query_heads,
        kv_heads,
        DIM,
        rows,
        rows,
        Some(0),
    );
    let projected = linear(&attention, &weights.output, rows, query_width, hidden, None);
    let residual = add(input, &projected);
    let post_normed = norm(&residual, &weights.post_norm, rows, hidden, EPSILON);
    let gate = linear(
        &post_normed,
        &weights.gate,
        rows,
        hidden,
        intermediate,
        None,
    );
    let up = linear(&post_normed, &weights.up, rows, hidden, intermediate, None);
    let silu = silu(&gate);
    let product: Vec<_> = silu
        .iter()
        .zip(&up)
        .map(|(&x, &y)| round(widen(x) * widen(y)))
        .collect();
    let down = linear(&product, &weights.down, rows, intermediate, hidden, None);
    let output = add(&residual, &down);
    LayerTrace {
        normalized,
        q,
        k,
        v,
        q_normed,
        k_normed,
        rotated_q,
        rotated_k,
        attention,
        projected,
        residual,
        post_normed,
        gate,
        up,
        silu,
        product,
        down,
        output,
    }
}

/// Host-only tests of oracle layout and observable BF16 rounding boundaries.
#[cfg(test)]
mod tests {
    use super::*;

    /// Extract exact BF16 bit patterns for oracle regression assertions.
    fn bits(values: &[BFloat16]) -> Vec<u16> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    /// Distinguish normalized-value rounding from a fused weighted RMSNorm result.
    #[test]
    fn norm_has_two_rounding_boundaries_and_weights_each_coordinate() {
        assert_eq!(
            bits(&norm(
                &[3.0, 4.0].map(round),
                &[2.0, -1.0].map(round),
                1,
                2,
                12.5
            )),
            [0x3f9a, 0xbf4d]
        );
        let weights = [BFloat16::from_bits(0x3f71), round(1.0)];
        let actual = norm(&[1.0, 2.0].map(round), &weights, 1, 2, 0.0);
        assert_eq!(actual[0].to_bits(), 0x3f19);
        // Multiplying the weight before narrowing the normalized value gives 0x3f18.
        assert_eq!(round(widen(weights[0]) / 2.5_f64.sqrt()).to_bits(), 0x3f18);
    }

    /// Verify output-major weights and bias addition before output narrowing.
    #[test]
    fn linear_uses_output_by_input_weights_and_adds_bias_before_narrowing() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(round);
        let weights = [1.0, -2.0, 3.0, 2.0, 1.0, -1.0].map(round);
        let bias = [0.5, -2.0].map(round);
        let actual = linear(&input, &weights, 2, 3, 2, Some(&bias));
        assert_eq!(
            actual.into_iter().map(widen).collect::<Vec<_>>(),
            [6.5, -1.0, 12.5, 5.0]
        );
        let actual = linear(
            &[BFloat16::from_bits(0x3f70)],
            &[BFloat16::from_bits(0x3f78)],
            1,
            1,
            1,
            Some(&[BFloat16::from_bits(0xbb80)]),
        );
        assert_eq!(actual[0].to_bits(), 0x3f68);
        // Narrowing the matrix product before adding bias instead gives 0x3f67.
    }

    /// Check ties-to-even residual rounding and signed SiLU outputs.
    #[test]
    fn residual_add_rounds_even_ties_and_silu_keeps_sign() {
        let left = [1.0, 1.0078125, -2.0].map(round);
        let right = [0.00390625, 0.00390625, 3.0].map(round);
        assert_eq!(bits(&add(&left, &right)), [0x3f80, 0x3f82, 0x3f80]);
        assert_eq!(
            bits(&silu(&[0.0, 1.0, -1.0].map(round))),
            [0, 0x3f3b, 0xbe8a]
        );
    }

    /// Verify gather order and exact preservation of signed zero and NaN payloads.
    #[test]
    fn embedding_preserves_order_duplicates_and_storage_bits() {
        let table = [0x8000, 0x7fc1, 0x3f80, 0xbf80, 0x4000, 0x4040].map(BFloat16::from_bits);
        assert_eq!(
            bits(&embedding(&table, &[2, 0, 2], 2)),
            [0x4000, 0x4040, 0x8000, 0x7fc1, 0x4000, 0x4040]
        );
        assert!(embedding(&table, &[], 2).is_empty());
    }

    /// Reject a token ID past the final complete embedding row.
    #[test]
    #[should_panic(expected = "embedding ID exceeds vocabulary")]
    fn embedding_rejects_out_of_range_ids() {
        embedding(&[round(1.0); 4], &[2], 2);
    }

    /// Reject truncated dense input instead of silently shortening a dot product.
    #[test]
    #[should_panic(expected = "input storage length mismatch")]
    fn linear_rejects_truncated_input() {
        linear(&[round(1.0)], &[round(1.0); 2], 1, 2, 1, None);
    }

    /// Verify both residual paths and distinct query/KV widths with zero projections.
    #[test]
    fn zero_projection_layer_preserves_both_residuals_with_distinct_head_widths() {
        let weights = LayerWeights {
            input_norm: vec![round(1.0); 2],
            query: vec![round(0.0); 256 * 2],
            key: vec![round(0.0); 128 * 2],
            value: vec![round(0.0); 128 * 2],
            output: vec![round(0.0); 2 * 256],
            q_norm: vec![round(1.0); 128],
            k_norm: vec![round(1.0); 128],
            post_norm: vec![round(1.0); 2],
            gate: vec![round(0.0); 3 * 2],
            up: vec![round(0.0); 3 * 2],
            down: vec![round(0.0); 2 * 3],
        };
        let input = [1.0, -2.0, 3.0, 4.0].map(round);
        let trace = layer(
            &input,
            &weights,
            2,
            3,
            2,
            1,
            &[round(1.0); 128],
            &[round(0.0); 128],
        );
        assert_eq!(bits(&trace.output), bits(&input));
        assert_eq!(bits(&trace.residual), bits(&input));
        assert_eq!(trace.q.len(), 2 * 256);
        assert_eq!(trace.k.len(), 2 * 128);
        assert_eq!(trace.attention.len(), 2 * 256);
        assert_eq!(trace.gate.len(), 2 * 3);
        assert!(trace.attention.iter().all(|&x| widen(x) == 0.0));
    }
}
