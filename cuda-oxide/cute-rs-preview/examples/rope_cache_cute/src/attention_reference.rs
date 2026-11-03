/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent FP64 grouped-query attention oracle over BF16 storage.
use crate::reference::{round, widen};
use cute_rs::BFloat16;

/// Computes scaled dot-product attention in token/head/coordinate order.
///
/// `query_start` selects causal attention at that absolute cache position;
/// `None` exposes every key to every query. Probabilities remain FP64, and
/// only the final output is rounded through FP32 to BF16. Inputs and output are
/// dense `[tokens, heads, dim]` arrays, with separate query and KV head counts.
/// Panics on inconsistent lengths, invalid grouping, or a causal range beyond
/// the cache. Future keys and values are excluded before evaluating arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    query: &[BFloat16],
    key: &[BFloat16],
    value: &[BFloat16],
    query_heads: usize,
    kv_heads: usize,
    dim: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_start: Option<usize>,
) -> Vec<BFloat16> {
    assert!(query_heads > 0, "query heads must be positive");
    assert!(kv_heads > 0, "KV heads must be positive");
    assert!(dim > 0, "head dimension must be positive");
    assert!(
        query_heads.is_multiple_of(kv_heads),
        "query heads must be a multiple of KV heads"
    );
    let query_len = query_tokens
        .checked_mul(query_heads)
        .and_then(|x| x.checked_mul(dim))
        .expect("query extent overflow");
    let kv_len = key_tokens
        .checked_mul(kv_heads)
        .and_then(|x| x.checked_mul(dim))
        .expect("KV extent overflow");
    assert_eq!(query.len(), query_len, "query storage length mismatch");
    assert_eq!(key.len(), kv_len, "key storage length mismatch");
    assert_eq!(value.len(), kv_len, "value storage length mismatch");
    if let Some(start) = query_start {
        let end = start
            .checked_add(query_tokens)
            .expect("causal query range overflow");
        assert!(end <= key_tokens, "causal query range exceeds key tokens");
    }
    assert!(
        query_tokens == 0 || key_tokens > 0,
        "nonempty queries require at least one key"
    );

    let mut output = Vec::with_capacity(query_len);
    let group_size = query_heads / kv_heads;
    let scale = 1.0 / (dim as f64).sqrt();
    for query_token in 0..query_tokens {
        // Restrict the iteration domain before touching K or V: masking a
        // computed NaN score or multiplying a masked NaN value by zero is unsafe.
        let visible_keys = query_start.map_or(key_tokens, |start| start + query_token + 1);
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_base = (query_token * query_heads + query_head) * dim;
            let mut weights = Vec::with_capacity(visible_keys);
            for key_token in 0..visible_keys {
                let key_base = (key_token * kv_heads + kv_head) * dim;
                let dot: f64 = (0..dim)
                    .map(|d| widen(query[query_base + d]) * widen(key[key_base + d]))
                    .sum();
                weights.push(dot * scale);
            }
            let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for weight in &mut weights {
                *weight = (*weight - maximum).exp();
            }
            let denominator: f64 = weights.iter().sum();
            for coordinate in 0..dim {
                let numerator: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(key_token, weight)| {
                        let index = (key_token * kv_heads + kv_head) * dim + coordinate;
                        weight * widen(value[index])
                    })
                    .sum();
                output.push(round(numerator / denominator));
            }
        }
    }
    output
}

/// Analytic CPU checks of causal windows, head grouping, scale, and stable softmax.
#[cfg(test)]
mod tests {
    use super::*;

    /// Widen oracle output for direct comparison with analytic FP64 fixture values.
    fn values(output: Vec<BFloat16>) -> Vec<f64> {
        output.into_iter().map(widen).collect()
    }

    /// Check uniform softmax over the causal prefix selected by a physical query offset.
    #[test]
    fn zero_queries_average_each_visible_prefix_at_a_nonzero_offset() {
        let query = [round(0.0); 4];
        let key = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0].map(round);
        let value = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0].map(round);
        assert_eq!(
            values(attention(&query, &key, &value, 1, 1, 2, 2, 3, Some(1))),
            [4.0, 6.0, 6.0, 8.0]
        );
    }

    /// Expose incorrect grouped-query head mapping with distinct KV values.
    #[test]
    fn adjacent_query_heads_share_the_correct_kv_head() {
        let query = [round(0.0); 8];
        let key = [round(0.0); 8];
        let value = [2.0, 4.0, 10.0, 20.0, 6.0, 8.0, 30.0, 40.0].map(round);
        assert_eq!(
            values(attention(&query, &key, &value, 4, 2, 2, 1, 2, None)),
            [4.0, 6.0, 4.0, 6.0, 20.0, 30.0, 20.0, 30.0]
        );
    }

    /// Check that the dot-product scale uses head width rather than head count.
    #[test]
    fn scores_use_inverse_square_root_head_dimension() {
        let query = [1.0, 0.0, 0.0, 0.0].map(round);
        let key = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].map(round);
        let value = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, -2.0].map(round);
        let actual = attention(&query, &key, &value, 1, 1, 4, 1, 2, None);
        // The second probability is sigmoid(1 / sqrt(4)), about 0.622459.
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [0x3f1f, 0x3f9f, 0xbf1f, 0xbf9f]
        );
    }

    /// Check max-subtracted FP64 softmax on large but finite input scores.
    #[test]
    fn extreme_finite_scores_do_not_overflow_the_softmax() {
        let query = [round(1024.0)];
        let key = [1024.0, 0.0, -1024.0].map(round);
        let value = [7.0, -2.0, 3.0].map(round);
        assert_eq!(
            values(attention(&query, &key, &value, 1, 1, 1, 1, 3, None)),
            [7.0]
        );
    }

    /// Ensure poisoned future K/V elements cannot contaminate a causal reference output.
    #[test]
    fn masked_future_keys_and_values_are_never_evaluated() {
        let query = [round(1.0); 2];
        let key = [0.0, 0.0, f64::NAN, f64::NAN].map(round);
        let value = [3.0, -4.0, f64::NAN, f64::NAN].map(round);
        assert_eq!(
            values(attention(&query, &key, &value, 1, 1, 2, 1, 2, Some(0))),
            [3.0, -4.0]
        );
    }

    /// Check the empty-query result when no key/value tokens exist.
    #[test]
    fn empty_queries_allow_an_empty_cache() {
        assert!(attention(&[], &[], &[], 4, 2, 128, 0, 0, Some(0)).is_empty());
    }

    /// Require the causal query range to lie within the available cache token prefix.
    #[test]
    #[should_panic(expected = "causal query range exceeds key tokens")]
    fn causal_range_must_fit_the_cache() {
        attention(
            &[round(0.0)],
            &[round(0.0); 2],
            &[round(0.0); 2],
            1,
            1,
            1,
            1,
            2,
            Some(2),
        );
    }

    /// Reject a query-head count that is not an integer multiple of KV heads.
    #[test]
    #[should_panic(expected = "query heads must be a multiple of KV heads")]
    fn fractional_head_grouping_is_rejected() {
        attention(&[], &[], &[], 3, 2, 128, 0, 0, None);
    }
}
