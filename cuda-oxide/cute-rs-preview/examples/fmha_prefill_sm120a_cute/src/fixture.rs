/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic BSHD inputs and an independent FP64 attention oracle.

/// FP16 values per attention head.
pub const HEAD_DIM: usize = 64;
/// Dot-product scale, equal to the inverse square root of the head dimension.
pub const SOFTMAX_SCALE: f64 = 0.125;
/// Absolute error allowance when comparing FP16 output with the FP64 oracle.
pub const ATOL: f64 = 2.0e-4;
/// Relative error allowance, multiplied by the oracle value's magnitude.
pub const RTOL: f64 = 2.0e-3;
/// Fixed seed used to generate query values independently of K and V.
const Q_SEED: u32 = 0xa511_e9b3;
/// Fixed seed used to generate key values independently of Q and V.
const K_SEED: u32 = 0x63d8_3595;
/// Fixed seed used to generate value inputs independently of Q and K.
const V_SEED: u32 = 0xc2b2_ae35;

/// Single-batch attention dimensions and mask selection in BSHD order.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Number of query rows.
    pub seq_q: usize,
    /// Number of key/value rows.
    pub seq_k: usize,
    /// Number of independent attention heads.
    pub heads: usize,
    /// Whether each query may attend only to its own and earlier positions.
    pub causal: bool,
}

impl Shape {
    #[allow(dead_code)]
    /// Check prefill tile constraints, causal dimensions, and 32-bit index bounds.
    pub fn validate(self) -> Result<(), String> {
        if self.seq_q == 0 || self.seq_k == 0 || self.heads == 0 {
            return Err("sequence lengths and head count must be positive".into());
        }
        if (self.seq_q != 1 && !self.seq_q.is_multiple_of(64)) || !self.seq_k.is_multiple_of(64) {
            return Err(
                "query length must be 1 or a multiple of 64; KV length must be a multiple of 64"
                    .into(),
            );
        }
        if self.causal && self.seq_q != self.seq_k {
            return Err("the initial causal specialization requires equal sequence lengths".into());
        }
        for length in [self.seq_q, self.seq_k] {
            let elements = length
                .checked_mul(self.heads)
                .and_then(|n| n.checked_mul(HEAD_DIM));
            if elements.is_none_or(|n| n > u32::MAX as usize) {
                return Err("tensor element count must fit u32 indexing".into());
            }
        }
        Ok(())
    }

    /// Number of FP16 query or output elements; call after shape validation.
    pub fn q_elements(self) -> usize {
        self.seq_q * self.heads * HEAD_DIM
    }

    /// Number of FP16 elements in either K or V; call after shape validation.
    pub fn kv_elements(self) -> usize {
        self.seq_k * self.heads * HEAD_DIM
    }
}

/// Deterministically mix an index and seed using wrapping 32-bit arithmetic.
fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

/// Generate reproducible FP16 values with the requested power-of-two scale.
fn input(length: usize, seed: u32, scale: u32) -> Vec<f16> {
    (0..length)
        .map(|i| {
            let integer = (mix32(i as u32 ^ seed) & 63) as i32 - 32;
            (integer as f32 * scale as f32 / 128.0) as f16
        })
        .collect()
}

/// Scale Q/K by an exact power of two to exercise larger logits; V is unchanged.
pub fn make_inputs(
    shape: Shape,
    input_scale: u32,
    zero_query: bool,
) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let q = if zero_query {
        vec![0.0 as f16; shape.q_elements()]
    } else {
        input(shape.q_elements(), Q_SEED, input_scale)
    };
    (
        q,
        input(shape.kv_elements(), K_SEED, input_scale),
        input(shape.kv_elements(), V_SEED, 1),
    )
}

/// FNV-1a over canonical little-endian FP16 bytes, matching reference/launch.py.
pub fn bits_hash(values: &[f16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Mathematical attention, with all arithmetic and probabilities in FP64.
/// This intentionally does not reproduce the kernel's FP16 probability rounding.
pub fn reference(shape: Shape, q: &[f16], k: &[f16], v: &[f16]) -> Vec<f64> {
    assert_eq!(q.len(), shape.q_elements());
    assert_eq!(k.len(), shape.kv_elements());
    assert_eq!(v.len(), shape.kv_elements());
    let q: Vec<f64> = q.iter().map(|&x| x as f64).collect();
    let k: Vec<f64> = k.iter().map(|&x| x as f64).collect();
    let v: Vec<f64> = v.iter().map(|&x| x as f64).collect();
    let mut output = vec![0.0; shape.q_elements()];
    let mut scores = vec![0.0; shape.seq_k];
    for head in 0..shape.heads {
        for row in 0..shape.seq_q {
            let q_base = (row * shape.heads + head) * HEAD_DIM;
            let end = if shape.causal {
                (row + 1).min(shape.seq_k)
            } else {
                shape.seq_k
            };
            let mut maximum = f64::NEG_INFINITY;
            for (column, score) in scores[..end].iter_mut().enumerate() {
                let k_base = (column * shape.heads + head) * HEAD_DIM;
                let mut dot = 0.0;
                for d in 0..HEAD_DIM {
                    dot += q[q_base + d] * k[k_base + d];
                }
                *score = dot * SOFTMAX_SCALE;
                maximum = maximum.max(*score);
            }
            let mut denominator = 0.0;
            for score in &mut scores[..end] {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            let mut accumulator = [0.0f64; HEAD_DIM];
            for (column, &weight) in scores[..end].iter().enumerate() {
                let v_base = (column * shape.heads + head) * HEAD_DIM;
                for d in 0..HEAD_DIM {
                    accumulator[d] += weight * v[v_base + d];
                }
            }
            for d in 0..HEAD_DIM {
                output[q_base + d] = accumulator[d] / denominator;
            }
        }
    }
    output
}

/// Error statistics from checking every output element against the oracle.
pub struct Verification {
    /// Number of output elements checked.
    pub elements: usize,
    /// Largest absolute difference from the FP64 oracle.
    pub max_abs_error: f64,
    /// Largest error divided by its allowed absolute-plus-relative bound.
    pub max_tolerance_ratio: f64,
}

/// Require every element to be finite and satisfy atol + rtol * abs(expected).
pub fn verify(got: &[f16], expected: &[f64]) -> Result<Verification, String> {
    if got.len() != expected.len() || got.is_empty() {
        return Err(format!(
            "output length {} differs from nonempty oracle length {}",
            got.len(),
            expected.len()
        ));
    }
    let mut result = Verification {
        elements: got.len(),
        max_abs_error: 0.0,
        max_tolerance_ratio: 0.0,
    };
    for (index, (&actual, &want)) in got.iter().zip(expected).enumerate() {
        let actual = actual as f64;
        if !actual.is_finite() || !want.is_finite() {
            return Err(format!(
                "nonfinite output/oracle at {index}: actual={actual}, expected={want}"
            ));
        }
        let error = (actual - want).abs();
        let bound = ATOL + RTOL * want.abs();
        if error > bound {
            return Err(format!(
                "output mismatch at {index}: actual={actual}, expected={want}, error={error}, bound={bound}"
            ));
        }
        result.max_abs_error = result.max_abs_error.max(error);
        result.max_tolerance_ratio = result.max_tolerance_ratio.max(error / bound);
    }
    Ok(result)
}

/// Tests for deterministic inputs, the attention oracle, and full-output validation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check fixture hashes and exact power-of-two scaling against the Python convention.
    #[test]
    fn fixture_bytes_match_python_and_scaling_preserves_values() {
        let shape = Shape {
            seq_q: 2,
            seq_k: 2,
            heads: 2,
            causal: false,
        };
        let (q, k, v) = make_inputs(shape, 1, false);
        assert_eq!(
            [bits_hash(&q), bits_hash(&k), bits_hash(&v)],
            [
                0xab18_87af_c727_f6b1,
                0xc758_962d_b6dc_1ae0,
                0xe543_f46a_99a9_3138
            ]
        );
        let (q16, k16, v16) = make_inputs(shape, 16, false);
        assert!(
            q16.iter()
                .zip(&q)
                .all(|(&a, &b)| a as f32 == b as f32 * 16.0)
        );
        assert!(
            k16.iter()
                .zip(&k)
                .all(|(&a, &b)| a as f32 == b as f32 * 16.0)
        );
        assert_eq!(v16, v);
        assert!(
            make_inputs(shape, 16, true)
                .0
                .iter()
                .all(|&x| x == 0.0 as f16)
        );
    }

    /// Check causal masking and independent head indexing using uniform logits.
    #[test]
    fn oracle_masks_future_keys_and_keeps_heads_separate() {
        let mut shape = Shape {
            seq_q: 2,
            seq_k: 2,
            heads: 2,
            causal: false,
        };
        let zeros = vec![0.0 as f16; 256];
        let values: Vec<f16> = [2.0, 10.0, 4.0, 14.0]
            .into_iter()
            .flat_map(|x| [x as f16; 64])
            .collect();
        let expected: Vec<f64> = [3.0, 12.0, 3.0, 12.0]
            .into_iter()
            .flat_map(|x| [x; 64])
            .collect();
        assert_eq!(reference(shape, &zeros, &zeros, &values), expected);
        shape.causal = true;
        let expected: Vec<f64> = [2.0, 10.0, 3.0, 12.0]
            .into_iter()
            .flat_map(|x| [x; 64])
            .collect();
        assert_eq!(reference(shape, &zeros, &zeros, &values), expected);
    }

    /// Check stable softmax with logits too large for unshifted exponentials.
    #[test]
    fn oracle_handles_logits_that_overflow_unshifted_exponentials() {
        let shape = Shape {
            seq_q: 1,
            seq_k: 2,
            heads: 1,
            causal: false,
        };
        let q = vec![100.0 as f16; 64];
        let k: Vec<f16> = [100.0, -100.0]
            .into_iter()
            .flat_map(|x| [x as f16; 64])
            .collect();
        let v: Vec<f16> = [3.0, -7.0]
            .into_iter()
            .flat_map(|x| [x as f16; 64])
            .collect();
        assert_eq!(reference(shape, &q, &k, &v), vec![3.0; 64]);
    }

    /// Ensure every element, including the tail, is checked for finite and accurate output.
    #[test]
    fn verifier_rejects_nonfinite_and_unsampled_tail_corruption() {
        let expected = [0.0, 0.01, -0.02, 0.03];
        let mut got: Vec<f16> = expected.iter().map(|&x| x as f16).collect();
        assert_eq!(verify(&got, &expected).unwrap().elements, 4);
        assert!(verify(&got[..3], &expected).is_err());
        for bad in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY, -0.03 as f16] {
            got[3] = bad;
            assert!(verify(&got, &expected).is_err());
        }
    }

    /// Check positive dimensions, tile boundaries, and equal-length causal sequences.
    #[test]
    fn shape_rejects_zero_and_asymmetric_causal_sequences() {
        let good = Shape {
            seq_q: 64,
            seq_k: 128,
            heads: 8,
            causal: false,
        };
        assert!(good.validate().is_ok());
        assert!(Shape { seq_q: 1, ..good }.validate().is_ok());
        assert!(Shape { heads: 0, ..good }.validate().is_err());
        assert!(Shape { seq_q: 0, ..good }.validate().is_err());
        assert!(Shape { seq_k: 0, ..good }.validate().is_err());
        assert!(
            Shape {
                causal: true,
                ..good
            }
            .validate()
            .is_err()
        );
        assert!(Shape { seq_q: 65, ..good }.validate().is_err());
    }
}
