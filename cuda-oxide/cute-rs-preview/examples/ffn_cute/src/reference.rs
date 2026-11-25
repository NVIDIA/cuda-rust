/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent FP64 matrix arithmetic and BF16 rounding for synthetic fixtures.
use cute_rs::BFloat16;

/// Integer RNE oracle independent of the device conversion helper.
pub fn round(value: f32) -> BFloat16 {
    let bits = value.to_bits();
    let high = (bits >> 16) as u16;
    if value.is_nan() {
        return BFloat16::from_bits(high | 0x40);
    }
    let low = bits & 0xffff;
    BFloat16::from_bits(
        high.wrapping_add(u16::from(low > 0x8000 || (low == 0x8000 && high & 1 != 0))),
    )
}
/// Widen stored bits independently from the device implementation.
pub fn widen(value: BFloat16) -> f64 {
    f64::from(f32::from_bits(u32::from(value.to_bits()) << 16))
}
/// Compute X times transposed stored weights using sequential FP64 dot products.
///
/// Input is `[rows, input_width]`; weights are `[output_width, input_width]`.
/// Both strides count BF16 elements. The caller supplies buffers covering every
/// indexed row; the returned `[rows, output_width]` matrix is dense. Each dot
/// product narrows through FP32 before integer round-to-nearest-even BF16.
#[allow(clippy::too_many_arguments)]
pub fn linear(
    input: &[BFloat16],
    weight: &[BFloat16],
    rows: usize,
    input_width: usize,
    output_width: usize,
    input_stride: usize,
    weight_stride: usize,
) -> Vec<BFloat16> {
    let mut result = Vec::with_capacity(rows * output_width);
    for row in 0..rows {
        for out in 0..output_width {
            let mut sum = 0.0_f64;
            for k in 0..input_width {
                sum +=
                    widen(input[row * input_stride + k]) * widen(weight[out * weight_stride + k]);
            }
            result.push(round(sum as f32));
        }
    }
    result
}
/// Compute SiLU in FP64; preserve its BF16 output before the BF16 product.
pub fn activate(gate: &[BFloat16], up: &[BFloat16]) -> (Vec<BFloat16>, Vec<BFloat16>) {
    assert_eq!(gate.len(), up.len());
    let silu: Vec<_> = gate
        .iter()
        .map(|&g| {
            let g = widen(g);
            let s = if g >= 0.0 {
                1.0 / (1.0 + (-g).exp())
            } else {
                let e = g.exp();
                e / (1.0 + e)
            };
            round((g * s) as f32)
        })
        .collect();
    let product = silu
        .iter()
        .zip(up)
        .map(|(&s, &u)| round((widen(s) * widen(u)) as f32))
        .collect();
    (silu, product)
}

/// CPU fixtures exposing transposed weights, BF16 ties, and activation rounding.
#[cfg(test)]
mod tests {
    use super::*;
    /// Check `[output, input]` orientation while ignoring NaN-padded row strides.
    #[test]
    fn asymmetric_output_by_input_weights_and_padding_are_observable() {
        let x = [1., 2., 3., f32::NAN, 4., 5., 6., f32::NAN].map(round);
        let w = [1., -2., 3., f32::NAN, 2., 1., -1., f32::NAN].map(round);
        assert_eq!(
            linear(&x, &w, 2, 3, 2, 4, 4)
                .into_iter()
                .map(widen)
                .collect::<Vec<_>>(),
            [6., 1., 12., 7.]
        );
        assert!(linear(&[], &w, 0, 3, 2, 3, 4).is_empty());
    }
    /// Check even-tie BF16 rounding, signed zero, and NaN preservation.
    #[test]
    fn independent_rounding_covers_ties_and_special_values() {
        assert_eq!(round(f32::from_bits(0x3f80_8000)).to_bits(), 0x3f80);
        assert_eq!(round(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82);
        assert_eq!(round(-0.0).to_bits(), 0x8000);
        assert!(widen(round(f32::from_bits(0x7f80_0001))).is_nan());
    }
    /// Distinguish the BF16 SiLU intermediate from rounding the full product only once.
    #[test]
    fn activation_keeps_the_silu_rounding_boundary() {
        let (silu, product) = activate(&[round(1.0)], &[BFloat16::from_bits(0x4001)]);
        assert_eq!(silu[0].to_bits(), 0x3f3b);
        assert_eq!(product[0].to_bits(), 0x3fbc);
        let fused =
            round((1.0_f64 / (1.0 + (-1.0_f64).exp()) * widen(BFloat16::from_bits(0x4001))) as f32);
        assert_ne!(product[0].to_bits(), fused.to_bits());
    }
}
