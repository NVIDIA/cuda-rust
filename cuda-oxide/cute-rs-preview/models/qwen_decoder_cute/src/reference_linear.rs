/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Per-shape reduction plans that mirror the reference's cuBLAS kernel choice.
//!
//! The reference computes every decode projection with `F.linear` on BF16
//! tensors, and cuBLAS picks a kernel from the row count and the weight shape.
//! Which kernel runs decides how the FP32 sum is formed, and therefore which
//! BF16 value comes out when the exact sum sits near a rounding boundary:
//!
//! ```text
//! one row, most shapes      FP32 FMA chains over k mod 16 (or 32) ─┐
//!                            merged by XOR offsets 8,4,2,1 (16,…,1)   ├─► one BF16 rounding
//! one row, 6144 → 2048      tensor-core K16 chains per 1280-wide slice,│
//! two rows, 3072 → 1024     … per 192-wide slice, slice sums in order ─┤
//! two rows, other shapes    one tensor-core K16 chain over the width ──┘
//! ```
//!
//! The table below was read off cuBLAS 13.0 on an RTX 5090 (torch
//! 2.14.1+cu130) with cancellation probes that expose each kernel's reduction
//! tree, then confirmed bit for bit on 1,784,832 saved reference activations
//! across every predictor and Talker decode shape. The tensor-core chain is
//! what the ordinary decode kernel already computes, so shapes outside the
//! table keep that kernel.
use cute_rs::Bf16;

/// How the reference reduces one decode projection shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReductionPlan {
    /// FP32 FMA chains over `k mod chains`, merged by XOR offsets `chains/2 .. 1`.
    FmaChains {
        /// Number of interleaved accumulators per output: 16 or 32.
        chains: usize,
    },
    /// One tensor-core K16 chain over the whole width.
    MmaChain,
    /// Tensor-core K16 chains per slice; FP32 slice sums added in slice order.
    MmaSplitK {
        /// Elements per slice; the last slice takes the remainder.
        slice_width: usize,
    },
}

/// Select the plan the reference uses for `rows` input rows and `[outputs, width]` weights.
pub(crate) fn reference_plan(rows: usize, width: usize, outputs: usize) -> ReductionPlan {
    match (rows, width, outputs) {
        (1, 1024, 1024) => ReductionPlan::FmaChains { chains: 32 },
        (1, 1024, 2048 | 3072) | (1, 2048, 1024 | 2048 | 3072 | 6144) | (1, 3072, 1024) => {
            ReductionPlan::FmaChains { chains: 16 }
        }
        (1, 6144, 2048) => ReductionPlan::MmaSplitK { slice_width: 1280 },
        (2, 3072, 1024) => ReductionPlan::MmaSplitK { slice_width: 192 },
        _ => ReductionPlan::MmaChain,
    }
}

/// CTAs for an FMA-chain launch: one warp covers `32 / (chains / 4)` outputs.
pub(crate) fn fma_chain_blocks(chains: usize, outputs: usize) -> usize {
    outputs.div_ceil(crate::reference_linear_kernel::kernels::outputs_per_warp(
        chains,
    ))
}

/// Host FP32 emulation of [`ReductionPlan::FmaChains`] for exact kernel checks.
///
/// `input` is one row of `width` values; `weight` is `[outputs, width]` with
/// element pitch `weight_stride`. Rust evaluates `w * x + acc` as an exact
/// product followed by one FP32 addition, exactly like the device entry.
pub(crate) fn emulate_fma_chains(
    input: &[Bf16],
    weight: &[Bf16],
    weight_stride: usize,
    width: usize,
    outputs: usize,
    chains: usize,
    bias: Option<&[Bf16]>,
) -> Vec<Bf16> {
    (0..outputs)
        .map(|n| {
            let row = &weight[n * weight_stride..n * weight_stride + width];
            let mut acc = vec![0.0f32; chains];
            for (k, (w, x)) in row.iter().zip(&input[..width]).enumerate() {
                acc[k % chains] += w.to_f32() * x.to_f32();
            }
            let mut offset = chains / 2;
            while offset >= 1 {
                for r in 0..offset {
                    acc[r] += acc[r + offset];
                }
                offset /= 2;
            }
            let total = match bias {
                Some(bias) => acc[0] + bias[n].to_f32(),
                None => acc[0],
            };
            Bf16::from_f32(total)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every production decode shape maps to the kernel the reference selects.
    #[test]
    fn production_shapes_follow_the_probed_table() {
        for (width, outputs) in [(1024, 2048), (1024, 3072), (2048, 1024), (3072, 1024)] {
            assert_eq!(
                reference_plan(1, width, outputs),
                ReductionPlan::FmaChains { chains: 16 }
            );
        }
        for (width, outputs) in [(2048, 2048), (2048, 3072), (2048, 6144)] {
            assert_eq!(
                reference_plan(1, width, outputs),
                ReductionPlan::FmaChains { chains: 16 }
            );
        }
        assert_eq!(
            reference_plan(1, 1024, 1024),
            ReductionPlan::FmaChains { chains: 32 }
        );
        assert_eq!(
            reference_plan(1, 6144, 2048),
            ReductionPlan::MmaSplitK { slice_width: 1280 }
        );
        assert_eq!(
            reference_plan(2, 3072, 1024),
            ReductionPlan::MmaSplitK { slice_width: 192 }
        );
        for (width, outputs) in [(1024, 1024), (1024, 2048), (1024, 3072), (2048, 1024)] {
            assert_eq!(reference_plan(2, width, outputs), ReductionPlan::MmaChain);
        }
        assert_eq!(reference_plan(1, 17, 7), ReductionPlan::MmaChain);
        assert_eq!(reference_plan(3, 2048, 2048), ReductionPlan::MmaChain);
    }

    /// Launch geometry covers every output with a whole warp per CTA.
    #[test]
    fn launch_geometry_covers_outputs() {
        assert_eq!(fma_chain_blocks(16, 1024), 128);
        assert_eq!(fma_chain_blocks(16, 3072), 384);
        assert_eq!(fma_chain_blocks(32, 1024), 256);
        assert_eq!(fma_chain_blocks(16, 9), 2);
    }

    /// The emulation reproduces the probed order on a cancellation triple.
    ///
    /// Elements 0 and 16 share one chain: `L + (-L)` cancels before the chain
    /// meets element 1, so the 1.0 placed at element 1 survives. Element 32 is
    /// added to the chain after `L`, where it is absorbed.
    #[test]
    fn emulation_follows_interleaved_chains_and_butterfly() {
        let width = 64;
        let one = Bf16::from_f32(1.0);
        let input = vec![one; width];
        let large = Bf16::from_f32(33_554_432.0);
        let mut weight = vec![Bf16::from_f32(0.0); 2 * width];
        weight[0] = large;
        weight[16] = Bf16::from_f32(-33_554_432.0);
        weight[1] = one;
        weight[width] = large;
        weight[width + 32] = one;
        weight[width + 48] = Bf16::from_f32(-33_554_432.0);
        let out = emulate_fma_chains(&input, &weight, width, width, 2, 16, None);
        assert_eq!(out[0].to_f32(), 1.0);
        assert_eq!(out[1].to_f32(), 0.0);
        let bias = [Bf16::from_f32(0.5); 2];
        let out = emulate_fma_chains(&input, &weight, width, width, 2, 16, Some(&bias));
        assert_eq!(out[0].to_f32(), 1.5);
    }
}
