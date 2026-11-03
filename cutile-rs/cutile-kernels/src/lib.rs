/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::too_many_arguments)]

//! Reusable cuTile kernels.
//!
//! Kernel modules are grouped by function: attention, norms, positional
//! encoding, KV-cache updates, argmax/sampling, embeddings, and pointwise
//! transformer utilities. Raw-pointer and benchmark-oriented kernels live under
//! `experimental`.

pub mod argmax;
pub mod attention;
pub mod embeddings;
pub mod experimental;
pub mod kv_cache;
pub mod norms;
pub mod pointwise;
pub mod positional;

pub use argmax::{argmax_blocks_f16, argmax_reduce_blocks_to_u32, lm_head_argmax_blocks_f16};
pub use attention::{
    flash_attn_causal_seq_dynpos_f16, flash_attn_causal_seq_f16, fmha_causal,
    fmha_decode_gqa_split, fmha_prefill_causal, fmha_prefill_gqa, splitk_reduce_merge,
};
pub use embeddings::embedding_batch_f16;
pub use kv_cache::{kv_cache_update_seq_dynpos_f16, kv_cache_update_seq_f16};
pub use norms::{add_rms_norm_f16, qk_norm_f16, rms_norm_f16};
pub use pointwise::{add_2d_f16, gather_row_f16, silu_mul_2d_f16};
pub use positional::{qk_rope_dynpos_f16, rope_seq_dynpos_f16, rope_seq_f16};

// Kinds that fire during a normal transformer inference path (decode CUDA graph
// + step-graph prefill) and therefore may need warmup to pre-pay JIT cost.
//
// Gemm / Gemv run via cuBLAS rather than cuTile JIT, but they remain in this
// list because first-call cuBLAS handle and workspace setup is not free.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelKind {
    EmbeddingBatch,
    Gemm,
    Gemv,
    RmsNorm,
    RopeSeq,
    KvCacheUpdateSeq,
    FlashAttnCausalSeq,
    AddVec,
    SiluMul,
    GatherRow,
    ArgmaxBlocks,
    AddRmsNorm,
    QkNorm,
    QkRope,
    QkNormRopeKvPrefill,
    QkNormRopeKvDecode,
    ArgmaxReduceBlocks,
}

impl KernelKind {
    pub const COUNT: usize = 17;

    pub const fn idx(self) -> usize {
        self as usize
    }
}

pub const TILE_KERNEL_KINDS: [KernelKind; 17] = [
    KernelKind::EmbeddingBatch,
    KernelKind::Gemm,
    KernelKind::Gemv,
    KernelKind::RmsNorm,
    KernelKind::RopeSeq,
    KernelKind::KvCacheUpdateSeq,
    KernelKind::FlashAttnCausalSeq,
    KernelKind::AddVec,
    KernelKind::SiluMul,
    KernelKind::GatherRow,
    KernelKind::ArgmaxBlocks,
    KernelKind::AddRmsNorm,
    KernelKind::QkNorm,
    KernelKind::QkRope,
    KernelKind::QkNormRopeKvPrefill,
    KernelKind::QkNormRopeKvDecode,
    KernelKind::ArgmaxReduceBlocks,
];
