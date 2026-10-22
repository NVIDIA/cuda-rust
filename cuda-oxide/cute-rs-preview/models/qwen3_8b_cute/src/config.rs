/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Architectural constants and shape contracts for `Qwen/Qwen3-8B`.
//!
//! All dimensions match the official `Qwen/Qwen3-8B` `config.json`:
//!
//! ```text
//! hidden_size        = 4096
//! intermediate_size  = 12288
//! num_hidden_layers  = 36
//! num_attention_heads= 32   (head_dim = 128 -> Q width = 4096)
//! num_key_value_heads= 8    (GQA 4:1        -> KV width = 1024)
//! vocab_size         = 151936
//! rms_norm_eps       = 1e-6
//! rope_theta         = 1_000_000.0
//! ```
//!
//! # Example
//!
//! ```
//! use qwen3_8b_cute::config::{GQA_GROUP_SIZE, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, Q_HEADS};
//!
//! assert_eq!(LAYERS, 36);
//! assert_eq!(HIDDEN, Q_HEADS * HEAD_DIM);
//! assert_eq!(Q_HEADS / KV_HEADS, GQA_GROUP_SIZE);
//! assert_eq!(INTERMEDIATE, 12_288);
//! ```

/// Model hidden dimension (`H = 4096`).
pub const HIDDEN: usize = 4096;

/// SwiGLU feed-forward intermediate dimension (`I = 12288`).
pub const INTERMEDIATE: usize = 12_288;

/// Number of transformer decoder layers in the full `Qwen/Qwen3-8B` stack.
pub const LAYERS: usize = 36;

/// Number of query attention heads per layer (`32`).
pub const Q_HEADS: usize = 32;

/// Number of key/value attention heads per layer (`8`, 4:1 grouped-query attention).
pub const KV_HEADS: usize = 8;

/// Queries sharing each KV head (`Q_HEADS / KV_HEADS = 4`).
pub const GQA_GROUP_SIZE: usize = Q_HEADS / KV_HEADS;

/// Per-head attention dimension (`D = 128`).
pub const HEAD_DIM: usize = 128;

/// Flattened query projection width (`Q_HEADS * HEAD_DIM = 4096`).
pub const Q_WIDTH: usize = Q_HEADS * HEAD_DIM;

/// Flattened key/value projection width (`KV_HEADS * HEAD_DIM = 1024`).
pub const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;

/// Token vocabulary size (`151936`).
pub const VOCAB: usize = 151_936;

/// RMSNorm epsilon (`1e-6`).
pub const EPSILON: f32 = 1e-6;

/// Rotary position embedding base frequency (`1_000_000.0`).
pub const ROPE_THETA: f32 = 1_000_000.0;

/// Maximum rotary position index (`40960`).
pub const MAX_POSITION: u32 = 40_960;

/// Number of logical K elements sharing one `Float8E4M3FN` block scale in NVFP4 (`16`).
pub const NVFP4_VEC_SIZE: usize = 16;

/// Total parameter count for one `Qwen3-8B` decoder layer (`q, k, v, o, gate, up, down` + norms).
///
/// ```
/// use qwen3_8b_cute::config::layer_parameter_count;
/// assert_eq!(layer_parameter_count(), 192_946_432);
/// ```
#[must_use]
pub const fn layer_parameter_count() -> usize {
    let qkv_o = (Q_WIDTH + 2 * KV_WIDTH + HIDDEN) * HIDDEN;
    let mlp = 3 * HIDDEN * INTERMEDIATE;
    let norms = 2 * HIDDEN + 2 * HEAD_DIM;
    qkv_o + mlp + norms
}

/// Approximate weight footprint in bytes for `layers` decoder layers plus embeddings and LM head.
///
/// In BF16, every parameter uses 2 bytes. In NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` scale per 16 values),
/// linear weights use `0.5` bytes per weight plus `1/16` byte (`0.0625` bytes) of `Float8E4M3FN` scale,
/// while embeddings and RMSNorm weights remain in BF16.
///
/// ```
/// use qwen3_8b_cute::config::{LAYERS, model_weight_bytes};
/// let bf16_bytes = model_weight_bytes(LAYERS, false);
/// let nvfp4_bytes = model_weight_bytes(LAYERS, true);
/// assert!(nvfp4_bytes < bf16_bytes / 2);
/// ```
#[must_use]
pub const fn model_weight_bytes(layers: usize, nvfp4_linears: bool) -> usize {
    let linear_params_per_layer =
        (Q_WIDTH + 2 * KV_WIDTH + HIDDEN) * HIDDEN + 3 * HIDDEN * INTERMEDIATE;
    let norm_params_per_layer = 2 * HIDDEN + 2 * HEAD_DIM;
    let embed_and_head_params = 2 * VOCAB * HIDDEN + HIDDEN;
    let linear_bytes = if nvfp4_linears {
        // 4 bits per E2M1 value + 8 bits of E4M3 scale per 16 values = 9 bits / 16 bytes per value
        (layers * linear_params_per_layer * 9) / 16
    } else {
        layers * linear_params_per_layer * 2
    };
    linear_bytes + (layers * norm_params_per_layer + embed_and_head_params) * 2
}
