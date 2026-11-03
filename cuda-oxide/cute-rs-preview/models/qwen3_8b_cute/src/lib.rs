/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Qwen/Qwen3-8B` transformer decoder in `cute-rs` with BF16 and NVFP4 (`Float4E2M1FN` + `Float8E4M3FN`, `SF_VEC = 16`) execution.
//!
//! ```text
//! Token IDs / Prompt Hidden [rows, 4096]
//!        │
//!        ▼   × 36 Decoder Layers (H=4096, I=12288, Q=32, KV=8, D=128)
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  RMSNorm(4096) ──► Q/K/V Projections (BF16 MMA or NVFP4 E2M1+E4M3)  │
//! │                ──► QK-Norm(128) + RoPE(θ=1e6) + GQA(32:8) KV Cache  │
//! │                ──► O Projection + Residual                          │
//! │  RMSNorm(4096) ──► Gate/Up Projections ──► SwiGLU ──► Down + Resid  │
//! └─────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! Final RMSNorm(4096)
//! ```

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

/// Architectural constants and memory-footprint helpers for `Qwen/Qwen3-8B`.
pub mod config;

/// True NVFP4 (`Float4E2M1FN` + `Float8E4M3FN` scale, `SF_VEC = 16`) quantization and GPU projection kernels.
pub mod nvfp4;

/// Multi-shard safetensors checkpoint loader and fast GPU synthetic weight generator.
pub mod checkpoint;

/// End-to-end `Qwen3-8B` BF16 and NVFP4 execution engine and RTX 5090 benchmark harness.
pub mod benchmark;

/// Shared host runner for causal grouped-query attention.
#[allow(dead_code)]
#[path = "../../../examples/rope_cache_cute/src/attention.rs"]
mod attention;

/// Shared causal grouped-query attention kernels (`[64, 128]` shared-memory KV tiles).
#[allow(dead_code)]
#[path = "../../../examples/rope_cache_cute/src/attention_kernel.rs"]
mod attention_kernel;

/// Shared one-/two-row BF16 decode linear projection runner.
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/decode_linear.rs"]
pub(crate) mod decode_linear;

/// Shared elementwise embedding, residual-add, and SiLU device kernels.
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/element_kernel.rs"]
pub(crate) mod element_kernel;

/// Shared BF16 Tensor-Core MMA projection and SwiGLU activation kernels.
#[allow(dead_code)]
#[path = "../../../examples/ffn_cute/src/kernel.rs"]
pub(crate) mod ffn_kernel;

/// Shared rotary and KV-cache append device kernels.
#[allow(dead_code)]
#[path = "../../../examples/rope_cache_cute/src/kernel.rs"]
mod kernel;

/// Typed synchronous BF16 decoder composition (`Batch<W>`, `DecoderCache`, `Runner`).
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/model.rs"]
pub mod model;

/// Shared general and warp-specialized RMSNorm device kernels.
#[allow(dead_code)]
#[path = "../../../examples/common/rmsnorm.rs"]
pub(crate) mod norm_kernel;

/// Shared RMSNorm host dispatch.
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/norm_specialized.rs"]
mod norm_specialized;

/// Shared rotary and causal GQA host operator (`KvCache`, `AttentionInputs`, `Runner`).
#[allow(dead_code)]
#[path = "../../../examples/rope_cache_cute/src/operator.rs"]
pub mod operator;

/// Shared cuBLAS-matching reduction-order planner for BF16 decode projections.
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/reference_linear.rs"]
pub(crate) mod reference_linear;

/// Shared cuBLAS-matching FMA-chain and split-K BF16 projection kernels.
#[allow(dead_code)]
#[path = "../../qwen_decoder_cute/src/reference_linear_kernel.rs"]
mod reference_linear_kernel;

/// Shared split-half rotary position embedding frequency and table generator.
#[allow(dead_code)]
#[path = "../../../examples/rope_cache_cute/src/rotary.rs"]
pub mod rotary;
