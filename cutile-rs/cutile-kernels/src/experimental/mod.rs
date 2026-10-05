/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Experimental cuTile kernels.
//!
//! This directory contains raw-pointer entry points and benchmark-oriented
//! kernels, grouped by function. APIs here are expected to move faster than the
//! root kernel modules.

pub mod attention;
pub mod fused_transformer;
pub mod kvbm;
pub mod moe;
pub mod norms;

pub use attention::{attention_decode_kernel_grouped, fmha_prefill_gqa_lpt};
pub use fused_transformer::{qk_norm_rope_kv_decode_raw_f16, qk_norm_rope_kv_prefill_raw_f16};
pub use kvbm::{copy_contiguous_to_stacked_f16, copy_stacked_to_contiguous_f16};
pub use moe::group_gemm_f16_nt_desc;
pub use norms::add_rms_norm_decode_raw_f16;
