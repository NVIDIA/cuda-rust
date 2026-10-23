/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CuTe runtime mappings, organized by capability and architecture.
//! The full-profile registry owns shared registrations exactly once.

mod common;
mod common_sm100;
pub(in crate::cute) mod copy;
mod emit;
pub(in crate::cute) mod epilogue;
mod independent_sm100;
pub(in crate::cute) mod mma;
pub(in crate::cute) mod pipeline;
mod registry;
pub(in crate::cute) mod scheduler;
mod sm100;
pub(in crate::cute) mod tensor;

pub(crate) use copy::registry::register_cute_attention_pack;
pub(crate) use emit::{Emit, integer_property, property};
pub(crate) use independent_sm100::register_cute_sm100_attention_pack;
pub(crate) use registry::register_cute_gemm_pack;
pub(crate) use sm100::register_cute_sm100_pack;
pub use tensor::elementwise::register_cute_elementwise_pack;
pub(crate) use tensor::gemv::register_cute_cutlass_profile_pack;

#[cfg(test)]
#[path = "tests/gemm.rs"]
mod gemm_tests;

#[cfg(test)]
#[path = "tests/cooperative.rs"]
mod cooperative_tests;
