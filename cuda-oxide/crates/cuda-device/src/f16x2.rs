// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two f16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.
//!
//! The packing here is the ALU format for SIMD arithmetic. For over-aligned
//! *memory element* types that make loads and stores single wide transactions
//! (e.g. eight packed halves moving as one 128-bit access), see
//! [`crate::vector`]. For multi-register *value* groups, see
//! [`crate::cusimd::CuSimd`].
//!
//! See also [`crate::f16`] for the scalar min/max forms that operate on a
//! single half rather than a packed pair.
//!
//! # `fma` modifiers
//!
//! Six forms are available. `ftz` flushes subnormal inputs and results to zero,
//! `sat` clamps the result into `[0.0, 1.0]`, and `relu` clamps negatives to
//! zero while canonicalizing NaN. `fma_f16x2`, `fma_ftz_f16x2`,
//! `fma_sat_f16x2`, and `fma_ftz_sat_f16x2` are native at PTX ISA 4.2; the two
//! ReLU forms, `fma_relu_f16x2` and `fma_ftz_relu_f16x2`, require `sm_80` and
//! PTX ISA 7.0.

include!("generated/f16x2.rs");
