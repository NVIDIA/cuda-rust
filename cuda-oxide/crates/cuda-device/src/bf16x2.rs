// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two bf16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.
//!
//! The packing here is the ALU format for SIMD arithmetic. For over-aligned
//! *memory element* types that make loads and stores single wide transactions,
//! see [`crate::vector`]. For multi-register *value* groups, see
//! [`crate::cusimd::CuSimd`].
//!
//! See also [`crate::bf16`] for the scalar min/max forms that operate on a
//! single bfloat16 rather than a packed pair.
//!
//! # `fma` modifiers
//!
//! Only `fma_bf16x2` and `fma_relu_bf16x2` exist. The `ftz` and `sat` forms
//! that [`crate::f16x2`] provides are absent from the PTX ISA for `bf16x2`, not
//! merely unimplemented here: ptxas rejects `fma.rn.ftz.bf16x2` with "Illegal
//! modifier `.ftz` for instruction `fma`" while accepting
//! `fma.rn.relu.bf16x2`. LLVM declares the missing forms as intrinsics, so they
//! fail at instruction selection rather than at assembly.

include!("generated/bf16x2.rs");
