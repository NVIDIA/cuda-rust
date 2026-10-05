/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar integer intrinsics.
//!
//! `min_relu_s32`/`max_relu_s32` clamp the comparison result to 0 if
//! negative, in one PTX instruction (`min.relu.s32`/`max.relu.s32`,
//! PTX ISA 8.0, `sm_90+`). Plain scalar `i32` min/max need no intrinsic:
//! ordinary Rust `min`/`max` already lowers to native `min.s32`/`max.s32`.
//! See [`crate::i16x2`] for the packed 16-bit forms.

include!("generated/int.rs");
