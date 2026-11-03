// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product intrinsics (`dp4a`, `dp2a`).
//!
//! These instructions perform packed-byte or packed-half dot products with
//! accumulation, useful for integer quantised inference on Ampere+ GPUs.
//!
//! # `dp4a`, 4-element byte dot product
//!
//! Treats `a` and `b` as vectors of 4 packed bytes, multiplies corresponding
//! elements, sums the products, and adds the scalar accumulator `c`:
//!
//! ```text
//! d = c + a.byte0*b.byte0 + a.byte1*b.byte1 + a.byte2*b.byte2 + a.byte3*b.byte3
//! ```
//!
//! # `dp2a`, 2-element half-word × byte dot product
//!
//! Treats `a` as two packed 16-bit values and `b` as packed bytes (lower 2
//! bytes selected by the `.lo` qualifier):
//!
//! ```text
//! d = c + a.half0*b.byte0 + a.half1*b.byte1
//! ```
//!
//! # Supported on
//!
//! - `sm_61+` (`dp4a`, `dp2a`)

include!("generated/dotprod.rs");
