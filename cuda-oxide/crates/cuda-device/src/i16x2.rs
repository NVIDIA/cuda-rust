/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed 16-bit integer min/max intrinsics.
//!
//! Each `u32` stores two 16-bit integers. The first value uses the low 16
//! bits. The second value uses the high 16 bits. `min_s16x2`/`max_s16x2`
//! compare the lanes as signed `i16`, `min_u16x2`/`max_u16x2` as unsigned
//! `u16`, and the `relu` forms clamp each signed lane's result to 0 if
//! negative.
//!
//! These are the packed halves of the PTX ISA 8.0 integer min/max
//! extensions (`sm_90+`); ptxas fuses adjacent chains of them into DPX
//! (`VIMNMX`-family) SASS. See [`crate::int`] for the scalar `.relu` forms
//! and [`crate::f16x2`] for the floating-point packed pairs.

include!("generated/i16x2.rs");
