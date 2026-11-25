/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVIDIA GPU hardware atoms (`cutlass.cute.nvgpu`), grouped by execution unit:
//!
//! - [`common`] — operand roles (`OperandA`, `OperandB`, `Mkl`, `Nkl`).
//! - [`cpasync`] — asynchronous G2S `cp.async` and TMA descriptor/tile copy atoms.
//! - [`warp`] — warp-level `mma.sync` and `ldmatrix` atoms (SM80 / SM120).
//! - [`tcgen05`] — 5th-generation Tensor Core UMMA and TMEM copy atoms (SM100).

pub mod common;
pub mod cpasync;
pub mod tcgen05;
pub mod warp;
