/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM120a prefill attention example and comparison entry point.

#![feature(f16)]

/// Shared CuTeDSL comparison process and interpreter selection.
mod comparison;
mod fixture;
mod kernel;
/// Host setup, correctness checks, and CUDA-event timing.
mod runner;

/// Run prefill verification, timing, or the requested CuTeDSL comparison.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
