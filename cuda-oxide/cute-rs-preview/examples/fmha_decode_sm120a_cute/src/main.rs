/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SM120a single-token attention example and comparison entry point.

#![feature(f16)]

/// Shared CuTeDSL comparison launcher used by both SM120a attention examples.
#[path = "../../fmha_prefill_sm120a_cute/src/comparison.rs"]
mod comparison;
/// Checked decode options and launch geometry.
mod config;
#[path = "../../fmha_prefill_sm120a_cute/src/fixture.rs"]
mod fixture;
mod kernel;
mod runner;

/// Run decode verification, timing, or the requested CuTeDSL comparison.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
