/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Materialize CUDA mapping plans for the upstream MLIR printer.

use crate::{MlirModule, TranslationError};

mod argument_locations;
mod graph;
mod tokens;

/// Published printer version and the downstream compatibility revision.
pub const INTEROP_REVISION: &str = "pliron-mlir-interop-0.1.0+cuda-oxide.2";

/// Render a checked mapping plan through the upstream MLIR printer.
pub fn render_module(module: &MlirModule) -> Result<String, TranslationError> {
    crate::translation::validate_target(module)?;
    graph::render(module)
}

#[cfg(test)]
mod tests;
