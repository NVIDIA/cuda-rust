/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Materialize CUDA mapping plans for the upstream MLIR printer.

use crate::{MlirModule, TranslationError};

mod graph;
mod tokens;

/// Upstream printer revision and the local generic extension revision.
pub const INTEROP_REVISION: &str = "295773c4ab9cd7c5c3a14c2082be564b9cf5af90+cuda-oxide.1";

/// Render a checked mapping plan through the upstream MLIR printer.
pub fn render_module(module: &MlirModule) -> Result<String, TranslationError> {
    crate::translation::validate_target(module)?;
    graph::render(module)
}

#[cfg(test)]
mod tests;
