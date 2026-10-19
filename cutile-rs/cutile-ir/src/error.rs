/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use thiserror::Error;

/// Errors produced by the tile-ir crate.
#[derive(Debug, Error)]
pub enum Error {
    #[error("bytecode write error: {0}")]
    BytecodeWrite(String),

    #[error("IR verification error: {0}")]
    Verification(String),

    #[error("builder error: {0}")]
    Builder(String),
}

pub type Result<T> = std::result::Result<T, Error>;
