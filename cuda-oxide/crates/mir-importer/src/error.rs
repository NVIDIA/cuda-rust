/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types for MIR translation.
//!
//! Provides categorized error types that integrate with pliron's error system,
//! enabling rich error messages with source locations and backtraces.
//!
//! # Usage
//!
//! ```rust,no_run
//! use mir_importer::{TranslationErr, TranslationResult};
//! use pliron::{input_err, input_err_noloc, location::Location};
//!
//! // With location (preferred because it identifies the MIR source):
//! fn reject_unsupported_construct(loc: Location) -> TranslationResult<()> {
//!     input_err!(loc, TranslationErr::unsupported("f16 type"))
//! }
//!
//! // Without location when no source span is available:
//! fn reject_type_mismatch() -> TranslationResult<()> {
//!     input_err_noloc!(TranslationErr::type_error("expected i32, got f32"))
//! }
//! ```

use thiserror::Error;

/// Categorized translation error types.
///
/// Used with pliron's `input_err!` macro to create errors with location info.
/// The error category helps distinguish between:
/// - Missing features (planned but not yet implemented)
/// - Type mismatches (usually indicates a bug)
/// - Invalid MIR patterns (shouldn't happen with valid Rust code)
#[derive(Error, Debug)]
pub enum TranslationErr {
    /// Feature not yet implemented (e.g., f16 type, async, etc.).
    #[error("Unsupported construct: {0}")]
    Unsupported(String),

    /// Type mismatch or invalid type conversion.
    #[error("Type error: {0}")]
    TypeError(String),

    /// Invalid MIR pattern (shouldn't occur with valid Rust input).
    #[error("Invalid operation: {0}")]
    InvalidOp(String),
}

impl TranslationErr {
    /// Creates an `Unsupported` error for missing features.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Creates a `TypeError` for type mismatches.
    pub fn type_error(msg: impl Into<String>) -> Self {
        Self::TypeError(msg.into())
    }

    /// Creates an `InvalidOp` for invalid MIR patterns.
    pub fn invalid_op(msg: impl Into<String>) -> Self {
        Self::InvalidOp(msg.into())
    }
}

/// Result type for translation operations.
///
/// Uses pliron's error type which provides:
/// - Source location from MIR spans (when available)
/// - Backtrace capture (with `RUST_BACKTRACE=1`)
/// - Pretty printing via `.disp(ctx)`
pub type TranslationResult<T> = pliron::result::Result<T>;
