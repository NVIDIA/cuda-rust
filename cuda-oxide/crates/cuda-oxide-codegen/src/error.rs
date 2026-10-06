/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// Errors from pipeline execution, categorized by stage.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum PipelineError {
    /// The requested MIR pass pipeline is invalid for this compilation.
    InvalidMirPassPipeline(String),
    /// Function has no MIR body (shouldn't happen for collected functions).
    NoBody(String),
    /// MIR→Pliron IR translation failed.
    Translation(String),
    /// Pliron IR verification failed (includes failing operation if found).
    Verification {
        name: String,
        message: String,
        operation: Option<String>,
    },
    /// MIR→LLVM lowering failed.
    Lowering(String),
    /// The lowered LLVM-dialect module failed structural verification.
    LoweredVerification {
        message: String,
        operation: Option<String>,
    },
    /// Standalone PTX contains declarations that require a link step.
    UnsupportedLinking { symbols: Vec<String> },
    /// Libdevice linking was requested, but the toolchain cannot perform it.
    LibdeviceUnavailable { message: String },
    /// LLVM IR export failed.
    Export(String),
    /// The requested CUDA target could not be parsed, or cannot lower a
    /// feature the module needs. Distinct from `PtxGeneration`: this is a
    /// problem with the requested target itself, decided before `llc` runs.
    /// `reason` is already fully phrased for a user; `target` is carried
    /// separately so callers can match on it.
    TargetSelection { target: String, reason: String },
    /// PTX generation via `llc` failed.
    PtxGeneration(String),
    /// The requested LLVM middle-end optimization failed.
    Optimization(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMirPassPipeline(message) => {
                write!(f, "invalid MIR pass pipeline: {message}")
            }
            Self::NoBody(name) => write!(f, "Function '{}' has no MIR body", name),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::Verification {
                name,
                message,
                operation,
            } => {
                writeln!(f, "Verification failed for '{}':", name)?;
                writeln!(f, "  {}", message)?;
                if let Some(op) = operation {
                    writeln!(f, "  Failed operation:\n{}", op)?;
                }
                Ok(())
            }
            Self::Lowering(msg) => write!(f, "Lowering failed: {}", msg),
            Self::LoweredVerification { message, operation } => {
                writeln!(f, "Verification failed for lowered LLVM module:")?;
                writeln!(f, "  {message}")?;
                if let Some(op) = operation {
                    writeln!(f, "  Failed operation:\n{op}")?;
                }
                Ok(())
            }
            Self::UnsupportedLinking { symbols } => write!(
                f,
                "standalone PTX cannot resolve external symbols: {symbols:?}"
            ),
            Self::LibdeviceUnavailable { message } => {
                write!(f, "libdevice linking is unavailable: {message}")
            }
            Self::Export(msg) => write!(f, "Export failed: {}", msg),
            Self::TargetSelection { reason, .. } => write!(f, "{reason}"),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
            Self::Optimization(msg) => write!(f, "LLVM optimization failed: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}
