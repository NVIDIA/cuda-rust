/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thin, fail-closed `PostMem2Reg` `dialect-mir` transcriber.
//!
//! Every `#[verify(...)]` GPU kernel is transcribed 1-to-1 line by line into
//! `CudaOxide.Mir.Program.MirFunc` (paired with `ControlFlowCfg` for barrier
//! uniformity, `mbarrier` handoffs, PTX proxy fences, and `TMEM` lifecycle) so
//! Lean 4 owns 100% of arithmetic, race, and hardware state-machine proofs:
//!
//! ```text
//!                     translate_function(ctx, mir_func)
//!                                    │
//!                     ┌──────────────┴──────────────┐
//!                     ▼                             ▼
//!                `scalar.rs`                   `program.rs`
//!            • `ScalarWord` AST        • 1-to-1 `MirFunc` blocks
//!            • `BitVec` semantics      • `ControlFlowCfg` (barriers & `TMEM`)
//! ```

/// Shared `BlockGraph` adjacency queries and Pliron operand/attribute/shape helpers.
mod ir_utils;
/// Line-by-line `PostMem2Reg` `dialect-mir` -> `CudaOxide.Mir.Program.MirFunc` transcriber.
pub mod program;
/// Single-block pure scalar integer `mir.func` transcriber (`arithmetic.mir`).
mod scalar;

use pliron::{
    context::{Context, Ptr},
    operation::{Operation, verify_operation},
};

use crate::{
    ExportError,
    spec::{KernelVerificationSpec, has_verification_attributes, parse_kernel_verification_spec},
};

/// Transcribed output of [`translate_function`] before rendering to `.lean` source text.
/// Carries either a single-block scalar `BitVec` expression or a full line-by-line `MirFunc`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TranslatedFunction {
    /// Pure single-block scalar function (`CudaOxide.Mir.Scalar.Expr`).
    Scalar {
        /// Bit width (`8`, `16`, `32`, or `64`).
        width: u32,
        /// Number of entry arguments.
        arity: usize,
        /// Ordered `let argN / vN := ...` lines.
        let_lines: Vec<String>,
        /// Final returned variable name (`"arg0"`, `"v1"`, ...).
        returned_var: String,
    },
    /// Verified 1-to-1 line-by-line `MirFunc` kernel (`CudaOxide.Mir.Program.MirFunc`).
    MirProgram {
        /// Number of kernel parameters (`0..arity`).
        arity: usize,
        /// Parsed `#[launch_contract]` and `#[verify(...)]` specification.
        spec: Box<KernelVerificationSpec>,
        /// Line-by-line transcribed `MirFunc` blocks, parameters, shared decls, and `ControlFlowCfg`.
        program: Box<program::MirProgramData>,
        /// `mir.assert` bounds-check operations proven redundant by `MirAssertRedundant`.
        proven_assert_ops: Vec<Ptr<Operation>>,
    },
}

/// Transcribe any supported `mir.func` (scalar function or verified GPU kernel)
/// into a [`TranslatedFunction`] ready for Lean 4 rendering.
pub fn translate_function(
    ctx: &Context,
    function: Ptr<Operation>,
) -> Result<TranslatedFunction, ExportError> {
    if !has_verification_attributes(ctx, function) {
        return scalar::translate_scalar_function(ctx, function);
    }
    verify_operation(function, ctx)
        .map_err(|error| ExportError::Verification(error.to_string()))?;
    let (arity, spec) = parse_kernel_verification_spec(ctx, function)?;
    program::translate_mir_program(ctx, function, arity, spec)
}
