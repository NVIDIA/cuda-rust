/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `dialect-mir` operation converters.
//!
//! This module contains submodules that convert standard `dialect-mir` ops
//! to their LLVM dialect equivalents. Each submodule handles a category of
//! operations with similar semantics and conversion patterns.
//!
//! # Module Organization
//!
//! | Module           | Operations                         | Description                  |
//! |------------------|------------------------------------|------------------------------|
//! | [`aggregate`]    | Struct, tuple, enum ops            | Composite type manipulation  |
//! | [`arithmetic`]   | Binary ops, comparisons, shifts    | Math and logic operations    |
//! | [`call`]         | Function calls                     | With argument flattening     |
//! | [`cast`]         | Type conversions                   | All cast variants            |
//! | [`constants`]    | `mir.constant`, `mir.float_constant`| Literal values              |
//! | [`control_flow`] | Return, goto, branch, assert       | Block terminators            |
//! | [`memory`]       | Load, store, ref, ptr_offset       | Memory access operations     |
//!
//! # Common Patterns
//!
//! All converters receive `(ctx, rewriter, op, operands_info)` from the
//! `DialectConversion` framework:
//!
//! - **Operands** are already type-converted — read them directly from `op`.
//! - Use `rewriter.insert_operation` to emit new LLVM ops.
//! - Use `rewriter.replace_operation` to map results and erase the MIR op.
//! - Use `operands_info` to recover pre-conversion types when needed.

pub mod aggregate;
pub mod arithmetic;
pub mod call;
pub mod cast;
pub mod constants;
pub mod control_flow;
pub mod memory;

#[cfg(test)]
pub(crate) mod test_util;
