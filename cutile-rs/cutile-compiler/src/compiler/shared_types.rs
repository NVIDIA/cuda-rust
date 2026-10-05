/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared type definitions used by compiler2.
//!
//! These are MLIR-free copies of types originally defined in
//! `compiler/_type.rs`, `compiler/_value.rs`, and `compiler/_function.rs`.

use crate::syn_utils::SingleMetaList;
use syn::Expr;

// ---------------------------------------------------------------------------
// Kind — classification of a Rust type within the CUDA Tile type system
// ---------------------------------------------------------------------------

/// Classification of a Rust type within the CUDA Tile type system.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Kind {
    /// Scalar element type (e.g. `f32`, `i32`).
    PrimitiveType,
    /// Shaped type with element and dimensions (e.g. `Tile<f32, {[128, 64]}>`).
    StructuredType,
    /// Tuple or array compound type; not compiled to CUDA Tile IR.
    Compound,
    /// User-defined struct; not compiled to CUDA Tile IR.
    Struct,
    /// String literal type; requires special handling.
    String,
    /// Compile-time enum value; currently used for `Option<T>` optional operands.
    Enum,
}

// ---------------------------------------------------------------------------
// BlockTerminator — how a compiled block transfers control at its end
// ---------------------------------------------------------------------------

/// How a compiled block transfers control at its end.
#[derive(Debug, Clone, Copy)]
pub enum BlockTerminator {
    /// Yield values to the enclosing region.
    Yield,
    /// Continue to the next loop iteration.
    Continue,
    /// Return from the function.
    Return,
    /// Break out of the enclosing loop.
    Break,
}

// ---------------------------------------------------------------------------
// LoopKind — which Tile IR loop op the innermost enclosing loop lowers to
// ---------------------------------------------------------------------------

/// The Tile IR op the innermost enclosing source loop lowers to. Decides
/// which early exits a block may emit: `cuda_tile.break` is only valid
/// inside `cuda_tile.loop` (and the `cuda_tile.if` regions nested in it),
/// never inside `cuda_tile.for`, which has no early exit at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopKind {
    /// A `for` over a range, dimension, or mapped-partition schedule:
    /// `cuda_tile.for`.
    For,
    /// `while` / `loop`: `cuda_tile.loop`.
    Loop,
}

// ---------------------------------------------------------------------------
// Mutability — mutability state of a variable binding
// ---------------------------------------------------------------------------

/// Mutability state of a variable binding during compilation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Mutability {
    /// Mutability has not been determined yet.
    Unset,
    /// The binding is mutable (`let mut`).
    Mutable,
    /// The binding is immutable (`let`).
    Immutable,
}

// ---------------------------------------------------------------------------
// EntryAttrs — parsed #[entry(...)] attributes
// ---------------------------------------------------------------------------

/// Parsed attributes from the `#[cuda_tile::entry(...)]` annotation on a kernel function.
pub struct EntryAttrs {
    pub(crate) entry_attrs: SingleMetaList,
}

impl EntryAttrs {
    pub(crate) fn get_entry_arg_expr(&self, name: &str) -> Option<&Expr> {
        self.entry_attrs.parse_custom_expr(name)
    }
    pub(crate) fn get_entry_arg_bool(&self, name: &str) -> bool {
        self.entry_attrs.parse_bool(name).unwrap_or(false)
    }
}
