/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AST pass pipeline for the cuTile compiler.
//!
//! Passes transform or annotate the syn AST in a defined order before IR
//! emission. Each pass reads the previous pass's output and produces a
//! progressively more resolved form. Name resolution runs once per module;
//! the remaining passes run per `#[entry]` function body just before it is
//! lowered to `cutile-ir`.
//!
//! ```text
//! Raw syn AST (from proc macro)
//!     ↓
//! [Name Resolution]         — rustc-style DefId/Res/Namespace symbol table;
//!                             resolve imports and paths           (per module)
//!     ↓
//! [Proof Analysis]          — collect `#[entry(preconditions = ...)]` facts
//!                             for IR emission to query
//!     ↓
//! [Node IDs]                — assign stable ids to semantic expressions so
//!                             type-check side tables can refer back to them
//!     ↓
//! [Type Inference]          — DSL-narrow inference: expression types,
//!                             method/impl selection, dispatch-wrapper calls
//!     ↓
//! [Typed Dispatch Lowering] — rewrite trait-dispatch wrapper calls
//!                             (`f(a, b)` → `a.method(b)`) from typeck results
//!     ↓
//! [IR Emission]             — translation to cutile-ir (compiler/compile_*)
//! ```
//!
//! Type inference and dispatch lowering are DSL-narrow and compiler1-compatible
//! rather than a full Rust type checker; generic instantiation is handled in
//! [`crate::generics`], not as a pass here.

pub mod name_resolution;
pub mod node_ids;
pub mod obligation;
pub mod proof_analysis;
pub mod type_inference;
pub mod typed_dispatch_lowering;
