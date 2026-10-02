/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build textual MLIR from any Pliron dialect.
//!
//! The engine has three small pieces:
//!
//! ```text
//! Pliron values and blocks
//!          │ explicit mappings
//!          ▼
//! typed MLIR syntax tree
//!          │ one renderer
//!          ▼
//! deterministic generic MLIR
//! ```
//!
//! A mapping pack registers the operations, types, and attributes it owns.
//! Unknown items are collected into one error before translation starts.
//! Equal-looking source and target names are never copied implicitly.

mod ast;
mod error;
mod registry;
mod render;
mod translate;

pub use ast::{
    MlirAttribute, MlirBlock, MlirBlockArgument, MlirBlockId, MlirFloatType, MlirLocation,
    MlirModule, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueId, MlirValueUse,
};
pub use error::{MissingMappings, TranslationError};
pub use registry::{
    AttributeTranslation, DropAttribute, FixedType, OneToOneOperation, OperationTranslation,
    TranslationRegistry, TypeTranslation,
};
pub use render::render_module;
pub use translate::{
    LocationMode, OperationInput, TranslationConfig, TranslationSession, translate_module,
};
