/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit mapping plans and checked translation of the prepared source module.

mod ast;
mod error;
mod registry;
mod translate;

pub(crate) use translate::validate_target;

pub use ast::{
    MlirAttribute, MlirBlock, MlirBlockArgument, MlirBlockId, MlirFloatType, MlirLocation,
    MlirModule, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueId, MlirValueUse,
};
pub use error::{MissingMappings, TranslationError};
pub use registry::{
    AttributeTranslation, DropAttribute, FixedType, OneToOneOperation, OperationTranslation,
    TranslationRegistry, TypeTranslation,
};
pub use translate::{
    LocationMode, OperationInput, TranslationConfig, TranslationSession, translate_module,
};
