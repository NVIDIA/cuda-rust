/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::BTreeSet;

use thiserror::Error;

/// Every source item that has no mapping in the selected profile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MissingMappings {
    pub operations: BTreeSet<String>,
    pub types: BTreeSet<String>,
    pub attributes: BTreeSet<String>,
}

impl MissingMappings {
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty() && self.types.is_empty() && self.attributes.is_empty()
    }
}

impl std::fmt::Display for MissingMappings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "the selected MLIR profile does not cover this Pliron module"
        )?;
        if !self.operations.is_empty() {
            writeln!(f, "  operations: {}", join(&self.operations))?;
        }
        if !self.types.is_empty() {
            writeln!(f, "  types: {}", join(&self.types))?;
        }
        if !self.attributes.is_empty() {
            writeln!(f, "  attributes: {}", join(&self.attributes))?;
        }
        Ok(())
    }
}

fn join(values: &BTreeSet<String>) -> String {
    values.iter().cloned().collect::<Vec<_>>().join(", ")
}

/// A checked translation failure.
#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("duplicate {kind} mapping for {id}")]
    DuplicateMapping { kind: &'static str, id: String },
    #[error("the translation registry is sealed")]
    RegistrySealed,
    #[error("{0}")]
    MissingMappings(MissingMappings),
    #[error("translation of {source_item} for profile {profile} failed: {message}")]
    Mapping {
        source_item: String,
        profile: String,
        message: String,
    },
    #[error("translated operation {source_item} did not define source result {result:?}")]
    MissingResult {
        source_item: String,
        result: crate::MlirValueId,
    },
    #[error("translated operation {source_item} defined source result {result:?} more than once")]
    DuplicateResult {
        source_item: String,
        result: crate::MlirValueId,
    },
    #[error("internal translation state is missing {kind} {id}")]
    MissingIndexedValue { kind: &'static str, id: String },
    #[error("invalid target MLIR: {0}")]
    InvalidTarget(String),
}
