/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The shared PTX ISA semantics ceiling for analyses and rewrite plans.
//!
//! The lossless parser accepts newer PTX spellings by design, but anything
//! that assigns control-flow or binding semantics (CFG recovery, register
//! renaming, scope flattening) must fail closed on a `.version` newer than
//! the one those semantics were audited against.

use ptx_parse::Document;
use std::fmt;

pub const SUPPORTED_PTX_MAJOR: u16 = 9;
pub const SUPPORTED_PTX_MINOR: u16 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PtxVersionError {
    Missing,
    Invalid { value: String },
    Unsupported { value: String, supported: String },
}

impl fmt::Display for PtxVersionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(formatter, "PTX semantics require a .version directive"),
            Self::Invalid { value } => {
                write!(formatter, "invalid PTX .version value {value:?}")
            }
            Self::Unsupported { value, supported } => write!(
                formatter,
                "PTX {value} is newer than the audited semantics ceiling {supported}"
            ),
        }
    }
}

impl std::error::Error for PtxVersionError {}

pub(crate) fn validate_ptx_version(document: &Document<'_>) -> Result<(), PtxVersionError> {
    let value = document
        .directives()
        .iter()
        .find(|directive| directive.name() == ".version")
        .map(|directive| directive.arguments().trim())
        .ok_or(PtxVersionError::Missing)?;
    let (major, minor) = value
        .split_once('.')
        .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
        .ok_or_else(|| PtxVersionError::Invalid {
            value: value.to_string(),
        })?;
    if (major, minor) > (SUPPORTED_PTX_MAJOR, SUPPORTED_PTX_MINOR) {
        return Err(PtxVersionError::Unsupported {
            value: value.to_string(),
            supported: format!("{SUPPORTED_PTX_MAJOR}.{SUPPORTED_PTX_MINOR}"),
        });
    }
    Ok(())
}
