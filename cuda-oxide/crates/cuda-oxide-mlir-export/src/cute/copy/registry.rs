/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Single ownership of cooperative-copy and matrix-load profile registrations.

use crate::cute::{copy::cooperative::CopyTranslation, mma::ldmatrix::MatrixTranslation};
use crate::{DropAttribute, TranslationError, TranslationRegistry};
use dialect_cute::{
    attributes::{
        CuteComposedLayoutAttr, CuteCopyAtomAttr, CuteDivisibilityAttr, CuteLayoutAttr,
        CuteMatrixRoleAttr,
    },
    ops::{CuteCopyG2SOp, CuteLdmatrixOp},
};

/// Register cooperative-copy and matrix-load operations plus their consumed plan attributes.
pub(crate) fn register_cute_attention_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<CuteComposedLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<CuteCopyAtomAttr>(DropAttribute)?;
    registry.register_attribute::<CuteDivisibilityAttr>(DropAttribute)?;
    registry.register_attribute::<CuteLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<CuteMatrixRoleAttr>(DropAttribute)?;
    registry.register_operation::<CuteCopyG2SOp>(CopyTranslation)?;
    registry.register_operation::<CuteLdmatrixOp>(MatrixTranslation)?;
    Ok(())
}
