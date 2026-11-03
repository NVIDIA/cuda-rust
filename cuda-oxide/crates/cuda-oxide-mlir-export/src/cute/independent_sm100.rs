/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Independent SM100 attention pack registration.

use crate::cute::{copy::sm100_attention::Tma, mma::sm100_attention::Mma};
use crate::{DropAttribute, TranslationError, TranslationRegistry};
use dialect_cute::sm100_attention_ops::*;

/// Register standalone SM100 attention MMA and TMA operations without GEMM-plan ownership.
pub(crate) fn register_cute_sm100_attention_pack(
    r: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    r.register_attribute::<CuteSm100MmaAttr>(DropAttribute)?;
    r.register_attribute::<CuteSm100TmaAttr>(DropAttribute)?;
    r.register_operation::<CuteSm100MmaSsOp>(Mma(false))?;
    r.register_operation::<CuteSm100MmaTsOp>(Mma(true))?;
    r.register_operation::<CuteSm100TmaLoadOp>(Tma(false))?;
    r.register_operation::<CuteSm100TmaStoreOp>(Tma(true))?;
    Ok(())
}
