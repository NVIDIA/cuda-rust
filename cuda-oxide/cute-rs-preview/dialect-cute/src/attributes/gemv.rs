/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed-value and scale-view contracts for the block-scaled GEMV flow.

use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

/// Which part of a block-scaled tensor is currently selected.
///
/// ```text
/// Full ── select (batch, row) ──► Row ── select K tile ──► KTile<64>
/// ```
#[pliron_attr(name = "cute.scaled_layout", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteScaledLayoutAttr {
    /// The complete bound value/scale tensor before selecting a row.
    Full,
    /// One selected row across its complete reduction dimension.
    Row,
    /// One selected reduction tile with the stored logical K width.
    KTile(#[doc = "Logical K elements in one selected fragment."] u64),
}

impl Verify for CuteScaledLayoutAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if matches!(self, Self::KTile(0)) {
            return verify_err_noloc!("cute.scaled_layout K tile width must be greater than zero");
        }
        Ok(())
    }
}
