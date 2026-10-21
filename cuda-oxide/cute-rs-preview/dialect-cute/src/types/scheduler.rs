/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent output-tile grids and their ordinary cursor provenance.

use pliron::{
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    result::Error,
    r#type::{Type, TypedHandle},
};

use crate::attributes::CuteTileGridAttr;

/// One logical output tile selected by a static scheduler.
///
/// The type remembers the static tile grid. Its defining scheduler operation
/// carries the current linear tile number; no `WorkTile` struct reaches the
/// runtime ABI.
///
/// ```text
/// current linear number + grid<16,16,1>
///                    │
///                    ▼
///        !cute.work_tile<grid<16,16,1>>
///                    │ coordinates
///                    ▼
///              linear, m, n, batch
/// ```
#[pliron_type(name = "cute.work_tile", format = "`<` $grid `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteWorkTileType {
    /// Static M/N/batch dimensions of the output tile grid.
    pub grid: CuteTileGridAttr,
}

impl CuteWorkTileType {
    /// Create the short-lived semantic handle for one scheduler iteration.
    pub fn get(ctx: &mut Context, grid: CuteTileGridAttr) -> TypedHandle<Self> {
        Type::instantiate(Self { grid }, ctx)
    }
}

impl Verify for CuteWorkTileType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.grid.verify(ctx)
    }
}
