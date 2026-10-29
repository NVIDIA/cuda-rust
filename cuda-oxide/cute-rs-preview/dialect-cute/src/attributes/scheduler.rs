/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Persistent output-tile grids and their ordinary cursor provenance.

use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

/// The logical output-tile grid owned by a static scheduler.
///
/// M changes first, then N, then batch (also called L):
///
/// ```text
/// grid<3, 2, 1>
///
/// linear:  0      1      2      3      4      5
/// tile:   0,0    1,0    2,0    0,1    1,1    2,1
/// ```
///
/// All three sizes are positive. Their product must fit in `u64`, which is
/// the scalar type carried through the scheduler loop.
#[pliron_attr(
    name = "cute.tile_grid",
    format = "`<` $m_tiles `,` $n_tiles `,` $batches `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteTileGridAttr {
    /// Number of logical output tiles along M.
    pub m_tiles: u64,
    /// Number of logical output tiles along N.
    pub n_tiles: u64,
    /// Number of independent output batches.
    pub batches: u64,
}

impl CuteTileGridAttr {
    /// Describe a grid where M changes first, then N, then batch.
    #[must_use]
    pub const fn new(m_tiles: u64, n_tiles: u64, batches: u64) -> Self {
        Self {
            m_tiles,
            n_tiles,
            batches,
        }
    }

    /// Return the number of logical work tiles when multiplication is safe.
    #[must_use]
    pub const fn total_tiles(self) -> Option<u64> {
        let Some(mn) = self.m_tiles.checked_mul(self.n_tiles) else {
            return None;
        };
        mn.checked_mul(self.batches)
    }
}

impl Verify for CuteTileGridAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.m_tiles == 0 || self.n_tiles == 0 || self.batches == 0 {
            return verify_err_noloc!("cute.tile_grid sizes must be greater than zero");
        }
        if self.total_tiles().is_none() {
            return verify_err_noloc!("cute.tile_grid tile count must fit in u64");
        }
        Ok(())
    }
}
