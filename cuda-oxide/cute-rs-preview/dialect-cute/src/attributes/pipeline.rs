/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Barrier-ring configuration and producer/consumer cursor contracts.

use pliron::{common_traits::Verify, context::Context, result::Error, verify_err_noloc};
use pliron_derive::pliron_attr;

/// Which side of a staged load owns one ring position.
///
/// ```text
/// Producer  waits for an empty stage, then starts TMA
/// Consumer  waits for a full stage, then reads and releases it
/// ```
#[pliron_attr(name = "cute.pipeline_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CutePipelineRoleAttr {
    /// Cursor side waiting for empty storage before starting an asynchronous transfer.
    Producer,
    /// Cursor side waiting for completed input before reading and releasing storage.
    Consumer,
}

/// Compile-time facts attached to the ordinary `(slot, phase)` scalars.
///
/// The attribute says how to interpret those scalars. It is not a runtime
/// state object:
///
/// ```text
/// pipeline_state<Producer, 3>
///                    │
///                    └── slot: u32, phase: u32
/// ```
#[pliron_attr(name = "cute.pipeline_state", format = "`<` $role `,` $stages `>`")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CutePipelineStateAttr {
    /// Producer or consumer interpretation of the cursor.
    pub role: CutePipelineRoleAttr,
    /// Number of reusable slots in the barrier or output-buffer ring.
    pub stages: u64,
}

impl CutePipelineStateAttr {
    /// Describe producer or consumer state for one fixed-size stage ring.
    #[must_use]
    pub const fn new(role: CutePipelineRoleAttr, stages: u64) -> Self {
        Self { role, stages }
    }

    /// Describe producer state for one fixed-size stage ring.
    #[must_use]
    pub const fn producer(stages: u64) -> Self {
        Self::new(CutePipelineRoleAttr::Producer, stages)
    }

    /// Describe consumer state for one fixed-size stage ring.
    #[must_use]
    pub const fn consumer(stages: u64) -> Self {
        Self::new(CutePipelineRoleAttr::Consumer, stages)
    }
}

impl Verify for CutePipelineStateAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 || self.stages > u64::from(u32::MAX) {
            return verify_err_noloc!(
                "cute.pipeline_state stages must be between 1 and {}, got {}",
                u32::MAX,
                self.stages
            );
        }
        Ok(())
    }
}
