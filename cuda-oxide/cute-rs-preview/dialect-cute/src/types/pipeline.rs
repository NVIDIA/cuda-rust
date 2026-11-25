/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Barrier-ring configuration and producer/consumer cursor contracts.

use pliron::{
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    location::Location,
    result::Error,
    r#type::{Type, TypedHandle},
    verify_err,
};

/// The shared barrier ring used by TMA loads.
///
/// This is a compiler-only handle. The real storage is still the shared
/// pointer consumed by `cute.tma_load_pipeline_make`:
///
/// ```text
/// shared barrier base ── make<3, 8 warps, 17408 bytes>
///                              │
///                              ▼
///               !cute.tma_load_pipeline<3, 8, 17408>
/// ```
///
/// The handle must disappear before ABI or LLVM lowering. Slot and phase are
/// deliberately *not* part of this type; ordinary `u32` values carry them
/// through the loop.
#[pliron_type(
    name = "cute.tma_load_pipeline",
    format = "`<` $stages `,` $consumer_warps `,` $transaction_bytes `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteTmaLoadPipelineType {
    /// Number of reusable slots in the barrier or output-buffer ring.
    pub stages: u64,
    /// Consumer warps that release each full stage after reading it.
    pub consumer_warps: u32,
    /// Total bytes expected by each full barrier for one input stage.
    pub transaction_bytes: u32,
}

impl CuteTmaLoadPipelineType {
    /// Create a compiler-only TMA load-pipeline handle.
    pub fn get(
        ctx: &mut Context,
        stages: u64,
        consumer_warps: u32,
        transaction_bytes: u32,
    ) -> TypedHandle<Self> {
        Type::instantiate(
            Self {
                stages,
                consumer_warps,
                transaction_bytes,
            },
            ctx,
        )
    }

    /// Return bytes occupied by `[full; stages]` and `[empty; stages]`.
    ///
    /// Each hardware barrier occupies eight bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> Option<u64> {
        self.stages
            .checked_mul(2)
            .and_then(|count| count.checked_mul(8))
    }
}

impl Verify for CuteTmaLoadPipelineType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 || self.stages > u64::from(u32::MAX) {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline stages must be between 1 and {}",
                u32::MAX
            );
        }
        if !(1..=32).contains(&self.consumer_warps) {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline consumer warps must be between 1 and 32"
            );
        }
        if self.transaction_bytes == 0 {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline transaction bytes must be greater than zero"
            );
        }
        if self.storage_bytes().is_none() {
            return verify_err!(
                Location::Unknown,
                "cute.tma_load_pipeline barrier storage size overflowed u64"
            );
        }
        Ok(())
    }
}
