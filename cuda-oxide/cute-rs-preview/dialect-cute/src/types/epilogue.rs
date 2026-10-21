/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared output placement and asynchronous store synchronization contracts.

use dialect_mir::types::MirFP16Type;
use pliron::{
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    location::Location,
    result::Error,
    r#type::{Type, TypeHandle, TypedHandle},
    verify_err,
};

use crate::attributes::CuteEpiloguePlanAttr;

/// Static meaning of the shared result allocation used by one tiled MMA.
///
/// This type is carried only by operation attributes. Runtime values remain
/// the same ordinary pointer, warp/lane scalars, and FP32 accumulator used by
/// the Rust kernel:
///
/// ```text
/// AS3 f16 pointer + 64xf32 accumulator
///              │  TypeAttr<!cute.epilogue_tile<...>>
///              ▼
///       two 128x64 shared halves
/// ```
#[pliron_type(name = "cute.epilogue_tile", format = "`<` $storage `,` $plan `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CuteEpilogueTileType {
    /// MIR element type physically stored in memory.
    pub storage: TypeHandle,
    /// Static layout and instruction contract used by this fixture or type.
    pub plan: CuteEpiloguePlanAttr,
}

impl CuteEpilogueTileType {
    /// Attach the current epilogue placement to its physical storage type.
    pub fn get(
        ctx: &mut Context,
        storage: TypeHandle,
        plan: CuteEpiloguePlanAttr,
    ) -> TypedHandle<Self> {
        Type::instantiate(Self { storage, plan }, ctx)
    }

    /// Return the element capacity of one output half, or `None` on overflow.
    #[must_use]
    pub fn half_elements(&self) -> Option<u64> {
        self.plan.half_elements()
    }

    /// Return the element capacity of the complete output tile, or `None` on overflow.
    #[must_use]
    pub fn full_elements(&self) -> Option<u64> {
        self.plan.full_elements()
    }

    /// Return one output half's byte capacity, or `None` for an invalid size.
    #[must_use]
    pub fn half_bytes(&self) -> Option<u64> {
        self.plan.half_bytes(2)
    }

    /// Return the complete output tile's byte capacity, or `None` for an invalid size.
    #[must_use]
    pub fn full_bytes(&self) -> Option<u64> {
        self.plan.full_bytes(2)
    }
}

impl Verify for CuteEpilogueTileType {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        if self
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
        {
            return verify_err!(
                Location::Unknown,
                "cute.epilogue_tile v0 storage must be f16"
            );
        }
        self.plan.verify(ctx)?;
        if self.half_elements() != Some(8192)
            || self.full_elements() != Some(16384)
            || self.half_bytes() != Some(16 * 1024)
            || self.full_bytes() != Some(32 * 1024)
        {
            return verify_err!(
                Location::Unknown,
                "cute.epilogue_tile v0 must be two 16 KiB halves in one 32 KiB allocation"
            );
        }
        Ok(())
    }
}
