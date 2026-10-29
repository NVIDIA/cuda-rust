/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared placement and asynchronous output for the SM120 GEMM epilogue.
//!
//! Runtime values stay ordinary MIR values. Static attributes explain the
//! shared tile and the store protocol:
//!
//! ```text
//! 64xf32 accumulator
//!         │
//!         ▼
//! epilogue_store_fragment ──► two 128x64 shared f16 halves
//!         │                              │
//!         ▼                              ▼
//! proxy fence + counted sync       two tma_store_2d
//!                                          │
//!                                          ▼
//!                                  one committed group
//! ```
//!
//! The TMA store queue is hardware state owned by the issuing thread. Its
//! acquire, commit, and tail operations therefore have no fake SSA handle or
//! loop-carried counter. Their order in the CFG is the state.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::builtin::{
    attributes::TypeAttr,
    op_interfaces::{NOpdsInterface, NResultsInterface},
    types::{IntegerType, Signedness},
};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::{
    CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
    CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTmaStorePipelineAttr,
};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use crate::types::{CuteEpilogueTileType, CuteTmaViewType, MmaCarrierShape, mma_carrier_shape};

/// Check that the SSA value has an unsigned 32-bit integer type.
fn is_u32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned())
}

/// Check that the SSA value has an unsigned 64-bit integer type.
fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

/// Downcast a type handle to the static shared output-tile description.
fn epilogue_tile_of_type(ctx: &Context, ty: TypeHandle) -> Option<CuteEpilogueTileType> {
    ty.deref(ctx)
        .downcast_ref::<CuteEpilogueTileType>()
        .cloned()
}

/// Read a TMA transport view from an SSA operand's type.
fn tma_view_of(ctx: &Context, value: Value) -> Option<CuteTmaViewType> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<CuteTmaViewType>()
        .cloned()
}

/// Check mutable shared storage with the required element type.
fn shared_pointer_matches(ctx: &Context, value: Value, storage: TypeHandle) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some_and(|pointer| {
            pointer.pointee == storage
                && pointer.is_mutable
                && pointer.address_space == address_space::SHARED
        })
}

/// Require a valid epilogue tile attribute before checking carrier operands.
fn checked_tile(
    tile: Option<CuteEpilogueTileType>,
    ctx: &Context,
) -> Result<CuteEpilogueTileType, String> {
    let tile = tile.ok_or_else(|| "must carry a cute.epilogue_tile TypeAttr".to_owned())?;
    tile.verify(ctx)
        .map_err(|error| format!("carries an invalid epilogue tile: {error}"))?;
    Ok(tile)
}

/// Build the supported 128-by-64 swizzled FP16 half-tile placement.
fn current_half_layout() -> ComposedLayout {
    let outer: Layout = "(128,64):(64,1)"
        .parse()
        .expect("fixed epilogue half layout is valid");
    ComposedLayout::new(Swizzle::new(3, 3, 3), 0, outer, OffsetUnit::Elements)
        .expect("fixed epilogue half swizzle is valid")
}

mod store;
mod tile;

pub use store::{
    CuteTmaStore2dSemanticOp, CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp, CuteTmaStoreTailOp,
};
pub use tile::{
    CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp,
    CuteEpilogueWarpSliceOp,
};

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
