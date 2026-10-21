/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-tensor and tiled-MMA meaning for the GEMM compute loop.
//!
//! Runtime values stay exactly as they are in MIR today. Typed attributes say
//! what those values mean:
//!
//! ```text
//! shared pointer + capacity ── overlay ─────────────────────────────┐
//!                                                                  │
//! lane ── tiled_mma_slice ──────────────────────────────────────┐   │
//!                                                              │   │
//! scales A/B ── load_scales ── slice K=64 ──────────────────┐   │   │
//! A shared tile ─────────────── load_a ───────────────────┐  │   │   │
//! B shared tile ─────────────── partition_b ───────────┐  │  │   │   │
//! zero / previous C ───────────────────────────────────┴──┴──┴───┴───┘
//!                                                       │
//!                                                       ▼
//!                                                  tiled_gemm
//! ```
//!
//! `partition_b` is intentionally lazy. It returns only B's shared pointer,
//! capacity, warp-N position, and K half. `tiled_gemm` can therefore load two
//! B fragments, use them immediately, and then load the next pair. No eager
//! eight-fragment B aggregate exists in this schema.
//!
//! Rust may temporarily store a shared view inside an ordinary aggregate.
//! Consumers therefore repeat its static `!cute.smem_tensor` TypeAttr instead
//! of requiring their pointer/capacity operands to come directly from the
//! overlay operation. Normal and no-inline MIR can flatten to the same ABI.
//!
//! Backend continuations consume the same ordinary carriers: they can erase
//! forwarding operations into native loads and MMA sites or group the values
//! into CuTe MLIR tensors and fragments.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::builtin::{
    attributes::TypeAttr,
    op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    types::{FP32Type, IntegerType},
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
    CuteMmaCarrierKindAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
    CuteTiledMmaPlanAttr, SM1XX_BLOCK_SCALE_ATOM_BYTES,
};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use crate::types::{CuteSmemTensorType, MmaCarrierShape, mma_carrier_shape};

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

/// Check for the FP32 scalar type required by accumulator carriers.
fn is_f32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<FP32Type>()
        .is_some()
}

/// Downcast static metadata to the shared tensor's format and placement.
fn smem_tensor_of_type(ctx: &Context, ty: TypeHandle) -> Option<CuteSmemTensorType> {
    ty.deref(ctx).downcast_ref::<CuteSmemTensorType>().cloned()
}

/// Require mutable shared storage with the view's exact storage element type.
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

/// Check the shared pointer and unsigned capacity carrier against the static tensor view.
fn shared_carriers_match_view(
    ctx: &Context,
    base: Value,
    capacity: Value,
    view: &CuteSmemTensorType,
) -> bool {
    view.tensor_view(ctx).is_some_and(|tensor| {
        shared_pointer_matches(ctx, base, tensor.storage) && is_u64(ctx, capacity)
    })
}

/// Build the supported dense placement for one stage of block-scale carriers.
fn canonical_scale_placement() -> ComposedLayout {
    let inner: Layout = "(32,4):(4,1)"
        .parse()
        .expect("fixed SM120 scale placement is valid");
    ComposedLayout::new(Swizzle::IDENTITY, 0, inner, OffsetUnit::Elements)
        .expect("identity scale placement is valid")
}

/// Build the supported swizzled placement for one stage of packed FP4 values.
fn canonical_data_placement() -> ComposedLayout {
    let inner: Layout = "(128,32):(32,1)"
        .parse()
        .expect("fixed SM120 data placement is valid");
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements)
        .expect("fixed SM120 data swizzle is valid")
}

/// Clone an optional borrowed tiled-MMA attribute for a consumer.
fn plan_of_attr(
    plan: Option<std::cell::Ref<'_, CuteTiledMmaPlanAttr>>,
) -> Option<CuteTiledMmaPlanAttr> {
    plan.map(|plan| plan.clone())
}

/// Restrict tiled MMA to the supported SM120 shape, atom, and ownership layouts.
fn check_plan(plan: &CuteTiledMmaPlanAttr, ctx: &Context) -> Result<(), &'static str> {
    plan.verify(ctx)
        .map_err(|_| "has an invalid tiled-MMA plan")?;
    let valid_atom = plan.atom == crate::attributes::CuteMmaAtomAttr::mxf4_m16n8k64()
        || plan.atom == crate::attributes::CuteMmaAtomAttr::nvf4_m16n8k64();
    if !valid_atom
        || plan.cta_m != 128
        || plan.cta_n != 128
        || plan.cta_k != 128
        || plan.warp_m != 4
        || plan.warp_n != 2
        || plan.b_load_group != 2
        || plan.shared_layout.0 != canonical_data_placement()
    {
        return Err("v0 supports the 128x128x128 SM120 MXFP4/NVFP4 plan");
    }
    let canonical = CuteTiledMmaPlanAttr::mxf4_128x128x128(plan.shared_layout.0.clone());
    if plan.m_ownership != canonical.m_ownership || plan.n_ownership != canonical.n_ownership {
        return Err("v0 needs the SM120 2x8 warp-cell ownership maps");
    }
    Ok(())
}

/// Require the packed-value format, operand role, and shared placement needed by tiled MMA.
fn check_data_view(
    ctx: &Context,
    view: &CuteSmemTensorType,
    plan: &CuteTiledMmaPlanAttr,
    role: CuteTensorRoleAttr,
) -> Result<(), &'static str> {
    view.verify(ctx)
        .map_err(|_| "uses an invalid shared tensor type")?;
    let tensor = view.tensor_view(ctx).ok_or("must wrap a tensor view")?;
    if tensor.format != CuteTensorFormatAttr::E2M1
        || tensor.role != role
        || tensor.layout != CuteTensorLayoutAttr::KMajor
        || tensor.alignment.0 < 16
        || tensor
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
        || view.placement != plan.shared_layout
        || view.storage_elements() != Some(4096)
        || view.storage_bytes(ctx) != Some(8192)
    {
        return Err("must be the matching 128x128 packed-E2M1 shared tile");
    }
    Ok(())
}

/// Require the scale format, operand role, and canonical stage placement needed by tiled MMA.
fn check_scale_view(
    ctx: &Context,
    view: &CuteSmemTensorType,
    plan: &CuteTiledMmaPlanAttr,
    role: CuteTensorRoleAttr,
) -> Result<(), &'static str> {
    view.verify(ctx)
        .map_err(|_| "uses an invalid shared tensor type")?;
    let tensor = view.tensor_view(ctx).ok_or("must wrap a tensor view")?;
    let storage_is_u32 = tensor
        .storage
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned());
    if tensor.format != plan.atom.scale_format
        || tensor.role != role
        || tensor.layout != CuteTensorLayoutAttr::BlockScaleKMajor(plan.atom.values_per_scale)
        || tensor.alignment.0 < 4
        || !storage_is_u32
        || view.placement.0 != canonical_scale_placement()
        || view.storage_elements() != Some(SM1XX_BLOCK_SCALE_ATOM_BYTES / 4)
        || view.storage_bytes(ctx) != Some(SM1XX_BLOCK_SCALE_ATOM_BYTES)
    {
        return Err("must be the matching 128-row canonical UE8M0/E4M3 shared scale atom");
    }
    Ok(())
}

/// Require the ordinary aggregate's register counts to match its semantic fragment kind.
fn check_carrier(
    ctx: &Context,
    value: Value,
    expected: MmaCarrierShape,
) -> Result<(), &'static str> {
    if mma_carrier_shape(ctx, value.get_type(ctx)) == Some(expected) {
        Ok(())
    } else {
        Err("has the wrong ordinary MIR carrier shape")
    }
}

mod fragment;
mod gemm;
mod ldmatrix;
mod load;
mod tensor;

pub use fragment::{CuteFragmentFillOp, CuteFragmentSliceKOp};
pub use gemm::CuteTiledGemmOp;
pub use ldmatrix::CuteLdmatrixOp;
pub use load::{CuteMmaLoadAOp, CuteMmaLoadScalesOp, CuteMmaPartitionBOp};
pub use tensor::{CuteSmemTensorOverlayOp, CuteTiledMmaSliceOp};

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
