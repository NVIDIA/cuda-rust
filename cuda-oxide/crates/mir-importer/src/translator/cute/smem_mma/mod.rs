/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural Rust facts for the shared-memory MXFP4 MMA boundary.
//!
//! The runtime carriers stay ordinary Rust values. This module only decodes
//! the exact types that give those values tensor and tiled-MMA meaning.

pub(crate) mod emit;

use dialect_cute::attributes::CuteTensorRoleAttr;
use dialect_cute::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use rustc_public::ty::{FloatTy, GenericArgs, RigidTy, Ty, TyKind, UintTy};

use super::static_config::decode_smem_layout;
use super::tensor::{canonical_adt_path, exact_schema, type_arg};

/// Exact defining path of `cute_rs::copy::tiled::SharedTensor`; public reexports do not change it.
const SHARED_TENSOR: &str = "cute_rs::copy::tiled::SharedTensor";
/// Exact defining path of `cute_rs::numeric::Mxf4E2M1`; public reexports do not change it.
const MXF4_E2M1: &str = "cute_rs::numeric::Mxf4E2M1";
/// Exact defining path of `cute_rs::numeric::UE8M0x4`; public reexports do not change it.
const UE8M0_X4: &str = "cute_rs::numeric::UE8M0x4";
/// Exact defining path of `cute_rs::block_scaled::layouts::Sm120ScaleAtom`; public reexports do not change it.
const SM120_SCALE_ATOM: &str = "cute_rs::block_scaled::layouts::Sm120ScaleAtom";
/// Exact defining path of `cute_rs::block_scaled::Mkl`; public reexports do not change it.
const MKL: &str = "cute_rs::block_scaled::Mkl";
/// Exact defining path of `cute_rs::block_scaled::Nkl`; public reexports do not change it.
const NKL: &str = "cute_rs::block_scaled::Nkl";
/// Exact defining path of `cute_rs::mma::block_scaled::tiled::Mxf4TiledMma`; public reexports do not change it.
const TILED_MMA: &str = "cute_rs::mma::block_scaled::tiled::Mxf4TiledMma";
/// Exact defining path of `cute_rs::mma::block_scaled::fragments::Mxf4AccumulatorTile2x8`; public reexports do not change it.
const ACCUMULATOR: &str = "cute_rs::mma::block_scaled::fragments::Mxf4AccumulatorTile2x8";
/// Exact defining path of `cute_rs::mma::block_scaled::fragments::Mxf4ScaleTile128`; public reexports do not change it.
const SCALE_STAGE: &str = "cute_rs::mma::block_scaled::fragments::Mxf4ScaleTile128";
/// Exact defining path of `cute_rs::mma::block_scaled::fragments::Mxf4ScalePairs128`; public reexports do not change it.
const SCALE_K64: &str = "cute_rs::mma::block_scaled::fragments::Mxf4ScalePairs128";
/// Exact defining path of `cute_rs::mma::block_scaled::fragments::Mxf4BTileK64`; public reexports do not change it.
const B_TILE_K64: &str = "cute_rs::mma::block_scaled::fragments::Mxf4BTileK64";

/// Whether shared storage holds packed matrix data or scale words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SharedTensorKind {
    /// Packed FP4 matrix values carried in shared FP16 storage.
    Data,
    /// Packed UE8M0 scale words in the canonical shared atom.
    Scale,
}

/// Static kind, role and physical placement decoded from a shared tensor type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedTensorRust {
    /// Distinguishes packed-data storage from packed scale words.
    pub kind: SharedTensorKind,
    /// Matrix-A or matrix-B role carried by the exact source marker.
    pub role: CuteTensorRoleAttr,
    /// Physical shared-memory placement, including composition and offset units.
    pub placement: ComposedLayout,
}

/// Shared placement retained by a tiled-MMA or B-tile source carrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TiledMmaRust {
    /// Physical shared-memory placement, including composition and offset units.
    pub placement: ComposedLayout,
}

/// Return an ADT's exact defining path and generic arguments, or no match.
fn adt(ty: &Ty) -> Option<(String, GenericArgs)> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return None;
    };
    Some((canonical_adt_path(&def), args))
}

/// Match the exact marker definition and require its expected empty generic schema.
fn exact_marker(ty: &Ty, expected: &str) -> Result<bool, String> {
    let Some((path, args)) = adt(ty) else {
        return Ok(false);
    };
    if path != expected {
        return Ok(false);
    }
    exact_schema(&args, &[], expected)?;
    Ok(true)
}

/// Decode the matrix-A or matrix-B role from its exact marker identity.
fn role(ty: &Ty) -> Result<CuteTensorRoleAttr, String> {
    if exact_marker(ty, MKL)? {
        Ok(CuteTensorRoleAttr::Mkl)
    } else if exact_marker(ty, NKL)? {
        Ok(CuteTensorRoleAttr::Nkl)
    } else {
        Err(format!("shared tensor role must be `{MKL}` or `{NKL}`"))
    }
}

/// Recognize the compiler's primitive FP16 type.
fn is_f16(ty: &Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Float(FloatTy::F16)))
}

/// Recognize the compiler's primitive unsigned 32-bit type.
fn is_u32(ty: &Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Uint(UintTy::U32)))
}

/// Return the fixed 32-by-4 word layout of one SM120 scale atom.
fn scale_placement() -> ComposedLayout {
    let outer = "(32,4):(4,1)"
        .parse()
        .expect("fixed SM120 scale placement is valid");
    ComposedLayout::from_layout(outer, OffsetUnit::Elements)
}

/// Return the fixed swizzled 128-by-32 packed-data placement.
pub(crate) fn canonical_data_placement() -> ComposedLayout {
    let outer: Layout = "(128,32):(32,1)"
        .parse()
        .expect("fixed SM120 packed-data placement is valid");
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, outer, OffsetUnit::Elements)
        .expect("fixed SM120 packed-data swizzle is valid")
}

/// Decode packed data or scale tensor markers, preserving layout and matrix role.
pub(crate) fn decode_shared_tensor(ty: &Ty) -> Result<Option<SharedTensorRust>, String> {
    let Some((path, args)) = adt(ty) else {
        return Ok(None);
    };
    if path != SHARED_TENSOR {
        return Ok(None);
    }
    exact_schema(&args, &["Type", "Type", "Type", "Type"], SHARED_TENSOR)?;
    let element = type_arg(&args, 0, SHARED_TENSOR)?;
    let storage = type_arg(&args, 1, SHARED_TENSOR)?;
    let layout = type_arg(&args, 2, SHARED_TENSOR)?;
    let role = role(&type_arg(&args, 3, SHARED_TENSOR)?)?;

    if exact_marker(&element, MXF4_E2M1)? && is_f16(&storage) {
        return Ok(Some(SharedTensorRust {
            kind: SharedTensorKind::Data,
            role,
            placement: decode_smem_layout(&layout)?,
        }));
    }
    if exact_marker(&element, UE8M0_X4)?
        && is_u32(&storage)
        && exact_marker(&layout, SM120_SCALE_ATOM)?
    {
        return Ok(Some(SharedTensorRust {
            kind: SharedTensorKind::Scale,
            role,
            placement: scale_placement(),
        }));
    }
    Err("shared MMA tensor must be packed E2M1/f16 data or UE8M0x4/u32 scales".to_string())
}

/// Decode a value directly or through one reference or raw-pointer layer.
fn behind_one_pointer<T>(
    ty: &Ty,
    decode: impl FnOnce(&Ty) -> Result<Option<T>, String>,
) -> Result<Option<T>, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode(&pointee)
        }
        _ => decode(ty),
    }
}

/// Decode a shared tensor value or one borrowed/raw-pointer receiver.
pub(crate) fn decode_shared_tensor_receiver(ty: &Ty) -> Result<Option<SharedTensorRust>, String> {
    behind_one_pointer(ty, decode_shared_tensor)
}

/// Recognize the tiled-MMA definition and decode its shared placement.
pub(crate) fn decode_tiled_mma(ty: &Ty) -> Result<Option<TiledMmaRust>, String> {
    let Some((path, args)) = adt(ty) else {
        return Ok(None);
    };
    if path != TILED_MMA {
        return Ok(None);
    }
    exact_schema(&args, &["Type"], TILED_MMA)?;
    Ok(Some(TiledMmaRust {
        placement: decode_smem_layout(&type_arg(&args, 0, TILED_MMA)?)?,
    }))
}

/// Decode a tiled-MMA value or one borrowed/raw-pointer receiver.
pub(crate) fn decode_tiled_mma_receiver(ty: &Ty) -> Result<Option<TiledMmaRust>, String> {
    behind_one_pointer(ty, decode_tiled_mma)
}

/// Recognize one exact source ADT rather than a type with a similar printed name.
fn exact_adt(ty: &Ty, expected: &str) -> Result<bool, String> {
    exact_marker(ty, expected)
}

/// Recognize the fixed accumulator carrier directly or behind one pointer.
pub(crate) fn is_accumulator(ty: &Ty) -> Result<bool, String> {
    behind_one_pointer(ty, |ty| exact_adt(ty, ACCUMULATOR).map(Some))
        .map(|value| value.unwrap_or(false))
}

/// Recognize the complete 128-row scale-tile carrier.
pub(crate) fn is_scale_stage(ty: &Ty) -> Result<bool, String> {
    exact_adt(ty, SCALE_STAGE)
}

/// Recognize the scale-pair carrier selected for one K64 half.
pub(crate) fn is_scale_k64(ty: &Ty) -> Result<bool, String> {
    exact_adt(ty, SCALE_K64)
}

/// Decode the B-tile lifetime/layout schema and recover its shared placement.
pub(crate) fn decode_b_tile(ty: &Ty) -> Result<Option<TiledMmaRust>, String> {
    let Some((path, args)) = adt(ty) else {
        return Ok(None);
    };
    if path != B_TILE_K64 {
        return Ok(None);
    }
    exact_schema(&args, &["Lifetime", "Type"], B_TILE_K64)?;
    Ok(Some(TiledMmaRust {
        placement: decode_smem_layout(&type_arg(&args, 1, B_TILE_K64)?)?,
    }))
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check that scale placement uses exactly one 128-word atom.
    #[test]
    fn fixed_scale_placement_has_one_128_word_atom() {
        assert_eq!(scale_placement().outer().checked_size(), Some(128));
    }

    /// Check that data placement fixes the canonical swizzled 128-by-32 tile.
    #[test]
    fn canonical_data_placement_locks_the_128_by_32_swizzled_tile() {
        let placement = canonical_data_placement();
        assert_eq!(placement.outer().checked_size(), Some(128 * 32));
        assert_eq!(placement.inner(), Swizzle::new(2, 3, 3));
        assert_eq!(placement.offset(), 0);
        assert_eq!(placement.unit(), OffsetUnit::Elements);
    }
}
