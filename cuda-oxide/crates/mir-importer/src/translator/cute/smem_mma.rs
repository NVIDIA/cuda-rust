/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural Rust facts for the shared-memory MXFP4 MMA boundary.
//!
//! The runtime carriers stay ordinary Rust values. This module only decodes
//! the exact types that give those values tensor and tiled-MMA meaning.

use dialect_cute::attributes::CuteTensorRoleAttr;
use dialect_cute::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use rustc_public::ty::{FloatTy, GenericArgs, RigidTy, Ty, TyKind, UintTy};

use super::static_config::decode_smem_layout;
use super::tensor::{canonical_adt_path, exact_schema, type_arg};

const SHARED_TENSOR: &str = "cute_rs::tiled_copy::SharedTensor";
const MXF4_E2M1: &str = "cute_rs::numeric::Mxf4E2M1";
const UE8M0_X4: &str = "cute_rs::numeric::UE8M0x4";
const SM120_SCALE_ATOM: &str = "cute_rs::block_scaled::Sm120ScaleAtom";
const MKL: &str = "cute_rs::block_scaled::Mkl";
const NKL: &str = "cute_rs::block_scaled::Nkl";
const TILED_MMA: &str = "cute_rs::block_scaled_mma::Mxfp4TiledMma";
const ACCUMULATOR: &str = "cute_rs::block_scaled_mma::Mxf4AccumulatorTile2x8";
const SCALE_STAGE: &str = "cute_rs::block_scaled_mma::Mxf4ScaleTile128";
const SCALE_K64: &str = "cute_rs::block_scaled_mma::Mxf4ScalePairs128";
const B_TILE_K64: &str = "cute_rs::block_scaled_mma::Mxf4BTileK64";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SharedTensorKind {
    Data,
    Scale,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SharedTensorRust {
    pub kind: SharedTensorKind,
    pub role: CuteTensorRoleAttr,
    pub placement: ComposedLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TiledMmaRust {
    pub placement: ComposedLayout,
}

fn adt(ty: &Ty) -> Option<(String, GenericArgs)> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return None;
    };
    Some((canonical_adt_path(&def), args))
}

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

fn role(ty: &Ty) -> Result<CuteTensorRoleAttr, String> {
    if exact_marker(ty, MKL)? {
        Ok(CuteTensorRoleAttr::Mkl)
    } else if exact_marker(ty, NKL)? {
        Ok(CuteTensorRoleAttr::Nkl)
    } else {
        Err(format!("shared tensor role must be `{MKL}` or `{NKL}`"))
    }
}

fn is_f16(ty: &Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Float(FloatTy::F16)))
}

fn is_u32(ty: &Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Uint(UintTy::U32)))
}

fn scale_placement() -> ComposedLayout {
    let inner = "(32,4):(4,1)"
        .parse()
        .expect("fixed SM120 scale placement is valid");
    ComposedLayout::from_layout(inner, OffsetUnit::Elements)
}

pub(crate) fn canonical_data_placement() -> ComposedLayout {
    let inner: Layout = "(128,32):(32,1)"
        .parse()
        .expect("fixed SM120 packed-data placement is valid");
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements)
        .expect("fixed SM120 packed-data swizzle is valid")
}

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

pub(crate) fn decode_shared_tensor_receiver(ty: &Ty) -> Result<Option<SharedTensorRust>, String> {
    behind_one_pointer(ty, decode_shared_tensor)
}

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

pub(crate) fn decode_tiled_mma_receiver(ty: &Ty) -> Result<Option<TiledMmaRust>, String> {
    behind_one_pointer(ty, decode_tiled_mma)
}

fn exact_adt(ty: &Ty, expected: &str) -> Result<bool, String> {
    exact_marker(ty, expected)
}

pub(crate) fn is_accumulator(ty: &Ty) -> Result<bool, String> {
    behind_one_pointer(ty, |ty| exact_adt(ty, ACCUMULATOR).map(Some))
        .map(|value| value.unwrap_or(false))
}

pub(crate) fn is_scale_stage(ty: &Ty) -> Result<bool, String> {
    exact_adt(ty, SCALE_STAGE)
}

pub(crate) fn is_scale_k64(ty: &Ty) -> Result<bool, String> {
    exact_adt(ty, SCALE_K64)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_scale_placement_has_one_128_word_atom() {
        assert_eq!(scale_placement().inner().checked_size(), Some(128));
    }

    #[test]
    fn canonical_data_placement_locks_the_128_by_32_swizzled_tile() {
        let placement = canonical_data_placement();
        assert_eq!(placement.inner().checked_size(), Some(128 * 32));
        assert_eq!(placement.outer(), Swizzle::new(2, 3, 3));
        assert_eq!(placement.offset(), 0);
        assert_eq!(placement.unit(), OffsetUnit::Elements);
    }
}
