/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust type decoding for the block-scaled GEMV flow.
//!
//! The four Rust view types keep one selection story visible:
//!
//! ```text
//! BlockScaledTensor -> BlockScaledThreadRow -> BlockScaledTile64
//!                                               |
//!                                               v load
//!                                      LoadedBlockScaledTile64
//! ```
//!
//! Only the exact E2M1 + UE8M0, 16-values-per-scale path becomes a ghost CuTe
//! type. Other block-scaled types keep their ordinary Rust representation.

use crate::error::{TranslationErr, TranslationResult};
use dialect_cute::attributes::{
    CuteScaledLayoutAttr, CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorFormatAttr,
    CuteTensorLayoutAttr, CuteTensorRoleAttr,
};
use dialect_cute::types::{CuteFragmentType, CuteScaledViewType, CuteTensorViewType};
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::Context;
use pliron::input_error_noloc;
use pliron::r#type::TypeHandle;
use rustc_public::ty::{GenericArgs, RigidTy, Ty, TyKind};

use super::tensor::{canonical_adt_path, const_arg, exact_schema, type_arg, unsigned_const};

const BLOCK_SCALED_TENSOR: &str = "cute_rs::block_scaled::BlockScaledTensor";
const BLOCK_SCALED_THREAD_ROW: &str = "cute_rs::block_scaled::BlockScaledThreadRow";
const BLOCK_SCALED_TILE_64: &str = "cute_rs::block_scaled::BlockScaledTile64";
const LOADED_BLOCK_SCALED_TILE_64: &str = "cute_rs::block_scaled::LoadedBlockScaledTile64";
const K_MAJOR: &str = "cute_rs::block_scaled::KMajor";
const MKL: &str = "cute_rs::block_scaled::Mkl";
const NKL: &str = "cute_rs::block_scaled::Nkl";
const E2M1: &str = "cute_rs::numeric::E2M1";
const UE8M0: &str = "cute_rs::numeric::UE8M0";

/// Which stage of the block-scaled view chain one Rust ADT represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockScaledStage {
    Full,
    Row,
    KTile64,
    Fragment64,
}

/// Static facts used to classify calls and build ghost types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockScaledRust {
    pub role: CuteTensorRoleAttr,
    pub stage: BlockScaledStage,
}

fn adt_path(ty: &Ty) -> Option<(String, GenericArgs)> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return None;
    };
    Some((canonical_adt_path(&def), args))
}

fn exact_marker(ty: &Ty, expected: &str) -> Result<(), String> {
    let Some((path, args)) = adt_path(ty) else {
        return Err(format!("`{expected}` must be an ADT marker"));
    };
    if path != expected {
        return Err(format!("expected `{expected}`, found `{path}`"));
    }
    exact_schema(&args, &[], expected)
}

fn decode_role(ty: &Ty) -> Result<CuteTensorRoleAttr, String> {
    let Some((path, args)) = adt_path(ty) else {
        return Err("block-scaled role must be an ADT marker".to_string());
    };
    exact_schema(&args, &[], &path)?;
    match path.as_str() {
        MKL => Ok(CuteTensorRoleAttr::Mkl),
        NKL => Ok(CuteTensorRoleAttr::Nkl),
        _ => Err(format!(
            "block-scaled role must be `{MKL}` or `{NKL}`, found `{path}`"
        )),
    }
}

fn decode_k_major_role(ty: &Ty) -> Result<CuteTensorRoleAttr, String> {
    let Some((path, args)) = adt_path(ty) else {
        return Err("block-scaled value layout must be KMajor".to_string());
    };
    if path != K_MAJOR {
        return Err(format!("expected `{K_MAJOR}`, found `{path}`"));
    }
    exact_schema(&args, &["Type"], K_MAJOR)?;
    decode_role(&type_arg(&args, 0, K_MAJOR)?)
}

/// Decode one exact Rust ADT in the GEMV block-scaled chain.
pub(crate) fn decode_block_scaled(ty: &Ty) -> Result<Option<BlockScaledRust>, String> {
    let Some((path, args)) = adt_path(ty) else {
        return Ok(None);
    };

    let decoded = match path.as_str() {
        BLOCK_SCALED_TENSOR => {
            exact_schema(
                &args,
                &["Lifetime", "Type", "Type", "Const", "Type"],
                BLOCK_SCALED_TENSOR,
            )?;
            exact_marker(&type_arg(&args, 1, BLOCK_SCALED_TENSOR)?, E2M1)?;
            exact_marker(&type_arg(&args, 2, BLOCK_SCALED_TENSOR)?, UE8M0)?;
            let values_per_scale = unsigned_const(
                &const_arg(&args, 3, BLOCK_SCALED_TENSOR)?,
                "block-scaled values per scale",
            )?;
            if values_per_scale != 16 {
                return Err(format!(
                    "GEMV block-scaled views require 16 values per scale, found {values_per_scale}"
                ));
            }
            BlockScaledRust {
                role: decode_k_major_role(&type_arg(&args, 4, BLOCK_SCALED_TENSOR)?)?,
                stage: BlockScaledStage::Full,
            }
        }
        BLOCK_SCALED_THREAD_ROW | BLOCK_SCALED_TILE_64 => {
            exact_schema(&args, &["Lifetime", "Type"], &path)?;
            BlockScaledRust {
                role: decode_role(&type_arg(&args, 1, &path)?)?,
                stage: if path == BLOCK_SCALED_THREAD_ROW {
                    BlockScaledStage::Row
                } else {
                    BlockScaledStage::KTile64
                },
            }
        }
        LOADED_BLOCK_SCALED_TILE_64 => {
            exact_schema(&args, &["Type"], LOADED_BLOCK_SCALED_TILE_64)?;
            BlockScaledRust {
                role: decode_role(&type_arg(&args, 0, LOADED_BLOCK_SCALED_TILE_64)?)?,
                stage: BlockScaledStage::Fragment64,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(decoded))
}

/// Decode a block-scaled value behind exactly one reference or raw pointer.
pub(crate) fn decode_block_scaled_receiver(ty: &Ty) -> Result<Option<BlockScaledRust>, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode_block_scaled(&pointee)
        }
        _ => decode_block_scaled(ty),
    }
}

fn scaled_view_type(
    ctx: &mut Context,
    role: CuteTensorRoleAttr,
    layout: CuteScaledLayoutAttr,
) -> TypeHandle {
    let logical: TypeHandle = FP32Type::get(ctx).into();
    let storage: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let values: TypeHandle = CuteTensorViewType::get_with_facts(
        ctx,
        logical,
        storage,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        1,
        CuteTensorFormatAttr::E2M1,
        role,
        CuteTensorLayoutAttr::KMajor,
    )
    .into();
    let scales: TypeHandle = CuteTensorViewType::get_with_facts(
        ctx,
        logical,
        storage,
        CuteTensorAddressSpaceAttr::Gmem,
        CuteTensorAccessAttr::ReadOnly,
        1,
        CuteTensorFormatAttr::UE8M0,
        role,
        CuteTensorLayoutAttr::BlockScaleKMajor(16),
    )
    .into();
    CuteScaledViewType::get(ctx, values, scales, role, layout).into()
}

/// Type-translation hook called before generic ADT lowering.
pub(crate) fn try_translate_block_scaled_type(
    ctx: &mut Context,
    rust_ty: &Ty,
) -> Option<TranslationResult<TypeHandle>> {
    let decoded = match decode_block_scaled(rust_ty) {
        Ok(Some(decoded)) => decoded,
        Ok(None) => return None,
        Err(error) => {
            return Some(Err(input_error_noloc!(TranslationErr::unsupported(
                format!("invalid cute-rs block-scaled type: {error}")
            ))));
        }
    };

    Some(Ok(match decoded.stage {
        BlockScaledStage::Full => scaled_view_type(ctx, decoded.role, CuteScaledLayoutAttr::Full),
        BlockScaledStage::Row => scaled_view_type(ctx, decoded.role, CuteScaledLayoutAttr::Row),
        BlockScaledStage::KTile64 => {
            scaled_view_type(ctx, decoded.role, CuteScaledLayoutAttr::KTile(64))
        }
        BlockScaledStage::Fragment64 => {
            let source = scaled_view_type(ctx, decoded.role, CuteScaledLayoutAttr::KTile(64));
            CuteFragmentType::get(ctx, source).into()
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_names_keep_the_source_flow_visible() {
        assert_ne!(BlockScaledStage::Full, BlockScaledStage::Row);
        assert_ne!(BlockScaledStage::KTile64, BlockScaledStage::Fragment64);
    }

    #[test]
    fn scaled_types_keep_role_and_selection_stage() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let matrix = scaled_view_type(
            &mut ctx,
            CuteTensorRoleAttr::Mkl,
            CuteScaledLayoutAttr::KTile(64),
        );
        let matrix_type = matrix.deref(&ctx);
        let matrix = matrix_type
            .downcast_ref::<CuteScaledViewType>()
            .expect("scaled view");
        assert_eq!(matrix.role, CuteTensorRoleAttr::Mkl);
        assert_eq!(matrix.k_width(), Some(64));
    }
}
