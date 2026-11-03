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

pub(crate) mod emit;

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

/// Exact defining path of `cute_rs::block_scaled::tensor::BlockScaledTensor`; public reexports do not change it.
const BLOCK_SCALED_TENSOR: &str = "cute_rs::block_scaled::tensor::BlockScaledTensor";
/// Exact defining path of `cute_rs::block_scaled::gemv::BlockScaledThreadRow`; public reexports do not change it.
const BLOCK_SCALED_THREAD_ROW: &str = "cute_rs::block_scaled::gemv::BlockScaledThreadRow";
/// Exact defining path of `cute_rs::block_scaled::gemv::BlockScaledTile64`; public reexports do not change it.
const BLOCK_SCALED_TILE_64: &str = "cute_rs::block_scaled::gemv::BlockScaledTile64";
/// Exact defining path of `cute_rs::block_scaled::gemv::LoadedBlockScaledTile64`; public reexports do not change it.
const LOADED_BLOCK_SCALED_TILE_64: &str = "cute_rs::block_scaled::gemv::LoadedBlockScaledTile64";
/// Exact defining path of `cute_rs::block_scaled::gemv::Nvfp4ThreadRow`; public reexports do not change it.
const NVFP4_THREAD_ROW: &str = "cute_rs::block_scaled::gemv::Nvfp4ThreadRow";
/// Exact defining path of `cute_rs::block_scaled::gemv::Nvfp4Tile64`; public reexports do not change it.
const NVFP4_TILE_64: &str = "cute_rs::block_scaled::gemv::Nvfp4Tile64";
/// Exact defining path of `cute_rs::block_scaled::gemv::LoadedNvfp4Tile64`; public reexports do not change it.
const LOADED_NVFP4_TILE_64: &str = "cute_rs::block_scaled::gemv::LoadedNvfp4Tile64";
/// Exact defining path of `cute_rs::block_scaled::layouts::KMajor`; public reexports do not change it.
const K_MAJOR: &str = "cute_rs::block_scaled::layouts::KMajor";
/// Exact defining path of `cute_rs::block_scaled::Mkl`; public reexports do not change it.
const MKL: &str = "cute_rs::block_scaled::Mkl";
/// Exact defining path of `cute_rs::block_scaled::Nkl`; public reexports do not change it.
const NKL: &str = "cute_rs::block_scaled::Nkl";
/// Exact defining path of `cute_rs::numeric::Float4E2M1FN`; public reexports do not change it.
const FLOAT4_E2M1FN: &str = "cute_rs::numeric::Float4E2M1FN";
/// Exact defining path of `cute_rs::numeric::Float8E8M0FNU`; public reexports do not change it.
const FLOAT8_E8M0FNU: &str = "cute_rs::numeric::Float8E8M0FNU";
/// Exact defining path of `cute_rs::numeric::Float8E4M3FN`; public reexports do not change it.
const FLOAT8_E4M3FN: &str = "cute_rs::numeric::Float8E4M3FN";

/// Which stage of the block-scaled view chain one Rust ADT represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockScaledStage {
    /// The complete scaled matrix/vector view before row selection.
    Full,
    /// One selected output row or vector batch.
    Row,
    /// A selected 64-element K tile that has not been loaded.
    KTile64,
    /// Loaded packed values and scales for one K64 dot product.
    Fragment64,
}

/// Static facts used to classify calls and build ghost types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockScaledRust {
    /// Matrix or vector role carried by the source layout marker.
    pub role: CuteTensorRoleAttr,
    /// Scale format (`UE8M0` or `E4M3`).
    pub scale_format: CuteTensorFormatAttr,
    /// Selection stage represented by the scaled source view.
    pub stage: BlockScaledStage,
}

/// Return the defining path and generic arguments of an ADT, or no match for other types.
fn adt_path(ty: &Ty) -> Option<(String, GenericArgs)> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return None;
    };
    Some((canonical_adt_path(&def), args))
}

/// Match the exact marker definition and require its expected empty generic schema.
fn exact_marker(ty: &Ty, expected: &str) -> Result<(), String> {
    let Some((path, args)) = adt_path(ty) else {
        return Err(format!("`{expected}` must be an ADT marker"));
    };
    if path != expected {
        return Err(format!("expected `{expected}`, found `{path}`"));
    }
    exact_schema(&args, &[], expected)
}

/// Decode the exact role marker; reject unsupported or look-alike role types.
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

/// Require a K-major layout and decode its matrix or vector role.
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
            exact_marker(&type_arg(&args, 1, BLOCK_SCALED_TENSOR)?, FLOAT4_E2M1FN)?;
            let scale_ty = type_arg(&args, 2, BLOCK_SCALED_TENSOR)?;
            let scale_format = if exact_marker(&scale_ty, FLOAT8_E8M0FNU).is_ok() {
                CuteTensorFormatAttr::UE8M0
            } else {
                exact_marker(&scale_ty, FLOAT8_E4M3FN)?;
                CuteTensorFormatAttr::E4M3
            };
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
                scale_format,
                stage: BlockScaledStage::Full,
            }
        }
        BLOCK_SCALED_THREAD_ROW | BLOCK_SCALED_TILE_64 => {
            exact_schema(&args, &["Lifetime", "Type"], &path)?;
            BlockScaledRust {
                role: decode_role(&type_arg(&args, 1, &path)?)?,
                scale_format: CuteTensorFormatAttr::UE8M0,
                stage: if path == BLOCK_SCALED_THREAD_ROW {
                    BlockScaledStage::Row
                } else {
                    BlockScaledStage::KTile64
                },
            }
        }
        NVFP4_THREAD_ROW | NVFP4_TILE_64 => {
            exact_schema(&args, &["Lifetime", "Type"], &path)?;
            BlockScaledRust {
                role: decode_role(&type_arg(&args, 1, &path)?)?,
                scale_format: CuteTensorFormatAttr::E4M3,
                stage: if path == NVFP4_THREAD_ROW {
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
                scale_format: CuteTensorFormatAttr::UE8M0,
                stage: BlockScaledStage::Fragment64,
            }
        }
        LOADED_NVFP4_TILE_64 => {
            exact_schema(&args, &["Type"], LOADED_NVFP4_TILE_64)?;
            BlockScaledRust {
                role: decode_role(&type_arg(&args, 0, LOADED_NVFP4_TILE_64)?)?,
                scale_format: CuteTensorFormatAttr::E4M3,
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

/// Build the IR scaled-view type for the decoded role, scale format, and selection stage.
fn scaled_view_type(
    ctx: &mut Context,
    role: CuteTensorRoleAttr,
    scale_format: CuteTensorFormatAttr,
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
        scale_format,
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
        BlockScaledStage::Full => scaled_view_type(
            ctx,
            decoded.role,
            decoded.scale_format,
            CuteScaledLayoutAttr::Full,
        ),
        BlockScaledStage::Row => scaled_view_type(
            ctx,
            decoded.role,
            decoded.scale_format,
            CuteScaledLayoutAttr::Row,
        ),
        BlockScaledStage::KTile64 => scaled_view_type(
            ctx,
            decoded.role,
            decoded.scale_format,
            CuteScaledLayoutAttr::KTile(64),
        ),
        BlockScaledStage::Fragment64 => {
            let source = scaled_view_type(
                ctx,
                decoded.role,
                decoded.scale_format,
                CuteScaledLayoutAttr::KTile(64),
            );
            CuteFragmentType::get(ctx, source).into()
        }
    }))
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check that each source view maps to its expected block-scaled stage.
    #[test]
    fn stage_names_keep_the_source_flow_visible() {
        assert_ne!(BlockScaledStage::Full, BlockScaledStage::Row);
        assert_ne!(BlockScaledStage::KTile64, BlockScaledStage::Fragment64);
    }

    /// Check that scaled IR views retain their role and selection stage.
    #[test]
    fn scaled_types_keep_role_and_selection_stage() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);

        let matrix = scaled_view_type(
            &mut ctx,
            CuteTensorRoleAttr::Mkl,
            CuteTensorFormatAttr::UE8M0,
            CuteScaledLayoutAttr::KTile(64),
        );
        let nvfp4 = scaled_view_type(
            &mut ctx,
            CuteTensorRoleAttr::Nkl,
            CuteTensorFormatAttr::E4M3,
            CuteScaledLayoutAttr::KTile(64),
        );
        let matrix_type = matrix.deref(&ctx);
        let matrix = matrix_type
            .downcast_ref::<CuteScaledViewType>()
            .expect("scaled view");
        assert_eq!(matrix.role, CuteTensorRoleAttr::Mkl);
        assert_eq!(matrix.k_width(), Some(64));

        let nvfp4_type = nvfp4.deref(&ctx);
        let nvfp4 = nvfp4_type
            .downcast_ref::<CuteScaledViewType>()
            .expect("nvfp4 scaled view");
        assert_eq!(nvfp4.role, CuteTensorRoleAttr::Nkl);
        assert_eq!(nvfp4.k_width(), Some(64));
    }
}
