/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural Rust facts for the SM120 output-tile boundary.
//!
//! None of these Rust types becomes a ghost. The importer uses their exact
//! identity to attach one epilogue plan to the same pointer, warp/lane, and
//! accumulator carriers already present in MIR.

pub(crate) mod emit;

use dialect_cute::attributes::{CuteEpiloguePlanAttr, CuteTiledMmaPlanAttr};
use dialect_cute::types::CuteEpilogueTileType;
use dialect_mir::types::MirFP16Type;
use pliron::context::Context;
use pliron::r#type::TypeHandle;
use rustc_public::mir;
use rustc_public::ty::{GenericArgKind, GenericArgs, RigidTy, Ty, TyKind};

use super::smem_mma::canonical_data_placement;
use super::tensor::{canonical_adt_path, const_arg, exact_schema, unsigned_const};

/// Exact defining path of `cute_rs::epilogue::sm120::Sm120Epilogue128x128`; public reexports do not change it.
const EPILOGUE_TILE: &str = "cute_rs::epilogue::sm120::Sm120Epilogue128x128";
/// Exact defining path of `cute_rs::epilogue::sm120::Sm120EpilogueWarp128x128`; public reexports do not change it.
const EPILOGUE_WARP_SLICE: &str = "cute_rs::epilogue::sm120::Sm120EpilogueWarp128x128";
/// Exact defining path of `cute_rs::mma::block_scaled::fragments::Mxf4AccumulatorTile2x8`; public reexports do not change it.
const ACCUMULATOR: &str = "cute_rs::mma::block_scaled::fragments::Mxf4AccumulatorTile2x8";
/// Exact defining path of `cute_rs::pipeline::PipelineTmaStore`; public reexports do not change it.
const TMA_STORE_PIPELINE: &str = "cute_rs::pipeline::PipelineTmaStore";

/// Recognize one exact source ADT rather than a type with a similar printed name.
fn exact_adt(ty: &Ty, expected: &str) -> Result<Option<GenericArgs>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    if canonical_adt_path(&def) != expected {
        return Ok(None);
    }
    Ok(Some(args))
}

/// Decode a value directly or through one reference or raw-pointer layer.
fn behind_one_pointer(
    ty: &Ty,
    decode: impl FnOnce(&Ty) -> Result<bool, String>,
) -> Result<bool, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode(&pointee)
        }
        _ => decode(ty),
    }
}

/// Check an exact marker identity without accepting unrelated generic arguments.
fn is_exact_marker(ty: &Ty, expected: &str) -> Result<bool, String> {
    let Some(args) = exact_adt(ty, expected)? else {
        return Ok(false);
    };
    exact_schema(&args, &[], expected)?;
    Ok(true)
}

/// Recognize the fixed SM120 shared epilogue tile.
pub(crate) fn is_epilogue_tile(ty: &Ty) -> Result<bool, String> {
    is_exact_marker(ty, EPILOGUE_TILE)
}

/// Recognize the epilogue tile directly or behind one pointer.
pub(crate) fn is_epilogue_tile_receiver(ty: &Ty) -> Result<bool, String> {
    behind_one_pointer(ty, is_epilogue_tile)
}

/// Recognize the fixed warp-owned epilogue view.
pub(crate) fn is_epilogue_warp_slice(ty: &Ty) -> Result<bool, String> {
    is_exact_marker(ty, EPILOGUE_WARP_SLICE)
}

/// Recognize an epilogue warp view directly or behind one pointer.
pub(crate) fn is_epilogue_warp_slice_receiver(ty: &Ty) -> Result<bool, String> {
    behind_one_pointer(ty, is_epilogue_warp_slice)
}

/// Recognize the FP32 accumulator carrier accepted by the epilogue.
pub(crate) fn is_accumulator(ty: &Ty) -> Result<bool, String> {
    behind_one_pointer(ty, |ty| is_exact_marker(ty, ACCUMULATOR))
}

/// Decode and validate the store pipeline's constant stage count.
pub(crate) fn decode_store_pipeline(ty: &Ty) -> Result<Option<u32>, String> {
    let decode = |ty: &Ty| -> Result<Option<u32>, String> {
        let Some(args) = exact_adt(ty, TMA_STORE_PIPELINE)? else {
            return Ok(None);
        };
        exact_schema(&args, &["Const"], TMA_STORE_PIPELINE)?;
        let stages = unsigned_const(
            &const_arg(&args, 0, TMA_STORE_PIPELINE)?,
            "TMA store-pipeline stage count",
        )?;
        let stages = u32::try_from(stages)
            .map_err(|_| "TMA store-pipeline stage count does not fit u32".to_owned())?;
        if stages != 1 {
            return Err(format!(
                "v0 needs one shared epilogue buffer, got {stages} store-pipeline stages"
            ));
        }
        Ok(Some(stages))
    };
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode(&pointee)
        }
        _ => decode(ty),
    }
}

/// Read the one const generic carried by `tma_half` or a store-pipeline helper.
pub(crate) fn one_unsigned_const(func: &mir::Operand, what: &str) -> Result<u64, String> {
    let mir::Operand::Constant(constant) = func else {
        return Err(format!("{what} must be a direct constant function call"));
    };
    let TyKind::RigidTy(RigidTy::FnDef(_, args)) = constant.const_.ty().kind() else {
        return Err(format!("{what} callee is not a FnDef"));
    };
    let constants = args
        .0
        .iter()
        .filter_map(|arg| match arg {
            GenericArgKind::Const(value) => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    if constants.len() != 1 {
        return Err(format!(
            "{what} needs exactly one const generic, found {}",
            constants.len()
        ));
    }
    unsigned_const(constants[0], what)
}

/// Construct the fixed two-half SM120 epilogue IR type.
pub(crate) fn tile_type(ctx: &mut Context) -> TypeHandle {
    let storage: TypeHandle = MirFP16Type::get(ctx).into();
    let mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(canonical_data_placement());
    let plan = CuteEpiloguePlanAttr::sm120_mxf4_128x128(mma);
    CuteEpilogueTileType::get(ctx, storage, plan).into()
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check that the fixed epilogue type retains two 16 KiB halves.
    #[test]
    fn fixed_tile_type_keeps_two_16_kib_halves() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let ty = tile_type(&mut ctx);
        let ty = ty.deref(&ctx);
        let tile = ty.downcast_ref::<CuteEpilogueTileType>().unwrap();
        assert_eq!(tile.half_bytes(), Some(16 * 1024));
        assert_eq!(tile.full_bytes(), Some(32 * 1024));
    }
}
