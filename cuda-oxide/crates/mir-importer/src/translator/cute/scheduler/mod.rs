/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural Rust types for the persistent scheduler boundary.
//!
//! The scheduler itself remains two ordinary `usize` fields. Only the
//! short-lived selected tile becomes a semantic handle:
//!
//! ```text
//! StaticPersistentTileScheduler<M,N,B>   ordinary Rust state
//!                    │ get_current_work
//!                    ▼
//!          WorkTileInfo<M,N,B>           !cute.work_tile<grid<M,N,B>>
//! ```

pub(crate) mod emit;

use crate::error::{TranslationErr, TranslationResult};
use dialect_cute::attributes::CuteTileGridAttr;
use dialect_cute::types::CuteWorkTileType;
use pliron::context::Context;
use pliron::input_error_noloc;
use pliron::r#type::TypeHandle;
use rustc_public::ty::{GenericArgs, RigidTy, Ty, TyKind};

use super::tensor::{canonical_adt_path, const_arg, exact_schema, unsigned_const};

/// Exact defining path of `cute_rs::scheduler::StaticPersistentTileScheduler`; public reexports do not change it.
const SCHEDULER: &str = "cute_rs::scheduler::StaticPersistentTileScheduler";
/// Exact defining path of `cute_rs::scheduler::WorkTileInfo`; public reexports do not change it.
const WORK_TILE: &str = "cute_rs::scheduler::WorkTileInfo";

/// Recognize one exact source ADT rather than a type with a similar printed name.
fn exact_adt(ty: &Ty, expected: &str) -> Result<Option<GenericArgs>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    if canonical_adt_path(&def) != expected {
        return Ok(None);
    }
    exact_schema(&args, &["Const", "Const", "Const"], expected)?;
    Ok(Some(args))
}

/// Decode constant M/N/batch grid dimensions from a scheduler or work-tile ADT.
fn grid_from_args(args: &GenericArgs, owner: &str) -> Result<CuteTileGridAttr, String> {
    let m_tiles = unsigned_const(&const_arg(args, 0, owner)?, "M tile count")?;
    let n_tiles = unsigned_const(&const_arg(args, 1, owner)?, "N tile count")?;
    let batches = unsigned_const(&const_arg(args, 2, owner)?, "batch count")?;
    let grid = CuteTileGridAttr::new(m_tiles, n_tiles, batches);
    if m_tiles == 0 || n_tiles == 0 || batches == 0 {
        return Err("scheduler tile-grid sizes must be greater than zero".to_string());
    }
    if grid.total_tiles().is_none() {
        return Err("scheduler tile-grid product must fit in u64".to_string());
    }
    Ok(grid)
}

/// Decode the ordinary two-scalar scheduler ADT.
pub(crate) fn decode_scheduler(ty: &Ty) -> Result<Option<CuteTileGridAttr>, String> {
    exact_adt(ty, SCHEDULER)?
        .map(|args| grid_from_args(&args, SCHEDULER))
        .transpose()
}

/// Decode the semantic work-tile ADT.
pub(crate) fn decode_work_tile(ty: &Ty) -> Result<Option<CuteTileGridAttr>, String> {
    exact_adt(ty, WORK_TILE)?
        .map(|args| grid_from_args(&args, WORK_TILE))
        .transpose()
}

/// Decode scheduler facts from a value or one borrowed/raw-pointer receiver.
fn decode_behind_one_pointer(
    ty: &Ty,
    decode: fn(&Ty) -> Result<Option<CuteTileGridAttr>, String>,
) -> Result<Option<CuteTileGridAttr>, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode(&pointee)
        }
        _ => decode(ty),
    }
}

/// Recover the constant tile grid from a scheduler receiver.
pub(crate) fn decode_scheduler_receiver(ty: &Ty) -> Result<Option<CuteTileGridAttr>, String> {
    decode_behind_one_pointer(ty, decode_scheduler)
}

/// Recover the constant tile grid from a work-tile receiver.
pub(crate) fn decode_work_tile_receiver(ty: &Ty) -> Result<Option<CuteTileGridAttr>, String> {
    decode_behind_one_pointer(ty, decode_work_tile)
}

/// Build a work-tile IR type that retains the complete logical grid.
fn work_tile_type(ctx: &mut Context, grid: CuteTileGridAttr) -> TypeHandle {
    CuteWorkTileType::get(ctx, grid).into()
}

/// Type-translation hook called before generic ADT lowering.
pub(crate) fn try_translate_work_tile_type(
    ctx: &mut Context,
    rust_ty: &Ty,
) -> Option<TranslationResult<TypeHandle>> {
    match decode_work_tile(rust_ty) {
        Ok(Some(grid)) => Some(Ok(work_tile_type(ctx, grid))),
        Ok(None) => None,
        Err(error) => Some(Err(input_error_noloc!(TranslationErr::unsupported(
            format!("invalid cute-rs scheduler work tile: {error}")
        )))),
    }
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod tests {
    use super::*;

    /// Check that work-tile types retain all three grid dimensions.
    #[test]
    fn work_tile_type_keeps_the_whole_grid() {
        let mut ctx = Context::new();
        dialect_cute::register(&mut ctx);
        let ty = work_tile_type(&mut ctx, CuteTileGridAttr::new(16, 16, 1));
        let ty = ty.deref(&ctx);
        let tile = ty.downcast_ref::<CuteWorkTileType>().unwrap();
        assert_eq!(tile.grid, CuteTileGridAttr::new(16, 16, 1));
    }
}
