/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural decoding for the first high-level tensor flow.
//!
//! The Rust types keep the story visible:
//!
//! ```text
//! Tensor<_, Contiguous1D>
//!          -> Tensor<_, Zipped1D<N>>
//!          -> Tensor<_, Tile1D<N>>
//! ```
//!
//! This module turns only those exact `cute-rs` ADTs into the ghost
//! `!cute.tensor_view` type. Other `Tensor` layouts continue through the
//! ordinary Rust-struct path until their own high-level operation set exists.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types::translate_type;
use dialect_cute::attributes::{
    CuteTensorAccessAttr, CuteTensorAddressSpaceAttr, CuteTensorLayoutAttr,
};
use dialect_cute::types::CuteTensorViewType;
use pliron::context::Context;
use pliron::input_error_noloc;
use pliron::r#type::TypeHandle;
use rustc_public::CrateDef;
use rustc_public::ty::{
    AdtDef, GenericArgKind, GenericArgs, RigidTy, Ty, TyConst, TyConstKind, TyKind,
};

const TENSOR: &str = "cute_rs::tensor::Tensor";
const TENSOR_MUT: &str = "cute_rs::tensor::TensorMut";
const CONTIGUOUS_1D: &str = "cute_rs::tensor::Contiguous1D";
const ZIPPED_1D: &str = "cute_rs::tensor::Zipped1D";
const TILE_1D: &str = "cute_rs::tensor::Tile1D";
const REGISTER_TILE: &str = "cute_rs::tensor::RegisterTile";

/// Rust-side facts needed by call classification and operation emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorViewRust {
    pub logical: Ty,
    pub storage: Ty,
    pub access: CuteTensorAccessAttr,
    pub layout: CuteTensorLayoutAttr,
}

/// Rust-side facts for `RegisterTile<T, N>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegisterTileRust {
    pub element: Ty,
    pub size: u64,
}

pub(super) fn canonical_adt_path(def: &AdtDef) -> String {
    let crate_name = def.krate().name.to_string();
    let mut segments = Vec::new();
    let mut current = Some(def.def_id());
    while let Some(def_id) = current {
        let printed = def_id.name();
        let segment = printed.as_str().rsplit("::").next().unwrap_or_default();
        segments.push(segment.to_owned());
        current = def_id.parent();
    }
    super::canonical_path_from_leaf_segments(&crate_name, segments)
}

pub(super) fn exact_schema(
    args: &GenericArgs,
    expected: &[&str],
    owner: &str,
) -> Result<(), String> {
    let found: Vec<_> = args
        .0
        .iter()
        .map(|arg| match arg {
            GenericArgKind::Type(_) => "Type",
            GenericArgKind::Const(_) => "Const",
            GenericArgKind::Lifetime(_) => "Lifetime",
        })
        .collect();
    if found == expected {
        Ok(())
    } else {
        Err(format!(
            "`{owner}` expects generics [{}], found [{}]",
            expected.join(", "),
            found.join(", ")
        ))
    }
}

pub(super) fn type_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<Ty, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Type(value)) => Ok(*value),
        _ => Err(format!("`{owner}` generic {index} must be a type")),
    }
}

pub(super) fn const_arg(args: &GenericArgs, index: usize, owner: &str) -> Result<TyConst, String> {
    match args.0.get(index) {
        Some(GenericArgKind::Const(value)) => Ok(value.clone()),
        _ => Err(format!("`{owner}` generic {index} must be a const")),
    }
}

pub(super) fn unsigned_const(value: &TyConst, what: &str) -> Result<u64, String> {
    let raw = match value.kind() {
        TyConstKind::Value(_, allocation) => allocation
            .read_uint()
            .map_err(|error| format!("cannot read {what}: {error:?}"))?,
        _ => u128::from(
            value
                .eval_target_usize()
                .map_err(|error| format!("cannot evaluate {what}: {error:?}"))?,
        ),
    };
    u64::try_from(raw).map_err(|_| format!("{what} value {raw} does not fit in u64"))
}

fn decode_layout(ty: &Ty) -> Result<Option<CuteTensorLayoutAttr>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    let path = canonical_adt_path(&def);
    match path.as_str() {
        CONTIGUOUS_1D => {
            exact_schema(&args, &[], CONTIGUOUS_1D)?;
            Ok(Some(CuteTensorLayoutAttr::Contiguous1D))
        }
        ZIPPED_1D | TILE_1D => {
            exact_schema(&args, &["Const"], &path)?;
            let tile = unsigned_const(&const_arg(&args, 0, &path)?, "tensor tile width")?;
            if tile == 0 {
                return Err("tensor tile width must be greater than zero".to_string());
            }
            Ok(Some(if path == ZIPPED_1D {
                CuteTensorLayoutAttr::Zipped1D(tile)
            } else {
                CuteTensorLayoutAttr::Tile1D(tile)
            }))
        }
        _ => Ok(None),
    }
}

/// Decode an exact v0 `Tensor` or `TensorMut` ADT.
///
/// `Ok(None)` means either “not a tensor” or “a tensor using a layout not in
/// v0”; both intentionally keep the existing generic ADT translation.
pub(crate) fn decode_tensor_view(ty: &Ty) -> Result<Option<TensorViewRust>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    let path = canonical_adt_path(&def);
    let access = match path.as_str() {
        TENSOR => CuteTensorAccessAttr::ReadOnly,
        TENSOR_MUT => CuteTensorAccessAttr::ReadWrite,
        _ => return Ok(None),
    };
    exact_schema(&args, &["Lifetime", "Type", "Type", "Type"], &path)?;
    let logical = type_arg(&args, 1, &path)?;
    let Some(layout) = decode_layout(&type_arg(&args, 2, &path)?)? else {
        return Ok(None);
    };
    let storage = type_arg(&args, 3, &path)?;
    Ok(Some(TensorViewRust {
        logical,
        storage,
        access,
        layout,
    }))
}

/// Decode a tensor view behind exactly one Rust reference or raw pointer.
pub(crate) fn decode_tensor_view_receiver(ty: &Ty) -> Result<Option<TensorViewRust>, String> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, pointee, _) | RigidTy::RawPtr(pointee, _)) => {
            decode_tensor_view(&pointee)
        }
        _ => decode_tensor_view(ty),
    }
}

/// Decode an exact `RegisterTile<T, N>`.
pub(crate) fn decode_register_tile(ty: &Ty) -> Result<Option<RegisterTileRust>, String> {
    let TyKind::RigidTy(RigidTy::Adt(def, args)) = ty.kind() else {
        return Ok(None);
    };
    if canonical_adt_path(&def) != REGISTER_TILE {
        return Ok(None);
    }
    exact_schema(&args, &["Type", "Const"], REGISTER_TILE)?;
    let element = type_arg(&args, 0, REGISTER_TILE)?;
    let size = unsigned_const(&const_arg(&args, 1, REGISTER_TILE)?, "register tile width")?;
    if size == 0 {
        return Err("register tile width must be greater than zero".to_string());
    }
    Ok(Some(RegisterTileRust { element, size }))
}

/// Type-translation hook called before generic ADT lowering.
pub(crate) fn try_translate_tensor_view_type(
    ctx: &mut Context,
    rust_ty: &Ty,
) -> Option<TranslationResult<TypeHandle>> {
    let decoded = match decode_tensor_view(rust_ty) {
        Ok(Some(decoded)) => decoded,
        Ok(None) => return None,
        Err(error) => {
            return Some(Err(input_error_noloc!(TranslationErr::unsupported(
                format!("invalid cute-rs tensor view type: {error}")
            ))));
        }
    };

    Some((|| {
        let logical = translate_type(ctx, &decoded.logical)?;
        let storage = translate_type(ctx, &decoded.storage)?;
        let alignment = decoded
            .storage
            .layout()
            .map_err(|error| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "cannot read cute-rs tensor storage alignment: {error:?}"
                )))
            })?
            .shape()
            .abi_align;
        Ok(CuteTensorViewType::get(
            ctx,
            logical,
            storage,
            CuteTensorAddressSpaceAttr::Gmem,
            decoded.access,
            alignment,
            decoded.layout,
        )
        .into())
    })())
}
