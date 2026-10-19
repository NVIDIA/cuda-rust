/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion: maps the old compiler's type info to tile-ir types.
//!
//! The primary conversion is `convert_type`, which parses the Tile IR
//! type string stored in `TileRustType.cuda_tile_ty_str` via
//! `cutile_ir::ir::Type::parse()`. This is a transitional pattern — ideally
//! `TileRustType` would store `cutile_ir::ir::Type` directly.

use super::tile_rust_type::TileRustType;
use cutile_ir::ir::{PointerType, ScalarType, TileElementType, TileType, Type};
use quote::ToTokens;
use syn::FnArg;

/// Convert a `cuda_tile_name` string to a `cutile_ir::ir::ScalarType`.
pub fn scalar_from_name(name: &str) -> Option<ScalarType> {
    match name {
        "i1" => Some(ScalarType::I1),
        "i4" => Some(ScalarType::I4),
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f16" => Some(ScalarType::F16),
        "bf16" => Some(ScalarType::BF16),
        "f32" => Some(ScalarType::F32),
        "tf32" => Some(ScalarType::TF32),
        "f64" => Some(ScalarType::F64),
        "f8e4m3fn" | "f8E4M3FN" => Some(ScalarType::F8E4M3FN),
        "f8e5m2" | "f8E5M2" => Some(ScalarType::F8E5M2),
        "f8e8m0fnu" | "f8E8M0FNU" => Some(ScalarType::F8E8M0FNU),
        "f8e5m3fnu" | "f8E5M3FNU" => Some(ScalarType::F8E5M3FNU),
        "f4e2m1fn" | "f4E2M1FN" => Some(ScalarType::F4E2M1FN),
        "f4e2m1fnx2" => Some(ScalarType::I8),
        // Rust-facing names (unsigned maps to signed CUDA types)
        "bool" => Some(ScalarType::I1),
        "u8" => Some(ScalarType::I8),
        "u16" => Some(ScalarType::I16),
        "u32" => Some(ScalarType::I32),
        "u64" => Some(ScalarType::I64),
        _ => None,
    }
}

/// Returns the pre-parsed `cutile_ir::ir::Type` from a `TileRustType`.
///
/// Prefers the eagerly parsed `tile_ir_ty` field. Falls back to
/// `cuda_tile_name` for scalar types if the full type wasn't available.
pub fn convert_type(old: &TileRustType) -> Option<Type> {
    if let Some(ty) = &old.tile_ir_ty {
        return Some(ty.clone());
    }
    // Fallback: for primitive types without a full type, use the cuda_tile_name.
    let name = old.cuda_tile_name.as_deref()?;
    scalar_from_name(name).map(Type::Scalar)
}

/// Build a tile type from element type name and shape.
pub fn make_tile_type(element_name: &str, shape: &[i64]) -> Option<Type> {
    let scalar = scalar_from_name(element_name)?;
    Some(Type::Tile(TileType {
        shape: shape.to_vec(),
        element_type: TileElementType::Scalar(scalar),
    }))
}

/// Build a tensor view type from element type name, shape, and strides.
pub fn make_tensor_view_type(element_name: &str, shape: &[i64], strides: &[i64]) -> Option<Type> {
    let scalar = scalar_from_name(element_name)?;
    Some(Type::TensorView(cutile_ir::ir::TensorViewType {
        element_type: scalar,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
    }))
}

/// Build a scalar tile type (rank-0 tile).
pub fn make_scalar_tile_type(element_name: &str) -> Option<Type> {
    make_tile_type(element_name, &[])
}

/// Compile a function parameter type from the generated entry point.
///
/// The entry point generator produces parameters of known forms:
/// - `PointerTile<*mut T, {[]}>` → `tile<ptr<T>>`
/// - Primitive scalar (i32, f32, etc.) → `tile<scalar>`
/// - `Tile<T, {[shapes]}>` → `tile<shape x T>`
pub fn compile_entry_param_type(param: &FnArg) -> Option<Type> {
    let FnArg::Typed(typed) = param else {
        return None;
    };
    let ty_str = typed.ty.to_token_stream().to_string();

    // PointerTile<*mut T, {[]}> → tile<ptr<T>>
    if ty_str.contains("PointerTile") {
        let elem = extract_pointer_element_type(&ty_str)?;
        let scalar = scalar_from_name(&elem)?;
        return Some(Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Pointer(Box::new(PointerType { pointee: scalar })),
        }));
    }

    // Try as a primitive Rust type → scalar tile.
    let rust_type = ty_str.trim().to_string();
    if let Some(scalar) = rust_scalar_type(&rust_type) {
        return Some(Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(scalar),
        }));
    }

    None
}

/// Map a Rust primitive type name to a tile-ir scalar type.
fn rust_scalar_type(name: &str) -> Option<ScalarType> {
    match name {
        "bool" => Some(ScalarType::I1),
        "i8" | "u8" => Some(ScalarType::I8),
        "i16" | "u16" => Some(ScalarType::I16),
        "i32" | "u32" => Some(ScalarType::I32),
        "i64" | "u64" => Some(ScalarType::I64),
        "f16" => Some(ScalarType::F16),
        "bf16" => Some(ScalarType::BF16),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "f8e8m0fnu" => Some(ScalarType::F8E8M0FNU),
        "f8e5m3fnu" => Some(ScalarType::F8E5M3FNU),
        "f4e2m1fn" => Some(ScalarType::F4E2M1FN),
        "f4e2m1fnx2" => Some(ScalarType::I8),
        "i4" => Some(ScalarType::I4),
        _ => None,
    }
}

/// Extract the element type name from a PointerTile type string.
fn extract_pointer_element_type(ty_str: &str) -> Option<String> {
    let after_mut = ty_str.split("mut").nth(1)?;
    let trimmed = after_mut.trim();
    let end = trimmed.find([',', '>', ' '])?;
    Some(trimmed[..end].to_string())
}
