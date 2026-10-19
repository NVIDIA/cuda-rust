/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lifetime-free type representation for compiler2.
//!
//! Each `TileRustType` eagerly parses its dialect type string into a
//! `cutile_ir::ir::Type` at construction time, stored in `tile_ir_ty`.
//! The string form is retained in `cuda_tile_ty_str` for debug output.

use super::shared_types::Kind;
use crate::error::JITError;
use crate::generics::{GenericVars, TypeInstance};
use crate::types::*;
use cutile_ir::ir::Type as TileIrType;
use quote::ToTokens;
use std::collections::HashMap;
use syn::ItemImpl;

/// Build a `TypeInstance::StructuredType` for a synthetic tile type with element info.
fn synthetic_tile_instance(rust_ty: syn::Type, element_name: &str, shape: &[i32]) -> TypeInstance {
    let elem_ty = syn::parse_str::<syn::Type>(element_name).unwrap_or(rust_ty.clone());
    TypeInstance::StructuredType(crate::generics::TypeInstanceStructuredType {
        generic_ty: rust_ty.clone(),
        instance_ty: rust_ty,
        primitive_type: Some(crate::generics::TypInstancePrimitiveType::ElementType(
            crate::generics::TypeInstanceElementType {
                generic_ty: elem_ty.clone(),
                instance_ty: elem_ty,
                rust_element_instance_ty: element_name.to_string(),
            },
        )),
        shape: shape.to_vec(),
    })
}

/// Build a `TypeInstance::StructuredType` for a synthetic pointer tile type.
fn synthetic_ptr_instance(
    rust_ty: syn::Type,
    element_name: &str,
    is_mutable: bool,
) -> TypeInstance {
    TypeInstance::StructuredType(crate::generics::TypeInstanceStructuredType {
        generic_ty: rust_ty.clone(),
        instance_ty: rust_ty.clone(),
        primitive_type: Some(crate::generics::TypInstancePrimitiveType::PtrType(
            crate::generics::TypeInstancePtrType {
                generic_ty: rust_ty.clone(),
                instance_ty: rust_ty,
                rust_element_instance_ty: element_name.to_string(),
                is_mutable,
            },
        )),
        shape: vec![],
    })
}

/// A compiled type binding: maps a Rust `syn::Type` to its CUDA Tile type metadata.
///
/// Lifetime-free — all fields are owned. The tile-ir type is eagerly parsed at
/// construction and stored in `tile_ir_ty`.
#[derive(Debug, Clone)]
pub struct TileRustType {
    pub(crate) kind: Kind,
    pub(crate) cuda_tile_name: Option<String>,
    /// The dialect type string (e.g. `"!cuda_tile.tile<128x64xf32>"`).
    /// Retained for debug output; the parsed form is in `tile_ir_ty`.
    pub(crate) cuda_tile_ty_str: Option<String>,
    /// Eagerly parsed tile-ir type. `None` when the type string is absent
    /// or could not be parsed (deferred types).
    pub(crate) tile_ir_ty: Option<TileIrType>,
    pub(crate) params: Vec<TypeParam>,
    pub(crate) rust_ty: syn::Type,
    pub(crate) type_instance: TypeInstance,
}

impl TileRustType {
    pub(crate) fn get_instantiated_rust_element_type(
        &self,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<String> {
        self.type_instance.get_rust_element_instance_ty()
    }
    pub(crate) fn get_cuda_tile_element_type(
        &self,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<Option<String>, JITError> {
        let type_instance = self.type_instance.get_instantiated_type();
        match self.kind {
            Kind::PrimitiveType => Ok(Some(get_cuda_tile_element_type_primitive(
                type_instance,
                primitives,
            ))),
            Kind::StructuredType => Ok(get_cuda_tile_element_type_structured(
                type_instance,
                primitives,
            )),
            _ => JITError::generic(&format!(
                "cannot get element type for {:?} values",
                self.kind
            )),
        }
    }
    pub(crate) fn get_cuda_tile_element_type_prefix(
        &self,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<super::shared_utils::ElementTypePrefix, JITError> {
        let cuda_elem_ty_str = self
            .get_cuda_tile_element_type(primitives)?
            .ok_or_else(|| {
                JITError::generic_err(&format!(
                    "unable to determine element type for `{}`",
                    self.rust_ty.to_token_stream()
                ))
            })?;
        super::shared_utils::ElementTypePrefix::new(&cuda_elem_ty_str)
    }
    pub(crate) fn get_cuda_tile_type_str(&self) -> Option<String> {
        self.cuda_tile_ty_str.clone()
    }

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    pub(crate) fn new_primitive_type(
        cuda_tile_name: String,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
        mut params: Vec<TypeParam>,
        type_instance: TypeInstance,
    ) -> Result<TileRustType, JITError> {
        let rust_ty = type_instance.get_source_type().clone();
        let type_param_str = params
            .iter_mut()
            .map(|tp| tp.instantiate(generic_vars, primitives))
            .collect::<Result<Vec<_>, _>>()?
            .join(",");
        let type_str = format!("{}<{}>", cuda_tile_name, type_param_str);
        let tile_ir_ty = TileIrType::parse(&type_str);
        Ok(TileRustType {
            cuda_tile_ty_str: Some(type_str),
            tile_ir_ty,
            cuda_tile_name: Some(cuda_tile_name),
            params,
            rust_ty,
            type_instance,
            kind: Kind::PrimitiveType,
        })
    }

    pub(crate) fn new_structured_type(
        cuda_tile_name: String,
        generic_args: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
        mut params: Vec<TypeParam>,
        type_instance: TypeInstance,
    ) -> Result<TileRustType, JITError> {
        let rust_ty = type_instance.get_source_type().clone();
        let type_str = if params.is_empty() {
            cuda_tile_name.to_string()
        } else {
            let type_param_str = params
                .iter_mut()
                .map(|tp| tp.instantiate(generic_args, primitives))
                .collect::<Result<Vec<_>, _>>()?
                .join(",");
            format!("{}<{}>", cuda_tile_name, type_param_str)
        };
        let tile_ir_ty = TileIrType::parse(&type_str);
        Ok(TileRustType {
            cuda_tile_ty_str: Some(type_str),
            tile_ir_ty,
            cuda_tile_name: Some(cuda_tile_name),
            params,
            rust_ty,
            type_instance,
            kind: Kind::StructuredType,
        })
    }

    /// Build a `TileRustType` for `Tile<element, shape>` directly.
    /// Returns `None` if `element_name` is not a concrete scalar type
    /// (e.g. a generic like `"E"`) — callers should fall back to `compile_type`.
    pub(crate) fn from_tile(element_name: &str, shape: &[i32]) -> Option<TileRustType> {
        let ir_shape: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let tile_ir_ty = super::_type::make_tile_type(element_name, &ir_shape)?;
        let shape_str = if shape.is_empty() {
            element_name.to_string()
        } else {
            let dims = shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!("{dims}x{element_name}")
        };
        let type_str = format!("!cuda_tile.tile<{shape_str}>");
        let rust_ty_str = if shape.is_empty() {
            format!("Tile<{element_name}, {{[]}}>")
        } else {
            format!("Tile<{element_name}, {{ {shape:?} }}>")
        };
        let rust_ty = syn::parse_str::<syn::Type>(&rust_ty_str).ok()?;
        Some(TileRustType {
            kind: Kind::StructuredType,
            cuda_tile_name: Some("!cuda_tile.tile".into()),
            cuda_tile_ty_str: Some(type_str),
            tile_ir_ty: Some(tile_ir_ty),
            params: vec![],
            rust_ty: rust_ty.clone(),
            type_instance: synthetic_tile_instance(rust_ty, element_name, shape),
        })
    }

    /// Build a `TileRustType` for a scalar tile `Tile<element, {[]}>`.
    pub(crate) fn from_scalar_tile(element_name: &str) -> Option<TileRustType> {
        Self::from_tile(element_name, &[])
    }

    /// Build a `TileRustType` for a scalar pointer tile
    /// `PointerTile<*mut element, {[]}>` / `PointerTile<*const element, {[]}>`.
    /// Constness is a Rust-facing property only: the Tile IR pointer type
    /// carries no mutability, so both build the same IR type.
    pub(crate) fn from_scalar_ptr(element_name: &str, is_mutable: bool) -> Option<TileRustType> {
        let scalar = super::_type::scalar_from_name(element_name)?;
        let tile_ir_ty = TileIrType::Tile(cutile_ir::ir::TileType {
            shape: vec![],
            element_type: cutile_ir::ir::TileElementType::Pointer(Box::new(
                cutile_ir::ir::PointerType { pointee: scalar },
            )),
        });
        let type_str = format!("!cuda_tile.tile<!cuda_tile.ptr<{element_name}>>");
        let prefix = crate::types::ptr_prefix(is_mutable);
        let rust_ty =
            syn::parse_str::<syn::Type>(&format!("PointerTile<{prefix} {element_name}, {{[]}}>"))
                .ok()?;
        Some(TileRustType {
            kind: Kind::PrimitiveType,
            cuda_tile_name: Some("!cuda_tile.tile".into()),
            cuda_tile_ty_str: Some(type_str),
            tile_ir_ty: Some(tile_ir_ty),
            params: vec![],
            rust_ty: rust_ty.clone(),
            type_instance: synthetic_ptr_instance(rust_ty, element_name, is_mutable),
        })
    }

    /// Re-labels the Rust-facing constness of this type's raw-pointer
    /// component — `*mut E` <-> `*const E`, scalar or inside a
    /// `PointerTile`. The Tile IR type is deliberately untouched: IR
    /// pointers carry no constness, which is exactly what makes
    /// `cast_const`/`cast_mut` free.
    pub(crate) fn set_pointer_constness(&mut self, is_mutable: bool) {
        let (from, to) = if is_mutable {
            ("* const", "* mut")
        } else {
            ("* mut", "* const")
        };
        let relabel = |ty: &syn::Type| -> Option<syn::Type> {
            let s = quote::quote!(#ty).to_string();
            if s.contains(from) {
                syn::parse_str::<syn::Type>(&s.replace(from, to)).ok()
            } else {
                None
            }
        };
        if let Some(new_ty) = relabel(&self.rust_ty) {
            self.rust_ty = new_ty;
        }
        let relabel_ptr_instance = |inst: &mut crate::generics::TypeInstancePtrType| {
            inst.is_mutable = is_mutable;
            if let Some(new_ty) = relabel(&inst.instance_ty) {
                inst.instance_ty = new_ty;
            }
            if let Some(new_ty) = relabel(&inst.generic_ty) {
                inst.generic_ty = new_ty;
            }
        };
        match &mut self.type_instance {
            TypeInstance::PtrType(inst) => relabel_ptr_instance(inst),
            TypeInstance::StructuredType(structured) => {
                if let Some(new_ty) = relabel(&structured.instance_ty) {
                    structured.instance_ty = new_ty;
                }
                if let Some(new_ty) = relabel(&structured.generic_ty) {
                    structured.generic_ty = new_ty;
                }
                if let Some(crate::generics::TypInstancePrimitiveType::PtrType(inst)) =
                    &mut structured.primitive_type
                {
                    relabel_ptr_instance(inst);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn new_structure(
        cuda_tile_name: String,
        type_instance: TypeInstance,
    ) -> TileRustType {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty_str: None,
            tile_ir_ty: None,
            cuda_tile_name: Some(cuda_tile_name),
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::Struct,
        }
    }

    pub(crate) fn new_compound(type_instance: TypeInstance) -> TileRustType {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty_str: None,
            tile_ir_ty: None,
            cuda_tile_name: None,
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::Compound,
        }
    }

    pub(crate) fn new_string(type_instance: TypeInstance) -> TileRustType {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty_str: None,
            tile_ir_ty: None,
            cuda_tile_name: None,
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::String,
        }
    }

    pub(crate) fn new_enum(type_instance: TypeInstance) -> TileRustType {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty_str: None,
            tile_ir_ty: None,
            cuda_tile_name: None,
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::Enum,
        }
    }
}
