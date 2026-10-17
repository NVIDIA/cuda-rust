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
use super::type_identity::{Referent, TypeIdentity};
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

/// Mutability of each `&` in `&mut &T`, outermost first.
fn reference_mutabilities(mut ty: &syn::Type) -> Vec<bool> {
    let mut mutabilities = vec![];
    while let syn::Type::Reference(reference) = ty {
        mutabilities.push(reference.mutability.is_some());
        ty = &reference.elem;
    }
    mutabilities
}

/// `Partition<..>` -> `{prefix}Partition<..>`.
fn prefix_type_name(ty: &mut syn::Type, prefix: &str) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    let Some(segment) = path.path.segments.last_mut() else {
        return false;
    };
    segment.ident = syn::Ident::new(&format!("{prefix}{}", segment.ident), segment.ident.span());
    true
}

/// A compiled type binding: maps a Rust `syn::Type` to its CUDA Tile type metadata.
///
/// Lifetime-free — all fields are owned. The tile-ir type is eagerly parsed at
/// construction and stored in `tile_ir_ty`. Fields change only through the
/// methods below, which keep the spelling, the instance and the tile-ir type
/// describing one type.
#[derive(Debug, Clone)]
pub struct TileRustType {
    kind: Kind,
    cuda_tile_name: Option<String>,
    /// The dialect type string (e.g. `"!cuda_tile.tile<128x64xf32>"`).
    /// Retained for debug output; the parsed form is in `tile_ir_ty`.
    cuda_tile_ty_str: Option<String>,
    /// Eagerly parsed tile-ir type. `None` when the type string is absent
    /// or could not be parsed (deferred types).
    tile_ir_ty: Option<TileIrType>,
    params: Vec<TypeParam>,
    /// The type as spelled in the scope that owns the value.
    rust_ty: syn::Type,
    /// `rust_ty`'s referent, instantiated in that scope.
    type_instance: TypeInstance,
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

    pub(crate) fn kind(&self) -> Kind {
        self.kind.clone()
    }
    pub(crate) fn cuda_tile_name(&self) -> Option<&str> {
        self.cuda_tile_name.as_deref()
    }
    pub(crate) fn tile_ir_ty(&self) -> Option<&TileIrType> {
        self.tile_ir_ty.as_ref()
    }
    pub(crate) fn params(&self) -> &[TypeParam] {
        &self.params
    }
    pub(crate) fn rust_ty(&self) -> &syn::Type {
        &self.rust_ty
    }
    pub(crate) fn type_instance(&self) -> &TypeInstance {
        &self.type_instance
    }

    pub(crate) fn identity(&self) -> TypeIdentity {
        TypeIdentity {
            references: reference_mutabilities(&self.rust_ty),
            referent: Referent::of(&self.type_instance),
        }
    }

    /// `rust_ty` spells the type in its producer's scope (`{ [N] }` or
    /// `{ [4] }`); types are compared by identity.
    pub(crate) fn same_type(&self, other: &TileRustType) -> bool {
        self.identity() == other.identity()
    }

    /// Rebinds the value to `spelling`, its type as written in the scope it
    /// moves into, with `instance` the instantiation there. Unchecked: an
    /// inlined call's arguments and result already have these types in
    /// rustc-checked source. The tile-ir type stays the value's.
    pub(crate) fn adopt(&mut self, spelling: syn::Type, instance: TypeInstance) {
        self.rust_ty = spelling;
        self.type_instance = instance;
    }

    /// Records an op's pointer attribute on `lowered`, this type's tile-ir
    /// type. The Rust type is unaffected.
    pub(crate) fn attach_pointer_attribute(
        &mut self,
        lowered: TileIrType,
        attribute: cutile_ir::ir::PointerAttribute,
    ) {
        self.tile_ir_ty = Some(lowered.with_pointer_attribute(attribute));
        self.cuda_tile_ty_str = None;
    }

    /// `T` -> `&T`; the instance keeps describing the referent.
    pub(crate) fn wrap_reference(
        &mut self,
        lifetime: Option<syn::Lifetime>,
        mutability: Option<syn::Token![mut]>,
    ) {
        self.rust_ty = syn::Type::Reference(syn::TypeReference {
            and_token: Default::default(),
            lifetime,
            mutability,
            elem: Box::new(self.rust_ty.clone()),
        });
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
        let relabel_ptr_instance = |inst: &mut crate::generics::TypeInstancePtrType| {
            inst.is_mutable = is_mutable;
            if let Some(new_ty) = relabel(&inst.instance_ty) {
                inst.instance_ty = new_ty;
            }
            if let Some(new_ty) = relabel(&inst.generic_ty) {
                inst.generic_ty = new_ty;
            }
        };
        if let Some(new_ty) = relabel(&self.rust_ty) {
            self.rust_ty = new_ty;
        }
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

    /// Renames the type in its spelling and its instance alike; the tile-ir
    /// type is unchanged. Returns `false` when the type has no name.
    pub(crate) fn prefix_type_name(&mut self, prefix: &str) -> bool {
        if !prefix_type_name(&mut self.rust_ty, prefix) {
            return false;
        }
        if let TypeInstance::StructuredType(structured) = &mut self.type_instance {
            prefix_type_name(&mut structured.generic_ty, prefix);
            prefix_type_name(&mut structured.instance_ty, prefix);
        }
        true
    }

    /// A view whose tile-ir type its op computed (e.g. a partition view).
    pub(crate) fn new_view(
        cuda_tile_name: String,
        tile_ir_ty: TileIrType,
        type_instance: TypeInstance,
    ) -> TileRustType {
        Self::bare(
            Kind::StructuredType,
            Some(cuda_tile_name),
            Some(tile_ir_ty),
            type_instance,
        )
    }

    pub(crate) fn new_structure(
        cuda_tile_name: String,
        type_instance: TypeInstance,
    ) -> TileRustType {
        Self::bare(Kind::Struct, Some(cuda_tile_name), None, type_instance)
    }

    pub(crate) fn new_compound(type_instance: TypeInstance) -> TileRustType {
        Self::bare(Kind::Compound, None, None, type_instance)
    }

    pub(crate) fn new_string(type_instance: TypeInstance) -> TileRustType {
        Self::bare(Kind::String, None, None, type_instance)
    }

    pub(crate) fn new_enum(type_instance: TypeInstance) -> TileRustType {
        Self::bare(Kind::Enum, None, None, type_instance)
    }

    /// A type without type parameters, spelled by `type_instance`'s source.
    fn bare(
        kind: Kind,
        cuda_tile_name: Option<String>,
        tile_ir_ty: Option<TileIrType>,
        type_instance: TypeInstance,
    ) -> TileRustType {
        TileRustType {
            kind,
            cuda_tile_name,
            cuda_tile_ty_str: None,
            tile_ir_ty,
            params: vec![],
            rust_ty: type_instance.get_source_type().clone(),
            type_instance,
        }
    }
}

#[cfg(test)]
mod same_type_tests {
    use super::*;

    fn primitives() -> HashMap<(String, String), ItemImpl> {
        ["f32", "f16"]
            .into_iter()
            .map(|name| {
                let ident = syn::Ident::new(name, proc_macro2::Span::call_site());
                let item: ItemImpl = syn::parse_quote! {
                    #[cuda_tile::ty(name = #name)]
                    impl ElementType for #ident {}
                };
                (("ElementType".to_string(), name.to_string()), item)
            })
            .collect()
    }

    fn generic_vars() -> GenericVars {
        let mut vars = GenericVars::empty_unchecked();
        vars.inst_i32.insert("N".into(), 4);
        vars.inst_i32.insert("BM".into(), 4);
        vars.inst_types.insert("E".into(), "f32".into());
        vars
    }

    fn ty(spelling: &str) -> TileRustType {
        let rust_ty = syn::parse_str::<syn::Type>(spelling).unwrap();
        let (vars, primitives) = (generic_vars(), primitives());
        let instance = vars.instantiate_type(&rust_ty, &primitives).unwrap();
        TileRustType::new_structured_type(
            "!cuda_tile.tile".into(),
            &vars,
            &primitives,
            vec![],
            instance,
        )
        .unwrap()
    }

    fn reference_to(spelling: &str) -> TileRustType {
        let mut reference = ty(spelling);
        reference.wrap_reference(None, None);
        reference
    }

    #[test]
    fn symbolic_shape_is_its_instantiation() {
        assert!(ty("Tile<f32, { [N] }>").same_type(&ty("Tile<f32, { [4] }>")));
        assert!(ty("Tile<f32, { [N] }>").same_type(&TileRustType::from_tile("f32", &[4]).unwrap()));
    }

    #[test]
    fn different_symbols_with_equal_values_are_the_same_type() {
        assert!(ty("Tile<f32, { [N] }>").same_type(&ty("Tile<f32, { [BM] }>")));
    }

    #[test]
    fn generic_element_is_its_instantiation() {
        assert!(ty("Tile<E, { [N] }>").same_type(&ty("Tile<f32, { [4] }>")));
        assert!(
            ty("PointerTile<* mut E, { [N] }>").same_type(&ty("PointerTile<* mut f32, { [4] }>"))
        );
    }

    #[test]
    fn different_shape_is_rejected() {
        assert!(!ty("Tile<f32, { [N] }>").same_type(&ty("Tile<f32, { [8] }>")));
        assert!(!ty("Tile<f32, { [4] }>").same_type(&ty("Tile<f32, { [4, 1] }>")));
    }

    #[test]
    fn different_element_type_is_rejected() {
        assert!(!ty("Tile<E, { [4] }>").same_type(&ty("Tile<f16, { [4] }>")));
    }

    #[test]
    fn different_pointer_mutability_is_rejected() {
        assert!(!ty("PointerTile<* mut E, { [N] }>")
            .same_type(&ty("PointerTile<* const f32, { [4] }>")));
        assert!(!TileRustType::from_scalar_ptr("f32", true)
            .unwrap()
            .same_type(&TileRustType::from_scalar_ptr("f32", false).unwrap()));
    }

    #[test]
    fn reference_is_not_its_referent() {
        assert!(!reference_to("Tile<f32, { [4] }>").same_type(&ty("Tile<f32, { [4] }>")));
        assert!(reference_to("Tile<f32, { [N] }>").same_type(&reference_to("Tile<f32, { [4] }>")));
    }
}
