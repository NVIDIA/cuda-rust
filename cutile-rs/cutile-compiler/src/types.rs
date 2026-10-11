/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type parameter definitions and element-type resolution helpers.
//! Maps Rust generic type parameters to their CUDA Tile IR representations.

// Helper module for type parameters.
use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use crate::generics::{
    GenericVars, TypInstancePrimitiveType, TypeInstance, TypeInstanceStructuredType,
};
use crate::syn_utils::{
    get_ident_generic_args, get_meta_list, get_type_ident, maybe_generic_args,
    strip_generic_args_lifetimes, SingleMetaList,
};
use quote::ToTokens;
use std::collections::HashMap;
use syn::{
    Expr, ExprLit, FnArg, GenericArgument, ItemImpl, Lit, Signature, Type, TypeReference, UnOp,
};

/// A type parameter slot in a CUDA Tile type definition (e.g. element type, shape, strides).
#[derive(Debug, Clone)]
pub enum TypeParam {
    /// Scalar element type parameter.
    Primitive(TypeParamPrimitive),
    /// Shaped element type parameter (shape × element).
    ShapedPrimitive(TypeParamShapedPrimitive),
    /// Stride layout parameter.
    Strides(TypeParamStrides),
    /// Dimension mapping parameter.
    DimMap(TypeParamDimMap),
    /// Tile shape parameter.
    Tile(TypeParamTile),
    // TODO (hme): Can we rename this to Type?
    /// Nested tensor view type parameter.
    TensorView(TypeParamTensorView),
    /// Padding value parameter (e.g. `padding_value = neg_inf`).
    Padding(TypeParamPadding),
}

impl TypeParam {
    /// Returns the parameter name, if set.
    pub fn name(&self) -> Option<String> {
        match self {
            TypeParam::Primitive(tp) => tp.name.clone(),
            TypeParam::ShapedPrimitive(tp) => tp.name.clone(),
            TypeParam::Strides(tp) => tp.name.clone(),
            TypeParam::Tile(tp) => tp.name.clone(),
            TypeParam::TensorView(tp) => tp.name.clone(),
            TypeParam::DimMap(tp) => tp.name.clone(),
            TypeParam::Padding(tp) => tp.name.clone(),
        }
    }
    /// Resolves this type parameter to a concrete Tile IR type string.
    pub fn instantiate(
        &mut self,
        generic_args: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        match self {
            TypeParam::Primitive(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::ShapedPrimitive(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::Strides(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::Tile(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::TensorView(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::DimMap(tp) => tp.instantiate(generic_args, primitives),
            TypeParam::Padding(tp) => tp.instantiate(),
        }
    }
    /// Constructs the appropriate `TypeParam` variant from a parameter name and Rust type.
    pub fn derive_param_from_type(
        name: String,
        rust_ty: syn::Type,
        cuda_tile_ty: Option<String>,
        type_instance: Option<TypeInstance>,
    ) -> TypeParam {
        match name.as_str() {
            "E" => TypeParam::Primitive(TypeParamPrimitive {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
            }),
            "!cuda_tile.ptr<E>" => TypeParam::Primitive(TypeParamPrimitive {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
            }),
            "{D}xE" => TypeParam::ShapedPrimitive(TypeParamShapedPrimitive {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
                type_instance,
            }),
            "{D}xP" => TypeParam::ShapedPrimitive(TypeParamShapedPrimitive {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
                type_instance,
            }),
            "tile" => TypeParam::Tile(TypeParamTile {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
                type_instance,
            }),
            "strides" => TypeParam::Strides(TypeParamStrides {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
            }),
            "dim_map" => TypeParam::DimMap(TypeParamDimMap {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
                type_instance,
            }),
            "tensor_view" => TypeParam::TensorView(TypeParamTensorView {
                name: Some(name),
                rust_ty,
                cuda_tile_ty,
                type_instance,
            }),
            "padding" | "padding_value" => TypeParam::Padding(TypeParamPadding {
                name: Some(name),
                padding_value: None,
            }),
            _ => panic!("Unexpected TypeParam name {name}"), // Internal error: unknown TypeParam variant
        }
    }
}

/// A type parameter combining shape dimensions with an element type (e.g. `{D}xE`).
#[derive(Debug, Clone)]
pub struct TypeParamShapedPrimitive {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
    pub type_instance: Option<TypeInstance>,
}

impl TypeParamShapedPrimitive {
    fn instantiate(
        &mut self,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        let TypeInstance::StructuredType(structured_type_instance) =
            generic_vars.instantiate_type(&self.rust_ty, primitives)?
        else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "TypeParamShapedPrimitive Unexpected type {:#?}",
                self.rust_ty
            ));
        };
        let mlir_var_type =
            MLIRVariadicArg::from_structured_type_instance(&structured_type_instance, primitives);
        // let mlir_var_type = get_variadic_type_arg(&self.rust_ty, None, generic_vars, primitives);
        // println!("TypeParamShapedPrimitive: {:#?}", mlir_var_type);
        let result = mlir_var_type.mlir_str("x", true);
        self.cuda_tile_ty = Some(result.clone());
        self.type_instance = Some(TypeInstance::StructuredType(structured_type_instance));
        Ok(result)
    }
}

/// A scalar element type parameter (e.g. `E` → `f32`).
#[derive(Debug, Clone)]
pub struct TypeParamPrimitive {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
}

impl TypeParamPrimitive {
    fn instantiate(
        &mut self,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        let type_instance = generic_vars.instantiate_type(&self.rust_ty, primitives)?;
        let rust_ty_instance = type_instance.get_instantiated_type();
        let element_type = get_cuda_tile_element_type_primitive(rust_ty_instance, primitives);
        match self.rust_ty {
            Type::Path(_) => {
                let result = element_type.clone();
                self.cuda_tile_ty = Some(result.clone());
                Ok(result)
            }
            Type::Ptr(_) => {
                let result = format!("!cuda_tile.ptr<{}>", element_type);
                self.cuda_tile_ty = Some(result.clone());
                Ok(result)
            }
            _ => SourceLocation::unknown().jit_error_result(&format!(
                "unsupported primitive type `{}`",
                self.rust_ty.to_token_stream()
            )),
        }
    }
}

/// A stride layout parameter (e.g. `strides=[?,?,?]`).
#[derive(Debug, Clone)]
pub struct TypeParamStrides {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
}

// TODO (hme): Move this closer to definition of the type?
impl TypeParamStrides {
    fn instantiate(
        &mut self,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        let TypeInstance::StructuredType(structured_type_instance) =
            generic_vars.instantiate_type(&self.rust_ty, primitives)?
        else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "TypeParamStrides Unexpected type {:#?}",
                self.rust_ty
            ));
        };
        let mlir_var_type =
            MLIRVariadicArg::from_structured_type_instance(&structured_type_instance, primitives);
        // let mlir_var_type = get_variadic_type_arg(&self.rust_ty, None, generic_vars, primitives);
        let result = format!("strides=[{}]", mlir_var_type.mlir_str(",", false));
        self.cuda_tile_ty = Some(result.clone());
        Ok(result)
    }
}

/// A dimension mapping parameter (e.g. `dim_map=[0,1,2]`).
#[derive(Debug, Clone)]
pub struct TypeParamDimMap {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
    pub type_instance: Option<TypeInstance>,
}

impl TypeParamDimMap {
    fn instantiate(
        &mut self,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        let TypeInstance::StructuredType(structured_type_instance) =
            generic_vars.instantiate_type(&self.rust_ty, primitives)?
        else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "TypeParamDimMap Unexpected type {:#?}",
                self.rust_ty
            ));
        };
        let mlir_var_type =
            MLIRVariadicArg::from_structured_type_instance(&structured_type_instance, primitives);
        // let mlir_var_type = get_variadic_type_arg(&self.rust_ty, None, generic_vars, primitives);
        let result = format!("dim_map=[{}]", mlir_var_type.mlir_str(",", false));
        self.cuda_tile_ty = Some(result.clone());
        self.type_instance = Some(TypeInstance::StructuredType(structured_type_instance));
        Ok(result)
    }
}

/// A tile shape parameter (e.g. `tile=(128x64)`).
#[derive(Debug, Clone)]
pub struct TypeParamTile {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
    pub type_instance: Option<TypeInstance>,
}

impl TypeParamTile {
    fn instantiate(
        &mut self,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        let TypeInstance::StructuredType(structured_type_instance) =
            generic_vars.instantiate_type(&self.rust_ty, primitives)?
        else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "TypeParamTile Unexpected type {:#?}",
                self.rust_ty
            ));
        };
        let mlir_var_type =
            MLIRVariadicArg::from_structured_type_instance(&structured_type_instance, primitives);
        // let mlir_var_type = get_variadic_type_arg(&self.rust_ty, None, generic_args, primitives);
        let result = format!("tile=({})", mlir_var_type.mlir_str("x", false));
        self.cuda_tile_ty = Some(result.clone());
        self.type_instance = Some(TypeInstance::StructuredType(structured_type_instance));
        Ok(result)
    }
}

/// A nested tensor view type parameter used by partition views.
#[derive(Debug, Clone)]
pub struct TypeParamTensorView {
    pub name: Option<String>,
    pub rust_ty: syn::Type,
    pub cuda_tile_ty: Option<String>,
    pub type_instance: Option<TypeInstance>,
}

impl TypeParamTensorView {
    fn instantiate(
        &mut self,
        _generic_args: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<String, JITError> {
        if self.cuda_tile_ty.is_some() {
            return Ok(self.cuda_tile_ty.clone().unwrap());
        }
        // This is the tensor_view type param provided to e.g. partition views.
        // It cannot be instantiated without information provided when the type was constructed.
        SourceLocation::unknown().jit_error_result(&format!(
            "Cannot instantiate TypeParamTensorView from type {:?}",
            self.rust_ty.to_token_stream().to_string()
        ))
    }
}

/// A padding value parameter for partition views (e.g. `padding_value = neg_inf`).
#[derive(Debug, Clone)]
pub struct TypeParamPadding {
    pub name: Option<String>,
    /// The padding value string (e.g. "neg_inf", "zero"). Set after creation
    /// from the string literal argument.
    pub padding_value: Option<String>,
}

impl TypeParamPadding {
    fn instantiate(&self) -> Result<String, JITError> {
        match &self.padding_value {
            Some(val) => Ok(format!("padding_value = {val}")),
            None => SourceLocation::unknown().jit_error_result(
                "Padding TypeParam has no value set; the string literal was not resolved",
            ),
        }
    }
}

impl From<syn::Type> for TypeParamShapedPrimitive {
    fn from(rust_ty: syn::Type) -> Self {
        Self {
            rust_ty,
            name: None,
            cuda_tile_ty: None,
            type_instance: None,
        }
    }
}

impl From<syn::Type> for TypeParamStrides {
    fn from(rust_ty: syn::Type) -> Self {
        Self {
            rust_ty,
            name: None,
            cuda_tile_ty: None,
        }
    }
}

impl From<syn::Type> for TypeParamTile {
    fn from(rust_ty: syn::Type) -> Self {
        Self {
            rust_ty,
            name: None,
            cuda_tile_ty: None,
            type_instance: None,
        }
    }
}

/// Shape and element type information formatted for Tile IR type string construction.
#[derive(Debug)]
pub struct MLIRVariadicArg {
    pub primitive_type_str: Option<String>,
    pub shape: Vec<String>,
}

impl MLIRVariadicArg {
    /// Builds an `MLIRVariadicArg` from a structured type instance's shape and element info.
    pub fn from_structured_type_instance(
        inst: &TypeInstanceStructuredType,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> MLIRVariadicArg {
        let primitive_type_str = match &inst.primitive_type {
            Some(TypInstancePrimitiveType::ElementType(primitive_type)) => {
                let rust_element_instance_ty = &primitive_type.rust_element_instance_ty;
                get_cuda_tile_element_type_from_rust_primitive_str(
                    rust_element_instance_ty,
                    primitives,
                )
            }
            Some(TypInstancePrimitiveType::PtrType(primitive_type)) => {
                let rust_element_instance_ty = &primitive_type.rust_element_instance_ty;
                if let Some(ptr_attrs) =
                    get_pointer_primitives_attrs(primitive_type.is_mutable, primitives)
                {
                    if let Some(cuda_tile_element_type) =
                        get_cuda_tile_element_type_from_rust_primitive_str(
                            rust_element_instance_ty,
                            primitives,
                        )
                    {
                        // This is okay because the Pointer trait impl is for all E: ElementType.
                        let type_name = ptr_attrs.parse_string("pointer_type").unwrap();
                        Some(format!("{}<{}>", type_name, cuda_tile_element_type))
                    } else {
                        panic!("Failed to get cuda tile type for {inst:#?}")
                    }
                } else {
                    panic!("Failed to get cuda tile type for {inst:#?}")
                }
            }
            None => None,
        };
        let shape = inst
            .shape
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        MLIRVariadicArg {
            primitive_type_str,
            shape,
        }
    }
    /// Formats as a Tile IR type fragment (e.g. `128x64xf32`) with the given delimiter.
    pub fn mlir_str(&self, delim: &str, include_element_type: bool) -> String {
        let mlir_shape = self
            .shape
            .iter()
            .map(|s| {
                if s == "- 1" || s == "-1" {
                    "?".to_string()
                } else {
                    s.clone()
                }
            })
            .collect::<Vec<_>>();
        let x = match (include_element_type, &self.primitive_type_str) {
            (true, Some(elem_type)) => [mlir_shape, vec![elem_type.clone()]].concat(),
            _ => mlir_shape,
        };
        x.join(delim)
    }
}

/// Looks up the `cuda_tile::ty` attribute for a primitive trait impl.
pub fn get_primitives_attrs(
    trait_name: &str,
    rust_type_name: &str,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<SingleMetaList> {
    match primitives.get(&(trait_name.to_string(), rust_type_name.to_string())) {
        Some(item_impl) => get_meta_list("cuda_tile :: ty", &item_impl.attrs),
        None => None,
    }
}

/// The `Pointer` impl registration key for a given constness. Every
/// consumer of pointer attrs goes through this so the `*mut`/`*const`
/// pair never leaks into ad-hoc string keys.
pub fn pointer_impl_key(is_mutable: bool) -> &'static str {
    if is_mutable {
        "* mut E"
    } else {
        "* const E"
    }
}

/// The Rust-source pointer prefix for a given constness. Tile IR has a
/// single pointer type, so constness lives only in the Rust-facing type
/// strings built from this.
pub fn ptr_prefix(is_mutable: bool) -> &'static str {
    if is_mutable {
        "* mut"
    } else {
        "* const"
    }
}

/// Looks up the `Pointer` impl attrs matching the pointer's constness.
pub fn get_pointer_primitives_attrs(
    is_mutable: bool,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<SingleMetaList> {
    get_primitives_attrs("Pointer", pointer_impl_key(is_mutable), primitives)
}

/// Returns `true` if the Rust type string is a registered element type.
pub fn is_element_type(
    rust_primitive: &str,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> bool {
    get_primitives_attrs("ElementType", rust_primitive, primitives).is_some()
}

/// Returns `true` if the Rust type string is a pointer to a registered element type.
pub fn is_element_type_ptr(
    rust_ptr: &str,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> bool {
    // Check if this is an element type pointer.
    let rust_primitive = if rust_ptr.starts_with("* mut ") {
        rust_ptr.split("* mut ").collect::<Vec<_>>()[1]
    } else if rust_ptr.starts_with("* const ") {
        rust_ptr.split("* const ").collect::<Vec<_>>()[1]
    } else {
        ""
    };
    // println!("is_element_type_ptr? {rust_primitive}");
    get_primitives_attrs("ElementType", rust_primitive, primitives).is_some()
}

/// Parses a pointer type string, returning `(is_mutable, pointee_type)`.
pub fn get_ptr_type(rust_ptr: &str) -> Option<(bool, String)> {
    // This also serves to check whether this is actually a pointer.
    let res = if rust_ptr.starts_with("* mut ") {
        (
            true,
            rust_ptr.split("* mut ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else if rust_ptr.starts_with("* const ") {
        (
            false,
            rust_ptr.split("* const ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else {
        return None;
    };
    Some(res)
}

/// Like [`get_ptr_type`] but also resolves generic type variables.
pub fn get_ptr_type_instance(
    rust_ptr: &str,
    generic_vars: &GenericVars,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<(bool, String)> {
    // This also serves to check whether this is actually a pointer.
    let (prefix, maybe_element_type) = if rust_ptr.starts_with("* mut ") {
        (
            true,
            rust_ptr.split("* mut ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else if rust_ptr.starts_with("* const ") {
        (
            false,
            rust_ptr.split("* const ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else {
        return None;
    };
    if is_element_type(&maybe_element_type, primitives) {
        Some((prefix, maybe_element_type))
    } else {
        generic_vars
            .inst_types
            .get(&maybe_element_type)
            .map(|element_type| (prefix, element_type.to_string()))
    }
}

/// Maps a Rust primitive name (e.g. `"f32"`) to its CUDA Tile element type string.
pub fn get_cuda_tile_element_type_from_rust_primitive_str(
    rust_primitive: &str,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<String> {
    match get_primitives_attrs("ElementType", rust_primitive, primitives) {
        Some(attrs) => attrs.parse_string("name"),
        None => None,
    }
}

/// Returns the Rust identifier string for a primitive type.
pub fn get_rust_element_type_primitive(ty: &syn::Type) -> String {
    let type_ident = get_type_ident(ty);
    assert!(
        type_ident.is_some(),
        "get_element_type_primitive failed for {ty:#?}"
    );
    type_ident.unwrap().to_string()
}

/// Returns the CUDA Tile element type string for a Rust primitive type.
pub fn get_cuda_tile_element_type_primitive(
    ty: &syn::Type,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> String {
    let rust_elem_ty_str = get_rust_element_type_primitive(ty);
    let element_type_attrs =
        get_primitives_attrs("ElementType", rust_elem_ty_str.as_str(), primitives);
    assert!(
        element_type_attrs.is_some(),
        "get_cuda_tile_element_type_primitive failed for {ty:#?}"
    );
    let element_type_attrs = element_type_attrs.unwrap();
    let element_type_param = element_type_attrs.parse_string("name");
    assert!(
        element_type_param.is_some(),
        "get_cuda_tile_element_type_primitive failed for {ty:#?}"
    );
    element_type_param.unwrap()
}

/// Extracts the Rust element type name from a structured type's generic arguments.
pub fn get_element_type_structured(
    ty: &syn::Type,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<String> {
    let (_type_ident, type_generic_args) = get_ident_generic_args(ty);
    let mut element_type: Option<String> = None;
    for generic_arg in &type_generic_args.args {
        if let GenericArgument::Type(type_param) = generic_arg {
            match type_param {
                syn::Type::Path(type_path) => {
                    let ident_str = type_path.path.segments.last().unwrap().ident.to_string();
                    if get_primitives_attrs("ElementType", &ident_str, primitives).is_some() {
                        element_type = Some(ident_str);
                        break;
                    }
                }
                syn::Type::Ptr(type_ptr) => {
                    let ident_str = match get_type_ident(type_param) {
                        Some(ident) => ident.to_string(),
                        None => panic!("Unable to extract ident from pointer {type_ptr:#?}"),
                    };
                    if get_primitives_attrs("ElementType", &ident_str, primitives).is_some() {
                        element_type = Some(ident_str);
                        break;
                    }
                }
                syn::Type::Reference(type_ref) => {
                    element_type = get_element_type_structured(&type_ref.elem, primitives)
                }
                _ => {}
            }
        }
    }
    element_type
}

/// Returns the CUDA Tile element type string for a structured type.
pub fn get_cuda_tile_element_type_structured(
    ty: &syn::Type,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Option<String> {
    let rust_element_type = get_element_type_structured(ty, primitives)?;
    get_cuda_tile_element_type_from_rust_primitive_str(&rust_element_type, primitives)
}

/// Infers the `syn::Type` of a literal expression from its suffix or kind.
pub fn get_lit_type(lit_expr: &ExprLit) -> Option<syn::Type> {
    match &lit_expr.lit {
        Lit::Int(lit) => {
            if !lit.suffix().is_empty() {
                Some(syn::parse2(lit.suffix().parse().unwrap()).unwrap())
            } else {
                None
            }
        }
        Lit::Float(lit) => {
            if !lit.suffix().is_empty() {
                Some(syn::parse2(lit.suffix().parse().unwrap()).unwrap())
            } else {
                None
            }
        }
        Lit::Bool(_bool_lit) => Some(syn::parse2("bool".parse().unwrap()).unwrap()),
        Lit::Str(_str_lit) => Some(syn::parse2("str".parse().unwrap()).unwrap()),
        _ => None,
    }
}

/// Parses a possibly-negated integer literal expression as an `i32`.
pub fn parse_signed_literal_as_i32(expr: &Expr) -> Result<i32, JITError> {
    let expected_literal = || {
        JITError::generic_err(&format!(
            "expected an integer literal or a negated integer literal, got `{}`",
            expr.to_token_stream()
        ))
    };
    let (lit, negative) = match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit), ..
        }) => (lit, false),
        Expr::Unary(unary) if matches!(unary.op, UnOp::Neg(_)) => {
            let Expr::Lit(ExprLit {
                lit: Lit::Int(lit), ..
            }) = unary.expr.as_ref()
            else {
                return Err(expected_literal());
            };
            (lit, true)
        }
        _ => return Err(expected_literal()),
    };
    let out_of_range = || {
        JITError::generic_err(&format!(
            "integer literal `{}` is outside the i32 range",
            expr.to_token_stream()
        ))
    };
    // Parse the magnitude in a wider type so that -2147483648 remains valid.
    let magnitude = lit.base10_parse::<i64>().map_err(|_| out_of_range())?;
    let value = if negative {
        magnitude.checked_neg().ok_or_else(out_of_range)?
    } else {
        magnitude
    };
    i32::try_from(value).map_err(|_| out_of_range())
}

// pub fn get_expr_mutability(expr: &Expr) -> bool {
//     match expr {
//         Expr::Reference(ref_expr) => {}
//     }
// }

/// Returns a per-parameter mutability flag for a function signature.
pub fn get_sig_param_mutability(sig: &Signature) -> Vec<bool> {
    let mut result = vec![];
    for arg in &sig.inputs {
        let is_mutable = match arg {
            FnArg::Receiver(receiver) => receiver.mutability.is_some(),
            FnArg::Typed(fn_param) => {
                let pat_mutability = get_pat_mutability(&fn_param.pat);
                let ty_mutability = get_type_mutability(&fn_param.ty);
                pat_mutability || ty_mutability
            }
        };
        result.push(is_mutable);
    }
    result
}

/// Returns `true` if the pattern is marked mutable.
pub fn get_pat_mutability(pat: &syn::Pat) -> bool {
    match pat {
        syn::Pat::Reference(ref_pat) => ref_pat.mutability.is_some(),
        syn::Pat::Ident(identifier) => identifier.mutability.is_some(),
        syn::Pat::Tuple(_) => {
            // Tuple patterns don't have mutability at the pattern level
            false
        }
        _ => panic!("Unexpected argument pattern"),
    }
}

/// Returns `true` if the type is a mutable reference.
pub fn get_type_mutability(ty: &Type) -> bool {
    match ty {
        Type::Reference(TypeReference {
            mutability: Some(_),
            ..
        }) => true,
        Type::Reference(TypeReference {
            mutability: None, ..
        }) => false,
        _ => false,
    }
}

/// Tries to extract a const generic array from a type's generic arguments.
/// `None` means no array was found; malformed arrays return an error.
pub fn try_extract_cga(
    ty: &Type,
    generic_vars: &GenericVars,
) -> Result<Option<Vec<i32>>, JITError> {
    let Some(mut args) = maybe_generic_args(ty) else {
        return Ok(None);
    };
    strip_generic_args_lifetimes(&mut args);
    let mut result = None;
    for arg in &args.args {
        if let Some(array) = crate::generics::get_cga_from_generic_argument(arg, generic_vars)? {
            result = Some(array);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod dimension_parse_tests {
    use super::*;

    fn expr(source: &str) -> Expr {
        syn::parse_str(source).unwrap()
    }

    fn vars() -> GenericVars {
        let mut vars = GenericVars::empty_unchecked();
        vars.inst_i32.insert("N".into(), 3);
        vars.inst_array.insert("D".into(), vec![16, 32]);
        vars.len2array.insert("R".into(), "D".into());
        vars
    }

    #[test]
    fn signed_literals_include_i32_min() {
        for (source, expected) in [
            ("0", 0),
            ("128", 128),
            ("-1", -1),
            ("0x20", 32),
            ("16_i32", 16),
            ("2147483647", i32::MAX),
            ("-2147483648", i32::MIN),
        ] {
            assert_eq!(
                parse_signed_literal_as_i32(&expr(source)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn overflowing_literals_return_errors() {
        for source in ["2147483648", "-2147483649", "18446744073709551616"] {
            let error = parse_signed_literal_as_i32(&expr(source)).unwrap_err();
            assert!(
                error.to_string().contains("outside the i32 range"),
                "{source}: {error}"
            );
        }
    }

    #[test]
    fn unsupported_literals_return_errors() {
        for source in ["true", "1.5", "!1", "1 + 2", "N", "-N", "-(1)"] {
            let error = parse_signed_literal_as_i32(&expr(source)).unwrap_err();
            assert!(
                error.to_string().contains("expected an integer literal"),
                "{source}: {error}"
            );
        }
    }

    #[test]
    fn const_generic_arrays_resolve_literals_variables_and_indices() {
        for (source, expected) in [
            ("Shape<{[128, -1, N, D[1]]}>", vec![128, -1, 3, 32]),
            ("Shape<{[-1; N]}>", vec![-1; 3]),
            ("Shape<{[-1; R]}>", vec![-1; 2]),
            ("Shape<{[D[0]; 2]}>", vec![16; 2]),
            ("Shape<{[1; 0]}>", vec![]),
            ("Shape<D>", vec![16, 32]),
        ] {
            let ty: Type = syn::parse_str(source).unwrap();
            assert_eq!(
                try_extract_cga(&ty, &vars()).unwrap(),
                Some(expected),
                "{source}"
            );
        }
    }

    #[test]
    fn malformed_const_generic_arrays_return_errors() {
        for (source, expected) in [
            ("Shape<{[2147483648]}>", "outside the i32 range"),
            ("Shape<{[true]}>", "expected an integer literal"),
            ("Shape<{[1 + 2]}>", "unsupported dimension expression"),
            ("Shape<{[M]}>", "undefined generic parameter"),
            ("Shape<{[M[0]]}>", "undefined const generic array"),
            ("Shape<{[D[-1]]}>", "out of bounds"),
            ("Shape<{[D[2]]}>", "out of bounds"),
            ("Shape<{[D[0 + 0]]}>", "expected an integer literal"),
            ("Shape<{[D[2147483648]]}>", "outside the i32 range"),
            (
                "Shape<{[get_dims()[0]]}>",
                "expected a const generic array name",
            ),
            ("Shape<{[1; -1]}>", "must be non-negative"),
            ("Shape<{[1; M]}>", "undefined generic parameter"),
            ("Shape<{[1; 2147483648]}>", "outside the i32 range"),
            ("Shape<{}>", "single array expression"),
            ("Shape<{let x = 1; [x]}>", "single array expression"),
            ("Shape<{1}>", "expected a const generic array expression"),
        ] {
            let ty: Type = syn::parse_str(source).unwrap();
            let error = try_extract_cga(&ty, &vars()).unwrap_err();
            assert!(error.to_string().contains(expected), "{source}: {error}");
        }
    }

    #[test]
    fn missing_array_is_distinct_from_malformed_array() {
        for source in ["f32", "Tensor<f32>", "Marker<3>"] {
            let ty: Type = syn::parse_str(source).unwrap();
            assert_eq!(try_extract_cga(&ty, &vars()).unwrap(), None, "{source}");
        }
    }
}
