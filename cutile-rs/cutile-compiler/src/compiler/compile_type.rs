/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type compilation for compiler2: translates Rust `syn::Type` AST nodes into
//! `TileRustType` representations.
//!
//! Types are eagerly parsed into `TileRustType.tile_ir_ty` at construction
//! and accessed via `types::convert_type()`.

use super::_function::CUDATileFunctionCompiler;
use super::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::{GenericVars, Instantiable, TypeInstance, TypeInstanceUserType};
use crate::syn_utils::*;
use crate::types::*;
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;

impl<'m> CUDATileFunctionCompiler<'m> {
    pub fn compile_type(
        &self,
        ty: &syn::Type,
        generic_vars: &GenericVars,
        type_params: &HashMap<String, TypeParam>,
    ) -> Result<Option<TileRustType>, JITError> {
        // Consts named in the type resolve in the scope of the module whose
        // source spelled it (the kernel's, or the inlined callee's).
        let normalized_ty = self
            .modules
            .normalize_type_aliases_in(ty, Some(&self.current_module()))?;
        let ty = &normalized_ty;
        let _ty_debug_str = ty.to_token_stream().to_string();
        let mut ty_attrs: Option<SingleMetaList> = None;
        let mut structure: Option<(String, &syn::ItemStruct)> = None;
        let mut _maybe_generic_element_type: Option<(String, &syn::ItemImpl)> = None;
        let type_instance: Option<TypeInstance>;

        match ty {
            // Array, Slice, and Tuple compile to the same compiler representation (compound values).
            syn::Type::Tuple(tuple) => {
                if tuple.elems.is_empty() {
                    return Ok(None);
                } else {
                    let unknown_type_instance = TypeInstanceUserType::instantiate(
                        ty,
                        generic_vars,
                        self.modules.primitives(),
                    )
                    .unwrap();
                    let type_instance = TypeInstance::UserType(unknown_type_instance);
                    return Ok(Some(TileRustType::new_compound(type_instance)));
                }
            }
            syn::Type::Array(_) => {
                let unknown_type_instance =
                    TypeInstanceUserType::instantiate(ty, generic_vars, self.modules.primitives())
                        .unwrap();
                let type_instance = TypeInstance::UserType(unknown_type_instance);
                return Ok(Some(TileRustType::new_compound(type_instance)));
            }
            syn::Type::Slice(_) => {
                let unknown_type_instance =
                    TypeInstanceUserType::instantiate(ty, generic_vars, self.modules.primitives())
                        .unwrap();
                let type_instance = TypeInstance::UserType(unknown_type_instance);
                return Ok(Some(TileRustType::new_compound(type_instance)));
            }
            syn::Type::Reference(ref_ty) => {
                let mut res = self.compile_type(&ref_ty.elem, generic_vars, type_params)?;
                match &mut res {
                    Some(cuda_tile_ty) => {
                        cuda_tile_ty.rust_ty = ty.clone();
                    }
                    None => return self.jit_error_result(&ty.span(), "Failed to compile type"),
                }
                return Ok(res);
            }
            syn::Type::Path(_) => match get_type_ident(ty) {
                Some(ident) => {
                    let type_name = ident.to_string();
                    if type_name == "Option" {
                        let option_type_instance = TypeInstanceUserType::instantiate(
                            ty,
                            generic_vars,
                            self.modules.primitives(),
                        )
                        .unwrap();
                        return Ok(Some(TileRustType::new_enum(TypeInstance::UserType(
                            option_type_instance,
                        ))));
                    }
                    if let Some(item_struct) = self.modules.structs().get(type_name.as_str()) {
                        ty_attrs = self.modules.get_cuda_tile_type_attrs(type_name.as_str());
                        if ty_attrs.is_none() {
                            let unknown_type_instance = TypeInstanceUserType::instantiate(
                                ty,
                                generic_vars,
                                self.modules.primitives(),
                            )
                            .unwrap();
                            let type_instance = TypeInstance::UserType(unknown_type_instance);
                            return Ok(Some(TileRustType::new_structure(
                                type_name.clone(),
                                type_instance,
                            )));
                        }
                        structure = Some((type_name.clone(), item_struct));
                        type_instance =
                            Some(generic_vars.instantiate_type(ty, self.modules.primitives())?);
                    } else {
                        let local_type_instance =
                            generic_vars.instantiate_type(ty, self.modules.primitives())?;
                        if let TypeInstance::StringType(_string_inst) = local_type_instance {
                            return Ok(Some(TileRustType::new_string(TypeInstance::StringType(
                                _string_inst,
                            ))));
                        }
                        let Some(element_type_instance_str) =
                            local_type_instance.get_rust_element_instance_ty()
                        else {
                            return self.jit_error_result(&ty.span(), "Failed to compile type");
                        };
                        if let Some(element_type_impl) = self.modules.primitives().get(&(
                            "ElementType".to_string(),
                            element_type_instance_str.to_string(),
                        )) {
                            ty_attrs = self
                                .modules
                                .get_primitives_attrs("ElementType", &element_type_instance_str);
                            _maybe_generic_element_type =
                                Some((type_name.clone(), element_type_impl));
                        }
                        if ty_attrs.is_none() {
                            return self.jit_error_result(&ty.span(), "Failed to compile type");
                        }
                        type_instance = Some(local_type_instance);
                    }
                }
                None => return self.jit_error_result(&ty.span(), "Failed to compile type"),
            },
            syn::Type::Ptr(_) => {
                let type_name = get_type_ident(ty);
                if type_name.is_none() {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                }
                let type_name = type_name.unwrap().to_string();
                let local_type_instance =
                    generic_vars.instantiate_type(ty, self.modules.primitives())?;
                let Some(element_type_instance_str) =
                    local_type_instance.get_rust_element_instance_ty()
                else {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                };
                if let Some(element_type_impl) = self.modules.primitives().get(&(
                    "ElementType".to_string(),
                    element_type_instance_str.to_string(),
                )) {
                    ty_attrs = self
                        .modules
                        .get_primitives_attrs("ElementType", &element_type_instance_str);
                    _maybe_generic_element_type = Some((type_name.clone(), element_type_impl));
                }
                if ty_attrs.is_none() {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                }
                type_instance = Some(local_type_instance);
            }
            _ => return self.jit_error_result(&ty.span(), "Failed to compile type"),
        };

        // Compile cuda_tile type.
        if ty_attrs.is_none() {
            return self.jit_error_result(&ty.span(), "Unexpected type");
        }
        let ty_attrs = ty_attrs.unwrap();
        if structure.is_some() {
            let _ty_str = ty.to_token_stream().to_string();
            let cuda_tile_attr_name_vec = ty_attrs.name_as_vec().unwrap();
            let attr_name = *cuda_tile_attr_name_vec.last().unwrap();
            if attr_name != "ty" {
                return self
                    .jit_error_result(&ty.span(), &format!("Unexpected attribute: {ty_attrs:#?}"));
            }
            let type_name = ty_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile compiling type {} using attrs {ty_attrs:#?}",
                        ty.to_token_stream()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            let params = ty_attrs.parse_string_arr("type_params").unwrap_or(vec![]);
            let mut args: Vec<TypeParam> = vec![];
            for param in &params {
                if let Some(type_param) = type_params.get(param) {
                    args.push(type_param.clone());
                    continue;
                }
                match param.as_str() {
                    "{D}xE" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    "{D}xP" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    "tile" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    "strides" | "traversal_strides" => return Ok(None),
                    "tensor_view" => return Ok(None),
                    _ => {
                        return self.jit_error_result(
                            &ty.span(),
                            &format!(
                                "Unexpected param {param} for {:?}",
                                ty.to_token_stream().to_string()
                            ),
                        )
                    }
                }
            }
            let type_params_optional = ty_attrs
                .parse_string_arr("type_params_optional")
                .unwrap_or(vec![]);
            for optional_type_param in &type_params_optional {
                if let Some(type_param) = type_params.get(optional_type_param) {
                    args.push(type_param.clone());
                    continue;
                }
                // Required-but-deferred params: if not yet resolved, the type
                // cannot be fully compiled at this point.
                match optional_type_param.as_str() {
                    "tensor_view" | "strides" => return Ok(None),
                    _ => {}
                }
            }
            let type_instance = type_instance.expect("Failed to instantiate type.");
            return Ok(Some(TileRustType::new_structured_type(
                type_name,
                generic_vars,
                self.modules.primitives(),
                args,
                type_instance,
            )?));
        }
        // Expecting a primitive cuda_tile type (either E or *mut E).
        if let Some(TypeInstance::ElementType(element_instance)) = type_instance {
            let Some(scalar_attrs) = self.modules.get_primitives_attrs("Scalar", "E") else {
                return self
                    .jit_error_result(&ty.span(), "misconfigured Scalar impl in core module");
            };
            let type_name = scalar_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile type {} using attrs {scalar_attrs:#?}",
                        element_instance.generic_ty.to_token_stream()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            let args: Vec<TypeParam> = vec![TypeParam::derive_param_from_type(
                "E".to_string(),
                element_instance.generic_ty.clone(),
                None,
                Some(TypeInstance::ElementType(element_instance.clone())),
            )];
            Ok(Some(TileRustType::new_primitive_type(
                type_name,
                generic_vars,
                self.modules.primitives(),
                args,
                TypeInstance::ElementType(element_instance),
            )?))
        } else if let Some(TypeInstance::PtrType(ptr_instance)) = type_instance {
            let Some(pointer_attrs) = self.modules.get_primitives_attrs(
                "Pointer",
                crate::types::pointer_impl_key(ptr_instance.is_mutable),
            ) else {
                return self
                    .jit_error_result(&ty.span(), "misconfigured Pointer impl in core module");
            };
            let type_name = pointer_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile compiling type {} using attrs {pointer_attrs:#?}",
                        ty.to_token_stream()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            let args: Vec<TypeParam> = vec![TypeParam::derive_param_from_type(
                "!cuda_tile.ptr<E>".to_string(),
                ptr_instance.generic_ty.clone(),
                None,
                Some(TypeInstance::PtrType(ptr_instance.clone())),
            )];
            Ok(Some(TileRustType::new_primitive_type(
                type_name,
                generic_vars,
                self.modules.primitives(),
                args,
                TypeInstance::PtrType(ptr_instance),
            )?))
        } else {
            self.jit_error_result(
                &ty.span(),
                &format!("Unable to instantiate Scalar or Pointer impls: type_instance={type_instance:#?}"),
            )
        }
    }
}
