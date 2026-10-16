/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The instantiated type behind a spelling. `Tile<f32, { [N] }>` with
//! `N = 4` and `Tile<f32, { [4] }>` have the same identity.

use crate::generics::{
    carries_metadata_args, TypInstancePrimitiveType, TypeInstance, TypeInstancePtrType,
};
use quote::ToTokens;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TypeIdentity {
    /// Mutability of each `&`, outermost first.
    pub(crate) references: Vec<bool>,
    pub(crate) referent: Referent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Referent {
    Element(Element),
    Shaped {
        name: String,
        element: Option<Element>,
        shape: Vec<i32>,
    },
    Token,
    Str,
    /// Not instantiated (tuples, arrays, `Option`, plain structs, shaped
    /// types with metadata arguments): the spelling's tokens.
    Unresolved(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Element {
    Scalar(String),
    Pointer { mutable: bool, pointee: String },
}

impl Element {
    fn of(primitive: &TypInstancePrimitiveType) -> Element {
        match primitive {
            TypInstancePrimitiveType::ElementType(element) => {
                Element::Scalar(element.rust_element_instance_ty.clone())
            }
            TypInstancePrimitiveType::PtrType(pointer) => Element::pointer(pointer),
        }
    }

    fn pointer(instance: &TypeInstancePtrType) -> Element {
        Element::Pointer {
            mutable: instance.is_mutable,
            pointee: instance.rust_element_instance_ty.clone(),
        }
    }
}

impl Referent {
    pub(crate) fn of(instance: &TypeInstance) -> Referent {
        match instance {
            TypeInstance::ElementType(element) => {
                Referent::Element(Element::Scalar(element.rust_element_instance_ty.clone()))
            }
            TypeInstance::PtrType(pointer) => Referent::Element(Element::pointer(pointer)),
            TypeInstance::TokenType(_) => Referent::Token,
            TypeInstance::StringType(_) => Referent::Str,
            TypeInstance::StructuredType(structured) => match type_name(&structured.instance_ty) {
                Some(name) if !carries_metadata_args(&name) => Referent::Shaped {
                    name,
                    element: structured.primitive_type.as_ref().map(Element::of),
                    shape: structured.shape.clone(),
                },
                _ => Referent::unresolved(&structured.instance_ty),
            },
            TypeInstance::UserType(user) => Referent::unresolved(&user.maybe_generic_ty),
        }
    }

    fn unresolved(ty: &syn::Type) -> Referent {
        Referent::Unresolved(ty.to_token_stream().to_string())
    }
}

/// `T` in `&mut &T`.
pub(crate) fn referent(mut ty: &syn::Type) -> &syn::Type {
    while let syn::Type::Reference(reference) = ty {
        ty = &reference.elem;
    }
    ty
}

fn type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = referent(ty) else {
        return None;
    };
    Some(path.path.segments.last()?.ident.to_string())
}
