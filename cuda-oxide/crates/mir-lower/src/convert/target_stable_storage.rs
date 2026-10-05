/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Target-stable physical storage for values whose semantic LLVM type contains
//! address-space-sensitive leaves.
//!
//! Modern NVVM represents address-space-3 pointers as 32-bit values while the
//! legacy/PTX paths use 64-bit pointers. Any physical aggregate image that must
//! be identical before the backend mode is selected therefore uses CUDA generic
//! pointers as the stable carrier and converts shared pointers at value
//! boundaries. Enum payload storage also uses this module for its bool-byte
//! canonicalization.

use llvm_export::op_interfaces::{CastOpInterface, CastOpWithNNegInterface};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

#[derive(Clone, Copy)]
pub(crate) struct StorageRewriteOptions {
    pub canonicalize_bool: bool,
}

#[derive(Clone)]
pub(crate) struct StorageRewrite {
    pub ty: TypeHandle,
    pub shared_pointer_leaves: u64,
    pub array_shared_pointer_leaves: u64,
}

enum TypeShape {
    Bool,
    Pointer(u32),
    Array(TypeHandle, u64),
    Vector(TypeHandle),
    Struct {
        fields: Vec<TypeHandle>,
        layout: llvm_types::StructLayout,
    },
    Identity,
}

/// Rewrite a semantic LLVM value type into a backend-independent physical
/// storage type.
///
/// Shared pointers become generic pointers recursively. Bool leaves may also
/// be canonicalized to `i8` when requested by the caller, which is required by
/// enum byte storage but not by the internal packed-aggregate call ABI.
pub(crate) fn target_stable_storage_type(
    ctx: &mut Context,
    semantic_ty: TypeHandle,
    options: StorageRewriteOptions,
    role: &str,
) -> std::result::Result<StorageRewrite, anyhow::Error> {
    rewrite_storage_type(ctx, semantic_ty, options, role)
}

fn rewrite_storage_type(
    ctx: &mut Context,
    semantic_ty: TypeHandle,
    options: StorageRewriteOptions,
    role: &str,
) -> std::result::Result<StorageRewrite, anyhow::Error> {
    let shape = {
        let ty_ref = semantic_ty.deref(ctx);
        if options.canonicalize_bool
            && ty_ref
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 1)
        {
            TypeShape::Bool
        } else if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
            TypeShape::Pointer(pointer.address_space())
        } else if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            TypeShape::Array(array.elem_type(), array.size())
        } else if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
            TypeShape::Vector(vector.elem_type())
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            TypeShape::Struct {
                fields: struct_ty.fields().collect(),
                layout: struct_ty.layout(),
            }
        } else {
            TypeShape::Identity
        }
    };

    match shape {
        TypeShape::Bool => Ok(StorageRewrite {
            ty: IntegerType::get(ctx, 8, Signedness::Signless).into(),
            shared_pointer_leaves: 0,
            array_shared_pointer_leaves: 0,
        }),
        TypeShape::Pointer(address_space) if address_space == llvm_types::address_space::SHARED => {
            Ok(StorageRewrite {
                ty: llvm_types::PointerType::get_generic(ctx).into(),
                shared_pointer_leaves: 1,
                array_shared_pointer_leaves: 0,
            })
        }
        TypeShape::Pointer(_) | TypeShape::Identity => Ok(StorageRewrite {
            ty: semantic_ty,
            shared_pointer_leaves: 0,
            array_shared_pointer_leaves: 0,
        }),
        TypeShape::Array(element_ty, count) => {
            let element = rewrite_storage_type(ctx, element_ty, options, role)?;
            let shared_pointer_leaves = element
                .shared_pointer_leaves
                .checked_mul(count)
                .ok_or_else(|| {
                    anyhow::anyhow!("{role}: shared-pointer array rewrite size overflow")
                })?;
            let ty = if element.ty == element_ty {
                semantic_ty
            } else {
                llvm_types::ArrayType::get(ctx, element.ty, count).into()
            };
            Ok(StorageRewrite {
                ty,
                shared_pointer_leaves,
                array_shared_pointer_leaves: shared_pointer_leaves,
            })
        }
        TypeShape::Vector(element_ty) => {
            let element = rewrite_storage_type(ctx, element_ty, options, role)?;
            if element.ty != element_ty || element.shared_pointer_leaves != 0 {
                return Err(anyhow::anyhow!(
                    "{role}: vectors containing bool or shared-memory pointer elements are not supported"
                ));
            }
            Ok(StorageRewrite {
                ty: semantic_ty,
                shared_pointer_leaves: 0,
                array_shared_pointer_leaves: 0,
            })
        }
        TypeShape::Struct { fields, layout } => {
            let mut storage_fields = Vec::with_capacity(fields.len());
            let mut changed = false;
            let mut shared_pointer_leaves = 0_u64;
            let mut array_shared_pointer_leaves = 0_u64;
            for field in fields {
                let storage = rewrite_storage_type(ctx, field, options, role)?;
                changed |= storage.ty != field;
                shared_pointer_leaves = shared_pointer_leaves
                    .checked_add(storage.shared_pointer_leaves)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{role}: shared-pointer leaf count overflow while rewriting a struct payload"
                        )
                    })?;
                array_shared_pointer_leaves = array_shared_pointer_leaves
                    .checked_add(storage.array_shared_pointer_leaves)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "{role}: shared-pointer leaf count overflow while rewriting a struct payload"
                        )
                    })?;
                storage_fields.push(storage.ty);
            }
            let ty = if changed {
                llvm_types::StructType::get_unnamed(ctx, (storage_fields, layout)).into()
            } else {
                semantic_ty
            };
            Ok(StorageRewrite {
                ty,
                shared_pointer_leaves,
                array_shared_pointer_leaves,
            })
        }
    }
}

enum ValueCoercion {
    BoolToByte,
    ByteToBool,
    SharedToGeneric,
    GenericToShared,
    Array {
        target_element: TypeHandle,
        count: u64,
    },
    Struct(Vec<TypeHandle>),
    Unsupported,
}

/// Convert a value between its semantic type and a target-stable storage type.
///
/// The conversion is symmetric and recursively rebuilds arrays and structs so
/// pointer address-space conversions remain explicit SSA operations rather than
/// bit reinterpretations.
pub(crate) fn coerce_target_stable_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: TypeHandle,
    role: &str,
) -> Result<Value> {
    let source_ty = value.get_type(ctx);
    if source_ty == target_ty {
        return Ok(value);
    }

    let coercion = {
        let source_ref = source_ty.deref(ctx);
        let target_ref = target_ty.deref(ctx);

        let source_integer = source_ref
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        let target_integer = target_ref
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        if source_integer == Some(1) && target_integer == Some(8) {
            ValueCoercion::BoolToByte
        } else if source_integer == Some(8) && target_integer == Some(1) {
            ValueCoercion::ByteToBool
        } else if let (Some(source_pointer), Some(target_pointer)) = (
            source_ref.downcast_ref::<llvm_types::PointerType>(),
            target_ref.downcast_ref::<llvm_types::PointerType>(),
        ) {
            match (
                source_pointer.address_space(),
                target_pointer.address_space(),
            ) {
                (llvm_types::address_space::SHARED, llvm_types::address_space::GENERIC) => {
                    ValueCoercion::SharedToGeneric
                }
                (llvm_types::address_space::GENERIC, llvm_types::address_space::SHARED) => {
                    ValueCoercion::GenericToShared
                }
                _ => ValueCoercion::Unsupported,
            }
        } else if let (Some(source_array), Some(target_array)) = (
            source_ref.downcast_ref::<llvm_types::ArrayType>(),
            target_ref.downcast_ref::<llvm_types::ArrayType>(),
        ) {
            if source_array.size() == target_array.size() {
                ValueCoercion::Array {
                    target_element: target_array.elem_type(),
                    count: source_array.size(),
                }
            } else {
                ValueCoercion::Unsupported
            }
        } else if let (Some(source_struct), Some(target_struct)) = (
            source_ref.downcast_ref::<llvm_types::StructType>(),
            target_ref.downcast_ref::<llvm_types::StructType>(),
        ) {
            if source_struct.num_fields() == target_struct.num_fields() {
                ValueCoercion::Struct(target_struct.fields().collect())
            } else {
                ValueCoercion::Unsupported
            }
        } else {
            ValueCoercion::Unsupported
        }
    };

    match coercion {
        ValueCoercion::BoolToByte => {
            let zext = llvm::ZExtOp::new_with_nneg(ctx, value, target_ty, false);
            rewriter.insert_operation(ctx, zext.get_operation());
            Ok(zext.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::ByteToBool => {
            let trunc = llvm::TruncOp::new(ctx, value, target_ty);
            rewriter.insert_operation(ctx, trunc.get_operation());
            Ok(trunc.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::SharedToGeneric | ValueCoercion::GenericToShared => {
            let cast = llvm::AddrSpaceCastOp::new(ctx, value, target_ty);
            rewriter.insert_operation(ctx, cast.get_operation());
            Ok(cast.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::Array {
            target_element,
            count,
        } => {
            let undef = llvm::UndefOp::new(ctx, target_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let mut current = undef.get_operation().deref(ctx).get_result(0);
            for index in 0..count {
                let index = u32::try_from(index).map_err(|_| {
                    pliron::input_error_noloc!("{role} array has more than u32::MAX elements")
                })?;
                let extract = llvm::ExtractValueOp::new(ctx, value, vec![index])?;
                rewriter.insert_operation(ctx, extract.get_operation());
                let element = extract.get_operation().deref(ctx).get_result(0);
                let converted =
                    coerce_target_stable_value(ctx, rewriter, element, target_element, role)?;
                let insert = llvm::InsertValueOp::new(ctx, current, converted, vec![index]);
                rewriter.insert_operation(ctx, insert.get_operation());
                current = insert.get_operation().deref(ctx).get_result(0);
            }
            Ok(current)
        }
        ValueCoercion::Struct(fields) => {
            let undef = llvm::UndefOp::new(ctx, target_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let mut current = undef.get_operation().deref(ctx).get_result(0);
            for (index, target_field) in fields.into_iter().enumerate() {
                let index = u32::try_from(index).map_err(|_| {
                    pliron::input_error_noloc!("{role} struct has more than u32::MAX fields")
                })?;
                let extract = llvm::ExtractValueOp::new(ctx, value, vec![index])?;
                rewriter.insert_operation(ctx, extract.get_operation());
                let field = extract.get_operation().deref(ctx).get_result(0);
                let converted =
                    coerce_target_stable_value(ctx, rewriter, field, target_field, role)?;
                let insert = llvm::InsertValueOp::new(ctx, current, converted, vec![index]);
                rewriter.insert_operation(ctx, insert.get_operation());
                current = insert.get_operation().deref(ctx).get_result(0);
            }
            Ok(current)
        }
        ValueCoercion::Unsupported => pliron::input_err_noloc!(
            "{role} type mismatch: {} cannot be adapted to {}",
            source_ty.deref(ctx).disp(ctx),
            target_ty.deref(ctx).disp(ctx)
        ),
    }
}
