/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[type_interface_impl]` registrations for MIR → LLVM type conversion.
//!
//! Each type that `convert_type` can handle implements both:
//! - [`MirConvertibleType`] — marker badge for `can_convert_type`
//! - [`MirTypeConversion`] — returns a function pointer that does the conversion
//!
//! The function-pointer pattern avoids the borrow-checker conflict between
//! `type_cast` (borrows ctx immutably) and conversion (needs `&mut ctx`).

use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirFP16Type, MirPtrType, MirSliceType,
    MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::derive::type_interface_impl;

use crate::type_conversion_interface::{ConvertMirTypeFn, MirConvertibleType, MirTypeConversion};

use super::types::{
    StructLayoutInfo, build_struct_slot_map, build_union_storage_type, convert_enum_to_llvm,
    convert_type, make_disjoint_slice_struct, make_slice_struct,
};

// =============================================================================
// `dialect-mir` types
// =============================================================================

#[type_interface_impl]
impl MirConvertibleType for MirFP16Type {}

#[type_interface_impl]
impl MirTypeConversion for MirFP16Type {
    fn converter(&self) -> ConvertMirTypeFn {
        |_ty, ctx| Ok(llvm_types::HalfType::get(ctx).into())
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirDisjointSliceType {}

#[type_interface_impl]
impl MirTypeConversion for MirDisjointSliceType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let space_tys = ty
                .deref(ctx)
                .downcast_ref::<MirDisjointSliceType>()
                .map(|slice_ty| slice_ty.space_tys.clone())
                .unwrap_or_default();
            make_disjoint_slice_struct(ctx, &space_tys)
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirSliceType {}

#[type_interface_impl]
impl MirTypeConversion for MirSliceType {
    fn converter(&self) -> ConvertMirTypeFn {
        // `MirPointerKind` is semantic-only. `&[T]`, `&mut [T]`, and raw
        // slice pointers all keep the same physical `{ptr, i64}` layout.
        |_ty, ctx| Ok(make_slice_struct(ctx))
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirPtrType {}

#[type_interface_impl]
impl MirTypeConversion for MirPtrType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            // Deliberately erase Rust pointer/reference kind here. It is not
            // a physical LLVM pointer property and is not sufficient by
            // itself to justify `noalias`, `readonly`, or related metadata.
            let address_space = ty
                .deref(ctx)
                .downcast_ref::<MirPtrType>()
                .unwrap()
                .address_space;
            Ok(llvm_types::PointerType::get(ctx, address_space).into())
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirTupleType {}

#[type_interface_impl]
impl MirTypeConversion for MirTupleType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let layout = {
                let r = ty.deref(ctx);
                StructLayoutInfo::of_tuple(r.downcast_ref::<MirTupleType>().unwrap())
            };
            Ok(build_struct_slot_map(ctx, &layout)?.llvm_struct_ty)
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirStructType {}

#[type_interface_impl]
impl MirTypeConversion for MirStructType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let layout = {
                let r = ty.deref(ctx);
                StructLayoutInfo::of_struct(r.downcast_ref::<MirStructType>().unwrap())
            };
            Ok(build_struct_slot_map(ctx, &layout)?.llvm_struct_ty)
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirUnionType {}

#[type_interface_impl]
impl MirTypeConversion for MirUnionType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let union_ty = {
                let r = ty.deref(ctx);
                r.downcast_ref::<MirUnionType>().unwrap().clone()
            };
            build_union_storage_type(ctx, &union_ty)
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirEnumType {}

#[type_interface_impl]
impl MirTypeConversion for MirEnumType {
    fn converter(&self) -> ConvertMirTypeFn {
        // `{tag, variant fields...}`, plus a trailing `[N x i8]` pad when
        // rustc's total size is known and larger than the structural size.
        // See `convert_enum_to_llvm` for the full layout contract.
        |ty, ctx| convert_enum_to_llvm(ctx, ty)
    }
}

#[type_interface_impl]
impl MirConvertibleType for MirArrayType {}

#[type_interface_impl]
impl MirTypeConversion for MirArrayType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let (elem_ty, size) = {
                let r = ty.deref(ctx);
                let a = r.downcast_ref::<MirArrayType>().unwrap();
                (a.element_type(), a.size())
            };
            let llvm_elem_ty = convert_type(ctx, elem_ty)?;
            Ok(llvm_types::ArrayType::get(ctx, llvm_elem_ty, size).into())
        }
    }
}

// =============================================================================
// Pliron Builtin Types
// =============================================================================

#[type_interface_impl]
impl MirConvertibleType for IntegerType {}

#[type_interface_impl]
impl MirTypeConversion for IntegerType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let width = ty.deref(ctx).downcast_ref::<IntegerType>().unwrap().width();
            Ok(IntegerType::get(ctx, width, Signedness::Signless).into())
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for FP32Type {}

#[type_interface_impl]
impl MirTypeConversion for FP32Type {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, _ctx| Ok(ty)
    }
}

#[type_interface_impl]
impl MirConvertibleType for FP64Type {}

#[type_interface_impl]
impl MirTypeConversion for FP64Type {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, _ctx| Ok(ty)
    }
}

// =============================================================================
// LLVM Dialect Types (passthrough / recursive element conversion)
// =============================================================================

#[type_interface_impl]
impl MirConvertibleType for llvm_types::HalfType {}

#[type_interface_impl]
impl MirTypeConversion for llvm_types::HalfType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, _ctx| Ok(ty)
    }
}

#[type_interface_impl]
impl MirConvertibleType for llvm_types::PointerType {}

#[type_interface_impl]
impl MirTypeConversion for llvm_types::PointerType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, _ctx| Ok(ty)
    }
}

#[type_interface_impl]
impl MirConvertibleType for llvm_types::ArrayType {}

#[type_interface_impl]
impl MirTypeConversion for llvm_types::ArrayType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let (elem_ty, size) = {
                let r = ty.deref(ctx);
                let a = r.downcast_ref::<llvm_types::ArrayType>().unwrap();
                (a.elem_type(), a.size())
            };
            let llvm_elem_ty = convert_type(ctx, elem_ty)?;
            Ok(llvm_types::ArrayType::get(ctx, llvm_elem_ty, size).into())
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for llvm_types::VectorType {}

#[type_interface_impl]
impl MirTypeConversion for llvm_types::VectorType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let (elem_ty, size) = {
                let r = ty.deref(ctx);
                let v = r.downcast_ref::<llvm_types::VectorType>().unwrap();
                (v.elem_type(), v.num_elements())
            };
            let llvm_elem_ty = convert_type(ctx, elem_ty)?;
            Ok(llvm_types::VectorType::get(
                ctx,
                llvm_elem_ty,
                size,
                llvm_types::VectorTypeKind::Fixed,
            )
            .into())
        }
    }
}

#[type_interface_impl]
impl MirConvertibleType for llvm_types::StructType {}

#[type_interface_impl]
impl MirTypeConversion for llvm_types::StructType {
    fn converter(&self) -> ConvertMirTypeFn {
        |ty, ctx| {
            let (fields, layout) = {
                let r = ty.deref(ctx);
                let struct_ty = r.downcast_ref::<llvm_types::StructType>().unwrap();
                (struct_ty.fields().collect::<Vec<_>>(), struct_ty.layout())
            };
            let llvm_fields: Vec<_> = fields
                .into_iter()
                .map(|f| convert_type(ctx, f))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(llvm_types::StructType::get_unnamed(ctx, (llvm_fields, layout)).into())
        }
    }
}
