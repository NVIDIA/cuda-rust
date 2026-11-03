/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared recursive MIR shape vocabulary for packed-AS3 carriers.
//!
//! This does not decide which contexts may materialize a carrier. Internal
//! device ABI and compiler-owned local storage keep their separate admission,
//! array-budget, and provenance checks.

use dialect_mir::types::{MirArrayType, MirFP16Type, MirPtrType, MirStructType, MirTupleType};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::Context;
use pliron::r#type::TypeHandle;

/// Return `true` when `ty` is a zero-sized MIR array, struct, or tuple.
fn mir_type_is_zero_sized(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        return array_ty.size() == 0 || mir_type_is_zero_sized(ctx, array_ty.element_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return struct_ty
            .field_types
            .iter()
            .all(|field| mir_type_is_zero_sized(ctx, *field));
    }
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty
            .get_types()
            .iter()
            .all(|field| mir_type_is_zero_sized(ctx, *field));
    }
    false
}

/// The structural subset understood by the packed-AS3 storage rewriter.
/// Admission remains the caller's responsibility; vectors and all unrelated
/// aggregate kinds must fail closed.
pub(crate) fn packed_shared_mir_shape_is_supported(ctx: &Context, mir_ty: TypeHandle) -> bool {
    let children = {
        let ty_ref = mir_ty.deref(ctx);
        if ty_ref.is::<IntegerType>()
            || ty_ref.is::<MirFP16Type>()
            || ty_ref.is::<llvm_types::HalfType>()
            || ty_ref.is::<FP32Type>()
            || ty_ref.is::<FP64Type>()
            || ty_ref.is::<MirPtrType>()
            || ty_ref.is::<llvm_types::PointerType>()
        {
            return true;
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            Some(struct_ty.field_types.clone())
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            Some(tuple_ty.get_types().to_vec())
        } else {
            ty_ref
                .downcast_ref::<MirArrayType>()
                .map(|array_ty| vec![array_ty.element_type()])
        }
    };

    children.is_some_and(|children| {
        children.into_iter().all(|child| {
            mir_type_is_zero_sized(ctx, child) || packed_shared_mir_shape_is_supported(ctx, child)
        })
    })
}
