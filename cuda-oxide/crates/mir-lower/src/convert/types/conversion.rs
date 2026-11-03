/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `convert_type` dispatch, ZST detection, and the slice fat-pointer builders.

use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::r#type::{TypeHandle, type_cast};

use crate::type_conversion_interface::MirTypeConversion;

// =============================================================================
// Zero-Sized Type (ZST) Detection
// =============================================================================

/// Check if a type is zero-sized (empty struct).
///
/// Zero-sized types include:
/// - Empty structs `struct {}`
/// - PhantomData markers (which become empty structs in MIR)
/// - Structs where all fields are themselves zero-sized
///
/// # Why This Matters
///
/// LLVM's NVPTX backend doesn't support empty struct types in function
/// signatures. We strip these during type conversion to avoid:
/// `LLVM ERROR: Empty parameter types are not supported`
///
/// # Background
///
/// Rust's `#[inline(always)]` attribute is stored in `codegen_fn_attrs`, which
/// is not exposed through the stable_mir API. Since we intercept MIR and generate
/// our own LLVM IR, we don't propagate inline hints. When LLVM decides not to
/// inline a function, the empty struct parameters/returns cause NVPTX to crash.
///
/// By stripping ZSTs at the LLVM type level, we avoid this issue regardless of
/// inlining decisions.
pub fn is_zero_sized_type(ctx: &Context, ty: TypeHandle) -> bool {
    if let Some(array_ty) = ty.deref(ctx).downcast_ref::<llvm_types::ArrayType>() {
        return array_ty.size() == 0 || is_zero_sized_type(ctx, array_ty.elem_type());
    }

    // Check if LLVM StructType with zero fields
    if let Some(struct_ty) = ty.deref(ctx).downcast_ref::<llvm_types::StructType>() {
        let num_fields = struct_ty.num_fields();
        if num_fields == 0 {
            return true;
        }
        // Also check if ALL fields are zero-sized (nested PhantomData)
        return struct_ty.fields().all(|f| is_zero_sized_type(ctx, f));
    }
    false
}

// =============================================================================
// Type Conversion
// =============================================================================

/// Convert a `dialect-mir` type to its LLVM dialect equivalent.
///
/// Dispatches via `MirTypeConversion` type interface — each supported type
/// registers a converter function pointer through `#[type_interface_impl]`
/// in [`crate::convert::type_interface_impls`].
///
/// The function-pointer indirection avoids a borrow-checker conflict:
/// `type_cast` borrows `ctx` immutably, but conversion needs `&mut ctx`.
/// We extract the `Copy` function pointer, drop the borrow, then call it.
pub fn convert_type(ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle, anyhow::Error> {
    // Phase 1: extract a Copy function pointer while ctx is immutably borrowed.
    let converter_fn = {
        let ty_ref = ty.deref(ctx);
        type_cast::<dyn MirTypeConversion>(&*ty_ref).map(|conv| conv.converter())
    };
    // Phase 2: borrow dropped — ctx is free for &mut.
    if let Some(conv_fn) = converter_fn {
        return conv_fn(ty, ctx);
    }

    let type_display = ty.deref(ctx).disp(ctx).to_string();
    Err(anyhow::anyhow!(
        "Unsupported type conversion: {}\n\
         Supported: integers, fp32, fp64, pointers, slices, tuples, structs, enums, arrays, vectors.",
        type_display
    ))
}

/// Create the LLVM struct type used for slice representations.
///
/// Slices are represented as fat pointers: `{ ptr, i64 }` where:
/// - `ptr` is a generic address space (0) pointer to the data
/// - `i64` is the number of elements (not bytes)
///
/// # Layout
///
/// ```text
/// struct {
///     ptr: !llvm.ptr,     ; offset 0, size 8
///     len: i64,           ; offset 8, size 8
/// }                       ; total size: 16 bytes
/// ```
///
/// # Address Space
///
/// The pointer uses generic address space (0) because:
/// - Slices passed to kernels may point to global memory
/// - The kernel doesn't know at compile time which memory space
/// - Generic pointers can be used with any memory type
///
/// # Usage
///
/// This type is used for:
/// - `&[T]` slice arguments
/// - `DisjointSlice<T>` (unique-ownership slice) arguments
/// - Any other fat pointer representation
pub(crate) fn make_slice_struct(ctx: &mut Context) -> TypeHandle {
    let ptr_ty = llvm_types::PointerType::get_generic(ctx);
    let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    llvm_types::StructType::get_unnamed(
        ctx,
        (
            vec![ptr_ty.into(), len_ty.into()],
            llvm_types::StructLayout::Unpacked,
        ),
    )
    .into()
}

/// The fat pointer, followed by an index space's runtime layout fields.
///
/// With no such fields this is exactly [`make_slice_struct`], so a slice over
/// an index space fixed in its type keeps its two-field representation.
pub(crate) fn make_disjoint_slice_struct(
    ctx: &mut Context,
    space_tys: &[TypeHandle],
) -> anyhow::Result<TypeHandle> {
    if space_tys.is_empty() {
        return Ok(make_slice_struct(ctx));
    }
    let ptr_ty = llvm_types::PointerType::get_generic(ctx);
    let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let mut fields: Vec<TypeHandle> = vec![ptr_ty.into(), len_ty.into()];
    for space_ty in space_tys {
        fields.push(convert_type(ctx, *space_ty)?);
    }
    Ok(
        llvm_types::StructType::get_unnamed(ctx, (fields, llvm_types::StructLayout::Unpacked))
            .into(),
    )
}
