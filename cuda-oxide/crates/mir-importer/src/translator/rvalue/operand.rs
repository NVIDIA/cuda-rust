/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [`translate_operand`] plus float-constant and shared-array helpers.

use super::coerce::{cast_to_declared_rust_pointer_type_if_needed, erase_thin_pointer_kind};
use super::const_alloc::{
    array_to_slice_unsize_info, interior_array_to_slice_unsize_info, slice_len_from_constant,
    translate_static_array_as_slice,
};
use super::const_bytes::{
    constant_bytes, translate_array_value_constant, translate_constant_value_from_bytes,
    translate_struct_constant, translate_tuple_constant, translate_zero_sized_constant_value,
};
use super::const_enum::{read_uint_from_bytes, translate_enum_constant};
use super::const_union::translate_union_constant;
use super::place_read::translate_place;
use super::promoted::translate_ptr_to_array_constant;
use super::static_global::{get_static_pointer_info, translate_static_global_pointer};
use super::statics::{
    shared_static_source_identity, shared_static_source_name, static_target_from_constant,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_iket::{ops::IketSentinelTokenOp, types::IketRangeTokenType};
use dialect_mir::attributes::MirFP16Attr;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::MirRefOp;
use dialect_mir::types::MirFP16Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_err_noloc, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
use std::num::NonZeroUsize;

fn read_float_constant_bits(
    constant: &mir::ConstOperand,
    kind_name: &str,
    byte_width: usize,
    loc: Location,
) -> TranslationResult<u128> {
    let bytes = constant_bytes(constant, kind_name, loc.clone())?;

    if bytes.len() < byte_width {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{kind_name} constant needs {byte_width} bytes, found {}",
                bytes.len()
            ))
        );
    }

    Ok(read_uint_from_bytes(&bytes[..byte_width]))
}

fn translate_float_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    const_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirFloatConstantOp;
    use pliron::builtin::attributes::{FPDoubleAttr, FPSingleAttr};

    /// Bit pattern decoded from the MIR constant, tagged by float width.
    enum FloatBits {
        F16(u16),
        F32(u32),
        F64(u64),
    }

    // Decode the constant bytes before allocating the op, so a decode
    // error cannot leave an orphan operation behind in the context.
    let bits = if const_ty.deref(ctx).is::<MirFP16Type>() {
        FloatBits::F16(read_float_constant_bits(constant, "f16", 2, loc.clone())? as u16)
    } else if const_ty.deref(ctx).is::<FP32Type>() {
        FloatBits::F32(read_float_constant_bits(constant, "f32", 4, loc.clone())? as u32)
    } else if const_ty.deref(ctx).is::<FP64Type>() {
        FloatBits::F64(read_float_constant_bits(constant, "f64", 8, loc.clone())? as u64)
    } else {
        unreachable!("translate_float_constant called with a non-float type");
    };

    let op = Operation::new(
        ctx,
        MirFloatConstantOp::get_concrete_op_info(),
        vec![const_ty],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let float_op = MirFloatConstantOp::new(op);

    match bits {
        FloatBits::F16(bits) => {
            float_op.set_attr_float_value_f16(ctx, MirFP16Attr::from_bits(bits));
        }
        FloatBits::F32(bits) => {
            float_op.set_attr_float_value(ctx, FPSingleAttr::from(f32::from_bits(bits)));
        }
        FloatBits::F64(bits) => {
            float_op.set_attr_float_value_f64(ctx, FPDoubleAttr::from(f64::from_bits(bits)));
        }
    }

    if let Some(prev) = prev_op {
        float_op.get_operation().insert_after(ctx, prev);
    } else {
        float_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let value = float_op.get_operation().deref(ctx).get_result(0);
    Ok((value, Some(float_op.get_operation())))
}

/// Translate a MIR Operand to a pliron IR [`Value`].
/// Returns the value and the last inserted operation (for proper ordering).
///
/// Handles Copy, Move (via translate_place) and Constant operands.
pub fn translate_operand(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            // Get the value from the place
            translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc)
        }
        mir::Operand::Constant(constant) => {
            // Get the Rust type of this constant
            let rust_ty = constant.ty();

            // Check if this is a pointer to SharedArray (static shared memory)
            if is_shared_array_pointer(&rust_ty) {
                // Extract element type, size, and alignment from SharedArray<T, N, ALIGN>
                let (elem_ty, array_size, alignment) = extract_shared_array_info(ctx, &rust_ty)?;

                // Allocation creates physical storage with Erased provenance.
                // The Rust constant's exact reference/raw-pointer kind is
                // established explicitly after the producer.
                let declared_ptr_ty = types::translate_type(ctx, &rust_ty)?;
                let storage_ptr_ty =
                    erase_thin_pointer_kind(ctx, declared_ptr_ty).ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(
                            "SharedArray constant did not translate to a thin pointer type"
                        ))
                    })?;

                // Create a MirSharedAllocOp to represent the shared memory allocation
                // This will be lowered to an LLVM global with addrspace(3)
                //
                // NOTE: We include the alloc key in the operation so the LLVM lowering
                // phase can deduplicate multiple references to the same static.
                use dialect_mir::ops::MirSharedAllocOp;
                let op = Operation::new(
                    ctx,
                    MirSharedAllocOp::get_concrete_op_info(),
                    vec![storage_ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let shared_alloc = MirSharedAllocOp::new(op);

                // Set the element type, size, and alloc key attributes
                use pliron::builtin::attributes::{IntegerAttr, StringAttr, TypeAttr};
                shared_alloc.set_attr_elem_type(ctx, TypeAttr::new(elem_ty));
                let size_attr = IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(
                        ctx,
                        64,
                        pliron::builtin::types::Signedness::Signless,
                    ),
                    pliron::utils::apint::APInt::from_u64(
                        array_size as u64,
                        std::num::NonZeroUsize::new(64).unwrap(),
                    ),
                );
                shared_alloc.set_attr_size(ctx, size_attr);

                // Full debug resolves an injective static key.  Off and line
                // tables deliberately retain the historical allocation key and
                // avoid the extra stable-MIR identity work entirely.
                let source_identity = value_map
                    .debug_variables()
                    .then(|| shared_static_source_identity(constant))
                    .flatten();
                let alloc_key = if let Some(identity) = &source_identity {
                    identity.key.clone()
                } else {
                    format!("{:?}", constant.const_)
                };
                shared_alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key));

                // Record which Rust `static` this is. The alloc key above is
                // opaque and lowering mints an anonymous `__shared_mem_N`
                // symbol, so without this the generated shared-memory blocks
                // cannot be attributed back to source.
                if let Some(identity) = source_identity {
                    shared_alloc.set_attr_source_name(ctx, StringAttr::new(identity.name));
                    shared_alloc.set_attr_source_key(ctx, StringAttr::new(identity.key));
                } else if let Some(source_name) = shared_static_source_name(constant) {
                    shared_alloc.set_attr_source_name(ctx, StringAttr::new(source_name));
                }

                // Set alignment if specified (non-zero)
                if alignment > 0 {
                    shared_alloc.set_alignment_value(ctx, alignment as u64);
                }

                if let Some(prev) = prev_op {
                    shared_alloc.get_operation().insert_after(ctx, prev);
                } else {
                    shared_alloc.get_operation().insert_at_front(block_ptr, ctx);
                }

                let storage = shared_alloc.get_operation().deref(ctx).get_result(0);
                let (value, last_op) = cast_to_declared_rust_pointer_type_if_needed(
                    ctx,
                    storage,
                    declared_ptr_ty,
                    block_ptr,
                    Some(shared_alloc.get_operation()),
                    loc.clone(),
                    MirPointerKindAuthorityAttr::StaticAddress,
                );

                return Ok((value, last_op));
            }

            // Check if this is a pointer to Barrier (static barrier in shared memory)
            if is_barrier_pointer(&rust_ty) {
                // Barrier is a single 64-bit value in shared memory (mbarrier state)
                let elem_ty = pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into();

                // Keep the allocation itself Erased, then establish the Rust
                // constant's declared pointer category at a visible boundary.
                let declared_ptr_ty = types::translate_type(ctx, &rust_ty)?;
                let storage_ptr_ty =
                    erase_thin_pointer_kind(ctx, declared_ptr_ty).ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(
                            "Barrier constant did not translate to a thin pointer type"
                        ))
                    })?;

                // Create a MirSharedAllocOp for the barrier
                use dialect_mir::ops::MirSharedAllocOp;
                let op = Operation::new(
                    ctx,
                    MirSharedAllocOp::get_concrete_op_info(),
                    vec![storage_ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let shared_alloc = MirSharedAllocOp::new(op);

                // Set attributes: element type (i64), size (1 element)
                use pliron::builtin::attributes::{IntegerAttr, StringAttr, TypeAttr};
                shared_alloc.set_attr_elem_type(ctx, TypeAttr::new(elem_ty));
                let size_attr = IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(
                        ctx,
                        64,
                        pliron::builtin::types::Signedness::Signless,
                    ),
                    pliron::utils::apint::APInt::from_u64(
                        1, // Single barrier element
                        std::num::NonZeroUsize::new(64).unwrap(),
                    ),
                );
                shared_alloc.set_attr_size(ctx, size_attr);

                let source_identity = value_map
                    .debug_variables()
                    .then(|| shared_static_source_identity(constant))
                    .flatten();
                let alloc_key = if let Some(identity) = &source_identity {
                    identity.key.clone()
                } else {
                    format!("{:?}", constant.const_)
                };
                shared_alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key));

                // A `Barrier` static occupies shared memory too, so name it
                // for the same attribution reason as `SharedArray` above.
                if let Some(identity) = source_identity {
                    shared_alloc.set_attr_source_name(ctx, StringAttr::new(identity.name));
                    shared_alloc.set_attr_source_key(ctx, StringAttr::new(identity.key));
                } else if let Some(source_name) = shared_static_source_name(constant) {
                    shared_alloc.set_attr_source_name(ctx, StringAttr::new(source_name));
                }

                if let Some(prev) = prev_op {
                    shared_alloc.get_operation().insert_after(ctx, prev);
                } else {
                    shared_alloc.get_operation().insert_at_front(block_ptr, ctx);
                }

                let storage = shared_alloc.get_operation().deref(ctx).get_result(0);
                let (value, last_op) = cast_to_declared_rust_pointer_type_if_needed(
                    ctx,
                    storage,
                    declared_ptr_ty,
                    block_ptr,
                    Some(shared_alloc.get_operation()),
                    loc.clone(),
                    MirPointerKindAuthorityAttr::StaticAddress,
                );

                return Ok((value, last_op));
            }

            // Ordinary Rust `static` / `static mut` values in device code live in
            // CUDA global memory (addrspace 1) by default. SharedArray/Barrier
            // statics have already been intercepted above and remain addrspace 3.
            // Statics tagged `#[constant]` (detected by the mangled symbol
            // prefix) instead lower into constant memory (addrspace 4).
            if let Some((pointee_ty, origin)) = get_static_pointer_info(&rust_ty)
                && let Some(static_target) = static_target_from_constant(constant, loc.clone())?
            {
                let is_mutable = origin.is_mutable();
                let static_ty = static_target.static_def.ty();
                let pointee_mir_ty = types::translate_type(ctx, &pointee_ty)?;
                let static_mir_ty = types::translate_type(ctx, &static_ty)?;

                // A zero addend may name either the whole static or a subobject
                // at byte zero (for example `&ARRAY[0]` or `&STRUCT.first`).
                // Array→slice unsize remains special because it needs a fat
                // pointer carrying the evaluated metadata word. Other sized
                // pointees continue through byte-address normalization below.
                if static_target.byte_offset == 0
                    && pointee_mir_ty != static_mir_ty
                    && let Some((elem_ty, array_len)) =
                        array_to_slice_unsize_info(&static_ty, &pointee_ty, loc.clone())?
                {
                    // The emitted length is the constant's own metadata
                    // word, not the array's N: a zero-addend prefix
                    // subslice (`split_at(k).0` over the static) stores
                    // k there. The array length only bounds it.
                    let len = slice_len_from_constant(constant, loc.clone())?;
                    if len > array_len {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "constant slice over device static {} stores length {}, \
                                 which exceeds the static array's length {}",
                                static_target.static_def.name(),
                                len,
                                array_len
                            ))
                        );
                    }
                    return translate_static_array_as_slice(
                        ctx,
                        &static_target.static_def,
                        elem_ty,
                        len,
                        origin,
                        0,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    );
                }

                if static_target.byte_offset != 0
                    && let Some((elem_ty, remaining_len)) = interior_array_to_slice_unsize_info(
                        &static_ty,
                        &pointee_ty,
                        static_target.byte_offset,
                        loc.clone(),
                    )?
                {
                    let len = slice_len_from_constant(constant, loc.clone())?;

                    if len > remaining_len {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "constant slice over device static {} stores length {}, \
                                 which exceeds the selected array region's remaining length {}",
                                static_target.static_def.name(),
                                len,
                                remaining_len,
                            ))
                        );
                    }

                    return translate_static_array_as_slice(
                        ctx,
                        &static_target.static_def,
                        elem_ty,
                        len,
                        origin,
                        static_target.byte_offset,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    );
                }

                // Every remaining path emits a thin pointer, including a
                // zero-addend pointer to a first field/element. A slice, str,
                // trait object, or another DST requires metadata and cannot be
                // represented here. Note that `layout()` succeeds for DSTs
                // such as `[f32]`, so check the returned shape explicitly.
                let pointee_layout = pointee_ty.layout().map_err(|e| {
                    input_error!(
                        loc.clone(),
                        TranslationErr::unsupported(format!(
                            "constant pointer into device static {} has byte offset {}, \
                             but pointee type {:?} does not have a sized layout: {:?}",
                            static_target.static_def.name(),
                            static_target.byte_offset,
                            pointee_ty,
                            e
                        ))
                    )
                })?;
                if !pointee_layout.shape().is_sized() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "constant pointer into device static {} has byte offset {}, \
                             but pointee type {:?} is unsized; cuda-oxide does not yet \
                             preserve the fat-pointer metadata this slice or DST pointer needs",
                            static_target.static_def.name(),
                            static_target.byte_offset,
                            pointee_ty
                        ))
                    );
                }

                // The materialized constant pointer must carry the exact
                // translated Rust operand type. Slot stores and mem2reg are
                // type-strict, so shapes like `_1 = const &STATIC;
                // &(*_1).field` need the operand normalized here rather
                // than leaving the physical-address-space pointer type
                // exposed to the rest of the function body.
                let result_ptr_ty = types::translate_type(ctx, &rust_ty)?;

                return translate_static_global_pointer(
                    ctx,
                    &static_target.static_def,
                    pointee_mir_ty,
                    result_ptr_ty,
                    is_mutable,
                    static_target.byte_offset,
                    block_ptr,
                    prev_op,
                    loc.clone(),
                );
            }

            let const_ty_ptr = types::translate_type(ctx, &rust_ty)?;

            // `RangeToken<R>` is intentionally a Rust ZST. Optimized MIR can
            // therefore materialize a token operand as an independent ZST
            // constant instead of preserving the call-result SSA edge. Keep a
            // well-typed semantic placeholder here; the frontend-provided
            // static range key pairs range_start/range_end during lowering.
            if const_ty_ptr.deref(ctx).is::<IketRangeTokenType>() {
                let op = IketSentinelTokenOp::new(ctx).get_operation();
                op.deref_mut(ctx).set_loc(loc);
                if let Some(prev) = prev_op {
                    op.insert_after(ctx, prev);
                } else {
                    op.insert_at_front(block_ptr, ctx);
                }
                return Ok((op.deref(ctx).get_result(0), Some(op)));
            }

            // ZSTs have no runtime bytes, but they still need a value with the
            // exact translated type. This is critical for marker structs,
            // unit, and zero-sized unions.
            if types::is_zst_type(ctx, const_ty_ptr) {
                return translate_zero_sized_constant_value(
                    ctx,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }

            // A fully-uninitialized constant allocation (`MaybeUninit::uninit()`
            // and similar: every byte uninit, no provenance) has no defined
            // bytes to materialize for any type; its value is `undef`.
            if let ConstantKind::Allocated(alloc) = constant.const_.kind()
                && !alloc.bytes.is_empty()
                && alloc.bytes.iter().all(|b| b.is_none())
                && alloc.provenance.ptrs.is_empty()
            {
                use dialect_mir::ops::MirUndefOp;
                let op = MirUndefOp::new(ctx, const_ty_ptr).get_operation();
                op.deref_mut(ctx).set_loc(loc);
                if let Some(prev) = prev_op {
                    op.insert_after(ctx, prev);
                } else {
                    op.insert_at_front(block_ptr, ctx);
                }
                return Ok((op.deref(ctx).get_result(0), Some(op)));
            }

            // Check if this is a struct type (non-ZST)
            // For struct constants, we need to construct the struct from its field values.
            let is_struct = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirStructType>();
            let is_tuple = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirTupleType>();

            // Check if this is a float type (f16, f32, or f64)
            let is_float_16 = const_ty_ptr.deref(ctx).is::<MirFP16Type>();
            let is_float_32 = const_ty_ptr.deref(ctx).is::<FP32Type>();
            let is_float_64 = const_ty_ptr.deref(ctx).is::<FP64Type>();
            let is_float = is_float_16 || is_float_32 || is_float_64;

            // Check if this is an enum type
            let is_enum = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirEnumType>();
            let is_union = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirUnionType>();

            // Check if this is a pointer to an array (byte strings, or typed arrays like [f64; 3])
            let is_ptr_to_array = const_ty_ptr
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirPtrType>()
                .map(|ptr_ty| {
                    ptr_ty
                        .pointee
                        .deref(ctx)
                        .is::<dialect_mir::types::MirArrayType>()
                })
                .unwrap_or(false);

            // Check if this is a bare array value constant (e.g. `const TABLE: [f32; N]`
            // referenced as `TABLE[runtime_idx]`, which materialises the whole array
            // as an operand rather than a pointer to it).
            let is_array_value = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirArrayType>();

            if is_float {
                return translate_float_constant(
                    ctx,
                    constant,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }

            // Debug repr kept for diagnostics only (CUDA_OXIDE_DEBUG_CONST and
            // the unsupported-constant error at the bottom); constant VALUES
            // are read through typed rustc_public APIs, never parsed from it.
            let const_str = format!("{:?}", constant.const_);

            // Handle pointer-to-array constants (byte strings, typed arrays like [f64; 3], etc.)
            if is_ptr_to_array {
                return translate_ptr_to_array_constant(
                    ctx,
                    constant,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }

            // Handle bare array value constants (e.g. `TABLE[runtime_idx]` where
            // `TABLE: [f32; N]` materialises the whole array by value).
            if is_array_value {
                return translate_array_value_constant(
                    ctx,
                    constant,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }

            if is_struct {
                // Non-ZST struct constant - extract field values and construct the struct
                translate_struct_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_tuple {
                translate_tuple_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_enum {
                translate_enum_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_union {
                translate_union_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirPtrType>()
            {
                // Pointer type constant - could be:
                // 1. A raw pointer constant (like core::ptr::null()) - just bytes,
                //    no provenance
                // 2. A reference to a constant struct (like &(8..16)) - need
                //    struct + mir.ref
                // 3. A reference to any other promoted constant (like the `&77`
                //    that -O const-folds out of `Option<&u32>::unwrap_or(&77)`,
                //    issue #132) - follow the allocation provenance, materialize
                //    the pointee constant, then mir.ref
                //
                // Only constants WITHOUT provenance may take the raw-pointer
                // path; a provenance entry always names a real allocation, and
                // ignoring it would lower the reference to `inttoptr 0` (a null
                // pointer).

                // Extract pointer type info before further borrows
                let (pointee_ty, is_mutable, pointee_is_struct) = {
                    let ty_ref = const_ty_ptr.deref(ctx);
                    let ptr_ty = ty_ref
                        .downcast_ref::<dialect_mir::types::MirPtrType>()
                        .unwrap();
                    let pointee = ptr_ty.pointee;
                    let mutable = ptr_ty.is_mutable;
                    let is_struct = pointee.deref(ctx).is::<dialect_mir::types::MirStructType>();
                    (pointee, mutable, is_struct)
                };

                // Check if the constant has actual struct data (not all zeros)
                // Handle both Allocated constants and promoted constants (Ty variant)
                //
                // Debug: print constant info for reference-to-struct types
                if pointee_is_struct && std::env::var("CUDA_OXIDE_DEBUG_CONST").is_ok() {
                    eprintln!(
                        "[DEBUG] Ptr-to-struct constant: kind={:?}, str={:?}",
                        constant.const_.kind(),
                        const_str
                    );
                }

                let has_struct_data = if pointee_is_struct {
                    match constant.const_.kind() {
                        ConstantKind::Allocated(alloc) => {
                            // For promoted constants like &(8..16), the bytes are zeros
                            // (pointer placeholder) but provenance indicates a real allocation.
                            // Check for provenance OR non-zero bytes.
                            let has_provenance = !alloc.provenance.ptrs.is_empty();
                            let has_nonzero_bytes = alloc
                                .raw_bytes()
                                .ok()
                                .map(|bytes| bytes.iter().any(|&b| b != 0))
                                .unwrap_or(false);
                            has_provenance || has_nonzero_bytes
                        }
                        ConstantKind::Ty(_) => {
                            // Promoted constants (like &(8..16)) are Ty variants
                            // These contain the actual struct data
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                };

                if has_struct_data {
                    // This is a reference to a constant struct (like &(8..16))

                    // Create the struct constant, then use mir.ref to get a pointer
                    let (struct_val, last_op) = translate_struct_constant(
                        ctx,
                        constant,
                        &rust_ty,
                        pointee_ty,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Now create mir.ref to get a pointer to the struct
                    use dialect_mir::ops::MirRefOp;
                    let ref_op = Operation::new(
                        ctx,
                        MirRefOp::get_concrete_op_info(),
                        vec![const_ty_ptr], // Result is pointer to struct
                        vec![struct_val],   // Operand is the struct value
                        vec![],
                        0,
                    );
                    ref_op.deref_mut(ctx).set_loc(loc);

                    let mir_ref = MirRefOp::new(ref_op);

                    mir_ref
                        .set_attr_mutable(ctx, dialect_mir::attributes::MutabilityAttr(is_mutable));
                    mir_ref.set_pointer_kind_authority(
                        ctx,
                        MirPointerKindAuthorityAttr::StaticAddress,
                    );

                    if let Some(prev) = last_op {
                        mir_ref.get_operation().insert_after(ctx, prev);
                    } else {
                        mir_ref.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let ptr_val = mir_ref.get_operation().deref(ctx).get_result(0);
                    return Ok((ptr_val, Some(mir_ref.get_operation())));
                }

                // Reference to a non-struct promoted constant (issue #132).
                //
                // Under -O, MIR const-folds e.g. the `None` arm of
                // `Option<&u32>::unwrap_or(&77)` into a constant of type `&u32`
                // whose data bytes are a pointer placeholder and whose
                // provenance entry names the allocation holding the literal
                // `77`. Struct pointees were already handled above; follow the
                // provenance for every other pointee type too, materialize the
                // pointee through the shared constant-from-bytes path, and take
                // its address with mir.ref (mem2reg/lowering turn that into an
                // alloca + store + address; sound because promoted constants
                // are immutable).
                let backing_alloc: Option<&rustc_public::ty::Allocation> =
                    match constant.const_.kind() {
                        ConstantKind::Allocated(alloc) => Some(alloc),
                        ConstantKind::Ty(ty_const) => match ty_const.kind() {
                            rustc_public::ty::TyConstKind::Value(_, alloc) => Some(alloc),
                            _ => None,
                        },
                        _ => None,
                    };

                if let Some(alloc) = backing_alloc
                    && let Some(&(prov_pos, prov)) = alloc.provenance.ptrs.first()
                {
                    use rustc_public::mir::alloc::GlobalAlloc;
                    let alloc_id = prov.0;

                    // The pointer's own data bytes encode the byte offset into
                    // the target allocation (zero for plain promoted literals
                    // like `&77`). The struct/array provenance branches assume
                    // offset zero; here the slice below honors a non-zero
                    // offset, and an unreadable offset is a hard error rather
                    // than a silently wrong address.
                    let ptr_width =
                        rustc_public::target::MachineInfo::target_pointer_width().bytes();
                    let target_offset = alloc
                        .read_partial_uint(prov_pos..prov_pos + ptr_width)
                        .map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to read pointer constant provenance offset: {:?}",
                                e
                            )))
                        })? as usize;

                    // The pointee is decoded from raw bytes below, with no
                    // provenance map to resolve pointers nested inside it, so
                    // a pointee that itself contains relocations must fail
                    // loudly before its placeholder bytes decode as addresses.
                    let reject_pointee_relocations = |relocations: usize| -> TranslationResult<()> {
                        if relocations != 0 {
                            return input_err!(
                                loc.clone(),
                                TranslationErr::unsupported(format!(
                                    "Promoted constant's pointee contains {relocations} pointer \
                                         relocation(s); cuda-oxide cannot yet preserve nested \
                                         pointer provenance"
                                ))
                            );
                        }
                        Ok(())
                    };
                    let target_bytes: Vec<u8> = match GlobalAlloc::from(alloc_id) {
                        GlobalAlloc::Memory(target_alloc) => {
                            reject_pointee_relocations(target_alloc.provenance.ptrs.len())?;
                            target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                                target_alloc
                                    .bytes
                                    .iter()
                                    .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                    .collect::<Vec<u8>>()
                            })
                        }
                        GlobalAlloc::Static(static_def) => {
                            let target_alloc = static_def.eval_initializer().map_err(|e| {
                                input_error_noloc!(TranslationErr::unsupported(format!(
                                    "Failed to evaluate static initializer for pointer constant: {:?}",
                                    e
                                )))
                            })?;
                            reject_pointee_relocations(target_alloc.provenance.ptrs.len())?;
                            target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                                target_alloc
                                    .bytes
                                    .iter()
                                    .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                    .collect::<Vec<u8>>()
                            })
                        }
                        other => {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Pointer constant provenance points to non-memory allocation: {:?}",
                                    other
                                ))
                            );
                        }
                    };

                    if target_offset > target_bytes.len() {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Pointer constant provenance offset {} exceeds target allocation size {}",
                                target_offset,
                                target_bytes.len()
                            ))
                        );
                    }

                    // The shared materializer needs the pointee's Rust type for
                    // enum-layout queries and ZST detection.
                    let Some((pointee_rust_ty, _)) = get_static_pointer_info(&rust_ty) else {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Pointer constant with provenance has unsupported Rust type: {:?}",
                                rust_ty
                            ))
                        );
                    };

                    let (pointee_val, last_op) = translate_constant_value_from_bytes(
                        ctx,
                        &pointee_rust_ty,
                        pointee_ty,
                        &target_bytes[target_offset..],
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Take the address of the materialized value, exactly like
                    // the struct branch above.
                    let ref_op = Operation::new(
                        ctx,
                        MirRefOp::get_concrete_op_info(),
                        vec![const_ty_ptr], // Result is pointer to the pointee
                        vec![pointee_val],  // Operand is the materialized value
                        vec![],
                        0,
                    );
                    ref_op.deref_mut(ctx).set_loc(loc);

                    let mir_ref = MirRefOp::new(ref_op);
                    mir_ref
                        .set_attr_mutable(ctx, dialect_mir::attributes::MutabilityAttr(is_mutable));
                    mir_ref.set_pointer_kind_authority(
                        ctx,
                        MirPointerKindAuthorityAttr::StaticAddress,
                    );

                    if let Some(prev) = last_op {
                        mir_ref.get_operation().insert_after(ctx, prev);
                    } else {
                        mir_ref.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let ptr_val = mir_ref.get_operation().deref(ctx).get_result(0);
                    return Ok((ptr_val, Some(mir_ref.get_operation())));
                }

                // Raw pointer constant (like core::ptr::null()).
                //
                // Only reachable for constants WITHOUT provenance (true null or
                // int-to-ptr values); provenance-carrying constants returned
                // above. Create an integer constant with the pointer value
                // (0 for null), then convert it to a pointer type using
                // MirCastOp
                use dialect_mir::ops::MirCastOp;

                // No provenance, so the data bytes ARE the address. Read the
                // first pointer-width bytes as a target-endian uint:
                //
                //   bytes: [00 00 00 00 00 00 00 00] -> 0x00 (null)
                //   bytes: [2a 00 00 00 00 00 00 00] -> 0x2a (without_provenance(42))
                //
                // A constant with no backing allocation, too few bytes, or an
                // uninit byte in the range has no readable address; post-mono
                // MIR should never produce that here, so fail loudly instead
                // of defaulting to null.
                let Some(alloc) = backing_alloc else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "raw pointer constant has no backing allocation to read an \
                             address from (constant kind: {:?})",
                            constant.const_.kind()
                        ))
                    );
                };
                debug_assert!(
                    alloc.provenance.ptrs.is_empty(),
                    "provenance-carrying pointer constants must return earlier"
                );
                let ptr_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
                let ptr_val = alloc.read_partial_uint(0..ptr_width).map_err(|e| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "raw pointer constant needs {ptr_width} initialized address \
                         bytes, but reading them failed: {e:?}"
                    )))
                })? as u64;

                // Create integer constant (i64) for the pointer value
                let i64_ty = pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Signless,
                );
                let apint = APInt::from_u64(ptr_val, NonZeroUsize::new(64).unwrap());
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);

                use dialect_mir::ops::MirConstantOp;
                let int_op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![i64_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                int_op.deref_mut(ctx).set_loc(loc.clone());

                let const_op = MirConstantOp::new(int_op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                let int_val = const_op.get_operation().deref(ctx).get_result(0);

                // Cast integer to pointer type using MirCastOp
                let cast_op = Operation::new(
                    ctx,
                    MirCastOp::get_concrete_op_info(),
                    vec![const_ty_ptr], // Result is the pointer type
                    vec![int_val],      // Operand is the integer value
                    vec![],
                    0,
                );
                cast_op.deref_mut(ctx).set_loc(loc);
                let cast = MirCastOp::new(cast_op);
                cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
                if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, const_ty_ptr) {
                    cast.set_pointer_kind_authority(
                        ctx,
                        MirPointerKindAuthorityAttr::StaticAddress,
                    );
                }

                cast_op.insert_after(ctx, const_op.get_operation());

                let ptr_val_result = cast_op.deref(ctx).get_result(0);

                Ok((ptr_val_result, Some(cast_op)))
            } else if const_ty_ptr.deref(ctx).is::<IntegerType>() {
                // Integer constant
                let (width_val, signedness) = {
                    let const_ty_obj = const_ty_ptr.deref(ctx);
                    let int_ty = const_ty_obj
                        .downcast_ref::<IntegerType>()
                        .expect("already checked is::<IntegerType>()");
                    (int_ty.width(), int_ty.signedness())
                };

                let byte_size = (width_val as usize).div_ceil(8);

                // Walk to the backing allocation directly instead of going
                // through `constant_bytes`: that helper zero-fills uninit
                // bytes, and an integer's value bytes must all be present.
                //
                //   bytes: [ff 7f] -> 0x7fff  (fine)
                //   bytes: [-- 7f] -> error   (uninit byte, not 0x7f00)
                //
                // No fallback: a constant we cannot read here is a compiler
                // bug, never a zero.
                let alloc = match constant.const_.kind() {
                    ConstantKind::Allocated(alloc) => alloc,
                    ConstantKind::Ty(ty_const) => match ty_const.kind() {
                        rustc_public::ty::TyConstKind::Value(_, alloc) => alloc,
                        other => {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "integer constant is not an evaluated value \
                                     (TyConstKind::{other:?}), so its bytes cannot be read"
                                ))
                            );
                        }
                    },
                    other => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "integer constant has no byte-backed allocation \
                                 (constant kind: {other:?})"
                            ))
                        );
                    }
                };
                if !alloc.provenance.ptrs.is_empty() {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "integer constant carries pointer provenance; its bytes are \
                             a relocation placeholder, not a number",
                        )
                    );
                }
                if alloc.bytes.len() < byte_size {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "integer constant has {} data bytes, expected at least \
                             {byte_size} for i{width_val}",
                            alloc.bytes.len()
                        ))
                    );
                }
                let int_val = alloc.read_partial_uint(0..byte_size).map_err(|e| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "integer constant has uninitialized bytes in its value range: {e:?}"
                    )))
                })?;

                let width = NonZeroUsize::new(width_val as usize).unwrap();
                let apint = APInt::from_u128(int_val, width);

                let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(ctx, width_val, signedness),
                    apint,
                );

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![const_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);

                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                let val = const_op.get_operation().deref(ctx).get_result(0);

                Ok((val, Some(const_op.get_operation())))
            } else {
                // No matching type handler — report what we got so it's clear what needs support.
                let pliron_ty_dbg = format!("{:?}", const_ty_ptr.deref(ctx));
                Err(input_error_noloc!(TranslationErr::unsupported(format!(
                    "Unsupported constant type in translate_constant.\n\
                     \n  Rust type : {:?}\
                     \n  pliron type: {}\
                     \n  const repr : {}\
                     \n\
                     \nThe type dispatch (ZST -> ptr_to_array -> array -> struct -> tuple -> enum -> union -> float -> pointer -> integer)\n\
                     did not match this constant. A new handler may need to be added.",
                    rust_ty, pliron_ty_dbg, const_str
                ))))
            }
        }
        mir::Operand::RuntimeChecks(_) => {
            // RuntimeChecks variants (UbChecks, ContractChecks, OverflowChecks)
            // evaluate to `false` on GPU -- runtime safety checks are disabled.
            //
            // Emits a `mir.constant false : i1` and inserts it into the current
            // block. The op *must* be linked before returning; callers use the
            // returned `last_op` as the insertion point for subsequent ops.
            use dialect_mir::ops::MirConstantOp;
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::{IntegerType, Signedness};
            use pliron::utils::apint::APInt;

            let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let value = APInt::from_u64(
                u64::from(crate::DEVICE_RUNTIME_CHECKS_VALUE),
                std::num::NonZeroUsize::new(1).unwrap(),
            );
            let const_attr =
                IntegerAttr::new(IntegerType::get(ctx, 1, Signedness::Signless), value);

            let op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![bool_ty],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let const_op = MirConstantOp::new(op);
            const_op.set_attr_value(ctx, const_attr);

            match prev_op {
                Some(p) => op.insert_after(ctx, p),
                None => op.insert_at_front(block_ptr, ctx),
            }

            let val = const_op.get_operation().deref(ctx).get_result(0);

            Ok((val, Some(const_op.get_operation())))
        }
    }
}

/// Check if a type is a pointer to cuda-device's SharedArray.
fn is_shared_array_pointer(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            // Check if the pointee is SharedArray
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => {
                    types::is_cuda_device_adt(&adt_def, "SharedArray")
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Check if a type is a pointer to cuda-device's Barrier (mbarrier state in
/// shared memory).
fn is_barrier_pointer(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            // Check if the pointee is Barrier
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => {
                    types::is_cuda_device_adt(&adt_def, "Barrier")
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Extract element type, size, and alignment from a pointer to SharedArray<T, N, ALIGN>.
/// Returns (element_type, size, alignment) where alignment is 0 for natural alignment.
fn extract_shared_array_info(
    ctx: &mut Context,
    ty: &rustc_public::ty::Ty,
) -> TranslationResult<(pliron::r#type::TypeHandle, usize, usize)> {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => {
                    if !types::is_cuda_device_adt(&adt_def, "SharedArray") {
                        return input_err_noloc!(TranslationErr::unsupported(format!(
                            "Expected cuda_device::SharedArray, got {}",
                            adt_def.trimmed_name()
                        )));
                    }

                    let generic_args = &substs.0;

                    // SharedArray substs arrive in declaration order, with
                    // the ALIGN default already materialized by rustc:
                    //
                    //   SharedArray<T, N, ALIGN = 0>
                    //     [0] T      type
                    //     [1] N      const usize (element count)
                    //     [2] ALIGN  const usize (0 = natural alignment)
                    //
                    // Reads are positional on purpose: if N fails to
                    // evaluate that's an error, never "slide over and read
                    // ALIGN as N". If cuda-device ever reorders these
                    // generics, this read must change in the same commit.
                    debug_assert!(
                        generic_args.len() == 2 || generic_args.len() == 3,
                        "SharedArray is declared <T, const N, const ALIGN = 0>; \
                         got {} generic args",
                        generic_args.len()
                    );

                    let Some(GenericArgKind::Type(elem_ty)) = generic_args.first() else {
                        return input_err_noloc!(TranslationErr::unsupported(
                            "SharedArray substs missing the element type at position 0"
                        ));
                    };

                    let Some(GenericArgKind::Const(n_const)) = generic_args.get(1) else {
                        return input_err_noloc!(TranslationErr::unsupported(
                            "SharedArray substs missing the N const generic at position 1"
                        ));
                    };
                    let size = facts::eval_usize_const(n_const, "SharedArray N", None)? as usize;

                    // ALIGN = 0 means natural alignment (the declaration
                    // default). Only a genuinely two-generic SharedArray may
                    // omit it; a present-but-unreadable position 2 is an
                    // error, silently dropping a user-requested
                    // over-alignment would miscompile.
                    let alignment = match generic_args.get(2) {
                        Some(GenericArgKind::Const(align_const)) => {
                            facts::eval_usize_const(align_const, "SharedArray ALIGN", None)?
                                as usize
                        }
                        Some(_) => {
                            return input_err_noloc!(TranslationErr::unsupported(
                                "SharedArray substs position 2 is not the ALIGN const generic"
                            ));
                        }
                        None => 0,
                    };

                    let translated_elem_ty = types::translate_type(ctx, elem_ty)?;
                    Ok((translated_elem_ty, size, alignment))
                }
                _ => input_err_noloc!(TranslationErr::unsupported(
                    "Expected ADT type for SharedArray"
                )),
            }
        }
        _ => input_err_noloc!(TranslationErr::unsupported("Expected raw pointer type")),
    }
}
