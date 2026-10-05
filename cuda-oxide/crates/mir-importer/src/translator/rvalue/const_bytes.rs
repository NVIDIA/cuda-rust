/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Relocation-free constant decoding from raw allocation bytes.

use super::coerce::cast_struct_fields_to_expected_types;
use super::const_alloc::{
    translate_array_constant_from_alloc, translate_struct_constant_from_alloc,
    translate_tuple_constant_from_alloc, validate_array_value_element_type,
};
use super::const_enum::{read_uint_from_bytes, translate_enum_constant_from_bytes};
use super::promoted::constant_allocation;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use dialect_mir::attributes::MirFP16Attr;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{MirCastOp, MirConstructStructOp, MirUndefOp};
use dialect_mir::types::MirFP16Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
use std::num::NonZeroUsize;

/// Lower a bare `MirArrayType` value constant (e.g. `const TABLE: [f32; N] =
/// [..]` indexed by runtime value) to a `MirConstructArrayOp`. Element stride
/// and aggregate field offsets come from rustc layout. Thin pointer fields
/// inside elements that relocate to device statics are materialized via
/// [`MirGlobalAllocOp`] per field.
pub(super) fn translate_array_value_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let element_ty = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let Some(array_ty) = ty_obj.downcast_ref::<dialect_mir::types::MirArrayType>() else {
            return input_err!(
                loc,
                TranslationErr::unsupported("translate_array_value_constant: expected array type")
            );
        };
        array_ty.element_type()
    };

    // Bare array values support primitive scalars, enums, initialized unions,
    // tuples, structs, and nested arrays of those. Struct elements are decoded
    // through the existing layout-aware aggregate path while pointer-to-array
    // promotion remains intentionally unchanged; arrays nested inside struct
    // constants have their own layout-aware entry point.
    validate_array_value_element_type(ctx, element_ty, &loc)?;

    let rust_array_ty = constant.const_.ty();
    let alloc = match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => alloc.clone(),
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => alloc.clone(),
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Array value constant must be backed by bytes, found TyConstKind::{other:?}"
                    ))
                );
            }
        },
        ConstantKind::ZeroSized => {
            return translate_array_value_constant_inner(
                ctx,
                constant,
                const_ty_ptr,
                rust_array_ty,
                block_ptr,
                prev_op,
                loc,
            );
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Array value constant must be Allocated or Ty::Value, got {other:?}"
                ))
            );
        }
    };

    translate_array_constant_from_alloc(
        ctx,
        &alloc,
        0,
        &rust_array_ty,
        const_ty_ptr,
        block_ptr,
        prev_op,
        loc,
    )
}

pub(super) fn rust_type_layout_size(
    ty: rustc_public::ty::Ty,
    loc: Location,
) -> TranslationResult<usize> {
    ty.layout()
        .map(|layout| layout.shape().size.bytes())
        .map_err(|error| {
            input_error!(
                loc,
                TranslationErr::unsupported(format!(
                    "Failed to query rustc layout for constant type {ty:?}: {error:?}"
                ))
            )
        })
}

pub(super) fn rust_array_type_info(
    ty: rustc_public::ty::Ty,
    loc: Location,
) -> TranslationResult<(rustc_public::ty::Ty, u64)> {
    match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Array(element_ty, count)) => {
            let count = count.eval_target_usize().map_err(|error| {
                input_error!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Failed to evaluate array constant length: {error:?}"
                    ))
                )
            })?;
            Ok((element_ty, count))
        }
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Array constant expected a Rust array type, got {other:?}"
            ))
        ),
    }
}

/// Build a `MirConstructArrayOp` (and the necessary scalar / nested-array
/// element ops) from a slice of raw bytes for an `array_ty`. Recurses on
/// `MirArrayType` element types so multi-dimensional arrays (`[[T; M]; N]`,
/// etc.) are handled by repeated decomposition.
fn build_array_op_from_bytes(
    ctx: &mut Context,
    array_ty: TypeHandle,
    rust_array_ty: rustc_public::ty::Ty,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};

    // Element type + count.
    let (element_ty_ptr, element_count) = {
        let arr_ty_obj = array_ty.deref(ctx);
        let arr_ty = arr_ty_obj
            .downcast_ref::<dialect_mir::types::MirArrayType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "build_array_op_from_bytes: expected array type"
                ))
            })?;
        (arr_ty.element_type(), arr_ty.size())
    };

    let (rust_element_ty, rust_element_count) = rust_array_type_info(rust_array_ty, loc.clone())?;
    if rust_element_count != element_count {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Array constant length mismatch: Rust type has {rust_element_count} elements, dialect type has {element_count}"
            ))
        );
    }
    let element_byte_size = rust_type_layout_size(rust_element_ty, loc.clone())?;

    let element_count_usize = usize::try_from(element_count).map_err(|_| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Array constant element count {element_count} does not fit usize"
        )))
    })?;
    let expected_bytes = element_count_usize
        .checked_mul(element_byte_size)
        .ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Array constant byte size overflowed: {} elements x {} bytes each",
                element_count, element_byte_size
            )))
        })?;
    if bytes.len() != expected_bytes {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Array constant has {} bytes but requires exactly {} ({} elements x {} bytes each)",
                bytes.len(),
                expected_bytes,
                element_count,
                element_byte_size
            ))
        );
    }

    #[derive(Clone, Copy)]
    enum ElemKind {
        F64,
        F32,
        F16,
        Int { width: u32, signedness: Signedness },
        Array,
        Tuple,
        Struct,
    }
    let elem_kind = {
        let elem_obj = element_ty_ptr.deref(ctx);
        if elem_obj.is::<FP64Type>() {
            ElemKind::F64
        } else if elem_obj.is::<FP32Type>() {
            ElemKind::F32
        } else if elem_obj.is::<MirFP16Type>() {
            ElemKind::F16
        } else if let Some(int_ty) = elem_obj.downcast_ref::<IntegerType>() {
            ElemKind::Int {
                width: int_ty.width(),
                signedness: int_ty.signedness(),
            }
        } else if elem_obj.is::<dialect_mir::types::MirArrayType>() {
            ElemKind::Array
        } else if elem_obj.is::<dialect_mir::types::MirTupleType>() {
            ElemKind::Tuple
        } else if elem_obj.is::<dialect_mir::types::MirStructType>() {
            ElemKind::Struct
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Array constant element type is not supported by byte lowering: {:?}. \
                     Byte lowering handles primitive scalars, tuples, structs, or nested \
                     arrays of those. Enum elements decode from a constant \
                     allocation instead, so an enum array that reaches byte lowering (e.g. \
                     one with a zero-sized element) cannot be materialized here.",
                    elem_obj
                ))
            );
        }
    };

    let mut element_values = Vec::with_capacity(element_count_usize);
    let mut last_op = prev_op;

    for i in 0..element_count_usize {
        let chunk = &bytes[i * element_byte_size..(i + 1) * element_byte_size];

        let (elem_val, elem_last_op) = match elem_kind {
            ElemKind::F64 => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(chunk);
                let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                    rustc_public::target::Endian::Little => f64::from_le_bytes(buf),
                    rustc_public::target::Endian::Big => f64::from_be_bytes(buf),
                };
                let float_attr = pliron::builtin::attributes::FPDoubleAttr::from(float_val);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value_f64(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::F32 => {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(chunk);
                let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                    rustc_public::target::Endian::Little => f32::from_le_bytes(buf),
                    rustc_public::target::Endian::Big => f32::from_be_bytes(buf),
                };
                let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::F16 => {
                let bits = read_uint_from_bytes(chunk) as u16;
                let float_attr = MirFP16Attr::from_bits(bits);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value_f16(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::Int { width, signedness } => {
                let val = read_uint_from_bytes(chunk);
                let apint = APInt::from_u128(val, NonZeroUsize::new(width as usize).unwrap());
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                    IntegerType::get(ctx, width, signedness),
                    apint,
                );

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = last_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    const_op.get_operation().deref(ctx).get_result(0),
                    Some(const_op.get_operation()),
                )
            }
            ElemKind::Array => build_array_op_from_bytes(
                ctx,
                element_ty_ptr,
                rust_element_ty,
                chunk,
                block_ptr,
                last_op,
                loc.clone(),
            )?,
            ElemKind::Tuple | ElemKind::Struct => translate_constant_value_from_bytes(
                ctx,
                &rust_element_ty,
                element_ty_ptr,
                chunk,
                block_ptr,
                last_op,
                loc.clone(),
            )?,
        };

        element_values.push(elem_val);
        last_op = elem_last_op;
    }

    use dialect_mir::ops::MirConstructArrayOp;
    let construct_op = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty],
        element_values,
        vec![],
        0,
    );
    construct_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        construct_op.insert_after(ctx, prev);
    } else {
        construct_op.insert_at_front(block_ptr, ctx);
    }
    last_op = Some(construct_op);

    let array_val = construct_op.deref(ctx).get_result(0);
    Ok((array_val, last_op))
}

/// Extract the raw allocation bytes for a bare array value, then recursively
/// build the corresponding `MirConstructArrayOp`.
fn translate_array_value_constant_inner(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    array_ty: TypeHandle,
    rust_array_ty: rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let bytes = constant_bytes(constant, "Array", loc.clone())?;

    build_array_op_from_bytes(
        ctx,
        array_ty,
        rust_array_ty,
        &bytes,
        block_ptr,
        prev_op,
        loc,
    )
}

/// ## How it works
///
/// 1. Resolve the struct's own allocation (following by-ref provenance when the
///    constant is a promoted `&Struct`)
/// 2. Decode each field at its rustc layout offset from that allocation
/// 3. Thin pointer fields that relocate to device statics become
///    [`MirGlobalAllocOp`] results; other fields use the byte decoder
/// 4. Create MirConstructStructOp with those operands
///
/// Each field is read at the byte offset rustc's layout records for it, so
/// padding between fields and any reordering rustc applies are both accounted
/// for. A field's size comes from the same layout, which is why a padded struct
/// nested inside another is sliced at its true width rather than at the sum of
/// its fields.
///
/// Aggregate **const** values with thin pointers to device statics are
/// materialized per field; device-global initializers use a separate
/// allocation-level relocation path instead of this value reconstruction.
pub(super) fn translate_struct_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    // Confirm the MIR type is a struct before touching the allocation.
    {
        let ty_obj = const_ty_ptr.deref(ctx);
        if ty_obj
            .downcast_ref::<dialect_mir::types::MirStructType>()
            .is_none()
        {
            return input_err!(
                loc,
                TranslationErr::unsupported("translate_struct_constant called on non-struct type")
            );
        }
    }

    // The constant's Rust type decides how to read the allocation. A
    // reference or raw pointer means stable_mir handed over a promoted
    // constant indirectly (`&(8..16)`): the allocation is a thin pointer whose
    // provenance names the allocation with the actual struct data, and the
    // layout query must run on the pointee. An aggregate type means the
    // allocation IS the struct's memory image. Deciding by "has provenance"
    // conflated the two: a by-value struct with a pointer field would have had
    // its own bytes silently replaced by the first pointee's.
    let by_ref_pointee = match rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, pointee, _))
        | rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::RawPtr(pointee, _)) => {
            Some(pointee)
        }
        _ => None,
    };
    let struct_rust_ty = by_ref_pointee.unwrap_or(*rust_ty);

    let alloc = match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => {
            if by_ref_pointee.is_some() {
                use rustc_public::mir::alloc::GlobalAlloc;
                let Some(&(prov_pos, prov)) = alloc.provenance.ptrs.first() else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "Reference-to-struct constant has no provenance to follow".to_string()
                        )
                    );
                };
                // The pointer's data bytes encode the byte offset into the
                // target allocation. Field decoding below starts at byte zero
                // of that target, so an interior reference must fail loudly.
                let ptr_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
                let ref_offset = alloc
                    .read_partial_uint(prov_pos..prov_pos + ptr_width)
                    .map_err(|e| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Failed to read struct constant provenance offset: {:?}",
                            e
                        )))
                    })?;
                if ref_offset != 0 {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Reference-to-struct constant points at interior offset {ref_offset}; \
                             cuda-oxide cannot yet decode interior references to constants"
                        ))
                    );
                }
                let alloc_id = prov.0;
                match GlobalAlloc::from(alloc_id) {
                    GlobalAlloc::Memory(target_alloc) => target_alloc,
                    GlobalAlloc::Static(static_def) => {
                        static_def.eval_initializer().map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to evaluate static initializer for struct constant: {:?}",
                                e
                            )))
                        })?
                    }
                    other => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Struct constant provenance points to non-memory allocation: {:?}",
                                other
                            ))
                        );
                    }
                }
            } else {
                // The allocation is the struct's own memory image (may contain
                // thin-pointer relocations to device statics).
                alloc.clone()
            }
        }
        ConstantKind::ZeroSized => {
            return translate_struct_constant_from_bytes(
                ctx,
                &struct_rust_ty,
                const_ty_ptr,
                &[],
                block_ptr,
                prev_op,
                loc,
            );
        }
        _ => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Struct constant must be Allocated, got: {:?}. \
                     Consider using inline construction: `let s = MyStruct {{ field: value }};`",
                    constant.const_.kind()
                ))
            );
        }
    };

    translate_struct_constant_from_alloc(
        ctx,
        &alloc,
        0,
        &struct_rust_ty,
        const_ty_ptr,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Byte image for a tuple constant, or `None` when a sized tuple has no
/// backing allocation.
///
/// Undefined bytes in an allocation are padding; they are zeroed
/// deterministically while the provenance map stays available separately for
/// pointer-field reconstruction. `ConstantKind::ZeroSized`-style constants
/// (e.g. `((), ())`) carry no allocation at all; a zero-byte layout is
/// reproduced exactly by an empty image.
fn tuple_constant_byte_image(
    allocation: Option<&rustc_public::ty::Allocation>,
    layout_size: usize,
) -> Option<Vec<u8>> {
    match allocation {
        Some(allocation) => Some(
            allocation
                .bytes
                .iter()
                .map(|byte| byte.unwrap_or(0))
                .collect(),
        ),
        None if layout_size == 0 => Some(Vec::new()),
        None => None,
    }
}

/// Translate a non-empty tuple constant from its own allocation image.
///
/// Unlike `constant_bytes`, this must not follow the first provenance entry:
/// for a by-value tuple, that entry names one pointer field's target, while the
/// allocation itself still contains the tuple's scalar fields and padding.
/// Pointer fields consume their relocation entries through the
/// allocation-aware decoder below.
pub(super) fn translate_tuple_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let layout_size = rust_ty
        .layout()
        .map_err(|error| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to query layout for tuple constant: {error:?}"
                ))
            )
        })?
        .shape()
        .size
        .bytes();

    let Some(allocation) = constant_allocation(constant) else {
        let bytes = tuple_constant_byte_image(None, layout_size).ok_or_else(|| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Tuple constant of {layout_size} byte(s) must be backed by an allocation, \
                     found {:?}",
                    constant.const_.kind()
                ))
            )
        })?;
        return translate_tuple_constant_from_bytes(
            ctx,
            rust_ty,
            const_ty_ptr,
            &bytes,
            block_ptr,
            prev_op,
            loc,
        );
    };

    translate_tuple_constant_from_alloc(
        ctx,
        allocation,
        0,
        rust_ty,
        const_ty_ptr,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Translate a tuple constant from bytes using rustc's field offsets.
fn translate_tuple_constant_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let (field_types, mir_field_offsets, mir_memory_order, mir_total_size, mir_abi_align) = {
        let ty_ref = const_ty_ptr.deref(ctx);
        let tuple_ty = ty_ref
            .downcast_ref::<dialect_mir::types::MirTupleType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_tuple_constant called on non-tuple type"
                ))
            })?;
        (
            tuple_ty.get_types().to_vec(),
            tuple_ty.field_offsets().to_vec(),
            tuple_ty.memory_order(),
            tuple_ty.total_size(),
            tuple_ty.abi_align(),
        )
    };

    let rust_field_types = match rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(fields)) => {
            fields.to_vec()
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant expected Rust tuple type, got {:?}",
                    other
                ))
            );
        }
    };

    if field_types.len() != rust_field_types.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Tuple constant type mismatch: MIR has {} fields, Rust type has {}",
                field_types.len(),
                rust_field_types.len()
            ))
        );
    }

    let layout = rust_ty.layout().map_err(|error| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Failed to query layout for tuple constant: {error:?}"
            ))
        )
    })?;
    let shape = layout.shape();
    let tuple_size = shape.size.bytes();
    if bytes.len() != tuple_size {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Tuple constant has {} bytes but rustc layout requires exactly {tuple_size}",
                bytes.len()
            ))
        );
    }

    let field_offsets = match &shape.fields {
        rustc_public::abi::FieldsShape::Primitive if field_types.is_empty() => vec![],
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => offsets
            .iter()
            .map(|offset| offset.bytes())
            .collect::<Vec<_>>(),
        fields => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant fields use unsupported layout shape {fields:?}"
                ))
            );
        }
    };
    if field_offsets.len() != field_types.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Tuple constant layout has {} offsets for {} fields",
                field_offsets.len(),
                field_types.len()
            ))
        );
    }
    if !field_types.is_empty() {
        let rust_field_offsets = field_offsets
            .iter()
            .map(|offset| *offset as u64)
            .collect::<Vec<_>>();
        let rust_memory_order = shape.fields.fields_by_offset_order();
        if mir_field_offsets != rust_field_offsets
            || mir_memory_order != rust_memory_order
            || mir_total_size != tuple_size as u64
            || mir_abi_align != shape.abi_align
        {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant layout disagrees between rustc and dialect-mir: rustc offsets/order/size/alignment {:?}/{:?}/{}/{}, dialect {:?}/{:?}/{}/{}",
                    rust_field_offsets,
                    rust_memory_order,
                    tuple_size,
                    shape.abi_align,
                    mir_field_offsets,
                    mir_memory_order,
                    mir_total_size,
                    mir_abi_align
                ))
            );
        }
    }

    let mut values = Vec::with_capacity(field_types.len());
    let mut current_prev_op = prev_op;

    for (field_idx, (field_ty, rust_field_ty)) in field_types
        .iter()
        .copied()
        .zip(rust_field_types.iter())
        .enumerate()
    {
        let byte_offset = field_offsets[field_idx];
        let byte_size = rust_type_layout_size(*rust_field_ty, loc.clone())?;

        let field_end = byte_offset.checked_add(byte_size).ok_or_else(|| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Tuple constant field {field_idx} overflowed offset computation"
                ))
            )
        })?;
        let field_bytes = if byte_size == 0 {
            &[][..]
        } else if field_end <= bytes.len() {
            &bytes[byte_offset..field_end]
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant has insufficient bytes for field {} (need {} bytes at offset {}, have {})",
                    field_idx,
                    byte_size,
                    byte_offset,
                    bytes.len()
                ))
            );
        };

        let (value, new_prev_op) = translate_constant_value_from_bytes(
            ctx,
            rust_field_ty,
            field_ty,
            field_bytes,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        values.push(value);
        current_prev_op = new_prev_op;
    }

    use dialect_mir::ops::MirConstructTupleOp;
    let op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = current_prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok((op.deref(ctx).get_result(0), Some(op)))
}

/// Storage size of a struct constant from its recorded layout, or `None` when
/// that layout is missing.
///
/// rustc's size counts the padding between and after fields, so it is what a
/// reader must stride by. Summing the field sizes would under-report any padded
/// struct and hand a short byte slice to whoever reads it.
///
/// Recorded offsets are what separates a known layout from a failed query:
/// `translator/types.rs` stores empty offsets and a zero size when `Ty::layout()`
/// fails. A zero size on its own does not imply a failure, since a struct whose
/// fields are all zero-sized is genuinely zero bytes wide. `is_zst_type` does not
/// cover those, because it calls a struct zero-sized only when it has no fields
/// at all, so a `PhantomData` newtype reaches here with one field and a size of
/// zero.
fn struct_storage_size(field_count: usize, offset_count: usize, total_size: u64) -> Option<usize> {
    if offset_count == 0 && field_count > 0 {
        None
    } else {
        Some(total_size as usize)
    }
}

pub(super) fn constant_storage_size(ctx: &Context, ty_ptr: TypeHandle) -> Option<usize> {
    let ty_ref = ty_ptr.deref(ctx);
    if types::is_zst_type(ctx, ty_ptr) {
        Some(0)
    } else if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        Some((int_ty.width() as usize).div_ceil(8))
    } else if ty_ref.is::<MirFP16Type>() {
        Some(2)
    } else if ty_ref.is::<FP32Type>() {
        Some(4)
    } else if ty_ref.is::<FP64Type>() {
        Some(8)
    } else if ty_ref.is::<dialect_mir::types::MirPtrType>() {
        Some(rustc_public::target::MachineInfo::target_pointer_width().bytes())
    } else if let Some(st) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
        struct_storage_size(
            st.field_types().len(),
            st.field_offsets().len(),
            st.total_size(),
        )
    } else if let Some(union_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirUnionType>() {
        usize::try_from(union_ty.total_size()).ok()
    } else if let Some(at) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        let elem = at.element_type();
        let n = at.size() as usize;
        Some(constant_storage_size(ctx, elem)? * n)
    } else {
        None
    }
}

/// Whether a constant-storage type contains a pointer-bearing leaf.
///
/// Raw-byte constant materialization is only sound when this returns false.
/// Provenance-aware aggregate paths use the same predicate to decide whether
/// they need typed reconstruction instead.
pub(super) fn constant_type_contains_pointer(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<dialect_mir::types::MirPtrType>()
        || ty_ref.is::<dialect_mir::types::MirSliceType>()
        || ty_ref.is::<dialect_mir::types::MirDisjointSliceType>()
    {
        return true;
    }
    if let Some(array) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        let element = array.element_type();
        drop(ty_ref);
        return constant_type_contains_pointer(ctx, element);
    }
    let children: Vec<TypeHandle> =
        if let Some(tuple) = ty_ref.downcast_ref::<dialect_mir::types::MirTupleType>() {
            tuple.get_types().to_vec()
        } else if let Some(structure) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
            structure.field_types().to_vec()
        } else if let Some(enumeration) = ty_ref.downcast_ref::<dialect_mir::types::MirEnumType>() {
            enumeration.all_field_types.clone()
        } else if let Some(union_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirUnionType>() {
            union_ty.field_types().to_vec()
        } else {
            return false;
        };
    drop(ty_ref);
    children
        .into_iter()
        .any(|child| constant_type_contains_pointer(ctx, child))
}

/// Translate a struct value from raw bytes plus the Rust type/layout metadata.
///
/// This is the byte-slice counterpart to [`translate_struct_constant`] and is
/// used whenever a constant field has a struct type (e.g. `NonZero<T>` wrappers
/// inside enum payloads). Each field is parsed recursively so nested newtypes
/// are handled automatically.
fn translate_struct_constant_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    struct_bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use rustc_public::ty::{RigidTy, TyKind};

    let field_types: Vec<TypeHandle> = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let struct_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirStructType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_struct_constant_from_bytes called on non-struct type"
                ))
            })?;
        struct_ty.field_types().to_vec()
    };

    let layout = rust_ty.layout().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to query layout for struct constant: {:?}",
            e
        )))
    })?;
    let shape = layout.shape();

    // A zero-sized struct span holds no bytes and cannot carry relocations,
    // and not every type that lands here as a MirStructType is an ADT:
    // function items and non-capturing closures have no ADT metadata to
    // consult. Synthesize such values from the dialect type alone.
    if shape.size.bytes() == 0 {
        return translate_zero_sized_constant_value(ctx, const_ty_ptr, block_ptr, prev_op, loc);
    }

    let field_offsets: Vec<usize> = match &shape.fields {
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
            offsets.iter().map(|offset| offset.bytes()).collect()
        }
        rustc_public::abi::FieldsShape::Primitive => vec![0; field_types.len()],
        rustc_public::abi::FieldsShape::Union { .. } => vec![0; field_types.len()],
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Struct constant fields use unsupported shape {:?}",
                    other
                ))
            );
        }
    };

    let (adt_def, substs) = match rust_ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => (adt_def, substs),
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Expected ADT Rust type for struct constant, got {:?}",
                    other
                ))
            );
        }
    };

    // Structs have a single variant in the ADT metadata.
    let variants = adt_def.variants();
    let struct_variant = variants.first().ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(
            "Struct ADT has no variants in metadata"
        ))
    })?;

    let mut field_values = Vec::with_capacity(field_types.len());
    let mut current_prev_op = prev_op;

    for (field_idx, field_ty_ptr) in field_types.iter().copied().enumerate() {
        let fields = struct_variant.fields();
        let rust_field = fields.get(field_idx).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Struct constant field {} is missing in rustc ADT metadata ({} field(s) recorded)",
                field_idx,
                fields.len()
            )))
        })?;
        let rust_field_ty = rust_field.ty_with_args(&substs);
        let field_layout = rust_field_ty.layout().map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to query layout for struct field {}: {:?}",
                field_idx, e
            )))
        })?;
        let field_size = field_layout.shape().size.bytes();
        let field_offset = *field_offsets.get(field_idx).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Missing layout offset for struct field {}",
                field_idx
            )))
        })?;

        if field_size == 0 {
            let (zst_val, new_prev_op) = translate_zero_sized_constant_value(
                ctx,
                field_ty_ptr,
                block_ptr,
                current_prev_op,
                loc.clone(),
            )?;
            field_values.push(zst_val);
            current_prev_op = new_prev_op;
            continue;
        }

        let field_end = field_offset.checked_add(field_size).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Struct field {} offset {} + size {} overflowed",
                field_idx, field_offset, field_size
            )))
        })?;
        if field_end > struct_bytes.len() {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Struct constant has {} bytes, but field {} needs [{}..{})",
                    struct_bytes.len(),
                    field_idx,
                    field_offset,
                    field_end
                ))
            );
        }

        let field_bytes = &struct_bytes[field_offset..field_end];
        let (field_val, new_prev_op) = translate_constant_value_from_bytes(
            ctx,
            &rust_field_ty,
            field_ty_ptr,
            field_bytes,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        field_values.push(field_val);
        current_prev_op = new_prev_op;
    }

    let (casted_field_values, prev_after_casts) = cast_struct_fields_to_expected_types(
        ctx,
        field_values,
        const_ty_ptr,
        block_ptr,
        current_prev_op,
        loc.clone(),
    );

    let op = Operation::new(
        ctx,
        MirConstructStructOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        casted_field_values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_after_casts {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok((op.deref(ctx).get_result(0), Some(op)))
}

/// Translate one field-sized byte slice into a constant value.
pub(super) fn translate_constant_value_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    ty_ptr: TypeHandle,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let is_enum = {
        let ty_ref = ty_ptr.deref(ctx);
        ty_ref.is::<dialect_mir::types::MirEnumType>()
    };
    if is_enum {
        return translate_enum_constant_from_bytes(
            ctx, rust_ty, ty_ptr, bytes, block_ptr, prev_op, loc,
        );
    }

    // Aggregate decoders own their complete field model, including non-empty
    // aggregates whose every field is zero-sized. Dispatch them before the
    // generic ZST synthesizer, which only has the translated type and cannot
    // recover a Rust aggregate's active variant or field metadata.
    if ty_ptr.deref(ctx).is::<dialect_mir::types::MirTupleType>() {
        return translate_tuple_constant_from_bytes(
            ctx, rust_ty, ty_ptr, bytes, block_ptr, prev_op, loc,
        );
    }

    // Struct-typed constants (e.g. `NonZero<T>` wrappers inside enum payloads)
    // need per-field construction rather than a single scalar constant.
    let is_struct = {
        let ty_ref = ty_ptr.deref(ctx);
        ty_ref.is::<dialect_mir::types::MirStructType>()
    };
    if is_struct {
        return translate_struct_constant_from_bytes(
            ctx, rust_ty, ty_ptr, bytes, block_ptr, prev_op, loc,
        );
    }

    let is_zst = rust_ty
        .layout()
        .map(|layout| layout.shape().is_1zst())
        .map_err(|error| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to query layout for aggregate constant field {rust_ty:?}: {error:?}"
                ))
            )
        })?;
    if is_zst || types::is_zst_type(ctx, ty_ptr) {
        return translate_zero_sized_constant_value(ctx, ty_ptr, block_ptr, prev_op, loc);
    }

    if ty_ptr.deref(ctx).is::<dialect_mir::types::MirUnionType>() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "Initialized union constants require their rustc allocation so the initialization mask is preserved; byte-only union decoding is not supported"
                    .to_string()
            )
        );
    }

    enum ValueKind {
        Integer { width: u32, signedness: Signedness },
        Float16,
        Float32,
        Float64,
        Pointer,
        Unsupported(String),
    }

    let value_kind = {
        let ty_ref = ty_ptr.deref(ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            ValueKind::Integer {
                width: int_ty.width(),
                signedness: int_ty.signedness(),
            }
        } else if ty_ref.is::<MirFP16Type>() {
            ValueKind::Float16
        } else if ty_ref.is::<FP32Type>() {
            ValueKind::Float32
        } else if ty_ref.is::<FP64Type>() {
            ValueKind::Float64
        } else if ty_ref.is::<dialect_mir::types::MirPtrType>() {
            ValueKind::Pointer
        } else {
            ValueKind::Unsupported(format!("{:?}", ty_ref))
        }
    };

    match value_kind {
        ValueKind::Integer { width, signedness } => {
            let byte_size = (width as usize).div_ceil(8);
            if bytes.len() < byte_size {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Integer constant needs {} bytes, found {}",
                        byte_size,
                        bytes.len()
                    ))
                );
            }

            let int_val = read_uint_from_bytes(&bytes[..byte_size]);
            let width_nz = NonZeroUsize::new(width as usize).unwrap();
            let apint = APInt::from_u128(int_val, width_nz);
            let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(ctx, width, signedness),
                apint,
            );

            use dialect_mir::ops::MirConstantOp;
            let op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let const_op = MirConstantOp::new(op);
            const_op.set_attr_value(ctx, int_attr);

            if let Some(prev) = prev_op {
                const_op.get_operation().insert_after(ctx, prev);
            } else {
                const_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                const_op.get_operation().deref(ctx).get_result(0),
                Some(const_op.get_operation()),
            ))
        }
        ValueKind::Float16 => {
            if bytes.len() < 2 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f16 constant needs 2 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let bits = read_uint_from_bytes(&bytes[..2]) as u16;
            let float_attr = MirFP16Attr::from_bits(bits);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value_f16(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Float32 => {
            if bytes.len() < 4 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f32 constant needs 4 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let mut field_bytes = [0u8; 4];
            field_bytes.copy_from_slice(&bytes[..4]);
            let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                rustc_public::target::Endian::Little => f32::from_le_bytes(field_bytes),
                rustc_public::target::Endian::Big => f32::from_be_bytes(field_bytes),
            };
            let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Float64 => {
            if bytes.len() < 8 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f64 constant needs 8 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let mut field_bytes = [0u8; 8];
            field_bytes.copy_from_slice(&bytes[..8]);
            let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                rustc_public::target::Endian::Little => f64::from_le_bytes(field_bytes),
                rustc_public::target::Endian::Big => f64::from_be_bytes(field_bytes),
            };
            let float_attr = pliron::builtin::attributes::FPDoubleAttr::from(float_val);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value_f64(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Pointer => {
            let pointer_bytes = rustc_public::target::MachineInfo::target_pointer_width().bytes();
            if bytes.len() < pointer_bytes {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Pointer constant needs {} bytes, found {}",
                        pointer_bytes,
                        bytes.len()
                    ))
                );
            }

            let ptr_val = read_uint_from_bytes(&bytes[..pointer_bytes]) as u64;
            let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
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

            let const_value = const_op.get_operation().deref(ctx).get_result(0);
            let cast_op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![const_value],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc.clone());
            let cast = MirCastOp::new(cast_op);
            cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
            if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, ty_ptr) {
                cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
            }
            cast_op.insert_after(ctx, const_op.get_operation());

            Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
        }
        ValueKind::Unsupported(ty_name) => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Aggregate constant field type is not yet supported: {}",
                ty_name
            ))
        ),
    }
}

/// Build a zero-sized value while preserving its exact translated type.
pub(super) fn translate_zero_sized_constant_value(
    ctx: &mut Context,
    ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    enum ZeroSizedKind {
        Struct,
        EmptyTuple,
        Union,
        Unsupported(String),
    }

    let zero_sized_kind = {
        let ty_ref = ty_ptr.deref(ctx);
        if ty_ref.is::<dialect_mir::types::MirStructType>() {
            ZeroSizedKind::Struct
        } else if ty_ref.is::<dialect_mir::types::MirUnionType>() {
            ZeroSizedKind::Union
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirTupleType>() {
            if tuple_ty.get_types().is_empty() {
                ZeroSizedKind::EmptyTuple
            } else {
                ZeroSizedKind::Unsupported(
                    "Only empty tuple constants can be synthesized as zero-sized values"
                        .to_string(),
                )
            }
        } else {
            ZeroSizedKind::Unsupported(format!(
                "Zero-sized constant synthesis does not support type {:?}",
                ty_ref
            ))
        }
    };

    // A zero-sized struct can still carry (zero-sized) fields in its type, and
    // `MirConstructStructOp` requires one operand per field. Recursively
    // synthesize a ZST value for each field type (e.g. `TryFromIntError(())`,
    // which surfaces when building `core` for nvptx via `-Zbuild-std`).
    if matches!(zero_sized_kind, ZeroSizedKind::Struct) {
        let field_types: Vec<TypeHandle> = {
            let ty_ref = ty_ptr.deref(ctx);
            ty_ref
                .downcast_ref::<dialect_mir::types::MirStructType>()
                .map(|st| st.field_types.clone())
                .unwrap_or_default()
        };
        let mut operands = Vec::with_capacity(field_types.len());
        let mut cur_prev = prev_op;
        for fty in field_types {
            let (v, np) =
                translate_zero_sized_constant_value(ctx, fty, block_ptr, cur_prev, loc.clone())?;
            operands.push(v);
            cur_prev = np;
        }
        let op = Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![ty_ptr],
            operands,
            vec![],
            0,
        );
        op.deref_mut(ctx).set_loc(loc);
        if let Some(prev) = cur_prev {
            op.insert_after(ctx, prev);
        } else {
            op.insert_at_front(block_ptr, ctx);
        }
        return Ok((op.deref(ctx).get_result(0), Some(op)));
    }

    let op = match zero_sized_kind {
        ZeroSizedKind::Struct => unreachable!("handled above"),
        ZeroSizedKind::EmptyTuple => {
            use dialect_mir::ops::MirConstructTupleOp;
            Operation::new(
                ctx,
                MirConstructTupleOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            )
        }
        ZeroSizedKind::Union => MirUndefOp::new(ctx, ty_ptr).get_operation(),
        ZeroSizedKind::Unsupported(message) => {
            return input_err!(loc, TranslationErr::unsupported(message));
        }
    };
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok((op.deref(ctx).get_result(0), Some(op)))
}

/// Fetch the raw bytes backing a constant, following provenance for promoted
/// aggregate constants when necessary.
pub(crate) fn constant_bytes(
    constant: &mir::ConstOperand,
    kind_name: &str,
    loc: Location,
) -> TranslationResult<Vec<u8>> {
    use rustc_public::ty::TyConstKind;

    fn allocation_bytes_zeroing_uninit(alloc: &rustc_public::ty::Allocation) -> Vec<u8> {
        alloc.raw_bytes().ok().unwrap_or_else(|| {
            alloc
                .bytes
                .iter()
                .map(|opt: &Option<u8>| opt.unwrap_or(0))
                .collect::<Vec<u8>>()
        })
    }

    fn allocation_bytes(
        alloc: &rustc_public::ty::Allocation,
        kind_name: &str,
        loc: Location,
    ) -> TranslationResult<Vec<u8>> {
        use rustc_public::mir::alloc::GlobalAlloc;

        if let Some((_, prov)) = alloc.provenance.ptrs.first() {
            let alloc_id = prov.0;
            match GlobalAlloc::from(alloc_id) {
                GlobalAlloc::Memory(target_alloc) => {
                    Ok(allocation_bytes_zeroing_uninit(&target_alloc))
                }
                GlobalAlloc::Static(static_def) => {
                    let target_alloc = static_def.eval_initializer().map_err(|e| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Failed to evaluate static initializer for {} constant: {:?}",
                            kind_name, e
                        )))
                    })?;
                    Ok(allocation_bytes_zeroing_uninit(&target_alloc))
                }
                other => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "{} constant provenance points to non-memory allocation: {:?}",
                        kind_name, other
                    ))
                ),
            }
        } else {
            Ok(allocation_bytes_zeroing_uninit(alloc))
        }
    }

    match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => allocation_bytes(alloc, kind_name, loc),
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => allocation_bytes(alloc, kind_name, loc),
            TyConstKind::ZSTValue(_) => Ok(vec![]),
            other => input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{} constant must be backed by bytes, found TyConstKind::{:?}",
                    kind_name, other
                ))
            ),
        },
        ConstantKind::ZeroSized => Ok(vec![]),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{} constant must be Allocated or Ty::Value, got {:?}",
                kind_name, other
            ))
        ),
    }
}

#[cfg(test)]
mod tuple_constant_byte_image_tests {
    use super::tuple_constant_byte_image;
    use rustc_public::mir::Mutability;
    use rustc_public::ty::{Allocation, ProvenanceMap};

    #[test]
    fn zst_tuple_constant_without_allocation_is_an_empty_image() {
        // `ConstantKind::ZeroSized`-style tuple constants such as `((), ())`
        // carry no allocation; a zero-byte layout translates as empty bytes.
        assert_eq!(tuple_constant_byte_image(None, 0), Some(Vec::new()));
    }

    #[test]
    fn sized_tuple_constant_without_allocation_is_rejected() {
        assert_eq!(tuple_constant_byte_image(None, 16), None);
    }

    #[test]
    fn allocation_padding_bytes_are_zeroed_deterministically() {
        let allocation = Allocation {
            bytes: vec![Some(0xAB), None, None, Some(0xCD)],
            provenance: ProvenanceMap { ptrs: Vec::new() },
            align: 4,
            mutability: Mutability::Not,
        };
        assert_eq!(
            tuple_constant_byte_image(Some(&allocation), 4),
            Some(vec![0xAB, 0, 0, 0xCD])
        );
    }
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn struct_storage_size_reads_layout_presence_not_size() {
        // Ordinary padded struct: size is the padded width, not the field sum.
        assert_eq!(struct_storage_size(2, 2, 16), Some(16));

        // A struct with fields that is genuinely zero-sized, such as a
        // `PhantomData` newtype. rustc recorded an offset for the field, so the
        // size of zero is the answer rather than a missing one.
        assert_eq!(struct_storage_size(1, 1, 0), Some(0));

        // Field-less struct. `is_zst_type` takes these before the struct arm is
        // reached, but the predicate agrees with it.
        assert_eq!(struct_storage_size(0, 0, 0), Some(0));

        // `Ty::layout()` failed when the type was imported, which
        // `translator/types.rs` records as no offsets and a zero size.
        assert_eq!(struct_storage_size(2, 0, 0), None);

        // Same failure on a type whose size was recorded before the query failed.
        assert_eq!(struct_storage_size(1, 0, 8), None);
    }
}
