/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Relocation-aware `Allocation` decoding.

use super::coerce::{cast_struct_fields_to_expected_types, coerce_slice_data_pointee};
use super::const_bytes::{
    constant_storage_size, rust_array_type_info, rust_type_layout_size,
    translate_constant_value_from_bytes, translate_zero_sized_constant_value,
};
use super::const_enum::{read_uint_from_bytes, translate_enum_constant_from_alloc};
use super::const_union::translate_union_constant_from_alloc;
use super::static_global::{get_static_pointer_info, translate_static_global_pointer};
use super::statics::static_target_from_allocation_at;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{MirCastOp, MirConstantOp, MirConstructArrayOp, MirConstructStructOp};
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
use rustc_public::CrateDef;
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
use std::num::NonZeroUsize;

/// Detect zero-addend array→slice unsize: static `[T; N]` viewed as `[T]`.
///
/// Returns `(element_ty, N)` when the pointee is a slice of the same element
/// type as the static array. Other pointee mismatches stay unsupported.
/// `N` is an upper bound for validation only; the emitted slice length comes
/// from the constant's own fat-pointer metadata word, which is what makes
/// zero-addend prefix subslices (e.g. `split_at(2).0` over the static) carry
/// their true length instead of the whole array's.
///
/// The `static_elem == slice_elem` restriction is deliberate: a zero-addend
/// *flattening* view (e.g. `&NESTED[0]` over `[[f32; 2]; 3]` typed as
/// `&[f32]`) is valid Rust but stays a diagnosed support gap; accepting it
/// would need element-count arithmetic across the reinterpreted shape, not
/// just the stored metadata word.
pub(super) fn array_to_slice_unsize_info(
    static_ty: &rustc_public::ty::Ty,
    pointee_ty: &rustc_public::ty::Ty,
    loc: Location,
) -> TranslationResult<Option<(rustc_public::ty::Ty, u64)>> {
    use rustc_public::ty::{RigidTy, TyKind};

    match (static_ty.kind(), pointee_ty.kind()) {
        (
            TyKind::RigidTy(RigidTy::Array(static_elem, len_const)),
            TyKind::RigidTy(RigidTy::Slice(slice_elem)),
        ) if static_elem == slice_elem => {
            let len = len_const.eval_target_usize().map_err(|error| {
                input_error!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Failed to evaluate array length for static slice unsize: {error:?}"
                    ))
                )
            })?;
            Ok(Some((static_elem, len)))
        }
        _ => Ok(None),
    }
}

/// Detect an interior array→slice unsize within a device static.
///
/// The walk is intentionally limited to arrays. At each nesting level, the
/// byte addend selects one array element; when that array's element type
/// matches the slice element type, the remaining element count is returned.
/// This supports both an offset into a flat `[T; N]` and a slice over an array
/// nested inside outer arrays, while keeping structs, tuples, enums, and other
/// DST reinterpretations outside this lowering path.
pub(super) fn interior_array_to_slice_unsize_info(
    static_ty: &rustc_public::ty::Ty,
    pointee_ty: &rustc_public::ty::Ty,
    byte_offset: u64,
    loc: Location,
) -> TranslationResult<Option<(rustc_public::ty::Ty, u64)>> {
    use rustc_public::ty::{RigidTy, Ty, TyKind};

    let TyKind::RigidTy(RigidTy::Slice(slice_elem)) = pointee_ty.kind() else {
        return Ok(None);
    };

    fn find_region(
        array_ty: Ty,
        slice_elem: Ty,
        byte_offset: u64,
        loc: &Location,
    ) -> TranslationResult<Option<u64>> {
        let TyKind::RigidTy(RigidTy::Array(array_elem, len_const)) = array_ty.kind() else {
            return Ok(None);
        };

        let len = len_const.eval_target_usize().map_err(|error| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to evaluate array length for interior static slice unsize: {error:?}"
                ))
            )
        })?;
        let elem_size = rust_type_layout_size(array_elem, loc.clone())? as u64;
        let array_size = elem_size.checked_mul(len).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Array byte size overflowed while resolving an interior static slice: \
                 {len} elements x {elem_size} bytes"
            )))
        })?;

        if byte_offset > array_size {
            return Ok(None);
        }

        if array_elem == slice_elem {
            // A non-zero byte addend cannot distinguish positions between
            // zero-sized elements. Keep that case outside this path rather
            // than manufacturing an arbitrary element index.
            if elem_size == 0 || !byte_offset.is_multiple_of(elem_size) {
                return Ok(None);
            }

            let start = byte_offset / elem_size;
            if start > len {
                return Ok(None);
            }
            return Ok(Some(len - start));
        }

        // Descend only through the concrete outer element containing the
        // addend. An offset one-past this array has no child array region to
        // inspect, even though it may be valid for an empty slice at this
        // array's own element type (handled by the matching arm above).
        if elem_size == 0 || byte_offset >= array_size {
            return Ok(None);
        }

        let element_index = byte_offset / elem_size;
        if element_index >= len {
            return Ok(None);
        }

        find_region(array_elem, slice_elem, byte_offset % elem_size, loc)
    }

    Ok(find_region(*static_ty, slice_elem, byte_offset, &loc)?
        .map(|remaining_len| (slice_elem, remaining_len)))
}

/// Validate the relocation shape of one slice fat pointer stored inside an
/// allocation.
///
/// A slice occupies two pointer-width words. Exactly one relocation must back
/// the data word at `fat_ptr_offset`; the metadata word is a literal `usize`
/// and therefore must not overlap another relocation. Sibling relocations that
/// end exactly at the field start or begin exactly at the field end are fine.
///
/// Returns `(metadata_offset, fat_pointer_end)` on success. Generic over the
/// provenance payload so the boundary rules are unit testable without a rustc
/// session.
fn validate_slice_relocation_shape<P>(
    ptrs: &[(usize, P)],
    fat_ptr_offset: usize,
    pointer_width: usize,
) -> Result<(usize, usize), String> {
    if pointer_width == 0 {
        return Err("slice fat pointer has zero-width target pointers".to_string());
    }

    let metadata_offset = fat_ptr_offset
        .checked_add(pointer_width)
        .ok_or_else(|| "slice fat-pointer data-word end overflowed".to_string())?;
    let fat_pointer_end = metadata_offset
        .checked_add(pointer_width)
        .ok_or_else(|| "slice fat-pointer metadata-word end overflowed".to_string())?;

    let anchored = ptrs
        .iter()
        .filter(|(offset, _)| *offset == fat_ptr_offset)
        .count();
    if anchored != 1 {
        return Err(format!(
            "Slice fat pointer at byte {fat_ptr_offset} has {anchored} provenance entries at its \
             data-word start; expected exactly one"
        ));
    }

    let overlapping =
        relocation_offsets_overlapping_range(ptrs, fat_ptr_offset, fat_pointer_end, pointer_width);
    if let Some(other_offset) = overlapping
        .into_iter()
        .find(|offset| *offset != fat_ptr_offset)
    {
        return Err(format!(
            "Slice fat pointer at byte {fat_ptr_offset} has an additional relocation at byte \
             {other_offset}; the metadata word must remain literal usize bytes"
        ));
    }

    Ok((metadata_offset, fat_pointer_end))
}

/// Read the slice length from a fat-pointer image stored at an arbitrary
/// allocation offset.
fn slice_len_from_alloc_at(
    alloc: &rustc_public::ty::Allocation,
    fat_ptr_offset: usize,
    loc: Location,
) -> TranslationResult<u64> {
    let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let (metadata_offset, fat_pointer_end) =
        validate_slice_relocation_shape(&alloc.provenance.ptrs, fat_ptr_offset, pointer_width)
            .map_err(|message| input_error!(loc.clone(), TranslationErr::unsupported(message)))?;

    if fat_pointer_end > alloc.bytes.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Slice fat pointer at byte {fat_ptr_offset} needs bytes through \
                 {fat_pointer_end}, but the allocation is only {} bytes",
                alloc.bytes.len()
            ))
        );
    }

    alloc
        .read_partial_uint(metadata_offset..fat_pointer_end)
        .map(|len| len as u64)
        .map_err(|error| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to read slice length metadata at byte {metadata_offset}: {error:?}"
            )))
        })
}

/// Read the slice length from a standalone fat-pointer constant's metadata
/// word.
///
/// A `&[T]` / `*const [T]` constant is a two-word image: the data word (which
/// carries the provenance to the static, read by `static_target_from_constant`)
/// followed by the `usize` length. The length word is the source of truth for
/// the emitted slice: a const like `split_at(2).0` over a `[f32; 4]` static is
/// a zero-addend pointer whose stored length is 2, not the array's 4.
pub(super) fn slice_len_from_constant(
    constant: &mir::ConstOperand,
    loc: Location,
) -> TranslationResult<u64> {
    let ConstantKind::Allocated(alloc) = constant.const_.kind() else {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "static slice unsize constant is not an allocated constant".to_string()
            )
        );
    };

    let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let expected_size = pointer_width.checked_mul(2).ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(
            "static slice unsize constant pointer width overflowed".to_string()
        ))
    })?;
    if alloc.bytes.len() != expected_size {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "static slice unsize constant has {} bytes; expected exactly two \
                 pointer-width words ({expected_size} bytes)",
                alloc.bytes.len()
            ))
        );
    }

    slice_len_from_alloc_at(alloc, 0, loc)
}

/// Materialize a region of a device static as a fat `&[T]` / `*const [T]`.
///
/// A zero addend preserves the established whole-array path. A non-zero
/// addend reuses the byte-addressed static-pointer lowering to produce the
/// interior `*T` data pointer before pairing it with the length stored in the
/// constant's metadata word.
pub(super) fn translate_static_array_as_slice(
    ctx: &mut Context,
    static_def: &rustc_public::mir::mono::StaticDef,
    elem_ty: rustc_public::ty::Ty,
    len: u64,
    origin: facts::PointerOrigin,
    byte_offset: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirConstructSliceOp;

    let elem_mir_ty = types::translate_type(ctx, &elem_ty)?;
    let is_mutable = origin.is_mutable();

    let (data_ptr, prev_after_data) = if byte_offset == 0 {
        let static_ty = static_def.ty();
        let array_mir_ty = types::translate_type(ctx, &static_ty)?;

        // Thin pointer to the full array object (exact Rust `&[T; N]` shape).
        let thin_array_ptr_ty: TypeHandle =
            facts::mint_generic_ptr_type(ctx, array_mir_ty, origin).into();

        let (array_ptr, prev_after_array) = translate_static_global_pointer(
            ctx,
            static_def,
            array_mir_ty,
            thin_array_ptr_ty,
            is_mutable,
            0,
            block_ptr,
            prev_op,
            loc.clone(),
        )?;

        // Fat-pointer data slot is a generic `*T` / `*mut T`.
        coerce_slice_data_pointee(
            ctx,
            array_ptr,
            elem_mir_ty,
            is_mutable,
            block_ptr,
            prev_after_array,
            loc.clone(),
        )
    } else {
        let data_ptr_ty: TypeHandle = facts::mint_generic_ptr_type(ctx, elem_mir_ty, origin).into();

        translate_static_global_pointer(
            ctx,
            static_def,
            elem_mir_ty,
            data_ptr_ty,
            is_mutable,
            byte_offset,
            block_ptr,
            prev_op,
            loc.clone(),
        )?
    };

    let usize_ty = types::get_usize_type(ctx);
    let len_attr = pliron::builtin::attributes::IntegerAttr::new(
        usize_ty,
        APInt::from_u64(len, NonZeroUsize::new(64).unwrap()),
    );
    let len_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![],
        vec![],
        0,
    );
    len_op.deref_mut(ctx).set_loc(loc.clone());
    MirConstantOp::new(len_op).set_attr_value(ctx, len_attr);
    match prev_after_data {
        Some(prev) => len_op.insert_after(ctx, prev),
        None => len_op.insert_at_front(block_ptr, ctx),
    };
    let len_val = len_op.deref(ctx).get_result(0);

    let slice_ty = facts::mint_slice_type(ctx, elem_mir_ty, origin);
    let construct = Operation::new(
        ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![slice_ty.into()],
        vec![data_ptr, len_val],
        vec![],
        0,
    );
    construct.deref_mut(ctx).set_loc(loc);
    construct.insert_after(ctx, len_op);

    Ok((construct.deref(ctx).get_result(0), Some(construct)))
}

/// Return relocation starts whose pointer-width storage overlaps
/// `range_start..range_end`.
///
/// Unlike a simple "starts in range" check, this catches a relocation that
/// begins before the enum tag carrier but extends into it.
pub(super) fn relocation_offsets_overlapping_range<P>(
    ptrs: &[(usize, P)],
    range_start: usize,
    range_end: usize,
    pointer_width: usize,
) -> Vec<usize> {
    ptrs.iter()
        .map(|(pos, _)| *pos)
        .filter(|pos| {
            let relocation_end = pos.saturating_add(pointer_width);
            *pos < range_end && relocation_end > range_start
        })
        .collect()
}

/// Match the provenance entries of a thin-pointer field spanning
/// `pointer_offset..field_end`.
///
/// Returns the payload of the single relocation anchored at the field's base
/// offset, or `None` when no entry starts inside the field (null or exposed
/// provenance bytes). More than one entry anchored at the base, or an entry
/// starting strictly inside the field (fat or multi-word pointer bits), is a
/// hard error. Generic over the payload so the matching rules are unit
/// testable without a rustc session (`Prov` wraps an unconstructible
/// `AllocId`).
fn match_thin_pointer_relocation<P: Copy>(
    ptrs: &[(usize, P)],
    pointer_offset: usize,
    field_end: usize,
) -> Result<Option<P>, String> {
    let matches: Vec<P> = ptrs
        .iter()
        .filter(|(pos, _)| *pos == pointer_offset)
        .map(|&(_, prov)| prov)
        .collect();
    if matches.len() > 1 {
        return Err(format!(
            "Thin pointer field at offset {pointer_offset} has {} provenance entries; \
             expected at most one",
            matches.len()
        ));
    }

    // A thin pointer occupies one pointer-sized word; reject any additional
    // provenance that lands inside this field's byte range.
    if let Some(&(interior_pos, _)) = ptrs
        .iter()
        .find(|(pos, _)| *pos > pointer_offset && *pos < field_end)
    {
        return Err(format!(
            "Pointer field at offset {pointer_offset} has interior provenance at byte \
             {interior_pos}; fat or multi-word pointer provenance in aggregate constants \
             is not supported"
        ));
    }

    Ok(matches.first().copied())
}

/// Decode the byte addend stored under a thin-pointer relocation at
/// `pointer_offset..pointer_offset + ptr_width`.
///
/// The bytes under a relocation encode the offset into the target allocation
/// and are always initialized by rustc, so an uninitialized byte is a hard
/// error rather than a zero. Endianness is a parameter so the decoding is
/// unit testable without a rustc session.
fn decode_relocation_addend(
    bytes: &[Option<u8>],
    pointer_offset: usize,
    ptr_width: usize,
    endianness: rustc_public::target::Endian,
) -> Result<u128, String> {
    if ptr_width > 16 {
        return Err(format!(
            "relocation addend width {ptr_width} exceeds the 16-byte decode limit"
        ));
    }
    let field_end = pointer_offset
        .checked_add(ptr_width)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| {
            format!(
                "relocation addend at offset {pointer_offset} needs {ptr_width} bytes, \
                 but the allocation holds {}",
                bytes.len()
            )
        })?;
    let raw = bytes[pointer_offset..field_end]
        .iter()
        .copied()
        .collect::<Option<Vec<u8>>>()
        .ok_or_else(|| {
            format!("relocation addend at offset {pointer_offset} contains uninitialized bytes")
        })?;
    Ok(match endianness {
        rustc_public::target::Endian::Little => {
            raw.iter().enumerate().fold(0u128, |acc, (idx, byte)| {
                acc | ((*byte as u128) << (idx * 8))
            })
        }
        rustc_public::target::Endian::Big => raw
            .iter()
            .fold(0u128, |acc, byte| (acc << 8) | (*byte as u128)),
    })
}

/// Materialize a thin pointer field from an aggregate constant's allocation.
///
/// Aggregate **const** values with thin pointers to device statics are
/// materialized via [`MirGlobalAllocOp`] per field (addend taken from the
/// relocation's stored bytes). This does **not** mean device-static
/// *initializers* that themselves contain pointer relocations are supported —
/// those remain rejected by [`allocation_initializer_data`].
///
/// When the field has no provenance entry, falls back to the existing
/// inttoptr-of-bytes path (null / exposed provenance).
pub(super) fn translate_thin_pointer_at_alloc_offset(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    pointer_offset: usize,
    result_ptr_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use rustc_public::mir::alloc::GlobalAlloc;

    let ptr_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let field_end = pointer_offset.checked_add(ptr_width).ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Thin pointer field offset {pointer_offset} + width {ptr_width} overflowed"
            ))
        )
    })?;
    if field_end > alloc.bytes.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Thin pointer field at offset {pointer_offset} needs {ptr_width} bytes, \
                 but allocation is only {} bytes",
                alloc.bytes.len()
            ))
        );
    }

    let relocation =
        match_thin_pointer_relocation(&alloc.provenance.ptrs, pointer_offset, field_end)
            .map_err(|message| input_error!(loc.clone(), TranslationErr::unsupported(message)))?;

    if let Some(prov) = relocation {
        let addend = decode_relocation_addend(
            &alloc.bytes,
            pointer_offset,
            ptr_width,
            rustc_public::target::MachineInfo::target_endianness(),
        )
        .map_err(|message| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to read thin-pointer addend at offset {pointer_offset}: {message}"
                ))
            )
        })?;

        let (pointee_ty, is_mutable) = {
            let ty_ref = result_ptr_ty.deref(ctx);
            let ptr_ty = ty_ref
                .downcast_ref::<dialect_mir::types::MirPtrType>()
                .ok_or_else(|| {
                    input_error_noloc!(TranslationErr::unsupported(
                        "translate_thin_pointer_at_alloc_offset: expected MirPtrType"
                    ))
                })?;
            (ptr_ty.pointee, ptr_ty.is_mutable)
        };

        match GlobalAlloc::from(prov.0) {
            GlobalAlloc::Static(static_def) => {
                let byte_offset = u64::try_from(addend).map_err(|_| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "Device-static pointer addend {addend} does not fit u64"
                    )))
                })?;
                return translate_static_global_pointer(
                    ctx,
                    &static_def,
                    pointee_ty,
                    result_ptr_ty,
                    is_mutable,
                    byte_offset,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }
            GlobalAlloc::Memory(_) => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Aggregate constant thin pointer at offset {pointer_offset} points at \
                        an anonymous promoted allocation; promoted aggregate pointer constants \
                        are not yet supported"
                    ))
                );
            }
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Aggregate constant thin pointer at offset {pointer_offset} points at \
                         unsupported allocation kind: {other:?}"
                    ))
                );
            }
        }
    }

    // No provenance: keep the existing inttoptr-of-bytes behavior (null / exposed).
    let field_bytes: Vec<u8> = alloc.bytes[pointer_offset..field_end]
        .iter()
        .map(|opt| opt.unwrap_or(0))
        .collect();
    let ptr_val = read_uint_from_bytes(&field_bytes) as u64;
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let apint = APInt::from_u64(ptr_val, NonZeroUsize::new(64).unwrap());
    let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);

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
        vec![result_ptr_ty],
        vec![const_value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
    if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, result_ptr_ty) {
        cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
    }
    cast_op.insert_after(ctx, const_op.get_operation());
    Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
}

/// Materialize a slice fat-pointer field from an aggregate constant allocation.
///
/// The data word keeps rustc provenance and may carry a non-zero byte addend
/// into a device static. The metadata word is decoded independently as the
/// stored slice length. This deliberately supports only same-element
/// array-to-slice views over Rust statics, matching the standalone constant
/// path; anonymous promoted allocations and other DST metadata remain
/// fail-closed.
#[allow(clippy::too_many_arguments)]
fn translate_slice_at_alloc_offset(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    fat_ptr_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let Some((pointee_ty, origin)) = get_static_pointer_info(rust_ty) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Aggregate slice constant at byte {fat_ptr_offset} has unexpected Rust type \
                 {rust_ty:?}; expected a reference or raw pointer to a slice"
            ))
        );
    };

    let len = slice_len_from_alloc_at(alloc, fat_ptr_offset, loc.clone())?;
    let Some(static_target) = static_target_from_allocation_at(alloc, fat_ptr_offset)? else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Aggregate slice constant at byte {fat_ptr_offset} points at an anonymous or \
                 unsupported allocation; slice provenance currently requires a Rust device static"
            ))
        );
    };

    let static_ty = static_target.static_def.ty();
    let slice_region = if static_target.byte_offset == 0 {
        array_to_slice_unsize_info(&static_ty, &pointee_ty, loc.clone())?
    } else {
        interior_array_to_slice_unsize_info(
            &static_ty,
            &pointee_ty,
            static_target.byte_offset,
            loc.clone(),
        )?
    };

    let Some((elem_ty, available_len)) = slice_region else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Aggregate slice constant at byte {fat_ptr_offset} points into device static {} \
                 at byte addend {}, but its pointee type {pointee_ty:?} is not a supported \
                 same-element array-to-slice view of static type {static_ty:?}",
                static_target.static_def.name(),
                static_target.byte_offset
            ))
        );
    };

    if len > available_len {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Aggregate slice constant at byte {fat_ptr_offset} stores length {len}, which \
                 exceeds the selected region's remaining length {available_len} in device \
                 static {}",
                static_target.static_def.name()
            ))
        );
    }

    translate_static_array_as_slice(
        ctx,
        &static_target.static_def,
        elem_ty,
        len,
        origin,
        static_target.byte_offset,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Slice `size` bytes from `alloc` at `offset`, treating uninit as zero.
pub(super) fn alloc_slice_bytes_zeroing_uninit(
    alloc: &rustc_public::ty::Allocation,
    offset: usize,
    size: usize,
    what: &str,
    loc: &Location,
) -> TranslationResult<Vec<u8>> {
    let end = offset.checked_add(size).ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what}: offset {offset} + size {size} overflowed"
            ))
        )
    })?;
    if end > alloc.bytes.len() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what}: need [{offset}..{end}), but allocation is only {} bytes",
                alloc.bytes.len()
            ))
        );
    }
    Ok(alloc.bytes[offset..end]
        .iter()
        .map(|opt| opt.unwrap_or(0))
        .collect())
}

/// Whether any provenance entry starts inside `offset..offset + size`.
/// Generic over the payload so the predicate is unit testable without a
/// rustc session.
fn provenance_starts_in_range<P>(ptrs: &[(usize, P)], offset: usize, size: usize) -> bool {
    let end = offset.saturating_add(size);
    ptrs.iter().any(|(pos, _)| *pos >= offset && *pos < end)
}

fn alloc_has_provenance_in_range(
    alloc: &rustc_public::ty::Allocation,
    offset: usize,
    size: usize,
) -> bool {
    provenance_starts_in_range(&alloc.provenance.ptrs, offset, size)
}

/// Return the start offset of the first provenance entry that lies inside
/// the aggregate's byte range but inside none of its fields.
///
/// Field translation consumes (thin pointer) or rejects (every other kind)
/// the relocations under the bytes it decodes, so a survivor here sits in
/// padding: no field would ever consume it, and dropping it would silently
/// strip a pointer from the constant. Generic over the payload so the audit
/// is unit testable without a rustc session.
fn find_unconsumed_relocation<P>(
    ptrs: &[(usize, P)],
    aggregate_start: usize,
    aggregate_size: usize,
    field_ranges: &[(usize, usize)],
) -> Option<usize> {
    let aggregate_end = aggregate_start.saturating_add(aggregate_size);
    ptrs.iter().map(|(pos, _)| *pos).find(|&pos| {
        pos >= aggregate_start
            && pos < aggregate_end
            && !field_ranges
                .iter()
                .any(|&(start, size)| pos >= start && pos < start.saturating_add(size))
    })
}

/// Fail-closed audit run after all of an aggregate's fields have been
/// translated: any relocation inside the aggregate's byte range that no
/// field consumed is a hard error.
pub(super) fn audit_aggregate_relocations(
    alloc: &rustc_public::ty::Allocation,
    aggregate_start: usize,
    aggregate_size: usize,
    field_ranges: &[(usize, usize)],
    what: &str,
    loc: &Location,
) -> TranslationResult<()> {
    if let Some(pos) = find_unconsumed_relocation(
        &alloc.provenance.ptrs,
        aggregate_start,
        aggregate_size,
        field_ranges,
    ) {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} constant has a pointer relocation at byte {pos} that no field \
                 consumes; provenance in padding bytes cannot be preserved"
            ))
        );
    }
    Ok(())
}

/// Byte width of a constant field: rustc layout when available, the dialect
/// type's storage size as a fallback. Shared by the scalar decode path and
/// the per-aggregate relocation audit so both see the same field extents.
fn constant_field_byte_size(
    ctx: &Context,
    rust_ty: &rustc_public::ty::Ty,
    ty_ptr: TypeHandle,
    loc: &Location,
) -> TranslationResult<usize> {
    rust_type_layout_size(*rust_ty, loc.clone()).or_else(|_| {
        constant_storage_size(ctx, ty_ptr).ok_or_else(|| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Cannot determine storage size for constant field of type {rust_ty:?}"
                ))
            )
        })
    })
}

/// Decode one typed value from an allocation at an absolute byte offset,
/// preserving supported pointer provenance. Thin pointer fields resolve through
/// [`translate_thin_pointer_at_alloc_offset`]; slice fields pair the relocated
/// data word with their literal length metadata.
pub(super) fn translate_constant_value_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    absolute_byte_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use rustc_public::ty::{RigidTy, TyKind};

    let is_ptr = ty_ptr.deref(ctx).is::<dialect_mir::types::MirPtrType>();
    if is_ptr {
        return translate_thin_pointer_at_alloc_offset(
            ctx,
            alloc,
            absolute_byte_offset,
            ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let is_slice = ty_ptr.deref(ctx).is::<dialect_mir::types::MirSliceType>();
    if is_slice {
        let size = rust_type_layout_size(*rust_ty, loc.clone())?;
        let field_end = absolute_byte_offset.checked_add(size).ok_or_else(|| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Slice field at byte {absolute_byte_offset} with size {size} overflowed"
                ))
            )
        })?;
        let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
        let overlaps = relocation_offsets_overlapping_range(
            &alloc.provenance.ptrs,
            absolute_byte_offset,
            field_end,
            pointer_width,
        );
        if !overlaps.is_empty() {
            return translate_slice_at_alloc_offset(
                ctx,
                alloc,
                absolute_byte_offset,
                rust_ty,
                block_ptr,
                prev_op,
                loc,
            );
        }
        let bytes = alloc_slice_bytes_zeroing_uninit(
            alloc,
            absolute_byte_offset,
            size,
            "Slice field",
            &loc,
        )?;
        return translate_constant_value_from_bytes(
            ctx, rust_ty, ty_ptr, &bytes, block_ptr, prev_op, loc,
        );
    }

    let is_tuple = ty_ptr.deref(ctx).is::<dialect_mir::types::MirTupleType>();
    if is_tuple {
        return translate_tuple_constant_from_alloc(
            ctx,
            alloc,
            absolute_byte_offset,
            rust_ty,
            ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let is_struct = ty_ptr.deref(ctx).is::<dialect_mir::types::MirStructType>();
    if is_struct {
        return translate_struct_constant_from_alloc(
            ctx,
            alloc,
            absolute_byte_offset,
            rust_ty,
            ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let is_union = ty_ptr.deref(ctx).is::<dialect_mir::types::MirUnionType>();
    if is_union {
        return translate_union_constant_from_alloc(
            ctx,
            alloc,
            absolute_byte_offset,
            rust_ty,
            ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let is_array = ty_ptr.deref(ctx).is::<dialect_mir::types::MirArrayType>();
    if is_array {
        return translate_array_constant_from_alloc(
            ctx,
            alloc,
            absolute_byte_offset,
            rust_ty,
            ty_ptr,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let is_enum = ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>();
    if is_enum {
        let size = rust_type_layout_size(*rust_ty, loc.clone())?;
        if alloc_has_provenance_in_range(alloc, absolute_byte_offset, size) {
            return translate_enum_constant_from_alloc(
                ctx,
                alloc,
                absolute_byte_offset,
                rust_ty,
                ty_ptr,
                block_ptr,
                prev_op,
                loc,
            );
        }
    }

    let size = constant_field_byte_size(ctx, rust_ty, ty_ptr, &loc)?;
    // Fail closed: the bytes under a relocation are an addend into the target
    // allocation, not literal data. Decoding them as a non-pointer value
    // would silently strip the pointer they represent.
    if alloc_has_provenance_in_range(alloc, absolute_byte_offset, size) {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Constant field of type {rust_ty:?} at byte offset {absolute_byte_offset} \
                 overlaps a pointer relocation; only supported pointer or slice fields can \
                 carry provenance in aggregate constants"
            ))
        );
    }
    // ZSTs: layout size 0.
    if size == 0 || types::is_zst_type(ctx, ty_ptr) {
        // Still need Rust ADT metadata for empty aggregates.
        if matches!(
            rust_ty.kind(),
            TyKind::RigidTy(RigidTy::Tuple(_)) | TyKind::RigidTy(RigidTy::Adt(_, _))
        ) {
            let bytes = alloc_slice_bytes_zeroing_uninit(
                alloc,
                absolute_byte_offset,
                size,
                "ZST field",
                &loc,
            )?;
            return translate_constant_value_from_bytes(
                ctx, rust_ty, ty_ptr, &bytes, block_ptr, prev_op, loc,
            );
        }
        return translate_zero_sized_constant_value(ctx, ty_ptr, block_ptr, prev_op, loc);
    }

    let bytes = alloc_slice_bytes_zeroing_uninit(
        alloc,
        absolute_byte_offset,
        size,
        "Constant field",
        &loc,
    )?;
    translate_constant_value_from_bytes(ctx, rust_ty, ty_ptr, &bytes, block_ptr, prev_op, loc)
}

pub(super) fn translate_tuple_constant_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let field_types: Vec<TypeHandle> = {
        let ty_ref = const_ty_ptr.deref(ctx);
        let tuple_ty = ty_ref
            .downcast_ref::<dialect_mir::types::MirTupleType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_tuple_constant_from_alloc called on non-tuple type"
                ))
            })?;
        tuple_ty.get_types().to_vec()
    };

    let rust_field_types = match rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(fields)) => {
            fields.to_vec()
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant expected Rust tuple type, got {other:?}"
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

    let field_offsets = crate::translator::layout::aggregate_field_offsets(rust_ty, "Tuple", &loc)?;
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

    let mut values = Vec::with_capacity(field_types.len());
    let mut field_ranges = Vec::with_capacity(field_types.len());
    let mut current_prev_op = prev_op;
    for (field_idx, (field_ty, rust_field_ty)) in field_types
        .iter()
        .copied()
        .zip(rust_field_types.iter())
        .enumerate()
    {
        let abs = base_offset
            .checked_add(field_offsets[field_idx])
            .ok_or_else(|| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Tuple constant field {field_idx} offset overflowed"
                    ))
                )
            })?;
        let field_size = constant_field_byte_size(ctx, rust_field_ty, field_ty, &loc)?;
        field_ranges.push((abs, field_size));
        let (value, new_prev_op) = translate_constant_value_from_alloc(
            ctx,
            alloc,
            abs,
            rust_field_ty,
            field_ty,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        values.push(value);
        current_prev_op = new_prev_op;
    }

    let tuple_size = rust_type_layout_size(*rust_ty, loc.clone())?;
    audit_aggregate_relocations(alloc, base_offset, tuple_size, &field_ranges, "Tuple", &loc)?;

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

pub(super) fn translate_struct_constant_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: TypeHandle,
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
                    "translate_struct_constant_from_alloc called on non-struct type"
                ))
            })?;
        struct_ty.field_types().to_vec()
    };

    // A zero-sized struct span holds no bytes and cannot carry relocations,
    // and not every type that lands here as a MirStructType is an ADT:
    // function items and non-capturing closures have no ADT metadata to
    // consult. Synthesize such values from the dialect type alone.
    let struct_size = rust_ty
        .layout()
        .map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to query layout for struct constant: {:?}",
                e
            )))
        })?
        .shape()
        .size
        .bytes();
    if struct_size == 0 {
        return translate_zero_sized_constant_value(ctx, const_ty_ptr, block_ptr, prev_op, loc);
    }

    let field_offsets =
        crate::translator::layout::aggregate_field_offsets(rust_ty, "Struct", &loc)?;
    if field_offsets.len() != field_types.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Struct constant layout has {} field offsets, type has {} fields",
                field_offsets.len(),
                field_types.len()
            ))
        );
    }

    let (adt_def, substs) = match rust_ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => (adt_def, substs),
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Expected ADT Rust type for struct constant, got {other:?}"
                ))
            );
        }
    };
    let variants = adt_def.variants();
    let struct_variant = variants.first().ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(
            "Struct ADT has no variants in metadata"
        ))
    })?;

    let mut field_values = Vec::with_capacity(field_types.len());
    let mut field_ranges = Vec::with_capacity(field_types.len());
    let mut current_prev_op = prev_op;
    for (field_idx, field_ty_ptr) in field_types.iter().copied().enumerate() {
        let fields = struct_variant.fields();
        let rust_field = fields.get(field_idx).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Struct constant field {field_idx} is missing in rustc ADT metadata ({} field(s) recorded)",
                fields.len()
            )))
        })?;
        let rust_field_ty = rust_field.ty_with_args(&substs);
        let abs = base_offset
            .checked_add(field_offsets[field_idx])
            .ok_or_else(|| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Struct constant field {field_idx} offset overflowed"
                    ))
                )
            })?;
        let field_size = constant_field_byte_size(ctx, &rust_field_ty, field_ty_ptr, &loc)?;
        field_ranges.push((abs, field_size));
        let (field_val, new_prev_op) = translate_constant_value_from_alloc(
            ctx,
            alloc,
            abs,
            &rust_field_ty,
            field_ty_ptr,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        field_values.push(field_val);
        current_prev_op = new_prev_op;
    }

    let struct_size = rust_type_layout_size(*rust_ty, loc.clone())?;
    audit_aggregate_relocations(
        alloc,
        base_offset,
        struct_size,
        &field_ranges,
        "Struct",
        &loc,
    )?;

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
    op.deref_mut(ctx).set_loc(loc);
    if let Some(prev) = prev_after_casts {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
    Ok((op.deref(ctx).get_result(0), Some(op)))
}

/// Element kinds admitted by a bare array value constant
/// (`translate_array_value_constant`).
///
/// Primitive scalars, enums, initialized unions, tuples, structs, and nested
/// arrays are supported at this entry point. Nested arrays are walked
/// recursively so an unsupported leaf cannot hide behind nesting. Struct
/// elements reuse the same layout-aware aggregate decoders used when structs
/// appear inside other constants. Arrays inside struct or tuple constants are
/// dispatched through [`translate_constant_value_from_alloc`] and are not
/// governed by this gate.
pub(super) fn validate_array_value_element_type(
    ctx: &Context,
    element_ty: TypeHandle,
    loc: &Location,
) -> TranslationResult<()> {
    let nested_element_ty = {
        let elem_obj = element_ty.deref(ctx);
        if elem_obj.is::<IntegerType>()
            || elem_obj.is::<MirFP16Type>()
            || elem_obj.is::<FP32Type>()
            || elem_obj.is::<FP64Type>()
            || elem_obj.is::<dialect_mir::types::MirTupleType>()
            || elem_obj.is::<dialect_mir::types::MirStructType>()
            || elem_obj.is::<dialect_mir::types::MirEnumType>()
            || elem_obj.is::<dialect_mir::types::MirUnionType>()
        {
            return Ok(());
        }
        let Some(array_ty) = elem_obj.downcast_ref::<dialect_mir::types::MirArrayType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Array constant element type is not supported: {:?}. Supported array \
                     constants are primitive scalars, enums, initialized unions, tuples, \
                     structs, or nested arrays of those.",
                    elem_obj
                ))
            );
        };
        array_ty.element_type()
    };
    validate_array_value_element_type(ctx, nested_element_ty, loc)
}

pub(super) fn translate_array_constant_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    rust_array_ty: &rustc_public::ty::Ty,
    array_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let (element_ty_ptr, element_count) = {
        let arr_ty_obj = array_ty.deref(ctx);
        let arr_ty = arr_ty_obj
            .downcast_ref::<dialect_mir::types::MirArrayType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_array_constant_from_alloc: expected array type"
                ))
            })?;
        (arr_ty.element_type(), arr_ty.size())
    };

    let (rust_element_ty, rust_element_count) = rust_array_type_info(*rust_array_ty, loc.clone())?;
    if rust_element_count != element_count {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Array constant length mismatch: Rust type has {rust_element_count} elements, \
                 dialect type has {element_count}"
            ))
        );
    }
    let element_byte_size = rust_type_layout_size(rust_element_ty, loc.clone())?;
    let element_count_usize = usize::try_from(element_count).map_err(|_| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Array constant element count {element_count} does not fit usize"
        )))
    })?;

    let mut element_values = Vec::with_capacity(element_count_usize);
    let mut element_ranges = Vec::with_capacity(element_count_usize);
    let mut last_op = prev_op;
    for i in 0..element_count_usize {
        let abs = base_offset
            .checked_add(i.checked_mul(element_byte_size).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Array constant element {i} stride overflowed"
                )))
            })?)
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Array constant element {i} absolute offset overflowed"
                )))
            })?;
        element_ranges.push((abs, element_byte_size));
        let (elem_val, elem_last_op) = translate_constant_value_from_alloc(
            ctx,
            alloc,
            abs,
            &rust_element_ty,
            element_ty_ptr,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        element_values.push(elem_val);
        last_op = elem_last_op;
    }

    let array_size = rust_type_layout_size(*rust_array_ty, loc.clone())?;
    audit_aggregate_relocations(
        alloc,
        base_offset,
        array_size,
        &element_ranges,
        "Array",
        &loc,
    )?;

    let op = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty],
        element_values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
    Ok((op.deref(ctx).get_result(0), Some(op)))
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod aggregate_relocation_tests {
    use super::super::const_bytes::constant_type_contains_pointer;
    use super::super::const_union::{
        UnionConstantStorageKind, UnionConstantUse, classify_union_constant_storage,
        relocation_free_pointer_integer_union_storage_field,
        translate_pointer_integer_union_constant_from_storage,
        validate_device_static_union_storage,
    };
    use super::{
        decode_relocation_addend, find_unconsumed_relocation, match_thin_pointer_relocation,
        provenance_starts_in_range, relocation_offsets_overlapping_range,
        validate_array_value_element_type, validate_slice_relocation_shape,
    };
    use dialect_mir::ops::{MirCastOp, MirInsertFieldOp};
    use dialect_mir::types::{
        EnumVariant, MirArrayType, MirEnumType, MirPointerKind, MirPtrType, MirStructType,
        MirTupleType, MirUnionType, pointer_carriers_in_type,
    };
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::location::Location;
    use pliron::op::Op;
    use pliron::operation::Operation;
    use pliron::r#type::TypeHandle;
    use pliron::r#type::Typed;
    use rustc_public::target::Endian;

    #[test]
    fn relocation_overlap_detects_exact_and_left_crossing_pointer_words() {
        let ptrs = [(0usize, ()), (8, ()), (24, ())];
        assert_eq!(
            relocation_offsets_overlapping_range(&ptrs, 8, 16, 8),
            vec![8],
            "an exact full-width relocation covers the carrier"
        );
        assert_eq!(
            relocation_offsets_overlapping_range(&ptrs, 4, 12, 8),
            vec![0, 8],
            "overlap detection must include relocations starting before the carrier"
        );
        assert!(
            relocation_offsets_overlapping_range(&ptrs, 16, 24, 8).is_empty(),
            "touching a range boundary is not an overlap"
        );
    }

    #[test]
    fn aggregate_slice_relocation_accepts_nonzero_field_offset_with_siblings() {
        let ptrs = [(0usize, ()), (8, ()), (24, ())];
        assert_eq!(
            validate_slice_relocation_shape(&ptrs, 8, 8),
            Ok((16, 24)),
            "sibling relocations outside the fat-pointer bytes must not interfere"
        );
    }

    #[test]
    fn aggregate_slice_relocation_rejects_metadata_provenance() {
        let ptrs = [(8usize, ()), (16, ())];
        let error = validate_slice_relocation_shape(&ptrs, 8, 8)
            .expect_err("the metadata word must remain literal usize bytes");
        assert!(
            error.contains("additional relocation at byte 16"),
            "diagnostic must identify metadata provenance: {error}"
        );
    }

    #[test]
    fn aggregate_slice_relocation_requires_data_word_provenance_at_field_start() {
        let ptrs = [(12usize, ())];
        let error = validate_slice_relocation_shape(&ptrs, 8, 8)
            .expect_err("interior provenance cannot stand in for the slice data word");
        assert!(
            error.contains("0 provenance entries at its data-word start"),
            "diagnostic must require an anchored data relocation: {error}"
        );

        let duplicate = [(8usize, 1u8), (8, 2u8)];
        let error = validate_slice_relocation_shape(&duplicate, 8, 8)
            .expect_err("two data-word provenance entries are ambiguous");
        assert!(
            error.contains("2 provenance entries at its data-word start"),
            "diagnostic must count duplicate anchored relocations: {error}"
        );
    }

    #[test]
    fn aggregate_slice_relocation_rejects_left_crossing_pointer_storage() {
        let ptrs = [(4usize, ()), (8, ())];
        let error = validate_slice_relocation_shape(&ptrs, 8, 8)
            .expect_err("a relocation from the preceding bytes must not overlap the slice");
        assert!(
            error.contains("additional relocation at byte 4"),
            "diagnostic must identify the crossing relocation: {error}"
        );
    }

    #[test]
    fn relocation_matching_is_anchored_to_the_field_base() {
        let ptrs = [(0usize, 1u32), (16, 2)];
        assert_eq!(
            match_thin_pointer_relocation(&ptrs, 16, 24),
            Ok(Some(2)),
            "the entry at the field base must be matched"
        );
        assert_eq!(
            match_thin_pointer_relocation(&ptrs, 8, 16),
            Ok(None),
            "entries outside the field belong to sibling fields, not this one"
        );
    }

    #[test]
    fn relocation_matching_rejects_duplicate_entries_at_the_base() {
        let ptrs = [(8usize, 1u32), (8, 2)];
        let error = match_thin_pointer_relocation(&ptrs, 8, 16)
            .expect_err("two provenance entries at one offset must fail closed");
        assert!(
            error.contains("2 provenance entries"),
            "diagnostic must count the entries: {error}"
        );
    }

    #[test]
    fn relocation_matching_rejects_interior_provenance() {
        let ptrs = [(12usize, 7u32)];
        let error = match_thin_pointer_relocation(&ptrs, 8, 16)
            .expect_err("provenance strictly inside a thin field is fat-pointer bits");
        assert!(
            error.contains("interior provenance at byte 12"),
            "diagnostic must name the interior byte: {error}"
        );
    }

    #[test]
    fn relocation_addend_decodes_with_the_given_endianness() {
        let mut bytes = vec![Some(0u8); 16];
        bytes[8] = Some(0x28);
        assert_eq!(
            decode_relocation_addend(&bytes, 8, 8, Endian::Little),
            Ok(0x28),
            "little-endian addend must read the low byte first"
        );
        assert_eq!(
            decode_relocation_addend(&bytes, 8, 8, Endian::Big),
            Ok(0x28u128 << 56),
            "big-endian addend must read the high byte first"
        );
    }

    #[test]
    fn relocation_addend_rejects_uninitialized_and_out_of_bounds_bytes() {
        let mut bytes = vec![Some(0u8); 16];
        bytes[10] = None;
        let error = decode_relocation_addend(&bytes, 8, 8, Endian::Little)
            .expect_err("addend bytes under a relocation are always initialized");
        assert!(
            error.contains("uninitialized"),
            "diagnostic must name the failure: {error}"
        );

        let error = decode_relocation_addend(&bytes, 12, 8, Endian::Little)
            .expect_err("an addend past the allocation end must fail closed");
        assert!(
            error.contains("needs 8 bytes"),
            "diagnostic must name the missing width: {error}"
        );
    }

    #[test]
    fn non_pointer_fields_detect_overlapping_relocations() {
        let ptrs = [(4usize, ())];
        assert!(
            provenance_starts_in_range(&ptrs, 4, 4),
            "a relocation at the field base overlaps the field"
        );
        assert!(
            provenance_starts_in_range(&ptrs, 0, 8),
            "a relocation inside the field range overlaps the field"
        );
        assert!(
            !provenance_starts_in_range(&ptrs, 8, 8),
            "a relocation before the field does not start inside it"
        );
        assert!(
            !provenance_starts_in_range(&ptrs, 4, 0),
            "a zero-sized field cannot overlap any relocation"
        );
    }

    #[test]
    fn unconsumed_relocation_audit_flags_padding_only() {
        let padding_relocation = [(12usize, ())];
        assert_eq!(
            find_unconsumed_relocation(&padding_relocation, 0, 16, &[(0, 8), (8, 4)]),
            Some(12),
            "a relocation in padding is consumed by no field and must be reported"
        );
        assert_eq!(
            find_unconsumed_relocation(&padding_relocation, 0, 16, &[(0, 8), (8, 8)]),
            None,
            "a relocation covered by a field is that field's responsibility"
        );
        assert_eq!(
            find_unconsumed_relocation(&padding_relocation, 16, 16, &[(16, 8)]),
            None,
            "relocations outside the aggregate's range belong to its siblings"
        );
        assert_eq!(
            find_unconsumed_relocation(&padding_relocation, 0, 16, &[(0, 12), (12, 0)]),
            Some(12),
            "a zero-sized field consumes nothing"
        );
    }

    #[test]
    fn bare_array_elements_follow_the_documented_contract() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        assert!(
            validate_array_value_element_type(&ctx, u32_ty, &Location::Unknown).is_ok(),
            "primitive scalar elements remain supported"
        );

        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![u32_ty]).into();
        assert!(
            validate_array_value_element_type(&ctx, tuple_ty, &Location::Unknown).is_ok(),
            "tuple elements remain supported"
        );
        let nested_array_ty: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 4).into();
        assert!(
            validate_array_value_element_type(&ctx, nested_array_ty, &Location::Unknown).is_ok(),
            "nested array elements remain supported"
        );

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Side".into(),
            u8_ty,
            vec![2, 5],
            vec![
                EnumVariant::unit("Low".into()),
                EnumVariant::unit("High".into()),
            ],
            0,
            1,
            1,
        )
        .into();
        assert!(
            validate_array_value_element_type(&ctx, enum_ty, &Location::Unknown).is_ok(),
            "bare enum-array elements are supported"
        );
        let nested_enum_array: TypeHandle = MirArrayType::get(&mut ctx, enum_ty, 2).into();
        assert!(
            validate_array_value_element_type(&ctx, nested_enum_array, &Location::Unknown).is_ok(),
            "nesting preserves supported enum leaves"
        );

        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 4).into();
        let union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "Bits".into(),
            vec!["word".into(), "bytes".into()],
            vec![u32_ty, bytes_ty],
            4,
            4,
        )
        .into();
        assert!(
            validate_array_value_element_type(&ctx, union_ty, &Location::Unknown).is_ok(),
            "initialized union elements are supported"
        );
        let nested_union_array: TypeHandle = MirArrayType::get(&mut ctx, union_ty, 2).into();
        assert!(
            validate_array_value_element_type(&ctx, nested_union_array, &Location::Unknown).is_ok(),
            "nesting preserves supported union leaves"
        );
        assert!(
            !constant_type_contains_pointer(&ctx, union_ty),
            "the byte-materialized Bits shape is pointer-free"
        );

        let pointer_field_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let u8_pointer_field_ty: TypeHandle =
            MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let pointer_only_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "PointerOnly".into(),
            vec!["word".into(), "bytes".into()],
            vec![pointer_field_ty, u8_pointer_field_ty],
            8,
            8,
        )
        .into();
        assert!(
            constant_type_contains_pointer(&ctx, pointer_only_union_ty),
            "pointer-only union constants must use typed reconstruction"
        );
        assert_eq!(
            classify_union_constant_storage(
                &ctx,
                pointer_only_union_ty,
                UnionConstantUse::SsaReconstruction
            ),
            Ok(UnionConstantStorageKind::ThinPointer {
                field_index: 0,
                field_ty: pointer_field_ty,
            }),
            "compatible thin-pointer alternatives can share one provenance-bearing carrier"
        );

        assert!(
            validate_array_value_element_type(&ctx, pointer_only_union_ty, &Location::Unknown)
                .is_ok(),
            "bare arrays of pointer-only unions must reach the provenance-aware union decoder"
        );

        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let raw_const_pointer_field_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::RawConst)
                .into();
        let pointer_integer_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "PointerBits".into(),
            vec!["ptr".into(), "bits".into()],
            vec![raw_const_pointer_field_ty, u64_ty],
            8,
            8,
        )
        .into();
        let pointer_integer_error = classify_union_constant_storage(
            &ctx,
            pointer_integer_union_ty,
            UnionConstantUse::SsaReconstruction,
        )
        .expect_err("the provenance-aware classifier must keep pointer/integer overlap closed");
        assert!(
            pointer_integer_error.contains("pointer/integer union constants"),
            "diagnostic must explain the provenance-vs-bits conflict: {pointer_integer_error}"
        );
        assert_eq!(
            relocation_free_pointer_integer_union_storage_field(&ctx, pointer_integer_union_ty, 8),
            Some((1, u64_ty)),
            "a pointer-word ptr/u64 union is byte-image eligible when the allocation has no relocation"
        );
        assert_eq!(
            relocation_free_pointer_integer_union_storage_field(&ctx, pointer_integer_union_ty, 4),
            None,
            "the exception must not cross a target pointer-width mismatch"
        );

        let u32_pointer_integer_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "PointerNarrowBits".into(),
            vec!["ptr".into(), "bits".into()],
            vec![raw_const_pointer_field_ty, u32_ty],
            8,
            8,
        )
        .into();
        assert_eq!(
            relocation_free_pointer_integer_union_storage_field(
                &ctx,
                u32_pointer_integer_union_ty,
                8
            ),
            None,
            "partial-width integer alternatives stay outside the initial pointer-word exception"
        );

        let slice_ty: TypeHandle = dialect_mir::types::MirSliceType::get(&mut ctx, u32_ty).into();
        let fat_pointer_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "FatPointer".into(),
            vec!["slice".into(), "ptr".into()],
            vec![slice_ty, pointer_field_ty],
            16,
            8,
        )
        .into();
        let fat_pointer_error = classify_union_constant_storage(
            &ctx,
            fat_pointer_union_ty,
            UnionConstantUse::SsaReconstruction,
        )
        .expect_err("fat-pointer union storage must remain fail-closed");
        assert!(
            fat_pointer_error.contains("not a thin pointer"),
            "diagnostic must identify unsupported fat/nested storage: {fat_pointer_error}"
        );

        let struct_ty: TypeHandle = MirStructType::get(
            &mut ctx,
            "ArrayValueElement".into(),
            vec!["value".into()],
            vec![u32_ty],
        )
        .into();
        assert!(
            validate_array_value_element_type(&ctx, struct_ty, &Location::Unknown).is_ok(),
            "bare struct arrays are materialized by the layout-aware aggregate decoder"
        );

        let struct_array_ty: TypeHandle = MirArrayType::get(&mut ctx, struct_ty, 2).into();
        assert!(
            validate_array_value_element_type(&ctx, struct_array_ty, &Location::Unknown).is_ok(),
            "nesting preserves supported struct leaves"
        );

        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        assert!(
            validate_array_value_element_type(&ctx, ptr_ty, &Location::Unknown).is_err(),
            "direct pointer elements were never part of the bare array contract"
        );
    }

    #[test]
    fn relocation_free_pointer_integer_union_gate_is_raw_only() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let raw_mut_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, true, MirPointerKind::RawMut)
                .into();
        let raw_mut_union: TypeHandle = MirUnionType::get(
            &mut ctx,
            "RawMutBits".into(),
            vec!["ptr".into(), "bits".into()],
            vec![raw_mut_ty, u64_ty],
            8,
            8,
        )
        .into();
        assert_eq!(
            relocation_free_pointer_integer_union_storage_field(&ctx, raw_mut_union, 8),
            Some((1, u64_ty)),
            "RawMut carries no uniqueness claim and may use the integer storage view"
        );

        let erased_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let shared_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::SharedRef)
                .into();
        let unique_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, true, MirPointerKind::UniqueRef)
                .into();
        let global_raw_ty: TypeHandle = MirPtrType::get_with_kind(
            &mut ctx,
            u32_ty,
            false,
            dialect_mir::types::address_space::GLOBAL,
            MirPointerKind::RawConst,
        )
        .into();
        let fn_marker: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "FnPtrTarget".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            0,
        )
        .into();
        let fn_token_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, fn_marker, false).into();

        for (name, pointer_ty) in [
            ("Erased", erased_ty),
            ("SharedRef", shared_ty),
            ("UniqueRef", unique_ty),
            ("non-generic raw", global_raw_ty),
            ("function token", fn_token_ty),
        ] {
            let union_ty: TypeHandle = MirUnionType::get(
                &mut ctx,
                format!("{name}Bits"),
                vec!["ptr".into(), "bits".into()],
                vec![pointer_ty, u64_ty],
                8,
                8,
            )
            .into();
            assert_eq!(
                relocation_free_pointer_integer_union_storage_field(&ctx, union_ty, 8),
                None,
                "{name} must not acquire a pointer category from integer-initialized union storage"
            );
        }
    }

    #[test]
    fn pointer_only_union_classifier_preserves_reference_semantics() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_u32: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::SharedRef)
                .into();
        let shared_u8: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u8_ty, false, MirPointerKind::SharedRef)
                .into();
        let raw_u32: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::RawConst)
                .into();
        let raw_u8: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u8_ty, false, MirPointerKind::RawConst)
                .into();
        let raw_mut_u32: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, true, MirPointerKind::RawMut)
                .into();
        let unique_u32: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, true, MirPointerKind::UniqueRef)
                .into();

        let raw_views: TypeHandle = MirUnionType::get(
            &mut ctx,
            "RawViews".into(),
            vec!["word".into(), "byte".into()],
            vec![raw_u32, raw_u8],
            8,
            8,
        )
        .into();
        assert!(
            classify_union_constant_storage(&ctx, raw_views, UnionConstantUse::SsaReconstruction)
                .is_ok(),
            "same-kind raw pointers may safely use different pointee views"
        );

        for (name, fields, expected) in [
            (
                "MixedReferencePointees",
                vec![shared_u32, shared_u8],
                "reference pointee types",
            ),
            (
                "MixedRawKinds",
                vec![raw_u32, raw_mut_u32],
                "pointer storage semantics",
            ),
            ("UniqueReference", vec![unique_u32, unique_u32], "UniqueRef"),
        ] {
            let union_ty: TypeHandle = MirUnionType::get(
                &mut ctx,
                name.into(),
                vec!["first".into(), "second".into()],
                fields,
                8,
                8,
            )
            .into();
            let error = classify_union_constant_storage(
                &ctx,
                union_ty,
                UnionConstantUse::SsaReconstruction,
            )
            .expect_err("ambiguous pointer semantics must fail closed");
            assert!(
                error.contains(expected),
                "{name} diagnostic must identify {expected}: {error}"
            );
        }

        // Physical device-static storage emits exact bytes plus one
        // integer-width relocation slot and never mints a typed reference, so
        // same-kind reference alternatives with different pointee views stay
        // supported there (e.g. `union { word: &'static u32, byte: &'static u8 }`).
        let mixed_reference_pointees: TypeHandle = MirUnionType::get(
            &mut ctx,
            "MixedReferencePointees".into(),
            vec!["first".into(), "second".into()],
            vec![shared_u32, shared_u8],
            8,
            8,
        )
        .into();
        assert!(
            classify_union_constant_storage(
                &ctx,
                mixed_reference_pointees,
                UnionConstantUse::PhysicalStorage
            )
            .is_ok(),
            "physical initializer storage must keep mixed-pointee reference unions supported"
        );
    }

    #[test]
    fn pointer_integer_union_materialization_never_transmutes_to_a_pointer_carrier() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let raw_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::RawConst)
                .into();
        let union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "PointerOrBits".into(),
            vec!["pointer".into(), "bits".into()],
            vec![raw_ty, u64_ty],
            8,
            8,
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);

        let (value, last_op) = translate_pointer_integer_union_constant_from_storage(
            &mut ctx,
            union_ty,
            1,
            u64_ty,
            &[
                Some(0x88),
                Some(0x77),
                Some(0x66),
                Some(0x55),
                Some(0x44),
                Some(0x33),
                Some(0x22),
                Some(0x11),
            ],
            block,
            None,
            Location::Unknown,
        )
        .expect("the exact raw-pointer/integer storage shape must materialize");
        assert_eq!(value.get_type(&ctx), union_ty);

        let last_op = last_op.expect("non-empty storage must emit an insert operation");
        let insert = Operation::get_op::<MirInsertFieldOp>(last_op, &ctx)
            .expect("the final operation must insert the integer union field");
        assert_eq!(
            insert.get_attr_insert_index(&ctx).map(|index| index.0),
            Some(1)
        );
        assert!(insert.verify(&ctx).is_ok());

        for operation in block.deref(&ctx).iter(&ctx) {
            if let Some(cast) = Operation::get_op::<MirCastOp>(operation, &ctx) {
                assert!(cast.verify(&ctx).is_ok());
                let result_ty = cast
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx);
                assert!(
                    pointer_carriers_in_type(&ctx, result_ty).is_empty(),
                    "constant storage may transmute bytes to the integer field, never to a pointer-bearing union"
                );
            }
        }
    }

    #[test]
    fn device_static_union_storage_gate_admits_only_one_anchored_pointer_word() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let word_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let byte_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let thin_pointer_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "ThinPointerWord".into(),
            vec!["word".into(), "bytes".into()],
            vec![word_ptr_ty, byte_ptr_ty],
            8,
            8,
        )
        .into();

        assert_eq!(
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 8, 8, 8, &[(0, 8)]),
            Ok(()),
            "one naturally aligned pointer word with one anchored relocation is the \
             accepted shape"
        );

        // The example shape `union { word: &'static u32, byte: &'static u8 }`:
        // same-kind references with different pointee views. The storage gate
        // emits bytes plus one relocation slot and never mints a typed
        // reference, so this stays supported even though SSA reconstruction
        // of the same union fails closed.
        let shared_word_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u32_ty, false, MirPointerKind::SharedRef)
                .into();
        let shared_byte_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, u8_ty, false, MirPointerKind::SharedRef)
                .into();
        let mixed_reference_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "MixedReferenceWord".into(),
            vec!["word".into(), "byte".into()],
            vec![shared_word_ty, shared_byte_ty],
            8,
            8,
        )
        .into();
        assert_eq!(
            validate_device_static_union_storage(
                &ctx,
                mixed_reference_union_ty,
                8,
                8,
                8,
                &[(0, 8)]
            ),
            Ok(()),
            "mixed-pointee reference unions remain valid device-static storage"
        );

        let byte_image_error =
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 4, 8, 8, &[(0, 8)])
                .expect_err("a 32-bit pointer target must remain fail-closed");
        assert!(
            byte_image_error.contains("8-byte NVPTX pointers"),
            "diagnostic must name the pointer-width restriction: {byte_image_error}"
        );

        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 4).into();
        let pointer_free_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "Bits".into(),
            vec!["word".into(), "bytes".into()],
            vec![u32_ty, bytes_ty],
            4,
            4,
        )
        .into();
        let storage_error =
            validate_device_static_union_storage(&ctx, pointer_free_union_ty, 8, 4, 4, &[(0, 8)])
                .expect_err("byte-image unions take the literal-bytes path, not this gate");
        assert!(
            storage_error.contains("without thin-pointer storage"),
            "diagnostic must explain the thin-pointer requirement: {storage_error}"
        );

        let wide_union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            "TwoWords".into(),
            vec!["first".into(), "second".into()],
            vec![word_ptr_ty, byte_ptr_ty],
            16,
            8,
        )
        .into();
        let size_error =
            validate_device_static_union_storage(&ctx, wide_union_ty, 8, 16, 8, &[(0, 8)])
                .expect_err("a union wider than one pointer word must remain fail-closed");
        assert!(
            size_error.contains("size/alignment 16/8"),
            "diagnostic must report the rejected layout: {size_error}"
        );

        let initializer_error =
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 8, 16, 8, &[(0, 8)])
                .expect_err("an over-sized evaluated initializer must remain fail-closed");
        assert!(
            initializer_error.contains("initializer size/alignment 16/8"),
            "diagnostic must report the evaluated-initializer mismatch: {initializer_error}"
        );

        let uninit_error =
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 8, 8, 8, &[])
                .expect_err("zero relocations (e.g. a ZST-field initializer) must fail closed");
        assert!(
            uninit_error.contains("0 initializer relocations"),
            "diagnostic must count the missing relocation: {uninit_error}"
        );

        let multi_error = validate_device_static_union_storage(
            &ctx,
            thin_pointer_union_ty,
            8,
            8,
            8,
            &[(0, 4), (4, 4)],
        )
        .expect_err("two relocations cannot describe one pointer word");
        assert!(
            multi_error.contains("2 initializer relocations"),
            "diagnostic must count the extra relocations: {multi_error}"
        );

        let offset_error =
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 8, 8, 8, &[(4, 8)])
                .expect_err("a relocation off the word base must remain fail-closed");
        assert!(
            offset_error.contains("occupies byte 4"),
            "diagnostic must locate the misplaced relocation: {offset_error}"
        );

        let width_error =
            validate_device_static_union_storage(&ctx, thin_pointer_union_ty, 8, 8, 8, &[(0, 4)])
                .expect_err("a narrow relocation cannot carry full pointer provenance");
        assert!(
            width_error.contains("width 4"),
            "diagnostic must report the short relocation: {width_error}"
        );
    }
}
