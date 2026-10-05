/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Union constant storage classification and translation.

use super::const_alloc::{
    relocation_offsets_overlapping_range, translate_thin_pointer_at_alloc_offset,
};
use super::const_bytes::{
    constant_type_contains_pointer, rust_type_layout_size, translate_zero_sized_constant_value,
};
use super::promoted::constant_allocation;
use super::statics::GlobalInitializerData;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{
    MirCastOp, MirConstantOp, MirConstructArrayOp, MirInsertFieldOp, MirUndefOp,
};
use dialect_mir::types::MirPointerKind;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use std::num::NonZeroUsize;

/// Physical representation strategy for an initialized union constant.
///
/// rustc does not retain the source-level active field in an evaluated union
/// allocation. Pointer-free unions can therefore use the exact byte image,
/// while a provenance-bearing union is only reconstructible without guessing
/// an active field when every non-ZST alternative is the same class of thin
/// pointer storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnionConstantStorageKind {
    ByteImage,
    ThinPointer {
        field_index: usize,
        field_ty: TypeHandle,
    },
}

/// How a classified union constant is consumed by its caller.
///
/// SSA reconstruction materializes the carrier field's own reference type, so
/// a reference minted for the wrong alternative could claim validity for a
/// pointee the program never wrote. Device-static physical storage emits the
/// exact evaluated bytes plus one integer-width relocation slot and never
/// mints a typed reference, so same-kind reference alternatives may keep
/// different pointee views there; device code reads the storage through its
/// own field types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UnionConstantUse {
    SsaReconstruction,
    PhysicalStorage,
}

pub(super) fn classify_union_constant_storage(
    ctx: &Context,
    union_ty: TypeHandle,
    usage: UnionConstantUse,
) -> Result<UnionConstantStorageKind, String> {
    let (name, field_types) = {
        let ty_ref = union_ty.deref(ctx);
        let union_ty = ty_ref
            .downcast_ref::<dialect_mir::types::MirUnionType>()
            .ok_or_else(|| {
                "classify_union_constant_storage called on non-union type".to_string()
            })?;
        (union_ty.name().to_string(), union_ty.field_types().to_vec())
    };

    if !field_types
        .iter()
        .copied()
        .any(|field| constant_type_contains_pointer(ctx, field))
    {
        return Ok(UnionConstantStorageKind::ByteImage);
    }

    let mut carrier: Option<(usize, TypeHandle, u32, MirPointerKind, bool)> = None;
    for (field_index, field_ty) in field_types.into_iter().enumerate() {
        if types::is_zst_type(ctx, field_ty) {
            continue;
        }

        let pointer_semantics = {
            let field_ref = field_ty.deref(ctx);
            field_ref
                .downcast_ref::<dialect_mir::types::MirPtrType>()
                .map(|ptr| (ptr.address_space, ptr.kind, ptr.is_mutable))
        };

        let Some((address_space, pointer_kind, is_mutable)) = pointer_semantics else {
            if constant_type_contains_pointer(ctx, field_ty) {
                return Err(format!(
                    "Initialized union constant `{name}` contains pointer-bearing field \
                     {field_index} that is not a thin pointer; fat or nested pointer storage \
                     in union constants is not supported"
                ));
            }
            return Err(format!(
                "Initialized union constant `{name}` overlaps thin-pointer storage with \
                 non-pointer field {field_index}; pointer/integer union constants cannot \
                 preserve both provenance and exact integer bits"
            ));
        };

        if dialect_mir::types::is_opaque_fn_pointer_type(ctx, field_ty) {
            return Err(format!(
                "Initialized union constant `{name}` contains canonical function-pointer \
                 field {field_index}; function tokens cannot serve as data-pointer union storage"
            ));
        }
        if pointer_kind == MirPointerKind::UniqueRef {
            return Err(format!(
                "Initialized union constant `{name}` contains UniqueRef field {field_index}; \
                 constant reconstruction cannot establish uniqueness"
            ));
        }

        match carrier {
            None => {
                carrier = Some((
                    field_index,
                    field_ty,
                    address_space,
                    pointer_kind,
                    is_mutable,
                ));
            }
            Some((
                _,
                carrier_field_ty,
                carrier_address_space,
                carrier_kind,
                carrier_mutability,
            )) if carrier_address_space == address_space
                && carrier_kind == pointer_kind
                && carrier_mutability == is_mutable =>
            {
                if usage == UnionConstantUse::SsaReconstruction
                    && pointer_kind == MirPointerKind::SharedRef
                    && carrier_field_ty != field_ty
                {
                    return Err(format!(
                        "Initialized union constant `{name}` mixes reference pointee types; \
                         rustc's evaluated allocation does not retain the active union field, \
                         so reconstructing either reference view could invent pointee validity"
                    ));
                }
            }
            Some((_, _, carrier_address_space, carrier_kind, carrier_mutability)) => {
                return Err(format!(
                    "Initialized union constant `{name}` mixes pointer storage semantics \
                     (address space/kind/mutability {carrier_address_space}/{carrier_kind:?}/{carrier_mutability} \
                     and {address_space}/{pointer_kind:?}/{is_mutable}); one union carrier cannot \
                     preserve both representations and Rust pointer categories"
                ));
            }
        }
    }

    let Some((field_index, field_ty, _, _, _)) = carrier else {
        return Err(format!(
            "Initialized union constant `{name}` is pointer-bearing but has no non-ZST \
             thin-pointer field"
        ));
    };

    Ok(UnionConstantStorageKind::ThinPointer {
        field_index,
        field_ty,
    })
}

/// Return the full-width integer field through which a relocation-free mixed
/// raw-pointer/integer union can be reconstructed without producing a pointer.
///
/// Keep this deliberately narrower than [`classify_union_constant_storage`]:
/// the ordinary classifier remains the provenance-preserving authority used by
/// pointer-bearing constants and device-static union initializers. This helper
/// only admits one naturally aligned pointer word whose non-ZST alternatives
/// are semantically compatible generic raw pointers or full-width integers.
pub(super) fn relocation_free_pointer_integer_union_storage_field(
    ctx: &Context,
    union_ty: TypeHandle,
    pointer_width: usize,
) -> Option<(usize, TypeHandle)> {
    let (union_size, union_align, field_types) = {
        let ty_ref = union_ty.deref(ctx);
        let union_ty = ty_ref.downcast_ref::<dialect_mir::types::MirUnionType>()?;
        (
            union_ty.total_size(),
            union_ty.abi_align(),
            union_ty.field_types().to_vec(),
        )
    };

    let pointer_width_u64 = pointer_width as u64;
    if union_size != pointer_width_u64 || union_align != pointer_width_u64 {
        return None;
    }

    let integer_width = (pointer_width * 8) as u32;
    let mut pointer_semantics = None;
    let mut saw_pointer = false;
    let mut integer_field = None;

    for (field_index, field_ty) in field_types.into_iter().enumerate() {
        if types::is_zst_type(ctx, field_ty) {
            continue;
        }

        let field_ref = field_ty.deref(ctx);
        if let Some(pointer) = field_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            if !matches!(
                pointer.kind,
                MirPointerKind::RawConst | MirPointerKind::RawMut
            ) || pointer.address_space != dialect_mir::types::address_space::GENERIC
            {
                return None;
            }
            let semantics = (pointer.address_space, pointer.kind, pointer.is_mutable);
            match pointer_semantics {
                None => pointer_semantics = Some(semantics),
                Some(existing) if existing == semantics => {}
                Some(_) => return None,
            }
            saw_pointer = true;
            continue;
        }

        if field_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == integer_width)
        {
            integer_field.get_or_insert((field_index, field_ty));
            continue;
        }

        // Fat pointers, nested pointer aggregates, partial-width integers, and
        // unrelated scalar/aggregate alternatives keep the existing fail-closed
        // path. The byte-image exception is intentionally pointer-word exact.
        return None;
    }

    saw_pointer.then_some(())?;
    integer_field
}

/// Admit only the device-static union shape whose complete storage can be
/// represented by one provenance-preserving thin-pointer relocation.
///
/// Device-global initializers do not retain a source-level active union field,
/// so this path must prove the physical storage without guessing one. Keep the
/// scope deliberately narrower than ordinary union SSA lowering: one top-level
/// pointer-sized union, naturally aligned, one relocation at byte zero, and
/// every non-ZST alternative a representation-compatible thin pointer.
pub(super) fn validate_device_static_union_initializer(
    ctx: &Context,
    static_def: &rustc_public::mir::mono::StaticDef,
    union_ty: TypeHandle,
    initializer: &GlobalInitializerData,
    loc: Location,
) -> TranslationResult<()> {
    let relocation_slots: Vec<(u64, u32)> = initializer
        .relocations
        .iter()
        .map(|relocation| (relocation.source_offset, relocation.width_bytes))
        .collect();
    validate_device_static_union_storage(
        ctx,
        union_ty,
        rustc_public::target::MachineInfo::target_pointer_width().bytes(),
        initializer.bytes.len() as u64,
        initializer.alignment,
        &relocation_slots,
    )
    .map_err(|message| {
        input_error!(
            loc,
            TranslationErr::unsupported(format!(
                "device static {} contains {message}",
                static_def.name()
            ))
        )
    })
}

/// Pure storage gate behind [`validate_device_static_union_initializer`]:
/// accept only a union that is one naturally aligned pointer word whose
/// evaluated initializer is exactly one full-width relocation at byte zero.
/// The caller attaches the static's name and source location to rejections.
pub(super) fn validate_device_static_union_storage(
    ctx: &Context,
    union_ty: TypeHandle,
    pointer_width: usize,
    initializer_len: u64,
    initializer_align: u64,
    relocation_slots: &[(u64, u32)],
) -> Result<(), String> {
    let (union_name, union_size, union_align) = {
        let ty_ref = union_ty.deref(ctx);
        let union_ty = ty_ref
            .downcast_ref::<dialect_mir::types::MirUnionType>()
            .ok_or_else(|| {
                "a non-union type in the device-static union initializer gate".to_string()
            })?;
        (
            union_ty.name().to_string(),
            union_ty.total_size(),
            union_ty.abi_align(),
        )
    };

    let storage_kind =
        classify_union_constant_storage(ctx, union_ty, UnionConstantUse::PhysicalStorage).map_err(
            |message| {
                format!(
                    "union `{union_name}` whose initializer cannot preserve pointer provenance: \
                 {message}"
                )
            },
        )?;
    if !matches!(storage_kind, UnionConstantStorageKind::ThinPointer { .. }) {
        return Err(format!(
            "union `{union_name}` without thin-pointer storage; device-global union \
             initializers are supported only when every non-ZST alternative is a \
             representation-compatible thin pointer"
        ));
    }

    if pointer_width != 8 {
        return Err(format!(
            "union `{union_name}`, but cuda-oxide currently supports device-global union \
             relocations only for 8-byte NVPTX pointers"
        ));
    }
    let pointer_width_u64 = pointer_width as u64;

    if union_size != pointer_width_u64 || union_align != pointer_width_u64 {
        return Err(format!(
            "union `{union_name}` with size/alignment {union_size}/{union_align}; \
             device-global union relocations require exactly one naturally aligned \
             {pointer_width}-byte pointer word"
        ));
    }
    if initializer_len != pointer_width_u64 || initializer_align != pointer_width_u64 {
        return Err(format!(
            "union `{union_name}` with evaluated initializer size/alignment \
             {initializer_len}/{initializer_align}, expected {pointer_width}/{pointer_width}"
        ));
    }

    let [(source_offset, width_bytes)] = relocation_slots else {
        return Err(format!(
            "union `{union_name}` with {} initializer relocations; exactly one \
             thin-pointer relocation is required",
            relocation_slots.len()
        ));
    };
    if *source_offset != 0 || u64::from(*width_bytes) != pointer_width_u64 {
        return Err(format!(
            "union `{union_name}` whose relocation occupies byte {source_offset} with \
             width {width_bytes}, expected one {pointer_width}-byte slot at byte zero"
        ));
    }

    Ok(())
}

/// Verify that rustc and the MIR union agree on the exact stored size.
fn union_constant_storage_size(
    ctx: &Context,
    rust_ty: &rustc_public::ty::Ty,
    union_ty: TypeHandle,
    loc: &Location,
) -> TranslationResult<usize> {
    let rust_size = rust_type_layout_size(*rust_ty, loc.clone())?;
    let (name, mir_size) = {
        let ty_ref = union_ty.deref(ctx);
        let union_ty = ty_ref
            .downcast_ref::<dialect_mir::types::MirUnionType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "union_constant_storage_size called on non-union type"
                ))
            })?;
        let size = usize::try_from(union_ty.total_size()).map_err(|_| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Union constant `{}` size {} does not fit usize",
                union_ty.name(),
                union_ty.total_size()
            )))
        })?;
        (union_ty.name().to_string(), size)
    };

    if rust_size != mir_size {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Union constant `{name}` has rustc size {rust_size}, but MirUnionType records {mir_size}"
            ))
        );
    }
    Ok(rust_size)
}

/// Materialize a non-ZST union constant without guessing an active field.
///
/// rustc constant evaluation gives us the physical storage image and its
/// initialization mask, but not a source-level active-field identity. Pointer-free
/// unions therefore keep the exact byte image. Pointer-only unions use a typed
/// carrier selected from representation-compatible fields so relocation provenance
/// survives without claiming which source field initialized the allocation.
pub(super) fn translate_union_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    union_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let size = union_constant_storage_size(ctx, rust_ty, union_ty, &loc)?;
    if size == 0 {
        return translate_zero_sized_constant_value(ctx, union_ty, block_ptr, prev_op, loc);
    }

    let Some(alloc) = constant_allocation(constant) else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Initialized union constant of {size} byte(s) must be backed by an allocation, found {:?}",
                constant.const_.kind()
            ))
        );
    };

    translate_union_constant_from_alloc(ctx, alloc, 0, rust_ty, union_ty, block_ptr, prev_op, loc)
}

/// Materialize one union value from an allocation while preserving its storage semantics.
///
/// Pointer-free unions retain the existing byte-image path and exact initialization
/// mask. A union whose every non-ZST alternative is a compatible thin pointer
/// instead uses one typed pointer carrier so rustc relocation provenance never
/// becomes integer bytes. A naturally aligned pointer-word union that overlaps
/// compatible thin pointers with full-width integers may also use the byte image,
/// but only when no relocation overlaps its storage. Relocation-bearing mixed
/// unions, fat pointers, nested pointer aggregates, and ambiguous layouts remain
/// fail-closed.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_union_constant_from_alloc(
    ctx: &mut Context,
    alloc: &rustc_public::ty::Allocation,
    base_offset: usize,
    rust_ty: &rustc_public::ty::Ty,
    union_ty: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let size = union_constant_storage_size(ctx, rust_ty, union_ty, &loc)?;
    if size == 0 {
        return translate_zero_sized_constant_value(ctx, union_ty, block_ptr, prev_op, loc);
    }

    let end = base_offset.checked_add(size).ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Union constant byte range overflows: offset {base_offset} + size {size}"
            ))
        )
    })?;
    if end > alloc.bytes.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Union constant needs bytes [{base_offset}..{end}), but allocation is only {} bytes",
                alloc.bytes.len()
            ))
        );
    }

    let pointer_width = rustc_public::target::MachineInfo::target_pointer_width().bytes();
    let relocations = relocation_offsets_overlapping_range(
        &alloc.provenance.ptrs,
        base_offset,
        end,
        pointer_width,
    );
    let storage = &alloc.bytes[base_offset..end];

    // A mixed pointer/integer union initialized through the integer view carries
    // no relocation provenance. When every byte is initialized, the evaluated
    // byte image is the complete storage truth. Reconstruct it through the
    // full-width integer field so no inactive pointer alternative is produced.
    // Keep this no-relocation exception separate: #984's device-static gate and
    // relocation-bearing union constants use the stricter pointer classifier.
    if relocations.is_empty()
        && storage.iter().all(Option::is_some)
        && let Some((integer_field_index, integer_field_ty)) =
            relocation_free_pointer_integer_union_storage_field(ctx, union_ty, pointer_width)
    {
        return translate_pointer_integer_union_constant_from_storage(
            ctx,
            union_ty,
            integer_field_index,
            integer_field_ty,
            storage,
            block_ptr,
            prev_op,
            loc,
        );
    }

    let storage_kind =
        classify_union_constant_storage(ctx, union_ty, UnionConstantUse::SsaReconstruction)
            .map_err(|message| input_error!(loc.clone(), TranslationErr::unsupported(message)))?;

    match storage_kind {
        UnionConstantStorageKind::ByteImage => {
            if !relocations.is_empty() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Pointer-free union constant has relocation(s) overlapping byte offset(s) \
                         {relocations:?}; the union type does not contain storage that can \
                         preserve that provenance"
                    ))
                );
            }

            translate_union_constant_from_storage(ctx, union_ty, storage, block_ptr, prev_op, loc)
        }
        UnionConstantStorageKind::ThinPointer {
            field_index,
            field_ty,
        } => {
            if size != pointer_width {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Thin-pointer union constant has size {size}, but the target pointer width \
                         is {pointer_width}; over-aligned or padded pointer-union constants are \
                         not yet supported because their non-pointer bytes would need a separate \
                         initialization-mask representation"
                    ))
                );
            }

            if relocations.iter().any(|offset| *offset != base_offset) || relocations.len() > 1 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Thin-pointer union constant expects at most one relocation anchored at \
                         byte {base_offset}, found overlapping relocation start(s) {relocations:?}"
                    ))
                );
            }

            let pointer_end = base_offset + pointer_width;
            if alloc.bytes[base_offset..pointer_end]
                .iter()
                .any(|byte| byte.is_none())
            {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "Thin-pointer union constant contains uninitialized bytes in its pointer \
                         carrier; partially initialized pointer storage cannot be reconstructed"
                            .to_string()
                    )
                );
            }

            let (pointer, current_prev_op) = translate_thin_pointer_at_alloc_offset(
                ctx,
                alloc,
                base_offset,
                field_ty,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            let undef_op = MirUndefOp::new(ctx, union_ty).get_operation();
            undef_op.deref_mut(ctx).set_loc(loc.clone());
            match current_prev_op {
                Some(prev) => undef_op.insert_after(ctx, prev),
                None => undef_op.insert_at_front(block_ptr, ctx),
            }
            let undef_value = undef_op.deref(ctx).get_result(0);

            let insert_op = Operation::new(
                ctx,
                MirInsertFieldOp::get_concrete_op_info(),
                vec![union_ty],
                vec![undef_value, pointer],
                vec![],
                0,
            );
            insert_op.deref_mut(ctx).set_loc(loc);
            MirInsertFieldOp::new(insert_op).set_attr_insert_index(
                ctx,
                dialect_mir::attributes::FieldIndexAttr(field_index as u32),
            );
            insert_op.insert_after(ctx, undef_op);

            Ok((insert_op.deref(ctx).get_result(0), Some(insert_op)))
        }
    }
}

/// Build `[u8; size]` with one SSA value per physical byte and transmute it to
/// the union type. `None` bytes become `mir.undef`; no inactive byte is invented.
fn translate_union_constant_from_storage(
    ctx: &mut Context,
    union_ty: TypeHandle,
    storage: &[Option<u8>],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    if storage.is_empty() {
        return translate_zero_sized_constant_value(ctx, union_ty, block_ptr, prev_op, loc);
    }

    let (byte_array, array_op) =
        translate_constant_storage_byte_array(ctx, storage, block_ptr, prev_op, loc.clone());

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![union_ty],
        vec![byte_array],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
    cast_op.insert_after(ctx, array_op);

    Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
}

/// Reconstruct a relocation-free pointer/integer union through its integer
/// field, exactly as an ordinary Rust union aggregate is constructed.
///
/// The byte-array-to-integer transmute is representation-only and produces no
/// pointer carrier. Inserting that integer into an undefined union therefore
/// keeps the pointer alternatives inactive instead of asking a synthetic
/// transmute to establish their raw-pointer kinds.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_pointer_integer_union_constant_from_storage(
    ctx: &mut Context,
    union_ty: TypeHandle,
    integer_field_index: usize,
    integer_field_ty: TypeHandle,
    storage: &[Option<u8>],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    debug_assert!(!storage.is_empty());

    let (byte_array, array_op) =
        translate_constant_storage_byte_array(ctx, storage, block_ptr, prev_op, loc.clone());

    let integer_cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![integer_field_ty],
        vec![byte_array],
        vec![],
        0,
    );
    integer_cast_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(integer_cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
    integer_cast_op.insert_after(ctx, array_op);
    let integer_value = integer_cast_op.deref(ctx).get_result(0);

    let undef_op = MirUndefOp::new(ctx, union_ty).get_operation();
    undef_op.deref_mut(ctx).set_loc(loc.clone());
    undef_op.insert_after(ctx, integer_cast_op);
    let undef_value = undef_op.deref(ctx).get_result(0);

    let insert_op = Operation::new(
        ctx,
        MirInsertFieldOp::get_concrete_op_info(),
        vec![union_ty],
        vec![undef_value, integer_value],
        vec![],
        0,
    );
    insert_op.deref_mut(ctx).set_loc(loc);
    MirInsertFieldOp::new(insert_op).set_attr_insert_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(integer_field_index as u32),
    );
    insert_op.insert_after(ctx, undef_op);

    Ok((insert_op.deref(ctx).get_result(0), Some(insert_op)))
}

/// Build the exact `[u8; size]` SSA image used by constant-storage
/// reconstruction. `None` bytes remain `mir.undef`.
fn translate_constant_storage_byte_array(
    ctx: &mut Context,
    storage: &[Option<u8>],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    debug_assert!(!storage.is_empty());

    let byte_ty = IntegerType::get(ctx, 8, Signedness::Unsigned);
    let byte_ty_handle: TypeHandle = byte_ty.into();
    let mut bytes = Vec::with_capacity(storage.len());
    let mut current_prev_op = prev_op;

    for byte in storage {
        let op = match byte {
            Some(value) => {
                let attr = pliron::builtin::attributes::IntegerAttr::new(
                    byte_ty,
                    APInt::from_u64(
                        u64::from(*value),
                        NonZeroUsize::new(8).expect("u8 width is non-zero"),
                    ),
                );
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![byte_ty_handle],
                    vec![],
                    vec![],
                    0,
                );
                MirConstantOp::new(op).set_attr_value(ctx, attr);
                op
            }
            None => MirUndefOp::new(ctx, byte_ty_handle).get_operation(),
        };
        op.deref_mut(ctx).set_loc(loc.clone());
        match current_prev_op {
            Some(prev) => op.insert_after(ctx, prev),
            None => op.insert_at_front(block_ptr, ctx),
        }
        bytes.push(op.deref(ctx).get_result(0));
        current_prev_op = Some(op);
    }

    let byte_array_ty: TypeHandle =
        dialect_mir::types::MirArrayType::get(ctx, byte_ty_handle, storage.len() as u64).into();
    let array_op = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![byte_array_ty],
        bytes,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    match current_prev_op {
        Some(prev) => array_op.insert_after(ctx, prev),
        None => array_op.insert_at_front(block_ptr, ctx),
    }
    (array_op.deref(ctx).get_result(0), array_op)
}
