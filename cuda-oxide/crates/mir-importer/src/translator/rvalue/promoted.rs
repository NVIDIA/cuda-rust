/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Promoted immutable device globals and their layout gates.

use super::coerce::cast_to_declared_rust_pointer_type_if_needed;
use super::const_bytes::rust_type_layout_size;
use super::static_global::{
    StaticGlobalMaterializationState, ensure_initializer_relocation_targets,
    set_global_initializer_hex_attr, set_global_initializer_relocations_attr,
    stored_type_union_name,
};
use super::statics::{
    GlobalInitializerData, GlobalInitializerRelocation, bytes_to_hex,
    encode_global_initializer_relocations, promoted_array_initializer,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirPointerKindAuthorityAttr;
use dialect_mir::ops::{MirConstantOp, MirGlobalAllocOp};
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
use pliron::{input_err, input_error_noloc};
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
use std::num::NonZeroUsize;

/// Translate a pointer-to-array constant to MIR operations.
///
/// Handles both byte string literals (`&[u8; N]`, e.g. `b"hello\0"`) and typed
/// array constants (`&[f64; 3]`, `&[u32; 4]`, etc.). The function:
/// 1. Extracts raw bytes from the constant's allocation
/// 2. Groups bytes into element-sized chunks based on the array element type
/// 3. Creates typed constants for each element (u8, u32, f32, f64, etc.)
/// 4. Returns a pointer to the constructed array
pub(super) fn translate_ptr_to_array_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    const_ty_ptr: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    // Extract array type from the pointer type. A pointer-to-array constant can
    // outlive this function, so lowering it as `array value + mir.ref` would
    // return a pointer to function-local stack storage. Materialize it as an
    // immutable device global instead.
    let array_ty = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let ptr_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_ptr_to_array_constant: expected pointer type"
                ))
            })?;

        let arr_ty_obj = ptr_ty.pointee.deref(ctx);
        if arr_ty_obj
            .downcast_ref::<dialect_mir::types::MirArrayType>()
            .is_none()
        {
            return input_err!(
                loc,
                TranslationErr::unsupported(
                    "translate_ptr_to_array_constant: expected array pointee"
                )
            );
        }
        ptr_ty.pointee
    };

    let rust_array_ty = match constant.const_.ty().kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::RawPtr(pointee, _))
        | rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, pointee, _)) => {
            pointee
        }
        _ => constant.const_.ty(),
    };
    let expected_size = rust_type_layout_size(rust_array_ty, loc.clone())?;
    if expected_size != 0
        && let Some(union_name) = stored_type_union_name(rust_array_ty, &mut Vec::new())
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "promoted array initializer contains union `{union_name}`; initialized union storage is not yet supported"
            ))
        );
    }

    validate_ptr_to_array_constant_type(ctx, array_ty, loc.clone())?;
    // The same byte-size agreement the bare-array promotion demands: the
    // global's declared type is the converted dialect type while its
    // initializer bytes come from rustc's evaluated allocation, so the two
    // layouts must agree before the byte image is trusted. Unlike that path
    // there is no element-wise lowering to fall back to here, so disagreement
    // is an error rather than a bail-out. Zero size stays exempt because
    // `dialect_stored_size` cannot tell a genuine zero from an unrecorded
    // layout, and an empty initializer has no bytes to misplace.
    let stored_size = dialect_stored_size(ctx, array_ty);
    if expected_size > 0 && stored_size != Some(expected_size as u64) {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "translate_ptr_to_array_constant: converted array storage size {stored_size:?} disagrees with rustc layout size {expected_size}"
            ))
        );
    }
    let initializer = promoted_array_initializer(constant, expected_size, "array", loc.clone())?;
    let promoted_anchor = materialize_promoted_initializer_targets(
        ctx,
        &initializer.relocations,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let global_alloc = emit_promoted_immutable_global(
        ctx,
        array_ty,
        &initializer,
        block_ptr,
        promoted_anchor,
        loc.clone(),
    );

    let global_ptr = global_alloc.get_operation().deref(ctx).get_result(0);
    let (ptr_val, last_op) = cast_to_declared_rust_pointer_type_if_needed(
        ctx,
        global_ptr,
        const_ty_ptr,
        block_ptr,
        Some(global_alloc.get_operation()),
        loc,
        MirPointerKindAuthorityAttr::StaticAddress,
    );
    Ok((ptr_val, last_op))
}

/// Materialize an evaluated constant allocation as an immutable device global.
///
/// Deduplicated by type, bytes, allocation alignment, and relocation identity.
/// Pointer placeholder bytes alone are not enough: two constants can have
/// identical byte images and addends while their rustc provenance targets
/// different statics or their backing allocations require different alignment.
///
/// The global is marked immutable, which is what makes it useful beyond simply
/// having an address: the exporter writes LLVM `constant`, so `opt` may treat
/// reads of it as invariant. Pointer slots remain symbolic relocation metadata
/// all the way to the exporter instead of being reconstructed from placeholder
/// integer bytes.
pub(crate) fn emit_promoted_immutable_global(
    ctx: &mut Context,
    value_ty: TypeHandle,
    initializer: &GlobalInitializerData,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> MirGlobalAllocOp {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let initializer_hex = bytes_to_hex(&initializer.bytes);
    let global_key = promoted_constant_dedup_key(ctx, value_ty, initializer);
    let global_ptr_ty = MirPtrType::get_global(ctx, value_ty, false);
    let validation_ty = promoted_global_validation_type(ctx, value_ty, initializer.bytes.len());
    let global_op = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    global_op.deref_mut(ctx).set_loc(loc);

    let global_alloc = MirGlobalAllocOp::new(global_op);
    global_alloc.set_attr_global_type(ctx, TypeAttr::new(validation_ty));
    global_alloc.set_attr_global_key(ctx, StringAttr::new(global_key));
    set_global_initializer_hex_attr(ctx, global_alloc.get_operation(), &initializer_hex);
    if !initializer.relocations.is_empty() {
        let encoded = encode_global_initializer_relocations(&initializer.relocations);
        set_global_initializer_relocations_attr(ctx, global_alloc.get_operation(), &encoded);
    }
    if initializer.alignment > 0 {
        global_alloc.set_alignment_value(ctx, initializer.alignment);
    }
    global_alloc.mark_immutable(ctx);

    if let Some(prev) = prev_op {
        global_alloc.get_operation().insert_after(ctx, prev);
    } else {
        global_alloc.get_operation().insert_at_front(block_ptr, ctx);
    }

    global_alloc
}

/// Ensure every Rust static referenced by a promoted initializer exists as a
/// MIR global before the promoted table is emitted.
///
/// Without this step the old element-wise pointer decoder used to materialize
/// the targets as a side effect. Once the entire table stays in one global, the
/// relocation metadata is the only reference and the targets must be made
/// explicit here.
fn materialize_promoted_initializer_targets(
    ctx: &mut Context,
    relocations: &[GlobalInitializerRelocation],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let mut state = StaticGlobalMaterializationState {
        globals: std::collections::HashMap::new(),
        last_op: prev_op,
    };
    ensure_initializer_relocation_targets(
        ctx,
        relocations,
        "promoted constant table",
        block_ptr,
        loc,
        &mut state,
    )?;
    Ok(state.last_op)
}

/// Whether an array's elements are cheap enough to be worth promoting the whole
/// array to an immutable global and copying it in.
///
/// This is the element gate for the immutable-global promotion optimization
/// used by bare array values ([`translate_array_constant_into_alloca`]) and for
/// the pointer-to-array form ([`translate_ptr_to_array_constant`], via
/// [`validate_ptr_to_array_constant_type`]). It is deliberately narrower than
/// bare-array value admission: an unpromotable bare value can fall back to
/// element-wise materialization, while a pointer-to-array constant cannot.
///
/// Admits primitive scalars, thin pointers, enums carrying no payload, tuples
/// and structs whose every field is itself admissible, and nested arrays of any
/// of those. Pointer admission is still relocation-gated later: only provenance
/// that can be represented by [`GlobalInitializerRelocation`] reaches promotion.
/// `ty` is
/// the array type, and nesting is walked so an unsupported leaf cannot hide
/// inside it. A zero-length array passes for any element type: its initializer
/// is empty and nothing can ever be read through it, which is what admits a
/// promoted empty-slice constant such as `&[]` (rustc promotes it to `&[T; 0]`)
/// regardless of `T`.
///
/// Tuples are admitted when every field is itself promotable. That only became
/// worth doing once a tuple field read stopped going through a copy of the whole
/// array: while that copy stood it dominated, so promoting such a table changed
/// nothing measurable and merely added a global to the module image. With the
/// read addressed in place, the promotion is what removes the depot — the two
/// only pay off together.
///
/// Structs are admitted recursively, including thin pointer fields. Unions,
/// fat-pointer storage, payload-carrying enums, and any other unsupported leaf
/// remain outside this path. A payload-carrying enum stays out because reading
/// one back still
/// round-trips the payload through memory: the address walker resolves enum
/// payload fields for writes, not for reads.
///
/// A struct's recorded rustc layout must be byte-faithful in the LLVM storage
/// selected by lowering. Ordinary layouts use the natural struct path; layouts
/// such as `repr(C, packed)` may use the packed struct path when the recorded
/// field ranges are non-overlapping and the packed representation can reproduce
/// every offset and the final size exactly. Layouts that neither representation
/// can express remain on the fallback path. See
/// [`struct_layout_matches_llvm_storage`].
fn promotable_array_element(ctx: &Context, ty: TypeHandle) -> bool {
    use dialect_mir::types::{MirArrayType, MirEnumType, MirStructType, MirTupleType};

    let obj = ty.deref(ctx);
    if let Some(array) = obj.downcast_ref::<MirArrayType>() {
        // No element values exist, so the element restriction is vacuous.
        if array.size() == 0 {
            return true;
        }
        return promotable_array_element(ctx, array.element_type());
    }
    if let Some(tuple) = obj.downcast_ref::<MirTupleType>() {
        let fields = tuple.get_types().to_vec();
        drop(obj);
        return fields
            .into_iter()
            .all(|field| promotable_array_element(ctx, field));
    }
    if let Some(structure) = obj.downcast_ref::<MirStructType>() {
        let fields = structure.field_types().to_vec();
        let field_offsets = structure.field_offsets().to_vec();
        let mem_to_decl = structure.mem_to_decl.clone();
        let total_size = structure.total_size;
        let has_zero_byte_over_alignment = total_size == 0 && structure.abi_align > 1;
        drop(obj);

        // A zero-byte `repr(align(N))` struct can raise the alignment of an
        // enclosing tuple without contributing storage to its LLVM shape. Keep
        // that established alignment-sensitive path out of immutable-global
        // promotion; ordinary stored structs still recurse through their fields.
        if has_zero_byte_over_alignment {
            return false;
        }

        // Immutable promotion copies rustc's evaluated byte image into the
        // exact storage type selected by lowering. Accept either the ordinary
        // natural representation or the exact packed representation introduced
        // for layout-divergent structs. Overlapping or otherwise unrepresentable
        // field maps still fail closed.
        if !struct_layout_matches_llvm_storage(
            ctx,
            &fields,
            &field_offsets,
            &mem_to_decl,
            total_size,
        ) {
            return false;
        }

        return fields
            .into_iter()
            .all(|field| promotable_array_element(ctx, field));
    }
    if let Some(enumeration) = obj.downcast_ref::<MirEnumType>() {
        // No variant carries a field, so the whole element *is* its discriminant
        // and reading one is a single load.
        return enumeration.total_size() > 0
            && enumeration
                .variant_field_counts
                .iter()
                .all(|&count| count == 0);
    }
    obj.is::<dialect_mir::types::MirPtrType>()
        || obj.is::<IntegerType>()
        || obj.is::<MirFP16Type>()
        || obj.is::<FP32Type>()
        || obj.is::<FP64Type>()
}

/// Whether rustc's recorded struct layout is byte-faithful in one of the LLVM
/// struct representations available to lowering.
///
/// The natural representation remains preferred. When it cannot reproduce the
/// recorded offsets, lowering can use a packed LLVM struct with explicit byte
/// padding between stored fields. Such a representation is exact iff the stored
/// fields are non-overlapping in rustc memory order and every field range plus
/// the tail fits inside `total_size`. Alignment is intentionally not part of the
/// packed check: LLVM packed structs permit an otherwise naturally aligned field
/// at an unaligned byte offset, while the enclosing allocation keeps rustc's ABI
/// alignment separately.
///
/// Unknown layouts retain the previous optimistic answer here; the promotion
/// path's stored-size agreement check rejects a non-empty initializer whose
/// concrete size cannot be established.
fn struct_layout_matches_llvm_storage(
    ctx: &Context,
    field_types: &[TypeHandle],
    field_offsets: &[u64],
    mem_to_decl: &[usize],
    total_size: u64,
) -> bool {
    if field_offsets.is_empty() || total_size == 0 {
        return true;
    }
    if field_offsets.len() != field_types.len() {
        return false;
    }

    if struct_layout_matches_llvm_natural(ctx, field_types, field_offsets, mem_to_decl, total_size)
    {
        return true;
    }

    // Empty `mem_to_decl` means identity (declaration order = memory order).
    let identity: Vec<usize>;
    let memory_order: &[usize] = if mem_to_decl.is_empty() {
        identity = (0..field_types.len()).collect();
        &identity
    } else {
        mem_to_decl
    };

    let mut end = 0u64;
    for &decl_idx in memory_order {
        if decl_idx >= field_types.len() {
            return false;
        }

        let Some((size, _)) = llvm_natural_size_align(ctx, field_types[decl_idx]) else {
            return false;
        };
        if size == 0 {
            continue;
        }

        let offset = field_offsets[decl_idx];
        if offset < end {
            return false;
        }
        let Some(field_end) = offset.checked_add(size) else {
            return false;
        };
        if field_end > total_size {
            return false;
        }
        end = field_end;
    }

    end <= total_size
}

/// Type used by initialized-global validation for a promoted constant.
///
/// `mir.global_alloc` keeps the semantic result pointer separately from its
/// `global_type` storage metadata. Pointer-free initializers are emitted as an
/// exact `[N x i8]` allocation by lowering. For a semantic type that contains a
/// byte-faithful packed struct, use that physical byte view for global-layout
/// validation while retaining the semantic result pointer. Later typed GEPs and
/// loads still use `value_ty`, whose packed representation is selected by #941.
/// Ordinary layouts keep their semantic validation unchanged.
fn promoted_global_validation_type(
    ctx: &mut Context,
    value_ty: TypeHandle,
    byte_len: usize,
) -> TypeHandle {
    use dialect_mir::types::MirArrayType;

    // A zero-byte allocation has no field bytes whose physical placement
    // needs the packed-storage fallback. Retain the semantic type so the
    // immutable-global root remains an exact authority for rustc-promoted
    // `&mut [T; 0]`; lowering still checks its zero size and required
    // alignment before emitting `[0 x i8]` storage.
    if byte_len == 0 {
        return value_ty;
    }

    if !type_contains_packed_struct_storage(ctx, value_ty) {
        return value_ty;
    }

    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    MirArrayType::get(ctx, byte_ty, byte_len as u64).into()
}

/// Whether `ty` contains a struct whose exact LLVM storage must be packed.
fn type_contains_packed_struct_storage(ctx: &Context, ty: TypeHandle) -> bool {
    use dialect_mir::types::{MirArrayType, MirStructType, MirTupleType};

    let obj = ty.deref(ctx);
    if let Some(array) = obj.downcast_ref::<MirArrayType>() {
        let element_ty = array.element_type();
        drop(obj);
        return type_contains_packed_struct_storage(ctx, element_ty);
    }
    if let Some(tuple) = obj.downcast_ref::<MirTupleType>() {
        let fields = tuple.get_types().to_vec();
        drop(obj);
        return fields
            .into_iter()
            .any(|field| type_contains_packed_struct_storage(ctx, field));
    }
    if let Some(structure) = obj.downcast_ref::<MirStructType>() {
        let fields = structure.field_types().to_vec();
        let field_offsets = structure.field_offsets().to_vec();
        let mem_to_decl = structure.mem_to_decl.clone();
        let total_size = structure.total_size;
        drop(obj);

        let uses_packed_storage = !struct_layout_matches_llvm_natural(
            ctx,
            &fields,
            &field_offsets,
            &mem_to_decl,
            total_size,
        ) && struct_layout_matches_llvm_storage(
            ctx,
            &fields,
            &field_offsets,
            &mem_to_decl,
            total_size,
        );
        return uses_packed_storage
            || fields
                .into_iter()
                .any(|field| type_contains_packed_struct_storage(ctx, field));
    }

    false
}

/// Whether rustc's recorded struct layout is one the lowering's non-packed
/// LLVM struct actually reproduces at the byte level.
///
/// The lowering places fields at their recorded offsets by inserting explicit
/// `[N x i8]` padding slots, which can only ADD bytes: a field can never land
/// below its natural LLVM alignment, and LLVM still rounds the struct's size
/// up to its natural alignment. So the built type agrees with rustc's byte
/// image exactly when every stored field's offset is naturally aligned and
/// non-overlapping in memory order, and `total_size` is a multiple of the
/// struct's natural alignment. `repr(packed)` breaks the former (a `u32` at
/// offset 1) and usually the latter (a 5-byte total), and either divergence
/// makes a byte-image copy land fields at the wrong bytes.
///
/// A struct with no recorded layout (`field_offsets` empty or `total_size`
/// zero) answers `true`: the lowering builds no padded layout to diverge
/// from, and the promotion path's stored-size agreement check already fails
/// closed on the unknown size. Any field whose natural size or alignment
/// cannot be established answers `false`: no verdict means no promotion.
fn struct_layout_matches_llvm_natural(
    ctx: &Context,
    field_types: &[TypeHandle],
    field_offsets: &[u64],
    mem_to_decl: &[usize],
    total_size: u64,
) -> bool {
    if field_offsets.is_empty() || total_size == 0 {
        return true;
    }
    if field_offsets.len() != field_types.len() {
        return false;
    }
    // Empty `mem_to_decl` means identity (declaration order = memory order).
    let identity: Vec<usize>;
    let memory_order: &[usize] = if mem_to_decl.is_empty() {
        identity = (0..field_types.len()).collect();
        &identity
    } else {
        mem_to_decl
    };

    let mut end: u64 = 0;
    let mut max_align: u64 = 1;
    for &decl_idx in memory_order {
        if decl_idx >= field_types.len() {
            return false;
        }
        let Some((size, align)) = llvm_natural_size_align(ctx, field_types[decl_idx]) else {
            return false;
        };
        // Zero-sized fields are stripped from the LLVM struct: no slot, no
        // bytes, no alignment contribution (over-aligned ZSTs are refused
        // before this walk runs).
        if size == 0 {
            continue;
        }
        let offset = field_offsets[decl_idx];
        if !offset.is_multiple_of(align) || offset < end {
            return false;
        }
        end = offset + size;
        max_align = max_align.max(align);
    }
    total_size >= end && total_size.is_multiple_of(max_align)
}

/// Natural size and alignment of the LLVM storage a dialect type converts to,
/// or `None` when the walk cannot establish them.
///
/// "Natural" means what LLVM's datalayout assigns the converted type with no
/// packing: leaves are their own width and self-aligned, arrays inherit their
/// element's alignment, and aggregates align to their most-aligned stored
/// field because the padding slots between fields are `[N x i8]` with
/// alignment one. Aggregates answer with rustc's `total_size` for their size;
/// that matches the built LLVM type only when their own layout is natural,
/// which [`struct_layout_matches_llvm_storage`] validates recursively (via
/// [`promotable_array_element`]) before the answer is trusted. A field-less
/// enum stores only its discriminant, so it aligns as that integer does.
fn llvm_natural_size_align(ctx: &Context, ty: TypeHandle) -> Option<(u64, u64)> {
    use dialect_mir::types::{MirArrayType, MirEnumType, MirPtrType, MirStructType, MirTupleType};

    let obj = ty.deref(ctx);
    if let Some(array) = obj.downcast_ref::<MirArrayType>() {
        let element_ty = array.element_type();
        let count = array.size();
        drop(obj);
        let (element_size, element_align) = llvm_natural_size_align(ctx, element_ty)?;
        return Some((element_size.checked_mul(count)?, element_align));
    }
    if let Some(tuple) = obj.downcast_ref::<MirTupleType>() {
        let fields = tuple.get_types().to_vec();
        let total_size = tuple.total_size();
        drop(obj);
        let align = aggregate_natural_align(ctx, &fields)?;
        return Some((total_size, align));
    }
    if let Some(structure) = obj.downcast_ref::<MirStructType>() {
        let fields = structure.field_types().to_vec();
        let total_size = structure.total_size;
        drop(obj);
        let align = aggregate_natural_align(ctx, &fields)?;
        return Some((total_size, align));
    }
    if let Some(enumeration) = obj.downcast_ref::<MirEnumType>() {
        let discriminant_ty = enumeration.discriminant_ty;
        let total_size = enumeration.total_size();
        drop(obj);
        let (_, align) = llvm_natural_size_align(ctx, discriminant_ty)?;
        return Some((total_size, align));
    }
    let size = if obj.is::<MirPtrType>() {
        // cuda-oxide lowers device code for 64-bit NVPTX. Keep this helper
        // independent of rustc's session TLS so it remains unit-testable.
        8
    } else if let Some(integer) = obj.downcast_ref::<IntegerType>() {
        // `bool` arrives as `i1` and occupies a byte.
        u64::from(integer.width().div_ceil(8)).max(1)
    } else if obj.is::<MirFP16Type>() {
        2
    } else if obj.is::<FP32Type>() {
        4
    } else if obj.is::<FP64Type>() {
        8
    } else {
        return None;
    };
    // Every scalar Rust hands this path is self-aligned at a power-of-two
    // width; anything else has no natural alignment to report.
    size.is_power_of_two().then_some((size, size))
}

/// Natural alignment of the LLVM struct built for an aggregate's fields:
/// the maximum over the stored (non-zero-sized) fields, one when nothing is
/// stored. `None` when some field's alignment cannot be established.
fn aggregate_natural_align(ctx: &Context, fields: &[TypeHandle]) -> Option<u64> {
    let mut align: u64 = 1;
    for &field in fields {
        let (field_size, field_align) = llvm_natural_size_align(ctx, field)?;
        if field_size > 0 {
            align = align.max(field_align);
        }
    }
    Some(align)
}

/// Bytes a dialect type occupies, as the converted LLVM storage will lay it out,
/// or `None` when that cannot be established from the type alone.
///
/// Aggregates answer with the `total_size` rustc gave them, which is exactly what
/// the storage builders reproduce: they pad to reach each field's recorded offset
/// and pad the tail to reach `total_size`. Leaves answer with their own width, and
/// `i1` counts as the one byte Rust gives a `bool` rather than an eighth of one.
///
/// A zero answer is reported as `None`: the dialect uses `total_size() == 0` both
/// for a genuine zero-sized type and for a size it does not know, and only the
/// caller's own size comparison could tell them apart.
fn dialect_stored_size(ctx: &Context, ty: TypeHandle) -> Option<u64> {
    use dialect_mir::types::{MirArrayType, MirEnumType, MirPtrType, MirStructType, MirTupleType};

    let obj = ty.deref(ctx);
    let size = if let Some(array) = obj.downcast_ref::<MirArrayType>() {
        let element = dialect_stored_size(ctx, array.element_type())?;
        element.checked_mul(array.size())?
    } else if let Some(tuple) = obj.downcast_ref::<MirTupleType>() {
        tuple.total_size()
    } else if let Some(structure) = obj.downcast_ref::<MirStructType>() {
        structure.total_size()
    } else if let Some(enumeration) = obj.downcast_ref::<MirEnumType>() {
        enumeration.total_size()
    } else if obj.is::<MirPtrType>() {
        // cuda-oxide lowers device code for 64-bit NVPTX. Keep this helper
        // independent of rustc's session TLS so it remains unit-testable.
        8
    } else if let Some(integer) = obj.downcast_ref::<IntegerType>() {
        // `bool` arrives as `i1` and occupies a byte.
        u64::from(integer.width().div_ceil(8)).max(1)
    } else if obj.is::<MirFP16Type>() {
        2
    } else if obj.is::<FP32Type>() {
        4
    } else if obj.is::<FP64Type>() {
        8
    } else {
        return None;
    };
    (size > 0).then_some(size)
}

/// Whether `local` is written exactly once — by the assignment being translated
/// — and never has an address handed out.
///
/// This chooses *which of two correct lowerings* to use, never correctness: both
/// give the local a private copy of the constant, and whether that copy is later
/// deleted is `opt`'s own sound decision. So an unrecognised write form here only
/// costs performance, which is why the statement match ends in a catch-all rather
/// than an error.
///
/// It has to exist because the two lowerings fail in opposite directions. When
/// the local really is read-only the copy disappears; when it is written the
/// copy survives, and NVPTX expands a surviving `memcpy` into a *byte* loop —
/// measurably worse than the element-wise stores it replaced. Rejecting every
/// borrow, not just mutable ones, keeps this on the shape that motivates it
/// (`TABLE[i]`, which projects a place and borrows nothing).
fn constant_local_is_written_once(body: &mir::Body, local: mir::Local) -> bool {
    let mut assignments = 0usize;
    for block in body.blocks.iter() {
        for statement in block.statements.iter() {
            match &statement.kind {
                mir::StatementKind::Assign(place, rvalue) => {
                    if place.local == local {
                        // A projected write is a write to part of the local.
                        if !place.projection.is_empty() {
                            return false;
                        }
                        assignments += 1;
                    }
                    // Any borrow or raw pointer, of either mutability, is a
                    // path this scan cannot follow.
                    match rvalue {
                        mir::Rvalue::Ref(_, _, source) if source.local == local => return false,
                        mir::Rvalue::AddressOf(_, source) if source.local == local => return false,
                        _ => {}
                    }
                }
                mir::StatementKind::SetDiscriminant { place, .. } if place.local == local => {
                    return false;
                }
                _ => {}
            }
        }
        // A call writes its destination.
        if let mir::TerminatorKind::Call { destination, .. } = &block.terminator.kind
            && destination.local == local
        {
            return false;
        }
    }
    assignments == 1
}

/// Fill an addressable local from a fully-constant array by copying an immutable
/// device global into it, instead of building the array in registers first.
///
/// `Ok(None)` means "not this shape" and the caller keeps the ordinary path.
///
/// # Why a copy and not just the global's address
///
/// The local is ordinary mutable storage — `let mut t = TABLE; t[0] = x;` is
/// legal — so handing out the global's address would be wrong in general. A
/// `memcpy` from `constant` storage is unconditionally correct instead, and
/// `opt` supplies the proof that removes it: `isOnlyCopiedFromConstantMemory`
/// deletes the copy and rewrites the reads to the global wherever the local is
/// never written.
///
/// [`constant_local_is_written_once`] gates the path on the same property, so
/// the two agree. It is not redundant: a copy that survives is *worse* than what
/// it replaced, because NVPTX expands a surviving `memcpy` into a byte loop.
///
/// # Why this is worth doing at all
///
/// The value form cannot reach the global no matter how it is spelled: LLVM
/// splits every first-class-aggregate load, so a whole-array load from the
/// global is folded straight back into one store per element. Only a `memcpy`
/// survives to the point where the proof can run.
///
/// What this replaces, for `TABLE[i]` with a runtime `i`, is one `st.local` per
/// element in *every thread* — a per-thread copy of data the module image
/// already carries — followed by `ld.local` reads of thread-private memory.
pub(crate) fn translate_array_constant_into_alloca(
    ctx: &mut Context,
    body: &mir::Body,
    dest_place: &mir::Place,
    constant: &mir::ConstOperand,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    use pliron::builtin::attributes::IntegerAttr;

    // Whole-local assignment only. With a projection the copy would have to land
    // at an offset, which the ordinary element-wise path already handles.
    if !dest_place.projection.is_empty() {
        return Ok(None);
    }
    let Some(dest_addr) = value_map.get_slot(dest_place.local) else {
        return Ok(None);
    };
    if !constant_local_is_written_once(body, dest_place.local) {
        return Ok(None);
    }

    let rust_ty = constant.const_.ty();
    let Ok(value_ty) = types::translate_type(ctx, &rust_ty) else {
        return Ok(None);
    };
    if !value_ty.deref(ctx).is::<dialect_mir::types::MirArrayType>() {
        return Ok(None);
    }
    // Elements whose whole-element read is a single scalar-like load, or whose
    // fields are each addressed in place: primitive scalars, field-less enums,
    // recursively promotable tuples and structs, and nested arrays of those.
    if !promotable_array_element(ctx, value_ty) {
        return Ok(None);
    }
    let Ok(expected_size) = rust_type_layout_size(rust_ty, loc.clone()) else {
        return Ok(None);
    };
    // The copy is sized from the destination's *converted* type, while the bytes
    // come from rustc's evaluated allocation. Require the two to agree before
    // trusting a byte image: a padded tuple or an enum reaches its layout through
    // `build_struct_slot_map` / `build_enum_slot_map`, and if either ever stopped
    // reproducing rustc's size, this is the check that keeps a byte image from
    // being copied over a differently-sized local.
    if dialect_stored_size(ctx, value_ty) != Some(expected_size as u64) {
        return Ok(None);
    }
    // Preserve both the evaluated byte image and any supported pointer
    // relocations. Unsupported relocation targets keep the existing element-wise
    // fallback rather than partially promoting the table.
    let Ok(initializer) = promoted_array_initializer(constant, expected_size, "array", loc.clone())
    else {
        return Ok(None);
    };
    let promoted_anchor = materialize_promoted_initializer_targets(
        ctx,
        &initializer.relocations,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Past every promotion bail-out: emit one immutable global and copy it into the local.
    let global_alloc = emit_promoted_immutable_global(
        ctx,
        value_ty,
        &initializer,
        block_ptr,
        promoted_anchor,
        loc.clone(),
    );
    let src = global_alloc.get_operation().deref(ctx).get_result(0);

    // `mir.memcpy`'s count is in destination-pointee elements, and the
    // destination slot points at the whole array, so one element is the whole
    // table.
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
    let count_attr = IntegerAttr::new(i64_ty, APInt::from_i64(1, NonZeroUsize::new(64).unwrap()));
    let count_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    count_op.deref_mut(ctx).set_loc(loc.clone());
    MirConstantOp::new(count_op).set_attr_value(ctx, count_attr);
    count_op.insert_after(ctx, global_alloc.get_operation());
    let count = count_op.deref(ctx).get_result(0);

    // Typed builder: stamps the elem_type fact from dest_addr (the whole
    // promoted aggregate type; count is 1), read back by lowering.
    let memcpy_op = dialect_mir::ops::MirMemcpyOp::build(ctx, dest_addr, src, count)?;
    memcpy_op.deref_mut(ctx).set_loc(loc);
    memcpy_op.insert_after(ctx, count_op);

    Ok(Some(memcpy_op))
}

/// Enforce the pointer-to-array constant element boundary, as a hard error.
///
/// The admission question is [`promotable_array_element`], the same predicate
/// used by the bare array value path's immutable-global optimization. A bare
/// array that fails this gate can still fall back to element-wise materialization;
/// a pointer-to-array constant has no such fallback, so failure here is an input
/// error. Structs pass only when every field is recursively promotable.
fn validate_ptr_to_array_constant_type(
    ctx: &Context,
    ty: TypeHandle,
    loc: Location,
) -> TranslationResult<()> {
    if promotable_array_element(ctx, ty) {
        return Ok(());
    }

    input_err!(
        loc,
        TranslationErr::unsupported(format!(
            "Array constant element type is not supported: {:?}. Supported promoted array constants are primitive scalars (integers, f16, f32, f64), thin pointers/references with supported static relocations, field-less enums, tuples and structs recursively composed of supported fields, or nested arrays of those.",
            ty.deref(ctx)
        ))
    )
}

pub(super) fn constant_allocation(
    constant: &mir::ConstOperand,
) -> Option<&rustc_public::ty::Allocation> {
    match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => Some(alloc),
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            rustc_public::ty::TyConstKind::Value(_, alloc) => Some(alloc),
            _ => None,
        },
        _ => None,
    }
}

fn promoted_constant_dedup_key(
    ctx: &Context,
    ty: TypeHandle,
    initializer: &GlobalInitializerData,
) -> String {
    let relocations = encode_global_initializer_relocations(&initializer.relocations);
    promoted_constant_dedup_key_from_parts(
        ctx,
        ty,
        &initializer.bytes,
        initializer.alignment,
        &relocations,
    )
}

fn promoted_constant_dedup_key_from_parts(
    ctx: &Context,
    ty: TypeHandle,
    bytes: &[u8],
    alignment: u64,
    relocations: &str,
) -> String {
    // This string is only an in-pass map key; it never becomes the emitted
    // symbol name. Keep the full type, byte image, allocation alignment, and
    // relocation encoding so constants with identical bytes but different
    // storage requirements or provenance targets cannot alias.
    let ty = ty.deref(ctx).disp(ctx).to_string();
    let bytes = bytes_to_hex(bytes);
    let fingerprint = format!(
        "__cuda_oxide_promoted_type{}:{ty}:bytes{}:{bytes}:align{alignment}:relocs{}:{relocations}",
        ty.len(),
        bytes.len() / 2,
        relocations.len()
    );
    dialect_mir::ops::encode_promoted_global_key(&fingerprint)
}

#[cfg(test)]
mod pointer_array_constant_type_tests {
    use super::validate_ptr_to_array_constant_type;
    use dialect_mir::types::{
        EnumVariant, MirArrayType, MirEnumType, MirPtrType, MirStructType, MirTupleType,
    };
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::context::Context;
    use pliron::location::Location;
    use pliron::r#type::TypeHandle;

    #[test]
    fn pointer_array_constant_boundary_admits_recursive_promotable_aggregates() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let primitive_array: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 3).into();
        let nested_primitive_array: TypeHandle =
            MirArrayType::get(&mut ctx, primitive_array, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, nested_primitive_array, Location::Unknown)
                .is_ok(),
            "recursively nested primitive arrays remain supported"
        );

        let struct_ty: TypeHandle = MirStructType::get(
            &mut ctx,
            "PointerArrayElement".into(),
            vec!["value".into()],
            vec![u32_ty],
        )
        .into();
        let struct_array: TypeHandle = MirArrayType::get(&mut ctx, struct_ty, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, struct_array, Location::Unknown).is_ok(),
            "pointer-to-array constants admit structs whose fields are promotable"
        );

        // The reference form shares the immutable-promotion gate. A packed
        // struct whose selected packed LLVM storage reproduces rustc exactly
        // must therefore be accepted here as well.
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let packed_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedPointerArrayElement".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let packed_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, packed_struct_ty, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, packed_struct_array, Location::Unknown,)
                .is_ok(),
            "const R: &[Packed; N] = &TABLE must share the packed immutable global"
        );

        let nested_struct_array: TypeHandle = MirArrayType::get(&mut ctx, struct_array, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, nested_struct_array, Location::Unknown)
                .is_ok(),
            "nesting preserves a promotable struct leaf"
        );

        // The shared predicate gates both bare-value promotion and this
        // reference form. The initializer is rustc's evaluated byte image and
        // the size-agreement check rejects any layout the dialect reproduces
        // differently, so recursive tuples and structs travel this path without
        // rebuilding fields here.
        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![u32_ty]).into();
        let tuple_array: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, tuple_array, Location::Unknown).is_ok(),
            "const R: &[(u32,); N] = &TABLE must pass the same gate the bare table passes"
        );

        // A tuple containing a promotable struct is recursively admissible too.
        let tuple_with_struct_ty: TypeHandle =
            MirTupleType::get(&mut ctx, vec![u32_ty, struct_ty]).into();
        let tuple_with_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, tuple_with_struct_ty, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, tuple_with_struct_array, Location::Unknown)
                .is_ok(),
            "a promotable struct field keeps its tuple in the reference form"
        );

        // Pointer-bearing structs are promotable now that immutable globals
        // preserve rustc relocation metadata instead of flattening pointer
        // placeholder bytes.
        let pointer_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let pointer_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PointerBearingElement".into(),
            vec!["pointer".into()],
            vec![pointer_ty],
            vec![],
            vec![0],
            8,
            8,
        )
        .into();
        let pointer_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, pointer_struct_ty, 2).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, pointer_struct_array, Location::Unknown)
                .is_ok(),
            "pointer-bearing structs are admitted by relocation-aware promotion"
        );
    }

    #[test]
    fn pointer_array_constant_boundary_matches_the_bare_array_gate_for_enums() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let fieldless_variants = vec!["Add", "Sub"]
            .into_iter()
            .map(|name| EnumVariant {
                name: name.into(),
                field_types: vec![],
                field_offsets: vec![],
                field_sizes: vec![],
            })
            .collect();
        let op_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Op".into(),
            u32_ty,
            vec![0, 1],
            fieldless_variants,
            0,
            4,
            4,
        )
        .into();
        let op_array: TypeHandle = MirArrayType::get(&mut ctx, op_ty, 8).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, op_array, Location::Unknown).is_ok(),
            "const R: &[Op; N] = &OPS must pass the same gate the bare table passes"
        );

        let payload_variants = vec![
            EnumVariant {
                name: "None".into(),
                field_types: vec![],
                field_offsets: vec![],
                field_sizes: vec![],
            },
            EnumVariant {
                name: "Some".into(),
                field_types: vec![u32_ty],
                field_offsets: vec![4],
                field_sizes: vec![4],
            },
        ];
        let maybe_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Maybe".into(),
            u32_ty,
            vec![0, 1],
            payload_variants,
            0,
            8,
            4,
        )
        .into();
        let maybe_array: TypeHandle = MirArrayType::get(&mut ctx, maybe_ty, 4).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, maybe_array, Location::Unknown).is_err(),
            "a payload-carrying enum stays out of the reference form too"
        );

        // The `&[]` shape: a promoted empty-slice constant keeps its
        // any-element admission now that the check routes through the shared
        // predicate.
        let struct_ty: TypeHandle = MirStructType::get(
            &mut ctx,
            "EmptySliceElement".into(),
            vec!["value".into()],
            vec![u32_ty],
        )
        .into();
        let empty_struct_array: TypeHandle = MirArrayType::get(&mut ctx, struct_ty, 0).into();
        assert!(
            validate_ptr_to_array_constant_type(&ctx, empty_struct_array, Location::Unknown)
                .is_ok(),
            "a zero-length array has nothing readable, whatever its element type"
        );
    }
}

#[cfg(test)]
mod promotable_array_element_tests {
    use super::super::const_alloc::validate_array_value_element_type;
    use super::{dialect_stored_size, promotable_array_element, promoted_global_validation_type};
    use dialect_mir::types::{
        EnumVariant, MirArrayType, MirEnumType, MirPtrType, MirStructType, MirTupleType,
    };
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::context::Context;
    use pliron::location::Location;
    use pliron::r#type::TypeHandle;

    /// A field-less enum with a recorded layout, as rustc gives a `#[repr(u32)]`
    /// C-like enum.
    fn fieldless_enum(ctx: &mut Context, u32_ty: TypeHandle) -> TypeHandle {
        let variants = vec!["Add", "Sub"]
            .into_iter()
            .map(|name| EnumVariant {
                name: name.into(),
                field_types: vec![],
                field_offsets: vec![],
                field_sizes: vec![],
            })
            .collect();
        MirEnumType::get_with_layout(ctx, "Op".into(), u32_ty, vec![0, 1], variants, 0, 4, 4).into()
    }

    /// The same shape, except one variant carries a payload.
    fn payload_enum(ctx: &mut Context, u32_ty: TypeHandle) -> TypeHandle {
        let variants = vec![
            EnumVariant {
                name: "None".into(),
                field_types: vec![],
                field_offsets: vec![],
                field_sizes: vec![],
            },
            EnumVariant {
                name: "Some".into(),
                field_types: vec![u32_ty],
                field_offsets: vec![4],
                field_sizes: vec![4],
            },
        ];
        MirEnumType::get_with_layout(ctx, "Maybe".into(), u32_ty, vec![0, 1], variants, 0, 8, 4)
            .into()
    }

    #[test]
    fn promotion_admits_recursive_promotable_aggregates() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let scalars: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, scalars),
            "a scalar array is the shape this already promoted"
        );
        let nested: TypeHandle = MirArrayType::get(&mut ctx, scalars, 2).into();
        assert!(
            promotable_array_element(&ctx, nested),
            "nesting must not lose a promotable leaf"
        );

        let op_ty = fieldless_enum(&mut ctx, u32_ty);
        let op_array: TypeHandle = MirArrayType::get(&mut ctx, op_ty, 8).into();
        assert!(
            promotable_array_element(&ctx, op_array),
            "a field-less enum element is its discriminant, so one load reads it"
        );
        let nested_ops: TypeHandle = MirArrayType::get(&mut ctx, op_array, 2).into();
        assert!(
            promotable_array_element(&ctx, nested_ops),
            "nested field-less enum arrays stay promotable"
        );

        // Payload-carrying enums still use a value path that round-trips their
        // payload through memory, so they remain outside immutable-global promotion.
        let maybe_ty = payload_enum(&mut ctx, u32_ty);
        let maybe_array: TypeHandle = MirArrayType::get(&mut ctx, maybe_ty, 4).into();
        assert!(
            !promotable_array_element(&ctx, maybe_array),
            "a payload-carrying enum round-trips its payload through memory"
        );

        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![u32_ty, u32_ty]).into();
        let tuple_array: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, tuple_array),
            "a tuple of promotable fields is promotable now that a field read \
             addresses in place instead of copying the array"
        );
        let nested_tuple_arrays: TypeHandle = MirArrayType::get(&mut ctx, tuple_array, 2).into();
        assert!(
            promotable_array_element(&ctx, nested_tuple_arrays),
            "nesting must not lose a promotable tuple leaf"
        );

        let struct_ty: TypeHandle = MirStructType::get(
            &mut ctx,
            "Element".into(),
            vec!["value".into()],
            vec![u32_ty],
        )
        .into();
        let struct_array: TypeHandle = MirArrayType::get(&mut ctx, struct_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, struct_array),
            "a struct of promotable fields is promotable"
        );

        // Recursive aggregates remain promotable when every leaf is promotable.
        let tuple_with_struct: TypeHandle =
            MirTupleType::get(&mut ctx, vec![u32_ty, struct_ty]).into();
        let tuple_with_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, tuple_with_struct, 4).into();
        assert!(
            promotable_array_element(&ctx, tuple_with_struct_array),
            "a promotable struct field keeps its tuple promotable"
        );
        let nested_struct_arrays: TypeHandle =
            MirArrayType::get(&mut ctx, tuple_with_struct_array, 2).into();
        assert!(
            promotable_array_element(&ctx, nested_struct_arrays),
            "nesting preserves recursive promotability"
        );

        // A zero-byte over-aligned struct carries an ABI constraint that is not
        // visible in the lowered storage shape. Preserve the established
        // alignment-sensitive value path for aggregates containing such a leaf.
        let over_aligned_zst_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OverAlignedZst".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        let tuple_with_over_aligned_zst: TypeHandle =
            MirTupleType::get(&mut ctx, vec![over_aligned_zst_ty, u32_ty]).into();
        let over_aligned_zst_array: TypeHandle =
            MirArrayType::get(&mut ctx, tuple_with_over_aligned_zst, 2).into();
        assert!(
            !promotable_array_element(&ctx, over_aligned_zst_array),
            "a zero-byte over-aligned struct must keep its containing aggregate on the alignment-sensitive path"
        );

        // Thin pointers are scalar load leaves once their initializer
        // provenance is preserved as relocation metadata. Pin their concrete
        // storage size too, so admission cannot later fail the byte-size gate.
        let pointer_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let pointer_array: TypeHandle = MirArrayType::get(&mut ctx, pointer_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, pointer_array),
            "a thin pointer is itself a promotable scalar load leaf"
        );
        assert_eq!(
            dialect_stored_size(&ctx, pointer_array),
            Some(32),
            "four NVPTX64 pointers occupy 32 bytes"
        );

        // Use a full rustc-style layout so this assertion exercises
        // struct_layout_matches_llvm_natural instead of its unknown-layout
        // early return.
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointer_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PointerBearingElement".into(),
            vec!["pointer".into(), "tag".into()],
            vec![pointer_ty, u8_ty],
            vec![],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let pointer_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, pointer_struct_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, pointer_struct_array),
            "a naturally laid out struct containing a thin pointer is promotable"
        );

        let struct_with_payload_enum: TypeHandle = MirStructType::get(
            &mut ctx,
            "PayloadEnumElement".into(),
            vec!["value".into()],
            vec![maybe_ty],
        )
        .into();
        let payload_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, struct_with_payload_enum, 4).into();
        assert!(
            !promotable_array_element(&ctx, payload_struct_array),
            "a payload-enum leaf must keep its containing struct out of promotion"
        );
    }

    #[test]
    fn packed_struct_promotion_requires_byte_faithful_storage() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        // `#[repr(C, packed)] struct Packed { tag: u8, value: u32 }`: rustc
        // records `value` at offset 1 and a five-byte total. The natural LLVM
        // struct diverges, but the packed LLVM representation is exactly
        // `<{ i8, i32 }>` and therefore preserves both offsets and stride.
        let packed_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let packed_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, packed_struct_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, packed_struct_array),
            "a byte-faithful packed struct must use immutable-global promotion"
        );
        assert_eq!(
            dialect_stored_size(&ctx, packed_struct_array),
            Some(20),
            "four packed five-byte elements must retain rustc's stride"
        );

        let empty_packed_array: TypeHandle =
            MirArrayType::get(&mut ctx, packed_struct_ty, 0).into();
        assert!(
            promotable_array_element(&ctx, empty_packed_array),
            "a zero-length packed array remains vacuously promotable"
        );
        assert_eq!(
            promoted_global_validation_type(&mut ctx, empty_packed_array, 0),
            empty_packed_array,
            "zero-byte promotion must retain the semantic [Packed; 0] type"
        );
        let packed_byte_len = dialect_stored_size(&ctx, packed_struct_array).unwrap() as usize;
        let packed_validation =
            promoted_global_validation_type(&mut ctx, packed_struct_array, packed_byte_len);
        let packed_validation_ref = packed_validation.deref(&ctx);
        let packed_validation_array = packed_validation_ref
            .downcast_ref::<MirArrayType>()
            .expect("packed validation storage must be an array");
        assert_eq!(packed_validation_array.size, 20);
        assert_eq!(packed_validation_array.element_ty, u8_ty);
        drop(packed_validation_ref);
        // `repr(C, packed(2))` keeps one explicit byte between the fields and
        // a six-byte stride. Natural LLVM still cannot place the u32 at offset
        // two, while a packed struct plus one byte padding slot can.
        let packed2_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed2".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 2],
            6,
            2,
        )
        .into();
        let packed2_array: TypeHandle = MirArrayType::get(&mut ctx, packed2_struct_ty, 3).into();
        assert!(
            promotable_array_element(&ctx, packed2_array),
            "packed(2) must be admitted when explicit padding reproduces rustc exactly"
        );
        assert_eq!(
            dialect_stored_size(&ctx, packed2_array),
            Some(18),
            "packed(2) array promotion must preserve the six-byte element stride"
        );

        // A naturally laid-out outer struct may contain a packed child. The
        // child's selected storage is already byte-faithful, so recursive
        // promotion must not re-impose the child's natural alignment.
        let outer_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OuterWithPacked".into(),
            vec!["head".into(), "inner".into()],
            vec![u32_ty, packed_struct_ty],
            vec![],
            vec![0, 4],
            12,
            4,
        )
        .into();
        let outer_array: TypeHandle = MirArrayType::get(&mut ctx, outer_ty, 2).into();
        assert!(
            promotable_array_element(&ctx, outer_array),
            "a recursively byte-faithful packed child must remain promotable"
        );

        // Synthetic overlapping field ranges model layouts that no sequential
        // natural or packed LLVM struct can express. Keep this fail-closed even
        // though the individual leaves are promotable.
        let overlapping_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Overlapping".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 0],
            4,
            1,
        )
        .into();
        let overlapping_array: TypeHandle = MirArrayType::get(&mut ctx, overlapping_ty, 2).into();
        assert!(
            !promotable_array_element(&ctx, overlapping_array),
            "overlapping storage must remain outside immutable-global promotion"
        );

        // Bare-array admission stays broader than promotion, so a layout that
        // fails the promotion gate still has its existing element-wise fallback.
        assert!(
            validate_array_value_element_type(&ctx, overlapping_ty, &Location::Unknown).is_ok(),
            "the bare-array fallback must remain available for an unpromotable struct"
        );

        let natural_struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Natural".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 4],
            8,
            4,
        )
        .into();
        let natural_struct_array: TypeHandle =
            MirArrayType::get(&mut ctx, natural_struct_ty, 4).into();
        assert!(
            promotable_array_element(&ctx, natural_struct_array),
            "natural-layout struct promotion must remain unchanged"
        );
    }

    /// Coupling oracle: every struct layout the promotion gate admits must be
    /// one mir-lower actually lowers byte-faithfully (natural, or packed with
    /// explicit padding). The promotion gate swaps the global's validation
    /// type to a byte view, so this agreement is the only thing standing
    /// between an admitted layout and typed reads through a divergent
    /// natural-layout fallback. If this test fails after changing either
    /// side's layout walk, the two predicates have drifted.
    #[test]
    fn promoted_struct_layouts_agree_with_the_mir_lower_storage_selection() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        // (name, offsets, total_size, align) for each struct shape the
        // promotion test admits; every one must be byte-faithful downstream.
        let admitted = [
            ("Packed", vec![0u64, 1], 5u64, 1u64),
            ("Packed2", vec![0, 2], 6, 2),
            ("Natural", vec![0, 4], 8, 4),
        ];
        for (name, offsets, total_size, align) in admitted {
            let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
                &mut ctx,
                name.into(),
                vec!["tag".into(), "value".into()],
                vec![u8_ty, u32_ty],
                vec![],
                offsets,
                total_size,
                align,
            )
            .into();
            let array_ty: TypeHandle = MirArrayType::get(&mut ctx, struct_ty, 2).into();
            assert!(
                promotable_array_element(&ctx, array_ty),
                "`{name}` must stay in the promotion corpus this test guards"
            );
            assert!(
                mir_lower::convert::types::struct_value_lowering_is_byte_faithful(
                    &mut ctx, struct_ty
                )
                .expect("slot map must build for an admitted layout"),
                "`{name}` is admitted for promotion but mir-lower would fall \
                 back to a divergent natural layout; the two layout walks have \
                 drifted"
            );
        }

        // The known-unfaithful shape must stay rejected on BOTH sides.
        let overlapping_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Overlapping".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![],
            vec![0, 0],
            4,
            1,
        )
        .into();
        let overlapping_array: TypeHandle = MirArrayType::get(&mut ctx, overlapping_ty, 2).into();
        assert!(
            !promotable_array_element(&ctx, overlapping_array),
            "overlapping storage must remain outside promotion"
        );
        assert!(
            !mir_lower::convert::types::struct_value_lowering_is_byte_faithful(
                &mut ctx,
                overlapping_ty
            )
            .expect("slot map must still build for the overlapping layout"),
            "mir-lower must agree that overlapping storage is not byte-faithful"
        );
    }

    #[test]
    fn an_enum_without_a_recorded_layout_is_not_promoted() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        // `total_size` 0 means the layout was never recorded, so the byte image
        // has nothing to be checked against.
        let variants = vec![EnumVariant {
            name: "Only".into(),
            field_types: vec![],
            field_offsets: vec![],
            field_sizes: vec![],
        }];
        let unsized_enum: TypeHandle =
            MirEnumType::get(&mut ctx, "Unknown".into(), u32_ty, vec![0], variants).into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, unsized_enum, 4).into();
        assert!(
            !promotable_array_element(&ctx, array),
            "an enum with no recorded size must not be promoted"
        );
    }
}

#[cfg(test)]
mod promoted_global_tests {
    use super::*;

    #[test]
    fn promoted_immutable_globals_are_marked_and_dedup_by_type_and_initializer() {
        use dialect_mir::types::MirArrayType;
        use pliron::builtin::ops::ModuleOp;
        use pliron::linked_list::ContainsLinkedList;

        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "promoted_globals".try_into().unwrap());
        let module_region = module.get_operation().deref(&ctx).get_region(0);
        let block = {
            let existing = module_region.deref(&ctx).iter(&ctx).next();
            match existing {
                Some(block) => block,
                None => {
                    let block = BasicBlock::new(&mut ctx, None, vec![]);
                    block.insert_at_back(module_region, &ctx);
                    block
                }
            }
        };

        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let table_ty: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 2).into();
        let bytes = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40]; // [1.0f32, 2.0f32]
        let initializer = GlobalInitializerData {
            bytes,
            alignment: 4,
            relocations: Vec::new(),
        };

        let first = emit_promoted_immutable_global(
            &mut ctx,
            table_ty,
            &initializer,
            block,
            None,
            Location::Unknown,
        );

        // The marker is the point: without it the exporter writes `global` and
        // `opt` may not delete the per-thread copy this path exists to remove.
        assert!(
            first.is_immutable(&ctx),
            "promoted global must be immutable"
        );
        let first_key = String::from(first.get_attr_global_key(&ctx).expect("dedup key").clone());

        // The same table reached again, through another function or spelling,
        // must produce the same key.
        let again = emit_promoted_immutable_global(
            &mut ctx,
            table_ty,
            &initializer,
            block,
            None,
            Location::Unknown,
        );
        let again_key = String::from(again.get_attr_global_key(&ctx).expect("dedup key").clone());
        assert_eq!(
            first_key, again_key,
            "same type, bytes, and relocations must share a key"
        );

        // A different byte image must not alias.
        let other_initializer = GlobalInitializerData {
            bytes: vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f],
            alignment: 4,
            relocations: Vec::new(),
        };
        let other = emit_promoted_immutable_global(
            &mut ctx,
            table_ty,
            &other_initializer,
            block,
            None,
            Location::Unknown,
        );
        let other_key = String::from(other.get_attr_global_key(&ctx).expect("dedup key").clone());
        assert_ne!(first_key, other_key, "distinct bytes must not share a key");
    }

    #[test]
    fn promoted_dedup_key_distinguishes_relocation_identity() {
        use dialect_mir::types::MirArrayType;

        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let table_ty: TypeHandle = MirArrayType::get(&mut ctx, i64_ty, 1).into();
        let bytes = [0u8; 8];

        // Pointer placeholder bytes and addends may be identical. The target key
        // is rustc's provenance identity and must therefore participate in the
        // promoted-global dedup key.
        let target_a = "v1 1 0 8 1 0 8 target_a ";
        let target_b = "v1 1 0 8 1 0 8 target_b ";

        let key_a = promoted_constant_dedup_key_from_parts(&ctx, table_ty, &bytes, 8, target_a);
        let key_b = promoted_constant_dedup_key_from_parts(&ctx, table_ty, &bytes, 8, target_b);
        assert_ne!(
            key_a, key_b,
            "identical bytes with different relocation targets must not alias"
        );

        let differently_aligned =
            promoted_constant_dedup_key_from_parts(&ctx, table_ty, &bytes, 16, target_a);
        assert_ne!(
            key_a, differently_aligned,
            "identical type, bytes, and relocations with different allocation alignment must not alias"
        );
    }
}
