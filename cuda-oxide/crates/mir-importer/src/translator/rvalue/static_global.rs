/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static global materialization state machine.

use super::const_union::validate_device_static_union_initializer;
use super::promoted_alloc::ensure_promoted_allocation_global;
use super::statics::{
    GlobalInitializerRelocation, RelocationTarget, bytes_to_hex,
    encode_global_initializer_relocations, require_initialized_static_storage, static_global_key,
    static_initializer_data,
};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use dialect_mir::ops::{MirCastOp, MirConstantOp, MirGlobalAllocOp, MirPtrOffsetOp};
use dialect_mir::types::MirPointerKind;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_public::CrateDef;
use rustc_public::CrateDefType;
use std::num::NonZeroUsize;

/// One device global emitted for a Rust static or an anonymous rustc
/// allocation, as seen by the function that materialized it.
#[derive(Clone, Copy)]
pub(super) struct MaterializedStaticGlobal {
    pub(super) base_ptr: Value,
    pub(super) global_op: Ptr<Operation>,
    pub(super) allocation_size: u64,
}

impl MaterializedStaticGlobal {
    /// Record a newly emitted `mir.global_alloc` and its allocation size.
    pub(super) fn new(global_op: Ptr<Operation>, base_ptr: Value, allocation_size: u64) -> Self {
        Self {
            base_ptr,
            global_op,
            allocation_size,
        }
    }
}

/// Globals already emitted for one constant operand, keyed by `global_key`.
///
/// Registering a global before walking its initializer's relocations keeps
/// self-referential and mutually recursive allocation graphs finite. Lowering
/// still deduplicates by key across the whole module.
pub(super) struct StaticGlobalMaterializationState {
    pub(super) globals: std::collections::HashMap<String, MaterializedStaticGlobal>,
    pub(super) last_op: Option<Ptr<Operation>>,
}

/// Materialize one device static and every allocation reachable from its
/// initializer.
///
/// The current global is registered before traversing its relocations. That
/// makes self-references and mutually recursive static graphs finite while the
/// lowering pass still performs module-wide deduplication by `global_key`.
fn ensure_static_global_alloc(
    ctx: &mut Context,
    static_def: &rustc_public::mir::mono::StaticDef,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    loc: Location,
    state: &mut StaticGlobalMaterializationState,
) -> TranslationResult<MaterializedStaticGlobal> {
    let global_key = static_global_key(static_def);
    if let Some(existing) = state.globals.get(&global_key) {
        return Ok(*existing);
    }

    let address_space =
        require_initialized_static_storage(static_def, "a constant pointer", loc.clone())?;
    let initializer = static_initializer_data(static_def, loc.clone())?;
    let allocation_size = initializer.bytes.len() as u64;
    let initializer_hex = bytes_to_hex(&initializer.bytes);
    let static_ty = static_def.ty();
    let global_ty = types::translate_type(ctx, &static_ty)?;

    if let Some(union_name) = stored_type_union_name(static_ty, &mut Vec::new()) {
        if global_ty
            .deref(ctx)
            .is::<dialect_mir::types::MirUnionType>()
        {
            validate_device_static_union_initializer(
                ctx,
                static_def,
                global_ty,
                &initializer,
                loc.clone(),
            )?;
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "device static {} contains nested union `{union_name}`; device-global \
                     union initializer relocations are supported only for a top-level \
                     thin-pointer union",
                    static_def.name()
                ))
            );
        }
    }
    let global_ptr_ty: TypeHandle =
        dialect_mir::types::MirPtrType::get(ctx, global_ty, is_mutable, address_space).into();

    let global_op = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_ptr_ty],
        vec![],
        vec![],
        0,
    );
    global_op.deref_mut(ctx).set_loc(loc.clone());

    let global_alloc = MirGlobalAllocOp::new(global_op);

    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    global_alloc.set_attr_global_type(ctx, TypeAttr::new(global_ty));
    global_alloc.set_attr_global_key(ctx, StringAttr::new(global_key.clone()));
    set_global_initializer_hex_attr(ctx, global_alloc.get_operation(), &initializer_hex);
    if !initializer.relocations.is_empty() {
        let encoded = encode_global_initializer_relocations(&initializer.relocations);
        set_global_initializer_relocations_attr(ctx, global_alloc.get_operation(), &encoded);
    }

    if initializer.alignment > 0 {
        global_alloc.set_alignment_value(ctx, initializer.alignment);
    }

    match state.last_op {
        Some(previous) => global_alloc.get_operation().insert_after(ctx, previous),
        None => global_alloc.get_operation().insert_at_front(block_ptr, ctx),
    }
    state.last_op = Some(global_alloc.get_operation());

    let materialized = MaterializedStaticGlobal::new(
        global_alloc.get_operation(),
        global_alloc.get_operation().deref(ctx).get_result(0),
        allocation_size,
    );
    state.globals.insert(global_key, materialized);

    let owner_description = format!("device static {}", static_def.name());
    ensure_initializer_relocation_targets(
        ctx,
        &initializer.relocations,
        &owner_description,
        block_ptr,
        loc,
        state,
    )?;

    Ok(materialized)
}

/// Materialize and validate every allocation an initializer points into.
///
/// Rust statics get their own globals; anonymous rustc allocations get the
/// immutable per-allocation global from [`ensure_promoted_allocation_global`].
/// The owner is already registered before recursion reaches this helper, so
/// self-references and mutually recursive graphs remain finite.
pub(super) fn ensure_initializer_relocation_targets(
    ctx: &mut Context,
    relocations: &[GlobalInitializerRelocation],
    owner_description: &str,
    block_ptr: Ptr<BasicBlock>,
    loc: Location,
    state: &mut StaticGlobalMaterializationState,
) -> TranslationResult<()> {
    for relocation in relocations {
        let target = match relocation.target {
            RelocationTarget::Static(static_def) => {
                ensure_static_global_alloc(ctx, &static_def, false, block_ptr, loc.clone(), state)?
            }
            RelocationTarget::Memory(alloc_id) => ensure_promoted_allocation_global(
                ctx,
                alloc_id,
                None,
                block_ptr,
                loc.clone(),
                state,
            )?,
        };
        if relocation.target_addend > target.allocation_size {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{owner_description} relocation at byte {} points {} bytes into {}, but the target allocation is only {} bytes",
                    relocation.source_offset,
                    relocation.target_addend,
                    relocation.target.describe(),
                    target.allocation_size
                ))
            );
        }
    }
    Ok(())
}

pub(super) fn translate_static_global_pointer(
    ctx: &mut Context,
    static_def: &rustc_public::mir::mono::StaticDef,
    result_pointee_ty: TypeHandle,
    result_ptr_ty: TypeHandle,
    is_mutable: bool,
    byte_offset: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let mut state = StaticGlobalMaterializationState {
        globals: std::collections::HashMap::new(),
        last_op: prev_op,
    };
    let materialized = ensure_static_global_alloc(
        ctx,
        static_def,
        is_mutable,
        block_ptr,
        loc.clone(),
        &mut state,
    )?;

    pointer_into_materialized_global(
        ctx,
        materialized,
        state.last_op.unwrap_or(materialized.global_op),
        &format!("device static {}", static_def.name()),
        result_pointee_ty,
        result_ptr_ty,
        byte_offset,
        loc,
    )
}

/// Form the constant pointer `&global + byte_offset`, typed as the Rust operand.
///
/// `materialized` is a device global already emitted for a Rust static or an
/// anonymous rustc allocation, and `insert_after` is the last op emitted for it
/// (its relocation targets included). `description` names the allocation in
/// diagnostics.
///
/// The address arithmetic keeps the global's own mutability, because a
/// generic cast cannot change it; only the final retype establishes the
/// operand's pointer kind (see [`retype_static_pointer_result`]).
pub(super) fn pointer_into_materialized_global(
    ctx: &mut Context,
    materialized: MaterializedStaticGlobal,
    insert_after: Ptr<Operation>,
    description: &str,
    result_pointee_ty: TypeHandle,
    result_ptr_ty: TypeHandle,
    byte_offset: u64,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    // Rust const evaluation permits forming a pointer one past the end of
    // an allocation (offset == allocation size); only offsets strictly
    // beyond the allocation are impossible for rustc to have produced.
    // Forming the pointer is what is translated here, so the check must
    // not add a pointee-size term.
    if byte_offset > materialized.allocation_size {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "constant pointer to {description} has byte offset {byte_offset}, \
                 but the allocation is only {} bytes",
                materialized.allocation_size
            ))
        );
    }

    let base_ptr = materialized.base_ptr;

    let (base_pointee_ty, address_space, is_mutable) = {
        let base_ty = base_ptr.get_type(ctx);
        let base_ty = base_ty.deref(ctx);
        let base_ptr_ty = base_ty
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .expect("MirGlobalAllocOp must return MirPtrType");
        (
            base_ptr_ty.pointee,
            base_ptr_ty.address_space,
            base_ptr_ty.is_mutable,
        )
    };

    // Preserve the existing direct path. It avoids generating unnecessary
    // casts and pointer arithmetic for ordinary references to whole statics.
    // The result is still normalized to the exact translated Rust operand
    // type: slot stores and mem2reg are type-strict, so the physical
    // address-space pointer must not leak into the function body. A zero-addend
    // relocation may still name the first element of an array static (or an
    // equivalent differently typed view). Keep those on the byte-addressed
    // path below so `StaticAddress` establishes only the final kind/address
    // space after generic Erased casts have normalized the pointee shape.
    if byte_offset == 0 && base_pointee_ty == result_pointee_ty {
        let (result, last_op) =
            retype_static_pointer_result(ctx, base_ptr, result_ptr_ty, insert_after, loc);
        return Ok((result, Some(last_op)));
    }

    // mir.ptr_offset scales by sizeof(pointee). Cast to u8 first so the
    // rustc addend is interpreted as bytes rather than static elements.
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();

    let byte_ptr_ty: TypeHandle =
        dialect_mir::types::MirPtrType::get(ctx, byte_ty, is_mutable, address_space).into();

    // The address arithmetic stays in the static's physical address space
    // (LLVM GEPs cannot change address spaces); the exact-Rust-type
    // normalization happens once, at the end.
    let interior_ptr_ty: TypeHandle =
        dialect_mir::types::MirPtrType::get(ctx, result_pointee_ty, is_mutable, address_space)
            .into();

    // 1. *StaticType addrspace(N) -> *u8 addrspace(N)
    let to_byte_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![byte_ptr_ty],
        vec![base_ptr],
        vec![],
        0,
    );
    to_byte_op.deref_mut(ctx).set_loc(loc.clone());

    let to_byte_cast = MirCastOp::new(to_byte_op);
    to_byte_cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    to_byte_cast.get_operation().insert_after(ctx, insert_after);

    let byte_ptr = to_byte_cast.get_operation().deref(ctx).get_result(0);

    // 2. Materialize the rustc byte addend as usize.
    let offset_ty = types::get_usize_type(ctx);
    let offset_attr = pliron::builtin::attributes::IntegerAttr::new(
        offset_ty,
        APInt::from_u64(
            byte_offset,
            NonZeroUsize::new(64).expect("usize must have non-zero width"),
        ),
    );

    let offset_const_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![offset_ty.into()],
        vec![],
        vec![],
        0,
    );
    offset_const_op.deref_mut(ctx).set_loc(loc.clone());

    let offset_const = MirConstantOp::new(offset_const_op);
    offset_const.set_attr_value(ctx, offset_attr);
    offset_const
        .get_operation()
        .insert_after(ctx, to_byte_cast.get_operation());

    let offset_value = offset_const.get_operation().deref(ctx).get_result(0);

    // 3. Apply the addend. Since the pointer now points to u8, one element
    // equals exactly one byte.
    let ptr_offset_op = Operation::new(
        ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![byte_ptr_ty],
        vec![byte_ptr, offset_value],
        vec![],
        0,
    );
    ptr_offset_op.deref_mut(ctx).set_loc(loc.clone());
    ptr_offset_op.insert_after(ctx, offset_const.get_operation());

    let offset_byte_ptr = ptr_offset_op.deref(ctx).get_result(0);

    // 4. *u8 addrspace(N) -> *ResultPointee addrspace(N)
    let result_cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![interior_ptr_ty],
        vec![offset_byte_ptr],
        vec![],
        0,
    );
    result_cast_op.deref_mut(ctx).set_loc(loc.clone());

    let result_cast = MirCastOp::new(result_cast_op);
    result_cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    result_cast.get_operation().insert_after(ctx, ptr_offset_op);

    // 5. Normalize to the exact translated Rust operand type (lowering
    // emits an `addrspacecast` when the address spaces differ).
    let result = result_cast.get_operation().deref(ctx).get_result(0);
    let (result, last_op) =
        retype_static_pointer_result(ctx, result, result_ptr_ty, result_cast.get_operation(), loc);
    Ok((result, Some(last_op)))
}

/// Retype a materialized static-pointer `value` to the exact translated Rust
/// operand type.
///
/// `MirGlobalAllocOp` results (and interior-pointer arithmetic built on
/// them) carry the static's physical address space, but slot stores and
/// mem2reg are type-strict: the constant operand must have the exact
/// translated Rust type. Lowering turns this `PtrToPtr` cast into an
/// `addrspacecast` when the address spaces differ, which
/// `InferAddressSpaces` later folds back through for direct loads.
///
/// `StaticAddress` grants `*mut T` only over writable storage. A `*mut T` into
/// read-only storage (`&42 as *const u32 as *mut u32`) is still a valid Rust
/// value, formed by an `as` cast, so it is retyped as `*const T` first and then
/// cast raw-to-raw under `RustCast`, the authority for that source cast. The
/// storage stays read-only; only the pointer's kind changes.
fn retype_static_pointer_result(
    ctx: &mut Context,
    value: Value,
    result_ptr_ty: TypeHandle,
    insert_after: Ptr<Operation>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    if value.get_type(ctx) == result_ptr_ty {
        return (value, insert_after);
    }

    let value_is_mutable = value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .is_some_and(|pointer| pointer.is_mutable);
    let read_only_raw_mut = result_ptr_ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .filter(|pointer| pointer.kind == MirPointerKind::RawMut && !value_is_mutable)
        .map(|pointer| (pointer.pointee, pointer.address_space));
    if let Some((pointee, address_space)) = read_only_raw_mut {
        let const_origin =
            facts::pointer_origin_of_raw_mutability(rustc_public::mir::Mutability::Not);
        let const_ptr_ty: TypeHandle =
            facts::mint_ptr_type(ctx, pointee, address_space, const_origin).into();
        let (const_ptr, after) =
            retype_static_pointer_result(ctx, value, const_ptr_ty, insert_after, loc.clone());

        let cast_op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_ptr_ty],
            vec![const_ptr],
            vec![],
            0,
        );
        cast_op.deref_mut(ctx).set_loc(loc);
        let cast = MirCastOp::new(cast_op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
        cast_op.insert_after(ctx, after);
        return (cast_op.deref(ctx).get_result(0), cast_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_ptr_ty],
        vec![value],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    if result_ptr_ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .is_some_and(|pointer| pointer.kind != MirPointerKind::Erased)
    {
        cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
    }
    cast_op.insert_after(ctx, insert_after);

    (cast_op.deref(ctx).get_result(0), cast_op)
}

/// Return the first union stored inline in `ty`.
///
/// Pointer pointees are deliberately not followed: their bytes are not part of
/// the containing allocation, and initializer relocations are collected through
/// rustc provenance separately. Arrays, tuples, structs, and enum payloads are
/// inline and must be searched recursively.
pub(super) fn stored_type_union_name(
    ty: rustc_public::ty::Ty,
    visited: &mut Vec<rustc_public::ty::Ty>,
) -> Option<String> {
    use rustc_public::ty::{AdtKind, RigidTy, TyKind};

    if visited.contains(&ty) {
        return None;
    }
    visited.push(ty);

    match ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => {
            if matches!(adt_def.kind(), AdtKind::Union) {
                return Some(adt_def.trimmed_name());
            }
            for variant in adt_def.variants() {
                for field in variant.fields() {
                    if let Some(name) = stored_type_union_name(field.ty_with_args(&substs), visited)
                    {
                        return Some(name);
                    }
                }
            }
            None
        }
        TyKind::RigidTy(RigidTy::Array(element, _)) | TyKind::RigidTy(RigidTy::Slice(element)) => {
            stored_type_union_name(element, visited)
        }
        TyKind::RigidTy(RigidTy::Tuple(elements)) => {
            for element in elements.iter() {
                if let Some(name) = stored_type_union_name(*element, visited) {
                    return Some(name);
                }
            }
            None
        }
        _ => None,
    }
}

pub(super) fn set_global_initializer_hex_attr(
    ctx: &mut Context,
    op: Ptr<Operation>,
    initializer_hex: &str,
) {
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;

    let key = Identifier::try_new("global_initializer_hex".to_string()).expect("valid identifier");
    op.deref_mut(ctx)
        .attributes
        .set(key, StringAttr::new(initializer_hex.to_string()));
}

pub(super) fn set_global_initializer_relocations_attr(
    ctx: &mut Context,
    op: Ptr<Operation>,
    relocations: &str,
) {
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;

    let key = Identifier::try_new("global_initializer_relocations".to_string())
        .expect("valid identifier");
    op.deref_mut(ctx)
        .attributes
        .set(key, StringAttr::new(relocations.to_string()));
}

/// Check if a type is a pointer/reference to a static allocation.
/// Returns `(pointee_ty, origin)` when the type can carry a static address.
/// Keeping the origin here prevents constant materialization from collapsing
/// `&T`/`&mut T` and raw pointers before they enter `dialect-mir`.
pub(super) fn get_static_pointer_info(
    ty: &rustc_public::ty::Ty,
) -> Option<(rustc_public::ty::Ty, facts::PointerOrigin)> {
    facts::pointer_origin_of_ty(ty)
}
