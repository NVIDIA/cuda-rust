/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Storing into an enum payload whose storage type differs from its usage
//! type.
//!
//! Two payload types are not stored in the form they are used:
//!
//! ```text
//!   used as:     bool (i1)             ptr to shared (addrspace 3)
//!   stored as:   full i8 byte          generic ptr
//! ```
//!
//! Building or unpacking a whole enum VALUE converts between the two forms on
//! the spot. A raw POINTER to the payload bytes cannot: it escapes, and every
//! load or store made through it later uses the usage type against bytes laid
//! out in the storage type. mir-lower refuses to hand out such a pointer.
//!
//! That refusal fired on an ordinary field write:
//!
//! ```rust,ignore
//! if let Flag::On(b) = &mut flag { *b = value }
//! ```
//!
//! MIR turns this into `(_flag as On).0 = value`, and the obvious translation
//! is "take the payload's address, store through it". This module translates
//! it without any payload address by rebuilding the whole enum:
//!
//! ```text
//!   (_flag as On).0 = v   ->   _e  = load _flag             // whole enum out
//!                              _e' = construct_enum On(v)   // new payload in
//!                              store _e' -> _flag           // whole enum back
//! ```
//!
//! Why this is correct:
//!
//! - `construct_enum` converts each payload to its storage form on the way
//!   in. That is exactly the conversion a raw address could not carry.
//! - Valid MIR only writes a variant's field when the enum already holds that
//!   variant, so rebuilding the same variant keeps the discriminant.
//! - In a variant with several fields, the untouched ones are read out of the
//!   loaded value and passed back in unchanged.

use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public_bridge::IndexedVal;

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;

/// The enum place, variant and field an assignment destination names, when it
/// ends in a payload whose bytes use canonical storage.
pub(crate) struct CanonicalPayloadStore {
    /// The enum place: the destination without its `Downcast` and `Field`.
    pub(crate) enum_place: mir::Place,
    /// The enum's Rust type.
    pub(crate) enum_rust_ty: rustc_public::ty::Ty,
    /// Variant named by the `Downcast`.
    pub(crate) variant: usize,
    /// Field within that variant.
    pub(crate) field: usize,
}

/// A canonical enum-payload destination whose enclosing enum address was
/// materialized before a call.
pub(crate) struct PreparedCanonicalPayloadStore {
    enum_ptr: Value,
    store: CanonicalPayloadStore,
}

/// Classify an assignment destination.
///
/// `None` for every destination the ordinary address path still owns: a
/// payload whose storage equals its semantic type, a place that names no
/// payload, or an enum type this importer cannot resolve.
pub(crate) fn classify(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
) -> TranslationResult<Option<CanonicalPayloadStore>> {
    let projection = &place.projection;
    if projection.len() < 2 {
        return Ok(None);
    }
    let (mir::ProjectionElem::Downcast(variant), mir::ProjectionElem::Field(field, field_ty)) = (
        &projection[projection.len() - 2],
        &projection[projection.len() - 1],
    ) else {
        return Ok(None);
    };

    let enum_place = mir::Place {
        local: place.local,
        projection: projection[..projection.len() - 2].to_vec(),
    };
    let Ok(enum_rust_ty) = enum_place.ty(body.locals()) else {
        return Ok(None);
    };
    let enum_ty = types::translate_type(ctx, &enum_rust_ty)?;
    if !enum_ty.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
        return Ok(None);
    }

    let field_type = types::translate_type(ctx, field_ty)?;
    if !rvalue::enum_payload_needs_storage_coercion_pub(ctx, field_type) {
        return Ok(None);
    }

    Ok(Some(CanonicalPayloadStore {
        enum_place,
        enum_rust_ty,
        variant: IndexedVal::to_index(variant),
        field: *field,
    }))
}

/// Materialize the enclosing enum address for a canonical payload call
/// destination.
///
/// Keeping the typed payload metadata together with the address lets the
/// result write happen after the callee without re-evaluating the MIR place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_call_store(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    store: CanonicalPayloadStore,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(PreparedCanonicalPayloadStore, Option<Ptr<Operation>>)>> {
    let Some((enum_ptr, prev)) = rvalue::translate_place_address(
        ctx,
        body,
        value_map,
        &store.enum_place,
        /* is_mutable */ true,
        block_ptr,
        prev_op,
        loc,
    )?
    else {
        return Ok(None);
    };

    Ok(Some((
        PreparedCanonicalPayloadStore { enum_ptr, store },
        prev,
    )))
}

/// Rebuild a canonical enum payload through an address prepared before a call.
pub(crate) fn rebuild_and_store_prepared(
    ctx: &mut Context,
    prepared: &PreparedCanonicalPayloadStore,
    new_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let enum_ty = types::translate_type(ctx, &prepared.store.enum_rust_ty)?;
    rebuild_and_store_at_ptr(
        ctx,
        &prepared.store,
        prepared.enum_ptr,
        enum_ty,
        new_value,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Rebuild the enum around `new_value` and store it back.
///
/// `Ok(None)` when the enum place has no address to load from and store to,
/// which leaves the caller on its existing path and its loud refusal.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rebuild_and_store(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    store: &CanonicalPayloadStore,
    new_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Option<Ptr<Operation>>>> {
    let enum_ty = types::translate_type(ctx, &store.enum_rust_ty)?;

    let Some((enum_ptr, prev)) = rvalue::translate_place_address(
        ctx,
        body,
        value_map,
        &store.enum_place,
        /* is_mutable */ true,
        block_ptr,
        prev_op,
        loc.clone(),
    )?
    else {
        return Ok(None);
    };

    Ok(Some(Some(rebuild_and_store_at_ptr(
        ctx, store, enum_ptr, enum_ty, new_value, block_ptr, prev, loc,
    )?)))
}

#[allow(clippy::too_many_arguments)]
fn rebuild_and_store_at_ptr(
    ctx: &mut Context,
    store: &CanonicalPayloadStore,
    enum_ptr: Value,
    enum_ty: TypeHandle,
    new_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let (load_op, enum_value) = emit_load(ctx, enum_ptr, enum_ty, block_ptr, prev_op, loc.clone());
    let mut prev = Some(load_op);

    let field_count = {
        let ty_obj = enum_ty.deref(ctx);
        let enum_ty_ref = ty_obj
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .expect("classify resolved this enum type");
        match enum_ty_ref.get_variant(store.variant) {
            Some(variant) => variant.field_types.len(),
            None => {
                return pliron::input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "payload store: variant {} is out of bounds for the enum",
                        store.variant
                    ))
                );
            }
        }
    };

    let mut payload_values = Vec::with_capacity(field_count);
    for field in 0..field_count {
        if field == store.field {
            payload_values.push(new_value);
            continue;
        }
        let field_rust_ty = payload_field_type(&store.enum_rust_ty, store.variant, field, &loc)?;
        let (value, next) = rvalue::apply_enum_field_projection_pub(
            ctx,
            enum_value,
            &store.enum_rust_ty,
            IndexedVal::to_val(store.variant),
            field,
            &field_rust_ty,
            block_ptr,
            prev,
            loc.clone(),
        )?;
        payload_values.push(value);
        prev = next;
    }

    let construct = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructEnumOp::get_concrete_op_info(),
        vec![enum_ty],
        payload_values,
        vec![],
        0,
    );
    construct.deref_mut(ctx).set_loc(loc.clone());
    dialect_mir::ops::MirConstructEnumOp::new(construct).set_attr_construct_enum_variant_index(
        ctx,
        dialect_mir::attributes::VariantIndexAttr(store.variant as u32),
    );
    match prev {
        Some(p) => construct.insert_after(ctx, p),
        None => construct.insert_at_front(block_ptr, ctx),
    }
    let rebuilt = construct.deref(ctx).get_result(0);

    Ok(emit_store(
        ctx,
        rebuilt,
        enum_ptr,
        block_ptr,
        Some(construct),
        loc,
    ))
}

/// The Rust type of one payload field, read from the enum's own definition.
fn payload_field_type(
    enum_rust_ty: &rustc_public::ty::Ty,
    variant: usize,
    field: usize,
    loc: &Location,
) -> TranslationResult<rustc_public::ty::Ty> {
    use rustc_public::ty::{RigidTy, TyKind};

    let TyKind::RigidTy(RigidTy::Adt(adt_def, args)) = enum_rust_ty.kind() else {
        return pliron::input_err!(
            loc.clone(),
            TranslationErr::unsupported("payload store: the destination is not an ADT".to_string())
        );
    };
    let variants = adt_def.variants();
    let Some(variant_def) = variants.get(variant) else {
        return pliron::input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "payload store: variant {variant} is out of bounds"
            ))
        );
    };
    let fields = variant_def.fields();
    let Some(field_def) = fields.get(field) else {
        return pliron::input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("payload store: field {field} is out of bounds"))
        );
    };
    Ok(field_def.ty_with_args(&args))
}

fn emit_store(
    ctx: &mut Context,
    value: Value,
    slot: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![slot, value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    match prev_op {
        Some(p) => op.insert_after(ctx, p),
        None => op.insert_at_front(block_ptr, ctx),
    }
    op
}

fn emit_load(
    ctx: &mut Context,
    slot: Value,
    pointee: TypeHandle,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Ptr<Operation>, Value) {
    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![slot],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    match prev_op {
        Some(p) => op.insert_after(ctx, p),
        None => op.insert_at_front(block_ptr, ctx),
    }
    let value = op.deref(ctx).get_result(0);
    (op, value)
}
