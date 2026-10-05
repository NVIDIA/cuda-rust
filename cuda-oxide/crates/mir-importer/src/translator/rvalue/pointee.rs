/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `PointeeKind` and pointee/type utilities.

use crate::translator::facts;
use dialect_mir::ops::{MirExtractFieldOp, MirPtrOffsetOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Describes what a pointer points to (array vs. anything else) for
/// address-computation dispatch.
pub(super) enum PointeeKind {
    /// Pointee is `[T; N]` (carries `T`). Element addressing GEPs through
    /// the array type via `mir.array_element_addr`.
    Array(TypeHandle),
    /// Pointee is any other type. When an `Index` / `ConstantIndex`
    /// projection meets such a pointer, MIR typing guarantees the indexed
    /// place is a slice whose data pointer (produced by the fat-pointer
    /// `Deref` arm) points directly at the elements, so element addressing
    /// is a plain `mir.ptr_offset` keeping the pointer's own type.
    Direct,
}

pub(super) fn indexed_element_ptr_type(
    ctx: &mut Context,
    current_ptr_ty: TypeHandle,
    pointee_kind: PointeeKind,
    _addr_space: u32,
    _is_mutable: bool,
) -> TypeHandle {
    match pointee_kind {
        PointeeKind::Array(element_ty) => projected_pointer_type(
            ctx,
            current_ptr_ty,
            element_ty,
            /* legacy requested mutability, ignored */ false,
        )
        .expect("indexed array base must be a MirPtrType"),
        PointeeKind::Direct => current_ptr_ty,
    }
}

/// Emit the address of element `index_val` behind `current`, which is either
/// a pointer to a whole array (`&arr[i]`: `mir.array_element_addr`) or a
/// pointer to a single ELEMENT, i.e. the data pointer of a fat slice value
/// extracted by the Deref arm above (`(*slice)[i]`: element-size pointer
/// arithmetic via `mir.ptr_offset`). Returns the emitted op and the element
/// address it produces.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_indexed_element_addr(
    ctx: &mut Context,
    current: Value,
    index_val: Value,
    pointee_kind: PointeeKind,
    _addr_space: u32,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    current_prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Ptr<Operation>, Value) {
    use dialect_mir::ops::MirArrayElementAddrOp;

    let addr_op = match pointee_kind {
        PointeeKind::Array(element_ty) => {
            let elem_ptr_ty =
                projected_pointer_type(ctx, current.get_type(ctx), element_ty, is_mutable)
                    .expect("indexed array base must be a MirPtrType");
            Operation::new(
                ctx,
                MirArrayElementAddrOp::get_concrete_op_info(),
                vec![elem_ptr_ty],
                vec![current, index_val],
                vec![],
                0,
            )
        }
        PointeeKind::Direct => {
            // The pointee IS the element type, so indexing is plain
            // element-size pointer arithmetic and the result keeps the
            // pointer's own type.
            let ptr_ty = current.get_type(ctx);
            Operation::new(
                ctx,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![current, index_val],
                vec![],
                0,
            )
        }
    };
    addr_op.deref_mut(ctx).set_loc(loc);
    match current_prev_op {
        Some(p) => addr_op.insert_after(ctx, p),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let result = addr_op.deref(ctx).get_result(0);
    (addr_op, result)
}

/// Inspect a pointer value and return its pointee kind + address space, or
/// `None` if the value's type isn't a `MirPtrType`.
pub(super) fn pointer_pointee_kind(ctx: &Context, ptr_value: Value) -> Option<(PointeeKind, u32)> {
    pointer_type_pointee_kind(ctx, ptr_value.get_type(ctx))
}

/// Inspect a pointer type and return its pointee kind + address space, or
/// `None` if the type isn't a `MirPtrType`.
pub(super) fn pointer_type_pointee_kind(
    ctx: &Context,
    ptr_ty: TypeHandle,
) -> Option<(PointeeKind, u32)> {
    let ty_ref = ptr_ty.deref(ctx);
    let mir_ptr_ty = ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()?;
    let pointee = mir_ptr_ty.pointee;
    let addr_space = mir_ptr_ty.address_space;
    let pointee_ref = pointee.deref(ctx);
    let kind = if let Some(arr_ty) = pointee_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
    {
        PointeeKind::Array(arr_ty.element_type())
    } else {
        PointeeKind::Direct
    };
    Some((kind, addr_space))
}

pub(super) fn mir_ptr_pointee(ctx: &Context, ptr_ty: TypeHandle) -> Option<TypeHandle> {
    ptr_ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .map(|ptr_ty| ptr_ty.pointee)
}

/// Build a pointer to `pointee` while preserving the address space of
/// `base_ptr_ty`.
///
/// Address-producing projections such as field access are LLVM GEPs, and a
/// GEP cannot change the address space of its base pointer.
pub(super) fn projected_pointer_type(
    ctx: &mut Context,
    base_ptr_ty: TypeHandle,
    pointee: TypeHandle,
    _is_mutable: bool,
) -> Option<TypeHandle> {
    let (origin, address_space) = {
        let base_ptr_ty = base_ptr_ty.deref(ctx);
        let base_ptr_ty = base_ptr_ty.downcast_ref::<dialect_mir::types::MirPtrType>()?;
        (
            facts::pointer_origin_of_ptr_carrier(base_ptr_ty),
            base_ptr_ty.address_space,
        )
    };

    Some(facts::mint_ptr_type(ctx, pointee, address_space, origin).into())
}

pub(super) fn is_empty_tuple_type(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<dialect_mir::types::MirTupleType>()
        .is_some_and(|tt| tt.get_types().is_empty())
}

/// Whether `ty` is a tuple carrying a field that owns zero bytes while
/// demanding ABI alignment above 1, at any array nesting depth.
///
/// Code-shape guard for the read classifier: the address path's final
/// `mir.load` states the loaded field's natural alignment only, so a tuple
/// whose recorded ABI alignment comes from a zero-sized `repr(align(N))`
/// field must keep the value path, which moves the whole aggregate at its
/// recorded alignment. Gate shape from PR #715 (vyncint), with the byte-size
/// gap closed: detection compares byte sizes, not element counts, so
/// `[Align32; 2]` still reads as zero bytes.
pub(super) fn tuple_has_over_aligned_zst_field(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<dialect_mir::types::MirTupleType>()
        .is_some_and(|tuple_ty| {
            tuple_ty
                .get_types()
                .iter()
                .any(|field| is_over_aligned_zst_type(ctx, *field))
        })
}

/// Whether `ty` occupies zero bytes while its recorded ABI alignment exceeds
/// 1: a `repr(align(N))` ZST, possibly wrapped in arrays.
fn is_over_aligned_zst_type(ctx: &Context, ty: TypeHandle) -> bool {
    recorded_byte_size_and_abi_align(ctx, ty)
        .is_some_and(|(byte_size, abi_align)| byte_size == 0 && abi_align > 1)
}

/// Byte size and ABI alignment of `ty`, when the dialect records them.
///
/// Aggregates carry rustc's `total_size` and `abi_align` directly. Arrays
/// multiply the element's byte size by the element count
/// (`MirArrayType::size()` is a count, not a byte size) and align like their
/// element. Scalars and pointers return `None`: they always occupy at least
/// one byte, so they can never be a zero-sized alignment carrier.
fn recorded_byte_size_and_abi_align(ctx: &Context, ty: TypeHandle) -> Option<(u64, u64)> {
    use dialect_mir::types::{
        MirArrayType, MirEnumType, MirStructType, MirTupleType, MirUnionType,
    };

    let ty_ref = ty.deref(ctx);
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        let element_ty = array_ty.element_type();
        let element_count = array_ty.size();
        return recorded_byte_size_and_abi_align(ctx, element_ty).map(
            |(element_size, element_align)| {
                (element_size.saturating_mul(element_count), element_align)
            },
        );
    }
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return Some((tuple_ty.total_size(), tuple_ty.abi_align()));
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return Some((struct_ty.total_size(), struct_ty.abi_align));
    }
    if let Some(enum_ty) = ty_ref.downcast_ref::<MirEnumType>() {
        return Some((enum_ty.total_size(), enum_ty.abi_align()));
    }
    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return Some((union_ty.total_size(), union_ty.abi_align()));
    }
    None
}

pub(super) fn slice_like_element_type(ctx: &Context, ty: TypeHandle) -> Option<TypeHandle> {
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirSliceType>()
        .map(|slice_ty| slice_ty.element_type())
        .or_else(|| {
            ty_ref
                .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                .map(|slice_ty| slice_ty.element_type())
        })
}

/// Return the projection-internal data-pointer type for a fat slice value.
///
/// Ordinary slices deliberately erase their Rust pointer kind while retaining
/// the carrier's exact machine mutability. `DisjointSlice` instead has a fixed
/// `RawMut` field contract. In both cases extraction is representation-only:
/// it cannot invent writable evidence or a stronger Rust category.
pub(super) fn erased_slice_data_pointer_type(
    ctx: &mut Context,
    slice_value: Value,
    element_ty: TypeHandle,
) -> Option<TypeHandle> {
    let slice_ty = slice_value.get_type(ctx);
    let ordinary_mutability = slice_ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirSliceType>()
        .map(|slice| slice.is_mutable);
    if let Some(is_mutable) = ordinary_mutability {
        return Some(
            dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, is_mutable).into(),
        );
    }
    if slice_ty
        .deref(ctx)
        .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
        .is_some()
    {
        return Some(
            facts::mint_generic_ptr_type(ctx, element_ty, facts::abi_disjoint_slice_data_ptr())
                .into(),
        );
    }
    None
}

pub(super) fn rust_ty_is_slice(ty: &rustc_public::ty::Ty) -> bool {
    matches!(
        ty.kind(),
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(_))
    )
}

pub(super) fn normalize_slice_value_to_data_ptr(
    ctx: &mut Context,
    value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Option<Ptr<Operation>>) {
    let slice_element_ty = {
        let value_ty = value.get_type(ctx);
        let value_ty_ref = value_ty.deref(ctx);
        value_ty_ref
            .downcast_ref::<dialect_mir::types::MirSliceType>()
            .map(|slice_ty| slice_ty.element_type())
    };
    let Some(element_ty) = slice_element_ty else {
        return (value, prev_op);
    };

    let Some(data_ptr_ty) = erased_slice_data_pointer_type(ctx, value, element_ty) else {
        return (value, prev_op);
    };
    let extract_ptr = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![data_ptr_ty],
        vec![value],
        vec![],
        0,
    );
    extract_ptr.deref_mut(ctx).set_loc(loc);
    MirExtractFieldOp::new(extract_ptr)
        .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
    match prev_op {
        Some(prev) => extract_ptr.insert_after(ctx, prev),
        None => extract_ptr.insert_at_front(block_ptr, ctx),
    }

    (extract_ptr.deref(ctx).get_result(0), Some(extract_ptr))
}

#[cfg(test)]
// Tests build kinded fixture types directly; production code mints via facts::PointerOrigin.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use dialect_mir::types::{MirArrayType, MirPtrType, MirStructType, MirTupleType};
    use pliron::builtin::types::{IntegerType, Signedness};

    #[test]
    fn over_aligned_zst_tuple_fields_force_the_value_fallback_gate() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        // `#[repr(align(32))] struct Align32;`: zero bytes, ABI alignment 32.
        let align32_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();

        // `(Align32, u8)` as rustc lays it out: one payload byte padded out
        // to a 32-byte, align-32 allocation.
        let pair_ty: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![align32_ty, u8_ty],
            vec![],
            vec![0, 0],
            32,
            32,
        )
        .into();
        assert!(
            tuple_has_over_aligned_zst_field(&ctx, pair_ty),
            "a zero-byte repr(align(32)) field must force the value path"
        );

        // The byte-size gap in PR #715's gate: `[Align32; 2]` has element
        // count 2 but still owns zero bytes and still demands align 32, at
        // any array nesting depth.
        let align32_x2_ty: TypeHandle = MirArrayType::get(&mut ctx, align32_ty, 2).into();
        let align32_x2x3_ty: TypeHandle = MirArrayType::get(&mut ctx, align32_x2_ty, 3).into();
        for wrapped_ty in [align32_x2_ty, align32_x2x3_ty] {
            let tuple_ty: TypeHandle = MirTupleType::get_with_layout(
                &mut ctx,
                vec![wrapped_ty, u8_ty],
                vec![],
                vec![0, 0],
                32,
                32,
            )
            .into();
            assert!(
                tuple_has_over_aligned_zst_field(&ctx, tuple_ty),
                "a zero-byte array of over-aligned ZSTs must not slip through on element count"
            );
        }

        // Ordinary tuples keep the address path: rustc's reordered
        // `(u8, u32)` layout (u8 at offset 4, u32 at offset 0).
        let plain_ty: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![u8_ty, u32_ty],
            vec![1, 0],
            vec![4, 0],
            8,
            4,
        )
        .into();
        assert!(
            !tuple_has_over_aligned_zst_field(&ctx, plain_ty),
            "tuples without over-aligned ZST fields must stay on the address path"
        );

        // An align-1 ZST field, the unit tuple, does not trip the gate:
        // zero bytes alone raise nothing.
        let unit_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
        let with_unit_ty: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![unit_ty, u32_ty],
            vec![],
            vec![0, 0],
            4,
            4,
        )
        .into();
        assert!(
            !tuple_has_over_aligned_zst_field(&ctx, with_unit_ty),
            "plain ZST fields raise no alignment and must not force the fallback"
        );
    }

    #[test]
    fn projected_pointer_type_preserves_base_address_space() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let field_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        let aggregate_ty: TypeHandle = MirStructType::get(
            &mut ctx,
            "ProjectedAddressSpaceTest".into(),
            vec!["field".into()],
            vec![field_ty],
        )
        .into();

        for address_space in [1, 4] {
            let base_ptr_ty: TypeHandle =
                MirPtrType::get(&mut ctx, aggregate_ty, false, address_space).into();

            let projected_ty = projected_pointer_type(&mut ctx, base_ptr_ty, field_ty, false)
                .expect("base type must be a MIR pointer");

            let (projected_pointee, projected_mutability, projected_address_space) = {
                let projected_ty = projected_ty.deref(&ctx);
                let projected_ptr = projected_ty
                    .downcast_ref::<MirPtrType>()
                    .expect("projected type must remain a MIR pointer");

                (
                    projected_ptr.pointee,
                    projected_ptr.is_mutable,
                    projected_ptr.address_space,
                )
            };

            assert_eq!(
                projected_pointee, field_ty,
                "field projection must change the pointee type"
            );
            assert!(
                !projected_mutability,
                "field projection must preserve the requested mutability"
            );
            assert_eq!(
                projected_address_space, address_space,
                "field projection must preserve the base pointer address space"
            );
        }
    }
}
