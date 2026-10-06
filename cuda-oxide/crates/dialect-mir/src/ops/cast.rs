/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR cast operations.
//!
//! This module defines type conversion operations for the MIR dialect.

use crate::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
use crate::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPointerCarrier, MirPointerKind, MirPtrType,
    MirSliceType, MirStructType, MirTupleType, MirUnionType, is_opaque_fn_pointer_type,
    pointer_carriers_in_type, pointer_kinds_in_type,
};
use pliron::{
    builtin::{
        attributes::StringAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        type_interfaces::FloatTypeInterface,
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{Typed, type_cast, type_impls},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use super::{
    aggregate::{MirArrayElementAddrOp, MirFieldAddrOp},
    memory::{MirAllocaOp, MirGlobalAllocOp, MirPtrOffsetOp, MirSharedAllocOp},
};

// ============================================================================
// MirCastOp
// ============================================================================

/// MIR cast operation — type conversion with preserved semantic intent.
///
/// The `cast_kind` attribute records the MIR `CastKind` so the lowering can
/// dispatch correctly (e.g. `Transmute` → `bitcast`/`extractvalue`, not `sitofp`).
///
/// # Operands
///
/// | Name    | Type     |
/// |---------|----------|
/// | `value` | Any type |
///
/// # Results
///
/// | Name  | Type        |
/// |-------|-------------|
/// | `res` | Target type |
///
/// # Attributes
///
/// | Name                     | Type                          | Description                                                     |
/// |--------------------------|-------------------------------|-----------------------------------------------------------------|
/// | `cast_kind`              | `MirCastKindAttr`             | Semantic cast kind from MIR.                                    |
/// | `pointer_kind_authority` | `MirPointerKindAuthorityAttr` | Optional Rust boundary authorizing a new concrete pointer kind. |
///
/// # Verification
///
/// Requires `cast_kind` and checks that operand/result types match the kind:
/// - **IntToInt** / **FloatToFloat**: operand and result are both integer or both float types.
/// - **IntToFloat** / **FloatToInt**: operand and result are the appropriate integer/float pair.
/// - **PointerExposeAddress**: operand is pointer, result is integer.
/// - **PointerWithExposedProvenance**: operand is integer and result is raw,
///   or the exact immutable opaque function-pointer carrier; never a Rust
///   reference or arbitrary writable `Erased` storage.
/// - Generic pointer transitions preserve carrier mutability and may preserve
///   a concrete kind or erase it, but establishing/changing a concrete kind
///   requires a compatible `pointer_kind_authority`.
/// - **PtrToPtr** / **FnPtrToPtr** require pointer-carrying operand and result
///   type graphs; `FnPtrToPtr` additionally requires the canonical immutable
///   function-pointer source. An authority cannot reinterpret an integer as a
///   pointer.
/// - `Reborrow`, `RawAddress`, `StaticAddress`, and `AbiBoundary` accept only
///   top-level pointer/slice results with matching pointee/element shape
///   (`StaticAddress` pointer-to-pointer conversions must start from `Erased`
///   physical/static storage and it also covers the explicit integer-to-raw case).
/// - Only `RustCast` on `Transmute` can authorize nested aggregate pointer
///   transitions. Other casts require a structurally corresponding source for
///   every target carrier, including `Erased`, so integer bytes cannot become
///   writable pointer evidence.
/// - `RustCast` must match rustc's cast semantics: only `Transmute` may
///   reinterpret arbitrary pointer-bearing bytes; pointer coercions preserve
///   the carrier structure and obey their raw/function-pointer restrictions.
///   ReifyFnPointer and ClosureFnPointer are resolved by the importer into a
///   canonical function token; their legacy `mir.cast` forms fail closed
///   because a zero-sized source value contains no function-address bits to
///   lower.
#[pliron_op(
    name = "mir.cast",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (
        cast_kind: MirCastKindAttr,
        pointer_kind_authority: MirPointerKindAuthorityAttr
    )
)]
pub struct MirCastOp;

type PointerCarrier = MirPointerCarrier;

#[derive(Clone, Copy)]
enum ErasedStorageOrigin {
    Static,
    Compiler,
}

/// Check that an `Erased` thin pointer has the storage origin required by an
/// origin-sensitive authority. This is a verifier admissibility check, not a
/// dynamic borrow/provenance tag or an optimizer alias proof.
///
/// Merely checking the immediate kind is insufficient: a generic
/// `RawConst -> Erased` cast could otherwise hide a raw pointer immediately
/// before `StaticAddress` manufactures `SharedRef`. Walk only the closed set
/// of representation/address operations used by the importer, require every
/// value in the chain to remain `Erased`, and fail closed at block arguments,
/// loads, calls, marked casts, or any other producer.
fn has_erased_storage_lineage(
    ctx: &Context,
    mut value: Value,
    expected_origin: ErasedStorageOrigin,
) -> bool {
    let mut visited = Vec::new();
    loop {
        let value_is_erased = value
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .is_some_and(|pointer| pointer.kind == MirPointerKind::Erased);
        if !value_is_erased {
            return false;
        }

        let Some(defining_op) = value.defining_op() else {
            return false;
        };
        if visited.contains(&defining_op) {
            return false;
        }
        visited.push(defining_op);

        let has_expected_root = match expected_origin {
            ErasedStorageOrigin::Static => {
                Operation::get_op::<MirGlobalAllocOp>(defining_op, ctx).is_some()
                    || Operation::get_op::<MirSharedAllocOp>(defining_op, ctx).is_some()
            }
            ErasedStorageOrigin::Compiler => {
                Operation::get_op::<MirAllocaOp>(defining_op, ctx).is_some()
            }
        };
        if has_expected_root {
            return true;
        }

        if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx) {
            let is_unmarked_ptr_retype = cast
                .get_attr_cast_kind(ctx)
                .is_some_and(|kind| *kind == MirCastKindAttr::PtrToPtr)
                && cast.get_attr_pointer_kind_authority(ctx).is_none();
            if !is_unmarked_ptr_retype {
                return false;
            }
            value = defining_op.deref(ctx).get_operand(0);
            continue;
        }

        let is_pointer_transport = Operation::get_op::<MirPtrOffsetOp>(defining_op, ctx).is_some()
            || matches!(expected_origin, ErasedStorageOrigin::Compiler)
                && (Operation::get_op::<MirFieldAddrOp>(defining_op, ctx).is_some()
                    || Operation::get_op::<MirArrayElementAddrOp>(defining_op, ctx).is_some());
        if is_pointer_transport {
            value = defining_op.deref(ctx).get_operand(0);
            continue;
        }

        return false;
    }
}

/// Rustc may promote `&mut []` to immutable static storage. Establishing a
/// `UniqueRef` from static storage is normally invalid, but it is vacuously
/// sound for an exact zero-length array: no byte can be read, written, or
/// aliased through the reference. Keep this exception deliberately narrower
/// than general ZST handling so it cannot authorize non-empty promoted data.
fn is_promoted_empty_unique_ref_transition(
    ctx: &Context,
    source_value: Value,
    source_ty: pliron::r#type::TypeHandle,
    target_ty: pliron::r#type::TypeHandle,
) -> bool {
    let source = source_ty.deref(ctx);
    let target = target_ty.deref(ctx);
    let (Some(source), Some(target)) = (
        source.downcast_ref::<MirPtrType>(),
        target.downcast_ref::<MirPtrType>(),
    ) else {
        return false;
    };

    source.kind == MirPointerKind::Erased
        && !source.is_mutable
        && target.kind == MirPointerKind::UniqueRef
        && target.is_mutable
        && source.pointee == target.pointee
        && target
            .pointee
            .deref(ctx)
            .downcast_ref::<MirArrayType>()
            .is_some_and(|array| array.size == 0)
        && source_value.defining_op().is_some_and(|defining_op| {
            Operation::get_op::<MirGlobalAllocOp>(defining_op, ctx).is_some_and(|global| {
                let initializer_key = "global_initializer_hex".try_into().unwrap();
                let has_empty_initializer = defining_op
                    .deref(ctx)
                    .attributes
                    .get::<StringAttr>(&initializer_key)
                    .is_some_and(|initializer| String::from((*initializer).clone()).is_empty());
                let relocations_key = "global_initializer_relocations".try_into().unwrap();
                let has_no_relocations = defining_op
                    .deref(ctx)
                    .attributes
                    .get::<StringAttr>(&relocations_key)
                    .is_none();
                let required_alignment = required_pointee_alignment(ctx, target.pointee);
                global.is_immutable(ctx)
                    && has_empty_initializer
                    && has_no_relocations
                    && required_alignment.is_some_and(|required| {
                        global.get_alignment_value(ctx).is_some_and(|alignment| {
                            alignment.is_power_of_two() && alignment >= required
                        })
                    })
                    && global
                        .get_attr_global_type(ctx)
                        .is_some_and(|global_type| global_type.get_type(ctx) == target.pointee)
            })
        })
}

/// Required alignment of a promoted empty pointee.
///
/// Even a zero-length reference must be aligned. Aggregate MIR types retain
/// rustc's exact ABI alignment; arrays inherit it from their element. Scalar
/// and pointer leaves use the NVPTX64 natural alignment modeled by lowering.
/// Unknown leaves fail closed instead of relying on a later lowering error.
fn required_pointee_alignment(ctx: &Context, ty: pliron::r#type::TypeHandle) -> Option<u64> {
    let ty = ty.deref(ctx);
    if let Some(array) = ty.downcast_ref::<MirArrayType>() {
        return required_pointee_alignment(ctx, array.element_ty);
    }
    if let Some(tuple) = ty.downcast_ref::<MirTupleType>() {
        return if tuple.abi_align() > 0 {
            Some(tuple.abi_align())
        } else if tuple.types.is_empty() && tuple.total_size == 0 {
            Some(1)
        } else {
            None
        };
    }
    if let Some(structure) = ty.downcast_ref::<MirStructType>() {
        return if structure.abi_align > 0 {
            Some(structure.abi_align)
        } else if structure.field_types.is_empty() && structure.total_size == 0 {
            Some(1)
        } else {
            None
        };
    }
    if let Some(enumeration) = ty.downcast_ref::<MirEnumType>() {
        return (enumeration.abi_align() > 0).then(|| enumeration.abi_align());
    }
    if let Some(union) = ty.downcast_ref::<MirUnionType>() {
        return (union.abi_align() > 0).then(|| union.abi_align());
    }
    if ty.is::<MirSliceType>() {
        return Some(8);
    }
    if let Some(disjoint) = ty.downcast_ref::<MirDisjointSliceType>() {
        let mut alignment = 8;
        for &space_ty in &disjoint.space_tys {
            alignment = alignment.max(required_pointee_alignment(ctx, space_ty)?);
        }
        return Some(alignment);
    }
    if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        let size = u64::from(integer.width()).div_ceil(8).max(1);
        return Some(size.next_power_of_two().min(16));
    }
    if ty.is::<MirPtrType>() {
        return Some(8);
    }
    if let Some(float) = type_cast::<dyn FloatTypeInterface>(&*ty) {
        let size = u64::try_from(float.get_semantics().bits).ok()?.div_ceil(8);
        return Some(size.next_power_of_two().min(16));
    }
    None
}

/// Pair every pointer carrier in `target` with the carrier at the same
/// structural position in `source`, when one exists. This makes cast
/// verification cover pointers nested inside aggregates rather than only a
/// top-level thin/fat pointer. Pointer pointees are intentionally leaves: a
/// cast changes the pointer value's type, not values in the referenced
/// allocation.
fn pointer_kind_transitions(
    ctx: &Context,
    source: pliron::r#type::TypeHandle,
    target: pliron::r#type::TypeHandle,
) -> Vec<(Option<PointerCarrier>, PointerCarrier)> {
    fn visit(
        ctx: &Context,
        source: Option<pliron::r#type::TypeHandle>,
        target: pliron::r#type::TypeHandle,
        visited: &mut Vec<(
            Option<pliron::r#type::TypeHandle>,
            pliron::r#type::TypeHandle,
        )>,
        transitions: &mut Vec<(Option<PointerCarrier>, PointerCarrier)>,
    ) {
        if visited.contains(&(source, target)) {
            return;
        }
        visited.push((source, target));

        let target_obj = target.deref(ctx);
        if let Some(target_pointer) = target_obj.downcast_ref::<MirPtrType>() {
            let source_carrier = source.and_then(|source| {
                source
                    .deref(ctx)
                    .downcast_ref::<MirPtrType>()
                    .map(|pointer| PointerCarrier {
                        kind: pointer.kind,
                        is_mutable: pointer.is_mutable,
                    })
            });
            transitions.push((
                source_carrier,
                PointerCarrier {
                    kind: target_pointer.kind,
                    is_mutable: target_pointer.is_mutable,
                },
            ));
            return;
        }
        if let Some(target_slice) = target_obj.downcast_ref::<MirSliceType>() {
            let source_carrier = source.and_then(|source| {
                source
                    .deref(ctx)
                    .downcast_ref::<MirSliceType>()
                    .map(|slice| PointerCarrier {
                        kind: slice.kind,
                        is_mutable: slice.is_mutable,
                    })
            });
            transitions.push((
                source_carrier,
                PointerCarrier {
                    kind: target_slice.kind,
                    is_mutable: target_slice.is_mutable,
                },
            ));
            return;
        }
        if target_obj.downcast_ref::<MirDisjointSliceType>().is_some() {
            let source_carrier = source.and_then(|source| {
                (source == target).then_some(PointerCarrier {
                    kind: MirPointerKind::RawMut,
                    is_mutable: true,
                })
            });
            transitions.push((
                source_carrier,
                PointerCarrier {
                    kind: MirPointerKind::RawMut,
                    is_mutable: true,
                },
            ));
            return;
        }

        let source_obj = source.map(|source| source.deref(ctx));
        let (target_children, source_children): (Vec<_>, Vec<_>) =
            if let Some(target_array) = target_obj.downcast_ref::<MirArrayType>() {
                let source_children = source_obj
                    .as_deref()
                    .and_then(|source_obj| source_obj.downcast_ref::<MirArrayType>())
                    .filter(|source_array| source_array.size == target_array.size)
                    .map(|source_array| vec![source_array.element_ty])
                    .unwrap_or_default();
                (vec![target_array.element_ty], source_children)
            } else if let Some(tuple) = target_obj.downcast_ref::<MirTupleType>() {
                let source_children = source_obj
                    .as_deref()
                    .and_then(|source_obj| source_obj.downcast_ref::<MirTupleType>())
                    .filter(|source_tuple| {
                        source_tuple.types.len() == tuple.types.len()
                            && source_tuple.memory_order() == tuple.memory_order()
                            && source_tuple.field_offsets == tuple.field_offsets
                            && source_tuple.total_size == tuple.total_size
                            && source_tuple.abi_align == tuple.abi_align
                    })
                    .map(|source_tuple| source_tuple.types.clone())
                    .unwrap_or_default();
                (tuple.types.clone(), source_children)
            } else if let Some(target_struct) = target_obj.downcast_ref::<MirStructType>() {
                let source_children = source_obj
                    .as_deref()
                    .and_then(|source_obj| source_obj.downcast_ref::<MirStructType>())
                    .filter(|source_struct| {
                        source_struct.field_names == target_struct.field_names
                            && source_struct.field_types.len() == target_struct.field_types.len()
                            && source_struct.memory_order() == target_struct.memory_order()
                            && source_struct.field_offsets == target_struct.field_offsets
                            && source_struct.total_size == target_struct.total_size
                            && source_struct.abi_align == target_struct.abi_align
                            && source_struct.abi_kind == target_struct.abi_kind
                    })
                    .map(|source_struct| source_struct.field_types.clone())
                    .unwrap_or_default();
                (target_struct.field_types.clone(), source_children)
            } else if let Some(union_ty) = target_obj.downcast_ref::<MirUnionType>() {
                // Union fields are overlapping alternative views, not independent
                // structural positions. Only an unchanged union type proves that
                // a target pointer field corresponds to the same source view.
                let source_children = source
                    .filter(|source| *source == target)
                    .map(|_| union_ty.field_types.clone())
                    .unwrap_or_default();
                (union_ty.field_types.clone(), source_children)
            } else if let Some(enum_ty) = target_obj.downcast_ref::<MirEnumType>() {
                // Variant fields may share physical bytes. As with unions, only
                // an unchanged enum representation supplies a safe correspondence.
                let source_children = source
                    .filter(|source| *source == target)
                    .map(|_| enum_ty.all_field_types.clone())
                    .unwrap_or_default();
                (enum_ty.all_field_types.clone(), source_children)
            } else {
                return;
            };

        for (index, target_child) in target_children.into_iter().enumerate() {
            visit(
                ctx,
                source_children.get(index).copied(),
                target_child,
                visited,
                transitions,
            );
        }
    }

    let mut transitions = Vec::new();
    visit(ctx, Some(source), target, &mut Vec::new(), &mut transitions);
    transitions
}

/// Whether a non-representational pointer boundary keeps the value's carrier
/// shape. These authorities may change Rust pointer kind (and, for an address
/// normalization, address space), but they may not reinterpret an integer as
/// a reference or silently change the pointed-to Rust type.
fn matching_top_level_pointer_shape(
    ctx: &Context,
    source: pliron::r#type::TypeHandle,
    target: pliron::r#type::TypeHandle,
) -> bool {
    let source = source.deref(ctx);
    let target = target.deref(ctx);

    match (
        source.downcast_ref::<MirPtrType>(),
        target.downcast_ref::<MirPtrType>(),
    ) {
        (Some(source), Some(target)) => source.pointee == target.pointee,
        _ => match (
            source.downcast_ref::<MirSliceType>(),
            target.downcast_ref::<MirSliceType>(),
        ) {
            (Some(source), Some(target)) => source.element_ty == target.element_ty,
            _ => false,
        },
    }
}

fn is_raw_kind(kind: MirPointerKind) -> bool {
    matches!(kind, MirPointerKind::RawConst | MirPointerKind::RawMut)
}

/// Whether `ty` contains the canonical opaque function-pointer carrier at any
/// by-value aggregate position. Pointer pointees are deliberately leaves: a
/// data pointer to a struct containing a function pointer does not itself
/// carry a function-pointer token.
fn contains_opaque_fn_pointer(
    ctx: &Context,
    ty: pliron::r#type::TypeHandle,
    visited: &mut Vec<pliron::r#type::TypeHandle>,
) -> bool {
    if visited.contains(&ty) {
        return false;
    }
    visited.push(ty);

    if is_opaque_fn_pointer_type(ctx, ty) {
        return true;
    }

    let ty_obj = ty.deref(ctx);
    let children = if let Some(array) = ty_obj.downcast_ref::<MirArrayType>() {
        vec![array.element_ty]
    } else if let Some(tuple) = ty_obj.downcast_ref::<MirTupleType>() {
        tuple.types.clone()
    } else if let Some(struct_ty) = ty_obj.downcast_ref::<MirStructType>() {
        struct_ty.field_types.clone()
    } else if let Some(union_ty) = ty_obj.downcast_ref::<MirUnionType>() {
        union_ty.field_types.clone()
    } else if let Some(enum_ty) = ty_obj.downcast_ref::<MirEnumType>() {
        enum_ty.all_field_types.clone()
    } else {
        Vec::new()
    };

    children
        .into_iter()
        .any(|child| contains_opaque_fn_pointer(ctx, child, visited))
}

/// Keep canonical function-pointer tokens at the same recursive aggregate
/// positions. Their carrier is intentionally `Erased`, so ordinary carrier
/// comparison alone cannot distinguish them from erased data storage.
///
/// A source-level `Transmute` may explicitly reinterpret an aggregate. Other
/// legal function-pointer creations/exposures are top-level and are checked
/// separately by their exact cast-kind rules.
fn nested_opaque_fn_pointer_positions_match(
    ctx: &Context,
    source: pliron::r#type::TypeHandle,
    target: pliron::r#type::TypeHandle,
) -> bool {
    fn visit(
        ctx: &Context,
        source: pliron::r#type::TypeHandle,
        target: pliron::r#type::TypeHandle,
        visited: &mut Vec<(pliron::r#type::TypeHandle, pliron::r#type::TypeHandle)>,
    ) -> bool {
        if source == target || visited.contains(&(source, target)) {
            return true;
        }
        visited.push((source, target));

        let source_is_opaque = is_opaque_fn_pointer_type(ctx, source);
        let target_is_opaque = is_opaque_fn_pointer_type(ctx, target);
        if source_is_opaque || target_is_opaque {
            return source_is_opaque == target_is_opaque;
        }

        let source_obj = source.deref(ctx);
        let target_obj = target.deref(ctx);
        let corresponding_children: Option<(Vec<_>, Vec<_>)> =
            if let (Some(source_array), Some(target_array)) = (
                source_obj.downcast_ref::<MirArrayType>(),
                target_obj.downcast_ref::<MirArrayType>(),
            ) {
                (source_array.size == target_array.size)
                    .then(|| (vec![source_array.element_ty], vec![target_array.element_ty]))
            } else if let (Some(source_tuple), Some(target_tuple)) = (
                source_obj.downcast_ref::<MirTupleType>(),
                target_obj.downcast_ref::<MirTupleType>(),
            ) {
                (source_tuple.types.len() == target_tuple.types.len()
                    && source_tuple.memory_order() == target_tuple.memory_order()
                    && source_tuple.field_offsets == target_tuple.field_offsets
                    && source_tuple.total_size == target_tuple.total_size
                    && source_tuple.abi_align == target_tuple.abi_align)
                    .then(|| (source_tuple.types.clone(), target_tuple.types.clone()))
            } else if let (Some(source_struct), Some(target_struct)) = (
                source_obj.downcast_ref::<MirStructType>(),
                target_obj.downcast_ref::<MirStructType>(),
            ) {
                (source_struct.field_names == target_struct.field_names
                    && source_struct.field_types.len() == target_struct.field_types.len()
                    && source_struct.memory_order() == target_struct.memory_order()
                    && source_struct.field_offsets == target_struct.field_offsets
                    && source_struct.total_size == target_struct.total_size
                    && source_struct.abi_align == target_struct.abi_align
                    && source_struct.abi_kind == target_struct.abi_kind)
                    .then(|| {
                        (
                            source_struct.field_types.clone(),
                            target_struct.field_types.clone(),
                        )
                    })
            } else {
                None
            };

        if let Some((source_children, target_children)) = corresponding_children {
            return source_children
                .into_iter()
                .zip(target_children)
                .all(|(source, target)| visit(ctx, source, target, visited));
        }

        !contains_opaque_fn_pointer(ctx, source, &mut Vec::new())
            && !contains_opaque_fn_pointer(ctx, target, &mut Vec::new())
    }

    visit(ctx, source, target, &mut Vec::new())
}

/// Whether a sized pointee has the exact trailing-array shape that rustc may
/// unsize into `target` while preserving every other field position.
fn pointee_unsizes_to(
    ctx: &Context,
    source: pliron::r#type::TypeHandle,
    target: pliron::r#type::TypeHandle,
    visited: &mut Vec<(pliron::r#type::TypeHandle, pliron::r#type::TypeHandle)>,
) -> bool {
    if visited.contains(&(source, target)) {
        return true;
    }
    visited.push((source, target));

    let source_obj = source.deref(ctx);
    if let Some(array) = source_obj.downcast_ref::<MirArrayType>() {
        return array.element_ty == target;
    }
    let Some(source_struct) = source_obj.downcast_ref::<MirStructType>() else {
        return false;
    };
    let target_obj = target.deref(ctx);
    let Some(target_struct) = target_obj.downcast_ref::<MirStructType>() else {
        return false;
    };

    if source_struct.name != target_struct.name
        || source_struct.field_names != target_struct.field_names
        || source_struct.field_types.len() != target_struct.field_types.len()
        || source_struct.memory_order() != target_struct.memory_order()
        || source_struct.field_offsets != target_struct.field_offsets
        || source_struct.abi_align != target_struct.abi_align
        || source_struct.abi_kind != target_struct.abi_kind
    {
        return false;
    }
    let Some(tail_index) = source_struct.memory_order().last().copied() else {
        return false;
    };
    for (index, (source_field, target_field)) in source_struct
        .field_types
        .iter()
        .zip(&target_struct.field_types)
        .enumerate()
    {
        if index == tail_index {
            if !pointee_unsizes_to(ctx, *source_field, *target_field, visited) {
                return false;
            }
        } else if source_field != target_field {
            return false;
        }
    }
    true
}

/// Recognize the only unsize representation this dialect currently lowers:
/// a thin pointer to an array (possibly down a trailing struct-field chain)
/// becoming a fat MirSlice carrier with the same kind and mutability.
fn supported_unsize_shape(
    ctx: &Context,
    source_ty: pliron::r#type::TypeHandle,
    target_ty: pliron::r#type::TypeHandle,
) -> bool {
    let source_obj = source_ty.deref(ctx);
    let target_obj = target_ty.deref(ctx);
    let (Some(source), Some(target)) = (
        source_obj.downcast_ref::<MirPtrType>(),
        target_obj.downcast_ref::<MirSliceType>(),
    ) else {
        return false;
    };
    source.kind == target.kind
        && source.is_mutable == target.is_mutable
        && pointee_unsizes_to(ctx, source.pointee, target.element_ty, &mut Vec::new())
}

/// Check rustc's raw array-pointer decay shape: `*[T; N] -> *T` in the same
/// address space. Pointer constness is checked separately by the cast-kind
/// matrix because a mutable source may be weakened to const.
fn supported_array_to_pointer_shape(
    ctx: &Context,
    source_ty: pliron::r#type::TypeHandle,
    target_ty: pliron::r#type::TypeHandle,
) -> bool {
    let source_ty = source_ty.deref(ctx);
    let target_ty = target_ty.deref(ctx);
    let (Some(source), Some(target)) = (
        source_ty.downcast_ref::<MirPtrType>(),
        target_ty.downcast_ref::<MirPtrType>(),
    ) else {
        return false;
    };
    let source_pointee = source.pointee.deref(ctx);
    let Some(source_array) = source_pointee.downcast_ref::<MirArrayType>() else {
        return false;
    };
    source_array.element_ty == target.pointee && source.address_space == target.address_space
}

/// Check that a `RustCast` marker agrees with the actual rustc `CastKind`.
///
/// `RustCast` identifies an explicit source cast; it is not carte blanche to
/// claim any pointer category. Only `Transmute` can reinterpret arbitrary
/// pointer-bearing representations. Every narrower cast keeps the source
/// language's corresponding restrictions.
fn rust_cast_transition_is_admissible(
    ctx: &Context,
    cast_kind: MirCastKindAttr,
    source_ty: pliron::r#type::TypeHandle,
    target_ty: pliron::r#type::TypeHandle,
    source: Option<PointerCarrier>,
    target: Option<PointerCarrier>,
) -> bool {
    match cast_kind {
        MirCastKindAttr::Transmute => true,
        MirCastKindAttr::PtrToPtr => matches!(
            (source, target),
            (Some(source), Some(target))
                if is_raw_kind(source.kind) && is_raw_kind(target.kind)
        ),
        MirCastKindAttr::FnPtrToPtr => matches!(
            (source, target),
            (Some(source), Some(target))
                if source.kind == MirPointerKind::Erased
                    && is_opaque_fn_pointer_type(ctx, source_ty)
                    && is_raw_kind(target.kind)
        ),
        MirCastKindAttr::PointerCoercionUnsize => supported_unsize_shape(ctx, source_ty, target_ty),
        MirCastKindAttr::Subtype => source_ty == target_ty,
        MirCastKindAttr::PointerCoercionMutToConst => matches!(
            (source, target),
            (Some(source), Some(target))
                if source.kind == MirPointerKind::RawMut
                    && target.kind == MirPointerKind::RawConst
                    && matching_top_level_pointer_shape(ctx, source_ty, target_ty)
        ),
        MirCastKindAttr::PointerCoercionArrayToPointer => matches!(
            (source, target),
            (Some(source), Some(target))
                if is_raw_kind(source.kind)
                    && is_raw_kind(target.kind)
                    && supported_array_to_pointer_shape(ctx, source_ty, target_ty)
                    && !(source.kind == MirPointerKind::RawConst
                        && target.kind == MirPointerKind::RawMut)
        ),
        MirCastKindAttr::PointerWithExposedProvenance => {
            target.is_some_and(|target| is_raw_kind(target.kind))
        }
        // These operations create or adjust opaque function-pointer carriers.
        // Function pointers remain Erased and may not establish ref/raw kinds.
        MirCastKindAttr::PointerCoercionReifyFnPointer
        | MirCastKindAttr::PointerCoercionUnsafeFnPointer
        | MirCastKindAttr::PointerCoercionClosureFnPointer => false,
        // Numeric casts and PointerExposeAddress cannot carry this authority;
        // the compatibility check diagnoses them before this helper matters.
        MirCastKindAttr::IntToInt
        | MirCastKindAttr::IntToFloat
        | MirCastKindAttr::FloatToInt
        | MirCastKindAttr::FloatToFloat
        | MirCastKindAttr::PointerExposeAddress => false,
    }
}

impl MirCastOp {
    /// Create a new MirCastOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCastOp { op }
    }

    /// Mark this cast as an explicit Rust semantic boundary that is allowed
    /// to establish or change a concrete pointer kind.
    pub fn set_pointer_kind_authority(
        &self,
        ctx: &mut Context,
        authority: MirPointerKindAuthorityAttr,
    ) {
        self.set_attr_pointer_kind_authority(ctx, authority);
    }
}

impl Verify for MirCastOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let loc = op.loc();

        // Structural: one operand, one result (OneOpdInterface / OneResultInterface guarantee count).
        let opd_val = op.get_operand(0);
        let res_val = op.get_result(0);
        let opd_ty = opd_val.get_type(ctx);
        let res_ty = res_val.get_type(ctx);
        let opd_ty_obj = opd_ty.deref(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        let cast_kind = match self.get_attr_cast_kind(ctx) {
            Some(r) => r.clone(),
            None => return verify_err!(loc, "MirCastOp must have a cast_kind attribute"),
        };

        match &cast_kind {
            MirCastKindAttr::IntToInt => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToInt cast requires integer operand type");
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToInt cast requires integer result type");
                }
            }
            MirCastKindAttr::IntToFloat => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToFloat cast requires integer operand type");
                }
                if !type_impls::<dyn FloatTypeInterface>(&*res_ty_obj) {
                    return verify_err!(loc, "IntToFloat cast requires float result type");
                }
            }
            MirCastKindAttr::FloatToInt => {
                if !type_impls::<dyn FloatTypeInterface>(&*opd_ty_obj) {
                    return verify_err!(loc, "FloatToInt cast requires float operand type");
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "FloatToInt cast requires integer result type");
                }
            }
            MirCastKindAttr::FloatToFloat => {
                if !type_impls::<dyn FloatTypeInterface>(&*opd_ty_obj) {
                    return verify_err!(loc, "FloatToFloat cast requires float operand type");
                }
                if !type_impls::<dyn FloatTypeInterface>(&*res_ty_obj) {
                    return verify_err!(loc, "FloatToFloat cast requires float result type");
                }
            }
            MirCastKindAttr::PointerExposeAddress => {
                if opd_ty_obj.downcast_ref::<MirPtrType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerExposeAddress cast requires pointer operand type"
                    );
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerExposeAddress cast requires integer result type"
                    );
                }
            }
            MirCastKindAttr::PointerWithExposedProvenance => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance cast requires integer operand type"
                    );
                }
                let Some(result_ptr) = res_ty_obj.downcast_ref::<MirPtrType>() else {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance cast requires pointer result type"
                    );
                };
                if !matches!(
                    result_ptr.kind,
                    MirPointerKind::Erased | MirPointerKind::RawConst | MirPointerKind::RawMut
                ) {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance can only materialize an Erased or raw pointer kind"
                    );
                }
                if result_ptr.kind == MirPointerKind::Erased
                    && !is_opaque_fn_pointer_type(ctx, res_ty)
                {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance may materialize Erased only as the canonical immutable function-pointer carrier"
                    );
                }
            }
            // PtrToPtr, FnPtrToPtr, Transmute, PointerCoercion*, Subtype: operand/result can be
            // ptr, struct, tuple, etc.; lowering handles the details. No strict type check here.
            MirCastKindAttr::PtrToPtr
            | MirCastKindAttr::FnPtrToPtr
            | MirCastKindAttr::Transmute
            | MirCastKindAttr::PointerCoercionUnsize
            | MirCastKindAttr::PointerCoercionMutToConst
            | MirCastKindAttr::PointerCoercionArrayToPointer
            | MirCastKindAttr::PointerCoercionReifyFnPointer
            | MirCastKindAttr::PointerCoercionUnsafeFnPointer
            | MirCastKindAttr::PointerCoercionClosureFnPointer
            | MirCastKindAttr::Subtype => {}
        }

        // Pointer kind is semantic provenance. A representation-only cast may
        // preserve it or deliberately forget it, but must never recover a
        // concrete category from Erased or switch concrete categories. Those
        // transitions require an explicit, verifier-visible Rust boundary.
        let transitions = pointer_kind_transitions(ctx, opd_ty, res_ty);
        let top_level_target_carrier = res_ty_obj
            .downcast_ref::<MirPtrType>()
            .map(|pointer| PointerCarrier {
                kind: pointer.kind,
                is_mutable: pointer.is_mutable,
            })
            .or_else(|| {
                res_ty_obj
                    .downcast_ref::<MirSliceType>()
                    .map(|slice| PointerCarrier {
                        kind: slice.kind,
                        is_mutable: slice.is_mutable,
                    })
            });
        let top_level_source_carrier = opd_ty_obj
            .downcast_ref::<MirPtrType>()
            .map(|pointer| PointerCarrier {
                kind: pointer.kind,
                is_mutable: pointer.is_mutable,
            })
            .or_else(|| {
                opd_ty_obj
                    .downcast_ref::<MirSliceType>()
                    .map(|slice| PointerCarrier {
                        kind: slice.kind,
                        is_mutable: slice.is_mutable,
                    })
            });
        let authority = self
            .get_attr_pointer_kind_authority(ctx)
            .map(|authority| authority.clone());
        let target_pointer_carriers = pointer_carriers_in_type(ctx, res_ty);
        let concrete_target_kinds: Vec<_> = transitions
            .iter()
            .map(|(_, target)| target.kind)
            .filter(|target| *target != MirPointerKind::Erased)
            .collect();

        let source_is_opaque_fn_pointer = is_opaque_fn_pointer_type(ctx, opd_ty);
        let target_is_opaque_fn_pointer = is_opaque_fn_pointer_type(ctx, res_ty);

        // Opaque function pointers are a deliberately narrow Erased carrier,
        // not general-purpose storage. Validate the source-language coercion
        // shape before the generic pointer transition rules see it.
        match &cast_kind {
            MirCastKindAttr::FnPtrToPtr
                if !source_is_opaque_fn_pointer
                    || !top_level_target_carrier.is_some_and(|target| is_raw_kind(target.kind)) =>
            {
                return verify_err!(
                    loc,
                    "MirCastOp FnPtrToPtr requires the canonical opaque function-pointer source and a raw-pointer result"
                );
            }
            MirCastKindAttr::PointerCoercionReifyFnPointer
            | MirCastKindAttr::PointerCoercionClosureFnPointer => {
                return verify_err!(
                    loc,
                    "MirCastOp {:?} is not a lowerable MIR operation; the importer must materialize the canonical function token directly",
                    cast_kind
                );
            }
            MirCastKindAttr::PointerCoercionUnsafeFnPointer
                if !source_is_opaque_fn_pointer || !target_is_opaque_fn_pointer =>
            {
                return verify_err!(
                    loc,
                    "MirCastOp UnsafeFnPointer requires canonical opaque function-pointer operand and result types"
                );
            }
            _ if source_is_opaque_fn_pointer != target_is_opaque_fn_pointer
                && !matches!(
                    cast_kind,
                    MirCastKindAttr::FnPtrToPtr
                        | MirCastKindAttr::PointerWithExposedProvenance
                        | MirCastKindAttr::Transmute
                ) =>
            {
                return verify_err!(
                    loc,
                    "MirCastOp {:?} cannot reinterpret the canonical function-pointer carrier",
                    cast_kind
                );
            }
            _ => {}
        }

        let is_explicit_aggregate_transmute = cast_kind == MirCastKindAttr::Transmute
            && authority.as_ref() == Some(&MirPointerKindAuthorityAttr::RustCast);
        if !source_is_opaque_fn_pointer
            && !target_is_opaque_fn_pointer
            && !is_explicit_aggregate_transmute
            && !nested_opaque_fn_pointer_positions_match(ctx, opd_ty, res_ty)
        {
            return verify_err!(
                loc,
                "MirCastOp cannot reinterpret a nested canonical function-pointer carrier without an explicit Rust Transmute"
            );
        }

        // These source-level coercions have narrower structural meaning than
        // their carrier-kind transition alone can express. Enforce that
        // meaning even when no authority is needed because the pointer kind
        // happens to be preserved (for example, RawMut -> RawMut array
        // decay). Otherwise a hand-built op could label an arbitrary
        // same-kind pointer cast as ArrayToPointer or MutToConst.
        if matches!(
            cast_kind,
            MirCastKindAttr::PointerCoercionMutToConst
                | MirCastKindAttr::PointerCoercionArrayToPointer
        ) && !rust_cast_transition_is_admissible(
            ctx,
            cast_kind.clone(),
            opd_ty,
            res_ty,
            top_level_source_carrier,
            top_level_target_carrier,
        ) {
            return verify_err!(
                loc,
                "MirCastOp {:?} has an unsupported pointer coercion shape",
                cast_kind
            );
        }

        if cast_kind == MirCastKindAttr::Transmute
            && !target_pointer_carriers.is_empty()
            && opd_ty != res_ty
            && authority.as_ref() != Some(&MirPointerKindAuthorityAttr::RustCast)
        {
            return verify_err!(
                loc,
                "MirCastOp Transmute producing a pointer carrier requires RustCast authority"
            );
        }

        // Unsize is a narrow thin-to-fat operation in this dialect; Subtype
        // erases only source-language distinctions that are absent from the
        // translated type and must therefore be type-identical here. Do not
        // accept equal flattened carrier lists in rearranged aggregates.
        if (cast_kind == MirCastKindAttr::PointerCoercionUnsize
            && !supported_unsize_shape(ctx, opd_ty, res_ty))
            || (cast_kind == MirCastKindAttr::Subtype && opd_ty != res_ty)
        {
            return verify_err!(
                loc,
                "MirCastOp {:?} has an unsupported or representation-changing carrier shape",
                cast_kind
            );
        }

        // These rustc cast kinds operate on pointer carriers. The lowering
        // also supports aggregate fat-pointer carriers, so inspect the whole
        // type graph rather than requiring only a top-level MirPtrType. An
        // authority label must not make structurally impossible casts such as
        // integer -> UniqueRef valid; exposed-provenance and transmute are the
        // explicit operations for integer/arbitrary-byte reinterpretation.
        if matches!(
            cast_kind,
            MirCastKindAttr::PtrToPtr | MirCastKindAttr::FnPtrToPtr
        ) && (pointer_kinds_in_type(ctx, opd_ty).is_empty() || transitions.is_empty())
        {
            return verify_err!(
                loc,
                "MirCastOp {:?} requires pointer-carrying operand and result types",
                cast_kind
            );
        }

        // Declaration-order pairing is sufficient only when the aggregate
        // type itself is unchanged. A representation-reinterpreting cast can
        // move a declared pointer field to a byte offset that previously held
        // an integer (or another unrelated value), even if both declarations
        // list the same pointer kind at the same field index. Conservatively
        // require the explicit rustc-cast authority whenever a changed
        // aggregate result contains a concrete pointer carrier.
        if top_level_target_carrier.is_none()
            && !concrete_target_kinds.is_empty()
            && opd_ty != res_ty
            && authority.as_ref() != Some(&MirPointerKindAuthorityAttr::RustCast)
        {
            return verify_err!(
                loc,
                "MirCastOp cannot reinterpret an aggregate containing a concrete pointer kind without RustCast authority"
            );
        }

        if let Some(authority) = authority.as_ref() {
            if source_is_opaque_fn_pointer && *authority != MirPointerKindAuthorityAttr::RustCast {
                return verify_err!(
                    loc,
                    "MirCastOp cannot use {:?} to reinterpret an opaque function-pointer value as a data pointer",
                    authority
                );
            }
            if concrete_target_kinds.is_empty()
                && !(*authority == MirPointerKindAuthorityAttr::RustCast
                    && cast_kind == MirCastKindAttr::Transmute
                    && !target_pointer_carriers.is_empty())
            {
                return verify_err!(
                    loc,
                    "MirCastOp pointer-kind authority requires a concrete pointer carrier in the result type"
                );
            }

            if *authority != MirPointerKindAuthorityAttr::RustCast
                && top_level_target_carrier.is_none()
            {
                return verify_err!(
                    loc,
                    "MirCastOp pointer-kind authority {:?} only applies to a top-level pointer/slice result; nested aggregate transitions require RustCast",
                    authority
                );
            }

            let authority_matches_cast = match authority {
                MirPointerKindAuthorityAttr::Reborrow
                | MirPointerKindAuthorityAttr::RawAddress
                | MirPointerKindAuthorityAttr::AbiBoundary => {
                    cast_kind == MirCastKindAttr::PtrToPtr
                }
                MirPointerKindAuthorityAttr::StaticAddress => matches!(
                    cast_kind,
                    MirCastKindAttr::PtrToPtr | MirCastKindAttr::PointerWithExposedProvenance
                ),
                MirPointerKindAuthorityAttr::RustCast => matches!(
                    cast_kind,
                    MirCastKindAttr::PtrToPtr
                        | MirCastKindAttr::FnPtrToPtr
                        | MirCastKindAttr::PointerWithExposedProvenance
                        | MirCastKindAttr::Transmute
                        | MirCastKindAttr::PointerCoercionUnsize
                        | MirCastKindAttr::PointerCoercionMutToConst
                        | MirCastKindAttr::PointerCoercionArrayToPointer
                        | MirCastKindAttr::PointerCoercionUnsafeFnPointer
                        | MirCastKindAttr::Subtype
                ),
                // Inline assembly is a producer authority. It is never valid
                // on a cast, even when the cast result carries a pointer.
                MirPointerKindAuthorityAttr::InlineAsm => false,
            };
            if !authority_matches_cast {
                return verify_err!(
                    loc,
                    "MirCastOp pointer-kind authority {:?} is incompatible with cast kind {:?}",
                    authority,
                    cast_kind
                );
            }

            if *authority == MirPointerKindAuthorityAttr::RustCast
                && !rust_cast_transition_is_admissible(
                    ctx,
                    cast_kind.clone(),
                    opd_ty,
                    res_ty,
                    top_level_source_carrier,
                    top_level_target_carrier,
                )
            {
                return verify_err!(
                    loc,
                    "MirCastOp RustCast authority is incompatible with {:?} pointer transition",
                    cast_kind
                );
            }

            if matches!(
                authority,
                MirPointerKindAuthorityAttr::Reborrow
                    | MirPointerKindAuthorityAttr::RawAddress
                    | MirPointerKindAuthorityAttr::AbiBoundary
                    | MirPointerKindAuthorityAttr::StaticAddress
            ) && cast_kind == MirCastKindAttr::PtrToPtr
                && !matching_top_level_pointer_shape(ctx, opd_ty, res_ty)
            {
                return verify_err!(
                    loc,
                    "MirCastOp pointer-kind authority {:?} requires pointer-to-pointer or slice-to-slice operands with matching pointee/element types",
                    authority
                );
            }

            let promoted_empty_unique_ref = *authority
                == MirPointerKindAuthorityAttr::StaticAddress
                && cast_kind == MirCastKindAttr::PtrToPtr
                && is_promoted_empty_unique_ref_transition(ctx, opd_val, opd_ty, res_ty);

            if let Some(target_kind) =
                concrete_target_kinds
                    .iter()
                    .copied()
                    .find(|target_kind| !match authority {
                        MirPointerKindAuthorityAttr::Reborrow => target_kind.is_reference(),
                        MirPointerKindAuthorityAttr::RawAddress => matches!(
                            target_kind,
                            MirPointerKind::RawConst | MirPointerKind::RawMut
                        ),
                        MirPointerKindAuthorityAttr::StaticAddress => {
                            matches!(
                                target_kind,
                                MirPointerKind::SharedRef
                                    | MirPointerKind::RawConst
                                    | MirPointerKind::RawMut
                            ) || *target_kind == MirPointerKind::UniqueRef
                                && promoted_empty_unique_ref
                        }
                        MirPointerKindAuthorityAttr::RustCast
                        | MirPointerKindAuthorityAttr::AbiBoundary => true,
                        MirPointerKindAuthorityAttr::InlineAsm => false,
                    })
            {
                return verify_err!(
                    loc,
                    "MirCastOp pointer-kind authority {:?} cannot establish target kind {:?}",
                    authority,
                    target_kind
                );
            }

            // The authority says which Rust event happened, but it does not
            // make every source semantically admissible. In particular, an
            // immutable pointer cannot become `&mut T` or `*mut T` merely by
            // labelling a PtrToPtr cast as a reborrow/raw-address operation.
            if cast_kind == MirCastKindAttr::PtrToPtr
                && *authority != MirPointerKindAuthorityAttr::RustCast
            {
                let source_carrier = top_level_source_carrier
                    .expect("matching_top_level_pointer_shape accepted a non-pointer source");
                let target_carrier = top_level_target_carrier
                    .expect("non-RustCast PtrToPtr authority accepted a non-pointer target");
                // Generic casts and projections preserve carrier mutability,
                // including for Erased carriers. Establishing a *mutable*
                // concrete kind therefore needs either an already-writable
                // concrete category or an Erased thin/fat carrier whose
                // preserved mutability bit says the underlying address is
                // writable.
                let source_is_writable = match source_carrier.kind {
                    MirPointerKind::Erased => source_carrier.is_mutable,
                    MirPointerKind::UniqueRef | MirPointerKind::RawMut => true,
                    MirPointerKind::SharedRef | MirPointerKind::RawConst => false,
                };

                if *authority == MirPointerKindAuthorityAttr::StaticAddress
                    && source_carrier.kind == MirPointerKind::Erased
                    && !has_erased_storage_lineage(ctx, opd_val, ErasedStorageOrigin::Static)
                {
                    return verify_err!(
                        loc,
                        "MirCastOp StaticAddress requires Erased storage rooted in mir.global_alloc or mir.shared_alloc"
                    );
                }
                if *authority == MirPointerKindAuthorityAttr::AbiBoundary
                    && source_carrier.kind == MirPointerKind::Erased
                    && !has_erased_storage_lineage(ctx, opd_val, ErasedStorageOrigin::Compiler)
                {
                    return verify_err!(
                        loc,
                        "MirCastOp AbiBoundary requires an Erased source rooted in mir.alloca, or an already exact concrete pointer kind"
                    );
                }

                let admissible = match authority {
                    MirPointerKindAuthorityAttr::Reborrow => match target_carrier.kind {
                        MirPointerKind::SharedRef => true,
                        MirPointerKind::UniqueRef => source_is_writable,
                        _ => false,
                    },
                    MirPointerKindAuthorityAttr::RawAddress => match target_carrier.kind {
                        MirPointerKind::RawConst => true,
                        MirPointerKind::RawMut => source_is_writable,
                        _ => false,
                    },
                    MirPointerKindAuthorityAttr::StaticAddress => match target_carrier.kind {
                        MirPointerKind::SharedRef | MirPointerKind::RawConst => {
                            source_carrier.kind == MirPointerKind::Erased
                        }
                        MirPointerKind::RawMut => {
                            source_carrier.kind == MirPointerKind::Erased && source_is_writable
                        }
                        MirPointerKind::UniqueRef => promoted_empty_unique_ref,
                        _ => false,
                    },
                    MirPointerKindAuthorityAttr::AbiBoundary => {
                        (source_carrier.kind == MirPointerKind::Erased
                            || source_carrier.kind == target_carrier.kind)
                            && (target_carrier.kind.expected_mutability() != Some(true)
                                || source_is_writable)
                    }
                    MirPointerKindAuthorityAttr::RustCast => true,
                    MirPointerKindAuthorityAttr::InlineAsm => false,
                };
                if !admissible {
                    return verify_err!(
                        loc,
                        "MirCastOp pointer-kind authority {:?} cannot establish target kind {:?} from source kind {:?}",
                        authority,
                        target_carrier.kind,
                        source_carrier.kind
                    );
                }
            }
        }

        if let Some((source_carrier, target_carrier)) = transitions.iter().copied().find(
            |(source_carrier, target_carrier)| match source_carrier {
                Some(source_carrier) => {
                    !source_carrier
                        .kind
                        .can_retype_generically_to(target_carrier.kind)
                        || (authority.is_none()
                            && source_carrier.is_mutable != target_carrier.is_mutable)
                }
                None => {
                    // Creating an opaque top-level function-pointer carrier
                    // is the one non-Transmute operation that legitimately
                    // has no source pointer carrier. Exposed-provenance
                    // materialization has the same explicit integer->Erased
                    // shape. Aggregate casts and ordinary pointer casts may
                    // not manufacture even Erased carriers: writable Erased
                    // is evidence accepted by later Rust boundaries.
                    let creates_opaque_top_level_pointer = target_is_opaque_fn_pointer
                        && matches!(cast_kind, MirCastKindAttr::PointerWithExposedProvenance);
                    let creates_structurally_supported_fat_pointer = cast_kind
                        == MirCastKindAttr::PointerCoercionUnsize
                        && supported_unsize_shape(ctx, opd_ty, res_ty);
                    !(creates_opaque_top_level_pointer
                        || creates_structurally_supported_fat_pointer)
                }
            },
        ) && authority.is_none()
        {
            return verify_err!(
                loc,
                "MirCastOp cannot generically retype pointer carrier {:?} to {:?} without an explicit Rust pointer-kind authority",
                source_carrier,
                target_carrier
            );
        }

        Ok(())
    }
}

/// Register cast operations into the given context.
pub fn register(ctx: &mut Context) {
    MirCastOp::register(ctx);
}
