/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pointer provenance/overlap analysis over lowered LLVM aggregate storage.

use llvm_export::types as llvm_types;
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::llvm_type_size_align;
use crate::convert::enum_payload_storage::MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES;

pub(super) fn llvm_type_contains_pointer(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<llvm_types::PointerType>() {
        return true;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_pointer(ctx, array.elem_type());
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_pointer(ctx, vector.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_pointer(ctx, field));
    }
    false
}

/// One pointer-valued leaf in an LLVM aggregate's physical storage.
///
/// Offsets are absolute within the enclosing enum. Keeping the address space
/// in the identity prevents an AS0 pointer carrier from being treated as the
/// same storage as an AS1 payload pointer merely because both are eight bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LlvmPointerStorage {
    pub(super) offset: u64,
    pub(super) size: u64,
    pub(super) address_space: u32,
}

/// Why a pointer-storage walk refused a type or an overlap.
///
/// Distinguishing bounded array expansion from genuine provenance loss keeps
/// the user-facing diagnostic specific: an oversized pointer array reports the
/// same "rewrite requires N pointer conversions" contract as the payload
/// storage gate instead of a misleading provenance error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PointerOverlapRejection {
    /// Expanding the walked type's fixed arrays into per-leaf records would
    /// exceed [`MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES`]; `required` is the
    /// total number of array-expanded pointer leaves.
    OverArrayLeafBound { required: u64 },
    /// The type holds pointer storage this walk cannot represent, or the
    /// overlap would pun a pointer against non-matching bytes.
    ProvenanceLoss,
}

/// Total pointer leaves in `ty`, through arrays and structs, with checked
/// arithmetic. Pointer vectors count zero; every walk that expands leaves
/// fails closed on them separately.
fn count_pointer_leaves(ctx: &Context, ty: TypeHandle) -> Option<u64> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<llvm_types::PointerType>() {
        return Some(1);
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return count_pointer_leaves(ctx, array.elem_type())?.checked_mul(array.size());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let mut total = 0u64;
        for field in fields {
            total = total.checked_add(count_pointer_leaves(ctx, field)?)?;
        }
        return Some(total);
    }
    Some(0)
}

/// Pointer leaves that expanding `ty`'s fixed arrays contributes to a
/// pointer-storage walk.
///
/// Leaves outside arrays are proportional to the source text and stay
/// unbounded, exactly like struct nesting in `enum_payload_storage_type`.
/// `[&T; N]` expands from three tokens into `N` records, so only
/// array-expanded leaves count against
/// [`MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES`].
fn count_array_pointer_leaves(ctx: &Context, ty: TypeHandle) -> Option<u64> {
    let ty_ref = ty.deref(ctx);
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return count_pointer_leaves(ctx, array.elem_type())?.checked_mul(array.size());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let mut total = 0u64;
        for field in fields {
            total = total.checked_add(count_array_pointer_leaves(ctx, field)?)?;
        }
        return Some(total);
    }
    Some(0)
}

/// Record every pointer-valued leaf in `ty` at its natural LLVM byte offset.
///
/// This is deliberately a physical-layout walk rather than a simple
/// `contains_pointer` predicate. A niche carrier may be one field inside an
/// aggregate payload, for example the pointer at byte 8 in
/// `Option<(usize, &T)>`. In that case the aggregate and the carrier overlap,
/// but they agree exactly about which bytes hold the pointer.
///
/// Expanding an arbitrary array into one record per pointer would let a valid
/// but enormous type consume unbounded verifier memory, so the walk first
/// counts the array-expanded leaves and refuses anything over
/// [`MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES`] with the bound-specific
/// rejection before allocating a single record.
fn collect_llvm_pointer_storage(
    ctx: &Context,
    ty: TypeHandle,
    base_offset: u64,
    out: &mut Vec<LlvmPointerStorage>,
) -> std::result::Result<(), PointerOverlapRejection> {
    fn collect(
        ctx: &Context,
        ty: TypeHandle,
        base_offset: u64,
        out: &mut Vec<LlvmPointerStorage>,
    ) -> Option<()> {
        let ty_ref = ty.deref(ctx);
        if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
            let (size, _) = llvm_type_size_align(ctx, ty)?;
            out.push(LlvmPointerStorage {
                offset: base_offset,
                size,
                address_space: pointer.address_space(),
            });
            return Some(());
        }
        if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            let element_ty = array.elem_type();
            if !llvm_type_contains_pointer(ctx, element_ty) {
                return Some(());
            }
            let (element_size, _) = llvm_type_size_align(ctx, element_ty)?;
            for index in 0..array.size() {
                let element_offset = element_size.checked_mul(index)?;
                collect(
                    ctx,
                    element_ty,
                    base_offset.checked_add(element_offset)?,
                    out,
                )?;
            }
            return Some(());
        }
        if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
            // Pointer vectors carry vector-specific ABI alignment and cast
            // semantics. Keep them fail-closed rather than treating them like
            // fixed arrays.
            return (!llvm_type_contains_pointer(ctx, vector.elem_type())).then_some(());
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            let fields: Vec<_> = struct_ty.fields().collect();
            let packed = struct_ty.layout() == llvm_types::StructLayout::Packed;
            let mut end = 0u64;
            for field in fields {
                let (field_size, field_align) = llvm_type_size_align(ctx, field)?;
                let field_offset = if packed {
                    end
                } else {
                    let field_align = field_align.max(1);
                    let remainder = end % field_align;
                    if remainder == 0 {
                        end
                    } else {
                        end.checked_add(field_align - remainder)?
                    }
                };
                collect(ctx, field, base_offset.checked_add(field_offset)?, out)?;
                end = field_offset.checked_add(field_size)?;
            }
            return Some(());
        }

        // All pointer-bearing LLVM types understood by this lowering are
        // handled above. Unknown pointer containers must fail closed.
        (!llvm_type_contains_pointer(ctx, ty)).then_some(())
    }

    let required =
        count_array_pointer_leaves(ctx, ty).ok_or(PointerOverlapRejection::ProvenanceLoss)?;
    if required > MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES {
        return Err(PointerOverlapRejection::OverArrayLeafBound { required });
    }
    collect(ctx, ty, base_offset, out).ok_or(PointerOverlapRejection::ProvenanceLoss)
}

/// Analyze how a slotless incoming pointer-bearing field overlaps the enum's
/// already-selected storage, and report how to back every one of its pointers
/// with a real `ptr` slot.
///
/// Returns `Ok(extra)` when the field is representable without erasing any
/// pointer's provenance: every pointer leaf that coincides with an existing
/// claim reuses that claim, and each remaining ("extra") pointer leaf lands on
/// bytes no other claim covers, so it can be backed by its own fresh `ptr`
/// slot. `extra` lists exactly those leaves for the caller to add as claims.
/// An empty `extra` is the exact-match case (e.g. `Option<(usize, &T)>`, whose
/// single pointer already coincides with the niche carrier); a non-empty
/// `extra` is the multi-pointer payload case (e.g. `Option<(&mut [T],
/// &mut [T])>` from `split_at_mut_checked`, whose second slice pointer needs a
/// slot of its own beside the carrier).
///
/// Returns [`PointerOverlapRejection::ProvenanceLoss`] when the field cannot
/// be represented without punning a pointer against non-pointer bits: an
/// existing pointer slot the incoming field does not also carry at the same
/// offset/size/address space, or an extra pointer leaf that would overlap an
/// existing (necessarily non-pointer) claim. That genuine pointer/integer
/// union stays fail-closed — there is no single LLVM slot type that is both
/// provenance-carrying and integer-exact.
///
/// Returns [`PointerOverlapRejection::OverArrayLeafBound`] when a walked type
/// holds more array-expanded pointer leaves than the bounded rewrite limit,
/// so the caller can report the bound instead of a provenance error.
pub(super) fn analyze_pointer_overlap(
    ctx: &Context,
    incoming_offset: u64,
    incoming_size: u64,
    incoming_ty: TypeHandle,
    colliding_claims: &[&(u64, u64, TypeHandle)],
) -> std::result::Result<Vec<LlvmPointerStorage>, PointerOverlapRejection> {
    use PointerOverlapRejection::ProvenanceLoss;

    let incoming_end = incoming_offset
        .checked_add(incoming_size)
        .ok_or(ProvenanceLoss)?;

    let mut incoming = Vec::new();
    collect_llvm_pointer_storage(ctx, incoming_ty, incoming_offset, &mut incoming)?;

    let mut existing = Vec::new();
    for &&(offset, _size, claim_ty) in colliding_claims {
        let mut regions = Vec::new();
        collect_llvm_pointer_storage(ctx, claim_ty, offset, &mut regions)?;
        existing.extend(regions.into_iter().filter(|region| {
            let Some(region_end) = region.offset.checked_add(region.size) else {
                return true;
            };
            region.offset < incoming_end && incoming_offset < region_end
        }));
    }

    // Every existing pointer leaf overlapping this field must be one the field
    // also carries at the same offset/size/address space; otherwise a pointer
    // slot would be backed by non-matching bytes (an address-space mismatch, or
    // a pointer claim where the field has integer bits).
    for leaf in &existing {
        if !incoming.contains(leaf) {
            return Err(ProvenanceLoss);
        }
    }

    // Each incoming pointer leaf that does not reuse an existing claim needs its
    // own slot, and may only get one if it lands on otherwise-unclaimed bytes. A
    // leaf overlapping a claim it did not match is a pointer/non-pointer pun and
    // stays fail-closed.
    let mut extra = Vec::new();
    for leaf in &incoming {
        if existing.contains(leaf) {
            continue;
        }
        let leaf_end = leaf.offset.checked_add(leaf.size).ok_or(ProvenanceLoss)?;
        let overlaps_claim = colliding_claims.iter().any(|&&(o, s, _)| {
            let Some(claim_end) = o.checked_add(s) else {
                return true;
            };
            o < leaf_end && leaf.offset < claim_end
        });
        if overlaps_claim {
            return Err(ProvenanceLoss);
        }
        extra.push(*leaf);
    }

    Ok(extra)
}

pub(crate) fn llvm_type_contains_pointer_in_address_space(
    ctx: &Context,
    ty: TypeHandle,
    address_space: u32,
) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
        return pointer.address_space() == address_space;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_pointer_in_address_space(ctx, array.elem_type(), address_space);
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_pointer_in_address_space(ctx, vector.elem_type(), address_space);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_pointer_in_address_space(ctx, field, address_space));
    }
    false
}

/// Whether a physical by-value image contains a packed struct whose bytes
/// include a pointer in `address_space`.
///
/// Direct pointers in an unpacked aggregate do not trigger this predicate:
/// their ABI can preserve address-space semantics without observing the
/// pointer's raw storage width. A pointer nested anywhere under an LLVM packed
/// struct does trigger it because whole-value construction/load/store/ABI
/// traffic observes that packed physical image. This distinction keeps normal
/// AS3 aggregate handling intact while rejecting the target-dependent packed
/// case (modern NVVM p3:32 versus 64-bit PTX/legacy).
pub(crate) fn llvm_packed_struct_contains_pointer_in_address_space(
    ctx: &Context,
    ty: TypeHandle,
    address_space: u32,
) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_packed_struct_contains_pointer_in_address_space(
            ctx,
            array.elem_type(),
            address_space,
        );
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_packed_struct_contains_pointer_in_address_space(
            ctx,
            vector.elem_type(),
            address_space,
        );
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        if struct_ty.layout() == llvm_types::StructLayout::Packed
            && fields
                .iter()
                .copied()
                .any(|field| llvm_type_contains_pointer_in_address_space(ctx, field, address_space))
        {
            return true;
        }
        return fields.into_iter().any(|field| {
            llvm_packed_struct_contains_pointer_in_address_space(ctx, field, address_space)
        });
    }
    false
}
