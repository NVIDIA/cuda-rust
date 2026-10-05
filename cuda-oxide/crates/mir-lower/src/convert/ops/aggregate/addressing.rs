/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::{anyhow_to_pliron, byte_offset_gep};
use crate::convert::enum_payload_storage::enum_payload_storage_type;
use crate::convert::types::{
    StructLayoutInfo, build_enum_slot_map, build_struct_slot_map, convert_type,
    llvm_type_size_align, mir_element_stride, mir_type_abi_align,
};
use dialect_mir::ops::{MirConstantOp, MirFieldAddrOp};
use dialect_mir::types::{MirEnumType, MirPtrType, MirStructType, MirTupleType, MirUnionType};
use llvm_export::attributes::GepNoWrapFlags;
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

// ============================================================================
// MirFieldAddrOp Conversion
// ============================================================================

/// Convert `mir.field_addr` to `llvm.getelementptr`.
///
/// Computes the address of a struct field using GEP. This is needed when
/// Rust code takes `&mut self.field` — we need the ADDRESS of the field,
/// not a COPY of its value.
///
/// The GEP field index and the struct type it indexes into both come from
/// [`build_struct_slot_map`], so the index accounts for reordering,
/// `[N x i8]` padding slots and stripped ZSTs (ZST-ness is decided on the
/// converted LLVM field type, like the value-level sites).
///
/// Taking the address of a ZST field emits a distinct zero-offset `i8` GEP
/// off the base pointer (a ZST field lives at byte 0 of the struct), the
/// same idiom the union branch uses. It must NOT forward the base SSA value
/// itself: a 1:1 `replace_operation_with_values` that aliases the op's
/// result to an already-existing value records the result's pointee type
/// (the ZST field's own zero-field type) onto that value's conversion type
/// history, so a sibling `field_addr` on the same base would later resolve
/// the base's pointee to the ZST type instead of the struct and fail with
/// "field_addr index N out of bounds for struct with 0 fields". The
/// distinct GEP result leaves the base pointer's recorded pointee intact
/// and carries the field's pointee type on its own result, which is what
/// nested projections through the field address look up.
pub(crate) fn convert_field_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr_operand = op.deref(ctx).get_operand(0);

    let field_addr_op = MirFieldAddrOp::new(op);
    let field_index = match field_addr_op.get_attr_field_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("MirFieldAddrOp missing field_index attribute"),
    };

    let mir_ptr_pointee =
        match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, ptr_operand) {
            Some(r) => r.pointee,
            None => {
                return pliron::input_err_noloc!("MirFieldAddrOp operand must be pointer type");
            }
        };

    let union_field_count = mir_ptr_pointee
        .deref(ctx)
        .downcast_ref::<MirUnionType>()
        .map(MirUnionType::field_count);
    if let Some(field_count) = union_field_count {
        if field_index >= field_count {
            return pliron::input_err_noloc!(
                "field_addr index {} out of bounds for union with {} fields",
                field_index,
                field_count
            );
        }
        // Every union field begins at byte zero. Emit an explicit zero-offset
        // GEP instead of forwarding the base SSA value directly: the distinct
        // result keeps dialect conversion's pointer-type history unambiguous
        // for repeated field accesses and for `union.struct_field.inner`.
        use llvm_export::ops::GepIndex;
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
            ctx,
            ptr_operand,
            vec![GepIndex::Constant(0)],
            i8_ty,
            GepNoWrapFlags::INBOUNDS,
        );
        rewriter.insert_operation(ctx, gep.get_operation());
        rewriter.replace_operation(ctx, op, gep.get_operation());
        return Ok(());
    }

    // An enum payload field, addressed by its position in the flattened
    // `all_field_types`. Variants share bytes, so the slot map resolves each
    // field one of two ways: its own LLVM slot, or, when its bytes are already
    // held by a differently typed field of another variant, a byte offset into
    // the enum. Both give the address of the same storage, which is what makes
    // a write through the returned pointer land in the enum rather than a copy.
    let enum_name = mir_ptr_pointee
        .deref(ctx)
        .downcast_ref::<MirEnumType>()
        .map(|enum_ty| enum_ty.name().to_string());
    if let Some(enum_name) = enum_name {
        let map = build_enum_slot_map(ctx, mir_ptr_pointee).map_err(anyhow_to_pliron)?;
        let Some(slot_entry) = map.field_slots.get(field_index).copied() else {
            return pliron::input_err_noloc!(
                "field_addr index {} out of bounds for enum with {} payload fields",
                field_index,
                map.field_slots.len()
            );
        };

        // Enum payload bytes hold one CANONICAL storage type that can differ
        // from the field's semantic type: a bool is physically an i8 byte and
        // a shared-memory pointer is stored as a generic pointer, with the
        // value paths (construct/extract) coercing exactly at that boundary.
        // The address computed here ESCAPES this site: the loads and stores
        // made through it happen at arbitrary other sites and are typed with
        // the SEMANTIC type, so no storage coercion can be attached at
        // address-formation time. Handing the address out anyway would let an
        // i1 store leave the byte's upper seven bits undefined for the i8 and
        // niche-tag readers, or write a shared-pointer representation into
        // bytes every other reader interprets (and, on modern NVVM, sizes) as
        // a generic pointer. Fail closed instead, the same way the slot map
        // already rejects bool bytes hidden behind unions.
        let semantic_ty = map.field_llvm_types[field_index];
        let storage_ty = enum_payload_storage_type(ctx, semantic_ty).map_err(anyhow_to_pliron)?;
        if storage_ty != semantic_ty {
            return pliron::input_err_noloc!(
                "field_addr: cannot hand out the in-place address of payload field {} of enum `{}`: its bytes use canonical storage type {} while its semantic type is {}, and loads or stores made through an escaped payload address are typed with the semantic type, which the canonical bytes cannot honor; shared reads of such payloads are compiled through a value copy automatically, and a write that stays inside its function is compiled by rebuilding the enum around the new payload; a borrow that escapes into a call keeps no such rewrite and is refused here",
                field_index,
                enum_name,
                storage_ty.deref(ctx).disp(ctx),
                semantic_ty.deref(ctx).disp(ctx)
            );
        }

        use llvm_export::ops::GepIndex;
        let gep = match slot_entry {
            Some(slot) => llvm::GetElementPtrOp::new_with_no_wrap_flags(
                ctx,
                ptr_operand,
                vec![GepIndex::Constant(0), GepIndex::Constant(slot)],
                map.llvm_struct_ty,
                GepNoWrapFlags::INBOUNDS,
            ),
            None => {
                // No slot of its own: address the bytes directly, off the
                // ORIGINAL enum pointer, so a write through the result lands
                // in the enum. A zero-sized field lands at its offset like
                // any other and simply spans nothing. The offset is always
                // present: the slot map builds `field_offsets` and
                // `field_slots` from one walk over the same fields, and the
                // bounds check above already validated the index.
                let offset = map.field_offsets[field_index];
                let offset = u32::try_from(offset).map_err(|_| {
                    pliron::input_error_noloc!(
                        "field_addr: payload byte offset {} of enum `{}` exceeds u32",
                        offset,
                        enum_name
                    )
                })?;
                let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
                llvm::GetElementPtrOp::new_with_no_wrap_flags(
                    ctx,
                    ptr_operand,
                    vec![GepIndex::Constant(offset)],
                    i8_ty,
                    GepNoWrapFlags::INBOUNDS,
                )
            }
        };
        rewriter.insert_operation(ctx, gep.get_operation());
        rewriter.replace_operation(ctx, op, gep.get_operation());
        return Ok(());
    }

    // Carried alongside the layout so the field address can record what it
    // proves about its own alignment; see `stamp_field_address_alignment`.
    let aggregate_abi_align;
    let layout = {
        let pointee_ref = mir_ptr_pointee.deref(ctx);
        if let Some(struct_ty) = pointee_ref.downcast_ref::<MirStructType>() {
            aggregate_abi_align = struct_ty.abi_align;
            StructLayoutInfo::of_struct(struct_ty)
        } else if let Some(tuple_ty) = pointee_ref.downcast_ref::<MirTupleType>() {
            aggregate_abi_align = tuple_ty.abi_align();
            StructLayoutInfo::of_tuple(tuple_ty)
        } else {
            return pliron::input_err_noloc!(
                "MirFieldAddrOp pointer must point to a struct, tuple, union or enum type, got {}",
                mir_ptr_pointee.deref(ctx).disp(ctx)
            );
        }
    };

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let slot = match map.decl_to_llvm.get(field_index) {
        Some(Some(slot)) => *slot,
        Some(None) => {
            // ZST field: it has no storage; the struct address stands in for
            // the field address. Emit an explicit zero-offset GEP instead of
            // forwarding the base SSA value directly (mirrors the union branch
            // above): the distinct result keeps dialect conversion's pointer-type
            // history unambiguous for repeated field accesses off the same base
            // pointer. Forwarding the base value here type-puns it to the field's
            // pointee, corrupting the base pointer's recorded type so a sibling
            // field_addr on the same base later resolves to the wrong (0-field)
            // pointee -- the "field_addr index N out of bounds for struct with 0
            // fields" failure on a struct with >= 2 ZST fields (e.g. a closure
            // that captures other ZST closures, like iter map_fold).
            use llvm_export::ops::GepIndex;
            let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
            let gep = llvm::GetElementPtrOp::new_with_no_wrap_flags(
                ctx,
                ptr_operand,
                vec![GepIndex::Constant(0)],
                i8_ty,
                GepNoWrapFlags::INBOUNDS,
            );
            rewriter.insert_operation(ctx, gep.get_operation());
            rewriter.replace_operation(ctx, op, gep.get_operation());
            return Ok(());
        }
        None => {
            return pliron::input_err_noloc!(
                "field_addr index {} out of bounds for struct with {} fields",
                field_index,
                map.decl_to_llvm.len()
            );
        }
    };

    let rustc_offset = layout.field_offsets.get(field_index).copied();

    // Preserve the #859 address-path contract independently of the value
    // representation. `build_struct_slot_map` retains the offsets the same
    // fields would have under natural LLVM layout; when those differ from
    // rustc's recorded offsets, keep using the original aggregate pointer plus
    // a byte GEP. The semantic value type may now be an LLVM packed struct, but
    // changing this established field-address path is outside this change.
    if let Some(expected_offset) = rustc_offset {
        let actual_offset = map
            .natural_slot_offsets
            .as_ref()
            .and_then(|offsets| offsets.get(slot as usize).copied())
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "field_addr: cannot determine LLVM byte offset of slot {} for field {}",
                    slot,
                    field_index
                )
            })?;
        if map.layout_diverges && actual_offset != expected_offset {
            let field_ptr = byte_offset_gep(ctx, rewriter, ptr_operand, expected_offset);
            let gep = field_ptr
                .defining_op()
                .expect("byte_offset_gep always returns a GEP result");
            stamp_field_address_alignment(ctx, gep, aggregate_abi_align, Some(expected_offset));
            rewriter.replace_operation(ctx, op, gep);
            return Ok(());
        }
    }

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Constant(slot)];

    let gep_op = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        ptr_operand,
        gep_indices,
        map.llvm_struct_ty,
        GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep_op.get_operation());
    stamp_field_address_alignment(
        ctx,
        gep_op.get_operation(),
        aggregate_abi_align,
        rustc_offset,
    );
    rewriter.replace_operation(ctx, op, gep_op.get_operation());

    Ok(())
}

/// Record on a field address what its alignment provably is.
///
/// A load reads its alignment from its own result type, and a scalar records
/// none -- so `p.x` on an `#[repr(C, align(8))]` struct would export with LLVM's
/// default `align 4`, dropping the guarantee rustc gave the aggregate. That is
/// what stops LoadStoreVectorizer from fusing two adjacent field reads into one
/// wide access; a whole-element copy of the same struct already vectorizes,
/// because there the load's result type *is* the aggregate.
///
/// The aggregate is aligned to `abi_align`, so its field at byte `offset` is
/// aligned to `gcd(abi_align, offset)`, and to `abi_align` itself at offset
/// zero. Both numbers are rustc's own layout, so this claims nothing the source
/// did not already guarantee.
///
/// Stamps nothing when the aggregate records no alignment (`abi_align == 0`,
/// which also stands for "layout unknown") or the field has no recorded offset.
/// A wrong alignment here is a miscompile, so every uncertain case declines and
/// leaves the previous, weaker-but-sound behaviour.
fn stamp_field_address_alignment(
    ctx: &mut Context,
    gep: Ptr<Operation>,
    abi_align: u64,
    field_offset: Option<u64>,
) {
    const fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    if abi_align == 0 {
        return;
    }
    let Some(offset) = field_offset else {
        return;
    };
    let provable = if offset == 0 {
        abi_align
    } else {
        gcd(abi_align, offset)
    };
    // Every rustc layout has a power-of-two `abi_align`, but dialect-mir only
    // verifier-enforces that for unions and enums; a malformed hand-built
    // struct or tuple layout could reach here with e.g. 12, and a
    // non-power-of-two `align N` is invalid LLVM IR that llc rejects.
    if !provable.is_power_of_two() {
        return;
    }
    if let Ok(align) = u32::try_from(provable) {
        llvm_export::ops::set_address_alignment(ctx, gep, align);
    }
}

// ============================================================================
// MirArrayElementAddrOp Conversion
// ============================================================================

/// Convert `mir.array_element_addr` to `llvm.getelementptr`.
///
/// ```text
/// &arr[i]  ──►  getelementptr T, ptr %arr_ptr, i64 %i
/// ```
///
/// - The element type `T` comes from the op's OWN result type
///   (`mir.ptr<T>`), which the dialect verifier ties to the operand
///   array's element type. Same address as the old two-index
///   `[N x T]` GEP, minus the dead `[N x T]` base.
/// - Operand type history is not usable here: a kind-only `mir.cast`
///   lowers to a plain value forwarding, history does not follow that
///   edge, and a stale hit would stride by the wrong element size.
pub(crate) fn convert_array_element_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let arr_ptr = op.deref(ctx).get_operand(0);
    let index = op.deref(ctx).get_operand(1);

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let element_ty = result_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|mir_ptr| mir_ptr.pointee)
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                "mir.array_element_addr result must be a MIR pointer type; \
                 element sizing has no fact to derive from"
            )
        })?;

    let llvm_element_ty = convert_type(ctx, element_ty).map_err(anyhow_to_pliron)?;

    // The typed GEP strides by the LLVM allocation size of the element type.
    // Keep the rustc-stride comparison as a fail-closed backstop for any
    // element representation whose LLVM allocation size differs.
    {
        let rustc_stride = mir_element_stride(ctx, element_ty);
        let llvm_size = llvm_type_size_align(ctx, llvm_element_ty).map(|(size, _)| size);
        if let (Some(stride), Some(llvm_size)) = (rustc_stride, llvm_size)
            && stride != llvm_size
        {
            return pliron::input_err_noloc!(
                "addressing elements of an array whose LLVM allocation size differs from \
                 rustc's stored stride is not supported: rustc strides by {} bytes but \
                 the LLVM element type occupies {}",
                stride,
                llvm_size
            );
        }
    }

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Value(index)];

    let element_align = element_address_provable_alignment(ctx, arr_ptr, element_ty, index);

    let gep_op = llvm::GetElementPtrOp::new_with_no_wrap_flags(
        ctx,
        arr_ptr,
        gep_indices,
        llvm_element_ty,
        GepNoWrapFlags::INBOUNDS,
    );
    rewriter.insert_operation(ctx, gep_op.get_operation());
    if let Some(align) = element_align {
        llvm_export::ops::set_address_alignment(ctx, gep_op.get_operation(), align);
    }
    rewriter.replace_operation(ctx, op, gep_op.get_operation());

    Ok(())
}

/// Alignment an array-element address provably has, in bytes.
///
/// The field path records this already ([`stamp_field_address_alignment`]), but
/// an element address recorded nothing, so `lanes[0] + lanes[1]` on an
/// `#[repr(C, align(8))]` `[f32; 2]` still exported two `align 4` loads and
/// LoadStoreVectorizer refused to fuse them. Reading the same element into a
/// local first happens to work, because SROA then loads the whole aggregate and
/// the alignment comes from its type — so the cost depended on whether the
/// source copied the element or read through it.
///
/// Two facts combine. The element's `abi_align` equals the array's (a Rust
/// array aligns to its element), and whatever the *base pointer* already
/// proved is stronger when the array is itself a field of an over-aligned
/// aggregate: `&table[i].lanes` carries the outer struct's alignment, which
/// the element type alone does not know. Element `i` then sits at byte
/// `i * stride`, so it is aligned to `gcd(base, i * stride)`, and to `base`
/// itself at index zero. A runtime index can land on any element, so it gets
/// `gcd(base, stride)`, which every element satisfies.
///
/// Answers `None`, leaving the previous behaviour, when neither the base nor
/// the element type records an alignment, or the stride is unknown. As on the
/// field path, a wrong answer here is a miscompile, so every uncertain case
/// declines rather than guesses. That is why the stride comes from
/// [`mir_element_stride`], which is exact or `None`: an LLVM-level size
/// approximation would guess 8 for a MIR aggregate element it does not
/// model, and e.g. `[[f32; 3]; 4]` under an align-8 base would then claim
/// align 8 on element addresses that are only 4-aligned.
fn element_address_provable_alignment(
    ctx: &Context,
    arr_ptr: Value,
    element_ty: TypeHandle,
    index: Value,
) -> Option<u32> {
    const fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    // What the base address already proved, else what the element type
    // states (identical to the array's own alignment).
    let base_align = arr_ptr
        .defining_op()
        .and_then(|def| llvm_export::ops::address_alignment(ctx, def))
        .map(u64::from)
        .or_else(|| mir_type_abi_align(ctx, element_ty))?;
    if base_align == 0 {
        return None;
    }

    let stride = mir_element_stride(ctx, element_ty)?;
    if stride == 0 {
        return None;
    }

    let provable = match constant_index_value(ctx, index) {
        Some(0) => base_align,
        Some(i) => gcd(base_align, i.checked_mul(stride)?),
        // Any element is reachable, so claim only what every stride preserves.
        None => gcd(base_align, stride),
    };

    // Same guard as the field path: dialect-mir does not verifier-enforce a
    // power-of-two `abi_align` for every aggregate, and a non-power-of-two
    // `align N` is invalid LLVM IR that llc rejects.
    if !provable.is_power_of_two() {
        return None;
    }
    u32::try_from(provable).ok()
}

/// The constant an index operand holds, if it is one.
///
/// `APInt::to_u64` truncates wider values, so a >64-bit constant could be
/// misread as a small offset multiplier. Fail closed on such widths, as
/// `integer_constant_u64` in the extract-element fast path does.
fn constant_index_value(ctx: &Context, index: Value) -> Option<u64> {
    let defining_op = index.defining_op()?;
    if let Some(constant) = Operation::get_op::<MirConstantOp>(defining_op, ctx) {
        let value = constant.get_attr_value(ctx)?.value();
        return (value.bw() <= 64).then(|| value.to_u64());
    }
    let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
    let attribute = constant.get_value(ctx);
    let integer = (&*attribute as &dyn pliron::attribute::Attribute)
        .downcast_ref::<pliron::builtin::attributes::IntegerAttr>()?;
    let value = integer.value();
    (value.bw() <= 64).then(|| value.to_u64())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;

    use dialect_mir::ops as mir;
    use dialect_mir::types::{
        EnumCarrierKind, EnumEncoding, EnumLayoutKind, EnumVariant, MirArrayType, MirPtrType,
        MirStructType, MirTupleType,
    };
    use llvm_export::types as llvm_types;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::common_traits::Verify;

    use super::super::test_support::*;

    fn byte_gep_constant_offset(ctx: &Context, gep: &llvm::GetElementPtrOp) -> Option<u64> {
        use llvm_export::ops::GepIndex;

        let indices = gep.indices(ctx);
        let [GepIndex::Value(offset)] = indices.as_slice() else {
            return None;
        };
        let defining_op = offset.defining_op()?;
        let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
        let attribute = constant.get_value(ctx);
        (&*attribute as &dyn pliron::attribute::Attribute)
            .downcast_ref::<IntegerAttr>()
            .map(|integer| integer.value().to_u64())
    }

    fn assert_byte_addressed_field(
        ctx: &Context,
        module: Ptr<Operation>,
        expected_offset: u64,
        expected_alignment: u32,
    ) {
        let body = kernel_blocks(ctx, module);
        let geps = find_all::<llvm::GetElementPtrOp>(ctx, &body);
        assert_eq!(geps.len(), 1, "one field_addr must lower to one GEP");

        let gep = &geps[0];
        assert_eq!(
            gep.src_elem_type(ctx),
            IntegerType::get(ctx, 8, Signedness::Signless).into(),
            "a layout-mismatched field must be addressed in byte units"
        );
        assert_eq!(
            byte_gep_constant_offset(ctx, gep),
            Some(expected_offset),
            "the byte GEP must use rustc's exact field offset"
        );
        assert_eq!(
            llvm_export::ops::address_alignment(ctx, gep.get_operation()),
            Some(expected_alignment),
            "the byte GEP must retain the alignment proved by rustc's aggregate layout"
        );
    }

    /// Sibling ZST field addresses off one base pointer must each lower to
    /// their own zero-offset `i8` GEP, never to the base SSA value itself.
    /// Forwarding the base (the pre-fix behavior) recorded the first field's
    /// zero-field pointee type onto the base pointer's conversion history, so
    /// the second `field_addr` resolved the base's pointee to "struct with 0
    /// fields" and lowering aborted. This is the `iter().map(..).sum()` shape:
    /// a composed closure holding two captureless (zero-sized) closures.
    #[test]
    fn sibling_zst_field_addrs_lower_to_distinct_zero_offset_geps() {
        use llvm_export::ops::GepIndex;

        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let zst_f = empty_struct_ty(&mut ctx, "MapClosure");
        let zst_g = empty_struct_ty(&mut ctx, "SumClosure");

        // struct Composed { f: MapClosure (ZST), g: SumClosure (ZST), acc: i64 }
        let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Composed".to_string(),
            vec!["f".to_string(), "g".to_string(), "acc".to_string()],
            vec![zst_f, zst_g, i64_ty],
            vec![0, 1, 2],
            vec![0, 0, 0],
            8,
            8,
        )
        .into();

        let base_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, struct_ty, true).into();
        // Each field_addr result records the field's own pointee type; for the
        // ZST fields that zero-field pointee is exactly what poisoned the base
        // pointer's history when the result aliased the base.
        let f_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, zst_f, true).into();
        let g_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, zst_g, true).into();
        let acc_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, i64_ty, true).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![base_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        for (field_index, result_ty) in [(0u32, f_ptr_ty), (1, g_ptr_ty), (2, acc_ptr_ty)] {
            let op = MirFieldAddrOp::build(&mut ctx, base, result_ty, field_index)
                .expect("field_addr build");
            op.insert_at_back(block, &ctx);
        }
        append_mir_return(&mut ctx, block, vec![]);

        // Pre-fix this failed with "field_addr index 1 out of bounds for
        // struct with 0 fields" on the second (sibling) ZST field_addr.
        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), 3, "each field_addr must lower to its own GEP");
        assert!(
            geps.iter().all(|gep| gep.verify(&ctx).is_ok()),
            "every field-address GEP must satisfy LLVM dialect verification"
        );

        let zst_geps: Vec<_> = geps
            .iter()
            .filter(|gep| {
                gep.src_elem_type(&ctx) == i8_ty
                    && matches!(gep.indices(&ctx).as_slice(), [GepIndex::Constant(0)])
            })
            .collect();
        assert_eq!(
            zst_geps.len(),
            2,
            "both ZST field addresses must lower to zero-offset i8 GEPs"
        );

        let gep_base = |gep: &llvm::GetElementPtrOp| gep.get_operation().deref(&ctx).get_operand(0);
        let gep_result =
            |gep: &llvm::GetElementPtrOp| gep.get_operation().deref(&ctx).get_result(0);
        assert_eq!(
            gep_base(zst_geps[0]),
            gep_base(zst_geps[1]),
            "both ZST field addresses must be taken off the same base pointer"
        );
        assert_ne!(
            gep_result(zst_geps[0]),
            gep_base(zst_geps[0]),
            "a ZST field address must be a value distinct from the base pointer"
        );
        assert_ne!(
            gep_result(zst_geps[1]),
            gep_base(zst_geps[1]),
            "a ZST field address must be a value distinct from the base pointer"
        );
        assert_ne!(
            gep_result(zst_geps[0]),
            gep_result(zst_geps[1]),
            "each sibling ZST field address must get its own GEP result"
        );

        // The non-ZST sibling still resolves the base's pointee to the struct
        // (slot 0 holds `acc`): the base pointer's type history stayed intact.
        let acc_geps: Vec<_> = geps
            .iter()
            .filter(|gep| {
                matches!(
                    gep.indices(&ctx).as_slice(),
                    [GepIndex::Constant(0), GepIndex::Constant(0)]
                )
            })
            .collect();
        assert_eq!(
            acc_geps.len(),
            1,
            "the non-ZST field address must index the struct's layout slot"
        );
    }

    /// `mir.field_addr` on a TUPLE pointee (the `#693` shape: `let (a, b) =
    /// TABLE[i];`) must resolve the GEP index through the tuple's memory-order
    /// layout, exactly like the struct path above, not through the
    /// declaration index directly.
    ///
    /// `(u8, u32)` is rustc's own layout for this pair: the `u32` field is
    /// placed FIRST in memory for alignment, so declaration field 0 (`u8`,
    /// `.0`) lives at memory slot 1 and declaration field 1 (`u32`, `.1`)
    /// lives at memory slot 0. A GEP index equal to the declaration index
    /// would silently address the WRONG field's bytes.
    #[test]
    fn field_addr_tuple_pointee_resolves_memory_order_gep_index() {
        use llvm_export::ops::GepIndex;

        let mut ctx = make_ctx();

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        // decl order [u8, u32]; memory order [u32, u8] (mem_to_decl = [1, 0]).
        let tuple_ty: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![u8_ty, u32_ty],
            vec![1, 0],
            vec![4, 0],
            8,
            4,
        )
        .into();

        let tuple_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, tuple_ty, false).into();
        let u8_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let u32_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![tuple_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        // Declaration field 0 (`.0`, u8) first, then declaration field 1
        // (`.1`, u32) -- source order, not memory order.
        for (field_index, result_ty) in [(0u32, u8_ptr_ty), (1, u32_ptr_ty)] {
            let op = MirFieldAddrOp::build(&mut ctx, base, result_ty, field_index)
                .expect("field_addr build");
            op.insert_at_back(block, &ctx);
        }
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), 2, "each field_addr must lower to its own GEP");
        assert!(
            geps.iter().all(|gep| gep.verify(&ctx).is_ok()),
            "every field-address GEP must satisfy LLVM dialect verification"
        );

        // `.0` (u8, declared first) must land at MEMORY slot 1.
        let field0_geps: Vec<_> = geps
            .iter()
            .filter(|gep| {
                matches!(
                    gep.indices(&ctx).as_slice(),
                    [GepIndex::Constant(0), GepIndex::Constant(1)]
                )
            })
            .collect();
        assert_eq!(
            field0_geps.len(),
            1,
            "declaration field 0 (u8) must resolve to its memory slot 1, not slot 0"
        );

        // `.1` (u32, declared second) must land at MEMORY slot 0.
        let field1_geps: Vec<_> = geps
            .iter()
            .filter(|gep| {
                matches!(
                    gep.indices(&ctx).as_slice(),
                    [GepIndex::Constant(0), GepIndex::Constant(0)]
                )
            })
            .collect();
        assert_eq!(
            field1_geps.len(),
            1,
            "declaration field 1 (u32) must resolve to its memory slot 0, not slot 1"
        );
    }

    /// A `#[repr(C, packed)]`-style layout can place a naturally aligned
    /// scalar at an offset LLVM's ordinary struct layout cannot express. The
    /// field address must therefore use rustc's byte offset rather than a
    /// typed struct GEP that would silently land at byte 4.
    #[test]
    fn packed_field_addr_uses_rustc_byte_offset_and_alignment_one() {
        let mut ctx = make_ctx();

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();

        let base_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, packed_ty, false).into();
        let field_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let (module, block) = build_kernel(&mut ctx, vec![base_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        let field_addr =
            MirFieldAddrOp::build(&mut ctx, base, field_ptr_ty, 1).expect("field_addr build");
        field_addr.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        assert_byte_addressed_field(&ctx, module, 1, 1);
    }

    /// `repr(packed(2))` is not equivalent to byte alignment: the same u32
    /// field is allowed to sit at byte 2 and the address still proves align 2.
    #[test]
    fn packed_two_field_addr_uses_rustc_byte_offset_and_alignment_two() {
        let mut ctx = make_ctx();

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed2".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![0, 1],
            vec![0, 2],
            6,
            2,
        )
        .into();

        let base_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, packed_ty, false).into();
        let field_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let (module, block) = build_kernel(&mut ctx, vec![base_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        let field_addr =
            MirFieldAddrOp::build(&mut ctx, base, field_ptr_ty, 1).expect("field_addr build");
        field_addr.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        assert_byte_addressed_field(&ctx, module, 2, 2);
    }

    /// The decision must compare physical offsets, not merely the selected
    /// field's own alignment. In `{ u8, u32, u8 }` the final u8 is naturally
    /// byte-aligned, but LLVM has already shifted it because the packed u32
    /// before it was placed at byte 4 instead of byte 1.
    #[test]
    fn packed_trailing_byte_field_uses_accumulated_rustc_offset() {
        let mut ctx = make_ctx();

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedTrailing".into(),
            vec!["head".into(), "value".into(), "tail".into()],
            vec![u8_ty, u32_ty, u8_ty],
            vec![0, 1, 2],
            vec![0, 1, 5],
            6,
            1,
        )
        .into();

        let base_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, packed_ty, false).into();
        let field_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let (module, block) = build_kernel(&mut ctx, vec![base_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        let field_addr =
            MirFieldAddrOp::build(&mut ctx, base, field_ptr_ty, 2).expect("field_addr build");
        field_addr.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        assert_byte_addressed_field(&ctx, module, 5, 1);
    }

    /// Packed LLVM element types carry the same allocation size as rustc, so
    /// typed array GEPs now stride by the correct packed byte count.
    #[test]
    fn packed_array_element_addressing_uses_packed_stride() {
        use dialect_mir::ops::MirArrayElementAddrOp;

        let mut ctx = make_ctx();

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let array_ty: TypeHandle = MirArrayType::get(&mut ctx, packed_ty, 4).into();
        let base_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, array_ty, false).into();
        let elem_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, packed_ty, false).into();

        let (module, block) = build_kernel(&mut ctx, vec![base_ptr_ty, u64_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);
        let index = block.deref(&ctx).get_argument(1);

        let elem_addr = Operation::new(
            &mut ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![elem_ptr_ty],
            vec![base, index],
            vec![],
            0,
        );
        elem_addr.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("packed element addressing must use the packed LLVM stride");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::GetElementPtrOp>(&ctx, &body),
            1,
            "packed array element addressing should lower to one typed GEP"
        );
    }

    /// Taking the in-place address of a bool payload must fail loudly: the
    /// payload's canonical storage is an i8 byte, and a semantic i1 store
    /// through the escaped address would leave that byte's upper seven bits
    /// undefined for every i8 reader (including a niche tag sharing them).
    #[test]
    fn bool_payload_field_addr_fails_closed() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![bool_ty], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();
        let enum_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
        let bool_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, bool_ty, true).into();

        let (module, block) = build_kernel(&mut ctx, vec![enum_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);
        let op = MirFieldAddrOp::build(&mut ctx, base, bool_ptr_ty, 0).expect("field_addr build");
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let err = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("addressing a bool enum payload in place must fail to lower");
        assert!(
            err.err.to_string().contains("canonical storage type"),
            "unexpected error: {}",
            err.err
        );
    }

    /// Same gate for the slot arm: a shared-memory pointer payload is stored
    /// as a GENERIC pointer (its slot is typed `ptr`), so an in-place address
    /// would let a semantic `ptr addrspace(3)` store write a representation
    /// every other reader interprets as a generic pointer.
    #[test]
    fn shared_pointer_payload_field_addr_fails_closed() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, true).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OptionShared".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![shared], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                carrier_address_space: llvm_types::address_space::GENERIC,
                niche_start: 0,
                niche_variant_start: 0,
                niche_variant_end: 0,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let enum_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
        let payload_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, shared, true).into();

        let (module, block) = build_kernel(&mut ctx, vec![enum_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);
        let op =
            MirFieldAddrOp::build(&mut ctx, base, payload_ptr_ty, 0).expect("field_addr build");
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let err = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("addressing a shared-pointer enum payload in place must fail to lower");
        assert!(
            err.err.to_string().contains("canonical storage type"),
            "unexpected error: {}",
            err.err
        );
    }

    /// A payload with no slot of its own is addressed at its byte offset off
    /// the ORIGINAL enum pointer: no stack spill is introduced, so a write
    /// through the result lands in the enum rather than in a copy.
    #[test]
    fn slotless_payload_field_addr_geps_original_storage_at_byte_offset() {
        use llvm_export::ops::GepIndex;
        use pliron::builtin::types::FP32Type;

        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Either".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Real".into(), vec![f32_ty], vec![4], vec![4]),
                EnumVariant::new_with_layout("Bits".into(), vec![u32_ty], vec![4], vec![4]),
            ],
            0,
            8,
            4,
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(
            slot_map.field_slots,
            vec![Some(1), None],
            "Real's f32 claims byte 4 first; Bits shares those bytes slotless"
        );

        let enum_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
        let u32_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, true).into();
        let (module, block) = build_kernel(&mut ctx, vec![enum_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);
        let op = MirFieldAddrOp::build(&mut ctx, base, u32_ptr_ty, 1).expect("field_addr build");
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            0,
            "an in-place payload address must not spill the enum to a stack copy"
        );
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), 1, "one field_addr lowers to one GEP");
        let gep = &geps[0];
        // The MIR entry block and its arguments are the ORIGINALS (moved by
        // `inline_region`), so the enum pointer argument keeps its identity
        // through lowering and the GEP must be based directly on it.
        assert_eq!(
            gep.get_operation().deref(&ctx).get_operand(0),
            base,
            "the byte-offset GEP must address the ORIGINAL enum storage"
        );
        assert!(
            matches!(gep.indices(&ctx).as_slice(), [GepIndex::Constant(4)]),
            "the slotless payload must be addressed at its rustc byte offset"
        );
        assert_eq!(
            gep.src_elem_type(&ctx),
            IntegerType::get(&ctx, 8, Signedness::Signless).into(),
            "byte addressing must step in i8 units"
        );
    }

    /// The capability split, read side: a payload whose storage IS its
    /// semantic type keeps the address path even for a SHARED borrow. The
    /// borrow lowers to one GEP into the ORIGINAL enum storage and the read
    /// to one load through it: no stack spill, no value copy. (Non-canonical
    /// payloads never get here for shared borrows; the importer punts them
    /// to the value-copy fallback before an address is formed.)
    #[test]
    fn canonical_payload_shared_read_loads_through_gep_without_copy() {
        use llvm_export::ops::GepIndex;
        use pliron::builtin::types::FP32Type;

        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Slot".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Occupied".into(), vec![f32_ty], vec![4], vec![4]),
                EnumVariant::unit("Empty".into()),
            ],
            0,
            8,
            4,
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(
            slot_map.field_slots,
            vec![Some(1)],
            "an f32 payload owns its LLVM slot; storage equals semantic type"
        );

        // Immutable pointer types model the shared borrow.
        let enum_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, false).into();
        let f32_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, f32_ty, false).into();
        let (module, block) = build_kernel(&mut ctx, vec![enum_ptr_ty], vec![f32_ty]);
        let base = block.deref(&ctx).get_argument(0);
        let addr = MirFieldAddrOp::build(&mut ctx, base, f32_ptr_ty, 0).expect("field_addr build");
        addr.insert_at_back(block, &ctx);
        let payload_ptr = addr.deref(&ctx).get_result(0);

        let load = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![f32_ty],
            vec![payload_ptr],
            vec![],
            0,
        );
        load.insert_at_back(block, &ctx);
        let loaded = load.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![loaded]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            0,
            "a canonical payload read must not spill or copy the enum"
        );
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), 1, "one field_addr lowers to one GEP");
        let gep = &geps[0];
        assert_eq!(
            gep.get_operation().deref(&ctx).get_operand(0),
            base,
            "the GEP must address the ORIGINAL enum storage"
        );
        assert!(
            matches!(
                gep.indices(&ctx).as_slice(),
                [GepIndex::Constant(0), GepIndex::Constant(1)]
            ),
            "an own-slot payload is addressed through its struct slot"
        );
        let loads = find_all::<llvm::LoadOp>(&ctx, &body);
        assert_eq!(
            loads.len(),
            1,
            "the read is a single load through the payload address"
        );
        let gep_result = gep.get_operation().deref(&ctx).get_result(0);
        assert_eq!(
            loads[0].get_operation().deref(&ctx).get_operand(0),
            gep_result,
            "the load must go through the GEP result, not a copy"
        );
        assert_eq!(
            loads[0]
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx),
            f32_ty,
            "the load reads the payload at its semantic type"
        );
    }
}
