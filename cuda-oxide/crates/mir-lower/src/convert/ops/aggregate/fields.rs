/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::{anyhow_to_pliron, spill_enum_value};
use crate::convert::types::{
    StructLayoutInfo, StructSlotMap, build_struct_slot_map, build_union_storage_type, convert_type,
    is_zero_sized_type, make_slice_struct,
};
use dialect_mir::ops::{MirExtractFieldOp, MirInsertFieldOp};
use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// How the MIR-level field indices of an aggregate operand map onto the
/// lowered LLVM aggregate.
pub(super) enum AggregateSlots {
    /// Lowered from a `MirStructType`/`MirTupleType`: use the slot map the
    /// type converter built (accounts for reordering, `[N x i8]` padding
    /// slots and stripped ZST fields).
    Mapped(StructSlotMap),
    /// The MIR index is already the final LLVM index. Sound only for
    /// aggregates whose lowered layout is index-preserving by construction:
    /// arrays and slice fat pointers (`{ ptr, i64 }`).
    Identity,
}

/// Resolve how field indices of `aggregate` map onto its lowered type.
///
/// Recover-or-error (issue #128): when the operand has no recorded
/// `MirStructType`/`MirTupleType` conversion history, identity indexing is
/// only sound for aggregates the converter lowers without reordering,
/// padding, or ZST stripping: arrays and slice fat pointers. Anything
/// else is a lowering bug; guessing identity there silently reads or
/// writes the wrong field, so we error out loudly instead.
pub(super) fn resolve_aggregate_slots(
    ctx: &mut Context,
    operands_info: &OperandsInfo,
    aggregate: Value,
) -> Result<AggregateSlots> {
    let layout = operands_info
        .lookup_most_recent_of_type::<MirStructType>(ctx, aggregate)
        .map(|struct_ref| StructLayoutInfo::of_struct(&struct_ref))
        .or_else(|| {
            operands_info
                .lookup_most_recent_of_type::<MirTupleType>(ctx, aggregate)
                .map(|tuple_ref| StructLayoutInfo::of_tuple(&tuple_ref))
        });

    if let Some(layout) = layout {
        let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;
        return Ok(AggregateSlots::Mapped(map));
    }

    // Arrays keep their element indices: `[N x T]` has no reorder, no
    // padding, no ZST stripping.
    let is_array_history = operands_info
        .lookup_most_recent_of_type::<MirArrayType>(ctx, aggregate)
        .is_some();
    // Slices lower to the `{ ptr, i64 }` fat pointer, where index 0 = ptr
    // and index 1 = len by construction.
    let is_slice_history = operands_info
        .lookup_most_recent_of_type::<MirSliceType>(ctx, aggregate)
        .is_some()
        || operands_info
            .lookup_most_recent_of_type::<MirDisjointSliceType>(ctx, aggregate)
            .is_some();
    if is_array_history || is_slice_history {
        return Ok(AggregateSlots::Identity);
    }

    // No conversion history at all (e.g. a slice reconstructed in the entry
    // prologue, which is born as an LLVM struct). Identity is still fine if
    // the current type is a slice fat pointer or an LLVM array. A disjoint
    // slice with a runtime row width is the same case one word longer: every
    // runtime index space today carries a single `u32` row width, and the struct the type
    // converter builds for it (`make_disjoint_slice_struct`) is
    // index-preserving by construction, exactly like the two-field fat
    // pointer. Both shape checks share the two-field caveat that an unnamed
    // user struct with the identical lowered layout would be accepted too;
    // that has been the accepted trade-off for `{ ptr, i64 }` since issue
    // #128, and a struct needing a slot map still errors below whenever its
    // lowered shape differs (padding slots, reordering, other field types).
    let aggregate_ty = aggregate.get_type(ctx);
    let slice_struct_ty = make_slice_struct(ctx);
    let row_width_slice_struct_ty = {
        // The one runtime layout current `SpaceLayout` impls produce: a
        // single u32 row width. Built through the real constructor so this
        // check cannot drift from the lowered shape. A future space with a
        // different `Data` shape is not listed here and thus fails closed
        // into the refuse-to-guess error until this site learns it.
        let width_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        crate::convert::types::make_disjoint_slice_struct(ctx, &[width_ty])
            .map_err(anyhow_to_pliron)?
    };
    let is_llvm_array = aggregate_ty
        .deref(ctx)
        .is::<llvm_export::types::ArrayType>();
    if aggregate_ty == slice_struct_ty || aggregate_ty == row_width_slice_struct_ty || is_llvm_array
    {
        return Ok(AggregateSlots::Identity);
    }

    let ty_disp = aggregate_ty.deref(ctx).disp(ctx).to_string();
    pliron::input_err_noloc!(
        "Cannot map field indices for aggregate of type {ty_disp}: no struct/tuple \
         conversion history was recorded for this operand, and identity indexing is \
         only sound for arrays and slice fat pointers (with or without the runtime \
         row-width word). Refusing to guess a field mapping (issue #128)."
    )
}

/// Convert `mir.extract_field` to `llvm.extractvalue`.
///
/// Handles scalar-lowered newtype case: if the operand is a scalar (e.g., `ThreadIndex`),
/// no extraction is needed.
///
/// The declaration-order field index is mapped to the LLVM slot via
/// [`resolve_aggregate_slots`], which shares the type converter's view of
/// the struct (reorder, `[N x i8]` padding slots, stripped ZSTs). If
/// extracting a ZST field, we return undef of its (empty) type.
pub(crate) fn convert_extract_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let aggregate = op.deref(ctx).get_operand(0);

    let extract_op = MirExtractFieldOp::new(op);
    let decl_index = match extract_op.get_attr_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("Missing index attribute on extract_field"),
    };

    if operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, aggregate)
        .is_some()
    {
        return convert_extract_union_field(
            ctx,
            rewriter,
            op,
            aggregate,
            decl_index,
            operands_info,
        );
    }

    let is_scalar = aggregate
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some();

    if is_scalar {
        rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
        return Ok(());
    }

    let llvm_index = match resolve_aggregate_slots(ctx, operands_info, aggregate)? {
        AggregateSlots::Mapped(map) => match map.decl_to_llvm.get(decl_index) {
            Some(Some(slot)) => *slot,
            Some(None) => {
                // ZST field: stripped from the LLVM struct, so there is
                // nothing to extract. Materialize undef of its empty type.
                let zst_ty = map.field_llvm_types[decl_index];
                let undef_op = llvm::UndefOp::new(ctx, zst_ty);
                rewriter.insert_operation(ctx, undef_op.get_operation());
                rewriter.replace_operation(ctx, op, undef_op.get_operation());
                return Ok(());
            }
            None => {
                return pliron::input_err_noloc!(
                    "extract_field index {} out of bounds for aggregate with {} fields",
                    decl_index,
                    map.decl_to_llvm.len()
                );
            }
        },
        AggregateSlots::Identity => decl_index as u32,
    };

    let llvm_extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![llvm_index])?;
    rewriter.insert_operation(ctx, llvm_extract.get_operation());
    rewriter.replace_operation(ctx, op, llvm_extract.get_operation());

    Ok(())
}

/// Convert `mir.insert_field` to `llvm.insertvalue`.
///
/// Operands: `[aggregate, new_value]`
/// Returns a new aggregate with the field at `insert_index` replaced.
///
/// The declaration-order field index is mapped to the LLVM slot via
/// [`resolve_aggregate_slots`] (arrays keep their element index). If
/// inserting a ZST field, we return the original aggregate unchanged.
pub(crate) fn convert_insert_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let aggregate = op.deref(ctx).get_operand(0);
    let new_value = op.deref(ctx).get_operand(1);

    let insert_op = MirInsertFieldOp::new(op);
    let decl_index = match insert_op.get_attr_insert_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("Missing insert_index attribute on insert_field"),
    };

    if operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, aggregate)
        .is_some()
    {
        return convert_insert_union_field(
            ctx,
            rewriter,
            op,
            aggregate,
            new_value,
            decl_index,
            operands_info,
        );
    }

    let llvm_index = match resolve_aggregate_slots(ctx, operands_info, aggregate)? {
        AggregateSlots::Mapped(map) => match map.decl_to_llvm.get(decl_index) {
            Some(Some(slot)) => *slot,
            Some(None) => {
                // ZST field: stripped from the LLVM struct, so inserting
                // into it is a no-op. Forward the aggregate unchanged.
                rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
                return Ok(());
            }
            None => {
                return pliron::input_err_noloc!(
                    "insert_field index {} out of bounds for aggregate with {} fields",
                    decl_index,
                    map.decl_to_llvm.len()
                );
            }
        },
        AggregateSlots::Identity => decl_index as u32,
    };

    let llvm_insert = llvm::InsertValueOp::new(ctx, aggregate, new_value, vec![llvm_index]);
    rewriter.insert_operation(ctx, llvm_insert.get_operation());
    rewriter.replace_operation(ctx, op, llvm_insert.get_operation());

    Ok(())
}

fn union_type_of_operand(
    ctx: &Context,
    operands_info: &OperandsInfo,
    value: Value,
) -> Result<MirUnionType> {
    operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, value)
        .map(|union_ty| union_ty.clone())
        .ok_or_else(|| {
            pliron::create_error!(
                pliron::location::Location::Unknown,
                pliron::result::ErrorKind::VerificationFailed,
                pliron::result::StringError(
                    "Expected MirUnionType conversion history for union value".to_string()
                )
            )
        })
}

/// Read one typed view of a union's shared bytes.
fn convert_extract_union_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    union_value: Value,
    field_index: usize,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let union_ty = union_type_of_operand(ctx, operands_info, union_value)?;
    let Some(field_mir_ty) = union_ty.get_field_type(field_index) else {
        return pliron::input_err_noloc!(
            "union field index {} is out of bounds for `{}`",
            field_index,
            union_ty.name()
        );
    };
    let field_llvm_ty = convert_type(ctx, field_mir_ty).map_err(anyhow_to_pliron)?;
    if is_zero_sized_type(ctx, field_llvm_ty) {
        let undef = llvm::UndefOp::new(ctx, field_llvm_ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        rewriter.replace_operation(ctx, op, undef.get_operation());
        return Ok(());
    }

    let storage_ty = build_union_storage_type(ctx, &union_ty).map_err(anyhow_to_pliron)?;
    let ptr = spill_enum_value(ctx, rewriter, union_value, storage_ty, union_ty.abi_align());
    let load = llvm::LoadOp::new(ctx, ptr, field_llvm_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, load.get_operation());
    rewriter.replace_operation(ctx, op, load.get_operation());
    Ok(())
}

/// Write one typed view at byte zero while preserving the rest of the union.
fn convert_insert_union_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    union_value: Value,
    new_value: Value,
    field_index: usize,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let union_ty = union_type_of_operand(ctx, operands_info, union_value)?;
    let Some(field_mir_ty) = union_ty.get_field_type(field_index) else {
        return pliron::input_err_noloc!(
            "union field index {} is out of bounds for `{}`",
            field_index,
            union_ty.name()
        );
    };
    let field_llvm_ty = convert_type(ctx, field_mir_ty).map_err(anyhow_to_pliron)?;
    if is_zero_sized_type(ctx, field_llvm_ty) {
        rewriter.replace_operation_with_values(ctx, op, vec![union_value]);
        return Ok(());
    }

    let storage_ty = build_union_storage_type(ctx, &union_ty).map_err(anyhow_to_pliron)?;
    let ptr = spill_enum_value(ctx, rewriter, union_value, storage_ty, union_ty.abi_align());
    let store = llvm::StoreOp::new(ctx, new_value, ptr);
    llvm_export::ops::set_op_alignment(ctx, store.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, store.get_operation());

    let load = llvm::LoadOp::new(ctx, ptr, storage_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, load.get_operation());
    rewriter.replace_operation(ctx, op, load.get_operation());
    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::FieldIndexAttr;
    use dialect_mir::ops as mir;

    use llvm_export::types as llvm_types;

    use super::super::test_support::*;

    /// Extracting a ZST field must not emit `extract_value`: the field has no
    /// storage in the lowered LLVM struct, so lowering materializes an undef
    /// zero-sized value instead.
    #[test]
    fn extract_zst_field_lowers_to_undef_without_extract_value() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (struct_ty, zst_ty) = padded_struct_with_zst_ty(&mut ctx);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i8_ty, i64_ty], vec![]);
        let a = block.deref(&ctx).get_argument(0);
        let b = block.deref(&ctx).get_argument(1);
        let marker = append_empty_struct_value(&mut ctx, block, zst_ty);

        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructStructOp::get_concrete_op_info(),
            vec![struct_ty],
            vec![a, marker, b],
            vec![],
            0,
        );
        construct.insert_at_back(block, &ctx);
        let aggregate = construct.deref(&ctx).get_result(0);

        let extract = Operation::new(
            &mut ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![zst_ty],
            vec![aggregate],
            vec![],
            0,
        );
        MirExtractFieldOp::new(extract).set_attr_index(&ctx, FieldIndexAttr(1));
        extract.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);

        assert_eq!(
            count_ops::<llvm::ExtractValueOp>(&ctx, &body),
            0,
            "extracting a stripped ZST field must not emit llvm.extractvalue"
        );

        let zst_un_defs = find_all::<llvm::UndefOp>(&ctx, &body)
            .into_iter()
            .filter(|op| {
                let result_ty = op.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
                is_zero_sized_type(&ctx, result_ty)
            })
            .count();

        assert_eq!(
            zst_un_defs, 2,
            "one undef should build the ZST value and one should materialize the extracted ZST"
        );
    }

    /// The row-width arm of [`resolve_aggregate_slots`]' no-history fallback
    /// is defensive symmetry: no current lowering path produces a runtime-width
    /// slice value with an empty conversion history, so no end-to-end pipeline
    /// test can reach it. This exercises the fallback directly: a value born as
    /// the lowered `{ ptr, i64, i32 }` row-width shape (a block argument, which
    /// carries no history) must resolve to identity indexing exactly like the
    /// two-field `{ ptr, i64 }` fat pointer, and any other unrecognized
    /// no-history shape must keep failing closed into the refuse-to-guess
    /// error (issue #128).
    #[test]
    fn no_history_fallback_resolves_row_width_slice_shape_and_stays_closed_otherwise() {
        let mut ctx = make_ctx();

        // The exact struct the type converter lowers a runtime-width
        // disjoint slice to, built through the same constructor the fallback
        // uses so the test cannot drift from the lowered shape.
        let width_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let row_width_ty = crate::convert::types::make_disjoint_slice_struct(&mut ctx, &[width_ty])
            .expect("building the row-width slice struct must succeed");

        // A three-field control shape that is NOT the row-width slice: i64
        // where the u32 row-width word belongs.
        use llvm_export::types::PointerTypeExt;
        let ptr_ty: TypeHandle = llvm_types::PointerType::get_generic(&mut ctx).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let foreign_ty: TypeHandle = llvm_types::StructType::get_unnamed(
            &ctx,
            (
                vec![ptr_ty, i64_ty, i64_ty],
                llvm_types::StructLayout::Unpacked,
            ),
        )
        .into();

        let (_module, block) = build_kernel(&mut ctx, vec![row_width_ty, foreign_ty], vec![]);
        let row_width_value = block.deref(&ctx).get_argument(0);
        let foreign_value = block.deref(&ctx).get_argument(1);

        // Block arguments have no recorded conversion history at all.
        let no_history = OperandsInfo::default();

        let resolved = resolve_aggregate_slots(&mut ctx, &no_history, row_width_value)
            .expect("the row-width slice shape with no history must resolve");
        assert!(
            matches!(resolved, AggregateSlots::Identity),
            "the row-width {{ ptr, i64, u32 }} shape is index-preserving and must map identically"
        );

        let refused = resolve_aggregate_slots(&mut ctx, &no_history, foreign_value);
        assert!(
            refused.is_err(),
            "a no-history shape that is not a slice fat pointer (with or without \
             the row-width word) must fail closed"
        );
    }
}
