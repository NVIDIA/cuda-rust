/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::common::anyhow_to_pliron;
use crate::convert::types::{
    StructLayoutInfo, build_struct_slot_map, convert_type,
    llvm_packed_struct_contains_pointer_in_address_space, make_slice_struct,
    packed_shared_internal_abi_info,
};
use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert `mir.construct_struct` to a chain of `llvm.insertvalue` operations.
///
/// Builds a struct by:
/// 1. Creating an `undef` value of the lowered struct type
/// 2. Inserting each operand at the LLVM slot its field landed in
///
/// Operand order matches field order in the struct type (declaration order).
/// The LLVM struct type and the slot of each field both come from
/// [`build_struct_slot_map`], so the insert indices skip `[N x i8]` padding
/// slots exactly the way the type converter laid them out. ZST fields
/// (e.g. PhantomData) have no slot and are skipped.
pub(crate) fn convert_construct_struct(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let layout = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirStructType>() {
            Some(s) => StructLayoutInfo::of_struct(s),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructStructOp result type must be MirStructType"
                );
            }
        }
    };

    if operands.len() != layout.field_types.len() {
        return pliron::input_err_noloc!(
            "construct_struct has {} operands for a struct with {} fields",
            operands.len(),
            layout.field_types.len()
        );
    }

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    // A divergent rustc layout is constructible by value only when the slot
    // map proved that a sequential LLVM packed struct reproduces every byte.
    // Overlapping/union-like legacy struct models remain unrepresentable.
    if !map.by_value_layout_faithful {
        return pliron::input_err_noloc!(
            "constructing a struct whose rustc layout cannot be represented by an LLVM \
             struct value is not supported; keep the value behind a pointer and access \
             fields through their byte-accurate address path"
        );
    }

    // A packed struct containing AS3 remains target-dependent as a physical
    // memory image. The internal-call ABI exception is safe to construct in SSA
    // because its return boundary recursively rebuilds every supported AS3 leaf
    // into a target-stable generic-pointer carrier before the aggregate crosses
    // the function ABI.
    if llvm_packed_struct_contains_pointer_in_address_space(
        ctx,
        map.llvm_struct_ty,
        llvm_types::address_space::SHARED,
    ) && packed_shared_internal_abi_info(ctx, result_ty)
        .map_err(anyhow_to_pliron)?
        .is_none()
    {
        return pliron::input_err_noloc!(
            "constructing a packed aggregate containing a shared-memory pointer by value is \
             target-mode dependent and is not yet supported"
        );
    }

    let undef_op = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    // Walk in memory order so the insertvalue chain ascends slot indices.
    for &decl_idx in &layout.mem_to_decl {
        let Some(slot) = map.decl_to_llvm[decl_idx] else {
            continue; // ZST field: no slot in the LLVM struct.
        };

        let insert_op =
            llvm::InsertValueOp::new(ctx, current_struct, operands[decl_idx], vec![slot]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_struct = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.construct_tuple` to a chain of `llvm.insertvalue` operations.
///
/// Tuples are represented as LLVM structs. Same construction pattern as
/// structs, and like structs the element slots come from
/// [`build_struct_slot_map`] (identity order, no padding; ZST elements are
/// stripped and skipped).
pub(crate) fn convert_construct_tuple(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let layout = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirTupleType>() {
            Some(t) => StructLayoutInfo::of_tuple(t),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructTupleOp result type must be MirTupleType"
                );
            }
        }
    };

    if operands.len() != layout.field_types.len() {
        return pliron::input_err_noloc!(
            "construct_tuple has {} operands for a tuple with {} elements",
            operands.len(),
            layout.field_types.len()
        );
    }

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let undef_op = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_tuple = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    for (mir_idx, operand) in operands.iter().enumerate() {
        let Some(slot) = map.decl_to_llvm[mir_idx] else {
            continue; // ZST element: no slot in the LLVM struct.
        };

        let insert_op = llvm::InsertValueOp::new(ctx, current_tuple, *operand, vec![slot]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_tuple = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.construct_slice` to `llvm.undef` + two `llvm.insertvalue`s.
///
/// `MirSliceType` lowers to the `{ ptr, i64 }` fat-pointer struct, where
/// field 0 is the data pointer and field 1 is the element count by
/// construction (the same layout the entry prologue's `reconstruct_slice`
/// and the Unsize cast path build). `MirSliceType::kind` is intentionally
/// semantic-only: all reference/raw-pointer kinds share this physical layout
/// and no LLVM alias metadata is inferred here.
pub(crate) fn convert_construct_slice(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, data_val, len_val) = {
        let mir_op = op.deref(ctx);
        (
            mir_op.get_result(0).get_type(ctx),
            mir_op.get_operand(0),
            mir_op.get_operand(1),
        )
    };

    if !result_ty.deref(ctx).is::<MirSliceType>() {
        return pliron::input_err_noloc!("MirConstructSliceOp result type must be MirSliceType");
    }

    let slice_struct_ty = make_slice_struct(ctx);

    let undef_op = llvm::UndefOp::new(ctx, slice_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let undef_val = undef_op.get_operation().deref(ctx).get_result(0);

    let insert_ptr = llvm::InsertValueOp::new(ctx, undef_val, data_val, vec![0]);
    rewriter.insert_operation(ctx, insert_ptr.get_operation());
    let with_ptr = insert_ptr.get_operation().deref(ctx).get_result(0);

    let insert_len = llvm::InsertValueOp::new(ctx, with_ptr, len_val, vec![1]);
    rewriter.insert_operation(ctx, insert_len.get_operation());

    rewriter.replace_operation(ctx, op, insert_len.get_operation());

    Ok(())
}

/// Convert `mir.construct_disjoint_slice` to a chain of `llvm.insertvalue`
/// operations.
///
/// The same shape as [`convert_construct_slice`], one insert longer per
/// runtime layout word the index space carries. The struct type comes from
/// [`make_disjoint_slice_struct`], the same constructor the type converter
/// uses, so the field order here cannot drift from the lowered layout that
/// [`resolve_aggregate_slots`](super::fields::resolve_aggregate_slots) indexes identically.
pub(crate) fn convert_construct_disjoint_slice(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (mir_op.get_result(0).get_type(ctx), operands)
    };

    let space_tys = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirDisjointSliceType>() {
            Some(slice_ty) => slice_ty.space_types().to_vec(),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructDisjointSliceOp result type must be MirDisjointSliceType"
                );
            }
        }
    };

    // The verifier already pins the operand count to the result type. Check
    // it here too: this pass runs on operations the verifier accepted, and a
    // short operand list would otherwise index past the end below.
    let expected_operands = 2 + space_tys.len();
    if operands.len() != expected_operands {
        return pliron::input_err_noloc!(
            "MirConstructDisjointSliceOp expects {expected_operands} operands, got {}",
            operands.len()
        );
    }

    let slice_struct_ty = crate::convert::types::make_disjoint_slice_struct(ctx, &space_tys)
        .map_err(anyhow_to_pliron)?;

    let undef_op = llvm::UndefOp::new(ctx, slice_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current = undef_op.get_operation().deref(ctx).get_result(0);
    let mut last_insert = undef_op.get_operation();

    for (slot, operand) in operands.into_iter().enumerate() {
        let insert = llvm::InsertValueOp::new(ctx, current, operand, vec![slot as u32]);
        rewriter.insert_operation(ctx, insert.get_operation());
        current = insert.get_operation().deref(ctx).get_result(0);
        last_insert = insert.get_operation();
    }

    rewriter.replace_operation(ctx, op, last_insert);

    Ok(())
}

/// Convert `mir.construct_array` to a chain of `llvm.insertvalue` operations.
///
/// Arrays are represented as LLVM arrays. Same construction pattern as structs:
/// 1. Create `undef` of the array type
/// 2. Insert each element at its index
pub(crate) fn convert_construct_array(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let (element_ty, array_size) = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirArrayType>() {
            Some(a) => (a.element_type(), a.size()),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructArrayOp result type must be MirArrayType"
                );
            }
        }
    };

    let llvm_element_ty = convert_type(ctx, element_ty).map_err(anyhow_to_pliron)?;
    let llvm_array_ty = llvm_export::types::ArrayType::get(ctx, llvm_element_ty, array_size);

    let undef_op = llvm::UndefOp::new(ctx, llvm_array_ty.into());
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_array = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    for (i, operand) in operands.iter().enumerate() {
        let insert_op = llvm::InsertValueOp::new(ctx, current_array, *operand, vec![i as u32]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_array = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;

    use dialect_mir::ops as mir;
    use dialect_mir::types::{
        MirArrayType, MirPointerKind, MirPtrType, MirSliceType, MirStructType,
    };
    use llvm_export::types as llvm_types;

    use pliron::common_traits::Verify;

    use crate::convert::types::llvm_type_size_align;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    use super::super::test_support::*;

    /// `mir.construct_slice` lowers to the canonical fat-pointer value:
    /// `undef { ptr, i64 }`, then insert data pointer at slot 0 and length at slot 1.
    #[test]
    fn construct_slice_lowers_to_ptr_len_insert_values() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let usize_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, i8_ty, false).into();
        let slice_ty: TypeHandle = MirSliceType::get(&mut ctx, i8_ty).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, usize_ty], vec![]);
        let data_ptr = block.deref(&ctx).get_argument(0);
        let len = block.deref(&ctx).get_argument(1);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructSliceOp::get_concrete_op_info(),
            vec![slice_ty],
            vec![data_ptr, len],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1]],
            "slice construction must insert data pointer at slot 0 and length at slot 1"
        );
        let first_insert = inserts[0].get_operation();
        let second_insert = inserts[1].get_operation();
        assert_eq!(
            first_insert.deref(&ctx).get_operand(1),
            data_ptr,
            "slice slot 0 must receive the original data pointer"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(0),
            first_insert.deref(&ctx).get_result(0),
            "the length insertion must consume the aggregate produced by the pointer insertion"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(1),
            len,
            "slice slot 1 must receive the original length"
        );
        assert!(
            inserts.iter().all(|insert| insert.verify(&ctx).is_ok()),
            "both slice insertions must satisfy LLVM dialect verification"
        );
        assert_eq!(
            count_ops::<llvm::UndefOp>(&ctx, &body),
            1,
            "slice construction should start from one undef aggregate"
        );
    }

    /// `mir.construct_disjoint_slice` lowers to the same insert chain as the
    /// fat pointer for an index space with no runtime layout.
    #[test]
    fn construct_disjoint_slice_lowers_to_ptr_len_insert_values() {
        let mut ctx = make_ctx();

        let f32_ty: TypeHandle = pliron::builtin::types::FP32Type::get(&ctx).into();
        let usize_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let ptr_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, f32_ty, true, MirPointerKind::RawMut)
                .into();
        let slice_ty: TypeHandle = MirDisjointSliceType::get(&mut ctx, f32_ty).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, usize_ty], vec![]);
        let data_ptr = block.deref(&ctx).get_argument(0);
        let len = block.deref(&ctx).get_argument(1);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructDisjointSliceOp::get_concrete_op_info(),
            vec![slice_ty],
            vec![data_ptr, len],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1]],
            "a space-free disjoint slice must insert pointer at slot 0 and length at slot 1"
        );
        assert_eq!(
            inserts[0].get_operation().deref(&ctx).get_operand(1),
            data_ptr,
            "slot 0 must receive the original data pointer"
        );
        assert_eq!(
            inserts[1].get_operation().deref(&ctx).get_operand(1),
            len,
            "slot 1 must receive the original length"
        );
        assert_eq!(
            count_ops::<llvm::UndefOp>(&ctx, &body),
            1,
            "construction should start from one undef aggregate"
        );
    }

    /// The runtime row width is the third operand and must land in slot 2.
    /// Writing it into the length slot, or dropping it, gives a slice whose
    /// row width reads back as something else at every access site.
    #[test]
    fn construct_disjoint_slice_places_the_row_width_after_ptr_and_len() {
        let mut ctx = make_ctx();

        let f32_ty: TypeHandle = pliron::builtin::types::FP32Type::get(&ctx).into();
        let usize_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let width_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let ptr_ty: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, f32_ty, true, MirPointerKind::RawMut)
                .into();
        let slice_ty: TypeHandle =
            MirDisjointSliceType::get_with_space(&mut ctx, f32_ty, vec![width_ty]).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, usize_ty, width_ty], vec![]);
        let data_ptr = block.deref(&ctx).get_argument(0);
        let len = block.deref(&ctx).get_argument(1);
        let width = block.deref(&ctx).get_argument(2);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructDisjointSliceOp::get_concrete_op_info(),
            vec![slice_ty],
            vec![data_ptr, len, width],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1], vec![2]],
            "the row width must occupy slot 2, after the pointer and the length"
        );
        assert_eq!(
            inserts[2].get_operation().deref(&ctx).get_operand(1),
            width,
            "slot 2 must receive the row width operand, not the length"
        );
        assert_eq!(
            inserts[1].get_operation().deref(&ctx).get_operand(1),
            len,
            "slot 1 must still receive the length"
        );
    }

    /// Explicit rustc layout must be respected: field `b` is at byte offset 8,
    /// so the lowered LLVM struct has a padding slot between `a` and `b`.
    /// The ZST marker field is stripped and receives no insert_value.
    #[test]
    fn construct_struct_uses_layout_slots_and_skips_zst() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (struct_ty, zst_ty) = padded_struct_with_zst_ty(&mut ctx);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i8_ty, i64_ty], vec![]);
        let a = block.deref(&ctx).get_argument(0);
        let b = block.deref(&ctx).get_argument(1);
        let marker = append_empty_struct_value(&mut ctx, block, zst_ty);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructStructOp::get_concrete_op_info(),
            vec![struct_ty],
            vec![a, marker, b],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![2]],
            "non-ZST fields must be inserted at their layout slots, skipping padding and ZSTs"
        );
        let first_insert = inserts[0].get_operation();
        let second_insert = inserts[1].get_operation();
        assert_eq!(
            first_insert.deref(&ctx).get_operand(1),
            a,
            "struct slot 0 must receive field `a`"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(0),
            first_insert.deref(&ctx).get_result(0),
            "field `b` must be inserted into the aggregate containing field `a`"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(1),
            b,
            "struct slot 2 must receive field `b`"
        );
        assert!(
            inserts.iter().all(|insert| insert.verify(&ctx).is_ok()),
            "both struct insertions must satisfy LLVM dialect verification"
        );
    }

    /// A by-value packed struct is represented by an LLVM packed struct, so
    /// construction can use the ordinary insertvalue chain without changing
    /// rustc's field offsets or total byte size.
    #[test]
    fn packed_struct_construction_by_value_uses_packed_layout() {
        use dialect_mir::ops::MirConstructStructOp;

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

        let (module, block) = build_kernel(&mut ctx, vec![u8_ty, u32_ty], vec![]);
        let tag = block.deref(&ctx).get_argument(0);
        let value = block.deref(&ctx).get_argument(1);

        let construct = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![tag, value],
            vec![],
            0,
        );
        construct.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("constructing a pointer-free packed struct by value must lower");

        let body = kernel_blocks(&ctx, module);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1]],
            "packed construction must keep declaration fields in their packed LLVM slots"
        );
        let result_ty = inserts
            .last()
            .expect("packed construction must emit insertvalue")
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        let result_ty_ref = result_ty.deref(&ctx);
        let struct_ty = result_ty_ref
            .downcast_ref::<llvm_types::StructType>()
            .expect("packed construction result must be an LLVM struct");
        assert_eq!(struct_ty.layout(), llvm_types::StructLayout::Packed);
        assert_eq!(llvm_type_size_align(&ctx, result_ty), Some((5, 1)));
    }

    #[test]
    fn packed_struct_construction_with_one_direct_shared_pointer_stays_semantic_in_ssa() {
        use dialect_mir::ops::MirConstructStructOp;

        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedShared".into(),
            vec!["tag".into(), "ptr".into()],
            vec![u8_ty, shared_ty],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![u8_ty, shared_ty], vec![]);
        let tag = block.deref(&ctx).get_argument(0);
        let ptr = block.deref(&ctx).get_argument(1);
        let construct = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![tag, ptr],
            vec![],
            0,
        );
        construct.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("one direct AS3 pointer in a packed SSA aggregate must lower");

        let body = kernel_blocks(&ctx, module);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1]],
            "packed AS3 construction must preserve the semantic packed field slots"
        );

        let result_ty = inserts
            .last()
            .expect("packed AS3 construction must emit insertvalue")
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        let result_ty_ref = result_ty.deref(&ctx);
        let struct_ty = result_ty_ref
            .downcast_ref::<llvm_types::StructType>()
            .expect("packed AS3 construction result must be an LLVM struct");
        assert_eq!(struct_ty.layout(), llvm_types::StructLayout::Packed);
        assert_eq!(llvm_type_size_align(&ctx, result_ty), Some((9, 1)));

        let pointer_ty = struct_ty.field_type(1);
        let pointer_ty_ref = pointer_ty.deref(&ctx);
        let pointer = pointer_ty_ref
            .downcast_ref::<llvm_types::PointerType>()
            .expect("second packed field must remain a pointer");
        assert_eq!(
            pointer.address_space(),
            llvm_types::address_space::SHARED,
            "construction must keep the semantic pointer in AS3"
        );
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            0,
            "construction itself must not genericize the shared pointer"
        );
    }

    #[test]
    fn packed_struct_construction_with_multiple_direct_shared_pointers_stays_semantic_in_ssa() {
        use dialect_mir::ops::MirConstructStructOp;

        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedPair".into(),
            vec!["tag".into(), "left".into(), "right".into()],
            vec![u8_ty, shared_ty, shared_ty],
            vec![0, 1, 2],
            vec![0, 1, 9],
            17,
            1,
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![u8_ty, shared_ty, shared_ty], vec![]);
        let tag = block.deref(&ctx).get_argument(0);
        let left = block.deref(&ctx).get_argument(1);
        let right = block.deref(&ctx).get_argument(2);
        let construct = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![tag, left, right],
            vec![],
            0,
        );
        construct.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("multiple direct AS3 leaves in a packed SSA aggregate must lower");

        let body = kernel_blocks(&ctx, module);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1], vec![2]],
            "all direct shared leaves must remain in their semantic packed slots"
        );

        let result_ty = inserts
            .last()
            .expect("packed AS3 construction must emit insertvalue")
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        let result_ty_ref = result_ty.deref(&ctx);
        let struct_ty = result_ty_ref
            .downcast_ref::<llvm_types::StructType>()
            .expect("packed AS3 construction result must be an LLVM struct");
        assert_eq!(struct_ty.layout(), llvm_types::StructLayout::Packed);
        assert_eq!(llvm_type_size_align(&ctx, result_ty), Some((17, 1)));
        for slot in [1usize, 2] {
            let pointer_ty = struct_ty.field_type(slot);
            let pointer_ty_ref = pointer_ty.deref(&ctx);
            let pointer = pointer_ty_ref
                .downcast_ref::<llvm_types::PointerType>()
                .expect("direct shared leaves must remain pointer-typed");
            assert_eq!(pointer.address_space(), llvm_types::address_space::SHARED);
        }
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            0,
            "semantic construction must not genericize any direct shared leaf"
        );
    }

    #[test]
    fn packed_struct_construction_with_nested_shared_pointers_stays_semantic_in_ssa() {
        use dialect_mir::ops::MirConstructStructOp;

        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let inner_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "SharedPair".into(),
            vec!["left".into(), "right".into()],
            vec![shared_ty, shared_ty],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedNestedShared".into(),
            vec!["tag".into(), "pair".into()],
            vec![u8_ty, inner_ty],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![u8_ty, shared_ty, shared_ty], vec![]);
        let tag = block.deref(&ctx).get_argument(0);
        let left = block.deref(&ctx).get_argument(1);
        let right = block.deref(&ctx).get_argument(2);

        let inner = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![inner_ty],
            vec![left, right],
            vec![],
            0,
        );
        inner.insert_at_back(block, &ctx);
        let inner_value = inner.deref(&ctx).get_result(0);

        let outer = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![tag, inner_value],
            vec![],
            0,
        );
        outer.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("nested AS3 leaves in a packed SSA aggregate must lower");

        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            0,
            "semantic construction must keep nested shared leaves in AS3"
        );

        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        let packed_result = inserts
            .iter()
            .filter_map(|insert| {
                let result_ty = insert
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx);
                let is_packed = result_ty
                    .deref(&ctx)
                    .downcast_ref::<llvm_types::StructType>()
                    .is_some_and(|ty| ty.layout() == llvm_types::StructLayout::Packed);
                is_packed.then_some(result_ty)
            })
            .last()
            .expect("outer packed construction must produce a packed LLVM struct");

        assert_eq!(llvm_type_size_align(&ctx, packed_result), Some((17, 1)));
        assert!(
            llvm_packed_struct_contains_pointer_in_address_space(
                &ctx,
                packed_result,
                llvm_types::address_space::SHARED,
            ),
            "the semantic packed value must retain its recursively nested AS3 leaves"
        );
    }

    #[test]
    fn packed_struct_construction_with_bounded_shared_pointer_array_stays_semantic_in_ssa() {
        use dialect_mir::ops::{MirConstructArrayOp, MirConstructStructOp};

        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let array_ty: TypeHandle = MirArrayType::get(&mut ctx, shared_ty, 2).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedSharedArray".into(),
            vec!["tag".into(), "ptrs".into()],
            vec![u8_ty, array_ty],
            vec![0, 1],
            vec![0, 1],
            17,
            1,
        )
        .into();

        let (module, block) = build_kernel(&mut ctx, vec![u8_ty, shared_ty, shared_ty], vec![]);
        let tag = block.deref(&ctx).get_argument(0);
        let left = block.deref(&ctx).get_argument(1);
        let right = block.deref(&ctx).get_argument(2);

        let array = Operation::new(
            &mut ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_ty],
            vec![left, right],
            vec![],
            0,
        );
        array.insert_at_back(block, &ctx);
        let array_value = array.deref(&ctx).get_result(0);

        let outer = Operation::new(
            &mut ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![tag, array_value],
            vec![],
            0,
        );
        outer.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module)
            .expect("a bounded AS3 array in a packed SSA aggregate must lower");

        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            0,
            "semantic construction must keep array elements in AS3"
        );

        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);
        let packed_result = inserts
            .iter()
            .filter_map(|insert| {
                let result_ty = insert
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx);
                let is_packed = result_ty
                    .deref(&ctx)
                    .downcast_ref::<llvm_types::StructType>()
                    .is_some_and(|ty| ty.layout() == llvm_types::StructLayout::Packed);
                is_packed.then_some(result_ty)
            })
            .last()
            .expect("outer packed array construction must produce a packed LLVM struct");
        assert_eq!(llvm_type_size_align(&ctx, packed_result), Some((17, 1)));
        assert!(
            llvm_packed_struct_contains_pointer_in_address_space(
                &ctx,
                packed_result,
                llvm_types::address_space::SHARED,
            ),
            "the semantic packed value must retain AS3 array elements"
        );
    }
}
