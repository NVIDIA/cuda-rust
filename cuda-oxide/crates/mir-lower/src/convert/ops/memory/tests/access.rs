/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::disallowed_methods)]

use super::*;
use llvm_export::attributes::GepNoWrapFlags;
use llvm_export::op_interfaces::VolatilityOpInterface;

#[test]
fn convert_alloca_lowers_to_llvm_alloca() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    alloca_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(
        count_ops::<llvm::AllocaOp>(&ctx, &body),
        1,
        "expected exactly one llvm.alloca"
    );
    let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
    // Element type should round-trip through convert_type as i32.
    let elem_ty = alloca.result_pointee_type(&ctx);
    assert!(elem_ty.deref(&ctx).is::<IntegerType>());
}

#[test]
fn convert_alloca_preserves_nested_array_element_alignment() {
    let mut ctx = make_ctx();
    let tuple_ty = over_aligned_tuple_ty(&mut ctx);
    let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 2).into();
    let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, outer, true);
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    alloca_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
        Some(32)
    );
}

#[test]
fn convert_alloca_preserves_debug_local_metadata() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    let alloca_op = Operation::new(
        &mut ctx,
        mir::MirAllocaOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![],
        vec![],
        0,
    );
    llvm::set_debug_local_variable(
        &mut ctx,
        alloca_op,
        llvm::DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: Some(1),
            ty: llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    alloca_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
    let info = llvm::debug_local_variable(&ctx, alloca.get_operation())
        .expect("debug local metadata should survive lowering");

    assert_eq!(info.name, "x");
    assert_eq!(info.argument_index, Some(1));
    assert_eq!(
        info.ty,
        llvm::DebugLocalTypeKind::Basic {
            name: "i32".to_string(),
            size_bits: 32,
            encoding: "DW_ATE_signed",
        }
    );
}

#[test]
fn convert_store_lowers_to_llvm_store() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    // Kernel takes (ptr, val) so we can store one into the other.
    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i32_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    store_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(
        count_ops::<llvm::StoreOp>(&ctx, &body),
        1,
        "expected one llvm.store"
    );
    // The original mir.store must be gone.
    assert_eq!(count_ops::<mir::MirStoreOp>(&ctx, &body), 0);

    // convert_store swaps operand order: mir.store is [ptr, value] but
    // llvm.store takes (value, ptr). Verify that mapping survived.
    let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
    let addr_ty = store.get_operand_address(&ctx).get_type(&ctx);
    assert!(addr_ty.deref(&ctx).is::<PointerType>(), "operand 1 is ptr");
}

#[test]
fn convert_store_preserves_volatile() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i32_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let store_op = Operation::new(
        &mut ctx,
        mir::MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    mir::MirStoreOp::new(store_op).set_volatile(&mut ctx, true);
    store_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
    assert!(
        store.is_volatile(&ctx),
        "volatile mir.store must lower to a volatile llvm.store"
    );
}

#[test]
fn convert_load_lowers_to_llvm_load() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty],
        vec![ptr_val],
        vec![],
        0,
    );
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 1);
    assert_eq!(count_ops::<mir::MirLoadOp>(&ctx, &body), 0);
}

#[test]
fn convert_load_preserves_volatile() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let load_op = Operation::new(
        &mut ctx,
        mir::MirLoadOp::get_concrete_op_info(),
        vec![i32_ty],
        vec![ptr_val],
        vec![],
        0,
    );
    mir::MirLoadOp::new(load_op).set_volatile(&mut ctx, true);
    load_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let load = find_first::<llvm::LoadOp>(&ctx, &body).unwrap();
    assert!(
        load.is_volatile(&ctx),
        "volatile mir.load must lower to a volatile llvm.load"
    );
}

#[test]
fn convert_ref_lowers_to_alloca_then_store() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let mir_ptr_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty, false, MirPointerKind::SharedRef);

    // Take a u32 by value, build `&x`.
    let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
    let arg = block.deref(&ctx).get_argument(0);

    let ref_op_ptr = Operation::new(
        &mut ctx,
        mir::MirRefOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![arg],
        vec![],
        0,
    );
    let ref_op = mir::MirRefOp::new(ref_op_ptr);
    ref_op.set_mutable(&mut ctx, false);
    ref_op.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    ref_op_ptr.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    assert_eq!(
        count_ops::<llvm::AllocaOp>(&ctx, &body),
        1,
        "ref must materialize via alloca"
    );
    assert_eq!(
        count_ops::<llvm::StoreOp>(&ctx, &body),
        1,
        "ref must store the value into the alloca"
    );
    assert_eq!(count_ops::<mir::MirRefOp>(&ctx, &body), 0);
}

#[test]
fn convert_ref_preserves_tuple_alignment_on_alloca_and_store() {
    let mut ctx = make_ctx();
    let tuple_ty = over_aligned_tuple_ty(&mut ctx);
    let mir_ptr_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, tuple_ty, false, MirPointerKind::SharedRef);
    let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

    let undef = mir::MirUndefOp::new(&mut ctx, tuple_ty);
    undef.get_operation().insert_at_back(block, &ctx);
    let value = undef.get_operation().deref(&ctx).get_result(0);
    let ref_op = Operation::new(
        &mut ctx,
        mir::MirRefOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![value],
        vec![],
        0,
    );
    let mir_ref = mir::MirRefOp::new(ref_op);
    mir_ref.set_mutable(&mut ctx, false);
    mir_ref.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    ref_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
    let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
        Some(32)
    );
    assert_eq!(
        llvm_export::ops::op_alignment(&ctx, store.get_operation()),
        Some(32)
    );
}

#[test]
fn convert_ref_preserves_over_aligned_union_array_layout_and_alignment() {
    for abi_align in [32, 64] {
        let mut ctx = make_ctx();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let union_ty: TypeHandle = MirUnionType::get(
            &mut ctx,
            format!("Align{abi_align}Union"),
            vec!["word".into()],
            vec![u32_ty],
            abi_align,
            abi_align,
        )
        .into();
        let array_ty: TypeHandle = MirArrayType::get(&mut ctx, union_ty, 3).into();
        let mir_ptr_ty =
            MirPtrType::get_generic_with_kind(&mut ctx, array_ty, false, MirPointerKind::SharedRef);
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let undef = mir::MirUndefOp::new(&mut ctx, array_ty);
        undef.get_operation().insert_at_back(block, &ctx);
        let value = undef.get_operation().deref(&ctx).get_result(0);
        let ref_op = Operation::new(
            &mut ctx,
            mir::MirRefOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![value],
            vec![],
            0,
        );
        let mir_ref = mir::MirRefOp::new(ref_op);
        mir_ref.set_mutable(&mut ctx, false);
        mir_ref.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
        ref_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
        let llvm_array_ty = alloca.result_pointee_type(&ctx);
        let llvm_array_data = llvm_array_ty.deref(&ctx);
        let llvm_array = llvm_array_data
            .downcast_ref::<ArrayType>()
            .expect("over-aligned union array must remain an LLVM array");

        assert_eq!(llvm_array.size(), 3);
        assert_eq!(
            crate::convert::types::llvm_type_size_align(&ctx, llvm_array.elem_type()),
            Some((abi_align, 16))
        );
        assert_eq!(
            crate::convert::types::llvm_type_size_align(&ctx, llvm_array_ty),
            Some((abi_align * 3, 16))
        );
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
            Some(abi_align as u32)
        );
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, store.get_operation()),
            Some(abi_align as u32)
        );
    }
}

#[test]
fn convert_ptr_offset_lowers_to_gep_with_pointee_elem_type() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let off_val = block.deref(&ctx).get_argument(1);

    let off_op = Operation::new(
        &mut ctx,
        mir::MirPtrOffsetOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![ptr_val, off_val],
        vec![],
        0,
    );
    off_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).expect("expected one llvm.gep");
    // Element type must come from the MirPtrType pointee (i32), not the
    // i8 fallback used when no operand-type info is available.
    let elem_ty = gep.src_elem_type(&ctx);
    let elem_ty_ref = elem_ty.deref(&ctx);
    let int_ty = elem_ty_ref
        .downcast_ref::<IntegerType>()
        .expect("gep src_elem_type should be IntegerType");
    assert_eq!(int_ty.width(), 32, "gep elem type must be i32 (pointee)");
    assert!(
        gep.no_wrap_flags(&ctx).contains(GepNoWrapFlags::INBOUNDS),
        "ordinary pointer offsets retain the in-bounds contract"
    );
}

#[test]
fn convert_wrapping_ptr_offset_lowers_to_non_inbounds_gep() {
    let mut ctx = make_ctx();
    let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
    let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

    let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let off_val = block.deref(&ctx).get_argument(1);

    let off_op = Operation::new(
        &mut ctx,
        mir::MirPtrOffsetOp::get_concrete_op_info(),
        vec![mir_ptr_ty.into()],
        vec![ptr_val, off_val],
        vec![],
        0,
    );
    mir::MirPtrOffsetOp::new(off_op).set_inbounds(&mut ctx, false);
    off_op.insert_at_back(block, &ctx);
    append_mir_return(&mut ctx, block, vec![]);

    crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

    let body = kernel_blocks(&ctx, module_ptr);
    let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).expect("expected one llvm.gep");
    assert!(
        gep.no_wrap_flags(&ctx).is_empty(),
        "wrapping pointer offsets must not promise in-bounds arithmetic"
    );
}
