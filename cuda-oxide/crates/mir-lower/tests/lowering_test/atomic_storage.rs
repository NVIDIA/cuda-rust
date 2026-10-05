/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::common::{append_return, build_test_kernel, lowered_kernel_body, make_test_ctx};
use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::atomic::{
    AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp,
    NvvmAtomicRmwOp, NvvmAtomicStoreOp,
};
use llvm_export::attributes::ICmpPredicateAttr;
use llvm_export::op_interfaces::{FastMathFlags, IntBinArithOpWithOverflowFlag};
use llvm_export::ops as llvm;
use llvm_export::types::PointerType;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface};
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};

fn pointer_space(ctx: &Context, value: pliron::value::Value) -> u32 {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .unwrap()
        .address_space()
}

#[test]
fn private_rmw_preserves_value_semantics_and_old_result() -> anyhow::Result<()> {
    for width in [32, 64] {
        for kind in [
            AtomicRmwKind::Add,
            AtomicRmwKind::Sub,
            AtomicRmwKind::And,
            AtomicRmwKind::Or,
            AtomicRmwKind::Xor,
            AtomicRmwKind::Xchg,
            AtomicRmwKind::Min,
            AtomicRmwKind::Max,
            AtomicRmwKind::UMin,
            AtomicRmwKind::UMax,
        ] {
            let mut ctx = make_test_ctx();
            let signedness = if matches!(kind, AtomicRmwKind::Min | AtomicRmwKind::Max) {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let ty: TypeHandle = IntegerType::get(&ctx, width, signedness).into();
            let storage = MirPtrType::get(&mut ctx, ty, true, 5);
            let output = MirPtrType::get_global(&mut ctx, ty, true);
            let (module, entry) =
                build_test_kernel(&mut ctx, vec![storage.into(), ty, output.into()]);
            let address = entry.deref(&ctx).get_argument(0);
            let value = entry.deref(&ctx).get_argument(1);
            let output = entry.deref(&ctx).get_argument(2);
            let rmw = NvvmAtomicRmwOp::build(
                &mut ctx,
                address,
                value,
                ty,
                kind.clone(),
                AtomicOrdering::SeqCst,
                AtomicScope::System,
            )
            .get_operation();
            rmw.insert_at_back(entry, &ctx);
            let old = rmw.deref(&ctx).get_result(0);
            NvvmAtomicStoreOp::build(
                &mut ctx,
                old,
                output,
                AtomicOrdering::Relaxed,
                AtomicScope::System,
            )
            .get_operation()
            .insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);
            mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
            module
                .deref(&ctx)
                .verify(&ctx)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let body = lowered_kernel_body(&ctx, module);
            assert!(!body.iter().any(|&op| {
                Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some()
                    || Operation::get_op::<llvm::CallOp>(op, &ctx).is_some()
            }));
            let load = *body
                .iter()
                .find(|&&op| Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some())
                .unwrap();
            let loaded = load.deref(&ctx).get_result(0);
            let store = *body
                .iter()
                .find(|&&op| Operation::get_op::<llvm::StoreOp>(op, &ctx).is_some())
                .unwrap();
            assert_eq!(pointer_space(&ctx, store.deref(&ctx).get_operand(1)), 5);
            let observed = *body
                .iter()
                .find(|&&op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_some())
                .unwrap();
            assert_eq!(
                observed.deref(&ctx).get_operand(1),
                loaded,
                "RMW must return pre-update value"
            );
            for &op in &body {
                if let Some(add) = Operation::get_op::<llvm::AddOp>(op, &ctx) {
                    assert_eq!(add.integer_overflow_flag(&ctx), Default::default());
                }
                if let Some(sub) = Operation::get_op::<llvm::SubOp>(op, &ctx) {
                    assert_eq!(sub.integer_overflow_flag(&ctx), Default::default());
                }
            }
            let predicate = match kind {
                AtomicRmwKind::Min => Some(ICmpPredicateAttr::SLT),
                AtomicRmwKind::Max => Some(ICmpPredicateAttr::SGT),
                AtomicRmwKind::UMin => Some(ICmpPredicateAttr::ULT),
                AtomicRmwKind::UMax => Some(ICmpPredicateAttr::UGT),
                _ => None,
            };
            if let Some(predicate) = predicate {
                let cmp = body
                    .iter()
                    .find_map(|&op| Operation::get_op::<llvm::ICmpOp>(op, &ctx))
                    .unwrap();
                assert_eq!(cmp.predicate(&ctx), predicate);
                let selected = *body
                    .iter()
                    .find(|&&op| Operation::get_op::<llvm::SelectOp>(op, &ctx).is_some())
                    .unwrap();
                assert_eq!(selected.deref(&ctx).get_operand(1), loaded);
                assert_eq!(selected.deref(&ctx).get_operand(2), value);
                assert_eq!(
                    store.deref(&ctx).get_operand(0),
                    selected.deref(&ctx).get_result(0)
                );
            } else if matches!(kind, AtomicRmwKind::Xchg) {
                assert_eq!(store.deref(&ctx).get_operand(0), value);
            }
        }
    }
    Ok(())
}

#[test]
fn private_float_and_pointer_operations_keep_typed_values() -> anyhow::Result<()> {
    for shape in [0, 1, 2, 3] {
        let mut ctx = make_test_ctx();
        let ty: TypeHandle = match shape {
            0 => dialect_mir::types::MirFP16Type::get(&ctx).into(),
            1 => FP32Type::get(&ctx).into(),
            2 => FP64Type::get(&ctx).into(),
            _ => {
                let i64 = IntegerType::get(&ctx, 64, Signedness::Unsigned);
                MirPtrType::get_generic(&mut ctx, i64.into(), true).into()
            }
        };
        let storage = MirPtrType::get(&mut ctx, ty, true, 5);
        let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into(), ty]);
        let address = entry.deref(&ctx).get_argument(0);
        let value = entry.deref(&ctx).get_argument(1);
        NvvmAtomicStoreOp::build(
            &mut ctx,
            value,
            address,
            AtomicOrdering::Release,
            AtomicScope::System,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            ty,
            AtomicOrdering::Acquire,
            AtomicScope::System,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            value,
            ty,
            if shape == 3 {
                AtomicRmwKind::Xchg
            } else {
                AtomicRmwKind::FAdd
            },
            AtomicOrdering::AcqRel,
            AtomicScope::System,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);
        mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
        module
            .deref(&ctx)
            .verify(&ctx)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let body = lowered_kernel_body(&ctx, module);
        assert!(!body.iter().any(
            |&op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_some()
                || Operation::get_op::<llvm::CallOp>(op, &ctx).is_some()
        ));
        for op in body {
            if let Some(add) = Operation::get_op::<llvm::FAddOp>(op, &ctx) {
                assert_eq!(add.fast_math_flags(&ctx), Default::default());
            }
            if Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some() {
                assert_eq!(pointer_space(&ctx, op.deref(&ctx).get_operand(0)), 5);
                if shape == 3 {
                    assert_eq!(pointer_space(&ctx, op.deref(&ctx).get_result(0)), 0);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn private_compare_exchange_selects_new_or_old_but_returns_old() -> anyhow::Result<()> {
    for pointer_value in [false, true] {
        let mut ctx = make_test_ctx();
        let integer: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let ty = if pointer_value {
            MirPtrType::get_generic(&mut ctx, integer, true).into()
        } else {
            integer
        };
        let storage = MirPtrType::get(&mut ctx, ty, true, 5);
        let output = MirPtrType::get_global(&mut ctx, ty, true);
        let (module, entry) =
            build_test_kernel(&mut ctx, vec![storage.into(), ty, ty, output.into()]);
        let address = entry.deref(&ctx).get_argument(0);
        let expected = entry.deref(&ctx).get_argument(1);
        let replacement = entry.deref(&ctx).get_argument(2);
        let output = entry.deref(&ctx).get_argument(3);
        let cas = NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            address,
            expected,
            replacement,
            ty,
            AtomicOrdering::SeqCst,
            AtomicOrdering::SeqCst,
            AtomicScope::System,
        )
        .get_operation();
        cas.insert_at_back(entry, &ctx);
        let old = cas.deref(&ctx).get_result(0);
        NvvmAtomicStoreOp::build(
            &mut ctx,
            old,
            output,
            AtomicOrdering::Relaxed,
            AtomicScope::System,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);
        mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
        module
            .deref(&ctx)
            .verify(&ctx)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let body = lowered_kernel_body(&ctx, module);
        let load = *body
            .iter()
            .find(|&&op| Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some())
            .unwrap();
        let old = load.deref(&ctx).get_result(0);
        let compare = body
            .iter()
            .find_map(|&op| Operation::get_op::<llvm::ICmpOp>(op, &ctx))
            .unwrap();
        assert_eq!(compare.predicate(&ctx), ICmpPredicateAttr::EQ);
        assert_eq!(compare.get_operation().deref(&ctx).get_operand(0), old);
        assert_eq!(compare.get_operation().deref(&ctx).get_operand(1), expected);
        let select = *body
            .iter()
            .find(|&&op| Operation::get_op::<llvm::SelectOp>(op, &ctx).is_some())
            .unwrap();
        assert_eq!(
            select.deref(&ctx).get_operand(0),
            compare.get_operation().deref(&ctx).get_result(0)
        );
        assert_eq!(select.deref(&ctx).get_operand(1), replacement);
        assert_eq!(select.deref(&ctx).get_operand(2), old);
        let store = *body
            .iter()
            .find(|&&op| Operation::get_op::<llvm::StoreOp>(op, &ctx).is_some())
            .unwrap();
        assert_eq!(
            store.deref(&ctx).get_operand(0),
            select.deref(&ctx).get_result(0)
        );
        let observed = *body
            .iter()
            .find(|&&op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_some())
            .unwrap();
        assert_eq!(observed.deref(&ctx).get_operand(1), old);
        assert!(
            !body
                .iter()
                .any(|&op| Operation::get_op::<llvm::AtomicCmpxchgOp>(op, &ctx).is_some())
        );
    }
    Ok(())
}

#[test]
fn generic_storage_dispatches_by_address_and_merges_old_values() -> anyhow::Result<()> {
    let mut ctx = make_test_ctx();
    let ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let storage = MirPtrType::get_generic(&mut ctx, ty, true);
    let output = MirPtrType::get_global(&mut ctx, ty, true);
    let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into(), ty, output.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let output = entry.deref(&ctx).get_argument(2);
    let rmw = NvvmAtomicRmwOp::build(
        &mut ctx,
        address,
        value,
        ty,
        AtomicRmwKind::Add,
        AtomicOrdering::AcqRel,
        AtomicScope::Device,
    )
    .get_operation();
    rmw.insert_at_back(entry, &ctx);
    let old = rmw.deref(&ctx).get_result(0);
    NvvmAtomicStoreOp::build(
        &mut ctx,
        old,
        output,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);
    mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
    module
        .deref(&ctx)
        .verify(&ctx)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let body = lowered_kernel_body(&ctx, module);
    let observed = body
        .iter()
        .filter_map(|&op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
        .find(|asm| {
            String::from((*asm.get_attr_llvm_inline_asm_template(&ctx).unwrap()).clone())
                .starts_with("st.")
        })
        .unwrap()
        .get_operation();
    let continuation = observed.deref(&ctx).get_parent_block().unwrap();
    let mut queries = vec![];
    let mut atomic_spaces = vec![];
    let mut incoming = vec![];
    let mut fences = 0;
    for &op in &body {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(name) = call.callee(&ctx)
        {
            queries.push(name.to_string());
            assert_eq!(op.deref(&ctx).get_operand(0), address);
        }
        if Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some() {
            atomic_spaces.push(pointer_space(&ctx, op.deref(&ctx).get_operand(0)));
        }
        if Operation::get_op::<llvm::BrOp>(op, &ctx).is_some()
            && op.deref(&ctx).get_successor(0) == continuation
        {
            incoming.push(op.deref(&ctx).get_operand(0));
        }
        if let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx)
            && String::from((*asm.get_attr_llvm_inline_asm_template(&ctx).unwrap()).clone())
                .starts_with("fence.")
        {
            fences += 1;
        }
    }
    assert_eq!(queries, ["llvm_nvvm_isspacep_local"]);
    assert_eq!(atomic_spaces, [0]);
    assert_eq!(incoming.len(), 2);
    assert_eq!(
        fences, 2,
        "both original fences remain on the nonlocal path"
    );
    let load = *body
        .iter()
        .find(|&&op| Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some())
        .unwrap();
    assert_eq!(pointer_space(&ctx, load.deref(&ctx).get_operand(0)), 5);
    assert_eq!(incoming[0], load.deref(&ctx).get_result(0));
    let observed = body
        .iter()
        .filter_map(|&op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
        .find(|asm| {
            String::from((*asm.get_attr_llvm_inline_asm_template(&ctx).unwrap()).clone())
                .starts_with("st.")
        })
        .unwrap()
        .get_operation();
    let continuation = observed.deref(&ctx).get_parent_block().unwrap();
    assert_eq!(
        observed.deref(&ctx).get_operand(1),
        continuation.deref(&ctx).get_argument(0)
    );
    Ok(())
}

#[test]
fn atomic_storage_rejects_readonly_and_unsupported_address_spaces() {
    for space in [2, 4, 6, 101] {
        for operation in 0..4 {
            let mut ctx = make_test_ctx();
            let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
            let storage = MirPtrType::get(&mut ctx, ty, true, space);
            let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into(), ty]);
            let address = entry.deref(&ctx).get_argument(0);
            let value = entry.deref(&ctx).get_argument(1);
            let op = match operation {
                0 => NvvmAtomicLoadOp::build(
                    &mut ctx,
                    address,
                    ty,
                    AtomicOrdering::Relaxed,
                    AtomicScope::Device,
                )
                .get_operation(),
                1 => NvvmAtomicStoreOp::build(
                    &mut ctx,
                    value,
                    address,
                    AtomicOrdering::Relaxed,
                    AtomicScope::Device,
                )
                .get_operation(),
                2 => NvvmAtomicRmwOp::build(
                    &mut ctx,
                    address,
                    value,
                    ty,
                    AtomicRmwKind::Add,
                    AtomicOrdering::Relaxed,
                    AtomicScope::Device,
                )
                .get_operation(),
                _ => NvvmAtomicCmpxchgOp::build(
                    &mut ctx,
                    address,
                    value,
                    value,
                    ty,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                    AtomicScope::Device,
                )
                .get_operation(),
            };
            op.insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);
            let error = mir_lower::lower_mir_to_llvm(&mut ctx, module)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("atomic storage address space") && error.contains("unsupported"),
                "{error}"
            );
        }
    }
}

// A cluster-shared address must remain cluster-shared. Testing only the CTA
// window and treating the remainder as global would corrupt remote DSMEM.
#[test]
fn known_nonlocal_storage_keeps_its_address_space_without_dispatch() -> anyhow::Result<()> {
    for space in [1, 3, 7] {
        let mut ctx = make_test_ctx();
        let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let storage = MirPtrType::get(&mut ctx, ty, true, space);
        let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into(), ty]);
        let address = entry.deref(&ctx).get_argument(0);
        let value = entry.deref(&ctx).get_argument(1);
        NvvmAtomicLoadOp::build(
            &mut ctx,
            address,
            ty,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicStoreOp::build(
            &mut ctx,
            value,
            address,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicRmwOp::build(
            &mut ctx,
            address,
            value,
            ty,
            AtomicRmwKind::Add,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            address,
            value,
            value,
            ty,
            AtomicOrdering::Acquire,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .get_operation()
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);
        mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
        module
            .deref(&ctx)
            .verify(&ctx)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let body = lowered_kernel_body(&ctx, module);
        let mut atomic_spaces = vec![];
        let mut accesses = 0;
        for op in body {
            assert!(Operation::get_op::<llvm::CallOp>(op, &ctx).is_none());
            if Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some()
                || Operation::get_op::<llvm::AtomicCmpxchgOp>(op, &ctx).is_some()
            {
                atomic_spaces.push(pointer_space(&ctx, op.deref(&ctx).get_operand(0)));
            }
            if Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_some() {
                accesses += 1;
                assert_eq!(pointer_space(&ctx, op.deref(&ctx).get_operand(0)), 0);
            }
        }
        assert_eq!(atomic_spaces, [space, space]);
        assert_eq!(accesses, 2);
    }
    Ok(())
}

#[test]
fn shared_storage_preserves_generic_pointer_atomic_values() -> anyhow::Result<()> {
    let mut ctx = make_test_ctx();

    let pointee: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let ty: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, true).into();
    let storage = MirPtrType::get_shared(&mut ctx, ty, true);

    let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into(), ty]);
    let address = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);

    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        ty,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    NvvmAtomicStoreOp::build(
        &mut ctx,
        value,
        address,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    NvvmAtomicRmwOp::build(
        &mut ctx,
        address,
        value,
        ty,
        AtomicRmwKind::Xchg,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    NvvmAtomicCmpxchgOp::build(
        &mut ctx,
        address,
        value,
        value,
        ty,
        AtomicOrdering::Acquire,
        AtomicOrdering::Relaxed,
        AtomicScope::Device,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);

    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;

    module
        .deref(&ctx)
        .verify(&ctx)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let body = lowered_kernel_body(&ctx, module);

    let mut atomic_spaces = vec![];
    let mut accesses = 0;

    for op in body {
        assert!(
            Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
            "known shared storage must not require runtime address-space dispatch"
        );

        if Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some() {
            atomic_spaces.push(pointer_space(&ctx, op.deref(&ctx).get_operand(0)));

            assert_eq!(
                pointer_space(&ctx, op.deref(&ctx).get_operand(1)),
                0,
                "atomic exchange value must remain a generic pointer"
            );
            assert_eq!(
                pointer_space(&ctx, op.deref(&ctx).get_result(0)),
                0,
                "atomic exchange result must remain a generic pointer"
            );
        }

        if Operation::get_op::<llvm::AtomicCmpxchgOp>(op, &ctx).is_some() {
            atomic_spaces.push(pointer_space(&ctx, op.deref(&ctx).get_operand(0)));

            assert_eq!(
                pointer_space(&ctx, op.deref(&ctx).get_operand(1)),
                0,
                "compare-exchange expected value must remain a generic pointer"
            );
            assert_eq!(
                pointer_space(&ctx, op.deref(&ctx).get_operand(2)),
                0,
                "compare-exchange replacement value must remain a generic pointer"
            );
        }

        if Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_some() {
            accesses += 1;
            assert_eq!(
                pointer_space(&ctx, op.deref(&ctx).get_operand(0)),
                0,
                "inline PTX load/store uses the generic address representation"
            );
        }
    }

    assert_eq!(atomic_spaces, [3, 3]);
    assert_eq!(accesses, 2);

    Ok(())
}

#[test]
fn private_atomic_does_not_remove_a_standalone_fence() -> anyhow::Result<()> {
    use dialect_nvvm::ops::atomic::NvvmAtomicFenceOp;
    let mut ctx = make_test_ctx();
    let ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let storage = MirPtrType::get(&mut ctx, ty, true, 5);
    let (module, entry) = build_test_kernel(&mut ctx, vec![storage.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    NvvmAtomicLoadOp::build(
        &mut ctx,
        address,
        ty,
        AtomicOrdering::SeqCst,
        AtomicScope::System,
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    NvvmAtomicFenceOp::build(&mut ctx, AtomicOrdering::SeqCst, AtomicScope::System)
        .get_operation()
        .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);
    mir_lower::lower_mir_to_llvm(&mut ctx, module).map_err(|e| anyhow::anyhow!("{e}"))?;
    module
        .deref(&ctx)
        .verify(&ctx)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let body = lowered_kernel_body(&ctx, module);
    assert_eq!(
        body.iter()
            .filter(|&&op| Operation::get_op::<llvm::LoadOp>(op, &ctx).is_some())
            .count(),
        1
    );
    let callees: Vec<_> = body
        .into_iter()
        .filter_map(|op| {
            let call = Operation::get_op::<llvm::CallOp>(op, &ctx)?;
            if let CallOpCallable::Direct(name) = call.callee(&ctx) {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(callees, ["llvm_nvvm_membar_sys"]);
    Ok(())
}
