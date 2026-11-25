/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! References to SSA values need fresh storage at their original control-flow position.

// These fixtures intentionally forge pointer contracts to test verifier rejection.
#![allow(clippy::disallowed_methods)]

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use cuda_oxide_mlir_export::{MlirAttribute, MlirOperation, MlirType};
use dialect_mir::{
    attributes::MirPointerKindAuthorityAttr,
    ops::{MirConstructStructOp, MirFuncOp, MirGotoOp, MirRefOp, MirReturnOp},
    types::{MirArrayType, MirPointerKind, MirPtrType, MirStructType, MirTupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};

/// Build the same full-profile context used by CuTe device compilation.
fn context() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    ctx
}

/// Append one source operation with no successors or nested regions.
fn append<O: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let operation = Operation::new(ctx, O::get_concrete_op_info(), results, operands, vec![], 0);
    operation.insert_at_back(block, ctx);
    operation
}

/// Put values in a second block so accidental allocation hoisting is observable.
fn function(
    ctx: &mut Context,
    args: Vec<TypeHandle>,
) -> (ModuleOp, Ptr<BasicBlock>, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, "value_reference".try_into().unwrap());
    let signature = FunctionType::get(ctx, args.clone(), vec![]);
    let function = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(ctx, function, TypeAttr::new(signature.into()))
        .set_symbol_name(ctx, "test".try_into().unwrap());
    module.append_operation(ctx, function, 0);
    let region = function.deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, args);
    let body = BasicBlock::new(ctx, None, vec![]);
    entry.insert_at_back(region, ctx);
    body.insert_at_back(region, ctx);
    let branch = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![body],
        0,
    );
    branch.insert_at_back(entry, ctx);
    (module, entry, body)
}

/// Materialize a reference with the explicitly supplied Rust boundary evidence.
fn reference(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    value: Value,
    pointer: TypeHandle,
    mutable: bool,
    authority: MirPointerKindAuthorityAttr,
) {
    let reference = append::<MirRefOp>(ctx, block, vec![pointer], vec![value]);
    let reference = MirRefOp::new(reference);
    reference.set_mutable(ctx, mutable);
    reference.set_pointer_kind_authority(ctx, authority);
}

/// Return the translated test function, retaining individual operations for assertions.
fn translate(ctx: &Context, module: &ModuleOp) -> Result<MlirOperation, String> {
    fn find(function: &MlirOperation) -> Option<&MlirOperation> {
        if function.name == "func.func" {
            return Some(function);
        }
        function
            .regions
            .iter()
            .flat_map(|region| &region.blocks)
            .flat_map(|block| &block.operations)
            .find_map(find)
    }
    let translated = CutlassFullCuteMlir22::new("sm_120a")
        .unwrap()
        .translate_module(ctx, module)
        .map_err(|error| error.to_string())?;
    Ok(find(&translated.root)
        .expect("translated test function")
        .clone())
}

/// FnMut conversion closures have no captures, but still require a mutable reference.
#[test]
fn zero_sized_closure_reference_materializes_in_its_original_block() {
    let mut ctx = context();
    let closure = MirStructType::get_with_full_layout(
        &mut ctx,
        "ConversionClosure".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        1,
    )
    .into();
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, closure, true, 0, MirPointerKind::UniqueRef).into();
    let (module, _, body) = function(&mut ctx, vec![]);
    let value = append::<MirConstructStructOp>(&mut ctx, body, vec![closure], vec![])
        .deref(&ctx)
        .get_result(0);
    reference(
        &mut ctx,
        body,
        value,
        pointer,
        true,
        MirPointerKindAuthorityAttr::Reborrow,
    );
    append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
    let function = translate(&ctx, &module).unwrap();
    let blocks = &function.regions[0].blocks;
    assert!(
        blocks[0]
            .operations
            .iter()
            .all(|operation| operation.name != "llvm.alloca")
    );
    let operations = &blocks[1].operations;
    let alloca = operations
        .iter()
        .find(|operation| operation.name == "llvm.alloca")
        .unwrap();
    let store = operations
        .iter()
        .find(|operation| operation.name == "llvm.store")
        .unwrap();
    assert_eq!(store.operands[1].id, alloca.results[0].id);
    assert_eq!(
        alloca.properties["alignment"],
        MlirAttribute::Integer {
            value: 1,
            ty: MlirType::Integer(64)
        }
    );
}

/// Distinct references must retain distinct storage, even when borrowing the same SSA value.
#[test]
fn scalar_references_initialize_distinct_slots_with_the_operand() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, scalar, false, 0, MirPointerKind::SharedRef).into();
    let (module, entry, body) = function(&mut ctx, vec![scalar]);
    let value = entry.deref(&ctx).get_argument(0);
    for _ in 0..2 {
        reference(
            &mut ctx,
            body,
            value,
            pointer,
            false,
            MirPointerKindAuthorityAttr::Reborrow,
        );
    }
    append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
    let function = translate(&ctx, &module).unwrap();
    let blocks = &function.regions[0].blocks;
    let operations = &blocks[1].operations;
    let allocas: Vec<_> = operations
        .iter()
        .filter(|operation| operation.name == "llvm.alloca")
        .collect();
    let stores: Vec<_> = operations
        .iter()
        .filter(|operation| operation.name == "llvm.store")
        .collect();
    assert_eq!(allocas.len(), 2);
    assert_eq!(stores.len(), 2);
    assert_ne!(allocas[0].results[0].id, allocas[1].results[0].id);
    for (alloca, store) in allocas.iter().zip(stores) {
        assert_eq!(store.operands[0].id, blocks[0].arguments[0].id);
        assert_eq!(store.operands[1].id, alloca.results[0].id);
        for operation in [*alloca, store] {
            assert_eq!(
                operation.properties["alignment"],
                MlirAttribute::Integer {
                    value: 4,
                    ty: MlirType::Integer(64)
                }
            );
        }
    }
}

/// Explicit Rust over-alignment belongs on the allocation and initialization store.
#[test]
fn value_reference_preserves_explicit_abi_alignment() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let tuple =
        MirTupleType::get_with_layout(&mut ctx, vec![scalar], vec![0], vec![0], 32, 32).into();
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, tuple, false, 0, MirPointerKind::SharedRef).into();
    let (module, entry, body) = function(&mut ctx, vec![tuple]);
    let value = entry.deref(&ctx).get_argument(0);
    reference(
        &mut ctx,
        body,
        value,
        pointer,
        false,
        MirPointerKindAuthorityAttr::Reborrow,
    );
    append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
    let function = translate(&ctx, &module).unwrap();
    for operation in function.regions[0].blocks[1]
        .operations
        .iter()
        .filter(|operation| matches!(operation.name.as_str(), "llvm.alloca" | "llvm.store"))
    {
        assert_eq!(
            operation.properties["alignment"],
            MlirAttribute::Integer {
                value: 32,
                ty: MlirType::Integer(64)
            }
        );
    }
}

/// An array inherits its element's explicit ABI alignment, beyond natural field alignment.
#[test]
fn array_reference_preserves_overaligned_element_abi() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let element = MirStructType::get_with_full_layout(
        &mut ctx,
        "AlignedValue".into(),
        vec!["value".into()],
        vec![scalar],
        vec![0],
        vec![0],
        32,
        32,
    )
    .into();
    let array = MirArrayType::get(&mut ctx, element, 2).into();
    let pointer =
        MirPtrType::get_with_kind(&mut ctx, array, false, 0, MirPointerKind::SharedRef).into();
    let (module, entry, body) = function(&mut ctx, vec![array]);
    let value = entry.deref(&ctx).get_argument(0);
    reference(
        &mut ctx,
        body,
        value,
        pointer,
        false,
        MirPointerKindAuthorityAttr::Reborrow,
    );
    append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
    let function = translate(&ctx, &module).unwrap();
    let memory: Vec<_> = function.regions[0].blocks[1]
        .operations
        .iter()
        .filter(|operation| matches!(operation.name.as_str(), "llvm.alloca" | "llvm.store"))
        .collect();
    assert_eq!(memory.len(), 2);
    for operation in memory {
        assert_eq!(
            operation.properties["alignment"],
            MlirAttribute::Integer {
                value: 32,
                ty: MlirType::Integer(64)
            }
        );
    }
}

/// Dropping source attributes must not permit forged kinds, mutability, or address spaces.
#[test]
fn references_reject_invalid_source_pointer_contracts() {
    for case in 0..5 {
        let mut ctx = context();
        let scalar = FP32Type::get(&ctx).into();
        let wrong = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let (pointee, mutable, address_space, kind, authority) = match case {
            0 => (
                scalar,
                false,
                0,
                MirPointerKind::RawConst,
                MirPointerKindAuthorityAttr::Reborrow,
            ),
            1 => (
                scalar,
                true,
                0,
                MirPointerKind::UniqueRef,
                MirPointerKindAuthorityAttr::Reborrow,
            ),
            2 => (
                scalar,
                false,
                3,
                MirPointerKind::SharedRef,
                MirPointerKindAuthorityAttr::Reborrow,
            ),
            3 => (
                wrong,
                false,
                0,
                MirPointerKind::SharedRef,
                MirPointerKindAuthorityAttr::Reborrow,
            ),
            _ => (
                scalar,
                false,
                0,
                MirPointerKind::SharedRef,
                MirPointerKindAuthorityAttr::RustCast,
            ),
        };
        let pointer =
            MirPtrType::get_with_kind(&mut ctx, pointee, mutable, address_space, kind).into();
        let (module, entry, body) = function(&mut ctx, vec![scalar]);
        let value = entry.deref(&ctx).get_argument(0);
        reference(&mut ctx, body, value, pointer, false, authority);
        append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
        assert!(
            translate(&ctx, &module).is_err(),
            "accepted invalid reference case {case}"
        );
    }
}

/// A reference cannot bypass missing canonical bool or generic pointer storage adapters.
#[test]
fn references_reject_payloads_requiring_storage_adapters() {
    let mut ctx = context();
    let scalar = FP32Type::get(&ctx).into();
    let flag = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let flags = MirArrayType::get(&mut ctx, flag, 2).into();
    let capture = MirStructType::get(
        &mut ctx,
        "BoolCapture".into(),
        vec!["flag".into()],
        vec![flag],
    )
    .into();
    let shared = MirPtrType::get_shared(&mut ctx, scalar, false).into();
    for payload in [flag, flags, capture, shared] {
        let pointer =
            MirPtrType::get_with_kind(&mut ctx, payload, false, 0, MirPointerKind::SharedRef)
                .into();
        let (module, entry, body) = function(&mut ctx, vec![payload]);
        let value = entry.deref(&ctx).get_argument(0);
        reference(
            &mut ctx,
            body,
            value,
            pointer,
            false,
            MirPointerKindAuthorityAttr::Reborrow,
        );
        append::<MirReturnOp>(&mut ctx, body, vec![], vec![]);
        assert!(
            translate(&ctx, &module).is_err(),
            "accepted payload requiring a storage adapter"
        );
    }
}
