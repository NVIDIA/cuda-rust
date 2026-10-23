/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Transparent numeric wrappers retain Rust identity across CuTe vector copies.

use cuda_oxide_mlir_export::render_module;
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_cute::{
    attributes::CuteTensorAccessAttr,
    tensor_ops::{
        CuteTensorLoadIntoOp, CuteTensorMakeOp, CuteTensorSliceOp, CuteTensorStoreElementAbsOp,
        CuteTensorStoreFromOp, CuteTensorZippedDivideOp,
    },
};
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirPtrType, MirStructType, StructAbiKind, address_space},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};

/// Append one detached operation to the test function.
fn append<O: Op>(block: Ptr<BasicBlock>, operation: O, ctx: &Context) -> Ptr<Operation> {
    let operation = operation.get_operation();
    operation.insert_at_back(block, ctx);
    operation
}

/// Copies use the scalar representation; scalar tails retain the aggregate's exact stride.
#[test]
fn transparent_u16_tensor_uses_bit_preserving_vector_copies_and_tails() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(
        &mut ctx,
        Identifier::try_from("transparent_tensor").unwrap(),
    );
    let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let wrapper: TypeHandle = MirStructType::get_with_full_layout_and_abi(
        &mut ctx,
        "NumericBits".into(),
        vec!["0".into()],
        vec![u16_ty],
        vec![0],
        vec![0],
        2,
        2,
        StructAbiKind::TransparentScalar,
    )
    .into();
    let read_ptr: TypeHandle =
        MirPtrType::get(&mut ctx, wrapper, false, address_space::GENERIC).into();
    let write_ptr: TypeHandle =
        MirPtrType::get(&mut ctx, wrapper, true, address_space::GENERIC).into();
    let carrier_ptr: TypeHandle =
        MirPtrType::get(&mut ctx, wrapper, true, address_space::LOCAL).into();
    let arguments = vec![read_ptr, write_ptr, carrier_ptr, u64_ty, u64_ty, wrapper];
    let function_type = FunctionType::get(&ctx, arguments.clone(), vec![]);
    let function_op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
    function.set_symbol_name(&mut ctx, Identifier::try_from("copy_numeric_bits").unwrap());
    module.append_operation(&mut ctx, function_op, 0);
    let region = function_op.deref(&ctx).get_region(0);
    let entry = BasicBlock::new(&mut ctx, None, arguments);
    entry.insert_at_back(region, &ctx);
    let args = entry.deref(&ctx).arguments().collect::<Vec<_>>();

    for (pointer, access) in [
        (args[0], CuteTensorAccessAttr::ReadOnly),
        (args[1], CuteTensorAccessAttr::ReadWrite),
    ] {
        let make = append(
            entry,
            CuteTensorMakeOp::new(&mut ctx, pointer, args[3], wrapper, wrapper, access, 2),
            &ctx,
        );
        let view = make.deref(&ctx).get_result(0);
        let zipped = append(
            entry,
            CuteTensorZippedDivideOp::new(&mut ctx, view, 8),
            &ctx,
        );
        let divided = zipped.deref(&ctx).get_result(0);
        let slice = append(
            entry,
            CuteTensorSliceOp::new(&mut ctx, divided, args[4]),
            &ctx,
        );
        let tile = slice.deref(&ctx).get_result(0);
        if access == CuteTensorAccessAttr::ReadOnly {
            append(
                entry,
                CuteTensorLoadIntoOp::new(&mut ctx, tile, args[2], 16),
                &ctx,
            );
        } else {
            append(
                entry,
                CuteTensorStoreFromOp::new(&mut ctx, args[2], tile, 16),
                &ctx,
            );
            append(
                entry,
                CuteTensorStoreElementAbsOp::new(&mut ctx, tile, args[4], args[5]),
                &ctx,
            );
        }
    }
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, &ctx);

    let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let text = render_module(&target).unwrap();
    assert!(text.contains("!cute.memref<i16, gmem"), "{text}");
    assert!(
        text.contains("!cute_nvgpu.atom.universal_copy<i16, 128b>"),
        "{text}"
    );
    assert_eq!(text.matches("\"cute.copy\"").count(), 2, "{text}");
    assert!(text.contains("elem_type = !llvm.struct<(i16)>"), "{text}");
    assert!(text.contains("\"llvm.store\""), "{text}");
    assert!(
        !text.contains("\"arith.extf\"") && !text.contains("\"arith.truncf\""),
        "{text}"
    );
}
