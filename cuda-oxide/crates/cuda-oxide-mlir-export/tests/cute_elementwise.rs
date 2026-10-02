/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::num::NonZero;

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_cute::{
    attributes::CuteTensorAccessAttr,
    tensor_ops::{
        CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
        CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
        CuteTensorZippedDivideOp,
    },
};
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirFP16Type, MirPtrType, MirSliceType, address_space},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
};
use pliron_mlir_export::render_module;

fn append<O: Op>(block: Ptr<BasicBlock>, operation: O, ctx: &Context) -> Ptr<Operation> {
    let operation = operation.get_operation();
    operation.insert_at_back(block, ctx);
    operation
}

#[test]
fn all_eight_elementwise_ops_map_to_cutlass_cute_and_core_mlir() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, Identifier::try_from("cute_elementwise").unwrap());
    let f32_type: TypeHandle = FP32Type::get(&ctx).into();
    let f16_type: TypeHandle = MirFP16Type::get(&ctx).into();
    let u64_type: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let i1_type: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    // Rust slice fields arrive through the generic pointer ABI even though
    // the resulting CuTe view is global memory. Exercise that real bridge
    // here; the exporter must spell the AS0 -> AS1 transition explicitly.
    let global_ro: TypeHandle =
        MirPtrType::get(&mut ctx, f32_type, false, address_space::GENERIC).into();
    let global_rw: TypeHandle =
        MirPtrType::get(&mut ctx, f32_type, true, address_space::GENERIC).into();
    let local_rw: TypeHandle =
        MirPtrType::get(&mut ctx, f32_type, true, address_space::LOCAL).into();
    let global_ro_f16: TypeHandle =
        MirPtrType::get(&mut ctx, f16_type, false, address_space::GENERIC).into();
    let local_rw_f16: TypeHandle =
        MirPtrType::get(&mut ctx, f16_type, true, address_space::LOCAL).into();
    let arguments = vec![
        global_ro,
        global_rw,
        local_rw,
        u64_type,
        u64_type,
        f32_type,
        global_ro_f16,
        local_rw_f16,
    ];
    let results = vec![i1_type, u64_type];
    let function_type = FunctionType::get(&ctx, arguments.clone(), results);
    let function_op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(&mut ctx, function_op, TypeAttr::new(function_type.into()));
    function.set_symbol_name(&mut ctx, Identifier::try_from("elementwise").unwrap());
    module.append_operation(&mut ctx, function_op, 0);
    let region = function_op.deref(&ctx).get_region(0);
    let entry = BasicBlock::new(&mut ctx, None, arguments);
    entry.insert_at_back(region, &ctx);
    let arguments = entry.deref(&ctx).arguments().collect::<Vec<_>>();
    let read_ptr = arguments[0];
    let write_ptr = arguments[1];
    let carrier = arguments[2];
    let len = arguments[3];
    let tile_index = arguments[4];
    let scalar = arguments[5];
    let read_ptr_f16 = arguments[6];
    let carrier_f16 = arguments[7];

    let read_make = append(
        entry,
        CuteTensorMakeOp::new(
            &mut ctx,
            read_ptr,
            len,
            f32_type,
            f32_type,
            CuteTensorAccessAttr::ReadOnly,
            4,
        ),
        &ctx,
    );
    let read = read_make.deref(&ctx).get_result(0);
    let read_zipped = append(
        entry,
        CuteTensorZippedDivideOp::new(&mut ctx, read, 4),
        &ctx,
    );
    let read_zipped = read_zipped.deref(&ctx).get_result(0);
    let read_tile = append(
        entry,
        CuteTensorSliceOp::new(&mut ctx, read_zipped, tile_index),
        &ctx,
    );
    let read_tile = read_tile.deref(&ctx).get_result(0);
    let is_full = append(entry, CuteTensorIsFullOp::new(&mut ctx, read_tile), &ctx);
    let base = append(entry, CuteTensorBaseOp::new(&mut ctx, read_tile), &ctx);
    append(
        entry,
        CuteTensorLoadIntoOp::new(&mut ctx, read_tile, carrier, 16),
        &ctx,
    );

    let write_make = append(
        entry,
        CuteTensorMakeOp::new(
            &mut ctx,
            write_ptr,
            len,
            f32_type,
            f32_type,
            CuteTensorAccessAttr::ReadWrite,
            4,
        ),
        &ctx,
    );
    let write = write_make.deref(&ctx).get_result(0);
    let write_zipped = append(
        entry,
        CuteTensorZippedDivideOp::new(&mut ctx, write, 4),
        &ctx,
    );
    let write_zipped = write_zipped.deref(&ctx).get_result(0);
    let write_tile = append(
        entry,
        CuteTensorSliceOp::new(&mut ctx, write_zipped, tile_index),
        &ctx,
    );
    let write_tile = write_tile.deref(&ctx).get_result(0);
    append(
        entry,
        CuteTensorStoreFromOp::new(&mut ctx, carrier, write_tile, 16),
        &ctx,
    );
    append(
        entry,
        CuteTensorStoreElementAbsOp::new(&mut ctx, write_tile, tile_index, scalar),
        &ctx,
    );

    // The production example has a second instantiation: eight f16 values
    // still form one 16-byte transaction. Keep that real type/shape pair in
    // the parser fixture rather than assuming the f32 spelling generalizes.
    let read_make_f16 = append(
        entry,
        CuteTensorMakeOp::new(
            &mut ctx,
            read_ptr_f16,
            len,
            f16_type,
            f16_type,
            CuteTensorAccessAttr::ReadOnly,
            2,
        ),
        &ctx,
    );
    let read_f16 = read_make_f16.deref(&ctx).get_result(0);
    let read_zipped_f16 = append(
        entry,
        CuteTensorZippedDivideOp::new(&mut ctx, read_f16, 8),
        &ctx,
    );
    let read_zipped_f16 = read_zipped_f16.deref(&ctx).get_result(0);
    let read_tile_f16 = append(
        entry,
        CuteTensorSliceOp::new(&mut ctx, read_zipped_f16, tile_index),
        &ctx,
    );
    let read_tile_f16 = read_tile_f16.deref(&ctx).get_result(0);
    append(
        entry,
        CuteTensorLoadIntoOp::new(&mut ctx, read_tile_f16, carrier_f16, 16),
        &ctx,
    );
    let is_full = is_full.deref(&ctx).get_result(0);
    let base = base.deref(&ctx).get_result(0);
    let return_op = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![is_full, base],
        vec![],
        0,
    );
    return_op.insert_at_back(entry, &ctx);

    // The profile runs the shared whole-module CuTe verifier, then
    // composes builtin, MIR, NVVM, and CuTe mappings.
    let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let text = render_module(&target);

    assert!(text.contains("\"cute.make_view\""), "{text}");
    assert!(text.contains("\"cute.zipped_divide\""), "{text}");
    assert!(text.contains("\"cute.slice\""), "{text}");
    assert!(
        text.contains("!cute_nvgpu.atom.universal_copy<f32, 128b>"),
        "{text}"
    );
    assert!(
        text.contains("!cute_nvgpu.atom.universal_copy<f16, 128b>"),
        "{text}"
    );
    assert_eq!(text.matches("\"cute.copy\"").count(), 3, "{text}");
    assert_eq!(
        text.matches("operandSegmentSizes = array<i32: 1, 1, 1, 0>")
            .count(),
        3,
        "{text}"
    );
    assert_eq!(text.matches("\"arith.ori\"").count(), 2, "{text}");
    assert!(text.contains("\"arith.andi\""), "{text}");
    assert!(text.contains("value = 4611686018427387903 : i64"), "{text}");
    assert!(text.contains("predicate = 9 : i64"), "{text}");
    assert!(text.contains("\"llvm.getelementptr\""), "{text}");
    assert!(text.contains("\"llvm.store\""), "{text}");
    assert_eq!(text.matches("\"llvm.addrspacecast\"").count(), 6, "{text}");
    assert_eq!(
        text.matches("(!llvm.ptr) -> !llvm.ptr<1>").count(),
        3,
        "{text}"
    );
    assert_eq!(
        text.matches("(!llvm.ptr<5>) -> !llvm.ptr").count(),
        3,
        "{text}"
    );
    assert!(!text.contains("cute.tensor_"), "{text}");
}

#[test]
fn production_kernel_gets_one_device_module_and_the_six_argument_launch_abi() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, Identifier::try_from("elementwise_abi").unwrap());
    let f32_type: TypeHandle = FP32Type::get(&ctx).into();
    let slice_type: TypeHandle = MirSliceType::get(&mut ctx, f32_type).into();
    let function_type = FunctionType::get(&ctx, vec![slice_type, slice_type, slice_type], vec![]);
    let kernel_operation = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let kernel = MirFuncOp::new(
        &mut ctx,
        kernel_operation,
        TypeAttr::new(function_type.into()),
    );
    kernel.set_symbol_name(&mut ctx, Identifier::try_from("add_f32").unwrap());
    {
        let u32_type = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let width = NonZero::new(32).unwrap();
        let mut operation = kernel_operation.deref_mut(&ctx);
        operation.attributes.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
        for (name, value) in [
            ("reqntid_x", 256),
            ("reqntid_y", 1),
            ("reqntid_z", 1),
            ("cluster_dim_x", 2),
            ("cluster_dim_y", 1),
            ("cluster_dim_z", 1),
        ] {
            operation.attributes.set(
                Identifier::try_from(name).unwrap(),
                IntegerAttr::new(u32_type, APInt::from_u32(value, width)),
            );
        }
    }
    module.append_operation(&mut ctx, kernel_operation, 0);
    let kernel_region = kernel_operation.deref(&ctx).get_region(0);
    let kernel_entry = BasicBlock::new(&mut ctx, None, vec![slice_type, slice_type, slice_type]);
    kernel_entry.insert_at_back(kernel_region, &ctx);
    append(
        kernel_entry,
        MirReturnOp::new(Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )),
        &ctx,
    );

    let helper_type = FunctionType::get(&ctx, vec![], vec![]);
    let helper_operation = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let helper = MirFuncOp::new(
        &mut ctx,
        helper_operation,
        TypeAttr::new(helper_type.into()),
    );
    helper.set_symbol_name(&mut ctx, Identifier::try_from("device_helper").unwrap());
    module.append_operation(&mut ctx, helper_operation, 0);
    let helper_region = helper_operation.deref(&ctx).get_region(0);
    let helper_entry = BasicBlock::new(&mut ctx, None, vec![]);
    helper_entry.insert_at_back(helper_region, &ctx);
    append(
        helper_entry,
        MirReturnOp::new(Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )),
        &ctx,
    );

    let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
    let target = profile.translate_module(&ctx, &module).unwrap();
    let root = &target.root;
    assert_eq!(
        root.attributes.get("gpu.container_module"),
        Some(&pliron_mlir_export::MlirAttribute::Unit)
    );
    assert_eq!(root.regions[0].blocks[0].operations.len(), 1);
    let device_module = &root.regions[0].blocks[0].operations[0];
    assert_eq!(device_module.name, "gpu.module");
    assert_eq!(
        device_module.properties.get("sym_name"),
        Some(&pliron_mlir_export::MlirAttribute::String("kernels".into()))
    );
    assert_eq!(device_module.regions[0].blocks[0].operations.len(), 2);

    let kernel = &device_module.regions[0].blocks[0].operations[0];
    assert_eq!(
        kernel.attributes.get("cute.kernel"),
        Some(&pliron_mlir_export::MlirAttribute::Unit)
    );
    assert_eq!(
        kernel.attributes.get("gpu.kernel"),
        Some(&pliron_mlir_export::MlirAttribute::Unit)
    );
    assert_eq!(
        kernel.attributes.get("nvvm.reqntid"),
        Some(&pliron_mlir_export::MlirAttribute::DenseI32Array(vec![
            256, 1, 1
        ]))
    );
    let pliron_mlir_export::MlirAttribute::Type(pliron_mlir_export::MlirType::Function {
        inputs,
        ..
    }) = kernel.properties.get("function_type").unwrap()
    else {
        panic!("kernel function_type is not a function type")
    };
    let pointer = pliron_mlir_export::MlirType::dialect("!llvm.ptr").unwrap();
    let length = pliron_mlir_export::MlirType::Integer(64);
    let expected_inputs = vec![
        pointer.clone(),
        length.clone(),
        pointer.clone(),
        length.clone(),
        pointer,
        length,
    ];
    assert_eq!(inputs, &expected_inputs);
    let entry = &kernel.regions[0].blocks[0];
    assert_eq!(
        entry
            .arguments
            .iter()
            .map(|argument| argument.ty.clone())
            .collect::<Vec<_>>(),
        expected_inputs
    );
    assert_eq!(
        entry
            .operations
            .iter()
            .filter(|operation| operation.name == "llvm.mlir.undef")
            .count(),
        3
    );
    assert_eq!(
        entry
            .operations
            .iter()
            .filter(|operation| operation.name == "llvm.insertvalue")
            .count(),
        6
    );

    let helper = &device_module.regions[0].blocks[0].operations[1];
    assert_eq!(helper.name, "func.func");
    assert!(!helper.attributes.contains_key("gpu.kernel"));

    let text = render_module(&target);
    assert_eq!(text.matches("\"gpu.module\"").count(), 1, "{text}");
    assert!(text.contains("gpu.container_module = unit"), "{text}");
    assert!(
        text.contains("nvvm.reqntid = array<i32: 256, 1, 1>"),
        "{text}"
    );
    assert!(
        text.contains("nvvm.cluster_dim = array<i32: 2, 1, 1>"),
        "{text}"
    );
    assert!(!text.contains("cluster_dim_x"), "{text}");
}
