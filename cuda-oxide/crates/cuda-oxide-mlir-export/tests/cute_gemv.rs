/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_cute::{
    attributes::CuteTensorRoleAttr,
    gemv_ops::{
        CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
        CuteScaledViewRowOp, CuteTensorMake2DOp,
    },
};
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirPtrType, address_space},
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
    identifier::Identifier,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};
use pliron_mlir_export::{MlirModule, render_module};

fn append<O: Op>(block: Ptr<BasicBlock>, operation: O, ctx: &Context) -> Ptr<Operation> {
    let operation = operation.get_operation();
    operation.insert_at_back(block, ctx);
    operation
}

fn build_gemv_module(ctx: &mut Context) -> ModuleOp {
    dialect_mir::register(ctx);
    dialect_cute::register(ctx);

    let module = ModuleOp::new(ctx, Identifier::try_from("nvfp4_gemv").unwrap());
    let u8_type: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let u64_type: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let f32_type: TypeHandle = FP32Type::get(ctx).into();
    let pointer_type: TypeHandle =
        MirPtrType::get(ctx, u8_type, false, address_space::GENERIC).into();
    let argument_types = vec![
        pointer_type,
        u64_type,
        u64_type,
        u64_type,
        pointer_type,
        u64_type,
        pointer_type,
        u64_type,
        u64_type,
        u64_type,
        pointer_type,
        u64_type,
        u64_type,
        u64_type,
        u64_type,
        f32_type,
    ];
    let function_type = FunctionType::get(ctx, argument_types.clone(), vec![f32_type]);
    let function_operation = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let function = MirFuncOp::new(ctx, function_operation, TypeAttr::new(function_type.into()));
    function.set_symbol_name(ctx, Identifier::try_from("nvfp4_gemv_step").unwrap());
    module.append_operation(ctx, function_operation, 0);
    let region = function_operation.deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, argument_types);
    entry.insert_at_back(region, ctx);
    let arguments = entry.deref(ctx).arguments().collect::<Vec<_>>();

    let a_values = append(
        entry,
        CuteTensorMake2DOp::new_e2m1(
            ctx,
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            CuteTensorRoleAttr::Mkl,
            1,
        ),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let a_scales = append(
        entry,
        CuteTensorMake2DOp::new_ue8m0(
            ctx,
            arguments[4],
            arguments[5],
            arguments[2],
            arguments[3],
            CuteTensorRoleAttr::Mkl,
            1,
            16,
        ),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let a = append(
        entry,
        CuteScaledViewMakeOp::new(ctx, a_values, a_scales),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let a_row = append(
        entry,
        CuteScaledViewRowOp::new(ctx, a, arguments[14], arguments[2]),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let a_tile = append(
        entry,
        CuteScaledViewKTileOp::new(ctx, a_row, arguments[14]),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let a_fragment = append(entry, CuteScaledViewLoadOp::new(ctx, a_tile, 16, 4), ctx)
        .deref(ctx)
        .get_result(0);

    let b_values = append(
        entry,
        CuteTensorMake2DOp::new_e2m1(
            ctx,
            arguments[6],
            arguments[7],
            arguments[8],
            arguments[9],
            CuteTensorRoleAttr::Nkl,
            1,
        ),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let b_scales = append(
        entry,
        CuteTensorMake2DOp::new_ue8m0(
            ctx,
            arguments[10],
            arguments[11],
            arguments[8],
            arguments[9],
            CuteTensorRoleAttr::Nkl,
            1,
            16,
        ),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let b = append(
        entry,
        CuteScaledViewMakeOp::new(ctx, b_values, b_scales),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let b_row = append(
        entry,
        CuteScaledViewRowOp::new(ctx, b, arguments[14], arguments[8]),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let b_tile = append(
        entry,
        CuteScaledViewKTileOp::new(ctx, b_row, arguments[14]),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    let b_fragment = append(entry, CuteScaledViewLoadOp::new(ctx, b_tile, 16, 4), ctx)
        .deref(ctx)
        .get_result(0);

    let dot = append(
        entry,
        CuteDotOp::new(ctx, a_fragment, b_fragment, arguments[15]),
        ctx,
    )
    .deref(ctx)
    .get_result(0);
    append(
        entry,
        MirReturnOp::new(Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![dot],
            vec![],
            0,
        )),
        ctx,
    );
    module
}

fn translate_gemv_target(ctx: &Context, module: &ModuleOp) -> MlirModule {
    CutlassFullCuteMlir22::new("sm_120a")
        .unwrap()
        .translate_module(ctx, module)
        .unwrap()
}

fn translate_gemv(ctx: &Context, module: &ModuleOp) -> String {
    render_module(&translate_gemv_target(ctx, module))
}

#[test]
fn production_profile_maps_gemv_to_exact_narrow_types_and_ordered_dot() {
    let mut ctx = Context::new();
    let module = build_gemv_module(&mut ctx);
    let text = translate_gemv(&ctx, &module);

    assert!(
        text.contains("!cute.ptr<f4E2M1FN, gmem, align<16>>"),
        "{text}"
    );
    assert!(
        text.contains("!cute.ptr<f8E8M0FNU, gmem, align<4>>"),
        "{text}"
    );
    assert_eq!(
        text.matches("\"cute.memref.load_vec\"").count(),
        6,
        "{text}"
    );
    assert_eq!(text.matches("\"nvgpu.cvt_fpext\"").count(), 6, "{text}");
    assert_eq!(text.matches("vector<32xf4E2M1FN>").count(), 8, "{text}");
    assert_eq!(text.matches("vector<4xf8E8M0FNU>").count(), 4, "{text}");
    assert_eq!(text.matches("\"arith.mulf\"").count(), 192, "{text}");
    assert_eq!(text.matches("\"arith.addf\"").count(), 64, "{text}");
    assert_eq!(text.matches("\"llvm.addrspacecast\"").count(), 6, "{text}");
    assert_eq!(text.matches("!cute.i64<divby 16>").count(), 8, "{text}");
    assert_eq!(text.matches("!cute.i64<divby 4>").count(), 4, "{text}");
    assert!(!text.contains("cute.scaled_view_"), "{text}");
    assert!(!text.contains("cute.tensor_make_2d"), "{text}");
    assert_eq!(text.matches("\"gpu.module\"").count(), 1, "{text}");
}
