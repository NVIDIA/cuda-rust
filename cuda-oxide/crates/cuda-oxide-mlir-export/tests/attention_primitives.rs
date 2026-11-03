/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Attention arithmetic retains exact precision selectors and lane ordering.
use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_mir::ops::{MirFuncOp, MirReturnOp};
use dialect_nvvm::ops::*;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
};

/// Attach the canonical catalog identity and append the source intrinsic to the test block.
fn mark(ctx: &mut Context, op: Ptr<Operation>, block: Ptr<BasicBlock>, marker: &str) {
    op.deref_mut(ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new(marker.into()),
    );
    op.insert_at_back(block, ctx);
    if op.deref(ctx).get_num_results() == 1 {
        let value = op.deref(ctx).get_result(0);
        let args = block.deref(ctx).arguments().collect::<Vec<_>>();
        let ptr = args
            .iter()
            .rev()
            .find(|arg| {
                arg.get_type(ctx)
                    .deref(ctx)
                    .downcast_ref::<dialect_mir::types::MirPtrType>()
                    .is_some_and(|p| p.pointee == value.get_type(ctx))
            })
            .copied()
            .expect("output pointer");
        let store = Operation::new(
            ctx,
            dialect_mir::ops::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![ptr, value],
            vec![],
            0,
        );
        dialect_mir::ops::MirStoreOp::new(store).set_volatile(ctx, true);
        store.insert_at_back(block, ctx);
    }
}

/// Check attention arithmetic uses the reviewed precise native intrinsic spellings.
#[test]
fn full_cute_attention_numeric_pack_keeps_precise_intrinsics() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let module = ModuleOp::new(&mut ctx, "numeric".try_into().unwrap());
    let f32: TypeHandle = FP32Type::get(&ctx).into();
    let i32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let pred = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let shared = dialect_mir::types::MirPtrType::get(&mut ctx, i32, true, 3).into();
    let global = dialect_mir::types::MirPtrType::get(&mut ctx, i32, false, 1).into();
    let u64 = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let bar = dialect_mir::types::MirPtrType::get(&mut ctx, u64, true, 3).into();
    let f_out = dialect_mir::types::MirPtrType::get(&mut ctx, f32, true, 1).into();
    let u_out = dialect_mir::types::MirPtrType::get(&mut ctx, i32, true, 1).into();
    let boolean = IntegerType::get(&ctx, 1, Signedness::Signless).into();
    let b_out = dialect_mir::types::MirPtrType::get(&mut ctx, boolean, true, 1).into();
    let packed_out = dialect_mir::types::MirPtrType::get(&mut ctx, u64, true, 1).into();
    let args = vec![
        f32, f32, f32, i32, i32, pred, shared, global, bar, f_out, u_out, b_out, u64, u64, u64,
        packed_out,
    ];
    let fty = FunctionType::get(&ctx, args.clone(), vec![]);
    let op = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let f = MirFuncOp::new(&mut ctx, op, TypeAttr::new(fty.into()));
    f.set_symbol_name(&mut ctx, "numeric_probe".try_into().unwrap());
    module.append_operation(&mut ctx, op, 0);
    let block = BasicBlock::new(&mut ctx, None, args);
    block.insert_at_back(op.deref(&ctx).get_region(0), &ctx);
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
    // Distinct dynamic packed words retain both FP32 lanes through native arithmetic.
    let add = AddF32x2Op::build(&mut ctx, a[12], a[13]);
    mark(&mut ctx, add, block, "v1:i0995");
    let mul = MulF32x2Op::build(&mut ctx, a[12], a[13]);
    mark(&mut ctx, mul, block, "v1:i0999");
    let fma = FmaF32x2Op::build(&mut ctx, a[12], a[13], a[14]);
    mark(&mut ctx, fma, block, "v1:i1001");
    let shuffle = ShflSyncDownF32Op::build(&mut ctx, a[3], a[0], a[4]);
    mark(&mut ctx, shuffle, block, "v1:i0056");
    let shuffle = ShflSyncBflyI32Op::build(&mut ctx, a[3], a[4], a[4]);
    mark(&mut ctx, shuffle, block, "v1:i0051");
    // Ordinary Rust selection and saturation survive the same consumer boundary.
    use dialect_mir::{
        ops::{MirCallOp, MirStoreOp},
        rust_intrinsics::{CALLEE_SATURATING_SUB, CALLEE_SELECT_UNPREDICTABLE},
    };
    let select = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![i32],
        vec![a[5], a[3], a[4]],
        vec![],
        0,
    );
    MirCallOp::new(select)
        .set_attr_callee(&ctx, StringAttr::new(CALLEE_SELECT_UNPREDICTABLE.into()));
    select.insert_at_back(block, &ctx);
    let selected = select.deref(&ctx).get_result(0);
    let sub = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![i32],
        vec![selected, a[4]],
        vec![],
        0,
    );
    MirCallOp::new(sub).set_attr_callee(&ctx, StringAttr::new(CALLEE_SATURATING_SUB.into()));
    sub.insert_at_back(block, &ctx);
    let value = sub.deref(&ctx).get_result(0);
    let store = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![a[10], value],
        vec![],
        0,
    );
    MirStoreOp::new(store).set_volatile(&mut ctx, true);
    store.insert_at_back(block, &ctx);

    for (kind, ftz, marker) in [
        (ScalarMathOperationAttr::Ex2, false, "v1:i0819"),
        (ScalarMathOperationAttr::Lg2, false, "v1:i0786"),
        (ScalarMathOperationAttr::Rcp, true, "v1:i0788"),
    ] {
        let op = ScalarMathOp::build(
            &mut ctx,
            a[0],
            ScalarMathFormatAttr::F32,
            kind,
            ScalarMathPrecisionAttr::Approx,
            if ftz {
                ScalarMathSubnormalAttr::Ftz
            } else {
                ScalarMathSubnormalAttr::Preserve
            },
        );
        mark(&mut ctx, op, block, marker);
    }
    let op = ScalarArithmeticOp::build(
        &mut ctx,
        a[..3].to_vec(),
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Fma,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    mark(&mut ctx, op, block, "v1:i0418");
    let op = CvtF16x2F32Op::build(&mut ctx, a[0], a[1]);
    mark(&mut ctx, op, block, "v1:i0081");
    let op = ReduxSyncFmaxOp::build(&mut ctx, a[3], a[0]);
    mark(&mut ctx, op, block, "v1:i1007");
    let op = SetMaxNRegIncOp::build(&mut ctx, 192);
    mark(&mut ctx, op, block, "v1:i0915");
    let op = CpAsyncCa4Op::build(&mut ctx, a[6], a[7]);
    mark(&mut ctx, op, block, "v1:i0086");
    let op = CpAsyncMbarrierArriveNoIncOp::build(&mut ctx, a[8]);
    mark(&mut ctx, op, block, "v1:i0103");
    let op = VoteSyncAllOp::build(&mut ctx, a[3], a[5]);
    mark(&mut ctx, op, block, "v1:i0040");
    let op = BarWarpSyncOp::build(&mut ctx, a[3]);
    mark(&mut ctx, op, block, "v1:i0049");
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(block, &ctx);
    let result = CutlassFullCuteMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&ctx, &module)
        .unwrap();
    let text = cuda_oxide_mlir_export::render_module(&result).unwrap();
    for name in [
        "llvm.nvvm.ex2.approx.f32",
        "llvm.nvvm.lg2.approx.f",
        "llvm.nvvm.rcp.approx.ftz.f",
        "llvm.nvvm.fma.rn.f",
        "nvvm.redux.sync",
        "nvvm.cp.async.shared.global",
        "nvvm.cp.async.mbarrier.arrive",
        "nvvm.vote.sync",
        "nvvm.bar.warp.sync",
        "llvm.nvvm.ff2f16x2.rn",
        "nvvm.add.packed.f32x2",
        "nvvm.mul.packed.f32x2",
        "nvvm.fma.packed.f32x2",
    ] {
        assert!(text.contains(name), "{text}");
    }
    assert!(text.contains("192 : i32"), "{text}");
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}
