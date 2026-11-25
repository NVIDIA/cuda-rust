/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Standalone SM100 attention MMA and TMA source-plan integration.

use cuda_oxide_mlir_export::{CutlassFullCuteMlir22, MlirConsumerProfile};
use dialect_cute::sm100_attention_ops::*;
use dialect_mir::{
    ops::{MirFuncOp, MirReturnOp},
    types::{MirFP16Type, MirPtrType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
};

/// Check standalone attention MMA/TMA plans lower without requiring the paired-GEMM lifecycle.
#[test]
fn independent_attention_mma_and_rank_three_copy_lower_without_gemm_story() {
    for group in [1, 2] {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        dialect_cute::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "attention".try_into().unwrap());
        let f16 = MirFP16Type::get(&ctx).into();
        let u32: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let b: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get(&mut ctx, f16, true, 3).into();
        let desc: TypeHandle = MirPtrType::get(&mut ctx, u64, false, 0).into();
        let barrier: TypeHandle = MirPtrType::get(&mut ctx, u64, true, 3).into();
        let args = vec![u32, shared, shared, b, u32, desc, barrier, u32, u32];
        let ty = FunctionType::get(&ctx, args.clone(), vec![]);
        let f = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, f, TypeAttr::new(ty.into()))
            .set_symbol_name(&mut ctx, "attention_probe".try_into().unwrap());
        module.append_operation(&mut ctx, f, 0);
        let block = BasicBlock::new(&mut ctx, None, args);
        block.insert_at_back(f.deref(&ctx).get_region(0), &ctx);
        let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
        let plan = CuteSm100MmaAttr {
            m: 128,
            n: 128,
            k: 128,
            cta_group: group,
            a_mn: false,
            b_mn: false,
            a_layout: shared_layout(128 / group, 128, false).unwrap(),
            b_layout: shared_layout(128 / group, 128, false).unwrap(),
        };
        CuteSm100MmaSsOp::new(
            &mut ctx,
            vec![a[0], a[1], a[2], a[3], a[7], a[8]],
            plan.clone(),
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        if group == 1 {
            let mut pv = plan.clone();
            pv.b_mn = true;
            pv.b_layout = shared_layout(128, 128, true).unwrap();
            CuteSm100MmaTsOp::new(&mut ctx, vec![a[0], a[4], a[2], a[3], a[8]], pv)
                .get_operation()
                .insert_at_back(block, &ctx);
        }
        let copy = CuteSm100TmaAttr {
            layout: plan.a_layout,
            cta_group: group,
        };
        CuteSm100TmaLoadOp::new(
            &mut ctx,
            vec![a[5], a[1], a[4], a[4], a[4], a[6], a[4]],
            copy.clone(),
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        CuteSm100TmaStoreOp::new(&mut ctx, vec![a[5], a[1], a[4], a[4], a[4]], copy)
            .get_operation()
            .insert_at_back(block, &ctx);
        Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(block, &ctx);
        let target = CutlassFullCuteMlir22::new("sm_100a")
            .unwrap()
            .translate_module(&ctx, &module)
            .unwrap();
        let text = cuda_oxide_mlir_export::render_module(&target).unwrap();
        assert_eq!(
            text.matches("\"nvvm.tcgen05.mma\"").count(),
            if group == 1 { 16 } else { 8 }
        );
        // The allocation bases construct descriptors before dynamic byte offsets
        // are converted to the descriptor's 16-byte address units.
        assert_eq!(
            text.matches("\"cute_nvgpu.make_umma_smem_desc\"").count(),
            if group == 1 { 3 } else { 2 }
        );
        assert_eq!(
            text.matches("\"arith.shrui\"").count(),
            if group == 1 { 3 } else { 2 }
        );
        assert!(!text.contains("llvm.getelementptr"), "{text}");
        for shift in text.lines().filter(|line| line.contains("\"arith.shrui\"")) {
            let (_, operands) = shift.split_once("\"arith.shrui\"(").unwrap();
            let (input, rest) = operands.split_once(", ").unwrap();
            let amount = rest.split_once(')').unwrap().0;
            assert!(
                text.lines().any(
                    |line| line.trim_start().starts_with(&format!("{amount} = "))
                        && line.contains("value = 4 : i64")
                ),
                "{text}"
            );
            assert!(
                text.lines()
                    .any(|line| line.trim_start().starts_with(&format!("{input} = "))
                        && line.contains("\"arith.extui\"")
                        && line.contains("(i32) -> (i64)")),
                "{text}"
            );
        }
        assert!(text.contains("#cute_nvgpu.major<k>"), "{text}");
        assert!(text.contains("num_cta = "), "{text}");
        assert!(
            text.contains("operandSegmentSizes = array<i32: 1, 1, 1, 3, 0, 0, 0>"),
            "{text}"
        );
        assert!(!text.contains("\"arith.shli\""), "{text}");
        if group == 2 {
            assert!(text.contains("num_cta = 2 : i32"), "{text}");
            let mapped = text
                .lines()
                .find(|line| line.contains("\"nvvm.mapa\""))
                .unwrap();
            let (_, operands) = mapped.split_once("\"nvvm.mapa\"(").unwrap();
            let rank = operands
                .split_once(", ")
                .unwrap()
                .1
                .split_once(')')
                .unwrap()
                .0;
            assert!(
                text.lines()
                    .any(|line| line.trim_start().starts_with(&format!("{rank} = "))
                        && line.contains("value = 0 : i32")),
                "{text}"
            );
        }
        if group == 1 {
            assert!(text.contains("#cute_nvgpu.major<mn>"), "{text}");
        }
    }
}
