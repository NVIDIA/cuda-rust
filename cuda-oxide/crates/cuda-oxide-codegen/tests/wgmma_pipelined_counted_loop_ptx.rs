/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end probes for production BF16 WGMMA counted pipelines.
//!
//! The tests start from the canonical pointer-form MIR shape recognized by
//! `mir-lower` and cover both supported production depths: two distinct
//! `[[f32; 8]; 4]` accumulator slots with `wait_group<1>` and three slots with
//! `wait_group<2>`. Each slot owns one affine descriptor pair. The fusion pass
//! must replace the counted CFG with one value-form carrier, which then lowers
//! to one convergent inline-PTX counted pipeline and survives
//! `ptxas -arch=sm_90a` without stack or spill traffic.

#![cfg(unix)]

use cuda_oxide_codegen::experimental::{CodegenModule, CompileOptions, Compiler, Target};
use dialect_mir::{
    ops::{
        MirAddOp, MirCondBranchOp, MirConstantOp, MirFuncOp, MirGotoOp, MirLtOp, MirNotOp,
        MirReturnOp,
    },
    types::{MirArrayType, MirPtrType},
};
use dialect_nvvm::ops::{
    WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaMmaM64N64K16F32Bf16Op,
    WgmmaWaitGroupSyncAlignedOp,
};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OperandSegmentInterface, SymbolOpInterface},
        types::{FP32Type, FunctionType},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZeroUsize;

const ACCUMULATOR_LEN: u32 = 32;
const RESULT_COUNT: u32 = 64;
const THREE_SLOT_RESULT_COUNT: u32 = 96;
const TRIP_COUNT: u64 = 4;

fn append_unsigned_constant(
    ctx: &mut Context,
    block: pliron::context::Ptr<BasicBlock>,
    ty: pliron::r#type::TypedHandle<IntegerType>,
    value: u64,
) -> Value {
    let width = usize::try_from(ty.deref(ctx).width()).expect("integer width must fit usize");
    let constant = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(constant).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(
                value,
                NonZeroUsize::new(width).expect("nonzero integer width"),
            ),
        ),
    );
    constant.insert_at_back(block, ctx);
    constant.deref(ctx).get_result(0)
}

fn append_wait_group(
    ctx: &mut Context,
    block: pliron::context::Ptr<BasicBlock>,
    u64_ty: pliron::r#type::TypedHandle<IntegerType>,
    pending: u64,
) {
    let value = append_unsigned_constant(ctx, block, u64_ty, pending);
    WgmmaWaitGroupSyncAlignedOp::build(ctx, value).insert_at_back(block, ctx);
}

fn append_mma(
    ctx: &mut Context,
    block: pliron::context::Ptr<BasicBlock>,
    accumulator: Value,
    desc_a: Value,
    desc_b: Value,
) {
    Operation::new(
        ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![accumulator, desc_a, desc_b],
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}

fn append_u64_add(
    ctx: &mut Context,
    block: pliron::context::Ptr<BasicBlock>,
    u64_ty: pliron::r#type::TypedHandle<IntegerType>,
    value: Value,
    step: u64,
) -> Value {
    let step = append_unsigned_constant(ctx, block, u64_ty, step);
    let add = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![value, step],
        vec![],
        0,
    );
    add.insert_at_back(block, ctx);
    add.deref(ctx).get_result(0)
}

fn build_wgmma_pipelined_counted_loop_kernel(
    module: &mut CodegenModule,
    kernel_name: &str,
    slot_count: usize,
    max_pending_groups: u64,
) {
    module.edit(|ctx, module| {
        let module_region = module.get_operation().deref(ctx).get_region(0);
        let module_block = module_region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("codegen module must contain its top-level block");

        let f32_ty = FP32Type::get(ctx);
        let row_ty = MirArrayType::get(ctx, f32_ty.into(), 8);
        let accumulator_ty = MirArrayType::get(ctx, row_ty.into(), 4);
        let accumulator_ptr_ty = MirPtrType::get_generic(ctx, accumulator_ty.into(), true);
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let u64_type: pliron::r#type::TypeHandle = u64_ty.into();

        let mut argument_types: Vec<pliron::r#type::TypeHandle> =
            vec![accumulator_ptr_ty.into(); slot_count];
        argument_types.extend(vec![u64_type; slot_count * 2]);
        let function_type = FunctionType::get(ctx, argument_types.clone(), vec![]);
        let function_op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, function_op, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, kernel_name.try_into().unwrap());

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let preheader = BasicBlock::new(ctx, None, argument_types);
        preheader.insert_at_back(function_region, ctx);

        let mut header_types: Vec<pliron::r#type::TypeHandle> = vec![u32_ty.into()];
        header_types.extend(vec![u64_type; slot_count * 2]);
        let header = BasicBlock::new(ctx, None, header_types);
        header.insert_at_back(function_region, ctx);
        let latch = BasicBlock::new(ctx, None, vec![]);
        latch.insert_at_back(function_region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(function_region, ctx);

        let accumulators = (0..slot_count)
            .map(|slot| preheader.deref(ctx).get_argument(slot))
            .collect::<Vec<_>>();
        let desc_bases = (0..slot_count * 2)
            .map(|index| preheader.deref(ctx).get_argument(slot_count + index))
            .collect::<Vec<_>>();

        WgmmaFenceSyncAlignedOp::build(ctx).insert_at_back(preheader, ctx);
        let i0 = append_unsigned_constant(ctx, preheader, u32_ty, 0);
        let mut initial_values = vec![i0];
        initial_values.extend(desc_bases.iter().copied());
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            initial_values,
            vec![header],
            0,
        )
        .insert_at_back(preheader, ctx);

        let i = header.deref(ctx).get_argument(0);
        let descriptors = (0..slot_count * 2)
            .map(|index| header.deref(ctx).get_argument(1 + index))
            .collect::<Vec<_>>();
        let bound = append_unsigned_constant(ctx, header, u32_ty, TRIP_COUNT);
        let lt = Operation::new(
            ctx,
            MirLtOp::get_concrete_op_info(),
            vec![i1_ty.into()],
            vec![i, bound],
            vec![],
            0,
        );
        lt.insert_at_back(header, ctx);
        let lt_value = lt.deref(ctx).get_result(0);
        let not_lt = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1_ty.into()],
            vec![lt_value],
            vec![],
            0,
        );
        not_lt.insert_at_back(header, ctx);
        let not_lt_value = not_lt.deref(ctx).get_result(0);
        let (branch_operands, segment_sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![not_lt_value], vec![], vec![]]);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            branch_operands,
            vec![exit, latch],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(branch, ctx)
            .expect("MirCondBranchOp")
            .set_operand_segment_sizes(ctx, segment_sizes);
        branch.insert_at_back(header, ctx);

        for slot in 0..slot_count {
            append_mma(
                ctx,
                latch,
                accumulators[slot],
                descriptors[slot * 2],
                descriptors[slot * 2 + 1],
            );
            WgmmaCommitGroupSyncAlignedOp::build(ctx).insert_at_back(latch, ctx);
            append_wait_group(ctx, latch, u64_ty, max_pending_groups);
        }

        let one = append_unsigned_constant(ctx, latch, u32_ty, 1);
        let i_next = Operation::new(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![u32_ty.into()],
            vec![i, one],
            vec![],
            0,
        );
        i_next.insert_at_back(latch, ctx);
        let i_next = i_next.deref(ctx).get_result(0);

        let mut next_values = vec![i_next];
        for (index, descriptor) in descriptors.iter().copied().enumerate() {
            next_values.push(append_u64_add(
                ctx,
                latch,
                u64_ty,
                descriptor,
                16 * (index as u64 + 1),
            ));
        }

        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            next_values,
            vec![header],
            0,
        )
        .insert_at_back(latch, ctx);

        append_wait_group(ctx, exit, u64_ty, 0);
        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(exit, ctx);

        function.get_operation().insert_at_back(module_block, ctx);
    });

    module
        .mark_kernel_entry(kernel_name)
        .expect("kernel entry must be marked");
}

fn find_ptxas() -> std::path::PathBuf {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|variable| std::env::var(variable).ok())
        .filter(|root| !root.trim().is_empty())
        .map(|root| std::path::PathBuf::from(root).join("bin/ptxas"))
        .chain(std::iter::once(std::path::PathBuf::from(
            "/usr/local/cuda/bin/ptxas",
        )))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("ptxas"))
}

fn used_register_count(ptxas_stderr: &str) -> Option<u32> {
    ptxas_stderr.lines().find_map(|line| {
        let start = line.find("Used ")? + "Used ".len();
        let remainder = &line[start..];
        let end = remainder.find(" registers")?;
        remainder[..end].trim().parse().ok()
    })
}

#[test]
fn pointer_form_two_slot_counted_pipeline_compiles_spill_free_for_sm_90a() {
    let mut module = CodegenModule::new("wgmma_pipelined_counted_loop").unwrap();
    build_wgmma_pipelined_counted_loop_kernel(
        &mut module,
        "wgmma_pipelined_counted_loop_kernel",
        2,
        1,
    );

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());
    let ptx = compiler
        .compile(&mut module, &options)
        .expect("pointer-form two-slot counted WGMMA pipeline must compile to PTX")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX must be UTF-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry is missing:\n{text}"
    );
    assert!(
        text.contains(".target sm_90a"),
        "PTX must target sm_90a:\n{text}"
    );
    assert!(
        text.contains("wgmma_pipelined_counted_loop_kernel"),
        "counted-pipeline kernel is missing:\n{text}"
    );
    assert!(
        text.contains("L__wgmma_pipeline_loop_") && text.contains("L__wgmma_pipeline_done_"),
        "production counted-pipeline labels are missing; pointer-form MIR did not fuse:\n{text}"
    );
    assert!(
        !text.contains("${:uid}"),
        "llc must expand every ${{:uid}} escape into a unique label suffix:\n{text}"
    );
    assert_eq!(text.matches("bra.uni").count(), 2);
    assert_eq!(text.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        text.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
            .count(),
        2
    );
    assert_eq!(text.matches("wgmma.commit_group.sync.aligned").count(), 2);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 1").count(), 2);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 0").count(), 1);

    for recurrence in [
        "add.u64 %desc_a0",
        "add.u64 %desc_b0",
        "add.u64 %desc_a1",
        "add.u64 %desc_b1",
    ] {
        assert!(
            text.contains(recurrence),
            "descriptor recurrence `{recurrence}` is missing from the PTX loop:\n{text}"
        );
    }
    assert!(
        !text.contains(".local") && !text.contains("ld.local") && !text.contains("st.local"),
        "the production counted pipeline unexpectedly materializes local memory:\n{text}"
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let directory = std::env::temp_dir().join(format!(
        "wgmma_pipelined_counted_loop_ptx_{}_{}",
        std::process::id(),
        unique,
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let ptx_path = directory.join("pipelined_counted_loop.ptx");
    let cubin_path = directory.join("pipelined_counted_loop.cubin");
    std::fs::write(&ptx_path, &ptx).unwrap();

    let ptxas = find_ptxas();
    let ptxas_result = std::process::Command::new(&ptxas)
        .arg("-arch=sm_90a")
        .arg("--compile-only")
        .arg("-v")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();
    let _ = std::fs::remove_dir_all(&directory);

    let output = ptxas_result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nSet CUDA_TOOLKIT_PATH or CUDA_HOME, or put ptxas on PATH.",
            ptxas.display()
        )
    });
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    assert!(
        output.status.success(),
        "ptxas rejected the production two-slot counted WGMMA pipeline:\n{stderr}\n\nPTX:\n{text}"
    );
    assert!(
        stderr.contains("0 bytes stack frame"),
        "ptxas reported a nonzero stack frame or omitted the expected report:\n{stderr}"
    );
    assert!(
        stderr.contains("0 bytes spill stores"),
        "ptxas reported spill stores:\n{stderr}"
    );
    assert!(
        stderr.contains("0 bytes spill loads"),
        "ptxas reported spill loads:\n{stderr}"
    );

    let used_registers =
        used_register_count(&stderr).expect("ptxas output must report register usage");
    assert!(
        used_registers >= RESULT_COUNT,
        "the production path did not keep both accumulator slots live: ptxas used only {used_registers} registers\n{stderr}"
    );
    assert!(used_registers >= ACCUMULATOR_LEN);

    eprintln!("{stderr}");
}

#[test]
fn pointer_form_three_slot_counted_pipeline_compiles_spill_free_for_sm_90a() {
    let mut module = CodegenModule::new("wgmma_three_slot_pipelined_counted_loop").unwrap();
    build_wgmma_pipelined_counted_loop_kernel(
        &mut module,
        "wgmma_three_slot_pipelined_counted_loop_kernel",
        3,
        2,
    );

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());
    let ptx = compiler
        .compile(&mut module, &options)
        .expect("pointer-form three-slot counted WGMMA pipeline must compile to PTX")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX must be UTF-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry is missing:\n{text}"
    );
    assert!(
        text.contains(".target sm_90a"),
        "PTX must target sm_90a:\n{text}"
    );
    assert!(
        text.contains("wgmma_three_slot_pipelined_counted_loop_kernel"),
        "three-slot counted-pipeline kernel is missing:\n{text}"
    );
    assert!(
        text.contains("L__wgmma_pipeline_loop_") && text.contains("L__wgmma_pipeline_done_"),
        "production three-slot counted-pipeline labels are missing; pointer-form MIR did not fuse:\n{text}"
    );
    assert!(
        !text.contains("${:uid}"),
        "llc must expand every ${{:uid}} escape into a unique label suffix:\n{text}"
    );
    assert_eq!(text.matches("bra.uni").count(), 2);
    assert_eq!(text.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        text.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
            .count(),
        3
    );
    assert_eq!(text.matches("wgmma.commit_group.sync.aligned").count(), 3);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 2").count(), 3);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 0").count(), 1);

    for recurrence in [
        "add.u64 %desc_a0",
        "add.u64 %desc_b0",
        "add.u64 %desc_a1",
        "add.u64 %desc_b1",
        "add.u64 %desc_a2",
        "add.u64 %desc_b2",
    ] {
        assert!(
            text.contains(recurrence),
            "descriptor recurrence `{recurrence}` is missing from the three-slot PTX loop:\n{text}"
        );
    }
    assert!(
        !text.contains(".local") && !text.contains("ld.local") && !text.contains("st.local"),
        "the production three-slot counted pipeline unexpectedly materializes local memory:\n{text}"
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let directory = std::env::temp_dir().join(format!(
        "wgmma_three_slot_pipelined_counted_loop_ptx_{}_{}",
        std::process::id(),
        unique,
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let ptx_path = directory.join("three_slot_pipelined_counted_loop.ptx");
    let cubin_path = directory.join("three_slot_pipelined_counted_loop.cubin");
    std::fs::write(&ptx_path, &ptx).unwrap();

    let ptxas = find_ptxas();
    let ptxas_result = std::process::Command::new(&ptxas)
        .arg("-arch=sm_90a")
        .arg("--compile-only")
        .arg("-v")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();
    let _ = std::fs::remove_dir_all(&directory);

    let output = ptxas_result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nSet CUDA_TOOLKIT_PATH or CUDA_HOME, or put ptxas on PATH.",
            ptxas.display()
        )
    });
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    assert!(
        output.status.success(),
        "ptxas rejected the production three-slot counted WGMMA pipeline:\n{stderr}\n\nPTX:\n{text}"
    );
    assert!(
        stderr.contains("0 bytes stack frame"),
        "ptxas reported a nonzero stack frame or omitted the expected report:\n{stderr}"
    );
    assert!(
        stderr.contains("0 bytes spill stores"),
        "ptxas reported spill stores:\n{stderr}"
    );
    assert!(
        stderr.contains("0 bytes spill loads"),
        "ptxas reported spill loads:\n{stderr}"
    );

    let used_registers =
        used_register_count(&stderr).expect("ptxas output must report register usage");
    assert!(
        used_registers >= THREE_SLOT_RESULT_COUNT,
        "the production path did not keep all three accumulator slots live: ptxas used only {used_registers} registers\n{stderr}"
    );
    assert!(used_registers >= ACCUMULATOR_LEN);

    eprintln!("{stderr}");
}
