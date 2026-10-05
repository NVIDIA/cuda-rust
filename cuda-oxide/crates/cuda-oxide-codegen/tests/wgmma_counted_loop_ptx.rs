/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proves that the fused counted K-loop WGMMA template survives the real
//! toolchain, not just the LLVM dialect.
//!
//! The counted-loop template is the first inline-PTX region in this codebase
//! that contains local labels, `${:uid}` operand escapes, and `bra.uni`
//! branches. This probe builds the canonical pointer-form K-loop (trip count
//! 4, one MMA in the latch, affine descriptor recurrences), lets the deferred
//! accumulator fusion rewrite it into the counted-loop asm scope, and then
//! requires `llc` and `ptxas -arch=sm_90a` to accept both BF16 and F16 forms
//! with zero spill bytes while all 32 accumulator values stay live.

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
    WgmmaMmaM64N64K16F32F16Op, WgmmaWaitGroupSyncAlignedOp,
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
const TRIP_COUNT: u64 = 4;
const DESC_A_STEP: u64 = 16;
const DESC_B_STEP: u64 = 32;

#[derive(Clone, Copy, Debug)]
enum WgmmaInputKind {
    Bf16,
    F16,
}

impl WgmmaInputKind {
    fn ptx_suffix(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16.bf16",
            Self::F16 => "f16.f16",
        }
    }
}

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

/// Build the canonical counted K-loop that the deferred accumulator fusion
/// recognizes:
///
/// ```text
/// preheader: fence; goto header(0, desc_a_base, desc_b_base)
/// header(i, desc_a, desc_b): if !(i < 4) exit else latch
/// latch: mma(acc, desc_a, desc_b); goto header(i + 1, desc_a + 16, desc_b + 32)
/// exit: commit_group; wait_group<0>; return
/// ```
fn build_wgmma_counted_loop_kernel(module: &mut CodegenModule, input_kind: WgmmaInputKind) {
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

        // Kernel signature:
        //
        // wgmma_counted_loop_kernel(
        //     accumulator: *mut [[f32; 8]; 4],
        //     desc_a_base: u64,
        //     desc_b_base: u64,
        // )
        let argument_types: Vec<pliron::r#type::TypeHandle> =
            vec![accumulator_ptr_ty.into(), u64_ty.into(), u64_ty.into()];

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
        function.set_symbol_name(ctx, "wgmma_counted_loop_kernel".try_into().unwrap());

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let preheader = BasicBlock::new(ctx, None, argument_types);
        preheader.insert_at_back(function_region, ctx);
        let header = BasicBlock::new(ctx, None, vec![u32_ty.into(), u64_ty.into(), u64_ty.into()]);
        header.insert_at_back(function_region, ctx);
        let latch = BasicBlock::new(ctx, None, vec![]);
        latch.insert_at_back(function_region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(function_region, ctx);

        let accumulator = preheader.deref(ctx).get_argument(0);
        let desc_a_base = preheader.deref(ctx).get_argument(1);
        let desc_b_base = preheader.deref(ctx).get_argument(2);

        // preheader: fence; i0 = 0; goto header(i0, desc_a_base, desc_b_base)
        WgmmaFenceSyncAlignedOp::build(ctx).insert_at_back(preheader, ctx);
        let i0 = append_unsigned_constant(ctx, preheader, u32_ty, 0);
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![i0, desc_a_base, desc_b_base],
            vec![header],
            0,
        )
        .insert_at_back(preheader, ctx);

        // header(i, desc_a, desc_b): if !(i < TRIP_COUNT) exit else latch.
        let i = header.deref(ctx).get_argument(0);
        let desc_a = header.deref(ctx).get_argument(1);
        let desc_b = header.deref(ctx).get_argument(2);
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

        // latch: one MMA per K iteration and affine descriptor recurrences.
        let mma_op_info = match input_kind {
            WgmmaInputKind::Bf16 => WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
            WgmmaInputKind::F16 => WgmmaMmaM64N64K16F32F16Op::get_concrete_op_info(),
        };
        Operation::new(
            ctx,
            mma_op_info,
            vec![],
            vec![accumulator, desc_a, desc_b],
            vec![],
            0,
        )
        .insert_at_back(latch, ctx);

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

        let desc_a_step = append_unsigned_constant(ctx, latch, u64_ty, DESC_A_STEP);
        let desc_a_next = Operation::new(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![u64_ty.into()],
            vec![desc_a, desc_a_step],
            vec![],
            0,
        );
        desc_a_next.insert_at_back(latch, ctx);
        let desc_a_next = desc_a_next.deref(ctx).get_result(0);

        let desc_b_step = append_unsigned_constant(ctx, latch, u64_ty, DESC_B_STEP);
        let desc_b_next = Operation::new(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![u64_ty.into()],
            vec![desc_b, desc_b_step],
            vec![],
            0,
        );
        desc_b_next.insert_at_back(latch, ctx);
        let desc_b_next = desc_b_next.deref(ctx).get_result(0);

        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![i_next, desc_a_next, desc_b_next],
            vec![header],
            0,
        )
        .insert_at_back(latch, ctx);

        // exit: the only place where the asynchronous lifetime becomes visible.
        WgmmaCommitGroupSyncAlignedOp::build(ctx).insert_at_back(exit, ctx);
        let pending = append_unsigned_constant(ctx, exit, u64_ty, 0);
        WgmmaWaitGroupSyncAlignedOp::build(ctx, pending).insert_at_back(exit, ctx);
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
        .mark_kernel_entry("wgmma_counted_loop_kernel")
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

fn assert_counted_k_loop_wgmma_template_compiles_spill_free_for_sm_90a(input_kind: WgmmaInputKind) {
    let mut module = CodegenModule::new("wgmma_counted_loop").unwrap();
    build_wgmma_counted_loop_kernel(&mut module, input_kind);

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("counted K-loop WGMMA kernel must compile to PTX")
        .into_ptx();

    let text = String::from_utf8(ptx.clone()).expect("PTX must be UTF-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry is missing:\n{text}",
    );
    assert!(
        text.contains(".target sm_90a"),
        "PTX must target sm_90a:\n{text}",
    );
    assert!(
        text.contains("wgmma_counted_loop_kernel"),
        "counted-loop kernel is missing:\n{text}",
    );

    // The counted-loop fusion must have produced its template. The deferred
    // pointer-form fallback has no labels and no branches, so requiring them
    // here keeps this probe from passing vacuously.
    assert!(
        text.contains("L__wgmma_loop_") && text.contains("L__wgmma_done_"),
        "the counted-loop labels are missing; the K-loop did not fuse:\n{text}",
    );
    assert!(
        !text.contains("${:uid}"),
        "llc must expand every ${{:uid}} escape into a unique label suffix:\n{text}",
    );
    assert_eq!(
        text.matches("bra.uni").count(),
        2,
        "the counted loop must keep exactly its guard and back-edge branches:\n{text}",
    );

    assert_eq!(
        text.matches("wgmma.fence.sync.aligned").count(),
        1,
        "the counted loop must contain exactly one fence:\n{text}",
    );
    let expected_mma = format!(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.{}",
        input_kind.ptx_suffix()
    );
    assert_eq!(
        text.matches(&expected_mma).count(),
        1,
        "the loop body must contain exactly one {input_kind:?} MMA:\n{text}",
    );
    assert_eq!(
        text.matches("wgmma.commit_group.sync.aligned").count(),
        1,
        "the counted loop must contain exactly one commit:\n{text}",
    );
    assert_eq!(
        text.matches("wgmma.wait_group.sync.aligned 0").count(),
        1,
        "the counted loop must fully drain the group:\n{text}",
    );
    assert!(
        !text.contains(".local") && !text.contains("ld.local") && !text.contains("st.local"),
        "the counted loop unexpectedly materializes local memory:\n{text}",
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);

    let directory = std::env::temp_dir().join(format!(
        "wgmma_counted_loop_ptx_{input_kind:?}_{}_{}",
        std::process::id(),
        unique,
    ));

    std::fs::create_dir_all(&directory).unwrap();

    let ptx_path = directory.join("counted_loop.ptx");
    let cubin_path = directory.join("counted_loop.cubin");
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
            "could not run {}: {error}\n\
             Set CUDA_TOOLKIT_PATH or CUDA_HOME, or put ptxas on PATH.",
            ptxas.display(),
        )
    });

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    assert!(
        output.status.success(),
        "ptxas rejected the counted K-loop template:\n{stderr}\n\nPTX:\n{text}",
    );
    assert!(
        stderr.contains("0 bytes spill stores"),
        "ptxas reported spill stores:\n{stderr}",
    );
    assert!(
        stderr.contains("0 bytes spill loads"),
        "ptxas reported spill loads:\n{stderr}",
    );

    let used_registers =
        used_register_count(&stderr).expect("ptxas output must report register usage");

    assert!(
        used_registers >= ACCUMULATOR_LEN,
        "the probe did not keep all 32 accumulator values live across the loop: \
         ptxas used only {used_registers} registers\n{stderr}",
    );

    eprintln!("{stderr}");
}

#[test]
fn bf16_counted_k_loop_wgmma_template_compiles_spill_free_for_sm_90a() {
    assert_counted_k_loop_wgmma_template_compiles_spill_free_for_sm_90a(WgmmaInputKind::Bf16);
}

#[test]
fn f16_counted_k_loop_wgmma_template_compiles_spill_free_for_sm_90a() {
    assert_counted_k_loop_wgmma_template_compiles_spill_free_for_sm_90a(WgmmaInputKind::F16);
}
