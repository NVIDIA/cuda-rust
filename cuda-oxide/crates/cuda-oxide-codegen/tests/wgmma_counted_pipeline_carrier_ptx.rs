/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proves that BF16 WGMMA counted pipelines can keep multiple independent
//! accumulator slots spill-free while loop control and affine descriptor
//! recurrences remain inside the same convergent inline-PTX lifetime.
//!
//! This is a backend feasibility probe. It intentionally bypasses the production
//! counted-loop recognizer and combined carrier that a follow-up implementation
//! may add. The file covers both the existing two-slot baseline and the next
//! three-slot feasibility step:
//!
//! ```text
//! baseline:
//!   64 tied f32 accumulator values
//!   4 descriptor bases
//!   4 descriptor deltas
//!   wait_group<1> throttling
//!
//! three-slot probe:
//!   96 tied f32 accumulator values
//!   6 descriptor bases
//!   6 descriptor deltas
//!   wait_group<2> throttling
//!
//! both:
//!   1 trip count
//!   one PTX loop
//!   final wait_group<0>
//! ```
//!
//! `ptxas -arch=sm_90a` must accept both generated PTX kernels with zero stack
//! frame and zero spill traffic.

#![cfg(unix)]

use cuda_oxide_codegen::experimental::{CodegenModule, CompileOptions, Compiler, Target};
use dialect_mir::{
    ops::{MirConstantOp, MirFuncOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp},
    types::MirPtrType,
};
use dialect_nvvm::ops::InlinePtxOp;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::SymbolOpInterface,
        types::{FP32Type, FunctionType},
    },
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    utils::apint::APInt,
};
use std::num::NonZeroUsize;

const ACCUMULATOR_LEN: usize = 32;
const SLOT_COUNT: usize = 2;
const RESULT_COUNT: usize = ACCUMULATOR_LEN * SLOT_COUNT;
const CONTROL_COUNT: usize = 9;

const THREE_SLOT_COUNT: usize = 3;
const THREE_SLOT_RESULT_COUNT: usize = ACCUMULATOR_LEN * THREE_SLOT_COUNT;
const THREE_SLOT_CONTROL_COUNT: usize = 13;

fn accumulator_register_list(slot: usize) -> String {
    let base = slot * ACCUMULATOR_LEN;
    (base..base + ACCUMULATOR_LEN)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn counted_pipeline_template() -> String {
    // LLVM inline-asm numbering includes outputs first and then inputs.
    //
    // 0..63   : f32 outputs
    // 64..127 : tied f32 inputs
    // 128..136: u64 control inputs
    let control_base = RESULT_COUNT * 2;
    let desc_a0_base = control_base;
    let desc_b0_base = control_base + 1;
    let desc_a1_base = control_base + 2;
    let desc_b1_base = control_base + 3;
    let desc_a0_step = control_base + 4;
    let desc_b0_step = control_base + 5;
    let desc_a1_step = control_base + 6;
    let desc_b1_step = control_base + 7;
    let trip_count = control_base + 8;

    let slot0 = accumulator_register_list(0);
    let slot1 = accumulator_register_list(1);

    let mut template = String::from(
        "{\n\
         \x20   .reg .u64 %desc_a0;\n\
         \x20   .reg .u64 %desc_b0;\n\
         \x20   .reg .u64 %desc_a1;\n\
         \x20   .reg .u64 %desc_b1;\n\
         \x20   .reg .u64 %remaining;\n\
         \x20   .reg .pred %loop_more;\n",
    );

    template.push_str(&format!("    mov.u64 %desc_a0, ${desc_a0_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b0, ${desc_b0_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_a1, ${desc_a1_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b1, ${desc_b1_base};\n"));
    template.push_str(&format!("    mov.u64 %remaining, ${trip_count};\n"));

    template.push_str("    wgmma.fence.sync.aligned;\n");
    template.push_str("    setp.eq.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_counted_pipeline_done_${:uid};\n");
    template.push_str("L__wgmma_counted_pipeline_loop_${:uid}:\n");

    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
         {{{slot0}}}, %desc_a0, %desc_b0, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 1;\n");

    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
         {{{slot1}}}, %desc_a1, %desc_b1, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 1;\n");

    template.push_str(&format!(
        "    add.u64 %desc_a0, %desc_a0, ${desc_a0_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_b0, %desc_b0, ${desc_b0_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_a1, %desc_a1, ${desc_a1_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_b1, %desc_b1, ${desc_b1_step};\n"
    ));

    template.push_str("    sub.u64 %remaining, %remaining, 1;\n");
    template.push_str("    setp.ne.u64 %loop_more, %remaining, 0;\n");
    template.push_str("    @%loop_more bra.uni L__wgmma_counted_pipeline_loop_${:uid};\n");
    template.push_str("L__wgmma_counted_pipeline_done_${:uid}:\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn counted_pipeline_constraints() -> String {
    let mut constraints = vec!["=f".to_owned(); RESULT_COUNT];
    constraints.extend((0..RESULT_COUNT).map(|index| index.to_string()));
    constraints.extend((0..CONTROL_COUNT).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

fn counted_three_slot_pipeline_template() -> String {
    // LLVM inline-asm numbering includes outputs first and then inputs.
    //
    // 0..95    : f32 outputs
    // 96..191  : tied f32 inputs
    // 192..204 : u64 control inputs
    let control_base = THREE_SLOT_RESULT_COUNT * 2;
    let desc_a0_base = control_base;
    let desc_b0_base = control_base + 1;
    let desc_a1_base = control_base + 2;
    let desc_b1_base = control_base + 3;
    let desc_a2_base = control_base + 4;
    let desc_b2_base = control_base + 5;
    let desc_a0_step = control_base + 6;
    let desc_b0_step = control_base + 7;
    let desc_a1_step = control_base + 8;
    let desc_b1_step = control_base + 9;
    let desc_a2_step = control_base + 10;
    let desc_b2_step = control_base + 11;
    let trip_count = control_base + 12;

    let slot0 = accumulator_register_list(0);
    let slot1 = accumulator_register_list(1);
    let slot2 = accumulator_register_list(2);

    let mut template = String::from(
        "{\n\
         \x20   .reg .u64 %desc_a0;\n\
         \x20   .reg .u64 %desc_b0;\n\
         \x20   .reg .u64 %desc_a1;\n\
         \x20   .reg .u64 %desc_b1;\n\
         \x20   .reg .u64 %desc_a2;\n\
         \x20   .reg .u64 %desc_b2;\n\
         \x20   .reg .u64 %remaining;\n\
         \x20   .reg .pred %loop_more;\n",
    );

    template.push_str(&format!("    mov.u64 %desc_a0, ${desc_a0_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b0, ${desc_b0_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_a1, ${desc_a1_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b1, ${desc_b1_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_a2, ${desc_a2_base};\n"));
    template.push_str(&format!("    mov.u64 %desc_b2, ${desc_b2_base};\n"));
    template.push_str(&format!("    mov.u64 %remaining, ${trip_count};\n"));

    template.push_str("    wgmma.fence.sync.aligned;\n");
    template.push_str("    setp.eq.u64 %loop_more, %remaining, 0;\n");
    template
        .push_str("    @%loop_more bra.uni L__wgmma_three_slot_counted_pipeline_done_${:uid};\n");
    template.push_str("L__wgmma_three_slot_counted_pipeline_loop_${:uid}:\n");

    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
         {{{slot0}}}, %desc_a0, %desc_b0, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 2;\n");

    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
         {{{slot1}}}, %desc_a1, %desc_b1, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 2;\n");

    template.push_str(&format!(
        "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
         {{{slot2}}}, %desc_a2, %desc_b2, 1, 1, 1, 0, 0;\n"
    ));
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 2;\n");

    template.push_str(&format!(
        "    add.u64 %desc_a0, %desc_a0, ${desc_a0_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_b0, %desc_b0, ${desc_b0_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_a1, %desc_a1, ${desc_a1_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_b1, %desc_b1, ${desc_b1_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_a2, %desc_a2, ${desc_a2_step};\n"
    ));
    template.push_str(&format!(
        "    add.u64 %desc_b2, %desc_b2, ${desc_b2_step};\n"
    ));

    template.push_str("    sub.u64 %remaining, %remaining, 1;\n");
    template.push_str("    setp.ne.u64 %loop_more, %remaining, 0;\n");
    template
        .push_str("    @%loop_more bra.uni L__wgmma_three_slot_counted_pipeline_loop_${:uid};\n");
    template.push_str("L__wgmma_three_slot_counted_pipeline_done_${:uid}:\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
    template.push('}');
    template
}

fn counted_three_slot_pipeline_constraints() -> String {
    let mut constraints = vec!["=f".to_owned(); THREE_SLOT_RESULT_COUNT];
    constraints.extend((0..THREE_SLOT_RESULT_COUNT).map(|index| index.to_string()));
    constraints.extend((0..THREE_SLOT_CONTROL_COUNT).map(|_| "l".to_owned()));
    constraints.push("~{memory}".to_owned());
    constraints.join(",")
}

fn build_wgmma_counted_pipeline_carrier_kernel(module: &mut CodegenModule) {
    module.edit(|ctx, module| {
        let module_region = module.get_operation().deref(ctx).get_region(0);
        let module_block = module_region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("codegen module must contain its top-level block");

        let f32_ty = FP32Type::get(ctx);
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let output_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), true);

        let mut argument_types: Vec<pliron::r#type::TypeHandle> =
            Vec::with_capacity(1 + RESULT_COUNT + CONTROL_COUNT);
        let f32_arg_ty: pliron::r#type::TypeHandle = f32_ty.into();
        let u64_arg_ty: pliron::r#type::TypeHandle = u64_ty.into();
        argument_types.push(output_ptr_ty.into());
        argument_types.extend(vec![f32_arg_ty; RESULT_COUNT]);
        argument_types.extend(vec![u64_arg_ty; CONTROL_COUNT]);

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
        function.set_symbol_name(
            ctx,
            "wgmma_counted_pipeline_carrier_kernel".try_into().unwrap(),
        );

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(function_region, ctx);

        let output = entry.deref(ctx).get_argument(0);
        let accumulator_inputs = (0..RESULT_COUNT)
            .map(|index| entry.deref(ctx).get_argument(index + 1))
            .collect::<Vec<_>>();
        let controls = (0..CONTROL_COUNT)
            .map(|index| entry.deref(ctx).get_argument(1 + RESULT_COUNT + index))
            .collect::<Vec<_>>();

        let template = counted_pipeline_template();
        let constraints = counted_pipeline_constraints();

        let mut asm_inputs = accumulator_inputs;
        asm_inputs.extend(controls);

        let inline_ptx = InlinePtxOp::build(
            ctx,
            vec![f32_ty.into(); RESULT_COUNT],
            asm_inputs,
            &template,
            &constraints,
            true,
            true,
        );
        let results = (0..RESULT_COUNT)
            .map(|index| inline_ptx.deref(ctx).get_result(index))
            .collect::<Vec<_>>();
        inline_ptx.insert_at_back(entry, ctx);

        for (index, result) in results.into_iter().enumerate() {
            let index_op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(index_op).set_attr_value(
                ctx,
                IntegerAttr::new(
                    i32_ty,
                    APInt::from_u32(index as u32, NonZeroUsize::new(32).unwrap()),
                ),
            );
            let index_value = index_op.deref(ctx).get_result(0);
            index_op.insert_at_back(entry, ctx);

            let output_element_op = Operation::new(
                ctx,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![output_ptr_ty.into()],
                vec![output, index_value],
                vec![],
                0,
            );
            let output_element = output_element_op.deref(ctx).get_result(0);
            output_element_op.insert_at_back(entry, ctx);

            Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![output_element, result],
                vec![],
                0,
            )
            .insert_at_back(entry, ctx);
        }

        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);

        function.get_operation().insert_at_back(module_block, ctx);
    });

    module
        .mark_kernel_entry("wgmma_counted_pipeline_carrier_kernel")
        .expect("kernel entry must be marked");
}

fn build_wgmma_three_slot_counted_pipeline_carrier_kernel(module: &mut CodegenModule) {
    module.edit(|ctx, module| {
        let module_region = module.get_operation().deref(ctx).get_region(0);
        let module_block = module_region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("codegen module must contain its top-level block");

        let f32_ty = FP32Type::get(ctx);
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let output_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), true);

        let mut argument_types: Vec<pliron::r#type::TypeHandle> =
            Vec::with_capacity(1 + THREE_SLOT_RESULT_COUNT + THREE_SLOT_CONTROL_COUNT);
        let f32_arg_ty: pliron::r#type::TypeHandle = f32_ty.into();
        let u64_arg_ty: pliron::r#type::TypeHandle = u64_ty.into();
        argument_types.push(output_ptr_ty.into());
        argument_types.extend(vec![f32_arg_ty; THREE_SLOT_RESULT_COUNT]);
        argument_types.extend(vec![u64_arg_ty; THREE_SLOT_CONTROL_COUNT]);

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
        function.set_symbol_name(
            ctx,
            "wgmma_three_slot_counted_pipeline_carrier_kernel"
                .try_into()
                .unwrap(),
        );

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(function_region, ctx);

        let output = entry.deref(ctx).get_argument(0);
        let accumulator_inputs = (0..THREE_SLOT_RESULT_COUNT)
            .map(|index| entry.deref(ctx).get_argument(index + 1))
            .collect::<Vec<_>>();
        let controls = (0..THREE_SLOT_CONTROL_COUNT)
            .map(|index| {
                entry
                    .deref(ctx)
                    .get_argument(1 + THREE_SLOT_RESULT_COUNT + index)
            })
            .collect::<Vec<_>>();

        let template = counted_three_slot_pipeline_template();
        let constraints = counted_three_slot_pipeline_constraints();

        let mut asm_inputs = accumulator_inputs;
        asm_inputs.extend(controls);

        let inline_ptx = InlinePtxOp::build(
            ctx,
            vec![f32_ty.into(); THREE_SLOT_RESULT_COUNT],
            asm_inputs,
            &template,
            &constraints,
            true,
            true,
        );
        let results = (0..THREE_SLOT_RESULT_COUNT)
            .map(|index| inline_ptx.deref(ctx).get_result(index))
            .collect::<Vec<_>>();
        inline_ptx.insert_at_back(entry, ctx);

        for (index, result) in results.into_iter().enumerate() {
            let index_op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(index_op).set_attr_value(
                ctx,
                IntegerAttr::new(
                    i32_ty,
                    APInt::from_u32(index as u32, NonZeroUsize::new(32).unwrap()),
                ),
            );
            let index_value = index_op.deref(ctx).get_result(0);
            index_op.insert_at_back(entry, ctx);

            let output_element_op = Operation::new(
                ctx,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![output_ptr_ty.into()],
                vec![output, index_value],
                vec![],
                0,
            );
            let output_element = output_element_op.deref(ctx).get_result(0);
            output_element_op.insert_at_back(entry, ctx);

            Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![output_element, result],
                vec![],
                0,
            )
            .insert_at_back(entry, ctx);
        }

        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(entry, ctx);

        function.get_operation().insert_at_back(module_block, ctx);
    });

    module
        .mark_kernel_entry("wgmma_three_slot_counted_pipeline_carrier_kernel")
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
fn counted_two_slot_wgmma_pipeline_compiles_spill_free_for_sm_90a() {
    let mut module = CodegenModule::new("wgmma_counted_pipeline_carrier").unwrap();
    build_wgmma_counted_pipeline_carrier_kernel(&mut module);

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("counted two-slot WGMMA pipeline carrier must compile to PTX")
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
        text.contains("wgmma_counted_pipeline_carrier_kernel"),
        "counted-pipeline carrier kernel is missing:\n{text}"
    );

    assert!(
        text.contains("L__wgmma_counted_pipeline_loop_")
            && text.contains("L__wgmma_counted_pipeline_done_"),
        "counted-pipeline labels are missing:\n{text}"
    );
    assert!(
        !text.contains("${:uid}"),
        "llc must expand every ${{:uid}} escape into a unique label suffix:\n{text}"
    );
    assert_eq!(
        text.matches("bra.uni").count(),
        2,
        "the counted pipeline must keep exactly its guard and back-edge branches:\n{text}"
    );

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
        "the counted pipeline unexpectedly materializes local memory:\n{text}"
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let directory = std::env::temp_dir().join(format!(
        "wgmma_counted_pipeline_carrier_ptx_{}_{}",
        std::process::id(),
        unique,
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let ptx_path = directory.join("counted_pipeline.ptx");
    let cubin_path = directory.join("counted_pipeline.cubin");
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
        "ptxas rejected the counted two-slot WGMMA pipeline carrier:\n{stderr}\n\nPTX:\n{text}"
    );
    assert!(
        stderr.contains("0 bytes stack frame"),
        "ptxas reported a nonzero stack frame or omitted the expected stack-frame report:\n{stderr}"
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
        used_registers >= RESULT_COUNT as u32,
        "the probe did not keep both 32-value accumulator slots live: ptxas used only {used_registers} registers\n{stderr}"
    );

    eprintln!("{stderr}");
}

#[test]
fn counted_three_slot_wgmma_pipeline_compiles_spill_free_for_sm_90a() {
    let mut module = CodegenModule::new("wgmma_three_slot_counted_pipeline_carrier").unwrap();
    build_wgmma_three_slot_counted_pipeline_carrier_kernel(&mut module);

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("counted three-slot WGMMA pipeline carrier must compile to PTX")
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
        text.contains("wgmma_three_slot_counted_pipeline_carrier_kernel"),
        "three-slot counted-pipeline carrier kernel is missing:\n{text}"
    );

    assert!(
        text.contains("L__wgmma_three_slot_counted_pipeline_loop_")
            && text.contains("L__wgmma_three_slot_counted_pipeline_done_"),
        "three-slot counted-pipeline labels are missing:\n{text}"
    );
    assert!(
        !text.contains("${:uid}"),
        "llc must expand every ${{:uid}} escape into a unique label suffix:\n{text}"
    );
    assert_eq!(
        text.matches("bra.uni").count(),
        2,
        "the three-slot counted pipeline must keep exactly its guard and back-edge branches:\n{text}"
    );

    assert_eq!(text.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        text.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
            .count(),
        THREE_SLOT_COUNT
    );
    assert_eq!(
        text.matches("wgmma.commit_group.sync.aligned").count(),
        THREE_SLOT_COUNT
    );
    assert_eq!(
        text.matches("wgmma.wait_group.sync.aligned 2").count(),
        THREE_SLOT_COUNT
    );
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
        "the three-slot counted pipeline unexpectedly materializes local memory:\n{text}"
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let directory = std::env::temp_dir().join(format!(
        "wgmma_three_slot_counted_pipeline_carrier_ptx_{}_{}",
        std::process::id(),
        unique,
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let ptx_path = directory.join("three_slot_counted_pipeline.ptx");
    let cubin_path = directory.join("three_slot_counted_pipeline.cubin");
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
        "ptxas rejected the counted three-slot WGMMA pipeline carrier:\n{stderr}\n\nPTX:\n{text}"
    );
    assert!(
        stderr.contains("0 bytes stack frame"),
        "ptxas reported a nonzero stack frame or omitted the expected stack-frame report:\n{stderr}"
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
        used_registers >= THREE_SLOT_RESULT_COUNT as u32,
        "the probe did not keep all three 32-value accumulator slots live: ptxas used only {used_registers} registers\n{stderr}"
    );

    eprintln!("{stderr}");
}
