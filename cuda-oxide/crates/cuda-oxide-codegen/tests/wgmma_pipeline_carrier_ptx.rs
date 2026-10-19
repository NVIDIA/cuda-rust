/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proves that a two-slot BF16 WGMMA pipeline with `wait_group<1>` can keep
//! sixty-four tied `f32` accumulator values spill-free through code generation.

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
const DESCRIPTOR_COUNT: usize = 4;

fn accumulator_register_list(slot: usize) -> String {
    let base = slot * ACCUMULATOR_LEN;
    (base..base + ACCUMULATOR_LEN)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_wgmma_pipeline_carrier_kernel(module: &mut CodegenModule) {
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
            Vec::with_capacity(1 + RESULT_COUNT + DESCRIPTOR_COUNT);
        let f32_arg_ty: pliron::r#type::TypeHandle = f32_ty.into();
        let u64_arg_ty: pliron::r#type::TypeHandle = u64_ty.into();
        argument_types.push(output_ptr_ty.into());
        argument_types.extend(vec![f32_arg_ty; RESULT_COUNT]);
        argument_types.extend(vec![u64_arg_ty; DESCRIPTOR_COUNT]);

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
        function.set_symbol_name(ctx, "wgmma_pipeline_carrier_kernel".try_into().unwrap());

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(function_region, ctx);

        let output = entry.deref(ctx).get_argument(0);
        let accumulator_inputs = (0..RESULT_COUNT)
            .map(|index| entry.deref(ctx).get_argument(index + 1))
            .collect::<Vec<_>>();
        let descriptors = (0..DESCRIPTOR_COUNT)
            .map(|index| entry.deref(ctx).get_argument(1 + RESULT_COUNT + index))
            .collect::<Vec<_>>();

        let slot0 = accumulator_register_list(0);
        let slot1 = accumulator_register_list(1);
        let descriptor_base = RESULT_COUNT * 2;
        let template = format!(
            "{{\n\
             \x20   wgmma.fence.sync.aligned;\n\
             \x20   wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{slot0}}}, ${descriptor_base}, ${}, 1, 1, 1, 0, 0;\n\
             \x20   wgmma.commit_group.sync.aligned;\n\
             \x20   wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{slot1}}}, ${}, ${}, 1, 1, 1, 0, 0;\n\
             \x20   wgmma.commit_group.sync.aligned;\n\
             \x20   wgmma.wait_group.sync.aligned 1;\n\
             \x20   wgmma.wait_group.sync.aligned 0;\n\
             }}",
            descriptor_base + 1,
            descriptor_base + 2,
            descriptor_base + 3,
        );

        let mut constraints = vec!["=f".to_owned(); RESULT_COUNT];
        constraints.extend((0..RESULT_COUNT).map(|index| index.to_string()));
        constraints.extend((0..DESCRIPTOR_COUNT).map(|_| "l".to_owned()));
        constraints.push("~{memory}".to_owned());
        let constraints = constraints.join(",");

        let mut asm_inputs = accumulator_inputs;
        asm_inputs.extend(descriptors);
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
        .mark_kernel_entry("wgmma_pipeline_carrier_kernel")
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
fn bf16_wgmma_wait_group_one_keeps_two_accumulator_slots_without_spills() {
    let mut module = CodegenModule::new("wgmma_pipeline_carrier").unwrap();
    build_wgmma_pipeline_carrier_kernel(&mut module);

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());
    let ptx = compiler
        .compile(&mut module, &options)
        .expect("two-slot WGMMA pipeline carrier must compile to PTX")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX must be UTF-8");

    assert!(
        text.contains(".target sm_90a"),
        "PTX must target sm_90a:\n{text}"
    );
    assert!(
        text.contains("wgmma_pipeline_carrier_kernel"),
        "pipeline carrier kernel is missing:\n{text}"
    );
    assert_eq!(text.matches("wgmma.fence.sync.aligned").count(), 1);
    assert_eq!(
        text.matches("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16")
            .count(),
        2
    );
    assert_eq!(text.matches("wgmma.commit_group.sync.aligned").count(), 2);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 1").count(), 1);
    assert_eq!(text.matches("wgmma.wait_group.sync.aligned 0").count(), 1);
    assert!(
        !text.contains(".local") && !text.contains("ld.local") && !text.contains("st.local"),
        "the pipeline carrier unexpectedly materializes local memory:\n{text}"
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let directory = std::env::temp_dir().join(format!(
        "wgmma_pipeline_carrier_ptx_{}_{}",
        std::process::id(),
        unique,
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let ptx_path = directory.join("pipeline.ptx");
    let cubin_path = directory.join("pipeline.cubin");
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
        "ptxas rejected the two-slot WGMMA pipeline carrier:\n{stderr}\n\nPTX:\n{text}"
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
