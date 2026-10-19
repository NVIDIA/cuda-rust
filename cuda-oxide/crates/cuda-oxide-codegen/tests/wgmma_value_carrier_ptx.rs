/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proves that the codegen pipeline can carry tied WGMMA `f32` accumulator
//! values through one inline-PTX operation and produce spill-free `sm_90a` code.
//!
//! The probe covers the 32-value m64n64 BF16/F16/TF32 carriers and the
//! 64-value m64n128 BF16 carrier, for a single `wgmma.mma_async` and a chain
//! of several under one fence/commit/wait sequence in the same asm region.

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

const M64N64_ACCUMULATOR_LEN: usize = 32;
const M64N128_ACCUMULATOR_LEN: usize = 64;

#[derive(Clone, Copy, Debug)]
enum WgmmaCarrierKind {
    Bf16M64N64,
    F16M64N64,
    Tf32M64N64,
    Bf16M64N128,
}

impl WgmmaCarrierKind {
    fn accumulator_len(self) -> usize {
        match self {
            Self::Bf16M64N64 | Self::F16M64N64 | Self::Tf32M64N64 => M64N64_ACCUMULATOR_LEN,
            Self::Bf16M64N128 => M64N128_ACCUMULATOR_LEN,
        }
    }

    fn ptx_mnemonic(self) -> &'static str {
        match self {
            Self::Bf16M64N64 => "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16",
            Self::F16M64N64 => "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16",
            Self::Tf32M64N64 => "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32",
            Self::Bf16M64N128 => "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
        }
    }

    fn control_operands(self) -> &'static str {
        match self {
            Self::Bf16M64N64 | Self::F16M64N64 | Self::Bf16M64N128 => "1, 1, 1, 0, 0",
            Self::Tf32M64N64 => "1, 1, 1",
        }
    }
}

fn build_wgmma_value_carrier_kernel(
    module: &mut CodegenModule,
    mma_count: usize,
    kind: WgmmaCarrierKind,
) {
    module.edit(|ctx, module| {
        let accumulator_len = kind.accumulator_len();
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

        let descriptor_count = mma_count * 2;

        // Kernel signature:
        //
        // wgmma_value_carrier_kernel(
        //     output: *mut f32,
        //     acc0: f32,
        //     ...
        //     acc31: f32,
        //     desc_a0: u64,
        //     desc_b0: u64,
        //     ...              // one descriptor pair per chained MMA
        // )
        let mut argument_types: Vec<pliron::r#type::TypeHandle> =
            Vec::with_capacity(1 + accumulator_len + descriptor_count);

        argument_types.push(output_ptr_ty.into());

        let accumulator_types: Vec<pliron::r#type::TypeHandle> =
            vec![f32_ty.into(); accumulator_len];

        argument_types.extend(accumulator_types);

        for _ in 0..descriptor_count {
            argument_types.push(u64_ty.into());
        }

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
        function.set_symbol_name(ctx, "wgmma_value_carrier_kernel".try_into().unwrap());

        let function_region = function.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(function_region, ctx);

        let output = entry.deref(ctx).get_argument(0);
        let accumulator_inputs = (0..accumulator_len)
            .map(|index| entry.deref(ctx).get_argument(index + 1))
            .collect::<Vec<_>>();

        let descriptors = (0..descriptor_count)
            .map(|index| entry.deref(ctx).get_argument(1 + accumulator_len + index))
            .collect::<Vec<_>>();

        // Each output is tied to the corresponding accumulator input through
        // tied constraints. Every chained WGMMA instruction references all
        // output operands as one accumulator register tuple, the
        // same shape the value-form lowering template emits.
        let accumulator_registers = (0..accumulator_len)
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ");

        let desc_operand_base = accumulator_len * 2;

        let mut template = String::from("{\n    wgmma.fence.sync.aligned;\n");
        let ptx_mnemonic = kind.ptx_mnemonic();
        let control_operands = kind.control_operands();

        for mma_index in 0..mma_count {
            let desc_a = desc_operand_base + mma_index * 2;
            let desc_b = desc_a + 1;
            template.push_str(&format!(
                "    {ptx_mnemonic} \
                 {{{accumulator_registers}}}, ${desc_a}, ${desc_b}, {control_operands};\n"
            ));
        }

        template.push_str("    wgmma.commit_group.sync.aligned;\n");
        template.push_str("    wgmma.wait_group.sync.aligned 0;\n");
        template.push('}');

        let mut constraints = vec!["=f".to_owned(); accumulator_len];

        constraints.extend((0..accumulator_len).map(|index| index.to_string()));

        constraints.extend((0..descriptor_count).map(|_| "l".to_owned()));
        constraints.push("~{memory}".to_owned());

        let constraints = constraints.join(",");

        let mut asm_inputs = accumulator_inputs;
        asm_inputs.extend(descriptors);

        let inline_ptx = InlinePtxOp::build(
            ctx,
            vec![f32_ty.into(); accumulator_len],
            asm_inputs,
            &template,
            &constraints,
            true,
            true,
        );

        let accumulator_results = (0..accumulator_len)
            .map(|index| inline_ptx.deref(ctx).get_result(index))
            .collect::<Vec<_>>();

        inline_ptx.insert_at_back(entry, ctx);

        // Keep every result observably live across the inline-PTX boundary.
        //
        // Immediately after the asm, all thirty-two values must remain available.
        // Each value is then written to output[index], preventing the backend from
        // discarding unused outputs or shortening the probe to one live result.
        for (index, result) in accumulator_results.into_iter().enumerate() {
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
        .mark_kernel_entry("wgmma_value_carrier_kernel")
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

fn assert_spill_free_value_carrier(kind: WgmmaCarrierKind, mma_count: usize) {
    let mut module = CodegenModule::new("wgmma_value_carrier").unwrap();
    build_wgmma_value_carrier_kernel(&mut module, mma_count, kind);

    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt must be installed");
    let options = CompileOptions::new(Target::parse("sm_90a").unwrap());

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("WGMMA value carrier must compile to PTX")
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
        text.contains("wgmma_value_carrier_kernel"),
        "carrier kernel is missing:\n{text}",
    );
    assert_eq!(
        text.matches("wgmma.fence.sync.aligned").count(),
        1,
        "the WGMMA carrier must contain exactly one fence:\n{text}",
    );

    let accumulator_len = kind.accumulator_len();
    let expected_mma = kind.ptx_mnemonic();
    assert_eq!(
        text.matches(expected_mma).count(),
        mma_count,
        "the WGMMA carrier must chain exactly {mma_count} {kind:?} MMAs:\n{text}",
    );

    match kind {
        WgmmaCarrierKind::Bf16M64N64
        | WgmmaCarrierKind::F16M64N64
        | WgmmaCarrierKind::Bf16M64N128 => {
            assert!(
                text.contains("1, 1, 1, 0, 0;"),
                "K=16 WGMMA must carry transpose controls:\n{text}"
            );
        }
        WgmmaCarrierKind::Tf32M64N64 => {
            assert!(
                text.contains("1, 1, 1;"),
                "TF32 WGMMA must carry only scale controls:\n{text}"
            );
            assert!(
                !text.contains("1, 1, 1, 0, 0;"),
                "TF32 WGMMA must not carry transpose controls:\n{text}"
            );
        }
    }

    assert_eq!(
        text.matches("wgmma.commit_group.sync.aligned").count(),
        1,
        "the WGMMA carrier must contain exactly one commit:\n{text}",
    );

    assert_eq!(
        text.matches("wgmma.wait_group.sync.aligned 0").count(),
        1,
        "the WGMMA carrier must fully drain the group:\n{text}",
    );
    assert!(
        !text.contains(".local") && !text.contains("ld.local") && !text.contains("st.local"),
        "the carrier unexpectedly materializes local memory:\n{text}",
    );

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);

    let directory = std::env::temp_dir().join(format!(
        "wgmma_value_carrier_ptx_{:?}_{}_{}_{}",
        kind,
        mma_count,
        std::process::id(),
        unique,
    ));

    std::fs::create_dir_all(&directory).unwrap();

    let ptx_path = directory.join("carrier.ptx");
    let cubin_path = directory.join("carrier.cubin");
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
        "ptxas rejected the {accumulator_len}-value {kind:?} carrier:\n{stderr}\n\nPTX:\n{text}",
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
        used_registers >= accumulator_len as u32,
        "the probe did not keep all {accumulator_len} accumulator values live: \
     ptxas used only {used_registers} registers\n{stderr}",
    );

    eprintln!("{stderr}");
}

#[test]
fn bf16_wgmma_uses_thirty_two_tied_f32_values_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Bf16M64N64, 1);
}

#[test]
fn bf16_wgmma_chains_two_mma_async_under_one_commit_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Bf16M64N64, 2);
}

#[test]
fn f16_wgmma_uses_thirty_two_tied_f32_values_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::F16M64N64, 1);
}

#[test]
fn f16_wgmma_chains_two_mma_async_under_one_commit_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::F16M64N64, 2);
}

#[test]
fn bf16_m64n128_wgmma_uses_sixty_four_tied_f32_values_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Bf16M64N128, 1);
}

#[test]
fn bf16_m64n128_wgmma_chains_two_mma_async_under_one_commit_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Bf16M64N128, 2);
}

#[test]
fn tf32_wgmma_uses_thirty_two_tied_f32_values_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Tf32M64N64, 1);
}

#[test]
fn tf32_wgmma_chains_two_mma_async_under_one_commit_without_spills() {
    assert_spill_free_value_carrier(WgmmaCarrierKind::Tf32M64N64, 2);
}
