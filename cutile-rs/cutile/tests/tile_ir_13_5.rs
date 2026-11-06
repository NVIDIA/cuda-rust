/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA 13.5 frontend and bytecode contracts; no driver is needed.

use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::{BytecodeVersion, Opcode};

mod common;
#[path = "support/tile_ir_reference.rs"]
mod reference;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    unsafe fn multicast(input: *mut i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let base: PointerTile<*mut i32, { [] }> = unsafe { ptr_with_attr_multicast(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<i32, { [8] }> =
            unsafe { make_tensor_view(base, shape![8], strides, new_token_unordered()) };
        let tile_shape = shape![4];
        let token = new_token_unordered();
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(&tensor, tile_shape, padding::None, dim_map::Identity, token)
        };
        let (value, loaded): (Tile<i32, { [4] }>, Token) = unsafe {
            multimem_load_reduce_view_tko(
                &view,
                [0i32],
                multimem::Add,
                ordering::Acquire,
                scope::System,
                None,
                [true],
                shape![4],
            )
        };
        let reduced = unsafe {
            multimem_reduce_view_tko(
                &view,
                value,
                [0i32],
                multimem::Umax,
                ordering::Release,
                scope::System,
                Some(loaded),
                [true],
            )
        };
        let _stored = unsafe {
            multimem_store_view_tko(
                &view,
                value,
                [1i32],
                ordering::Release,
                scope::System,
                Some(reduced),
                [true],
            )
        };
    }

    #[cutile::entry()]
    unsafe fn unicast(input: *mut i32) {
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(input);
        let base = unsafe { ptr_with_attr_unicast(base) };
        let (value, loaded): (Tile<i32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                base,
                ordering::Acquire,
                Some(scope::System),
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        let _stored = unsafe {
            store_ptr_tko(
                base,
                value,
                ordering::Release,
                Some(scope::System),
                None,
                Some(loaded),
                Latency::<0>,
            )
        };
    }

    // Assembly-only: checked loads require a supported floating-point type
    // and ordered memory semantics. Execution also needs a validity protocol.
    #[cutile::entry()]
    fn validity(input: &Tensor<f32, { [8] }>) {
        let view: Partition<f32, { [4] }> = unsafe {
            make_partition_view(
                input,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let (_value, _token): (Tile<f32, { [4] }>, Token) = unsafe {
            load_view_tko_valid_check(
                &view,
                [0i32],
                None,
                ordering::Acquire,
                scope::Device,
                tma::Enabled,
                None,
                [true],
                shape![4],
                true,
            )
        };
    }

    #[cutile::entry()]
    fn atomic_inbounds(input: &Tensor<i32, { [8] }>) {
        let view: Partition<i32, { [4] }> = unsafe {
            make_partition_view(
                input,
                shape![4],
                padding::None,
                dim_map::Identity,
                new_token_unordered(),
            )
        };
        let value: Tile<i32, { [4] }> = constant(1, shape![4]);
        let _token = unsafe {
            atomic_red_view_tko_with_inbounds(
                &view,
                value,
                [0i32],
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                None,
                [true],
            )
        };
    }

    #[cutile::entry()]
    fn selection(input: &mut Tensor<i32, { [4] }>) {
        let condition: Tile<bool, { [4] }> = constant(true, shape![4]);
        let first: Tile<i32, { [4] }> = constant(1, shape![4]);
        let second: Tile<i32, { [4] }> = constant(2, shape![4]);
        let value: Tile<i32, { [4] }> = select(condition.likely::<true>(), first, second);
        let value: Tile<i32, { [4] }> = select(condition.likely::<false>(), value, second);
        input.store(value);
    }

    #[cutile::entry()]
    fn conditional(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        if condition != 0 {
            let value: Tile<i32, { [4] }> = constant(1, shape![4]);
            input.store(value);
        } else {
            let value: Tile<i32, { [4] }> = constant(2, shape![4]);
            input.store(value);
        }
    }

    fn hinted<const EXPECTED: bool>(condition: bool) -> bool {
        condition.likely::<EXPECTED>()
    }

    fn hinted_mask<const EXPECTED: bool, const S: [i32; 2]>(
        condition: Tile<bool, S>,
    ) -> Tile<bool, S> {
        condition.likely::<EXPECTED>()
    }

    #[cutile::entry()]
    fn hinted_matrix_selection(input: &mut Tensor<i32, { [2, 2] }>) {
        let data = input.load();
        let zero: Tile<i32, { [2, 2] }> = constant(0, shape![2, 2]);
        let mask = cmpi(data, zero, predicate::GreaterThan);
        let hinted = hinted_mask::<true, { [2, 2] }>(mask);
        let value = select(hinted, data, zero);
        input.store(value);
    }

    fn forwarded<const EXPECTED: bool>(condition: bool) -> bool {
        hinted::<EXPECTED>(condition)
    }

    #[cutile::entry()]
    fn hinted_conditional(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        let value: Tile<i32, { [4] }> = if (condition != 0).likely::<true>() {
            constant(1, shape![4])
        } else {
            constant(2, shape![4])
        };
        let unlikely = forwarded::<false>(condition != 1);
        let value = if unlikely {
            value
        } else {
            constant(3, shape![4])
        };
        input.store(value);
    }

    #[cutile::entry()]
    fn hinted_unit_region(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        let mut value: Tile<i32, { [4] }> = constant(1, shape![4]);
        if (condition != 0).likely::<true>() {
            value = constant(2, shape![4]);
        }
        input.store(value);
    }

    #[cutile::entry()]
    fn hinted_return(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        if (condition != 0).likely::<false>() {
            let value: Tile<i32, { [4] }> = constant(7, shape![4]);
            input.store(value);
            return;
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        input.store(value);
    }

    #[cutile::entry()]
    fn ordinary_return(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        if condition != 0 {
            let value: Tile<i32, { [4] }> = constant(7, shape![4]);
            input.store(value);
            return;
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        input.store(value);
    }

    #[cutile::entry()]
    fn hinted_loop(input: &mut Tensor<i32, { [4] }>, limit: i32) {
        let mut total: Tile<i32, { [4] }> = constant(0, shape![4]);
        let one: Tile<i32, { [4] }> = constant(1, shape![4]);
        let mut i = 0i32;
        while i < 8 {
            if (i >= limit).likely::<false>() {
                break;
            }
            i = i + 1;
            if (i % 2 != 0).likely::<true>() {
                continue;
            }
            total = total + one;
        }
        input.store(total);
    }

    #[cutile::entry()]
    fn hinted_selection(input: &mut Tensor<i32, { [4] }>) {
        let data = input.load();
        let zero: Tile<i32, { [4] }> = constant(0, shape![4]);
        let condition: Tile<bool, { [4] }> = cmpi(data, zero, predicate::GreaterThan);
        let yes: Tile<i32, { [4] }> = constant(11, shape![4]);
        let no: Tile<i32, { [4] }> = constant(22, shape![4]);
        let a: Tile<i32, { [4] }> = select(condition.likely::<true>(), yes, no);
        let b: Tile<i32, { [4] }> = select(condition.likely::<false>(), yes, no);
        // Hinting a copy must not annotate the original condition.
        let c: Tile<i32, { [4] }> = select(condition, yes, no);
        input.store(a + b + c);
    }

    #[cutile::entry()]
    fn hinted_reuse(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        let predicate = condition != 0;
        let first: Tile<i32, { [4] }> = if forwarded::<true>(predicate) {
            constant(1, shape![4])
        } else {
            constant(2, shape![4])
        };
        let second: Tile<i32, { [4] }> = if forwarded::<false>(predicate) {
            constant(3, shape![4])
        } else {
            constant(4, shape![4])
        };
        let third: Tile<i32, { [4] }> = if predicate {
            constant(5, shape![4])
        } else {
            constant(6, shape![4])
        };
        input.store(first + second + third);
    }

    #[cutile::entry()]
    fn generic_likelihood<const EXPECTED: bool>(input: &mut Tensor<i32, { [4] }>, condition: i32) {
        if forwarded::<EXPECTED>(condition != 0) {
            let value: Tile<i32, { [4] }> = constant(5, shape![4]);
            input.store(value);
        } else {
            let value: Tile<i32, { [4] }> = constant(6, shape![4]);
            input.store(value);
        }
    }

    #[cutile::entry(optimization_hints = (
        default = (min_ctas_per_sm = 1,),
        sm_100 = (num_worker_warps_per_cta = 2,),
    ))]
    fn default_residency_and_worker_warps() {}

    #[cutile::entry(optimization_hints = (sm_100 = (min_ctas_per_sm = 2,),))]
    fn min_ctas() {}
}

#[test]
fn compile_fabric_operations_and_preserve_classification() {
    common::with_test_stack(|| {
        for (kernel, classification) in [("multicast", "multicast"), ("unicast", "unicast")] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            assert!(artifacts
                .ir_text()
                .contains(&format!("ptr_attr<{classification}>")));
            let bytes = artifacts.bytecode().unwrap();
            reference::verify_if_configured(&bytes);
            assert!(cutile_ir::decode_bytecode(&bytes)
                .unwrap()
                .contains("v13.5"));
            if kernel == "multicast" {
                for opcode in [
                    Opcode::MultimemLoadReduceViewTko,
                    Opcode::MultimemReduceViewTko,
                    Opcode::MultimemStoreViewTko,
                ] {
                    assert!(artifacts.ir_text().contains(opcode.name()));
                }
            }
        }
    });
}

#[test]
fn fabric_operations_reject_old_versions_and_non_nvlink_targets() {
    common::with_test_stack(|| {
        for kernel in ["multicast", "unicast"] {
            for (version, target, diagnostic) in [
                (BytecodeVersion::V13_4, "sm_100", "13.5"),
                (BytecodeVersion::V13_5, "sm_120", "sm_100"),
            ] {
                let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                    .target(target)
                    .bytecode_version(version)
                    .compile()
                    .err()
                    .expect("unsupported combination must be refused");
                assert!(error.to_string().contains(diagnostic), "{error}");
            }
        }
    });
}

#[test]
fn minimum_residency_hint_is_versioned() {
    common::with_test_stack(|| {
        let compiled = KernelCompiler::new(kernels::__module_ast_self, "kernels", "min_ctas")
            .target("sm_100")
            .bytecode_version(BytecodeVersion::V13_5)
            .compile()
            .unwrap();
        assert!(compiled.ir_text().contains("min_ctas_per_sm"));
        reference::verify_if_configured(&compiled.bytecode().unwrap());
        let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", "min_ctas")
            .target("sm_100")
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()
            .err()
            .unwrap();
        assert!(error.to_string().contains("13.5"), "{error}");
    });
}

#[test]
fn new_view_attributes_and_selection_hints_are_versioned() {
    common::with_test_stack(|| {
        for (kernel, target, attribute) in [
            ("validity", "sm_107", "valid_check"),
            ("atomic_inbounds", "sm_100", "inbounds"),
            ("selection", "sm_100", "condition_hint"),
            ("hinted_conditional", "sm_100", "condition_hint"),
            ("hinted_unit_region", "sm_100", "condition_hint"),
        ] {
            let compiled = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target(target)
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            // Attributes on all operations are retained, even if their text printer omits them.
            assert!(
                format!("{:?}", compiled.module()).contains(attribute),
                "{kernel}"
            );
            reference::verify_if_configured(&compiled.bytecode().unwrap());
            let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target(target)
                .bytecode_version(BytecodeVersion::V13_4)
                .compile()
                .err()
                .expect("13.5 required");
            assert!(error.to_string().contains("13.5"), "{kernel}: {error}");
        }
        let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", "validity")
            .strides(&[("input", &[1])])
            .target("sm_100")
            .bytecode_version(BytecodeVersion::V13_5)
            .compile()
            .err()
            .unwrap();
        assert!(error.to_string().contains("sm_107"), "{error}");
    });
}

#[test]
fn if_flags_and_hints_are_accepted_by_the_reference_reader() {
    common::with_test_stack(|| {
        use cutile_ir::ir::{Attribute, OptimizationHints};
        let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", "conditional")
            .strides(&[("input", &[1])])
            .target("sm_100")
            .bytecode_version(BytecodeVersion::V13_5)
            .compile()
            .unwrap();
        reference::verify_if_configured(&artifacts.bytecode().unwrap());
        let mut module = artifacts.into_module();
        let region = module.op(module.functions[0]).regions[0];
        let block = module.region(region).blocks[0];
        let conditional = *module
            .block(block)
            .ops
            .iter()
            .find(|id| module.op(**id).opcode == Opcode::If)
            .unwrap();
        module.op_mut(conditional).attributes.push((
            "optimization_hints".into(),
            Attribute::OptimizationHints(OptimizationHints {
                entries: vec![(
                    "default".into(),
                    vec![(
                        "discardable_perf_hints".into(),
                        Attribute::String("{\"condition_hint\":\"LIKELY_TRUE\"}".into()),
                    )],
                )],
            }),
        ));
        cutile_ir::capabilities::TargetCapabilities::new(BytecodeVersion::V13_5, "sm_100")
            .validate_module(&module)
            .unwrap();
        let bytes =
            cutile_ir::bytecode::write_bytecode_version(&module, BytecodeVersion::V13_5).unwrap();
        reference::verify_if_configured(&bytes);
        assert!(
            cutile_ir::bytecode::write_bytecode_version(&module, BytecodeVersion::V13_4)
                .unwrap_err()
                .to_string()
                .contains("13.5")
        );
    });
}

#[test]
#[ignore = "requires a CUDA 13.5 assembler/backend; no GPU required"]
fn assemble_13_5_feature_kernels() {
    common::with_test_stack(|| {
        assert_eq!(
            std::env::var("CUTILE_BYTECODE_VERSION").as_deref(),
            Ok("13.5"),
            "set CUTILE_BYTECODE_VERSION=13.5 to prevent assembler fallback"
        );
        for (kernel, target) in [
            ("multicast", "sm_100"),
            ("unicast", "sm_100"),
            ("validity", "sm_107"),
            ("atomic_inbounds", "sm_100"),
            ("selection", "sm_100"),
            ("conditional", "sm_100"),
            ("hinted_conditional", "sm_100"),
            ("hinted_unit_region", "sm_100"),
            ("hinted_selection", "sm_100"),
            ("hinted_reuse", "sm_100"),
            ("hinted_loop", "sm_100"),
            ("default_residency_and_worker_warps", "sm_100"),
            ("min_ctas", "sm_100"),
        ] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target(target)
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            let cubin = cutile_compiler::cuda_tile_runtime_utils::compile_tile_ir_module(
                artifacts.module(),
                target,
            )
            .unwrap_or_else(|e| panic!("{kernel}: {e}"));
            assert!(
                cubin.starts_with(b"\x7fELF"),
                "{kernel}: expected cubin ELF"
            );
        }
    });
}

#[test]
fn residency_overrides_and_worker_warps_match_tile_ir() {
    use cutile_compiler::hints::CompileOptions;
    common::with_test_stack(|| {
        let compile = |options| {
            KernelCompiler::new(kernels::__module_ast_self, "kernels", "min_ctas")
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .options(options)
                .compile()
        };
        for value in [0, 1, 32] {
            let artifacts = compile(CompileOptions::default().min_ctas_per_sm(value)).unwrap();
            assert!(artifacts
                .ir_text()
                .contains(&format!("min_ctas_per_sm = {value}")));
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
        for value in [-1, 33] {
            let error = compile(CompileOptions::default().min_ctas_per_sm(value))
                .err()
                .unwrap();
            assert!(error.to_string().contains("min_ctas_per_sm"), "{error}");
        }
        for warps in [1, 2, 4, 8, 16, 32] {
            let artifacts = KernelCompiler::new(
                kernels::__module_ast_self,
                "kernels",
                "default_residency_and_worker_warps",
            )
            .target("sm_100")
            .bytecode_version(BytecodeVersion::V13_5)
            .options(CompileOptions::default().num_worker_warps_per_cta(warps))
            .compile()
            .unwrap();
            let hints = format!("{:?}", artifacts.module());
            assert!(hints.contains("default"));
            assert!(hints.contains("min_ctas_per_sm"));
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
    });
}

#[test]
fn hinted_conditions_preserve_results_and_captures() {
    common::with_test_stack(|| {
        for (kernel, count) in [("hinted_conditional", 2), ("hinted_unit_region", 1)] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap();
            let module = artifacts.module();
            let region = module.op(module.functions[0]).regions[0];
            let block = module.region(region).blocks[0];
            let conditionals = module
                .block(block)
                .ops
                .iter()
                .map(|id| module.op(*id))
                .filter(|op| op.opcode == Opcode::If)
                .collect::<Vec<_>>();
            assert_eq!(conditionals.len(), count);
            for op in &conditionals {
                assert!(
                    !op.result_types.is_empty(),
                    "branch result or captured mutation must escape"
                );
                assert!(format!("{:?}", op.attributes).contains("condition_hint"));
            }
            assert!(format!("{:?}", conditionals[0].attributes).contains("LIKELY_TRUE"));
            if count == 2 {
                assert!(format!("{:?}", conditionals[1].attributes).contains("LIKELY_FALSE"));
            }
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
    });
}

#[test]
fn likelihood_preserves_ordinary_control_flow_diagnostics() {
    common::with_test_stack(|| {
        for kernel in ["ordinary_return", "hinted_return"] {
            let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .err()
                .unwrap();
            assert!(
                error
                    .to_string()
                    .contains("`return` is only supported at the top level"),
                "{error}"
            );
        }
    });
}

#[test]
fn likelihood_is_forwarded_without_annotating_the_original_condition() {
    common::with_test_stack(|| {
        for (kernel, opcode) in [
            ("hinted_selection", Opcode::Select),
            ("hinted_reuse", Opcode::If),
        ] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", kernel)
                .strides(&[("input", &[1])])
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .unwrap();
            let module = artifacts.module();
            let region = module.op(module.functions[0]).regions[0];
            let block = module.region(region).blocks[0];
            let selects: Vec<_> = module
                .block(block)
                .ops
                .iter()
                .map(|id| module.op(*id))
                .filter(|op| op.opcode == opcode)
                .collect();
            assert_eq!(selects.len(), 3);
            assert!(format!("{:?}", selects[0].attributes).contains("LIKELY_TRUE"));
            assert!(format!("{:?}", selects[1].attributes).contains("LIKELY_FALSE"));
            assert!(!format!("{:?}", selects[2].attributes).contains("condition_hint"));
            assert_eq!(selects[0].operands[0], selects[1].operands[0]);
            assert_eq!(selects[0].operands[0], selects[2].operands[0]);
            reference::verify_if_configured(&artifacts.bytecode().unwrap());
        }
        let matrix = KernelCompiler::new(
            kernels::__module_ast_self,
            "kernels",
            "hinted_matrix_selection",
        )
        .strides(&[("input", &[2, 1])])
        .target("sm_100")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .unwrap();
        assert!(format!("{:?}", matrix.module()).contains("LIKELY_TRUE"));
        reference::verify_if_configured(&matrix.bytecode().unwrap());
        for (expected, name) in [(true, "LIKELY_TRUE"), (false, "LIKELY_FALSE")] {
            let artifacts =
                KernelCompiler::new(kernels::__module_ast_self, "kernels", "generic_likelihood")
                    .generics(vec![expected.to_string()])
                    .strides(&[("input", &[1])])
                    .target("sm_100")
                    .bytecode_version(BytecodeVersion::V13_5)
                    .compile()
                    .unwrap();
            assert!(format!("{:?}", artifacts.module()).contains(name));
        }
    });
}

#[test]
#[ignore = "requires a CUDA 13.5 assembler and GPU"]
fn likelihood_hints_preserve_gpu_results() -> Result<(), cutile::error::Error> {
    use cutile::prelude::*;
    assert!(common::supports_tile_ir(&[
        cutile_ir::requirements::Feature::ControlFlowHints
    ]));
    for condition in [0, 1, 2] {
        let input = api::zeros::<i32>(&[4]).sync()?.partition([4]);
        let (result, _) = kernels::hinted_reuse(input, condition).sync()?;
        assert_eq!(
            result.unpartition().to_host_vec().sync()?,
            vec![if condition == 0 { 12 } else { 9 }; 4]
        );
        let input = api::zeros::<i32>(&[4]).sync()?.partition([4]);
        let (result, _) = kernels::hinted_conditional(input, condition).sync()?;
        let expected = match condition {
            0 => 2,
            1 => 3,
            _ => 1,
        };
        assert_eq!(
            result.unpartition().to_host_vec().sync()?,
            vec![expected; 4]
        );
        let input = api::zeros::<i32>(&[4]).sync()?.partition([4]);
        let (result, _) = kernels::hinted_unit_region(input, condition).sync()?;
        assert_eq!(
            result.unpartition().to_host_vec().sync()?,
            vec![if condition == 0 { 1 } else { 2 }; 4]
        );
    }
    for limit in [0, 1, 3, 8] {
        let input = api::zeros::<i32>(&[4]).sync()?.partition([4]);
        let (result, _) = kernels::hinted_loop(input, limit).sync()?;
        assert_eq!(
            result.unpartition().to_host_vec().sync()?,
            vec![limit / 2; 4]
        );
    }
    let input = api::copy_host_vec_to_device(&Arc::new(vec![-2i32, 0, 1, 8]))
        .sync()?
        .partition([4]);
    let (result,) = kernels::hinted_selection(input).sync()?;
    assert_eq!(
        result.unpartition().to_host_vec().sync()?,
        vec![66, 66, 33, 33]
    );
    Ok(())
}
