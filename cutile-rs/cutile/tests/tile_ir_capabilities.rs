/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-free tests for the same feature gates used by JIT launches.

use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[test]
fn raw_op_metadata_matches_the_requirements_registry() {
    use cutile_compiler::syn_utils::SingleMetaList;
    use cutile_ir::requirements::{opcode_requirement, Architecture, Feature, OPCODES};
    use syn::visit::Visit;

    struct MetadataCheck(usize);
    impl<'ast> Visit<'ast> for MetadataCheck {
        fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
            if attr.path().segments.len() != 2
                || attr.path().segments[0].ident != "cuda_tile"
                || attr.path().segments[1].ident != "op"
            {
                return;
            }
            self.0 += 1;
            let metadata = SingleMetaList::from_attribute(attr.clone());
            let name = metadata.parse_string("name").expect("op name");
            // This legacy declaration has no dialect opcode or lowering.
            // It is not part of the supported Tile IR surface.
            if name == "cuda_tile.noti" {
                return;
            }
            let canonical_name = match name.strip_prefix("cuda_tile.").unwrap_or(&name) {
                "fpowf" => "pow",
                name => name,
            };
            let opcode = OPCODES
                .iter()
                .find(|op| canonical_name == op.name())
                .unwrap_or_else(|| panic!("{name} has no registry entry"));
            let requirement = opcode_requirement(*opcode);
            let since = if metadata.parse_string("pointer_attribute").is_some() {
                Feature::PointerAttribute.since()
            } else {
                requirement.since
            };
            assert_eq!(
                metadata.parse_string("since"),
                Some(format!("V{}_{}", since.major, since.minor)),
                "{name}"
            );
            let min_arch = match requirement.architecture {
                Architecture::AtLeast(sm) if sm > 80 => Some(format!("sm_{sm}")),
                _ => None,
            };
            assert_eq!(metadata.parse_string("min_arch"), min_arch, "{name}");
        }
    }
    let file = syn::parse_file(include_str!("../src/_core.rs")).unwrap();
    let mut check = MetadataCheck(0);
    check.visit_file(&file);
    assert!(check.0 > 80, "the test must visit the raw op declarations");
}

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    fn insert_kernel(output: &mut Tensor<f32, { [8] }>) {
        let small: Tile<f32, { [4] }> = constant(2.0, shape![4]);
        let large: Tile<f32, { [8] }> = constant(1.0, shape![8]);
        let index: Tile<i32, { [] }> = constant(1, shape![]);
        let result: Tile<f32, { [8] }> = unsafe { insert(small, large, [index]) };
        output.store(result);
    }

    #[cutile::entry()]
    fn pdl_tokens() {
        let launched: Token = unsafe { gdc_launch_dependents_tko(None) };
        let ready: Token = unsafe { gdc_wait_tko(Some(launched)) };
        let _fenced: Token = unsafe { memory_fence_alias_tko(ready) };
    }

    #[cutile::entry()]
    fn integer_power(output: &mut Tensor<f32, { [8] }>) {
        let base: Tile<f32, { [8] }> = constant(2.0, shape![8]);
        let exponent: Tile<i32, { [8] }> = constant(3, shape![8]);
        let result: Tile<f32, { [8] }> = unsafe { fpowi(base, exponent) };
        output.store(result);
    }

    #[cutile::entry()]
    fn invalid_exp_rounding(output: &mut Tensor<f32, { [4] }>) {
        let value: Tile<f32, { [4] }> = constant(0.0, shape![4]);
        let result = unsafe { exp_with_rounding(value, rounding::NearestAway) };
        output.store(result);
    }

    #[cutile::entry()]
    fn loop_return(output: &mut Tensor<i32, { [4] }>, exit: i32) {
        loop {
            if exit != 0 {
                return;
            }
            break;
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        output.store(value);
    }

    #[cutile::entry()]
    #[allow(unreachable_code, unused_variables)]
    fn folded_loop_return(output: &mut Tensor<i32, { [4] }>) {
        loop {
            if true {
                return;
            }
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        output.store(value);
    }

    #[cutile::entry()]
    fn for_return(output: &mut Tensor<i32, { [4] }>) {
        for _i in 0..2 {
            return;
        }
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        output.store(value);
    }

    fn helper_with_return(exit: i32) {
        loop {
            if exit != 0 {
                return;
            }
            break;
        }
    }

    #[cutile::entry()]
    fn inline_return(output: &mut Tensor<i32, { [4] }>, exit: i32) {
        helper_with_return(exit);
        let value: Tile<i32, { [4] }> = constant(9, shape![4]);
        output.store(value);
    }

    #[cutile::entry()]
    unsafe fn wide_assumptions() {
        let value: Tile<i64, { [1, 1, 1, 1, 1] }> = constant(0i64, shape![1, 1, 1, 1, 1]);
        let value = unsafe { assume_div_by_raw(value, 4611686018427387904u64, None, None) };
        let value = unsafe {
            assume_bounds_raw(
                value,
                Some(-9223372036854775808i64),
                Some(9223372036854775807i64),
            )
        };
        let _value = unsafe { assume_same_elements_raw(value, [1i64, 1i64, 1i64, 1i64, 1i64]) };
    }

    #[cutile::entry()]
    fn scale_format(output: &mut Tensor<f32, { [16, 16] }>) {
        let packed: Tile<u8, { [128] }> = constant(0u8, shape![128]);
        let unpacked: Tile<f4e2m1fn, { [256] }> = unpack(packed);
        let a: Tile<f4e2m1fn, { [16, 16] }> = unpacked.reshape(shape![16, 16]);
        let acc: Tile<f32, { [16, 16] }> = constant(0.0, shape![16, 16]);
        let sa: Tile<f8e5m3fnu, { [16, 1] }> = constant(f8e5m3fnu::ZERO, shape![16, 1]);
        let sb: Tile<f8e5m3fnu, { [1, 16] }> = constant(f8e5m3fnu::ZERO, shape![1, 16]);
        let result = mmaf_scaled(a, a, acc, sa, sb);
        output.store(result);
    }

    #[cutile::entry()]
    unsafe fn strided_options(input: &Tensor<f32, { [8] }>, output: &Tensor<f32, { [8] }>) {
        let src: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(input, shape![4], padding::Zero) };
        let dst: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(output, shape![4], padding::None) };
        let (value, loaded): (Tile<f32, { [4] }>, Token) = unsafe {
            load_view_raw(
                &src,
                [0i32],
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false],
                shape![4],
            )
        };
        let stored: Token = unsafe {
            store_view_raw(
                &dst,
                value,
                [0i32],
                Some(loaded),
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false],
            )
        };
        let _reduced: Token = unsafe {
            atomic_red_view_raw(
                &dst,
                value,
                [0i32],
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(stored),
            )
        };
    }

    #[cutile::entry()]
    unsafe fn gather_rank_three(
        input: &Tensor<f32, { [2, 4, 2] }>,
        output: &Tensor<f32, { [2, 4, 2] }>,
    ) {
        let src: GatherScatterView<f32, { [1, 4, 1] }, 1> =
            unsafe { make_gather_scatter_view(input, shape![1, 4, 1], padding::Zero) };
        let dst: GatherScatterView<f32, { [1, 4, 1] }, 1> =
            unsafe { make_gather_scatter_view(output, shape![1, 4, 1], padding::None) };
        let sparse: Tile<i32, { [4] }> = iota(shape![4]);
        let (value, loaded): (Tile<f32, { [1, 4, 1] }>, Token) = unsafe {
            load_view_raw(
                &src,
                (0i32, sparse, 0i32),
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false, false],
                shape![1, 4, 1],
            )
        };
        let _stored: Token = unsafe {
            store_view_raw(
                &dst,
                value,
                (0i32, sparse, 0i32),
                Some(loaded),
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [false, false, false],
            )
        };
    }

    #[cutile::entry()]
    unsafe fn allocation_and_conversion(output: &mut Tensor<i32, { [4] }>) {
        let _pointer: PointerTile<*mut f32, { [] }> =
            unsafe { alloca(4i64, 16i64, allocation::Local) };
        let x: Tile<f32, { [4] }> = constant(1.9, shape![4]);
        let y: Tile<i32, { [4] }> = ftoi(x, rounding::NearestIntToZero);
        output.store(y);
    }

    #[cutile::entry()]
    fn saturated_conversion(output: &mut Tensor<i32, { [4] }>) {
        let x: Tile<f32, { [4] }> = constant(1.9, shape![4]);
        let y: Tile<i32, { [4] }> = unsafe { ftoi_saturating(x, saturating::Enabled) };
        output.store(y);
    }

    #[cutile::entry()]
    unsafe fn classified_pointer(input: *mut f32) {
        let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(input);
        let classified: PointerTile<*mut f32, { [] }> = unsafe { ptr_with_attr_none(base) };
        let strides: Array<{ [1] }> = Array::<{ [1] }> { dims: &[1] };
        let tensor: Tensor<f32, { [8] }> =
            unsafe { make_tensor_view(classified, shape![8], strides, new_token_unordered()) };
        let _dims: [i32; 1] = get_tensor_shape(&tensor);
        let view: StridedView<f32, { [4] }, { [2] }, { [0] }> =
            unsafe { make_strided_view(&tensor, shape![4], padding::None) };
        let value: Tile<f32, { [4] }> = constant(2.0, shape![4]);
        let _stored: Token = unsafe {
            store_view_raw(
                &view,
                value,
                [0i32],
                None,
                ordering::Weak,
                scope::Device,
                tma::Disabled,
                None,
                [true],
            )
        };
    }
}

#[test]
fn insert_refuses_13_3_before_tileiras_with_source_span() {
    common::with_test_stack(|| {
        let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", "insert_kernel")
            .strides(&[("output", &[1])])
            .target("sm_120")
            .bytecode_version(BytecodeVersion::V13_3)
            .compile()
            .err()
            .expect("13.4 op must be refused");
        let cutile_compiler::error::JITError::Located(message, location) = error else {
            panic!("capability refusal must retain the call-site span");
        };
        assert_eq!(
            message,
            "cuda_tile.insert requires Tile IR 13.4 or newer; selected Tile IR 13.3, target sm_120"
        );
        assert!(location.file.ends_with("tile_ir_capabilities.rs"));
        assert!(location.line > 0);
    });
}

#[test]
fn scale_format_refuses_sm_120_even_with_13_4() {
    common::with_test_stack(|| {
        let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", "scale_format")
            .strides(&[("output", &[16, 1])])
            .target("sm_120")
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()
            .err()
            .expect("unsupported scale configuration must be refused");
        let cutile_compiler::error::JITError::Located(message, location) = error else {
            panic!("capability refusal must retain the call-site span");
        };
        assert_eq!(message, "cuda_tile.mmaf_scaled with f8E5M3FNU scales requires sm_107; selected Tile IR 13.4, target sm_120");
        assert!(location.file.ends_with("tile_ir_capabilities.rs"));
        assert!(location.line > 0);
    });
}

#[test]
fn scale_format_accepts_sm107_with_13_4() {
    common::with_test_stack(|| {
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "scale_format")
            .strides(&[("output", &[16, 1])])
            .target("sm_107")
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()
            .unwrap()
            .bytecode()
            .unwrap();
    });
}

#[test]
fn raw_13_4_ops_compile_and_serialize_without_a_driver() {
    common::with_test_stack(|| {
        for (name, strides, expected) in [
            ("insert_kernel", vec![("output", &[1][..])], "insert"),
            ("integer_power", vec![("output", &[1][..])], "fpowi"),
            ("pdl_tokens", vec![], "gdc_wait_tko"),
            ("classified_pointer", vec![], "ptr_attr<none>"),
        ] {
            let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", name)
                .strides(&strides)
                .target("sm_120")
                .bytecode_version(BytecodeVersion::V13_4)
                .compile()
                .unwrap();
            assert!(artifacts.ir_text().contains(expected));
            let bytes = artifacts.bytecode().unwrap();
            assert!(cutile_ir::decode_bytecode(&bytes)
                .unwrap()
                .contains("v13.4"));
        }
    });
}

#[test]
fn raw_13_3_gaps_compile_for_both_wire_versions() {
    common::with_test_stack(|| {
        for version in [BytecodeVersion::V13_3, BytecodeVersion::V13_4] {
            for (name, strides) in [
                (
                    "strided_options",
                    vec![("input", &[1][..]), ("output", &[1][..])],
                ),
                (
                    "gather_rank_three",
                    vec![("input", &[8, 2, 1][..]), ("output", &[8, 2, 1][..])],
                ),
                ("allocation_and_conversion", vec![("output", &[1][..])]),
                ("wide_assumptions", vec![]),
            ] {
                let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", name)
                    .strides(&strides)
                    .target("sm_120")
                    .bytecode_version(version)
                    .compile()
                    .unwrap_or_else(|e| panic!("{name} {version}: {e}"));
                artifacts.bytecode().unwrap();
            }
        }
        KernelCompiler::new(
            kernels::__module_ast_self,
            "kernels",
            "saturated_conversion",
        )
        .strides(&[("output", &[1])])
        .target("sm_120")
        .bytecode_version(BytecodeVersion::V13_4)
        .compile()
        .unwrap()
        .bytecode()
        .unwrap();
    });
}

#[test]
fn loop_return_is_versioned_and_does_not_escape_inline_helpers() {
    common::with_test_stack(|| {
        let compile = |name, version| {
            KernelCompiler::new(kernels::__module_ast_self, "kernels", name)
                .strides(&[("output", &[1])])
                .target("sm_120")
                .bytecode_version(version)
                .compile()
        };
        for name in ["loop_return", "folded_loop_return"] {
            compile(name, BytecodeVersion::V13_4)
                .unwrap()
                .bytecode()
                .unwrap();
            let error = compile(name, BytecodeVersion::V13_3)
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains("cuda_tile.return inside loop requires Tile IR 13.4"),
                "{error}"
            );
        }
        for name in ["for_return", "inline_return"] {
            let error = compile(name, BytecodeVersion::V13_4)
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains("`return` is only supported at the top level"),
                "{error}"
            );
        }
    });
}

#[test]
fn explicit_math_rounding_does_not_silently_use_the_default() {
    common::with_test_stack(|| {
        let error = KernelCompiler::new(
            kernels::__module_ast_self,
            "kernels",
            "invalid_exp_rounding",
        )
        .strides(&[("output", &[1])])
        .target("sm_120")
        .bytecode_version(BytecodeVersion::V13_4)
        .compile()
        .err()
        .expect("invalid explicit rounding must be rejected");
        assert!(
            error
                .to_string()
                .contains("explicit exp/tanh rounding must be rounding::Approx or rounding::Full"),
            "{error}"
        );
    });
}
