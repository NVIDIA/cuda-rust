/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::cuda_tile_runtime_utils::{
    serialize_tile_ir_bytecode, tileiras_fingerprint, TileirasOptions,
};
use cutile::jit_cache::l2_key;
use cutile::tile_kernel::CompileOptions;

mod common;

#[cutile::module]
mod compile_only_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn tile_math<const S: [i32; 1]>(output: &mut Tensor<f32, S>, scalar: f32) {
        let scalar_tile: Tile<f32, S> = broadcast_scalar(scalar, output.shape());
        let ones: Tile<f32, S> = broadcast_scalar(1.0f32, output.shape());
        output.store(scalar_tile + ones);
    }
}

#[test]
fn kernel_compiler_emits_ir_and_bytecode() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            compile_only_module::__module_ast_self,
            "compile_only_module",
            "tile_math",
        )
        .generics(vec!["32".into()])
        .strides(&[("output", &[1])])
        .target("sm_120")
        .compile()
        .expect("compile-only kernel compilation failed");

        let ir = artifacts.ir_text();
        assert!(!ir.trim().is_empty(), "expected non-empty Tile IR");
        assert!(
            ir.contains("entry"),
            "expected the compiled IR to contain an entry op.\nIR:\n{ir}"
        );

        let bytecode = artifacts
            .bytecode()
            .expect("bytecode serialization should succeed");
        assert!(!bytecode.is_empty(), "expected non-empty bytecode");
        assert_eq!(
            &bytecode[..8],
            &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00],
            "expected TileIR bytecode magic"
        );
    });
}

fn tile_math_compiler() -> KernelCompiler<fn() -> cutile::cutile_compiler::ast::Module> {
    KernelCompiler::new(
        compile_only_module::__module_ast_self as fn() -> cutile::cutile_compiler::ast::Module,
        "compile_only_module",
        "tile_math",
    )
    .generics(vec!["32".into()])
    .strides(&[("output", &[1])])
    .target("sm_120")
}

#[test]
fn kernel_compiler_l2_cache_key_matches_runtime_derivation() {
    common::with_test_stack(|| {
        let stats_before = cutile::jit_cache::stats();
        let backend_before = cutile::jit_cache::jit_backend_compile_count();
        let actual = tile_math_compiler()
            .l2_cache_key()
            .expect("cache-key derivation failed");
        assert_eq!(cutile::jit_cache::stats(), stats_before);
        assert_eq!(
            cutile::jit_cache::jit_backend_compile_count(),
            backend_before,
            "compile-only cache-key derivation must not run tileiras"
        );

        let artifacts = tile_math_compiler()
            .compile()
            .expect("compile-only kernel compilation failed");
        let (bytecode, version) = serialize_tile_ir_bytecode(artifacts.module())
            .expect("runtime bytecode serialization failed");
        let expected = l2_key(
            &bytecode,
            version,
            "sm_120",
            &TileirasOptions::from_compile_options(&CompileOptions::default()),
            tileiras_fingerprint(),
        );

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 64);
        assert!(actual.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(actual, actual.to_ascii_lowercase());
    });
}

#[test]
fn kernel_compiler_l2_cache_key_includes_target() {
    common::with_test_stack(|| {
        let sm_120 = tile_math_compiler()
            .l2_cache_key()
            .expect("sm_120 cache-key derivation failed");
        let sm_100 = KernelCompiler::new(
            compile_only_module::__module_ast_self,
            "compile_only_module",
            "tile_math",
        )
        .generics(vec!["32".into()])
        .strides(&[("output", &[1])])
        .target("sm_100")
        .l2_cache_key()
        .expect("sm_100 cache-key derivation failed");

        assert_ne!(sm_120, sm_100);
    });
}

#[cutile::module]
mod literal_constants {
    use cutile::core::*;

    #[cutile::entry()]
    fn false_mask(output: &mut Tensor<i32, { [32] }>) {
        let shape = output.shape();
        let mask: Tile<bool, { [32] }> = constant(false, shape);
        let one: Tile<i32, { [32] }> = constant(1i32, shape);
        let zero: Tile<i32, { [32] }> = constant(0i32, shape);
        let value: Tile<i32, { [32] }> = select(mask, one, zero);
        output.store(value);
    }

    #[cutile::entry()]
    fn true_mask(output: &mut Tensor<i32, { [32] }>) {
        let shape = output.shape();
        let mask: Tile<bool, { [32] }> = constant(true, shape);
        let one: Tile<i32, { [32] }> = constant(1i32, shape);
        let zero: Tile<i32, { [32] }> = constant(0i32, shape);
        let value: Tile<i32, { [32] }> = select(mask, one, zero);
        output.store(value);
    }

    #[cutile::entry()]
    fn unsigned_8(output: &mut Tensor<u8, { [32] }>) {
        let value: Tile<u8, { [32] }> = constant(255u8, output.shape());
        output.store(value);
    }

    #[cutile::entry()]
    fn unsigned_16(output: &mut Tensor<u16, { [32] }>) {
        let value: Tile<u16, { [32] }> = constant(65535u16, output.shape());
        output.store(value);
    }

    #[cutile::entry()]
    fn unsigned_32(output: &mut Tensor<u32, { [32] }>) {
        let value: Tile<u32, { [32] }> = constant(4294967295u32, output.shape());
        output.store(value);
    }

    #[cutile::entry()]
    fn unsigned_64(output: &mut Tensor<u64, { [32] }>) {
        let value: Tile<u64, { [32] }> = constant(18446744073709551615u64, output.shape());
        output.store(value);
    }

    #[cutile::entry()]
    fn scalar_i64_min(output: &mut Tensor<i64, { [32] }>) {
        let value = -9223372036854775808i64;
        output.store(value.broadcast(output.shape()));
    }

    #[cutile::entry()]
    fn scalar_u64(output: &mut Tensor<u64, { [32] }>) {
        let value = 0x9e3779b97f4a7c15u64;
        output.store(value.broadcast(output.shape()));
    }
}

fn check_literal_constant(kernel: &str, expected: &str) {
    let artifacts = KernelCompiler::new(
        literal_constants::__module_ast_self,
        "literal_constants",
        kernel,
    )
    .strides(&[("output", &[1])])
    .target("sm_80")
    .compile()
    .expect("literal kernel should compile without a GPU");
    let ir = artifacts.ir_text();
    assert!(ir.contains(expected), "{kernel}: missing {expected}\n{ir}");
    artifacts
        .bytecode()
        .expect("literal kernel should serialize");
}

#[test]
fn kernel_compiler_preserves_boolean_constants() {
    common::with_test_stack(|| {
        check_literal_constant("true_mask", "constant <i1: 1>");
        check_literal_constant("false_mask", "constant <i1: 0>");
    });
}

#[test]
fn kernel_compiler_preserves_unsigned_constants() {
    common::with_test_stack(|| {
        for (kernel, expected) in [
            ("unsigned_8", "constant <i8: -1>"),
            ("unsigned_16", "constant <i16: -1>"),
            ("unsigned_32", "constant <i32: -1>"),
            ("unsigned_64", "constant <i64: -1>"),
            ("scalar_u64", "constant <i64: -7046029254386353131>"),
            ("scalar_i64_min", "constant <i64: -9223372036854775808>"),
        ] {
            check_literal_constant(kernel, expected);
        }
    });
}

#[test]
fn kernel_compiler_rejects_out_of_range_integer_literals() {
    common::with_test_stack(|| {
        for body in [
            "let value: Tile<u8, { [32] }> = constant(256u8, output.shape()); output.store(value);",
            "let value = 256u8; output.store(value.broadcast(output.shape()));",
            "let value = -129i8; let tile: Tile<i8, { [32] }> = value.broadcast(output.shape()); let result: Tile<u8, { [32] }> = bitcast(tile); output.store(result);",
        ] {
            // Deliberately invalid source bypasses rustc to check the JIT diagnostic.
            let source = format!(
                "mod probe {{ use cutile::core::*; #[cutile::entry()] fn kernel(output: &mut Tensor<u8, {{ [32] }}>) {{ {body} }} }}"
            );
            let result = KernelCompiler::new(
                || cutile_compiler::ast::Module::new("probe", syn::parse_str(&source).unwrap()),
                "probe",
                "kernel",
            )
            .strides(&[("output", &[1])])
            .target("sm_80")
            .compile();
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("out-of-range literal should be rejected: {body}"),
            };
            assert!(error.to_string().contains("out of range for i8"), "{error}");
        }
    });
}

#[test]
#[ignore = "requires a CUDA device"]
fn literal_constants_execute_on_gpu() {
    common::with_test_stack(|| {
        use cutile::{api, tensor::*, tile_kernel::DeviceOp};
        use std::sync::Arc;

        macro_rules! check {
            ($kernel:ident, $ty:ty, $expected:expr) => {{
                let input: Arc<Vec<$ty>> = Arc::new(vec![42; 32]);
                let output = api::copy_host_vec_to_device(&input).sync().unwrap();
                let (output,) = literal_constants::$kernel(output.partition([32]))
                    .sync()
                    .unwrap();
                let host: Vec<$ty> = output.unpartition().to_host_vec().sync().unwrap();
                assert_eq!(host, vec![$expected; 32], stringify!($kernel));
            }};
        }
        check!(false_mask, i32, 0);
        check!(true_mask, i32, 1);
        check!(unsigned_8, u8, u8::MAX);
        check!(unsigned_16, u16, u16::MAX);
        check!(unsigned_32, u32, u32::MAX);
        check!(unsigned_64, u64, u64::MAX);
        check!(scalar_u64, u64, 0x9e3779b97f4a7c15u64);
        check!(scalar_i64_min, i64, i64::MIN);
    });
}
