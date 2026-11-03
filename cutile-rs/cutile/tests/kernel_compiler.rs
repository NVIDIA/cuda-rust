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
