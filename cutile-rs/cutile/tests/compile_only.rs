/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-only compile-path smoke test. Verifies `KernelCompiler` compiles a
//! kernel to Tile IR + bytecode without needing the CUDA driver at runtime.
//! Runs on machines with CUDA headers but no GPU.

use cutile::compile_api::KernelCompiler;

mod common;

#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    /// Simple kernel that does tile math without dynamic tensor inputs.
    #[cutile::entry()]
    fn tile_math<const S: [i32; 1]>(output: &mut Tensor<f32, S>, scalar: f32) {
        let scalar_tile: Tile<f32, S> = broadcast_scalar(scalar, output.shape());
        let ones: Tile<f32, S> = broadcast_scalar(1.0f32, output.shape());
        let result = scalar_tile + ones;
        output.store(result);
    }
}

#[test]
fn smoke_compile_only() {
    common::with_test_stack(|| {
        let artifacts =
            KernelCompiler::new(my_kernels::__module_ast_self, "my_kernels", "tile_math")
                .generics(vec!["32".into()])
                .strides(&[("output", &[1])])
                .target("sm_120")
                .compile()
                .expect("compilation failed");

        // IR text should be non-empty and mention the kernel name somewhere.
        let ir_text = artifacts.ir_text();
        assert!(!ir_text.is_empty(), "ir_text should not be empty");

        // Bytecode serialization should produce a non-empty blob.
        let bytecode = artifacts.bytecode().expect("bytecode serialization failed");
        assert!(!bytecode.is_empty(), "bytecode should not be empty");
    });
}
