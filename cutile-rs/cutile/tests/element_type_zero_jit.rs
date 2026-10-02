/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT-compile tests for `T::ZERO` usage through a generic type parameter.
//!
//! Verifies the cutile-compiler resolves `T::ZERO` against the monomorphized
//! type in the `constant` op's dense attribute. This was broken until the
//! `generic_args.inst_types` lookup was added to the Expr::Path branch in
//! `compile_cuda_tile_op.rs`.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod zero_kernel_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn zero_fill<T: ElementType, const BM: i32, const BN: i32>(out: &mut Tensor<T, { [BM, BN] }>) {
        let z: Tile<T, { [BM, BN] }> = constant(T::ZERO, shape![BM, BN]);
        out.store(z);
    }
}

use zero_kernel_module::__module_ast_self;

fn compile_zero_fill(ty: &str) -> String {
    let module_op_str = common::compile_to_ir(
        __module_ast_self,
        "zero_kernel_module",
        "zero_fill",
        &[ty.to_string(), 64.to_string(), 64.to_string()],
        &[("out", &[64, 1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed to compile");
    println!("=== MLIR for zero_fill<{ty}> ===\n{module_op_str}");
    module_op_str
}

#[test]
fn zero_fill_f32_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("f32");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("f32"), "expected f32 element type in MLIR");
    });
}

#[test]
fn zero_fill_f16_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("f16");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("f16"), "expected f16 element type in MLIR");
    });
}

#[test]
fn zero_fill_bf16_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("bf16");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("bf16"), "expected bf16 element type in MLIR");
    });
}

#[test]
fn zero_fill_i32_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("i32");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("i32"), "expected i32 element type in MLIR");
    });
}
