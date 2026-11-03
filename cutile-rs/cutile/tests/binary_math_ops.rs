/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod binary_math_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn minmax_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test min and max operations
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let rem_result: Tile<f32, S> = remf(x, y);
        let max_result: Tile<f32, S> = maxf(rem_result, y, nan::Disabled, ftz::Disabled);
        let min_result: Tile<f32, S> = minf(max_result, y, nan::Disabled, ftz::Disabled);
        output.store(min_result);
    }

    #[cutile::entry()]
    fn select_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test select operation
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);

        let mask: Tile<bool, S> = cmpf(x, y, predicate::LessThan, cmp_ordering::Ordered);
        let result: Tile<f32, S> = select(mask, x, y);
        output.store(result);
    }

    #[cutile::entry()]
    fn bf16_binary_arith_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Covers bf16 binary arithmetic lowering
        let x: Tile<bf16, S> = load_tile_mut(output);
        let y: Tile<bf16, S> = load_tile_mut(output);

        let sum: Tile<bf16, S> = x + y;
        let product: Tile<bf16, S> = sum * y;
        let result: Tile<bf16, S> = product / x;
        output.store(result);
    }

    // Exercises the trait-dispatched `addf` — Phase B proof point.
    // Direct call; resolves via the free-fn wrapper emitted by the
    // trait-dispatch macro (not the variadic `addf__N_N` rewriter).
    #[cutile::entry()]
    fn addf_shadow_dispatch_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = addf(x, y, rounding::NearestEven, ftz::Disabled);
        output.store(result);
    }

    // Nested addf: inner result flows into outer call without an intermediate
    // let-binding. This is the exact case today's variadic-expansion rewriter
    // fails to desugar — under trait dispatch it resolves via rustc inference.
    #[cutile::entry()]
    fn addf_shadow_dispatch_nested_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let z: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = addf(
            addf(x, y, rounding::NearestEven, ftz::Disabled),
            z,
            rounding::NearestEven,
            ftz::Disabled,
        );
        output.store(result);
    }

    // Case-3b proof point: reshape has two independent CGAs (S and R), so it
    // produces a rank-changing return type via an associated `Out`. Verifies
    // the multi-CGA code path through the trait-dispatch emitter.
    #[cutile::entry()]
    fn reshape_shadow_dispatch_kernel<const S: [i32; 2]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let target: Shape<{ [128] }> = Shape::<{ [128] }> { dims: &[128i32] };
        let flat: Tile<f32, { [128] }> = reshape(x, target);
        let back_shape: Shape<S> = output.shape();
        let back: Tile<f32, S> = reshape(flat, back_shape);
        output.store(back);
    }

    // Case-3c proof point: reduce_sum's return CGA (R) is not in any arg,
    // so `Out` must be a trait generic inferred from the return-type ascription.
    #[cutile::entry()]
    fn reduce_sum_shadow_dispatch_kernel<const S: [i32; 2]>(
        input: &mut Tensor<f32, S>,
        output: &mut Tensor<f32, { [1, 1] }>,
    ) {
        let tile: Tile<f32, S> = load_tile_mut(input);
        let reduced: Tile<f32, { [1, 1] }> = reduce_sum(tile, 1i32);
        output.store(reduced);
    }
}

use binary_math_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "binary_math_ops_module",
        function_name,
        generics,
        strides,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed.")
}

#[test]
fn compile_minmax() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("minmax_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== MIN/MAX MLIR ===\n{}", module_op_str);

        let expected_ops = ["remf", "maxf", "minf"];
        for op in expected_ops {
            assert!(
                module_op_str.contains(format!("= {}", op).as_str()),
                "Expected {} operation in MLIR output",
                op
            );
        }

        println!(
            "\n✓ All {} min/max operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_select() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("select_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== SELECT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= cmpf"),
            "Expected cmpf operation in MLIR output"
        );
        assert!(
            module_op_str.contains("select"),
            "Expected select operation in MLIR output"
        );

        println!("\n✓ select operation verified in MLIR output");
    });
}

#[test]
fn compile_bf16_binary_arith() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "bf16_binary_arith_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== BF16 BINARY ARITH MLIR ===\n{}", module_op_str);

        for op in ["addf", "mulf", "divf"] {
            assert!(
                module_op_str.contains(format!("= {}", op).as_str()),
                "Expected {} operation in MLIR output",
                op
            );
        }
        assert!(
            module_op_str.contains("bf16"),
            "Expected bf16 type in MLIR output"
        );
    });
}

#[test]
fn compile_addf_shadow_dispatch() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "addf_shadow_dispatch_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== ADDF TRAIT DISPATCH MLIR ===\n{}", module_op_str);
        assert!(
            module_op_str.contains("= addf"),
            "Expected addf operation in MLIR output"
        );
    });
}

#[test]
fn compile_reshape_shadow_dispatch() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "reshape_shadow_dispatch_kernel",
            &[8.to_string(), 16.to_string()],
            &[("output", &[8, 16])],
        );
        println!("\n=== RESHAPE TRAIT DISPATCH MLIR ===\n{}", module_op_str);
        assert!(
            module_op_str.contains("reshape"),
            "Expected reshape operation in MLIR output"
        );
    });
}

#[test]
fn compile_reduce_sum_shadow_dispatch() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "reduce_sum_shadow_dispatch_kernel",
            &[8.to_string(), 16.to_string()],
            &[("input", &[8, 16]), ("output", &[1, 1])],
        );
        println!(
            "\n=== REDUCE_SUM TRAIT DISPATCH MLIR ===\n{}",
            module_op_str
        );
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );
    });
}

#[test]
fn compile_addf_shadow_dispatch_nested() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "addf_shadow_dispatch_nested_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!(
            "\n=== ADDF TRAIT DISPATCH NESTED MLIR ===\n{}",
            module_op_str
        );
        // Nested call site should lower to two `addf` ops in the MLIR.
        let addf_count = module_op_str.matches("= addf").count();
        assert!(
            addf_count >= 2,
            "Expected at least 2 addf operations (nested), got {}",
            addf_count
        );
    });
}
