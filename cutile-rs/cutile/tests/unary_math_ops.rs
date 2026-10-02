/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod unary_math_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn unary_math_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test all floating-point unary math operations
        let x: Tile<f32, S> = load_tile_mut(output);

        // Arithmetic operations
        let t1: Tile<f32, S> = absf(x); // absolute value (float)
        let t2: Tile<f32, S> = negf(t1); // negation (float)
        let t3: Tile<f32, S> = rsqrt(t2, ftz::Disabled); // reciprocal square root

        // Exponential and logarithmic
        let t4: Tile<f32, S> = exp(t3); // exponential (e^x)
        let t5: Tile<f32, S> = exp2(t4, ftz::Disabled); // base-2 exponential (2^x)
        let t6: Tile<f32, S> = log(t5); // natural log
        let t7: Tile<f32, S> = log2(t6); // base-2 log

        // Trigonometric
        let t8: Tile<f32, S> = sin(t7); // sine
        let t9: Tile<f32, S> = cos(t8); // cosine
        let t10: Tile<f32, S> = tan(t9); // tangent

        // Hyperbolic trigonometric
        let t11: Tile<f32, S> = sinh(t10); // hyperbolic sine
        let t12: Tile<f32, S> = cosh(t11); // hyperbolic cosine
        let t13: Tile<f32, S> = tanh(t12); // hyperbolic tangent

        // Rounding
        let t14: Tile<f32, S> = ceil(t13); // ceiling
        let result: Tile<f32, S> = floor(t14); // floor

        output.store(result);
    }

    #[cutile::entry()]
    fn integer_unary_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer unary operations
        let x: Tile<i64, S> = load_tile_mut(output);

        // Integer arithmetic operations
        let t1: Tile<i64, S> = absi(x); // absolute value (int)
        let result: Tile<i64, S> = negi(t1, overflow::None); // negation (int)

        output.store(result);
    }

    #[cutile::entry()]
    fn sqrt_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test sqrt operation
        let x: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = sqrt(x, rounding::NegativeInf, ftz::Disabled); // square root
        output.store(result);
    }

    #[cutile::entry()]
    fn fma_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test fused multiply-add operation
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let z: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = fma(x, y, z, rounding::NearestEven, ftz::Disabled); // x * y + z
        output.store(result);
    }

    #[cutile::entry()]
    fn pow_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test power operation
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = pow(x, y); // x^y
        output.store(result);
    }

    #[cutile::entry()]
    fn atan2_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = atan2(x, y);
        output.store(result);
    }

    #[cutile::entry()]
    fn exp2_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = exp2(x, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn maxf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = maxf(x, y, nan::Disabled, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn minf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = minf(x, y, nan::Disabled, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn addf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = addf(x, y, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn subf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = subf(x, y, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn mulf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = mulf(x, y, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn divf_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = divf(x, y, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn fma_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let z: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = fma(x, y, z, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn rsqrt_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = rsqrt(x, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn sqrt_ftz_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = sqrt(x, rounding::NearestEven, ftz::Enabled);
        output.store(result);
    }

    #[cutile::entry()]
    fn unary_math_ops_bf16_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Verifies bf16 unary math operation lowering
        let x: Tile<bf16, S> = load_tile_mut(output);
        let t1: Tile<bf16, S> = absf(x);
        let t2: Tile<bf16, S> = negf(t1);
        let t3: Tile<bf16, S> = exp(t2);
        let result: Tile<bf16, S> = floor(t3);
        output.store(result);
    }
}

use unary_math_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "unary_math_ops_module",
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
fn compile_unary_math_ops() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "unary_math_ops_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== UNARY MATH OPS MLIR ===\n{}", module_op_str);

        let expected_ops = [
            "absf", "negf", "rsqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "sinh",
            "cosh", "tanh", "ceil", "floor",
        ];
        for op in expected_ops {
            assert!(
                module_op_str.contains(op),
                "Expected {} operation in MLIR output",
                op
            );
        }

        println!(
            "\n✓ All {} floating-point unary math operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_integer_unary_ops() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "integer_unary_ops_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== INTEGER UNARY OPS MLIR ===\n{}", module_op_str);

        let expected_ops = ["absi", "negi"];
        for op in expected_ops {
            assert!(
                module_op_str.contains(op),
                "Expected {} operation in MLIR output",
                op
            );
        }

        println!(
            "\n✓ All {} integer unary math operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_sqrt() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("sqrt_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== SQRT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("sqrt"),
            "Expected sqrt operation in MLIR output"
        );

        println!("\n✓ sqrt operation verified in MLIR output");
    });
}

#[test]
fn compile_fma() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("fma_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== FMA MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("fma"),
            "Expected fma operation in MLIR output"
        );

        println!("\n✓ fma operation verified in MLIR output");
    });
}

#[test]
fn compile_pow() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("pow_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== POW MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= pow"),
            "Expected pow operation in MLIR output"
        );

        println!("\n✓ pow operation verified in MLIR output");
    });
}

#[test]
fn compile_atan2() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("atan2_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== ATAN2 MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= atan2"),
            "Expected atan2 operation in MLIR output"
        );
    });
}

#[test]
fn compile_exp2_ftz() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("exp2_ftz_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== EXP2 FTZ MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("exp2"),
            "Expected exp2 operation in MLIR output"
        );
        assert!(
            module_op_str.contains("flush_to_zero"),
            "Expected flush_to_zero attribute in MLIR output"
        );
    });
}

#[test]
fn compile_maxf_ftz() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("maxf_ftz_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== MAXF FTZ MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("maxf"),
            "Expected maxf operation in MLIR output"
        );
        assert!(
            module_op_str.contains("flush_to_zero"),
            "Expected flush_to_zero attribute in MLIR output"
        );
    });
}

#[test]
fn compile_minf_ftz() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("minf_ftz_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== MINF FTZ MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("minf"),
            "Expected minf operation in MLIR output"
        );
        assert!(
            module_op_str.contains("flush_to_zero"),
            "Expected flush_to_zero attribute in MLIR output"
        );
    });
}

/// Helper: compile a kernel and assert it contains the expected op name and flush_to_zero.
fn assert_ftz_in_mlir(kernel_name: &'static str, expected_op: &'static str) {
    common::with_test_stack(move || {
        let module_op_str = compile_ir(kernel_name, &[128.to_string()], &[("output", &[1])]);
        println!("\n=== {kernel_name} MLIR ===\n{module_op_str}");

        assert!(
            module_op_str.contains(expected_op),
            "Expected {expected_op} operation in MLIR output"
        );
        assert!(
            module_op_str.contains("flush_to_zero"),
            "Expected flush_to_zero attribute in MLIR output"
        );
    });
}

#[test]
fn compile_addf_ftz() {
    assert_ftz_in_mlir("addf_ftz_kernel", "addf");
}

#[test]
fn compile_subf_ftz() {
    assert_ftz_in_mlir("subf_ftz_kernel", "subf");
}

#[test]
fn compile_mulf_ftz() {
    assert_ftz_in_mlir("mulf_ftz_kernel", "mulf");
}

#[test]
fn compile_divf_ftz() {
    assert_ftz_in_mlir("divf_ftz_kernel", "divf");
}

#[test]
fn compile_fma_ftz() {
    assert_ftz_in_mlir("fma_ftz_kernel", "fma");
}

#[test]
fn compile_rsqrt_ftz() {
    assert_ftz_in_mlir("rsqrt_ftz_kernel", "rsqrt");
}

#[test]
fn compile_sqrt_ftz() {
    assert_ftz_in_mlir("sqrt_ftz_kernel", "sqrt");
}

#[test]
fn compile_unary_math_ops_bf16() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "unary_math_ops_bf16_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== BF16 UNARY MATH OPS MLIR ===\n{}", module_op_str);

        for op in ["absf", "negf", "exp", "floor"] {
            assert!(
                module_op_str.contains(op),
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
