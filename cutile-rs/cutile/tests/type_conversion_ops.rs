/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::{api::*, tensor::*, tile_kernel::*};
use cutile_compiler::compiler::utils::CompileOptions;
use half::bf16;
use std::sync::Arc;

mod common;

#[cutile::module]
mod type_conversion_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn conversion_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer conversion operations
        let x: Tile<i64, S> = load_tile_mut(output);
        // Truncate to i32, then extend back to i64
        let truncated: Tile<i32, S> = trunci(x, overflow::None);
        let extended: Tile<i64, S> = exti(truncated);
        output.store(extended);
    }

    #[cutile::entry()]
    fn ptr_conversion_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test pointer conversion operations
        let x: Tile<i64, S> = load_tile_mut(output);
        // Convert to pointer, cast pointer type, convert back to int
        let ptrs: PointerTile<*mut i64, S> = int_to_ptr(x);
        let ptrs_f32: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs);
        let ptrs_back: PointerTile<*mut i64, S> = ptr_to_ptr(ptrs_f32);
        let ints: Tile<i64, S> = ptr_to_int(ptrs_back);
        output.store(ints);
    }

    #[cutile::entry()]
    fn exti_unsigned_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer extension with unsigned types (zero extension)
        // Note: Using i64 tensor but operating on u32 tiles to test unsigned extension
        let x: Tile<i64, S> = load_tile_mut(output);
        // Truncate to u32, then extend back to i64 with unsigned (zero extension)
        let truncated: Tile<u32, S> = trunci(x, overflow::None);
        let extended: Tile<i64, S> = exti(truncated);
        output.store(extended);
    }

    #[cutile::entry()]
    fn bf16_conversion_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Exercises bf16 <-> f32 tile conversion lowering
        let x: Tile<bf16, S> = load_tile_mut(output);
        let upcast: Tile<f32, S> = convert_tile(x);
        let downcast: Tile<bf16, S> = convert_tile(upcast);
        output.store(downcast);
    }

    #[cutile::entry()]
    fn bf16_to_f32_conversion_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<bf16, { [-1] }>,
    ) {
        // Runtime test kernel for bf16 -> f32 tile conversion
        let x: Tile<bf16, S> = load_tile_like(input, output);
        let y: Tile<f32, S> = convert_tile(x);
        output.store(y);
    }

    #[cutile::entry()]
    fn f32_to_bf16_conversion_kernel<const S: [i32; 1]>(
        output: &mut Tensor<bf16, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        // Runtime test kernel for f32 -> bf16 tile conversion
        let x: Tile<f32, S> = load_tile_like(input, output);
        let y: Tile<bf16, S> = convert_tile(x);
        output.store(y);
    }

    #[cutile::entry()]
    fn unannotated_load_tile_like_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let x = load_tile_like(input, output);
        output.store(x);
    }

    #[cutile::entry()]
    fn explicit_conversion_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let narrowed: Tile<f16, S> = ftof(x, rounding::NearestEven);
        let widened: Tile<f32, S> = ftof(narrowed, rounding::NearestEven);
        let ints: Tile<i32, S> = ftoi(widened, rounding::Zero);
        let floats: Tile<f32, S> = itof(ints, rounding::NearestEven);
        output.store(floats);
    }
}

use type_conversion_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "type_conversion_ops_module",
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
use type_conversion_ops_module::bf16_conversion_kernel;
use type_conversion_ops_module::bf16_to_f32_conversion_kernel;
use type_conversion_ops_module::f32_to_bf16_conversion_kernel;

#[test]
fn compile_conversion_ops() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "conversion_ops_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== CONVERSION OPS MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= trunci"),
            "Expected trunci operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= exti"),
            "Expected exti operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attribute in exti operation"
        );

        println!("\n✓ trunci and exti operations verified in MLIR output");
    });
}

#[test]
fn compile_ptr_conversion() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "ptr_conversion_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== PTR CONVERSION MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= int_to_ptr"),
            "Expected int_to_ptr operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= ptr_to_ptr"),
            "Expected ptr_to_ptr operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= ptr_to_int"),
            "Expected ptr_to_int operation in MLIR output"
        );

        println!("\n✓ Pointer conversion operations verified in MLIR output");
    });
}

#[test]
fn compile_exti_unsigned() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "exti_unsigned_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== EXTI UNSIGNED MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= exti"),
            "Expected exti operation in MLIR output"
        );
        assert!(
            module_op_str.contains("unsigned"),
            "Expected unsigned signedness attribute (zero extension)"
        );

        println!("\n✓ exti with unsigned types (zero extension) verified in MLIR output");
    });
}

#[test]
fn compile_bf16_conversion() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "bf16_conversion_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== BF16 CONVERSION MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= ftof"),
            "Expected floating-point conversion operation in MLIR output"
        );
        assert!(
            module_op_str.contains("bf16"),
            "Expected bf16 type in MLIR output"
        );
    });
}

#[test]
fn compile_bf16_to_f32_load_tile_like() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "bf16_to_f32_conversion_kernel",
            &[128.to_string()],
            &[("output", &[1]), ("input", &[1])],
        );
        println!(
            "\n=== BF16 -> F32 LOAD TILE LIKE MLIR ===\n{}",
            module_op_str
        );

        assert!(
            module_op_str.contains("load_view_tko"),
            "Expected load_view_tko operation in MLIR output"
        );
        assert!(
            module_op_str.contains("tile<128xbf16>"),
            "Expected source tile type in MLIR output"
        );
        assert!(
            module_op_str.contains("tile<128xf32>"),
            "Expected converted tile type in MLIR output"
        );
    });
}

#[test]
fn compile_unannotated_load_tile_like() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "unannotated_load_tile_like_kernel",
            &[128.to_string()],
            &[("output", &[1]), ("input", &[1])],
        );
        println!(
            "\n=== UNANNOTATED LOAD TILE LIKE MLIR ===\n{}",
            module_op_str
        );

        assert!(
            module_op_str.contains("load_view_tko"),
            "Expected load_view_tko operation in MLIR output"
        );
        assert!(
            module_op_str.contains("tile<128xf32>"),
            "Expected statically shaped tile type in MLIR output"
        );
        assert!(
            !module_op_str.contains("-9223372036854775808"),
            "Did not expect dynamic-shape sentinel in load_tile_like result"
        );
    });
}

#[test]
fn compile_explicit_conversion_ops() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "explicit_conversion_ops_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== EXPLICIT CONVERSION OPS MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.matches("= ftof").count() >= 2,
            "Expected two ftof operations in MLIR output"
        );
        assert!(
            module_op_str.contains("= ftoi"),
            "Expected ftoi operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= itof"),
            "Expected itof operation in MLIR output"
        );
    });
}

#[test]
fn execute_bf16_f32_roundtrip() {
    common::with_test_stack(|| {
        let input_host = Arc::new(vec![
            bf16::from_f32(-3.5),
            bf16::from_f32(-1.0),
            bf16::from_f32(-0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.125),
            bf16::from_f32(0.1),
            bf16::from_f32(1.1),
            bf16::from_f32(42.0),
        ]);

        let input: Tensor<bf16> = copy_host_vec_to_device(&input_host)
            .sync()
            .expect("Failed.");
        let (result,) = bf16_conversion_kernel(input.partition([4]))
            .sync()
            .expect("Failed.");

        let result_host: Vec<bf16> = result.unpartition().to_host_vec().sync().expect("Failed.");

        // This kernel performs bf16 -> f32 -> bf16, so bf16 bit patterns should round-trip.
        assert_eq!(
            result_host, *input_host,
            "Expected bf16 values to round-trip through f32 conversion"
        );
    });
}

#[test]
fn execute_bf16_to_f32_conversion() {
    common::with_test_stack(|| {
        let input_host = Arc::new(vec![
            bf16::from_f32(-3.5),
            bf16::from_f32(-1.0),
            bf16::from_f32(-0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.125),
            bf16::from_f32(0.1),
            bf16::from_f32(1.1),
            bf16::from_f32(42.0),
        ]);

        let input: Tensor<bf16> = copy_host_vec_to_device(&input_host)
            .sync()
            .expect("Failed.");
        let input = Arc::new(input);
        let output: Tensor<f32> = zeros(&[input_host.len()]).sync().expect("Failed.");

        let (result, _) = bf16_to_f32_conversion_kernel(output.partition([4]), input)
            .sync()
            .expect("Failed.");

        let result_host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        let expected: Vec<f32> = input_host.iter().map(|x| x.to_f32()).collect();

        assert_eq!(
            result_host, expected,
            "Expected bf16->f32 conversion output to match host-side bf16::to_f32"
        );
    });
}

#[test]
fn execute_f32_to_bf16_conversion() {
    common::with_test_stack(|| {
        let input_host = Arc::new(vec![-3.5f32, -1.0, -0.0, 0.0, 0.125, 0.1, 1.1, 42.0]);

        let input: Tensor<f32> = copy_host_vec_to_device(&input_host)
            .sync()
            .expect("Failed.");
        let input = Arc::new(input);
        let output: Tensor<bf16> = zeros(&[input_host.len()]).sync().expect("Failed.");

        let (result, _) = f32_to_bf16_conversion_kernel(output.partition([4]), input)
            .sync()
            .expect("Failed.");

        let result_host: Vec<bf16> = result.unpartition().to_host_vec().sync().expect("Failed.");
        let expected: Vec<bf16> = input_host.iter().map(|x| bf16::from_f32(*x)).collect();

        assert_eq!(
            result_host, expected,
            "Expected f32->bf16 conversion output to match host-side bf16::from_f32"
        );
    });
}
