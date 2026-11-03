/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod reduce_scan_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn scan_sum_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test scan_sum operation
        let tile: Tile<f32, S> = load_tile_mut(output);

        // Scan along dimension 0 (cumulative sum) - result has same shape
        let prefix_sums: Tile<f32, S> = scan_sum(tile, 0i32, reverse::Forward, 0.0f32);

        // Store the prefix sums
        output.store(prefix_sums);
    }

    #[cutile::entry()]
    fn reduce_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - sum
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for sum reduction
        let sum_scalar = reduce(tile, 0i32, 0.0f32, |acc, x| acc + x);

        // Reshape and store
        let sum_as_array: Tile<f32, { [1] }> = sum_scalar.reshape(shape![1]);
        result.store(sum_as_array);
    }

    #[cutile::entry()]
    fn reduce_product_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - product
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for product reduction
        let product_scalar = reduce(tile, 0i32, 1.0f32, |acc, x| acc * x);

        // Reshape and store
        let product_as_array: Tile<f32, { [1] }> = product_scalar.reshape(shape![1]);
        result.store(product_as_array);
    }

    #[cutile::entry()]
    fn reduce_max_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - max
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for max reduction with NEG_INFINITY as identity
        // f32::NEG_INFINITY works as identity because max(NEG_INFINITY, x) = x for any x
        //
        // TODO (np): Using maxf(acc, x, nan::Disabled, ftz::Disabled) here gives "cannot find function" error even though
        // it's imported via `use cutile::core::{*}`. This appears to be a scoping bug
        // where some functions aren't visible inside reduce/scan closures. Using max()
        // (scalar version) works correctly and is the right function for this context anyway.
        let max_scalar = reduce(tile, 0i32, f32::NEG_INFINITY, |acc, x| max(acc, x));

        // Reshape and store
        let max_as_array: Tile<f32, { [1] }> = max_scalar.reshape(shape![1]);
        result.store(max_as_array);
    }

    #[cutile::entry()]
    fn scan_closure_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test scan with custom closure - prefix product
        let tile: Tile<f32, S> = load_tile_mut(output);

        // Use closure for prefix product
        let prefix_products: Tile<f32, S> =
            scan(tile, 0i32, reverse::Forward, 1.0f32, |acc, x| acc * x);

        // Store the result
        output.store(prefix_products);
    }
}

use reduce_scan_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "reduce_scan_ops_module",
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
fn compile_scan_sum_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "scan_sum_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== SCAN_SUM MLIR ===\n{}", module_op_str);

        // Verify scan operation appears
        assert!(
            module_op_str.contains("scan"),
            "Expected scan operation in MLIR output"
        );

        // Verify it has a region with yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in scan region"
        );

        // Verify reverse attribute
        assert!(
            module_op_str.contains("reverse=false") || module_op_str.contains("reverse = false"),
            "Expected reverse=false in scan operation"
        );

        println!("\n✓ scan_sum operation verified (with prefix sum scan region)");
    });
}

#[test]
fn compile_reduce_closure_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "reduce_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
        );
        println!(
            "\n=== REDUCE WITH CLOSURE (SUM) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with add operation
        assert!(
            module_op_str.contains("addf") || module_op_str.contains("addi"),
            "Expected add operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (sum) operation verified");
    });
}

#[test]
fn compile_reduce_product_closure_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "reduce_product_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
        );
        println!(
            "\n=== REDUCE WITH CLOSURE (PRODUCT) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with multiply operation
        assert!(
            module_op_str.contains("mulf") || module_op_str.contains("muli"),
            "Expected multiply operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (product) operation verified");
    });
}

#[test]
fn compile_reduce_max_closure_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "reduce_max_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
        );
        println!(
            "\n=== REDUCE WITH CLOSURE (MAX) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with maxf operation
        assert!(
            module_op_str.contains("maxf"),
            "Expected maxf operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (max) operation verified");
    });
}

#[test]
fn compile_scan_closure_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "scan_closure_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!(
            "\n=== SCAN WITH CLOSURE (PREFIX PRODUCT) MLIR ===\n{}",
            module_op_str
        );

        // Verify scan operation appears
        assert!(
            module_op_str.contains("scan"),
            "Expected scan operation in MLIR output"
        );

        // Verify it has a region with multiply operation
        assert!(
            module_op_str.contains("mulf") || module_op_str.contains("muli"),
            "Expected multiply operation in scan region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in scan region"
        );

        // Verify reverse attribute
        assert!(
            module_op_str.contains("reverse=false") || module_op_str.contains("reverse = false"),
            "Expected reverse=false in scan operation"
        );

        println!("\n✓ scan with closure (prefix product) operation verified");
    });
}
