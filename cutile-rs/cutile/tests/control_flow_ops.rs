/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::{self, api::*, tensor::*, tile_kernel::*};
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod control_flow_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn control_flow_test_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        dynamic_value: i32,
    ) {
        // Test what control flow operations are generated
        let mut sum: Tile<f32, S> = load_tile_mut(output);

        // Test for loop (should generate cuda_tile.for)
        for _i in 0i32..10i32 {
            sum = sum + sum;
        }

        // Test if/else (should generate cuda_tile.if with yield)
        if dynamic_value < 5i32 {
            sum = sum + sum;
        } else {
            sum = sum - sum;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn break_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test break statement in infinite loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut i: i32 = 0;
        loop {
            sum = sum + sum;
            i = i + 1i32;
            if i >= 2i32 {
                break;
            }
        }
        output.store(sum);
    }

    #[cutile::entry()]
    fn if_return_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>, conditional: bool) {
        let mut val: Tile<i64, S> = output.load();
        let result: Tile<i64, S> = if conditional {
            val = val + val;
            constant(2, val.shape())
        } else {
            val = val + val + val;
            constant(3, val.shape())
        };
        val = val + result;
        output.store(val);
    }

    #[cutile::entry()]
    fn while_loop_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test while loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut counter: i32 = 0i32;

        while counter < 10i32 {
            sum = sum + sum;
            counter = counter + 1i32;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn infinite_loop_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test infinite loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut counter: i32 = 0i32;

        // Simulate loop with while instead
        while counter < 10i32 {
            sum = sum + sum;
            counter = counter + 1i32;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn step_by_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        for _i in (0i32..100i32).step_by(10) {
            sum = sum + sum;
        }
        output.store(sum);
    }

    #[cutile::entry()]
    fn assume_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test assume operation - provides optimization hints to compiler
        // Note: Using i64 tensor because bounded predicate only works with integers
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Tell compiler to assume tile values are non-negative (bounded<0, ?>)
        // This can enable additional optimizations
        // Using Melih's const-generic assume_bounds_lower with lower bound of 0
        let assumed_tile: Tile<i64, S> = unsafe { assume_bounds_lower::<_, 0>(tile) };

        // Use the assumed tile
        let result: Tile<i64, S> = assumed_tile + constant(1i64, output.shape());

        output.store(result);
    }

    #[cutile::entry()]
    fn assume_non_negative_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume values are non-negative (>= 0) using Melih's const-generic version
        let non_neg_tile: Tile<i64, S> = unsafe { assume_bounds_lower::<_, 0>(tile) };

        let result: Tile<i64, S> = non_neg_tile + constant(1i64, output.shape());
        output.store(result);
    }

    #[cutile::entry()]
    fn assume_div_by_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume all values are divisible by 16 (using Melih's const generic version)
        let aligned_tile: Tile<i64, S> = unsafe { assume_div_by::<_, 16>(tile) };

        let result: Tile<i64, S> = aligned_tile + constant(1i64, output.shape());
        output.store(result);
    }

    #[cutile::entry()]
    fn assume_same_elements_test_kernel<const S: [i32; 2]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume groups of size 2 along dim 0, size 4 along dim 1 have same elements
        // Using Melih's const-generic pattern instead of runtime parameters
        let same_tile: Tile<i64, S> = unsafe { assume_same_elements_2d::<_, 2, 4>(tile) };

        let result: Tile<i64, S> = same_tile + constant(1i64, output.shape());
        output.store(result);
    }

    /// Repro: for loop inside if doesn't propagate mutable variable.
    /// collect_mutated_variables_from_block doesn't recurse into
    /// nested control flow (for/while/if), so the if op doesn't
    /// yield acc, and post-if code uses the stale pre-if value.
    #[cutile::entry()]
    fn if_for_carry_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, flag: i32) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if flag > 0i32 {
            for _i in 0i32..10i32 {
                let ones: Tile<f32, S> = constant(1.0f32, output.shape());
                acc = acc + ones;
            }
        } else {
            acc = acc;
        }
        output.store(acc);
    }

    /// Same repro but with const generic flag (closer to Yinuo's report).
    #[cutile::entry()]
    fn if_for_carry_const_kernel<const S: [i32; 1], const FLAG: i32, const N: i32>(
        output: &mut Tensor<f32, S>,
    ) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if FLAG > 0i32 {
            for _i in 0i32..N {
                let ones: Tile<f32, S> = constant(1.0f32, output.shape());
                acc = acc + ones;
            }
        } else {
            acc = acc;
        }
        output.store(acc);
    }

    /// if/else as a tile expression: `let result = if cond { a } else { b };`
    #[cutile::entry()]
    fn if_else_tile_expr_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, flag: i32) {
        let ones: Tile<f32, S> = constant(1.0f32, output.shape());
        let twos: Tile<f32, S> = constant(2.0f32, output.shape());
        let result: Tile<f32, S> = if flag > 0i32 { ones } else { twos };
        output.store(result);
    }

    /// else-if as a tile expression: nested ifs in Rust source syntax.
    #[cutile::entry()]
    fn else_if_tile_expr_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, flag: i32) {
        let ones: Tile<f32, S> = constant(1.0f32, output.shape());
        let twos: Tile<f32, S> = constant(2.0f32, output.shape());
        let threes: Tile<f32, S> = constant(3.0f32, output.shape());
        let result: Tile<f32, S> = if flag < 0i32 {
            ones
        } else if flag < 1i32 {
            twos
        } else {
            threes
        };
        output.store(result);
    }

    /// else-if carrying a mutable variable through every branch.
    #[cutile::entry()]
    fn else_if_carry_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, flag: i32) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if flag < 0i32 {
            acc = acc + constant(1.0f32, output.shape());
        } else if flag < 1i32 {
            acc = acc + constant(2.0f32, output.shape());
        } else {
            acc = acc + constant(3.0f32, output.shape());
        }
        output.store(acc);
    }

    /// Nested mutation: for-in-if with runtime condition.
    /// The for loop mutates `acc` inside a dynamic if branch.
    #[cutile::entry()]
    fn nested_for_in_if_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, flag: i32) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if flag > 0i32 {
            for _i in 0i32..5i32 {
                let twos: Tile<f32, S> = constant(2.0f32, output.shape());
                acc = acc + twos;
            }
        } else {
            let ones: Tile<f32, S> = constant(1.0f32, output.shape());
            acc = acc + ones;
        }
        output.store(acc);
    }

    /// Nested mutation: for-in-if with const condition (const-folded path).
    #[cutile::entry()]
    fn nested_for_in_if_const_kernel<const S: [i32; 1], const FLAG: i32>(
        output: &mut Tensor<f32, S>,
    ) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if FLAG > 0i32 {
            for _i in 0i32..5i32 {
                let twos: Tile<f32, S> = constant(2.0f32, output.shape());
                acc = acc + twos;
            }
        } else {
            let ones: Tile<f32, S> = constant(1.0f32, output.shape());
            acc = acc + ones;
        }
        output.store(acc);
    }

    /// Deeply nested: if-in-for-in-if with const outer condition.
    /// Outer if is const-folded, for loop runs 4 times, inner if uses runtime flag.
    #[cutile::entry()]
    fn nested_if_for_if_kernel<const S: [i32; 1], const OUTER: i32>(
        output: &mut Tensor<f32, S>,
        inner_flag: i32,
    ) {
        let mut acc: Tile<f32, S> = constant(0.0f32, output.shape());
        if OUTER > 0i32 {
            for _i in 0i32..4i32 {
                if inner_flag > 0i32 {
                    let threes: Tile<f32, S> = constant(3.0f32, output.shape());
                    acc = acc + threes;
                } else {
                    acc = acc;
                }
            }
        } else {
            acc = acc;
        }
        output.store(acc);
    }
}

use control_flow_ops_module::{
    __module_ast_self, break_test_kernel, else_if_carry_kernel, else_if_tile_expr_kernel,
    if_else_tile_expr_kernel, if_for_carry_const_kernel, if_for_carry_kernel,
    if_return_test_kernel, nested_for_in_if_const_kernel, nested_for_in_if_kernel,
    nested_if_for_if_kernel,
};

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "control_flow_ops_module",
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
fn compile_control_flow_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "control_flow_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== CONTROL FLOW TEST MLIR ===\n{}", module_op_str);

        let has_for = module_op_str.contains(" for ");
        let has_if = module_op_str.contains("if %");
        let has_continue = module_op_str.contains("continue ");
        let has_return = module_op_str.contains("return");

        println!("\n=== Control Flow Operations Found ===");
        println!("for:      {}", if has_for { "✓" } else { "✗" });
        println!("if:       {}", if has_if { "✓" } else { "✗" });
        println!("continue: {}", if has_continue { "✓" } else { "✗" });
        println!("return:   {}", if has_return { "✓" } else { "✗" });

        assert!(has_for, "Expected 'for' loop in MLIR");
        assert!(has_if, "Expected 'if' in MLIR");
        assert!(has_continue, "Expected 'continue' as loop terminator");
        assert!(has_return, "Expected 'return' at function end");

        println!("\n✓ All control flow operations generated (for, if, continue, return)!");
    });
}

#[test]
fn execute_if_result_test() {
    common::with_test_stack(|| {
        let arg: Tensor<i64> = ones(&[16]).sync().expect("Failed.");
        // If true, double and add 2.
        let (result, _) = if_return_test_kernel(arg.partition([4]), true)
            .sync()
            .expect("Failed.");
        let result: Vec<i64> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(result.iter().all(|x| *x == 4));

        // If false, triple and add 3.
        let arg: Tensor<i64> = ones(&[16]).sync().expect("Failed.");
        let (result, _) = if_return_test_kernel(arg.partition([4]), false)
            .sync()
            .expect("Failed.");
        let result: Vec<i64> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(result.iter().all(|x| *x == 6));
    });
}

#[test]
fn execute_break_test() {
    common::with_test_stack(|| {
        // break_test_kernel loads output, doubles it twice (loop runs 2 iterations then breaks),
        // and stores the result. Starting from 1.0, we expect 1.0 * 2 * 2 = 4.0.
        let arg: Tensor<f32> = ones(&[16]).sync().expect("Failed.");
        let (result,) = break_test_kernel(arg.partition([4]))
            .sync()
            .expect("Failed.");
        let result: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(
            result.iter().all(|x| *x == 4.0),
            "Expected all elements to be 4.0, got: {:?}",
            result
        );
    });
}

#[test]
fn compile_break_test() {
    common::with_test_stack(|| {
        let module_op_str =
            compile_ir("break_test_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== BREAK TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains("cuda_tile.loop");
        let has_break = module_op_str.contains("break");

        assert!(has_loop, "Expected loop operation in IR");
        assert!(has_break, "Expected break operation in IR");

        println!("\n✓ break statement compiled successfully");
    });
}

#[test]
fn compile_while_loop_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "while_loop_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== WHILE LOOP TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains("cuda_tile.loop") || module_op_str.contains(" loop ");
        let has_break = module_op_str.contains("break ");

        assert!(has_loop, "Expected cuda_tile.loop operation in MLIR");
        assert!(has_break, "Expected break operation for while loop exit");

        println!("\n✓ while loop compiled to cuda_tile.loop with break!");
    });
}

#[test]
fn compile_loop_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "infinite_loop_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== LOOP TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains("cuda_tile.loop") || module_op_str.contains(" loop ");
        let has_break = module_op_str.contains("break ");

        assert!(has_loop, "Expected cuda_tile.loop operation in MLIR");
        assert!(has_break, "Expected break operation for loop exit");

        println!("\n✓ loop expression compiled to cuda_tile.loop with break!");
    });
}

#[test]
fn compile_step_by_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "step_by_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== STEP_BY TEST MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains(" for "),
            "Expected for loop in MLIR output"
        );
        assert!(
            module_op_str.contains(", step %"),
            "Expected step_by(10) to compile to a for-loop with step"
        );
    });
}

#[test]
fn compile_assume_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "assume_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== ASSUME MLIR ===\n{}", module_op_str);

        // Verify assume operation appears
        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );

        // Verify it has the bounded predicate attribute (bounded<0, ?> = non-negative)
        assert!(
            module_op_str.contains("bounded"),
            "Expected bounded predicate on assume operation"
        );

        println!(
            "\n✓ assume operation verified (compiler optimization hint with bounded predicate)"
        );
    });
}

#[test]
fn compile_assume_non_negative_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "assume_non_negative_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== ASSUME_NON_NEGATIVE MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("bounded<0, ?>"),
            "Expected bounded<0, ?> predicate on assume operation"
        );

        println!("\n✓ assume_non_negative operation verified with bounded<0, ?>");
    });
}

#[test]
fn compile_assume_div_by_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "assume_div_by_test_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== ASSUME_DIV_BY MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("div_by<16>"),
            "Expected div_by<16> predicate on assume operation"
        );

        println!("\n✓ assume_div_by operation verified with div_by<16>");
    });
}

#[test]
fn compile_assume_same_elements_test() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "assume_same_elements_test_kernel",
            &[4.to_string(), 8.to_string()],
            &[("output", &[2, 2])],
        );
        println!("\n=== ASSUME_SAME_ELEMENTS MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("same_elements<[2, 4]>"),
            "Expected same_elements<[2, 4]> predicate on assume operation"
        );

        println!("\n✓ assume_same_elements operation verified with same_elements<[2, 4]>");
    });
}

#[test]
fn if_for_carry_propagates_mutation() {
    // Repro: for loop inside if should propagate mutable variable updates.
    // flag=1 means the if body runs: acc += 1.0 ten times → acc = 10.0.
    // Bug: acc stays 0.0 because the if op doesn't yield the for's output.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        if_for_carry_kernel((&mut output).partition([128]), 1i32)
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 10.0).abs() < 1e-3,
            "Expected 10.0 (for loop ran 10 times), got {}",
            host[0]
        );
    });
}

#[test]
fn if_for_carry_const_propagates_mutation() {
    // Same as above but with const generic FLAG and N.
    // FLAG=1, N=10 → acc = 10.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        if_for_carry_const_kernel((&mut output).partition([128]))
            .generics(vec![
                "128".to_string(), // S
                "1".to_string(),   // FLAG
                "10".to_string(),  // N
            ])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 10.0).abs() < 1e-3,
            "Expected 10.0 (const FLAG=1, N=10), got {}",
            host[0]
        );
    });
}

#[test]
fn if_else_tile_expr_returns_value() {
    // if/else as an expression: `let result = if flag > 0 { ones } else { twos };`
    // flag=1 → result = 1.0, flag=0 → result = 2.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        if_else_tile_expr_kernel((&mut output).partition([128]), 1i32)
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 1.0).abs() < 1e-3,
            "flag=1: expected 1.0, got {}",
            host[0]
        );
    });
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        if_else_tile_expr_kernel((&mut output).partition([128]), 0i32)
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 2.0).abs() < 1e-3,
            "flag=0: expected 2.0, got {}",
            host[0]
        );
    });
}

#[test]
fn else_if_tile_expr_returns_value() {
    for (flag, expected) in [(-1i32, 1.0f32), (0i32, 2.0f32), (1i32, 3.0f32)] {
        common::with_test_stack(move || {
            let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
            else_if_tile_expr_kernel((&mut output).partition([128]), flag)
                .sync()
                .expect("kernel");
            let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
            assert!(
                (host[0] - expected).abs() < 1e-3,
                "flag={flag}: expected {expected}, got {}",
                host[0]
            );
        });
    }
}

#[test]
fn else_if_carries_mutable_values() {
    for (flag, expected) in [(-1i32, 1.0f32), (0i32, 2.0f32), (1i32, 3.0f32)] {
        common::with_test_stack(move || {
            let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
            else_if_carry_kernel((&mut output).partition([128]), flag)
                .sync()
                .expect("kernel");
            let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
            assert!(
                (host[0] - expected).abs() < 1e-3,
                "flag={flag}: expected {expected}, got {}",
                host[0]
            );
        });
    }
}

// ---- Nested mutation tests ------------------------------------------------

#[test]
fn nested_for_in_if_dynamic() {
    // Runtime flag=1: for loop runs 5 times, acc += 2.0 each → 10.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_for_in_if_kernel((&mut output).partition([128]), 1i32)
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 10.0).abs() < 1e-3,
            "flag=1: expected 10.0, got {}",
            host[0]
        );
    });
    // Runtime flag=0: else branch, acc += 1.0 → 1.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_for_in_if_kernel((&mut output).partition([128]), 0i32)
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 1.0).abs() < 1e-3,
            "flag=0: expected 1.0, got {}",
            host[0]
        );
    });
}

#[test]
fn nested_for_in_if_const_folded() {
    // Const FLAG=1: then branch const-folded in, for runs 5 times → 10.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_for_in_if_const_kernel((&mut output).partition([128]))
            .generics(vec!["128".into(), "1".into()])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 10.0).abs() < 1e-3,
            "FLAG=1: expected 10.0, got {}",
            host[0]
        );
    });
    // Const FLAG=0: else branch const-folded in, acc += 1.0 → 1.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_for_in_if_const_kernel((&mut output).partition([128]))
            .generics(vec!["128".into(), "0".into()])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 1.0).abs() < 1e-3,
            "FLAG=0: expected 1.0, got {}",
            host[0]
        );
    });
}

#[test]
fn nested_if_for_if_deep() {
    // Const OUTER=1, runtime inner_flag=1: for runs 4x, inner if adds 3.0 → 12.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_if_for_if_kernel((&mut output).partition([128]), 1i32)
            .generics(vec!["128".into(), "1".into()])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0] - 12.0).abs() < 1e-3,
            "OUTER=1, inner=1: expected 12.0, got {}",
            host[0]
        );
    });
    // Const OUTER=1, runtime inner_flag=0: for runs 4x, inner else is no-op → 0.0.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_if_for_if_kernel((&mut output).partition([128]), 0i32)
            .generics(vec!["128".into(), "1".into()])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0]).abs() < 1e-3,
            "OUTER=1, inner=0: expected 0.0, got {}",
            host[0]
        );
    });
    // Const OUTER=0: outer else is no-op → 0.0 regardless of inner_flag.
    common::with_test_stack(|| {
        let mut output = cutile::api::zeros::<f32>(&[128]).sync().expect("alloc");
        nested_if_for_if_kernel((&mut output).partition([128]), 1i32)
            .generics(vec!["128".into(), "0".into()])
            .sync()
            .expect("kernel");
        let host: Vec<f32> = output.dup().to_host_vec().sync().expect("to_host");
        assert!(
            (host[0]).abs() < 1e-3,
            "OUTER=0: expected 0.0, got {}",
            host[0]
        );
    });
}
