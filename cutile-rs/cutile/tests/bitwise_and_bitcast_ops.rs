/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod bitwise_and_bitcast_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn bitwise_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test bitwise operations
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);

        // Bitwise operations
        let t1: Tile<i64, S> = andi(x, y); // AND
        let t2: Tile<i64, S> = ori(t1, y); // OR
        let t3: Tile<i64, S> = xori(t2, y); // XOR
        let t4: Tile<i64, S> = shli(t3, y, overflow::None); // left shift
        let result: Tile<i64, S> = shri(t4, y); // right shift

        output.store(result);
    }

    #[cutile::entry()]
    fn bitcast_kernel<const S: [i32; 1]>(output: &mut Tensor<u32, S>) {
        // Test bitcast operation
        let x: Tile<u32, S> = load_tile_mut(output);
        let float_view: Tile<f32, S> = bitcast(x); // reinterpret as f32
        let back_to_int: Tile<u32, S> = bitcast(float_view); // back to u32
        output.store(back_to_int);
    }

    #[cutile::entry()]
    fn shri_unsigned_kernel<const S: [i32; 1]>(output: &mut Tensor<u32, S>) {
        // Test right shift with unsigned types (logical shift)
        let x: Tile<u32, S> = load_tile_mut(output);
        let shift_amount: Tile<u32, S> = load_tile_mut(output);
        let result: Tile<u32, S> = shri(x, shift_amount);
        output.store(result);
    }
}

use bitwise_and_bitcast_ops_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "bitwise_and_bitcast_ops_module",
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
fn compile_bitwise_ops() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "bitwise_ops_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== BITWISE OPS MLIR ===\n{}", module_op_str);

        let expected_ops = ["andi", "ori", "xori", "shli", "shri"];
        for op in expected_ops {
            assert!(
                module_op_str.contains(format!("= {}", op).as_str()),
                "Expected {} operation in MLIR output",
                op
            );
        }

        println!(
            "\n✓ All {} bitwise operations verified in MLIR output",
            expected_ops.len()
        );
    });
}

#[test]
fn compile_bitcast() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("bitcast_kernel", &[128.to_string()], &[("output", &[1])]);
        println!("\n=== BITCAST MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= bitcast"),
            "Expected bitcast operation in MLIR output"
        );

        println!("\n✓ bitcast operation verified in MLIR output");
    });
}

#[test]
fn compile_shri_unsigned() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "shri_unsigned_kernel",
            &[128.to_string()],
            &[("output", &[1])],
        );
        println!("\n=== SHRI UNSIGNED MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= shri"),
            "Expected shri operation in MLIR output"
        );
        assert!(
            module_op_str.contains("unsigned"),
            "Expected unsigned signedness attribute (logical shift)"
        );

        println!("\n✓ shri with unsigned types (logical shift) verified in MLIR output");
    });
}
