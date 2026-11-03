/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT-compile tests for `num_tiles(view, axis)`.
//!
//! Verifies that the intrinsic handler in `compile_intrinsic.rs` emits
//! `cuda_tile.get_index_space_shape` against the partition view and extracts
//! the axis-th result. No GPU execution — these tests only cover the
//! DSL → IR lowering path.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod num_tiles_kernels {
    use cutile::core::*;

    /// Calls `num_tiles` along both axes of a rank-2 partition.
    #[cutile::entry()]
    fn num_tiles_2d<const BM: i32, const BN: i32>(
        input: &Tensor<f32, { [-1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let part = input.partition(shape![BM, BN]);
        let nm: i32 = num_tiles(&part, 0);
        let nn: i32 = num_tiles(&part, 1);
        // Fold the pair of axis counts into a single scalar so the kernel
        // references both values (prevents DCE from dropping either call).
        let combined: Tile<i32, { [1] }> = broadcast_scalar(nm * 100i32 + nn, shape![1]);
        out.store(combined);
    }

    /// Calls `num_tiles` along axis 2 of a rank-3 partition.
    #[cutile::entry()]
    fn num_tiles_3d_axis2<const BM: i32, const BN: i32, const BK: i32>(
        input: &Tensor<f32, { [-1, -1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let part = input.partition(shape![BM, BN, BK]);
        let nk: i32 = num_tiles(&part, 2);
        let tile: Tile<i32, { [1] }> = broadcast_scalar(nk, shape![1]);
        out.store(tile);
    }

    /// Directly exercises the raw Tile IR view-shape query surfaces.
    #[cutile::entry()]
    fn raw_view_shape_queries<const BM: i32, const BN: i32>(
        input: &Tensor<f32, { [-1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let tensor_shape: [i32; 2] = get_tensor_shape(input);
        let part = input.partition(shape![BM, BN]);
        let index_shape: [i32; 2] = get_index_space_shape(&part);
        let combined: i32 = tensor_shape[0] + tensor_shape[1] + index_shape[0] + index_shape[1];
        let tile: Tile<i32, { [1] }> = broadcast_scalar(combined, shape![1]);
        out.store(tile);
    }

    /// Reads shape dimensions through ordinary indexing syntax.
    #[cutile::entry()]
    fn shape_index_queries(input: &Tensor<f32, { [-1, -1] }>, out: &mut Tensor<i32, { [1] }>) {
        let m: i32 = input.shape()[0];
        let n: i32 = input.shape()[1];
        let tile: Tile<i32, { [1] }> = broadcast_scalar(m * 100i32 + n, shape![1]);
        out.store(tile);
    }

    /// Uses a dynamic `num_tiles` loop bound over a statically-shaped tensor.
    ///
    /// The returned value must remain the SSA result from
    /// `get_index_space_shape`, but it should carry exact static bounds so the
    /// partition load inside the loop is proven safe at compile time.
    #[cutile::entry()]
    fn static_num_tiles_loop_bounds<const BK: i32, const K: i32>(
        input: &Tensor<f32, { [K] }>,
        out: &mut Tensor<f32, { [BK] }>,
    ) {
        let part = input.partition(shape![BK]);
        let nk: i32 = num_tiles(&part, 0);
        let mut tile: Tile<f32, { [BK] }> = broadcast_scalar(0.0f32, shape![BK]);
        for i in 0i32..nk {
            tile = tile + part.load([i]);
        }
        out.store(tile);
    }

    /// Explicit `Dim` values are source-level iterable proof objects.
    #[cutile::entry()]
    fn explicit_dim_loop_bounds<const BK: i32, const K: i32>(
        input: &Tensor<f32, { [K] }>,
        out: &mut Tensor<f32, { [BK] }>,
    ) {
        let part = input.partition(shape![BK]);
        let nk = num_tiles(&part, 0).into_dim();
        let mut tile: Tile<f32, { [BK] }> = broadcast_scalar(0.0f32, shape![BK]);
        for i in nk {
            tile = tile + part.load([i]);
        }
        out.store(tile);
    }
}

use num_tiles_kernels::__module_ast_self;

fn compile(kernel: &str, gen_args: &[String], strides: &[(&str, &[i32])]) -> String {
    let mlir = common::compile_to_ir(
        __module_ast_self,
        "num_tiles_kernels",
        kernel,
        gen_args,
        strides,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed to compile");
    println!("=== MLIR for {kernel} ===\n{mlir}");
    mlir
}

#[test]
fn rank2_emits_get_index_space_shape() {
    common::with_test_stack(|| {
        let mlir = compile(
            "num_tiles_2d",
            &[64.to_string(), 64.to_string()],
            &[("input", &[-1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
        assert!(
            mlir.contains("partition_view"),
            "expected partition_view operand in the op's input"
        );
    });
}

#[test]
fn rank3_emits_get_index_space_shape() {
    common::with_test_stack(|| {
        let mlir = compile(
            "num_tiles_3d_axis2",
            &[32.to_string(), 32.to_string(), 32.to_string()],
            &[("input", &[-1, -1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
    });
}

#[test]
fn raw_view_shape_queries_emit_view_ops() {
    common::with_test_stack(|| {
        let mlir = compile(
            "raw_view_shape_queries",
            &[64.to_string(), 32.to_string()],
            &[("input", &[-1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_tensor_shape"),
            "expected `get_tensor_shape` op in emitted MLIR"
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
    });
}

#[test]
fn shape_index_queries_compile_as_shape_value_access() {
    common::with_test_stack(|| {
        let mlir = compile(
            "shape_index_queries",
            &[],
            &[("input", &[-1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("muli") && mlir.contains("addi"),
            "expected shape indexing to compile as arithmetic over shape dimension values"
        );
    });
}

#[test]
fn static_num_tiles_loop_keeps_dynamic_upper_bound_but_proves_access() {
    common::with_test_stack(|| {
        let mlir = compile(
            "static_num_tiles_loop_bounds",
            &[8.to_string(), 32.to_string()],
            &[("input", &[1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `num_tiles` to lower to a dynamic `get_index_space_shape` result"
        );
        let num_tiles_value = mlir
            .lines()
            .find(|line| line.contains("get_index_space_shape"))
            .and_then(|line| {
                line.split_once('=')
                    .map(|(value, _)| value.trim().to_string())
            })
            .expect("expected to find the value produced by `get_index_space_shape`");
        let for_line = mlir
            .lines()
            .find(|line| line.contains(" = for "))
            .expect("expected a Tile IR loop");
        assert!(
            for_line.contains(&format!(" to {num_tiles_value},")),
            "expected the loop upper bound to use the dynamic `num_tiles` result; loop was `{for_line}`"
        );
        assert!(
            !mlir.contains("partition access out of bounds"),
            "expected static bounds on `num_tiles` to discharge the partition access check"
        );
    });
}

#[test]
fn explicit_dim_loop_iterates_as_dynamic_upper_bound_and_proves_access() {
    common::with_test_stack(|| {
        let mlir = compile(
            "explicit_dim_loop_bounds",
            &[8.to_string(), 32.to_string()],
            &[("input", &[1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `num_tiles` to lower to a dynamic `get_index_space_shape` result"
        );
        assert!(
            !mlir.contains("partition access out of bounds"),
            "expected explicit Dim iteration to discharge the partition access check"
        );
    });
}
