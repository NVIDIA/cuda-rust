/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage for example-style `load_tile_like` calls.
//!
//! The examples usually rely on inference (`let tile = load_tile_like(...)`)
//! instead of ascribing the returned tile type. These tests exercise the same
//! JIT path without requiring a CUDA device.

use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod load_tile_like_examples_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_refs_like<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn add_refs_like_unchecked<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry()]
    fn saxpy_like<const S: [i32; 2]>(
        y: &mut Tensor<f32, S>,
        a: f32,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = load_tile_like(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }

    #[cutile::entry()]
    fn generic_saxpy_like<T: ElementType, const S: [i32; 2]>(
        y: &mut Tensor<T, S>,
        a: T,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = load_tile_like(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }

    #[cutile::entry()]
    fn static_input_like<const B: i32, const N: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [N] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        z.store(tile_x);
    }

    #[cutile::entry()]
    fn permuted_partition_like<const BM: i32, const BN: i32, const DIM_MAP: [i32; 2]>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let dim_map = const_array!(DIM_MAP);
        let part = x.partition_permuted(shape![BM, BN], dim_map);
        let tile: Tile<f32, { [BM, BN] }> = part.load([pid.0, pid.1]);
        z.store(tile);
    }
}

use load_tile_like_examples_module::__module_ast_self;

fn compile(kernel: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    let mlir = common::compile_to_ir(
        __module_ast_self,
        "load_tile_like_examples_module",
        kernel,
        generics,
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
fn compiles_add_refs_style_1d_inference() {
    common::with_test_stack(|| {
        let mlir = compile(
            "add_refs_like",
            &[4.to_string()],
            &[("z", &[1]), ("x", &[1]), ("y", &[1])],
        );
        assert!(mlir.contains("load_view_tko"));
        assert_eq!(mlir.matches("load_view_tko").count(), 2);
        assert!(
            mlir.contains("padding_value = zero"),
            "load_tile_like should lower read-only input partitions with zero padding.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn unchecked_dynamic_1d_load_tile_like_uses_zero_padded_inputs() {
    common::with_test_stack(|| {
        let mlir = compile(
            "add_refs_like_unchecked",
            &[16384.to_string()],
            &[("z", &[1]), ("x", &[1]), ("y", &[1])],
        );
        assert_eq!(mlir.matches("load_view_tko").count(), 2);
        assert!(
            mlir.matches("partition_view<tile=(16384), padding_value = zero, tensor_view<?xf32")
                .count()
                >= 4,
            "Unchecked dynamic load_tile_like inputs should use zero-padded partition views.\nMLIR:\n{mlir}"
        );
        let store_line = mlir
            .lines()
            .find(|line| line.contains("store_view_tko"))
            .expect("expected store_view_tko in add_refs_like_unchecked MLIR");
        assert!(
            !store_line.contains("padding_value = zero"),
            "Mutable output store should remain non-padded.\nStore line:\n{store_line}\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn compiles_saxpy_style_2d_inference() {
    common::with_test_stack(|| {
        let mlir = compile(
            "saxpy_like",
            &[2.to_string(), 4.to_string()],
            &[("y", &[4, 1]), ("x", &[4, 1])],
        );
        assert!(mlir.contains("load_view_tko"));
        assert!(mlir.contains("tile<2x4xf32>"));
        assert!(
            !mlir.contains(" offset "),
            "Mutable output entry lowering should not offset the output pointer.\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("mini"),
            "Mutable output entry lowering should not compute per-block remaining dimensions.\nMLIR:\n{mlir}"
        );
        let store_line = mlir
            .lines()
            .find(|line| line.contains("store_view_tko"))
            .expect("expected store_view_tko in saxpy_like MLIR");
        assert!(
            mlir.contains("padding_value = zero"),
            "load_tile_like should lower read-only input partitions with zero padding.\nMLIR:\n{mlir}"
        );
        assert!(
            !store_line.contains("padding_value = zero"),
            "Tensor::store for mutable outputs should use padding::None.\nStore line:\n{store_line}\nMLIR:\n{mlir}"
        );
        assert!(
            !store_line.contains("[0"),
            "Tensor::store should index by tile-block id, not [0, 0].\nStore line:\n{store_line}\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn compiles_static_input_1d_inference() {
    common::with_test_stack(|| {
        let mlir = compile(
            "static_input_like",
            &[4.to_string(), 16.to_string()],
            &[("z", &[1]), ("x", &[1])],
        );
        assert!(mlir.contains("load_view_tko"));
        assert!(mlir.contains("tile<4xf32>"));
    });
}

#[test]
fn permuted_read_only_partition_uses_zero_padding() {
    common::with_test_stack(|| {
        let mlir = compile(
            "permuted_partition_like",
            &[2.to_string(), 4.to_string(), 0.to_string(), 1.to_string()],
            &[("z", &[4, 1]), ("x", &[4, 1])],
        );
        assert!(mlir.contains("load_view_tko"));
        assert!(
            mlir.contains("padding_value = zero"),
            "partition_permuted should lower read-only partitions with zero padding.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn compiles_generic_saxpy_style_2d_inference() {
    common::with_test_stack(|| {
        let mlir = compile(
            "generic_saxpy_like",
            &["f32".to_string(), 2.to_string(), 4.to_string()],
            &[("y", &[4, 1]), ("x", &[4, 1])],
        );
        assert!(mlir.contains("load_view_tko"));
        assert!(mlir.contains("tile<2x4xf32>"));
    });
}
