/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage: a reduce result keeps the symbolic const-generic
//! shape written in its type annotation.

use cutile::compile_api::KernelCompiler;

mod common;

#[cutile::module]
mod reduce_symbolic_shape_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn reduce_max_scaled<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like(x, y);
        let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
        let scale: Tile<f32, { [BM] }> = broadcast_scalar(2f32, shape![BM]);
        let scaled: Tile<f32, { [BM] }> = tile_x_max * scale;
        let out: Tile<f32, { [BM, BN] }> = scaled.reshape(shape![BM, 1]).broadcast(y.shape());
        y.store(out);
    }
}

use reduce_symbolic_shape_module::__module_ast_self;

#[test]
fn reduce_result_keeps_symbolic_shape() {
    common::with_test_stack(|| {
        KernelCompiler::new(
            __module_ast_self,
            "reduce_symbolic_shape_module",
            "reduce_max_scaled",
        )
        .generics(vec!["2".to_string(), "16".to_string()])
        .strides(&[("y", &[16, 1]), ("x", &[16, 1])])
        .target("sm_120")
        .compile()
        .expect("reduce result should type-check against its symbolic annotation");
    });
}
