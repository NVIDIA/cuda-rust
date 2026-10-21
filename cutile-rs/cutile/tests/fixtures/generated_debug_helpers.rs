/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cutile::module]
pub mod generated_debug_helpers {
    use cutile::core::*;

    pub fn array_function<const VALUES: [i32; 2]>(x: i32) -> i32 {
        x + VALUES[0] + VALUES[1]
    }

    #[cuda_tile::variadic_trait(N = 2)]
    pub trait ArrayMethod<const VALUES: [i32; N]> {
        fn array_method(self, shape: Shape<VALUES>) -> i32;
    }

    #[cuda_tile::variadic_trait_impl()]
    #[cuda_tile::variadic_impl(N = 2)]
    impl<const VALUES: [i32; N]> ArrayMethod<VALUES> for i32 {
        fn array_method(self, _shape: Shape<VALUES>) -> i32 {
            self + VALUES[0] + VALUES[1]
        }
    }

    pub fn nested_arrays(x: i32) -> i32 {
        let a = array_function::<{ [-17, 23] }>(x);
        a.array_method(shape![31, 37])
    }

    pub fn reductions(x: Tile<f32, { [32] }>) -> Tile<f32, { [] }> {
        let sum: Tile<f32, { [] }> = reduce_sum(x, 0);
        let product: Tile<f32, { [] }> = reduce_prod(x, 0);
        let minimum: Tile<f32, { [] }> = reduce_min(x, 0);
        let maximum: Tile<f32, { [] }> = reduce_max(x, 0);
        sum + product + minimum + maximum
    }

    pub fn checked_load(x: &Tensor<f32, { [-1] }>, index: i32) -> Tile<f32, { [32] }> {
        let p = x.partition(shape![32]);
        p.load([index])
    }

    pub fn first_tile(x: &Tensor<f32, { [-1] }>) -> Tile<f32, { [32] }> {
        let p = x.partition(shape![32]);
        p.load([0])
    }

    pub fn nested_first_tile(x: &Tensor<f32, { [-1] }>) -> Tile<f32, { [32] }> {
        first_tile(x)
    }
}
