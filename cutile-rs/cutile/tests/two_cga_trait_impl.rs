/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke tests for `#[cuda_tile::variadic_trait_impl]` and assoc-type
//! rewriting in `#[cuda_tile::variadic_impl]`.
//!
//! Drives macro permissiveness items #3 (methodless `variadic_trait_impl`)
//! and #4 (rank-instance rewriting in trait refs + assoc-type bindings).

#[cutile::module]
mod two_cga_trait_impl_module {
    use cutile::core::*;

    #[cuda_tile::variadic_trait(N = 6)]
    pub trait TwoCgaTraitOp<E: ElementType, const A: [i32; N], const B: [i32; N]> {
        fn two_cga_op(self, a: Tile<E, A>, b: Shape<B>);
    }

    #[cuda_tile::variadic_trait_impl()]
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const A: [i32; N], const B: [i32; N]> TwoCgaTraitOp<E, A, B> for () {
        fn two_cga_op(self, _a: Tile<E, A>, _b: Shape<B>) {
            unreachable!()
        }
    }

    /// Independent-length CGAs: `A: [i32; N]` and `B: [i32; M]` cover all
    /// (N, M) rank pairs. Drives macro permissiveness item #3 — the
    /// product space must be enumerated, not just the diagonal.
    #[cuda_tile::variadic_trait(N = 6, M = 6)]
    pub trait TwoCgaIndependent<E: ElementType, const A: [i32; N], const B: [i32; M]> {
        fn two_cga_independent(self, a: Tile<E, A>, b: Shape<B>);
    }

    #[cuda_tile::variadic_trait_impl()]
    #[cuda_tile::variadic_impl(N = 6, M = 6)]
    impl<E: ElementType, const A: [i32; N], const B: [i32; M]> TwoCgaIndependent<E, A, B> for () {
        fn two_cga_independent(self, _a: Tile<E, A>, _b: Shape<B>) {
            unreachable!()
        }
    }

    // Variant: `variadic_trait_impl` alone (no `variadic_impl`). Routes
    // through `desugar_variadic_trait_impl` so this is no longer the panic
    // path it used to be. Confirms macro permissiveness item #3.
    #[cuda_tile::variadic_trait(N = 6)]
    pub trait MethodlessTwoCga<E: ElementType, const D: [i32; N]> {
        fn methodless_two_cga(self, a: Tile<E, D>);
    }

    #[cuda_tile::variadic_trait_impl()]
    impl<E: ElementType, const D: [i32; N]> MethodlessTwoCga<E, D> for () {
        fn methodless_two_cga(self, _a: Tile<E, D>) {
            unreachable!()
        }
    }
}
