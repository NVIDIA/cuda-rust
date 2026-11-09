/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Library-style wrappers around foreign declarations, shaped like the Tile
//! C++ end-user APIs of tile libraries: tag values select options, traits
//! dispatch one name over tile ranks, and the library's constants are
//! inferred from the operands. Compile-only.

use cutile::compile_api::KernelCompiler;
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[cutile::module]
mod wrap_decls {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches `solver.tilelib`'s `cholesky`.
    #[cutile::foreign]
    pub unsafe fn cholesky<const Batch: i32, const M: i32, T: ElementType>(
        Function: &str,
        FillMode: &str,
        #[inout] A: Tile<T, { [Batch, M, M] }>,
    ) -> (Tile<T, { [Batch, M, M] }>, Tile<i32, { [Batch] }>);

    /// # Safety
    ///
    /// Matches `fft.tilelib`'s `fft`.
    #[cutile::foreign]
    pub unsafe fn fft<
        const AXIS: i32,
        const DIM0: i32,
        const DIM1: i32,
        const DIM2: i32,
        T: ElementType,
    >(
        #[inout] data: Tile<T, { [DIM0, DIM1, DIM2, 2] }>,
    ) -> Tile<T, { [DIM0, DIM1, DIM2, 2] }>;
}

#[cutile::module]
mod wrappers {
    use super::wrap_decls;
    use super::wrap_decls::cholesky;
    use cutile::core::*;

    /// A fill-mode tag; `NAME` is the string the library expects.
    pub trait FillMode {
        const NAME: &'static str;
    }
    /// Like Tile C++ `lower`.
    pub struct Lower;
    impl FillMode for Lower {
        const NAME: &'static str = "lower";
    }
    /// Like Tile C++ `upper`.
    pub struct Upper;
    impl FillMode for Upper {
        const NAME: &'static str = "upper";
    }

    pub trait CholeskyFactor {
        type Out;
        fn cholesky_factor_impl<F: FillMode>(self, fill: F) -> Self::Out;
    }

    impl<T: ElementType, const M: i32> CholeskyFactor for Tile<T, { [M, M] }> {
        type Out = (Tile<T, { [M, M] }>, Tile<i32, { [] }>);
        fn cholesky_factor_impl<F: FillMode>(
            self,
            fill: F,
        ) -> (Tile<T, { [M, M] }>, Tile<i32, { [] }>) {
            let a: Tile<T, { [1, M, M] }> = self.reshape(const_shape![1, M, M]);
            let (factor, info) = unsafe { cholesky::<1, M, T>("potrf", F::NAME, a) };
            (
                factor.reshape(const_shape![M, M]),
                info.reshape(const_shape![]),
            )
        }
    }

    impl<T: ElementType, const B: i32, const M: i32> CholeskyFactor for Tile<T, { [B, M, M] }> {
        type Out = (Tile<T, { [B, M, M] }>, Tile<i32, { [B] }>);
        fn cholesky_factor_impl<F: FillMode>(
            self,
            fill: F,
        ) -> (Tile<T, { [B, M, M] }>, Tile<i32, { [B] }>) {
            unsafe { cholesky::<B, M, T>("potrf", F::NAME, self) }
        }
    }

    /// Like Tile C++ `cholesky_factor(A, fill)`.
    pub fn cholesky_factor<A: CholeskyFactor, F: FillMode>(
        a: A,
        fill: F,
    ) -> <A as CholeskyFactor>::Out {
        a.cholesky_factor_impl(fill)
    }

    /// Like Tile C++ `axis<N>`.
    pub struct Axis<const N: i32>;

    pub trait Fft<A>: Sized {
        fn fft_impl(self, axis: A) -> Self;
    }

    impl<T: ElementType, const N: i32> Fft<Axis<0>> for Tile<T, { [N, 2] }> {
        fn fft_impl(self, _axis: Axis<0>) -> Tile<T, { [N, 2] }> {
            let x: Tile<T, { [1, 1, N, 2] }> = self.reshape(const_shape![1, 1, N, 2]);
            let y: Tile<T, { [1, 1, N, 2] }> = unsafe { wrap_decls::fft::<2, 1, 1, N, T>(x) };
            y.reshape(const_shape![N, 2])
        }
    }

    impl<T: ElementType, const D0: i32, const D1: i32> Fft<Axis<0>> for Tile<T, { [D0, D1, 2] }> {
        fn fft_impl(self, _axis: Axis<0>) -> Tile<T, { [D0, D1, 2] }> {
            let x: Tile<T, { [1, D0, D1, 2] }> = self.reshape(const_shape![1, D0, D1, 2]);
            let y: Tile<T, { [1, D0, D1, 2] }> = unsafe { wrap_decls::fft::<1, 1, D0, D1, T>(x) };
            y.reshape(const_shape![D0, D1, 2])
        }
    }

    impl<T: ElementType, const D0: i32, const D1: i32> Fft<Axis<1>> for Tile<T, { [D0, D1, 2] }> {
        fn fft_impl(self, _axis: Axis<1>) -> Tile<T, { [D0, D1, 2] }> {
            let x: Tile<T, { [1, D0, D1, 2] }> = self.reshape(const_shape![1, D0, D1, 2]);
            let y: Tile<T, { [1, D0, D1, 2] }> = unsafe { wrap_decls::fft::<2, 1, D0, D1, T>(x) };
            y.reshape(const_shape![D0, D1, 2])
        }
    }

    impl<T: ElementType, const D0: i32, const D1: i32, const D2: i32> Fft<Axis<2>>
        for Tile<T, { [D0, D1, D2, 2] }>
    {
        fn fft_impl(self, _axis: Axis<2>) -> Tile<T, { [D0, D1, D2, 2] }> {
            unsafe { wrap_decls::fft::<2, D0, D1, D2, T>(self) }
        }
    }

    /// Like Tile C++ `fft(data, axis<N>)`.
    pub fn fft<X: Fft<A>, A>(data: X, transform_axis: A) -> X {
        data.fft_impl(transform_axis)
    }

    /// A tag whose const generic sets only the size of a result.
    pub struct Pivots<const R: i32>;

    pub trait Factor<P> {
        type Out;
        fn factor_impl(self, pivots: P) -> Self::Out;
    }

    impl<T: ElementType, const M: i32, const N: i32, const R: i32> Factor<Pivots<R>>
        for Tile<T, { [M, N] }>
    {
        type Out = (Tile<T, { [M, N] }>, Tile<i32, { [R] }>);
        fn factor_impl(self, _pivots: Pivots<R>) -> (Tile<T, { [M, N] }>, Tile<i32, { [R] }>) {
            (self, constant(0i32, const_shape![R]))
        }
    }

    pub fn factor<A: Factor<P>, P>(a: A, pivots: P) -> A::Out {
        a.factor_impl(pivots)
    }

    /// A method called on the tag value itself.
    pub trait Forward {
        fn forward<X>(self, x: X) -> X;
    }

    impl<const R: i32> Forward for Pivots<R> {
        fn forward<X>(self, x: X) -> X {
            x
        }
    }

    pub fn forward<P: Forward>(x: Tile<f32, { [16] }>, tag: P) -> Tile<f32, { [16] }> {
        tag.forward(x)
    }
}

#[cutile::module]
mod wrap_kernels {
    use super::wrappers::{cholesky_factor, factor, fft, forward, Axis, Lower, Pivots, Upper};
    use cutile::core::*;

    #[cutile::entry()]
    fn factor_matrix<const M: i32>(
        a: &mut Tensor<f64, { [M, M] }>,
        info: &mut Tensor<i32, { [1] }>,
    ) {
        let (factor, status) = cholesky_factor(a.load(), Lower);
        a.store(factor);
        info.store(status.reshape(const_shape![1]));
    }

    #[cutile::entry()]
    fn factor_batch<const B: i32, const M: i32>(
        a: &mut Tensor<f64, { [B, M, M] }>,
        info: &mut Tensor<i32, { [B] }>,
    ) {
        let (factor, status) = cholesky_factor(a.load(), Upper);
        a.store(factor);
        info.store(status);
    }

    #[cutile::entry()]
    fn factor_with_pivots<T: ElementType, const M: i32, const N: i32, const R: i32>(
        a: &mut Tensor<T, { [M, N] }>,
        pivots: &mut Tensor<i32, { [R] }>,
    ) {
        let (f, p) = factor(a.load(), Pivots::<R>);
        a.store(f);
        pivots.store(p);
    }

    /// The impl's `M` and `N` name this entry's `N` and `M`, and its `R` this
    /// entry's `M`.
    #[cutile::entry()]
    fn factor_with_swapped_names<const N: i32, const M: i32>(
        a: &mut Tensor<f32, { [N, M] }>,
        pivots: &mut Tensor<i32, { [M] }>,
    ) {
        let (f, p) = factor(a.load(), Pivots::<M>);
        a.store(f);
        pivots.store(p);
    }

    #[cutile::entry()]
    fn factor_with_literal_pivots(
        a: &mut Tensor<f32, { [8, 4] }>,
        pivots: &mut Tensor<i32, { [2] }>,
    ) {
        let (f, p) = factor(a.load(), Pivots::<2>);
        a.store(f);
        pivots.store(p);
    }

    #[cutile::entry()]
    fn forward_literal_tag(x: &mut Tensor<f32, { [16] }>) {
        let t = x.load();
        x.store(forward(t, Pivots::<2>));
    }

    #[cutile::entry()]
    fn forward_caller_tag<const R: i32>(x: &mut Tensor<f32, { [16] }>) {
        let t = x.load();
        x.store(forward(t, Pivots::<R>));
    }

    #[cutile::entry()]
    fn fft_rank_1<const N: i32>(x: &mut Tensor<f32, { [N, 2] }>) {
        let signal = x.load();
        x.store(fft(signal, Axis::<0>));
    }

    #[cutile::entry()]
    fn fft_rank_2_axis_0<const D0: i32, const D1: i32>(x: &mut Tensor<f32, { [D0, D1, 2] }>) {
        let signal = x.load();
        x.store(fft(signal, Axis::<0>));
    }

    #[cutile::entry()]
    fn fft_rank_2_axis_1<const D0: i32, const D1: i32>(x: &mut Tensor<f32, { [D0, D1, 2] }>) {
        let signal = x.load();
        x.store(fft(signal, Axis::<1>));
    }

    #[cutile::entry()]
    fn fft_rank_3_axis_2(x: &mut Tensor<f64, { [8, 8, 2] }>) {
        let signal: Tile<f64, { [2, 4, 8, 2] }> = x.load().reshape(const_shape![2, 4, 8, 2]);
        let spectrum = fft(signal, Axis::<2>);
        x.store(spectrum.reshape(const_shape![8, 8, 2]));
    }
}

fn compile(
    entry: &str,
    generics: Vec<String>,
    strides: &[(&str, &[i32])],
) -> cutile::compile_api::CompileArtifacts {
    KernelCompiler::new(wrap_kernels::__module_ast_self, "wrap_kernels", entry)
        .generics(generics)
        .strides(strides)
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .unwrap_or_else(|e| panic!("compiling {entry}: {e}"))
}

fn call_line(ir: &str, callee: &str) -> String {
    ir.lines()
        .find(|l| l.contains(&format!("preview$call @{callee}")))
        .unwrap_or_else(|| panic!("no call to {callee} in\n{ir}"))
        .to_string()
}

#[test]
fn a_tag_value_selects_the_string_constant_and_a_matrix_is_batch_1() {
    common::with_test_stack(|| {
        let artifacts = compile(
            "factor_matrix",
            vec!["32".into()],
            &[("a", &[32, 1]), ("info", &[1])],
        );
        let ir = artifacts.ir_text();
        let call = call_line(&ir, "cholesky");
        assert!(
            call.contains(r#"@cholesky<["potrf", "lower", 1, 32, f64]>"#),
            "{call}"
        );
        assert!(
            call.contains(": (tile<1x32x32xf64>, tile<1xi32>, token)"),
            "{call}"
        );
        // The batch-1 info folds to a rank-0 tile.
        assert!(ir.contains("-> tile<i32>"), "{ir}");
    });
}

#[test]
fn a_batch_of_matrices_passes_through() {
    common::with_test_stack(|| {
        let artifacts = compile(
            "factor_batch",
            vec!["4".into(), "16".into()],
            &[("a", &[256, 16, 1]), ("info", &[1])],
        );
        let call = call_line(&artifacts.ir_text(), "cholesky");
        assert!(
            call.contains(r#"@cholesky<["potrf", "upper", 4, 16, f64]>"#),
            "{call}"
        );
    });
}

#[test]
fn signals_of_lower_rank_are_promoted_and_the_axis_shifted() {
    common::with_test_stack(|| {
        for (entry, generics, strides, expected) in [
            (
                "fft_rank_1",
                vec!["64".to_string()],
                &[2, 1][..],
                "@fft<[2, 1, 1, 64, f32]>",
            ),
            (
                "fft_rank_2_axis_0",
                vec!["4".to_string(), "64".to_string()],
                &[128, 2, 1][..],
                "@fft<[1, 1, 4, 64, f32]>",
            ),
            (
                "fft_rank_2_axis_1",
                vec!["4".to_string(), "64".to_string()],
                &[128, 2, 1][..],
                "@fft<[2, 1, 4, 64, f32]>",
            ),
            (
                "fft_rank_3_axis_2",
                vec![],
                &[16, 2, 1][..],
                "@fft<[2, 2, 4, 8, f64]>",
            ),
        ] {
            let artifacts = compile(entry, generics, &[("x", strides)]);
            let call = call_line(&artifacts.ir_text(), "fft");
            assert!(call.contains(expected), "{entry}: {call}");
        }
    });
}

/// Regression: a tag's const generic that appears only in the wrapper's
/// result type, `factor(a, Pivots::<R>)` returning `Tile<i32, {[R]}>`,
/// panicked the JIT with "Unexpected arg type Type".
#[test]
fn a_tag_const_generic_sizes_the_result() {
    common::with_test_stack(|| {
        let artifacts = compile(
            "factor_with_pivots",
            vec!["f32".into(), "8".into(), "4".into(), "2".into()],
            &[("a", &[4, 1]), ("pivots", &[1])],
        );
        let ir = artifacts.ir_text();
        assert!(ir.contains("tile<2xi32>"), "{ir}");
    });
}

/// The tag's const generic is matched by value, not by its spelling: a
/// literal, or a caller const whose name means something else in the impl.
#[test]
fn a_tag_const_generic_is_matched_by_value() {
    common::with_test_stack(|| {
        let ir = compile(
            "factor_with_swapped_names",
            vec!["8".into(), "2".into()],
            &[("a", &[2, 1]), ("pivots", &[1])],
        )
        .ir_text();
        assert!(
            ir.contains("tile<8x2xf32>") && ir.contains("tile<2xi32>"),
            "{ir}"
        );
        assert!(!ir.contains("tile<8xi32>"), "{ir}");
        let ir = compile(
            "factor_with_literal_pivots",
            vec![],
            &[("a", &[4, 1]), ("pivots", &[1])],
        )
        .ir_text();
        assert!(ir.contains("tile<2xi32>"), "{ir}");
    });
}

/// A trait method called on a const-generic tag value, `tag.forward(x)` with
/// `tag = Pivots::<2>`, failed: "unable to resolve generic type `Pivots :: < 2 >`".
#[test]
fn a_method_is_called_on_a_const_generic_tag() {
    common::with_test_stack(|| {
        compile("forward_literal_tag", vec![], &[("x", &[1])]);
        compile("forward_caller_tag", vec!["2".into()], &[("x", &[1])]);
    });
}
