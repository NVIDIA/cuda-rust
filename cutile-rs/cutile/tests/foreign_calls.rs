/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Foreign tile-library calls (`#[cutile::foreign]`) lower to Tile IR
//! `preview$call` at bytecode 13.5. Compile-only: needs neither a GPU nor
//! the tile library, which `tileiras` loads only when assembling.

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::{CompileOptions, TileLibrarySpec};
use cutile_ir::bytecode::BytecodeVersion;

mod common;

#[cutile::module]
mod lib_decls {
    use cutile::core::*;

    /// A string constant as a tag type: `NAME` is the string the library
    /// expects.
    pub trait Tag {
        const NAME: &'static str;
    }

    pub struct Potrf;
    impl crate::lib_decls::Tag for Potrf {
        const NAME: &'static str = "potrf";
    }
    pub struct Potrs;
    impl crate::lib_decls::Tag for Potrs {
        const NAME: &'static str = "potrs";
    }
    pub struct Lower;
    impl crate::lib_decls::Tag for Lower {
        const NAME: &'static str = "lower";
    }

    #[cutile::foreign(name = "fft")]
    pub unsafe fn fft<
        const AXIS: i32,
        const DIM0: i32,
        const DIM1: i32,
        const DIM2: i32,
        T: ElementType,
    >(
        #[inout] data: Tile<T, { [DIM0, DIM1, DIM2, 2] }>,
    ) -> Tile<T, { [DIM0, DIM1, DIM2, 2] }>;

    #[cutile::foreign]
    pub unsafe fn cholesky<const BATCH: i32, const M: i32, T: ElementType>(
        function: &str,
        fill_mode: &str,
        #[inout] a: Tile<T, { [BATCH, M, M] }>,
    ) -> (Tile<T, { [BATCH, M, M] }>, Tile<i32, { [BATCH] }>);

    #[cutile::foreign]
    pub unsafe fn potrs<const BATCH: i32, const M: i32, const K: i32, T: ElementType>(
        function: &str,
        fill_mode: &str,
        a: Tile<T, { [BATCH, M, M] }>,
        #[inout] b: Tile<T, { [BATCH, M, K] }>,
    ) -> Tile<T, { [BATCH, M, K] }>;

    /// A scalar operand and an output-only scalar result.
    #[cutile::foreign(name = "scale_tile")]
    pub unsafe fn scale<const N: i32, T: ElementType>(
        #[inout] x: Tile<T, { [N] }>,
        alpha: T,
    ) -> (Tile<T, { [N] }>, i32);

    /// An output-only f4E2M1FN result.
    #[cutile::foreign]
    pub unsafe fn quantize<const N: i32>(x: Tile<f32, { [N] }>) -> Tile<f4e2m1fn, { [N] }>;

    #[cutile::foreign]
    pub unsafe fn singleton(x: Tile<f32, { [32] }>) -> (Tile<f32, { [32] }>,);

    #[cutile::foreign]
    #[allow(clippy::unused_unit)]
    pub unsafe fn consume(x: Tile<f32, { [32] }>) -> ();
}

#[cutile::module]
mod kernels {
    use super::lib_decls::*;
    use cutile::core::*;

    /// FFT of each row of a `[B, N, 2]` tile, through the rank-4 declaration.
    fn fft_rows<const B: i32, const N: i32, T: ElementType>(
        x: Tile<T, { [B, N, 2] }>,
    ) -> Tile<T, { [B, N, 2] }> {
        let x4: Tile<T, { [1, B, N, 2] }> = x.reshape(const_shape![1, B, N, 2]);
        let y4 = unsafe { fft::<2, 1, B, N, T>(x4) };
        y4.reshape(const_shape![B, N, 2])
    }

    #[cutile::entry()]
    fn fft_kernel<const B: i32, const N: i32>(y: &mut Tensor<f32, { [B, N, 2] }>) {
        let t = y.load();
        y.store(fft_rows(t));
    }

    #[cutile::entry()]
    fn fft_inferred(y: &mut Tensor<f32, { [4, 64, 2] }>) {
        let t: Tile<f32, { [1, 4, 64, 2] }> = y.load().reshape(const_shape![1, 4, 64, 2]);
        let r = unsafe { fft::<1, _, _, _, _>(t) };
        y.store(r.reshape(const_shape![4, 64, 2]));
    }

    #[cutile::entry()]
    fn cholesky_solve<const M: i32, const K: i32>(
        x: &mut Tensor<f64, { [1, M, K] }>,
        info: &mut Tensor<i32, { [1] }>,
        a: &Tensor<f64, { [-1, -1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let a_tile: Tile<f64, { [1, M, M] }> = a.partition(shape![1, M, M]).load([pid.0, 0, 0]);
        let (l, status) = unsafe { cholesky::<1, M, f64>(Potrf::NAME, Lower::NAME, a_tile) };
        let b = x.load();
        x.store(unsafe { potrs::<1, M, K, f64>(Potrs::NAME, Lower::NAME, l, b) });
        info.store(status);
    }

    #[cutile::entry()]
    fn scale_kernel<const N: i32>(y: &mut Tensor<f32, { [N] }>, alpha: f32) {
        let (scaled, _status) = unsafe { scale::<N, f32>(y.load(), alpha) };
        y.store(scaled);
    }

    /// A literal scalar operand is a rank-0 constant of the parameter's type.
    #[cutile::entry()]
    fn scale_by_literal<const N: i32>(y: &mut Tensor<f32, { [N] }>) {
        let t = y.load();
        let (scaled, _status) = unsafe { scale::<N, f32>(t, 2.0) };
        y.store(scaled);
    }

    #[cutile::entry()]
    fn quantize_kernel(x: &mut Tensor<f32, { [64] }>, out: &mut Tensor<u8, { [32] }>) {
        let q = unsafe { quantize::<64>(x.load()) };
        let packed: Tile<u8, { [32] }> = pack(q);
        out.store(packed);
    }

    #[cutile::entry()]
    fn singleton_result(y: &mut Tensor<f32, { [32] }>) {
        let (first,) = unsafe { singleton(y.load()) };
        let second = unsafe { singleton(first) };
        y.store(second.0);
    }

    #[cutile::entry()]
    fn unit_result(y: &mut Tensor<f32, { [32] }>) {
        let t = y.load();
        unsafe { consume(t) };
        y.store(t);
    }
}

fn compile(
    entry: &str,
    generics: Vec<String>,
    strides: &[(&str, &[i32])],
) -> cutile::compile_api::CompileArtifacts {
    KernelCompiler::new(kernels::__module_ast_self, "kernels", entry)
        .generics(generics)
        .strides(strides)
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("compilation failed")
}

#[test]
fn singleton_foreign_results_keep_their_tuple_shape() {
    common::with_test_stack(|| {
        let artifacts = compile("singleton_result", vec![], &[("y", &[1])]);
        assert_eq!(call_lines(&artifacts.ir_text(), "singleton").len(), 2);
        artifacts.bytecode().expect("valid singleton call bytecode");
    });
}

#[test]
fn unit_foreign_results_remain_void_calls() {
    common::with_test_stack(|| {
        let artifacts = compile("unit_result", vec![], &[("y", &[1])]);
        let calls = call_lines(&artifacts.ir_text(), "consume");
        assert_eq!(calls.len(), 1);
        assert!(calls[0].contains("-> token"), "{calls:?}");
        artifacts.bytecode().expect("valid void call bytecode");
    });
}

#[test]
fn fft_call_infers_shape_generics_and_threads_a_token() {
    common::with_test_stack(|| {
        let artifacts = compile(
            "fft_kernel",
            vec!["4".into(), "64".into()],
            &[("y", &[128, 2, 1])],
        );
        let ir = artifacts.ir_text();
        assert!(ir.contains("preview$call @fft<[2, 1, 4, 64, f32]>"), "{ir}");
        // The module names the callee, never the library: the host
        // configures that.
        assert!(!ir.contains("tilelib"), "{ir}");
    });
}

#[test]
fn cholesky_solve_passes_string_constants_and_an_implicit_info_operand() {
    common::with_test_stack(|| {
        let artifacts = compile(
            "cholesky_solve",
            vec!["32".into(), "4".into()],
            &[("x", &[128, 4, 1]), ("info", &[1]), ("a", &[1024, 32, 1])],
        );
        let ir = artifacts.ir_text();
        assert!(
            ir.contains(r#"preview$call @cholesky<["potrf", "lower", 1, 32, f64]>"#),
            "{ir}"
        );
        assert!(
            ir.contains(r#"preview$call @potrs<["potrs", "lower", 1, 32, 4, f64]>"#),
            "{ir}"
        );
        assert_eq!(artifacts.module().foreign_callees(), ["cholesky", "potrs"]);
        // The zero-filled implicit `info` operand is passed after `A`.
        assert!(ir.contains("constant <i32: 0> : tile<1xi32>"), "{ir}");
        let call = ir
            .lines()
            .find(|l| l.contains("preview$call @cholesky<"))
            .unwrap_or_else(|| panic!("{ir}"));
        assert!(
            call.contains(
                ": (tile<1x32x32xf64>, tile<1xi32>, token) -> (tile<1x32x32xf64>, tile<1xi32>, token)"
            ),
            "{call}"
        );
        let bytes = artifacts.bytecode().expect("bytecode");
        assert_eq!(&bytes[8..12], &[13, 5, 1, 0], "13.5 with the preview bit");
    });
}

/// An output-only f4E2M1FN result is zero-filled through a legal constant:
/// a rank-0 f4 tile cannot carry the splat value.
#[test]
fn an_output_only_f4_result_is_a_legal_zero_tile() {
    common::with_test_stack(|| {
        let artifacts =
            KernelCompiler::new(kernels::__module_ast_self, "kernels", "quantize_kernel")
                .strides(&[("x", &[1]), ("out", &[1])])
                .target("sm_100")
                .bytecode_version(BytecodeVersion::V13_5)
                .compile()
                .expect("compilation failed");
        let ir = artifacts.ir_text();
        let call = ir
            .lines()
            .find(|l| l.contains("preview$call @quantize<[64]>"))
            .unwrap_or_else(|| panic!("{ir}"));
        assert!(
            call.contains(
                ": (tile<64xf32>, tile<64xf4e2m1fn>, token) -> (tile<64xf4e2m1fn>, token)"
            ),
            "{call}"
        );
        artifacts.bytecode().expect("bytecode");
    });
}

#[test]
fn scalar_operands_and_results_are_rank_0_tiles() {
    common::with_test_stack(|| {
        let artifacts = compile("scale_kernel", vec!["128".into()], &[("y", &[1])]);
        let ir = artifacts.ir_text();
        let call = ir
            .lines()
            .find(|l| l.contains("preview$call @scale_tile<[128, f32]>"))
            .unwrap_or_else(|| panic!("{ir}"));
        // x, alpha, the zero-filled i32 status, the token.
        assert!(
            call.contains(": (tile<128xf32>, tile<f32>, tile<i32>, token) -> (tile<128xf32>, tile<i32>, token)"),
            "{call}"
        );
        assert_eq!(artifacts.module().foreign_callees(), ["scale_tile"]);
    });
}

#[test]
fn turbofish_placeholders_are_inferred_from_the_arguments() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(kernels::__module_ast_self, "kernels", "fft_inferred")
            .strides(&[("y", &[128, 2, 1])])
            .target("sm_80")
            .bytecode_version(BytecodeVersion::V13_5)
            .compile()
            .expect("compilation failed");
        let ir = artifacts.ir_text();
        assert!(ir.contains("preview$call @fft<[1, 1, 4, 64, f32]>"), "{ir}");
    });
}

#[test]
fn foreign_calls_require_tile_ir_13_5() {
    common::with_test_stack(|| {
        let error = KernelCompiler::new(kernels::__module_ast_self, "kernels", "fft_kernel")
            .generics(vec!["4".into(), "64".into()])
            .strides(&[("y", &[128, 2, 1])])
            .target("sm_80")
            .bytecode_version(BytecodeVersion::V13_4)
            .compile()
            .err()
            .expect("13.4 cannot express preview$call");
        let message = error.to_string();
        assert!(message.contains("13.5"), "{message}");
    });
}

fn scale_l2_key(options: CompileOptions) -> Result<String, String> {
    KernelCompiler::new(kernels::__module_ast_self, "kernels", "scale_kernel")
        .generics(vec!["128".into()])
        .strides(&[("y", &[1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .options(options)
        .l2_cache_key()
        .map_err(|e| e.to_string())
}

/// The host configures the libraries a kernel links. Without one, a kernel
/// with foreign calls fails with an error that names the callee; a missing
/// file is reported; and the key covers the library's contents.
#[test]
fn tile_libraries_are_configured_on_the_host() {
    common::with_test_stack(|| {
        let message = scale_l2_key(CompileOptions::new()).expect_err("no library");
        assert!(
            message.contains("`scale_tile`") && message.contains("no tile library is configured"),
            "{message}"
        );

        let dir = std::env::temp_dir().join(format!("cutile_foreign_calls_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = TileLibrarySpec::new(dir.join("missing.tilelib"));
        let message =
            scale_l2_key(CompileOptions::new().tile_library(missing)).expect_err("missing library");
        assert!(message.contains("cannot be opened"), "{message}");

        let library = dir.join("scale.tilelib");
        std::fs::write(&library, b"v1").unwrap();
        let with = |spec: TileLibrarySpec| {
            scale_l2_key(CompileOptions::new().tile_library(spec)).expect("key")
        };
        let key = with(TileLibrarySpec::new(&library));
        assert_eq!(key, with(TileLibrarySpec::new(&library)));
        assert_ne!(
            key,
            with(TileLibrarySpec::new(&library).setting("version", "2")),
            "a provider setting is key material"
        );
        std::fs::write(&library, b"v2, rebuilt").unwrap();
        assert_ne!(
            key,
            with(TileLibrarySpec::new(&library)),
            "contents are key material"
        );
        let _ = std::fs::remove_dir_all(&dir);
    });
}

#[test]
fn literal_scalar_operands_are_materialized() {
    common::with_test_stack(|| {
        let artifacts = compile("scale_by_literal", vec!["32".into()], &[("y", &[1])]);
        let ir = artifacts.ir_text();
        assert!(ir.contains("constant <f32: 2"), "{ir}");
        assert!(ir.contains("preview$call @scale_tile<[32, f32]>"), "{ir}");
    });
}

/// Two binding modules that each declare a `Fill` tag, with different
/// strings. Each call passes the string of the tag its path names.
#[cutile::module]
mod lower_fill {
    use super::lib_decls::*;
    use cutile::core::*;

    pub struct Fill;
    impl crate::lib_decls::Tag for Fill {
        const NAME: &'static str = "lower";
    }

    pub fn factor_lower<const M: i32>(
        a: Tile<f64, { [1, M, M] }>,
    ) -> (Tile<f64, { [1, M, M] }>, Tile<i32, { [1] }>) {
        unsafe { cholesky::<1, M, f64>(Potrf::NAME, Fill::NAME, a) }
    }

    /// Returns this module's `Fill`.
    pub fn fill() -> Fill {
        Fill
    }

    pub struct FillMaker;
    impl FillMaker {
        /// Returns this module's `Fill`.
        pub fn fill(self) -> Fill {
            Fill
        }
    }
}

#[cutile::module]
mod upper_fill {
    use super::lib_decls::*;
    use cutile::core::*;

    pub struct Fill;
    impl crate::lib_decls::Tag for Fill {
        const NAME: &'static str = "upper";
    }

    pub fn factor_upper<const M: i32>(
        a: Tile<f64, { [1, M, M] }>,
    ) -> (Tile<f64, { [1, M, M] }>, Tile<i32, { [1] }>) {
        unsafe { cholesky::<1, M, f64>(Potrf::NAME, Fill::NAME, a) }
    }

    pub fn factor_with<F: crate::lib_decls::Tag, const M: i32>(
        a: Tile<f64, { [1, M, M] }>,
        _fill: F,
    ) -> (Tile<f64, { [1, M, M] }>, Tile<i32, { [1] }>) {
        unsafe { cholesky::<1, M, f64>(Potrf::NAME, F::NAME, a) }
    }

    /// Takes this module's `Fill` as a parameter of that concrete type: the
    /// tag reaches `factor_with`'s `F` through the declared type.
    pub fn factor_fill<const M: i32>(
        a: Tile<f64, { [1, M, M] }>,
        fill: Fill,
    ) -> (Tile<f64, { [1, M, M] }>, Tile<i32, { [1] }>) {
        factor_with(a, fill)
    }
}

#[cutile::module]
mod same_named_markers {
    use super::lib_decls::{cholesky, Potrf};
    use super::lower_fill::factor_lower;
    use super::upper_fill::factor_upper;
    #[allow(unused_imports)]
    use crate::lib_decls::Tag;
    use cutile::core::*;

    #[cutile::entry()]
    fn both_fills<const M: i32>(
        l: &mut Tensor<f64, { [1, M, M] }>,
        u: &mut Tensor<f64, { [1, M, M] }>,
    ) {
        let (lf, _) = factor_lower(l.load());
        l.store(lf);
        let (uf, _) = factor_upper(u.load());
        u.store(uf);
    }

    /// The fill mode is a kernel generic the host names.
    #[cutile::entry()]
    fn host_fill<F: crate::lib_decls::Tag, const M: i32>(a: &mut Tensor<f64, { [1, M, M] }>) {
        let (factor, _) = unsafe { cholesky::<1, M, f64>(Potrf::NAME, F::NAME, a.load()) };
        a.store(factor);
    }
}

/// Imports `lower_fill`'s `Fill` and calls `upper_fill::factor_fill`, whose
/// parameter is `upper_fill`'s `Fill`.
#[cutile::module]
mod marker_parameters {
    #[allow(unused_imports)]
    use super::lower_fill::Fill;
    use super::upper_fill;
    use cutile::core::*;

    #[cutile::entry()]
    fn concrete_marker_parameter<const M: i32>(a: &mut Tensor<f64, { [1, M, M] }>) {
        let (factor, _) = upper_fill::factor_fill(a.load(), upper_fill::Fill);
        a.store(factor);
    }
}

/// Passes `lower_fill`'s `Fill`, returned by a function and by a method, to
/// `upper_fill::factor_with`, whose module declares another `Fill`.
#[cutile::module]
mod returned_markers {
    use super::lower_fill;
    use super::upper_fill::factor_with;
    use cutile::core::*;

    #[cutile::entry()]
    fn returned_marker<const M: i32>(a: &mut Tensor<f64, { [1, M, M] }>) {
        let from_fn = lower_fill::fill();
        let (f1, _) = factor_with(a.load(), from_fn);
        let from_method = lower_fill::FillMaker.fill();
        let (f2, _) = factor_with(f1, from_method);
        a.store(f2);
    }
}

fn call_lines(ir: &str, callee: &str) -> Vec<String> {
    ir.lines()
        .filter(|l| l.contains(&format!("preview$call @{callee}")))
        .map(str::to_string)
        .collect()
}

#[test]
fn markers_with_one_name_pass_the_string_their_path_names() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            same_named_markers::__module_ast_self,
            "same_named_markers",
            "both_fills",
        )
        .generics(vec!["16".into()])
        .strides(&[("l", &[256, 16, 1]), ("u", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("each `Fill` resolves in its own module");
        let calls = call_lines(&artifacts.ir_text(), "cholesky");
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert!(
            calls[0].contains(r#"@cholesky<["potrf", "lower", 1, 16, f64]>"#),
            "{calls:?}"
        );
        assert!(
            calls[1].contains(r#"@cholesky<["potrf", "upper", 1, 16, f64]>"#),
            "{calls:?}"
        );
    });
}

/// A tag that reaches a foreign call through a callee's parameter of
/// concrete tag type is that callee's tag, not a same-named one the
/// kernel imports.
#[test]
fn a_concrete_marker_parameter_keeps_the_callee_marker() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            marker_parameters::__module_ast_self,
            "marker_parameters",
            "concrete_marker_parameter",
        )
        .generics(vec!["16".into()])
        .strides(&[("a", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("compiles");
        let calls = call_lines(&artifacts.ir_text(), "cholesky");
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert!(
            calls[0].contains(r#"@cholesky<["potrf", "upper", 1, 16, f64]>"#),
            "{calls:?}"
        );
    });
}

/// A tag that a function or a method returns is the tag its return type
/// names in the callee's module, not a same-named one where it is used.
#[test]
fn a_returned_marker_keeps_the_callee_marker() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            returned_markers::__module_ast_self,
            "returned_markers",
            "returned_marker",
        )
        .generics(vec!["16".into()])
        .strides(&[("a", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("compiles");
        let calls = call_lines(&artifacts.ir_text(), "cholesky");
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert!(
            calls
                .iter()
                .all(|call| call.contains(r#"@cholesky<["potrf", "lower", 1, 16, f64]>"#)),
            "{calls:?}"
        );
    });
}

fn compile_host_fill(fill: &str) -> Result<cutile::compile_api::CompileArtifacts, String> {
    KernelCompiler::new(
        same_named_markers::__module_ast_self,
        "same_named_markers",
        "host_fill",
    )
    .generics(vec![fill.into(), "16".into()])
    .strides(&[("a", &[256, 16, 1])])
    .target("sm_80")
    .bytecode_version(BytecodeVersion::V13_5)
    .compile()
    .map_err(|e| e.to_string())
}

/// A tag the host names by its module path is that module's tag. A
/// bare name that two modules declare with different strings is an error.
#[test]
fn a_host_named_marker_resolves_by_its_path() {
    common::with_test_stack(|| {
        let artifacts = compile_host_fill("upper_fill::Fill").expect("compiles");
        let calls = call_lines(&artifacts.ir_text(), "cholesky");
        assert!(
            calls[0].contains(r#"@cholesky<["potrf", "upper", 1, 16, f64]>"#),
            "{calls:?}"
        );

        let message = compile_host_fill("Fill")
            .err()
            .expect("the two `Fill` markers clash");
        assert!(
            message.contains("`Fill` is defined in several modules"),
            "{message}"
        );
        assert!(
            message.contains("`lower_fill`") && message.contains("`upper_fill`"),
            "{message}"
        );
    });
}

/// A tag that shares its name with the library's `Lower`.
#[cutile::module]
mod decoy_fill {
    pub struct Lower;
    impl crate::lib_decls::Tag for Lower {
        const NAME: &'static str = "decoy";
    }
}

/// Loads the decoy `Lower` next to the library's, and imports neither.
#[cutile::module]
mod host_fill_with_decoy {
    #[allow(unused_imports)]
    use super::decoy_fill;
    use super::lib_decls::{cholesky, Potrf};
    #[allow(unused_imports)]
    use crate::lib_decls::Tag;
    use cutile::core::*;

    #[cutile::entry()]
    fn host_fill<F: crate::lib_decls::Tag, const M: i32>(a: &mut Tensor<f64, { [1, M, M] }>) {
        let (factor, _) = unsafe { cholesky::<1, M, f64>(Potrf::NAME, F::NAME, a.load()) };
        a.store(factor);
    }
}

/// A bare host-named tag is ambiguous even when the declaration's own
/// module declares one of that name: the call does not use that one.
#[test]
fn a_bare_host_marker_is_not_taken_from_the_declarations_module() {
    common::with_test_stack(|| {
        let message = KernelCompiler::new(
            host_fill_with_decoy::__module_ast_self,
            "host_fill_with_decoy",
            "host_fill",
        )
        .generics(vec!["Lower".into(), "16".into()])
        .strides(&[("a", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .err()
        .expect("the two `Lower` markers clash")
        .to_string();
        assert!(
            message.contains("`Lower` is defined in several modules")
                && message.contains("`decoy_fill`")
                && message.contains("`lib_decls`"),
            "{message}"
        );
    });
}

/// Wrappers that re-export the library's `Lower` and dispatch on the tile
/// type through a trait, so the tag reaches the foreign call as the type
/// variable `F`.
#[cutile::module]
mod fill_wrappers {
    pub use super::lib_decls::Lower;
    use super::lib_decls::{cholesky, Potrf};
    #[allow(unused_imports)]
    use crate::lib_decls::Tag;
    use cutile::core::*;

    pub trait Factor {
        type Out;
        fn factor_impl<F: crate::lib_decls::Tag>(self, fill: F) -> Self::Out;
    }

    impl<T: ElementType, const M: i32> Factor for Tile<T, { [1, M, M] }> {
        type Out = (Tile<T, { [1, M, M] }>, Tile<i32, { [1] }>);
        fn factor_impl<F: crate::lib_decls::Tag>(
            self,
            fill: F,
        ) -> (Tile<T, { [1, M, M] }>, Tile<i32, { [1] }>) {
            unsafe { cholesky::<1, M, T>(Potrf::NAME, F::NAME, self) }
        }
    }

    pub fn factor<A: Factor, F: crate::lib_decls::Tag>(a: A, fill: F) -> <A as Factor>::Out {
        a.factor_impl(fill)
    }
}

#[cutile::module]
mod reexported_markers {
    use super::decoy_fill;
    use super::fill_wrappers::{factor, Lower};
    use cutile::core::*;

    #[cutile::entry()]
    fn through_wrappers<const M: i32>(
        a: &mut Tensor<f64, { [1, M, M] }>,
        b: &mut Tensor<f64, { [1, M, M] }>,
    ) {
        let (fa, _) = factor(a.load(), Lower);
        a.store(fa);
        let (fb, _) = factor(b.load(), decoy_fill::Lower);
        b.store(fb);
    }
}

/// `Lower` reaches the foreign call through the wrappers' `pub use` and the
/// trait method's `F`; `decoy_fill::Lower` takes the same route.
#[test]
fn markers_follow_reexports_and_type_variables() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            reexported_markers::__module_ast_self,
            "reexported_markers",
            "through_wrappers",
        )
        .generics(vec!["16".into()])
        .strides(&[("a", &[256, 16, 1]), ("b", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("compiles");
        let calls = call_lines(&artifacts.ir_text(), "cholesky");
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert!(
            calls[0].contains(r#"@cholesky<["potrf", "lower", 1, 16, f64]>"#),
            "{calls:?}"
        );
        assert!(
            calls[1].contains(r#"@cholesky<["potrf", "decoy", 1, 16, f64]>"#),
            "{calls:?}"
        );
    });
}

#[cutile::module]
mod alias_markers {
    use cutile::core::*;

    pub struct Target;
    impl crate::lib_decls::Tag for Target {
        const NAME: &'static str = "lower";
    }
    pub type Fill = Target;
    pub type Chained = Fill;
    pub type Generic<T> = T;

    #[cutile::foreign]
    pub unsafe fn alias_call(fill: &str, #[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;

    fn call_with_marker<F: crate::lib_decls::Tag>(
        x: Tile<f32, { [32] }>,
        _fill: F,
    ) -> Tile<f32, { [32] }> {
        unsafe { alias_call(F::NAME, x) }
    }

    pub fn alias_wrapper(x: Tile<f32, { [32] }>, fill: Chained) -> Tile<f32, { [32] }> {
        call_with_marker(x, fill)
    }

    pub fn tuple_alias_wrapper(
        x: Tile<f32, { [32] }>,
        fill: (Chained, i32),
    ) -> Tile<f32, { [32] }> {
        call_with_marker(x, fill.0)
    }
}

#[cutile::module]
mod alias_decoys {
    pub struct Fill;
    impl crate::lib_decls::Tag for Fill {
        const NAME: &'static str = "upper";
    }
    pub struct Target;
    impl crate::lib_decls::Tag for Target {
        const NAME: &'static str = "decoy";
    }
    pub struct Upper;
    impl crate::lib_decls::Tag for Upper {
        const NAME: &'static str = "upper";
    }
    pub type Chained = Upper;
}

#[cutile::module]
mod alias_kernels {
    #[allow(unused_imports)]
    use super::alias_decoys;
    use super::alias_markers::{
        self, alias_call, alias_wrapper, tuple_alias_wrapper, Chained, Fill, Generic, Target,
    };
    #[allow(unused_imports)]
    use crate::lib_decls::Tag;
    use cutile::core::*;

    #[cutile::entry()]
    fn aliased_fill(x: &mut Tensor<f32, { [32] }>) {
        let a = unsafe { alias_call(Fill::NAME, x.load()) };
        let b = unsafe { alias_call(Chained::NAME, a) };
        let c = unsafe { alias_call(<Generic<Fill>>::NAME, b) };
        let d = unsafe { alias_call(<Generic<Generic<alias_markers::Fill>>>::NAME, c) };
        x.store(d);
    }

    #[cutile::entry()]
    fn wrapped_fill(x: &mut Tensor<f32, { [32] }>) {
        let a = alias_wrapper(x.load(), Target);
        x.store(a);
    }

    #[cutile::entry()]
    fn tuple_fill(x: &mut Tensor<f32, { [32] }>) {
        let a = tuple_alias_wrapper(x.load(), (Target, 0i32));
        x.store(a);
    }
}

#[test]
fn marker_aliases_resolve_in_their_defining_scope() {
    common::with_test_stack(|| {
        for (entry, count) in [("aliased_fill", 4), ("wrapped_fill", 1), ("tuple_fill", 1)] {
            let artifacts =
                KernelCompiler::new(alias_kernels::__module_ast_self, "alias_kernels", entry)
                    .strides(&[("x", &[1])])
                    .target("sm_80")
                    .bytecode_version(BytecodeVersion::V13_5)
                    .compile()
                    .expect("marker aliases compile");
            let calls = call_lines(&artifacts.ir_text(), "alias_call");
            assert_eq!(calls.len(), count, "{calls:?}");
            assert!(
                calls
                    .iter()
                    .all(|call| call.contains(r#"@alias_call<["lower"]>"#)),
                "{entry}: {calls:?}"
            );
            artifacts.bytecode().expect("valid alias call bytecode");
        }
    });
}

/// Two libraries that each export a function named `scale`.
#[cutile::module]
mod ns_a {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches library A's `scale_a`.
    #[cutile::foreign(name = "scale_a")]
    pub unsafe fn scale(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;
}

#[cutile::module]
mod ns_b {
    use cutile::core::*;

    /// # Safety
    ///
    /// Matches library B's `scale_b`.
    #[cutile::foreign(name = "scale_b")]
    pub unsafe fn scale(#[inout] x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;
}

/// An ordinary device function that shares the foreign functions' name.
#[cutile::module]
mod ns_plain {
    use cutile::core::*;

    pub fn scale(x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }> {
        x + x
    }
}

#[cutile::module]
mod ns_kernels {
    use super::ns_a::*;
    use super::ns_b::scale as scale_b;
    #[allow(unused_imports)]
    use super::ns_plain;
    use cutile::core::*;

    /// Absolute item paths name each library's `scale`.
    #[cutile::entry()]
    fn absolute_paths(x: &mut Tensor<f32, { [32] }>) {
        let a = unsafe { crate::ns_a::scale(x.load()) };
        let b = unsafe { crate::ns_b::scale(a) };
        x.store(b);
    }

    /// `super::` paths name the same functions.
    #[cutile::entry()]
    fn relative_paths(x: &mut Tensor<f32, { [32] }>) {
        let a = unsafe { super::ns_b::scale(x.load()) };
        let b = unsafe { super::ns_a::scale(a) };
        x.store(b);
    }

    /// The glob import brings `ns_a::scale`; `ns_b::scale` is renamed.
    #[cutile::entry()]
    fn imported_names(x: &mut Tensor<f32, { [32] }>) {
        let a = unsafe { scale(x.load()) };
        let b = unsafe { scale_b(a) };
        x.store(b);
    }

    /// The ordinary function of the same name is inlined, not called.
    #[cutile::entry()]
    fn plain_and_foreign(x: &mut Tensor<f32, { [32] }>) {
        let a = crate::ns_plain::scale(x.load());
        let b = unsafe { crate::ns_a::scale(a) };
        x.store(b);
    }
}

fn compile_ns(entry: &str) -> String {
    KernelCompiler::new(ns_kernels::__module_ast_self, "ns_kernels", entry)
        .strides(&[("x", &[1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
        .ir_text()
}

/// Same-named foreign functions in different modules resolve through the
/// ordinary namespace rules: absolute and `super::` paths, glob imports and
/// renaming imports each reach the function they name.
#[test]
fn same_named_foreign_functions_resolve_per_module() {
    common::with_test_stack(|| {
        for (entry, order) in [
            ("absolute_paths", ["scale_a", "scale_b"]),
            ("relative_paths", ["scale_b", "scale_a"]),
            ("imported_names", ["scale_a", "scale_b"]),
        ] {
            let ir = compile_ns(entry);
            let callees: Vec<&str> = ir
                .lines()
                .filter_map(|l| l.split("preview$call @").nth(1))
                .map(|rest| rest.split(['(', '<']).next().unwrap())
                .collect();
            assert_eq!(callees, order, "{entry}:\n{ir}");
        }
        let ir = compile_ns("plain_and_foreign");
        assert_eq!(call_lines(&ir, "scale_a").len(), 1, "{ir}");
        assert!(call_lines(&ir, "scale_b").is_empty(), "{ir}");
        assert!(ir.contains("addf"), "the plain `scale` is inlined:\n{ir}");
    });
}

/// Consts that `string_args` names by their module path.
#[cutile::module]
mod string_consts {
    /// Shares its name, not its value, with `string_args::FILL`.
    pub const FILL: &str = "upper";
}

/// Every spelling of a string constant the JIT resolves.
#[cutile::module]
mod string_args {
    use super::lib_decls::*;
    use super::string_consts;
    use cutile::core::*;

    const FILL: &str = "lower";
    /// Not a literal: rustc evaluates it, the JIT does not.
    const COMPUTED: &str = concat!("lo", "wer");

    fn factor_with(fill: &str, a: Tile<f64, { [1, 16, 16] }>) -> Tile<f64, { [1, 16, 16] }> {
        let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, a) };
        factor
    }

    #[cutile::entry()]
    fn spellings(a: &mut Tensor<f64, { [1, 16, 16] }>) {
        let local = "upper";
        let (f1, _) = unsafe { cholesky::<1, 16, f64>("potrf", "lower", a.load()) };
        let (f2, _) = unsafe { cholesky::<1, 16, f64>(Potrf::NAME, FILL, f1) };
        let (f3, _) = unsafe { cholesky::<1, 16, f64>("potrf", local, f2) };
        let f4 = factor_with("upper", f3);
        let (f5, _) = unsafe { cholesky::<1, 16, f64>("potrf", string_consts::FILL, f4) };
        let (f6, _) = unsafe { cholesky::<1, 16, f64>("potrf", crate::string_consts::FILL, f5) };
        a.store(f6);
    }

    #[cutile::entry()]
    fn computed(a: &mut Tensor<f64, { [1, 16, 16] }>) {
        let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", COMPUTED, a.load()) };
        a.store(factor);
    }

    /// Each call passes the string `fill` holds where it is written.
    #[cutile::entry()]
    fn reassigned_in_sequence(a: &mut Tensor<f64, { [1, 16, 16] }>) {
        let mut fill = "lower";
        let (f1, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, a.load()) };
        fill = "upper";
        let (f2, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, f1) };
        a.store(f2);
    }

    /// `fill` is "upper" after the `if` only when `flag > 0`.
    #[cutile::entry()]
    fn reassigned_in_branch(a: &mut Tensor<f64, { [1, 16, 16] }>, flag: i32) {
        let mut fill = "lower";
        if flag > 0 {
            fill = "upper";
        }
        let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, a.load()) };
        a.store(factor);
    }

    /// The reassignment is inside an `unsafe {}` block in the branch.
    #[cutile::entry()]
    fn reassigned_in_unsafe_in_branch(a: &mut Tensor<f64, { [1, 16, 16] }>, flag: i32) {
        let mut fill = "lower";
        if flag > 0 {
            #[allow(unused_unsafe)]
            unsafe {
                fill = "upper";
            }
        }
        let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, a.load()) };
        a.store(factor);
    }

    /// A nested `if` that does not assign `fill` leaves it known: the call
    /// in the outer branch passes "upper".
    #[cutile::entry()]
    fn nested_if_keeps_the_string(a: &mut Tensor<f64, { [1, 16, 16] }>, flag: i32) {
        #[allow(unused_assignments)]
        let mut fill = "lower";
        let mut t = a.load();
        if flag > 0 {
            fill = "upper";
            if flag > 1 {
                t = a.load();
            }
            let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, t) };
            t = factor;
        }
        a.store(t);
    }

    /// The first iteration passes "lower", the next ones "upper".
    #[cutile::entry()]
    fn reassigned_in_loop(a: &mut Tensor<f64, { [1, 16, 16] }>, n: i32) {
        let mut fill = "lower";
        let mut t = a.load();
        for _ in 0..n {
            let (factor, _) = unsafe { cholesky::<1, 16, f64>("potrf", fill, t) };
            t = factor;
            fill = "upper";
        }
        a.store(t);
    }
}

fn compile_string_args(entry: &str) -> Result<String, String> {
    KernelCompiler::new(string_args::__module_ast_self, "string_args", entry)
        .strides(&[("a", &[256, 16, 1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .map(|artifacts| artifacts.ir_text())
        .map_err(|e| e.to_string())
}

/// A string constant can be a literal, an associated const, a `const`
/// item (also by its module path), a local bound to a literal, or a
/// wrapper's `&str` parameter.
#[test]
fn string_constants_resolve_when_the_kernel_is_compiled() {
    common::with_test_stack(|| {
        let ir = compile_string_args("spellings").expect("compiles");
        let fills: Vec<&str> = call_lines(&ir, "cholesky")
            .iter()
            .map(|l| {
                if l.contains(r#"<["potrf", "lower", 1, 16, f64]>"#) {
                    "lower"
                } else if l.contains(r#"<["potrf", "upper", 1, 16, f64]>"#) {
                    "upper"
                } else {
                    panic!("{l}")
                }
            })
            .collect();
        assert_eq!(
            fills,
            ["lower", "lower", "upper", "upper", "upper", "upper"],
            "{ir}"
        );
    });
}

/// A string the JIT cannot evaluate is a compile error that says what it
/// accepts.
#[test]
fn a_string_the_jit_cannot_evaluate_is_an_error() {
    common::with_test_stack(|| {
        let message = compile_string_args("computed").expect_err("not a literal");
        assert!(
            message.contains("`fill_mode` must be a string known when the kernel is compiled"),
            "{message}"
        );
        assert!(
            message.contains("got `COMPUTED`") && !message.contains("__cutile_node_id"),
            "the message shows the expression as written: {message}"
        );
    });
}

/// A `&str` local reassigned in a runtime `if` or loop holds a string known
/// only at run time, so passing it is a compile error. Reassigned outside
/// regions, it passes the string it holds at each call.
#[test]
fn a_string_reassigned_in_a_branch_or_loop_is_an_error() {
    common::with_test_stack(|| {
        for entry in [
            "reassigned_in_branch",
            "reassigned_in_loop",
            "reassigned_in_unsafe_in_branch",
        ] {
            let message = compile_string_args(entry).expect_err(entry);
            assert!(
                message.contains("`fill_mode` must be a string known when the kernel is compiled"),
                "{entry}: {message}"
            );
        }
        let ir = compile_string_args("reassigned_in_sequence").expect("compiles");
        let calls = call_lines(&ir, "cholesky");
        assert_eq!(calls.len(), 2, "{ir}");
        assert!(
            calls[0].contains(r#"<["potrf", "lower", 1, 16, f64]>"#)
                && calls[1].contains(r#"<["potrf", "upper", 1, 16, f64]>"#),
            "{calls:?}"
        );
        let ir = compile_string_args("nested_if_keeps_the_string").expect("compiles");
        let calls = call_lines(&ir, "cholesky");
        assert!(
            calls.len() == 1 && calls[0].contains(r#"<["potrf", "upper", 1, 16, f64]>"#),
            "{calls:?}"
        );
    });
}

/// A tag whose `NAME` differs between two traits.
#[cutile::module]
mod two_names {
    use super::lib_decls::cholesky;
    use cutile::core::*;

    pub trait Uplo {
        const NAME: &'static str;
    }

    pub struct Lower;
    impl crate::lib_decls::Tag for Lower {
        const NAME: &'static str = "lower";
    }
    impl Uplo for Lower {
        const NAME: &'static str = "L";
    }

    fn factor_with<F: crate::lib_decls::Tag + Uplo>(
        a: Tile<f64, { [1, 16, 16] }>,
        _fill: F,
    ) -> Tile<f64, { [1, 16, 16] }> {
        let (factor, _) = unsafe {
            cholesky::<1, 16, f64>(<F as Uplo>::NAME, <F as crate::lib_decls::Tag>::NAME, a)
        };
        factor
    }

    #[cutile::entry()]
    fn trait_named(a: &mut Tensor<f64, { [1, 16, 16] }>) {
        let f1 = factor_with(a.load(), Lower);
        let (f2, _) = unsafe {
            cholesky::<1, 16, f64>(
                <Lower as Uplo>::NAME,
                <Lower as crate::lib_decls::Tag>::NAME,
                f1,
            )
        };
        a.store(f2);
    }
}

/// `<T as Trait>::NAME` passes the `NAME` of that trait's impl, for a
/// generic parameter and for a tag type.
#[test]
fn a_trait_qualified_name_reads_that_traits_impl() {
    common::with_test_stack(|| {
        let ir = KernelCompiler::new(two_names::__module_ast_self, "two_names", "trait_named")
            .strides(&[("a", &[256, 16, 1])])
            .target("sm_80")
            .bytecode_version(BytecodeVersion::V13_5)
            .compile()
            .expect("compiles")
            .ir_text();
        let calls = call_lines(&ir, "cholesky");
        assert_eq!(calls.len(), 2, "{ir}");
        assert!(
            calls
                .iter()
                .all(|call| call.contains(r#"@cholesky<["L", "lower", 1, 16, f64]>"#)),
            "{calls:?}"
        );
    });
}

/// Two `#[cutile::module]`s named `lib` in different places, each declaring
/// a foreign `scale`: each renamed import reaches its own module's function.
mod same_leaf_a {
    #[cutile::module]
    pub mod lib {
        use cutile::core::*;
        /// # Safety
        ///
        /// Test declaration; never linked.
        #[cutile::foreign(name = "same_leaf_a_scale")]
        pub unsafe fn scale(x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;
    }
}
mod same_leaf_b {
    #[cutile::module]
    pub mod lib {
        use cutile::core::*;
        /// # Safety
        ///
        /// Test declaration; never linked.
        #[cutile::foreign(name = "same_leaf_b_scale")]
        pub unsafe fn scale(x: Tile<f32, { [32] }>) -> Tile<f32, { [32] }>;
    }
}
#[cutile::module]
mod same_leaf_kernels {
    use super::same_leaf_a::lib::scale as scale_a;
    use super::same_leaf_b::lib::scale as scale_b;
    use cutile::core::*;
    #[cutile::entry()]
    fn run(x: &mut Tensor<f32, { [32] }>) {
        let a = unsafe { scale_a(x.load()) };
        let b = unsafe { scale_b(a) };
        x.store(b);
    }
}

#[test]
fn foreign_functions_of_same_named_modules_stay_distinct() {
    common::with_test_stack(|| {
        let ir = KernelCompiler::new(
            same_leaf_kernels::__module_ast_self,
            "same_leaf_kernels",
            "run",
        )
        .strides(&[("x", &[1])])
        .target("sm_80")
        .bytecode_version(BytecodeVersion::V13_5)
        .compile()
        .expect("compile")
        .ir_text();
        assert!(
            ir.contains("preview$call @same_leaf_a_scale")
                && ir.contains("preview$call @same_leaf_b_scale"),
            "{ir}"
        );
    });
}
