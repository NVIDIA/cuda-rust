/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage: a const-generic tile type spelled with symbols
//! (`{ [BM] }`) is the same type as the concrete type a generic op produces
//! for the same instantiation (`{ [4] }`).

use cutile::compile_api::KernelCompiler;
use cutile_compiler::ast::Module;

mod common;

#[cutile::module]
mod type_coherence_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn reduce_sum_then_mul<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(sums * factor);
    }

    #[cutile::entry()]
    fn reduce_max_then_scale<const BM: i32, const BN: i32>(
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

    #[cutile::entry()]
    fn softmax_via_helper<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = x.load_like(y);
        y.store(normalize_rows(tile_x, y.shape()));
    }

    fn normalize_rows<const BM: i32, const BN: i32>(
        t: Tile<f32, { [BM, BN] }>,
        shape: Shape<{ [BM, BN] }>,
    ) -> Tile<f32, { [BM, BN] }> {
        let row_max: Tile<f32, { [BM] }> = reduce_max(t, 1i32);
        let shifted: Tile<f32, { [BM, BN] }> = t - row_max.reshape(shape![BM, 1]).broadcast(shape);
        let num: Tile<f32, { [BM, BN] }> = exp(shifted);
        let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1i32);
        num / denom.reshape(shape![BM, 1]).broadcast(shape)
    }

    #[cutile::entry()]
    fn softmax_via_literal_shape_helper<const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = x.load_like(y);
        y.store(normalize_rows_literal_shape(tile_x));
    }

    fn normalize_rows_literal_shape<const BM: i32, const BN: i32>(
        t: Tile<f32, { [BM, BN] }>,
    ) -> Tile<f32, { [BM, BN] }> {
        let denom: Tile<f32, { [BM] }> = reduce_sum(t, 1i32);
        t / denom.reshape(shape![BM, 1]).broadcast(shape![BM, BN])
    }

    #[cutile::entry()]
    fn scaled_sum_via_helper<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        out.store(scaled_sum(values, out.shape()));
    }

    fn scaled_sum<const N: i32>(
        values: Tile<f32, { [N, 2] }>,
        shape: Shape<{ [N] }>,
    ) -> Tile<f32, { [N] }> {
        let sums: Tile<f32, { [N] }> = reduce_sum(values, 1i32);
        let scale: Tile<f32, { [N] }> = broadcast_scalar(2f32, shape);
        sums * scale
    }

    #[cutile::entry()]
    fn tile_to_scalar_then_scalar_mul(out: &mut Tensor<f32, { [4] }>, x: f32) {
        let t: Tile<f32, { [] }> = scalar_to_tile(x);
        let s: f32 = tile_to_scalar(t);
        let y: f32 = s * x;
        out.store(broadcast_scalar(y, out.shape()));
    }

    #[cutile::entry()]
    fn generic_scalar_to_tile_then_tile_add<E: ElementType, const N: i32>(
        out: &mut Tensor<E, { [N] }>,
        x: E,
    ) {
        let t: Tile<E, { [] }> = scalar_to_tile(x);
        let u: Tile<E, { [] }> = broadcast_scalar(x, shape![]);
        let sum: Tile<E, { [] }> = t + u;
        out.store(sum.reshape(shape![1]).broadcast(out.shape()));
    }

    #[cutile::entry()]
    fn tuple_parameter(out: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let a: Tile<f32, { [4] }> = x.partition(shape![4]).load([0]);
        out.store(first((a, 1i32)));
    }

    fn first(pair: (Tile<f32, { [4] }>, i32)) -> Tile<f32, { [4] }> {
        pair.0
    }

    fn sum_rows<const M: i32>(values: Tile<f32, { [M, 2] }>) -> Tile<f32, { [M] }> {
        reduce_sum(values, 1i32)
    }

    fn sum_rows_annotated<const M: i32>(values: Tile<f32, { [M, 2] }>) -> Tile<f32, { [M] }> {
        let sums: Tile<f32, { [M] }> = reduce_sum(values, 1i32);
        sums
    }

    #[cutile::entry()]
    fn helper_result_times_symbolic<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(sum_rows(values) * factor);
    }

    #[cutile::entry()]
    fn symbolic_times_helper_result<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(factor * sum_rows(values));
    }

    #[cutile::entry()]
    fn annotated_helper_result_times_symbolic<const N: i32>(
        out: &mut Tensor<f32, { [N] }>,
        pairs: &Tensor<f32, { [-1, 2] }>,
        scale: &Tensor<f32, { [-1] }>,
    ) {
        let values: Tile<f32, { [N, 2] }> = pairs.partition(shape![N, 2]).load([0, 0]);
        let factor: Tile<f32, { [N] }> = scale.partition(shape![N]).load([0]);
        out.store(sum_rows_annotated(values) * factor);
    }
}

use type_coherence_module::__module_ast_self;

fn compile(function_name: &str, generics: &[&str], strides: &[(&str, &[i32])]) {
    let function_name = function_name.to_string();
    let generics: Vec<String> = generics.iter().map(|g| g.to_string()).collect();
    let strides: Vec<(String, Vec<i32>)> = strides
        .iter()
        .map(|(name, s)| (name.to_string(), s.to_vec()))
        .collect();
    common::with_test_stack(move || {
        let strides: Vec<(&str, &[i32])> = strides
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        KernelCompiler::new(
            __module_ast_self as fn() -> Module,
            "type_coherence_module",
            &function_name,
        )
        .generics(generics)
        .strides(&strides)
        .target("sm_120")
        .compile()
        .unwrap_or_else(|err| panic!("`{function_name}` should compile: {err}"));
    });
}

#[test]
fn reduce_sum_result_matches_symbolic_operand() {
    compile(
        "reduce_sum_then_mul",
        &["4"],
        &[("out", &[1]), ("pairs", &[2, 1]), ("scale", &[1])],
    );
}

#[test]
fn reduce_max_result_matches_symbolic_operand() {
    compile(
        "reduce_max_then_scale",
        &["2", "16"],
        &[("y", &[16, 1]), ("x", &[16, 1])],
    );
}

// #1378's helper repros. They already compile before this change (and at
// cutile-rs v0.4.0); they guard the inline boundary the change touches.
#[test]
fn helper_signature_type_matches_concrete_broadcast() {
    compile(
        "softmax_via_helper",
        &["4", "128"],
        &[("y", &[128, 1]), ("x", &[128, 1])],
    );
}

#[test]
fn helper_signature_type_matches_literal_shape_broadcast() {
    compile(
        "softmax_via_literal_shape_helper",
        &["4", "128"],
        &[("y", &[128, 1]), ("x", &[128, 1])],
    );
}

#[test]
fn helper_reduce_result_matches_symbolic_operand() {
    compile(
        "scaled_sum_via_helper",
        &["4"],
        &[("out", &[1]), ("pairs", &[2, 1])],
    );
}

// A reduction returned from a helper reaches the caller with concrete dims,
// whether or not the helper annotates it.
#[test]
fn helper_result_matches_symbolic_operand() {
    for function_name in [
        "helper_result_times_symbolic",
        "symbolic_times_helper_result",
        "annotated_helper_result_times_symbolic",
    ] {
        compile(
            function_name,
            &["4"],
            &[("out", &[1]), ("pairs", &[2, 1]), ("scale", &[1])],
        );
    }
}

#[test]
fn tuple_parameter_keeps_its_spelling() {
    compile("tuple_parameter", &[], &[("out", &[1]), ("x", &[1])]);
}

#[test]
fn tile_to_scalar_result_is_a_scalar() {
    compile("tile_to_scalar_then_scalar_mul", &[], &[("out", &[1])]);
}

#[test]
fn generic_scalar_to_tile_result_is_a_tile() {
    compile(
        "generic_scalar_to_tile_then_tile_add",
        &["f32", "4"],
        &[("out", &[1])],
    );
}

// Kernels rustc would reject, so they reach the JIT as raw source: the JIT
// must reject them too.
const MISMATCHED_MODULE: &str = r#"
mod mismatched_module {
    use cutile::core::*;

    fn takes_unknown(t: Unknown) -> Tile<f32, { [4] }> {
        t.tile
    }

    #[cutile::entry()]
    fn unresolvable_parameter(out: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let a: Tile<f32, { [4] }> = x.partition(shape![4]).load([0]);
        out.store(takes_unknown(a));
    }

    #[cutile::entry()]
    fn binary_shape(out: &mut Tensor<f32, { [4] }>, x: &Tensor<f32, { [-1] }>) {
        let a: Tile<f32, { [4] }> = x.partition(shape![4]).load([0]);
        let b: Tile<f32, { [8] }> = x.partition(shape![8]).load([0]);
        out.store(a * b);
    }

    #[cutile::entry()]
    fn binary_element(
        out: &mut Tensor<f32, { [4] }>,
        x: &Tensor<f32, { [-1] }>,
        h: &Tensor<f16, { [-1] }>,
    ) {
        let a: Tile<f32, { [4] }> = x.partition(shape![4]).load([0]);
        let b: Tile<f16, { [4] }> = h.partition(shape![4]).load([0]);
        out.store(a * b);
    }

    #[cutile::entry()]
    unsafe fn binary_pointer_mutability(
        out: &mut Tensor<f32, { [4] }>,
        x_ptr: *const f32,
        y_ptr: *mut f32,
    ) {
        let p: PointerTile<*const f32, { [] }> = pointer_to_tile(x_ptr);
        let q: PointerTile<*mut f32, { [] }> = pointer_to_tile(y_ptr);
        let _same = p == q;
        out.store(broadcast_scalar(0f32, out.shape()));
    }
}
"#;

fn compile_error(function_name: &str, strides: &[(&str, &[i32])]) -> String {
    let function_name = function_name.to_string();
    let strides: Vec<(String, Vec<i32>)> = strides
        .iter()
        .map(|(name, s)| (name.to_string(), s.to_vec()))
        .collect();
    common::with_test_stack(move || {
        let strides: Vec<(&str, &[i32])> = strides
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let module = || {
            Module::new(
                "mismatched_module",
                syn::parse_str(MISMATCHED_MODULE).expect("mismatched module parses"),
            )
        };
        match KernelCompiler::new(module, "mismatched_module", &function_name)
            .strides(&strides)
            .target("sm_120")
            .compile()
        {
            Ok(_) => panic!("`{function_name}` must not compile"),
            Err(err) => err.to_string(),
        }
    })
}

fn assert_mentions(message: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            message.contains(needle),
            "expected `{needle}` in: {message}"
        );
    }
}

#[test]
fn binary_op_rejects_different_shape() {
    let message = compile_error("binary_shape", &[("out", &[1]), ("x", &[1])]);
    assert_mentions(&message, &["requires operands of the same type"]);
}

#[test]
fn binary_op_rejects_different_element_type() {
    let message = compile_error("binary_element", &[("out", &[1]), ("x", &[1]), ("h", &[1])]);
    assert_mentions(&message, &["requires operands of the same type"]);
}

#[test]
fn binary_op_rejects_different_pointer_mutability() {
    let message = compile_error("binary_pointer_mutability", &[("out", &[1])]);
    assert_mentions(&message, &["requires operands of the same type"]);
}

#[test]
fn unresolvable_parameter_type_is_an_error() {
    let message = compile_error("unresolvable_parameter", &[("out", &[1]), ("x", &[1])]);
    assert_mentions(&message, &["parameter `t` of `takes_unknown`"]);
}
