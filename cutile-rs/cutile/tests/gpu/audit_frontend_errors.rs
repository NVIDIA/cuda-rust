/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Frontend inputs that used to panic the JIT (2026-08 audit) and now
//! produce spanned errors or compile: too few host-supplied generics, an
//! `if`/`else` producing a tuple, a `PARAM[i]` shape argument to a
//! per-dimension generic callee, and the bare `unchecked_accesses` entry
//! flag, which was silently ignored.

use cutile::prelude::*;

use crate::audit_common::{self, host};
use crate::common;

#[cutile::module]
mod frontend_errors_module {
    use cutile::core::*;

    /// Two generic parameters: a launcher that supplies fewer than two
    /// arguments used to index past the end of the list.
    #[cutile::entry()]
    fn two_generics<T: ElementType, const S: [i32; 1]>(out: &mut Tensor<T, S>) {
        let t: Tile<T, S> = load_tile_mut(out);
        out.store(t);
    }

    /// An `if`/`else` whose branches produce tuples has no single value to
    /// yield across the region boundary; this hit an `expect`.
    #[cutile::entry()]
    fn tuple_from_if(out: &mut Tensor<i32, { [1] }>, flag: i32) {
        let (a, b) = if flag > 0i32 {
            (1i32, 2i32)
        } else {
            (3i32, 4i32)
        };
        let t: Tile<i32, { [] }> = scalar_to_tile(a + b);
        out.store(t.reshape(shape![1]));
    }

    /// A per-dimension callee: its generics are inferred from the argument
    /// tile's shape.
    fn scale_rows<const M: i32, const N: i32>(t: Tile<f32, { [M, N] }>) -> Tile<f32, { [M, N] }> {
        t + t
    }

    /// The caller projects its const generic array (`S[0]`, `S[1]`) onto the
    /// callee's per-dimension parameters; inference hit an `unimplemented!`.
    #[cutile::entry()]
    fn projected_shape_callee<const S: [i32; 2]>(out: &mut Tensor<f32, S>) {
        let t: Tile<f32, { [S[0], S[1]] }> = load_tile_mut(out);
        out.store(scale_rows(t));
    }

    /// The bare flag form documented in the DSL reference. `unsafe fn` is
    /// required for it, so a rejected flag would fail the `unsafe` check
    /// only when spelled `= true`; the bare form was silently ignored and the
    /// kernel compiled with its checks.
    #[cutile::entry(unchecked_accesses)]
    unsafe fn bare_unchecked<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        idx: i32,
    ) {
        let p = x.partition(shape![B]);
        z.store(p.load([idx]));
    }

    /// The same access with checks on, for contrast.
    #[cutile::entry()]
    fn checked_access<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        idx: i32,
    ) {
        let p = x.partition(shape![B]);
        z.store(p.load([idx]));
    }
}

use frontend_errors_module::__module_ast_self;

fn compile(
    function_name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> Result<(String, cutile::compile_api::CheckPlacementCounts), String> {
    audit_common::compile(
        __module_ast_self,
        "frontend_errors_module",
        function_name,
        generics,
        strides,
    )
    .map_err(|err| err.to_string())
}

#[test]
fn too_few_host_generics_is_a_spanned_error() {
    common::with_test_stack(|| {
        // Missing both, then missing only the const array parameter.
        for generics in [&[][..], &["f32"][..]] {
            let err = compile("two_generics", generics, &[("out", &[1])])
                .expect_err("too few generic arguments must be rejected");
            assert!(
                err.contains("not enough generic arguments"),
                "unexpected diagnostic for {generics:?}: {err}"
            );
            assert!(
                err.contains("audit_frontend_errors.rs"),
                "expected a source location for {generics:?}: {err}"
            );
        }
        // A malformed const argument is a diagnostic too, not a parse panic.
        let err = compile("two_generics", &["f32", "wide"], &[("out", &[1])])
            .expect_err("a non-numeric const argument must be rejected");
        assert!(
            err.contains("is not an i32"),
            "unexpected diagnostic: {err}"
        );
        // The well-formed instantiation still compiles.
        compile("two_generics", &["f32", "16"], &[("out", &[1])]).expect("two_generics compiles");
    });
}

#[test]
fn tuple_producing_if_is_a_spanned_error() {
    common::with_test_stack(|| {
        let err = compile("tuple_from_if", &[], &[("out", &[1])])
            .expect_err("an `if` producing a tuple must be rejected");
        assert!(
            err.contains("compound value") && err.contains("tuple"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("audit_frontend_errors.rs"),
            "expected a source location: {err}"
        );
    });
}

#[test]
fn projected_shape_arguments_infer_per_dimension_callee_generics() {
    common::with_test_stack(|| {
        let (ir, _) = compile("projected_shape_callee", &["4", "8"], &[("out", &[8, 1])])
            .expect("`S[0]`/`S[1]` shape arguments must infer the callee's generics");
        assert!(
            ir.contains("tile<4x8xf32>"),
            "expected the projected tile shape:\n{ir}"
        );
        // And the inlined callee runs: every element doubles.
        let input = api::copy_host_vec_to_device(&Arc::new(
            (0..32).map(|v| v as f32).collect::<Vec<f32>>(),
        ))
        .reshape(&[4, 8])
        .sync()
        .expect("upload");
        let (out,) = frontend_errors_module::projected_shape_callee(input.partition([4, 8]))
            .generics(vec!["4".to_string(), "8".to_string()])
            .sync()
            .expect("projected_shape_callee");
        let expected: Vec<f32> = (0..32).map(|v| 2.0 * v as f32).collect();
        assert_eq!(host(&out.unpartition()), expected);
    });
}

#[test]
fn bare_unchecked_accesses_flag_means_true() {
    common::with_test_stack(|| {
        let (checked_ir, checked_counts) =
            compile("checked_access", &["16"], &[("z", &[1]), ("x", &[1])]).expect("checked");
        assert_eq!(
            checked_counts.in_place, 1,
            "the runtime index keeps an in-place check: {checked_counts:?}\n{checked_ir}"
        );
        let (unchecked_ir, unchecked_counts) =
            compile("bare_unchecked", &["16"], &[("z", &[1]), ("x", &[1])]).expect("unchecked");
        assert_eq!(
            (
                unchecked_counts.discharged,
                unchecked_counts.hoisted,
                unchecked_counts.in_place
            ),
            (0, 0, 0),
            "the bare flag must disable every check: {unchecked_counts:?}\n{unchecked_ir}"
        );
        assert!(
            !unchecked_ir.contains("assert"),
            "no device check may remain:\n{unchecked_ir}"
        );
    });
}
