/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `return` below the function body and `break` inside `for` have no Tile IR
//! lowering: the former silently fell through (the enclosing block's
//! terminator was emitted and `if idx >= n { return; } p.load([idx])`
//! performed the load), the latter produced a `cuda_tile.break` nested in
//! `cuda_tile.for`, which the assembler rejects. Both are now spanned
//! compile errors (2026-08 audit).

use cutile::prelude::*;

use crate::audit_common::{self, host};
use crate::common;

#[cutile::module]
mod control_flow_module {
    use cutile::core::*;

    /// A `return` under `if` used to be dropped, and the load executed.
    #[cutile::entry()]
    fn early_return_in_if(z: &mut Tensor<f32, { [16] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        let p = x.partition(shape![16]);
        let idx = get_tile_block_id().0;
        if idx >= n {
            return;
        }
        z.store(p.load([idx]));
    }

    /// `cuda_tile.break` is not valid inside `cuda_tile.for`.
    #[cutile::entry()]
    fn break_in_for(z: &mut Tensor<f32, { [16] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        let p = x.partition(shape![16]);
        let mut acc: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for i in 0i32..4i32 {
            if i >= n {
                break;
            }
            acc = acc + p.load([i]);
        }
        z.store(acc);
    }

    /// A top-level `return` stays supported.
    #[cutile::entry()]
    fn top_level_return(z: &mut Tensor<f32, { [16] }>) {
        let t: Tile<f32, { [16] }> = constant(1.0, shape![16]);
        z.store(t);
        return;
    }

    /// `break` inside `while` lowers to `cuda_tile.loop`, which supports it.
    #[cutile::entry()]
    fn break_in_while(z: &mut Tensor<f32, { [16] }>) {
        let mut acc: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        let mut i: i32 = 0i32;
        while i < 10i32 {
            acc = acc + constant(1.0, shape![16]);
            i = i + 1i32;
            if i >= 3i32 {
                break;
            }
        }
        z.store(acc);
    }
}

use control_flow_module::__module_ast_self;

fn compile(function_name: &str, strides: &[(&str, &[i32])]) -> Result<String, String> {
    audit_common::compile(
        __module_ast_self,
        "control_flow_module",
        function_name,
        &[],
        strides,
    )
    .map(|(ir, _)| ir)
    .map_err(|err| err.to_string())
}

#[test]
fn return_below_the_function_body_is_rejected() {
    common::with_test_stack(|| {
        let err = compile("early_return_in_if", &[("z", &[1]), ("x", &[1])])
            .expect_err("a `return` under `if` must be rejected");
        assert!(
            err.contains("`return` is only supported at the top level"),
            "unexpected diagnostic: {err}"
        );
        // Spanned: the diagnostic points into this file.
        assert!(
            err.contains("audit_control_flow.rs"),
            "expected a source location: {err}"
        );
        compile("top_level_return", &[("z", &[1])]).expect("a top-level `return` compiles");
    });
}

#[test]
fn break_inside_for_is_rejected_but_break_inside_while_runs() {
    common::with_test_stack(|| {
        let err = compile("break_in_for", &[("z", &[1]), ("x", &[1])])
            .expect_err("a `break` inside `for` must be rejected");
        assert!(
            err.contains("`break` inside a `for` loop is not supported"),
            "unexpected diagnostic: {err}"
        );
        assert!(
            err.contains("audit_control_flow.rs"),
            "expected a source location: {err}"
        );
        let (z,) = control_flow_module::break_in_while(api::zeros::<f32>(&[16]).partition([16]))
            .sync()
            .expect("break_in_while");
        assert_eq!(host(&z.unpartition()), vec![3.0f32; 16]);
    });
}
