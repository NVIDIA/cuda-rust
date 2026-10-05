/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `x.load_like(z)` is the method-form spelling of `load_tile_like(x, z)`.
//! Pins that both spellings lower to identical Tile IR (same ops, same
//! bounds-check obligations) and that the method form compiles and runs
//! in a `#[cutile::entry]` kernel.

use cutile::prelude::*;
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod load_like_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_free_fn<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like(x, z);
        let ty = load_tile_like(y, z);
        z.store(tx + ty);
    }

    #[cutile::entry()]
    fn add_method<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = x.load_like(z);
        let ty = y.load_like(z);
        z.store(tx + ty);
    }
}

use load_like_module::__module_ast_self as load_like_module_ast;
use load_like_module::add_method;

fn compile(name: &str) -> String {
    common::compile_to_ir(
        load_like_module_ast,
        "load_like_module",
        name,
        &["4".to_string()],
        &[("z", &[1]), ("x", &[1]), ("y", &[1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed to compile.")
}

/// Normalize a module dump to its live-op sequence: SSA numbers stripped,
/// dead `constant` lines dropped (the method form's extra inline hop
/// materializes shape-metadata constants the free-fn form doesn't; both
/// are DCE'd by tileiras). What remains — every view construction, load,
/// store, assume, and assert, in order — is exactly the placement-relevant
/// content.
fn live_ops(module: &str) -> Vec<String> {
    module
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.contains("= constant "))
        .map(|line| {
            let mut out = String::new();
            let mut chars = line.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '%' {
                    while chars.peek().is_some_and(|d| d.is_ascii_digit()) {
                        chars.next();
                    }
                    out.push('%');
                } else {
                    out.push(c);
                }
            }
            out
        })
        .collect()
}

#[test]
fn both_spellings_lower_to_the_same_live_ops() {
    common::with_test_stack(|| {
        let free_fn = compile("add_free_fn").replace("add_free_fn", "K");
        let method = compile("add_method").replace("add_method", "K");
        assert_eq!(
            live_ops(&free_fn),
            live_ops(&method),
            "load_like must lower to the same live ops as load_tile_like"
        );
        // Same check-placement outcomes, explicitly: neither spelling may
        // change the number of device asserts.
        assert_eq!(
            free_fn.matches("assert").count(),
            method.matches("assert").count()
        );
    });
}

#[test]
fn method_form_runs_end_to_end() {
    common::with_test_stack(|| {
        let len = 32usize;
        let z_host = add_method(
            api::zeros(&[len]).partition([4]),
            api::arange::<f32>(len),
            api::ones(&[len]),
        )
        .grid(((len / 4) as u32, 1, 1))
        .first()
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add_method kernel");
        for (i, v) in z_host.iter().enumerate() {
            assert_eq!(*v, i as f32 + 1.0, "index {i}");
        }
    });
}
