/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for generic raw view ops with a colliding `const D`.

use cutile::compile_api::KernelCompiler;

const TEST_STACK_SIZE: usize = 32_000_000;
const TARGET: &str = "sm_120";

fn g(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).to_string()).collect()
}

/// Generic-element load through `load_view_tko` with a colliding `const D`.
#[cutile::module]
pub mod load_collide_module {
    use cutile::core::*;

    #[cutile::entry(print_ir = false)]
    unsafe fn load_collide<E: ElementType, const D: i32, const HALF_D: i32, const LATENCY: i32>(
        x: &Tensor<E, { [-1, -1, D] }>,
        out: &mut Tensor<E, { [1, 1, HALF_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let x_part: Partition<E, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let t: Tile<E, { [1, 1, HALF_D] }> = load_view_tko(
            &x_part,
            [seq_idx, head_idx, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
        out.store(t);
    }
}

/// Generic-element store through `store_view_tko_mut` with a colliding `const D`.
#[cutile::module]
pub mod store_collide_module {
    use cutile::core::*;

    #[cutile::entry(print_ir = false)]
    unsafe fn store_collide<E: ElementType, const D: i32, const HALF_D: i32, const LATENCY: i32>(
        x: &Tensor<E, { [-1, -1, D] }>,
        out: &mut Tensor<E, { [1, 1, HALF_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let x_part: Partition<E, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let t: Tile<E, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let mut out_part: PartitionMut<E, { [1, 1, HALF_D] }> =
            out.partition_mut(const_shape![1, 1, HALF_D]);
        let _tok: Token = store_view_tko_mut(
            &mut out_part,
            t,
            [0i32, 0i32, 0i32],
            ordering::Weak,
            scope::TileBlock,
            Some(LATENCY),
            tma::Enabled,
        );
    }
}

type ModuleAst = fn() -> cutile_compiler::ast::Module;

fn compile(module_ast: ModuleAst, module_name: &str, function_name: &str, generics: Vec<String>) {
    let module_name = module_name.to_string();
    let function_name = function_name.to_string();
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(move || {
            let artifacts = KernelCompiler::new(module_ast, &module_name, &function_name)
                .target(TARGET)
                .generics(generics)
                .strides(&[("x", &[-1i32, -1, 1][..]), ("out", &[1i32, 1, 1][..])])
                .grid((1, 1, 1))
                .compile()
                .unwrap_or_else(|err| panic!("{function_name} failed to compile: {err}"));
            assert!(
                !artifacts.ir_text().trim().is_empty(),
                "{function_name} emitted empty IR"
            );
        })
        .expect("spawn compile thread")
        .join()
        .expect("compile thread panicked");
}

#[test]
fn generic_load_with_colliding_const_d_compiles_f16() {
    compile(
        load_collide_module::__module_ast_self,
        "load_collide_module",
        "load_collide",
        g(&["f16", "128", "64", "0"]),
    );
}

#[test]
fn generic_load_with_colliding_const_d_compiles_bf16() {
    compile(
        load_collide_module::__module_ast_self,
        "load_collide_module",
        "load_collide",
        g(&["bf16", "128", "64", "0"]),
    );
}

#[test]
fn generic_load_with_colliding_const_d_compiles_f32() {
    compile(
        load_collide_module::__module_ast_self,
        "load_collide_module",
        "load_collide",
        g(&["f32", "128", "64", "0"]),
    );
}

#[test]
fn generic_store_with_colliding_const_d_compiles_f16() {
    compile(
        store_collide_module::__module_ast_self,
        "store_collide_module",
        "store_collide",
        g(&["f16", "128", "64", "0"]),
    );
}

#[test]
fn generic_store_with_colliding_const_d_compiles_f32() {
    compile(
        store_collide_module::__module_ast_self,
        "store_collide_module",
        "store_collide",
        g(&["f32", "128", "64", "0"]),
    );
}

/// Concrete-element regression anchor.
#[test]
fn concrete_qk_rope_dynpos_f16_still_compiles() {
    let function_name = "qk_rope_dynpos_f16".to_string();
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(move || {
            let artifacts = KernelCompiler::new(
                cutile_kernels::positional::qk_rope_dynpos_f16_module::__module_ast_self,
                "qk_rope_dynpos_f16_module",
                &function_name,
            )
            .target(TARGET)
            .generics(g(&["128", "64", "0"]))
            .strides(&[
                ("q", &[-1i32, -1, 1][..]),
                ("k", &[-1i32, -1, 1][..]),
                ("inv_freq", &[1i32][..]),
                ("position_start", &[1i32][..]),
                ("out", &[1i32, 1, 1][..]),
            ])
            .grid((1, 1, 1))
            .compile()
            .unwrap_or_else(|err| panic!("{function_name} failed to compile: {err}"));
            assert!(!artifacts.ir_text().trim().is_empty());
        })
        .expect("spawn compile thread")
        .join()
        .expect("compile thread panicked");
}
