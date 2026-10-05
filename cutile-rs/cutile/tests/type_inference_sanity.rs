/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};

mod common;

#[cutile::module]
mod type_inference_sanity_module {
    use cutile::core::*;

    const GLOBAL_DIM: i32 = 4;
    const GLOBAL_FLAG: bool = true;
    const GLOBAL_SCALE: f32 = 1.0;

    struct TileCarrier {
        tile: Tile<f32, { [4] }>,
    }

    trait AutorefTrait {
        fn trait_by_ref(&self) -> i64;
        fn trait_by_mut(&mut self) -> i64;
    }

    impl AutorefTrait for Tensor<f32, { [4] }> {
        fn trait_by_ref(&self) -> i64 {
            0i64
        }

        fn trait_by_mut(&mut self) -> i64 {
            0i64
        }
    }

    trait HasI64Const {
        const VALUE: i64;
    }

    struct AssocConstHost;

    impl HasI64Const for AssocConstHost {
        const VALUE: i64 = 7;
    }

    fn expect_i64(value: i64) -> i64 {
        value
    }

    fn expect_u32(value: u32) -> u32 {
        value
    }

    fn expect_f32(value: f32) -> f32 {
        value
    }

    fn expect_tile_f32x4(tile: Tile<f32, { [4] }>) -> Tile<f32, { [4] }> {
        tile
    }

    fn expect_option_tile_f32x4(value: Option<Tile<f32, { [4] }>>) -> Option<Tile<f32, { [4] }>> {
        value
    }

    type TensorAlias = Tensor<f32, { [4] }>;
    type DynamicTensorAlias = Tensor<f32, { [-1] }>;
    type ScalarAlias = i32;
    type PtrAlias = *mut f32;

    fn signature_closure_probe<E: ElementType, F>(tile: Tile<E, { [4] }>, _f: F) -> Tile<E, { [4] }>
    where
        F: Fn(E, E) -> E,
    {
        tile
    }

    #[cutile::entry()]
    fn unsafe_block_tail_store_kernel(
        source: &Tensor<f32, { [-1, -1, 8] }>,
        out: &mut Tensor<f32, { [1, 4, 8] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let source_part = source.partition(shape![1, 1, 8]);
        let mut out_part = out.partition_mut(shape![1, 1, 8]);
        let s_start: i32 = pid.1 * 4i32;
        if s_start < seq_len {
            for s_local in 0i32..4i32 {
                let s_global: i32 = s_start + s_local;
                if s_global < seq_len {
                    let tile = source_part
                        .load([s_global, pid.0, pid.2])
                        .reshape(shape![1, 1, 8]);
                    out_part.store(tile, [0i32, s_local, 0i32]);
                }
            }
        }
    }

    #[cutile::entry()]
    fn if_tail_method_chain_kernel(
        source: &Tensor<f32, { [-1, -1, 8] }>,
        out: &mut Tensor<f32, { [1, 1, 8] }>,
        flag: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let source_part = source.partition(shape![1, 1, 8]);
        let tile = if flag > 0i32 {
            source_part
                .load([pid.0, pid.1, pid.2])
                .reshape(shape![1, 1, 8])
        } else {
            source_part
                .load([pid.0, pid.1, pid.2])
                .reshape(shape![1, 1, 8])
        };
        out.store(tile);
    }

    #[cutile::entry()]
    fn struct_literal_field_kernel(out: &mut Tensor<f32, { [4] }>) {
        let carrier = TileCarrier {
            tile: constant(1.0f32, shape![4]),
        };
        out.store(carrier.tile);
    }

    #[cutile::entry()]
    fn tuple_destructure_method_chain_kernel(
        source: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, { [4] }>,
    ) {
        let part = source.partition(shape![4]);
        let pair = (
            part.load([0i32]).reshape(shape![4]),
            part.load([1i32]).reshape(shape![4]),
        );
        let (first, _second) = pair;
        out.store(first);
    }

    #[cutile::entry()]
    fn index_result_method_arg_kernel(out: &mut Tensor<f32, { [4] }>) {
        // Values are all 0 because the partition below has exactly one tile;
        // what this kernel tests is that an array-index expression's TYPE
        // reaches the method argument, not which element it selects.
        let values = [0i32, 0i32, 0i32];
        let idx = values[1];
        let tile = constant(1.0f32, shape![4]);
        let mut out_part = out.partition_mut(shape![4]);
        out_part.store(tile, [idx]);
    }

    #[cutile::entry()]
    fn step_by_loop_local_kernel(out: &mut Tensor<f32, { [4] }>) {
        let mut out_part = out.partition_mut(shape![4]);
        // One tile, so one iteration: `step_by` lowering is the subject here.
        for i in (0i32..1i32).step_by(1) {
            let tile = constant(1.0f32, shape![4]);
            out_part.store(tile, [i]);
        }
    }

    #[cutile::entry()]
    fn step_by_cast_usize_kernel(out: &mut Tensor<f32, { [4] }>) {
        let mut out_part = out.partition_mut(shape![4]);
        let step = 1i32;
        // One tile, so one iteration; the cast in `step_by` is the subject.
        for i in (0i32..1i32).step_by(step as usize) {
            let tile = constant(1.0f32, shape![4]);
            out_part.store(tile, [i]);
        }
    }

    #[cutile::entry()]
    fn if_else_never_join_kernel(out: &mut Tensor<f32, { [4] }>, flag: i32) {
        let tile = if flag > 0i32 {
            return;
        } else {
            constant(1.0f32, shape![4])
        };
        out.store(tile);
    }

    #[cutile::entry()]
    fn unsuffixed_literal_context_kernel(
        source: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, { [4] }>,
        flag: i32,
    ) {
        let part = source.partition(shape![4]);
        let mut out_part = out.partition_mut(shape![4]);
        let default_int = 7;
        let default_float = 3.0;
        let _default_int_tile: Tile<i32, { [] }> = scalar_to_tile(default_int);
        let _default_float_tile: Tile<f64, { [] }> = scalar_to_tile(default_float);
        // One tile, so one iteration; the if/else join is the subject.
        for i in 0..1 {
            let indices = [0, i, 2];
            let idx = indices[1];
            let tile: Tile<f32, { [4] }> = if 0 < flag {
                constant(1.0, shape![4])
            } else {
                part.load([idx])
            };
            out_part.store(tile, [idx]);
        }
    }

    #[cutile::entry()]
    fn backward_literal_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let int_lit = 0;
        let int_alias = int_lit;
        let _resolved_i64 = expect_i64(int_alias);

        let float_lit = 1.0;
        let float_alias = float_lit;
        let resolved_f32 = expect_f32(float_alias);
        let _resolved_f32_tile: Tile<f32, { [] }> = scalar_to_tile(resolved_f32);

        let tile = constant(1.0, shape![4]);
        let tile = expect_tile_f32x4(tile);
        out.store(tile);
    }

    #[cutile::entry()]
    fn var_to_var_binary_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let lhs = 0;
        let rhs = 1;
        let _sum = lhs + rhs;
        let _resolved_lhs = expect_i64(lhs);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn var_to_var_branch_constraints_kernel(out: &mut Tensor<f32, { [4] }>, flag: bool) {
        let lhs = 0;
        let rhs = 1;
        let _joined = if flag { lhs } else { rhs };
        let _resolved_lhs = expect_i64(lhs);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn if_join_direct_expr_constraints_kernel(out: &mut Tensor<f32, { [4] }>, flag: bool) {
        let lhs = 0;
        let _pair = (if flag { lhs } else { 1i64 }, 2i64);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn if_join_diverging_branch_constraints_kernel(out: &mut Tensor<f32, { [4] }>, flag: bool) {
        let _pair = (
            if flag {
                return;
            } else {
                1i64
            },
            2i64,
        );

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn inline_array_index_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let idx = [0, 1][0];
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn local_array_index_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let values = [0, 1];
        let idx = values[0];
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn repeat_array_index_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let values = [0; 2];
        let idx = values[0];
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn inline_tuple_field_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let idx = (0,).0;
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn local_tuple_field_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let pair = (0,);
        let idx = pair.0;
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[allow(unused_assignments)]
    #[cutile::entry()]
    fn assignment_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let mut idx = 0;
        idx = 1i64;
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn assignment_accumulator_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let mut acc = 0.0;
        let update = expect_f32(1.0);
        acc = acc + update;
        let resolved_acc = expect_f32(acc);
        let _acc_tile: Tile<f32, { [] }> = scalar_to_tile(resolved_acc);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn cast_result_type_kernel(out: &mut Tensor<f32, { [4] }>) {
        let idx = 0i32 as u32;
        let _resolved_idx = expect_u32(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn bool_condition_context_kernel(out: &mut Tensor<f32, { [4] }>) {
        let idx = 0;
        let flag = idx < 1i64;
        if flag && true {
            let _resolved_idx = expect_i64(idx);
        }

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn reduce_closure_body_typeck_kernel(out: &mut Tensor<f32, { [4] }>) {
        let tile = constant(1.0f32, shape![4]);
        let _sum = reduce(tile, 0i32, 0.0f32, |acc, x| {
            let y = acc + x;
            y
        });
        out.store(tile);
    }

    #[cutile::entry()]
    fn signature_driven_closure_typeck_kernel(out: &mut Tensor<f32, { [4] }>) {
        let tile = constant(1.0f32, shape![4]);
        let _same = signature_closure_probe(tile, |acc, x| {
            let y = acc + x;
            y
        });
        out.store(tile);
    }

    #[cutile::entry()]
    fn while_loop_body_method_selection_kernel(
        source: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, { [4] }>,
    ) {
        let part = source.partition(shape![4]);
        let mut out_part = out.partition_mut(shape![4]);
        let mut i = 0;
        while i < 1i32 {
            let tile = part.load([i]);
            out_part.store(tile, [i]);
            i = i + 1i32;
        }
    }

    #[cutile::entry()]
    fn loop_body_method_selection_kernel(
        source: &Tensor<f32, { [-1] }>,
        out: &mut Tensor<f32, { [4] }>,
    ) {
        let part = source.partition(shape![4]);
        let mut out_part = out.partition_mut(shape![4]);
        let mut i = 0;
        loop {
            let tile = part.load([i]);
            out_part.store(tile, [i]);
            i = i + 1i32;
            if i >= 1i32 {
                break;
            }
        }
    }

    #[cutile::entry()]
    fn tuple_pattern_backward_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let (idx, scale) = (0, 1.0);
        let _resolved_idx = expect_i64(idx);
        let resolved_scale = expect_f32(scale);
        let _scale_tile: Tile<f32, { [] }> = scalar_to_tile(resolved_scale);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn annotated_tuple_pattern_context_kernel(out: &mut Tensor<f32, { [4] }>) {
        let (idx, scale): (i64, f32) = (0, 1.0);
        let _resolved_idx = expect_i64(idx);
        let _scale_tile: Tile<f32, { [] }> = scalar_to_tile(scale);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn array_pattern_element_constraints_kernel(out: &mut Tensor<f32, { [4] }>) {
        let [idx, other] = [0, 1];
        let _resolved_idx = expect_i64(idx);
        let _other_passthrough = other;

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn array_pattern_from_path_kernel(out: &mut Tensor<f32, { [4] }>) {
        let values = [0i32, 1i32];
        let [idx, other] = values;
        let _other_tile: Tile<i32, { [] }> = scalar_to_tile(other);

        let tile = constant(1.0f32, shape![4]);
        let mut out_part = out.partition_mut(shape![4]);
        out_part.store(tile, [idx]);
    }

    #[cutile::entry()]
    fn struct_pattern_field_kernel(out: &mut Tensor<f32, { [4] }>) {
        let carrier = TileCarrier {
            tile: constant(1.0f32, shape![4]),
        };
        let TileCarrier { tile } = carrier;
        out.store(tile);
    }

    #[cutile::entry()]
    fn block_local_const_item_kernel(out: &mut Tensor<f32, { [4] }>) {
        const IDX: i64 = 0;
        let _resolved_idx = expect_i64(IDX);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn option_constructor_context_kernel(out: &mut Tensor<f32, { [4] }>) {
        let tile = constant(1.0f32, shape![4]);
        let some: Option<Tile<f32, { [4] }>> = Some(tile);
        let none: Option<Tile<f32, { [4] }>> = None;
        let _some = expect_option_tile_f32x4(some);
        let _none = expect_option_tile_f32x4(none);
        out.store(tile);
    }

    #[cutile::entry()]
    fn autoref_trait_receiver_kernel(out: &mut Tensor<f32, { [4] }>) {
        let _by_ref = expect_i64(out.trait_by_ref());
        let _by_mut = expect_i64(out.trait_by_mut());

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn shared_reference_receiver_kernel(
        input: &Tensor<f32, { [4] }>,
        out: &mut Tensor<f32, { [4] }>,
    ) {
        let _by_ref = expect_i64(input.trait_by_ref());

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn qself_associated_const_typeck_kernel(out: &mut Tensor<f32, { [4] }>) {
        let idx = <AssocConstHost as HasI64Const>::VALUE;
        let _resolved_idx = expect_i64(idx);

        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn generic_associated_const_bound_kernel<T: ElementType>(out: &mut Tensor<T, { [4] }>) {
        let zero = T::ZERO;
        let tile: Tile<T, { [4] }> = constant(zero, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn generic_qself_associated_const_bound_kernel<T: ElementType>(out: &mut Tensor<T, { [4] }>) {
        let tile: Tile<T, { [4] }> = constant(<T as ElementType>::ZERO, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn where_bound_associated_const_kernel<T>(out: &mut Tensor<T, { [4] }>)
    where
        T: ElementType,
    {
        let tile: Tile<T, { [4] }> = constant(T::ZERO, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn type_alias_tensor_entry_kernel(
        input: &DynamicTensorAlias,
        out: &mut TensorAlias,
        scalar: ScalarAlias,
    ) {
        let _scalar: ScalarAlias = scalar;
        let part = input.partition(shape![4]);
        out.store(part.load([0i32]));
    }

    #[cutile::entry()]
    unsafe fn type_alias_pointer_entry_kernel(ptr: PtrAlias, len: ScalarAlias) {
        let _ptr = ptr;
        let _len = len;
    }

    #[cutile::entry()]
    fn global_const_shape_kernel(out: &mut Tensor<f32, { [GLOBAL_DIM] }>) {
        let tile = constant(1.0f32, shape![GLOBAL_DIM]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn global_bool_const_kernel(out: &mut Tensor<f32, { [4] }>) {
        if GLOBAL_FLAG {
            let _flag = GLOBAL_FLAG;
        }
        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn global_scalar_const_kernel(out: &mut Tensor<f32, { [4] }>) {
        let tile = constant(GLOBAL_SCALE, shape![4]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn cga_index_const_shape_kernel<const S: [i32; 3]>(out: &mut Tensor<f32, { [S[0], S[2]] }>) {
        let tile: Tile<f32, { [S[0], S[2]] }> = constant(1.0f32, shape![S[0], S[2]]);
        out.store(tile);
    }

    #[cutile::entry()]
    fn bool_const_generic_kernel<const DO_STORE: bool>(out: &mut Tensor<f32, { [4] }>) {
        if DO_STORE {
            let _flag = DO_STORE;
        }
        let tile = constant(1.0f32, shape![4]);
        out.store(tile);
    }
}

use type_inference_sanity_module::__module_ast_self;

fn compile_kernel(name: &str, shape_constraints: &[(&str, &[i32])]) -> String {
    compile_kernel_with_generic_args(name, &[], shape_constraints)
}

fn compile_kernel_with_generic_args(
    name: &str,
    generic_args: &[&str],
    shape_constraints: &[(&str, &[i32])],
) -> String {
    compile_kernel_result_with_generic_args(name, generic_args, shape_constraints)
        .expect("Failed to compile.")
}

fn compile_kernel_result_with_generic_args(
    name: &str,
    generic_args: &[&str],
    shape_constraints: &[(&str, &[i32])],
) -> Result<String, cutile_compiler::error::JITError> {
    let generic_args = generic_args
        .iter()
        .map(|arg| arg.to_string())
        .collect::<Vec<_>>();
    common::compile_to_ir(
        __module_ast_self,
        "type_inference_sanity_module",
        name,
        &generic_args,
        shape_constraints,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
}

fn typeck_dump(name: &str, shape_constraints: &[(&str, &[i32])]) -> String {
    typeck_dump_with_generic_args(name, &[], shape_constraints)
}

fn typeck_dump_with_generic_args(
    name: &str,
    generic_args: &[&str],
    shape_constraints: &[(&str, &[i32])],
) -> String {
    let modules = CUDATileModules::from_kernel(__module_ast_self())
        .expect("Failed to create CUDATileModules");
    let generic_args = generic_args
        .iter()
        .map(|arg| arg.to_string())
        .collect::<Vec<_>>();
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "type_inference_sanity_module",
        name,
        &generic_args,
        shape_constraints,
        &[],
        &[],
        None,
        "sm_120".to_string(),
        &CompileOptions::default(),
    )
    .expect("Failed to create compiler.");
    compiler
        .debug_typeck_dump()
        .expect("Failed to dump typeck results.")
}

fn assert_compiles_with_store(
    name: &'static str,
    shape_constraints: &'static [(&'static str, &'static [i32])],
) {
    common::with_test_stack(move || {
        let module_op_str = compile_kernel(name, shape_constraints);
        assert!(
            module_op_str.contains("store_view_tko"),
            "Expected `{name}` to emit a tile store.\n{module_op_str}"
        );
    });
}

fn assert_generic_compiles_with_store(
    name: &'static str,
    generic_args: &'static [&'static str],
    shape_constraints: &'static [(&'static str, &'static [i32])],
) {
    common::with_test_stack(move || {
        let module_op_str = compile_kernel_with_generic_args(name, generic_args, shape_constraints);
        assert!(
            module_op_str.contains("store_view_tko"),
            "Expected `{name}` to emit a tile store.\n{module_op_str}"
        );
    });
}

#[test]
fn unsafe_block_tail_types_feed_method_selection() {
    assert_compiles_with_store(
        "unsafe_block_tail_store_kernel",
        &[("source", &[8, 8, 8]), ("out", &[1, 4, 8])],
    );
}

#[test]
fn if_tail_method_chain_types_feed_later_store() {
    assert_compiles_with_store(
        "if_tail_method_chain_kernel",
        &[("source", &[8, 8, 8]), ("out", &[1, 1, 8])],
    );
}

#[test]
fn struct_literal_field_type_feeds_later_store() {
    assert_compiles_with_store("struct_literal_field_kernel", &[("out", &[4])]);
}

#[test]
fn tuple_destructure_types_feed_later_store() {
    assert_compiles_with_store(
        "tuple_destructure_method_chain_kernel",
        &[("source", &[8]), ("out", &[4])],
    );
}

#[test]
fn index_result_type_feeds_method_argument() {
    assert_compiles_with_store("index_result_method_arg_kernel", &[("out", &[4])]);
}

#[test]
fn step_by_loop_local_type_feeds_method_argument() {
    assert_compiles_with_store("step_by_loop_local_kernel", &[("out", &[4])]);
}

#[test]
fn step_by_cast_usize_stays_surface_type() {
    assert_compiles_with_store("step_by_cast_usize_kernel", &[("out", &[4])]);
}

#[test]
fn typeck_dump_records_step_by_cast_usize_surface_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump("step_by_cast_usize_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("usize"),
            "Expected `step as usize` to be recorded as a Rust surface type.\n{dump}"
        );
    });
}

#[test]
fn if_else_uses_else_type_when_then_never() {
    common::with_test_stack(|| {
        let dump = typeck_dump("if_else_never_join_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("Tile < f32 , { [4] } >"),
            "Expected else branch to determine the if expression tile type.\n{dump}"
        );
        assert!(
            dump.contains("method#"),
            "Expected inferred tile to feed the following store method selection.\n{dump}"
        );
    });
}

#[test]
fn unsuffixed_literals_compile_with_contextual_types_and_defaults() {
    assert_compiles_with_store(
        "unsuffixed_literal_context_kernel",
        &[("source", &[8]), ("out", &[4])],
    );
}

#[test]
fn typeck_dump_records_unsuffixed_literal_context_and_defaults() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "unsuffixed_literal_context_kernel",
            &[("source", &[8]), ("out", &[4])],
        );
        assert!(
            dump.contains("[i32 ; 3usize]"),
            "Expected unsuffixed index array to infer `[i32; 3]`.\n{dump}"
        );
        assert!(
            dump.contains("f64"),
            "Expected unconstrained unsuffixed float to use Rust's f64 fallback.\n{dump}"
        );
        assert!(
            dump.contains("Tile < f32 , { [4] } >"),
            "Expected annotated tile context to constrain `constant(1.0, ...)` to f32.\n{dump}"
        );
    });
}

#[test]
fn backward_constraints_feed_local_literal_and_tile_types() {
    assert_compiles_with_store("backward_literal_constraints_kernel", &[("out", &[4])]);
}

#[test]
fn typeck_dump_records_backward_constraints() {
    common::with_test_stack(|| {
        let dump = typeck_dump("backward_literal_constraints_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected later function argument context to constrain aliased integer literal to i64.\n{dump}"
        );
        assert!(
            dump.contains("f32"),
            "Expected later function argument context to constrain aliased float literal to f32.\n{dump}"
        );
        assert!(
            dump.contains("Tile < f32 , { [4] } >"),
            "Expected later tile context to constrain the tile-producing call.\n{dump}"
        );
    });
}

#[test]
fn var_to_var_binary_constraints_share_later_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump("var_to_var_binary_constraints_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected constraining one binary operand to constrain related unknowns.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected related integer inference vars to resolve together, not fallback independently.\n{dump}"
        );
    });
}

#[test]
fn var_to_var_branch_constraints_share_later_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump("var_to_var_branch_constraints_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected constraining one branch variable to constrain the joined branch variables.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected joined integer inference vars to resolve together, not fallback independently.\n{dump}"
        );
    });
}

#[test]
fn if_join_pushes_else_type_back_into_then_branch_var() {
    common::with_test_stack(|| {
        let dump = typeck_dump("if_join_direct_expr_constraints_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected typed else branch to constrain the unknown then-branch variable.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected branch-local integer inference var to resolve through the join, not fallback.\n{dump}"
        );
    });
}

#[test]
fn if_join_ignores_diverging_branch_for_value_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "if_join_diverging_branch_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("i64"),
            "Expected non-diverging branch to determine the if expression type.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected diverging branch not to force integer fallback in the joined value.\n{dump}"
        );
    });
}

#[test]
fn inline_array_index_backward_constraints_feed_elements() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "inline_array_index_backward_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("[i64 ; 2usize]"),
            "Expected later index value context to constrain the inline array element type.\n{dump}"
        );
    });
}

#[test]
fn local_array_index_backward_constraints_feed_origin_elements() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "local_array_index_backward_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("[i64 ; 2usize]"),
            "Expected later index value context to constrain the local array origin element type.\n{dump}"
        );
    });
}

#[test]
fn repeat_array_index_backward_constraints_feed_origin_element() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "repeat_array_index_backward_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("[i64 ; 2usize]"),
            "Expected later index value context to constrain the repeat array element type.\n{dump}"
        );
    });
}

#[test]
fn inline_tuple_field_backward_constraints_feed_elements() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "inline_tuple_field_backward_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("(i64 ,)"),
            "Expected later field value context to constrain the inline tuple field type.\n{dump}"
        );
    });
}

#[test]
fn local_tuple_field_backward_constraints_feed_origin_elements() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "local_tuple_field_backward_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("(i64 ,)"),
            "Expected later field value context to constrain the local tuple origin field type.\n{dump}"
        );
    });
}

#[test]
fn assignment_constraints_update_mutated_local() {
    common::with_test_stack(|| {
        let dump = typeck_dump("assignment_backward_constraints_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected assignment RHS to constrain the mutated local.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected initial integer literal to resolve through assignment, not fallback independently.\n{dump}"
        );
    });
}

#[test]
fn assignment_constraints_compile_with_mutated_local() {
    assert_compiles_with_store("assignment_backward_constraints_kernel", &[("out", &[4])]);
}

#[test]
fn assignment_constraints_update_accumulator_local() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "assignment_accumulator_constraints_kernel",
            &[("out", &[4])],
        );
        assert!(
            dump.contains("f32"),
            "Expected accumulator assignment to constrain initial float literal to f32.\n{dump}"
        );
        assert!(
            !dump.contains("f64"),
            "Expected accumulator initial literal to resolve through assignment, not fallback to f64.\n{dump}"
        );
    });
}

#[test]
fn assignment_accumulator_constraints_compile_with_mutated_local() {
    assert_compiles_with_store(
        "assignment_accumulator_constraints_kernel",
        &[("out", &[4])],
    );
}

#[test]
fn cast_result_type_feeds_later_context() {
    assert_compiles_with_store("cast_result_type_kernel", &[("out", &[4])]);
}

#[test]
fn typeck_dump_records_cast_result_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump("cast_result_type_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("u32"),
            "Expected cast expression to record its explicit result type.\n{dump}"
        );
    });
}

#[test]
fn bool_condition_context_constrains_comparison_operands() {
    assert_compiles_with_store("bool_condition_context_kernel", &[("out", &[4])]);
}

#[test]
fn typeck_dump_records_bool_condition_context() {
    common::with_test_stack(|| {
        let dump = typeck_dump("bool_condition_context_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("bool"),
            "Expected comparison/logical condition to infer bool.\n{dump}"
        );
        assert!(
            dump.contains("i64"),
            "Expected comparison operands to unify with the typed RHS.\n{dump}"
        );
        assert!(
            !dump.contains(": i32"),
            "Expected comparison operand literal not to fallback independently.\n{dump}"
        );
    });
}

#[test]
fn reduce_closure_body_is_typechecked_from_call_context() {
    common::with_test_stack(|| {
        let dump = typeck_dump("reduce_closure_body_typeck_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("Tile < f32 , { [] } >"),
            "Expected reduce closure parameters/body to be checked as scalar tiles.\n{dump}"
        );
    });
}

#[test]
fn closure_params_are_typechecked_from_fn_bound_signature() {
    common::with_test_stack(|| {
        let dump = typeck_dump("signature_driven_closure_typeck_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("Tile < f32 , { [] } >"),
            "Expected generic `Fn(E, E) -> E` bound to type closure params as scalar tiles.\n{dump}"
        );
        assert!(
            dump.contains("method#"),
            "Expected typed closure params to feed method selection in the closure body.\n{dump}"
        );
    });
}

#[test]
fn while_loop_body_types_feed_method_selection() {
    assert_compiles_with_store(
        "while_loop_body_method_selection_kernel",
        &[("source", &[8]), ("out", &[4])],
    );
}

#[test]
fn while_loop_typeck_dump_records_body_method_selection() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "while_loop_body_method_selection_kernel",
            &[("source", &[8]), ("out", &[4])],
        );
        assert!(
            dump.contains("core::load"),
            "Expected while body load method selection to be recorded.\n{dump}"
        );
        assert!(
            dump.contains("core::store"),
            "Expected while body store method selection to be recorded.\n{dump}"
        );
    });
}

#[test]
fn loop_body_types_feed_method_selection() {
    assert_compiles_with_store(
        "loop_body_method_selection_kernel",
        &[("source", &[8]), ("out", &[4])],
    );
}

#[test]
fn loop_typeck_dump_records_body_method_selection() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "loop_body_method_selection_kernel",
            &[("source", &[8]), ("out", &[4])],
        );
        assert!(
            dump.contains("core::load"),
            "Expected loop body load method selection to be recorded.\n{dump}"
        );
        assert!(
            dump.contains("core::store"),
            "Expected loop body store method selection to be recorded.\n{dump}"
        );
    });
}

#[test]
fn tuple_pattern_backward_constraints_feed_bindings() {
    assert_compiles_with_store(
        "tuple_pattern_backward_constraints_kernel",
        &[("out", &[4])],
    );
}

#[test]
fn annotated_tuple_pattern_pushes_context_to_initializer() {
    assert_compiles_with_store("annotated_tuple_pattern_context_kernel", &[("out", &[4])]);
}

#[test]
fn array_pattern_element_constraints_share_later_type() {
    assert_compiles_with_store("array_pattern_element_constraints_kernel", &[("out", &[4])]);
}

#[test]
fn array_pattern_from_path_feeds_method_argument() {
    assert_compiles_with_store("array_pattern_from_path_kernel", &[("out", &[4])]);
}

#[test]
fn struct_pattern_field_type_feeds_later_store() {
    assert_compiles_with_store("struct_pattern_field_kernel", &[("out", &[4])]);
}

#[test]
fn block_local_const_items_are_available_to_later_expressions() {
    assert_compiles_with_store("block_local_const_item_kernel", &[("out", &[4])]);
}

#[test]
fn option_constructors_use_expected_option_context() {
    common::with_test_stack(|| {
        let dump = typeck_dump("option_constructor_context_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("Option < Tile < f32 , { [4] } > >"),
            "Expected Some/None constructors to use the annotated Option context.\n{dump}"
        );
    });
}

#[test]
fn autoref_trait_receiver_methods_are_selected() {
    common::with_test_stack(|| {
        let dump = typeck_dump("autoref_trait_receiver_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("trait_by_ref -> i64"),
            "Expected method lookup to select a trait method through receiver autoref.\n{dump}"
        );
        assert!(
            dump.contains("trait_by_mut -> i64"),
            "Expected method lookup to select a mutable trait method through receiver autoref.\n{dump}"
        );
    });
}

#[test]
fn shared_reference_receiver_selects_shared_method() {
    common::with_test_stack(|| {
        let dump = typeck_dump(
            "shared_reference_receiver_kernel",
            &[("input", &[4]), ("out", &[4])],
        );
        assert!(
            dump.contains("trait_by_ref -> i64"),
            "Expected method lookup to select `&self` through a shared reference receiver.\n{dump}"
        );
        assert!(
            !dump.contains("trait_by_mut -> i64"),
            "Did not expect a mutable receiver method selection for a shared reference receiver.\n{dump}"
        );
    });
}

#[test]
fn qself_associated_const_infers_declared_type() {
    common::with_test_stack(|| {
        let dump = typeck_dump("qself_associated_const_typeck_kernel", &[("out", &[4])]);
        assert!(
            dump.contains("i64"),
            "Expected `<AssocConstHost as HasI64Const>::VALUE` to infer i64.\n{dump}"
        );
    });
}

#[test]
fn type_parameter_associated_const_uses_trait_bound() {
    common::with_test_stack(|| {
        let dump = typeck_dump_with_generic_args(
            "generic_associated_const_bound_kernel",
            &["f32"],
            &[("out", &[4])],
        );
        assert!(
            dump.contains("Tile < T , { [4] } >") || dump.contains("Tile < f32 , { [4] } >"),
            "Expected `T::ZERO` to feed the annotated `Tile<T, ...>` context.\n{dump}"
        );
    });
}

#[test]
fn qself_type_parameter_associated_const_uses_trait_bound() {
    common::with_test_stack(|| {
        let dump = typeck_dump_with_generic_args(
            "generic_qself_associated_const_bound_kernel",
            &["f32"],
            &[("out", &[4])],
        );
        assert!(
            dump.contains("Tile < T , { [4] } >") || dump.contains("Tile < f32 , { [4] } >"),
            "Expected `<T as ElementType>::ZERO` to feed the annotated `Tile<T, ...>` context.\n{dump}"
        );
    });
}

#[test]
fn qself_type_parameter_associated_const_compiles() {
    assert_generic_compiles_with_store(
        "generic_qself_associated_const_bound_kernel",
        &["f32"],
        &[("out", &[4])],
    );
}

#[test]
fn where_bound_associated_const_uses_trait_bound() {
    assert_generic_compiles_with_store(
        "where_bound_associated_const_kernel",
        &["f32"],
        &[("out", &[4])],
    );
}

#[test]
fn same_module_type_alias_tensor_entry_compiles() {
    assert_compiles_with_store(
        "type_alias_tensor_entry_kernel",
        &[("input", &[1]), ("out", &[1])],
    );
}

#[test]
fn same_module_type_alias_pointer_entry_compiles() {
    common::with_test_stack(|| {
        let module_op_str = compile_kernel("type_alias_pointer_entry_kernel", &[]);
        assert!(
            module_op_str.contains("entry @type_alias_pointer_entry_kernel_entry"),
            "Expected pointer-alias entry kernel to compile.\n{module_op_str}"
        );
    });
}

#[test]
fn module_level_i32_const_works_in_tensor_type_and_const_shape() {
    assert_compiles_with_store("global_const_shape_kernel", &[("out", &[4])]);
}

#[test]
fn module_level_bool_const_works_in_expression() {
    assert_compiles_with_store("global_bool_const_kernel", &[("out", &[4])]);
}

#[test]
fn module_level_scalar_const_works_as_expression_input() {
    assert_compiles_with_store("global_scalar_const_kernel", &[("out", &[4])]);
}

#[test]
fn const_generic_array_index_works_in_const_shape_and_types() {
    assert_generic_compiles_with_store(
        "cga_index_const_shape_kernel",
        &["4", "2", "8"],
        &[("out", &[4, 8])],
    );
}

#[test]
fn bool_const_generic_works_in_expression() {
    assert_generic_compiles_with_store("bool_const_generic_kernel", &["true"], &[("out", &[4])]);
}

#[test]
fn typeck_dump_records_bool_const_generic() {
    common::with_test_stack(|| {
        let dump =
            typeck_dump_with_generic_args("bool_const_generic_kernel", &["true"], &[("out", &[4])]);
        assert!(
            dump.contains("bool"),
            "Expected bool const generic to be available in typeck results.\n{dump}"
        );
    });
}

#[test]
fn unsupported_associated_const_value_has_specific_error() {
    common::with_test_stack(|| {
        let err = compile_kernel_result_with_generic_args(
            "qself_associated_const_typeck_kernel",
            &[],
            &[("out", &[4])],
        )
        .expect_err("Expected arbitrary associated const value emission to fail.");
        let message = err.to_string();
        assert!(
            message.contains("associated const values are not supported"),
            "Expected a targeted associated const diagnostic.\n{message}"
        );
    });
}

#[test]
fn typeck_dump_golden_for_struct_literal_field() {
    common::with_test_stack(|| {
        let dump = typeck_dump("struct_literal_field_kernel", &[("out", &[4])]);
        let expected = r#"expr#0: TileCarrier
expr#1: Tile < f32 , { [4] } >
expr#2: f32
expr#3: Shape < { [4] } >
expr#4: Token
method#4: core::store -> Token
expr#5: & mut Tensor < f32 , { [4] } >
expr#6: Tile < f32 , { [4] } >
expr#7: TileCarrier"#;
        assert_eq!(dump, expected);
    });
}

#[test]
fn typeck_dump_golden_for_step_by_loop_local() {
    common::with_test_stack(|| {
        let dump = typeck_dump("step_by_loop_local_kernel", &[("out", &[4])]);
        let expected = r#"expr#0: PartitionMut < 'a , f32 , { [4] } >
method#0: core::partition_mut -> _
expr#1: & mut Tensor < f32 , { [4] } >
expr#2: Shape < { [4] } >
expr#5: i32
expr#6: i32
expr#7: i32
expr#8: Tile < f32 , { [4] } >
expr#9: f32
expr#10: Shape < { [4] } >
expr#11: Token
method#11: core::store -> Token
expr#13: Tile < f32 , { [4] } >
expr#14: [i32 ; 1usize]
expr#15: i32"#;
        assert_eq!(dump, expected);
    });
}
