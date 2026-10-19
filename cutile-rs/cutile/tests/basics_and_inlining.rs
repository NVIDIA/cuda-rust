/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod basics_and_inlining_module {

    #![allow(unused_variables)]

    use cutile::core::*;

    // Inlining

    fn other_function<T: ElementType, const D: [i32; 3], const B: [i32; 3]>(
        y: Tile<T, D>,
        shape: Shape<B>,
    ) -> Tile<T, B> {
        reshape(y, shape)
    }

    #[cutile::entry]
    fn inlining_kernel<E: ElementType, const X: i32, const S: [i32; 3], const Y: i32>(
        x: f32,
        y: &mut Tensor<E, S>,
    ) {
        let tile_x: Tile<f32, S> = broadcast_scalar(x, y.shape());
        let empty: &[i32] = &[];
        let shape: Shape<{ [32, 512, 1024] }> = Shape::<{ [32, 512, 1024] }> { dims: empty };
        other_function(tile_x, shape);
    }

    #[cutile::entry]
    fn scalar_bool_condition_kernel<const CAUSAL: i32, const EVEN_K: i32>(
        output: &mut Tensor<i32, { [1] }>,
        mask_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let j = pid.0;
        if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
            let one: Tile<i32, { [1] }> = constant(1i32, const_shape![1]);
            output.store(one);
        }
    }

    // Various Rust->TileIR tests.

    pub struct SomeStruct {
        pub ptr: *mut f32,
        pub scalar: f32,
    }

    fn identity(x: *mut f32) -> *mut f32 {
        return x;
    }

    fn ones_shape<const N: usize>() -> [i32; N] {
        [1; N]
    }

    fn expects_i32(value: i32) -> i32 {
        value
    }

    fn expects_f32(value: f32) -> f32 {
        value
    }

    unsafe fn tensor_from_ptr<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }

    #[cutile::entry]
    unsafe fn inline_tensor_from_ptr_kernel<T: ElementType>(ptr: *mut T, len: i32) {
        let _tensor: Tensor<T, { [-1] }> = tensor_from_ptr(ptr, len);
    }

    #[cutile::entry]
    unsafe fn ptr_partition_load_kernel<T: ElementType>(ptr: *mut T, len: i32) {
        let tensor: Tensor<T, { [-1] }> = tensor_from_ptr(ptr, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![4i32];
        let _tile = tensor.partition(tile_shape).load([pid.0]);
    }

    /// Nibble unpacking shape: shifts and masks on integer tiles and scalars.
    #[cutile::entry]
    unsafe fn shift_ops_kernel(x: &Tensor<i32, { [-1] }>, z: &mut Tensor<i32, { [-1] }>, s: i32) {
        let tile: Tile<i32, { [64] }> = x.partition(const_shape![64]).load([0]);
        let four: Tile<i32, { [64] }> = constant(4i32, const_shape![64]);
        let mask: Tile<i32, { [64] }> = constant(0xFi32, const_shape![64]);
        let lo = tile & mask;
        let hi = (tile >> four) & mask;
        let scaled = lo << four;
        let _scalar_shift: i32 = s >> 1;
        z.partition_mut(const_shape![64]).store(hi + scaled, [0]);
    }

    #[cutile::entry]
    unsafe fn ptr_partition_mut_store_kernel(ptr: *mut f32, len: i32) {
        let mut tensor: Tensor<f32, { [-1] }> = tensor_from_ptr(ptr, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![4i32];
        let tile: Tile<f32, { [4] }> = constant(1.0, tile_shape);
        tensor.partition_mut(tile_shape).store(tile, [pid.0]);
    }

    #[cutile::entry]
    unsafe fn partition_mut_store_rank3_loop_kernel<const S: [i32; 3]>(out: &mut Tensor<f32, S>) {
        // The tile is one slice of the `[1, 4, 8]` slab along axis 1, so the
        // loop below sweeps that axis's four tiles exactly. Tiling by the whole
        // slab (as this kernel once did) makes the same sweep write four times
        // past it — silently, until stores became checked.
        let tile_shape = const_shape![1i32, 1i32, 8i32];
        let mut out_part: PartitionMut<f32, { [1, 1, 8] }> = out.partition_mut(tile_shape);
        for s_local in 0i32..4i32 {
            let tile: Tile<f32, { [1, 1, 8] }> = constant(1.0, tile_shape);
            out_part.store(tile, [0i32, s_local, 0i32]);
        }
    }

    #[cutile::entry]
    unsafe fn partition_mut_store_loaded_rank3_loop_kernel<
        const BLOCK_SIZE: i32,
        const BM_S: i32,
    >(
        source: &Tensor<f32, { [-1, -1, BLOCK_SIZE] }>,
        out: &mut Tensor<f32, { [1, BM_S, BLOCK_SIZE] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let s_tile_idx = pid.1;
        let d_block = pid.2;

        let source_part = source.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut out_part = out.partition_mut(const_shape![1, 1, BLOCK_SIZE]);

        let s_start: i32 = s_tile_idx * BM_S;
        if s_start < seq_len {
            for s_local in 0i32..BM_S {
                let s_global: i32 = s_start + s_local;
                if s_global < seq_len {
                    let tile = source_part
                        .load([s_global, head, d_block])
                        .reshape(const_shape![1, 1, BLOCK_SIZE]);
                    out_part.store(tile, [0i32, s_local, 0i32]);
                }
            }
        }
    }

    #[cutile::entry]
    unsafe fn basics_kernel<const S: [i32; 3]>(
        y: &mut Tensor<f32, { [128, -1] }>,
        #[allow(unused_variables)] w: &Tensor<f32, S>,
        ptr: *mut f32,
        scalar: f32,
        integer: u32,
    ) {
        let some_struct: SomeStruct = SomeStruct { ptr, scalar };
        let the_ptr = some_struct.ptr;
        let mut _result = identity(the_ptr);
        _result = the_ptr;

        let tile_scalar: Tile<f32, { [] }> = scalar_to_tile(scalar);
        let _scalar2: f32 = tile_to_scalar(tile_scalar);

        let tile_integer: Tile<u32, { [] }> = scalar_to_tile(integer);
        let _integer2: u32 = tile_to_scalar(tile_integer);

        let ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr);
        let _ptr2: *mut f32 = tile_to_pointer(ptr_tile);

        let num_pid: (i32, i32, i32) = get_num_tile_blocks();
        let shape: Shape<{ [128, 256] }> = Shape::<{ [128, 256] }> {
            dims: &[num_pid.0, 256i32],
        };

        let shape_dim_1: i32 = 128;
        let shape_dim_2: i32 = 256;
        let stride_dim_1: i32 = num_pid.0;
        let stride: Array<{ [-1, 128] }> = Array::<{ [-1, 128] }> {
            dims: &[stride_dim_1],
        };
        let dynamic_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[shape_dim_1, shape_dim_2],
        };

        unsafe {
            let token: Token = new_token_unordered();
            let _some_tensor: Tensor<f32, { [-1, -1] }> =
                make_tensor_view(ptr_tile, dynamic_shape, stride, token);
            let mut partition: PartitionMut<f32, { [128, 256] }> =
                make_partition_view_mut(y, shape, padding::None, token);
            let idx: [i32; 2] = [0i32, 0i32];
            let some_tile: Tile<f32, { [128, 256] }> = load_view_tko_mut(
                &partition,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            store_view_tko_mut(
                &mut partition,
                some_tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            let _store_token_2: Token = store_view_tko_mut(
                &mut partition,
                some_tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }

        let shape: Shape<{ [1, 1] }> = Shape::<{ [1, 1] }> {
            dims: &[1i32, 1i32],
        };
        let x: Tile<u32, { [1, 1] }> = reshape(tile_integer, shape);
        let x_shape: Shape<{ [128, 64] }> = Shape::<{ [128, 64] }> {
            dims: &[128i32, 64i32],
        };
        let x_shaped: Tile<u32, { [128, 64] }> = broadcast(x, x_shape);
        let trans_perm: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let _x_transpose: Tile<u32, { [64, 128] }> = permute(x_shaped, trans_perm);
        let _this_works: i32 = 1i32 + 2i32;

        let _multi_dim_const: Tile<f32, { [128, 64] }> = constant(0.0, x_shape);
        let _neg_const: Tile<f32, { [128, 64] }> = constant(-1.0, x_shape);

        // Test tuples.
        let tuple: (i32, i32) = (1i32, 2i32);
        let _tuple_0: i32 = tuple.0;

        let _ones: [i32; 2] = ones_shape::<2>();
        let _a_bool: bool = false;

        let xu32: u32 = 1;
        let yu32: u32 = 2;
        let _zu32 = xu32 / yu32;

        let _b: bool = xu32 == yu32;
        let _b: bool = xu32 > yu32;
        let _b: bool = xu32 >= yu32;
        let _b: bool = xu32 < yu32;
        let _b: bool = xu32 <= yu32;

        let xi32: i32 = 1;
        let yi32: i32 = 2;
        let _zi32 = xi32 / yi32;

        let _b: bool = xi32 == yi32;
        let _b: bool = xi32 > yi32;
        let _b: bool = xi32 >= yi32;
        let _b: bool = xi32 < yi32;
        let _b: bool = xi32 <= yi32;

        let an_f32: f32 = 1.0;
        let another_f32: f32 = 2.0;
        let _yet_another_f32: f32 = an_f32 / another_f32;

        let _b: bool = an_f32 == another_f32;
        let _b: bool = an_f32 > another_f32;
        let _b: bool = an_f32 >= another_f32;
        let _b: bool = an_f32 < another_f32;
        let _b: bool = an_f32 <= another_f32;

        let inferred_i32 = 7i32;
        let _inferred_i32_tile: Tile<i32, { [] }> = scalar_to_tile(inferred_i32);
        let inferred_f32 = 7.0f32;
        let _inferred_f32_tile: Tile<f32, { [] }> = scalar_to_tile(inferred_f32);
        let mut assigned_i32: i32 = 0i32;
        let _assigned_i32_initial_tile: Tile<i32, { [] }> = scalar_to_tile(assigned_i32);
        assigned_i32 = 9;
        let _assigned_i32_tile: Tile<i32, { [] }> = scalar_to_tile(assigned_i32);
        let mut assigned_f32: f32 = 0.0f32;
        let _assigned_f32_initial_tile: Tile<f32, { [] }> = scalar_to_tile(assigned_f32);
        assigned_f32 = 9.0;
        let _assigned_f32_tile: Tile<f32, { [] }> = scalar_to_tile(assigned_f32);
        let contextual_i32 = expects_i32(11);
        let _contextual_i32_tile: Tile<i32, { [] }> = scalar_to_tile(contextual_i32);
        let contextual_f32 = expects_f32(11.0);
        let _contextual_f32_tile: Tile<f32, { [] }> = scalar_to_tile(contextual_f32);
        let annotated_call_i32: i32 = expects_i32(13);
        let _annotated_call_i32_tile: Tile<i32, { [] }> = scalar_to_tile(annotated_call_i32);
        let annotated_call_f32: f32 = expects_f32(13.0);
        let _annotated_call_f32_tile: Tile<f32, { [] }> = scalar_to_tile(annotated_call_f32);

        // Convert things.
        let x: f32 = 0.0;
        let x: i32 = convert_scalar::<i32>(x);
        let x: f32 = convert_scalar::<f32>(x);
        let x: f64 = convert_scalar::<f64>(x);
        let _x: f16 = convert_scalar::<f16>(x);

        let shape: Shape<{ [128, 256] }> = Shape::<{ [128, 256] }> {
            dims: &[num_pid.0, 256i32],
        };
        let shape_dim_1: i32 = 128;
        let shape_dim_2: i32 = 256;
        let stride_dim_1: i32 = num_pid.0;
        let stride: Array<{ [-1, 128] }> = Array::<{ [-1, 128] }> {
            dims: &[stride_dim_1],
        };
        let dynamic_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[shape_dim_1, shape_dim_2],
        };

        unsafe {
            // Basic loop pattern with a tile.
            let token: Token = new_token_unordered();
            let _some_tensor: Tensor<f32, { [-1, -1] }> =
                make_tensor_view(ptr_tile, dynamic_shape, stride, token);
            let mut partition: PartitionMut<f32, { [128, 256] }> =
                make_partition_view_mut(y, shape, padding::None, token);
            let idx: [i32; 2] = [0i32, 0i32];
            let mut some_tile: Tile<f32, { [128, 256] }> = load_view_tko_mut(
                &partition,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            store_view_tko_mut(
                &mut partition,
                some_tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            store_view_tko_mut(
                &mut partition,
                some_tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            let loop_end: i32 = 10;
            for _i in 0..loop_end {
                let some_tile_2: Tile<f32, { [128, 256] }> = constant(2.0, shape);
                some_tile = some_tile + some_tile_2;
                store_view_tko_mut(
                    &mut partition,
                    some_tile,
                    idx,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                );
                continue;
            }
            let some_3: Tile<f32, { [128, 256] }> = some_tile + some_tile;
            store_view_tko_mut(
                &mut partition,
                some_3,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
        }

        let _basic_string = "a string.";

        let x: [i32; 2] = [1i32, 2i32];
        let _x_val: i32 = x[0];
        let repeat_x: [i32; 2] = [0; 2];
        let _repeat_x_val: i32 = repeat_x[0];
        let x: &[i32] = &[1i32, 2i32];
        let _x_val: i32 = x[0];

        const ARRAY: [i32; 2] = [1, 2];
        const X0: i32 = ARRAY[0];
        let _x1: i32 = ARRAY[1];

        if shape_dim_1 != shape_dim_2 {
            cuda_tile_assert!(shape_dim_1 != shape_dim_2, "Impossible");
        } else {
            cuda_tile_assert!(shape_dim_1 == shape_dim_2, "Impossible");
        }
    }

    #[cutile::entry]
    unsafe fn ptr_tile_reshape_kernel(ptr: *mut f32) {
        let ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr);
        let offset_ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr).offset(1);
        let reshaped: PointerTile<*mut f32, { [1] }> = ptr_tile.reshape(const_shape![1]);
        let _reshaped_offset: PointerTile<*mut f32, { [1] }> =
            offset_ptr_tile.reshape(const_shape![1]);
        let _broadcast: PointerTile<*mut f32, { [128] }> = reshaped.broadcast(const_shape![128]);
    }

    #[cutile::entry]
    fn negative_constant_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let shape = output.shape();
        let _neg_float: Tile<f32, S> = constant(-1.0, shape);
        let _neg_int: Tile<i32, S> = constant(-42i32, shape);
        let _neg_suffixed: Tile<f32, S> = constant(-2.5f32, shape);
    }

    // Distilled from grout's fmha_prefill_causal_mapped: `mma` unifies its
    // shared shape params against a mix of annotation-typed values (symbolic
    // `D`) and op-computed values (substituted `128`). Inline inference must
    // normalize both forms of the same dimension before unifying.
    #[cutile::entry]
    fn symbolic_dim_mma_kernel<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        mut out: MappedPartitionMut<f16, { [BM, 1, D] }, MAP_SHAPE>,
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
    ) {
        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let k_part = k.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for index in out.iter_indices() {
            let [q_m_idx, q_head_idx, _d0] = index.coords();
            let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);
            let tq_raw: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
            let tq: Tile<f16, { [BM, D] }> = tq_raw.reshape(const_shape![BM, D]);
            for j in 0i32..2i32 {
                let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([q_head_idx, j, 0i32]);
                let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
                let k_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
                let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                qk = mma(tq, k_trans, qk);
                let p: Tile<f16, { [BM, BN] }> = convert_tile(qk);
                let v_tile: Tile<f16, { [BN, D] }> = k_part
                    .load([q_head_idx, j, 0i32])
                    .reshape(const_shape![BN, D]);
                acc = mma(p, v_tile, acc);
            }
            let out_tile: Tile<f16, { [BM, 1, D] }> =
                convert_tile(acc.reshape(const_shape![BM, 1, D]));
            out.store(out_tile, index);
        }
    }
}

use basics_and_inlining_module::__module_ast_self;

fn compile_ir(function_name: &str, generics: &[String], strides: &[(&str, &[i32])]) -> String {
    common::compile_to_ir(
        __module_ast_self,
        "basics_and_inlining_module",
        function_name,
        generics,
        strides,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .expect("Failed.")
}

#[test]
fn compile_inline_tensor_from_ptr_helper() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("inline_tensor_from_ptr_kernel", &["f32".to_string()], &[]);
        assert!(
            module_op_str.contains("make_tensor_view"),
            "Expected inlined pointer helper to emit make_tensor_view.\n{module_op_str}"
        );
        println!("{module_op_str}");
    });
}

#[test]
fn compile_ptr_partition_load_helper() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("ptr_partition_load_kernel", &["f32".to_string()], &[]);
        assert!(
            module_op_str.contains("make_partition_view"),
            "Expected partition helper to emit make_partition_view.\n{module_op_str}"
        );
        assert!(
            module_op_str.contains("load_view_tko"),
            "Expected partition load helper to emit load_view_tko.\n{module_op_str}"
        );
    });
}

#[test]
fn compile_shift_ops_kernel() {
    common::with_test_stack(|| {
        let module_op_str =
            compile_ir("shift_ops_kernel", &[], &[("x", &[1][..]), ("z", &[1][..])]);
        assert!(
            module_op_str.contains("shri"),
            "Expected `>>` to lower to shri.\n{module_op_str}"
        );
        assert!(
            module_op_str.contains("shli"),
            "Expected `<<` to lower to shli.\n{module_op_str}"
        );
    });
}

#[test]
fn compile_ptr_partition_mut_store_helper() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("ptr_partition_mut_store_kernel", &[], &[]);
        assert!(
            module_op_str.contains("make_partition_view"),
            "Expected mutable partition helper to emit make_partition_view.\n{module_op_str}"
        );
        assert!(
            module_op_str.contains("store_view_tko"),
            "Expected mutable partition helper to emit store_view_tko.\n{module_op_str}"
        );
    });
}

#[test]
fn compile_partition_mut_store_rank3_loop() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "partition_mut_store_rank3_loop_kernel",
            &[1.to_string(), 4.to_string(), 8.to_string()],
            &[("out", &[1, 4, 8])],
        );
        assert!(
            module_op_str.contains("store_view_tko"),
            "Expected partition store helper to emit store_view_tko.\n{module_op_str}"
        );
    });
}

#[test]
fn compile_partition_mut_store_loaded_rank3_loop() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "partition_mut_store_loaded_rank3_loop_kernel",
            &[8.to_string(), 4.to_string()],
            &[("source", &[16, 16, 8]), ("out", &[1, 4, 8])],
        );
        assert!(
            module_op_str.contains("store_view_tko"),
            "Expected partition store helper to emit store_view_tko.\n{module_op_str}"
        );
    });
}

#[test]
fn compile_inlining() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "inlining_kernel",
            &[
                "f32".to_string(),
                1.to_string(),
                128.to_string(),
                256.to_string(),
                512.to_string(),
                2.to_string(),
            ],
            &[("y", &[1024, 1, 1])],
        );
        println!("{module_op_str}");
    });
}

#[test]
fn compile_scalar_bool_condition() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "scalar_bool_condition_kernel",
            &[1.to_string(), 0.to_string()],
            &[("output", &[1])],
        );
        println!("{module_op_str}");
    });
}

#[test]
fn compile_basics() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "basics_kernel",
            &[128.to_string(), 256.to_string(), 512.to_string()],
            &[("y", &[1024, 1]), ("w", &[1, 2, 3])],
        );
        println!("{module_op_str}");
    });
}

#[test]
fn compile_negative_constant() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "negative_constant_kernel",
            &[128.to_string()],
            &[("output", &[1024])],
        );
        assert!(module_op_str.contains("-1.0"));
        println!("{module_op_str}");
    });
}

#[test]
fn compile_ptr_tile_reshape() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir("ptr_tile_reshape_kernel", &[], &[]);
        println!("{module_op_str}");
        assert!(
            module_op_str.contains("reshape") && module_op_str.contains("tile<ptr<f32>>"),
            "Expected reshape operation on pointer tile type"
        );
        assert!(
            module_op_str.contains("broadcast"),
            "Expected broadcast operation on pointer tile type"
        );
    });
}

#[test]
fn symbolic_dim_mma_unifies_substituted_and_symbolic_shapes() {
    common::with_test_stack(|| {
        let mlir = compile_ir(
            "symbolic_dim_mma_kernel",
            &[
                "16".to_string(),
                "32".to_string(),
                "128".to_string(),
                "1".to_string(),
                "1".to_string(),
                "1".to_string(),
            ],
            &[
                ("out", &[128, 128, 1]),
                ("q", &[512, 128, 1]),
                ("k", &[512, 128, 1]),
            ],
        );
        assert!(
            mlir.contains("mma"),
            "expected an mma op in the symbolic-dim kernel:\n{mlir}"
        );
    });
}

// The bytecode debug section must chain "inlined at" info: ops compiled
// from other_function's body carry a CallSite linking their own source
// line to the call expression inside inlining_kernel.
#[test]
fn inlined_ops_carry_call_site_debug_info() {
    common::with_test_stack(|| {
        let artifacts = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "basics_and_inlining_module",
            "inlining_kernel",
        )
        .target("sm_120")
        .generics(vec![
            "f32".into(),
            "1".into(),
            "128".into(),
            "256".into(),
            "512".into(),
            "2".into(),
        ])
        .strides(&[("y", &[1024, 1, 1])])
        .compile()
        .expect("compile inlining_kernel");
        let bytecode = artifacts.bytecode().expect("bytecode");
        let dump =
            cutile_ir::bytecode::decoder::decode_bytecode(&bytecode).expect("decode bytecode");

        assert!(
            dump.contains("CallSite(callee=di["),
            "inlined ops must carry a call-site chain:\n{dump}"
        );
        // Both frames name this test file: the callee op (reshape inside
        // other_function) and the caller (the call in inlining_kernel).
        assert!(
            dump.contains("basics_and_inlining.rs"),
            "debug locations must name the source file:\n{dump}"
        );
        assert!(
            dump.contains(r#"name="inlining_kernel", linkage="inlining_kernel_entry""#),
            "subprogram must pair user name with the entry symbol:\n{dump}"
        );
        // The inlined callee gets its OWN subprogram (with a synthetic
        // linkage name — it has no symbol), so inline debug frames name
        // other_function, not the kernel entry.
        assert!(
            dump.contains(r#"name="other_function", linkage="other_function@"#),
            "inlined ops must be scoped to the callee's subprogram:\n{dump}"
        );
    });
}
