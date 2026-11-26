/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[deny(nonstandard_style, unused_variables, unreachable_code)]
mod host_lint_regression {
    #[test]
    fn generated_host_code_allows_device_only_lints() {}

    #[cutile::module]
    mod generated {
        use cutile::core::*;

        #[cutile::entry]
        fn printThenContinue(value: i32, unused: i32) {
            cuda_tile_print!("value={}\n", value);
            let _after_print = value;
        }
    }
}

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

    // A generic wrapper whose body is one trait-method call. Calling it on
    // the result of another call, `twice(twice(x))`, needs the inner call's
    // result type before the outer call's generic `X` can be resolved.
    pub trait Twice: Sized {
        fn twice_impl(self) -> Self;
    }

    impl<const N: i32> Twice for Tile<f32, { [N] }> {
        fn twice_impl(self) -> Tile<f32, { [N] }> {
            self + self
        }
    }

    pub fn twice<X: Twice>(x: X) -> X {
        x.twice_impl()
    }

    pub fn twice_plain<const N: i32>(x: Tile<f32, { [N] }>) -> Tile<f32, { [N] }> {
        x + x
    }

    #[cutile::entry]
    fn nested_generic_call_kernel<const N: i32>(y: &mut Tensor<f32, { [N] }>) {
        let x = y.load();
        y.store(twice(twice(x)));
    }

    #[cutile::entry]
    fn nested_generic_call_mixed_kernel<const N: i32>(y: &mut Tensor<f32, { [N] }>) {
        let x = y.load();
        y.store(twice_plain(twice(twice_plain(x))));
    }

    // A device function's turbofish may name a concrete element type, a
    // tag struct, or `_`, not only the caller's own type parameters.
    pub fn turbofish_ident<const N: i32, T: ElementType>(x: Tile<T, { [N] }>) -> Tile<T, { [N] }> {
        x
    }

    // A string tag: the turbofish and the argument must name it alike.
    pub trait TurbofishTag {
        const NAME: &'static str;
    }
    pub struct TurbofishLower;
    impl TurbofishTag for TurbofishLower {
        const NAME: &'static str = "lower";
    }

    pub fn turbofish_tagged<T: ElementType, J: TurbofishTag>(
        x: Tile<T, { [16] }>,
        _j: J,
    ) -> Tile<T, { [16] }> {
        x
    }

    pub struct TurbofishSel;
    impl TurbofishSel {
        pub fn sel<J: TurbofishTag>(self, x: Tile<f32, { [16] }>, _j: J) -> Tile<f32, { [16] }> {
            x
        }
    }

    #[cutile::entry]
    fn device_fn_turbofish_kernel(y: &mut Tensor<f32, { [16] }>) {
        let t = y.load();
        let a = turbofish_ident::<16, f32>(t);
        let b = turbofish_ident::<16, _>(a);
        let b = turbofish_ident::<_, f32>(b);
        let c = turbofish_tagged::<f32, TurbofishLower>(b, TurbofishLower);
        let c = turbofish_tagged::<_, TurbofishLower>(c, TurbofishLower);
        let c = TurbofishSel.sel::<TurbofishLower>(c, TurbofishLower);
        y.store(c + c);
    }

    #[cutile::entry]
    fn device_fn_turbofish_generic_kernel<const N: i32>(y: &mut Tensor<f32, { [N] }>) {
        let t = y.load();
        let a = turbofish_ident::<N, f32>(t);
        y.store(a + a);
    }

    // In a generic kernel a helper's result is spelled with the kernel's
    // values (`{[16]}`) and a declared or loaded tile with its symbols
    // (`{[N]}`): the two spellings are one type for a binary operator.
    #[cutile::entry]
    fn generic_extent_binop_kernel<const N: i32>(y: &mut Tensor<f32, { [N] }>, s: f32) {
        let a: Tile<f32, { [N] }> = twice_plain(y.load());
        let b: Tile<f32, { [N] }> = broadcast_scalar(s, const_shape![N]);
        let c = twice_plain(y.load());
        let d = y.load();
        y.store(a * b);
        y.store(d - c);
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

/// Regression: a generic wrapper call whose argument is another call,
/// `twice(twice(x))`, failed with "unable to resolve generic type" (or a
/// JIT panic) while the same calls as two `let` statements compiled.
#[test]
fn compile_nested_generic_call() {
    common::with_test_stack(|| {
        for kernel in [
            "nested_generic_call_kernel",
            "nested_generic_call_mixed_kernel",
        ] {
            let module_op_str = compile_ir(kernel, &[16.to_string()], &[("y", &[1])]);
            assert!(
                module_op_str.contains("addf"),
                "Expected the inlined wrappers to emit addf.\n{module_op_str}"
            );
        }
    });
}

/// Regression: a device function's turbofish with a concrete type
/// (`ident::<16, f32>`), a tag struct (`tagged::<Lower>`) or `_` failed
/// with "type `f32` is not a known generic type parameter" (or "unable to
/// extract type identifier from type argument" for `_`). A string tag
/// in the turbofish of a call that also infers it (`tagged::<_, Lower>`,
/// `sel.sel::<Lower>`) must not panic on a differently spelled tag.
#[test]
fn compile_device_fn_turbofish_types() {
    common::with_test_stack(|| {
        for (kernel, generics) in [
            ("device_fn_turbofish_kernel", vec![]),
            ("device_fn_turbofish_generic_kernel", vec![16.to_string()]),
        ] {
            let module_op_str = compile_ir(kernel, &generics, &[("y", &[1])]);
            assert!(
                module_op_str.contains("addf"),
                "Expected `{kernel}` to emit addf.\n{module_op_str}"
            );
        }
    });
}

/// Regression: in a generic kernel, `a * b` with `a` a helper's result
/// and `b` a declared `Tile<f32, {[N]}>` failed with "binary `Mul`
/// requires operands of the same type, but got `Tile < f32 , { [16] } >`
/// and `Tile < f32 , { [N] } >`"; storing `d - c` without annotations
/// failed with "Failed to determine typeck return type".
#[test]
fn compile_generic_extent_binop() {
    common::with_test_stack(|| {
        let module_op_str = compile_ir(
            "generic_extent_binop_kernel",
            &[16.to_string()],
            &[("y", &[1])],
        );
        for op in ["mulf", "subf"] {
            assert!(
                module_op_str.contains(op),
                "Expected the kernel to emit {op}.\n{module_op_str}"
            );
        }
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

// Calls by module path. Calls resolve in the calling module's scope, and
// modules are known by their last path segment, so
// `super::path_call_library::scale(x)` must reach the library's `scale` (or
// fail to compile when that module is not known to the JIT), never a
// same-named function of the calling module.

#[cutile::module]
mod path_call_library {
    use cutile::core::*;

    /// Multiplies by 7.
    pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(7.0f32, const_shape![128]);
        x * c
    }
}

/// A plain Rust module (not a cuTile module) that re-exports the library's
/// `scale`: the JIT has no source for it.
mod path_call_reexports {
    pub use super::path_call_library::scale;
}

/// Calls the library's `scale` by path, without `use`, next to its own
/// `scale`: the library module is not known to the JIT.
#[cutile::module]
mod path_call_unknown_module {
    #![allow(dead_code)]

    use cutile::core::*;

    /// Multiplies by 5.
    fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(5.0f32, const_shape![128]);
        x * c
    }

    #[cutile::entry()]
    fn path_call_kernel(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(super::path_call_library::scale(x));
    }

    /// Through a plain Rust module's `pub use`.
    #[cutile::entry()]
    fn reexport_call_kernel(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(crate::path_call_reexports::scale(x));
    }
}

/// Calls the library's `scale` by path; the library module is `use`d, so the
/// JIT knows it.
#[cutile::module]
mod path_call_known_module {
    use super::path_call_library;
    use cutile::core::*;

    #[cutile::entry()]
    fn path_call_kernel(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        let y = path_call_library::scale(x);
        out.store(super::path_call_library::scale(y));
    }
}

/// A plain Rust module with a nested module named like the library.
mod plain_parent {
    pub mod path_call_library {
        pub use super::super::path_call_library::scale;
    }
}

/// The library is loaded, but the calls name `scale` through plain modules:
/// an explicit import, and a relative path whose last segment names the
/// library. Neither may compile the library's `scale` behind Rust's back.
#[cutile::module]
mod calls_through_plain_modules {
    // Loads the library next to the plain modules.
    #[allow(unused_imports)]
    use super::path_call_library;
    use super::path_call_reexports::scale;
    use cutile::core::*;

    #[cutile::entry()]
    fn imported_from_plain_module(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(scale(x));
    }

    #[cutile::entry()]
    fn relative_path_to_plain_module(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(super::plain_parent::path_call_library::scale(x));
    }
}

#[test]
fn calls_through_plain_modules_are_compile_errors() {
    common::with_test_stack(|| {
        for (entry, expected) in [
            (
                "imported_from_plain_module",
                "imported from `path_call_reexports`",
            ),
            (
                "relative_path_to_plain_module",
                "is not known to the cuTile JIT",
            ),
        ] {
            let err = match compile_scalar_kernel(
                calls_through_plain_modules::__module_ast_self,
                "calls_through_plain_modules",
                entry,
            ) {
                Ok(ir) => panic!("{entry}: compiled the library's `scale`:\n{ir}"),
                Err(err) => err,
            };
            assert!(err.contains(expected), "{entry}: unexpected error: {err}");
        }
    });
}

#[test]
fn path_call_to_unknown_module_is_a_compile_error() {
    common::with_test_stack(|| {
        for (entry, path) in [
            ("path_call_kernel", "super::path_call_library::scale"),
            ("reexport_call_kernel", "crate::path_call_reexports::scale"),
        ] {
            let err = match compile_scalar_kernel(
                path_call_unknown_module::__module_ast_self,
                "path_call_unknown_module",
                entry,
            ) {
                Ok(ir) => {
                    panic!("{entry}: the path call compiled to the kernel's own `scale`:\n{ir}")
                }
                Err(err) => err,
            };
            assert!(
                err.contains(path) && err.contains("is not known to the cuTile JIT"),
                "{entry}: the error must name the call path; got: {err}"
            );
        }
    });
}

#[test]
fn path_call_to_known_module_reaches_that_module() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            path_call_known_module::__module_ast_self,
            "path_call_known_module",
            "path_call_kernel",
        )
        .expect("path calls into a used module compile");
        assert!(
            ir.contains("7.0") && !ir.contains("5.0"),
            "expected the library's constant 7.0:\n{ir}"
        );
    });
}

// Calls resolve in the calling module's scope: two modules can define the
// same function name.

#[cutile::module]
mod same_name_a {
    use cutile::core::*;

    /// Multiplies by 13.
    pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(13.0f32, const_shape![128]);
        x * c
    }
}

#[cutile::module]
mod same_name_b {
    use cutile::core::*;

    /// Multiplies by 17.
    pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(17.0f32, const_shape![128]);
        x * c
    }
}

#[cutile::module]
mod same_name_kernels {
    use super::same_name_a::scale;
    use super::same_name_b;
    use cutile::core::*;

    /// `scale` is imported from `same_name_a`; `same_name_b::scale` is
    /// called through the imported module.
    #[cutile::entry()]
    fn by_import_and_by_module(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(same_name_b::scale(scale(x)));
    }

    /// Both by `super::` path.
    #[cutile::entry()]
    fn by_relative_path(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(super::same_name_b::scale(super::same_name_a::scale(x)));
    }
}

/// `super::` written inside a nested `mod` starts from that `mod` in Rust,
/// and the JIT does not track which `mod` holds a call: the module's
/// `super::` calls resolve only to a name one loaded module defines, never
/// to another `scale`.
#[cutile::module]
mod same_name_nested_super {
    use cutile::core::*;

    /// Multiplies by 5.
    pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(5.0f32, const_shape![128]);
        x * c
    }

    pub mod inner {
        use cutile::core::*;

        pub fn helper(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            super::scale(x)
        }
    }

    #[cutile::entry()]
    fn super_in_a_nested_mod(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(inner::helper(x));
    }
}

/// The same, with a second loaded `scale`: the `super::` call is ambiguous.
#[cutile::module]
mod same_name_nested_super_ambiguous {
    use super::same_name_a;
    use cutile::core::*;

    pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        let c: Tile<f32, { [128] }> = constant(5.0f32, const_shape![128]);
        x * c
    }

    pub mod inner {
        use cutile::core::*;

        pub fn helper(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            super::scale(x)
        }
    }

    #[cutile::entry()]
    fn super_in_a_nested_mod(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(same_name_a::scale(inner::helper(x)));
    }
}

fn compile_scalar_kernel(
    module_ast: fn() -> cutile_compiler::ast::Module,
    module_name: &str,
    function_name: &str,
) -> Result<String, String> {
    common::compile_to_ir(
        module_ast,
        module_name,
        function_name,
        &[],
        &[("out", &[1])],
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .map_err(|err| format!("{err}"))
}

/// Whether the IR holds the constant `value` (`3.0` must not match `13.0`).
fn has_constant(ir: &str, value: &str) -> bool {
    ir.match_indices(value)
        .any(|(at, _)| !ir[..at].ends_with(|c: char| c.is_ascii_digit()))
}

#[test]
fn same_named_functions_resolve_per_module() {
    common::with_test_stack(|| {
        for entry in ["by_import_and_by_module", "by_relative_path"] {
            let ir = compile_scalar_kernel(
                same_name_kernels::__module_ast_self,
                "same_name_kernels",
                entry,
            )
            .unwrap_or_else(|err| panic!("{entry}: {err}"));
            assert!(
                has_constant(&ir, "13.0") && has_constant(&ir, "17.0"),
                "{entry}: expected both modules' constants 13.0 and 17.0:\n{ir}"
            );
        }
    });
}

#[test]
fn super_path_in_a_nested_mod_resolves_only_a_unique_name() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            same_name_nested_super::__module_ast_self,
            "same_name_nested_super",
            "super_in_a_nested_mod",
        )
        .unwrap_or_else(|err| panic!("{err}"));
        assert!(
            has_constant(&ir, "5.0"),
            "expected the module's own scale:\n{ir}"
        );

        let err = match compile_scalar_kernel(
            same_name_nested_super_ambiguous::__module_ast_self,
            "same_name_nested_super_ambiguous",
            "super_in_a_nested_mod",
        ) {
            Ok(ir) => panic!("the ambiguous `super::` call compiled:\n{ir}"),
            Err(err) => err,
        };
        assert!(
            err.contains("super::scale") && err.contains("`super::` paths are ambiguous"),
            "unexpected error: {err}"
        );
    });
}

// Modules are known by their path: cuTile modules in different places may
// share a short name, and each `use` reaches the module it names.

mod short_name_lib {
    /// Shares its short name with the kernel module that uses it.
    #[cutile::module]
    pub mod kernels {
        use cutile::core::*;

        /// Multiplies by 7.
        pub fn times_seven(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(7.0f32, const_shape![128]);
            x * c
        }
    }

    #[cutile::module]
    pub mod helpers {
        use cutile::core::*;

        /// Multiplies by 3.
        pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(3.0f32, const_shape![128]);
            x * c
        }

        pub mod nested {
            use cutile::core::*;

            /// Multiplies by 11.
            pub fn times_eleven(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
                let c: Tile<f32, { [128] }> = constant(11.0f32, const_shape![128]);
                x * c
            }
        }
    }
}

mod short_name_other {
    #[cutile::module]
    pub mod helpers {
        use cutile::core::*;

        /// Multiplies by 19.
        pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(19.0f32, const_shape![128]);
            x * c
        }
    }
}

mod short_name_app {
    #[cutile::module]
    pub mod kernels {
        use crate::short_name_lib::kernels::times_seven;
        use cutile::core::*;

        #[cutile::entry()]
        fn kernel_named_like_its_library(out: &mut Tensor<f32, { [128] }>) {
            let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
            out.store(times_seven(x));
        }
    }
}

#[cutile::module]
mod short_name_users {
    use crate::short_name_lib::helpers::nested::times_eleven;
    use crate::short_name_lib::helpers::scale as scale_by_3;
    use crate::short_name_other::helpers;
    use crate::short_name_other::helpers::scale as scale_by_19;
    use cutile::core::*;

    /// Two `helpers` modules, each reached by its own imports.
    #[cutile::entry()]
    fn libraries_named_alike(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        let y = helpers::scale(scale_by_19(scale_by_3(x)));
        out.store(times_eleven(y));
    }
}

/// How many constants of `value` the IR holds (`3.0` does not count `13.0`).
fn constant_count(ir: &str, value: &str) -> usize {
    ir.match_indices(value)
        .filter(|(at, _)| !ir[..*at].ends_with(|c: char| c.is_ascii_digit()))
        .count()
}

#[test]
fn modules_are_known_by_their_path() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            short_name_app::kernels::__module_ast_self,
            "kernels",
            "kernel_named_like_its_library",
        )
        .unwrap_or_else(|err| panic!("a kernel module named like its library: {err}"));
        assert_eq!(constant_count(&ir, "7.0"), 1, "{ir}");

        let ir = compile_scalar_kernel(
            short_name_users::__module_ast_self,
            "short_name_users",
            "libraries_named_alike",
        )
        .unwrap_or_else(|err| panic!("two libraries named alike: {err}"));
        for (value, count) in [("3.0", 1), ("19.0", 2), ("11.0", 1)] {
            assert_eq!(constant_count(&ir, value), count, "{value}:\n{ir}");
        }
    });
}

#[cutile::module]
mod private_import_source {
    use cutile::core::*;

    pub fn private_exp_source(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        x
    }
}

#[cutile::module]
mod private_import_wrapper {
    use super::private_import_source::private_exp_source as exp;
    use cutile::core::*;

    pub fn public_wrapper(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        exp(x)
    }
}

#[cutile::module]
mod import_regression_kernels {
    #[allow(unused_imports)]
    use super::path_call_library::{self as lib, scale};
    #[allow(unused_imports)]
    use super::private_import_wrapper::*;
    use cutile::core::*;

    #[cutile::entry()]
    fn private_import_is_not_exported(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(exp(x));
    }

    #[cutile::entry()]
    fn grouped_module_alias(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(lib::scale(x));
    }
}

#[test]
fn private_imports_do_not_shadow_core_operations_through_globs() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            import_regression_kernels::__module_ast_self,
            "import_regression_kernels",
            "private_import_is_not_exported",
        )
        .expect("private imports do not escape the wrapper");
        assert!(
            ir.contains("exp "),
            "expected the core exponential operation:\n{ir}"
        );
    });
}

#[test]
fn grouped_self_imports_name_the_module() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            import_regression_kernels::__module_ast_self,
            "import_regression_kernels",
            "grouped_module_alias",
        )
        .expect("the grouped module alias resolves");
        assert!(
            has_constant(&ir, "7.0"),
            "expected the library's scale:\n{ir}"
        );
    });
}

/// A local function named `coord` is that function, not core's `coord`,
/// which a renaming import still reaches.
#[cutile::module]
mod local_coord {
    use cutile::core::coord as at;
    use cutile::core::*;

    fn coord(t: (i32, i32)) -> i32 {
        t.0 + t.1
    }

    #[cutile::entry()]
    fn local_coord_kernel(out: &mut Tensor<i32, { [128] }>) {
        let c = coord((3i32, 4i32));
        let t: Tile<i32, { [128] }> = broadcast_scalar(c, shape![128]);
        out.store(t);
    }

    #[allow(deprecated)]
    #[cutile::entry()]
    fn renamed_core_coord(out: &mut Tensor<f32, { [1, 64] }>) {
        let cols = Dim::new(4);
        let mut p = out
            .partition_mut(shape![1, 16])
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let t: Tile<f32, { [1, 16] }> = constant(0.0, shape![1, 16]);
            p.store(t, at((0i32, j)));
        }
    }
}

#[test]
fn a_local_coord_function_is_not_cores() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            local_coord::__module_ast_self,
            "local_coord",
            "local_coord_kernel",
        )
        .expect("the local `coord` compiles");
        assert!(ir.contains("constant <i32: 7>"), "expected 3 + 4:\n{ir}");
        common::compile_to_ir(
            local_coord::__module_ast_self,
            "local_coord",
            "renamed_core_coord",
            &[],
            &[("out", &[64, 1])],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("core's `coord` under another name compiles");
    });
}

/// Marker namespaces nested in a module, written as `cutile::core`'s `ftz`
/// and `nan` are: each defines `Mode` and `Enabled`.
#[cutile::module]
mod nested_marker_namespaces {
    #![allow(dead_code)]
    use cutile::core::*;

    pub mod mode_a {
        pub trait Mode {}
        pub struct Enabled;
        impl Mode for Enabled {}
    }

    pub mod mode_b {
        pub trait Mode {}
        pub struct Enabled;
        impl Mode for Enabled {}
    }

    #[cutile::entry()]
    fn uses_no_marker(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(x);
    }
}

#[test]
fn same_named_types_in_nested_mods_load() {
    common::with_test_stack(|| {
        compile_scalar_kernel(
            nested_marker_namespaces::__module_ast_self,
            "nested_marker_namespaces",
            "uses_no_marker",
        )
        .expect("a kernel that names neither marker compiles");
    });
}

/// Functions a nested `mod` declares under the names of core's `exp` and
/// `log` and of an imported `scale`. In Rust, a plain `exp` or `scale`
/// outside the `mod` is core's or the imported one; the JIT does not track
/// which `mod` holds a call, so it rejects the plain names instead of
/// guessing. `log` is imported from the nested `mod` by name.
#[cutile::module]
mod nested_item_shadows {
    #![allow(dead_code)]
    use self::approx::log;
    use super::same_name_a::scale;
    use cutile::core::*;

    mod approx {
        use cutile::core::*;

        /// Multiplies by 29.
        pub fn exp(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(29.0f32, const_shape![128]);
            x * c
        }

        /// Multiplies by 5.
        pub fn scale(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(5.0f32, const_shape![128]);
            x * c
        }

        /// Multiplies by 37.
        pub fn log(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
            let c: Tile<f32, { [128] }> = constant(37.0f32, const_shape![128]);
            x * c
        }
    }

    #[cutile::entry()]
    fn calls_core_exp(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(exp(x));
    }

    #[cutile::entry()]
    fn calls_imported_scale(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(scale(x));
    }

    #[cutile::entry()]
    fn calls_nested_by_path(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(approx::exp(approx::scale(log(x))));
    }
}

#[test]
fn nested_items_do_not_replace_imported_or_core_items() {
    common::with_test_stack(|| {
        for (entry, nested, other) in [
            ("calls_core_exp", "`approx::exp`", "`core::exp`"),
            (
                "calls_imported_scale",
                "`approx::scale`",
                "`same_name_a::scale`",
            ),
        ] {
            let err = match compile_scalar_kernel(
                nested_item_shadows::__module_ast_self,
                "nested_item_shadows",
                entry,
            ) {
                Ok(ir) => panic!("{entry}: the plain name compiled:\n{ir}"),
                Err(err) => err,
            };
            assert!(
                err.contains(nested) && err.contains(other),
                "{entry}: the error must name both items: {err}"
            );
        }
        let ir = compile_scalar_kernel(
            nested_item_shadows::__module_ast_self,
            "nested_item_shadows",
            "calls_nested_by_path",
        )
        .expect("paths into the nested `mod` and its import compile");
        assert!(
            ["29.0", "5.0", "37.0"].iter().all(|c| has_constant(&ir, c)),
            "expected the nested functions:\n{ir}"
        );
    });
}

/// A library whose nested `mod`s each define a type alias `Elem`.
#[cutile::module]
mod nested_alias_lib {
    #![allow(dead_code)]
    use cutile::core::*;

    pub mod wide {
        pub type Elem = f64;
    }

    pub mod narrow {
        pub type Elem = i32;
    }

    pub fn same(x: Tile<f32, { [128] }>) -> Tile<f32, { [128] }> {
        x
    }
}

/// Defines its own `Elem` once, and loads the library above.
#[cutile::module]
mod own_alias_kernel {
    use super::nested_alias_lib::same;
    use cutile::core::*;

    type Elem = f32;

    #[cutile::entry()]
    fn uses_own_alias(out: &mut Tensor<f32, { [128] }>) {
        let x: Tile<Elem, { [128] }> = constant(1.0f32, const_shape![128]);
        out.store(same(x));
    }
}

#[test]
fn a_library_duplicate_does_not_hide_a_kernel_alias() {
    common::with_test_stack(|| {
        compile_scalar_kernel(
            own_alias_kernel::__module_ast_self,
            "own_alias_kernel",
            "uses_own_alias",
        )
        .unwrap_or_else(|err| panic!("the kernel's own `Elem` expands: {err}"));
    });
}

// A qualified const path names the module at that path: two loaded modules
// share the short name `tools`, and the kernel has a const of the same name.

mod side_c {
    #[cutile::module]
    pub mod tools {
        pub const N: i32 = 43;
    }
}

mod side_d {
    #[cutile::module]
    pub mod tools {
        pub const N: i32 = 47;
    }
}

#[cutile::module]
mod qualified_const_paths {
    // Loads both `tools` modules.
    #[allow(unused_imports)]
    use crate::side_c::tools as tools_c;
    #[allow(unused_imports)]
    use crate::side_d::tools as tools_d;
    use cutile::core::*;

    #[allow(dead_code)]
    const N: i32 = 53;

    #[cutile::entry()]
    fn qualified_const_kernel(out: &mut Tensor<i32, { [128] }>) {
        let n = crate::side_c::tools::N + 1000i32;
        let t: Tile<i32, { [128] }> = broadcast_scalar(n, shape![128]);
        out.store(t);
    }
}

/// `crate::side_c::tools::N` is side_c's `N` (43), not side_d's (47) nor the
/// kernel module's own (53).
#[test]
fn qualified_const_paths_name_the_module_at_that_path() {
    common::with_test_stack(|| {
        let ir = compile_scalar_kernel(
            qualified_const_paths::__module_ast_self,
            "qualified_const_paths",
            "qualified_const_kernel",
        )
        .unwrap_or_else(|err| panic!("{err}"));
        assert!(
            ir.contains("constant <i32: 1043>"),
            "expected side_c's N + 1000:\n{ir}"
        );
        for other in ["constant <i32: 1047>", "constant <i32: 1053>"] {
            assert!(!ir.contains(other), "unexpected {other}:\n{ir}");
        }
    });
}
