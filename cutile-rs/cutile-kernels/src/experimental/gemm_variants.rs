/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! FP16-input, FP32-output GEMM with selectable memory and MMA schedules.
//!
//! Tensors represent `A[M,K]`, `B[K,N]` and `C[M,N]`. Use an output partition `[BM,BN]`
//! and grid (ceil(M/BM), ceil(N/BN), 1), with target-supported MMA tile sizes.
//! L is a load-latency hint. MODE=0 uses TMA loads/stores, MODE=1 disables TMA
//! stores, and MODE=2 disables TMA loads and stores.
//! See the `kernel_variants` host example for launch and numerical validation.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    fn store_non_tma<const BM: i32, const BN: i32>(
        c: &mut Tensor<f32, { [BM, BN] }>,
        value: Tile<f32, { [BM, BN] }>,
    ) {
        let mut cp = c.partition_mut(shape![BM, BN]);
        check_partition_access_mut(&cp, [0i32, 0i32]);
        unsafe {
            store_view_tko_mut(
                &mut cp,
                value,
                [0i32, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
        }
    }

    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const L: i32, const MODE: i32>(
        c: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f16, { [-1, -1] }>,
        b: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ap = a.partition(shape![BM, BK]);
        let bp = b.partition(shape![BK, BN]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for j in 0i32..num_tiles(&ap, 1i32) {
            if MODE == 2i32 {
                check_partition_access(&ap, [pid.0, j]);
                check_partition_access(&bp, [j, pid.1]);
                let av: Tile<f16, { [BM, BK] }> = load_view_tko(
                    &ap,
                    [pid.0, j],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(L),
                    tma::Disabled,
                );
                let bv: Tile<f16, { [BK, BN] }> = load_view_tko(
                    &bp,
                    [j, pid.1],
                    ordering::Weak,
                    scope::TileBlock,
                    Some(L),
                    tma::Disabled,
                );
                acc = mma(av, bv, acc);
            } else {
                let av = ap.load_pipelined::<L>([pid.0, j]);
                let bv = bp.load_pipelined::<L>([j, pid.1]);
                acc = mma(av, bv, acc);
            }
        }
        if MODE == 0i32 {
            c.store(acc);
        } else {
            store_non_tma(c, acc);
        }
    }

    #[cutile::entry()]
    pub fn gemm_transposed<const BM: i32, const BN: i32, const BK: i32, const L: i32>(
        c: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f16, { [-1, -1] }>,
        b: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ap = a.partition(shape![BM, BK]);
        let bp = b.partition(shape![BK, BN]);
        let mut acc: Tile<f32, { [BN, BM] }> = constant(0.0f32, shape![BN, BM]);
        for j in 0i32..num_tiles(&ap, 1i32) {
            let av = ap.load_pipelined::<L>([pid.0, j]);
            let bv = bp.load_pipelined::<L>([j, pid.1]);
            let bt: Tile<f16, { [BN, BK] }> = permute(bv, const_array![1, 0]);
            let at: Tile<f16, { [BK, BM] }> = permute(av, const_array![1, 0]);
            acc = mma(bt, at, acc);
        }
        let result: Tile<f32, { [BM, BN] }> = permute(acc, const_array![1, 0]);
        c.store(result);
    }

    /// Gather GEMM for contiguous row-major inputs with masked tails.
    ///
    /// # Safety
    /// A/B must have strides `[K,1]`/`[N,1]` and matching K. C has shape `[M,N]`
    /// and does not alias inputs. Use the documented grid and partition.
    /// Padded coordinates must fit i32; AEL=BM*BK and BEL=BK*BN.
    /// Pointer offsets are computed in i64 before multiplication.
    #[cutile::entry()]
    pub unsafe fn gemm_gather<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const L: i32,
        const AEL: i32,
        const BEL: i32,
    >(
        c: &mut Tensor<f32, { [BM, BN] }>,
        a: &Tensor<f16, { [-1, -1] }>,
        b: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let ap = a.partition(shape![BM, BK]);
        let atoken: Token = a.token();
        let btoken: Token = b.token();
        let ashape: [i32; 2] = get_tensor_shape(a);
        let bshape: [i32; 2] = get_tensor_shape(b);
        let rows: Tile<i32, { [BM] }> = broadcast_scalar(pid.0 * BM, shape![BM]) + iota(shape![BM]);
        let cols: Tile<i32, { [BN] }> = broadcast_scalar(pid.1 * BN, shape![BN]) + iota(shape![BN]);
        let rows64: Tile<i64, { [BM] }> = exti(rows);
        let cols64: Tile<i64, { [BN] }> = exti(cols);
        let ar: Tile<i64, { [BM, BK] }> = rows64.reshape(shape![BM, 1]).broadcast(shape![BM, BK]);
        let bc: Tile<i64, { [BK, BN] }> = cols64.reshape(shape![1, BN]).broadcast(shape![BK, BN]);
        let as32: Tile<i32, { [] }> = scalar_to_tile(ashape[1]);
        let as64: Tile<i64, { [] }> = exti(as32);
        let am32: Tile<i32, { [] }> = scalar_to_tile(ashape[0]);
        let am64: Tile<i64, { [] }> = exti(am32);
        let bs32: Tile<i32, { [] }> = scalar_to_tile(bshape[1]);
        let bs64: Tile<i64, { [] }> = exti(bs32);
        let bk32: Tile<i32, { [] }> = scalar_to_tile(bshape[0]);
        let bk64: Tile<i64, { [] }> = exti(bk32);
        let astride: Tile<i64, { [BM, BK] }> = as64.reshape(shape![1, 1]).broadcast(shape![BM, BK]);
        let aheight: Tile<i64, { [BM, BK] }> = am64.reshape(shape![1, 1]).broadcast(shape![BM, BK]);
        let bstride: Tile<i64, { [BK, BN] }> = bs64.reshape(shape![1, 1]).broadcast(shape![BK, BN]);
        let bheight: Tile<i64, { [BK, BN] }> = bk64.reshape(shape![1, 1]).broadcast(shape![BK, BN]);
        let ar_unsigned: Tile<u64, { [BM, BK] }> = bitcast(ar);
        let aheight_unsigned: Tile<u64, { [BM, BK] }> = bitcast(aheight);
        let bc_unsigned: Tile<u64, { [BK, BN] }> = bitcast(bc);
        let bstride_unsigned: Tile<u64, { [BK, BN] }> = bitcast(bstride);
        let amask_rows: Tile<bool, { [BM, BK] }> = lt_tile(ar_unsigned, aheight_unsigned);
        let bmask_cols: Tile<bool, { [BK, BN] }> = lt_tile(bc_unsigned, bstride_unsigned);
        let arowoff: Tile<i64, { [BM, BK] }> = ar * astride;
        let abase: PointerTile<*const f16, { [] }> = pointer_to_tile(a.as_ptr());
        let abase: PointerTile<*const f16, { [1, 1] }> = abase.reshape(shape![1, 1]);
        let abase: PointerTile<*const f16, { [BM, BK] }> = abase.broadcast(shape![BM, BK]);
        let bbase: PointerTile<*const f16, { [] }> = pointer_to_tile(b.as_ptr());
        let bbase: PointerTile<*const f16, { [1, 1] }> = bbase.reshape(shape![1, 1]);
        let bbase: PointerTile<*const f16, { [BK, BN] }> = bbase.broadcast(shape![BK, BN]);
        let zero_tile: Tile<f16, { [] }> = constant(f16::ZERO, shape![]);
        let zero: f16 = tile_to_scalar(zero_tile);
        let kkbase: Tile<i32, { [BK] }> = iota(shape![BK]);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for j in 0i32..num_tiles(&ap, 1i32) {
            let kk: Tile<i32, { [BK] }> = broadcast_scalar(j * BK, shape![BK]) + kkbase;
            let kk64: Tile<i64, { [BK] }> = exti(kk);
            let ac: Tile<i64, { [BM, BK] }> = kk64.reshape(shape![1, BK]).broadcast(shape![BM, BK]);
            let br: Tile<i64, { [BK, BN] }> = kk64.reshape(shape![BK, 1]).broadcast(shape![BK, BN]);
            let ac_unsigned: Tile<u64, { [BM, BK] }> = bitcast(ac);
            let astride_unsigned: Tile<u64, { [BM, BK] }> = bitcast(astride);
            let br_unsigned: Tile<u64, { [BK, BN] }> = bitcast(br);
            let bheight_unsigned: Tile<u64, { [BK, BN] }> = bitcast(bheight);
            let amask: Tile<bool, { [BM, BK] }> =
                amask_rows & lt_tile(ac_unsigned, astride_unsigned);
            let bmask: Tile<bool, { [BK, BN] }> =
                lt_tile(br_unsigned, bheight_unsigned) & bmask_cols;
            let aoff: Tile<i64, { [BM, BK] }> = arowoff + ac;
            let boff: Tile<i64, { [BK, BN] }> = br * bstride + bc;
            let aptr: PointerTile<*const f16, { [BM, BK] }> = addptr_tile(abase, aoff);
            let aflat: PointerTile<*const f16, { [AEL] }> = aptr.reshape(shape![AEL]);
            let amask: Tile<bool, { [AEL] }> = amask.reshape(shape![AEL]);
            let (avf, _at): (Tile<f16, { [AEL] }>, Token) = unsafe {
                load_ptr_tko(
                    aflat,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(amask),
                    Some(zero),
                    Some(atoken),
                    Latency::<L>,
                )
            };
            let av: Tile<f16, { [BM, BK] }> = avf.reshape(shape![BM, BK]);
            let bptr: PointerTile<*const f16, { [BK, BN] }> = addptr_tile(bbase, boff);
            let bflat: PointerTile<*const f16, { [BEL] }> = bptr.reshape(shape![BEL]);
            let bmask: Tile<bool, { [BEL] }> = bmask.reshape(shape![BEL]);
            let (bvf, _bt): (Tile<f16, { [BEL] }>, Token) = unsafe {
                load_ptr_tko(
                    bflat,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(bmask),
                    Some(zero),
                    Some(btoken),
                    Latency::<L>,
                )
            };
            let bv: Tile<f16, { [BK, BN] }> = bvf.reshape(shape![BK, BN]);
            acc = mma(av, bv, acc);
        }
        c.store(acc);
    }
}
