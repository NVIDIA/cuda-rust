/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Causal FP16 attention with FP32 output and accumulation.
//!
//! Q, K, V and output use [sequence, batch_heads, D] views. Probabilities are
//! converted to FP16 before the second MMA. `scale` is the ordinary softmax
//! scale multiplied by log2(e), since these kernels use exp2.
//! Input/output extents must match; use grid (ceil(S/BM), batch_heads, 1).
//! `single` covers S <= BN; `tail` covers BN < S <= BN+TT (BN divisible by TT);
//! `causal` streams over the visible keys. Dimensions and tile sizes must
//! be positive and padded coordinates must fit i32. D is a supported MMA
//! width. NT=1 disables TMA for the small single-tile path.

#[cutile::module]
pub mod kernels {
    use cutile::core::*;
    fn qload<const BM: i32, const D: i32, const NT: i32, const S: i32>(
        q: &Tensor<f16, { [S, -1, D] }>,
        b: i32,
        mb: i32,
    ) -> Tile<f16, { [BM, D] }> {
        let qp = q.partition(shape![BM, 1, D]);
        check_partition_access(&qp, [mb, b, 0i32]);
        if NT == 1i32 {
            let loaded: Tile<f16, { [BM, 1, D] }> = load_view_tko(
                &qp,
                [mb, b, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
            loaded.reshape(shape![BM, D])
        } else {
            let loaded: Tile<f16, { [BM, 1, D] }> = load_view_tko(
                &qp,
                [mb, b, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            loaded.reshape(shape![BM, D])
        }
    }
    fn kload<const BN: i32, const D: i32, const NT: i32, const L: i32, const S: i32>(
        k: &Tensor<f16, { [S, -1, D] }>,
        b: i32,
        j: i32,
    ) -> Tile<f16, { [D, BN] }> {
        let kp = k.partition_permuted(shape![D, 1, BN], const_array![2, 1, 0]);
        check_partition_access(&kp, [0i32, b, j]);
        if NT == 1i32 {
            let loaded: Tile<f16, { [D, 1, BN] }> = load_view_tko(
                &kp,
                [0i32, b, j],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
            loaded.reshape(shape![D, BN])
        } else if L == 0i32 {
            let loaded: Tile<f16, { [D, 1, BN] }> = load_view_tko(
                &kp,
                [0i32, b, j],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            loaded.reshape(shape![D, BN])
        } else {
            let loaded: Tile<f16, { [D, 1, BN] }> = load_view_tko(
                &kp,
                [0i32, b, j],
                ordering::Weak,
                scope::TileBlock,
                Some(L),
                tma::Enabled,
            );
            loaded.reshape(shape![D, BN])
        }
    }
    fn vload<const BN: i32, const D: i32, const NT: i32, const L: i32, const S: i32>(
        v: &Tensor<f16, { [S, -1, D] }>,
        b: i32,
        j: i32,
    ) -> Tile<f16, { [BN, D] }> {
        let vp = v.partition(shape![BN, 1, D]);
        check_partition_access(&vp, [j, b, 0i32]);
        if NT == 1i32 {
            let loaded: Tile<f16, { [BN, 1, D] }> = load_view_tko(
                &vp,
                [j, b, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
            loaded.reshape(shape![BN, D])
        } else if L == 0i32 {
            let loaded: Tile<f16, { [BN, 1, D] }> = load_view_tko(
                &vp,
                [j, b, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            loaded.reshape(shape![BN, D])
        } else {
            let loaded: Tile<f16, { [BN, 1, D] }> = load_view_tko(
                &vp,
                [j, b, 0i32],
                ordering::Weak,
                scope::TileBlock,
                Some(L),
                tma::Enabled,
            );
            loaded.reshape(shape![BN, D])
        }
    }
    fn store_output<const BM: i32, const D: i32>(
        o: &mut Tensor<f32, { [BM, 1, D] }>,
        value: Tile<f32, { [BM, D] }>,
    ) {
        let mut op = o.partition_mut(shape![BM, 1, D]);
        check_partition_access_mut(&op, [0i32, 0i32, 0i32]);
        let stored: Tile<f32, { [BM, 1, D] }> = value.reshape(shape![BM, 1, D]);
        unsafe {
            store_view_tko_mut(
                &mut op,
                stored,
                [0i32, 0i32, 0i32],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
        }
    }
    fn scores<const BM: i32, const BN: i32, const D: i32>(
        q: Tile<f16, { [BM, D] }>,
        k: Tile<f16, { [D, BN] }>,
        mb: i32,
        nb: i32,
    ) -> Tile<f32, { [BM, BN] }> {
        let row_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let row_1d: Tile<i32, { [BM] }> = broadcast_scalar(mb * BM, shape![BM]) + row_iota;
        let rows: Tile<i32, { [BM, BN] }> = row_1d.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let col_iota: Tile<i32, { [BN] }> = iota(shape![BN]);
        let col_1d: Tile<i32, { [BN] }> = broadcast_scalar(nb * BN, shape![BN]) + col_iota;
        let cols: Tile<i32, { [BM, BN] }> = col_1d.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let z: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let neg: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
        select(ge_tile(rows, cols), mma(q, k, z), neg)
    }

    #[cutile::entry()]
    pub fn single<const BM: i32, const BN: i32, const D: i32, const NT: i32, const S: i32>(
        o: &mut Tensor<f32, { [BM, 1, D] }>,
        q: &Tensor<f16, { [S, -1, D] }>,
        k: &Tensor<f16, { [S, -1, D] }>,
        v: &Tensor<f16, { [S, -1, D] }>,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mb = pid.0;
        let b = pid.1;
        let qv = qload::<BM, D, NT, S>(q, b, mb);
        let kv = kload::<BN, D, NT, 0, S>(k, b, 0i32);
        let vv = vload::<BN, D, NT, 0, S>(v, b, 0i32);
        let sv = scores(qv, kv, mb, 0i32);
        let sm: Tile<f32, { [BM] }> = reduce_max(sv, 1i32);
        let m = sm.reshape(shape![BM, 1]) * broadcast_scalar(scale, shape![BM, 1]);
        let p = exp2(
            sv * broadcast_scalar(scale, shape![BM, BN]) - m.broadcast(shape![BM, BN]),
            ftz::Enabled,
        );
        let ps: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
        let l = ps.reshape(shape![BM, 1]);
        let pf: Tile<f16, { [BM, BN] }> = convert_tile(p);
        let zero: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        let acc = mma(pf, vv, zero);
        store_output(o, true_div(acc, l.broadcast(shape![BM, D])));
    }

    #[cutile::entry()]
    pub fn tail<const BM: i32, const BN: i32, const D: i32, const TT: i32, const S: i32>(
        o: &mut Tensor<f32, { [BM, 1, D] }>,
        q: &Tensor<f16, { [S, -1, D] }>,
        k: &Tensor<f16, { [S, -1, D] }>,
        v: &Tensor<f16, { [S, -1, D] }>,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mb = pid.0;
        let b = pid.1;
        let qv = qload::<BM, D, 1, S>(q, b, mb);
        let kv = kload::<BN, D, 0, 0, S>(k, b, 0i32);
        let vv = vload::<BN, D, 0, 0, S>(v, b, 0i32);
        let sv = scores(qv, kv, mb, 0i32);
        let sm: Tile<f32, { [BM] }> = reduce_max(sv, 1i32);
        let mut m = sm.reshape(shape![BM, 1]) * broadcast_scalar(scale, shape![BM, 1]);
        let mut l: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        if (mb + 1i32) * BM > BN {
            let k2 = kload::<TT, D, 0, 0, S>(k, b, BN / TT);
            let v2 = vload::<TT, D, 0, 0, S>(v, b, BN / TT);
            let s2 = scores(qv, k2, mb, BN / TT);
            let sm2: Tile<f32, { [BM] }> = reduce_max(s2, 1i32);
            m = max_tile(
                m,
                sm2.reshape(shape![BM, 1]) * broadcast_scalar(scale, shape![BM, 1]),
            );
            let p2 = exp2(
                s2 * broadcast_scalar(scale, shape![BM, TT]) - m.broadcast(shape![BM, TT]),
                ftz::Enabled,
            );
            let ps2: Tile<f32, { [BM] }> = reduce_sum(p2, 1i32);
            l = ps2.reshape(shape![BM, 1]);
            let pf2: Tile<f16, { [BM, TT] }> = convert_tile(p2);
            let zero: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
            acc = mma(pf2, v2, zero);
        }
        let p = exp2(
            sv * broadcast_scalar(scale, shape![BM, BN]) - m.broadcast(shape![BM, BN]),
            ftz::Enabled,
        );
        let ps: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
        l = l + ps.reshape(shape![BM, 1]);
        let pf: Tile<f16, { [BM, BN] }> = convert_tile(p);
        acc = mma(pf, vv, acc);
        store_output(o, true_div(acc, l.broadcast(shape![BM, D])));
    }

    #[cutile::entry()]
    pub fn causal<const BM: i32, const BN: i32, const D: i32, const S: i32>(
        o: &mut Tensor<f32, { [BM, 1, D] }>,
        q: &Tensor<f16, { [S, -1, D] }>,
        k: &Tensor<f16, { [S, -1, D] }>,
        v: &Tensor<f16, { [S, -1, D] }>,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mb = pid.0;
        let b = pid.1;
        let qv = qload::<BM, D, 0, S>(q, b, mb);
        let row_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let row_1d: Tile<i32, { [BM] }> = broadcast_scalar(mb * BM, shape![BM]) + row_iota;
        let rows: Tile<i32, { [BM, BN] }> = row_1d.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let col_iota: Tile<i32, { [BN] }> = iota(shape![BN]);
        let cols: Tile<i32, { [BM, BN] }> =
            col_iota.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let mut m: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        for j in 0i32..ceil_div((mb + 1i32) * BM, BN) {
            let kv = kload::<BN, D, 0, 4, S>(k, b, j);
            let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            let mut sv = mma(qv, kv, zero);
            if (j + 1i32) * BN > mb * BM {
                let neg: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
                sv = select(
                    ge_tile(rows, broadcast_scalar(j * BN, shape![BM, BN]) + cols),
                    sv,
                    neg,
                );
            }
            let sm: Tile<f32, { [BM] }> = reduce_max(sv, 1i32);
            let mn = max_tile(
                m,
                sm.reshape(shape![BM, 1]) * broadcast_scalar(scale, shape![BM, 1]),
            );
            let p = exp2(
                sv * broadcast_scalar(scale, shape![BM, BN]) - mn.broadcast(shape![BM, BN]),
                ftz::Enabled,
            );
            let alpha = exp2(m - mn, ftz::Enabled);
            let ps: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            l = l * alpha + ps.reshape(shape![BM, 1]);
            acc = acc * alpha.broadcast(shape![BM, D]);
            let vv = vload::<BN, D, 0, 4, S>(v, b, j);
            let pf: Tile<f16, { [BM, BN] }> = convert_tile(p);
            acc = mma(pf, vv, acc);
            m = mn;
        }
        store_output(o, true_div(acc, l.broadcast(shape![BM, D])));
    }
}
