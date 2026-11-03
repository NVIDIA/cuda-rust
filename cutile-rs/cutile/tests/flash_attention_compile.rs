/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage for the example flash-attention kernels.

use cutile::compile_api::KernelCompiler;

mod common;

#[cutile::module]
mod flash_attention_compile_module {
    use cutile::core::*;

    #[cutile::entry(
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 1, max_divisibility = 16,),
        )
    )]
    fn fmha<
        const BM: i32,
        const BN: i32,
        const D: i32,
        const H: i32,
        const CAUSAL: i32,
        const EVEN_K: i32,
    >(
        out: &mut Tensor<f32, { [1, BM, D] }>,
        q: &Tensor<f32, { [-1, -1, -1, -1] }>,
        k: &Tensor<f32, { [-1, -1, -1, -1] }>,
        v: &Tensor<f32, { [-1, -1, -1, -1] }>,
        qk_scale: f32,
        input_pos: i32,
        query_group_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let batch_idx = pid.0 / H;
        let q_head_idx = pid.0 % H;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;

        let two: Tile<f32, { [] }> = constant(2.0f32, shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: Tile<f32, { [BM, BN] }> = broadcast_scalar(qk_scale / log2, shape![BM, BN]);

        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);

        let q_part: Partition<f32, { [1, 1, BM, D] }> = q.partition(shape![1, 1, BM, D]);
        let tq: Tile<f32, { [BM, D] }> = q_part
            .load([batch_idx, q_head_idx, q_m_idx, 0i32])
            .reshape(shape![BM, D]);

        let k_seqlen: i32 = get_shape_dim(k.shape(), 2i32);
        let m_end: i32 = input_pos + (q_m_idx + 1i32) * BM;
        let mut mask_start: i32 = k_seqlen / BN;
        let mut tc: i32 = ceil_div(k_seqlen, BN);
        if CAUSAL == 1i32 {
            mask_start = (input_pos + q_m_idx * BM) / BN;
            let k_seqlen_tiles = k_seqlen / BN;
            mask_start = min(mask_start, k_seqlen_tiles);
            tc = ceil_div(min(m_end, k_seqlen), BN);
        }

        let k_part = k.partition(shape![1, 1, BN, D]);
        let v_part = v.partition(shape![1, 1, BN, D]);

        let offs_n_tile: Tile<i32, { [BN] }> = iota(shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> =
            offs_n_tile.reshape(shape![1, BN]).broadcast(shape![BM, BN]);

        let offs_m_iota: Tile<i32, { [BM] }> = iota(shape![BM]);
        let offs_m_iota = offs_m_iota.reshape(shape![BM, 1]);
        let offs_m: Tile<i32, { [BM, 1] }> =
            broadcast_scalar(q_m_idx * BM + input_pos, shape![BM, 1]) + offs_m_iota;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m.broadcast(shape![BM, BN]);
        let k_seqlen_tile: Tile<i32, { [BM, BN] }> = k_seqlen.broadcast(shape![BM, BN]);
        let mask_true: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);

        for j in 0i32..tc {
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, kv_head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = k_tile.transpose();
            let mut qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
            qk = mma(tq, k_tile_trans, qk);

            if (CAUSAL == 1i32 || EVEN_K == 0i32) && j >= mask_start {
                let offs_n: Tile<i32, { [BM, BN] }> =
                    broadcast_scalar(j * BN, shape![BM, BN]) + offs_n_tile;
                let mut mask: Tile<bool, { [BM, BN] }> = constant(true, shape![BM, BN]);
                if EVEN_K == 0i32 {
                    let lt_res: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, k_seqlen_tile);
                    mask = mask & lt_res;
                }
                if CAUSAL == 1i32 {
                    let ge_res: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
                    mask = mask & ge_res;
                }
                qk = qk + select(mask, mask_true, mask_false);
            }

            qk = qk * qk_scale;
            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(shape![BM, D]);

            let v_tile: Tile<f32, { [BN, D] }> = v_part
                .load([batch_idx, kv_head_idx, j, 0i32])
                .reshape(shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        out.store(true_div(acc, l_i.broadcast(shape![BM, D])).reshape(shape![1, BM, D]));
    }
}

use flash_attention_compile_module::__module_ast_self;

#[test]
fn compile_flash_attention_causal_example() {
    common::with_test_stack(|| {
        let artifacts =
            KernelCompiler::new(__module_ast_self, "flash_attention_compile_module", "fmha")
                .generics(vec![
                    "32".to_string(),
                    "32".to_string(),
                    "32".to_string(),
                    "8".to_string(),
                    "1".to_string(),
                    "1".to_string(),
                ])
                .strides(&[
                    ("out", &[1024, 32, 1]),
                    ("q", &[16384, 2048, 32, 1]),
                    ("k", &[16384, 2048, 32, 1]),
                    ("v", &[16384, 2048, 32, 1]),
                ])
                .target("sm_120")
                .compile()
                .expect("Failed to compile");

        let ir = artifacts.ir_text();
        assert!(ir.contains("mma"), "expected MMA op in IR.\nIR:\n{ir}");
        assert!(
            ir.contains("store_view_tko"),
            "expected output store op in IR.\nIR:\n{ir}"
        );

        let bytecode = artifacts
            .bytecode()
            .expect("bytecode serialization should succeed");
        assert!(!bytecode.is_empty(), "expected non-empty bytecode");
        assert_eq!(
            &bytecode[..8],
            &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00],
            "expected TileIR bytecode magic"
        );
    });
}
