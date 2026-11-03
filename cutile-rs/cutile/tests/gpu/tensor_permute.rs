/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: `partition_permuted` — load tiles in a permuted layout, then
//! store into a 3-D output that matches the permuted shape. Verifies the
//! permutation by reconstructing the expected layout from a plain-Rust loop
//! over the source `arange`.

use cutile::api::{arange, zeros, DeviceOpReshape};
use cutile::prelude::*;

use my_module::tensor_permute;

use crate::common;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry(
        unchecked_accesses=false,
        optimization_hints = (sm_120 = (max_divisibility = 16,),)
    )]
    unsafe fn tensor_permute<
        T: ElementType,
        const BBH: i32,
        const BB: i32,
        const BH: i32,
        const BM: i32,
        const BD: i32,
        const DIM_MAP: [i32; 4],
    >(
        src: &Tensor<T, { [-1, -1, -1, -1] }>,
        dst: &mut Tensor<T, { [BBH, BD, BM] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = src.shape()[1];
        let b_idx = pid.0 / (h / BH);
        let h_idx = pid.0 % (h / BH);
        let d_idx = pid.1;
        let m_idx = pid.2;

        let dim_map = const_array!(DIM_MAP);
        let src_part: Partition<T, { [BB, BH, BD, BM] }> =
            src.partition_permuted(shape![BB, BH, BD, BM], dim_map);
        let src_tile: Tile<T, { [BB, BH, BD, BM] }> = src_part.load([b_idx, h_idx, d_idx, m_idx]);
        let src_tile = src_tile.reshape(shape![BBH, BD, BM]);
        dst.store(src_tile);
    }
}

#[test]
fn smoke_tensor_permute() {
    common::with_test_stack(|| {
        // Small shapes — the example used (4, 32, 1024, 64); shrinking keeps
        // the test fast while still exercising the rank-4 → rank-3 permute.
        let (b, h, m, d) = (2usize, 4usize, 16usize, 8usize);
        let partition: [usize; 4] = [1, 2, 4, 4];
        let dim_map: [usize; 4] = [0, 1, 3, 2];

        let bbh = partition[dim_map[0]] * partition[dim_map[1]];
        let partition_shape_rank3 = [bbh, partition[dim_map[2]], partition[dim_map[3]]];

        let src: Arc<Tensor<f32>> = arange(b * h * m * d)
            .reshape(&[b, h, m, d])
            .sync()
            .expect("src reshape")
            .into();
        let dst = zeros(&[b * h, d, m])
            .sync()
            .expect("dst zeros")
            .partition(partition_shape_rank3);

        let mut generics: Vec<String> =
            [[bbh].as_slice(), partition.as_slice(), dim_map.as_slice()]
                .concat()
                .iter()
                .map(|x| x.to_string())
                .collect();
        generics.insert(0, "f32".to_string());

        let (src, dst) = unsafe { tensor_permute(src.clone(), dst) }
            .generics(generics)
            .sync()
            .expect("tensor_permute kernel");

        let out_host = dst.unpartition().to_host_vec().sync().expect("dst to_host");
        let src_host = src.to_host_vec().sync().expect("src to_host");

        // Expected: src_perm[b, h, d, m] = src[b, h, m, d], reshaped to [b*h, d, m].
        // Equivalently, out[b*h + h_idx, d_idx, m_idx] = src[b_idx, h_idx, m_idx, d_idx].
        for b_idx in 0..b {
            for h_idx in 0..h {
                for d_idx in 0..d {
                    for m_idx in 0..m {
                        let src_flat = ((b_idx * h + h_idx) * m + m_idx) * d + d_idx;
                        let dst_flat = ((b_idx * h + h_idx) * d + d_idx) * m + m_idx;
                        assert_eq!(
                            out_host[dst_flat], src_host[src_flat],
                            "mismatch at b={} h={} d={} m={}",
                            b_idx, h_idx, d_idx, m_idx
                        );
                    }
                }
            }
        }
    });
}
