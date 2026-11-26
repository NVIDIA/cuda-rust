# Provably Safe Warp & Block Reduction (`safe_warp_reduce`)

Stage 3 of `cuda-oxide`'s provably safe kernel pipeline:
1. **`warp_reduce_sum`**: 1D warp butterfly reduction (`warp::reduce_sum_f32`) with `#[verify(no_oob_idx(data, out), disjoint_writes(out), writes_all_elems(out))]`.
2. **`block_reduce_sum`**: 2-phase 256-thread block reduction across 8 warps using `WARP_SUMS: SharedArray<f32, 8>` and `#[verify(no_oob_idx(WARP_SUMS), no_data_race(WARP_SUMS), uniform_barriers)]`.

```bash
cargo oxide lean-export safe_warp_reduce
cargo oxide run safe_warp_reduce
```
