# Provably Safe Sub-Warp Staging (`safe_subwarp_stage`)

Stage 4 of `cuda-oxide`'s provably safe kernel pipeline:
1. **`subwarp_bit_split_stage`**: 16x16 sub-warp bitwise coordinate split (`m_local = tid >> 4`, `k_lane = tid & 15`) staging into `SUBWARP_TILE: SharedArray<f32, 256>` (`(tid >> 4) * 16 + (tid & 15) = tid`) and scanning across `k_lane * 16 + j`.
2. **`multibuf_subwarp_stage`**: 3-phase, 3-buffer (`S_X_DIRECT`, `S_XS_DIRECT`, `S_ACT_DIRECT`) sub-warp staging and reduction modeled on `fc1_relu2_requant_direct` in `nemotron3_moe`.

```bash
cargo oxide lean-export safe_subwarp_stage
cargo oxide run safe_subwarp_stage
```
