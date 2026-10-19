# bf16x2_fma

Behavior example for `cuda_device::bf16x2::fma_bf16x2`.

The kernel computes two packed bf16 lanes:

- lane 0: `2.0 * 3.0 + 7.0 = 13.0`
- lane 1: `4.0 * 5.0 + 11.0 = 31.0`

Run with:

```bash
cargo oxide run bf16x2_fma
```
