# Provably Safe Async TMA Copy (`safe_async_copy`)

Stage 5 of `cuda-oxide`'s provably safe kernel pipeline:
- **`async_copy_stage`**: 1-stage TMA bulk asynchronous global-to-shared copy (`cp_async_bulk_tensor_2d_g2s_cta`) synchronized via `mbarrier_arrive_expect_tx` and `mbarrier_try_wait_parity`, proved safe with `#[verify(no_oob_idx(TILE), no_data_race(TILE), uniform_barriers)]`.

```bash
cargo oxide lean-export safe_async_copy
cargo oxide run safe_async_copy
```
