# Persistent Two-CTA BF16 GEMM + PDL (`sm107_bf16_gemm_cute`)

Two CTAs team up as a `(2, 1, 1)` cluster to compute `C = A * B^T` (`BFloat16` inputs/outputs, `FP32` accumulation) on Vera Rubin (`sm_107a`) and Blackwell (`sm_100a`, `sm_103a`): each CTA loads half of `A` (`64x64`) and half of `B` (`128x64`) into shared memory, and one `tcgen05` instruction multiplies the combined `128x256` tile.

The `_pdl` variant adds **Programmatic Dependent Launch (`PDL`)**, letting a downstream kernel start prefetching descriptors and weights before the previous kernel finishes draining its last tile.

## Pipeline Overview

```text
  PDL Prologue (sm107_bf16_gemm_pdl):
    prefetch_descriptor(A, B) -> griddepcontrol_wait() -> griddepcontrol_launch_dependents()

  Global A/B (BF16) ──► TmaCopy<_, 2, BFloat16> (Warp 5) ──► 8-Stage SMEM Ring (192 KB)
                                                                   │
                                                                   ▼  TwoCtaRing<8>
                                                          MmaSs<..., BFloat16> (Warp 4)
                                                                   │
                                                                   ▼  TwoCtaRing<2> (TMEM Ping-Pong)
  Global C (BF16)   ◄── 256-bit Streaming Store          ◄── TmemFragment<128> + TensorSSA (Warps 0..3)
```

| Component         | `cute-rs` Type                                           |
|:------------------|:---------------------------------------------------------|
| `A` SMEM Layout   | `ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<64, 64>>`  |
| `B` SMEM Layout   | `ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<128, 64>>` |
| Tiled MMA         | `MmaSs<ALayout, BLayout, 128, 256, 64, 2, BFloat16>`     |
| Tile Scheduler    | `PersistentTileSchedulerParams<128, 256, 8, 2>`          |
| PDL Control       | `cute_rs::arch::{griddepcontrol_wait, griddepcontrol_launch_dependents}` |

## Vera Rubin (`sm_107a`, 216 SMs) vs. CuTeDSL 4.8.1

`N = 4096, K = 4096`, `1,000x` CUDA Graph replay (`128x256` 2-CTA schedule, `0.000000` normalized error):

| `M`    | `CuTeDSL` (`us`) | `cute-rs` (`us`) | Speedup | `cute-rs` `TFLOP/s` |
|:-------|:-----------------|:-----------------|:--------|:--------------------|
| `256`  | `16.432`         | `16.373`         | `1.00x` | `524.6`             |
| `512`  | `16.593`         | `16.549`         | `1.00x` | `1,038.1`           |
| `1024` | `25.342`         | `24.765`         | `1.02x` | `1,387.4`           |
| `2048` | `33.950`         | `33.247`         | `1.02x` | `2,067.0`           |
| `4096` | `62.244`         | `52.413`         | `1.19x` | `2,622.2`           |
| `8192` | `124.269`        | `104.001`        | `1.19x` | `2,643.0`           |

At `M <= 256`, narrower tiles (`128x64`) expose `4x` more clusters across 216 SMs (`6.74 us` at `M = 128`). From `M = 512` to `8192`, `cute-rs` (`G = 8` group-M L2 swizzle + 256-bit streaming stores) matches or beats an 8-schedule `CuTeDSL` sweep (`106.57 us` best on `256x256`).

## Run

```bash
# On Vera Rubin (sm_107a):
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run sm107_bf16_gemm_cute --arch sm_107a

# On Blackwell B200 (sm_100a):
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run sm107_bf16_gemm_cute --arch sm_100a
```
