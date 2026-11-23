# Rubin Persistent MXFP8 GEMM + B-Collector Reuse (`sm107_mxfp8_gemm_cute`)

Vera Rubin (`sm_107a`) doubles FP8 Tensor Core reduction depth to `K = 64` per instruction and adds a hardware **B-collector** buffer. Every 32 `Float8E4M3FN` values share one power-of-two `UE8M0` scale byte (`MXFP8`), and B-collector reuse (`CollectorFill` -> `CollectorLastUse`) stashes `B` inside the Tensor Core on the first `128`-row half of `A` so the second `128`-row half reuses `B` without reading shared memory again.

This example compares both schedules head-to-head (`128x128` `CollectorDiscard` vs. `256x128` B-collector reuse + `PDL`) against an exact CPU `MXFP8` reference.

## Pipeline Overview

```text
  Each K = 128 TMA Stage (50,688 B/stage, 4 Stages = 203 KB SMEM):
    A0, A1 (128x128 E4M3) + SFA0, SFA1 (512 B UE8M0) ──┐
    B      (128x128 E4M3) + SFB        (512 B UE8M0) ──┴──► PipelineTmaAsync<4> (Warp 5)
                                                                     │
                                                                     ▼
    Warp 4 (MMA Consumer):
      SM107BlockScaledMmaMXF8F6F4Op::copy (SFA0, SFA1, SFB -> TMEM)
      for sub_k in [0, 1]:
        MmaBKeep::gemm (TMEM[  0..128], A0, B, SFA0, SFB, sub_k)   // CollectorFill (stash B)
        MmaBReuse::gemm(TMEM[128..256], A1, B, SFA1, SFB, sub_k)   // CollectorLastUse (reuse B)
                                                                     │
                                                                     ▼  SingleCtaRing
    Warps 0..3 (Epilogue):
      TmemFragment<128>::load -> TensorSSA::to::<BFloat16>() -> 256-bit Streaming Store
```

| Component         | `cute-rs` Type                                                                       |
|:------------------|:-------------------------------------------------------------------------------------|
| `A` / `B` Layout  | `ComposedLayout<Swizzle<3, 4, 3>, 0, RowMajor<128, 128>>`                            |
| Scale Layout      | `Sm1xxBlockScaleKMajor<32>` (`512`-byte `(32, 4, 4)` `UE8M0` atoms)                  |
| B-Keep MMA        | `SM107BlockScaledMmaMXF8F6F4Op<ATileSmem, BTileSmem, 128, 128, 1, CollectorFill>`    |
| B-Reuse MMA       | `SM107BlockScaledMmaMXF8F6F4Op<ATileSmem, BTileSmem, 128, 128, 1, CollectorLastUse>` |
| PDL Control       | `cute_rs::arch::{griddepcontrol_wait, griddepcontrol_launch_dependents}`             |

## Vera Rubin (`sm_107a`, 216 SMs) vs. CuTeDSL 4.8.1

`N = 4096, K = 4096`, `1,000x` CUDA Graph replay (`0.000000` normalized error):

| `M`    | `CuTeDSL` `128x128` | `cute-rs` `128x128` | `CuTeDSL` `256x128` | `cute-rs` `256x128` | Parity  | Peak `TFLOP/s` |
|:-------|:--------------------|:--------------------|:--------------------|:--------------------|:--------|:---------------|
| `256`  | `8.449 us`          | `8.088 us`          | `10.459 us`         | `10.064 us`         | `1.04x` | `1,062.0`      |
| `512`  | `9.369 us`          | `8.989 us`          | `10.788 us`         | `10.468 us`         | `1.03x` | `1,911.2`      |
| `1024` | `14.867 us`         | `14.633 us`         | `12.155 us`         | `12.222 us`         | `0.99x` | `2,811.3`      |
| `2048` | `19.969 us`         | `20.192 us`         | `20.870 us`         | `21.731 us`         | `0.99x` | `3,403.3`      |
| `4096` | `30.812 us`         | `31.732 us`         | `30.075 us`         | `30.783 us`         | `0.98x` | `4,464.8`      |
| `8192` | `60.447 us`         | `59.031 us`         | `53.354 us`         | `53.122 us`         | `1.00x` | `5,174.5`      |

Both single-CTA schedules match Python `CuTeDSL` `4.8.1` at `0.98x - 1.04x` (`5.17 PFLOP/s` at `M = 8192`). Pairing two CTAs on a `256x256` tile halves per-CTA TMA traffic and pushes `M = 8192` further to `42.76 us` (`6.42 PFLOP/s`).

## Run

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir cargo oxide run sm107_mxfp8_gemm_cute --arch sm_107a
```
