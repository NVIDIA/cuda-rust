# `safe_tcgen05_matmul`

Provably safe Stage 6 Blackwell (`sm_100a`, NVIDIA B200) `128 x 128 x 16` FP16-input / FP32-accumulator Tensor Core matrix multiply (`C = A * B^T`) using asynchronous 2D TMA copies, `mbarrier` token ordering, PTX `generic -> async` proxy coherence, `tcgen05.mma`, and Tensor Memory (`TMEM`) with **zero `unsafe` blocks inside the kernel**.

## Pipeline Overview

```text
  [Global Memory] ──(2D TMA g2s)──► [SMEM_A, SMEM_B] ──(tcgen05.mma)──► [TMEM 128x128] ──(tcgen05.ld)──► [DisjointSlice C]
```

| Stage             | Hardware Mechanism                     | Compile-Time Proof Obligation                        |
|:------------------|:---------------------------------------|:-----------------------------------------------------|
| **Init**          | `mbarrier_init` + `fence.proxy.async`  | Proxy coherence + `sync_threads()` before wait       |
| **TMEM Alloc**    | `tcgen05_alloc(&TMEM_ADDR[0], 128)`    | Warp-0 collective + `TmemValidCols 128`              |
| **TMA Stage**     | `cp_async_bulk_tensor_2d_g2s_cta`      | `expect_tx <= 4096` (`NoOobTileAccess` + `NoDataRace`)|
| **Tensor Core**   | `tcgen05_mma_f16` + `tcgen05_commit`   | `StageReady` on `TMA_*_BAR` + `NoOobTmemAccess`      |
| **TMEM Epilogue** | `tcgen05_ld_32x32b_x1_raw` + `wait`    | `TmemReady(MMA_BAR)` + warpgroup 32-lane slice proof |
| **TMEM Dealloc**  | `tcgen05_dealloc(tmem_addr, 128)`      | Exit post-dominance + matching `128` column count    |

## Requirements

* CUDA Toolkit 13.1+ with `nvcc` and `ptxas`.
* Lean 4 (`lake` on `PATH` or `~/.elan/bin/lake`) for compile-time formal verification.
* Datacenter Blackwell GPU (`sm_100a`, such as NVIDIA B200) to execute on device; non-`sm_100a` hosts verify Lean 4 proofs and PTX emission cleanly.

## Build & Run

```bash
cargo oxide build safe_tcgen05_matmul --arch sm_100a
cargo oxide run safe_tcgen05_matmul --arch sm_100a
```
