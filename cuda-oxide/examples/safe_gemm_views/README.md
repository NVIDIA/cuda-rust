<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Provably Safe 2D Tiled SGEMM with Views (`safe_gemm_views`)

Demonstrates how `cuda-oxide` combines **Rust proof-carrying views** (`MatrixView32`, `DisjointSlice`) for global memory with **Lean 4 safety proofs** (`#[verify(...)]`) for shared-memory tiles (`TILE_A`, `TILE_B`) and `sync_threads()` barriers:

```text
  #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1),
                    requires = (k >= 1, a.len() >= m * k, b.len() >= k * n, c.len() >= m * n))]
  #[verify(no_oob_idx(TILE_A, TILE_B), no_data_race(TILE_A, TILE_B), uniform_barriers)]
  #[kernel(launch_context = lc)]
  pub fn sgemm_tiled_views(...) { ... }
        │
        ├──► Global memory (`a`, `b`, `c`):
        │      Checked once per thread strip via `MatrixView32` and `DisjointSlice`.
        │
        └──► Shared memory (`TILE_A`, `TILE_B`) & `sync_threads()` barriers:
               Verified at compile time by Lean 4 at `PostMem2Reg`:
                 • `no_oob_idx(TILE_A, TILE_B)`  (`CudaOxide.Contracts.NoOobTileAccess`)
                 • `no_data_race(TILE_A, TILE_B)` (`CudaOxide.Contracts.NoDataRace`)
                 • `uniform_barriers`             (`CudaOxide.Contracts.UniformBarriers`)
               Erases all 4 `SharedArray` `mir.assert` bounds checks before PTX lowering.
```

## Run

From the `cuda-oxide/` directory:

```bash
cargo oxide run safe_gemm_views
```

To inspect the exported Lean module and generated theorems directly:

```bash
cargo oxide lean-export crates/lean-exporter/fixtures/sgemm_tiled.mir --function sgemm_tiled_views
```
