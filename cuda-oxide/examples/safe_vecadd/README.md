<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Provably Safe 1D Vector Addition (`safe_vecadd`)

Demonstrates a provably safe 1D GPU kernel in `cuda-oxide`, combining compile-time Lean 4 safety proofs at the `PostMem2Reg` `dialect-mir` stage with host-side `#[launch_contract]` precondition checking.

```text
  #[launch_contract(domain = 1, block = (256, 1, 1),
                    requires = (a.len() >= c.len(), b.len() >= c.len()))]
  #[verify(no_oob_idx(a, b, c), disjoint_writes(c), writes_all_elems(c))]
  #[kernel]
  pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
        │
        ├──► Compile time: `lean-exporter` extracts `PostMem2Reg` `dialect-mir`
        │    and runs Lean 4 (`lake env lean`) to prove:
        │      • `no_oob_idx(a, b, c)` (`CudaOxide.Contracts.NoOobIdx`)
        │      • `disjoint_writes(c)`  (`CudaOxide.Contracts.DisjointWrites`)
        │      • `writes_all_elems(c)` (`CudaOxide.Contracts.WritesAllElems`)
        │
        └──► Launch time: `#[cuda_module]` checked launcher enforces
             `a.len() >= c.len()` and `b.len() >= c.len()` on the CPU
```

## Run

From the `cuda-oxide/` directory:

```bash
cargo oxide run safe_vecadd
```

To inspect the exported Lean module and generated theorems directly:

```bash
cargo oxide lean-export crates/lean-exporter/fixtures/vecadd.mir --function vecadd
```
