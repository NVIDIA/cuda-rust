# CuTe dialect

`dialect-cute` defines backend-neutral CuTe operations for cuda-oxide using
Pliron IR. Operations retain typed layouts and collective contracts alongside
ordinary MIR scalar and pointer carriers.

The public source boundaries are:

| Module | Responsibility |
| --- | --- |
| [`ops/`](src/ops) | Operation families and their local verification contracts. |
| [`attributes/`](src/attributes) | Static layouts, formats, roles, and plans. |
| [`types/`](src/types) | Tensor descriptions, fragment shapes, and compiler-only handles. |
| [`verify/`](src/verify) | Function-level control flow, provenance, and operation protocol checks. |

Prefer the grouped operation paths below. Each former root module is a public
alias to the same module, preserving its existing types and functions.

| Preferred path | Compatibility alias | Contents |
| --- | --- | --- |
| `ops::copy` | `copy_ops` | Direct and cooperative copies, raw TMA transport, cache-aware global stores. |
| `ops::tensor` | `tensor_ops` | Tensor construction, partitioning, selection, and transfers. |
| `ops::gemv` | `gemv_ops` | Scaled tensor views and block-scaled dot products. |
| `ops::gemm_tma` | `gemm_tma_ops` | Typed global/shared transport views and GEMM TMA copies. |
| `ops::mma` | `smem_mma_ops` | Shared tensor overlays, fragments, matrix loads, and tiled GEMM. |
| `ops::epilogue` | `epilogue_ops` | Shared output tiles and asynchronous TMA stores. |
| `ops::pipeline` | `pipeline_ops` | Load-pipeline construction, scalar state, and lifecycle. |
| `ops::scheduler` | `scheduler_ops` | Persistent work grids, coordinates, and cursor advancement. |
| `ops::sm100` | `sm100_ops` | Two-CTA tensor-memory MMA, cluster TMA, and pipelines. |
| `ops::sm100_attention` | `sm100_attention_ops` | Independent SM100 attention MMA and TMA contracts. |

The existing `ops::CuteCopyOp`, `ops::CuteCopyG2SOp`, `ops::CuteAssumeDivOp`,
`ops::CuteLdmatrixOp`, `ops::CuteTmaLoad2dOp`, and `ops::CuteTmaStore2dOp` imports
also remain available. Larger families separate production responsibilities
into private children and keep regression fixtures in `tests.rs`.

Run the host checks from the owning workspace, `cuda-oxide/`:

```bash
cargo test -p dialect-cute
cargo clippy -p dialect-cute --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p dialect-cute --no-deps
cargo fmt -p dialect-cute --check
```
