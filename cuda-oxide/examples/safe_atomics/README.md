# Provably Safe Scoped Atomics & Warp Votes (`safe_atomics`)

Demonstrates Lean 4 formal verification (`0` `unsafe` blocks inside kernels) across five GPU atomic and warp-collective patterns:

| Kernel                     | Scopes & Primitives                                   | Lean 4 Obligations Proved                                      |
|:---------------------------|:------------------------------------------------------|:---------------------------------------------------------------|
| `two_level_histogram`      | `SharedArray<BlockAtomicU32, 256>` + `DeviceAtomicU32`| `MirNoOobGlobal`, `MirNoOobShared`, `MirNoSharedRace`, `MirNoGlobalRace`, `UniformBarriers` |
| `ballot_warp_compaction`   | `warp::ballot` + `lanemask_lt` + `DeviceAtomicU32`    | `MirNoOobGlobal`, `MirNoGlobalRace`, `UniformBarriers`         |
| `cas_max_and_fault_ticket` | `compare_exchange` + `fetch_min`/`or` + `AcqRel`      | `MirNoOobGlobal`, `MirNoGlobalRace`, `MirDisjointWrites`, `UniformBarriers` |
| `per_block_cta_counters`   | Global `BlockAtomicU32` partitioned by `blockIdx_x()` | `MirNoOobGlobal`, `MirNoGlobalRace` (`bx1 = bx2`)              |
| `float_and_system_atomics` | `DeviceAtomicF32::fetch_add` + `SystemAtomicU32`      | `MirNoOobGlobal`, `MirNoGlobalRace`                            |

```bash
cargo oxide lean-export safe_atomics --function two_level_histogram
cargo oxide run safe_atomics
```
