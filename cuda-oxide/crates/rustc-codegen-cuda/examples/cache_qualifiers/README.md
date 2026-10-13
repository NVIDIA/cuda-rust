# cache_qualifiers

Demonstrates and benchmarks PTX cache-control streaming and eviction qualifiers
via `cuda_device::cache` (`load_streaming`, `load_l2_only`, `store_streaming`, `store_l2_only`).

In bandwidth-bound and memory-intensive GPU kernels, single-pass streaming data
(such as large linear reductions, streaming activations, or attention KV caches)
can pollute the SM's L1 data cache and L2 cache, evicting hot reusable working
sets. By using explicit cache qualifiers:

- **Streaming (`.cs`)**: Allocates cache lines with evict-first priority,
  preventing cold linear sweeps from displacing hot resident tiles.
- **L2-Only / Bypass L1 (`.cg`)**: Directs memory operations to cache at the
  global L2 level while bypassing L1 entirely, conserving L1 capacity for register
  spills and shared memory.

## Running

```bash
cargo oxide run cache_qualifiers
```
