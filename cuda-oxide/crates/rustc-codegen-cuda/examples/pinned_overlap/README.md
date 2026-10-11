# Pinned transfer overlap

This example makes the effect of CUDA page-locked host memory visible in two
ways:

1. It compares pageable and pinned HtoD/DtoH bandwidth from 1 MiB through 256 MiB.
2. It compares a serialized pageable upload -> kernel -> download loop with a
   three-stream pipeline using rotating pinned host stagers and persistent device buffers.

Run it from the repository root:

```bash
cargo oxide run pinned_overlap
```

See the [book chapter](https://github.com/NVIDIA/cuda-rust/blob/main/cuda-oxide/cuda-oxide-book/async-programming/overlapping-transfers-and-compute.md) for the API and pipeline explanation.

The program verifies every transformed element before printing its `SUCCESS`
marker. It reports median CUDA-event bandwidth and wall-clock pipeline time so
the result shows both the transfer improvement and the end-to-end overlap
speedup. The benchmark allocates its staging buffers before timing; ring slots
are only refilled after their completion event has fired. The pageable
bandwidth columns time the simple helper path, which allocates a fresh
destination each iteration; the pinned columns reuse persistent buffers, so
the gap reflects both the data path and the allocation cost.

## Results

Reference run on an RTX 4090 / PCIe Gen4 host (`sm_89`):

| Transfer size | Pageable HtoD | Pinned HtoD | Pageable DtoH | Pinned DtoH |
|---:|---:|---:|---:|---:|
| 1 MiB | 11.60 | 22.58 | 8.93 | 25.54 |
| 4 MiB | 15.23 | 23.61 | 11.99 | 26.66 |
| 16 MiB | 16.56 | 23.88 | 12.84 | 18.37 |
| 64 MiB | 14.70 | 19.55 | 3.53 | 16.95 |
| 256 MiB | 12.36 | 23.62 | 3.61 | 23.85 |

| Pipeline | Time |
|---|---:|
| Serialized pageable | 174.996 ms |
| Overlapped pinned | 99.332 ms |
| Speedup | 1.76x |
