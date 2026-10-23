# Public CUTLASS source provenance

The files under `cute/blackwell/kernel/attention/mla/` are unmodified public
NVIDIA CUTLASS v4.7.0 sources at commit
`dcf215af68a2d08d305076c152a06f201728cd53`. Their BSD-3-Clause notices are
preserved. The first-party adapter is `../cutedsl.py`.

| File | SHA-256 |
| --- | --- |
| [`mla_decode_fp16.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py) | `82116d97e611366d3122473ada3fde7e2550f1e06914ba1834fc448954dfd106` |
| [`mla_helpers.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_helpers.py) | `13b0b43a5417be6d79369354ab90dbaddd2c49ec49c982258bf90fe2f285a2ae` |

These files form the complete example-local dependency closure. Other imports
come from the pinned CUTLASS package, CUDA Python, and PyTorch. The directory
hierarchy supports upstream's absolute `cute.blackwell...` import unchanged.
The comparison adapter verifies these hashes before compilation.
