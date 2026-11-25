# Public CUTLASS reference

The copied source and helpers are byte-for-byte NVIDIA CUTLASS **v4.7.0**,
commit `dcf215af68a2d08d305076c152a06f201728cd53`, with their original BSD-3-Clause
license notices. The historical `blackwell/fmha/fmha.py` location moved to the
path below. The adapter is the first-party `../compare.py` function
`sm100_fmha_prefill_cutedsl`; the copied source remains unchanged.

Paths below are relative to upstream `examples/python/CuTeDSL/` and to this
local `reference/` directory.

| File | SHA-256 |
| --- | --- |
| [`cute/blackwell/kernel/attention/fmha/fmha.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py) | `c11801367b5681e531b756238920124ba4d2ca7809bdb4f9aacd9aaa9ae3a955` |
| [`helpers/fmha_helpers.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/helpers/fmha_helpers.py) | `cbe090eb5567b7cdbd8b4944cbd579084fd60379c463bcfb011e6d4b28ef20e1` |
| [`helpers/__init__.py`](https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/helpers/__init__.py) | `9cf1ae75e6473e5acf38b8ef7b498a14ff74fd54741a930708f6995fcacb7581` |

These are the complete local Python helper dependencies. Other imports are
provided by the packages in `requirements.txt`. The tested Torch wheel has
CUDA13.0 support (`2.13.0+cu130`). The Rust backend uses the repository's pinned
CUTLASS4.7 compiler, installed with `cargo oxide toolchain install cutlass`.
