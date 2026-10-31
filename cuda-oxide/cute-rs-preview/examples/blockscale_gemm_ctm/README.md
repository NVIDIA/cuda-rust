# MXFP4 GEMM with Rust CTM (ctm-rs)

This cuda-oxide example multiplies packed E2M1 matrices with UE8M0 scales per
32 K values, FP32 accumulation and FP16 output. The fixed problem is
**M=N=2048, K=1024** on an **SM120 Blackwell GPU**, such as RTX 5090.

```text
A/B + scales → TMA shared-memory ring → register fragments
            → block-scaled MMA → FP16 epilogue → TMA output
```

The 128×128 tiles use three pipeline stages, eight compute warps and one
producer warp. Typed register primitives keep operand roles explicit;
physical lane addresses, barriers and publication remain explicit Rust.
The Rust executable needs no Python runtime.

From `cuda-oxide/`, with the pinned CUTLASS 4.7 compiler installed using
`cargo oxide toolchain install cutlass`:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_MLIR_PROFILE=cutlass-primitives-mlir22 \
CUDA_OXIDE_CUTLASS_OPT_LEVEL=2 \
  cargo oxide run blockscale_gemm_ctm --arch sm_120a -- \
  --ctas 170 --warmup 100 --samples 31
```

Success checks all **4,194,304 FP16 output bit patterns** against a substantial
CPU oracle, checks repeated-launch output, and reports CUDA-event timings.
This runs Rust only. The [Python CTM reference](reference/README.md) runs the
same GEMM through CUTLASS Primitives and needs its Python dependencies;
`compare.py` beside this README times both under Nsight Systems. `validate.py`
covers 170-, 1- and 17-CTA grids plus memory/tensor and synchronization
sanitizers. From `cuda-oxide/`:

```sh
python3 scripts/check-primitives-api-compile.py --example blockscale \
  --output-dir /tmp/primitives-blockscale-evidence
python3 cute-rs-preview/examples/blockscale_gemm_ctm/validate.py \
  --binary /path/to/blockscale_gemm_ctm --output-dir /tmp/blockscale-results --sanitizer
```

The first command builds a fresh executable through the CUTLASS compiler and
records its manifest; use its `run-*/target/release/blockscale_gemm_ctm` as the
binary.

## Rust CTM versus Python CTM

RTX 5090 at `a0d5776b2` (2026-10-18), same inputs and 170 CTAs. Rust uses
CUTLASS compiler **4.7.0**; Python uses **4.8.0**; both use optimization level 2.
Both pass all output-bit and replay checks. Times are microseconds: median of
three round medians, with 100 warmups and 31 single-launch samples per round.

| Implementation | CUDA events (µs) | Nsight kernel duration (µs) |
| --- | ---: | ---: |
| Rust CTM (`ctm-rs`) | 13.536 | 11.616 |
| Python CTM (CUTLASS Primitives) | 19.520 | 10.880 |

Events include submission gaps; Nsight reports GEMM execution duration,
including device stalls. Python rebuilds tensor-map descriptors per launch;
Rust prepares them before timing. The two metrics come from separate runs,
with rotating order and unlocked clocks.

- [Python CTM API and validation](reference/README.md)
- [CuTe-RS counterpart](../blockscale_gemm_cute)
