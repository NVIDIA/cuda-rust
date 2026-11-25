# FlashInfer CuTeDSL SOTA Reference (`nemotron3_moe`)

This folder contains the standalone **FlashInfer CuTeDSL (`flashinfer.fused_moe.cute_dsl_fused_moe`, `w4a4` + `ActivationType.Relu2`)** benchmark harness used to produce the `FlashInfer CuTeDSL (us)` reference column in [`../src/main.rs`](../src/main.rs).

Both the Rust harness and [`benchmark_flashinfer.py`](benchmark_flashinfer.py) measure the **exact same 4-stage MoE layer** (`H = 1024, I = 1024, E = 8, top_k = 4`) inside a captured CUDA Graph replayed **`1,000×` per sample** (`20` samples averaged, `50` untimed graph warmups per sample) on NVIDIA B200 (`sm_100a`).

## Reproduce the FlashInfer CuTeDSL Numbers

In a Python environment with PyTorch (CUDA 13) and `flashinfer` (`>= 0.2.3` with `cute_dsl` Blackwell support) installed:

```bash
python3 reference/benchmark_flashinfer.py
```
