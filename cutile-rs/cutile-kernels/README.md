# cutile-kernels

Reusable cuTile Rust kernels.

Kernel modules are grouped by function: attention, norms, positional encoding,
KV-cache updates, argmax/sampling, embeddings, and pointwise transformer
utilities.

`experimental/` contains raw-pointer kernels and benchmark-oriented kernels,
including preliminary grouped GEMM/MoE work and f16 KVBM block-layout conversions.
