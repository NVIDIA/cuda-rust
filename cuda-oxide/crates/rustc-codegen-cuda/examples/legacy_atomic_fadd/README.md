# Legacy NVVM floating-point atomic add

Exercises device-scoped relaxed `f32` and `f64` atomic add through CUDA's LLVM 7 NVVM IR dialect.

```bash
cargo oxide run legacy_atomic_fadd --emit-nvvm-ir --arch sm_86
```

The run must report both counters at `256`, verify that each `fetch_add`
returned a permutation of the old values `0..255`, and print
`legacy_atomic_fadd: PASS`.
