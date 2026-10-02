# Pliron to MLIR export

This crate translates a Pliron module into textual MLIR.

It deliberately knows nothing about CUDA, CuTe, or a particular MLIR build.
Those details are supplied as mapping packs:

```text
Pliron module
      │
      ▼
mapping registry ──► small MLIR syntax tree ──► deterministic text
      ▲
      │
core / NVVM / CuTe mapping packs
```

Every source operation, type, and attribute needs an explicit mapping. A
matching name is never treated as permission to copy an unknown item through.
That makes dialect drift a clear translation error instead of malformed MLIR.
