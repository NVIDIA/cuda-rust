# generic_mono_in_lib

Regression for [issue #1365](https://github.com/NVIDIA/cuda-rust/issues/1365):
an all-generic `#[cuda_module]` in a library that monomorphizes locally must
still keep its `.oxart` archive member via the #72 artifact-anchor handshake.

```bash
cargo oxide run generic_mono_in_lib
cargo oxide run -- generic_mono_in_lib -- --verify-bundles
```
