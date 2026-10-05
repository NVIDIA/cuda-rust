# Opt-in removal of indexing bounds checks

Normally, every `a[i]` in a kernel compiles to a hidden safety check: "is
`i` inside the slice? if not, stop the kernel". In PTX that is a compare, a
guarded branch, and a `trap;` block. In hot loops those checks can dominate
the runtime (removing them safely is worth 2.4x on the naive `gemm_views`
kernel).

This example demonstrates the blunt instrument for removing them:

```rust
#[kernel(unchecked_indexing)]     // per kernel
```

```bash
CUDA_OXIDE_UNCHECKED_INDEXING=1   # whole build (env var)
cargo oxide build --unchecked-indexing   # whole build (flag)
```

The contract is the same as `slice::get_unchecked`: you promise every index
is in bounds. If you are wrong there is no trap and no error, just
undefined behavior, possibly silently corrupting unrelated memory. Prefer
the proof-carrying views (see `examples/gemm_views`) where they fit; use
this switch for code shapes they cannot express yet, after the checked
build has been proven correct (for example under `compute-sanitizer`).

Only indexing checks are removed. Arithmetic overflow, division by zero,
and every other safety check keep trapping. Range indexing (`&a[i..j]`)
also still traps.

## What the example runs

- `indexed_sum_checked` and `indexed_sum_unchecked`: byte-identical kernel
  bodies, indexing with a raw thread id the compiler cannot prove in
  bounds. The only difference is the attribute.
- `scaled_gather<T>` (generic, opted in) and `gather_then_check` (NOT opted
  in, but calls `scaled_gather`'s helper function): a regression guard
  proving the flag never leaks from an opted kernel into a caller that
  didn't ask for it.

The host runs all kernels on the GPU, compares results against a CPU
reference, then inspects the generated PTX and asserts:

```text
indexed_sum_checked    traps present   (default behavior intact)
indexed_sum_unchecked  zero traps      (the flag worked)
scaled_gather          zero traps      (works for generic kernels)
gather_then_check      traps present   (no leak into non-opted callers)
```

```bash
cargo oxide run unchecked_indexing
```
