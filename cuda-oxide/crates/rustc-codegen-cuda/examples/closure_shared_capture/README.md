# Closures Capturing Shared-Memory Pointers

A non-`move` closure that uses a shared-memory pointer captures the local
holding it by reference. The kernel borrows that local (`&s`) and passes the
borrow to the closure body, which is compiled as a separate function reading a
plain `&*mut T`.

| Kernel                 | Shared pointer source                  | Closure body               |
|------------------------|----------------------------------------|----------------------------|
| `dynamic_smem_closure` | `DynamicSharedArray::<u16>::get()`     | loop of stores             |
| `static_smem_closure`  | `addr_of_mut!` of a `static mut SharedArray` | loop of stores       |
| `cp_async_closure`     | `DynamicSharedArray::<u32>::get()`     | loop of `cp_async_cg_zfill_16` |

The loop keeps rustc's MIR inliner from inlining the closure, so the borrow
survives into codegen.

## Address spaces

A pointer local whose address is taken keeps the address space of its
declared Rust type, which is generic for `*mut T`. The shared pointer is
converted with `addrspacecast` when it is stored, so the borrow points at an
8-byte generic pointer, as the closure body expects; a narrowed slot would hold
a 4-byte `addrspace(3)` pointer under `--nvptx-short-ptr`. After LLVM
inlines the closure, address-space inference restores `st.shared` and
`cp.async.cg.shared.global`. An unoptimized build (`CUDA_OXIDE_NO_OPT=1`)
keeps generic `ld`/`st` for these accesses.

`verify-code-shape.sh` checks the pre-opt IR for the out-of-line closure call
and the `addrspacecast`, and the optimized PTX for shared-memory instructions.

## Run

```bash
cargo oxide run closure_shared_capture
```

Requires sm_80+ for `cp.async`.
