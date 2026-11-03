# promoted_ref_return

Regression for [#1724](https://github.com/NVIDIA/cuda-rust/issues/1724):
device helpers that return a `&'static` reference to constant memory.

A reference constant such as `const PAIR: &(u32, u32) = &(3, 4)`, or a borrow
rustc promotes to `'static` such as `&42`, points at an anonymous rustc
allocation. Safe code can return that reference from the function that names
it. Each such allocation must therefore live in a module-level read-only
device global, not in the frame of whichever function first mentions it.

Every helper here is `#[inline(never)]`, so the reference crosses a real call
boundary before the kernel reads through it. The helpers cover:

- named tuple and struct reference constants, and promoted literals;
- `&&7`, whose outer allocation holds a pointer to an inner one;
- `&PAIR.1`, an interior pointer that shares `PAIR`'s global;
- `&Some(11)`, a reference to a payload-carrying enum;
- by-value tuple and `Option<&u32>` constants whose pointer fields point at
  anonymous memory;
- `&&TABLE[2]`, a promoted pointer into a device static;
- device statics whose initializers are `&19` and `&[31, 37, 41]`;
- `&Option<u32>` and `&u32` into its payload from one constant, which must stay
  one allocation: the kernel checks that both payload addresses are equal;
- `*mut u32` constants formed with `as` casts from references, at offset 0,
  4 bytes into `PAIR`, and as a tuple field. The kernel only reads through
  them: the storage stays read-only.

`verify-code-shape.sh` checks the emitted LLVM IR and PTX: no helper has a stack
slot or a `.local` depot, the constants are read-only `addrspace(1)` globals,
`PAIR` has exactly one global, and nested pointers are relocations to the inner
allocation's global. It then builds two copies of the example, one with extra
kernels that carry their own constants spliced in at the `ADDED KERNEL SLOT`
comment, and checks from the `dialect-mir` dump (`CUDA_OXIDE_DUMP_MIR`) that
every helper's promoted globals keep the same keys. These globals are keyed by
their contents, so unrelated constants never rename them.

Run with:

```bash
cargo oxide run promoted_ref_return
```

Expected output:

```text
promoted_ref_return: PASS (20 values)
```
