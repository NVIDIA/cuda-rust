# Const-generic kernel entries

This example proves that const values participate in a kernel entry's compiled
identity.

```text
write_value::<4> -> write_value_TID_<hash A>
write_value::<8> -> write_value_TID_<hash B>
```

Both specializations have the same runtime parameter types. The PTX must still
contain two distinct `.entry` symbols, and each body must use its own folded
constant. The kernel also calls a const-generic `#[device]` helper, covering the
same forwarding rule for device functions. The raw-pointer kernels and device
helper are `unsafe`; the example also verifies that macro expansion preserves
that caller-visible contract.

A second kernel deliberately does not read its const parameter. Its `<4>` and
`<8>` specializations must still remain two exact, host-addressable PTX entries
even though their optimized instruction bodies are identical.

A third entry is never launched. Calling only `name_only_ptx_name::<4>()` must
still retain that specialization in PTX, so a returned name never points at a
missing entry.

The generated host helper keeps the lookup readable:

```rust
let four = kernels::write_value_ptx_name::<4>();
let eight = kernels::write_value_ptx_name::<8>();
assert_ne!(four, eight);
```

```bash
cargo oxide pipeline const_generic
cargo oxide run const_generic
```

The built executable can compare its own lookup names with generated PTX
without creating a CUDA context:

```bash
cargo oxide build const_generic
./cuda-oxide/crates/rustc-codegen-cuda/examples/const_generic/target/release/const_generic \
  --verify-ptx
```

The executable still links the CUDA driver library, so this explicit local
check requires `libcuda.so.1`. Compile-only CI does not execute it.

That check proves that every host name resolves to PTX, the two `write_value`
entries retain `#[launch_bounds(64)]`, and each body contains its own folded
constant.
