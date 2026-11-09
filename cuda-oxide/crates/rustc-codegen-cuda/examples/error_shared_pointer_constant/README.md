# `error_shared_pointer_constant`

Negative test for a constant whose type stores a pointer to a `Barrier`:

```rust
const INDIRECT: &&Barrier = &&Barrier::UNINIT;
```

A `&Barrier` or `*const SharedArray<T, N>` is a shared-memory
(`addrspace(3)`) pointer. A pointer stored inside a constant or a static is an
initializer relocation, and every relocation target is an initialized device
global in the global or constant address space: anonymous constant memory, an
ordinary static, or an ordinary static with a `Barrier` field. Loading the
stored pointer would reinterpret that global address as a shared one.

The importer must refuse the constant at compile time instead:

```bash
cargo oxide build error_shared_pointer_constant
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
stores a pointer to shared memory
```

A direct `&raw const` or `&raw mut` of a `SharedArray` or `Barrier` static
stays supported. For a pointer that reaches a shared static through a
type-erased pointer, see `error_shared_static_relocation`.
