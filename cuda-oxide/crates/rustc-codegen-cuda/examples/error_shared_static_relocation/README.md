# `error_shared_static_relocation`

Negative test for a constant that stores a type-erased pointer to a
`SharedArray` static:

```rust
static mut TILE: SharedArray<u32, 4> = SharedArray::UNINIT;
const INDIRECT: &*const u32 = &(&raw const TILE as *const u32);
```

A `SharedArray` or `Barrier` static is per-block shared memory with no
initializer, so it has no initialized device global. A direct `&raw const TILE`
operand reaches the block's shared allocation. A pointer to it stored inside a
constant (or another static) is instead an initializer relocation, and a
relocation can only point at storage that has its own initialized global.
Emitting one for `TILE` would create a second, zero-byte `addrspace(1)` global
whose address is not the block's shared memory.

The cast hides shared memory from the stored pointer's type, so the importer
must refuse the relocation at compile time from the static it points at:

```bash
cargo oxide build error_shared_static_relocation
```

Expected diagnostic (pinned by `cuda-oxide/scripts/smoketest.sh`):

```text
points at shared-memory static
```

Pointers to ordinary and `ConstantMemory` statics stored inside constants stay
supported; see `promoted_ref_return`. A stored pointer whose type names shared
memory is refused from its type; see `error_shared_pointer_constant`.
