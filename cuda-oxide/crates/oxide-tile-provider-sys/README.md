# oxide-tile-provider-sys

Raw Rust bindings to the Tile IR linkage provider C ABI.

A linkage provider is a shared object the Tile IR assembler loads while it
compiles a Tile kernel. The assembler looks up `tileir_create_linkage_provider`,
receives a vtable, and asks the provider for implementation descriptions and
then for device code. This crate is that ABI and nothing else; the callbacks
are implemented in `oxide-tile-provider`.

```text
LinkageProviderAPI.h  --bindgen at build time-->  OUT_DIR/bindings.rs
        ^                                                  |
TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR                include!() by src/lib.rs
```

## The header is not in this repository

The bindings are generated from `tile_ir/linkage_provider/LinkageProviderAPI.h`
in the Tile IR compiler's source tree. That header carries NVIDIA's proprietary
notice, ships in no CUDA toolkit and in no public repository, and describes
itself as an in-tree, pre-release interface without compatibility guarantees.
It is therefore read from outside the tree:

```bash
export TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR=/path/to/tile_ir/compiler/linkage_provider/include
cargo build -p oxide-tile-provider-sys
```

`upstream.lock` pins the header by SHA-256 and records where the bytes came
from. The build refuses a header with another hash, because the ABI is a 0.x
interface whose record layouts can change without a version bump, and a
provider built from the wrong header fails inside the assembler. Set
`TILEIR_LINKAGE_PROVIDER_UNPINNED=1` to build against a different header
anyway; re-pin once it has been checked against the assembler that loads it.

## Conventions

- Every C enum is a newtype over its integer representation. The header says
  unknown values are protocol errors, and the newtype keeps them representable
  so the runtime can report them instead of invoking undefined behaviour.
- `ABI_VERSION` and `ENTRY_SYMBOL` are the two constants a provider needs
  beyond the generated items.
- bindgen's layout tests run as this crate's unit tests, so `cargo test` checks
  the generated layout against libclang's reading of the header.
