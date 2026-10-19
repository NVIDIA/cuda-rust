# oxide-tile-provider

The runtime inside a cuda-oxide tile library that is a linkage provider.

A tile library can be static (descriptions and LTOIR packed into a `.tilelib`
ahead of time) or a provider: a shared object the Tile IR assembler loads while
it compiles a Tile kernel. The assembler asks the provider which
implementations exist, lets it rank the compatible ones, and then asks for the
device code behind the ones it selected. This crate answers those callbacks on
top of the `#[tile_export]` descriptions and the device code embedded in the
shared object.

```text
tileiras                                provider shared object
   |  tileir_create_linkage_provider          | version check, payload parsed once
   |  provide_impls(one call site)            | catalog bytecode for that declaration
   |  filter_and_rank(candidates)             | keep all, score 0
   |  instantiate(selected calls)             | LTOIR exporting the asked symbols
   |  free_* , destroy                        | buffers back, state dropped
```

The C ABI comes from `oxide-tile-provider-sys`. The catalogs are written with
`oxide-tilebc` from the same descriptions the static form uses, so the two
forms describe an export identically.

## Building a provider

`cargo oxide tilelib --provider` writes and builds the `cdylib`. By hand, a
crate with `crate-type = ["cdylib"]` and this one item is enough:

```rust,ignore
oxide_tile_provider::define_provider! {
    name: "relu",
    descriptions: include_str!("relu.tileexports.json"),
    bytecode_version: (13, 5),
    ltoir: include_bytes!("relu.ltoir"),
    arch: "sm_120",
}
```

`bytecode_version` is the version the assembler calls current: preview content
is read only at that version, so it is the number `tileiras --list-versions`
printed when the library was built.

## What the callbacks promise

- `provide_impls` returns a catalog only for declarations in the payload. A
  call to any other declaration gets the "no support" answer, so other
  providers or frontend implementations can serve it.
- `instantiate` serves a selected implementation only when its symbol is one
  the embedded code defines and the kernel's architecture is the one the code
  was compiled for. Anything else is reported as an unresolved call or an
  error with a message; nothing is guessed.
- Every buffer handed out is owned by the provider until the matching
  `free_*` callback; the state is immutable after creation, so the callbacks
  are safe to call from any thread. A panic never unwinds into the assembler.

`OXIDE_TILE_PROVIDER_LOG=1` in the assembler's environment prints every
callback with the call sites, symbols and architectures involved.
