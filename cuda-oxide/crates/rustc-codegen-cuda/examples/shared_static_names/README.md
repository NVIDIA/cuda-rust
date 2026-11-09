# shared_static_names

## Stable names for shared-memory and device-global statics

Each `SharedArray` static becomes a `.shared` global in the PTX, and each
ordinary `static` a kernel reads becomes a `.global`. Their names come from
the Rust static itself:

```text
__shared_mem_<crate id>_<hash of the static's mangled symbol>
__device_global_<crate id>_<hash of the static's mangled symbol>
```

The crate id is the crate's `StableCrateId`, which keeps the names distinct
when several crates' PTX is merged into one module. Neither part depends on
which other kernels or statics the crate has, so adding, removing or
reordering a kernel leaves the other kernels' PTX unchanged. Release and
full-debug builds name a static the same way.

## What This Example Does

1. **rotate_128**: one 128-entry `SharedArray`; each thread reads its
   neighbor's slot
2. **rotate_256**: two 256-entry `SharedArray`s and a `static` weight table,
   so it has both kinds of generated global

The host checks both kernels' outputs.

`scripts/check-shared-static-names.sh` builds this example, then copies of it
with extra kernels spliced in at the `ADDED KERNEL SLOT` comments, in release
and full-debug builds, and checks that both kernels and the declarations
of every global they use are unchanged. It needs no GPU.

## Running

```bash
cargo oxide run shared_static_names
just check-shared-static-names
```

## Expected Output

```text
=== shared_static_names (#1477) ===
SUCCESS: rotate_128 and rotate_256 match the host reference
```
