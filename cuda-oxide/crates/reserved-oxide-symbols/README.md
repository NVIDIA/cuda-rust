# reserved-oxide-symbols — INTERNAL

> **Not a public API.** This crate is `publish = false` and exists only to
> keep the macro side and the consumer side of the cuda-oxide naming
> contract in lockstep. The constants, builders, and predicates exposed
> here may change without notice between commits. External consumers
> should depend on `cuda-host`, `cuda-device`, or `cuda-macros` instead.

## What this crate owns

The `cuda_oxide_*` symbol namespace that `#[kernel]` and `#[device]`
mangle user functions into, and that the codegen backend, MIR-lowering,
and LLVM-export passes look for.

Every symbol prefix here contains the magic component `246e25db_`, which is
`sha256("cuda_oxide_ + rust")` truncated to 8 hex chars. The hash
exists purely to make accidental collisions impossible — a user is
never going to write `fn cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_foo()` by accident.

The crate owns these ten symbol constants. The `naming-guard` CI job keeps
reserved-prefix literals out of their consumers:

| Constant                    | Value                                               | What it names |
|-----------------------------|-----------------------------------------------------|---------------|
| `KERNEL_PREFIX`             | `cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_` | `#[kernel]` entry points |
| `DEVICE_PREFIX`             | `cuda_oxide_codegen_v1_cuda_oxide_device_246e25db_` | `#[device]` functions |
| `DEVICE_EXTERN_PREFIX`      | `cuda_oxide_device_extern_246e25db_`                | `extern` device declarations |
| `INSTANTIATE_PREFIX`        | `cuda_oxide_instantiate_246e25db_`                  | forced generic instantiations |
| `CONSTANT_PREFIX`           | `cuda_oxide_const_246e25db_`                        | `#[constant]` statics, for `cuModuleGetGlobal` lookup |
| `KERNEL_SCOPE_LOCAL`        | `cuda_oxide_kernel_scope_246e25db`                  | the hidden thread-index scope token injected by `#[kernel]` / `#[device]` |
| `ARTIFACT_ANCHOR_PREFIX`    | `cuda_oxide_artifact_anchor_246e25db_`              | the link anchor pinning a crate's `.oxart` section into the binary |
| `PTX_MERGE_REQUIRED_PREFIX` | `cuda_oxide_ptx_merge_required_246e25db_`           | statics marking modules whose generic kernels need PTX-bundle merging |
| `LEGACY_KERNEL_PREFIX`      | `cuda_oxide_kernel_246e25db_`                       | pre-scoped-cache kernels — recognized for a compatibility diagnostic, never emitted |
| `LEGACY_DEVICE_PREFIX`      | `cuda_oxide_device_246e25db_`                       | pre-scoped-cache device functions — same |

Internal operation-attribute keys also use this crate as their naming authority.
`KERNEL_REFERENCE_PARAM_VALIDITY_KEY_PREFIX` preserves the compiler metadata
prefix `cuda_oxide_kernel_reference_param_validity_`; it is not a link symbol.
Use `kernel_reference_param_validity_key(index)` to create a key and
`kernel_reference_param_validity_index_text(key)` to read its suffix. The latter
preserves malformed suffixes so LLVM export can diagnose invalid metadata.

The two legacy forms need care: `KERNEL_PREFIX` *contains* `LEGACY_KERNEL_PREFIX`
as a substring, since the modern name is the legacy one behind a
`cuda_oxide_codegen_v1_` scope. `starts_with` tells them apart; `contains` does
not, and reports every modern kernel as legacy. This is why the Layer-3
predicates exist — use them rather than matching strings by hand.

## Layered API

Three concentric layers, pick the one that fits the call site:

1. **Constants** — the raw prefix strings, for code that needs the literal.
2. **Builders** — `kernel_symbol(base) -> String`, etc., for the macro side.
3. **Predicates and extractors** — `is_kernel_symbol(name) -> bool`,
   `kernel_base_name(name) -> Option<&str>`, etc., for the consumer side.

The Layer-3 helpers hide the substring matching that used to be
duplicated across `rustc-codegen-cuda`, `llvm-export`, and `mir-lower`.

## Why "reserved"

The `cuda_oxide_*` namespace is **reserved**: user code may not define
functions whose name starts with it. The `#[kernel]` and `#[device]`
proc macros enforce this at the source-code level via a compile-error
guard, and the hash suffix defends against the macro-bypass case (a
plain function literally named `cuda_oxide_kernel_foo`, no macro). Both
defenses are needed: the macro guard catches honest mistakes early at
the source-code level; the hash suffix makes the actually-mangled name
collision-resistant against any code path that bypasses the macro.
