# Native library calls

`cuda_device::library::call_v1` is an unsafe integration point for library
authors. It specializes native device implementations from concrete Rust
calls, then supplies their LTO-IR to ordinary CUBIN materialization. The
compiler owns discovery and linking; the library owns its operation semantics,
native code, ABI and scratch requirements.

Use `cargo oxide build`, `run` or `test` with `--materialize-cubin`. The compiler
recognizes the actual marker definition, including calls reached through
generic dependency helpers. A same-named user function remains ordinary Rust.
The marker must be called directly; pass a wrapper helper rather than the
marker itself as a callable value.

The `native_library` example is an executable provider with no CUB dependency.
Its dependency helper is instantiated at one and four items per thread, and
two concrete kernel roots run through the embedded module loader:

```sh
cargo oxide run native_library --arch sm_120 --materialize-cubin
```

## Registration and call ABI

The marker's generic parameters are `T`, `Registration`, `Args`,
`const BLOCK_THREADS: usize` and `const ITEMS_PER_THREAD: usize`.
Both extents must fit positive `u32`. It accepts a registration function item
and one tuple of scalar/raw-pointer arguments; it returns unit.

The registration getter returns one constant UTF-8 JSON string. Its MIR must
contain one return block and one assignment of that constant to its result.
Closures, function pointers and computed getters are rejected. The compiler
reads the constant without running Rust device code. A Cargo build script can
embed the string with `cargo::rustc-env` and serve as the already-built native
provider. It must recognize the provider argument before its ordinary Cargo
build-script behavior.

Registration fields are:

| Field | Meaning |
| --- | --- |
| `version` | Integer `1` |
| `provider` | Nonempty library identifier |
| `executable` | Absolute path to the built host provider |
| `operation`, `algorithm` | Opaque recipe names interpreted by that provider |

The runtime tuple starts with a const or mutable memory pointer and a mutable
payload pointer. Remaining fields are supported fixed-width numeric values or
raw pointers. The provider declares the exact ordered types and pointer
mutability; the compiler compares them with the Rust tuple. It inserts a
generic `unsigned char*` scratch parameter at native argument index two.
Small signed and unsigned integer values retain their native extension rules.
The native function returns `void`.

## Host request and response

For each `(provider, executable)` pair, the compiler invokes the executable
once with `--oxide-library-request-v1`. Stdin is one JSON object with `version: 1`,
`target`, `cuda_root`, and an ordered `requests` array. Each request contains
`operation`, `algorithm`, `dtype`, `block: [B,1,1]`, and `items_per_thread`.
Repeated concrete requests are deduplicated across roots. Scalar descriptions
use `{"kind":"signed|unsigned|float", "bits":N}`; integer widths are
8/16/32/64 and float widths are 16/32/64. A provider may support a narrower set
and must diagnose unsupported requests.

Stdout contains only a binary response: eight bytes `OXDLIB01`, little-endian
u64 JSON-header length, little-endian u64 LTO-IR length, then the header and raw
LTO-IR. Diagnostics go to stderr; failure exits nonzero. Requests and response
headers are limited to 1 MiB, and the nonempty LTO-IR buffer to 256 MiB.

The header has `version: 1`, a 32-byte `batch_identity` array, and one
`specializations` entry per request, in the same order:

| Field | Meaning |
| --- | --- |
| `request_index` | Zero-based request position |
| `identity`, `symbol` | Unique 32-byte specialization identity and native link symbol |
| `arguments` | Ordered runtime tuple ABI, excluding the inserted scratch pointer |
| `cub_layout`, `scratch` | Native temporary-layout and required scratch `{size, alignment}` |
| `scratch_free` | Must agree with `scratch.size == 0` |
| `mutates_payload` | Semantic information; never permission to read an undefined value |

Argument descriptions use `kind: "const_pointer"`, `"mut_pointer"` or
`"value"`, with a `scalar` description. Layout fields must fit `u32`, and
alignment must be a positive power of two. Symbol names use letters, digits
and underscores and cannot start with a digit. Different batches must not
return the same symbol. Truncated/trailing frames, mismatched versions, duplicate
identities and ABI mismatches fail compilation.

The production provider is responsible for truthful layouts and identities
covering its sources, headers, recipes and toolchain. Returning owned bytes
avoids a second read of a mutable LTO file during finalization. Cargo/provider
invalidation must still arrange recompilation when those inputs change.

## Resources and unsafe obligations

Every reachable library call contributes its exact block shape and required
scratch to its kernel root. Conflicting shapes and incompatible declared launch
contracts or maximum bounds are errors. The compiled artifact carries the
resulting facts to launch preparation; `dynamic_shared = inferred` with
`LaunchConfig3D::auto_shared` uses the compiled byte count.

This version aliases all library scratch views at the dynamic shared base,
using the maximum required size and alignment. The caller must complete each
use before conflicting reuse. Mixing unplanned native dynamic shared views
with library calls is rejected. No barrier, allocation token or borrow-based
completion is supplied by the marker.

The library wrapper must enforce pointer validity, alignment, capacity,
initialization, nonaliasing, collective participation and resource reuse for
its native implementation. Convergent declarations preserve optimization
constraints; they do not prove participation or make the wrapper safe. The
marker is an unsafe compiler boundary, not a safe collective operation.
