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
provider. Read it through the registration macro so Cargo tracks native inputs:

```rust,ignore
fn registration() -> &'static str {
    cuda_device::library::registration!("MY_LIBRARY_REGISTRATION")
}
```

The macro expands to the same constant string. The provider executable must
recognize its protocol arguments before its ordinary Cargo build-script behavior.

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

## Checked block helpers

`cuda_device::library::checked_block_call_v1` derives its block shape from the
kernel's exact launch contract and checks supported participation and uniform
memory/control arguments. It remains unsafe for the memory, native ABI and
scratch reuse obligations described below.

The same generic helper and marker site may serve kernels with different exact
block shapes. The compiler emits helper bodies for each root and routes their
calls to the corresponding native specialization and scratch plan, including
through intermediate helpers. No block-size type argument or inlining promise
is required. Device statics keep their original shared identity. Helpers with
fixed external symbols need an explicit root binding and are diagnosed by this
route; unsupported indirect checked calls retain their existing diagnostics.

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
avoids a second read of a mutable LTO file during finalization. The registration
macro tracks native inputs before rustc emits dep-info;
manual registration getters must arrange equivalent dependency tracking.

## Synchronous scalar callbacks

`call_with_callback_v1` accepts the ordinary registration and runtime tuple,
followed by a thunk function item and an immutable `&Environment`. Its generic
parameters are `T, Registration, Args, Environment, Thunk, BLOCK_THREADS,
ITEMS_PER_THREAD`. The initial callback ABI supports `T = i32`:

```rust,ignore
unsafe extern "C" fn thunk<F: Fn(i32, i32) -> i32 + Copy>(
    environment: *const F, left: i32, right: i32,
) -> i32 {
    unsafe { (&*environment)(left, right) }
}

unsafe {
    cuda_device::library::call_with_callback_v1::<i32, _, _, F, _, B, ITEMS>(
        registration, arguments, thunk::<F>, &operation,
    );
}
```

The compiler requires the actual function-item type with this exact non-unwinding
C signature; runtime function pointers and closure-valued thunk witnesses are
rejected. The environment may be a function item or an `Fn` closure capturing
fixed-width numeric scalars by value. Reference, pointer and aggregate captures
are rejected. Rust owns the layout and keeps the immutable environment live for
the entire synchronous call. Native code only carries its opaque address back to
the thunk; it must not inspect, mutate, retain, free or publish that address.

The compiler discovers and retains the selected thunk even when no Rust call
invokes it directly. Its definition joins the kernel roots for optimization and
linking. Callback bodies may call ordinary scalar Rust helpers. CUDA resource
operations/types, foreign or indirect calls, inline assembly and unqualified
compiler intrinsics are rejected transitively. Resource-bearing callbacks need
an explicit root/storage binding and are outside this version. This raw marker
also requires block-uniform callable specialization and capture values; it does
not prove them, and lane-private environment addresses need not be equal.

Callback calls use a separate batch invoked with `--oxide-library-request-v2`.
The registration and input-query protocols stay at version one. The request and
response JSON use `version: 2`, and the response prefix is `OXDLIB02`; all existing
length, layout and ordering checks apply. Each request adds `callback`:

```json
{"symbol":"actual_emitted_thunk", "arguments":[
  {"kind":"const_pointer","scalar":{"kind":"unsigned","bits":8}},
  {"kind":"value","scalar":{"kind":"signed","bits":32}},
  {"kind":"value","scalar":{"kind":"signed","bits":32}}
], "result":{"kind":"signed","bits":32}}
```

The symbol is the compiler's final emitted identity. The byte-pointer spelling
is opaque and uses generic address space zero; it conveys no closure layout.
The native operation receives scratch at index two and the environment pointer
as its last argument. Its response `arguments` describes only the explicit Rust
tuple, excluding both compiler-inserted pointers. Native request deduplication
includes the callback descriptor, while runtime capture values do not affect
specialization identity. Rust body changes remain ordinary Rust build inputs.

## Native input dependencies

The registration macro tracks its environment variable, the provider executable,
CUDA_HOME/CUDA_PATH selection, and the device compilation settings. During CUBIN
materialization it invokes the provider with `--oxide-library-inputs-v1`. Ordinary
host builds and rustdoc do not query toolkit inputs or require installed CUDA.

The input query reads stdin through EOF. Its JSON request is
`{"version":1,"cuda_root":"/selected/toolkit","target":"compute_120"}`.
Stdout is one JSON response with `version: 1`, `paths` (absolute file or directory
paths) and `environment` (additional environment variable names). Requests and
responses are limited to 1 MiB. Diagnostics go to stderr; failures exit nonzero.
Unknown fields, invalid versions, relative/missing paths and invalid environment
names are errors. The query declares dependencies without compiling native code.

The library owns this inventory, including sources, headers, compiler tools and
any environment affecting their selection or behavior. Directories are tracked
recursively so adding a header also causes recompilation. The macro records
these dependencies in the library crate; Cargo propagates its rebuild to
consumers and concrete generic kernel instantiations. The compiler has no
library-specific header layout or native cache.

Cargo's timestamp freshness rules apply. Edits that preserve timestamps can
remain undetected; clean/rebuild the affected crates after such external changes.
The `native_library` example tracks its provider, CUDA tools/headers, host compiler
driver and relevant environment. It assumes the host compiler's installed system
headers and libraries remain unchanged during a build; its declarations are not
a general inventory of an arbitrary C++ toolchain.

## Debug information

Providers choose the debug information in their LTO-IR; the compiler preserves
it during linking. With CUDA 13.3, Rust `--lineinfo` and native nvcc
`--generate-line-info` can be combined, retaining both source line tables.
Rust `--device-debug` also works with a provider built without debug information,
as exercised by the `native_library` full-debug smoke route.

CUDA 13.3 rejects nvcc `-G` with `-dlto`. Linking Rust full debug with native line
information also fails in nvJitLink when it sees more than one debug compilation
unit. Select a supported combination explicitly; provider metadata is not
silently stripped to make these combinations link.

## Resources and unsafe obligations

Every reachable library call contributes its exact block shape and required
scratch to its kernel root. Conflicting shapes and incompatible declared launch
contracts or maximum bounds are errors. The compiled artifact carries the
resulting facts to launch preparation; `dynamic_shared = inferred` with
`LaunchConfig3D::auto_shared` uses the compiled byte count.

This version aliases all library scratch views at the dynamic shared base,
using the maximum required size and alignment. The caller must complete each
use before conflicting reuse. Mixing native dynamic shared views
with library calls is rejected. No barrier, allocation token or borrow-based
completion is supplied by the marker.

The library wrapper must enforce pointer validity, alignment, capacity,
initialization, nonaliasing, collective participation and resource reuse for
its native implementation. Convergent declarations preserve optimization
constraints; they do not prove participation or make the wrapper safe. The
marker is an unsafe compiler boundary, not a safe collective operation.

The provider reads the complete request through stdin EOF before writing its
response. Reserve stdout for the binary response frame and use stderr for
diagnostics. The compiler waits for provider completion; provider execution has
the same trust boundary as a dependency's Cargo build script.
