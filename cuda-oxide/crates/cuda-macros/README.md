# cuda-macros

Procedural macros for writing CUDA kernels in Rust. Provides `#[cuda_module]`
for typed embedded-module loading, `#[kernel]` for GPU entry points,
`#[device]`, `#[launch_bounds]`, `#[cluster_launch]`, `#[cooperative_launch]`,
`#[launch_contract]`, `gpu_printf!`, `ptx_asm!`, and the lower-level `cuda_launch!` / `cuda_launch_async!`
escape hatches. Both lower-level launch macros are caller-unsafe: prefer
`#[cuda_module]` with a launch contract unless you are launching a module
loaded at runtime by name.

## Attributes

### `#[kernel]` -- GPU Kernel Entry Point

Marks a function as a CUDA kernel. Generates:
1. An entry point renamed into the reserved `cuda_oxide_kernel_<hash>_<name>` namespace
   (with `#[no_mangle]`) so the codegen backend can find it. The hash makes the prefix
   unguessable for user code; see `cuda-oxide/crates/reserved-oxide-symbols/` for the contract.
2. Host lookup metadata used by typed launch APIs.
3. For a generic kernel, a readable `<name>_ptx_name::<...>()` helper. Generated
   marker types are internal plumbing and should not be named by application code.

The `<name>_ptx_name` sibling is part of the generated API, so that name must
remain free beside a generic kernel. Calling it also retains that concrete
specialization in device output; it cannot return a name for an omitted entry.

> **Reserved names.** The macros refuse to compile any function whose name starts with
> `cuda_oxide_` -- that namespace is reserved for cuda-oxide-internal mangling. The check
> is enforced at expansion time so the error points at the offending source line.

```rust
use cuda_device::{kernel, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

**Generic kernels** work in two modes:

```rust
// Mode 1: call-site specialization (PTX name from the function-item TypeId)
#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) { ... }
// Raw launch: unsafe { module.scale::<f32>(&stream, config, factor, &input, &mut out)? }
// Inspect its generated entry name: scale_ptx_name::<f32>()

// Mode 2: explicit instantiation list
#[kernel(f32, i32)]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) { ... }
// Generates named entry points: scale_f32, scale_i32
```

The legacy explicit list supports exactly one type parameter. Use Mode 1 for
const parameters, lifetimes, or mixed type/const kernels.

### `#[device]` -- Device Helper Functions and Externs

Device functions run on GPU but are not entry points. Works on both regular functions and `extern "C"` blocks:

```rust
#[device]
pub fn magnitude(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

// Extern device functions (e.g. from libdevice or cuBLASDx)
#[device]
extern "C" {
    fn __nv_expf(x: f32) -> f32;
}
```

| Feature              | `#[kernel]`          | `#[device]`          |
|----------------------|----------------------|----------------------|
| Entry point          | Yes (PTX `.entry`)   | No (PTX `.func`)     |
| Can return values    | No (must be `()`)    | Yes                  |
| Callable from host   | Via `#[cuda_module]` | No                   |
| Callable from device | Yes                  | Yes                  |

### `#[launch_bounds(max_threads, min_blocks)]`

Occupancy hints for register allocation. It may appear before or after
`#[kernel]`; generic kernels forward its compiler marker to every generated
entry. The first value is a maximum, not an exact block size.

```rust
#[kernel]
#[launch_bounds(256, 2)]  // max 256 threads, min 2 blocks per SM
pub fn optimized(out: DisjointSlice<f32>) { ... }
// PTX: .entry optimized .maxntid 256 .minnctapersm 2 { ... }
```

The values may come from a policy type. With a launch contract, the host checks
the concrete maximum for the selected policy before launching:

```rust
#[kernel]
#[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
#[launch_contract(domain = 1)]
pub fn transform<P: TransformPolicy>(out: DisjointSlice<f32>) { ... }

let prepared = module.prepare_transform::<SmallPolicy>(
    LaunchConfig1D::new(blocks, 64, 0),
)?;
```

### `#[launch_contract(...)]`

Declares a kernel's launch-time correctness assumptions. Inside
`#[cuda_module]`, this changes that kernel's generated API from a raw
`LaunchConfig` to a prepared, specialization-branded launch:

```rust
#[kernel]
#[launch_bounds(256)]
#[launch_contract(
    domain = 1,
    block = (256, 1, 1),
    dynamic_shared_range = (1024, 49152),
    dynamic_shared_alignment = 128,
    min_compute_capability = (9, 0),
)]
pub fn reduce(input: &[f32], mut out: DisjointSlice<f32>) { /* ... */ }

let prepared = module.prepare_reduce(LaunchConfig1D::new(blocks, 256, 8192))?;
module.reduce(&stream, &prepared, &input, &mut out)?;
```

Kernel configuration attributes may appear before or after `#[kernel]`. If an
attribute expands first, generic kernel generation forwards its exact internal
marker to the exported entry.

```text
prepare_reduce: dimensions + live CUDA limits -> PreparedLaunch<reduce>
reduce:         PreparedLaunch<reduce>         -> enqueue
reduce_unchecked: raw LaunchConfig             -> unsafe expert path
```

`block` is exact. If it is omitted, `#[launch_bounds]` supplies the compiled
maximum total threads per block. For example, a limit of 256 accepts both
`(256, 1, 1)` and `(16, 16, 1)`.

An exact `block` also reaches the device compiler, which emits
`.reqntid x, y, z` in place of `.maxntid`. The CUDA driver enforces that per
axis, so a launch whose block differs is refused even when it never passed
through preparation. Preparation and the driver therefore check the same
requirement on independent paths, and the raw `_unchecked` path is covered by
the second. The two directives cannot coexist on one entry, so a kernel
declaring both `#[launch_bounds]` and an exact `block` emits `.reqntid` alone.
The occupancy hint `.minnctapersm` is unaffected.

Dynamic shared memory is either exact
(`dynamic_shared = BYTES`) or an inclusive range. The byte extent is an author
promise; the alignment is a compiler-visible minimum and is merged with any
higher `DynamicSharedArray<T, ALIGN>` request in the body or a reachable local
helper. Prelinked external helpers retain their separately compiled alignment.
Launch-contract fields remain integer literals. The maximum supplied by
`#[launch_bounds]` may be a generic const expression and is evaluated for each
kernel specialization.

`domain` is deliberately explicit because calls through device helpers defeat
AST inference. `#[cuda_module]` adds a sealed trait bound to the complete
`DisjointSlice` parameter type, so Rust resolves aliases before checking the
rank:

```text
type Tile = Index2D<64>;  DisjointSlice<_, Tile>    + domain 2 -> accepted
type Tile = Index1D;      DisjointSlice<_, Tile>    + domain 2 -> type error
local struct named DisjointSlice                     -> type error
```

Mutable slice parameters, incompatible cluster axes, and blocks above the
emitted launch bounds are also rejected.

Opted-in kernels gain generated sync, borrowed-async, and owned-async safe
methods that consume the same prepared proof. Their raw paths use an
`_unchecked` suffix. Uncontracted methods keep their existing names, but are
also unsafe because no proof ties their raw configuration to the kernel.

All loaders for a module containing a contracted kernel are unsafe. Bundles are
currently identified at package granularity, so `load()` cannot distinguish a
library artifact from a same-package binary artifact by name alone. The caller
must prove that `load`, `load_async`, `load_named`, `load_async_named`, or
`from_module` binds code with the ABI and resource semantics declared by the
module. Generic loading also merges all PTX bundles, so those specializations
must match and have no conflicting entry definitions. Preparation and launch
are safe after this one-time binding.

Cluster and cooperative requirements are each checked against the live device,
and they may be combined. For a clustered cooperative kernel, preparation first
proves the cluster shape with `cuOccupancyMaxActiveClusters`, then requires the
whole grid to fit in that concurrent cluster capacity so a cooperative
`grid::sync()` cannot hang on non-resident blocks. Oversized grids fail
preparation with `LaunchContractError::CooperativeGridTooLarge`.

### `#[cluster_launch(x, y, z)]`

Compile-time thread block cluster dimensions (Hopper+). It may appear before or
after `#[kernel]`.

```rust
#[kernel]
#[cluster_launch(4, 1, 1)]  // 4 blocks per cluster
pub fn cluster_kernel(out: DisjointSlice<u32>) { ... }
// PTX: .entry cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
```

### `#[cooperative_launch]`

Marks a kernel for cooperative launch, the precondition for grid-wide
barriers (`cuda_device::grid::sync()`). It may appear before or after
`#[kernel]`. Unlike `#[cluster_launch]` this changes nothing in the PTX:
`#[cuda_module]` records the setting before nested attributes expand and
routes every generated launch method through `cuLaunchKernelEx` with
`CU_LAUNCH_ATTRIBUTE_COOPERATIVE` set. May be
combined with `#[cluster_launch]`; both attributes then go into the same
`cuLaunchKernelEx` call.

```rust
#[kernel]
#[cooperative_launch]
pub fn grid_sync_kernel(mut out: DisjointSlice<u32>) {
    // ... per-block work ...
    grid::sync();
    // ... grid-wide post-barrier work ...
}
```

### `#[convergent]`, `#[pure]`, `#[readonly]`

Semantic markers for the codegen backend (pass-through -- no code transformation).

### `#[constant]` -- Constant-Memory Statics

Marks a module-scope `static` as a CUDA constant-memory global. The static must
be typed `ConstantMemory<T>`. Its `UnsafeCell<T>` interior stops the compiler
from constant-folding the initializer, so the writes the host makes with
`cuMemcpyHtoD` stay visible to kernels.

```rust
#[cuda_module]
mod kernels {
    #[constant]
    static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;
}
```

The macro attaches a reserved `export_name` so the PTX symbol is resolvable via
`cuModuleGetGlobal`, and the host-side `#[cuda_module]` expansion generates the
setters:

- `module.set_<name>(&stream, &value)` -- stream-ordered async write, which
  orders correctly against surrounding launches. Prefer this one.
- `module.set_<name>_blocking(&value)` -- synchronous `cuMemcpyHtoD`, for
  one-shot initialization with no stream in scope.

Three restrictions worth knowing before you reach for it:

- The initializer must be all-zeros, which in practice means
  `ConstantMemory::UNINIT`. Non-zero initializers are not honored yet, so
  populate from the host before any kernel reads.
- The attribute must appear inside a `#[cuda_module]`. Anywhere else it
  silently produces an unreachable symbol with no setter.
- The identifier must not start with the reserved `cuda_oxide_` prefix.

## `#[cuda_module]` -- Typed Embedded Module Loading

Wrap an inline module containing `#[kernel]` functions to generate a typed
loader and per-kernel launch methods:

```rust
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
}

let module = kernels::load(&ctx)?;
// SAFETY: the raw geometry is 1-D and matches vecadd's indexing and resources.
unsafe {
    module.vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?;
}
```

When `cuda-host` is built with its `async` feature, async code can load the
same embedded module from a `cuda-async` device context:

```rust
let module = kernels::load_async(0)?;
let launch = unsafe {
    // SAFETY: the raw geometry is 1-D and matches vecadd's assumptions.
    module.vecadd_async(LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?
};
launch.sync()?;
```

Borrowed async methods return `AsyncKernelLaunch<'_>` and tie the lazy operation
to referenced buffers and borrowed scalar arguments. Owned async methods take
device buffers by value and return them after completion:

```rust
let launch = unsafe {
    // SAFETY: the raw geometry is 1-D and matches vecadd's assumptions.
    module.vecadd_async_owned(LaunchConfig::for_num_elems(N as u32), a_dev, b_dev, c_dev)?
};
let (a_dev, b_dev, c_dev) = launch.await?;
```

## `cuda_launch!` -- Unsafe Lower-Level Synchronous Kernel Launch

For kernels embedded in your own crate, use `#[cuda_module]` above: it reads
the kernel signatures at compile time and generates typed launch methods.
`cuda_launch!` is the unsafe escape hatch for the remaining case, modules
loaded at runtime by name, where no compile-time signature exists to check.

The macro verifies neither the argument list nor the kernel's launch contract.
The caller promises that argument count, order, and types match the actual
signature, pointer arguments are device-accessible, and the geometry/resources
satisfy the kernel's indexing and synchronization assumptions. A mismatch can
cause undefined behavior. Every use must therefore sit inside an `unsafe { }`
block:

```rust
// SAFETY: argument count, order, and types match vecadd's signature;
// all three buffers are live device allocations.
unsafe {
    cuda_launch! {
        kernel: vecadd,                                  // or scale::<f32> for generics
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        cluster_dim: (4, 1, 1),                          // optional, uses launch_kernel_ex
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }
}
```

### Argument Forms

| Syntax                | Kernel Parameter    | Marshalling                         |
|-----------------------|---------------------|-------------------------------------|
| `expr`                | `T` (scalar)        | `&mut value` as `*mut c_void`       |
| `slice(buf)`          | `&[T]`              | Device pointer + length (two args)  |
| `slice_mut(buf)`      | `DisjointSlice<T>`  | Device pointer + length (two args)  |
| `move \|..\| body`    | Closure `F`         | Whole closure environment by value  |
| `\|..\| body`         | Closure `F`         | Whole closure environment by value  |

### PTX Name Resolution

| Kernel Kind   | PTX Name                                                |
|:--------------|:--------------------------------------------------------|
| Non-generic   | Original function name (`vecadd`)                       |
| Generic       | `{name}_TID_{hex32}` (fixed length regardless of arity) |
| Closure-only  | Same as Generic — closure type is in the function item |

`{hex32}` is rustc's stable 128-bit type-id hash for the concrete generated
kernel function-item type, rendered as 32 lowercase hex chars. Its `FnDef`
contains the function identity and every ordered type and const argument. The
backend hashes `Instance::ty`; the host hashes
`&kernel_entry::<T, N>` through `cuda_host::type_id_u128_of_val`. Both use the
same region-erasing stable-hash pipeline, so lifetimes do not create variants
and the on-wire name stays a fixed `base.len() + 37` characters regardless of
generic arity.

The suffix is a build-time rendezvous key, not a permanent ABI. Rebuild host
code and PTX together when the pinned rustc toolchain or crate graph changes.

For generics, the macro forces monomorphization with a volatile pointer
trick so the kernel appears in the codegen unit even without a host-side
call.

## `cuda_launch_async!` -- Lower-Level Async Kernel Launch

Returns an immutable `AsyncKernelLaunch` implementing `DeviceOperation` for
`cuda-async` scheduling. The macro first builds inert launch data, then crosses
the same caller-unsafe raw finalization boundary as `cuda_launch!`. It accepts
the same argument forms, but has no `stream:` or `cluster_dim:` fields.

```rust
// SAFETY: ABI, lifetimes, geometry, and resources match vecadd.
let op = unsafe {
    cuda_launch_async! {
        kernel: vecadd,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }
};
```

This is a lower-level API. Prefer `#[cuda_module]`'s borrowed async methods for
stack-local use and owned async methods for spawned tasks:

```text
raw pointer async:
  op stores only a device address
  owner can be dropped before op runs

typed borrowed async:
  op borrows buffers until completion

typed owned async:
  op owns buffers and returns them after completion
```

## `gpu_printf!` -- Device-Side Printf

Compiles to CUDA's `vprintf` with C vararg promotion rules. Format string must use C-style specifiers.

```rust
gpu_printf!("thread %d: val = %f\n", tid as i32, val as f64);
```

## `ptx_asm!` -- Inline PTX

Supports CUDA inline PTX with `%0` operands, `in`, `out`, and `inout`
operands, CUDA register constraints `"h"`, `"r"`, `"l"`, `"q"`, `"f"`, and
`"d"`, immediate integer constraint `"n"`, and compile-time string constraint
`"C"`. The template follows CUDA's `%` convention (`%0` operands, `%%laneid`
literal registers), and `$` labels can be written normally. For the source
syntax, see CUDA's
[Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
reference.

By default, snippets are treated as side-effecting and stay inside their current
control flow. For snippets that only read explicit operands and write the
explicit outputs, add `options(register_only)`.

Use `options(register_only, may_diverge)` only for pure snippets that are safe
to move across divergent control flow. **Never** use it for `.sync` instructions,
collectives, or any snippet whose participating lanes matter; the optimizer may
move those snippets out of branch control flow.

```rust
let y: u32;
unsafe {
    ptx_asm!(
        "add.u32 %0, %1, %1;",
        out("=r") y,
        in("r") x,
        options(register_only),
    );
}
```

The surface supports up to 16 output operands across `out` and `inout`, up to
16 explicit `in` operands, and `clobber("memory")`. Every `out` constraint must
be `=`-prefixed, such as `"=r"`. Every `inout` constraint must be `+`-prefixed,
such as `"+r"`; its current value initializes the PTX output register and the
final register value is written back to the same Rust place. All `out` and
`inout` operands must appear before explicit `in` operands.

With two or more output operands, including any mixture of `out` and `inout`,
the marker returns a tuple under the hood and the macro writes its elements
back to the output places in declaration order:

```rust
let sum: u32;
let prod: u32;
unsafe {
    ptx_asm!(
        "add.u32 %0, %2, %3; mul.lo.u32 %1, %2, %3;",
        out("=r") sum,
        out("=r") prod,
        in("r") x,
        in("r") y,
        options(register_only),
    );
}
```

Read-write operands use `inout`:

```rust
let mut accumulator = initial;
unsafe {
    ptx_asm!(
        "add.u32 %0, %0, %1;",
        inout("+r") accumulator,
        in("r") increment,
        options(register_only),
    );
}
```

```rust
const MUL_MODE: &[u8; 6] = b".wide\0";

let product: u64;

unsafe {
    ptx_asm!(
        "mul%1.u32 %0, %2, %3;",
        out("=l") product,
        in("C") MUL_MODE,
        in("r") lhs,
        in("r") rhs,
        options(register_only),
    );
}
```

`in("C")` requires a compile-time byte string containing valid UTF-8. The
referenced operand is substituted directly into the PTX template and does not
become a runtime LLVM inline-assembly operand. One trailing zero byte is
removed when present. Constraint modifiers such as `"=C"` and `"+C"` are not
supported. `"C"` operands count toward the 16-input limit.

`options(register_only)` requires an `out` or `inout` operand and cannot be
combined with clobbers. `options(may_diverge)` must be paired with
`register_only`. More than 16 output operands are not implemented yet.

## Source Layout

```text
src/
├── lib.rs         # All proc-macro definitions (kernel, device, launch, etc.)
├── printf.rs      # gpu_printf! implementation
└── ptx_asm.rs     # ptx_asm! implementation
```

## Further Reading

- [cuda-device](../cuda-device/) -- re-exports these macros for convenience
- [cuda-host](../cuda-host/) -- `CudaKernel` / `GenericCudaKernel` traits used by generated code
- [cuda-core](../../../cuda-core/) -- `launch_kernel` / `launch_kernel_ex` called by generated code
