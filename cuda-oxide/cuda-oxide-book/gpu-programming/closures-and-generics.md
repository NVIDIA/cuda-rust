# Closures and Generics

Rust's zero-cost abstractions -- generics, closures, and trait bounds -- work on
the GPU. This is one of cuda-oxide's most distinctive capabilities: you can write
a single generic kernel that operates on any numeric type, or pass a closure from
the host to customize GPU behavior, all without runtime overhead.

## Generic kernels

A kernel can be generic over types, const values, and trait bounds, just like
any Rust function. The compiler monomorphizes each specialization into a
separate PTX entry point:

```rust
use core::ops::Mul;
use cuda_device::{DisjointSlice, kernel, thread};

#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(
    factor: T,
    input: &[T],
    mut out: DisjointSlice<T>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = input[i] * factor;
    }
}
```

### PTX naming

Each monomorphization produces a distinct PTX entry point. Non-generic kernels
keep their plain function name. Generic kernels (including closure-generic
kernels) get a `_TID_<hex32>` suffix where `<hex32>` is rustc's stable
type-id hash of the concrete generated kernel function item, rendered as 32
lowercase hex characters:

| Instantiation             | PTX entry point name |
|:--------------------------|:---------------------|
| `vecadd` (non-generic)    | `vecadd`             |
| `scale::<f32>`            | `scale_TID_<hex32>`  |
| `scale::<MyType>`         | `scale_TID_<hex32>`  |
| `tile::<4>`               | `tile_TID_<hex32>`   |
| `tile::<8>`               | `tile_TID_<hex32>`   |
| `map::<f32, _>` (closure) | `map_TID_<hex32>`    |

Both the host launcher and the device backend ask the pinned rustc toolchain
for the same `FnDef` hash within one unified build, so the strings match
byte-for-byte. The function item contains its definition plus every ordered
type and const argument. The on-wire name therefore remains fixed-length
regardless of generic arity. Borrow lifetimes are erased before hashing, so
they do not create duplicate GPU code.

Treat the suffix as a build artifact, not a permanent ABI: host code and PTX
must be built together with cuda-oxide's pinned rustc toolchain.

### Launching generic kernels

When launching, specify the type parameter on the generated typed method. That
forces the concrete instantiation and lets the loader look up the matching PTX
entry point:

```rust
use cuda_core::simt::LaunchConfig;

// SAFETY: config is 1D and both buffers cover every launched index.
unsafe {
    module.scale::<f32>(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        2.0f32,
        &input_dev,
        &mut output_dev,
    )
}
.expect("Launch failed");
```

The generated method forces monomorphization of `scale::<f32>` so the
instantiation appears in the compiled PTX even though it is never called
directly on the CPU.

### Compile-time kernel policies

A policy type groups tuning choices for one generic kernel:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::LaunchConfig1D;
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};

trait TransformPolicy {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

enum Small {}
impl TransformPolicy for Small {
    const MAX_THREADS: u32 = 64;
    const MIN_BLOCKS: u32 = 2;
    const UNROLL: u32 = 2;
}

enum Wide {}
impl TransformPolicy for Wide {
    const MAX_THREADS: u32 = 256;
    const MIN_BLOCKS: u32 = 1;
    const UNROLL: u32 = 4;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(P::MAX_THREADS, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub fn transform<P: TransformPolicy>(mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(value) = output.get_mut(index) {
            let mut step = 0;
            #[unroll(P::UNROLL)]
            while step < 8 {
                *value = value.rotate_left(1);
                step += 1;
            }
        }
    }
}

let launch = module
    .prepare_transform::<Small>(LaunchConfig1D::new(1, 64, 0))
    .expect("valid Small launch");
module
    .transform::<Small>(&stream, &launch, &mut output)
    .expect("launch Small kernel");

let wide_launch = module
    .prepare_transform::<Wide>(LaunchConfig1D::new(1, 256, 0))
    .expect("valid Wide launch");
module
    .transform::<Wide>(&stream, &wide_launch, &mut output)
    .expect("launch Wide kernel");
```

`Small` and `Wide` produce separate PTX kernels. The policy is not a kernel
argument, and the GPU does not choose a policy at runtime.

A policy can also carry metadata-only descriptions:

```rust
use cuda_device::config::{Atom, AtomKind, Block, Global, RowMajor, Shape1, Thread, Tile};

type BlockTile = Tile<Shape1<1024>, RowMajor, Global, Block>;

enum Rotate {}
impl AtomKind for Rotate {}
type ElementOp = Atom<Rotate, Shape1<1>, Thread>;
```

`BlockTile` describes 1,024 row-major global-memory elements handled by a
block. `ElementOp` names a one-element operation handled by a thread; a kernel
library decides what `Rotate` does. Neither type allocates memory, calculates
an address, or runs an operation.

`MAX_THREADS` is a maximum, not an exact block size. The prepared launch for
`Small` accepts at most 64 threads per block; `Wide` accepts at most 256. Use
`block = (x, y, z)` in `#[launch_contract]` only when the shape must be exact.
`MIN_BLOCKS` is a compiler occupancy hint, not a requested grid size.

`UNROLL = 4` groups four loop iterations. It does not stop the loop after four
iterations, and remaining iterations are still handled.

Generic policy expressions currently need Rust's `generic_const_exprs`
feature. Other named constants used in a contracted `launch_bounds` maximum
must be visible at module scope.

### Const-generic kernels

Const parameters work on kernel and device-function entry points:

```rust
#[kernel]
pub fn add_value<const VALUE: u32>(mut output: DisjointSlice<u32>) {
    let index = thread::index_1d();
    if let Some(element) = output.get_mut(index) {
        *element += VALUE;
    }
}
```

```text
add_value::<4> -> add_value_TID_<hash A>
add_value::<8> -> add_value_TID_<hash B>
```

Use ordinary Rust turbofish syntax on the generated launch method:

```rust
// SAFETY: config matches the kernels' 1D indexing and output bounds.
unsafe {
    module.add_value::<4>(&stream, config, &mut output)?;
    module.add_value::<{ Config::VALUE }>(&stream, config, &mut output)?;
}
```

The older `#[kernel(f32, i32)]` convenience form only supports one type
parameter. For const or mixed generics, use bare `#[kernel]` and specialize at
the launch site.

## Host closures as kernel arguments

cuda-oxide supports passing closures from the host to the GPU. This enables
powerful `map`-style patterns where the kernel's behavior is parameterized by
a function:

```rust
#[kernel]
pub fn map<F: Fn(i32) -> i32>(f: F, input: &[i32], mut out: DisjointSlice<i32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = f(input[i]);
    }
}
```

Launch with a closure:

```rust
let factor = 3i32;
// SAFETY: config is 1D and both buffers cover every launched index.
unsafe {
    module.map::<_>(&stream, config, move |x| x * factor, &input_dev, &mut output_dev)
}
.expect("Launch failed");
```

### How closure arguments travel

The closure passes through the launch as one value -- not as a list of
captured fields. The launcher pushes a single driver argument (the whole
closure struct, captures and all), and the kernel receives it as one
byval `.param`:

```text
host          factor = 3i32; cl = move |x| x * factor
              push one driver arg ─► closure struct { factor: i32 }

GPU kernel    .entry map_TID_<hex>(
                .param .align 4 .b8 f[4],    ; one byval closure
                .param .u64 input_ptr,        ; slice still (ptr, len)
                .param .u64 input_len,
                ...
              )
```

Slices keep their `(ptr, len)` flattening because that shape is shared by
the host launch helpers and the PTX entry-point layout. Only aggregate-
by-value parameters (closures and user structs passed by value) land as
one byval at the kernel boundary.

A closure with no captures is a zero-sized type -- the backend drops the
`.param` entirely, and the host launcher knows to skip it so the packet
stays aligned.

### PTX naming for closures

A closure-generic kernel gets the same `_TID_<hex32>` suffix as any other
generic kernel. The closure's anonymous type is part of the concrete function
item, so two distinct closure literals -- even ones with the
same `Fn` signature -- produce two distinct entry points:

| Closure                                | PTX entry point   |
|:---------------------------------------|:------------------|
| `move \|x\| x * factor` (one capture)  | `map_TID_<hex_a>` |
| `move \|x\| x + offset` (one capture)  | `map_TID_<hex_b>` |

## Move vs reference closures

The `move` keyword determines how captures are transferred to the GPU:

### Move closures (recommended default)

```rust
let factor = 3i32;
move |x| x * factor   // `factor` is copied into the closure struct
```

- The closure struct holds the capture by value (`{ factor: i32 }`).
- The kernel reads `factor` as a regular field of the byval closure.
- The host variable can be dropped after launch.
- Works on all systems -- no special hardware support needed.

### Reference closures (HMM)

```rust
let factor = 3i32;
|x| x * factor   // closure captures &factor
```

- The closure struct contains a **host pointer** to `factor`
  (`{ factor: &i32 }`).
- The whole closure still travels as one byval parameter; the kernel
  deref's that host pointer through **Hardware-Managed Memory (HMM)**,
  which migrates the host page on access.
- The host variable **must remain alive** until the kernel completes.
- Requires HMM support (Turing+ GPU, Linux 6.1.24+, CUDA 12.2+).

### When to use which

| Scenario                                       | Use                                                               |
|:-----------------------------------------------|:------------------------------------------------------------------|
| Small scalar captures (numbers, booleans)      | `move` (zero-copy overhead)                                       |
| Large struct captures                          | `move` if the kernel reads it many times; HMM if rarely accessed  |
| Prototyping                                    | Either works; `move` is more portable                             |
| Shared mutable state between host and device   | Reference (HMM) -- but beware synchronization                     |

:::{tip}
When in doubt, use `move` closures. They are simpler to reason about, work
everywhere, and avoid the synchronization hazards of shared host/device memory.
:::

## In-kernel closures

Closures defined and called entirely within device code work with normal Rust
semantics -- no host/device ABI is involved because everything is already on
the GPU:

```rust
#[kernel]
pub fn apply_transform(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();

    let transform = |x: f32| -> f32 {
        let clamped = if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };
        clamped * clamped
    };

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = transform(input[idx.get()]);
    }
}
```

In-kernel closures are inlined by the compiler and have zero overhead. They are
useful for factoring logic within a kernel without introducing a separate device
function.

## Cross-crate kernels

Kernels can be defined in a library crate and launched from a binary crate:

```rust
// In lib crate `my_kernels`:
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}
```

```rust
// In binary crate:
use my_kernels::kernels;

let module = kernels::load(&ctx)?;
// SAFETY: config is 1D and all three buffers cover every launched index.
unsafe { module.vecadd(&stream, config, &a, &b, &mut c) }
    .expect("Launch failed");
```

The compiler handles cross-crate kernel discovery through the marker traits
generated by `#[kernel]`. The typed module resolves the PTX name at compile time
and caches the loaded function handle.

:::{tip}
For generic cross-crate kernels, the monomorphization happens in the **calling**
crate (where the concrete type is known), so the PTX is generated as part of
the binary's compilation.
:::
