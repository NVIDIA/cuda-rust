# Tile Libraries

A Tile kernel works on whole tiles and never names a thread. A SIMT function
works on one element from one thread. A **tile library** joins the two. It
packages Rust device functions so the Tile compiler can call them from inside
a cuTile kernel, once per element or once per warp, and link them into the
kernel's cubin. The Rust side is `#[tile_export]`, the packaging is
`cargo oxide tilelib`, and the caller is cuTile Python's `foreign_call` today,
with [cutile-rs](https://nvidia.github.io/cuda-rust/cutile/) to follow.

```text
#[tile_export(tile_shape = 128)]
pub fn relu(x: f32) -> f32 { .. }                     your Rust function
        |
        |  cargo oxide tilelib <crate> --arch sm_120
        v
<crate>.ltoir          device code     void relu_impl(float x, float *relu_out)
<crate>.iface.tilebc   declaration     @relu(tile<128xf32> in, tile<128xf32> out) -> tile<128xf32>
<crate>.impl.tilebc    implementation  @relu_impl implements @relu, one invocation per element
        |
        |  tileiras --preview-create-tile-lib       (CUDA 13.5 or newer)
        v
<crate>.tilelib
        |
        |  tp.foreign_call(symbol_name="relu", ...)   inside a cuTile kernel
        v
the Tile compiler links relu_impl into the kernel's cubin
```

:::{seealso}
[Kernels and Device Functions](../gpu-programming/kernels-and-device-functions.md)
covers `#[device]` and the LTOIR route to CUDA C++. A tile library is the same
route with the Tile compiler as the caller.
:::

```{figure} images/tilelib-direct-linkage.svg
:align: center
:width: 100%

The direct-linkage path. cuda-oxide ships the code (LTOIR) and two
descriptions (interface and implementation bytecode); tileiras packs them
into one `.tilelib`, and a Tile kernel calls the Rust function from inside
its body.
```

## Writing an export

Put `#[tile_export(tile_shape = N)]` on an ordinary function. `N` is the tile size
the caller will pass. The function stays as written, so the rest of your
device code keeps calling it. The attribute adds two items next to it: an
`extern "C"` implementation on the native SIMT ABI, named `<name>_impl` by
default, and a host-only description that the compiler writes out as JSON.

The
[`tilelib_relu`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/tilelib_relu)
example exports three functions, and this chapter follows them:

```rust
use cuda_device::{tile_export, warp};

/// Elementwise `max(x, 0)`.
#[tile_export(tile_shape = 128)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Scale a tile in place. The cuTile side knows it as `scale_in_place`.
#[tile_export(
    tile_shape = 128,
    name = "scale_in_place",
    symbol = "scale_in_place_impl"
)]
pub fn scale(x: &mut f32, factor: f32) {
    *x *= factor;
}

/// Sum over the 32 lanes of each warp; every lane gets the warp total.
#[tile_export(tile_shape = 128, sync = "warp")]
pub fn warp_sum(x: f32) -> f32 {
    let mut total = x;
    let mut offset = 16;
    while offset > 0 {
        total += warp::shuffle_xor_f32_sync(u32::MAX, total, offset);
        offset /= 2;
    }
    total
}
```

What each one becomes in the device artifact:

| cuTile name      | Rust                                    | Implementation the call resolves to                |
|:-----------------|:----------------------------------------|:---------------------------------------------------|
| `relu`           | `fn relu(x: f32) -> f32`                | `void relu_impl(float x, float *relu_out)`         |
| `scale_in_place` | `fn scale(x: &mut f32, factor: f32)`    | `void scale_in_place_impl(float *x, float factor)` |
| `warp_sum`       | `fn warp_sum(x: f32) -> f32`, warp sync | `void warp_sum_impl(float x, float *warp_sum_out)` |

`warp_sum` uses warp shuffles, which are legal because `sync = "warp"` makes
all 32 lanes of a warp call it together. The
[Warp-Level Programming](warp-level-programming.md) chapter explains the
shuffle pattern.

### Attribute arguments

| Argument                    | Meaning                                          | Default            |
|:----------------------------|:-------------------------------------------------|:-------------------|
| `tile_shape = N`            | 1-D tile size, an integer literal (required)     | none               |
| `name = "..."`              | declaration name the cuTile caller uses          | the function name  |
| `symbol = "..."`            | implementation symbol in the device artifact     | `<name>_impl`      |
| `sync = "none"` or `"warp"` | threads that must call the function together     | `"none"`           |
| `arch = ["sm_120", ...]`    | compute capabilities the implementation supports | every architecture |

`arch` is normally omitted, which matches every target. When given, names are
compared after stripping `_` and lowercasing, so `"sm_120"` also matches the
`sm_120a` the Tile compiler builds for.

### Parameters and the ABI

Every parameter is a plain identifier with one of four types. `T` is one of
`f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.
Tile IR has no unsigned integers, so a `u32` parameter is an `i32` tile on the
Tile side, same bits.

| Rust parameter        | Access  | Footprint | ABI parameter                         |
|:----------------------|:--------|:----------|:--------------------------------------|
| `x: T`                | `in`    | 1         | `T` by value                          |
| `x: &T`               | `in`    | 1         | `*const T`                            |
| `x: &[T; F]`          | `in`    | F         | `*const T`                            |
| `x: &mut T`           | `inout` | 1         | `*mut T`                              |
| `x: &mut [T; F]`      | `inout` | F         | `*mut T`                              |
| `-> T` (return value) | `out`   | 1         | `*mut T`, appended last as `<fn>_out` |

The footprint is the number of contiguous tile elements one invocation owns.
Only an `in` scalar travels by value. Everything else is a pointer to the
invocation's own elements, and the implementation returns nothing: a return
value is stored through the trailing `out` pointer. On the Tile side every
parameter, the return slot included, is one argument of the declaration, typed
as a tile of the export's size. Slices, raw pointers, tuples, structs, `bool`,
generics, `self`, and patterns in parameter position are rejected.

### Rules the tooling enforces

The attribute reports these at compile time:

- `tile_shape` is present, an integer literal, and positive. A constant or an
  expression is not accepted: a static library ships one concrete symbol per
  export.
- The export has an output: a return value, or at least one `&mut` parameter.
  A declaration's results are its `out` and `inout` arguments.
- The `out` and `inout` parameters form one contiguous run.
  `fn f(a: &mut f32, b: f32) -> f32` is `[inout, in, out]` and is rejected;
  write `fn f(b: f32, a: &mut f32) -> f32`, with the `in` parameters first.
- Every parameter of one export has the same footprint, because the
  fragments of one implementation share their invocation count. A
  `&mut [i32; 4]` next to a `&i32` would need 32 invocations for one argument
  and 128 for the other, so `fn window(acc: &mut [i32; 4], bias: &i32)` is
  rejected and `bias: &[i32; 4]` is the accepted form.
- `sync = "warp"` needs a tile size that is a multiple of 32 times the
  footprint, so every invocation fills whole warps.
- `name` and `symbol` are ASCII identifiers outside the reserved
  `cuda_oxide_` prefix.

`cargo oxide tilelib` repeats the same checks when it reads the descriptions
back, so a hand-edited description cannot slip past them.

## Building the library

```bash
cargo oxide tilelib tilelib_relu --arch sm_120 --print
```

`--arch` is required because the LTOIR inside a library is built for one
architecture. In a standalone project, such as one made with `cargo oxide new`,
the crate name can be omitted. The command runs `emit-ltoir`, then writes the
description modules next to the device artifact:

```text
tilelib_relu.ltoir               device code for sm_120
tilelib_relu.tileexports.json    one description per export, written by the compiler
tilelib_relu.iface.tilebc        declarations, Tile IR bytecode at the assembler's version
tilelib_relu.impl.tilebc         implementations, same version
tilelib_relu.tilelib.mlir        readable form of both; --print shows it
tilelib_relu.tilelib             the library, when tileiras can build it
```

Before writing the modules it checks the emitted NVVM IR: every implementation
symbol must be a `define void` with the parameter list the ABI table predicts,
for example `define void @relu_impl(float, ptr)`. The assembler never looks
inside the code payload, so a missing or misdeclared symbol would otherwise
surface only when a cuTile program is compiled against the library. The two
bytecode modules are built by the `oxide-tilebc` crate, cuda-oxide's own
writer for these two kinds of module; the Tile stack's crates are not
involved.
writes its own Tile IR with.

### Reading the dump

`--print` shows both modules in Tile IR syntax. The text is the same as
`tilelib_relu.tilelib.mlir`. For `tilelib_relu`:

```text
// tilelib_relu.iface.tilebc: the interface a cuTile program calls
cuda_tile.module @tilelib_relu {
  cuda_tile.preview$func extern @relu(%x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}, %relu_out: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>}) -> !cuda_tile.tile<128xf32>
  cuda_tile.preview$func extern @scale_in_place(%x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}, %factor: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}) -> !cuda_tile.tile<128xf32>
  cuda_tile.preview$func extern @warp_sum(%x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}, %warp_sum_out: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>}) -> !cuda_tile.tile<128xf32>
}
// tilelib_relu.impl.tilebc: the implementations the LTOIR defines
cuda_tile.module @tilelib_relu {
  cuda_tile.preview$extern_impl @relu_impl implements @relu(%x: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<():()>, fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}, %relu_out: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<(1):(1)>, fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
  cuda_tile.preview$extern_impl @scale_in_place_impl implements @scale_in_place(%x: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<(1):(1)>, fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>}, %factor: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<():()>, fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<in>})
  cuda_tile.preview$extern_impl @warp_sum_impl implements @warp_sum(%x: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<():()>, fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}, %warp_sum_out: !cuda_tile.preview$fragment<1xf32, local, mem_layout = #cuda_tile.preview$layout<(1):(1)>, fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>> {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
    synchronization = #cuda_tile.preview$synchronization<warp>
}
```

How to read it:

- The interface has one `preview$func extern` per export. Every Rust parameter
  is a `tile<128xf32>` argument with its access, and the results are the
  `out` and `inout` arguments in order. `relu` returns one tile, its
  `relu_out` slot.
- The implementation has one `preview$extern_impl` per export, naming the
  symbol it `implements`. Every argument is a `fragment`: the piece of the tile
  one invocation sees. `mem_layout` is the buffer's own layout: `():()` for a
  by-value scalar, `(1):(1)` for one element behind a pointer.
- `fragmap` maps `(invocations, participants, footprint)` onto the tile.
  `(128, 1, 1):(1, 0, 0)` is one invocation per element, 128 of them.
  `(4, 32, 1):(32, 1, 0)` is 4 invocations of 32 lanes each, which is what
  `synchronization = warp` requires.

The summary line names the artifacts and their sizes:

```text
✓ Tile library descriptions written: 3 export(s), Tile IR 13.5 with the preview tag
```

### The tileiras requirement

The descriptions are Tile IR public-preview content, which exists from
bytecode 13.5 on. An assembler reads preview content only at the version it
calls current, the newest numbered version its `--list-versions` prints, so
the command asks the resolved `tileiras` first and writes the descriptions at
that version: 13.5 for the CUDA 13.5 assembler, the newer version a newer
assembler names, and 13.5 when no assembler can read them yet.
`tileiras --preview-create-tile-lib` arrives with CUDA 13.5, the first
assembler whose `--list-versions` names 13.5. When the resolved `tileiras`
lists 13.5 or newer, the command assembles the interface, the
implementations, and the LTOIR into `tilelib_relu.tilelib` (or the path given
with `-o`) and prints the `tileiras` invocation it used.

On CUDA 13.4 the command stops after the descriptions and says why:

```text
Tile library skipped: /usr/local/cuda/bin/tileiras lists bytecode versions 13.1, 13.2, 13.3, 13.4; the descriptions need 13.5 or newer
The installed tileiras cannot create tile libraries: the descriptions are Tile IR public-preview content, which exists from bytecode 13.5 on and needs the CUDA 13.5 or newer assembler.
```

It then prints the `tileiras --preview-create-tile-lib` command that turns the
written files into the library on a newer toolkit, and exits 2 so a script
notices the missing library. `--library-optional` makes that outcome exit 0
when the LTOIR and the descriptions are all you need. Both outcomes print a
line a script can match: `Tile library written to` on success,
`Tile library skipped:` otherwise. The assembler is resolved from
`CUTILE_TILEIRAS_PATH` first, then from `PATH`, the configured CUDA toolkit
root, and the standard install locations. The
[installation guide](../getting-started/installation.md) covers toolkit
selection.

## Calling it from cuTile Python

A cuTile kernel reaches the export through `cuda.tile_preview.foreign_call`.
This is the kernel from the example's `python/call_from_cutile.py`:

```python
import cuda.tile as ct
from cuda import tile_preview as tp

TILE = 128
PROVIDER = tp.Tilelib(path="tilelib_relu.tilelib", version="0.1.0")


def relu_tile(x, y):
    """One block runs `relu` over one 128-float tile of `x` into `y`."""
    pid = ct.bid(0)
    tile = ct.load(x, index=(pid,), shape=(TILE,))
    # Every declaration argument needs an input tile, the `out` slot included.
    result_slot = ct.zeros((TILE,), dtype=ct.float32)
    (out,) = tp.foreign_call(
        tilelibs=(PROVIDER,),
        symbol_name="relu",
        constant_params=(),
        inputs=(tile, result_slot),
        output_types=((ct.float32, (TILE,)),),
    )
    ct.store(y, index=(pid,), tile=out)
```

Three things to get right:

- `symbol_name` is the declaration name, `relu`, not the implementation symbol
  `relu_impl`. The Tile compiler resolves the call to the implementation when
  it links the library.
- Every declaration argument takes a tile, the `out` ones included. `relu` has
  two arguments, so `inputs` has two tiles; a zero tile stands in for the
  result slot. A scalar needs a 0-d tile, and a Python literal is rejected.
- `output_types` lists the results, one `(dtype, shape)` per `out` or `inout`
  argument in declaration order. `relu` has one.

`foreign_call` reads the library as a compile-time constant, so the `Tilelib`
is a module global rather than a kernel argument, and its file has to exist
when the kernel is traced. The script's `--dry-run` compiles the kernel to
bytecode without `tileiras`, which works on CUDA 13.4 today:

```text
bytecode: 844 bytes, header 0d050100 (version 13.5, preview flag set)
PASS: the caller side compiles; the program calls `relu` through preview$call (dry run, no cubin)
```

The full run needs the `.tilelib` from a CUDA 13.5 or newer `tileiras` and
a cuTile Python that lists that assembler's current bytecode version, because
preview content is read only at that version.

## Calling it from Tile C++

The same toolkit that brings the assembler teaches nvcc `--enable-tile` and
`--tile-preview simt`, and a Tile C++ kernel calls the library through a
foreign declaration. The example's `cpp/tile_caller.cu` declares all three
exports; this is the `relu` one:

```text
namespace ct = cuda::tiles;
using tile_f32 = ct::tile<float, ct::shape<128>>;

// @relu(%x: tile<128xf32> in, %relu_out: tile<128xf32> out) -> tile<128xf32>
__experimental_foreign_decl__ __experimental_foreign_name__("relu")
    tile_f32 relu(tile_f32 x, [[cutile::experimental_inout]] tile_f32 relu_out);

[[cutile::hint(0, num_worker_warps_per_cta = 4)]] __tile_global__ void kernel(float* x, float* y) {
    auto xs = ct::partition_view { ct::tensor_span { x, ct::shape<128> {} }, ct::shape<128> {} };
    auto ys = ct::partition_view { ct::tensor_span { y, ct::shape<128> {} }, ct::shape<128> {} };
    ys.store(relu(xs.load(0), ys.load(0)), 0);
}
```

A parameter the library reads carries no attribute; one it writes carries
`[[cutile::experimental_inout]]` and comes back as the call's result, in
argument order. The declaration's name is the library's declaration name,
not the implementation symbol. Compile and run:

```bash
nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
     -std=c++20 -arch=sm_120 -o tile_caller cpp/tile_caller.cu
./tile_caller
```

```text
PASS relu: 128 elements through the tile library
PASS scale_in_place: 128 elements through the tile library
PASS warp_sum: 128 elements through the tile library, every lane holding its warp's total
SUCCESS: 3/3 tile exports called from a Tile C++ kernel
```

Because nvcc, tileiras and the Tile C++ front end come from one toolkit, the
program and the library always agree on the bytecode version. That is the
shortest path to a running Tile kernel on top of a Rust library; the cuTile
Python path needs a cuTile Python that lists the same assembler's version.

## What happens at the boundary

The Tile compiler never hands your function a whole tile. It splits each tile
argument into fragments, one per invocation, and calls the implementation once
per fragment. With `sync = "none"` a fragment is one element, so a 128-element
tile means 128 calls, each from one thread. With `sync = "warp"` an invocation
is a whole warp: 32 lanes call together, each with its own element, and warp
intrinsics are legal inside. The calling thread is a real thread of the Tile
kernel, and `threadIdx.x` reads the hardware register, so lane and warp
identity work as in any SIMT kernel (see the
[Execution Model](../gpu-programming/execution-model.md)).

The pointers you receive point at the invocation's own buffer and nothing
else. `&mut f32` is one element and `&mut [f32; 4]` is four contiguous
elements; the function may not reach past them, and nothing outside the
invocation aliases them during the call. The function cannot allocate shared
memory, so do not declare a `SharedArray` in an export: the shared memory of a
Tile kernel is the Tile compiler's to place, and this release has no way to
request a scratch buffer from it (see
[Shared Memory and Synchronization](shared-memory-and-synchronization.md)
for what shared memory means in a SIMT kernel). Keep exports to registers,
the parameters they receive, and warp traffic.

## Not in this release

- Parametric tile sizes. `tile_shape` is one integer literal, so a library ships one
  export per tile size.
- Warpgroup and CTA scope. `sync` accepts `"none"` and `"warp"` only.
- Scratch memory. An export cannot ask for a shared-memory buffer.
- `bf16` parameters. The attribute accepts `f16` but not `bf16` yet.
- 2-D tiles. `tile_shape` has one extent.
- Linkage providers, the form of a library that compiles implementations on
  demand while the Tile compiler runs. They are not a public CUDA 13.5
  surface. A static library ships one implementation per declaration.
- A cutile-rs caller. Only cuTile Python has `foreign_call` today.

The [Supported Features](../appendix/supported-features.md) appendix tracks
the interop status alongside the rest of the compiler.

### Verifying the ABI on CUDA 13.4

The `.tilelib` needs the CUDA 13.5 or newer assembler, but the ABI can be
proven today. The example's
host program stands in for the Tile compiler:

```bash
cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120
```

It reads `tilelib_relu.tileexports.json`, generates `mock/caller.cu` with the
prototypes the ABI table predicts and one kernel per export, compiles that with
`nvcc` to LTOIR, links it with the Rust LTOIR through nvJitLink, the same
two-input link a consumed library gets, and runs one block of 128 threads per
export on the GPU. Every result is compared bit for bit:

```text
PASS relu: 128 elements through `relu_impl`, relu of a ramp from -4 to 4
PASS scale_in_place: 128 elements through `scale_in_place_impl`, the ramp scaled in place by 0.5
PASS warp_sum: 128 elements through `warp_sum_impl`, 32-lane sums over 0..128, every lane holding its warp's total
SUCCESS: 3/3 tile exports verified through the native ABI
```

The prototypes in `mock/caller.cu` are derived from the descriptions, not
written by hand, so the generated file is the ABI as the tooling understands
it. Without `nvcc`, or on a GPU that does not match `--arch`, the program
prints `skipping:` with the reason and exits 0.
