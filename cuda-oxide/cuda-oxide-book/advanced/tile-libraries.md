# Tile Libraries

A Tile kernel operates on whole tiles without naming individual threads. A SIMT
function operates on elements from individual threads or warps. A **tile
library** connects the two: it packages Rust device functions so the Tile
compiler can call them from inside a Tile kernel and link them into a single
`cubin`.

- **Annotate in Rust:** Mark device functions with `#[tile_export]`.
- **Package the library:** Run `cargo oxide tilelib` to emit compiled device
  code and Tile IR metadata, packaged into a `.tilelib` archive by `tileiras`
  (CUDA 13.5+).
- **Call from a Tile kernel:** Invoke the exported function from Tile C++ or
  cuTile Python, with [cutile-rs](https://nvidia.github.io/cuda-rust/cutile/)
  caller support to follow.

| Mode              | Rust Attribute                   | How It Works                                                        |
|:------------------|:---------------------------------|:--------------------------------------------------------------------|
| **Static mode**   | `#[tile_export(tile_shape = N)]` | Fixed tile size `N` is compiled ahead of time into `.tilelib`       |
| **Provider mode** | `#[tile_export]`                 | Caller chooses size `N`; provider compiles each size `N` on demand  |

:::{seealso}
[Kernels and Device Functions](../gpu-programming/kernels-and-device-functions.md)
covers `#[device]` and LTOIR linking with CUDA C++. A tile library uses the
same LTOIR pipeline with the Tile compiler as the caller.
:::

```{figure} images/tilelib-direct-linkage.svg
:align: center
:width: 100%

Static tile library flow. `cargo oxide tilelib` emits compiled device code
(`.ltoir`) plus interface and implementation descriptions (`.tilebc`);
`tileiras` packages them into a `.tilelib` and links called exports into the
Tile kernel's `cubin`.
```

---

## Writing Exports

Add `#[tile_export]` to a standard Rust function. Your original function stays
unchanged so other Rust device code can still call it directly. Alongside it,
the macro generates:

- An `extern "C"` wrapper (`<name>_impl` by default) using the Tile-to-SIMT C
  ABI.
- A host-side export description that the compiler writes to
  `<crate>.tileexports.json`.

The
[`tilelib_relu`](https://github.com/NVIDIA/cuda-rust/tree/main/cuda-oxide/crates/rustc-codegen-cuda/examples/tilelib_relu)
example demonstrates four export patterns:

```rust
use cuda_device::{tile_export, warp};

/// 1. Fixed-size elementwise export (128 threads, 1 element each).
#[tile_export(tile_shape = 128)]
pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// 2. In-place mutation with a custom Tile-side declaration name.
#[tile_export(
    tile_shape = 128,
    name = "scale_in_place",
    symbol = "scale_in_place_impl"
)]
pub fn scale(x: &mut f32, factor: f32) {
    *x *= factor;
}

/// 3. Warp-synchronized export (all 32 lanes of each warp call together).
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

/// 4. Parametric export (no tile_shape; caller supplies tile size N).
#[tile_export]
pub fn relu_any(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}
```

How each export maps to its generated C wrapper in the device artifact:

| Tile name        | Rust function                        | Generated SIMT wrapper                              |
|:-----------------|:-------------------------------------|:----------------------------------------------------|
| `relu`           | `fn relu(x: f32) -> f32`             | `void relu_impl(float x, float *relu_out)`          |
| `scale_in_place` | `fn scale(x: &mut f32, factor: f32)` | `void scale_in_place_impl(float *x, float factor)`  |
| `warp_sum`       | `fn warp_sum(x: f32) -> f32`         | `void warp_sum_impl(float x, float *warp_sum_out)`  |
| `relu_any`       | `fn relu_any(x: f32) -> f32`         | `void relu_any_impl__N<size>(float x, float *out)`  |

`warp_sum` can safely use warp shuffles because `sync = "warp"` guarantees all
32 lanes of a warp execute the function together. See
[Warp-Level Programming](warp-level-programming.md) for the butterfly reduction
pattern.

### Attribute Options

| Option                      | Meaning                                          | Default            |
|:----------------------------|:-------------------------------------------------|:-------------------|
| `tile_shape = N`            | Fixed 1D tile size (omit for parametric exports) | Parametric         |
| `name = "..."`              | Declaration name visible to the Tile caller      | Rust function name |
| `symbol = "..."`            | Linker symbol in the device artifact             | `<name>_impl`      |
| `sync = "none"` or `"warp"` | Threads that enter the function together         | `"none"`           |
| `arch = ["sm_120", ...]`    | Compute capabilities supported by the export     | All architectures  |

- Omit `tile_shape` when the caller should choose the tile size at each call
  site (shipped with `cargo oxide tilelib --provider`).
- Omit `arch` to match every target. When specified, architecture strings
  normalize underscores and case, so `"sm_120"` also matches `sm_120a`.

### Parameters and ABI

| Rust parameter        | Access  | Footprint | SIMT ABI parameter            |
|:----------------------|:--------|:----------|:------------------------------|
| `x: T`                | `in`    | 1         | `T` by value                  |
| `x: &T`               | `in`    | 1         | `*const T`                    |
| `x: &[T; F]`          | `in`    | `F`       | `*const T`                    |
| `x: &mut T`           | `inout` | 1         | `*mut T`                      |
| `x: &mut [T; F]`      | `inout` | `F`       | `*mut T`                      |
| `-> T` (return value) | `out`   | 1         | `*mut T` (appended last)      |

- **Supported element types (`T`):** `f16`, `f32`, `f64`, `i8`, `i16`, `i32`,
  `i64`, `u8`, `u16`, `u32`, and `u64`. Tile IR uses signless integers, so
  unsigned Rust types (`u32`) map to same-width integer tiles (`i32`) with
  identical bits.
- **Footprint (`F`):** The number of contiguous tile elements owned by a single
  invocation (`1` for scalars and single-element references, `F` for array
  references).
- **By-value vs. pointer passing:** Only read-only scalars (`x: T`) travel by
  value. Reference parameters and the return value travel as pointers to the
  invocation's private elements.
- **Tile-side signature:** On the Tile side, every parameter (including the
  trailing return slot) is passed as a tile argument, and the call returns all
  `inout` and `out` tiles in declaration order.

### Compile-Time Rules

Both `#[tile_export]` and `cargo oxide tilelib` enforce these invariants:

- **Positive integer literal `tile_shape`:** When given, `tile_shape` must be a
  positive integer literal (not a `const` or expression).
- **At least one output:** Every export must either return a value (`-> T`) or
  accept at least one `&mut` parameter.
- **Inputs before outputs:** All `in` parameters must precede `&mut` (`inout`)
  parameters so that `inout` and `out` slots form a single contiguous run. For
  example, write `fn f(b: f32, a: &mut f32) -> f32` rather than
  `fn f(a: &mut f32, b: f32) -> f32`.
- **Uniform footprint across parameters:** All parameters of an export must
  share the same footprint `F` so they require the same number of invocations
  (`tile_shape / F`). Pair `acc: &mut [i32; 4]` with `bias: &[i32; 4]`, not
  `bias: &i32`.
- **Whole warps for `sync = "warp"`:** Warp-synchronized exports require
  `tile_shape` to be a multiple of `32 * F`.
- **Unreserved identifiers:** `name` and `symbol` must be valid ASCII
  identifiers and cannot start with `cuda_oxide_`.

---

## Building a Static Library

Run `cargo oxide tilelib` with a target architecture to build a static tile
library:

```bash
cargo oxide tilelib tilelib_relu --arch sm_120 --print
```

In a standalone project created with `cargo oxide new`, you can omit the crate
name. The command compiles the device crate and writes the library files next
to the artifact:

| Generated File                   | Contents                                                             |
|:---------------------------------|:---------------------------------------------------------------------|
| `tilelib_relu.ltoir`             | Compiled device code (LTOIR) for `sm_120`                            |
| `tilelib_relu.tileexports.json`  | Export metadata emitted by the compiler                              |
| `tilelib_relu.iface.tilebc`      | Tile IR interface bytecode (declarations)                            |
| `tilelib_relu.impl.tilebc`       | Tile IR implementation bytecode (fragment layouts)                   |
| `tilelib_relu.tilelib.mlir`      | Human-readable dump of both bytecode modules                         |
| `tilelib_relu.tilelib`           | Packaged tile library (when CUDA 13.5+ `tileiras` is available)      |

What happens during the build:

- **ABI verification:** Before writing bytecode, `cargo oxide tilelib` checks
  the emitted NVVM IR to confirm every wrapper symbol (such as
  `define void @relu_impl(float, ptr)`) matches the ABI table.
- **Static filtering:** Only exports with a fixed `tile_shape` are included in a
  static library. Parametric exports are skipped with a reminder to pass
  `--provider`.
- **Assembler packaging (`tileiras`):** Creating the final `.tilelib` requires
  `tileiras --preview-create-tile-lib` from **CUDA 13.5 or newer** (resolved via
  `CUTILE_TILEIRAS_PATH`, `PATH`, or your configured CUDA toolkit). On CUDA
  13.4, the command still writes the `.ltoir` and `.tilebc` files, prints the
  `tileiras` command to run on a newer toolkit, and exits with code `2` (or `0`
  if you pass `--library-optional`).

### Inspecting the Tile IR Dump (`--print`)

Passing `--print` prints the human-readable `.tilelib.mlir` descriptions for the
interface and implementation modules:

```mlir
// tilelib_relu.iface.tilebc: declarations called by a Tile kernel
cuda_tile.module @tilelib_relu {
  cuda_tile.preview$func extern @relu(
      %x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %relu_out: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>}
  ) -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$func extern @scale_in_place(
      %x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<inout>},
      %factor: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>}
  ) -> !cuda_tile.tile<128xf32>

  cuda_tile.preview$func extern @warp_sum(
      %x: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %warp_sum_out: !cuda_tile.tile<128xf32> {cuda_tile.preview$access = #cuda_tile.preview$access<out>}
  ) -> !cuda_tile.tile<128xf32>
}

// tilelib_relu.impl.tilebc: SIMT implementations and fragment layouts
cuda_tile.module @tilelib_relu {
  cuda_tile.preview$extern_impl @relu_impl implements @relu(
      %x: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<():()>,
          fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %relu_out: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<out>})

  cuda_tile.preview$extern_impl @scale_in_place_impl implements @scale_in_place(
      %x: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<inout>},
      %factor: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<():()>,
          fragmap = #cuda_tile.preview$layout<(128, 1, 1):(1, 0, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<in>})

  cuda_tile.preview$extern_impl @warp_sum_impl implements @warp_sum(
      %x: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<():()>,
          fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<in>},
      %warp_sum_out: !cuda_tile.preview$fragment<1xf32, local,
          mem_layout = #cuda_tile.preview$layout<(1):(1)>,
          fragmap = #cuda_tile.preview$layout<(4, 32, 1):(32, 1, 0)>>
          {cuda_tile.preview$access = #cuda_tile.preview$access<out>})
      synchronization = #cuda_tile.preview$synchronization<warp>
}
```

How to read the two sections:

- **Interface (`preview$func extern`):** Lists each function as the Tile
  kernel sees it, taking `tile<128xf32>` inputs and returning all `inout` and
  `out` tiles.
- **Implementation (`preview$extern_impl`):** Maps each C symbol (`@relu_impl`)
  to the per-invocation `fragment` it receives. `mem_layout` is `():()` for a
  by-value scalar and `(1):(1)` for a pointer to 1 element.
- **Fragment map (`fragmap`):** Maps `(invocations, participants, footprint)`
  across the tile:
  - `(128, 1, 1):(1, 0, 0)` runs **128 independent thread invocations** of 1
    element each (`sync = "none"`).
  - `(4, 32, 1):(32, 1, 0)` runs **4 warp invocations of 32 lanes** each
    (`sync = "warp"`).

---

## Building a Provider Library (`--provider`)

```{figure} images/tilelib-provider-linkage.svg
:align: center
:width: 100%

Provider tile library flow. The `.tilelib` embeds a shared library that
`tileiras` loads while compiling a Tile kernel, generating and compiling
size-specific wrapper symbols on demand from a single Rust function body.
```

When an export omits `tile_shape`, its declaration is parametric in the tile
size: `@relu_any<[N: index]>(tile<[N]xf32>)`. Because compiled device functions
in Tile IR are never generic, calling `relu_any` at sizes 128 and 256 requires
two distinct symbols (`relu_any_impl__N128` and `relu_any_impl__N256`).

Instead of prebuilding symbols for every possible size, pass `--provider` to
package a linkage provider shared library:

```bash
cargo oxide tilelib tilelib_relu --arch sm_120 --provider
```

How provider mode works:

1. **Embed NVVM IR at build time:** `cargo oxide tilelib --provider` embeds your
   crate's NVVM IR (`.ll`) and export metadata into a generated `cdylib`
   (`lib<crate>_provider.so`), then combines the interface bytecode and `.so`
   into `<crate>.tilelib`. Both parametric and fixed-size exports are included.
2. **Load during Tile compilation:** When `tileiras` compiles a Tile kernel
   against the library, it loads the `.tilelib` shared object via `dlopen`.
3. **Resolve each call site:** For every foreign call in the kernel (such as
   `relu_any<128>` and `relu_any<256>`), `tileiras` queries the provider for
   matching implementations and assigns a concrete symbol per tile size.
4. **Compile requested sizes on demand:** `tileiras` requests the device code
   once for all chosen symbols. The provider appends a thin wrapper per symbol
   that tail-calls your Rust function, compiles the module to LTOIR with
   `libNVVM`, and hands it back to `tileiras` for final linking.

Set `OXIDE_TILE_PROVIDER_LOG=1` when compiling a Tile kernel to print every
query from `tileiras` and every symbol compiled by the provider.

---

## Calling from a Tile Kernel

### From Tile C++

With CUDA 13.5 or newer, `nvcc` supports `--enable-tile` and
`--tile-preview simt`. Declare exported functions with
`__experimental_foreign_decl__`:

```cpp
namespace ct = cuda::tiles;
using tile_f32 = ct::tile<float, ct::shape<128>>;

// Fixed-size export: @relu(tile<128xf32> in, tile<128xf32> out) -> tile<128xf32>
__experimental_foreign_decl__ __experimental_foreign_name__("relu")
    tile_f32 relu(tile_f32 x, [[cutile::experimental_inout]] tile_f32 relu_out);

[[cutile::hint(0, num_worker_warps_per_cta = 4)]] __tile_global__ void kernel(float* x, float* y) {
    auto xs = ct::partition_view { ct::tensor_span { x, ct::shape<128> {} }, ct::shape<128> {} };
    auto ys = ct::partition_view { ct::tensor_span { y, ct::shape<128> {} }, ct::shape<128> {} };
    ys.store(relu(xs.load(0), ys.load(0)), 0);
}
```

Compile and run with `nvcc`:

```bash
nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
     -std=c++20 -arch=sm_120 -o tile_caller cpp/tile_caller.cu
./tile_caller
```

For parametric exports in a `--provider` library, declare a C++ function
template parameterized by the tile extent `N` (see
`cpp/tile_caller_parametric.cu`):

```cpp
template <size_t N>
__experimental_foreign_decl__ __experimental_foreign_name__("relu_any")
    ct::tile<float, ct::shape<N>> relu_any(
        ct::tile<float, ct::shape<N>> x,
        [[cutile::experimental_inout]] ct::tile<float, ct::shape<N>> relu_any_out);

// Instantiates relu_any_impl__N128 and relu_any_impl__N256 on demand:
relu128.store(relu_any<128>(view<128>(ramp128).load(0), relu128.load(0)), 0);
relu256.store(relu_any<256>(view<256>(ramp256).load(0), relu256.load(0)), 0);
```

Key points for Tile C++ callers:

- **Input vs. output attributes:** Read-only parameters take no attribute;
  `inout` and `out` parameters take `[[cutile::experimental_inout]]` and are
  returned as the call result.
- **Use the declaration name:** Pass `"relu"` (not `"relu_impl"`) to
  `__experimental_foreign_name__`.
- **Matched toolchain:** Because `nvcc` and `tileiras` come from the same CUDA
  toolkit, their Tile IR bytecode versions always match.

### From cuTile Python

In cuTile Python, load the `.tilelib` at module scope and invoke exports with
`cuda.tile_preview.foreign_call` (from `python/call_from_cutile.py`):

```python
import cuda.tile as ct
from cuda import tile_preview as tp

TILE = 128
PROVIDER = tp.Tilelib(path="tilelib_relu.tilelib", version="0.1.0")


def relu_tile(x, y):
    """Run Rust `relu` over one 128-float tile per block."""
    pid = ct.bid(0)
    tile = ct.load(x, index=(pid,), shape=(TILE,))
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

Key points for cuTile Python callers:

- **Declaration name:** Pass `symbol_name="relu"`, not `"relu_impl"`.
- **Pass a tile for every parameter:** `inputs` must supply a tile for every
  declaration argument, including output slots (`result_slot = ct.zeros(...)`).
- **Specify `output_types`:** Provide one `(dtype, shape)` tuple per `inout` or
  `out` parameter in declaration order.
- **Parametric exports:** When calling a `--provider` export without a fixed
  `tile_shape`, pass the tile extent in `constant_params=(TILE,)` instead of
  `()`.

---

## Execution Model and Current Limits

### What Happens at the SIMT Boundary

The Tile compiler never passes an entire tile to a single Rust call. Instead, it
partitions each tile into per-invocation fragments in local memory or registers:

- **Per-element (`sync = "none"`):** A 128-element tile with footprint 1 runs
  128 calls, each on one thread.
- **Per-warp (`sync = "warp"`):** Each invocation is a full 32-lane hardware
  warp. Every lane receives its own element(s), and warp collectives
  (`cuda_device::warp`) are valid inside the function.
- **Private fragment pointers:** Reference parameters (`&T`, `&mut T`,
  `&[T; F]`) point strictly to that invocation's own elements (`1` or `F`
  contiguous values). Do not index past them, and never assume adjacent
  threads' fragments are contiguous in memory.
- **Registers only (no shared memory):** The Tile compiler owns shared-memory
  allocation for the kernel. Do not declare `SharedArray` inside a
  `#[tile_export]` function.

### Current Limitations

- **Synchronization scopes:** `sync` supports `"none"` and `"warp"` only
  (warpgroup and CTA scopes are not yet available).
- **Tile rank and element types:** Only 1D tiles and `f16`, `f32`, `f64`, and
  8/16/32/64-bit integer types are supported (`bf16` and 2D tiles are not yet
  supported).
- **No shared-memory scratch:** Exports cannot request a shared-memory scratch
  buffer from the Tile compiler.
- **Workspace requirement for `--provider`:** Building a `--provider` library
  currently requires running inside the `cuda-oxide` workspace so the generated
  provider crate can reference `oxide-tile-provider` by path.
- **Caller front ends:** Tile C++ and cuTile Python can call tile libraries
  today; `cutile-rs` caller support is planned.

See [Supported Features](../appendix/supported-features.md) for the full
compiler feature matrix.

### Verifying the ABI on CUDA 13.4

Even without the CUDA 13.5 `tileiras` assembler, you can test the generated SIMT
wrappers and ABI end-to-end on a GPU using the example's host harness:

```bash
cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120
```

The host runner reads `tilelib_relu.tileexports.json`, generates a CUDA C++
caller (`mock/caller.cu`) with the exact prototypes derived from the export
descriptions, links it with the Rust LTOIR via `nvJitLink`, and verifies all 5
exports on the GPU:

```text
PASS relu: 128 elements through `relu_impl`, relu of a ramp from -4 to 4
PASS scale_in_place: 128 elements through `scale_in_place_impl`, the ramp scaled in place by 0.5
PASS warp_sum: 128 elements through `warp_sum_impl`, 32-lane sums over 0..128, every lane holding its warp's total
PASS relu_any: 128 elements through `relu_any_impl`, relu of a ramp from -4 to 4
PASS warp_sum_any: 128 elements through `warp_sum_any_impl`, 32-lane sums over 0..128, every lane holding its warp's total
SUCCESS: 5/5 tile exports verified through the native ABI
```
