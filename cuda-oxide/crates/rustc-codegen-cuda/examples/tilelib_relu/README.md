# tilelib_relu

A tile library written in Rust. Five `#[tile_export]` functions compile to
one LTOIR module on the native SIMT ABI, and `tilelib_relu.tileexports.json`
describes them. A Tile kernel reaches them once they are packaged as
`tilelib_relu.tilelib`, which needs a CUDA 13.5 or newer tileiras: from Tile
C++ through a foreign declaration (`cpp/tile_caller.cu`,
`cpp/tile_caller_parametric.cu`), or from cuTile Python through
`foreign_call` (`python/call_from_cutile.py`). The host program here proves
the ABI on any CUDA 13.4 machine, through a generated CUDA C++ caller that
declares the implementations the way the Tile compiler would.

cuTile Rust can consume these libraries too, through `#[cutile::foreign]`
unsafe declarations and host-configured library paths. See
[Calling from cuTile Rust](#calling-from-cutile-rust) below.

```text
src/main.rs: #[tile_export] relu, scale_in_place, warp_sum   (tile_shape = 128)
             #[tile_export] relu_any, warp_sum_any           (no tile_shape)
    |
    |  cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120
    v
tilelib_relu.ll ---libNVVM---> tilelib_relu.ltoir        tilelib_relu.tileexports.json
 (NVVM IR, SIMT ABI)            (device code)              (one description per export)
                                      |                              |
          +---------------------------+------------------------------+
          |                                                          |
          v                                                          v
consumer 1: this host program (CUDA 13.4 is enough)    consumer 2: a Tile kernel (CUDA 13.5+)
  descriptions  -> mock/caller.cu                        cargo oxide tilelib tilelib_relu [--provider]
  nvcc -dc --keep -gencode arch=compute_120,code=lto_120   -> tilelib_relu.tilelib
  nvJitLink(tilelib_relu.ltoir + caller.ltoir) -> cubin  cpp/tile_caller*.cu, python/call_from_cutile.py
  launch call_<export>, compare, PASS x5
```

## The exports

| cuTile name      | Rust                                    | implementation the call resolves to                |
|------------------|-----------------------------------------|----------------------------------------------------|
| `relu`           | `fn relu(x: f32) -> f32`                | `void relu_impl(float x, float *relu_out)`         |
| `scale_in_place` | `fn scale(x: &mut f32, factor: f32)`    | `void scale_in_place_impl(float *x, float factor)` |
| `warp_sum`       | `fn warp_sum(x: f32) -> f32`, warp sync | `void warp_sum_impl(float x, float *warp_sum_out)` |
| `relu_any`       | `fn relu_any(x: f32) -> f32`, no `tile_shape` | `void relu_any_impl(float x, float *relu_any_out)`, one symbol per tile size |
| `warp_sum_any`   | `fn warp_sum_any(x: f32) -> f32`, warp sync, no `tile_shape` | `void warp_sum_any_impl(float x, float *warp_sum_any_out)`, one symbol per tile size |

`in` scalars travel by value, `&mut` parameters as pointers, and a return
value becomes a trailing `out` pointer. The host program does not restate
that table: it derives every prototype in `mock/caller.cu` from the
descriptions through `oxide_tile_export::TileExportDesc::abi_params`, so the
generated file is the ABI as the tooling understands it. It is written when
the program runs and is not tracked, like the other generated artifacts.

The first three exports name `tile_shape = 128`: their declarations are
`tile<128xf32>` and a static `.tilelib` can hold them. The last two leave
the tile size to the call: their declarations are `@relu_any<[N: index]>`
and `@warp_sum_any<[N: index]>`, the Tile compiler asks for one
implementation symbol per size a kernel uses, and only the provider form of
the library (`--provider`) can answer that.

The generated kernels launch one thread per invocation, tile size divided by
the export's footprint, and pass each thread the first of its footprint
elements; the five exports here have footprint 1. The parametric exports run
at 128 elements in the host program, which is as good a size as any for a
body that does not depend on the size. An export with `f16` elements is
skipped with a `SKIP` line and counted in the `SUCCESS` line rather than
failed, because nvcc has no scalar type that lowers to LLVM `half`.

## Run

```bash
# 1. Build the library and prove the ABI on the GPU (needs nvcc, libNVVM,
#    nvJitLink and a GPU matching --arch; CUDA 13.4 is enough).
cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120

# 2a. Package the three static exports as a tile library (needs a tileiras
#     that lists 13.5 or newer). The two parametric exports are left out
#     and the command says so.
cargo oxide tilelib tilelib_relu --arch sm_120

# 2b. Or package all five as a linkage provider: a shared object the
#     assembler loads while it compiles a kernel. Needs the Tile IR
#     compiler's linkage provider header, see below.
cargo oxide tilelib tilelib_relu --arch sm_120 --provider

# 3a. Call the static exports from a Tile C++ kernel (needs the same
#     toolkit's nvcc, which knows --enable-tile). Works with either library.
nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
     -std=c++20 -arch=sm_120 -o tile_caller cpp/tile_caller.cu && ./tile_caller

# 3b. Call relu_any at 128 and 256 elements and warp_sum_any at 64 from one
#     Tile C++ kernel; needs the provider library of step 2b.
nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
     -std=c++20 -arch=sm_120 -o tile_caller_parametric cpp/tile_caller_parametric.cu \
     && ./tile_caller_parametric

# 4. Or call relu and relu_any from cuTile Python (checked against cuTile Python 1.7).
python python/call_from_cutile.py --dry-run       # bytecode only, no tileiras
python python/call_from_cutile.py                 # full run, needs step 2b
```

Step 1 is not a Tile kernel. The host program stands in for the Tile
compiler: it writes a CUDA C++ caller from the descriptions and links it
against the LTOIR, which proves the ABI on CUDA 13.4. The end-to-end test, a
Tile kernel calling the Rust functions, is step 2 followed by step 3 or 4.
Run step 2 after step 1: `cargo oxide run` and `cargo oxide build` clear the
example's generated artifacts, the `.tilelib` among them. When the toolkit
that brings the 13.5 assembler is not the one on `PATH`, point
`CUTILE_TILEIRAS_PATH` at its `tileiras` for step 2 and run its `nvcc` for
step 3.

Step 2b builds the provider from the `oxide-tile-provider` runtime and the
crate's NVVM IR, as a generated crate under `target/oxide-provider/`. The
runtime binds the Tile IR compiler's linkage provider header, which is part
of that compiler's source tree and not of a CUDA toolkit; point
`TILEIR_LINKAGE_PROVIDER_INCLUDE_DIR` at the include directory that holds
`tile_ir/linkage_provider/LinkageProviderAPI.h`. With
`OXIDE_TILE_PROVIDER_LOG=1` in the environment of step 3 or 4, the provider
prints every question the assembler asked it and the symbols it compiled.

Step 3a prints one `PASS` line per export and then
`SUCCESS: 3/3 tile exports called from a Tile C++ kernel`; step 3b ends with
`SUCCESS: 3/3 parametric calls served by the tile library provider`. Each
foreign declaration in the two callers mirrors a declaration of the library:
parameters the library reads carry no attribute, parameters it writes carry
`[[cutile::experimental_inout]]` and come back as the result, the way the
MathDx tile libraries declare theirs. A parametric declaration is a function
template whose extent parameter is the declaration's `N`.

Step 1 prints one line per export and then the marker:

```text
PASS relu_any: 128 elements through `relu_any_impl`, relu of a ramp from -4 to 4
PASS relu: 128 elements through `relu_impl`, relu of a ramp from -4 to 4
PASS scale_in_place: 128 elements through `scale_in_place_impl`, the ramp scaled in place by 0.5
PASS warp_sum_any: 128 elements through `warp_sum_any_impl`, 32-lane sums over 0..128, every lane holding its warp's total
PASS warp_sum: 128 elements through `warp_sum_impl`, 32-lane sums over 0..128, every lane holding its warp's total
SUCCESS: 5/5 tile exports verified through the native ABI
```

Without nvcc it prints `skipping: nvcc not found` and exits 0.

## Calling from cuTile Rust

After `cargo oxide tilelib` packages the `#[tile_export]` functions, declare
their interfaces with `#[cutile::foreign] unsafe fn` inside a
`#[cutile::module]`. Match the library's names, concrete tile shapes, element
types, and input/output access modes, then configure the generated `.tilelib`
on the host with `TileLibrarySpec`. The
[cuTile Rust tile-library guide](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-book/guide/tile-libraries.md)
explains declarations, unsafe calls, and host configuration.

Use matching CUDA 13.5 preview tooling for the library and caller; the
library descriptors and caller bytecode must match the assembler's preview
version. The execution validation recorded in
[#1556](https://github.com/NVIDIA/cuda-rust/pull/1556) used the Tile C++ caller.
This example does not yet include an Oxide-to-cuTile Rust execution test.

## Current export limitations

- Exports are non-generic free functions with concrete, one-dimensional tile
  shapes. Static libraries are supported; on-demand linkage providers are not.
- Supported element types are `f16`, `f32`, `f64`, and signed/unsigned 8-, 16-,
  32-, and 64-bit integers; `bf16` is not accepted by `#[tile_export]`.
- All operands of an export must have the same per-invocation element count
  (footprint). Parameters use scalars, references, or references to fixed-size
  arrays; returns are scalar or unit, with at least one output required.
  Written operands (`out`/`inout`) must form one contiguous run.
- Synchronization is `none` or `warp`; warp exports need whole groups of
  `32 * footprint` elements. Callees cannot allocate shared memory; the export
  description has no scratch-buffer support.

References in the SIMT implementation access its tile fragments; they do not
add support for arbitrary pointer operands to cuTile Rust foreign declarations.

## What PASS proves

- The wrapper `#[tile_export]` generates has the calling convention the Tile
  compiler derives from the same description: a C++ caller written from the
  description alone links against it and gets the right numbers.
- The link works the way a tile library is consumed: two LTOIR inputs,
  symbols resolved by nvJitLink, no PTX in between.
- `sync = warp` behaves when every lane of a warp calls together: all 32
  lanes receive the warp total.

All checks compare bit for bit; the inputs are chosen so f32 cannot round
(relu copies or zeroes, 0.5 halves, the warp sums are small integers).

## What still needs CUDA 13.5 or newer

- `tilelib_relu.tilelib` itself: `tileiras --preview-create-tile-lib` exists
  from 13.5 on, and an assembler reads preview content only at the version
  it calls current, so the descriptions are written at the resolved
  assembler's newest listed version. When the installed tileiras lists only
  13.4, `cargo oxide tilelib` stops after writing the descriptions (and, with
  `--provider`, after building the shared object), says so, and exits 2; with
  `--library-optional` the descriptions alone count as success and it exits
  0, which is how the smoketest runs it.
- The Tile C++ callers: nvcc gains `--enable-tile` and `--tile-preview simt`
  with the same toolkit that brings the assembler, and the program and the
  library then agree on the bytecode version by construction.
- The cuTile side: `foreign_call` is a 13.5 bytecode feature. The dry run
  compiles it on 13.4 because it never calls tileiras. It was checked
  against cuTile Python 1.7 (checkout 770b493) with the `cuda-tile-preview`
  package built from that checkout's `experimental/tile_preview` folder:
  `cuda.tile_preview` is not on PyPI, and `call_from_cutile.py` imports
  private `cuda.tile._*` modules (`_bytecode`, `_cext`, `_compile`) that can
  move between releases. The full run also needs a cuTile Python that lists
  the assembler's current bytecode version: the script lists every version
  its encoder knows, and the probe against tileiras picks the newest one it
  accepts.
