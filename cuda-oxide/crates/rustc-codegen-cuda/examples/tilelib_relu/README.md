# tilelib_relu

A tile library written in Rust. Three `#[tile_export]` functions compile to
one LTOIR module on the native SIMT ABI, and `tilelib_relu.tileexports.json`
describes them. A Tile kernel reaches them once the two are packaged as
`tilelib_relu.tilelib`, which needs a CUDA 13.5 or newer tileiras: from Tile
C++ through a foreign declaration (`cpp/tile_caller.cu`), or from cuTile
Python through `foreign_call` (`python/call_from_cutile.py`). The host
program here proves the ABI on any CUDA 13.4 machine, through a generated
CUDA C++ caller that declares the implementations the way the Tile compiler
would.

```text
src/main.rs: #[tile_export] relu, scale_in_place, warp_sum
    |
    |  cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120
    v
tilelib_relu.ll ---libNVVM---> tilelib_relu.ltoir        tilelib_relu.tileexports.json
 (NVVM IR, SIMT ABI)            (device code)              (one description per export)
                                      |                              |
          +---------------------------+------------------------------+
          |                                                          |
          v                                                          v
consumer 1: this host program (CUDA 13.4 is enough)    consumer 2: cuTile (CUDA 13.5+)
  descriptions  -> mock/caller.cu                        cargo oxide tilelib tilelib_relu
  nvcc -dc --keep -gencode arch=compute_120,code=lto_120   -> tilelib_relu.tilelib
  nvJitLink(tilelib_relu.ltoir + caller.ltoir) -> cubin  python/call_from_cutile.py
  launch call_<export>, compare, PASS x3                   tp.foreign_call(symbol_name="relu")
```

## The exports

| cuTile name      | Rust                                    | implementation the call resolves to                |
|------------------|-----------------------------------------|----------------------------------------------------|
| `relu`           | `fn relu(x: f32) -> f32`                | `void relu_impl(float x, float *relu_out)`         |
| `scale_in_place` | `fn scale(x: &mut f32, factor: f32)`    | `void scale_in_place_impl(float *x, float factor)` |
| `warp_sum`       | `fn warp_sum(x: f32) -> f32`, warp sync | `void warp_sum_impl(float x, float *warp_sum_out)` |

`in` scalars travel by value, `&mut` parameters as pointers, and a return
value becomes a trailing `out` pointer. The host program does not restate
that table: it derives every prototype in `mock/caller.cu` from the
descriptions through `oxide_tile_export::TileExportDesc::abi_params`, so the
generated file is the ABI as the tooling understands it. It is written when
the program runs and is not tracked, like the other generated artifacts.

The generated kernels launch one thread per invocation, tile size divided by
the export's footprint, and pass each thread the first of its footprint
elements; the three exports here have footprint 1. An export with `f16`
elements is skipped with a `SKIP` line and counted in the `SUCCESS` line
rather than failed, because nvcc has no scalar type that lowers to LLVM
`half`.

## Run

```bash
# 1. Build the library and prove the ABI on the GPU (needs nvcc, libNVVM,
#    nvJitLink and a GPU matching --arch; CUDA 13.4 is enough).
cargo oxide run tilelib_relu --emit-nvvm-ir --arch sm_120

# 2. Package it as a tile library (needs a tileiras that lists 13.5 or newer).
cargo oxide tilelib tilelib_relu --arch sm_120

# 3. Call it from a Tile C++ kernel (needs the same toolkit's nvcc, which
#    knows --enable-tile; the kernel runs on the GPU and checks all three exports).
nvcc --enable-tile --tile-preview simt -Xtileiras=-l=tilelib_relu.tilelib \
     -std=c++20 -arch=sm_120 -o tile_caller cpp/tile_caller.cu && ./tile_caller

# 4. Or call it from cuTile Python (checked against cuTile Python 1.7).
python python/call_from_cutile.py --dry-run       # bytecode only, no tileiras
python python/call_from_cutile.py                 # full run, needs step 2
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

Step 3 prints one `PASS` line per export and then
`SUCCESS: 3/3 tile exports called from a Tile C++ kernel`. Each foreign
declaration in `cpp/tile_caller.cu` mirrors a declaration of the library:
parameters the library reads carry no attribute, parameters it writes carry
`[[cutile::experimental_inout]]` and come back as the result, the way the
MathDx tile libraries declare theirs.

Step 1 prints one line per export and then the marker:

```text
PASS relu: 128 elements through `relu_impl`, relu of a ramp from -4 to 4
PASS scale_in_place: 128 elements through `scale_in_place_impl`, the ramp scaled in place by 0.5
PASS warp_sum: 128 elements through `warp_sum_impl`, 32-lane sums over 0..128, every lane holding its warp's total
SUCCESS: 3/3 tile exports verified through the native ABI
```

Without nvcc it prints `skipping: nvcc not found` and exits 0.

## What PASS proves

- The wrapper `#[tile_export]` generates has the calling convention the Tile
  compiler derives from the same description: a C++ caller written from the
  description alone links against it and gets the right numbers.
- The link works the way a tile library is consumed: two LTOIR inputs,
  symbols resolved by nvJitLink, no PTX in between.
- `sync = warp` behaves when every lane of a warp calls together: all 32
  lanes receive the warp total.

All three checks compare bit for bit; the inputs are chosen so f32 cannot
round (relu copies or zeroes, 0.5 halves, the warp sums are small integers).

## What still needs CUDA 13.5 or newer

- `tilelib_relu.tilelib` itself: `tileiras --preview-create-tile-lib` exists
  from 13.5 on, and an assembler reads preview content only at the version
  it calls current, so the descriptions are written at the resolved
  assembler's newest listed version. When the installed tileiras lists only
  13.4,
  `cargo oxide tilelib` stops after writing the descriptions, says so, and
  exits 2; with `--library-optional` the descriptions alone count as
  success and it exits 0, which is how the smoketest runs it.
- The Tile C++ caller: nvcc gains `--enable-tile` and `--tile-preview simt`
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
