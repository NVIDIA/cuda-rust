# Tile Libraries

A **tile library** (`.tilelib`) packages device functions that a
kernel calls like any other function: an FFT of a tile, the Cholesky
factorization of a tile, and so on. Tile C++ calls them through
`__experimental_foreign_decl__` declarations and cuTile Python through
`cuda.tile_preview.foreign_call`; cuTile Rust calls them through
`#[cutile::foreign]` declarations.

```{note}
Tile libraries are a public preview of Tile IR 13.5. They need a CUDA 13.5
`tileiras`, and a tile library built for that same toolkit.
```

## Finding tile libraries

Start with NVIDIA's [Tile IR supported-library list](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#supported-libraries).
It links to providers as their packages become available. Tile libraries are
released separately from the CUDA Toolkit; installing CUDA 13.5 alone does not
install every provider's `.tilelib` files.

As of November 5, 2026, that list names:

| Provider | Functions | Availability listed by Tile IR |
| --- | --- | --- |
| MathDx Tile | FFT, linear algebra, compression | Upcoming; follow the supported-library list for package links. |
| [NVSHMEM](https://developer.nvidia.com/nvshmem) | GPU communication and synchronization | Starting with NVSHMEM 3.9; check the provider's downloads for release availability. |
| CCCL Coop | Cooperative CUB primitives, including scan, sort, and top-k | Upcoming; follow the supported-library list for package links. |

Use the provider's installation instructions to obtain a `.tilelib` and its
dependencies for your toolkit. Its interface determines the function names,
constant arguments, shapes, types, and architectures you can use. Then declare
the functions as below and pass the installed paths to
[`TileLibrarySpec`](#configuring-tile-libraries). The provider list describes
Tile IR support; cuTile Rust's [current operand limitations](#limitations)
also apply, including the lack of pointer operands in foreign calls.

The `solver.tilelib`, `fft_path`, and provider-library paths in this chapter
are illustrative configuration examples. They are not bundled downloads or
guaranteed package filenames. Replace them with the paths and declarations
from the provider's package.

### Reusable Rust kernels

For Rust implementations of attention, normalization, transformer utilities,
and NVFP4 MoE, see the
[`cutile-kernels` catalog](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-kernels/README.md)
and the
[`nvfp4_moe` example](https://github.com/NVIDIA/cuda-rust/blob/main/cutile-rs/cutile-examples/examples/nvfp4_moe.rs).
These are Rust source kernels launched from the host. A `.tilelib` supplies
device functions called from inside a kernel through `#[cutile::foreign]`.

## Declaring a foreign function

A foreign function is a body-less function inside a `#[cutile::module]`,
written like the declaration the tile library carries in its Tile IR
interface:

```text
preview$func extern @cholesky<[
    Function: !cuda_tile<preview$string>, FillMode: !cuda_tile<preview$string>,
    Batch: index, M: index, T: !cuda_tile<preview$Datatype>
  ]>(
    %A: tile<[Batch]x[M]x[M]xpreview$sym_t<T>> {cuda_tile.preview$access = #cuda_tile<preview$access<inout>>},
    %info: tile<[Batch]xi32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
  ) -> (tile<[Batch]x[M]x[M]xpreview$sym_t<T>>, tile<[Batch]xi32>)
```

```rust
#[cutile::module]
pub mod solver {
    use cutile::core::*;

    /// # Safety
    ///
    /// The declaration matches `solver.tilelib`'s `cholesky`.
    #[cutile::foreign]
    pub unsafe fn cholesky<const Batch: i32, const M: i32, T: ElementType>(
        Function: &str,
        FillMode: &str,
        #[inout] A: Tile<T, { [Batch, M, M] }>,
    ) -> (Tile<T, { [Batch, M, M] }>, Tile<i32, { [Batch] }>);   // `info` is output-only
}
```

- The declaration names a library function, not a library: the host chooses
  the libraries a kernel links (see [Configuring tile
  libraries](#configuring-tile-libraries)).
- `name = "..."` is the function's name in the library; it defaults to the
  Rust name.
- **The constant arguments** are the `&str` parameters, then the generic
  parameters, each group in order: a `name: &str` parameter is a string,
  `const N: i32` an `index`, and `T: ElementType` a datatype (`f32`, `f64`,
  ...). They correspond one to one to the constant parameters of the Tile
  IR declaration, the template parameters of a Tile C++ declaration and the
  `constant_params` of a cuTile Python `foreign_call`. When the library
  orders them differently, list every name in that order with
  `const_args = "..."`, e.g.
  `#[cutile::foreign(const_args = "Batch, Function, M, T, FillMode")]`.
- **Every other parameter is an operand the caller passes**, in the
  declaration's order: read-only by default, read-write with `#[inout]`. Scalars are rank-0
  tiles.
- **The results** start with the updated `#[inout]` operands. Every further
  result is an output-only operand (`info` above): it is not a parameter,
  and the call passes it zero-filled after the parameters. This is the form
  of Tile C++ `__experimental_foreign_decl__` declarations.
- The declaration may keep the library's names (`Batch`, `A`); the lints
  that would flag them are off for foreign declarations.
- A foreign function is declared `unsafe fn`, and every call is in an
  `unsafe` block (see [Safety](#safety)).

The declaration must match the library's interface exactly: `tileiras` checks
the call against the declaration the library carries.

### Safety

The compiler sees only the declaration, not the library's implementation, so
a foreign function is `unsafe fn`. A declaration that omits `unsafe` is an
error, and so is a call outside an `unsafe` block. Document the obligations
in a `# Safety` section of the declaration, even when there are none beyond
the ones below. The caller of a foreign function guarantees that:

- The declaration matches the library's interface: the name, the constant
  arguments and their order, the operand types and shapes, which operands are
  `#[inout]`, and the results. `tileiras` rejects some mismatches, not all.
- The library implements the function for the target architecture and for
  the constant arguments of the call.
- The function reads and writes only its operands and results. Foreign
  functions take tiles, which are values, so this holds for every library
  today. It becomes an obligation on the library once pointer operands are
  supported.

A wrapper that upholds these obligations for every input it accepts can be
a safe device function that calls the declaration in an `unsafe` block.

### String constants

A string constant is a `&str` parameter. The call passes a string that the
JIT resolves when it compiles the kernel, as it does for the format string
of `print_tko`:

- a literal: `cholesky::<1, 32, f64>("potrf", "lower", a)`;
- a `const` item whose value is a literal, by its name or its path:
  `const FILL: &str = "lower";`, `consts::FILL`;
- a local or a function parameter bound to one of these, so a wrapper can
  take `fill: &str` and pass it on. A local reassigned inside a runtime `if`,
  a loop, or a `{}` or `unsafe {}` block is not constant after it, or inside
  the loop: passing it there is a compile error;
- an associated const of a tag type, `Lower::NAME`, or `F::NAME` for a
  generic parameter `F` bound to a tag type. When a tag type defines `NAME` in two traits,
  name the trait: `<F as FillMode>::NAME`.

The string becomes part of the call's specialization: two calls with
different strings are two different library functions. Any other expression
(a value computed at run time, or `concat!`) is a compile error.

A tag type resolves by its path where it is written, as in Rust, through
imports and `pub use` re-exports. Two modules can each declare a `Lower`
whose `NAME` differs: each call passes the string of the `Lower` it names.
A tag that a function or method returns is the one its return type names in
that function's module.

### Mapping to Tile C++ and cuTile Python

| | Tile C++ | cuTile Python | cuTile Rust |
|---|---|---|---|
| Declaration | `__experimental_foreign_decl__` function template | none: every call is `foreign_call(...)` | `#[cutile::foreign]` body-less `unsafe fn` |
| Name in the library | `__experimental_foreign_name__("fft")` | `symbol_name="fft"` | `name = "fft"` (default: the Rust name) |
| Constant parameters | template parameters (`size_t`, `fixedString`, `typename`) | `constant_params=(int, str, dtype)` | `&str` parameters, then generic parameters (`const i32`, `ElementType`) |
| Read-write operands | `[[cutile::experimental_inout]]` | positional `inputs` | `#[inout]` |
| Output-only operands | implicit, from `multi_outputs` | passed explicitly | implicit: results after the `#[inout]` operands |
| Scalars | rank-0 tiles | rank-0 tiles | rank-0 tiles |
| Library | `-Xtileiras -l=path` | `Tilelib(path, version)` | `TileLibrarySpec::new(path)`, on the host |

## Calling a foreign function

Calls look like calls to any device function, inside an `unsafe` block.
Generic arguments the operands determine can be inferred; the others are
passed with a turbofish, where `_` lets the compiler infer the rest:

```rust
let (factor, info) = unsafe { cholesky::<_, _, _>("potrf", "lower", a_tile) };
```

A call lowers to a Tile IR `preview$call`, e.g.
`preview$call @cholesky<["potrf", "lower", 1, 32, f64]>(%a, %info0, %token)`,
where `%info0` is the zero-filled output-only operand.

Foreign calls are ordered like memory accesses: a call happens after every
earlier load, store, atomic and call, and before every later one, including
across loop iterations and in inlined helpers. Accesses with no call between
them stay unordered, as in a kernel without foreign calls.

`#[inout]` describes how the library uses the operand: the updated value is
the call's result, and the caller's tile is unchanged, as with any value.

Two modules can each declare a foreign function with the same Rust name;
each call reaches the one it names.

## Configuring tile libraries

Obtain a compatible provider package through
[Finding tile libraries](#finding-tile-libraries) before configuring these paths.

The host chooses the tile libraries that kernels link, so the same kernel
source can link a debug or a release build of a library, or one built for
each architecture:

```rust
use cutile::tile_kernel::{set_default_tile_libraries, CompileOptions, TileLibrarySpec};

// Libraries every kernel links, for the rest of the process.
set_default_tile_libraries([TileLibrarySpec::new("/opt/tilelibs/solver.tilelib")]);

// A library one launch links in addition to the defaults.
let (y,) = kernels::fft_kernel(y)
    .compile_options(CompileOptions::new().tile_library(TileLibrarySpec::new(fft_path)))
    .sync()?;
```

- The defaults (`set_default_tile_libraries`, `add_default_tile_library`)
  and a launch's `CompileOptions::tile_library` list are combined,
  canonicalized, sorted and de-duplicated before `tileiras` links them. A
  relative path is relative to the current directory. Adding a library that
  is already a default changes nothing.
- A kernel without foreign calls links nothing. A kernel with foreign calls
  and no configured library fails to compile with an error that names the
  library functions it calls.
- An *implementation-provider* library generates code for each call with a
  shared library it loads at link time. Declare every file it loads and
  every setting that changes the code it generates, so the caches miss when
  either changes:

  ```rust
  let solver = TileLibrarySpec::new(mathdx.join("lib/solver.tilelib"))
      .dependency(mathdx.join("lib/libmathdx_tile.so"))
      .setting("mathdx-version", "26.01");
  ```

## Limitations

- Operands and results are tiles and scalars; pointer operands are not
  supported yet.
- Constant parameters are `index`, string and datatype parameters; shape
  (tuple) parameters are not.
- A tile library must have been built for the `tileiras` that links it.
- A string constant must be a literal, a `const` with a literal value, or an
  associated const with a literal value. The JIT does not evaluate other
  constant expressions.
- Name resolution across modules has its own limits; see
  [Name Resolution](../reference/dsl-api.md#name-resolution).
