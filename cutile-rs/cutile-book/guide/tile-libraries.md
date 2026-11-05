# Tile Libraries

A **tile library** (`.tilelib`) provides precompiled tile functions that a
kernel calls like any other function: an FFT of a tile, the Cholesky
factorization of a tile, and so on. Tile C++ calls them through
`__experimental_foreign_decl__` declarations and cuTile Python through
`cuda.tile_preview.foreign_call`; cuTile Rust calls them through
`#[cutile::foreign]` declarations.

```{note}
Tile libraries are a public preview of Tile IR 13.5. They need a CUDA 13.5
`tileiras`, and a tile library built for that same toolkit.
```

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
  A raw pointer, `*const E` or `*mut E`, is a rank-0 `tile<ptr<E>>` operand:
  a global-memory address the library reads or writes through (see [Pointer
  operands](#pointer-operands)).
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
- The function reads and writes only its operands and results, and the
  memory its pointer operands are documented to reach. For a pointer
  operand, the caller also guarantees what the library documents for it:
  that the memory is valid, large enough and aligned, and that no other
  thread accesses it while the call runs, unless the library allows it.

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
| Pointers | `ct::tile<T*, ct::shape<>>` | `array.foreign_pointer` | `*const T` / `*mut T` |
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

## Pointer operands

Some library functions work on global memory directly: they read their
input from global memory, write their output there, or take a scratch
buffer. Their Tile IR declarations take a rank-0 tile of pointers, which a
foreign declaration writes as a raw pointer:

```text
preview$func extern @store_to(
    %x: tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
    %dst: tile<ptr<f32>> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>}
  ) -> ()
preview$func extern @load_from(
    %src: tile<ptr<f32>> {cuda_tile.preview$access = #cuda_tile<preview$access<in>>},
    %out: tile<32xf32> {cuda_tile.preview$access = #cuda_tile<preview$access<out>>}
  ) -> (tile<32xf32>)
```

```rust
/// # Safety
///
/// The library writes 32 `f32` through `dst`.
#[cutile::foreign]
pub unsafe fn store_to(x: Tile<f32, { [32] }>, dst: *mut f32);

/// # Safety
///
/// The library reads 32 `f32` through `src`.
#[cutile::foreign]
pub unsafe fn load_from(src: *const f32) -> Tile<f32, { [32] }>;
```

- A pointer operand is always `in`: the call passes the address, never
  gets one back. `*mut` states that the library writes through it, `*const`
  that it only reads; both lower to the same `tile<ptr<E>>`, and rustc
  rejects a `*const` where a `*mut` is declared. `#[inout]` pointers and
  pointer results are errors.
- `u8` and `i8` pointees are both `ptr<i8>`, as for tiles.
- Get the address from a tensor, or take a raw pointer kernel parameter:
  - `y.block_tile_ptr()` (`*mut E`, needs `&mut`) is the address of the current
    tile block's tile of a mutable tensor: the element `y.load()` returns
    first. Like `load` and `store`, it is relative to the block, for ranks 1
    to 3, and needs a static tile shape.
  - `x.as_ptr()` (`*const E`) is the base of the whole view; offset it
    with `PointerTile::offset` to reach a chunk:

  ```rust
  #[cutile::entry()]
  fn round_trip(x: &Tensor<f32, { [-1] }>, y: &mut Tensor<f32, { [32] }>) {
      let pid: (i32, i32, i32) = get_tile_block_id();
      // This block's chunk of the input.
      let base: PointerTile<*const f32, { [] }> = pointer_to_tile(x.as_ptr());
      let chunk: PointerTile<*const f32, { [] }> = base.offset(pid.0 * 32i32);
      let t = unsafe { load_from(tile_to_pointer(chunk)) };
      // This block's tile of `y`.
      unsafe { store_to(t, y.block_tile_ptr()) };
  }
  ```

  A block tile is contiguous memory only when the tensor's innermost stride
  is 1 and the tile spans whole rows of the outer axes; a library that takes
  a pointer to contiguous data needs that.
- The call is ordered like a memory access (see above), so a load after the
  call reads what the library wrote through a pointer, and the library sees
  every store before the call.

A safe wrapper is the usual Rust pattern: the declaration states the
library's ABI with raw pointers, and the wrapper takes tensors whose types
carry what the library needs, so it upholds the obligations for every input
it accepts:

```rust
/// Writes `x` to this block's 32-element tile of `y`.
pub fn store_tile(x: Tile<f32, { [32] }>, y: &mut Tensor<f32, { [32] }>) {
    // SAFETY: the tile is this block's own, so no other block touches it,
    // and it holds the 32 elements the library writes.
    unsafe { store_to(x, y.block_tile_ptr()) }
}
```

## Configuring tile libraries

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

- Operands and results are tiles and scalars, and operands can also be
  rank-0 global-memory pointers. Tiles of pointers of rank 1 or more, and
  pointer results, are not supported.
- Constant parameters are `index`, string and datatype parameters; shape
  (tuple) parameters are not.
- A tile library must have been built for the `tileiras` that links it.
- A string constant must be a literal, a `const` with a literal value, or an
  associated const with a literal value. The JIT does not evaluate other
  constant expressions.
- Name resolution across modules has its own limits; see
  [Name Resolution](../reference/dsl-api.md#name-resolution).
