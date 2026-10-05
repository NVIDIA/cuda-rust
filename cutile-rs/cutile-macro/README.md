# cuTile Rust Macros

Procedural macros for declaring cuTile Rust kernels with `cutile::module`.

<a id="two-track-design-macro-output-is-for-rustc-not-the-jit"></a>
## Macro expansion and JIT compilation

The macro runs at Rust compile time and emits three kinds of output:

1. **Rank-instance instantiation and shadow-trait dispatch.** Items that use a const-generic-array (CGA) generic — `const X: [i32; N]` — get materialized as concrete Rust that rustc can type-check. See *Rank-instance instantiation* and *Shadow dispatch* below.
2. **AST capture and registry registration.** `__module_ast_self()` stores the pre-expansion source text via `Span::source_text()` and re-parses it at runtime; a `linkme::distributed_slice` entry registers the module into a global registry the JIT consults. The JIT sees the *original generic* `fn foo<const S: [i32; N]>(...) { … real body … }`, not macro-emitted expansions. See *Linker-registry module discovery* below.
3. **Kernel launchers** for `#[entry]` functions.

The expanded Rust lets rustc type-check kernel bodies and report errors.
The JIT specializes ranks from the original source. Support for user-defined
structs or methods belongs in the JIT; the macro handles the Rust patterns
used by operation signatures in `_core.rs`.

## Linker-registry module discovery

`#[cutile::module]` emits two items used by the JIT to discover this module at runtime:

```rust
// Per-module AST builder. Returns just *this* module's `Module` value
// (re-parsed source text + span base). No recursive dep aggregation.
pub fn __module_ast_self() -> cutile_compiler::ast::Module { /* … */ }

// Self-registers this module into the global registry at link time.
#[linkme::distributed_slice(cutile_compiler::registry::CUTILE_MODULES)]
static __CUTILE_MODULE_ENTRY_FOO: cutile_compiler::registry::CutileModuleEntry =
    cutile_compiler::registry::CutileModuleEntry {
        absolute_path: ::std::module_path!(),
        build: __module_ast_self,
    };
```

`linkme`'s `#[distributed_slice]` exploits linker-section concatenation: each registration emits a `static` value tagged with a platform-specific section attribute (`#[link_section = "..."]`). At link time the linker concatenates every object file's same-named sections into one contiguous region; at runtime `CUTILE_MODULES` iterates that region as a slice of entries. The contract is between the macro and the linker, not Rust's module system — every crate that emits cuTile module entries with the agreed slice name participates in the same registry, regardless of visibility, semver, or path hierarchy.

### JIT-time dep discovery

`CUDATileModules::from_kernel(kernel)` walks the kernel's `use` statements iteratively against the registry:

1. Seed the working set with the kernel module.
2. For each `use path::*;` (or `use path::item;`) in the kernel, find the longest prefix that's registered in `CUTILE_MODULES`. If found and not yet visited, call its `build` closure to produce that module's AST, add it to the working set, recurse on its own `use` statements.
3. Continue until the working set stabilizes (cycle detection by visited absolute paths).
4. Build a `NameResolver` from the final working set.

`crate::*` paths are resolved against the owning module's crate-root segment (extracted from its `module_path!()`-derived absolute path). Unregistered paths (`std::`, `half::{bf16, f16}`, plain Rust submodules) are skipped silently.

### Re-exports need manual aliases

The registry uses string paths and can't follow Rust re-exports. If a crate exposes a module at a public path different from its canonical `module_path!()`, e.g.:

```rust
pub mod _core;             // canonical: `cutile::_core::core`
pub use _core::core;       // public:    `cutile::core`
```

then a `use cutile::core::*;` in user code won't match the registered `cutile::_core::core` entry. Until macro support lands, register an alias entry by hand next to the `pub use`:

```rust
#[linkme::distributed_slice(cutile_compiler::registry::CUTILE_MODULES)]
static __CUTILE_REEXPORT_CORE: cutile_compiler::registry::CutileModuleEntry =
    cutile_compiler::registry::CutileModuleEntry {
        absolute_path: "cutile::core",
        build: _core::core::__module_ast_self,
    };
```

### Caveats

- **Static linking only.** Modules inside a `dlopen`ed `.so` aren't reachable from the host's `CUTILE_MODULES` iteration.
- **Platform coverage.** `linkme` supports Linux, macOS, Windows, and Wasm. Currently verified on Linux only.
- **Mis-registered paths are silent misses.** A wrong `absolute_path` produces a JIT-time lookup miss rather than a link-time error. Mitigated by deriving the path from `module_path!()` at the static's location, which matches what the JIT looks up by construction.

## Rank-instance instantiation (`rank_instantiation.rs`)

For items annotated with `#[cuda_tile::variadic_struct]`, `#[cuda_tile::variadic_impl]`, or `#[cuda_tile::variadic_trait_impl]`, the macro emits one rank instance per rank, replacing each `const X: [i32; N]` with scalar consts `X_0, X_1, …, X_{R-1}` and suffixing CGA-bearing path segments accordingly.

```rust
// Source
#[cuda_tile::variadic_struct(N = 4)]
pub struct Tile<E: ElementType, const D: [i32; N]> { _type: PhantomData<E> }

// Macro output (abridged):
pub struct Tile_1<E: ElementType, const D_0: i32> { /* … */ }
pub struct Tile_2<E: ElementType, const D_0: i32, const D_1: i32> { /* … */ }
// … through Tile_4
```

Inherent impls instantiate the same way; their method bodies get walked by a `syn::visit_mut::VisitMut` pass (`RankInstantiator`) that substitutes the active CGA bindings into types, expression paths, struct literals, and `const_shape!` / `const_array!` macros.

## Shadow dispatch (`shadow_dispatch.rs`)

Functions marked `#[cuda_tile::variadic_op(...)]` and traits marked
`#[cuda_tile::variadic_trait(...)]` are rank-polymorphic. The macro generates
a *shadow trait* with the const generic arrays erased and one implementation
per rank. For free functions, it also generates a wrapper that dispatches
through the trait. User code calls the unsuffixed name; rustc selects the
rank-specific implementation through trait lookup.

Example — `addf` (same-shape case):

```rust
// Source
#[cuda_tile::variadic_op(N = 6)]
pub fn addf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
    lhs: Tile<E, S>, rhs: Tile<E, S>, rounding: R, ftz: F,
) -> Tile<E, S>;

// Macro output (abridged — trait + wrapper shown; 7 rank-instance impls elided)
trait Addf<R, F> {
    fn addf(self, rhs: Self, rounding: R, ftz: F) -> Self;
}
impl<E: ElementType, R: rounding::Mode, F: ftz::Mode, const S_0: i32>
    Addf<R, F> for Tile_1<E, S_0>
{ fn addf(self, rhs: Self, rounding: R, ftz: F) -> Self { unreachable!() } }
// ... ranks 0..=6

pub fn addf<__T, R, F>(lhs: __T, rhs: __T, rounding: R, ftz: F) -> __T
where __T: Addf<R, F> { lhs.addf(rhs, rounding, ftz) }
```

User code writes the call naturally; rustc resolves it through the trait:

```rust
fn kernel<const S: [i32; 2]>(x: Tile<f32, S>, y: Tile<f32, S>) -> Tile<f32, S> {
    addf(x, y, rounding::NearestEven, ftz::Disabled)
}
```

### Signature patterns the emitter handles

The emitter recognizes three shapes for the return type and emits the right trait form for each:

- **Same-shape (case 3a).** Return matches the first shape-bearing arg. Method returns `Self`. Examples: `addf`, `subf`, unary math, `fma`, comparisons that preserve shape.
- **Bound return (case 3b).** Return differs from `Self` but every generic in the return is also referenced by some argument. Method returns `Self::Out` with an associated type. Examples: `reshape`, `broadcast`, `constant`, comparisons with `bool` return.
- **Free return (case 3c).** Return contains a generic not present in any argument (an associated type would break coherence). `Out` becomes a trait generic that the caller ascribes at the return site. Examples: `permute`, `reduce_sum`, `bitcast`, `exti`.

The emitter also handles: nested-type recursion (`Option<Tile<bool, S>>`), reborrow on reference receivers (`&Tensor`, `&mut Tensor`), return-only lifetimes (lifted to trait generics), rank-dependent argument types (`idx: [i32; N]` promoted to a trait generic), and literal-CGA patterns (`Tile<E, {[]}>` → `Tile_0<E>`).

## Debugging

Generate backtraces on macro panics:

```bash
export RUSTFLAGS="-Zproc-macro-backtrace -Zmacro-backtrace"
export RUST_BACKTRACE=1
```

Dump generated kernel launcher code to a directory:

```bash
export DUMP_KERNEL_LAUNCHER_DIR="temp"
```

Inspect macro-emitted code for a specific module:

```bash
cargo expand -p cutile --lib
```

## Testing

```bash
cargo test -p cutile-macro
```
