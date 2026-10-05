# rustc_public -- Stable MIR

cuda-oxide does not invent its own Rust parser or type system. It piggybacks on
the real Rust compiler, intercepts the internal representation that `rustc`
produces after type-checking and monomorphization, and compiles *that* to PTX.
This chapter explains the intermediate representation cuda-oxide reads (MIR),
the stability layer it reads it through (`rustc_public`), and the bridge
pattern that connects the two worlds.

## What is MIR?

After `rustc` parses your source code, resolves names, checks types, and
desugars all the syntactic conveniences (closures, `for` loops, `?` operator),
it produces **MIR** -- the **Mid-level Intermediate Representation**. MIR is a
simplified, control-flow-oriented form of your program that looks much closer
to what a machine would execute than the abstract syntax tree you wrote.

MIR is where the heavy lifting happens:

- **Borrow checking** -- Rust's ownership rules are verified against MIR, not
  against your source code.
- **Optimizations** -- constant propagation, copy propagation, dead store
  elimination, and inlining all operate on MIR.
- **Monomorphization** -- generic functions are stamped out into concrete
  versions for each set of type parameters.

In a normal Rust compilation, MIR is lowered to LLVM IR (or Cranelift IR if
you are using the `cranelift` backend), and from there to machine code. But
what if you could intercept MIR *before* it reaches LLVM and do something else
with it -- like compile it to PTX for GPUs?

That is exactly what cuda-oxide does.

## The stability problem

There is a catch. MIR is `rustc`'s **internal** intermediate representation.
It was never designed as a public API. Types get renamed, enum variants get
reordered, fields appear and disappear -- all between consecutive nightly
releases, sometimes between consecutive *commits*. The compiler team is under
no obligation to keep any of it stable, and they don't.

If cuda-oxide consumed `rustc_middle` types directly, every nightly update
would be a game of whack-a-mole: something moves, something breaks, someone
spends a weekend patching compilation errors instead of writing GPU code.

This is where `rustc_public` comes in.

## What is rustc_public?

`rustc_public` (formerly known as `stable_mir`) is a **stable interface** to
the Rust compiler's internals. It lets tool developers -- verifiers, linters,
codegen backends like cuda-oxide -- perform analyses and code generation
without breaking every time the compiler's plumbing shifts.

The implementation lives in two crates inside the `rustc` repository:

| Crate                 | Role                                                                                                                                                                                                  |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rustc_public`        | The user-facing public API. Defines stable types for `Body`, `BasicBlock`, `Local`, `Place`, `Ty`, `StatementKind`, `TerminatorKind`, and the rest of MIR. Will eventually be published on crates.io. |
| `rustc_public_bridge` | The translation layer. Converts between `rustc_public` types and the real `rustc_middle` types that live inside the compiler.                                                                         |

The stable API covers the types cuda-oxide cares about most:

- **`Body`** -- the MIR of a single function (basic blocks, locals, types).
- **`BasicBlock`** -- a straight-line sequence of statements followed by a
  terminator.
- **`Local` and `Place`** -- variables and memory locations.
- **`Ty`** -- the full Rust type system: primitives, references, tuples,
  ADTs, function pointers, closures.
- **`StatementKind`** -- assignments, storage annotations, discriminant reads.
- **`TerminatorKind`** -- branches, calls, returns, asserts, drops.
- **`Instance`** -- a monomorphized function (concrete types filled in).

```{note}
The `rustc_public` effort is driven by the [Kani](https://github.com/model-checking/kani)
team at AWS (formal verification for Rust) and other projects that need stable
compiler access. cuda-oxide benefits from their work without having to
maintain the bridge itself.
```

## How cuda-oxide hooks in

### The CodegenBackend trait

Deep inside `rustc`, compilation is organized around a trait called
`CodegenBackend`. Its key method is `codegen_crate`, which receives a
`TyCtxt` -- the compiler's god-object containing all type information,
MIR bodies, and metadata for the current compilation -- and must produce
compiled output.

Normally, `rustc_codegen_llvm` implements this trait and turns MIR into
machine code via LLVM. cuda-oxide provides `CudaCodegenBackend`, which
**wraps** the LLVM backend rather than replacing it. This is a deliberate
design choice: cuda-oxide is not a full replacement for LLVM, it is a
specialist that handles the GPU side while letting LLVM do what LLVM does
best.

The wrapping flow looks like this:

1. `rustc` calls `CudaCodegenBackend::codegen_crate(tcx)`.
2. cuda-oxide intercepts, runs its **collector** to identify all device
   functions (kernels and their transitive callees), and enters the stable MIR
   context to compile them to PTX via `mir-importer`.
3. The PTX output is written to disk alongside the build artifacts.
4. cuda-oxide delegates the host code to the wrapped LLVM backend, which
   compiles it into a native binary as normal.

The result is a single `cargo` invocation that produces both a native host
binary *and* a PTX module, without requiring two separate toolchains or a
split build system.

### Entering the stable MIR context

Inside `codegen_crate`, cuda-oxide receives `rustc_middle` types -- the
internal, unstable kind. The `mir-importer` crate, which does the actual
MIR-to-Pliron-IR translation, uses `rustc_public` as its main MIR interface. To
cross the boundary, cuda-oxide uses the bridge:

```rust
// Inside codegen_crate():
let result = rustc_internal::run(tcx, || {
    // Now in stable MIR context
    let stable_instance = rustc_internal::stable(func.instance);
    let body = stable_instance.body().unwrap();
    // Feed to cuda-oxide pipeline
    mir_importer::run_pipeline(&functions, &config)
});
```

`rustc_internal::run(tcx, || { ... })` sets up a scoped context where stable
MIR queries are available. Inside the closure, `rustc_internal::stable()`
converts an internal `rustc_middle::ty::Instance<'tcx>` into its stable
counterpart `rustc_public::mir::mono::Instance`. From there, calling
`Instance::body()` retrieves the MIR through the stable API -- no direct
contact with `rustc_middle` needed.

## Thread-local context management

You might wonder why the bridge needs a special `run()` scope instead of just
passing a context object around. The answer is lifetime entanglement.

`TyCtxt<'tcx>` borrows data from the compiler's arena allocator. It can be
stored in a struct or passed between functions, but its `'tcx` lifetime must
remain within the compiler session. `rustc_public` uses **scoped thread-local
storage** so its public query methods can find the current context without
requiring a `TyCtxt` argument on every call.

The bridge sets up a thread-local variable (TLV):

| TLV                       | Stored value        | Purpose                                                   |
| :------------------------ | :------------------ | :-------------------------------------------------------- |
| `compiler_interface::TLV` | `Cell<*const ()>`    | Scoped access to the current `CompilerInterface` reference |

The pointer is erased internally; `compiler_interface::with()` recovers the
`CompilerInterface` reference for queries such as `local_crate()` and
`all_local_items()`. Conversions such as `rustc_internal::stable(instance)`
use `with_bridge()`, which accesses the same interface through `with()` and
borrows its translation tables. There is no second thread-local context.

For example, converting an instance and then asking for its MIR body uses
one context throughout:

```text
rustc_internal::run(tcx, || {
    stable_instance = stable(instance) -> with_bridge() -> with() -> tables
    stable_instance.body() -----------------------------> with() -> query
})
```

Queries outside an active scope panic; starting a nested `run()` returns an
error. The current design replaced the old `Container` and two-context
implementation in [rust-lang/rust#147923](https://github.com/rust-lang/rust/pull/147923).

## The bridge pattern

The `CompilerInterface` owns a **`Tables`** struct -- a bidirectional mapping
between `rustc`'s internal IDs and the stable API's types. When you
call `rustc_internal::stable(instance)`, the bridge looks up (or creates) the
corresponding stable ID in the tables. When the stable API needs to query the
compiler on your behalf -- say, to fetch a function's MIR body -- it goes
through the tables in the opposite direction to recover the internal type.

```text
    rustc_middle::ty::Instance<'tcx>
              │
              ▼
         ┌─────────┐
         │  Tables │  (bidirectional: internal ↔ stable)
         └─────────┘
              │
              ▼
    rustc_public::mir::mono::Instance
```

A few implementation details worth knowing:

- **Interior mutability** -- `CompilerInterface` holds the tables and
  compiler context in `RefCell`s. Each query borrows them on the current
  thread; overlapping mutable borrows are rejected at runtime.
- **Caching** -- once a type or instance is translated, the result is stored
  in the tables. Repeated lookups hit the cache instead of recomputing.
- **Automatic cleanup** -- when `rustc_internal::run()` returns, the scoped
  thread-local value is unset and the interface and its tables are dropped.
  Stable IDs refer to these tables; do not reuse them in a later context.

The driver in `rustc-codegen-cuda` opens the scope with
`rustc_public::rustc_internal::run()` and converts instances with `stable()`.
The `mir-importer` translator primarily consumes `rustc_public` types, but
some checks still consult `rustc_middle` through the bridge. For example,
grid-constant validation reads declared reference lifetimes and inlining
ownership from the compiler's internal representation. The importer does
not manage the bridge tables itself.

## What MIR looks like

Before cuda-oxide can translate MIR, it helps to know what MIR actually
*is*. Here is a simple function and its MIR:

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

```text
// MIR for `add`:
// _0: i32              (return place)
// _1: i32              (argument `x`)
// _2: i32              (argument `y`)
// _3: (i32, bool)      (temporary for checked arithmetic)
//
// bb0: {
//     _3 = CheckedAdd(_1, _2);
//     assert(!(_3.1), "attempt to compute `{} + {}`, which would overflow") -> bb1;
// }
// bb1: {
//     _0 = (_3.0);
//     return;
// }
```

A few things to notice:

- **Locals** are numbered. `_0` is always the return place (where the result
  goes). `_1`, `_2`, ... are function arguments. Higher-numbered locals are
  temporaries the compiler introduces.
- **Basic blocks** (`bb0`, `bb1`, ...) are straight-line sequences of
  statements. Every block ends with exactly one **terminator** that
  transfers control -- a branch, a call, a return, or an assert.
- **`CheckedAdd`** returns a tuple `(i32, bool)`. The `bool` is an overflow
  flag. The `assert` terminator checks it and either continues to `bb1` or
  panics. In debug builds this catches integer overflow; in release builds
  the check is optimized away.
- **No expressions are nested.** `x + y` in the source becomes two separate
  operations in MIR: compute the checked add, then extract the result. Every
  intermediate value gets its own local. This flat structure is what makes MIR
  easy for tools like cuda-oxide to consume -- no recursive expression trees,
  just a flat list of operations per block.

```{note}
You can see the MIR for any function by passing `--emit=mir` to `rustc`, or
by visiting the [Rust Playground](https://play.rust-lang.org/) and selecting
MIR output. It is a surprisingly readable format once you get used to the
local numbering.
```

### Why this matters for cuda-oxide

MIR's flat, explicit structure is what makes it feasible to build a GPU
compiler on top of it. Consider the alternative: if cuda-oxide operated on
Rust's AST or HIR (the high-level IR), it would have to handle closures,
method resolution, trait dispatch, type inference, and a hundred other
language features that are *already* resolved by the time MIR is produced.
By reading MIR, cuda-oxide gets a representation where generics are already
monomorphized, closures are already lowered to structs, and control flow is
already explicit. The `mir-importer` crate translates this into Pliron IR
(an MLIR-like framework), and from there the
[lowering pipeline](lowering-pipeline.md) takes it the rest of the way to PTX.

## Nightly pinning

Even with `rustc_public` providing a stable *API*, the bridge layer
(`rustc_public_bridge`) is still compiled against the compiler's internals
and is not independently versioned. In practice, this means a specific
version of cuda-oxide works with a specific nightly.

cuda-oxide pins to an exact nightly release via `rust-toolchain.toml`:

```toml
[toolchain]
channel = "nightly-2026-08-28"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
```

This pin guarantees reproducible builds: anyone who clones the repository
gets the same compiler, the same MIR shapes, and the same bridge behavior.
When updating the pin, the process is:

1. Bump the nightly date in `cuda-oxide/rust-toolchain.toml`. Rustup walks up
   from the standalone `rustc-codegen-cuda` workspace and finds that component
   pin, so no nested copy is needed. `cuda-oxide/scripts/check-toolchain-parity.sh`
   checks the scaffold `cargo oxide new` writes, the devcontainer, and the
   blocks and dated commands quoted in this book and the READMEs -- run it to
   find the stragglers.
2. Fix any `rustc_public` API changes (usually minor -- that is the whole
   point of the stable API).
3. Run the full test suite to verify that all examples still compile and
   produce correct PTX.
4. Celebrate, or revert and try next week's nightly.

```{note}
As `rustc_public` matures and moves toward a crates.io release, the coupling
to a specific nightly will loosen. The long-term goal is for cuda-oxide to
work with any sufficiently recent stable Rust toolchain -- but we are not
there yet.
```

---

Now that you understand how cuda-oxide talks to the Rust compiler, the next
chapter covers what happens when that conversation reaches the codegen
backend: [The Code Generator](rustc-codegen-cuda.md).
