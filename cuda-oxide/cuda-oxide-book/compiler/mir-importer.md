# The MIR Importer

The previous chapters explained how rustc produces Stable MIR
([rustc_public](rustc-public.md)) and how pliron provides the IR framework
([Pliron](pliron.md)). This chapter is where the two meet: `mir-importer`
takes the Stable MIR that rustc hands us and translates it into
`dialect-mir`, the pliron dialect that preserves Rust semantics. The
translator initially emits an alloca/load/store form -- cheap to produce,
easy to reason about, and a pliron identity on input. A subsequent
`pliron::opts::mem2reg` pass promotes those slots back into SSA form. cuda-oxide
then runs its [compiler optimizations](compiler-optimizations.md), beginning
with annotated loop unrolling, before lowering to the LLVM dialect.

But translation is only half the job. `mir-importer` also orchestrates the
*entire* compilation pipeline: translate, verify, optimize `dialect-mir`, lower,
export, and generate PTX. It is both the translator and the stage manager.

The crate lives in `cuda-oxide/crates/mir-importer` and is split into two parts:

- **`translator/`** -- the MIR-to-pliron translation logic (the interesting part).
- **`pipeline.rs`** -- the orchestration that chains every stage together (the
  responsible part).

---

## Pipeline Orchestration

Before diving into translation details, here is the big picture. The
`run_pipeline()` function is the entry point that `rustc-codegen-cuda` calls
after collecting device functions. It takes a list of `CollectedFunction`
structs and a `PipelineConfig`, then runs these stages:

```text
Step 1:  Translate Rust MIR → `dialect-mir`
Step 2:  Verify `dialect-mir` module
Step 3:  Run `pliron::opts::mem2reg` to promote alloca slots back into SSA
Step 4:  Apply `dialect-mir` optimizations (currently annotated loop unrolling)
Step 5:  Lower `dialect-mir` → LLVM dialect (via mir-lower)
Step 6:  For NVVM builds, adjust the LLVM dialect via nvvm-transforms
Step 7:  Export the LLVM dialect to textual LLVM IR (.ll)
Step 8:  Compile with llc, or with libNVVM and nvJitLink
```

Full variable-debug builds skip steps 3 and 4 so source variables remain in
stable memory locations for cuda-gdb.

Each `CollectedFunction` carries everything the pipeline needs to know about a
device function:

```rust
pub struct CollectedFunction {
    pub instance: Instance,
    pub is_kernel: bool,
    pub export_name: String,
}
```

`instance` is the monomorphized function from `rustc_public`. `is_kernel`
distinguishes kernel entry points from device helper functions (kernels get
special metadata in the LLVM IR so the NVPTX backend emits them as `.entry`
points). `export_name` is the symbol name that appears in the final PTX --
for device functions this is typically a fully qualified name (FQDN) that
matches what `CrateDef::name()` returns for the same function.

For each function, the pipeline:

1. Retrieves the MIR body via `instance.body()`.
2. Calls `translator::body::translate_body()` to produce the function's
   `dialect-mir` representation (using `mir.alloca` slots for locals), then
   appends it to the one `builtin.module` the whole run shares.
3. Runs pliron's verifier on the module to catch structural errors early --
   mismatched types, missing operands, broken dominance -- before they turn
   into cryptic LLVM failures downstream.
4. Runs `pliron::opts::mem2reg` to promote the alloca slots back into SSA
   values within `dialect-mir`.
5. Runs [compiler optimizations](compiler-optimizations.md), currently
   unrolling supported loops carrying `#[unroll]` or `#[unroll(N)]` requests.
6. Runs `lower_mir_to_llvm` (from the `mir-lower` crate) to lower every
   `dialect-mir` operation into its LLVM dialect equivalent via
   `DialectConversion`.
7. For NVVM builds, runs `nvvm-transforms` with the dialect selected for the
   target GPU. Ordinary PTX builds skip this step.
8. Exports the LLVM dialect module to a textual `.ll` string and compiles it
   with `llc`, or with libNVVM and nvJitLink for NVVM output.

If any step fails, the pipeline stops and returns a typed error (`NoBody`,
`Translation`, `Verification`, `Lowering`, `Export`, or `PtxGeneration`) with
enough context to diagnose the problem. No silent corruption, no mysterious
empty output files.

---

## Translation Architecture

The `translator/` directory is where Stable MIR becomes pliron IR. Each module
handles one level of MIR structure, and they compose neatly:

| Module       | Purpose                                                                    |
| :----------- | :--------------------------------------------------------------------------|
| `body`       | Function-level translation, alloca-slot seeding, FQDN name sanitization    |
| `block`      | Basic block translation coordinator                                        |
| `statement`  | Statement translations (assignments, storage)                              |
| `terminator` | Terminator translation (goto, call, return, FQDN-based intrinsic dispatch) |
| `rvalue`     | Expression translation (binops, casts, aggregates)                         |
| `types`      | Rust type to `dialect-mir` type conversion                                 |
| `values`     | MIR local → alloca slot mapping (`ValueMap`) + slot addrspace inference    |

The call flow follows MIR's structure top-down:

```text
pipeline::run_pipeline()
  └─ body::translate_body()               // once per collected function
       ├─ emit_entry_allocas()            // one mir.alloca per non-ZST local
       │     └─ SlotAddrSpaceMap::analyze // pointer slot addrspace inference
       └─ For each basic block:
            └─ block::translate_block()
                  ├─ statement::translate_statement()
                  │     └─ rvalue::translate_rvalue()
                  └─ terminator::translate_terminator()
```

`translate_body()` sets up the function's signature, creates pliron blocks
corresponding to MIR basic blocks, sanitizes the function name (converting
`::` to `__` in the FQDN from `instance.name()`), emits one `mir.alloca` per
non-ZST local at the top of the entry block, and stores the incoming
function arguments into their respective slots. Every non-entry block is
left argument-less -- cross-block data flow is carried by the alloca slots,
not by block arguments. Then it walks each block sequentially, translating
statements and terminators one by one.

`translate_statement()` handles the flat operations within a block --
assignments, storage live/dead markers, and discriminant writes. When an
assignment involves a right-hand side expression (a MIR `Rvalue`), it
delegates to `translate_rvalue()`, which handles binary operations, unary
operations, casts, aggregate construction, discriminant reads, pointer
arithmetic, and the other dozen-odd things Rust compiles to.

`translate_terminator()` handles the block-ending operations: `Goto`,
`SwitchInt`, `Call`, `Return`, `Assert`, `Drop`, and `Unreachable`. This is
also where intrinsic dispatch lives -- but that gets its own section below.

---

## The SSA Challenge (and how we defer it)

This is the trickiest part of the translation, and it deserves careful
explanation.

Rust MIR is *not* in strict SSA form. Locals (variables) are named storage
locations that any block can read or write. If `_3` is assigned in `bb0`, it
can be freely used in `bb1`, `bb5`, or anywhere else -- MIR does not care.

Pliron IR (MLIR-like) ultimately expects **strict SSA**: a value must dominate
every use, and if a value needs to flow from one block to another, it must be
passed explicitly as a **block argument**. You cannot just reach across
blocks and grab a local.

We resolve this tension in two phases:

1. **Importer: alloca + load/store.** The `mir-importer` deliberately does
   *not* construct SSA directly. Every non-ZST MIR local is backed by a
   single stack slot -- one `mir.alloca` emitted at the top of the entry
   block and recorded in `ValueMap`. Every write to the local becomes a
   `mir.store` into its slot; every read becomes a `mir.load`. Branch
   terminators are therefore all **zero-operand** and every non-entry block
   is **argument-less**: all cross-block data flow travels through the
   alloca slots, not through block arguments.

2. **`pliron::opts::mem2reg`: slots → SSA.** After the `dialect-mir` module
   verifies, `pipeline.rs` runs pliron's built-in `mem2reg` pass. It
   promotes every eligible alloca back into SSA values, rewiring each load
   to the reaching definition and inserting block arguments (the pliron
   spelling of phi nodes) wherever a value merges along multiple control
   paths. Address-taken slots -- the ones we genuinely need to keep on the
   stack -- are left alone for the `dialect-mir` → LLVM dialect lowering
   to translate into real `alloca`s.

Here is the problem in miniature. Given MIR where `_1` is written in `bb0`
and read in `bb1`:

```text
// Rust MIR
bb0: { _1 = 42_i32; goto -> bb1; }
bb1: { _0 = _1;     return; }
```

The importer first emits this alloca-based `dialect-mir` (before `mem2reg`):

```text
^bb0:
  %s1 = mir.alloca           : !mir.ptr<i32>
  %c  = mir.constant 42_i32  : i32
  mir.store %c, %s1
  mir.goto ^bb1                     // zero-operand; _1 flows via %s1
^bb1:                               // no block arguments
  %r = mir.load %s1 : i32
  mir.return %r : i32
```

After `pliron::opts::mem2reg` has promoted `%s1`, the same function becomes
SSA-form `dialect-mir`, with block arguments appearing only where they are
actually needed to merge reaching definitions:

```text
^bb0:
  %c = mir.constant 42_i32 : i32
  mir.goto ^bb1(%c : i32)
^bb1(%r : i32):
  mir.return %r : i32
```

(Functions with a single-predecessor successor, like the example above, end
up with a single block argument; joins with multiple reaching definitions
are where `mem2reg` introduces the nontrivial phi-style arguments.)

The upshot: the importer never runs a liveness analysis and never threads
values across blocks. All "which value is live where?" reasoning is deferred
to `mem2reg`, which already solves it correctly for the entire `dialect-mir`
module in one pass. The `translator/terminator/mod.rs` module docstring
carries the same worked example in inline-source form for quick reference.

---

## Type Translation

The `types` module converts Rust types (as seen through `rustc_public`) into
`dialect-mir` types. Most mappings are straightforward, but a few deserve
attention:

| Rust Type             | `dialect-mir` Type            | Notes                                |
| :-------------------- | :---------------------------- | :----------------------------------- |
| `i32`, `u64`, etc.    | `IntegerType`                 | With signedness tracking             |
| `f32`, `f64`          | `Float32Type` / `Float64Type` | Standard IEEE 754                    |
| `bool`                | `IntegerType(1)`              | 1-bit integer, as is tradition       |
| `(A, B, C)`           | `MirTupleType`                | Heterogeneous product type           |
| `&[T]`                | `MirSliceType`                | Pointer + length                     |
| `DisjointSlice<T>`    | `MirDisjointSliceType`        | Bounds-checked, typed-index mutable slice |
| `struct Foo`          | `MirStructType`               | With field offsets from rustc layout |
| `*mut T` / `*const T` | `MirPtrType`                  | With GPU address space               |
| `enum Option<T>`      | `MirEnumType`                 | Discriminant + variants              |

### Dynamic struct layout

Here is a subtlety that saves users a real headache. Consider this struct:

```rust
struct Extreme {
    a: u8,
    b: i128,
}
```

Rust's layout algorithm may reorder fields for alignment:

```text
User writes:      struct Extreme { a: u8, b: i128 }
rustc may layout: [b: i128 @ offset 0][a: u8 @ offset 16]
MirStructType:    mem_to_decl mapping, offsets, total_size
LLVM struct:      { i128, i8, [15 x i8] }   // explicit padding
```

cuda-oxide queries rustc for the exact byte offset of every field and builds
the struct type with explicit padding bytes. The `MirStructType` stores a
`mem_to_decl` mapping (memory order to declaration order), per-field offsets,
and the total size. When lowered to LLVM, padding is materialized as
`[N x i8]` arrays between fields.

The practical upshot: **`#[repr(C)]` is not required** for types shared between
host and device code. cuda-oxide matches rustc's layout automatically, so your
structs can use Rust's default `repr(Rust)` layout and the compiler will Do The
Right Thing on both sides. One less attribute to remember, one less footgun to
step on.

---

## Intrinsic Dispatch

When the translator encounters a `Call` terminator, it does not immediately
emit a `mir.call` operation. First, it checks whether the callee is a *known
intrinsic* -- a function from `cuda_device` that maps directly to a GPU hardware
instruction rather than a function with a body.

The `try_dispatch_intrinsic()` function matches on the **fully qualified domain
name (FQDN)** of the callee, obtained from `CrateDef::name()`:

```rust
match name {
    // condensed; the real arms also match re-export and ABI-alias paths
    "cuda_device::thread::threadIdx_x"  => emit_generated_nvvm_intrinsic(ReadPtxSregTidXOp),
    "cuda_device::thread::sync_threads" => { /* builds a Barrier0Op in place */ }
    // ... 100+ intrinsics
    _ => translate_as_normal_call()
}
```

Most arms are not written out by hand. Anything described by
`intrinsics/catalog.json` has its arm generated into a per-family module under
`cuda-oxide/crates/mir-importer/src/translator/terminator/intrinsics/generated/`.
Both arms above are of that kind, and so are the warp shuffles: the matched
strings are the masked `_sync` forms (`shuffle_xor_sync`,
`shuffle_xor_f32_sync`, `shuffle_xor_u64_sync`, and the
`shuffle_up_*`/`shuffle_down_*` families). The bare `warp::shuffle_xor` is
never a match string. It is an `#[inline(always)]` wrapper in
`cuda-oxide/crates/cuda-device/src/warp.rs` that supplies the full-warp mask
(`u32::MAX`) and calls `shuffle_xor_sync`, so the dispatcher only ever
intercepts the `_sync` call. The hand-written arms in `terminator/mod.rs`
and the `intrinsics/` modules beside the generated file hold the cases the
catalog does not describe.

The full FQDN (e.g. `cuda_device::thread::threadIdx_x`, not just `threadIdx_x`)
is used for matching to avoid ambiguity between identically-named functions in
different modules. The same FQDN is also used as the call target name for
non-generic, non-intrinsic calls -- the collector produces matching names, and
the lowering layer converts `::` to `__` on both sides.

If the function is a recognized intrinsic, the translator emits the
corresponding `dialect-nvvm` operation directly -- no function body, no call
overhead, just the hardware instruction. Thread indexing, warp shuffles,
barriers, shared memory operations, TMA bulk copies, and matrix multiply
instructions all go through this path.

If the function is *not* an intrinsic, it falls through to the normal path:
emit a `mir.call` operation that references the callee by symbol name. The
callee's body will have been translated separately (it is in the collected
function list too), so everything links up.

```{note}
The intrinsic dispatch table is the main extension point for adding new GPU
operations to cuda-oxide. If NVIDIA ships a new instruction and you want to
expose it, you add a function to `cuda_device`, add a `dialect-nvvm` op, and
add a match arm here. See [Adding New Intrinsics](adding-new-intrinsics.md)
for a step-by-step guide.
```

---

## Handling Unwind Paths

MIR models Rust's panic semantics faithfully. Every function call has two
possible successors -- a return target and an unwind target:

```text
_2 = mul(_1, _3) -> [return: bb1, unwind: bb2]
```

On a CPU, the unwind path matters: it runs destructors, unwinds the stack, and
either catches the panic or aborts the process. On a GPU, the CUDA toolchain
does not expose this capability today -- nvcc/ptxas strip landing pads and no
exception-handling infrastructure survives to PTX. The hardware itself *could*
support unwinding (absolute branches + per-thread call stack tracking
post-Volta are sufficient), but the compiler and runtime do not wire it up.
NVIDIA has an active project to add C++ exception support for automotive
safety; the current cuda-oxide design is forward-compatible with that work.

For now, cuda-oxide treats **all unwind paths as unreachable**. If a panic
would occur at runtime -- say, an integer overflow in debug mode or an
explicit `panic!()` -- the GPU traps and the kernel crashes. This is
semantically equivalent to `panic=abort` without requiring the user to set
the flag.

In practice, the translator simply ignores the unwind target in every `Call`
and `Assert` terminator, generating only the return-path branch. The unwind
blocks are never translated. They vanish, like they were never there.

The panic *message* vanishes with them. A block whose terminator is a diverging
call into `core::panicking` lowers to `nvvm.trap` + `mir.unreachable` and
nothing else -- its statements exist only to build what that dropped call would
have consumed (the message `&str`, the `format_args!` pieces), and the block has
no successor, so nothing they write can ever be read. Skipping them is also what
makes a message-carrying panic compilable in the first place: a materialized
`&str` constant has no device lowering.

This is not as scary as it sounds. Rust's borrow checker and type system
prevent most of the bugs that would cause panics. And for the ones that
slip through (array bounds checks, unwrap on `None`), a GPU trap is the
correct behavior anyway -- there is nothing useful a GPU thread can do to
"recover" from a logic error mid-kernel.

---

## Putting It All Together

Let's trace a simple kernel through the full pipeline to see how all the
pieces connect. Here is a vector addition kernel:

```rust
#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

After `mir-importer` translates the Stable MIR into `dialect-mir` (and
`pliron::opts::mem2reg` has promoted the alloca slots back into SSA), the
result looks something like this (simplified, with many details elided for
clarity):

```text
mir.func @vecadd(%a: mir.slice<f32>, %b: mir.slice<f32>,
                 %c: mir.disjoint_slice<f32>) {
^entry:
    %idx = nvvm.read_ptx_sreg_tid_x : i32
    %len = mir.extract_field %c[1]       // slice length
    %in_bounds = mir.lt %idx, %len
    mir.cond_br %in_bounds, ^compute, ^exit

^compute:
    %a_val = mir.load ...                // a[idx]
    %b_val = mir.load ...                // b[idx]
    %sum = mir.add %a_val, %b_val : f32
    mir.store %sum, ...                  // c[idx] = sum
    mir.goto ^exit

^exit:
    mir.return
}
```

A few things to notice:

- **`thread::index_1d()`** was dispatched as an intrinsic and became
  `nvvm.read_ptx_sreg_tid_x` -- a direct GPU register read, not a function
  call.
- **`DisjointSlice::get_mut()`** turned into a bounds check
  (`mir.lt`) and a conditional branch. The `if let Some` pattern in Rust
  became explicit control flow.
- **Block arguments** are absent here. Remember that `mir-importer` itself
  emits *every* branch terminator with zero operands and every non-entry
  block argument-less -- values cross block boundaries through alloca slots
  until `pliron::opts::mem2reg` runs. In this function, `mem2reg` can see
  that nothing needs to merge at `^compute` or `^exit` (no values survive
  across the branch), so it promotes the slots away without introducing any
  block arguments. A loopier kernel with values that live across a back
  edge would end up with `mir.goto ^header(%i, %acc)` and
  `^header(%i: i32, %acc: f32)` after `mem2reg`, which is pliron's spelling
  of phi nodes.
- **No unwind paths.** The original MIR had unwind targets on every
  operation that could panic. They are gone.

From here, the pipeline takes over:

1. **Verify** -- pliron checks that every operation's types match, every
   block's arguments are correct, and dominance holds.
2. **Optimize** -- `mir-transforms` rewrites supported annotated loops and
   cleans up the control flow it changes.
3. **Lower** -- `lower_mir_to_llvm` transforms `mir.add` into `llvm.fadd`,
   `mir.load` into `llvm.load`, `mir.slice` into an LLVM struct of pointer
   and length, and so on.
4. **Prepare NVVM IR when needed** -- `nvvm-transforms` converts operations to
   the LLVM form accepted by the selected libNVVM target. Ordinary PTX builds
   skip this step.
5. **Export** -- the LLVM dialect is printed as a textual `.ll` file with
   the appropriate `!nvvm.annotations` metadata marking `vecadd` as a
   kernel entry point.
6. **Compile** -- LLVM's NVPTX backend normally compiles the `.ll` to `.ptx`.
   NVVM builds instead use libNVVM and nvJitLink. The result is written next
   to the host binary.

`dialect-mir` captures Rust semantics faithfully. Before lowering it, cuda-oxide
can perform targeted, behavior-preserving rewrites. The first is explained in
[Compiler Optimizations](compiler-optimizations.md). The
[Lowering Pipeline](lowering-pipeline.md) then converts the result into something
LLVM can understand.
