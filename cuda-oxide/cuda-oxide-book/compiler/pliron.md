# Pliron -- Pliron IR (MLIR-like)

cuda-oxide does not compile Rust directly to PTX in one step. It lowers code
through a series of intermediate representations, each capturing a different
level of abstraction. The framework that makes this possible is **pliron** -- an
extensible compiler IR framework written in pure Rust, inspired by LLVM's MLIR.
We refer to the IR built on this framework as **Pliron IR**.

This chapter explains what upstream MLIR is, why pliron exists as an alternative,
and how its core data structures work. If you have never built a compiler, don't
worry -- nothing here requires prior compiler experience, just a working
knowledge of Rust.

## What is MLIR (and why should you care)?

LLVM IR is a fixed instruction set. It has roughly 70 opcodes (`add`, `load`,
`br`, `getelementptr`, and so on), and if your domain doesn't map cleanly to
those opcodes, tough luck -- you flatten your high-level concepts into low-level
instructions and hope the optimizer can reconstruct what you meant.

**MLIR** (Multi-Level Intermediate Representation) takes a different approach.
Instead of one fixed instruction set, MLIR gives you a framework for defining
*many* instruction sets -- called **dialects** -- each tailored to a specific
domain. A dialect is a collection of operations, types, and attributes that model
the concepts you actually care about.

The key idea is straightforward:

1. Define a dialect with operations that match your domain.
2. Write **passes** that transform one dialect into another (lowering).
3. Chain passes together until you reach something a backend can consume.

Consider how Triton (the GPU compiler behind PyTorch) uses MLIR. Python GPU code
first lowers to **TTIR**, a tensor-level IR where operations like "broadcast this
scalar across a tensor" are first-class. At that level, Triton can apply
domain-specific optimizations -- for instance, replacing a `splat + mul`
(broadcast a scalar, then multiply element-wise) with a single native
vector-scalar multiply. That optimization is trivial to express when you have
tensor operations in your IR and nearly impossible to recover after flattening
to LLVM IR.

cuda-oxide faces a similar challenge. We need to represent three very different
levels of abstraction:

| Level              | What it models                                                | Example operations                           |
| :----------------- | :------------------------------------------------------------ | :------------------------------------------- |
| **`dialect-mir`**  | Rust semantics -- tuples, enums, slices, checked arithmetic   | `mir.extract_field`, `mir.get_discriminant`  |
| **LLVM dialect**   | Machine-near operations -- integer math, memory, control flow | `llvm.add`, `llvm.load`, `llvm.br`           |
| **`dialect-nvvm`** | GPU intrinsics -- thread indexing, warp shuffles, TMA, WGMMA  | `nvvm.read_ptx_sreg_tid_x`, `nvvm.shfl_sync` |

Without an extensible IR, we would have to either jam Rust enums into LLVM IR
(losing semantic information) or build separate IR frameworks for each level of
the pipeline. MLIR lets us define all three as dialects in a single system and
lower between them with well-typed passes.

## Enter pliron

MLIR's dialect-and-lowering model is the right abstraction for cuda-oxide: keep
high-level semantics in the IR while progressively lowering toward code the GPU
backend can consume. The question is implementation fit. Upstream MLIR is part
of the LLVM C++ ecosystem, while cuda-oxide is a Rust compiler project built
around Rust crates, Rust types, and Cargo workflows.

[Pliron](https://github.com/vaivaswatha/pliron) is an MLIR-inspired extensible
compiler IR framework written in **pure Rust**. It follows the same conceptual
model -- operations, regions, basic blocks, types, attributes, dialects, passes
-- while fitting naturally into a Rust-native compiler stack.

| Aspect            | pliron                                   | Upstream MLIR                      |
| :---------------- | :--------------------------------------- | :--------------------------------- |
| Language          | Pure Rust                                | C++ with Python bindings           |
| Build system      | `cargo build`                            | CMake + full LLVM build            |
| DSLs required     | None -- just Rust macros                 | TableGen, ODS                      |
| Debugging         | Standard Rust tooling (`dbg!`, rust-gdb) | gdb on C++ templates               |
| Extensibility     | Add a dialect as a Rust crate            | C++ extension against LLVM headers |
| Dependency weight | One crate (git dependency)               | Gigabytes of LLVM build artifacts  |

For cuda-oxide, this means the entire compiler -- from MIR import to LLVM IR
export -- stays within the same Rust build and debugging environment as the rest
of the project. Pliron gives us the extensible IR structure we need without
moving the compiler out of the Rust ecosystem.

## Core data structures

Pliron's IR is built from a handful of core types. Understanding these is the
key to reading (and writing) dialect code.

### Context

The `Context` is the central data structure that owns *all* IR data --
operations, basic blocks, regions, types, attributes, and dialect registrations.
Think of it as the arena allocator for your entire compilation unit.

```rust
// Simplified -- the real Context has more fields
pub struct Context {
    operations: SlotMap<Ptr<Operation>, Operation>,
    basic_blocks: SlotMap<Ptr<BasicBlock>, BasicBlock>,
    regions: SlotMap<Ptr<Region>, Region>,
    dialects: HashMap<DialectName, Dialect>,
    type_store: TypeStore,     // uniqued (deduplicated) types
    // Attributes are not stored centrally -- see "Types and attributes" below.
}
```

Pliron uses **generational arenas** (from the `slotmap` crate) instead of
`Box`-allocated heap nodes. This gives you:

- **O(1) insert and remove** -- no tree rebalancing, no linked-list traversal.
- **Stable indices** -- inserting or removing an element does not invalidate
  other indices.
- **Generational versioning** -- each slot carries a generation counter. If you
  delete an operation and the slot gets reused, any old reference will have a
  stale generation and fail at runtime instead of silently reading garbage.

Every piece of IR is stored inside the Context. You never hold a direct `&mut
Operation` across function boundaries -- you hold a `Ptr<Operation>` and deref
it through the Context when needed.

### Operations

An **operation** is a single node in the IR graph. It has operands (inputs),
results (outputs), attributes (compile-time metadata), and optionally regions
(nested structure for things like function bodies and loop nests).

Every operation belongs to a dialect. The naming convention is
`dialect_name.op_name`:

```rust
#[pliron_op(name = "mir.func", dialect = "mir")]
pub struct MirFuncOp;

#[pliron_op(name = "nvvm.read_ptx_sreg_tid_x", dialect = "nvvm")]
pub struct ReadPtxSregTidXOp;

#[pliron_op(name = "llvm.add", dialect = "llvm")]
pub struct AddOp;
```

The `#[pliron_op(...)]` proc macro (from `pliron-derive`) generates the
boilerplate for registering the operation with its dialect, assigning it an
opcode, and wiring up the `Op` trait. You define the operation's semantics by
implementing `Verify`, `Printable`, and `Parsable`.

### Types and attributes

**Types** represent data types in the IR. Pliron's built-in dialect provides
integers (`i1`, `i32`, `i64`) and floats (`f32`, `f64`). cuda-oxide's MIR
dialect extends these with Rust-specific types:

```rust
#[pliron_type(name = "mir.tuple", dialect = "mir")]
pub struct MirTupleType {
    pub types: Vec<Ptr<TypeObj>>,
}

#[pliron_type(name = "mir.slice", dialect = "mir")]
pub struct MirSliceType {
    pub element_ty: Ptr<TypeObj>,
}

#[pliron_type(name = "mir.enum", dialect = "mir")]
pub struct MirEnumType {
    pub name: String,
    pub discriminant_ty: Ptr<TypeObj>,
    pub variant_names: Vec<String>,
    // ...
}
```

**Attributes** attach compile-time metadata to operations -- constant values,
flags, predicate kinds, cast kinds, and so on.

Types and attributes are stored differently:

- **Types are uniqued** (deduplicated). If you create two
  `MirTupleType { types: vec![i32, f32] }` instances, pliron stores only
  one copy and hands back the same pointer. Type equality becomes a
  pointer comparison instead of a deep structural compare.
- **Attributes are not uniqued by default.** Each operation carries its
  own attribute values inline. This matches how MLIR's "properties" work
  -- MLIR originally uniqued attributes too, found that mutation and
  per-op state were awkward, and introduced properties as the unstored,
  per-op alternative. Pliron skipped that detour and started with the
  property-shaped design.

If you do want to dedup an attribute (for example, a 4 KB constant lookup
table that many ops reference), opt into uniquing per-attribute via the
`uniqued_any` utility. The pattern is to wrap the heavy payload in a
`UniquedKey<T>` and store the key inside the attribute:

```rust
struct SomeData { /* large or expensive-to-compare payload */ }

#[pliron_attr(name = "...", dialect = "...")]
pub struct SomeDataAttr(pub UniquedKey<SomeData>);
```

Now `SomeDataAttr` is a small handle that compares by key; the underlying
`SomeData` lives once in the uniquing table.

### Ptr\<T\> -- safe arena references

`Ptr<T>` is pliron's equivalent of a pointer into the arena. Under the hood, it
is composed of two fields:

```rust
pub struct Ptr<T> {
    index: u32,           // slot index in the arena
    version: NonZeroU32,  // generational version
    _phantom: PhantomData<T>,
}
```

The **version** field is what makes this safe. When you delete an operation,
pliron bumps the generation counter on that slot. If someone later tries to
dereference an old `Ptr<Operation>` whose version no longer matches the slot's
current generation, the access fails instead of reading the new (unrelated)
occupant.

The `PhantomData<T>` ensures type safety at compile time -- you cannot
accidentally use a `Ptr<Operation>` to index into the `BasicBlock` arena. The
compiler won't let you.

```{note}
You interact with arena contents through the Context:

- `ptr.deref(ctx)` returns a shared reference (`&T`).
- `ptr.deref_mut(ctx)` returns an exclusive reference (`&mut T`).

This follows Rust's borrow model -- the Context is the owner, and you borrow
through it.
```

## Def-use chains (memory-safe)

In any compiler IR, you need to answer two questions constantly:

- **Use-def**: "Where is this value defined?" (Given a use, find the definition.)
- **Def-use**: "Where is this value used?" (Given a definition, find all uses.)

Traditional compilers implement these chains with raw pointers and manual
bookkeeping. Forget to update a use-list when you delete an operation? Dangling
pointer. Replace a value but miss one use? Stale reference. Welcome to your
afternoon of debugging a segfault in `opt -O2`.

Pliron implements def-use chains using Rust's type system. A `Value` in pliron
is an enum:

```rust
pub enum Value {
    OpResult { op: Ptr<Operation>, index: usize },
    BlockArgument { block: Ptr<BasicBlock>, index: usize },
}
```

Every value is either the result of an operation or an argument to a basic block.
Each definition tracks the set of all its uses, and each use stores a pointer
back to its definition. When you call `replace_all_uses_with`, both sides update
automatically.

This means:

- **No dangling references** -- removing an operation updates all use-lists.
- **No stale uses** -- replacing a value propagates to every consumer.
- **No segfaults during IR transforms** -- the borrow checker and generational
  arenas prevent the entire class of bugs that plague C++ compiler frameworks.

```{note}
If you are familiar with LLVM's `Value` / `Use` / `User` system, pliron's
design serves the same purpose. The difference is that LLVM implements it with
intrusive linked lists and raw `Value*` pointers, while pliron implements it
with arena indices and `HashSet<UseNode>`. The semantics are similar, but the
ownership model fits Rust IR transforms directly.
```

## Op interfaces

Most things you want to do across an entire IR -- verify it, print it,
lower it to another dialect -- naturally express as "every op that
implements interface `X` knows how to do `X`". Pliron makes that pattern
first-class through **op interfaces**: small Rust traits, marked with
`#[op_interface]`, that any op (or type, or attribute) can implement.

You define an interface like a normal Rust trait, with the `#[op_interface]`
attribute on top:

```rust
use pliron::derive::op_interface;

#[op_interface]
pub trait MirToLlvmConversion {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}
```

Each op implements the interface with `#[op_interface_impl]`:

```rust
use pliron::derive::op_interface_impl;

#[op_interface_impl]
impl MirToLlvmConversion for MirAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        // ...lower MirAddOp into one or more LLVM dialect ops...
    }
}
```

The framework then dispatches against the interface, not against a
specific concrete op type. A pass that lowers MIR to LLVM looks
roughly like:

```rust
// Inside DialectConversion::rewrite, for each op encountered:
if let Some(converter) = op_cast::<dyn MirToLlvmConversion>(op) {
    converter.convert(ctx, rewriter, operands_info)?;
}
```

If you add a new MIR op tomorrow and write its `#[op_interface_impl]`,
the lowering pass picks it up automatically -- no central match
statement to update, no enum to grow. The same pattern is used for
verification (`#[op_interface] trait Verify`), printing, and any other
cross-cutting behavior.

The same `#[op_interface]` / `#[op_interface_impl]` mechanism applies to
types and attributes too: a type can implement `dyn MemorySemantics`, a
constant attribute can implement `dyn TypedAttr`, and so on.

```{note}
Under the hood, `op_cast` is a small runtime lookup (one hash table
probe) that maps the op's concrete type to the interface implementation
the macros registered. It only fires during pass dispatch, not in the
hot path of IR construction. The benefit is that adding a dialect is
adding a crate -- never modifying a central enum or generics-tangled
function signature.
```

## How cuda-oxide uses pliron

The lowering path works with three dialects: cuda-oxide defines `dialect-mir`
and `dialect-nvvm` locally (each as its own crate) and consumes the LLVM
dialect from the upstream `pliron-llvm` crate. Two further local dialects sit
off that path -- `dialect-iket` for in-kernel event tracing and `dialect-ptx`
for structured terminal PTX -- and the sections below cover only the three
above. Registration is automatic: every
dialect, op, type, and attribute linked into the binary registers itself when
a `Context` is created (`Context::default` runs all link-time registrations),
so kernel authors and pass authors never have to think about dialect setup --
depending on the crate is the only thing you do.

### dialect-mir -- Rust semantics

`dialect-mir` captures Rust's mid-level IR as pliron operations, preserving
semantic information that would be lost if we lowered directly to LLVM.

- **Function definition**: `MirFuncOp` -- entry point for each device function,
  with typed block arguments matching the Rust function signature.
- **Arithmetic and comparison**: checked and unchecked binary ops (`mir.add`,
  `mir.sub`, `mir.eq`, `mir.lt`, ...) that preserve Rust's overflow semantics.
- **Aggregate types**: `MirTupleType`, `MirStructType`, `MirEnumType`,
  `MirSliceType`, `MirArrayType` -- first-class Rust compound types with
  operations like `mir.extract_field` and `mir.get_discriminant`.
- **Memory and control flow**: `mir.load`, `mir.store`, `mir.ref`,
  `mir.goto`, `mir.cond_br`, `mir.return` -- with GPU address-space
  tracking (`global`, `shared`, `local`, `tmem`).

### LLVM dialect -- machine-near IR

The LLVM dialect models LLVM IR as pliron operations, providing a 1:1 mapping
to textual `.ll` files. Its modeling lives upstream in the `pliron-llvm`
crate; cuda-oxide re-exports it (and adds the textual exporter) through the
`llvm-export` crate.

- **Arithmetic and casts**: all 19 LLVM binary ops (`llvm.add` through
  `llvm.frem`), plus 13 cast ops (`llvm.sext`, `llvm.trunc`, `llvm.bitcast`, ...).
- **Control flow**: `llvm.br`, `llvm.cond_br`, `llvm.switch`, `llvm.return`,
  `llvm.unreachable` -- with block arguments translated to PHI nodes on export.
- **Textual export**: the `llvm_export::export` module emits valid LLVM IR text,
  including `@llvm.used` arrays and `!nvvm.annotations` metadata for GPU kernels.

### dialect-nvvm -- GPU intrinsics

`dialect-nvvm` wraps LLVM's NVPTX backend intrinsics as typed pliron
operations.

- **Thread indexing**: `nvvm.read_ptx_sreg_tid_x`, `nvvm.read_ptx_sreg_ctaid_x`,
  `nvvm.barrier0` -- the building blocks of `thread::index_1d()`.
- **Warp-level primitives**: shuffle (`nvvm.shfl_sync`), vote, and reduce
  operations for warp-cooperative algorithms.
- **Accelerator ops**: TMA bulk copies, WGMMA matrix multiply-accumulate
  (Hopper), and tcgen05 tensor core operations (Blackwell) -- the hardware
  instructions behind the [Advanced GPU Features](../advanced/tensor-memory-accelerator.md)
  chapters.

Each dialect is covered in detail in [Pliron Dialects](mlir-dialects.md).
