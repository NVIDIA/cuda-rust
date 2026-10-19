# dialect-mir

A [pliron](https://github.com/vaivaswatha/pliron) dialect that represents Rust's Mid-level Intermediate Representation (MIR). This is the first IR in the cuda-oxide pipeline -- `mir-importer` translates rustc's MIR into this dialect, then `mir-lower` lowers it to the LLVM dialect for PTX generation.

```text
rustc MIR ──► mir-importer ──► dialect-mir ──► mir-lower ──► LLVM dialect ──► LLVM IR ──► PTX
```

## Types

The dialect defines nine types that preserve Rust-level semantics:

| Type                  | Description                                                  | Example                                                               |
|-----------------------|--------------------------------------------------------------|-----------------------------------------------------------------------|
| `MirTupleType`        | Heterogeneous tuples                                         | `mir.tuple<i32, f32, i64>`                                            |
| `MirPtrType`          | Thin pointers with address space, mutability, and source kind | `mir.ptr<f32, mutable: true, addrspace: 3, kind: UniqueRef>`           |
| `MirSliceType`        | Fat pointers with carrier mutability and source kind (ptr + len) | `mir.slice<f32, mutable: false, kind: SharedRef>`                  |
| `MirDisjointSliceType`| `DisjointSlice<T>` -- per-thread unique access               | `mir.disjoint_slice<f32, ...>`                                        |
| `MirStructType`       | Named structs with layout metadata                           | `mir.struct<"Point", [f32, f32]>`                                   |
| `MirUnionType`        | Rust unions -- each field a view of the same bytes           | `mir.union<"Repr", [a, b], [i32, f32], 4, 4>`                       |
| `MirEnumType`         | Rust enums with their exact rustc layout                     | `mir.enum<"Ordering", i8, ...>`                                     |
| `MirArrayType`        | Fixed-size arrays                                            | `mir.array<f32, 256>`                                                 |
| `MirFP16Type`         | IEEE 754 binary16 -- Rust's `f16`                            | `mir.fp16`                                                            |

`MirEnumType` records the enum's layout the way rustc computed it: Direct,
Niche, Single, or Empty; the physical integer/pointer carrier and absolute
byte offset when one exists; full-width niche arithmetic; the variant names
and declared discriminant VALUES; inhabitedness; exact field offsets/sizes;
and total size/alignment. `Unknown` is reserved for legacy hand-built IR and
is rejected by physical lowering. A size of zero can still be a known ZST
Single/Empty layout.

The textual type stores these as flattened parallel lists, for example:

```text
mir.enum<"Ordering", si8, ["Less", "Equal", "Greater"], [255, 0, 1],
         [0, 0, 0], [], [], [], 0, 1, 1,
         1, 1, 8, 0, 0, 0, 0, 0, 0, 0, [1, 1, 1]>
```

`Ordering::Less` is declared as -1, stored as the unsigned i8 bit pattern
255. In a Direct layout, the carrier slot holds these declared values; using
variant indices instead made `Ordering::Less` match the `Equal` arm (issue
#146).

For a niche layout such as `Option<&T>`, the carrier is the pointer itself:
null encodes `None`, and a non-null pointer is the untagged `Some` variant.
The device never adds a synthetic discriminant.

### Pointer and reference kinds

`MirPtrType` and `MirSliceType` retain the source-level category that produced a
pointer-like Rust value:

| Rust type | `MirPointerKind` | Meaning |
|-----------|------------------|---------|
| `&T` | `SharedRef` | Shared Rust reference |
| `&mut T` | `UniqueRef` | Mutable/unique Rust reference |
| `*const T` | `RawConst` | Immutable raw pointer |
| `*mut T` | `RawMut` | Mutable raw pointer |
| compiler-generated address | `Erased` | Storage/projection pointer with no Rust alias guarantee |

The `is_mutable` bit records the source pointer carrier's mutability spelling;
it is not a general storage-write permission and is not proof of uniqueness.
For example, a `SharedRef` carrier is immutable even though Rust may legally
mutate an `UnsafeCell` reached through it. `RawMut`, `UniqueRef`, and compiler
internal mutable `Erased` carriers set the bit, while only `UniqueRef`
originates from `&mut T`.

Pointer kind is propagated through Rust type import, references, raw-address
formation, slices, and compatible casts. The dialect verifier, rather than
importer convention alone, enforces this transition matrix:

| Producer or transition | Permitted result | Required authority |
|------------------------|------------------|--------------------|
| Generic cast, pointer offset, or projection | Preserve carrier mutability and the source kind, or erase a concrete kind to `Erased` | None |
| `Rvalue::Ref` / `mir.ref` | `SharedRef` for `&T`; `UniqueRef` for `&mut T` | `Reborrow` |
| `Rvalue::AddressOf` / `mir.ref` | `RawConst` or `RawMut`, matching source mutability | `RawAddress` |
| Typed constant, static, or promoted address | `SharedRef`, `RawConst`, or `RawMut`; `UniqueRef` only for rustc's promoted immutable `[T; 0]` backing of `&mut []` | `StaticAddress` |
| Explicit rustc cast, coercion, or transmute | The concrete kind declared by that cast | `RustCast` |
| Adaptation to a declared Rust function/intrinsic ABI | The exact concrete ABI type | `AbiBoundary` |
| Inline PTX output whose type comes from its Rust destination | The exact destination-derived pointer carrier | `InlineAsm` |
| Allocation (`alloca`, shared/global/extern storage) | `Erased` only; `alloca` is specifically mutable AS0 storage, while shared/global producers retain their declared carrier mutability and fixed address space | None |
| Integer with exposed provenance | `RawConst`/`RawMut`, or the exact immutable `Erased` `FnPtrTarget` carrier used for reified function tokens; never a Rust reference or arbitrary writable `Erased` storage | `StaticAddress` or `RustCast` when concrete; none for the opaque function token |

The four address/conversion authorities (`Reborrow`, `RawAddress`,
`StaticAddress`, and `AbiBoundary`) apply only to a top-level pointer or slice. They require an
actual pointer-to-pointer or slice-to-slice conversion with the same
pointee/element shape; `StaticAddress` also permits the explicit
integer-to-raw-pointer case. Only `RustCast` on an explicit `Transmute` may
authorize a pointer kind nested in a representation-reinterpreting aggregate.
Every target pointer carrier, including `Erased`, otherwise needs a
structurally corresponding source carrier with matching aggregate category,
cardinality, field order, offsets, size, alignment, and ABI. This keeps a cast
from turning integer bytes into writable `Erased` evidence, or claiming that a
pointer was preserved merely because source and target list one at the same
declaration index.
The source is constrained too: `UniqueRef` may be reborrowed only from an
already writable concrete pointer (`UniqueRef`/`RawMut`) or a writable
top-level `Erased` thin/fat carrier. The same requirement applies when
`RawAddress`, `StaticAddress`, or `AbiBoundary` establishes `RawMut`.
For a pointer-to-pointer conversion, `StaticAddress` specifically requires an
`Erased` physical/static source; it cannot relabel an arbitrary raw pointer as
a Rust reference or a different raw category. Recovering a concrete kind from
`Erased` with `StaticAddress` or `AbiBoundary` is currently thin-pointer-only:
the verifier traces a closed set of pointer casts/offsets back to a
`mir.global_alloc`/`mir.shared_alloc` or `mir.alloca`, respectively, and rejects
block arguments, loads, calls, marked casts, and unknown producers. The sole
static `UniqueRef` exception is rustc's promoted `&mut []`: the source must be
the direct result of an immutable `mir.global_alloc` whose declared type and
exact pointer pointee are the same `[T; 0]`, whose empty initializer is recorded,
which carries no relocation, and whose explicit allocation alignment satisfies
the full required alignment of `T` (natural scalar/pointer alignment as well as
any rustc ABI alignment retained by an aggregate). Unit/empty markers and the
slice/disjoint-slice value carriers have explicit conservative rules; a leaf
whose alignment is not represented in this dialect fails closed.
Non-empty arrays, arbitrary zero-sized types, raw-pointer laundering, and
unproven `Erased` values remain rejected.
Lowering treats `global_key`/shared `alloc_key` as allocation identities, not
symbol-name hints: a repeated key is reused only when its complete physical
declaration agrees (type/extent, alignment, address space, initializer,
relocations, and immutability as applicable). Promoted-global keys include the
evaluated allocation alignment. Conflicts fail closed before one address can be
redirected to under-aligned or differently typed storage.
Static shared allocations and per-function dynamic extern-shared declarations
are distinct storage categories, so even an adversarial key collision cannot
redirect one to the other's symbol or linkage.
`InlineAsm` is accepted only by `nvvm.inline_ptx`, only when every recursive
result pointer carrier is derived from the Rust destination type; a
pointer-free result must not carry it. `SharedRef`, `RawConst`, and immutable `Erased` carriers cannot directly
establish a mutable concrete kind. `AbiBoundary` may otherwise establish a
concrete kind from internal `Erased` storage or preserve the already exact
concrete kind; it does not relabel one concrete Rust category as another.

`RustCast` is also checked against rustc's cast kind; the label is not a
universal escape hatch. `Transmute` may reinterpret a pointer-bearing
representation. `PtrToPtr` changes only raw-pointer categories,
`FnPtrToPtr` converts only the canonical immutable `Erased` `FnPtrTarget`
function-pointer carrier to a raw pointer,
`MutToConst` is exactly `RawMut -> RawConst`, and `ArrayToPointer` stays raw
without turning const into mut. `Unsize` is the supported thin-to-fat trailing
array conversion with kind, mutability, field order, offsets, and ABI prefix
preserved; `Subtype` is type-identical after translation. The importer resolves
function-item and noncapturing-closure coercions and materializes the canonical
opaque token directly; the legacy `ReifyFnPointer` and `ClosureFnPointer`
`mir.cast` forms are rejected because their zero-sized operands contain no
address bits to lower. Exposed-provenance materialization may create only that
exact `Erased` function token or a raw pointer. Ordinary integer bytes cannot
manufacture writable `Erased` evidence for a later reborrow.

Generic representation normalization and local storage may preserve a concrete
kind or deliberately forget it by converting to `Erased`, but they never
recover a concrete kind from `Erased` or switch directly between two distinct
concrete Rust kinds. This prevents `SharedRef -> Erased -> UniqueRef` laundering
while still allowing legitimate reborrows such as `RawMut -> UniqueRef` at
`Rvalue::Ref`.

Pointer projections and offsets preserve address space and carrier mutability.
They may not produce the canonical `FnPtrTarget` carrier: it is a resolved
function value, never a data address. The same recursive identity rule keeps a
nested function token in the same aggregate position unless an explicit Rust
`Transmute` performs the reinterpretation.
Unmarked representation casts preserve carrier mutability and kind (or erase a
concrete kind), while an explicit pointer-representation cast may also change
address space.
That makes writable `Erased` a traceable input to a later authorized
`UniqueRef`/`RawMut` boundary rather than a property a generic operation can
invent. Reading through a writable address does not require first changing its
pointer type. Ordinary slice carriers use generic address space 0 and obey the
same preserve-or-erase and mutability-preservation rules.
`MirDisjointSliceType` has a fixed field-0 carrier contract:
`MirPtr<T, mutable, addrspace(0), RawMut>`.

Retags (formerly `StatementKind::Retag`, now the `WithRetag` flag on
`Rvalue::Use`) remain a codegen no-op. The dialect records the static pointer
category but does not attempt to model dynamic Stacked Borrows / Tree Borrows
tags or retag epochs.

Union fields are alternative typed views of shared storage, not simultaneously
live values. Recursive pointer-carrier discovery includes every declared union
field so casts and ABI shapes fail closed, but that traversal is not evidence
that an inactive pointer/reference alternative exists. A fully initialized,
relocation-free raw-pointer/integer union constant is reconstructed as
`[u8; pointer_width] -> integer`, followed by `mir.undef` plus
`mir.insert_field` of that full-width integer; no pointer-producing cast occurs.
A later pointer-typed `mir.extract_field` is the source's unsafe union-view read.
Any future alias, validity, or dereferenceability attributes must therefore be
union-blind unless the active field has been proven; they must not recursively
trust all pointer kinds declared by a union type.

MIR-to-LLVM lowering deliberately erases `MirPointerKind`. All four Rust source
categories keep the same LLVM pointer/fat-pointer representation, and this
change does not emit `noalias`, `readonly`, `dereferenceable`, or related
metadata. Pointer kind, transition authority, and the closed storage-lineage
check are auditable classifications, not optimizer proof capabilities. They do
not encode properties such as `Freeze`/`Unpin`, borrow scope or epoch, or an
active union field. Any future alias metadata therefore requires a separate,
first-class verified proof; it must not be inferred from pointer kind,
authority, lineage, or `is_mutable` alone. In particular, store legality
through `UnsafeCell` cannot be modeled by treating an immutable carrier as
globally read-only.

For GPU-specific abstractions such as `SharedArray`, the Rust reference kind is
still retained when the source type is a reference, while the CUDA address
space remains an independent property. The compiler does not currently turn a
`UniqueRef` to shared memory into cross-thread LLVM alias guarantees.

### Address Spaces

Pointers carry an NVPTX address space independently of their source kind:

| Space      | ID | PTX Qualifier | Use                           |
|------------|----|---------------|-------------------------------|
| Generic    | 0  | (none)        | Default, resolved at runtime  |
| Global     | 1  | `.global`     | Device VRAM                   |
| Shared     | 3  | `.shared`     | Per-block scratchpad          |
| Constant   | 4  | `.const`      | Read-only cached              |
| Local      | 5  | `.local`      | Per-thread stack/spill        |
| TensorMem  | 6  | `.param`      | Blackwell+ tcgen05 operands   |

## Operations

62 operations across 12 modules, one row per file in `src/ops/`:

| Module         | Ops | Description                                                                                              |
|----------------|-----|----------------------------------------------------------------------------------------------------------|
| `function`     | 1   | `MirFuncOp` -- function definition                                                                       |
| `control_flow` | 6   | return, goto, cond_branch, assert, unreachable, unroll_hint                                              |
| `memory`       | 11  | alloca, load, store, ref, assign, ptr_offset, memcpy, memmove, shared_alloc, global_alloc, extern_shared |
| `constants`    | 3   | integer, float, and undef constants                                                                      |
| `arithmetic`   | 15  | add/sub/mul/div/rem, checked variants, bitwise, shifts                                                   |
| `comparison`   | 7   | lt, le, gt, ge, eq, ne, cmp                                                                              |
| `aggregate`    | 10  | construct/extract/insert for structs, tuples, arrays, slices and disjoint slices; field and element address |
| `enum_ops`     | 4   | construct_enum, get_discriminant, set_discriminant, enum_payload                                          |
| `cast`         | 1   | type conversions (kind tracked via `MirCastKindAttr`)                                                    |
| `storage`      | 2   | storage_live, storage_dead (lifetime markers)                                                            |
| `call`         | 1   | function calls                                                                                           |
| `debug`        | 1   | dbg_value -- binds a value to a source-level variable                                                    |

`MirAllocaOp` implements `PromotableAllocationInterface` and `MirLoadOp` / `MirStoreOp` implement `PromotableOpInterface`, so pliron's `mem2reg` pass can promote scalar stack slots back into SSA. `MirUndefOp` is the default reaching definition the pass materialises when a load is not dominated by any store.

## Verification

Every operation implements pliron's `Verify` trait to catch bugs early during the import phase.
Both public lowering entry points also run a whole-tree producer gate: generic
`builtin.constant` may produce scalars, but cannot claim a direct or nested MIR
pointer carrier.

| Category     | What's Checked                                             |
|--------------|------------------------------------------------------------|
| Function     | Entry block args match function signature                  |
| Control flow | Condition is `i1`, successor block args match              |
| Memory       | Pointer types, pointee types, address spaces consistent    |
| Arithmetic   | Operands same type, result type matches                    |
| Comparison   | Operands same type, result is `i1`                         |
| Aggregate    | Struct/tuple types, index within bounds, element types     |
| Enum         | Discriminant type valid, payload types match variant       |
| Cast         | Cast kind, recursive pointer-kind transitions, authority compatibility, and exposed-provenance targets |
| Constants    | Type attribute present and well-formed                     |
| Call         | Callee exists, argument count and types match              |

This catches mismatches immediately after `mir-importer` translates from rustc, rather than deferring errors to LLVM.

## Attributes

The dialect defines eight domain-specific attribute types (following the pliron best practice of avoiding overloaded `IntegerAttr`), one row per `#[pliron_attr(...)]` in `src/attributes.rs`:

| Attribute                     | Rust Type                   | Description                                                                                                          |
|-------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------|
| `mir.cast_kind`               | `MirCastKindAttr`           | Preserves Rust cast intent (e.g. `IntToFloat`, `PtrToPtr`, `Transmute`) so lowering picks the right LLVM instruction |
| `mir.pointer_kind_authority`  | `MirPointerKindAuthorityAttr` | Names the semantic origin (`Reborrow`, `RawAddress`, `RustCast`, `StaticAddress`, `AbiBoundary`, or `InlineAsm`) of a pointer-kind transition |
| `mir.mutability`              | `MutabilityAttr`            | Boolean: `&` vs `&mut` for `mir.ref`                                                                                 |
| `mir.field_index`             | `FieldIndexAttr`            | Structural field index for `extract_field`, `insert_field`, `field_addr`, `enum_payload`                             |
| `mir.variant_index`           | `VariantIndexAttr`          | Enum variant index for `construct_enum`, `enum_payload`                                                              |
| `mir.fp16_attr`               | `MirFP16Attr`               | IEEE 754 binary16 value for `f16` constants, paired with `MirFP16Type`                                               |
| `mir.unroll`                  | `UnrollAttr`                | Unroll factor carried by `mir.unroll_hint` -- `0` means full unroll, `n >= 2` means `n` body copies per trip          |
| `mir.compiler_result_bundle`  | `CompilerResultBundleAttr`  | Marks an aggregate that exists only to adapt a compiler-owned multi-result op to a Rust aggregate return ABI          |

## Registration

```rust
use pliron::context::Context;
use dialect_mir::register;

let mut ctx = Context::new();
register(&mut ctx);  // Registers all ops, types, and attributes
```

## Source Layout

```text
src/
├── lib.rs                       # Dialect registration
├── types.rs                     # 9 MIR types + address_space constants
├── attributes.rs                # 8 domain-specific attributes
├── const_fold.rs                # Constant folding over dialect-mir ops
├── rust_intrinsics.rs           # Recognised core/std intrinsic calls
├── side_effects.rs              # Per-op side-effect classification
├── verification.rs              # Whole-tree pointer-producer gates shared by all lowerers
├── ops/
│   ├── mod.rs                   # Op module registry + re-exports
│   ├── function.rs              # MirFuncOp
│   ├── control_flow.rs          # Terminators, branches, unroll hints
│   ├── memory.rs                # Load, store, alloc, memcpy, shared memory
│   ├── constants.rs             # Integer and float literals
│   ├── arithmetic.rs            # Math, bitwise, shifts, checked ops
│   ├── comparison.rs            # Relational and equality
│   ├── aggregate.rs             # Struct, tuple, array, slice manipulation
│   ├── enum_ops.rs              # Enum construction and inspection
│   ├── cast.rs                  # Type conversions
│   ├── debug.rs                 # dbg_value source-variable bindings
│   ├── storage.rs               # Lifetime markers
│   └── call.rs                  # Function calls
```

## Further Reading

- [llvm-export](../llvm-export/) -- pliron-llvm shim + textual `.ll` exporter (lowering target)
- [dialect-nvvm](../dialect-nvvm/) -- NVVM GPU intrinsics
- [mir-importer](../mir-importer/) -- translates rustc MIR → `dialect-mir`
- [mir-lower](../mir-lower/) -- lowers `dialect-mir` → LLVM dialect
