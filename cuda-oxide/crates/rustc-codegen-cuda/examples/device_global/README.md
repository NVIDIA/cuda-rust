# device_global

Tests ordinary Rust `static mut` values in CUDA global memory, non-zero
immutable Rust static tables, and pointer relocations stored inside
device-global initializers, including slice fat pointers.

Run with:

```bash
cargo oxide run device_global
```

To validate the emitted AS1 DWARF shape (including same-path block statics,
duplicate leaf names, reachability, relocation-only targets, semantic types,
CU retention, and LLVM verification), build with full device debug and run the
shape gate:

```bash
CUDA_OXIDE_DEBUG=full cargo oxide build device_global --arch sm_120
./cuda-oxide/crates/rustc-codegen-cuda/examples/device_global/verify-debug-info.sh
```

For a live cuda-gdb lookup, stop inside the `device_global` kernel and keep the
expression language on automatic/Rust mode:

```gdb
set language auto
print device_global::DEVICE_COUNTER
print device_global::DEVICE_MARKER
print device_global::debug_left::SAME_LEAF
print device_global::debug_right::SAME_LEAF
```

Use the literal crate-qualified path, without a leading `::`. A bare
`DEVICE_COUNTER` is not required to resolve from inside the nested `kernels`
module. The two `SAME_LEAF` lookups prove module qualification disambiguates
same-named statics. Do not switch to C++; this is a Rust debugging contract.

The first kernel updates two ordinary device statics:

```rust
static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
```

The block-local identity kernel calls a non-inlined device helper that places
`static VALUE` and `static VALUE_REF` in opposite blocks of one function. The
two definitions of each leaf have the same source display path but different
rustc DefPath disambiguators. Their values (11 and 29), addresses, physical
globals, debug DIEs, and reference initializer relocations must stay distinct;
repeated reads of each reference must still deduplicate to one physical
definition.

The other kernels read non-zero immutable statics. One reads both the base
address and an interior constant pointer (`&STATIC_WEIGHTS[2]`, a 16-byte
addend), matching generated coefficient-table access patterns:

```rust
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], ...];
const STATIC_WEIGHT_PAIR: &[f32; 2] = &STATIC_WEIGHTS[2];
```

The subobject kernel takes references to fields and indexed elements of
statics (`&PADDED_STATIC.tag`, `&PADDED_STATIC.value`, `&STATIC_WEIGHTS[2]`).
The address arithmetic stays in the static's physical address space and a
single address-space cast restores the generic Rust pointer type at the
borrow boundary.

The static-initializer relocation kernel covers pointer values that are part
of another static's evaluated allocation:

```rust
static TARGET_A: u32 = 0x1234_5678;
static TARGET_B: u32 = 0xcafe_babe;
static REFERENCE: &u32 = &TARGET_A;
static REFERENCES: [&u32; 2] = [&TARGET_A, &TARGET_B];
static INTERIOR_REFERENCE: &f32 = &STATIC_WEIGHTS[2][1];
```

rustc stores each pointer as literal addend bytes plus a separate provenance
entry naming the target allocation. cuda-oxide keeps both components through
MIR lowering. The LLVM global uses byte-array fields for literal spans and an
integer-width relocation slot for every pointer. The exporter reconstructs
each slot with `getelementptr`, `addrspacecast`, and `ptrtoint` constant
expressions, so the device linker sees an actual relocation rather than a null
placeholder.

Slice fat-pointer initializers use the same relocation carrier for their data
word while keeping the length metadata as literal initializer data:

```rust
#[repr(C)]
struct SliceTarget {
    prefix: [u32; 2],
    view: [u32; 3],
    suffix: [u32; 3],
}

static SLICE_TARGET: SliceTarget = SliceTarget {
    prefix: [11, 17],
    view: [23, 31, 41],
    suffix: [47, 59, 61],
};
static SLICE_VIEW: &'static [u32] = &SLICE_TARGET.view;
```

The two-element `prefix` fixes `view` at byte offset 8 without relying on
range indexing in a static initializer, which is not const-stable on the
repository's current Rust toolchain. The data word therefore relocates to
`SLICE_TARGET + 8`, while the metadata word stores length `3`. The runtime
regression verifies both the non-zero data-pointer addend and the slice length.

A top-level union initializer is also supported when its complete storage is
exactly one naturally aligned pointer word and every non-ZST union alternative
is a representation-compatible thin pointer. The example initializes the union
with `&UNION_RELOCATION_TARGETS[2]`, so the same check also covers an 8-byte
non-zero target addend. Pointer/integer unions, fat or nested pointer storage,
mixed pointer address spaces, nested unions, and padded, over-aligned, or
under-aligned union storage remain fail-closed.

Packed `repr(C, packed)` statics are also covered. When either the allocation
alignment or a relocation's byte offset cannot satisfy the pointer carrier's
natural alignment, cuda-oxide uses a packed LLVM struct only as the physical
initializer carrier.
The relocation carrier remains separate from the semantic aggregate
representation; device code accesses unaligned packed pointer fields with
`addr_of!` plus `read_unaligned()`. For example, a one-byte tag followed by an
eight-byte pointer is emitted as `<{ [1 x i8], i64 }>` with allocation
alignment 1. A relocated top-level struct whose own layout remains ordinary
and non-divergent may also contain one direct packed struct field when that
field's explicit rustc layout is non-overlapping and a sequential LLVM packed
struct reproduces its offsets and size exactly. A packed top-level struct does
not stack this relaxation on a packed child, and deeper packed nesting remains
fail-closed.

The relocation coverage includes:

- a direct static-to-static reference;
- multiple pointer fields in one initializer;
- two fields sharing one target;
- a second independently materialized target;
- an interior pointer with a non-zero byte addend;
- a slice fat-pointer initializer with a relocated data word and literal length metadata;
- a non-zero slice data-pointer addend into the target static;
- a top-level thin-pointer union occupying one pointer word;
- packed/unaligned relocation slots, including literal prefix/suffix bytes;
- one direct nested packed relocation carrier inside an ordinary `repr(C)` struct;
- targets reachable only through another static initializer;
- modern opaque-pointer NVVM IR and legacy LLVM 7 typed-pointer NVVM IR.

The one-past-the-end kernel forms a constant pointer whose byte addend equals
the allocation size. Const eval permits forming, but not dereferencing, such a
pointer, so the translator materializes it; the kernel checks its distance from
the base equals the 32-byte allocation.

The edge-case kernel checks two byte-level details:

- `STATIC_NAN` keeps the complete `0x7fc01234` NaN payload instead of being
  rewritten to a canonical NaN.
- `PADDED_STATIC`, a `#[repr(C)] { u8, u32 }`, reads the `u32` from its Rust
  layout offset after three padding bytes.

Expected behavior:

| Static kind                  | Memory space          |
|------------------------------|-----------------------|
| Ordinary `static mut`        | Global `addrspace(1)` |
| `SharedArray` / `Barrier`    | Shared `addrspace(3)` |
| `DynamicSharedArray::get()`  | Shared `addrspace(3)` |

The example launches the kernel twice. `DEVICE_COUNTER` should persist across
launches, proving it is global device storage and not per-block shared memory.

Non-zero immutable static initializers are emitted as the exact evaluated byte
image in LLVM/PTX, so device code can read compile-time data without losing
padding, field offsets, or floating-point payload bits. Pointer-width slots are
excluded from the literal byte image and emitted as provenance-preserving LLVM
constant expressions.

Before emitting an initializer, cuda-oxide proves that its typed field loads
use the same offsets and size as rustc. For relocated structs, the physical
initializer carrier may follow rustc's explicit non-overlapping byte ranges
instead of requiring natural LLVM field alignment. This applies to a packed
top-level struct by itself and to one direct nested packed struct field when
the enclosing top-level struct remains naturally representable and the child's
sequential packed representation is exact. Relocation records are still checked
for pointer width, bounds, overlap, target identity, target address space, and
addend bounds. Unsupported layouts fail at compile time instead of producing a
wrong value.

The supported relocation scope is intentionally narrow: thin pointers and slice
fat pointers whose data word targets another device static in global or constant
memory, including zero and non-zero byte addends. For slices, the data word
carries provenance while the length metadata remains literal initializer data.
A top-level union is admitted only when the complete union is one naturally
aligned pointer word and every non-ZST alternative is a
representation-compatible thin pointer. Anonymous promoted allocations,
functions, vtables, trait-object metadata, other fat-pointer forms, unsized
pointees, nested unions, pointer/integer unions, and relocation targets outside
device static storage remain fail-closed. Packed or otherwise unaligned
thin-pointer slots are supported when the containing top-level struct has an
explicit, non-overlapping rustc layout. One direct nested packed struct is also
supported when the enclosing top-level layout is ordinary/non-divergent and the
child's explicit layout is non-overlapping and exactly representable as a
sequential LLVM packed struct. Packed-root plus packed-child and deeper packed
nesting remain fail-closed.
