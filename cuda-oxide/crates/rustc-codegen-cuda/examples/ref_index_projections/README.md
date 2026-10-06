# ref_index_projections

Minimal reproduction for a `rustc-codegen-cuda` miscompile where a closure that
takes `idx: usize` and reads a captured array through that index has the
**load's `getelementptr` lowered with `idx` replaced by literal 0**. Every call
to the closure therefore returns whatever the slot-0 value happens to be,
regardless of the argument actually passed.

The example is reduced from a CUDA kernel that indexed a two-slot wrapper
through a closure. The closure returned the slot-0 value for both `k` arms,
which later optimisation could then fold into a single path.

## What the example does

`src/main.rs` contains **32 kernels in four groups**. The first group's
**14 variants** bisect the trigger conditions:

| #  | Variant                              | Purpose                                        |
|:---|:-------------------------------------|:-----------------------------------------------|
| 1  | `test_unrolled_baseline`             | No closure; sanity check                       |
| 2  | `test_closure_indexes_into_array`    | Closure indexes a raw `[f32; 2]`               |
| 3  | `test_closure_indexes_via_match`     | Closure uses `match`, not `[idx]`              |
| 4  | `test_closure_into_struct_wrapper`   | Closure indexes through `Pair::get`            |
| 5  | `test_closure_pre_loaded_outside`    | Closure selects pre-loaded values              |
| 6  | `test_closure_node_ref_access`       | Closure indexes through `Pair::node`           |
| 7  | `test_closure_with_shuffle`          | One warp shuffle inside the closure            |
| 8  | `test_closure_two_shuffles`          | Two warp shuffles inside the closure           |
| 9  | `test_two_shuffles_no_indexed_load`  | Two shuffles without an indexed load           |
| 10 | `test_two_shuffles_raw_array`        | Two shuffles with a raw array                  |
| 11 | `test_two_shuffles_no_transparent`   | Two shuffles, non-transparent wrapper          |
| 12 | `test_two_shuffles_inlined`          | Same indexed access, no closure binding        |
| 13 | `test_closure_shuffle_with_captures` | Two shuffles plus extra captures               |
| 14 | `test_closure_via_array_literal`     | Results via `[compute(0), compute(1)]`         |

plus **6 address-walker regression kernels** that pin the borrow and
raw-pointer shapes the unified `translate_place_address` walker must lower:

| #  | Variant                              | Shape pinned                                   |
|:---|:-------------------------------------|:-----------------------------------------------|
| 15 | `test_mut_ref_writethrough`          | `&mut pair.0[k]` write-through                 |
| 16 | `test_constant_index_tail`           | `&(*pr).0[1]` ConstantIndex tail               |
| 17 | `test_raw_addr_of_const`             | `addr_of!(pr.0[k])` raw reads                  |
| 18 | `test_raw_addr_of_mut_writethrough`  | `addr_of_mut!(acc[k])` raw write               |
| 19 | `test_inline_never_node_fn`          | Exact issue #120 `node()` MIR shape            |
| 20 | `test_holder_deref_tail`             | `&hr.0[k]` with Deref inside the tail          |

plus **6 slice-value regression kernels** that pin direct and nested indexing
of projected unsized slice tails, including padded layouts:

| #  | Variant                                  | Shape pinned                                           |
|:---|:-----------------------------------------|:-------------------------------------------------------|
| 21 | `test_slice_tail_constant_index`         | `Field(tail) -> MirSliceType -> ConstantIndex`         |
| 22 | `test_slice_tail_runtime_index`          | `Field(tail) -> MirSliceType -> Index`                 |
| 23 | `test_slice_tail_padded_offset`          | `[u16]` tail at byte offset 10 behind padding          |
| 24 | `test_nested_slice_tail_constant_index`  | `Field(inner) -> Field(tail) -> ConstantIndex`         |
| 25 | `test_nested_slice_tail_runtime_index`   | `Field(inner) -> Field(tail) -> Index`                 |
| 26 | `test_nested_slice_tail_padded_offset`   | Nested padded DST tail with constant + runtime indexes |

plus **6 DST slice-tail address regression kernels** for issue #881:

| #  | Variant                                  | Shape pinned                                        |
|:---|:-----------------------------------------|:----------------------------------------------------|
| 27 | `test_slice_tail_write_constant_index`   | `Field(tail) -> ConstantIndex` mutable store        |
| 28 | `test_slice_tail_write_runtime_index`    | `Field(tail) -> Index` mutable store                |
| 29 | `test_slice_tail_borrow_constant_index`  | `&value.tail[1]` element borrow                     |
| 30 | `test_slice_tail_borrow_runtime_index`   | `&value.tail[k]` element borrow                     |
| 31 | `test_slice_tail_write_padded`           | Padded `[u16]` tail, constant + runtime writes      |
| 32 | `test_slice_tail_borrow_padded`          | Padded `[u16]` tail, constant + runtime borrows     |

Each kernel writes a difference (`r1 - r0`, or original-local readback for
the write-through variants) for inputs chosen so a correct implementation
must produce `+5.0` for every element. The harness prints `PASS` per kernel,
tracks failures, prints a final `SUCCESS` marker when every kernel passes,
and exits non-zero if any kernel reports a wrong diff.

The direct slice-value regressions construct a `SliceTail<[f32; 2]>` and
unsize it to `&SliceTail<[f32]>`. Deref of the fat struct reference must
preserve the runtime tail length long enough for `Field(tail)` to reconstruct
a `MirSliceType` value. `test_slice_tail_constant_index` then exercises a
literal `ConstantIndex`, while `test_slice_tail_runtime_index` uses a
data-derived runtime `Index`. Both normalize the semantic slice value to its
data pointer before reusing the existing pointer-offset + load lowering.
`test_slice_tail_padded_offset` repeats both indexing forms on the issue #870
repro layout (`head: u64`, `tag: u8`, `tail: [u16]`), whose tail sits at byte
offset 10 behind a padding byte: `SliceTail` places its tail at offset 4 with
no padding, so only the padded variant can catch a wrong-tail-offset bug in the
`Field(tail)` address computation.

The nested issue #880 regressions push that same slice tail one struct field
deeper. `NestedOuter<T>` ends in `NestedInner<T>`, so a value read walks
`Deref -> Field(inner) -> Field(tail) -> Index/ConstantIndex`. The outer fat
pointer's length metadata must remain paired with the projected address across
`Field(inner)` instead of being handed only to the immediately following
field. `test_nested_slice_tail_padded_offset` nests `PaddedTail<T>` as the final
field of another struct so the walk must preserve metadata while also honoring
both aggregate field offsets.

The issue #881 regressions exercise the corresponding address-producing path.
Whole-tail borrows such as `&value.tail` already rebuild the DST tail as a
`(data_ptr, len)` slice value. Element writes and borrows continue one
projection farther: after rebuilding that fat tail, the address walker
normalizes it back to its data pointer and reuses the existing
`Index`/`ConstantIndex` element-offset lowering. The padded variants verify
that this address arithmetic still starts at the real tail byte offset rather
than at a naive aggregate prefix.

## Trigger conditions

After bisection, the miscompile fires when **all of** the following are true:

1. The index-by-`usize` happens inside a Rust closure (not the surrounding
   function body).
2. The closure has ≥ 1 captured upvar (so it lowers as a function over
   `&Self`).
3. The closure body uses warp shuffles in a way that prevents LLVM from
   inlining it back into the caller (in practice: ≥ 2 calls, where
   `llvm.nvvm.shfl.sync.*` is marked `convergent`).
4. The array being indexed is reached **through a struct field projection**
   (e.g. `Pair(pub [f32; 2])`). Bare `[f32; 2]` upvars are lowered correctly.

When all four hold, the rustc MIR

```text
_4 = &((*_9).0: [f32; 2])[_2]
```

(place projection chain `[Deref, Field(0, [f32;2]), Index(_2)]`) is silently
truncated by the mir-importer to a `[Deref, Field(0)]` address, dropping the
runtime `Index`.

## Root cause

Before this fix, the `Rvalue::Ref` arm had five cases. Case 2
(`[Deref, Field, …]`) emitted a `MirFieldAddrOp` for the first field, then
walked the remaining projections in an inner loop that only handled further
`Field`s; every other variant hit `_ => break`. After the loop, the function
unconditionally returned the partial field address, **silently discarding**
any tail projections, including a runtime `Index`.

That arm lives in `cuda-oxide/crates/mir-importer/src/translator/rvalue/expr.rs`, and the
address walk it delegates to in `…/rvalue/place_addr.rs`.

The fix delegates the tail walk to the existing
`translate_place_addr_from_slot` helper (which is now also extended to handle
runtime `Index` by emitting `MirArrayElementAddrOp`). As maintainer
hardening, `Rvalue::Ref` and `Rvalue::AddressOf` now share a single
`translate_place_address` entry that walks the full projection list
(including `Deref`), and any projection that cannot be lowered to an address
fails the build loudly instead of silently returning a prefix address. With
the fix, the same MIR lowers to

```text
%v7 = getelementptr inbounds { [2 x float] }, ptr %v5, i32 0, i32 0   ; field 0
%v8 = getelementptr inbounds [2 x float], ptr %v7, i32 0, i64 %v3     ; index
%v9 = load float, ptr %v8                                              ; correct
```

and all 32 kernels report `PASS` (the harness prints a final `SUCCESS` marker
and exits non-zero if any kernel reports a wrong diff).

Issue #880 exposes a separate value-walker gap after the direct DST-tail
support from #873. The old `preserved_slice_tail_len` logic used a one-step
lookahead at `Deref`, so `Field(inner)` caused the fat pointer's length to be
dropped before `Field(tail)` could reconstruct the slice. The updated walker
carries that metadata alongside the projected address through nested
slice-tailed struct fields, only consuming it when the actual `[T]` tail field
is reached. Sized fields drop the metadata, and structurally inconsistent
projection shapes fail loudly.

## Build & run

From the cuda-oxide repository root:

```bash
cargo oxide run ref_index_projections
cargo oxide pipeline ref_index_projections    # dump MIR + LLVM IR
```

Requires a CUDA-capable GPU and the cuda-oxide rustc toolchain.
