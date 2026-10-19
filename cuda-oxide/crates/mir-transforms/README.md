# mir-transforms

Optimization passes over the `dialect-mir` IR.

These run in the middle of cuda-oxide's pipeline: after `mem2reg` has promoted
memory slots to plain SSA values, and before the IR is lowered to the LLVM
dialect on its way to PTX. Running here means a pass sees Rust-level structure
— typed aggregates, slices, checked arithmetic — that is gone by the time LLVM
IR exists.

The first pass is loop unrolling, requested by the `#[unroll]` /
`#[unroll(N)]` annotation, which the importer records as a `mir.unroll_hint`
operation inside the annotated loop. Further loop passes belong here too.

Two more sources ship with it. `canonicalize.rs` prepares a loop for the
unroller: it merges multiple back-edges (several `continue`s) into one
synthetic latch and forwards header-carried values out of each loop exit so
full unrolling can resolve uses past an early `break`.
`scalarize_borrowed_aggregate_reads.rs` rewrites bounded read-only array
loads behind non-promotable `mir.field_addr` / `mir.array_element_addr`
projections into value operations (`mir.extract_field` plus
`mir.extract_array_element`), for parameter entry slots before `mem2reg` and
for immutable aggregate pointer arguments with proven caller-private
provenance after it; anything it cannot prove fails closed and keeps the
memory access.
