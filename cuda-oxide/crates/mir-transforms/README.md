# mir-transforms

Required library analysis and lowering, plus optional optimization passes,
over the `dialect-mir` IR. These operate on Rust-level structure such as typed
aggregates, slices and checked arithmetic before lowering to LLVM.

Library checks run on the prepared module before native provider execution.
Normal preparation promotes memory slots to SSA with `mem2reg`; Full variable
debug retains unpromoted locals, which the same analysis also handles. The
shared Pliron `AnalysisManager` caches the prepared graph. Provider results
supply layouts for storage planning, then native ABI lowering runs before
optional post-mem2reg passes and annotated loop unrolling.

The library implementation is divided by responsibility:

- [`library/admission.rs`](src/library/admission.rs) checks root geometry;
  [`library/stateful.rs`](src/library/stateful.rs) proves collective
  participation, uniformity, output ownership and storage reuse.
- [`library/callback.rs`](src/library/callback.rs) checks callback bodies using
  retained MIR and original source facts.
- [`library/storage.rs`](src/library/storage.rs) allocates scratch from the
  proof and provider layouts; [`library/context_plans.rs`](src/library/context_plans.rs)
  assembles root and helper plans.
- [`library.rs`](src/library.rs) and [`library/contexts.rs`](src/library/contexts.rs)
  lower native operations and materialize helper contexts with those plans.

Loop unrolling is requested by the `#[unroll]` /
`#[unroll(N)]` annotation, which the importer records as a `mir.unroll_hint`
operation inside the annotated loop. Further loop passes belong here too.

`canonicalize.rs` prepares a loop for the
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
