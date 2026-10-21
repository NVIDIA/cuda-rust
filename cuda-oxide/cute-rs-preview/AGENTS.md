# cute-rs kernel architecture

Kernels intended to demonstrate or exercise cute-rs (including the Qwen
acceptance workload and `*_cute` examples) must express tensor computation
through cute-rs tensor views, layouts, partitions, register tiles, copies,
reductions and MMA operations. This is a hard requirement, alongside numerical
correctness and competitive measured performance.

- Keep instruction-level CTM operations underneath reusable cute-rs operations.
  CTM kernel declarations, launch contracts, thread identifiers and ABI storage
  types may remain at kernel entry points. Explicit `*_ctm` examples continue to
  demonstrate the lower-level instruction layer.
- Do not hand-code hardware fragment packing, lane-to-value maps, or shared-memory
  swizzle formulas in cute-rs workload kernels. Describe those mappings using
  typed layouts and derive accesses from them.
- When an abstraction is missing, extend cute-rs with a reusable, typed operation
  and test its invariants. Moving a workload-specific implementation behind a
  new name, or wrapping a raw instruction while leaving its layout logic in the
  kernel, does not satisfy this requirement.
- Keep operand roles, element formats, dimensions, storage bounds and mutable
  ownership explicit. State collective-participation and aliasing obligations
  at unsafe boundaries; do not imply the type system proves convergence.
- Preserve arithmetic order and observable rounding unless the task explicitly
  authorizes changing them. Test layout coverage and tails independently of
  device code, then validate GPU numerics, sanitizers and generated instructions.
- Benchmark representative end-to-end workloads against the preserved baseline.
  Better abstractions must not silently sacrifice performance. Record remaining
  performance gaps honestly; abstraction migration alone is not a speedup claim.
- Existing legacy violations are migration work, not a precedent for new code.
  Report any remaining violations explicitly when delivering a migration.

## Example and model documentation

- Keep each entry README approachable, following `examples/elementwise_cute`:
  explain the computation, show the CuTe data flow, give a runnable command,
  and describe the expected result.
- Document public and private APIs, including helpers, types, fields and
  constants. Explain shapes, units, ownership, readiness, rounding and error
  behavior where relevant. Unsafe APIs must state caller obligations; comments
  must not claim that types prove thread convergence or asynchronous completion.
- Put detailed numerical contracts, validation evidence, debugging, profiling
  and benchmark methodology in the workload's `reference/` folder, linked from
  its README. Keep essential prerequisites and known user-visible limitations
  in the entry README.
