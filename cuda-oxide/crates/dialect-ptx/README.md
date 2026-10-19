# dialect-ptx

`dialect-ptx` is CUDA Oxide's structured terminal PTX dialect. Its operation
tree can be constructed directly with `PtxBuilder`, emitted deterministically
with the dedicated PTX emitter, or projected from the lossless CST in
`ptx-parse`.

The two representations and writers have distinct authority:

- `ptx-parse::Document` owns exact external source, trivia, unknown syntax,
  and byte-preserving edits.
- `dialect-ptx` owns canonical structure for analysis, construction,
  transformation, and deterministic emission.
- `EditScript::apply_with_map` is the lossless patch path. It preserves all
  untouched bytes and returns original/normalized byte lineage.
- `emit_canonical_module` is the constructed/transformed-IR path. It verifies
  native CFG invariants and may normalize the complete module's formatting.

When operations originate in source, `Projection` keeps statement/scope and
byte-span lineage in a side table. Source lineage is not a required operation
attribute, so generated operations never need synthetic source locations.

One `ptx.callable` identity owns either a single-block `ptx.surface_body` or a
multi-block `ptx.cfg_body`; raising changes the body form without duplicating
callable identity or header attributes. Native indexed-branch tables derive
their emitted targets from ordered CFG successor slots, while fallthrough is
accepted only when it names the next emitted block.

The dialect currently models module and lexical scopes, callable declarations
and definitions, directives, labels, generic instructions, and a raw escape hatch.
Guard predicates (`@%p` / `@!%p`) are a typed attribute on instructions, and
callable name/kind/external are typed attributes verified against the header
text they print through. Typed ISA operations can be added incrementally
without making the lossless parser reject newer PTX spellings.

`Projection::control_flow` recovers a conservative intraprocedural CFG for
direct and indexed branches, predicated fallthrough, and terminal instructions.
It retains CST statement/scope lineage and fails closed for unsupported PTX
versions or unresolved targets instead of attaching guessed successors to the
operation tree.

`RegisterAlphaPlan` and `ScopeFlattenPlan` rewrite surface PTX and gate behind
the same version ceiling as the CFG. Rename plans also fail closed on
vector-element uses of a renamed register (`v.x` lexes as one word) and on
rename targets that would capture a label or callable name.

## Consumers

`ptx-schedule` is the in-tree consumer of the native CFG today, and the only
crate that depends on this one. `analyze_ptx` takes `ControlFlow::analyze`'s
blocks and successor edges to find back-edges, which is what lets a schedule
campaign perturb a spin loop whatever its label, predicate or branch spelling.

Two more are planned and do not depend on this crate yet: IKET PTX
instrumentation (`dialect-iket`), which needs verified block boundaries and
successor edges to place probes, and the Tile-to-SIMT interop epic (#96), which
splices SIMT regions into externally produced PTX. Longer term the raised
dialect is the substrate for a direct-to-PTX emission path that bypasses
textual `.ll` round-trips entirely.

## Transform conventions

Transforms that operate on the raised dialect must use pliron's rewriter,
dialect-conversion, and op-interface infrastructure, per the repo-wide rules;
manual walk-and-replace over raised operations is not acceptable. The
text-domain `EditScript` layer exists only for pre-raising normalization of
surface PTX (alpha-renaming, scope flattening). Once operations exist, edits
go through the IR, and the canonical emitter prints the result.
