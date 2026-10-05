# ptx-parse

Lossless structural views over PTX source text.

`Document::parse` borrows the source and owns only structural indices into it,
so every byte of the input -- including comments, whitespace and syntax this
crate does not recognise -- is still reachable through the original `&str`.
Nothing is reformatted, and nothing is discarded.

```text
PTX source text
     │  Document::parse
     ▼
Document<'source>        tokens, statements, scopes, diagnostics, coverage
     │                   labels, directives, callables, instructions
     │  EditScript
     ▼
edited PTX + byte lineage (AppliedEdits)
```

## What it deliberately does not do

This crate does not type-check the PTX ISA. Instructions are discovered
structurally, so an opcode introduced by a newer PTX version is retained with
the same source spans as one this crate has seen before. Consumers that need
ISA semantics layer that policy over `Instruction::head`.

That choice is what keeps the crate useful against a moving target: a PTX file
from a toolkit newer than this checkout still parses, and the parts a caller
does understand keep exact spans.

## The two representations

`dialect-ptx` is the sibling crate, and the split between them is deliberate:

| Crate         | Owns                                                                       |
|---------------|----------------------------------------------------------------------------|
| `ptx-parse`   | Exact external source, trivia, unknown syntax, byte-preserving edits       |
| `dialect-ptx` | Canonical structure for analysis, construction, transformation, emission   |

Use `EditScript::apply_with_map` when the requirement is to patch real source
and keep every untouched byte; use `dialect-ptx`'s `emit_canonical_module`
when the module was constructed or transformed in IR and normalised formatting
is acceptable.

## Structure

Every statement is one of six kinds -- `Directive`, `Instruction`,
`CallableHeader`, `Label`, `Preprocessor`, `Unknown` -- and each carries exact
source and token ranges. `Unknown` is a first-class outcome rather than an
error: it is how unrecognised syntax keeps its bytes and its spans.

`Document` indexes those statements several ways so callers do not re-scan:

- by scope (`statements_in_scope`, `scopes`)
- by span (`labels_in`, `directives_in`)
- by statement (`labels_for_statement`, `directive_for_statement`,
  `callable_for_statement`)
- by name (`callables_named`), plus `definitions()` for callables with bodies

`diagnostics()` reports what the parse noticed without failing, and
`coverage()` reports how much of the source was structurally classified.
`ParseError` is reserved for input the crate cannot index at all.

## Editing

`EditScript` collects `insert`, `delete` and `replace` operations against
original offsets and applies them in one pass:

- `apply` returns the edited string.
- `apply_with_map` returns `AppliedEdits`, which maps offsets in both
  directions (`original_to_output`, `output_to_original`,
  `original_range_to_output`) with an explicit `MapBias` at edit boundaries.

The mapping is the reason to prefer `apply_with_map`: a caller that recorded
spans against the original text can move them onto the edited text without
re-parsing.

## Consumers

| Crate                 | Uses it for                                                |
|-----------------------|------------------------------------------------------------|
| `cuda-oxide-codegen`  | Reading and patching emitted PTX (`export.rs`, `ptx.rs`)   |
| `cuda-host`           | Editing embedded PTX artifacts (`embedded.rs`)             |
| `cuda-intrinsics-gen` | Checking the PTX each generated intrinsic emits            |
| `cargo-oxide`         | Inspecting PTX for the CLI                                 |
| `dialect-ptx`         | Projecting parsed source into the structured dialect       |
| `ptx-schedule`        | Finding schedule-sensitive sites and inserting sleeps      |

## License

Apache-2.0. See [LICENSE](https://github.com/NVlabs/cuda-oxide/blob/main/LICENSE).
