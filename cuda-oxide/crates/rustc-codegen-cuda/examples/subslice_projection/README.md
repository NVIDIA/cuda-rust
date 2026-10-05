# subslice_projection

Regression coverage for rustc MIR `ProjectionElem::Subslice` in the MIR importer.

Rustc emits `Subslice` for the `middle @ ..` portion of array and slice patterns. The two forms have different codegen semantics:

- arrays use `Subslice { from, to, from_end: false }`; the result is the sized array place `[T; to - from]` starting at element `from`;
- slices use `Subslice { from, to, from_end: true }`; the result keeps a data pointer advanced by `from` elements and metadata `old_len - from - to`.

Before this fix, the place walkers under `cuda-oxide/crates/mir-importer/src/translator/rvalue/` had no `Subslice` lowering. The value walker reported `Projection element ... not yet implemented in iterative mode`, while the address walker returned `Ok(None)`. Mutable borrows then failed loudly because falling back to a reference to a copy would lose write-through semantics.

## Coverage

The example contains five cases:

| Case | MIR property checked |
|---|---|
| `array value` | sized array subslice loaded by value |
| `array shared ref` | shared reference aliases the projected array region |
| `array mutable ref` | mutable reference writes through to original array storage |
| `slice metadata` | slice data pointer advances and length becomes `old_len - from - to` |
| `slice mutable ref` | mutable slice subslice writes through to original storage |

The helper functions are `#[inline(never)]` so optimized MIR retains the relevant projection in a separate body.

Typical MIR shapes are expected to include:

```text
Subslice { from: 1, to: 3, from_end: false }   # [u32; 4] -> [u32; 2]
Subslice { from: 1, to: 1, from_end: true }    # [u32] -> [u32]
```

## Run

From the cuda-oxide repository root:

```bash
cargo oxide run subslice_projection
cargo oxide pipeline subslice_projection
```

The executable prints one verdict per case and exits non-zero if any result differs from the expected value.
