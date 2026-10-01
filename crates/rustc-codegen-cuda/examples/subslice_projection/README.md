# subslice_projection

Regression coverage for rustc MIR `ProjectionElem::Subslice` in the MIR importer.

Rustc emits `Subslice` for the `middle @ ..` portion of array and slice patterns. The two forms have different codegen semantics:

- arrays use `Subslice { from, to, from_end: false }`; the result is the sized array place `[T; to - from]` starting at element `from`;
- slices use `Subslice { from, to, from_end: true }`; the result keeps a data pointer advanced by `from` elements and metadata `old_len - from - to`.

`Subslice` lowering already supports sized arrays and slice fat pointers. This regression extends that support to `ConstantIndex { from_end: true }` immediately following a slice `Subslice`. The index must use the rebuilt subslice metadata: `subslice_len = old_len - from - to`, then `index = subslice_len - offset`. Coverage includes both value reads and address-based mutable writes, plus a field projection after the from-end index.

## Coverage

The example contains seven cases:

| Case | MIR property checked |
|---|---|
| `array value` | sized array subslice loaded by value |
| `array shared ref` | shared reference aliases the projected array region |
| `array mutable ref` | mutable reference writes through to original array storage |
| `slice metadata` | slice data pointer advances and length becomes `old_len - from - to` |
| `slice mutable ref` | mutable slice subslice writes through to original storage |
| `slice subslice from-end field` | rebuilt slice metadata drives a from-end index followed by struct field projections |
| `slice subslice from-end mutable` | two distinct from-end offsets write through the rebuilt subslice while unrelated elements remain unchanged |

The helper functions are `#[inline(never)]` so optimized MIR retains the relevant projection in a separate body.

Typical MIR shapes are expected to include:

```text
Subslice { from: 1, to: 3, from_end: false }   # [u32; 4] -> [u32; 2]
Subslice { from: 1, to: 1, from_end: true }    # [u32] -> [u32]
Subslice -> ConstantIndex { offset: 1, from_end: true }
Subslice -> ConstantIndex { offset: 2, from_end: true }
Subslice -> ConstantIndex { offset: 1, from_end: true } -> Field
```

## Run

From the cuda-oxide repository root:

```bash
cargo oxide run subslice_projection
cargo oxide pipeline subslice_projection
```

The executable prints one verdict per case and exits non-zero if any result differs from the expected value.
