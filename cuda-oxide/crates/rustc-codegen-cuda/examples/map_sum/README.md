# map_sum

Regression test for `iter().map(...).sum()` — a struct with two zero-sized fields
addressed off a shared base pointer.

`Iterator::sum` composes closures via `core`'s `map_fold`
(`move |acc, elt| g(acc, f(elt))`), which captures the map closure `f` and the
`Sum::sum` fold closure `g` as upvars. Both are zero-sized, so the composed
closure is a ZST struct with two ZST fields, and its body borrows both upvars off
the same base pointer.

In `convert_field_addr`, the ZST-field branch forwarded the base SSA value
directly for the first field. That type-punned the base pointer to that field's
pointee in dialect conversion's type history, so the *sibling* `field_addr` for
the second field resolved its base pointee to the wrong (zero-field) type and
failed to lower:

```
field_addr index 1 out of bounds for struct with 0 fields
```

The fix emits an explicit zero-offset GEP (a distinct result) for the ZST field,
mirroring the union branch, so the base pointer's recorded type stays intact.

## Run

```
cargo oxide run map_sum
```

Two kernels — an `i64` and a `usize` `iter().map(..).sum()` (the two element types
that occur in practice) — each write `out[tid]` and the host verifies the result,
so the test fails if the reduction is wrong, not merely if codegen aborts.
