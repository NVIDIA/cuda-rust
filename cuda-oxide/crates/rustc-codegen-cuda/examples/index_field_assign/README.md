# index_field_assign

Regression test for assigning through a 2-level `Index -> Field` place — writing
`arr[i].field = value` where `arr` is a local array of structs.

The statement translator's 2-level assignment-projection match enumerated
`(Deref, Field)`, `(Field, Field)`, `(Deref, Index)`, `(Index, Index)`, and
`(Field, Index)`, but had no arm for `(Index, Field)` / `(ConstantIndex, Field)`.
Assigning to `arr[i].field` therefore failed to lower:

```
2-level projection Index(_) -> Field(_, _) not yet implemented for assignment
```

The fix adds the missing arm, delegating to `store_through_place_address` — the
same address-walk-and-store helper the sibling index arms (`(Deref, Index)`,
`(Index, Index)`, `(Field, Index)`) and the 3+ projection fallback already use,
and which `Rvalue::Ref` uses to take `&place`.

## Run

```
cargo oxide run index_field_assign
```

Each of 256 threads fills a local `[Cell; 4]` by assigning to `arr[i].a` /
`arr[i].b` (the runtime `Index -> Field` place) and bumps `arr[0].a` (a
`ConstantIndex -> Field` write), then writes the array's reduction to `out[tid]`;
the host verifies every thread's result (`14*tid + 106`), so the test fails if
any write is dropped or misrouted, not merely if codegen aborts.
