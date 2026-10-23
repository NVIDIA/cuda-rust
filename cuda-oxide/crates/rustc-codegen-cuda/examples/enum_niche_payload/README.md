# enum_niche_payload

Regression test for an enum whose niche carrier shares bytes with an aggregate
payload: `Option<(f32, f32, &mut f32)>`, the rms_norm `next_cell` shape.

Only the pointer leaf used to get a typed slot. The two floats fell into
`[N x i8]` filler, construction spilled them byte by byte, and the payload
read loaded them back as one wide value. SROA cannot rebuild mixed-type
slices, so the spill slot survived `opt -O2` as a local depot.

The fix gives every leaf of such a payload a typed slot at its rustc byte
offset, reusing the carrier's slot where the types match, and builds and
reads the value with `insertvalue`/`extractvalue`. Size, alignment, field
offsets, and the niche encoding are unchanged; a payload the rule cannot
prove keeps the old lowering.

## Depot

Measured with the same device-only kernel source against each tree's backend:

| entry | e027ac32 (parent) | with this fix |
|---|---|---|
| `smooth_triplets` | 16 B in the entry, 16 B in the outlined `Cells::next_cell` | none |

`lease_borrow` and `lease_raw` are layout-neutral value checks, not part of
the depot criterion: neither emits a `.local` on the parent.

## Run

```
cargo oxide run enum_niche_payload
cargo oxide build enum_niche_payload --verbose
crates/rustc-codegen-cuda/examples/enum_niche_payload/target/release/enum_niche_payload --verify-ptx
```

`--verify-ptx` fails on the parent: `smooth_triplets` carries the 16 B depot.
