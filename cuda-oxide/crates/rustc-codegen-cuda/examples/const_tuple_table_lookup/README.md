# const_tuple_table_lookup

Reading one field of a runtime-indexed tuple-array constant.

## What this tests

`const PAIRS: [(u8, u32); 256]` read as `let (a, b) = PAIRS[idx]`. Before this
fix, `mir.field_addr` verified only struct, union and enum pointees, so a
tuple field projection fell back to the value path, which loads the whole
table as one first-class-aggregate value -- something LLVM splits back into a
per-element store -- **once per field projected**. Reading both `a` and `b`
from the same element cost two independent whole-table copies.

`const_table_lookup` (#684) fixed the same pathology for a scalar table
(`const T: [f32; N]`); this is the tuple-element case (issue #693).

Measured on an RTX 5060 (sm_120): same example, `cargo oxide inspect`,
before and after this diff with nothing else changed.

| | `st.local` | local depot |
|---|---:|---:|
| before | 878 | 4096 bytes |
| after | 512 | 2048 bytes |

The remaining 512 stores are the table's own base-array materialization into
the depot, a separate, pre-existing limit of #684's byte-image path (which
only trusts primitive-scalar or nested-array elements, so a tuple-element
table keeps its own per-thread copy). What this fix removes is the
*duplication*: before, both fields independently re-triggered a full 256-entry
materialization; after, both fields resolve through one shared
`mir.field_addr`-computed element address.

`tuple_field_store` covers the WRITE side the same verifier change unlocks:
`arr[j].1 = x` through a runtime index and a write through a `&mut`
tuple-field borrow both lower to `mir.field_addr` + `mir.store` on a tuple
pointee, which previously failed dialect verification loudly. The
rustc-reordered `(u8, u32)` element (u32 first in memory) makes the bit-exact
check lock the memory-slot vs declaration-index distinction for stores, as
the read kernel already does for loads.

`sum_lookup` reads a table this fix does not change (`ROW: [u32; 4]`, scalar
elements, single index), run in the same binary as a contrast: its lowering
is untouched by this diff, so its correctness check rules out an unrelated
regression in the ordinary array-constant path.

## Usage

```bash
cargo oxide run const_tuple_table_lookup
```

To run the repository smoketest:

```bash
cuda-oxide/scripts/smoketest.sh -x -v '^const_tuple_table_lookup$'
```

## Expected output

```text
tuple_field_lookup: 16384 elements, exact match
tuple_field_store: 16384 elements, exact match
sum_lookup: 16384 elements, exact match
SUCCESS: tuple-field table lookups match the CPU reference
```

Each thread's index is a multiplicative hash of its thread index, spreading
the table lookup across the table rather than every lane sharing one entry, so
a kernel that read the wrong entry or the wrong field fails the check rather
than passing quietly.
