# slice_get_mut

Regression example for issue #58: mutable element writes through
slice-shaped (fat) pointers were rejected by the mir-importer.

A `&mut [T]` is a fat pointer: at the ABI level it is a (data pointer,
length) pair, not a single address. Calling `a.get_mut(i)` on a
`&mut [f32; N]` first unsizes the array reference to a fat `&mut [f32]`,
and the inlined `slice::get_mut` body then takes `&mut (*fat)[i]`, i.e.
the place projection chain `[Deref, Index(i)]` over the fat local.

Before the fix, the address walker refused that chain:

```text
Unsupported construct: cannot compute a mutable in-memory address through
fat-pointer deref (projection [Deref, Index(10)]); slices scalarize to
(ptr, len) and a single load would misread the fat value as a thin address
```

and the direct assignment form `a[i] = v` failed separately with
"2-level projection Deref -> Index(...) not yet implemented for
assignment".

The fix loads the whole fat value, extracts field 0 (the thin data
pointer, which addresses the ORIGINAL elements), and continues the
projection walk with a plain pointer offset, so both shared and mutable
borrows stay sound.

When the slice element is itself an array, the first index after the fat
slice deref must still offset by one whole slice element. For
`&mut [[f32; 2]]`, `rows[row][0]` must write row `row`, column 0; treating
the extracted data pointer as a pointer to one array object writes inside
row 0 instead. The example covers both a mutable reference to that place
and a direct projected assignment, including runtime and literal row indices.

## Kernels

| Kernel                       | Shape pinned                                |
|:-----------------------------|:--------------------------------------------|
| `write_get_mut_map`          | `a.get_mut(i).map(\|e\| *e = v)` (issue #58) |
| `write_get_mut_if_let`       | `if let Some(e) = a.get_mut(i)`             |
| `write_index_assign`         | `a[i] = v` through thin `&mut [f32; N]`     |
| `write_slice_index_assign`   | `s[i] = v` through fat `&mut [f32]`         |
| `write_mut_ref_index`        | `&mut a[i]` write (pre-existing guard)      |
| `write_struct_field_get_mut` | `get_mut` on `[Cell; 2]`, then field write  |
| `write_nested_slice_row_ref` | `let e = &mut rows[row][0]; *e = v` through fat `&mut [[f32; 2]]` |
| `write_nested_slice_const_row_ref` | `let e = &mut rows[1][0]; *e = v` through fat `&mut [[f32; 2]]` |
| `write_nested_slice_row_assign` | `rows[row][0] = v` through fat `&mut [[f32; 2]]` |
| `write_nested_slice_wide_nonzero_col` | `rows[row][2] = v` through fat `&mut [[f32; 3]]` (wider element, non-zero column) |

Each kernel writes a distinct, index-dependent pattern into a zeroed
buffer; the host reads everything back and checks every lane, so a write
that lands in a temporary copy (or always in lane 0) fails loudly. The
harness prints `PASS` per kernel, a final `SUCCESS` marker when all pass,
and exits non-zero otherwise.

## Build & run

From the cuda-oxide repository root:

```bash
cargo oxide run slice_get_mut
cargo oxide pipeline slice_get_mut    # dump MIR + LLVM IR
```

Requires a CUDA-capable GPU and the cuda-oxide rustc toolchain.
